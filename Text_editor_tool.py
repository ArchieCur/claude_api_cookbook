from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message
import os
import shutil
import json

# Load env variables and create client
client = Anthropic()
model = "claude-sonnet-4-5"


# ============================================
# Helper Functions
# ============================================

def add_user_message(messages, message):
    """Add a user message to the conversation history."""
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    """Add an assistant message to the conversation history."""
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
    """Make an API call to Claude with optional tools."""
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    """Extract all text content from a message."""
    return "\n".join([block.text for block in message.content if block.type == "text"])


# ============================================
# Text Editor Tool Implementation
# ============================================

class TextEditorTool:
    """A file editing tool with backup/restore capabilities.

    Provides safe file operations including:
    - view: Read file contents with optional line range
    - str_replace: Replace exact text (requires unique match)
    - create: Create new files
    - insert: Insert text at specific line
    - undo_edit: Restore from backup

    All operations are sandboxed to base_dir for security.
    """

    def __init__(self, base_dir: str = "", backup_dir: str = ""):
        """Initialize the text editor tool.

        Args:
            base_dir: Root directory for file operations (defaults to cwd)
            backup_dir: Directory for backup files (defaults to .backups/)
        """
        self.base_dir = base_dir or os.getcwd()
        self.backup_dir = backup_dir or os.path.join(self.base_dir, ".backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    def _validate_path(self, file_path: str) -> str:
        """Validate and normalize file path to prevent directory traversal."""
        abs_path = os.path.normpath(os.path.join(self.base_dir, file_path))
        if not abs_path.startswith(self.base_dir):
            raise ValueError(
                f"Access denied: Path '{file_path}' is outside the allowed directory"
            )
        return abs_path

    def _backup_file(self, file_path: str) -> str:
        """Create a timestamped backup of the file."""
        if not os.path.exists(file_path):
            return ""
        file_name = os.path.basename(file_path)
        backup_path = os.path.join(
            self.backup_dir, f"{file_name}.{os.path.getmtime(file_path):.0f}"
        )
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _restore_backup(self, file_path: str) -> str:
        """Restore the most recent backup of a file."""
        file_name = os.path.basename(file_path)
        backups = [
            f for f in os.listdir(self.backup_dir) if f.startswith(file_name + ".")
        ]
        if not backups:
            raise FileNotFoundError(f"No backups found for {file_path}")

        latest_backup = sorted(backups, reverse=True)[0]
        backup_path = os.path.join(self.backup_dir, latest_backup)

        shutil.copy2(backup_path, file_path)
        return f"Successfully restored {file_path} from backup"

    def _count_matches(self, content: str, old_str: str) -> int:
        """Count occurrences of a string in content."""
        return content.count(old_str)

    def view(self, file_path: str, view_range: list = None) -> str:
        """View file contents or directory listing.

        Args:
            file_path: Path to file or directory
            view_range: Optional [start_line, end_line] (-1 for end)

        Returns:
            File contents with line numbers or directory listing
        """
        try:
            abs_path = self._validate_path(file_path)

            # Handle directory listing
            if os.path.isdir(abs_path):
                try:
                    return "\n".join(os.listdir(abs_path))
                except PermissionError:
                    raise PermissionError(
                        "Permission denied. Cannot list directory contents."
                    )

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")

            # Apply line range if specified
            if view_range:
                start, end = view_range
                if end == -1:
                    end = len(lines)
                lines = lines[start - 1 : end]
                start_num = start
            else:
                start_num = 1

            # Add line numbers
            result = []
            for i, line in enumerate(lines, start_num):
                result.append(f"{i}: {line}")

            return "\n".join(result)

        except UnicodeDecodeError:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                "File contains non-text content and cannot be displayed.",
            )
        except ValueError as e:
            raise ValueError(str(e))
        except PermissionError:
            raise PermissionError("Permission denied. Cannot access file.")
        except Exception as e:
            raise type(e)(str(e))

    def str_replace(self, file_path: str, old_str: str, new_str: str) -> str:
        """Replace exact text match in file (requires unique match).

        Args:
            file_path: Path to file
            old_str: Exact text to replace
            new_str: Replacement text

        Returns:
            Success message

        Raises:
            ValueError: If 0 or multiple matches found
        """
        try:
            abs_path = self._validate_path(file_path)

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            match_count = self._count_matches(content, old_str)

            if match_count == 0:
                raise ValueError(
                    "No match found for replacement. Please check your text and try again."
                )
            elif match_count > 1:
                raise ValueError(
                    f"Found {match_count} matches for replacement text. "
                    "Please provide more context to make a unique match."
                )

            # Create backup before modifying
            self._backup_file(abs_path)

            # Perform the replacement
            new_content = content.replace(old_str, new_str)

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return "Successfully replaced text at exactly one location."

        except ValueError as e:
            raise ValueError(str(e))
        except PermissionError:
            raise PermissionError("Permission denied. Cannot modify file.")
        except Exception as e:
            raise type(e)(str(e))

    def create(self, file_path: str, file_text: str) -> str:
        """Create a new file with specified content.

        Args:
            file_path: Path for new file
            file_text: Content for the file

        Returns:
            Success message

        Raises:
            FileExistsError: If file already exists
        """
        try:
            abs_path = self._validate_path(file_path)

            if os.path.exists(abs_path):
                raise FileExistsError(
                    "File already exists. Use str_replace to modify it."
                )

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(file_text)

            return f"Successfully created {file_path}"

        except ValueError as e:
            raise ValueError(str(e))
        except PermissionError:
            raise PermissionError("Permission denied. Cannot create file.")
        except Exception as e:
            raise type(e)(str(e))

    def insert(self, file_path: str, insert_line: int, new_str: str) -> str:
        """Insert text after specified line number.

        Args:
            file_path: Path to file
            insert_line: Line number to insert after (0 for beginning)
            new_str: Text to insert

        Returns:
            Success message
        """
        try:
            abs_path = self._validate_path(file_path)

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            # Create backup before modifying
            self._backup_file(abs_path)

            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Handle line endings
            if lines and not lines[-1].endswith("\n"):
                new_str = "\n" + new_str

            # Insert at the beginning if insert_line is 0
            if insert_line == 0:
                lines.insert(0, new_str + "\n")
            # Insert after the specified line
            elif insert_line > 0 and insert_line <= len(lines):
                lines.insert(insert_line, new_str + "\n")
            else:
                raise IndexError(
                    f"Line number {insert_line} is out of range. "
                    f"File has {len(lines)} lines."
                )

            with open(abs_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            return f"Successfully inserted text after line {insert_line}"

        except ValueError as e:
            raise ValueError(str(e))
        except PermissionError:
            raise PermissionError("Permission denied. Cannot modify file.")
        except Exception as e:
            raise type(e)(str(e))

    def undo_edit(self, file_path: str) -> str:
        """Restore file from most recent backup.

        Args:
            file_path: Path to file

        Returns:
            Success message

        Raises:
            FileNotFoundError: If no backups exist
        """
        try:
            abs_path = self._validate_path(file_path)

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            return self._restore_backup(abs_path)

        except ValueError as e:
            raise ValueError(str(e))
        except FileNotFoundError:
            raise FileNotFoundError("No previous edits to undo")
        except PermissionError:
            raise PermissionError("Permission denied. Cannot restore file.")
        except Exception as e:
            raise type(e)(str(e))


# ============================================
# Tool Execution
# ============================================

# Initialize the text editor tool
text_editor_tool = TextEditorTool()


def run_tool(tool_name, tool_input):
    """Execute text editor tool commands.

    Dispatches to appropriate TextEditorTool method based on command.
    """
    if tool_name == "str_replace_editor":
        command = tool_input["command"]

        if command == "view":
            return text_editor_tool.view(
                tool_input["path"],
                tool_input.get("view_range")
            )
        elif command == "str_replace":
            return text_editor_tool.str_replace(
                tool_input["path"],
                tool_input["old_str"],
                tool_input["new_str"]
            )
        elif command == "create":
            return text_editor_tool.create(
                tool_input["path"],
                tool_input["file_text"]
            )
        elif command == "insert":
            return text_editor_tool.insert(
                tool_input["path"],
                tool_input["insert_line"],
                tool_input["new_str"],
            )
        elif command == "undo_edit":
            return text_editor_tool.undo_edit(tool_input["path"])
        else:
            raise Exception(f"Unknown text editor command: {command}")
    else:
        raise Exception(f"Unknown tool name: {tool_name}")


def run_tools(message):
    """Process all tool_use blocks in a message."""
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks = []

    for tool_request in tool_requests:
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


# ============================================
# Text Editor Schema
# ============================================

def get_text_edit_schema(model):
    """Get the text editor tool schema for the specified model.

    The text editor tool is a built-in tool provided by Anthropic.
    Instead of defining the full schema manually, we reference the
    built-in type that Claude already understands.

    Args:
        model: Model version (used for future compatibility)

    Returns:
        Tool schema configuration
    """
    return {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }


# ============================================
# Conversation Loop
# ============================================

def run_conversation(messages):
    """Run a conversation loop with text editor tool support.

    Continues until Claude provides a final response without tool use.
    """
    while True:
        response = chat(
            messages,
            tools=[get_text_edit_schema(model)],
        )

        add_assistant_message(messages, response)

        # Print any text response from Claude
        text_response = text_from_message(response)
        if text_response:
            print(text_response)

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

    return messages


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Create and Edit a File")
    print("=" * 60)

    # Create a test directory for examples
    test_dir = os.path.join(os.getcwd(), "test_files")
    os.makedirs(test_dir, exist_ok=True)

    # Use TextEditorTool directly (without Claude)
    editor = TextEditorTool(base_dir=test_dir)

    # Example 1a: Create a new file
    print("\n1a. Creating a new file...")
    try:
        result = editor.create(
            "example.py",
            'def greet(name):\n    print(f"Hello, {name}!")\n\ngreet("World")'
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}")

    # Example 1b: View the file
    print("\n1b. Viewing the file...")
    try:
        content = editor.view("example.py")
        print(content)
    except Exception as e:
        print(f"Error: {e}")

    # Example 1c: Replace text
    print("\n1c. Replacing text...")
    try:
        result = editor.str_replace(
            "example.py",
            'print(f"Hello, {name}!")',
            'return f"Hello, {name}!"'
        )
        print(result)
        print("\nUpdated content:")
        print(editor.view("example.py"))
    except Exception as e:
        print(f"Error: {e}")

    # Example 1d: Insert text
    print("\n1d. Inserting text...")
    try:
        result = editor.insert(
            "example.py",
            3,
            "\n# Call the function"
        )
        print(result)
        print("\nUpdated content:")
        print(editor.view("example.py"))
    except Exception as e:
        print(f"Error: {e}")

    # Example 1e: Undo last edit
    print("\n1e. Undoing last edit...")
    try:
        result = editor.undo_edit("example.py")
        print(result)
        print("\nRestored content:")
        print(editor.view("example.py"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Example 2: Claude Using Text Editor Tool")
    print("=" * 60)
    print("\nNote: Uncomment the code below to test with Claude API")

    """
    messages = []
    add_user_message(
        messages,
        "Create a simple Python function in a file called 'calculator.py' that adds two numbers."
    )

    run_conversation(messages)
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Text Editor Tool Pattern:
1. Built-in tool type - Uses Anthropic's text_editor_20250728 type
2. Command-based interface - Single tool with multiple commands
3. Automatic backups - All edits create timestamped backups
4. Path validation - Sandboxed to base_dir for security
5. Unique match requirement - str_replace requires exactly one match

Available Commands:
- view: Read file contents with optional line range
- str_replace: Replace exact text (requires unique match)
- create: Create new files
- insert: Insert text at specific line number
- undo_edit: Restore from most recent backup

Security Features:
- Path validation prevents directory traversal
- Operations sandboxed to base_dir
- Automatic backup before modifications
- Error handling for permissions and file operations

When to Use Text Editor Tool:
- File manipulation tasks (reading, editing, creating)
- Code generation and modification
- Configuration file updates
- Any task requiring precise file operations

Built-in Tool Benefits:
- No need to define complex JSON schema
- Claude has optimized prompting for this tool
- Standardized interface across applications
- Regular updates from Anthropic
""")

    # Cleanup test directory
    # shutil.rmtree(test_dir, ignore_errors=True)
