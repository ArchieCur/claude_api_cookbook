from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import ToolParam
import json

# Load env variables and create client
client = Anthropic()
model = "claude-sonnet-4-5"


# ============================================
# Helper Functions for Streaming
# ============================================

def add_user_message(messages, message):
    """Add a user message to the conversation history.

    Handles both string messages and structured content blocks.
    """
    if isinstance(message, list):
        user_message = {
            "role": "user",
            "content": message,
        }
    else:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(user_message)


def add_assistant_message(messages, message):
    """Add an assistant message to the conversation history.

    Handles Message objects by extracting their content blocks.
    """
    if isinstance(message, list):
        assistant_message = {
            "role": "assistant",
            "content": message,
        }
    elif hasattr(message, "content"):
        content_list = []
        for block in message.content:
            if block.type == "text":
                content_list.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_list.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        assistant_message = {
            "role": "assistant",
            "content": content_list,
        }
    else:
        # String messages need to be wrapped in a list with text block
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(assistant_message)


def chat_stream(
    messages,
    system=None,
    temperature=1.0,
    stop_sequences=[],
    tools=None,
    tool_choice=None,
    betas=[],
):
    """Create a streaming API call to Claude.

    Returns a stream context manager that yields events as they arrive.
    Use with the 'with' statement to ensure proper cleanup.
    """
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tool_choice:
        params["tool_choice"] = tool_choice

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    if betas:
        params["betas"] = betas

    return client.beta.messages.stream(**params)


def text_from_message(message):
    """Extract all text content from a message, skipping tool_use blocks."""
    return "\n".join([block.text for block in message.content if block.type == "text"])


# ============================================
# Tool Functions
# ============================================

def process_data(data, operation="validate"):
    """Generic data processing tool for demonstration.

    Args:
        data: Dictionary containing the data to process
        operation: Type of operation (validate, transform, analyze)

    Returns:
        Success message indicating the operation completed
    """
    print(f"[Processing data with operation: {operation}]")
    return f"Data processed successfully with {operation} operation"


def save_record(record_type, metadata):
    """Generic record saving tool for demonstration.

    Args:
        record_type: The type of record being saved
        metadata: Dictionary containing record metadata

    Returns:
        Success message with record type
    """
    print(f"[Saving {record_type} record with metadata: {metadata}]")
    return f"{record_type.title()} record saved successfully"


# ============================================
# Tool Schemas
# ============================================

# Example 1: Simple schema with primitive types
process_data_schema = ToolParam({
    "name": "process_data",
    "description": "Process data with a specified operation",
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "The data to process",
            },
            "operation": {
                "type": "string",
                "description": "The operation to perform: validate, transform, or analyze",
            },
        },
        "required": ["data", "operation"],
    },
})

# Example 2: Complex nested schema (demonstrates fine-grained streaming benefits)
save_record_schema = ToolParam({
    "name": "save_record",
    "description": "Save a record with metadata",
    "input_schema": {
        "type": "object",
        "properties": {
            "record_type": {
                "type": "string",
                "description": "Type of record being saved",
            },
            "metadata": {
                "type": "object",
                "description": "Metadata about the record",
                "properties": {
                    "priority": {
                        "type": "string",
                        "description": "Priority level (low, medium, high)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags for categorization",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes or description",
                    },
                },
                "required": ["priority", "tags"],
            },
        },
        "required": ["record_type", "metadata"],
    },
})


# ============================================
# Tool Execution
# ============================================

TOOL_FUNCTIONS = {
    "process_data": process_data,
    "save_record": save_record,
}


def run_tool(tool_name, tool_input):
    """Execute a tool by name with the provided input."""
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        return TOOL_FUNCTIONS[tool_name](**tool_input)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_tools(message):
    """Process all tool_use blocks in a message and return tool results."""
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
# Streaming Conversation Loop
# ============================================

def run_conversation(messages, tools=[], tool_choice=None, fine_grained=False):
    """Run a streaming conversation with optional fine-grained tool calling.

    Args:
        messages: Conversation history
        tools: List of tool schemas to make available
        tool_choice: Optional tool choice specification
        fine_grained: If True, enables fine-grained tool streaming for faster response

    The fine_grained parameter enables InputJsonEvent streaming, which shows
    tool inputs as they're being generated rather than waiting for complete
    JSON validation. This provides faster visual feedback but may show
    incomplete JSON during streaming.
    """
    while True:
        with chat_stream(
            messages,
            tools=tools,
            betas=["fine-grained-tool-streaming-2025-05-14"] if fine_grained else [],
            tool_choice=tool_choice,
        ) as stream:
            # Process streaming events
            for chunk in stream:
                # Handle text content streaming
                if chunk.type == "text":
                    print(chunk.text, end="", flush=True)

                # Handle tool call start
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        print(f'\n>>> Tool Call: "{chunk.content_block.name}"')

                # Handle tool input streaming (only with fine-grained enabled)
                if chunk.type == "input_json" and chunk.partial_json:
                    print(chunk.partial_json, end="", flush=True)

                # Handle content block end
                if chunk.type == "content_block_stop":
                    print()

            # Get the final complete message after streaming finishes
            response = stream.get_final_message()

        # Add assistant's response to conversation history
        add_assistant_message(messages, response)

        # Check if Claude wants to use tools
        if response.stop_reason != "tool_use":
            # No more tool calls - conversation is complete
            break

        # Execute tools and collect results
        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

        # If tool_choice was specified, stop after first tool use
        if tool_choice:
            break

    return messages


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Standard Streaming (without fine-grained)")
    print("=" * 60)

    messages = []
    add_user_message(
        messages,
        "Process this data with a validation operation: {\"user_id\": 123, \"email\": \"test@example.com\"}"
    )

    run_conversation(
        messages,
        tools=[process_data_schema],
        fine_grained=False,  # Standard streaming
    )

    print("\n" + "=" * 60)
    print("Example 2: Fine-Grained Streaming (faster tool input display)")
    print("=" * 60)

    messages = []
    add_user_message(
        messages,
        "Save a task record with high priority, tags 'urgent' and 'review', and notes explaining this is a critical item"
    )

    run_conversation(
        messages,
        tools=[save_record_schema],
        fine_grained=True,  # Fine-grained streaming - shows tool inputs as they generate
    )

    print("\n" + "=" * 60)
    print("Example 3: Forced Tool Use with Fine-Grained Streaming")
    print("=" * 60)

    messages = []
    add_user_message(
        messages,
        "Create a sample data processing request"
    )

    run_conversation(
        messages,
        tools=[process_data_schema, save_record_schema],
        tool_choice={"type": "tool", "name": "process_data"},
        fine_grained=True,
    )

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Fine-Grained Tool Streaming Benefits:
1. Faster visual feedback - tool inputs appear as they generate
2. Better user experience for complex nested objects
3. No waiting for complete JSON validation before display

When to use fine_grained=True:
- Tools with complex nested schemas (objects within objects)
- Tools that generate long string values
- Interactive applications where responsiveness matters

When to use fine_grained=False (standard):
- Simple tool schemas with few fields
- When you only care about the final result
- When you want guaranteed valid JSON at each step

Key Events in Fine-Grained Streaming:
- chunk.type == "text" - Text content from Claude
- chunk.type == "content_block_start" - Start of tool call
- chunk.type == "input_json" - Tool input chunks (fine-grained only)
  - chunk.partial_json - The JSON chunk
  - chunk.snapshot - Cumulative JSON so far
- chunk.type == "content_block_stop" - End of content block

Technical Details:
- Requires beta flag: "fine-grained-tool-streaming-2025-05-14"
- Uses client.beta.messages.stream() instead of standard create()
- InputJsonEvent provides both partial_json and snapshot properties
- Standard streaming buffers and validates; fine-grained streams immediately
""")
