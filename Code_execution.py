"""
CODE EXECUTION AND FILES API

CODE EXECUTION:
Claude can write and execute Python code in a secure, sandboxed environment.
This enables data analysis, visualization, calculations, file processing, and more.

CRITICAL: Code execution is STATELESS
- Each execution starts with a completely clean slate
- No variables, imports, or state from previous executions persist
- Every execution must redeclare all variables and reimport all libraries
- This is fundamentally different from a Jupyter notebook where state persists

FILES API:
Upload files to Claude's servers for analysis, processing, or reference.
Files remain available for the duration of your session and can be:
- Uploaded for Claude to analyze
- Downloaded after Claude processes them
- Listed to see what's available
- Deleted when no longer needed

SUPPORTED FILE TYPES:

Documents:
- PDF (.pdf) - application/pdf
- Text (.txt, .md, .py, .js, .html, .css) - text/plain
- CSV (.csv) - text/csv
- JSON (.json) - application/json
- XML (.xml) - application/xml

Spreadsheets:
- Excel (.xlsx) - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
- Excel 97-2003 (.xls) - application/vnd.ms-excel

Images:
- JPEG (.jpg, .jpeg) - image/jpeg
- PNG (.png) - image/png
- GIF (.gif) - image/gif
- WebP (.webp) - image/webp

BETA FEATURES:
Both Code Execution and Files API are in beta and require special headers.
Beta features may have breaking changes and are subject to rate limits.

USE CASES:

Code Execution:
- Data analysis and statistics
- Data visualization and plotting
- Mathematical calculations
- File format conversions
- Text processing and analysis
- Image manipulation
- Scientific computing

Files API:
- Upload datasets for analysis
- Share documents for processing
- Provide reference materials
- Retrieve generated outputs (plots, reports, processed files)
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from anthropic import Anthropic

# Initialize client with beta headers
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    default_headers={
        "anthropic-beta": "code-execution-2025-08-25, files-api-2025-04-14"
    },
)

# Use specific model version that supports code execution
model = "claude-sonnet-4-5-20250929"


# ============================================================================
# FILES API - FILE MANAGEMENT OPERATIONS
# ============================================================================


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file based on its extension.

    Raises ValueError if extension is not supported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    mime_type_map = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/plain",
        ".py": "text/plain",
        ".js": "text/plain",
        ".html": "text/plain",
        ".css": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    mime_type = mime_type_map.get(extension)

    if not mime_type:
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported: {', '.join(mime_type_map.keys())}"
        )

    return mime_type


def upload_file(file_path: str) -> Any:
    """
    Upload a file to Claude's servers.

    Returns file metadata including file_id needed for referencing the file.
    """
    path = Path(file_path)
    mime_type = get_mime_type(file_path)
    filename = path.name

    with open(file_path, "rb") as file:
        return client.beta.files.upload(file=(filename, file, mime_type))


def list_files() -> Any:
    """
    List all uploaded files.

    Returns metadata for all files currently uploaded.
    """
    return client.beta.files.list()


def download_file(file_id: str, output_filename: Optional[str] = None) -> None:
    """
    Download a file from Claude's servers.

    If output_filename is not provided, uses the original filename.
    """
    file_content = client.beta.files.download(file_id)

    if not output_filename:
        file_metadata = get_file_metadata(file_id)
        file_content.write_to_file(file_metadata.filename)
    else:
        file_content.write_to_file(output_filename)


def delete_file(file_id: str) -> Any:
    """
    Delete a file from Claude's servers.

    Removes the file and frees up storage.
    """
    return client.beta.files.delete(file_id)


def get_file_metadata(file_id: str) -> Any:
    """
    Get metadata for a specific file.

    Returns information like filename, size, type, upload time.
    """
    return client.beta.files.retrieve_metadata(file_id)


# ============================================================================
# CODE EXECUTION - BASIC PATTERN
# ============================================================================


def execute_code(
    user_request: str, system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ask Claude to write and execute code to fulfill a request.

    Claude will:
    1. Write Python code to accomplish the task
    2. Execute the code in a sandboxed environment
    3. Return results including any output or errors

    Returns the full message response including code and execution results.
    """
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt if system_prompt else [],
        messages=[{"role": "user", "content": user_request}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    )
    return message


# ============================================================================
# CODE EXECUTION WITH FILE UPLOAD
# ============================================================================


def analyze_file_with_code(
    file_path: str, analysis_request: str, system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload a file and ask Claude to analyze it using code execution.

    Common use cases:
    - Data analysis on CSV files
    - Statistical analysis
    - Data visualization
    - File format conversion
    - Text analysis
    """
    file_metadata = upload_file(file_path)

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt if system_prompt else [],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_request},
                    {"type": "container_upload", "file_id": file_metadata.id},
                ],
            }
        ],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    )
    return message


# ============================================================================
# CODE EXECUTION WITH MULTIPLE FILES
# ============================================================================


def analyze_multiple_files(
    file_paths: List[str], analysis_request: str, system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload multiple files and analyze them together.

    Use cases:
    - Compare datasets
    - Merge multiple CSV files
    - Cross-reference documents
    - Batch processing
    """
    uploaded_files = [upload_file(path) for path in file_paths]

    content = [{"type": "text", "text": analysis_request}]
    for file_metadata in uploaded_files:
        content.append({"type": "container_upload", "file_id": file_metadata.id})

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt if system_prompt else [],
        messages=[{"role": "user", "content": content}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    )
    return message


# ============================================================================
# MULTI-TURN CODE EXECUTION
# ============================================================================


class CodeExecutionSession:
    """
    Manage a multi-turn code execution conversation.

    IMPORTANT: Remember that code execution is stateless. Each execution
    starts fresh. If you need to reference previous results, you must
    explicitly ask Claude to regenerate or reload data.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = system_prompt
        self.uploaded_files: List[str] = []

    def upload_file(self, file_path: str) -> str:
        """Upload a file and track it for this session."""
        file_metadata = upload_file(file_path)
        self.uploaded_files.append(file_metadata.id)
        return file_metadata.id

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: Any) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": message.content})

    def execute(self, user_request: str) -> Any:
        """
        Execute a code request and maintain conversation history.

        NOTE: Code execution is stateless. Each call starts with a clean slate.
        Previous variables and imports do NOT persist between executions.
        """
        self.add_user_message(user_request)

        message = client.messages.create(
            model=model,
            max_tokens=8192,
            system=self.system_prompt if self.system_prompt else [],
            messages=self.messages,
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        )

        self.add_assistant_message(message)
        return message

    def cleanup(self) -> None:
        """Delete all uploaded files for this session."""
        for file_id in self.uploaded_files:
            try:
                delete_file(file_id)
            except Exception as e:
                print(f"Failed to delete file {file_id}: {e}")


# ============================================================================
# HELPER FUNCTIONS FOR PROCESSING RESPONSES
# ============================================================================


def extract_code_blocks(message: Any) -> List[str]:
    """
    Extract all code blocks from a message response.

    Returns a list of code strings that were executed.
    """
    code_blocks = []
    for block in message.content:
        if hasattr(block, "type") and block.type == "tool_use":
            if hasattr(block, "input") and "code" in block.input:
                code_blocks.append(block.input["code"])
    return code_blocks


def extract_execution_results(message: Any) -> List[Dict[str, Any]]:
    """
    Extract execution results from a message response.

    Returns a list of result dictionaries containing output, errors, etc.
    """
    results = []
    for block in message.content:
        if hasattr(block, "type") and block.type == "tool_result":
            results.append(
                {
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": getattr(block, "is_error", False),
                }
            )
    return results


def get_text_response(message: Any) -> str:
    """
    Extract the text response from a message.

    Returns the assistant's explanation/narrative, excluding code blocks.
    """
    text_blocks = []
    for block in message.content:
        if hasattr(block, "type") and block.type == "text":
            text_blocks.append(block.text)
    return "\n".join(text_blocks)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple code execution (no file upload)
    print("Example 1: Simple mathematical calculation")
    print("=" * 60)

    response = execute_code(
        """
        Calculate the first 10 Fibonacci numbers and create a simple plot showing
        the growth pattern. Use matplotlib for visualization.

        Remember: You must import all libraries in your code since execution is stateless.
        """
    )

    print("Claude's response:")
    print(get_text_response(response))
    print("\nCode executed:")
    for code in extract_code_blocks(response):
        print(code)
    print()

    # Example 2: Data analysis with file upload
    print("\nExample 2: Analyze a CSV file")
    print("=" * 60)

    # Note: Replace with actual file path
    # response = analyze_file_with_code(
    #     file_path="data/sales.csv",
    #     analysis_request="""
    #     Analyze this sales data:
    #     1. Calculate summary statistics (mean, median, std)
    #     2. Identify trends over time
    #     3. Create visualizations showing key patterns
    #     4. Highlight any anomalies or outliers
    #
    #     Generate at least 2-3 detailed plots.
    #
    #     CRITICAL: Remember to import all necessary libraries (pandas, matplotlib, etc.)
    #     at the start of your code since execution is stateless.
    #     """
    # )
    #
    # print("Analysis complete!")
    # print(get_text_response(response))
    print("(Skipped - requires actual CSV file)")
    print()

    # Example 3: Multi-turn conversation with code execution
    print("\nExample 3: Multi-turn code execution session")
    print("=" * 60)

    session = CodeExecutionSession(
        system_prompt="""You are a data analyst. When executing code:
        1. Always import required libraries at the start
        2. Include clear comments explaining your approach
        3. Create visualizations when helpful
        4. Remember that each execution is stateless - no variables persist"""
    )

    response1 = session.execute(
        """
        Generate a random dataset with 1000 samples containing:
        - Age (20-80)
        - Income (30k-200k)
        - Spending (10k-100k)

        Calculate basic statistics and show correlation between variables.
        """
    )
    print("Turn 1:")
    print(get_text_response(response1))
    print()

    response2 = session.execute(
        """
        Create a detailed visualization showing the relationship between
        income and spending from the data you just generated.

        IMPORTANT: Since execution is stateless, you need to regenerate
        the dataset first before creating the visualization.
        """
    )
    print("Turn 2:")
    print(get_text_response(response2))
    print()

    # Example 4: Working with multiple files
    print("\nExample 4: Analyze multiple files together")
    print("=" * 60)

    # Note: Replace with actual file paths
    # response = analyze_multiple_files(
    #     file_paths=["data/q1_sales.csv", "data/q2_sales.csv", "data/q3_sales.csv"],
    #     analysis_request="""
    #     Combine these quarterly sales files and:
    #     1. Merge them into a single dataset
    #     2. Calculate quarterly trends
    #     3. Compare performance across quarters
    #     4. Create a comprehensive visualization
    #
    #     Remember to import all required libraries (pandas, matplotlib, etc.).
    #     """
    # )
    print("(Skipped - requires actual CSV files)")
    print()

    # Example 5: File management operations
    print("\nExample 5: File management")
    print("=" * 60)

    print("Listing uploaded files:")
    files = list_files()
    for file in files.data:
        metadata = get_file_metadata(file.id)
        print(f"  - {metadata.filename} ({metadata.id})")
        print(f"    Size: {metadata.size} bytes")
        print(f"    Type: {metadata.mime_type}")

    # Cleanup example
    # print("\nCleaning up files:")
    # for file in files.data:
    #     delete_file(file.id)
    #     print(f"  Deleted: {file.id}")


"""
PATTERN SUMMARY:

1. Simple Code Execution
   - Use case: Mathematical calculations, data generation, simple analysis
   - Pattern: execute_code(request)
   - No file upload needed

2. File Analysis with Code
   - Use case: Analyze uploaded CSV, Excel, text files
   - Pattern: analyze_file_with_code(file_path, request)
   - Single file upload + analysis

3. Multiple File Analysis
   - Use case: Compare, merge, or analyze multiple files together
   - Pattern: analyze_multiple_files(file_paths, request)
   - Upload multiple files in one request

4. Multi-Turn Code Session
   - Use case: Iterative analysis, refinement, follow-up questions
   - Pattern: CodeExecutionSession class
   - Maintains conversation history

5. File Management
   - Use case: Upload, list, download, delete files
   - Pattern: upload_file(), list_files(), download_file(), delete_file()
   - Manage file lifecycle

CRITICAL REMINDERS:

Stateless Execution:
✗ Variables do NOT persist between executions
✗ Libraries do NOT persist between executions
✗ File handles do NOT persist between executions
✓ Each execution starts completely fresh
✓ Must redeclare/reimport everything every time

Best Practices:
1. Always remind Claude that execution is stateless
2. Import all libraries at the start of each code block
3. Regenerate or reload data if needed across turns
4. Use descriptive variable names for clarity
5. Include error handling in generated code
6. Clean up uploaded files when done

File Upload Tips:
- Files remain available throughout your session
- Reference uploaded files by their file_id
- Use container_upload type in message content
- Delete files when no longer needed to free storage

Code Execution Limits:
- Sandboxed environment (no network access, no file system access beyond uploaded files)
- Time limits on execution (typically 60 seconds)
- Memory limits apply
- Only standard Python libraries available (pandas, numpy, matplotlib, etc.)

WHEN TO USE CODE EXECUTION:

DO use when:
✓ Need numerical calculations or data analysis
✓ Want data visualizations or plots
✓ Processing structured data (CSV, JSON, Excel)
✓ Need deterministic, reproducible results
✓ Mathematical or statistical operations
✓ File format conversions

DON'T use when:
✗ Simple text processing (use Claude's text capabilities)
✗ Need external API calls (code execution is sandboxed)
✗ Real-time data or streaming
✗ File system operations beyond uploaded files
✗ Tasks requiring network access
"""
