"""
MODEL CONTEXT PROTOCOL (MCP) - Complete Implementation Guide

MCP is a protocol that enables AI applications to connect to external data sources
and tools through a standardized interface. Instead of manually writing tool schemas
and integration code, MCP provides a framework where:

- Server authors define tools, resources, and prompts once
- Client applications can use any MCP server without custom integration
- End users can add MCP servers to their applications via configuration

THE THREE MCP PRIMITIVES:

1. TOOLS (Model-controlled)
   - Claude decides when to call them
   - Used to give Claude capabilities (read files, call APIs, execute code)
   - Results are used by Claude to formulate responses

2. RESOURCES (App-controlled)
   - Your application decides when to fetch them
   - Used to add context to messages (documents, data, search results)
   - Fetched when user @mentions them or app needs context

3. PROMPTS (User-controlled)
   - End user explicitly triggers them
   - Pre-tested, reusable prompt templates
   - Triggered via slash commands (/format, /summarize) or UI buttons

REAL-WORLD USAGE:

You typically build EITHER a server OR a client, not both:

BUILD AN MCP SERVER when:
- Creating an integration for a service (GitHub, databases, file systems)
- Exposing your company's internal tools/data to AI applications
- Publishing a reusable server for the community

BUILD AN MCP CLIENT when:
- Creating an application that uses existing MCP servers
- Building a chat interface that connects to various services
- Integrating MCP capabilities into your product

END USERS typically:
- Just install/configure MCP servers in their apps (like Claude Desktop)
- Don't write any code

ARCHITECTURE FLOW:

User → Your App (MCP Client) → MCP Server → External Service (GitHub, DB, etc.)
     ← Your App ← MCP Server ← External Service

Your App → Claude (with tools from MCP)
     ← Claude (tool use decision)

Your App → MCP Server (execute tool)
     ← MCP Server (tool result)

Your App → Claude (tool result)
     ← Claude (final response)

TRANSPORT:
MCP is transport-agnostic. Common options:
- Standard I/O (stdio) - Most common for local servers
- HTTP/SSE - For remote servers
- WebSocket - For bidirectional communication

This guide covers Standard I/O transport.
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from contextlib import AsyncExitStack

if TYPE_CHECKING:
    from mcp import ClientSession, StdioServerParameters, types  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
    from pydantic import AnyUrl  # type: ignore

# ============================================================================
# DEPENDENCIES
# ============================================================================
"""
Required packages:

pip install anthropic python-dotenv "mcp[cli]>=1.8.0"

Or with pyproject.toml:

[project]
dependencies = [
    "anthropic>=0.51.0",
    "mcp[cli]>=1.8.0",
    "python-dotenv>=1.1.0",
]
"""

# ============================================================================
# MCP SERVER - DEFINING TOOLS, RESOURCES, AND PROMPTS
# ============================================================================
"""
MCP SERVERS expose capabilities to AI applications.

Use FastMCP framework for simple server creation:
- Automatic schema generation from function signatures
- Decorator-based API (@mcp.tool, @mcp.resource, @mcp.prompt)
- Built-in stdio transport support
"""

# SERVER EXAMPLE: Document Management Server
SERVER_CODE = '''
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from pydantic import Field

# Initialize server
mcp = FastMCP("DocumentMCP", log_level="ERROR")

# Sample data store
docs = {
    "report.pdf": "Q4 sales report showing 15% growth...",
    "notes.md": "Meeting notes from product review...",
    "data.csv": "timestamp,value,category\\n2024-01-01,100,A\\n...",
}


# ============================================================================
# DEFINING TOOLS
# ============================================================================
# Tools are actions Claude can perform.
# Claude decides when to call them based on user requests.

@mcp.tool(
    name="read_document",
    description="Read the contents of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="ID of the document to read"),
) -> str:
    """
    Tools are defined using the @mcp.tool() decorator.

    Key features:
    - Function parameters become tool input schema automatically
    - Use Pydantic Field() for parameter descriptions
    - Type hints define parameter types
    - Return value is the tool result
    - Raise exceptions for errors
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found")

    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing text. Replacement must match exactly.",
)
def edit_document(
    doc_id: str = Field(description="ID of the document to edit"),
    old_text: str = Field(description="Text to replace (must match exactly)"),
    new_text: str = Field(description="New text to insert"),
) -> str:
    """
    Tools can have multiple parameters and perform mutations.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found")

    if old_text not in docs[doc_id]:
        raise ValueError(f"Text to replace not found in document")

    docs[doc_id] = docs[doc_id].replace(old_text, new_text)
    return f"Successfully updated {doc_id}"


@mcp.tool(
    name="search_documents",
    description="Search for documents containing a specific term.",
)
def search_documents(
    query: str = Field(description="Search term to find in documents"),
) -> List[str]:
    """
    Tools can return complex types (lists, dicts).
    SDK handles serialization automatically.
    """
    results = [
        doc_id for doc_id, content in docs.items()
        if query.lower() in content.lower()
    ]
    return results


# ============================================================================
# DEFINING RESOURCES
# ============================================================================
# Resources expose data that your application can fetch and add to context.
# Your application decides when to fetch them (e.g., @mentions).

# DIRECT RESOURCES (no parameters, static data)
@mcp.resource("docs://documents", mime_type="application/json")
def list_documents() -> List[str]:
    """
    Direct resources have no parameters.

    Key features:
    - Custom URI scheme (docs://)
    - MIME type tells clients the format
    - Return value automatically serialized
    - Used for lists, catalogs, static data
    """
    return list(docs.keys())


# TEMPLATED RESOURCES (parameterized, dynamic data)
@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
def get_document(doc_id: str) -> str:
    """
    Templated resources have URI parameters.

    Key features:
    - Parameters in URI become function arguments
    - Multiple parameters supported: "docs://{type}/{id}"
    - Used for fetching specific items
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found")
    return docs[doc_id]


@mcp.resource("docs://metadata/{doc_id}", mime_type="application/json")
def get_metadata(doc_id: str) -> Dict[str, Any]:
    """
    Resources can return structured data.
    JSON MIME type + dict return = automatic serialization.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found")

    return {
        "id": doc_id,
        "size": len(docs[doc_id]),
        "word_count": len(docs[doc_id].split()),
    }


# ============================================================================
# DEFINING PROMPTS
# ============================================================================
# Prompts are pre-tested, reusable prompt templates.
# Users explicitly trigger them via slash commands or UI actions.

@mcp.prompt(
    name="summarize",
    description="Generate a concise summary of a document.",
)
def summarize_document(
    doc_id: str = Field(description="ID of the document to summarize"),
) -> List[base.Message]:
    """
    Prompts return a list of messages that form the complete prompt.

    Key features:
    - Can return multi-turn conversations
    - Use base.UserMessage, base.AssistantMessage
    - Can reference tools and resources
    - Pre-tested for quality
    """
    prompt = f"""
    Your task is to create a concise summary of the document.

    Document to summarize:
    <document_id>{doc_id}</document_id>

    Instructions:
    1. Use the 'read_document' tool to fetch the document contents
    2. Create a 2-3 sentence summary capturing the main points
    3. Focus on key insights and actionable information

    Provide only the summary, no explanation.
    """

    return [base.UserMessage(prompt)]


@mcp.prompt(
    name="format_markdown",
    description="Reformat a document using markdown syntax with headers and lists.",
)
def format_markdown(
    doc_id: str = Field(description="ID of the document to format"),
) -> List[base.Message]:
    """
    Prompts can orchestrate tool usage.
    """
    prompt = f"""
    Reformat the document to use proper markdown syntax.

    Document ID: {doc_id}

    Steps:
    1. Read the document using 'read_document' tool
    2. Add markdown headers (##, ###), bullet points, and formatting
    3. Use 'edit_document' tool to apply changes
    4. Return the formatted version

    Make it well-structured and easy to read.
    """

    return [base.UserMessage(prompt)]


@mcp.prompt(
    name="compare_documents",
    description="Compare two documents and highlight key differences.",
)
def compare_documents(
    doc_id_1: str = Field(description="ID of the first document"),
    doc_id_2: str = Field(description="ID of the second document"),
) -> List[base.Message]:
    """
    Prompts can have multiple parameters.
    """
    prompt = f"""
    Compare these two documents and identify key differences:

    Document 1: {doc_id_1}
    Document 2: {doc_id_2}

    Use the 'read_document' tool to fetch both documents.

    Provide a structured comparison highlighting:
    - Major differences in content
    - Unique information in each
    - Common themes
    """

    return [base.UserMessage(prompt)]


# ============================================================================
# RUN THE SERVER
# ============================================================================
if __name__ == "__main__":
    # Run server with stdio transport (standard for MCP)
    mcp.run(transport="stdio")
'''

# To run the server:
# python mcp_server.py


# ============================================================================
# MCP CLIENT - CONNECTING TO AND USING MCP SERVERS
# ============================================================================
"""
MCP CLIENTS connect to MCP servers and expose their capabilities to AI applications.

Key responsibilities:
- Connect to servers via stdio, HTTP, or WebSocket
- List available tools, resources, and prompts
- Execute tool calls
- Fetch resources
- Get prompt templates
"""

# Import MCP client dependencies
try:
    from mcp import ClientSession, StdioServerParameters, types  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
    from pydantic import AnyUrl  # type: ignore

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    types = None  # type: ignore
    stdio_client = None  # type: ignore
    AnyUrl = None  # type: ignore
    print("MCP not installed. Install with: pip install 'mcp[cli]>=1.8.0'")


class MCPClient:
    """
    Wrapper around MCP ClientSession for easier usage.

    Handles connection lifecycle and provides clean API for:
    - Listing and calling tools
    - Reading resources
    - Listing and getting prompts
    """

    def __init__(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize MCP client.

        Args:
            command: Command to run the MCP server (e.g., "python", "uv", "node")
            args: Arguments for the command (e.g., ["mcp_server.py"])
            env: Optional environment variables for the server process

        Examples:
            # Python server
            MCPClient(command="python", args=["mcp_server.py"])

            # With uv
            MCPClient(command="uv", args=["run", "mcp_server.py"])

            # Node.js server
            MCPClient(command="node", args=["server.js"])
        """
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        """
        Connect to the MCP server and initialize the session.

        This spawns the server process and establishes communication.
        """
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )

        # Create stdio transport
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio_read, stdio_write = stdio_transport

        # Create and initialize session
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(stdio_read, stdio_write)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        """Get the active session, raising error if not connected."""
        if self._session is None:
            raise ConnectionError(
                "Not connected. Call connect() first or use async context manager."
            )
        return self._session

    # ========================================================================
    # TOOLS
    # ========================================================================

    async def list_tools(self) -> List[types.Tool]:
        """
        List all tools available from the MCP server.

        Returns list of Tool objects with:
        - name: Tool identifier
        - description: What the tool does
        - inputSchema: JSON schema for parameters
        """
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> types.CallToolResult:
        """
        Execute a tool with the given input.

        Args:
            tool_name: Name of the tool to call
            tool_input: Dictionary of parameter name -> value

        Returns:
            CallToolResult with content and error status
        """
        return await self.session().call_tool(tool_name, tool_input)

    # ========================================================================
    # RESOURCES
    # ========================================================================

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI (e.g., "docs://documents", "docs://documents/report.pdf")

        Returns:
            Parsed resource content based on MIME type:
            - application/json -> Parsed dict/list
            - text/* -> String
        """
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]

        if isinstance(resource, types.TextResourceContents):
            # Parse JSON resources
            if resource.mimeType == "application/json":
                return json.loads(resource.text)

            # Return text resources as-is
            return resource.text

        # Handle other resource types if needed
        return resource

    # ========================================================================
    # PROMPTS
    # ========================================================================

    async def list_prompts(self) -> List[types.Prompt]:
        """
        List all prompts available from the MCP server.

        Returns list of Prompt objects with:
        - name: Prompt identifier
        - description: What the prompt does
        - arguments: List of required/optional parameters
        """
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str],
    ) -> List[types.PromptMessage]:
        """
        Get a prompt template with arguments filled in.

        Args:
            prompt_name: Name of the prompt
            arguments: Dictionary mapping parameter names to values

        Returns:
            List of messages forming the complete prompt
        """
        result = await self.session().get_prompt(prompt_name, arguments)
        return result.messages

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def cleanup(self) -> None:
        """Close the connection and clean up resources."""
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# ============================================================================
# INTEGRATION - BRIDGING MCP AND CLAUDE
# ============================================================================
"""
The integration layer converts between MCP and Claude API formats.
"""

from anthropic import Anthropic
from anthropic.types import Message, ToolResultBlockParam

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def convert_mcp_tools_to_claude_format(
    mcp_tools: List[types.Tool],
) -> List[Dict[str, Any]]:
    """
    Convert MCP tool definitions to Claude API format.

    MCP tools have name, description, and inputSchema.
    Claude expects the same format, so this is mostly a pass-through.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
        }
        for tool in mcp_tools
    ]


async def execute_tool_with_mcp(
    mcp_client: MCPClient,
    tool_name: str,
    tool_input: Dict[str, Any],
) -> ToolResultBlockParam:
    """
    Execute a tool via MCP and return result in Claude format.

    Handles errors and formats the result for Claude.
    """
    try:
        result = await mcp_client.call_tool(tool_name, tool_input)

        # Extract text content from result
        content_list = [
            item.text
            for item in result.content
            if isinstance(item, types.TextContent)
        ]
        content_json = json.dumps(content_list)

        return {
            "type": "tool_result",
            "content": content_json,
            "is_error": result.isError if hasattr(result, "isError") else False,
        }

    except Exception as e:
        return {
            "type": "tool_result",
            "content": json.dumps({"error": str(e)}),
            "is_error": True,
        }


async def process_claude_tool_calls(
    mcp_client: MCPClient,
    message: Message,
    tool_use_id_map: Optional[Dict[str, str]] = None,
) -> List[ToolResultBlockParam]:
    """
    Process all tool use blocks from Claude's response.

    Args:
        mcp_client: MCP client to execute tools with
        message: Claude's response containing tool_use blocks
        tool_use_id_map: Optional mapping to track tool use IDs

    Returns:
        List of tool results to send back to Claude
    """
    tool_results = []

    for block in message.content:
        if block.type == "tool_use":
            result = await execute_tool_with_mcp(
                mcp_client,
                block.name,
                block.input,
            )
            result["tool_use_id"] = block.id
            tool_results.append(result)

    return tool_results


def convert_mcp_prompt_to_claude_messages(
    prompt_messages: List[types.PromptMessage],
) -> List[Dict[str, Any]]:
    """
    Convert MCP prompt messages to Claude message format.

    MCP prompts return messages that can be added directly to
    the conversation history.
    """
    converted = []

    for msg in prompt_messages:
        role = "user" if msg.role == "user" else "assistant"

        # Handle different content formats
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, dict) and "text" in msg.content:
            content = msg.content["text"]
        elif isinstance(msg.content, list):
            # Handle list of content blocks
            content = [
                {"type": "text", "text": item.get("text", "")}
                if isinstance(item, dict)
                else {"type": "text", "text": str(item)}
                for item in msg.content
            ]
        else:
            content = str(msg.content)

        converted.append({"role": role, "content": content})

    return converted


async def add_resources_to_context(
    mcp_client: MCPClient,
    user_query: str,
) -> str:
    """
    Extract @mentions from query and fetch resources.

    Example: "Tell me about @report.pdf" -> fetches docs://documents/report.pdf

    Returns formatted context to add to the message.
    """
    # Extract @mentions
    words = user_query.split()
    mentions = [word[1:] for word in words if word.startswith("@")]

    if not mentions:
        return ""

    # Fetch document list
    doc_ids = await mcp_client.read_resource("docs://documents")

    # Fetch mentioned documents
    context_parts = []
    for mention in mentions:
        if mention in doc_ids:
            content = await mcp_client.read_resource(f"docs://documents/{mention}")
            context_parts.append(
                f'<document id="{mention}">\n{content}\n</document>'
            )

    return "\n".join(context_parts)


# ============================================================================
# COMPLETE EXAMPLE - MCP-ENABLED CHAT APPLICATION
# ============================================================================

async def mcp_chat_example():
    """
    Complete example of using MCP with Claude.

    This demonstrates:
    1. Connecting to an MCP server
    2. Listing available tools
    3. Converting tools to Claude format
    4. Having a conversation with tool use
    5. Handling resources and prompts
    """
    # Connect to MCP server
    async with MCPClient(
        command="python",
        args=["mcp_server.py"],
    ) as mcp_client:

        # Get available tools
        mcp_tools = await mcp_client.list_tools()
        claude_tools = convert_mcp_tools_to_claude_format(mcp_tools)

        print(f"Connected to MCP server with {len(claude_tools)} tools:")
        for tool in claude_tools:
            print(f"  - {tool['name']}: {tool['description']}")
        print()

        # Example 1: Simple tool use
        print("Example 1: Using tools")
        print("=" * 60)

        messages = [
            {
                "role": "user",
                "content": "What documents are available? Please read the first one.",
            }
        ]

        # First turn
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )

        print("Claude's response:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
            elif block.type == "tool_use":
                print(f"\n[Tool use: {block.name}]")
        print()

        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = await process_claude_tool_calls(mcp_client, response)
            messages.append({"role": "user", "content": tool_results})

            # Get final response
            final_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                tools=claude_tools,
                messages=messages,
            )

            print("Final response:")
            for block in final_response.content:
                if block.type == "text":
                    print(block.text)
            print()

        # Example 2: Using resources
        print("\nExample 2: Using resources with @mentions")
        print("=" * 60)

        user_query = "Summarize @report.pdf and @notes.md"
        context = await add_resources_to_context(mcp_client, user_query)

        enhanced_query = f"""
        {user_query}

        Here's the content:
        {context}

        Note: The @ symbol is just for mentioning. Actual doc names don't include @.
        """

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": enhanced_query}],
        )

        print("Response:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
        print()

        # Example 3: Using prompts
        print("\nExample 3: Using prompts (slash commands)")
        print("=" * 60)

        # List available prompts
        prompts = await mcp_client.list_prompts()
        print("Available prompts:")
        for prompt in prompts:
            print(f"  /{prompt.name}: {prompt.description}")
        print()

        # Get and use a prompt
        prompt_messages = await mcp_client.get_prompt(
            "summarize",
            {"doc_id": "report.pdf"}
        )

        # Convert to Claude format
        messages = convert_mcp_prompt_to_claude_messages(prompt_messages)

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )

        print("Prompt execution result:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
        print()


# ============================================================================
# TESTING AND DEVELOPMENT
# ============================================================================
"""
MCP INSPECTOR - Development Tool

The MCP Inspector is a web-based UI for testing MCP servers during development.

To use:
1. Install the inspector: npm install -g @modelcontextprotocol/inspector
2. Run your server with inspector: mcp-inspector python mcp_server.py
3. Open the web UI to:
   - View all tools, resources, and prompts
   - Test tool execution with different parameters
   - Fetch resources and see results
   - Execute prompts
   - Debug issues

Note: The inspector is under active development and may have updates.
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    if not MCP_AVAILABLE:
        print("MCP not installed. Install with: pip install 'mcp[cli]>=1.8.0'")
        sys.exit(1)

    # Run the example
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(mcp_chat_example())


"""
PATTERN SUMMARY:

================================================================================
MCP SERVER PATTERNS
================================================================================

1. Tools (@mcp.tool)
   - Model-controlled: Claude decides when to call
   - Use for actions: reading, writing, searching, calculating
   - Function signature → automatic schema generation
   - Pydantic Field() for parameter descriptions
   - Raise exceptions for errors

2. Resources (@mcp.resource)
   - App-controlled: Your code decides when to fetch
   - Use for data: documents, lists, search results
   - Direct resources: Static data, no parameters
   - Templated resources: URI parameters → function arguments
   - MIME types: application/json, text/plain, etc.
   - Automatic serialization based on return type

3. Prompts (@mcp.prompt)
   - User-controlled: Explicit user action (slash commands, buttons)
   - Use for workflows: predefined tasks, multi-step operations
   - Return list of messages
   - Can reference tools and resources
   - Pre-tested for quality

4. Server Initialization
   - FastMCP("ServerName", log_level="ERROR")
   - Define tools, resources, prompts with decorators
   - Run with: mcp.run(transport="stdio")

================================================================================
MCP CLIENT PATTERNS
================================================================================

1. Connection
   - MCPClient(command, args, env)
   - Use async context manager
   - Stdio transport for local servers

2. Tools
   - list_tools() → Get available tools
   - call_tool(name, input) → Execute tool
   - Convert to Claude format for API

3. Resources
   - read_resource(uri) → Fetch data
   - Automatic parsing based on MIME type
   - Use for @mentions, context injection

4. Prompts
   - list_prompts() → Get available prompts
   - get_prompt(name, args) → Get filled template
   - Convert to Claude messages

================================================================================
INTEGRATION PATTERNS
================================================================================

1. Tool Flow
   - List tools from MCP → Convert to Claude format
   - Claude decides to use tool → Extract tool_use blocks
   - Execute via MCP → Get results
   - Send results to Claude → Get final response

2. Resource Flow
   - Parse user query for @mentions
   - Fetch resources from MCP
   - Add to message context
   - Send to Claude

3. Prompt Flow
   - User triggers slash command
   - Get prompt from MCP with arguments
   - Convert to message format
   - Add to conversation
   - Send to Claude

4. Multi-Client Pattern
   - Connect to multiple MCP servers
   - Aggregate tools from all servers
   - Route tool calls to correct server
   - Combine capabilities

================================================================================
BEST PRACTICES
================================================================================

Tools:
✓ Focus on atomic actions
✓ Clear, specific descriptions
✓ Validate inputs and provide good error messages
✓ Keep tool logic simple and focused
✗ Don't make tools that do too many things
✗ Don't expose dangerous operations without safeguards

Resources:
✓ Use appropriate MIME types
✓ Direct resources for lists/catalogs
✓ Templated resources for specific items
✓ Return structured data when helpful
✗ Don't fetch huge amounts of data
✗ Don't perform expensive operations in resources

Prompts:
✓ Test thoroughly with different inputs
✓ Write detailed, specific instructions
✓ Focus on server's core purpose
✓ Include clear descriptions
✓ Design to work with your tools/resources
✗ Don't write vague or generic prompts
✗ Don't create prompts unrelated to your server's purpose

Development:
✓ Use MCP Inspector for testing
✓ Test each primitive independently
✓ Handle errors gracefully
✓ Log important events
✓ Document your server's capabilities
✗ Don't skip error handling
✗ Don't assume inputs are valid

================================================================================
WHEN TO USE EACH PRIMITIVE
================================================================================

Use TOOLS when:
✓ You want Claude to autonomously decide when to act
✓ The action requires Claude's judgment
✓ It's an operation (read, write, search, calculate)
✓ You're giving Claude new capabilities

Use RESOURCES when:
✓ You're adding context to messages
✓ User explicitly references data (@mentions)
✓ You need to inject information into prompts
✓ It's data retrieval, not actions

Use PROMPTS when:
✓ You have pre-tested workflows
✓ Users need explicit control (slash commands)
✓ It's a complex multi-step task
✓ You want consistent, reproducible results

================================================================================
ARCHITECTURE DECISIONS
================================================================================

Build a SERVER when:
✓ Creating integration for a service (GitHub, database, API)
✓ Exposing company tools/data to AI
✓ Publishing reusable functionality
✓ You want others to use your integration

Build a CLIENT when:
✓ Creating an application that uses MCP servers
✓ Building custom chat interface
✓ Integrating MCP into existing product
✓ You want to use existing MCP servers

Build BOTH when:
✓ Learning MCP (like this course!)
✓ Creating end-to-end solution
✗ Not typical in production

================================================================================
COMMON PATTERNS
================================================================================

1. Document Management Server
   - Tools: read, write, edit, search documents
   - Resources: list documents, get specific document, metadata
   - Prompts: summarize, format, compare documents

2. Database Server
   - Tools: query, insert, update, delete
   - Resources: list tables, get schema, get row counts
   - Prompts: generate report, analyze data, find anomalies

3. API Integration Server
   - Tools: create, update, delete resources via API
   - Resources: list items, get specific items, search results
   - Prompts: common workflows for the API

4. File System Server
   - Tools: read file, write file, list directory, search
   - Resources: file contents, directory listings, file metadata
   - Prompts: organize files, find duplicates, summarize folder

5. Multi-Service Aggregator
   - Connect to multiple MCP servers
   - Combine tools from all servers
   - Route operations to appropriate server
   - Unified interface for user

================================================================================
TROUBLESHOOTING
================================================================================

Connection Issues:
- Verify server command and args are correct
- Check server logs for errors
- Ensure server is using stdio transport
- Test server independently first

Tool Execution Issues:
- Verify tool name matches exactly
- Check input schema matches
- Look for server-side errors
- Test with MCP Inspector first

Resource Issues:
- Verify URI format is correct
- Check MIME type handling
- Ensure resource exists
- Test with Inspector

Prompt Issues:
- Verify all required arguments provided
- Check message format conversion
- Test prompt independently
- Ensure references to tools/resources are valid
"""
