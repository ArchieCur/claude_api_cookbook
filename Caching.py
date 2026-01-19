"""
PROMPT CACHING - Cost and Performance Optimization

Prompt caching reduces costs and latency by caching frequently used content.
When you send the same content repeatedly (system prompts, tools, documents, images),
caching can dramatically reduce both response time and token costs.

CACHE DURATION EXPLAINED:

The default cache duration is 5 minutes, and it gets automatically refreshed
(extending another 5 minutes) each time the cached content is used—at no additional cost.
This means if you're making requests every few minutes, your cache effectively stays
warm indefinitely.

The 1-hour cache duration is an optional upgrade that costs more, but guarantees
the cache stays live for a full hour regardless of usage patterns. This is useful
if you have sporadic requests that might be more than 5 minutes apart but less
than an hour apart.

WHAT CAN BE CACHED:

Cache breakpoints can be added to:
- System prompts (most common use case)
- Tool definitions (for multi-turn tool conversations)
- Image blocks (for analyzing the same image multiple times)
- Tool use and tool result blocks (for multi-turn workflows)
- Document/RAG context (for multiple queries against the same documents)

CACHE BREAKPOINTS:

- You can add up to 4 cache breakpoints per request
- Cache breakpoints are added using the "cache_control" parameter
- Only one cache_control type is available: {"type": "ephemeral"}

PROCESSING ORDER:

Claude processes your request components in a specific order:
1. Tools first
2. Then system prompt
3. Then messages

This order matters when deciding where to place cache breakpoints for optimal reuse.

BEST PRACTICES:

1. Cache content that:
   - Is large (>1000 tokens)
   - Repeats across requests (system prompts, few-shot examples, tools)
   - Changes infrequently (static instructions, tool definitions)

2. Place cache breakpoints at content boundaries:
   - End of system prompt
   - Last tool definition
   - After large document blocks in messages
   - After image blocks for multi-question analysis

3. Structure requests to maximize cache hits:
   - Keep cached content at the beginning of the request
   - Only vary the user query/input
   - Maintain consistent ordering of cached elements

4. Cost optimization:
   - Cache writes cost more than regular tokens (25% markup)
   - Cache reads cost significantly less (90% discount)
   - Break-even point is typically 2-3 requests with the same cached content

IMPORTANT: Minimum cacheable content is 1024 tokens (about 750 words).
Anything smaller won't be cached.
"""

import os
import base64
from typing import Dict, List, Any, Optional
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-5"


# ============================================================================
# PATTERN 1: CACHING LARGE SYSTEM PROMPTS
# ============================================================================
# Use case: System prompts with detailed instructions, examples, or guidelines
# that remain constant across many requests.


def chat_with_cached_system(
    user_message: str,
    system_prompt: str,
    use_cache: bool = True,
) -> Any:
    """
    Send a message with a system prompt that can be cached.

    The system prompt is cached if it's large enough (1024+ tokens).
    Subsequent calls with the same system prompt will reuse the cached version.
    """
    if use_cache:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = system_prompt

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return message


# ============================================================================
# PATTERN 2: CACHING TOOL DEFINITIONS
# ============================================================================
# Use case: Multi-turn conversations where the same tools are available
# throughout the entire conversation.


def create_cached_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add cache_control to the last tool in the list.

    Since Claude processes tools first, caching tool definitions ensures
    they're reused across multiple turns in a conversation.
    """
    if not tools:
        return tools

    cached_tools = tools.copy()
    if len(cached_tools) > 0:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    return cached_tools


def chat_with_cached_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    use_cache: bool = True,
) -> Any:
    """
    Multi-turn conversation with cached tool definitions.

    The tool definitions are cached on the first request and reused
    for all subsequent turns.
    """
    if use_cache:
        tools = create_cached_tools(tools)

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    return message


# ============================================================================
# PATTERN 3: CACHING IMAGES FOR MULTI-QUESTION ANALYSIS
# ============================================================================
# Use case: Analyzing the same image with multiple different questions.


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_cache(
    image_path: str,
    question: str,
    use_cache: bool = True,
) -> Any:
    """
    Analyze an image with caching enabled.

    The image is cached, allowing multiple questions about the same image
    without re-uploading and re-processing it.
    """
    image_base64 = encode_image(image_path)

    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64,
        },
    }

    if use_cache:
        image_block["cache_control"] = {"type": "ephemeral"}

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    image_block,
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return message


# ============================================================================
# PATTERN 4: CACHING DOCUMENT/RAG CONTEXT
# ============================================================================
# Use case: Multiple questions against the same document or context.


def chat_with_cached_context(
    context: str,
    question: str,
    use_cache: bool = True,
) -> Any:
    """
    Ask questions about a document with the document cached.

    The document context is cached, allowing multiple questions
    without re-processing the entire document each time.
    """
    context_block = {"type": "text", "text": context}

    if use_cache:
        context_block["cache_control"] = {"type": "ephemeral"}

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    context_block,
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return message


# ============================================================================
# PATTERN 5: MULTI-BREAKPOINT CACHING STRATEGY
# ============================================================================
# Use case: Complex workflows using system prompt + tools + context,
# maximizing cache efficiency with up to 4 breakpoints.


def chat_with_multi_breakpoint_cache(
    system_prompt: str,
    tools: List[Dict[str, Any]],
    context: str,
    user_question: str,
) -> Any:
    """
    Advanced caching strategy using multiple cache breakpoints.

    Cache breakpoints are strategically placed at:
    1. End of tool definitions (processed first by Claude)
    2. End of system prompt (processed second)
    3. End of document context (processed third, in messages)

    This maximizes cache reuse for complex, multi-turn workflows.
    """
    cached_tools = create_cached_tools(tools)

    cached_system = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    cached_context_block = {
        "type": "text",
        "text": context,
        "cache_control": {"type": "ephemeral"},
    }

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=cached_system,
        tools=cached_tools,
        messages=[
            {
                "role": "user",
                "content": [
                    cached_context_block,
                    {"type": "text", "text": user_question},
                ],
            }
        ],
    )
    return message


# ============================================================================
# PATTERN 6: CACHE STATISTICS AND MONITORING
# ============================================================================
# Use case: Track cache performance to optimize costs.


def print_cache_stats(message: Any) -> None:
    """
    Print cache usage statistics from the API response.

    Helps you understand cache efficiency and cost savings.
    """
    usage = message.usage

    print("Token Usage:")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")

    if hasattr(usage, "cache_creation_input_tokens"):
        print(f"  Cache creation tokens: {usage.cache_creation_input_tokens}")
        print(f"  Cache read tokens: {usage.cache_read_input_tokens}")

        if usage.cache_read_input_tokens > 0:
            savings_pct = (
                usage.cache_read_input_tokens / usage.input_tokens * 100
                if usage.input_tokens > 0
                else 0
            )
            print(f"  Cache hit rate: {savings_pct:.1f}%")


# ============================================================================
# PATTERN 7: CONDITIONAL CACHING FOR DEVELOPMENT VS PRODUCTION
# ============================================================================
# Use case: Disable caching during development, enable in production.


class CachedConversation:
    """
    Conversation manager with configurable caching.

    Useful for toggling caching on/off based on environment
    or testing different caching strategies.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_cache: bool = True,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.enable_cache = enable_cache
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def send_message(self, user_message: str) -> Any:
        """Send a message with optional caching based on configuration."""
        self.add_message("user", user_message)

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "messages": self.messages,
        }

        if self.system_prompt and self.enable_cache:
            params["system"] = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif self.system_prompt:
            params["system"] = self.system_prompt

        if self.tools and self.enable_cache:
            params["tools"] = create_cached_tools(self.tools)
        elif self.tools:
            params["tools"] = self.tools

        message = client.messages.create(**params)
        self.add_message("assistant", message.content[0].text)

        return message


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Cached system prompt for consistent behavior
    print("Example 1: Caching a large system prompt")
    print("=" * 60)

    large_system_prompt = """You are an expert data analyst.

When analyzing data, follow these steps:
1. Understand the context and objectives
2. Examine data quality and completeness
3. Identify patterns and anomalies
4. Draw evidence-based conclusions
5. Provide actionable recommendations

Guidelines:
- Always cite specific data points when making claims
- Consider multiple interpretations
- Acknowledge limitations and uncertainties
- Use clear, non-technical language for explanations

""" * 10  # Repeat to exceed 1024 token minimum

    response1 = chat_with_cached_system(
        "What are the key principles of data analysis?", large_system_prompt
    )
    print("First request (cache write):")
    print_cache_stats(response1)
    print()

    response2 = chat_with_cached_system(
        "How should I handle missing data?", large_system_prompt
    )
    print("Second request (cache hit):")
    print_cache_stats(response2)
    print()

    # Example 2: Cached tool definitions for multi-turn conversation
    print("\nExample 2: Caching tool definitions")
    print("=" * 60)

    sample_tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        },
    ]

    messages = [{"role": "user", "content": "What tools do you have available?"}]

    response = chat_with_cached_tools(messages, sample_tools, use_cache=True)
    print("Tool definitions cached:")
    print_cache_stats(response)
    print()

    # Example 3: Document Q&A with cached context
    print("\nExample 3: Caching document context for multiple questions")
    print("=" * 60)

    long_document = """
    [Product Requirements Document]

    Product Name: Smart Task Manager
    Version: 2.0

    Overview:
    Smart Task Manager is a productivity application that helps users
    organize, prioritize, and complete tasks efficiently using AI-powered
    recommendations and intelligent scheduling.

    Key Features:
    - Natural language task creation
    - AI-powered priority recommendations
    - Smart scheduling based on user habits
    - Cross-platform synchronization
    - Collaborative task management

    """ * 20  # Repeat to exceed 1024 token minimum

    response1 = chat_with_cached_context(
        long_document, "What are the key features?", use_cache=True
    )
    print("First question (cache write):")
    print_cache_stats(response1)
    print()

    response2 = chat_with_cached_context(
        long_document, "What's the product version?", use_cache=True
    )
    print("Second question (cache hit):")
    print_cache_stats(response2)
    print()

    # Example 4: Multi-breakpoint caching strategy
    print("\nExample 4: Multi-breakpoint caching (system + tools + context)")
    print("=" * 60)

    response = chat_with_multi_breakpoint_cache(
        system_prompt=large_system_prompt,
        tools=sample_tools,
        context=long_document,
        user_question="Analyze this document and suggest next steps",
    )
    print("Multi-breakpoint caching:")
    print_cache_stats(response)
    print()

    # Example 5: Conversation with configurable caching
    print("\nExample 5: Managed conversation with caching")
    print("=" * 60)

    conversation = CachedConversation(
        system_prompt=large_system_prompt, tools=sample_tools, enable_cache=True
    )

    response = conversation.send_message("Hello! What can you help me with?")
    print("Conversation turn 1:")
    print_cache_stats(response)
    print()

    response = conversation.send_message("Can you analyze some data for me?")
    print("Conversation turn 2 (cached system + tools):")
    print_cache_stats(response)
    print()


"""
PATTERN SUMMARY:

1. Cached System Prompts
   - Best for: Instructions, examples, guidelines that don't change
   - Cache location: system parameter with cache_control
   - Benefit: Consistent behavior with 90% cost reduction on cache hits

2. Cached Tool Definitions
   - Best for: Multi-turn conversations with the same available tools
   - Cache location: Last tool in tools array with cache_control
   - Benefit: Tools available across conversation without repeated processing

3. Cached Images
   - Best for: Multiple questions about the same image
   - Cache location: Image block in messages with cache_control
   - Benefit: Image processed once, queried many times

4. Cached Document Context
   - Best for: Q&A over large documents, RAG applications
   - Cache location: Document text block in messages with cache_control
   - Benefit: Document processed once, multiple queries answered

5. Multi-Breakpoint Strategy
   - Best for: Complex workflows combining system + tools + context
   - Cache locations: Tools (1st), system (2nd), context (3rd)
   - Benefit: Maximum cache efficiency with up to 4 breakpoints

6. Cache Monitoring
   - Best for: Understanding cache performance and cost savings
   - Method: Parse usage statistics from API response
   - Benefit: Optimize caching strategy based on real metrics

7. Conditional Caching
   - Best for: Different behavior in dev vs production
   - Method: Toggle cache_control based on environment
   - Benefit: Flexibility and easier debugging

COST CONSIDERATIONS:

Cache Write Costs:
- Base input tokens: 1x cost
- Cache write: 1.25x cost (25% markup)
- Cache write happens on first request with new content

Cache Read Costs:
- Regular input tokens: 1x cost
- Cache read: 0.1x cost (90% discount)
- Cache read happens on subsequent requests with same cached content

Break-Even Analysis:
- 1 request: Slightly more expensive (25% markup on cache write)
- 2 requests: Roughly break-even point
- 3+ requests: Cost savings increase significantly

Example Calculation:
- 10,000 token system prompt
- 5 requests

Without caching:
  5 requests × 10,000 tokens × 1x = 50,000 token cost

With caching:
  1 cache write: 10,000 tokens × 1.25x = 12,500
  4 cache reads: 4 × 10,000 tokens × 0.1x = 4,000
  Total: 16,500 token cost
  Savings: 67% cost reduction

WHEN TO USE CACHING:

DO use caching when:
✓ Content is large (>1024 tokens)
✓ Content repeats across multiple requests
✓ Content changes infrequently
✓ You have multi-turn conversations
✓ You're doing RAG/document Q&A
✓ You're analyzing the same image multiple times

DON'T use caching when:
✗ Content is small (<1024 tokens)
✗ Content is unique per request
✗ Content changes frequently
✗ Single-turn, one-off requests
✗ Content varies with each user query

"""
