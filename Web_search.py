from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message

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
# Web Search Tool Schemas
# ============================================

def get_web_search_schema(
    max_uses=5,
    allowed_domains=None,
    blocked_domains=None
):
    """Get the web search tool schema with optional domain filtering.

    IMPORTANT: Web search must be enabled in your organization settings:
    https://console.anthropic.com/settings/privacy

    The web search tool is executed by Anthropic's servers - you don't need
    to implement the search functionality yourself. Claude will automatically
    perform web searches and receive the results.

    Args:
        max_uses: Maximum number of searches Claude can perform (default: 5)
        allowed_domains: List of domains to restrict searches to (whitelist)
        blocked_domains: List of domains to exclude from searches (blacklist)

    Returns:
        Tool schema configuration for web search

    Note: You cannot use both allowed_domains and blocked_domains simultaneously.
    """
    schema = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }

    if allowed_domains:
        schema["allowed_domains"] = allowed_domains
    elif blocked_domains:
        schema["blocked_domains"] = blocked_domains

    return schema


# ============================================
# Pre-configured Web Search Schemas
# ============================================

# Unrestricted web search (any domain)
unrestricted_web_search = get_web_search_schema(max_uses=5)

# Academic/Research focused search
academic_web_search = get_web_search_schema(
    max_uses=5,
    allowed_domains=["nih.gov", "arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov"]
)

# News-focused search
news_web_search = get_web_search_schema(
    max_uses=5,
    allowed_domains=["reuters.com", "apnews.com", "bbc.com", "npr.org"]
)

# Technical documentation search
tech_docs_web_search = get_web_search_schema(
    max_uses=5,
    allowed_domains=["docs.python.org", "developer.mozilla.org", "stackoverflow.com", "github.com"]
)

# Search with blocked social media domains
no_social_media_search = get_web_search_schema(
    max_uses=5,
    blocked_domains=["facebook.com", "twitter.com", "instagram.com", "tiktok.com"]
)


# ============================================
# Web Search Result Processing
# ============================================

def render_search_results(message):
    """Extract and display web search results from Claude's response.

    Web search results are included in the message content blocks.
    This function helps visualize what Claude found.
    """
    print("\n" + "=" * 60)
    print("SEARCH RESULTS ANALYSIS")
    print("=" * 60)

    # Check for tool use blocks (Claude requesting search)
    tool_uses = [block for block in message.content if block.type == "tool_use"]
    if tool_uses:
        print("\n[Claude performed web searches]")
        for tool_use in tool_uses:
            if tool_use.name == "web_search":
                print(f"  Query: {tool_use.input.get('query', 'N/A')}")

    # Extract text response
    text_response = text_from_message(message)
    if text_response:
        print("\n[Claude's Response]")
        print(text_response)

    print("\n" + "=" * 60)


def chat_with_web_search(user_message, web_search_schema):
    """Run a conversation with web search capability.

    Args:
        user_message: The user's query
        web_search_schema: Web search configuration to use

    Returns:
        The final message from Claude
    """
    messages = []
    add_user_message(messages, user_message)

    response = chat(messages, tools=[web_search_schema])

    # Display results
    render_search_results(response)

    return response


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Web Search Tool Examples")
    print("=" * 60)
    print("""
IMPORTANT: Before running these examples, ensure web search is enabled
in your organization settings at:
https://console.anthropic.com/settings/privacy

If web search is not enabled, these examples will fail.
""")

    print("\n" + "=" * 60)
    print("Example 1: Academic Research Search")
    print("=" * 60)
    print("Searching only academic/government health sources...")

    """
    # Uncomment to run
    response = chat_with_web_search(
        "What's the best exercise for gaining leg muscle according to recent research?",
        academic_web_search
    )
    """

    print("\n" + "=" * 60)
    print("Example 2: News Search")
    print("=" * 60)
    print("Searching only reputable news sources...")

    """
    # Uncomment to run
    response = chat_with_web_search(
        "What are the latest developments in artificial intelligence?",
        news_web_search
    )
    """

    print("\n" + "=" * 60)
    print("Example 3: Technical Documentation Search")
    print("=" * 60)
    print("Searching only technical documentation sites...")

    """
    # Uncomment to run
    response = chat_with_web_search(
        "How do I use asyncio in Python?",
        tech_docs_web_search
    )
    """

    print("\n" + "=" * 60)
    print("Example 4: Unrestricted Search")
    print("=" * 60)
    print("Searching across all domains...")

    """
    # Uncomment to run
    response = chat_with_web_search(
        "What are the best restaurants in San Francisco?",
        unrestricted_web_search
    )
    """

    print("\n" + "=" * 60)
    print("Example 5: Custom Domain Filtering")
    print("=" * 60)
    print("Creating a custom search limited to specific domains...")

    """
    # Uncomment to run
    custom_search = get_web_search_schema(
        max_uses=3,
        allowed_domains=["wikipedia.org", "britannica.com"]
    )

    response = chat_with_web_search(
        "Tell me about the history of the Roman Empire",
        custom_search
    )
    """

    print("\n" + "=" * 60)
    print("Example 6: Blocking Specific Domains")
    print("=" * 60)
    print("Searching while excluding certain domains...")

    """
    # Uncomment to run
    response = chat_with_web_search(
        "What are people saying about the latest iPhone?",
        no_social_media_search
    )
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Web Search Tool Pattern:
1. Built-in tool executed by Anthropic - no implementation needed
2. Must be enabled in organization settings first
3. Configure with type "web_search_20250305"
4. Optional domain filtering (whitelist OR blacklist, not both)
5. Rate limiting with max_uses parameter

Web Search Configuration:
- type: "web_search_20250305" (required)
- name: "web_search" (required)
- max_uses: Maximum searches per conversation (default: 5)
- allowed_domains: Whitelist of allowed domains (optional)
- blocked_domains: Blacklist of blocked domains (optional)

Domain Filtering Rules:
- Use allowed_domains for restrictive whitelisting
- Use blocked_domains for permissive blacklisting
- Cannot use both simultaneously
- Domains should be base domains (e.g., "nih.gov" not "www.nih.gov")

When to Use Web Search:
- Current events and recent information
- Research requiring up-to-date sources
- Fact-checking and verification
- Finding specific resources or documentation
- Any query requiring real-time information

Best Practices:
- Set appropriate max_uses to control API costs
- Use domain filtering for quality control
- Academic searches: Use .gov, .edu, research domains
- News searches: Use reputable news outlets
- Technical searches: Use official documentation sites
- Always verify sources in production applications

Differences from Other Tools:
- Text Editor: You implement, Anthropic provides schema type
- Web Search: Anthropic implements AND provides schema type
- Custom Tools: You implement AND define schema

Setup Requirements:
1. Enable web search in Anthropic Console settings
2. Organization admin must enable the feature
3. Check privacy settings at: https://console.anthropic.com/settings/privacy
4. Once enabled, web search works across all API calls with the tool

Common Use Cases:
- Research assistants with trusted sources
- News aggregation with quality filtering
- Technical support with documentation search
- Educational apps with academic source restriction
- Content moderation avoiding social media
""")
