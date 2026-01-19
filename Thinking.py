from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message

# Load env variables and create client
client = Anthropic()
model = "claude-sonnet-4-5"  # Extended thinking requires Sonnet 4+


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


def chat(
    messages,
    system=None,
    temperature=1.0,
    stop_sequences=[],
    tools=None,
    thinking=False,
    thinking_budget=1024,
):
    """Make an API call to Claude with optional extended thinking.

    Args:
        messages: Conversation history
        system: System prompt
        temperature: Sampling temperature (0-1)
        stop_sequences: Sequences that stop generation
        tools: Tool definitions
        thinking: Enable extended thinking mode
        thinking_budget: Token budget for thinking (max tokens Claude can use for reasoning)

    Returns:
        Message object with response (may include RedactedThinkingBlock)
    """
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if thinking:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    """Extract all text content from a message, skipping thinking blocks."""
    return "\n".join([block.text for block in message.content if block.type == "text"])


def has_thinking(message):
    """Check if a message contains a thinking block.

    Args:
        message: Message object from Claude

    Returns:
        True if message includes redacted thinking
    """
    return any(block.type == "redacted_thinking" for block in message.content)


def get_thinking_block(message):
    """Extract the thinking block from a message.

    Note: The thinking content is redacted/encoded in production API responses.
    You cannot read the actual thinking text.

    Args:
        message: Message object from Claude

    Returns:
        RedactedThinkingBlock if present, None otherwise
    """
    for block in message.content:
        if block.type == "redacted_thinking":
            return block
    return None


def get_token_usage(message):
    """Extract token usage information from a message.

    Extended thinking uses additional input tokens during the thinking phase.

    Args:
        message: Message object from Claude

    Returns:
        Dictionary with token usage breakdown
    """
    usage = message.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }


# ============================================
# Extended Thinking Test
# ============================================

# Magic string to verify extended thinking is working
# This is a special test string recognized by Claude
THINKING_TEST_STRING = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"


def test_thinking_enabled():
    """Test if extended thinking is working properly.

    Uses a special trigger string that causes Claude to return
    a visible RedactedThinkingBlock.

    Returns:
        True if thinking is enabled and working
    """
    messages = []
    add_user_message(messages, THINKING_TEST_STRING)

    response = chat(messages, thinking=True, thinking_budget=1024)

    return has_thinking(response)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Extended Thinking Examples")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Example 1: Testing Extended Thinking")
    print("=" * 60)

    if test_thinking_enabled():
        print("✓ Extended thinking is enabled and working!")
    else:
        print("✗ Extended thinking is not working. Check model version (needs Sonnet 4+)")

    print("\n" + "=" * 60)
    print("Example 2: Complex Math Problem (With Thinking)")
    print("=" * 60)

    """
    # Uncomment to run
    messages = []
    add_user_message(
        messages,
        "If a train travels 120 miles in 2 hours, then speeds up and travels 200 miles in the next 2.5 hours, what was its average speed for the entire journey?"
    )

    # With extended thinking
    response = chat(messages, thinking=True, thinking_budget=2048)

    print("Claude's Response:")
    print(text_from_message(response))
    print("\n" + "-" * 60)

    if has_thinking(response):
        print("✓ Claude used extended thinking for this response")

    usage = get_token_usage(response)
    print(f"\nToken Usage:")
    print(f"  Input: {usage['input_tokens']}")
    print(f"  Output: {usage['output_tokens']}")
    print(f"  Total: {usage['total_tokens']}")
    """

    print("\n" + "=" * 60)
    print("Example 3: Comparison - With vs Without Thinking")
    print("=" * 60)

    """
    # Uncomment to run
    complex_problem = \"\"\"
    A farmer has 100 feet of fencing and wants to build a rectangular garden.
    One side of the garden will be against a barn (no fence needed there).
    What dimensions will give the maximum area?
    \"\"\"

    # WITHOUT thinking
    messages1 = []
    add_user_message(messages1, complex_problem)
    response1 = chat(messages1, thinking=False)

    print("WITHOUT Extended Thinking:")
    print(text_from_message(response1))
    print(f"Tokens used: {get_token_usage(response1)['total_tokens']}")

    # WITH thinking
    messages2 = []
    add_user_message(messages2, complex_problem)
    response2 = chat(messages2, thinking=True, thinking_budget=3000)

    print("\n" + "-" * 60)
    print("WITH Extended Thinking:")
    print(text_from_message(response2))
    print(f"Tokens used: {get_token_usage(response2)['total_tokens']}")
    print(f"Used thinking: {has_thinking(response2)}")
    """

    print("\n" + "=" * 60)
    print("Example 4: Different Thinking Budgets")
    print("=" * 60)

    """
    # Uncomment to run
    problem = "Explain the proof of the Pythagorean theorem using similar triangles."

    budgets = [512, 1024, 2048, 4096]

    for budget in budgets:
        messages = []
        add_user_message(messages, problem)
        response = chat(messages, thinking=True, thinking_budget=budget)

        print(f"\nBudget: {budget} tokens")
        print(f"Response length: {len(text_from_message(response))} chars")
        print(f"Total tokens: {get_token_usage(response)['total_tokens']}")
        print(f"Used thinking: {has_thinking(response)}")
        print("-" * 40)
    """

    print("\n" + "=" * 60)
    print("Example 5: Extended Thinking with Tools")
    print("=" * 60)

    """
    # Uncomment to run
    from anthropic.types import ToolParam

    calculator_tool = ToolParam({
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    })

    messages = []
    add_user_message(
        messages,
        "I have $1000 to invest. If I earn 7% annual interest compounded quarterly for 5 years, how much will I have?"
    )

    response = chat(
        messages,
        thinking=True,
        thinking_budget=2048,
        tools=[calculator_tool]
    )

    print("Response with thinking + tools:")
    print(text_from_message(response))
    print(f"\nUsed thinking: {has_thinking(response)}")
    print(f"Stop reason: {response.stop_reason}")
    """

    print("\n" + "=" * 60)
    print("Example 6: Multi-Turn Conversation with Thinking")
    print("=" * 60)

    """
    # Uncomment to run
    messages = []

    # Turn 1: Complex question
    add_user_message(
        messages,
        "I'm planning a road trip from New York to Los Angeles (about 2,800 miles). My car gets 28 mpg, gas costs $3.50/gallon, and I want to drive no more than 8 hours per day at an average speed of 60 mph. How many days will it take and what will the gas cost?"
    )

    response1 = chat(messages, thinking=True, thinking_budget=3000)
    add_assistant_message(messages, response1)

    print("Turn 1 Response:")
    print(text_from_message(response1))
    print(f"Used thinking: {has_thinking(response1)}")

    # Turn 2: Follow-up
    add_user_message(
        messages,
        "If I increase my daily driving to 10 hours, how much time and money would I save?"
    )

    response2 = chat(messages, thinking=True, thinking_budget=2000)
    add_assistant_message(messages, response2)

    print("\n" + "-" * 60)
    print("Turn 2 Response:")
    print(text_from_message(response2))
    print(f"Used thinking: {has_thinking(response2)}")
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Extended Thinking Overview:
- Allows Claude to engage in deeper reasoning before responding
- The thinking process is internal and redacted in API responses
- Requires Sonnet 4 or later models
- Controlled by a token budget parameter

Configuration:
  thinking = {
      "type": "enabled",
      "budget_tokens": 1024  # Max tokens for internal reasoning
  }

Token Budget Guidelines:
- 512 tokens: Quick reasoning tasks
- 1024 tokens: Standard complexity (DEFAULT)
- 2048 tokens: Complex problems requiring multiple steps
- 4096 tokens: Very complex reasoning, proofs, or multi-step logic
- 8192+ tokens: Extremely complex mathematical or logical problems

When to Use Extended Thinking:

✓ USE IT FOR:
- Complex mathematical problems
- Multi-step logical reasoning
- Planning and strategy tasks
- Proofs and derivations
- Problems requiring careful analysis
- Situations where accuracy > speed

✗ DON'T USE IT FOR:
- Simple factual questions
- Creative writing
- Basic conversation
- Quick lookups
- Tasks where speed matters more than depth

Performance Characteristics:

Response Time:
- WITHOUT thinking: 1-3 seconds typical
- WITH thinking: 2-5+ seconds (depends on budget)
- Longer budget = potentially longer response time

Token Cost:
- Thinking tokens count toward input tokens
- Larger budgets may use more tokens even if not fully consumed
- Cost = (input + thinking) + output tokens

Quality Trade-offs:
- Better reasoning and accuracy with thinking
- More consistent step-by-step problem solving
- Reduced hallucinations on complex topics
- Worth the latency for critical tasks

Integration Tips:

1. Start with 1024 token budget, adjust based on results
2. Monitor token usage to optimize budgets
3. Use thinking selectively (not for every request)
4. Combine with tools for best results on complex tasks
5. Test with and without thinking to measure impact

Production Considerations:

Cost Optimization:
- Use thinking only when complexity justifies it
- Set appropriate budgets (don't over-allocate)
- Monitor actual usage vs budget allocation

User Experience:
- Add loading indicators for thinking-enabled requests
- Consider async processing for very high budgets
- Provide feedback when thinking is in use

Debugging:
- Use has_thinking() to verify thinking was used
- Monitor token usage for budget tuning
- Test with THINKING_TEST_STRING to verify setup

Common Patterns:

Pattern 1: Adaptive Thinking
def solve_problem(problem, complexity="medium"):
    budgets = {"simple": 512, "medium": 1024, "complex": 2048}
    budget = budgets.get(complexity, 1024)

    messages = [{"role": "user", "content": problem}]
    return chat(messages, thinking=True, thinking_budget=budget)

Pattern 2: Thinking + Tools
# Extended thinking helps Claude reason about when and how to use tools
response = chat(
    messages,
    thinking=True,
    thinking_budget=2048,
    tools=available_tools
)

Pattern 3: Conditional Thinking
def chat_with_optional_thinking(message, use_thinking=False):
    messages = [{"role": "user", "content": message}]

    if use_thinking:
        return chat(messages, thinking=True, thinking_budget=1024)
    else:
        return chat(messages, thinking=False)

Best Practices:

1. Budget Allocation:
   - Start conservative, increase if needed
   - Monitor actual usage to right-size budgets
   - Higher budget ≠ always better results

2. Use Cases:
   - Evaluate if thinking adds value for your use case
   - A/B test responses with and without thinking
   - Measure accuracy vs latency trade-off

3. Error Handling:
   - Thinking requires Sonnet 4+, fails gracefully on older models
   - Check model compatibility before enabling
   - Provide fallback without thinking if needed

4. Monitoring:
   - Track thinking usage frequency
   - Monitor token consumption patterns
   - Analyze quality improvements

Technical Notes:

Thinking Block Structure:
- Type: "redacted_thinking"
- Content is encoded/redacted (not readable)
- Indicates thinking occurred, not what was thought
- Use has_thinking() to detect presence

Model Requirements:
- Claude Sonnet 4 or later
- Not available on Opus 3.5 or earlier models
- Feature availability may vary by region

API Compatibility:
- Works with streaming (thinking happens first, then streams response)
- Compatible with tools, system prompts, and all other features
- Can be used in multi-turn conversations

Future Considerations:
- Thinking visibility may evolve in future API versions
- Budget optimization strategies may improve over time
- Monitor Anthropic docs for updates to this feature
""")
