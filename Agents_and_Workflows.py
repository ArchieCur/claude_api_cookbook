"""
AGENTS AND WORKFLOWS - Architectural Patterns for AI Systems

This template covers the fundamental architectural decisions for building
reliable AI systems with Claude: when to use workflows vs agents, and the
key patterns within each approach.

GOLDEN RULE: Generally use workflows and only resort to agents when truly required.

WORKFLOWS vs AGENTS:

┌─────────────────────────────────┬─────────────────────────────────┐
│ WORKFLOWS                       │ AGENTS                          │
├─────────────────────────────────┼─────────────────────────────────┤
│ Predefined series of calls      │ Given goal + tools, figures     │
│ Solve known problems            │ out how to complete task        │
│                                 │                                 │
│ BENEFITS:                       │ BENEFITS:                       │
│ • Higher accuracy               │ • More flexible UX              │
│ • Easier to test                │ • Combines tools creatively     │
│ • Predictable behavior          │ • Handles varied tasks          │
│                                 │                                 │
│ DOWNSIDES:                      │ DOWNSIDES:                      │
│ • Less flexible                 │ • Lower success rate            │
│ • Constrained UX                │ • Harder to test/evaluate       │
│                                 │                                 │
│ USE WHEN:                       │ USE WHEN:                       │
│ • Well-defined processes        │ • Unpredictable requests        │
│ • Know steps ahead of time      │ • Need creative problem-solving │
│ • Need reliability              │ • Can't predetermine steps      │
└─────────────────────────────────┴─────────────────────────────────┘

KEY INSIGHT:
"Workflows provide the reliability and predictability that most production
applications need, while agents offer flexibility for scenarios where the
exact requirements can't be predetermined."
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Callable
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-5"


# ============================================================================
# WORKFLOW PATTERN 1: PARALLELIZATION
# ============================================================================
"""
PARALLELIZATION WORKFLOW

Problem: Complex single prompts asking Claude to consider multiple criteria,
         compare options, or make cross-domain decisions are unreliable.

Solution: Split into focused sub-tasks, run in parallel, aggregate results.

Pattern:
    User Task
       ↓
    ┌──┴──┬──────┬──────┐
    ↓     ↓      ↓      ↓
  Sub1  Sub2   Sub3   Sub4  (Parallel execution)
    ↓     ↓      ↓      ↓
    └──┬──┴──────┴──────┘
       ↓
   Aggregator
       ↓
    Output

Benefits:
✓ Focused attention - Each sub-task handles one aspect
✓ Easier optimization - Tune each independently
✓ Better scalability - Add more evaluations as needed
✓ Improved reliability - Specialized prompts more consistent

When to Use:
- Complex decisions decomposable into independent evaluations
- Multiple criteria to evaluate
- Comparing several options
- Cross-domain decisions (security + performance + usability)

Key: Sub-tasks don't need to be identical - each can have specialized
     prompts, tools, or evaluation criteria.
"""


async def parallel_code_review(code: str) -> Dict[str, Any]:
    """
    Example: Parallel code review across multiple dimensions.

    Instead of one prompt asking for security, performance, and style,
    run three specialized reviews in parallel.
    """
    # Define specialized review prompts
    security_prompt = f"""
    Review this code for security issues only:
    - SQL injection vulnerabilities
    - XSS risks
    - Authentication/authorization issues
    - Insecure data handling

    Code:
    {code}

    Provide severity (high/medium/low) and specific issues found.
    """

    performance_prompt = f"""
    Review this code for performance issues only:
    - Inefficient algorithms
    - N+1 queries
    - Memory leaks
    - Unnecessary computations

    Code:
    {code}

    Provide severity and specific issues found.
    """

    style_prompt = f"""
    Review this code for style and maintainability only:
    - Code clarity and readability
    - Consistent naming conventions
    - Documentation quality
    - Best practices adherence

    Code:
    {code}

    Provide severity and specific issues found.
    """

    # Run reviews in parallel
    tasks = [
        client.messages.create(
            model=model, max_tokens=1024, messages=[{"role": "user", "content": p}]
        )
        for p in [security_prompt, performance_prompt, style_prompt]
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate results
    aggregated = {
        "security": results[0].content[0].text,
        "performance": results[1].content[0].text,
        "style": results[2].content[0].text,
    }

    # Optional: Use Claude to synthesize final recommendation
    synthesis_prompt = f"""
    Synthesize these parallel code reviews into a final recommendation:

    Security Review:
    {aggregated['security']}

    Performance Review:
    {aggregated['performance']}

    Style Review:
    {aggregated['style']}

    Provide:
    1. Overall severity (critical/high/medium/low)
    2. Top 3 priorities to address
    3. Approval recommendation (approve/request changes/reject)
    """

    final = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": synthesis_prompt}]
    )

    aggregated["final_recommendation"] = final.content[0].text
    return aggregated


# ============================================================================
# WORKFLOW PATTERN 2: CHAINING
# ============================================================================
"""
CHAINING WORKFLOW

Problem: Long prompts with many requirements cause Claude to ignore constraints.

Solution: Split large task into smaller sequential non-parallelizable subtasks.

Pattern:
    Input → Task 1 → Task 2 → Task 3 → Output

Key Characteristics:
- Sequential, dependent tasks (not parallel)
- Each task focuses on ONE aspect
- Can insert non-LLM processing between steps
- Output of one feeds into next

Benefits:
✓ Better constraint adherence (focused prompts)
✓ Easier debugging (inspect each step)
✓ Can validate/transform between steps
✓ More reliable results

When to Use:
- Complex tasks with multiple requirements
- Claude consistently ignores constraints in long prompts
- Need to process/validate outputs between steps
- Want focused, manageable interactions per step
"""


async def chained_content_generation(topic: str, style: str, length: int) -> str:
    """
    Example: Generate content through sequential refinement.

    Instead of one prompt: "Write article, add examples, check accuracy, format"
    Chain it: Draft → Add Examples → Fact Check → Format
    """
    # Step 1: Generate draft
    draft_prompt = f"""
    Write a {length}-word article about: {topic}

    Focus only on creating a clear, comprehensive draft.
    Don't worry about examples or formatting yet.
    """

    draft_response = await client.messages.create(
        model=model, max_tokens=2048, messages=[{"role": "user", "content": draft_prompt}]
    )
    draft = draft_response.content[0].text

    # Step 2: Add examples (depends on draft)
    examples_prompt = f"""
    Add 2-3 concrete examples to this article:

    {draft}

    Focus only on adding relevant, specific examples.
    Don't rewrite the entire article.
    """

    examples_response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": examples_prompt}],
    )
    with_examples = examples_response.content[0].text

    # Step 3: Fact check (depends on examples)
    factcheck_prompt = f"""
    Review this article for factual accuracy:

    {with_examples}

    Identify any claims that need correction or clarification.
    Provide a corrected version if needed.
    """

    factcheck_response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": factcheck_prompt}],
    )
    fact_checked = factcheck_response.content[0].text

    # Step 4: Format (depends on fact-checked content)
    format_prompt = f"""
    Format this article in {style} style:

    {fact_checked}

    Apply proper formatting, headings, and structure.
    Don't change the content, only the formatting.
    """

    final_response = await client.messages.create(
        model=model, max_tokens=2048, messages=[{"role": "user", "content": format_prompt}]
    )

    return final_response.content[0].text


# ============================================================================
# WORKFLOW PATTERN 3: ROUTING
# ============================================================================
"""
ROUTING WORKFLOW

Problem: Generic prompts that try to handle everything are less effective.

Solution: Route requests to specialized pipelines based on category.

Pattern:
         User Input
             ↓
          Router
             ↓
    ┌────┬───┴───┬────┐
    ↓    ↓       ↓    ↓
 Pipeline1 P2   P3   P4  (Specialized workflows)
    ↓    ↓       ↓    ↓
 Output

Key: User input sent to ONE pipeline, not broadcast to all.

Benefits:
- Each pipeline optimized for its category
- Better results than generic one-size-fits-all
- Specialized prompts and tools per category
- Easier to maintain individual pipelines

When to Use:
- Handling diverse request types
- Different categories need different tools/approaches
- Want to optimize for specific scenarios
- Generic prompts performing poorly
"""


async def route_customer_request(user_request: str) -> Dict[str, Any]:
    """
    Example: Route customer requests to specialized handlers.

    Categories: technical_support, billing, product_inquiry, general
    """
    # Step 1: Categorize the request
    router_prompt = f"""
    Categorize this customer request into ONE of these categories:
    - technical_support: Technical issues, bugs, how-to questions
    - billing: Payment, refunds, subscription questions
    - product_inquiry: Product features, comparisons, recommendations
    - general: Everything else

    Customer request: "{user_request}"

    Respond with ONLY the category name.
    """

    router_response = await client.messages.create(
        model=model, max_tokens=50, messages=[{"role": "user", "content": router_prompt}]
    )
    category = router_response.content[0].text.strip().lower()

    # Step 2: Route to specialized pipeline
    if category == "technical_support":
        return await handle_technical_support(user_request)
    elif category == "billing":
        return await handle_billing(user_request)
    elif category == "product_inquiry":
        return await handle_product_inquiry(user_request)
    else:
        return await handle_general(user_request)


async def handle_technical_support(request: str) -> Dict[str, Any]:
    """Specialized pipeline for technical support."""
    # This pipeline might have access to:
    # - Error logs
    # - Documentation
    # - Troubleshooting tools

    prompt = f"""
    You are a technical support specialist.

    Tools available:
    - Check error logs
    - Search documentation
    - Run diagnostics

    User issue: {request}

    Provide step-by-step troubleshooting guidance.
    """

    response = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )

    return {
        "category": "technical_support",
        "response": response.content[0].text,
        "tools_used": ["documentation", "diagnostics"],
    }


async def handle_billing(request: str) -> Dict[str, Any]:
    """Specialized pipeline for billing."""
    # This pipeline might have access to:
    # - Payment processor API
    # - Subscription database
    # - Refund policies

    prompt = f"""
    You are a billing specialist.

    Access to:
    - Payment history
    - Subscription details
    - Refund policies

    User question: {request}

    Provide clear information about billing, with specific policy references.
    """

    response = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )

    return {
        "category": "billing",
        "response": response.content[0].text,
        "requires_escalation": False,
    }


async def handle_product_inquiry(request: str) -> Dict[str, Any]:
    """Specialized pipeline for product inquiries."""
    prompt = f"""
    You are a product specialist.

    User question: {request}

    Provide detailed product information, comparisons, and recommendations.
    """

    response = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )

    return {"category": "product_inquiry", "response": response.content[0].text}


async def handle_general(request: str) -> Dict[str, Any]:
    """Fallback pipeline for general requests."""
    prompt = f"""
    You are a helpful assistant.

    User request: {request}

    Provide a helpful response.
    """

    response = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )

    return {"category": "general", "response": response.content[0].text}


# ============================================================================
# AGENT PATTERN: TOOL DESIGN PRINCIPLES
# ============================================================================
"""
AGENTS AND TOOL DESIGN

Key Principle: Tools should be ABSTRACT, not hyper-specialized.

"The real power of agents lies in their ability to combine simple tools
in unexpected ways."

✓ GOOD - Abstract Tools (Composable):
- Bash: Run commands
- Glob: Find files
- Grep: Search file contents
- Read: Read a file
- Write: Create a file
- Edit: Edit a file
- WebFetch: Fetch a URL

✗ BAD - Hyper-specialized Tools (Limited):
- Refactor: Too specific, can be done with Read + Edit
- RunTests: Too specific, can be done with Bash
- CreateMigration: Too specific, can be done with Read + Write
- InstallDependencies: Too specific, can be done with Bash

Why Abstract Tools Win:
- Composable in unexpected ways
- More flexible for varied tasks
- Fewer tools to maintain
- Claude can be creative with combinations

Example:
Instead of "RunTests" tool, Claude uses:
- Bash + "pytest" or "npm test" or "cargo test"

This flexibility allows Claude to adapt to different project structures,
test frameworks, and languages without needing specialized tools.
"""

# Example tool definitions (abstract, not hyper-specialized)
ABSTRACT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates file if it doesn't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command and return the output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a pattern in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py')",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ============================================================================
# AGENT PATTERN: ENVIRONMENT INSPECTION
# ============================================================================
"""
ENVIRONMENT INSPECTION - Critical for Reliable Agents

Core Principle: Claude MUST be able to observe the results of its actions.

"Provide tools and instructions that let Claude observe the results of
its actions."

Why This Matters:

1. Better Progress Tracking
   - Claude gauges how close it is to completing a task
   - Knows when to continue vs when to finish

2. Error Handling
   - Unexpected results detected
   - Claude can correct mistakes autonomously

3. Quality Assurance
   - Output verified before considering task complete
   - Catch issues before they compound

4. Adaptive Behavior
   - Claude adjusts approach based on observations
   - Learns from what works/doesn't work

Pattern: Action → Observation → Verification → Correction (if needed)

Examples:
- Files: Read contents before AND after modifications
- UI: Take screenshots after interactions
- APIs: Check responses for expected data
- Content: Validate output against requirements

Contrast:
❌ Bad: Write file, assume it worked
✓ Good: Write file → Read it back → Verify contents
"""


async def agent_with_verification(goal: str, tools: List[Dict]) -> str:
    """
    Example: Agent that verifies its work at each step.

    This pattern ensures Claude can observe and correct its actions.
    """
    system_prompt = """
    You are an autonomous agent that completes tasks using available tools.

    CRITICAL: After each action, verify the results before proceeding.

    Examples:
    - After writing a file, read it back to confirm contents
    - After executing a command, check the output for errors
    - After making changes, validate they meet requirements

    If verification fails, correct the issue before continuing.

    Only consider the task complete after verifying all steps succeeded.
    """

    messages = [{"role": "user", "content": goal}]

    max_iterations = 10
    for iteration in range(max_iterations):
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Agent completed or needs to use tools
        if response.stop_reason == "end_turn":
            # Extract final response
            final_text = next(
                (block.text for block in response.content if block.type == "text"), ""
            )
            return final_text

        elif response.stop_reason == "tool_use":
            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools (simplified - would need actual implementations)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Simulate tool execution
                    result = f"Tool {block.name} executed successfully"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

    return "Agent exceeded maximum iterations"


# ============================================================================
# DECISION FRAMEWORK: WORKFLOW VS AGENT
# ============================================================================
"""
WHEN TO USE WORKFLOWS VS AGENTS

Use this decision tree to choose the right approach:

START: Do you know the exact steps needed to complete the task?
│
├─ YES → Can you predict all variations of the task?
│   │
│   ├─ YES → Do you need high reliability and predictability?
│   │   │
│   │   ├─ YES → USE WORKFLOW
│   │   │        Consider: Chaining, Parallelization, or Routing
│   │   │
│   │   └─ NO  → Could use either
│   │            Choose based on UX flexibility needs
│   │
│   └─ NO  → Can you categorize requests into known types?
│       │
│       ├─ YES → USE ROUTING WORKFLOW
│       │        Route to specialized pipelines
│       │
│       └─ NO  → USE AGENT
│                Need flexibility for varied requests
│
└─ NO → Are requests unpredictable or highly varied?
    │
    ├─ YES → USE AGENT
    │        Claude figures out the approach
    │
    └─ NO  → Can you break it into sequential steps?
        │
        ├─ YES → USE CHAINING WORKFLOW
        │        Focus on one aspect at a time
        │
        └─ NO  → USE AGENT
                 Let Claude determine the strategy

GOLDEN RULE: Default to workflows. Use agents only when necessary.
"""


def recommend_architecture(
    task_description: str,
    knows_steps: bool,
    predictable_variations: bool,
    needs_high_reliability: bool,
    can_categorize: bool,
    unpredictable_requests: bool,
) -> Dict[str, Any]:
    """
    Decision helper for choosing architecture.

    Args:
        task_description: What you're trying to accomplish
        knows_steps: Do you know the exact steps?
        predictable_variations: Can you predict all variations?
        needs_high_reliability: Is reliability critical?
        can_categorize: Can requests be categorized into types?
        unpredictable_requests: Are requests highly varied?

    Returns:
        Recommendation with rationale
    """
    if knows_steps and predictable_variations and needs_high_reliability:
        return {
            "recommendation": "WORKFLOW (Chaining or Parallelization)",
            "rationale": "Known steps + predictable variations + need reliability",
            "pattern": "Chain sequential tasks or parallelize independent evaluations",
            "benefits": [
                "Higher accuracy",
                "Easier to test",
                "Predictable behavior",
                "Better for production",
            ],
            "example": "Code review: Parallel security, performance, style checks",
        }

    elif knows_steps and not predictable_variations and can_categorize:
        return {
            "recommendation": "ROUTING WORKFLOW",
            "rationale": "Known steps but variations exist, can categorize",
            "pattern": "Route to specialized pipelines per category",
            "benefits": [
                "Optimized per category",
                "Better than generic approach",
                "Maintainable pipelines",
            ],
            "example": "Customer support: Route to technical, billing, or product teams",
        }

    elif unpredictable_requests or not knows_steps:
        return {
            "recommendation": "AGENT",
            "rationale": "Unpredictable requests or unknown steps",
            "pattern": "Provide abstract tools, let Claude figure out approach",
            "benefits": [
                "Flexible UX",
                "Handles varied tasks",
                "Creative tool combinations",
            ],
            "tradeoffs": [
                "Lower success rate",
                "Harder to test",
                "More expensive",
            ],
            "example": "Claude Code: User asks to fix bug (unknown steps needed)",
        }

    else:
        return {
            "recommendation": "CHAINING WORKFLOW",
            "rationale": "Can break into sequential focused steps",
            "pattern": "Chain tasks, each focusing on one aspect",
            "benefits": [
                "Better constraint adherence",
                "Easier debugging",
                "Can validate between steps",
            ],
            "example": "Content generation: Draft → Examples → Fact Check → Format",
        }


# ============================================================================
# PATTERN SUMMARY
# ============================================================================
"""
QUICK REFERENCE GUIDE

================================================================================
WORKFLOWS
================================================================================

1. PARALLELIZATION
   Pattern: Task → [Sub1, Sub2, Sub3] (parallel) → Aggregator → Result
   Use when: Independent evaluations, multiple criteria, comparing options
   Example: Code review (security || performance || style)

2. CHAINING
   Pattern: Input → Task1 → Task2 → Task3 → Output (sequential)
   Use when: Complex requirements, sequential dependencies, need focused steps
   Example: Content generation (Draft → Examples → Fact Check → Format)

3. ROUTING
   Pattern: Input → Router → Specialized Pipeline → Output
   Use when: Diverse request types, category-specific optimization needed
   Example: Customer support (Technical / Billing / Product / General)

================================================================================
AGENTS
================================================================================

TOOL DESIGN:
- Abstract, composable tools (Read, Write, Execute, List)
- NOT hyper-specialized (Refactor, RunTests, InstallDeps)
- Let Claude combine tools creatively

ENVIRONMENT INSPECTION:
- Claude must observe results of actions
- Action → Observation → Verification → Correction
- Read file after writing, check command output, validate results

================================================================================
DECISION FRAMEWORK
================================================================================

Use WORKFLOWS when:
✓ Well-defined processes
✓ Know steps ahead of time
✓ Predictable variations
✓ Need reliability (production)
✓ Easier testing required

Use AGENTS when:
✓ Unpredictable requests
✓ Don't know exact steps
✓ Need creative problem-solving
✓ Flexible UX required
✓ Can't predetermine requirements

Golden Rule: Default to workflows, use agents when necessary.

================================================================================
PRODUCTION CONSIDERATIONS
================================================================================

WORKFLOWS:
- Easier to monitor (known steps)
- Better cost predictability (fixed API calls)
- Simpler error handling (catch at each step)
- More deterministic output
- Easier to explain to users

AGENTS:
- Higher API costs (exploratory behavior)
- Need robust timeout handling
- Require comprehensive logging
- Benefit from prompt caching
- Need clear success criteria

HYBRID APPROACH:
- Use routing workflow to classify
- Route simple requests to workflows
- Route complex/unpredictable to agents
- Best of both worlds
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Examples demonstrating when to use each pattern.
    """

    # Example 1: Parallelization for code review
    print("Example 1: Parallel Code Review")
    print("=" * 60)
    sample_code = """
    def login(username, password):
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        return db.execute(query)
    """
    # result = asyncio.run(parallel_code_review(sample_code))
    print("(Would run parallel security, performance, and style reviews)")
    print()

    # Example 2: Chaining for content generation
    print("\nExample 2: Chained Content Generation")
    print("=" * 60)
    # result = asyncio.run(chained_content_generation(
    #     topic="Prompt caching with Claude",
    #     style="technical blog",
    #     length=500
    # ))
    print("(Would chain: Draft → Examples → Fact Check → Format)")
    print()

    # Example 3: Routing customer requests
    print("\nExample 3: Routing Customer Requests")
    print("=" * 60)
    requests = [
        "My payment was charged twice",
        "How do I reset my password?",
        "What's the difference between Pro and Enterprise plans?",
    ]
    for req in requests:
        # result = asyncio.run(route_customer_request(req))
        print(f"Request: {req}")
        print("(Would route to appropriate specialized pipeline)")
        print()

    # Example 4: Architecture recommendation
    print("\nExample 4: Architecture Recommendation")
    print("=" * 60)

    recommendation = recommend_architecture(
        task_description="Code review system",
        knows_steps=True,
        predictable_variations=True,
        needs_high_reliability=True,
        can_categorize=True,
        unpredictable_requests=False,
    )

    print(f"Task: {recommendation['recommendation']}")
    print(f"Rationale: {recommendation['rationale']}")
    print(f"Pattern: {recommendation['pattern']}")
    print(f"Example: {recommendation['example']}")
    print()

    print("\nFor detailed implementations, refer to the pattern examples above.")
    print("Remember: Default to workflows, use agents when necessary!")
