"""
MULTI-AGENT COORDINATOR - Coordinating Specialized Agents at Scale

Multi-agent orchestration lets one coordinator agent delegate to a roster of
specialized sub-agents. Each agent runs in its own session thread with an
isolated context window, so each specialist stays focused. All agents share
the same container and filesystem, so files written by one are readable by
another.

THREE COORDINATION PATTERNS:

  PARALLELIZATION                SPECIALIZATION              ESCALATION
  Fan out independent            Route to domain-focused     Consult a more
  subtasks simultaneously        agents with targeted        capable model for
  and synthesize results.        system prompts + tools.     a hard sub-task.

  Coordinator                    Coordinator                 Coordinator
  ┌──┴──┬──────┐                 ├── Security Agent          ├── Sonnet workers
  ↓     ↓      ↓                 ├── Docs Agent              └── Opus advisor
 Sub1  Sub2   Sub3               └── Data Agent                  (hard cases)

THREADING MODEL:

  The coordinator reports on the primary thread (the session-level event
  stream). Sub-agent activity surfaces there as thread events:
    - session.thread_created       a new sub-agent thread opened
    - agent.thread_message_received  sub-agent produced output
  To drill into a sub-agent's full reasoning, stream its session thread by ID.

LIMITS:
  - Max 20 unique agents in a coordinator's roster
  - Max 25 concurrent threads per session
  - One level of coordination only — sub-agents cannot spawn sub-agents

BETA: Requires managed-agents-2026-04-01 header (SDK sets automatically).
"""

import os
import time
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Code sample the coordinator will review and test.
# In production this would come from a file, a PR diff, or a CI pipeline.
SAMPLE_CODE = '''
def calculate_discount(price, discount_pct, user_type="standard"):
    """Apply a discount to a price based on user type."""
    if user_type == "premium":
        discount_pct = discount_pct * 1.5

    discount = price * (discount_pct / 100)
    final_price = price - discount

    return final_price


def process_order(items, user_type="standard"):
    """Process a list of order items and return the total."""
    total = 0
    for item in items:
        price = item["price"]
        qty = item["quantity"]
        discount = item.get("discount_pct", 0)
        line_total = calculate_discount(price * qty, discount, user_type)
        total = total + line_total
    return total
'''


def create_agents():
    """Create the two specialists and the coordinator that delegates to them."""

    print("Creating specialist agents...")

    # Sub-agents are created first so their IDs can be referenced in the
    # coordinator's roster. Model choice: Sonnet for capable, cost-efficient
    # implementation work.
    reviewer = client.beta.agents.create(
        name="Code Reviewer",
        model="claude-sonnet-4-6",
        system=(
            "You are a senior code reviewer. When given code to review:\n"
            "1. Analyze for correctness, security issues, and edge cases\n"
            "2. Write your findings to /mnt/session/outputs/review.md\n"
            "3. Cite specific lines and explain why each issue matters\n"
            "4. Note what the code does well — not every review is all negatives"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"  Code Reviewer:   {reviewer.id}")

    test_writer = client.beta.agents.create(
        name="Test Writer",
        model="claude-sonnet-4-6",
        system=(
            "You are a test engineer who writes pytest suites. When given code:\n"
            "1. Write tests to /mnt/session/outputs/tests.py\n"
            "2. Cover happy paths, edge cases, and error conditions\n"
            "3. Use descriptive names: test_<what>_<condition>_<expected>\n"
            "4. Add a module docstring summarizing what the suite covers"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"  Test Writer:     {test_writer.id}")

    # The coordinator uses Opus for the synthesis and judgment work.
    # The agent_toolset gives it filesystem access to read sub-agent outputs.
    # The multiagent config declares the roster — this is what enables delegation.
    print("\nCreating coordinator...")
    coordinator = client.beta.agents.create(
        name="Engineering Lead",
        model="claude-opus-4-7",
        system=(
            "You are an engineering lead coordinating a code quality workflow.\n"
            "For each code review task:\n"
            "1. Delegate a code review to the Code Reviewer\n"
            "2. Delegate test writing to the Test Writer\n"
            "3. Kick both off in parallel — do not wait for one before starting the other\n"
            "4. Once both have written their outputs, read them and write a synthesis\n"
            "   to /mnt/session/outputs/summary.md covering: top issues, test approach,\n"
            "   and your recommended next steps for the developer"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
        multiagent={
            "type": "coordinator",
            "agents": [
                {"type": "agent", "id": reviewer.id},
                {"type": "agent", "id": test_writer.id},
            ],
        },
    )
    print(f"  Engineering Lead (coordinator): {coordinator.id}")

    return coordinator, reviewer, test_writer


def create_environment():
    """Create a cloud sandbox for the session to run in."""
    print("\nCreating environment...")
    environment = client.beta.environments.create(
        name="code-review-sandbox",
        config={
            "type": "cloud",
            # unrestricted lets agents use web_search if needed for docs lookups.
            # Use "limited" with an allowed_hosts list for tighter isolation.
            "networking": {"type": "unrestricted"},
        },
    )
    print(f"  Environment: {environment.id}")
    return environment


def run_review(coordinator, environment):
    """Start a session and send the code review task to the coordinator."""

    print("\nCreating session...")
    session = client.beta.sessions.create(
        agent=coordinator.id,
        environment_id=environment.id,
        title="Code Review: discount + order processing module",
    )
    print(f"  Session: {session.id}")

    print("\nSending work to coordinator...")
    client.beta.sessions.events.send(
        session.id,
        events=[{
            "type": "user.message",
            "content": (
                "Please coordinate a full code quality pass on this module.\n"
                "Run the review and the test writing in parallel, then synthesize.\n\n"
                f"```python\n{SAMPLE_CODE}\n```"
            ),
        }],
    )

    return session


def poll_until_idle(session_id, timeout=300, poll_interval=15):
    """
    Poll the session until it idles (work complete) or terminates (error).

    Alternative: subscribe to the SSE event stream via
    client.beta.sessions.events.stream(session_id) for real-time updates
    without polling. Useful when you need to react as sub-agents report in
    rather than waiting for the full job to finish.
    """
    print(f"\nWaiting for agents (polling every {poll_interval}s, timeout {timeout}s)...")
    elapsed = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        session = client.beta.sessions.retrieve(session_id)
        status = session.status
        print(f"  [{elapsed:>3}s] status={status}")

        if status == "idle":
            print("  Coordinator finished — session is idle.")
            return True
        elif status == "terminated":
            print("  Session terminated with a terminal error.")
            return False

    print(f"  Timed out after {timeout}s.")
    return False


def retrieve_outputs(session_id):
    """Fetch and display output files the agents wrote to /mnt/session/outputs/."""
    print("\nRetrieving output files from session filesystem...")
    files = client.beta.files.list(scope_id=session_id)

    if not files.data:
        print("  No output files found.")
        return

    for f in files.data:
        print(f"\n{'═' * 64}")
        print(f"  {f.filename}  ({f.id})")
        print(f"{'═' * 64}")
        content = client.beta.files.download(f.id)
        # Files are plain text (markdown, Python). For binary outputs,
        # use content.write_to_file("/local/path") instead.
        print(content.text)


# ============================================================================
# SETUP
# ============================================================================

def main():
    print("=== Multi-Agent Coordinator Demo ===\n")

    coordinator, reviewer, test_writer = create_agents()
    environment = create_environment()

    # ============================================================================
    # DEMO
    # ============================================================================

    session = run_review(coordinator, environment)
    success = poll_until_idle(session.id)

    if success:
        retrieve_outputs(session.id)

    # ============================================================================
    # CLEANUP
    # ============================================================================

    print("\nCleaning up resources...")
    # Archiving makes agents read-only. Existing sessions continue unaffected.
    # In production, skip this — you want to reuse these agents across sessions.
    client.beta.agents.archive(reviewer.id)
    client.beta.agents.archive(test_writer.id)
    client.beta.agents.archive(coordinator.id)
    client.beta.environments.archive(environment.id)
    print("  Agents and environment archived.")


if __name__ == "__main__":
    main()
