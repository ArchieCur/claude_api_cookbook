"""
OUTCOMES WITH RUBRIC - Goal-Directed Sessions with Measurable Quality

An outcome elevates a session from conversation to work. You define what
"done" looks like with a rubric, and the agent iterates until it gets there.

HOW IT WORKS:

  You ──▶ user.define_outcome (description + rubric)
              │
              ▼
         Agent works
              │
              ▼
         Grader evaluates artifact against rubric
         (separate context window — isolated from agent's implementation choices)
              │
         ┌────┴──────────────────────────┐
         │                               │
    satisfied                     needs_revision
    (session idles)                      │
                                    Agent revises
                                         │
                                   Grader re-evaluates
                                         │
                              max_iterations_reached
                              (agent may do one final pass, then idles)

OUTCOME RESULTS:
  satisfied             All rubric criteria met — session idles
  needs_revision        Agent starts another iteration
  max_iterations_reached  No further grader cycles; agent may do a final pass
  failed                Rubric fundamentally does not match the task

WRITING GOOD RUBRICS:
  The grader scores each criterion independently, so vague criteria produce
  noisy evaluations. Write explicit, testable statements:

    GOOD:  "Raises a descriptive ValueError if a required column is missing"
    BAD:   "Handles errors appropriately"

    GOOD:  "Output JSON is parseable by json.loads() without error"
    BAD:   "Produces valid JSON"

  If you do not have a rubric on hand: give Claude a known-good example and
  ask it to analyze what makes that artifact good. Turn that analysis into
  explicit criteria. This often produces better rubrics than writing from scratch.

BETA: Requires managed-agents-2026-04-01 header (SDK sets automatically).
"""

import os
import time
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Terminal states: the grader will not run again after any of these.
# "needs_revision" is the only non-terminal state — the agent keeps working.
TERMINAL_RESULTS = {"satisfied", "max_iterations_reached", "failed", "interrupted"}

# ============================================================================
# RUBRIC
# Each criterion is a testable statement the grader checks independently.
# The more specific each line, the more reliable the grader's scoring.
# ============================================================================

RUBRIC = """
# CSV-to-JSON Pipeline Rubric

## Correctness
- Reads from the CSV file path provided as the first command-line argument
- Writes valid JSON to the path provided as the second command-line argument
- Handles missing / empty cell values without crashing — output null in JSON

## Code Quality
- Uses context managers (with statements) for all file I/O
- Includes a __main__ guard (if __name__ == "__main__":)
- Has a module-level docstring describing what the script does
- Type hints are present on all function signatures

## Error Handling
- Raises a descriptive ValueError if a required column is missing
- Prints a usage message and exits cleanly if called with wrong argument count
- FileNotFoundError from a bad input path is caught and reported with a clear message

## Output Quality
- Numeric columns are written as JSON numbers, not strings
- The output JSON file is valid and parseable by json.loads() without modification
- Running the script end-to-end with a sample CSV produces a correct JSON file
"""


# ============================================================================
# SETUP
# ============================================================================

def create_agent():
    """
    Create an agent configured to write and test Python scripts.
    The agent toolset gives it bash (to run its own script), read, write, and glob.
    """
    print("Creating agent...")
    agent = client.beta.agents.create(
        name="Pipeline Engineer",
        model="claude-sonnet-4-6",
        system=(
            "You are a Python engineer who writes clean, production-quality scripts.\n"
            "When given a coding task with a rubric:\n"
            "1. Write the script to /mnt/session/outputs/pipeline.py\n"
            "2. Create a small sample CSV at /mnt/session/outputs/sample.csv to test with\n"
            "3. Run your script against the sample CSV to verify it works end-to-end\n"
            "4. Fix any issues before considering the task complete\n"
            "5. If the grader returns feedback, address every cited gap before resubmitting"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"  Agent: {agent.id}")
    return agent


def create_environment():
    """Create a cloud sandbox with package manager access for pip installs."""
    print("Creating environment...")
    environment = client.beta.environments.create(
        name="pipeline-demo",
        config={
            "type": "cloud",
            "networking": {
                "type": "limited",
                # Allow pip so the agent can install pandas or other libraries
                # if it decides to use them. Deny arbitrary outbound traffic.
                "allow_package_managers": True,
                "allow_mcp_servers": False,
                "allowed_hosts": [],
            },
        },
    )
    print(f"  Environment: {environment.id}")
    return environment


# ============================================================================
# DEMO
# ============================================================================

def run_outcome_session(agent, environment):
    """
    Create a session and immediately define an outcome.
    The agent starts working as soon as it receives the define_outcome event —
    no additional user.message is required.
    """
    print("\nCreating session...")
    session = client.beta.sessions.create(
        agent=agent.id,
        environment_id=environment.id,
        title="Build CSV-to-JSON pipeline",
    )
    print(f"  Session: {session.id}")

    print("\nDefining outcome...")
    client.beta.sessions.events.send(
        session.id,
        events=[{
            "type": "user.define_outcome",
            "description": (
                "Write a Python script pipeline.py that converts any CSV file to JSON. "
                "Accept the input CSV path and output JSON path as command-line arguments. "
                "Create a sample CSV, run the script against it, and confirm the output is correct."
            ),
            # Pass the rubric as inline text. For rubrics you reuse across many sessions,
            # upload once via the Files API and pass {"type": "file", "file_id": rubric_id}
            # (requires the files-api-2025-04-14 beta header).
            "rubric": {
                "type": "text",
                "content": RUBRIC,
            },
            # How many grader ↔ agent revision cycles before the outcome resolves.
            # Default 3, max 20. Raise for tasks where quality matters more than speed.
            "max_iterations": 4,
        }],
    )
    print("  Outcome defined — agent is working...")
    return session


def poll_outcome(session_id, timeout=600, poll_interval=20):
    """
    Poll session.outcome_evaluations for grader results.

    The evaluations array grows by one entry per grader cycle. Each entry shows
    the iteration number (0-indexed) and the result. We track which iterations
    we have already logged so we can print new ones as they appear.

    Alternative: listen on the SSE event stream for span.outcome_evaluation_end
    events to get real-time grader feedback without polling.
    """
    print(f"\nPolling for grader results (every {poll_interval}s, timeout {timeout}s)...")
    seen_iterations = set()
    elapsed = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        session = client.beta.sessions.retrieve(session_id)
        evaluations = session.outcome_evaluations

        for ev in evaluations:
            key = ev.iteration
            if key not in seen_iterations:
                seen_iterations.add(key)
                _print_evaluation(ev)

                if ev.result in TERMINAL_RESULTS:
                    return ev

        # Fall back to session status in case evaluations are not yet populated
        if session.status == "idle" and evaluations:
            return evaluations[-1]

        print(f"  [{elapsed:>3}s] waiting... ({len(evaluations)} grader cycle(s) so far)")

    print(f"  Timed out after {timeout}s without a terminal result.")
    return None


def _print_evaluation(ev):
    """Pretty-print a single grader evaluation."""
    result_label = {
        "satisfied": "SATISFIED",
        "needs_revision": "NEEDS REVISION",
        "max_iterations_reached": "MAX ITERATIONS REACHED",
        "failed": "FAILED",
    }.get(ev.result, ev.result.upper())

    print(f"\n{'─' * 64}")
    print(f"  Grader — Iteration {ev.iteration}: {result_label}")
    print(f"{'─' * 64}")
    if ev.explanation:
        # Show the full grader explanation — this is what drives the next revision
        for line in ev.explanation.splitlines():
            print(f"  {line}")


def retrieve_deliverables(session_id):
    """
    Fetch output files the agent wrote to /mnt/session/outputs/.
    The Files API is scoped to the session — only files from this run appear.
    """
    print("\nRetrieving deliverables from session filesystem...")
    files = client.beta.files.list(scope_id=session_id)

    if not files.data:
        print("  No output files found.")
        return

    print(f"  Found {len(files.data)} file(s):\n")
    for f in files.data:
        print(f"{'═' * 64}")
        print(f"  {f.filename}  ({f.id})")
        print(f"{'═' * 64}")
        content = client.beta.files.download(f.id)
        # For text files (Python, JSON, CSV). Use content.write_to_file(path)
        # for binary outputs like .xlsx or images.
        print(content.text)
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== Outcomes with Rubric Demo ===\n")

    agent = create_agent()
    environment = create_environment()

    session = run_outcome_session(agent, environment)
    final_eval = poll_outcome(session.id)

    if final_eval:
        print(f"\nFinal outcome: {final_eval.result}")
        if final_eval.result == "satisfied":
            retrieve_deliverables(session.id)
        else:
            print(
                "  The agent did not fully satisfy the rubric. "
                "You may send a user.message to continue the session conversationally, "
                "or send a new user.define_outcome to kick off a fresh outcome cycle "
                "with revised instructions."
            )

    # ============================================================================
    # CLEANUP
    # ============================================================================

    print("\nCleaning up resources...")
    # After a satisfied outcome, the session retains its history and files.
    # You can continue it as a conversational session or chain a new outcome.
    client.beta.agents.archive(agent.id)
    client.beta.environments.archive(environment.id)
    print("  Agent and environment archived.")


if __name__ == "__main__":
    main()
