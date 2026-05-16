"""
ADVISOR STRATEGY - Tiered Model Routing for Cost-Efficient Multi-Agent Systems

Multi-agent coordination (see 01_multi_agent_coordinator.py) answers HOW to
split work across agents. The Advisor Strategy answers WHICH MODEL to give
each agent based on the nature of its task.

The core insight: Opus-level reasoning changes outcomes for judgment and
synthesis work. It does not change outcomes for mechanical tasks — it just
costs more. Route deliberately.

THREE-TIER MODEL:

  ┌───────────────────────────────────────────────────────────────────┐
  │  Tier 1 — Opus Advisor (Coordinator)                             │
  │  claude-opus-4-7                                                  │
  │                                                                   │
  │  Plans. Delegates. Synthesizes. Does NOT do mechanical work.      │
  │  Reserves its reasoning for judgment calls, architectural         │
  │  decisions, and synthesis of conflicting sub-agent outputs.       │
  └────────────────────────────┬──────────────────────────────────────┘
                               │ delegates to:
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌───────────────┐   ┌────────────────┐   ┌─────────────────────┐
  │  Tier 2       │   │  Tier 2        │   │  Tier 3             │
  │  Sonnet       │   │  Sonnet        │   │  Haiku              │
  │  Analyst      │   │  Doc Writer    │   │  Style Checker      │
  │               │   │                │   │                     │
  │  Security +   │   │  Docstrings,   │   │  Naming, PEP 8,    │
  │  correctness  │   │  usage examples│   │  formatting checks  │
  │  review       │   │  API reference │   │  (mechanical)       │
  └───────────────┘   └────────────────┘   └─────────────────────┘

ROUTING GUIDE — pick the right tier for each task:

  HAIKU  "Could a checklist or regex do this?"
    ✓  Style and formatting checks          ✓  Short content reformatting
    ✓  Structured data extraction           ✗  Anything requiring judgment

  SONNET  "Does this require real reasoning and judgment?"
    ✓  Code review, analysis, debugging     ✓  Writing docs, reports, prose
    ✓  Most implementation work             ✗  Mechanical tasks (use Haiku)
                                            ✗  High-stakes synthesis (use Opus)

  OPUS  "Does extra reasoning capability change the outcome here?"
    ✓  Coordinating + synthesizing complex, conflicting inputs
    ✓  Architectural decisions with long-term consequences
    ✓  Judgment calls where nuance matters and errors are expensive
    ✗  Anything a Sonnet worker handles well (save Opus for what only it can do)

WHY THIS MATTERS:
  Opus costs significantly more per token than Sonnet, which costs more than
  Haiku. A system that runs Opus on every task pays a premium for capability
  it does not need. The advisor strategy captures Opus-quality outcomes
  on the tasks that require it while containing cost everywhere else.

DEMO SCENARIO:
  A Python auth module with a SQL injection vulnerability. Three specialists
  review it in parallel. The Opus advisor synthesizes a prioritized action
  plan — the judgment call only it should make.

BETA: Requires managed-agents-2026-04-01 header (SDK sets automatically).
"""

import os
import time
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# The code sample under review — deliberately flawed to give each tier
# something meaningful to find. Security issues for the analyst, style
# gaps for the style checker, missing docs for the doc writer.
CODE_SAMPLE = '''
class UserAuthManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self.failed_attempts = {}

    def login(self, username, password):
        query = f"SELECT * FROM users WHERE username=\'{username}\'"
        user = self.db.execute(query)
        if user and user["password"] == password:
            self.failed_attempts[username] = 0
            return True
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        return False

    def reset_password(self, username, new_password):
        self.db.execute(
            f"UPDATE users SET password=\'{new_password}\' WHERE username=\'{username}\'"
        )

    def is_locked(self, username):
        return self.failed_attempts.get(username, 0) >= 5
'''


# ============================================================================
# SETUP — Create one agent per tier
# ============================================================================

def create_haiku_style_checker():
    """
    Tier 3 — Haiku for mechanical style checking.
    Fast and cheap. Its task is pattern matching, not reasoning.
    Haiku's speed here means the coordinator gets style feedback quickly
    without waiting on a more capable (and slower) model to do checklist work.
    """
    print("  [Haiku]  Creating Style Checker...")
    agent = client.beta.agents.create(
        name="Style Checker",
        model="claude-haiku-4-5",
        system=(
            "You are a code style checker. When given Python code:\n"
            "1. Check ONLY for: missing docstrings, PEP 8 naming violations, "
            "missing type hints on function signatures, and lines over 88 characters\n"
            "2. Write a brief checklist report to /mnt/session/outputs/style_report.md\n"
            "3. Do NOT analyze logic, security, or correctness — that is another agent's job\n"
            "4. Keep it short: one bullet per issue, file path and line number if possible"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"            {agent.id}")
    return agent


def create_sonnet_analyst():
    """
    Tier 2 — Sonnet for deep security and correctness analysis.
    This task requires real reasoning: understanding attack surfaces, tracing
    data flow, and evaluating severity. Sonnet is the right level for this.
    """
    print("  [Sonnet] Creating Security Analyst...")
    agent = client.beta.agents.create(
        name="Security Analyst",
        model="claude-sonnet-4-6",
        system=(
            "You are a security and correctness analyst. When given Python code:\n"
            "1. Identify security vulnerabilities (injection, insecure storage, auth flaws)\n"
            "2. Identify correctness issues and dangerous edge cases\n"
            "3. Write findings to /mnt/session/outputs/analysis_report.md\n"
            "4. Prioritize by severity — Critical and High issues first\n"
            "5. For each issue: state what it is, why it is dangerous, and how to fix it"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"            {agent.id}")
    return agent


def create_sonnet_doc_writer():
    """
    Tier 2 — Sonnet for documentation writing.
    Good documentation requires understanding intent, anticipating reader
    confusion, and crafting clear prose. Reasoning-dependent — Sonnet tier.
    """
    print("  [Sonnet] Creating Documentation Writer...")
    agent = client.beta.agents.create(
        name="Documentation Writer",
        model="claude-sonnet-4-6",
        system=(
            "You are a technical documentation writer. When given Python code:\n"
            "1. Write docstrings for the class and every method (Google style)\n"
            "2. Write a usage example showing how to instantiate and call the class\n"
            "3. Note any parameters or return values that need special attention\n"
            "4. Write everything to /mnt/session/outputs/documentation.md\n"
            "5. Assume the reader is a developer who will maintain this code"
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"            {agent.id}")
    return agent


def create_opus_advisor(style_checker, analyst, doc_writer):
    """
    Tier 1 — Opus as the coordinating advisor.

    The advisor's job is to delegate, synthesize, and make the judgment call
    about what matters most. It does NOT do the style checking, analysis, or
    writing itself — that would waste Opus-level reasoning on work a cheaper
    model handles well.

    The advisory report is where Opus earns its cost: deciding which issues
    are blockers vs. nice-to-haves requires the kind of nuanced judgment that
    distinguishes Opus from Sonnet.
    """
    print("  [Opus]   Creating Engineering Advisor (coordinator)...")
    agent = client.beta.agents.create(
        name="Engineering Advisor",
        model="claude-opus-4-7",
        system=(
            "You are a senior engineering advisor coordinating a code quality review.\n\n"
            "YOUR ROLE IS TO DELEGATE AND SYNTHESIZE — not to do the work yourself.\n\n"
            "For each code review task:\n"
            "1. Send the code to the Style Checker for formatting and style issues\n"
            "2. Send the code to the Security Analyst for vulnerabilities and correctness\n"
            "3. Send the code to the Documentation Writer to draft proper docs\n"
            "4. Run ALL THREE IN PARALLEL — do not wait for one before starting the others\n\n"
            "5. Once all three have written their output files, read them and write a\n"
            "   prioritized action plan to /mnt/session/outputs/advisory_report.md\n\n"
            "The advisory report must:\n"
            "  - Open with a one-line verdict: BLOCK (critical issues), CAUTION (fixable), or SHIP\n"
            "  - List critical/high security issues that must be fixed before any deployment\n"
            "  - Give the developer a clear ordered action list: fix this first, then this, then this\n"
            "  - Close with whether the documentation is production-ready\n\n"
            "The verdict and prioritization are YOUR judgment call — that is why you are here."
        ),
        tools=[{"type": "agent_toolset_20260401"}],
        multiagent={
            "type": "coordinator",
            "agents": [
                {"type": "agent", "id": style_checker.id},
                {"type": "agent", "id": analyst.id},
                {"type": "agent", "id": doc_writer.id},
            ],
        },
    )
    print(f"            {agent.id}")
    return agent


def create_environment():
    print("  Creating environment...")
    environment = client.beta.environments.create(
        name="advisor-strategy-demo",
        config={
            "type": "cloud",
            "networking": {"type": "unrestricted"},
        },
    )
    print(f"  Environment: {environment.id}\n")
    return environment


# ============================================================================
# DEMO
# ============================================================================

def run_advisory_session(advisor, environment):
    """Send the code to the Opus advisor and let it delegate to its roster."""
    print("Creating session and sending code for review...")
    session = client.beta.sessions.create(
        agent=advisor.id,
        environment_id=environment.id,
        title="Auth module advisory review",
    )

    client.beta.sessions.events.send(
        session.id,
        events=[{
            "type": "user.message",
            "content": (
                "Please coordinate a full quality review of this Python auth module.\n"
                "Delegate style, security analysis, and documentation in parallel.\n"
                "Then give me your prioritized advisory report.\n\n"
                f"```python\n{CODE_SAMPLE}\n```"
            ),
        }],
    )
    print(f"  Session: {session.id}")
    return session


def poll_until_idle(session_id, timeout=300, poll_interval=15):
    """Poll until the advisor finishes synthesizing all sub-agent outputs."""
    print(f"\nWaiting for advisor to complete (polling every {poll_interval}s)...")
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        session = client.beta.sessions.retrieve(session_id)
        print(f"  [{elapsed:>3}s] status={session.status}")
        if session.status == "idle":
            print("  Advisor has finished.")
            return True
        if session.status == "terminated":
            print("  Session terminated with an error.")
            return False
    print(f"  Timed out after {timeout}s.")
    return False


def retrieve_outputs(session_id):
    """Display all output files — style report, analysis, docs, and advisory report."""
    print("\nRetrieving output files...")
    files = client.beta.files.list(scope_id=session_id)

    if not files.data:
        print("  No output files found.")
        return

    # Show the advisory report last — it is the synthesis and deserves the final word
    advisory = [f for f in files.data if "advisory" in f.filename]
    supporting = [f for f in files.data if "advisory" not in f.filename]

    for f in supporting + advisory:
        print(f"\n{'═' * 64}")
        tier = _tier_label(f.filename)
        print(f"  {f.filename}  {tier}")
        print(f"{'═' * 64}")
        content = client.beta.files.download(f.id)
        print(content.text)


def _tier_label(filename):
    """Return a tier indicator for each output file so the source is clear."""
    if "style" in filename:
        return "[Tier 3 — Haiku Style Checker]"
    if "analysis" in filename:
        return "[Tier 2 — Sonnet Security Analyst]"
    if "documentation" in filename:
        return "[Tier 2 — Sonnet Doc Writer]"
    if "advisory" in filename:
        return "[Tier 1 — Opus Advisor synthesis]"
    return ""


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== Advisor Strategy Demo ===")
    print("Three tiers, one coordinator, parallel execution.\n")

    # ── SETUP ─────────────────────────────────────────────────────────────────
    print("Creating agents (one per tier):")
    style_checker = create_haiku_style_checker()
    analyst = create_sonnet_analyst()
    doc_writer = create_sonnet_doc_writer()
    advisor = create_opus_advisor(style_checker, analyst, doc_writer)
    environment = create_environment()

    # ── DEMO ──────────────────────────────────────────────────────────────────
    session = run_advisory_session(advisor, environment)
    success = poll_until_idle(session.id)
    if success:
        retrieve_outputs(session.id)

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    print("\nCleaning up resources...")
    for agent in [style_checker, analyst, doc_writer, advisor]:
        client.beta.agents.archive(agent.id)
    client.beta.environments.archive(environment.id)
    print("  All agents and environment archived.")
    print("\nAdvisor strategy complete.")
    print("Carry this pattern forward:")
    print("  Haiku  → mechanical, fast, cheap")
    print("  Sonnet → reasoning, implementation, writing")
    print("  Opus   → judgment, synthesis, decisions that matter")


if __name__ == "__main__":
    main()
