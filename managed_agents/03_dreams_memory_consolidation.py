"""
DREAMS - Async Memory Consolidation for Long-Running Agents

Agents write to their memory stores incrementally as they work. Over many
sessions a store accumulates duplicates, contradictions, and stale entries.
Dreams let Claude clean that up.

HOW IT WORKS:

  Past sessions           Input memory store
  (transcripts           (accumulated, messy)
   to mine)                      │
        │                        │
        └──────────┬─────────────┘
                   ▼
            dream.create()   ← async job; returns immediately
                   │
                   ▼
         ┌─────────────────────────────────────────┐
         │  Dream job (runs in the background)     │
         │                                         │
         │  1. Reads the input store               │
         │  2. Reads session transcripts           │
         │  3. Merges duplicates                   │
         │  4. Resolves contradictions             │
         │     (most recent value wins)            │
         │  5. Surfaces patterns as new insights   │
         └─────────────────────────────────────────┘
                   │
                   ▼
         Output memory store (new, clean)
         Input store UNCHANGED — review output before adopting it

SPEED LEVERS:
  Dreams have no explicit fast/slow mode, but two levers matter most:
  1. Model:    claude-sonnet-4-6 is faster; claude-opus-4-7 produces
               richer insights but takes longer. Use Sonnet for demos
               and routine maintenance, Opus for deep quarterly cleanups.
  2. Sessions: Runtime scales roughly linearly with transcript length.
               Two brief sessions consolidate much faster than ten long ones.
               Start small to evaluate quality before scaling up.

WHEN TO RUN:
  - After a long project sprint where the agent updated memory frequently
  - When you notice the agent acting on stale or contradictory information
  - On a regular schedule (e.g., weekly) for agents with active memory writes

BETA: Requires managed-agents-2026-04-01 AND dreaming-2026-04-21 headers.
      The SDK sets both automatically when you use client.beta.dreams.
"""

import os
import time
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

TERMINAL_DREAM_STATUSES = {"completed", "failed", "cancelled", "archived"}

# ============================================================================
# SEED DATA
# Deliberately messy memory entries — the kind that accumulate over real
# agent deployments. Each problem type is labelled so the before/after
# comparison makes the dream's work visible.
# ============================================================================

SEED_MEMORIES = [
    # ── Contradiction: indentation style ─────────────────────────────────────
    # An early entry that was never cleaned up when the convention changed.
    {
        "path": "/preferences/indentation.md",
        "content": "Use tabs for Python indentation. Tabs are the project standard.",
    },
    {
        "path": "/preferences/code_style.md",
        "content": "Follow PEP 8. Use 4-space indentation — never tabs. Black formatter enforced in CI.",
    },

    # ── Contradiction: stale vs current Python version ────────────────────────
    {
        "path": "/project/python_version.md",
        "content": "Project uses Python 3.8. Set up Q1 2024.",
    },
    {
        "path": "/project/runtime.md",
        "content": "Runtime: Python 3.11.4. Upgraded September 2024. Python 3.8 support dropped.",
    },

    # ── Near-duplicate: database info split across two entries ────────────────
    {
        "path": "/project/database.md",
        "content": "Primary database: PostgreSQL 14.",
    },
    {
        "path": "/project/database_notes.md",
        "content": "Database: PostgreSQL 14. Runs on port 5432. Managed via Supabase.",
    },

    # ── Overlap: user profile spread across two entries ───────────────────────
    {
        "path": "/user/name.md",
        "content": "User's name is Morgan.",
    },
    {
        "path": "/user/profile.md",
        "content": "User: Morgan. Prefers concise responses. Works in EST timezone. Uses dark mode.",
    },
]


# ============================================================================
# SETUP
# ============================================================================

def create_and_seed_memory_store():
    """
    Create the input memory store and populate it with messy content.
    In a real deployment these entries accumulate organically — we seed them
    here so the demo produces a visible before/after comparison.
    """
    print("Creating memory store...")
    store = client.beta.memory_stores.create(
        name="Coding Assistant Memory",
        description=(
            "Persistent memory for a coding assistant: "
            "user preferences, project configuration, and coding conventions."
        ),
    )
    print(f"  Store: {store.id}")

    print(f"\nSeeding {len(SEED_MEMORIES)} memories (with deliberate issues)...")
    for entry in SEED_MEMORIES:
        client.beta.memory_stores.memories.create(
            store.id,
            path=entry["path"],
            content=entry["content"],
        )
        print(f"  + {entry['path']}")

    return store


def create_agent():
    """Create a coding assistant that reads and interacts with its memory store."""
    print("\nCreating agent...")
    agent = client.beta.agents.create(
        name="Coding Assistant",
        model="claude-sonnet-4-6",
        system=(
            "You are a coding assistant with persistent memory. "
            "At the start of each task, check your memory store for relevant context. "
            "If you notice inconsistencies between entries, note them explicitly. "
            "Be brief and factual in your responses."
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"  Agent: {agent.id}")
    return agent


def create_environment():
    """Minimal sandbox — no network access needed for memory-reading tasks."""
    print("Creating environment...")
    environment = client.beta.environments.create(
        name="memory-demo",
        config={
            "type": "cloud",
            "networking": {
                "type": "limited",
                "allow_package_managers": False,
                "allow_mcp_servers": False,
                "allowed_hosts": [],
            },
        },
    )
    print(f"  Environment: {environment.id}")
    return environment


# ============================================================================
# DEMO — Phase 1: Run brief sessions to generate transcripts
# The dream mines these transcripts for patterns alongside the memory store.
# Brief, focused tasks keep runtime short while still producing useful signals.
# ============================================================================

def run_brief_session(agent, environment, store, task, title):
    """
    Run a single short session and wait for it to idle.
    Returns the completed session object so we can pass its ID to the dream.
    """
    print(f"\n  Session: '{title}'")
    session = client.beta.sessions.create(
        agent=agent.id,
        environment_id=environment.id,
        title=title,
        resources=[{
            "type": "memory_store",
            "memory_store_id": store.id,
            # read_write so the agent can update entries it finds stale or wrong.
            # Those writes appear in the session transcript, giving the dream
            # more signal about what the agent actually observed.
            "access": "read_write",
            "instructions": "Your persistent memory. Check relevant sections before responding.",
        }],
    )

    client.beta.sessions.events.send(
        session.id,
        events=[{"type": "user.message", "content": task}],
    )

    return _poll_session_until_idle(session, timeout=120, poll_interval=10)


def _poll_session_until_idle(session, timeout=120, poll_interval=10):
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        session = client.beta.sessions.retrieve(session.id)
        if session.status in ("idle", "terminated"):
            print(f"    → {session.status} after {elapsed}s")
            return session
    print(f"    → timed out after {timeout}s")
    return session


# ============================================================================
# DEMO — Phase 2: Create and poll the dream
# ============================================================================

def create_dream(store, session_ids):
    """
    Launch the dream job.

    The dream reads the input store and mines the session transcripts.
    It produces a new output store — the input store is never modified,
    so you can compare and discard if you do not like the result.
    """
    print("\nCreating dream...")
    dream = client.beta.dreams.create(
        inputs=[
            {"type": "memory_store", "memory_store_id": store.id},
            # Pass the session IDs whose transcripts the dream should mine.
            # Max 100 sessions per dream. For large deployments, pass the
            # most recent sessions; older ones add cost without much new signal.
            {"type": "sessions", "session_ids": session_ids},
        ],
        # Sonnet for speed. Use claude-opus-4-7 for richer consolidation
        # when quality matters more than turnaround time.
        model="claude-sonnet-4-6",
        instructions=(
            "Focus on coding conventions, project configuration, and user preferences. "
            "Merge duplicate entries, resolve contradictions in favor of the most recent "
            "value, and consolidate overlapping information into single authoritative entries. "
            "Surface any patterns the sessions reveal about the agent's working style. "
            "Ignore one-off debugging notes."
        ),
    )
    print(f"  Dream: {dream.id}  initial status: {dream.status}")
    return dream


def poll_dream(dream_id, timeout=900, poll_interval=20):
    """
    Poll until the dream reaches a terminal status.

    The output store ID appears in dream.outputs[] once the job starts running
    (before it completes). We retrieve it after completion for simplicity.
    Runtime typically 3-8 minutes for a small store + 2 brief sessions with Sonnet.
    """
    print(f"\nPolling dream (every {poll_interval}s, timeout {timeout // 60}m)...")
    elapsed = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        dream = client.beta.dreams.retrieve(dream_id)
        tokens = dream.usage.input_tokens if dream.usage else "—"
        print(f"  [{elapsed:>3}s] status={dream.status:<12} input_tokens={tokens}")

        if dream.status in TERMINAL_DREAM_STATUSES:
            return dream

    print(f"  Timed out after {timeout}s.")
    return client.beta.dreams.retrieve(dream_id)


def get_output_store_id(dream):
    """Extract the consolidated output store ID from the dream's outputs array."""
    for output in dream.outputs:
        if output.type == "memory_store":
            return output.memory_store_id
    return None


# ============================================================================
# COMPARE — Before / After
# ============================================================================

def compare_stores(input_store_id, output_store_id):
    """
    Print the contents of both stores so the consolidation is visible.
    Fewer entries in the output store is the expected result: duplicates
    merged, contradictions resolved, user profile consolidated.
    """
    print("\n" + "═" * 64)
    print("  BEFORE — Input Store (original, messy)")
    print("═" * 64)
    _print_store(input_store_id)

    print("\n" + "═" * 64)
    print("  AFTER  — Output Store (consolidated by dream)")
    print("═" * 64)
    _print_store(output_store_id)

    print(
        "\n  Attach the output store (not the input store) to future sessions.\n"
        "  The input store is unchanged — archive it once you are satisfied\n"
        "  with the consolidated output."
    )


def _print_store(store_id):
    """List every memory in a store and print its path and content."""
    page = client.beta.memory_stores.memories.list(
        store_id,
        path_prefix="/",
        order_by="path",
        depth=5,
    )
    entries = page.data
    print(f"\n  {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}:\n")

    for item in entries:
        print(f"  ┌─ {item.path}")
        try:
            mem = client.beta.memory_stores.memories.retrieve(
                item.id,
                memory_store_id=store_id,
            )
            for line in mem.content.splitlines():
                print(f"  │  {line}")
        except Exception:
            # Directory-level nodes have no content of their own
            print("  │  [directory node]")
        print("  │")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== Dreams Memory Consolidation Demo ===\n")
    print("This demo runs in three phases:")
    print("  1. Seed a messy memory store")
    print("  2. Run two brief sessions to generate transcripts")
    print("  3. Dream over those sessions to produce a clean store")
    print("  Expected total runtime: 6–15 minutes\n")

    # ── SETUP ─────────────────────────────────────────────────────────────────
    store = create_and_seed_memory_store()
    agent = create_agent()
    environment = create_environment()

    # ── DEMO: Phase 1 — Sessions ───────────────────────────────────────────────
    print("\n── Phase 1: Generating session transcripts ──")
    session1 = run_brief_session(
        agent, environment, store,
        task=(
            "Review your project configuration in memory. "
            "What Python version is the project using? "
            "Note any inconsistencies you find."
        ),
        title="Project config review",
    )
    session2 = run_brief_session(
        agent, environment, store,
        task=(
            "Check your coding style preferences in memory. "
            "Summarize the indentation rule in one sentence."
        ),
        title="Coding style check",
    )

    # ── DEMO: Phase 2 — Dream ─────────────────────────────────────────────────
    print("\n── Phase 2: Consolidating memory ──")
    dream = create_dream(store, [session1.id, session2.id])
    final_dream = poll_dream(dream.id)

    if final_dream.status == "completed":
        output_store_id = get_output_store_id(final_dream)
        if output_store_id:
            print(f"\n  Output store: {output_store_id}")
            compare_stores(store.id, output_store_id)
        else:
            print("  Dream completed but no output store found in dream.outputs.")

    elif final_dream.status == "failed":
        print(f"\n  Dream failed.")
        if hasattr(final_dream, "error") and final_dream.error:
            print(f"  Error: {final_dream.error}")

    else:
        print(f"\n  Dream ended with status: {final_dream.status}")

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    print("\nCleaning up resources...")
    client.beta.agents.archive(agent.id)
    client.beta.environments.archive(environment.id)
    # Archive the original messy store — the consolidated output store is
    # the one to carry forward into future sessions.
    client.beta.memory_stores.archive(store.id)
    print("  Agent, environment, and original memory store archived.")
    print("  Output store retained — attach it to future sessions as your new baseline.")


if __name__ == "__main__":
    main()
