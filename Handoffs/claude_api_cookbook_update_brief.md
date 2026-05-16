# claude_api_cookbook — Update Brief
### Managed Agents API: Multi-agent Sessions, Outcomes, Dreams, and Webhooks

**Prepared for:** New Claude Code session
**Repo:** https://github.com/ArchieCur/claude_api_cookbook
**Prepared by:** Claude Sonnet 4.6 + ArchieCur, 2026-05-15

---

## Context

The `claude_api_cookbook` is a comprehensive collection of Python examples and patterns for building applications with the Anthropic Claude API. It currently covers prompting, tool use, RAG systems, agents, and workflows. It is a practitioner's resource — working code people can copy and adapt, not reference documentation.

During a session working on the `claude_code_field_guide` repo, four new Managed Agents API features were reviewed and documented. They were determined to belong in the **cookbook** rather than the field guide because they are programmatic API features (not Claude Code CLI features) suited for developers building production agent systems.

The field guide's `managed_agents/README.md` already contains a brief orientation to these features and explicitly points readers to this cookbook for deep implementation patterns. That pointer is waiting to be fulfilled.

---

## What to Add: Overview

All four features are part of the **Managed Agents API** (research preview). Every request requires:

```python
# All Managed Agents requests use the beta namespace
# The SDK sets the required header automatically:
# anthropic-beta: managed-agents-2026-04-01

client = anthropic.Anthropic()
# Access via: client.beta.sessions, client.beta.agents, client.beta.dreams, etc.
```

---

## Feature 1: Multi-agent Sessions

**What it is:** One agent acts as a coordinator and delegates to a roster of sub-agents. Each agent runs in its own session thread with an isolated context window. All agents share the same container and filesystem. The coordinator reports on the primary thread; sub-agents are spawned at runtime.

**Best for:** Parallelization (fan out independent subtasks), specialization (route to domain-focused agents), escalation (consult a more capable model for hard subtasks).

**Key API shape:**

```python
import anthropic

client = anthropic.Anthropic()

# 1. Create sub-agents
reviewer = client.beta.agents.create(
    name="Code Reviewer",
    model="claude-sonnet-4-6",
    system="You review code for correctness, security, and performance. Be specific and concise.",
)

test_writer = client.beta.agents.create(
    name="Test Writer",
    model="claude-sonnet-4-6",
    system="You write pytest test cases. Cover happy paths, edge cases, and error conditions.",
)

# 2. Create coordinator with roster
coordinator = client.beta.agents.create(
    name="Engineering Lead",
    model="claude-opus-4-7",
    system="You coordinate engineering work. Delegate code review to the reviewer and test writing to the test writer. Synthesize their output.",
    tools=[{"type": "agent_toolset_20260401"}],
    multiagent={
        "type": "coordinator",
        "agents": [
            {"type": "agent", "id": reviewer.id},
            {"type": "agent", "id": test_writer.id},
        ],
    },
)

# 3. Create a session and send work
session = client.beta.sessions.create(
    agent=coordinator.id,
    environment_id=environment.id,
)

client.beta.sessions.events.send(
    session.id,
    events=[{
        "type": "user.message",
        "content": "Review the authentication module in src/auth.py and write tests for it.",
    }],
)
```

**Threads:** The session-level event stream is the primary thread. Sub-agent activity surfaces there as `session.thread_created`, `agent.thread_message_received`, etc. To drill into a specific sub-agent's reasoning, stream its session thread directly.

**Limits:** Max 20 unique agents in a coordinator's roster. Max 25 concurrent threads per session. Coordination is one level deep only — sub-agents cannot spawn their own sub-agents.

**Suggested file:** `managed_agents/01_multi_agent_coordinator.py`

---

## Feature 2: Outcomes

**What it is:** Define what "done" looks like with a rubric. The agent works toward the target, and a separate grader agent evaluates the artifact against the rubric in its own context window (avoiding influence from the main agent's implementation choices). The agent iterates until the outcome is satisfied or max iterations is reached.

**Best for:** Tasks with a clear deliverable and measurable quality bar — financial models, data pipelines, generated documents, code that must pass specific criteria.

**Key API shape:**

```python
import anthropic
import time

client = anthropic.Anthropic()

RUBRIC = """
# Data Pipeline Rubric

## Correctness
- Reads from the specified CSV file path
- Outputs a valid JSON file with the correct schema
- Handles missing values without crashing

## Code Quality
- Uses context managers for file I/O
- Includes a __main__ guard
- Has a docstring describing inputs and outputs

## Error Handling
- Raises a descriptive ValueError for missing required columns
- Logs errors before raising
"""

# Create a session
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=environment.id,
    title="Build CSV-to-JSON pipeline",
)

# Define the outcome — agent starts working immediately on receipt
client.beta.sessions.events.send(
    session.id,
    events=[{
        "type": "user.define_outcome",
        "description": "Write a Python script that converts sales_data.csv to output.json",
        "rubric": {"type": "text", "content": RUBRIC},
        "max_iterations": 5,  # default 3, max 20
    }],
)

# Poll for outcome result
while True:
    session = client.beta.sessions.retrieve(session.id)
    evaluations = session.outcome_evaluations
    if evaluations:
        latest = evaluations[-1]
        print(f"Result: {latest.result}")
        if latest.result not in ("satisfied", "needs_revision"):
            break
        if latest.result == "satisfied":
            break
    time.sleep(10)

# Retrieve output files
files = client.beta.files.list(scope_id=session.id)
for f in files.data:
    print(f.id, f.filename)
```

**Outcome results:**

| Result | Meaning |
|:---|:---|
| `satisfied` | All rubric criteria met — session idles |
| `needs_revision` | Agent starts another iteration |
| `max_iterations_reached` | Agent may do one final pass, then idles |
| `failed` | Rubric fundamentally doesn't match the task |

**Rubric writing tip:** Use explicit, gradeable criteria. "The CSV contains a price column with numeric values" grades cleanly. "The data looks good" does not. If you don't have a rubric, give Claude a known-good example and ask it to analyze what makes it good, then turn that analysis into criteria.

**Suggested file:** `managed_agents/02_outcomes_with_rubric.py`

---

## Feature 3: Dreams

**What it is:** An async job that reads an existing memory store alongside past session transcripts and produces a new, reorganized memory store. Duplicates are merged, stale or contradicted entries are replaced with the latest value, and new patterns are surfaced as insights. The input store is never modified.

**Best for:** Long-running agent deployments where memory accumulates over many sessions and starts to degrade with contradictions and duplicates. Run periodically to keep memory stores clean.

**Requires an additional beta header** — `dreaming-2026-04-21`. The SDK sets it automatically when you use `client.beta.dreams`.

**Key API shape:**

```python
import anthropic
import time

client = anthropic.Anthropic()

# Create a dream — provide the store to clean + sessions to mine for insights
dream = client.beta.dreams.create(
    inputs=[
        {"type": "memory_store", "memory_store_id": store_id},
        {"type": "sessions", "session_ids": [session_a, session_b, session_c]},
    ],
    model="claude-opus-4-7",
    instructions="Focus on coding-style preferences and architectural decisions. Ignore one-off debugging notes.",
)

print(f"Dream created: {dream.id} — status: {dream.status}")

# Poll for completion — typically minutes to tens of minutes
while dream.status in ("pending", "running"):
    time.sleep(10)
    dream = client.beta.dreams.retrieve(dream.id)
    print(f"status={dream.status} input_tokens={dream.usage.input_tokens}")

if dream.status == "completed":
    # Get the output store ID
    output_store_id = next(
        o.memory_store_id for o in dream.outputs if o.type == "memory_store"
    )
    print(f"New memory store ready: {output_store_id}")

    # Attach the new store to a future session
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=environment_id,
        resources=[
            {"type": "memory_store", "memory_store_id": output_store_id}
        ],
    )
else:
    print(f"Dream ended with status: {dream.status}")
    if dream.error:
        print(f"Error: {dream.error}")
```

**Supported models:** `claude-opus-4-7`, `claude-sonnet-4-6`
**Max sessions per dream:** 100
**Billing:** Standard token rates — cost scales roughly linearly with number and length of input sessions. Start with a small batch to evaluate curation quality before scaling up.

**Suggested file:** `managed_agents/03_dreams_memory_consolidation.py`

---

## Feature 4: Webhooks

**What it is:** Subscribe to major session lifecycle events without polling. Each delivery is small (type + ID only — not the full object). When you receive an event, fetch the resource by ID. Retries are handled automatically; the same `event.id` delivered twice means it's a retry.

**Best for:** Production integrations where you need to react to agent state changes — notify a user when their job is done, trigger a downstream step when a session idles, alert on terminal errors.

**Setup:** Register an endpoint at **Manage > Webhooks** in [Console](https://platform.claude.com). Store the `whsec_`-prefixed signing secret securely — it's shown only once.

**Key API shape (Flask example):**

```python
import anthropic
from flask import Flask, request

# Set ANTHROPIC_WEBHOOK_SIGNING_KEY in your environment
client = anthropic.Anthropic()
app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # unwrap() verifies the signature AND checks the payload is < 5 minutes old
        event = client.beta.webhooks.unwrap(
            request.get_data(as_text=True),
            headers=dict(request.headers),
        )
    except Exception:
        return "invalid signature", 400

    match event.data.type:
        case "session.status_idled":
            session = client.beta.sessions.retrieve(event.data.id)
            handle_session_idle(session)

        case "session.status_terminated":
            session = client.beta.sessions.retrieve(event.data.id)
            handle_terminal_error(session)

        case "session.outcome_evaluation_ended":
            session = client.beta.sessions.retrieve(event.data.id)
            for outcome in session.outcome_evaluations:
                print(f"{outcome.outcome_id}: {outcome.result}")

    return "", 200


def handle_session_idle(session):
    print(f"Session {session.id} is waiting for input or completed work.")


def handle_terminal_error(session):
    print(f"Session {session.id} hit a terminal error.")


if __name__ == "__main__":
    app.run(port=5000)
```

**Supported session events:**

| Event | Trigger |
|:---|:---|
| `session.status_run_started` | Agent execution kicked off |
| `session.status_idled` | Agent awaiting input |
| `session.status_rescheduled` | Transient error, session retrying |
| `session.status_terminated` | Session hit a terminal error |
| `session.thread_created` | New multiagent thread opened |
| `session.outcome_evaluation_ended` | One outcome evaluation iteration completed |

**Delivery notes:** Ordering is not guaranteed — use `created_at` to sort. Redirects (`3xx`) count as failures. Endpoint auto-disables after ~20 consecutive failures.

**Suggested file:** `managed_agents/04_webhooks.py`

---

## Bonus Pattern: The Advisor Strategy

Discussed at the Code with Claude conference. A tiered model configuration for multi-agent systems that balances capability and cost:

| Role | Model | Reasoning |
|:---|:---|:---|
| Coordinator / Advisor | `claude-opus-4-7` | Highest reasoning for judgment calls, synthesis, architectural decisions |
| General workers | `claude-sonnet-4-6` | Strong capability at lower cost for implementation work |
| Tool-heavy tasks | `claude-haiku-4-5` | Fast and cheap for search, lookup, formatting, mechanical tasks |

Demonstrate this as an end-to-end example in `managed_agents/05_advisor_strategy.py` — a three-tier coordinator that routes different subtask types to the appropriate model tier.

---

## Suggested Repo Structure

Add a new top-level folder:

```
managed_agents/
  README.md                         # What the Managed Agents API is and when to use it
  01_multi_agent_coordinator.py     # Coordinator + roster, threading, event streaming
  02_outcomes_with_rubric.py        # Define outcome, poll for result, retrieve output files
  03_dreams_memory_consolidation.py # Create dream, poll status, attach output store
  04_webhooks.py                    # Flask webhook handler with signature verification
  05_advisor_strategy.py            # End-to-end Opus/Sonnet/Haiku tiered system
```

---

## Local Source Files

The following files in `e:\Cowork\Code_with_Claude\` contain the full documentation used to write this brief. Read them for complete API reference, all language SDKs, and edge cases:

| File | Covers |
|:---|:---|
| `Multiagent sessions.md` | Coordinator setup, threads, event types, tool permissions routing |
| `Define_outcomes.md` | Outcome events, rubric format, retrieving deliverables via Files API |
| `dreams.md` | Full lifecycle, error types, archive/cancel, billing, limits |
| `Managed_Agents_Subscribe_to_webhooks.md` | Webhook registration, vault events, delivery behavior |

The field guide's orientation doc (cross-reference): `e:\Cowork\Field_Guide_updates\managed_agents\README.md`

---

## Style Notes for the Cookbook

- **Python first** — the cookbook is Python-focused; include TypeScript only if a pattern is significantly different
- **Working examples** — every file should run end-to-end with minimal setup (set `ANTHROPIC_API_KEY`, create an agent/environment, go)
- **Comments explain the why** — not what the code does, but why this pattern matters or what to watch out for
- **Keep it practical** — use realistic scenarios (financial analysis, code review, data pipelines) not toy examples
- **Note research preview status** — each file should have a header comment noting the beta header requirement and that behavior may change

---

*Prepared by Claude Sonnet 4.6 + ArchieCur, 2026-05-15*
