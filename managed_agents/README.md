# Managed Agents API

A collection of Python implementation patterns for building production agent systems with the Anthropic Managed Agents API (research preview).

---

## What It Is

The Managed Agents API is a different execution model from the standard Messages API. Instead of stateless, single-turn requests, you define a persistent **agent** — a reusable configuration bundling a model, system prompt, and tools — and run it inside a sandboxed cloud **environment** that provides a real filesystem, shell, and network. A **session** is one run of an agent in an environment: the agent reads and writes files, executes code, browses the web, and maintains state across multiple turns, all without you managing any infrastructure.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MANAGED AGENTS MODEL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Agent (config)          Environment (sandbox)                 │
│   ┌─────────────┐         ┌──────────────────────┐             │
│   │ model       │         │ filesystem            │             │
│   │ system      │ ──────▶ │ shell / bash          │             │
│   │ tools       │         │ network access        │             │
│   │ skills      │         │ /mnt/memory/  ◀─────────── memory  │
│   └─────────────┘         │ /mnt/session/ ◀─────────── outputs │
│          │                └──────────────────────┘             │
│          │                           │                          │
│          └───────────── Session ─────┘                          │
│                     (persistent run)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**When to use Managed Agents instead of the Messages API:**
- The task requires writing and reading files, executing code, or browsing the web autonomously
- You need state to persist across multiple turns without managing it yourself
- You want to coordinate multiple specialized agents working in parallel
- You need a measurable quality bar, not just a response

**Beta access:** Request access at the Anthropic Console. All requests require the `managed-agents-2026-04-01` beta header, which the Python SDK sets automatically when you call `client.beta.*`.

---

## Core Concepts

| Concept | What it is | ID prefix |
|---|---|---|
| **Agent** | Reusable, versioned config: model + system + tools | `agent_...` |
| **Environment** | Cloud sandbox with filesystem, shell, and network | `env_...` |
| **Session** | One run of an agent in an environment | `sesn_...` |
| **Memory Store** | Persistent key-value store that survives across sessions | `memstore_...` |
| **Dream** | Async job that consolidates a memory store from past sessions | `drm_...` |

---

## Files in This Folder

| File | Feature | Best for |
|---|---|---|
| `01_multi_agent_coordinator.py` | **Multi-agent sessions** | Parallelizing subtasks, specialization, escalation |
| `02_outcomes_with_rubric.py` | **Outcomes** | Tasks with a clear deliverable and measurable quality bar |
| `03_dreams_memory_consolidation.py` | **Dreams** | Cleaning up memory stores that have accumulated over many sessions |
| `04_webhooks.py` | **Webhooks** | Reacting to session lifecycle events without polling |
| `05_advisor_strategy.py` | **Advisor strategy** | Tiered model routing for cost-efficient multi-agent systems |

---

## Choosing the Right Pattern

```
Is your task a single deliverable with a measurable quality bar?
  ├─ Yes ──▶  02_outcomes_with_rubric.py
  └─ No
      │
      Does it decompose into parallel or specialized subtasks?
        ├─ Yes ──▶  01_multi_agent_coordinator.py
        │           (+ 05_advisor_strategy.py for tiered model costs)
        └─ No (single agent, conversational session)
            │
            Does the agent run many sessions over time?
              ├─ Yes, memory is accumulating ──▶  03_dreams_memory_consolidation.py
              └─ Need to react to state changes ──▶  04_webhooks.py
```

---

## Prerequisites

```bash
pip install anthropic python-dotenv flask
```

Set your API key:
```env
ANTHROPIC_API_KEY=your_api_key_here
```

Each file is self-contained and creates its own agent and environment inline. Run any file directly:
```bash
python 01_multi_agent_coordinator.py
python 02_outcomes_with_rubric.py
python 03_dreams_memory_consolidation.py
python 04_webhooks.py
python 05_advisor_strategy.py
```

> **Note on resources:** Agents and environments are designed to be created once and reused across many sessions. Each file creates them inline for self-containment and cleans up afterward. In production, create them once, store the IDs in config or environment variables, and reference them by ID.

---

## The `client.beta.*` Namespace

All Managed Agents API calls live under `client.beta`:

```python
import anthropic
client = anthropic.Anthropic()

client.beta.agents          # create, update, archive agents
client.beta.environments    # create, update, archive environments
client.beta.sessions        # create sessions, send events, retrieve status
client.beta.sessions.events # send user events (messages, outcomes, confirmations)
client.beta.memory_stores   # create and manage memory stores
client.beta.memory_stores.memories   # read/write individual memories
client.beta.dreams          # create and poll dream jobs
client.beta.webhooks        # verify and unwrap webhook deliveries
client.beta.files           # retrieve output files from a session
```

The SDK sets the required `anthropic-beta: managed-agents-2026-04-01` header automatically. Dreams additionally require `dreaming-2026-04-21`, which the SDK also sets automatically when you call `client.beta.dreams`.

---

## Advisor Strategy: Tiered Model Selection

A pattern introduced at Code with Claude for multi-agent systems. Match model capability to task complexity:

| Role | Model | Why |
|---|---|---|
| Coordinator / Advisor | `claude-opus-4-7` | Highest reasoning for synthesis, judgment, architectural decisions |
| General workers | `claude-sonnet-4-6` | Strong capability at lower cost for implementation work |
| Tool-heavy / mechanical tasks | `claude-haiku-4-5` | Fast and cheap for search, formatting, lookup |

See `05_advisor_strategy.py` for an end-to-end example.

---

*Research preview — APIs and behavior may change. Check the [Anthropic docs](https://docs.anthropic.com/) for the latest.*
