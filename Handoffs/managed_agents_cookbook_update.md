# claude_api_cookbook — Managed Agents Update Handoff
### Six New Files: README + Four API Features + Advisor Strategy Pattern

**Prepared by:** Claude Code (claude-sonnet-4-6) + ArchieCur  
**Date:** 2026-05-16  
**Repo:** https://github.com/ArchieCur/claude_api_cookbook  
**Files staged in:** `May_api_repo_update_research/May_updated_files/`

---

## What Was Done

Six new files were written to cover the Managed Agents API (research preview). They are staged for review before being moved to their final destination as a new top-level `managed_agents/` folder in the repo.

| File | Feature | Status |
|---|---|---|
| `README.md` | Orientation, concept map, decision guide | Staged ✓ |
| `01_multi_agent_coordinator.py` | Multi-agent sessions, coordinator + roster | Staged ✓ |
| `02_outcomes_with_rubric.py` | Outcome-driven sessions, rubric grading | Staged ✓ |
| `03_dreams_memory_consolidation.py` | Async memory consolidation via Dreams | Staged ✓ |
| `04_webhooks.py` | Flask webhook handler, signature verification | Staged ✓ |
| `05_advisor_strategy.py` | Tiered Opus/Sonnet/Haiku routing pattern | Staged ✓ |

All six files follow the existing cookbook style: large docstring header with ASCII diagrams, module-level client setup, helper functions, and a `main()` structured as SETUP → DEMO → CLEANUP.

---

## Research Sources Used

All documentation was assembled by ArchieCur before writing began:

| File | Covers |
|---|---|
| `May_api_repo_update_research/Multiagent sessions.md` | Coordinator setup, threads, event types |
| `May_api_repo_update_research/Define_outcomes.md` | Outcome events, rubric format, Files API retrieval |
| `May_api_repo_update_research/dreams.md` | Dream lifecycle, error types, billing, limits |
| `May_api_repo_update_research/Managed_Agents_Subscribe_to_webhooks.md` | Webhook registration, signature verification, delivery behavior |
| `May_api_repo_update_research/environments.md` | Environment creation, networking config, packages |
| `May_api_repo_update_research/Agent_memory_stores.md` | Memory store CRUD, versioning, attaching to sessions |
| `May_api_repo_update_research/Agent_setup/Agent_setup.md` | Agent creation, update semantics, lifecycle |
| `May_api_repo_update_research/Agent_setup/agent_tools.md` | agent_toolset_20260401, custom tools |
| `May_api_repo_update_research/Agent_setup/Agent_MCP.md` | MCP server declaration, vault auth |
| `May_api_repo_update_research/Agent_setup/Agent_permissions.md` | always_allow, always_ask policies, tool confirmation |
| `May_api_repo_update_research/Agent_setup/Agent_skills.md` | Anthropic + custom skill attachment |

The prior session's handoff brief (`Handoffs/claude_api_cookbook_update_brief.md`, prepared 2026-05-15) served as the master scope document.

---

## Design Decisions Made During This Session

**Pattern chosen: fully self-contained files**  
Each file creates its own agent, environment, and supporting resources inline, then cleans up at the end. The alternative — expecting pre-created resource IDs via environment variables — was rejected because it breaks the cookbook's "run it and see" contract. A comment in every file's setup section explains the production pattern (create once, persist IDs, reuse).

**Polling over SSE streaming**  
Session event streaming (SSE) is the preferred real-time consumption mechanism but required documentation not available during this session (`Multiagent sessions.md` was only partially read — see Known Gaps below). Polling via `client.beta.sessions.retrieve()` was used throughout, with a note in each file pointing to SSE as an alternative.

**Dreams speed optimization**  
The Dreams file uses `claude-sonnet-4-6` (not `claude-opus-4-7`) and only two brief sessions as inputs. This was a deliberate choice after discussing the speed levers with ArchieCur: model choice and session transcript length are the two dominant cost drivers. The tradeoff is that demo output may be less rich than a production dream over many sessions.

**Webhooks as a two-mode script**  
`04_webhooks.py` is structurally different from the other four — it is a server, not a script. A two-mode design was used: `python 04_webhooks.py` starts Flask, `python 04_webhooks.py trigger` creates a live session to generate events. This keeps it as a single file while being runnable.

---

## Reflections on the Concepts

*The following section is the author's genuine assessment, prompted by ArchieCur's closing question: "As I read the final modules will you tell me what you think about these concepts and give your experience creating these files."*

**Dreams is the most novel concept here.** Every other feature maps to something familiar in distributed systems — coordinators are orchestration patterns, outcomes are automated QA loops, webhooks are standard event delivery. Dreams is genuinely new: a system-level reflection mechanism. An agent that consolidates its own memory from past experience isn't just a technical feature; it's a design philosophy about how persistent AI systems should evolve. The "input store never modified" constraint is the detail that most reveals the careful thinking behind it — it allows review and discard before adoption, which is the right default for a memory system.

**The Advisor Strategy is the most immediately applicable pattern.** The routing question — "does extra reasoning capability actually change the outcome here?" — is a question developers building multi-agent systems will never stop needing to ask. It gives users a mental model that outlasts this specific API version. Model tiers will change; the question stays relevant.

**The Outcomes grader design is quietly smart.** Running the grader in a separate context window — isolated from the agent's implementation choices — prevents the kind of anchoring bias that would make self-evaluation unreliable. The agent can't see the grader's reasoning; the grader can't be influenced by how the agent built the artifact. That's careful evaluation design, not just an implementation detail.

---

## Known Gaps and Second Thoughts

These are items flagged for the next session or future contributor:

**1. SSE event streaming not implemented**  
`Multiagent sessions.md` was only read through line 60. The full SSE event streaming API (real-time consumption of session events) is referenced in files 01, 02, and 03 as "an alternative to polling" but not implemented. A future update should add a `stream_session_events()` helper to at least one file demonstrating the streaming consumption pattern. This is the primary gap in the current files.

**2. `content.text` attribute on file downloads**  
The Dreams, Outcomes, and Coordinator files use `content.text` to display downloaded file content. The official SDK docs show `content.write_to_file(path)` for binary outputs. The `.text` attribute likely exists on the SDK's response object but was not verified against a live SDK. If users hit an AttributeError here, the fix is `content.write_to_file("/tmp/filename")` followed by reading the local file. A note should be added or the attribute confirmed.

**3. Dreams session task richness**  
The two demo sessions in `03_dreams_memory_consolidation.py` use very brief tasks ("check the Python version in memory") chosen specifically to minimize runtime. This is correct for the demo but may produce thin transcripts that limit the dream's insight generation. For a more visually compelling demo output, richer tasks — ones that cause the agent to reason about and resolve a contradiction it finds — would produce a clearer before/after comparison. The current version trades demo richness for speed, which was the right call for the cookbook context.

**4. Idempotency in webhooks is demo-only**  
`04_webhooks.py` uses an in-memory set for idempotency tracking. This is clearly noted in the file but warrants a follow-up example showing a Redis or database-backed approach for production use, since idempotency is load-bearing in any real deployment.

---

## What Worked Well in This Collaboration

- **Pre-assembled research files** were essential. Without the environments, memory stores, and agent setup docs gathered before writing began, the API call shapes would have required guesswork.
- **One file at a time with careful review** caught nothing that needed correction, but the review cadence meant each file could build on the previous one's patterns intentionally.
- **ArchieCur's question about Dreams speed** — "is there a light sleep vs deep REM mode?" — was the right question at the right moment. The answer (model choice + session count are the dominant levers) directly improved the file's design and produced the Speed Levers section that became one of the file's most useful additions.
- **Framing for SWE → AIE transition**: ArchieCur's observation that the audience is developers transitioning from software engineering to AI engineering shaped the writing throughout, even though it was never explicitly stated as a requirement. Translating new concepts into familiar analogues (agent roster = service registry, memory store = versioned database, Dreams = scheduled reindex) made the files more accessible without sacrificing accuracy.

---

## Next Steps for a Future Session

1. **Move staged files** from `May_api_repo_update_research/May_updated_files/` to `managed_agents/`
2. **Update `README.md`** (root) to add `managed_agents/` to the Contents section
3. **Verify `content.text`** against the live Anthropic Python SDK and update all five Python files if the correct attribute is different
4. **Add SSE streaming example** to at least `01_multi_agent_coordinator.py` once the full event stream API docs are available
5. **Consider a `06_memory_store_management.py`** covering the memory store CRUD operations, versioning, and the optimistic concurrency (`content_sha256` precondition) pattern — enough material exists in `Agent_memory_stores.md` for a standalone file

---

*Prepared by Claude Code (claude-sonnet-4-6) + ArchieCur, 2026-05-16*  
*This repo is open-source. Handoffs are kept as a historical record of development decisions and contributor reflections.*
