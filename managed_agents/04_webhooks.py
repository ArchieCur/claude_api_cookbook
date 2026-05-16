"""
WEBHOOKS - React to Session Events Without Polling

Webhooks notify your server of major session lifecycle changes. Each delivery
is intentionally small — it carries the event type and the object ID, not the
full object. When you receive an event, fetch the resource by ID for fresh data.

WHY THIS DESIGN:
  Keeping deliveries small means retries are cheap and stale data is impossible.
  A retry delivers the same event.id — detect it and discard rather than
  double-processing.

EVENT FLOW:

  Session state change
         │
         ▼
  Anthropic POSTs to your endpoint
  ┌──────────────────────────────────────────┐
  │  POST /webhook                           │
  │  X-Webhook-Signature: v1=<hmac>          │
  │                                          │
  │  { "type": "event",                      │
  │    "id": "event_01ABC...",               │
  │    "created_at": "2026-05-16T...",       │
  │    "data": {                             │
  │      "type": "session.status_idled",     │
  │      "id":   "sesn_01XYZ..."            │
  │    }                                     │
  │  }                                       │
  └──────────────────────────────────────────┘
         │
         ▼
  Your server: verify signature → fetch session by ID → handle → return 2xx

SUPPORTED SESSION EVENTS:
  session.status_run_started      Agent execution kicked off
  session.status_idled            Agent awaiting input or completed work
  session.status_rescheduled      Transient error — session retrying automatically
  session.status_terminated       Terminal error — session will not recover
  session.thread_created          New multiagent sub-agent thread opened
  session.outcome_evaluation_ended  One outcome grader cycle completed

DELIVERY BEHAVIOR:
  - Ordering is NOT guaranteed. Use created_at to sort if order matters.
  - Anthropic retries at least once. Same event.id = retry — safe to discard.
  - Redirects (3xx) count as failures.
  - Endpoint auto-disables after ~20 consecutive failures.

SETUP (one-time):
  1. Register your endpoint at Console → Manage → Webhooks.
     URL must be HTTPS on port 443 with a public hostname.
  2. Copy the whsec_... signing secret shown at creation (shown once only).
  3. Set environment variables:
       ANTHROPIC_API_KEY=your_key
       ANTHROPIC_WEBHOOK_SIGNING_KEY=whsec_...
  4. For local development: expose localhost with `ngrok http 5000`
     and register the ngrok HTTPS URL in Console.

BETA: Requires managed-agents-2026-04-01 header (SDK sets automatically).
"""

import os
import sys
import time
from anthropic import Anthropic
from flask import Flask, request

# The SDK reads ANTHROPIC_WEBHOOK_SIGNING_KEY from the environment automatically.
# It also reads ANTHROPIC_API_KEY. Both must be set before starting the server.
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
app = Flask(__name__)

# ============================================================================
# IDEMPOTENCY
# In production use a database or distributed cache (Redis, DynamoDB) keyed
# on event.id with a TTL longer than Anthropic's retry window.
# This in-memory set works for single-process demos only.
# ============================================================================
_processed_event_ids: set[str] = set()


# ============================================================================
# WEBHOOK HANDLER
# ============================================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Main webhook entry point.

    Two responsibilities before any business logic:
      1. Verify the signature — rejects tampered or replayed payloads.
      2. Check idempotency — discards retries we already handled.

    Return any 2xx to acknowledge. Do not return 3xx (treated as failure).
    For slow handlers, acknowledge immediately and process in a background task.
    """
    # ── Signature verification ────────────────────────────────────────────────
    # unwrap() verifies the HMAC signature AND rejects payloads older than
    # 5 minutes, protecting against replay attacks.
    # It raises an exception on any failure — never skip this step.
    try:
        event = client.beta.webhooks.unwrap(
            request.get_data(as_text=True),
            headers=dict(request.headers),
        )
    except Exception as exc:
        app.logger.warning("Webhook signature verification failed: %s", exc)
        return "invalid signature", 400

    # ── Idempotency check ─────────────────────────────────────────────────────
    # Anthropic retries failed deliveries. The same event.id arriving twice
    # is a retry — acknowledge it immediately without re-processing.
    if event.id in _processed_event_ids:
        app.logger.info("Duplicate event %s — discarding", event.id)
        return "", 200

    _processed_event_ids.add(event.id)

    # ── Dispatch on event type ────────────────────────────────────────────────
    event_type = event.data.type
    object_id = event.data.id
    app.logger.info("Received: %s  id=%s", event_type, object_id)

    if event_type == "session.status_idled":
        session = client.beta.sessions.retrieve(object_id)
        handle_session_idled(session)

    elif event_type == "session.status_terminated":
        session = client.beta.sessions.retrieve(object_id)
        handle_session_terminated(session)

    elif event_type == "session.outcome_evaluation_ended":
        session = client.beta.sessions.retrieve(object_id)
        handle_outcome_evaluation_ended(session)

    elif event_type == "session.thread_created":
        # A sub-agent thread opened inside a multiagent session.
        # object_id is the session ID; the thread itself is visible on the
        # session's event stream.
        handle_thread_created(object_id)

    elif event_type == "session.status_rescheduled":
        # Transient error — Anthropic is retrying automatically.
        # Log it for observability; no action required.
        app.logger.warning("Session %s hit a transient error and is rescheduling.", object_id)

    elif event_type == "session.status_run_started":
        # Agent execution has kicked off. Useful for audit logs or dashboards.
        app.logger.info("Session %s started running.", object_id)

    else:
        # Vault events (vault.created, vault_credential.archived, etc.) and
        # any future event types land here. Log and acknowledge.
        app.logger.info("Unhandled event type: %s", event_type)

    return "", 200


# ============================================================================
# EVENT HANDLERS
# Each handler receives a freshly fetched object, not the raw webhook payload.
# ============================================================================

def handle_session_idled(session):
    """
    The agent has finished its current work and is waiting for input.
    This fires after normal task completion AND after outcome satisfaction.

    Common uses:
      - Notify the end user that their job is done
      - Trigger a downstream step in a pipeline
      - Send a follow-up message to continue a multi-step workflow
    """
    print(f"\n[IDLED] Session '{session.title or session.id}'")
    print(f"  Status:   {session.status}")

    # If this session ran an outcome, check the result
    if session.outcome_evaluations:
        latest = session.outcome_evaluations[-1]
        print(f"  Outcome:  {latest.result} (iteration {latest.iteration})")
        if latest.result == "satisfied":
            _notify_user(session.id, "Your task has been completed successfully.")

    # Example: send a follow-up message when session idles
    # client.beta.sessions.events.send(
    #     session.id,
    #     events=[{"type": "user.message", "content": "Please summarize what you built."}],
    # )


def handle_session_terminated(session):
    """
    The session hit a terminal error and will not recover.
    Alert on-call or log to your error tracker.
    """
    print(f"\n[TERMINATED] Session '{session.title or session.id}'")
    print(f"  This session has ended with a terminal error.")
    print(f"  Review the session event stream for details: {session.id}")
    # In production: page on-call, open an incident, notify the user


def handle_outcome_evaluation_ended(session):
    """
    One grader cycle has completed for an outcome-oriented session.
    Fires after each iteration, not just the final one — useful for
    progress dashboards that show the agent refining its work in real time.
    """
    if not session.outcome_evaluations:
        return

    latest = session.outcome_evaluations[-1]
    print(f"\n[OUTCOME EVAL] Session '{session.title or session.id}'")
    print(f"  Iteration: {latest.iteration}  Result: {latest.result}")

    if latest.result == "satisfied":
        print("  All rubric criteria met. Retrieving deliverables...")
        files = client.beta.files.list(scope_id=session.id)
        for f in files.data:
            print(f"  Output file: {f.filename} ({f.id})")

    elif latest.result == "needs_revision":
        print("  Agent is revising. Next grader cycle pending.")

    elif latest.result == "max_iterations_reached":
        print("  Max iterations reached. Review what was produced.")


def handle_thread_created(session_id):
    """
    A new sub-agent thread opened inside a multiagent session.
    Useful for tracking how many agents a coordinator has spun up.
    """
    print(f"\n[THREAD CREATED] in session {session_id}")
    print("  A sub-agent has been delegated work by the coordinator.")


def _notify_user(session_id, message):
    """Placeholder for your user notification logic (email, Slack, push, etc.)."""
    print(f"  [notify] Session {session_id}: {message}")


# ============================================================================
# TRIGGER — generate test webhook events
# Run in a separate terminal: python 04_webhooks.py trigger
# Your Flask server must already be running and registered in Console.
# ============================================================================

def trigger_test_events():
    """
    Create a real session to generate webhook events.
    The session does minimal work — enough to fire idled and run_started events.

    Prerequisites:
      - Flask server must be running (python 04_webhooks.py)
      - Your server's URL must be registered in Console → Manage → Webhooks
      - The registered endpoint must be subscribed to session events
    """
    print("=== Trigger Mode: Creating test session ===\n")

    print("Creating minimal agent + environment...")
    agent = client.beta.agents.create(
        name="Webhook Test Agent",
        model="claude-haiku-4-5",
        system="You are a test agent. When given a task, complete it in one sentence.",
        tools=[{"type": "agent_toolset_20260401"}],
    )
    environment = client.beta.environments.create(
        name="webhook-test",
        config={"type": "cloud", "networking": {"type": "limited",
            "allow_package_managers": False, "allow_mcp_servers": False, "allowed_hosts": []}},
    )
    print(f"  Agent: {agent.id}")
    print(f"  Environment: {environment.id}")

    print("\nCreating session and sending a message...")
    session = client.beta.sessions.create(
        agent=agent.id,
        environment_id=environment.id,
        title="Webhook test session",
    )
    client.beta.sessions.events.send(
        session.id,
        events=[{"type": "user.message", "content": "Write the word 'hello' to a file called hello.txt."}],
    )
    print(f"  Session: {session.id}")
    print("\nExpect these webhook events on your server:")
    print("  1. session.status_run_started  — agent begins execution")
    print("  2. session.status_idled        — agent finishes and waits")

    # Wait for events to fire, then clean up
    print("\nWaiting 60s for events to fire...")
    time.sleep(60)

    print("\nCleaning up trigger resources...")
    client.beta.agents.archive(agent.id)
    client.beta.environments.archive(environment.id)
    print("  Done.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "trigger":
        trigger_test_events()
        return

    # Default mode: start the webhook server
    signing_key = os.environ.get("ANTHROPIC_WEBHOOK_SIGNING_KEY", "")
    if not signing_key.startswith("whsec_"):
        print("WARNING: ANTHROPIC_WEBHOOK_SIGNING_KEY is not set or invalid.")
        print("  Set it to the whsec_... secret from Console → Manage → Webhooks.")
        print("  The server will start, but all webhook deliveries will be rejected.\n")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting webhook server on port {port}")
    print(f"Register this URL in Console: https://your-domain.com/webhook")
    print(f"For local testing: ngrok http {port}")
    print(f"Then run in a second terminal: python 04_webhooks.py trigger\n")

    # Use debug=False in production. A production deployment should use
    # gunicorn or uvicorn behind a reverse proxy, not Flask's dev server.
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
