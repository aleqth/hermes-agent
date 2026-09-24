"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def owned_run_state() -> dict | None:
    """Read the dispatcher's durable run, never infer acceptance from a tool call.

    No migration, model request, or subprocess on this turn-end path. A dead or
    stale worker must never resume its successor's work. Missing run identity is
    explicitly unknown; a tool invocation does not fill that gap.
    """
    tid = owned_kanban_task()
    if not tid:
        return None
    state = {"task_id": tid, "status": "unavailable", "error": "Cannot read this worker's owned board run."}
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
        from hermes_cli import kanban_db as kb
        with closing(sqlite3.connect(kb.kanban_db_path().resolve().as_uri() + "?mode=ro",
                                    uri=True, timeout=1)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")  # One snapshot for run identity + task details.
            status = kb.goal_run_status(conn, tid, run_id)
            task = kb.get_task(conn, tid)
            state.update(status=status or "superseded", run_id=run_id,
                         error=(task.last_failure_error or "") if task and status == "running" else "")
    except (OSError, ValueError, sqlite3.Error):
        pass
    return state


def guard_kanban_final_response(agent, final_response):
    """A bounded recovery exit stays visibly OPEN, including persisted output.

    This guards worker lifecycle truth, not arbitrary chat or semantic proof.
    It does not mark the card blocked or spend an additional model call.
    """
    state = owned_run_state()
    agent._kanban_terminal_rejected_reason = None
    agent._kanban_terminal_status = state["status"] if state else "not_applicable"
    if state and state["status"] in {"running", "unavailable", "superseded"}:
        detail = state["error"] or ("This run no longer owns the task." if state["status"] == "superseded"
                                    else "No terminal board transition was accepted.")
        agent._kanban_terminal_rejected_reason = detail
        return (f"Task {state['task_id']} remains OPEN for this worker: {detail} "
                "Work and acceptance remain on the board. Resume the same task from its saved context; "
                "do not recreate it or claim completion.")
    return final_response


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
    ):
        return None

    state = owned_run_state()
    if state and state["status"] not in {"running", "unavailable"}:
        return None
    tid = state["task_id"] if state else (task_id or "this task")
    if state and state["status"] == "unavailable":
        return (
            f"[System: Task `{tid}` ownership could not be verified. Use kanban_show to read the "
            "saved task and run. Do not mutate it without current ownership or report it complete. "
            "If the board is unavailable, preserve the exact error and next recovery action.]"
        )
    detail = (state or {}).get("error", "")
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still running in your owned run. A called or rejected terminal "
        "tool is not an accepted handoff. The saved board state is authoritative.\n"
        + (f"Last failure: {detail[:4096]}\n" if detail else "") +
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked by an external dependency you cannot resolve. A failed query or rejected "
        "check calls for inspection and repair in this run. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Use kanban_comment to save changed paths, observed evidence, the failed check and "
        "next executable action before yielding. Preserve this task ID and workspace. "
        "Two identical failures require a changed hypothesis; do not repeat unchanged work.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
