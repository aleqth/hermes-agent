"""Durable acceptance for ordinary work sessions, without a per-tool subprocess.

An enabled profile records each original request before the model runs. The
model uses this module's CLI to declare frozen checks, recover, and verify.
An open task survives turns/restarts; a final answer cannot erase it. This is
artifact acceptance, not an automatic semantic judge or a security sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.kanban_readback_acceptance import collect_readback, freeze_contract
from hermes_constants import get_hermes_home


def _now():
    return datetime.now(timezone.utc).isoformat()


def _user_stop_request(request):
    """Recognize standalone user control messages, never model checkpoint prose.

    Keep this conservative: "stop wasting time and fix it" is still work.
    Longer/ambiguous instructions remain subject to normal agent judgment.
    """
    return bool(re.fullmatch(
        r"(?:(?:please|sorry|oops|oh)[ ,]+)*"
        r"(?:stop|pause|cancel|wrong (?:chat|conversation))"
        r"(?: (?:here|now|please|lol|thanks))?[.! ]*",
        request.strip(), re.I,
    ))


def _answer_only_request(request):
    """Conservative, non-model exemption for short explanatory questions.

    Ambiguous/action-bearing requests stay on the work path. This is a routing
    convenience, not a universal semantic classifier or a security boundary.
    """
    request = request.strip()
    if request.startswith("/answer "):
        return True
    if request.casefold().rstrip(".! ") in {"hi", "hello", "thanks", "thank you"}:
        return True
    if len(request) > 400 or re.search(r"[\n;]", request):
        return False
    if re.search(r"\b(?:and|also|then|fix|repair|build|create|implement|finish|deploy|send|update|write|delete|run|execute|edit|change|add|modify|install|publish|push|commit|buy|pay|remove|complete|ship)\b", request, re.I):
        return False
    return bool(re.fullmatch(r"(?:what (?:is|are|does|did)|why (?:is|are|does|did)|how (?:does|do|is|are)|who (?:is|are)|when (?:is|was|did)|where (?:is|are)|explain)\s+[^?!]+[?.]?", request, re.I))


@contextmanager
def _db(home):
    root = Path(home) / "interactive-acceptance"
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "tasks.db", timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, session TEXT NOT NULL, request TEXT NOT NULL,
          contract TEXT, state TEXT NOT NULL, owner_turn TEXT NOT NULL,
          next_action TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS turns (
          id TEXT PRIMARY KEY, session TEXT NOT NULL, request TEXT NOT NULL,
          task TEXT, disposition TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS current_turn (
          session TEXT PRIMARY KEY, turn TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY, task TEXT, turn TEXT NOT NULL,
          kind TEXT NOT NULL, payload TEXT NOT NULL, observed_at TEXT NOT NULL);
        """)
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _event(conn, turn, kind, payload):
    conn.execute("INSERT INTO events(task,turn,kind,payload,observed_at) VALUES(?,?,?,?,?)",
                 (turn["task"], turn["id"], kind, json.dumps(payload, sort_keys=True), _now()))


def _owned(conn, session, turn_id):
    turn = conn.execute("SELECT * FROM turns WHERE id=? AND session=?", (turn_id, session)).fetchone()
    current = conn.execute("SELECT turn FROM current_turn WHERE session=?", (session,)).fetchone()
    if not turn or not current or current[0] != turn_id:
        raise ValueError("superseded or unknown turn; resume the current session task")
    return dict(turn)


def start_turn(home, session, request):
    if not session or not isinstance(request, str) or not request.strip():
        raise ValueError("session identity and original request are required")
    with _db(home) as conn:
        turn_id = uuid.uuid4().hex
        # A new request may have a different outcome. Never silently use an old
        # contract to accept it. Resuming is explicit and auditable below.
        disposition = "user_stop" if _user_stop_request(request) else "undecided"
        conn.execute("INSERT INTO turns VALUES(?,?,?,?,?,?)", (turn_id, session, request, None, disposition, _now()))
        conn.execute("INSERT OR REPLACE INTO current_turn VALUES(?,?)", (session, turn_id))
        turn = _owned(conn, session, turn_id)
        _event(conn, turn, "intake", {"resumed_task": None})
        return turn


def operate(home, session, turn_id, action, *, manifest=None, reason=None, next_action=None, task_id=None):
    """All lifecycle writes recheck the owning turn under the same transaction."""
    with _db(home) as conn:
        turn = _owned(conn, session, turn_id)
        if turn["disposition"] == "user_stop" and action != "status":
            raise ValueError("explicit user stop: preserve saved work until a new user request resumes it")
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (turn["task"],)).fetchone()
        if action == "resume":
            if task or not task_id or not isinstance(reason, str) or not reason.strip():
                raise ValueError("resume needs an unbound intake, saved task ID, and why this request continues that outcome")
            saved = conn.execute("SELECT * FROM tasks WHERE id=? AND session=? AND state='OPEN'", (task_id, session)).fetchone()
            if not saved:
                raise ValueError("no OPEN task with that ID in this session")
            conn.execute("UPDATE tasks SET owner_turn=? WHERE id=?", (turn_id, task_id))
            conn.execute("UPDATE turns SET task=?,disposition='work' WHERE id=?", (task_id, turn_id))
            turn["task"] = task_id
            _event(conn, turn, "resumed", {"reason": reason, "request": turn["request"]})
        elif action == "declare":
            contract = freeze_contract("verify:" + str(Path(manifest).resolve()))
            if task and task["contract"] != contract:
                raise ValueError("acceptance is frozen; repair the result, do not weaken or replace its checks")
            if not task:
                task_id = "iw_" + uuid.uuid4().hex
                conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)", (task_id, session, turn["request"], contract, "OPEN", turn_id, "execute declared work and verify actual artifacts", _now()))
                conn.execute("UPDATE turns SET task=?,disposition='work' WHERE id=?", (task_id, turn_id))
                turn["task"] = task_id
            _event(conn, turn, "declared", {"contract_sha256": hashlib.sha256(contract.encode()).hexdigest()})
        elif action == "verify":
            if not task:
                raise ValueError("declare acceptance before verifying work")
            result = collect_readback(task["contract"])
            conn.execute("UPDATE tasks SET state=?,next_action=? WHERE id=? AND owner_turn=?", ("VERIFIED" if result["ok"] else "OPEN", "recipient readback" if result["ok"] else result["detail"], task["id"], turn_id))
            _event(conn, turn, "readback", result)
        elif action == "answer":
            if task:
                raise ValueError("an existing work task cannot be replaced by an answer disposition; use checkpoint for a status-only reply")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("answer requires why this request is informational rather than accepted work")
            request = turn["request"].strip()
            if not _answer_only_request(request):
                raise ValueError("answer-only requires an unambiguous explanatory question, greeting, or explicit /answer prefix; model classification cannot exempt work")
            if conn.execute("SELECT 1 FROM tasks WHERE session=? AND state='OPEN'", (session,)).fetchone():
                raise ValueError("saved work remains OPEN; resume it and checkpoint a status-only reply")
            conn.execute("UPDATE turns SET disposition='answer' WHERE id=?", (turn_id,))
            _event(conn, turn, "answer_only", {"reason": reason})
        elif action == "checkpoint":
            if not task or not reason or not next_action:
                raise ValueError("checkpoint requires an existing task, observed obstruction and next executable action")
            conn.execute("UPDATE tasks SET state='OPEN',next_action=? WHERE id=?", (next_action, task["id"]))
            _event(conn, turn, "checkpoint", {"reason": reason, "next_action": next_action})
        elif action != "status":
            raise ValueError("unknown acceptance action")
        return _snapshot(conn, _owned(conn, session, turn_id))


def _snapshot(conn, turn):
    task = conn.execute("SELECT * FROM tasks WHERE id=?", (turn["task"],)).fetchone()
    events = conn.execute("SELECT seq,kind,payload,observed_at FROM events WHERE task=? OR turn=? ORDER BY seq DESC LIMIT 8", (turn["task"], turn["id"])).fetchall()
    pending = conn.execute("SELECT id,request,contract,next_action FROM tasks WHERE session=? AND state='OPEN' ORDER BY created_at", (turn["session"],)).fetchall()
    return {"turn": dict(turn), "task": dict(task) if task else None, "open_tasks": [dict(row) for row in pending],
            "events": [{**dict(e), "payload": json.loads(e["payload"])} for e in reversed(events)]}


def _command(home, session, turn):
    package_root = Path(__file__).resolve().parents[1]
    return (f"PYTHONPATH={shlex.quote(str(package_root))} {shlex.quote(sys.executable)} -P -m agent.interactive_acceptance "
            f"--home {shlex.quote(str(home))} --session {shlex.quote(session)} --turn {shlex.quote(turn)}")


def prepare_turn(agent, user_message):
    """Append current-turn guidance; never change the cached system/history prefix."""
    agent._interactive_acceptance = None
    agent._interactive_acceptance_error = None
    agent._interactive_acceptance_nudges = 0
    agent._interactive_acceptance_decision = None
    from hermes_cli.config import load_config_readonly
    from agent.delegation_context import owned_kanban_task, is_delegated_child_context
    config = load_config_readonly().get("interactive_acceptance", {})
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return user_message
    platform = getattr(agent, "platform", "") or "cli"
    if platform not in config.get("platforms", ["cli", "desktop", "api", "telegram", "discord"]):
        return user_message
    if owned_kanban_task() or is_delegated_child_context() or getattr(agent, "_jev_doctor_claimed_status", None):
        return user_message
    from agent.message_content import flatten_message_text
    request = user_message if isinstance(user_message, str) else flatten_message_text(user_message)
    home = get_hermes_home()
    try:
        record = start_turn(home, str(agent.session_id), request)
        command = _command(home, record["session"], record["id"])
        agent._interactive_acceptance = {"home": str(home), "session": record["session"], "turn": record["id"], "command": command}
        snapshot = operate(home, record["session"], record["id"], "status")
        pending = snapshot["open_tasks"]
        state = ("Saved OPEN work: " + json.dumps(pending, ensure_ascii=False) if pending else "No existing work task.")
        guidance = ("\n\n[Hermes durable task intake]\n" + state + "\n"
            "For requested work, declare concrete acceptance BEFORE changing the result: write a hermes.readback/v1 manifest (outcome; checks with id, absolute path, kind file_sha256+sha256 or json_equals+pointer list+expected), then run the command below with declare --manifest /absolute/file.json. "
            "Use real requested artifact content, never a self-written passed flag. Keep the full user outcome; these artifact checks do not prove deployment or arbitrary semantic claims. "
            "Execute, then run verify. Failed verification means repair and retry in this SAME task. Existing authorization carries forward; do not ask again to do accepted local work. "
            "If this request continues a saved outcome, first run resume --task TASK_ID --reason 'how this request continues that outcome'; its acceptance stays frozen. For a distinct outcome declare a NEW manifest; older tasks remain OPEN. Never use an old task's checks to accept a new outcome. "
            "For a short explanatory question or greeting with no saved work, run answer --reason 'informational'; a conservative independent request check must accept it. The explicit user /answer prefix also selects that path. For an ambiguous question rejected by answer, declare checks for a saved answer artifact and deliver the answer; do not ask the user to rephrase. "
            "For a real external dependency, user stop, or status-only question about an existing task, run checkpoint --reason 'observed state' --next-action 'specific action'; it stays OPEN. "
            "Never use answer/checkpoint to abandon executable accepted work. A new turn loads saved OPEN state; explicitly resume the matching outcome.\n"
            + command + " <declare|resume|verify|status|answer|checkpoint>\n[/Hermes durable task intake]")
        if record["disposition"] == "user_stop":
            guidance = ("\n\n[Hermes durable task intake]\n"
                "The original user message explicitly stops work in this conversation. Acknowledge it and stop. "
                "Existing unfinished tasks remain saved. Do not resume work, create a handoff task merely to pass "
                "acceptance, or claim completion. A later user request can resume the original task.\n"
                "[/Hermes durable task intake]")
    except Exception as exc:
        agent._interactive_acceptance_error = f"intake unavailable: {type(exc).__name__}"
        guidance = "\n\n[Hermes task intake is unavailable. Preserve OPEN and report the actual storage failure; do not claim verified completion.]"
    if isinstance(user_message, str):
        return user_message + guidance
    return [*user_message, {"type": "text", "text": guidance}]


def inspect_turn(agent):
    binding = getattr(agent, "_interactive_acceptance", None)
    error = getattr(agent, "_interactive_acceptance_error", None)
    if isinstance(error, str) and error:
        return {"status": "OPEN", "reason": error}
    if not isinstance(binding, dict):
        return {"status": "not_applicable"}
    try:
        with _db(binding["home"]) as conn:
            turn = _owned(conn, binding["session"], binding["turn"])
            if turn["disposition"] == "user_stop":
                return {"status": "OPEN", "user_stopped": True,
                        "reason": "user requested a stop; unfinished work remains saved"}
            snapshot = _snapshot(conn, turn)
            task = snapshot["task"]
            if task:
                if task["owner_turn"] != turn["id"]:
                    raise ValueError("task ownership changed")
                if task["state"] == "VERIFIED":
                    readback = collect_readback(task["contract"])
                    if readback["ok"]:
                        return {"status": "VERIFIED", "task_id": task["id"], "readback": readback}
                    conn.execute("UPDATE tasks SET state='OPEN',next_action=? WHERE id=?", (readback["detail"], task["id"]))
                    _event(conn, turn, "final_readback_rejected", readback)
                return {"status": "OPEN", "task_id": task["id"], "reason": task["next_action"]}
            if turn["disposition"] == "answer":
                return {"status": "ANSWER_ONLY", "scope": "informational reply, no task completion verified"}
            return {"status": "OPEN", "reason": "request has no declared acceptance or informational disposition"}
    except Exception as exc:
        return {"status": "OPEN", "reason": f"acceptance unavailable: {type(exc).__name__}: {exc}"}


def stop_nudge(agent):
    decision = inspect_turn(agent)
    if decision.get("user_stopped") or decision["status"] != "OPEN" or getattr(agent, "_interactive_acceptance_nudges", 0) >= 2:
        return None
    binding = getattr(agent, "_interactive_acceptance", None)
    if not binding:
        return None
    agent._interactive_acceptance_nudges = getattr(agent, "_interactive_acceptance_nudges", 0) + 1
    return (f"Your existing task remains OPEN: {decision['reason']}. Continue the authorized next action now, recover failed checks in this turn, and verify the actual outcome. "
            f"Do not ask again for existing authorization. Inspect saved state: {binding['command']} status. "
            "For a genuine external dependency or explicit user stop/status-only request, preserve a checkpoint; never turn that into a DONE claim.")


def guard_final(agent, response):
    decision = inspect_turn(agent)
    agent._interactive_acceptance_decision = decision
    if decision.get("user_stopped"):
        return "Stopped. Any unfinished work remains saved and has not been marked complete."
    if decision["status"] == "OPEN":
        return f"Task OPEN — {decision.get('task_id', 'intake')}: {decision['reason']}. The task is saved for continuation; completion has not been verified."
    return response


def finish_user_stop(agent, messages, conversation_history=None):
    """Persist a control reply without invoking a model, tool, or output hook.

    The caller has already staged the original user row. Use the ordinary
    persister so staged CLI rows and gateway history keep their dedup markers.
    """
    from agent.message_metadata import append_message
    decision = inspect_turn(agent)
    if not decision.get("user_stopped"):
        raise ValueError("no original user stop at this intake")
    response = guard_final(agent, "")
    append_message(messages, {"role": "assistant", "content": response})
    agent._last_turn_usage = None
    agent._session_messages = messages
    try:
        persisted = agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        persisted = False
    error = "stop_transcript_persistence_failed" if persisted is False else None
    # The control turn was handled; its saved work task remains OPEN. Marking
    # this as an interruption suppresses its acknowledgement in chat gateways.
    return {"final_response": response, "messages": messages, "api_calls": 0,
            "completed": not bool(error), "partial": False, "interrupted": False,
            "failed": bool(error), "error": error, "turn_exit_reason": "user_stop",
            "agent_persisted": persisted is True, "interactive_acceptance": decision}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--session", required=True)
    parser.add_argument("--turn", required=True)
    parser.add_argument("action", choices=["declare", "resume", "verify", "status", "answer", "checkpoint"])
    parser.add_argument("--task")
    parser.add_argument("--manifest")
    parser.add_argument("--reason")
    parser.add_argument("--next-action")
    args = parser.parse_args(argv)
    try:
        result = operate(args.home or get_hermes_home(), args.session, args.turn, args.action,
                         manifest=args.manifest, reason=args.reason, next_action=args.next_action, task_id=args.task)
        print(json.dumps(result, indent=2))
        return 1 if args.action == "verify" and result["task"]["state"] != "VERIFIED" else 0
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc), "task_remains": "OPEN"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
