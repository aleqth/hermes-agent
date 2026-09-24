"""Board-scoped intake policy: require a verifier before creating new work.

Policy lives in the board database, so CLI, tools and profile workers use the
same rule even when their current-board pointer or HERMES_HOME differs. Existing
cards retain their declarations; enabling this never rewrites historical work.
"""
from __future__ import annotations

import sqlite3


def verification_required(conn: sqlite3.Connection) -> bool:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='board_acceptance_policy'").fetchone()
    if not exists:
        return False  # Legacy boards remain compatible until explicitly enabled.
    row = conn.execute("SELECT required FROM board_acceptance_policy WHERE singleton=1").fetchone()
    if row is None or row[0] not in (0, 1):
        raise ValueError("Board acceptance policy is invalid; restore its explicit setting before creating work.")
    return bool(row[0])


def set_verification_required(conn: sqlite3.Connection, required: bool) -> None:
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS board_acceptance_policy (singleton INTEGER PRIMARY KEY CHECK(singleton=1), required INTEGER NOT NULL CHECK(required IN (0,1)))")
        conn.execute("INSERT INTO board_acceptance_policy VALUES (1,?) ON CONFLICT(singleton) DO UPDATE SET required=excluded.required", (int(required),))


def enforce_new_task_contract(conn: sqlite3.Connection, contract: str) -> None:
    """Called inside the task-creation transaction, before any task is inserted."""
    if contract == "local-only" and verification_required(conn):
        raise ValueError(
            "This board requires verified completion for new tasks. No task was created. "
            "Declare completion_contract='verify:/absolute/manifest.json' with a hermes.readback/v1 "
            "outcome and consumer checks, or declare OWNER/REPO or an exact GitHub PR URL. "
            "Missing/local-only acceptance cannot dispatch work here; repair intake and retry."
        )


def command(args) -> int:
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli.kanban_output import _json_out

    require = bool(getattr(args, "require_verified", False))
    allow = bool(getattr(args, "allow_unverified", False))
    if require and allow:
        raise ValueError("Choose either --require-verified or --allow-unverified.")
    with connect_closing() as conn:
        if require or allow:
            set_verification_required(conn, require)
        state = {"require_verified_completion": verification_required(conn), "scope": "new_tasks", "existing_tasks_changed": False}
    if not _json_out(args, state):
        print("New tasks require verified completion." if state["require_verified_completion"] else "New tasks may use legacy local-only completion.")
    return 0
