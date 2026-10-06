"""Intake cannot silently drop verification, including across profile scope."""
import argparse
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_acceptance_policy import set_verification_required, verification_required
from hermes_cli.kanban_db_connect import connect_closing


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-a"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()


def test_intake_is_bound_to_actual_database_and_preserves_existing_tasks(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    with connect_closing() as a:
        old = kb.create_task(a, title="Existing history", idempotency_key="existing")
        original = tuple(a.execute("SELECT * FROM tasks WHERE id=?", (old,)).fetchone())
        set_verification_required(a, True)
        assert kb.create_task(a, title="Retry existing", idempotency_key="existing") == old
        for contract in (None, "local-only"):
            with pytest.raises(ValueError, match="No task was created"):
                kb.create_task(a, title="Missing verifier", completion_contract=contract)
        assert a.execute("SELECT count(*) FROM tasks").fetchone()[0] == 1
        assert tuple(a.execute("SELECT * FROM tasks WHERE id=?", (old,)).fetchone()) == original
        # A different profile's current board must neither bypass A nor restrict B.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-b"))
        kb.init_db()
        with connect_closing() as b:
            assert not verification_required(b)
            assert kb.create_task(b, title="Legacy board B")
            assert verification_required(a)
            with pytest.raises(ValueError, match="repair intake"):
                kb.create_task(a, title="Cannot escape through profile B")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-a"))
        assert verification_required(a)
        artifact = tmp_path / "output.json"
        manifest = tmp_path / "acceptance.json"
        manifest.write_text(json.dumps({"schema":"hermes.readback/v1", "outcome":"Consumer reads count 2", "checks":[{"id":"count","kind":"json_equals","path":str(artifact),"pointer":["count"],"expected":2}]}))
        tid = kb.create_task(a, title="Verified work", completion_contract="verify:"+str(manifest))
        run = kb.claim_task(a, tid).current_run_id
        assert not kb.complete_task(a, tid, summary="premature", expected_run_id=run)
        artifact.write_text('{"count":2}')
        assert kb.complete_task(a, tid, summary="Actual consumer readback", expected_run_id=run)
        assert kb.create_task(a, title="Existing CI verifier", completion_contract="owner/repository")


def test_cli_and_tool_use_the_same_intake_policy(tmp_path, monkeypatch, capsys):
    _home(tmp_path, monkeypatch)
    from hermes_cli.kanban import build_parser, kanban_command
    from tools import kanban_tools
    from tools.registry import registry
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban","acceptance-policy","--require-verified","--json"])
    assert kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out)["require_verified_completion"]
    denied = json.loads(registry.dispatch("kanban_create", {"title":"Agent forgot acceptance", "assignee":"worker"}))
    assert "requires verified completion" in denied["error"]
    with connect_closing() as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    args = parser.parse_args(["kanban","create","CLI forgot acceptance","--json"])
    assert kanban_command(args) != 0
    assert "No task was created" in capsys.readouterr().err


def test_present_but_invalid_policy_does_not_silently_disable_verification(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    with connect_closing() as conn:
        set_verification_required(conn, True)
        conn.execute("DELETE FROM board_acceptance_policy")
        with pytest.raises(ValueError, match="policy is invalid"):
            kb.create_task(conn, title="Invalid policy cannot dispatch")
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
