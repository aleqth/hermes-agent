"""A worker cannot replace consumer readback with its own completion assertion."""
import hashlib
import json

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


def _manifest(tmp_path):
    artifact = tmp_path / "result.json"
    expected = {"result": "verified", "count": 3}
    data = json.dumps(expected).encode()
    manifest = tmp_path / "acceptance.json"
    manifest.write_text(json.dumps({
        "schema": "hermes.readback/v1", "outcome": "Persist the requested usable result",
        "checks": [
            {"id": "exact-artifact", "kind": "file_sha256", "path": str(artifact),
             "sha256": hashlib.sha256(data).hexdigest()},
            {"id": "consumer-value", "kind": "json_equals", "path": str(artifact),
             "pointer": ["result"], "expected": "verified"}]}))
    return manifest, artifact, data


def test_worker_rejection_recovery_and_restart_use_frozen_readback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    from tools import kanban_tools  # registers the real handlers
    from tools.registry import registry
    manifest, artifact, data = _manifest(tmp_path)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Return verified output", completion_contract="verify:" + str(manifest))
        run = kb.claim_task(conn, tid)
        run_id = run.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    # Intake is frozen in SQLite, not a mutable source manifest or worker receipt.
    manifest.write_text("{}")
    args = {"summary": "Done; all checks passed", "metadata": {"receipt": {"ok": True}}}
    denied = json.loads(registry.dispatch("kanban_complete", args))
    assert "retry kanban_complete in this same run" in denied["error"]
    with connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).current_run_id == run_id
    artifact.write_text('{"result":"wrong","count":3}')
    assert "error" in json.loads(registry.dispatch("kanban_complete", args))
    artifact.write_bytes(data)
    accepted = json.loads(registry.dispatch("kanban_complete", args))
    assert accepted.get("task_id") == tid and "error" not in accepted
    with connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
        rows = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='artifact_acceptance' ORDER BY id", (tid,)).fetchall()
        receipts = [json.loads(r[0]) for r in rows]
        assert [r["ok"] for r in receipts] == [False, False, True]
        assert len({r["contract_sha256"] for r in receipts}) == 1
        assert all(c["ok"] for c in receipts[-1]["checks"])


def test_acceptance_survives_handoffs_and_rejects_stale_artifacts_or_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    from hermes_cli import kanban_readback_acceptance as readback
    manifest, artifact, data = _manifest(tmp_path)
    kb.init_db()
    with connect() as conn:
        for state in ("ready", "blocked", "review"):
            tid = kb.create_task(conn, title=state, completion_contract="verify:" + str(manifest))
            if state == "blocked":
                assert kb.block_task(conn, tid, reason="External dependency")
            elif state == "review":
                assert kb.request_review(conn, tid, summary="Implementation ready")
            assert not kb.complete_task(conn, tid, summary="Assert success")
            assert kb.get_task(conn, tid).status == state
            artifact.write_bytes(data)
            assert kb.complete_task(conn, tid, summary="Recipient output verified")
            artifact.unlink()
        tid = kb.create_task(conn, title="stale", completion_contract="verify:" + str(manifest))
        first = kb.claim_task(conn, tid).current_run_id
        assert kb.block_task(conn, tid, reason="Reassign")
        assert kb.unblock_task(conn, tid)
        second = kb.claim_task(conn, tid).current_run_id
        artifact.write_bytes(data)
        assert not kb.complete_task(conn, tid, summary="Old run", expected_run_id=first)
        original = readback.collect_readback
        def changed_after_read(contract):
            receipt = original(contract)
            artifact.write_text("changed after verification")
            return receipt
        with monkeypatch.context() as patch:
            patch.setattr(readback, "collect_readback", changed_after_read)
            assert not kb.complete_task(conn, tid, summary="Stale readback", expected_run_id=second)
        assert kb.get_task(conn, tid).status == "running"
        assert "stale" in kb.get_task(conn, tid).last_failure_error
        artifact.write_bytes(data)
        assert kb.complete_task(conn, tid, summary="Fresh readback", expected_run_id=second)
        legacy = kb.create_task(conn, title="Existing local task", completion_contract="local-only")
        assert kb.complete_task(conn, legacy, summary="Existing local behavior")


def test_completed_review_edit_retains_worker_provenance_and_verifier_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    manifest, artifact, data = _manifest(tmp_path)
    artifact.write_bytes(data)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Review the delivered artifact", completion_contract="verify:" + str(manifest))
        run = kb.claim_task(conn, tid).current_run_id
        assert kb.complete_task(conn, tid, summary="Worker readback", expected_run_id=run,
            metadata={"worker_session_id": "original-worker-session", "artifacts": [str(artifact)], "review": "pending"})
        before = [tuple(r) for r in conn.execute("SELECT id,kind,run_id,payload FROM task_events WHERE task_id=? AND kind IN ('artifact_acceptance','completed') ORDER BY id", (tid,))]
        assert kb.edit_task(conn, tid, result="Independent review recorded", metadata={"review": "accepted", "review_revision": "v2"})
        meta = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run,)).fetchone()[0])
        assert meta["worker_session_id"] == "original-worker-session"
        assert meta["artifacts"] == [str(artifact)]
        assert meta["review"] == "accepted" and meta["review_revision"] == "v2"
        after = [tuple(r) for r in conn.execute("SELECT id,kind,run_id,payload FROM task_events WHERE task_id=? AND kind IN ('artifact_acceptance','completed') ORDER BY id", (tid,))]
        assert after == before
        assert kb.get_task(conn, tid).status == "done"
