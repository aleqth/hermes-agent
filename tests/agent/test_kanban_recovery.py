"""Rejected terminal calls, stubborn narration, and restart cannot erase work."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.kanban_stop import build_kanban_stop_nudge, owned_run_state
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


def _task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(key, raising=False)
    artifact = tmp_path / "deliverable.json"
    manifest = tmp_path / "acceptance.json"
    manifest.write_text(json.dumps({"schema": "hermes.readback/v1", "outcome": "Consumer gets count 3",
        "checks": [{"id": "consumer-count", "kind": "json_equals", "path": str(artifact),
                    "pointer": ["count"], "expected": 3}]}))
    with connect_closing() as conn:
        tid = kb.create_task(conn, title="Make the deliverable", completion_contract="verify:" + str(manifest))
        run = kb.claim_task(conn, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    return tid, run, artifact, manifest


@pytest.mark.parametrize("repair", [True, False])
def test_actual_turn_recovers_or_exits_open_with_bounded_cost(tmp_path, monkeypatch, repair):
    from tools import kanban_tools
    from tools.registry import registry
    from run_agent import AIAgent
    from hermes_state import SessionDB

    tid, run, artifact, _ = _task(tmp_path, monkeypatch)
    rejected = registry.dispatch("kanban_complete", {"summary": "Done"})
    assert "error" in json.loads(rejected)
    history = [{"role": "assistant", "tool_calls": [{"function": {"name": "kanban_complete"}}]},
               {"role": "tool", "name": "kanban_complete", "content": rejected}]
    assert "consumer-count" in build_kanban_stop_nudge(messages=history)

    db = SessionDB(db_path=tmp_path / "home" / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(session_id="worker-recovery", session_db=db, api_key="test-key",
                        base_url="https://example.invalid/v1", provider="openai-compat", model="test/model",
                        max_iterations=8, quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "stable prompt"
    agent.save_trajectories = False
    agent.compression_enabled = False
    calls = []
    def model_response(kwargs):
        calls.append(kwargs)
        if repair and len(calls) == 2:
            artifact.write_text('{"count":3}')
            assert json.loads(registry.dispatch("kanban_complete", {"summary": "Readback passed"}))["ok"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="DONE", tool_calls=None), finish_reason="stop")], model="test/model", usage=None)
    agent._interruptible_api_call = model_response
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("Produce the accepted deliverable")
    assert len(calls) == (2 if repair else 3)  # Fixed two-nudge bound; no judge call.
    assert all(c["messages"][0]["content"] == "stable prompt" for c in calls)
    assert result["completed"] is repair
    saved = db.get_messages("worker-recovery")[-1]["content"]
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
    if repair:
        assert saved == "DONE" and task.status == "done"
    else:
        assert "remains OPEN" in saved and "consumer-count" in saved
        assert result["failure_reason"] == "kanban_task_open"
        assert task.status == "running" and task.current_run_id == run
    db.close()


def test_restart_mounts_frozen_contract_and_checkpoint_and_fences_old_worker(tmp_path, monkeypatch):
    from hermes_cli.kanban_db_dispatch import _record_task_failure
    tid, run, artifact, manifest = _task(tmp_path, monkeypatch)
    with connect_closing() as conn:
        assert not kb.complete_task(conn, tid, summary="DONE", expected_run_id=run)
        kb.add_comment(conn, tid, "first-worker", "Schema discovered: deliverable.json has count. Next: write actual count 3 and retry.")
        _record_task_failure(conn, tid, "worker process exited", outcome="failed", release_claim=True, end_run=True)
        successor = kb.claim_task(conn, tid).current_run_id
    manifest.write_text("{}")
    assert owned_run_state()["status"] == "superseded"
    assert build_kanban_stop_nudge(messages=[]) is None
    # A fresh interpreter, no transcript and no source manifest, reads everything
    # needed from the same durable card and completes through the real board gate.
    child = r'''
import json,os
from pathlib import Path
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
from agent.kanban_stop import owned_run_state
tid=os.environ['HERMES_KANBAN_TASK'];run=int(os.environ['HERMES_KANBAN_RUN_ID'])
with connect_closing() as conn:
 context=kb.build_worker_context(conn,tid)
 assert 'consumer-count' in context and 'Schema discovered' in context and 'Latest verifier observation' in context
 task=kb.get_task(conn,tid);spec=json.loads(task.completion_contract[len('verify-v1:'):])
 check=spec['checks'][0];Path(check['path']).write_text(json.dumps({'count':check['expected']}))
 assert kb.complete_task(conn,tid,summary='Restarted consumer readback',expected_run_id=run)
print(json.dumps({'status':owned_run_state()['status'],'contract_retained':True,'checkpoint_retained':True}))
'''
    env = {**os.environ, "HERMES_KANBAN_RUN_ID": str(successor)}
    proc = subprocess.run([sys.executable, "-c", child], env=env, cwd=Path(__file__).resolve().parents[2],
                          text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(proc.stdout.splitlines()[-1]) == {"status": "done", "contract_retained": True, "checkpoint_retained": True}
    with connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert not kb.complete_task(conn, tid, summary="stale overwrite", expected_run_id=run)
