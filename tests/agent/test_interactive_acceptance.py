"""Ordinary turns must retain work, recover, and persist truthful completion."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import interactive_acceptance as ia


def _manifest(root):
    artifact = root / "answer.json"
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": "hermes.readback/v1", "outcome": "Consumer receives the computed total",
        "checks": [{"id": "total", "path": str(artifact), "kind": "json_equals", "pointer": ["total"], "expected": 42}]}))
    return artifact, manifest


@pytest.mark.parametrize("repair", [True, False])
def test_ordinary_installed_loop_recovers_or_persists_open(tmp_path, monkeypatch, repair):
    from run_agent import AIAgent
    from hermes_state import SessionDB
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("interactive_acceptance:\n  enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB"):
        monkeypatch.delenv(name, raising=False)
    artifact, manifest = _manifest(tmp_path)
    db = SessionDB(db_path=home / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(session_id="interactive-worker", session_db=db, api_key="test-key",
                        base_url="https://example.invalid/v1", provider="openai-compat", model="test/model",
                        max_iterations=8, quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent._cached_system_prompt = "stable prompt"
    agent.save_trajectories = False
    agent.compression_enabled = False
    calls = []
    def response(kwargs):
        calls.append(kwargs)
        binding = agent._interactive_acceptance
        if len(calls) == 1:
            ia.operate(home, binding["session"], binding["turn"], "declare", manifest=manifest)
            assert ia.operate(home, binding["session"], binding["turn"], "verify")["task"]["state"] == "OPEN"
        elif repair:
            artifact.write_text('{"total":42}')
            assert ia.operate(home, binding["session"], binding["turn"], "verify")["task"]["state"] == "VERIFIED"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="DONE", tool_calls=None), finish_reason="stop")], model="test/model", usage=None)
    agent._interruptible_api_call = response
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("Compute and save the requested total")
    assert len(calls) == (2 if repair else 3)
    assert all(c["messages"][0]["content"] == "stable prompt" for c in calls)
    assert result["completed"] is repair
    assert result["interactive_acceptance"]["status"] == ("VERIFIED" if repair else "OPEN")
    messages = db.get_messages("interactive-worker")
    assert messages[0]["content"] == "Compute and save the requested total"
    assert messages[-1]["content"] == ("DONE" if repair else result["final_response"])
    assert repair or "Task OPEN" in messages[-1]["content"]
    if repair:
        artifact.write_text('{"total":0}')
        assert ia.guard_final(agent, "DONE").startswith("Task OPEN")
    db.close()


def test_restart_freezes_acceptance_fences_old_turn_and_isolates_profiles(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    artifact, manifest = _manifest(tmp_path)
    first = ia.start_turn(homes[0], "same-session", "Compute total")
    original = ia.operate(homes[0], first["session"], first["id"], "declare", manifest=manifest)
    other = ia.start_turn(homes[1], "same-session", "/answer Explain the format")
    ia.operate(homes[1], other["session"], other["id"], "answer", reason="informational question")
    resumed = ia.start_turn(homes[0], "same-session", "continue")
    assert resumed["task"] is None
    state = ia.operate(homes[0], resumed["session"], resumed["id"], "status")
    assert state["open_tasks"][0]["id"] == original["task"]["id"]
    ia.operate(homes[0], resumed["session"], resumed["id"], "resume", task_id=original["task"]["id"], reason="continue the saved computation")
    with pytest.raises(ValueError, match="superseded"):
        ia.operate(homes[0], first["session"], first["id"], "verify")
    with pytest.raises(ValueError, match="cannot be replaced"):
        ia.operate(homes[0], resumed["session"], resumed["id"], "answer", reason="give up")
    manifest.write_text(manifest.read_text().replace('42', '0'))
    with pytest.raises(ValueError, match="frozen"):
        ia.operate(homes[0], resumed["session"], resumed["id"], "declare", manifest=manifest)
    artifact.write_text('{"total":42}')
    command = [sys.executable, "-P", "-m", "agent.interactive_acceptance", "--home", str(homes[0]),
               "--session", resumed["session"], "--turn", resumed["id"], "verify"]
    proc = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                          text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(proc.stdout)["task"]["state"] == "VERIFIED"
    assert ia.operate(homes[1], other["session"], other["id"], "status")["task"] is None


def test_new_work_cannot_use_answer_escape_or_previous_contract(tmp_path):
    artifact, manifest = _manifest(tmp_path)
    first = ia.start_turn(tmp_path, "session", "Compute total")
    with pytest.raises(ValueError, match="explicit /answer"):
        ia.operate(tmp_path, "session", first["id"], "answer", reason="pretend it is informational")
    old = ia.operate(tmp_path, "session", first["id"], "declare", manifest=manifest)
    second = ia.start_turn(tmp_path, "session", "Build a distinct result")
    agent = SimpleNamespace(_interactive_acceptance={"home": str(tmp_path), "session": "session", "turn": second["id"]})
    artifact.write_text('{"total":42}')
    assert ia.inspect_turn(agent)["status"] == "OPEN"
    assert ia.operate(tmp_path, "session", second["id"], "status")["task"] is None
    manifest2 = tmp_path / "manifest2.json"
    manifest2.write_text(json.dumps({"schema":"hermes.readback/v1", "outcome":"Distinct result",
        "checks":[{"id":"different", "kind":"json_equals", "path":str(tmp_path / "different.json"), "pointer":["x"], "expected":7}]}))
    new = ia.operate(tmp_path, "session", second["id"], "declare", manifest=manifest2)
    assert new["task"]["id"] != old["task"]["id"]
    assert len(new["open_tasks"]) == 2
    assert new["task"]["request"] == "Build a distinct result"


@pytest.mark.parametrize("user_request,allowed", [
    ("What does DADA mean?", True), ("How does Hermes work?", True),
    ("Explain the format", True), ("What is DADA? Also fix the runtime", False),
    ("Can you fix it?", False), ("Build the result", False),
    ("What is broken? Then repair it", False), ("Please finish", False),
])
def test_answer_routing_uses_original_request(tmp_path, user_request, allowed):
    turn = ia.start_turn(tmp_path, "chat", user_request)
    if allowed:
        state = ia.operate(tmp_path, "chat", turn["id"], "answer", reason="explanation")
        assert state["turn"]["disposition"] == "answer"
    else:
        with pytest.raises(ValueError):
            ia.operate(tmp_path, "chat", turn["id"], "answer", reason="model calls this informational")


def test_disabled_next_turn_clears_previous_decision():
    agent = SimpleNamespace(_interactive_acceptance_decision={"status": "OPEN"})
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert ia.prepare_turn(agent, "Hello") == "Hello"
    assert agent._interactive_acceptance_decision is None
    assert ia.inspect_turn(agent)["status"] == "not_applicable"


@pytest.mark.parametrize("user_request", ["wrong chat lol", "Stop.", "please pause"])
def test_explicit_stop_ends_real_loop_once_and_preserves_open_work(tmp_path, monkeypatch, user_request):
    from run_agent import AIAgent
    from hermes_state import SessionDB
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("interactive_acceptance:\n  enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB"):
        monkeypatch.delenv(name, raising=False)
    artifact, manifest = _manifest(tmp_path)
    first = ia.start_turn(home, "stop-worker", "Compute the total")
    pending = ia.operate(home, "stop-worker", first["id"], "declare", manifest=manifest)["task"]
    db = SessionDB(db_path=home / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(session_id="stop-worker", session_db=db, api_key="test-key",
                        base_url="https://example.invalid/v1", provider="openai-compat", model="test/model",
                        max_iterations=4, quiet_mode=True, skip_context_files=True, skip_memory=True, platform="desktop")
    agent._cached_system_prompt = "stable prompt"
    agent.save_trajectories = False
    agent.compression_enabled = False
    calls = []
    def response(kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="DONE", tool_calls=None),
            finish_reason="stop")], model="test/model", usage=None)
    agent._interruptible_api_call = response
    with (patch("hermes_cli.plugins.invoke_hook", return_value=[]),
          patch("agent.turn_stop_gates._verify_on_stop_nudge", return_value="continue anyway") as other_gate):
        result = agent.run_conversation(user_request)
    assert len(calls) == 1
    other_gate.assert_not_called()
    assert calls[0]["messages"][0]["content"] == "stable prompt"
    assert result["completed"] is False
    assert result["interactive_acceptance"]["status"] == "OPEN"
    assert result["interactive_acceptance"]["user_stopped"] is True
    rows = db.get_messages("stop-worker")
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["content"] == user_request
    assert rows[-1]["content"] == result["final_response"] != "DONE"
    binding = agent._interactive_acceptance
    state = ia.operate(home, "stop-worker", binding["turn"], "status")
    assert state["task"] is None
    assert state["open_tasks"][0]["id"] == pending["id"]
    assert not artifact.exists()
    db.close()


def test_user_stop_cannot_be_forged_by_checkpoint_and_next_request_can_resume(tmp_path):
    artifact, manifest = _manifest(tmp_path)
    for request in ["stop wasting time and fix it", "wrong chat, but build it here", "explain the stop button"]:
        first = ia.start_turn(tmp_path, "chat", request)
        task = ia.operate(tmp_path, "chat", first["id"], "declare", manifest=manifest)["task"]
        ia.operate(tmp_path, "chat", first["id"], "checkpoint", reason="model calls this a user stop", next_action="finish the work")
        agent = SimpleNamespace(_interactive_acceptance={"home": str(tmp_path), "session": "chat", "turn": first["id"], "command": "status"})
        assert ia.stop_nudge(agent) is not None
    stopped = ia.start_turn(tmp_path, "chat", "wrong chat lol")
    for action in ["resume", "declare", "verify", "answer", "checkpoint"]:
        with pytest.raises(ValueError, match="user stop"):
            ia.operate(tmp_path, "chat", stopped["id"], action, manifest=manifest, task_id=task["id"],
                       reason="ignore stop", next_action="continue")
    resumed = ia.start_turn(tmp_path, "chat", "Continue the saved work here")
    ia.operate(tmp_path, "chat", resumed["id"], "resume", task_id=task["id"], reason="user resumed")
    artifact.write_text('{"total":42}')
    proc = subprocess.run([sys.executable, "-P", "-m", "agent.interactive_acceptance", "--home", str(tmp_path),
                          "--session", "chat", "--turn", resumed["id"], "verify"], cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                          text=True, capture_output=True, check=True, timeout=30)
    assert json.loads(proc.stdout)["task"]["state"] == "VERIFIED"


@pytest.mark.parametrize("verified", [False, True])
def test_codex_app_server_checks_acceptance_before_durable_final(tmp_path, monkeypatch, verified):
    from agent.codex_runtime import run_codex_app_server_turn
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from unittest.mock import MagicMock
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(session_id="codex-owned", session_db=db, api_key="test-key",
                        base_url="https://example.invalid/v1", provider="openai-compat", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    record = ia.start_turn(tmp_path, "codex-owned", "Compute total")
    artifact, manifest = _manifest(tmp_path)
    ia.operate(tmp_path, "codex-owned", record["id"], "declare", manifest=manifest)
    if verified:
        artifact.write_text('{"total":42}')
        ia.operate(tmp_path, "codex-owned", record["id"], "verify")
    agent._interactive_acceptance = {"home": str(tmp_path), "session": "codex-owned", "turn": record["id"]}
    agent._codex_session = MagicMock()
    agent._codex_session_prompt = None
    agent._codex_session.run_turn.return_value = SimpleNamespace(
        interrupted=False, error=None, thread_id="thread-1", turn_id="turn-1", tool_iterations=0,
        final_text="DONE", projected_messages=[{"role": "assistant", "content": "DONE", "api_content": "DONE"}],
        should_retire=False)
    messages = [{"role": "user", "content": "Compute total"}]
    agent._flush_messages_to_session_db(messages)
    result = run_codex_app_server_turn(agent, user_message="Compute total", original_user_message="Compute total",
                                      messages=messages, effective_task_id="task")
    assert result["completed"] is verified
    assert result["interactive_acceptance"]["status"] == ("VERIFIED" if verified else "OPEN")
    rows = db.get_messages("codex-owned")
    assert len(rows) == 2
    assert rows[-1]["content"] == result["final_response"]
    assert messages[-1]["api_content"] == result["final_response"]
    assert verified or result["final_response"].startswith("Task OPEN")
    assert agent.session_api_calls == 1
    db.close()
