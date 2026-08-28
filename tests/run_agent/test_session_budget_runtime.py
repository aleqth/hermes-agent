from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.session_budget import SessionBudgetDecision


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_provider_budget_block_makes_zero_provider_calls():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    blocked = SessionBudgetDecision(
        "block", "Provider call blocked before spend: test rail", 1, 2, 3, 4, 5
    )
    with (
        patch(
            "agent.session_budget.evaluate_provider_call_budget",
            return_value=blocked,
        ),
        patch.object(agent, "_persist_session"),
    ):
        result = agent.run_conversation("do the expensive thing")

    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert "blocked before spend" in result["final_response"]
    assert not agent.client.chat.completions.create.called


def test_auto_rollover_forces_child_segment_then_calls_provider():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.compression_enabled = True
    agent.compression_in_place = True
    agent.session_id = "session-parent"
    agent.session_total_tokens = 10_000_001
    agent.session_input_tokens = 9_900_000
    agent.session_output_tokens = 100_001
    agent.session_prompt_tokens = 9_900_000
    agent.session_completion_tokens = 100_001
    agent.session_cache_read_tokens = 50_000
    agent.session_cache_write_tokens = 1_000
    agent.session_reasoning_tokens = 42
    agent.session_api_calls = 12
    agent.session_estimated_cost_usd = 3.14
    agent.session_cost_status = "estimated"
    agent.session_cost_source = "test"
    message = SimpleNamespace(
        content="continued after handoff",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="gemini/test",
        usage=None,
    )
    rollover = SessionBudgetDecision(
        "rollover", "segment complete", 9_990_000, 10_010_000, 70, 0, 0, True
    )
    ok = SessionBudgetDecision("ok", "fresh segment", 0, 20_000, 0, 0, 0, True)
    compression_modes = []

    def _evaluate_budget(*_args, **_kwargs):
        if agent.session_id == "session-parent":
            return rollover
        assert agent.session_id == "session-child"
        for counter in (
            "session_total_tokens",
            "session_input_tokens",
            "session_output_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_api_calls",
        ):
            assert getattr(agent, counter) == 0
        assert agent.session_estimated_cost_usd == 0.0
        assert agent.session_cost_status == "unknown"
        assert agent.session_cost_source == "none"
        return ok

    def _rotate(messages, system_message, **_kwargs):
        compression_modes.append(agent.compression_in_place)
        agent.session_id = "session-child"
        agent._last_compaction_in_place = False
        return list(messages), "durable handoff prompt"

    history = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "working"},
    ]
    with (
        patch(
            "agent.session_budget.evaluate_provider_call_budget",
            side_effect=_evaluate_budget,
        ) as evaluate_budget,
        patch.object(agent.context_compressor, "should_compress", return_value=False),
        patch.object(agent, "_compress_context", side_effect=_rotate),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "keep going", conversation_history=history
        )

    assert compression_modes == [False]
    assert agent.compression_in_place is True
    assert agent.session_id == "session-child"
    assert result["completed"] is True
    assert result["final_response"] == "continued after handoff"
    assert evaluate_budget.call_count == 2
    assert agent.client.chat.completions.create.call_count == 1
