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
