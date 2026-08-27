from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.session_budget import (
    SessionBudgetDecision,
    evaluate_provider_call_budget,
    measure_inline_images,
)


def _agent(**overrides):
    values = {
        "session_id": "session-1",
        "session_total_tokens": 0,
        "session_cache_read_tokens": 0,
        "session_cache_write_tokens": 0,
        "session_api_calls": 0,
        "max_tokens": 32000,
        "model": "moonshotai/kimi-k3",
        "provider": "openrouter",
        "_session_db": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _policy(**overrides):
    values = {
        "enabled": True,
        "max_session_tokens": 1_000_000,
        "max_prompt_tokens": 180_000,
        "max_session_api_calls": 80,
        "max_inline_image_bytes": 12_000_000,
        "max_inline_images": 4,
        "warn_ratio": 0.75,
    }
    values.update(overrides)
    return values


def test_counts_openai_and_anthropic_inline_images():
    messages = [
        {"image_url": {"url": "data:image/png;base64,AAAA"}},
        {"source": {"type": "base64", "data": "BBBB"}},
    ]
    size, count = measure_inline_images(messages)
    assert count == 2
    assert size >= 4 + len("data:image/png;base64,AAAA")


def test_blocks_cache_heavy_resumed_session_before_provider_call():
    db = MagicMock()
    db.get_session.return_value = {
        "input_tokens": 120_000,
        "output_tokens": 10_000,
        "cache_read_tokens": 830_000,
        "cache_write_tokens": 0,
        "api_call_count": 30,
    }
    with patch("agent.session_budget._load_policy", return_value=_policy()):
        result = evaluate_provider_call_budget(
            _agent(_session_db=db),
            approx_input_tokens=20_000,
            request_messages=[],
        )
    assert result.blocked
    assert result.current_tokens == 960_000
    assert "before spend" in result.message


def test_blocks_api_call_runaway_even_when_tokens_are_small():
    with patch("agent.session_budget._load_policy", return_value=_policy()):
        result = evaluate_provider_call_budget(
            _agent(session_api_calls=80),
            approx_input_tokens=1000,
            request_messages=[],
        )
    assert result.blocked
    assert "80 API calls" in result.message


def test_blocks_replayed_screenshot_fanout():
    image = "data:image/png;base64," + ("A" * 100)
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(max_inline_images=2),
    ):
        result = evaluate_provider_call_budget(
            _agent(),
            approx_input_tokens=1000,
            request_messages=[{"content": [{"type": "image_url", "image_url": {"url": image}}]}] * 3,
        )
    assert result.blocked
    assert "3 inline images" in result.message


def test_disabled_policy_is_noop():
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(enabled=False, max_session_tokens=1),
    ):
        result = evaluate_provider_call_budget(
            _agent(session_total_tokens=10_000),
            approx_input_tokens=10_000,
            request_messages=[],
        )
    assert result.status == "ok"
    assert not result.blocked


def test_decision_block_property():
    decision = SessionBudgetDecision("block", "stop", 1, 2, 3, 4, 5)
    assert decision.blocked


def test_gemini_route_gets_larger_workload_without_widening_other_models():
    policy = _policy(
        model_overrides={"gemini": {"max_session_tokens": 10_000_000}}
    )
    with patch("agent.session_budget._load_policy", return_value=policy):
        gemini = evaluate_provider_call_budget(
            _agent(
                session_total_tokens=1_100_000,
                model="3.7-flash",
                provider="gemini",
            ),
            approx_input_tokens=1000,
            request_messages=[],
        )
        kimi = evaluate_provider_call_budget(
            _agent(session_total_tokens=1_100_000),
            approx_input_tokens=1000,
            request_messages=[],
        )

    assert gemini.status == "ok"
    assert not gemini.blocked
    assert kimi.blocked
    assert "1,000,000-token limit" in kimi.message
