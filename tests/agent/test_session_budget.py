from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.session_budget import (
    InlineImageSanitization,
    SessionBudgetDecision,
    evaluate_provider_call_budget,
    measure_inline_images,
    sanitize_inline_images_for_provider,
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


def _image(value: str):
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + value},
    }


def test_provider_copy_keeps_newest_unique_images_without_mutating_history():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "old"}, _image("A")]},
        {"role": "user", "content": [_image("B"), _image("A")]},
        {"role": "user", "content": [_image("C"), _image("D"), _image("E")]},
    ]
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(max_inline_images=4),
    ):
        sanitized, report = sanitize_inline_images_for_provider(_agent(), messages)

    assert report == InlineImageSanitization(
        original_count=6,
        original_bytes=measure_inline_images(messages)[0],
        kept_count=4,
        kept_bytes=measure_inline_images(sanitized)[0],
        duplicate_count=1,
        overflow_count=1,
    )
    assert measure_inline_images(messages)[1] == 6
    assert measure_inline_images(sanitized)[1] == 4
    assert "data:image/png;base64,A" in str(sanitized)
    assert "data:image/png;base64,B" not in str(sanitized)
    assert "data:image/png;base64,E" in str(sanitized)
    assert "Conversation history was preserved" in report.notice


def test_provider_copy_trims_oldest_images_to_byte_cap_but_keeps_oversize_newest():
    messages = [{"role": "user", "content": [_image("A" * 20), _image("B" * 20)]}]
    newest_size = len("data:image/png;base64," + ("B" * 20))
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(max_inline_images=4, max_inline_image_bytes=newest_size),
    ):
        sanitized, report = sanitize_inline_images_for_provider(_agent(), messages)
    assert report.kept_count == 1
    assert report.overflow_count == 1
    assert "data:image/png;base64,B" in str(sanitized)

    oversized = [{"role": "user", "content": [_image("Z" * 100)]}]
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(max_inline_image_bytes=10),
    ):
        retained, report = sanitize_inline_images_for_provider(_agent(), oversized)
        decision = evaluate_provider_call_budget(
            _agent(), approx_input_tokens=1000, request_messages=retained
        )
    assert report.kept_count == 1
    assert not report.changed
    assert decision.blocked
    assert "inline image payloads total" in decision.message


def test_provider_copy_counts_and_sanitizes_gemini_inline_data():
    image = {"inline_data": {"mime_type": "image/png", "data": "AAAA"}}
    messages = [{"role": "user", "parts": [image, image, image, image, image]}]
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(max_inline_images=4),
    ):
        sanitized, report = sanitize_inline_images_for_provider(_agent(), messages)
    assert report.original_count == 5
    assert report.kept_count == 1
    assert report.duplicate_count == 4
    assert measure_inline_images(sanitized) == (4, 1)


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


def test_gemini_segment_requests_rollover_instead_of_blocking():
    policy = _policy(
        model_overrides={
            "gemini": {
                "max_session_tokens": 10_000_000,
                "auto_rollover": True,
            }
        }
    )
    with patch("agent.session_budget._load_policy", return_value=policy):
        result = evaluate_provider_call_budget(
            _agent(
                session_total_tokens=9_980_000,
                model="gemini-3-flash",
                provider="gemini",
            ),
            approx_input_tokens=20_000,
            request_messages=[],
        )

    assert result.status == "rollover"
    assert result.rollover_requested
    assert not result.blocked
    assert "fresh child segment" in result.message


def test_auto_rollover_does_not_bypass_api_call_budget():
    with patch(
        "agent.session_budget._load_policy",
        return_value=_policy(auto_rollover=True),
    ):
        result = evaluate_provider_call_budget(
            _agent(session_api_calls=80, session_total_tokens=9_999_000),
            approx_input_tokens=20_000,
            request_messages=[],
        )

    assert result.blocked
    assert not result.rollover_requested
    assert "80 API calls" in result.message
