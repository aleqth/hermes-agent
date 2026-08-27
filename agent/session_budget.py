"""Fail-closed, pre-provider workload limits for long Hermes sessions.

Tool hooks run too late to prevent the model request that selected a tool.
This module therefore evaluates local, profile-scoped policy immediately
before each provider call.  It performs no network I/O and never mutates the
conversation, preserving prompt-cache stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SessionBudgetDecision:
    status: str
    message: str
    current_tokens: int
    projected_tokens: int
    current_api_calls: int
    inline_image_bytes: int
    inline_image_count: int

    @property
    def blocked(self) -> bool:
        return self.status == "block"


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _row_value(row: Any, key: str) -> int:
    if row is None:
        return 0
    try:
        value = row.get(key, 0) if isinstance(row, Mapping) else row[key]
    except (KeyError, TypeError, IndexError, AttributeError):
        return 0
    return _positive_int(value)


def measure_inline_images(value: Any) -> tuple[int, int]:
    """Return encoded payload bytes and image count in a request structure."""
    if isinstance(value, str):
        marker = "data:image/"
        start = value.find(marker)
        if start < 0:
            return 0, 0
        # Content-part URLs are normally the whole string.  When a data URL is
        # embedded in JSON tool arguments, counting the remainder is a safe
        # upper bound and avoids parsing a multi-megabyte string again.
        return len(value) - start, value.count(marker)
    if isinstance(value, list) or isinstance(value, tuple):
        total_bytes = 0
        total_count = 0
        for item in value:
            item_bytes, item_count = measure_inline_images(item)
            total_bytes += item_bytes
            total_count += item_count
        return total_bytes, total_count
    if isinstance(value, dict):
        total_bytes = 0
        total_count = 0
        source_type = str(value.get("type") or "").lower()
        source_data = value.get("data")
        if source_type == "base64" and isinstance(source_data, str):
            total_bytes += len(source_data)
            total_count += 1
        for key, item in value.items():
            if key == "data" and source_type == "base64":
                continue
            item_bytes, item_count = measure_inline_images(item)
            total_bytes += item_bytes
            total_count += item_count
        return total_bytes, total_count
    return 0, 0


def _load_policy() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        config = {}
    policy = config.get("usage_limits") if isinstance(config, dict) else None
    return policy if isinstance(policy, dict) else {}


def _policy_for_route(
    policy: dict[str, Any],
    *,
    model: str,
    provider: str,
) -> dict[str, Any]:
    """Apply substring overrides to the concrete provider/model route.

    Matching the combined route lets operators express a provider-family
    budget (for example ``gemini``) without depending on every model id
    containing the provider name.  Later matching entries intentionally win,
    so a broad provider override can be refined by a model-specific one.
    """
    effective = dict(policy)
    overrides = policy.get("model_overrides") or {}
    if not isinstance(overrides, dict):
        return effective
    route_label = f"{provider}:{model}".lower()
    for fragment, values in overrides.items():
        if (
            str(fragment or "").lower() in route_label
            and isinstance(values, dict)
        ):
            effective.update(values)
    return effective


def _persisted_usage(agent: Any) -> tuple[int, int]:
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        return 0, 0
    try:
        row = db.get_session(session_id)
    except Exception:
        return 0, 0
    tokens = (
        _row_value(row, "input_tokens")
        + _row_value(row, "output_tokens")
        + _row_value(row, "cache_read_tokens")
        + _row_value(row, "cache_write_tokens")
    )
    return tokens, _row_value(row, "api_call_count")


def evaluate_provider_call_budget(
    agent: Any,
    *,
    approx_input_tokens: int,
    request_messages: Any,
) -> SessionBudgetDecision:
    """Evaluate the next provider request before it can consume credits."""
    policy = _policy_for_route(
        _load_policy(),
        model=str(getattr(agent, "model", "") or ""),
        provider=str(getattr(agent, "provider", "") or ""),
    )
    persisted_tokens, persisted_calls = _persisted_usage(agent)
    live_tokens = (
        _positive_int(getattr(agent, "session_total_tokens", 0))
        + _positive_int(getattr(agent, "session_cache_read_tokens", 0))
        + _positive_int(getattr(agent, "session_cache_write_tokens", 0))
    )
    live_calls = _positive_int(getattr(agent, "session_api_calls", 0))
    current_tokens = max(live_tokens, persisted_tokens)
    current_calls = max(live_calls, persisted_calls)
    prompt_tokens = _positive_int(approx_input_tokens)
    requested_output = _positive_int(getattr(agent, "max_tokens", 0))
    projected_tokens = current_tokens + prompt_tokens + requested_output
    image_bytes, image_count = measure_inline_images(request_messages)

    def decision(status: str, message: str) -> SessionBudgetDecision:
        return SessionBudgetDecision(
            status=status,
            message=message,
            current_tokens=current_tokens,
            projected_tokens=projected_tokens,
            current_api_calls=current_calls,
            inline_image_bytes=image_bytes,
            inline_image_count=image_count,
        )

    if not policy.get("enabled", False):
        return decision("ok", "session workload limits disabled")

    session_cap = _positive_int(policy.get("max_session_tokens"))
    prompt_cap = _positive_int(policy.get("max_prompt_tokens"))
    call_cap = _positive_int(policy.get("max_session_api_calls"))
    image_byte_cap = _positive_int(policy.get("max_inline_image_bytes"))
    image_count_cap = _positive_int(policy.get("max_inline_images"))
    label = (
        f"session={getattr(agent, 'session_id', None) or 'new'} "
        f"current={current_tokens:,} projected={projected_tokens:,} "
        f"calls={current_calls} images={image_count}/{image_bytes:,}B"
    )

    if prompt_cap and prompt_tokens >= prompt_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: prompt ~{prompt_tokens:,} tokens "
            f"reaches the {prompt_cap:,}-token request limit. Start /new or "
            f"compact first. {label}",
        )
    if session_cap and projected_tokens > session_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: projected session workload "
            f"exceeds the {session_cap:,}-token limit. Preserve a handoff and "
            f"start /new. {label}",
        )
    if call_cap and current_calls >= call_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: this session already made "
            f"{current_calls} API calls (limit {call_cap}). Preserve a handoff "
            f"and start /new. {label}",
        )
    if image_count_cap and image_count > image_count_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: {image_count} inline images "
            f"exceed the request limit of {image_count_cap}. Keep only the newest "
            f"proof image or start /new. {label}",
        )
    if image_byte_cap and image_bytes > image_byte_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: inline image payloads total "
            f"{image_bytes:,} bytes (limit {image_byte_cap:,}). Downscale or keep "
            f"only the newest proof image. {label}",
        )

    warn_ratio = policy.get("warn_ratio", 0.75)
    try:
        warn_ratio = min(1.0, max(0.0, float(warn_ratio)))
    except (TypeError, ValueError):
        warn_ratio = 0.75
    if session_cap and projected_tokens >= int(session_cap * warn_ratio):
        return decision(
            "warn",
            f"Session workload warning: {label}; hard limit={session_cap:,}. "
            "Finish this bounded step and prepare /new.",
        )
    return decision("ok", label)
