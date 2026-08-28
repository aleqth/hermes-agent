"""Fail-closed, pre-provider workload limits for long Hermes sessions.

Tool hooks run too late to prevent the model request that selected a tool.
This module therefore evaluates local, profile-scoped policy immediately
before each provider call.  It performs no network I/O and never mutates the
conversation, preserving prompt-cache stability.
"""

from __future__ import annotations

import hashlib
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
    auto_rollover: bool = False

    @property
    def blocked(self) -> bool:
        return self.status == "block"

    @property
    def rollover_requested(self) -> bool:
        return self.status == "rollover" and self.auto_rollover


@dataclass(frozen=True)
class InlineImageSanitization:
    """What changed in the provider-bound copy of a multimodal request."""

    original_count: int = 0
    original_bytes: int = 0
    kept_count: int = 0
    kept_bytes: int = 0
    duplicate_count: int = 0
    overflow_count: int = 0

    @property
    def removed_count(self) -> int:
        return self.duplicate_count + self.overflow_count

    @property
    def changed(self) -> bool:
        return self.removed_count > 0

    @property
    def notice(self) -> str:
        if not self.changed:
            return ""
        return (
            f"Images: using the newest {self.kept_count} unique inline "
            f"image{'s' if self.kept_count != 1 else ''}; omitted "
            f"{self.removed_count} older or repeated "
            f"image{'s' if self.removed_count != 1 else ''} from this "
            "provider request. Conversation history was preserved."
        )


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
        inline_data = value.get("inline_data")
        if isinstance(inline_data, Mapping) and isinstance(inline_data.get("data"), str):
            total_bytes += len(inline_data["data"])
            total_count += 1
        for key, item in value.items():
            if key == "data" and source_type == "base64":
                continue
            if key == "inline_data" and isinstance(inline_data, Mapping):
                continue
            item_bytes, item_count = measure_inline_images(item)
            total_bytes += item_bytes
            total_count += item_count
        return total_bytes, total_count
    return 0, 0


def _inline_image_candidate(value: Any) -> tuple[str, int] | None:
    """Return a stable fingerprint and encoded size for one image part."""
    if not isinstance(value, Mapping):
        return None
    part_type = str(value.get("type") or "").lower()
    payload = ""
    media_type = ""

    if part_type in {"image_url", "input_image"}:
        image_url = value.get("image_url")
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")
        if not isinstance(image_url, str):
            image_url = value.get("url")
        if isinstance(image_url, str) and image_url.startswith("data:image/"):
            payload = image_url
    elif part_type == "image":
        source = value.get("source")
        if (
            isinstance(source, Mapping)
            and str(source.get("type") or "").lower() == "base64"
            and isinstance(source.get("data"), str)
        ):
            payload = source["data"]
            media_type = str(source.get("media_type") or "")

    inline_data = value.get("inline_data")
    if (
        not payload
        and isinstance(inline_data, Mapping)
        and isinstance(inline_data.get("data"), str)
    ):
        payload = inline_data["data"]
        media_type = str(inline_data.get("mime_type") or inline_data.get("mimeType") or "")

    if not payload:
        return None
    digest = hashlib.sha256()
    digest.update(media_type.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(payload.encode("utf-8", errors="replace"))
    return digest.hexdigest(), len(payload)


def sanitize_inline_images_for_provider(
    agent: Any,
    request_messages: Any,
) -> tuple[Any, InlineImageSanitization]:
    """Trim only the provider-bound request to its newest unique images.

    Hermes' durable message/session history is deliberately not mutated. The
    request tree is rebuilt with shared immutable strings, so even multi-MB
    data URLs are not duplicated in memory. Newest messages and newest image
    parts win. If the newest single image exceeds the byte cap it is retained
    so the hard budget guard can still fail closed with an actionable error.
    """
    policy = _policy_for_route(
        _load_policy(),
        model=str(getattr(agent, "model", "") or ""),
        provider=str(getattr(agent, "provider", "") or ""),
    )
    if not policy.get("enabled", False):
        return request_messages, InlineImageSanitization()

    count_cap = _positive_int(policy.get("max_inline_images"))
    byte_cap = _positive_int(policy.get("max_inline_image_bytes"))
    candidates: list[tuple[tuple[Any, ...], str, int]] = []

    def collect(value: Any, path: tuple[Any, ...]) -> None:
        candidate = _inline_image_candidate(value)
        if candidate is not None:
            candidates.append((path, candidate[0], candidate[1]))
            return
        if isinstance(value, list) or isinstance(value, tuple):
            for index, item in enumerate(value):
                collect(item, path + (index,))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, path + (key,))

    collect(request_messages, ())
    if not candidates:
        return request_messages, InlineImageSanitization()

    kept_paths: set[tuple[Any, ...]] = set()
    seen: set[str] = set()
    kept_bytes = 0
    duplicate_count = 0
    overflow_count = 0
    for path, fingerprint, size in reversed(candidates):
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        if count_cap and len(kept_paths) >= count_cap:
            overflow_count += 1
            continue
        if byte_cap and kept_paths and kept_bytes + size > byte_cap:
            overflow_count += 1
            continue
        kept_paths.add(path)
        kept_bytes += size

    report = InlineImageSanitization(
        original_count=len(candidates),
        original_bytes=sum(item[2] for item in candidates),
        kept_count=len(kept_paths),
        kept_bytes=kept_bytes,
        duplicate_count=duplicate_count,
        overflow_count=overflow_count,
    )
    if not report.changed:
        return request_messages, report

    candidate_paths = {item[0] for item in candidates}
    dropped_paths = candidate_paths - kept_paths
    dropped = object()

    def rebuild(value: Any, path: tuple[Any, ...]) -> Any:
        if path in dropped_paths:
            return dropped
        if isinstance(value, list):
            result = []
            for index, item in enumerate(value):
                child = rebuild(item, path + (index,))
                if child is not dropped:
                    result.append(child)
            return result
        if isinstance(value, tuple):
            result = []
            for index, item in enumerate(value):
                child = rebuild(item, path + (index,))
                if child is not dropped:
                    result.append(child)
            return tuple(result)
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                child = rebuild(item, path + (key,))
                if child is not dropped:
                    result[key] = child
            return result
        return value

    return rebuild(request_messages, ()), report


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
            auto_rollover=bool(policy.get("auto_rollover", False)),
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
    # This is an independent runaway guard, not a segment-size trigger. Check
    # it before token rollover so reaching both rails cannot erase a runaway
    # call condition by rotating counters.
    if call_cap and current_calls >= call_cap:
        return decision(
            "block",
            f"Provider call blocked before spend: this session already made "
            f"{current_calls} API calls (limit {call_cap}). Preserve a handoff "
            f"and start /new. {label}",
        )
    if session_cap and projected_tokens > session_cap:
        if policy.get("auto_rollover", False):
            return decision(
                "rollover",
                f"Session workload reached its {session_cap:,}-token segment "
                f"limit. Preserve a durable handoff and continue in a fresh "
                f"child segment before the next provider call. {label}",
            )
        return decision(
            "block",
            f"Provider call blocked before spend: projected session workload "
            f"exceeds the {session_cap:,}-token limit. Preserve a handoff and "
            f"start /new. {label}",
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
            + (
                "Hermes will roll this conversation into a durable child "
                "segment automatically."
                if policy.get("auto_rollover", False)
                else "Finish this bounded step and prepare /new."
            ),
        )
    return decision("ok", label)
