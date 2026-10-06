"""Reject obvious example credentials only where Google's API receives them.

This is a candidate filter, not key validation or rotation policy. It neither
edits stored rows nor rejects unknown future Google key formats or proxy keys.
"""
import re
from urllib.parse import urlsplit


def is_google_placeholder(provider, entry, model_config):
    if provider != "gemini" or entry.auth_type != "api_key":
        return False
    if not re.fullmatch(r"(?:key|api_key|test|dummy|placeholder)[_-]?\d*", entry.runtime_api_key.strip(), re.I):
        return False
    default = "https://generativelanguage.googleapis.com/v1beta"
    endpoint = entry.runtime_base_url or ""
    model_config = model_config if isinstance(model_config, dict) else {}
    if not endpoint or endpoint.rstrip("/") == default:
        if model_config.get("provider") in {"gemini", "google", "google-gemini", "google-ai-studio"}:
            endpoint = model_config.get("base_url") or endpoint
    host = urlsplit(endpoint or default).hostname
    return host in {"generativelanguage.googleapis.com", "aiplatform.googleapis.com"} or bool(
        host and re.fullmatch(r"[a-z0-9-]+-aiplatform\.googleapis\.com", host))
