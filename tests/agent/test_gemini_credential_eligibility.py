"""A placeholder pool row must not shadow real candidates or profile env keys."""
import json
from pathlib import Path

import pytest
import yaml

from agent.credential_pool import CredentialPool, PooledCredential


def entry(key, ident, priority=0, provider="gemini", base_url=None):
    return PooledCredential(provider=provider, id=ident, label=ident, auth_type="api_key",
        priority=priority, source="manual", access_token=key, base_url=base_url)


@pytest.mark.parametrize("strategy", ["fill_first", "round_robin", "least_used", "random"])
def test_placeholder_eligibility_preserves_candidates_and_provider_boundaries(tmp_path, monkeypatch, strategy):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "profile"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "gemini"},
        "credential_pool_strategies": {"gemini": strategy}}))
    rows = [entry("key_1", "placeholder"), entry("AIza" + "a" * 35, "first", 1),
            entry("AQ." + "b" * 60, "second", 2)]
    pool = CredentialPool("gemini", rows)
    keys_before = {e.id: e.access_token for e in pool.entries()}
    selected = [pool.select().id for _ in range(4)]
    assert "placeholder" not in selected
    if strategy == "fill_first": assert selected == ["first"] * 4
    if strategy in {"round_robin", "least_used"}: assert selected == ["first", "second", "first", "second"]
    assert {e.id: e.access_token for e in pool.entries()} == keys_before
    assert pool.reclaim("placeholder") is None
    assert CredentialPool("gemini", [entry("key_1", "proxy", base_url="https://proxy.example/v1")]).select().id == "proxy"
    assert CredentialPool("custom", [entry("key_1", "other", provider="custom")]).select().id == "other"


def test_real_resolution_uses_each_profiles_env_after_only_placeholders(tmp_path, monkeypatch):
    from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.runtime_provider import resolve_runtime_provider
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = []
    for name in ("a", "b"):
        home = tmp_path / name; home.mkdir()
        key = "AIza" + name * 35
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "gemini", "default": "gemini-3.8-flash"}}))
        (home / ".env").write_text("GOOGLE_API_KEY=" + key + "\n")
        (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"gemini": [{
            "id": "placeholder", "label": "placeholder", "auth_type": "api_key", "priority": 0,
            "source": "manual", "access_token": "key_1", "last_status": "ok"}]}}))
        homes.append((home, key))
    previous = is_multiplex_active(); set_multiplex_active(True)
    try:
        for home, expected in [homes[0], homes[1], homes[0]]:
            ht = set_hermes_home_override(home)
            st = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
            try:
                resolved = resolve_runtime_provider(requested="gemini", target_model="gemini-3.8-flash")
                assert resolved["api_key"] == expected
                assert resolved["base_url"].startswith("https://generativelanguage.googleapis.com/")
                persisted = json.loads((home / "auth.json").read_text())["credential_pool"]["gemini"]
                assert next(e for e in persisted if e["id"] == "placeholder")["access_token"] == "key_1"
            finally:
                reset_secret_scope(st); reset_hermes_home_override(ht)
    finally:
        set_multiplex_active(previous)
