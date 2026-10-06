#!/usr/bin/env python3
"""ONE Doctor referee for external closes.

Architecture (.out = proposal; .ankl = record):
  A close counts only when an EXTERNAL read matches bytes — file sha256 or a
  live HTTP GET whose body is hashed. An agent's own status JSON never closes.

This module is the shared judge used by:
  - Hermes kanban readback acceptance
  - PR 204 path_writer_board /api/close (judge_close)
  - JEV verify_closure evidence re-hash (via verify_file_bytes)
  - Harness doctor_review EXTERNAL check

json_equals is advisory / non-closing. It may describe structure, but it cannot
alone mark a task done.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

CLOSING_KINDS = frozenset({"file_sha256", "http_sha256", "exact-byte-close", "exact_byte_close"})
NON_CLOSING_KINDS = frozenset({"json_equals"})
BLOCKED_CONTROL_URL = "http://127.0.0.1:8098/api/queue/control"
_PASS_FIXTURE_RE = re.compile(
    r"doctor\s*:\s*pass|doctor\s+pass|doctor\s*:\s*verified",
    re.IGNORECASE,
)
_BARE_LABELS = {
    "independently accepted",
    "harness accepted",
    "doctor: verified",
    "doctor pass",
    "doctor: pass",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExternalRead:
    status: int
    body: bytes
    fetches: int


def is_closing_kind(kind: str | None) -> bool:
    return str(kind or "") in CLOSING_KINDS


def is_bare_label(text: str) -> bool:
    sample = str(text or "").strip().lower()
    if sample in _BARE_LABELS:
        return True
    return sample.startswith("harness:") or sample.startswith("harness label")


def is_pass_fixture(text: str) -> bool:
    return bool(_PASS_FIXTURE_RE.search(str(text or "")))


def refuse(reason: str, detail: str) -> dict:
    return {"ok": False, "closed": False, "reason": reason, "detail": detail, "closing": True}


def verify_file_bytes(path: Path | str, expected_sha256: str) -> dict:
    """Re-hash a file on disk. External because the referee reads the bytes."""
    expected = str(expected_sha256 or "").lower().strip()
    if not SHA256_RE.fullmatch(expected):
        return refuse("bad_digest", "file_sha256 requires a 64-char hex digest")
    target = Path(path)
    try:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        return refuse("unreadable", f"evidence unreadable: {exc}")
    if digest != expected:
        return refuse("mismatch", "external file bytes do not match the claim")
    return {
        "ok": True,
        "closed": True,
        "reason": "match",
        "closing": True,
        "kind": "file_sha256",
        "sha256": digest,
        "path": str(target),
    }


def http_get(url: str, *, timeout: float = 15.0) -> ExternalRead:
    target = str(url or "").strip()
    if not target or target == BLOCKED_CONTROL_URL:
        return ExternalRead(status=0, body=b"", fetches=0)
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ExternalRead(status=0, body=b"", fetches=0)
    request = urllib.request.Request(target, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status = int(getattr(response, "status", 0) or 0)
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp is not None else b""
        status = int(exc.code or 0)
    except (urllib.error.URLError, TimeoutError, OSError):
        return ExternalRead(status=0, body=b"", fetches=1)
    return ExternalRead(status=status, body=payload, fetches=1)


def verify_http_sha256(url: str, expected_sha256: str, *, timeout: float = 15.0) -> dict:
    expected = str(expected_sha256 or "").lower().strip()
    if not SHA256_RE.fullmatch(expected):
        return refuse("bad_digest", "http_sha256 requires a 64-char hex digest")
    read = http_get(url, timeout=timeout)
    if read.fetches < 1:
        return refuse("no_fetch", "an external live read is required")
    digest = hashlib.sha256(read.body).hexdigest()
    if digest != expected:
        return refuse("mismatch", "live-read bytes do not match the claim")
    return {
        "ok": True,
        "closed": True,
        "reason": "match",
        "closing": True,
        "kind": "http_sha256",
        "sha256": digest,
        "url": url,
        "status": read.status,
        "fetches": read.fetches,
    }


def judge_close(
    *,
    path: str,
    claimed: bytes,
    prior: Optional[bytes],
    read: Optional[ExternalRead],
) -> dict:
    """PR 204 exact-byte close: external read body must equal claimed bytes."""
    artifact = str(path or "")
    if artifact.endswith(".out"):
        return refuse("proposal", "a .out is a proposal")
    body = read.body if read is not None else b""
    text = body.decode("utf-8", errors="replace")
    fetches = int(read.fetches) if read is not None else 0
    if fetches < 1:
        if is_pass_fixture(text):
            return refuse("pass_fixture", "a pass line with zero fetches is not a close")
        if is_bare_label(text) or is_bare_label(claimed.decode("utf-8", errors="replace")):
            return refuse("label", "a harness label is not a close")
        return refuse("no_fetch", "an external read is required")
    if is_pass_fixture(text):
        return refuse("pass_fixture", "a pass line with zero fetches is not a close")
    if is_bare_label(text):
        return refuse("label", "a harness label is not a close")
    if prior is not None and body == prior:
        return refuse("unchanged", "http 200 with unchanged bytes is not a close")
    if body != claimed:
        return refuse("mismatch", "external bytes do not match the claim")
    digest = hashlib.sha256(body).hexdigest()
    return {
        "ok": True,
        "closed": True,
        "reason": "match",
        "closing": True,
        "receipt": {
            "kind": "exact-byte-close",
            "sha256": digest,
            "fetches": fetches,
            "status": int(read.status) if read is not None else 0,
            "match": True,
            "record": "ankl" if artifact.endswith(".ankl") else "",
            "reopenable": artifact.endswith(".ankl"),
            "referee": "doctor_external_close",
        },
    }


def evaluate_manifest_check(check: dict, *, data: bytes | None = None) -> dict:
    """Evaluate one hermes.readback/v1 check. json_equals is never closing."""
    kind = check.get("kind")
    observed: dict[str, Any] = {
        "id": check.get("id"),
        "path": check.get("path"),
        "kind": kind,
        "ok": False,
        "closing": is_closing_kind(kind),
    }
    if kind == "json_equals":
        observed["closing"] = False
        observed["detail"] = "json_equals is advisory/non-closing (self-certifying status JSON cannot close)"
        if data is None:
            observed["detail"] = "artifact missing for advisory json_equals"
            return observed
        try:
            value = json.loads(data)
            for part in check.get("pointer") or []:
                if isinstance(part, str) and not isinstance(value, dict):
                    raise ValueError("pointer mismatch")
                if isinstance(part, int) and not isinstance(value, list):
                    raise ValueError("pointer mismatch")
                value = value[part]
            matched = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
                check.get("expected"), sort_keys=True, allow_nan=False
            )
            observed.update(ok=matched, detail="advisory matched" if matched else "advisory differed")
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            observed["detail"] = "advisory json_equals unavailable"
        return observed

    if kind == "file_sha256":
        path = check.get("path")
        expected = check.get("sha256")
        result = verify_file_bytes(path, expected)
        observed.update(
            ok=result.get("ok") is True,
            sha256=result.get("sha256"),
            detail=result.get("detail") or result.get("reason") or ("matched" if result.get("ok") else "mismatch"),
        )
        return observed

    if kind == "http_sha256":
        url = check.get("url") or check.get("path")
        result = verify_http_sha256(url, check.get("sha256"))
        observed.update(
            ok=result.get("ok") is True,
            sha256=result.get("sha256"),
            url=url,
            detail=result.get("detail") or result.get("reason") or ("matched" if result.get("ok") else "mismatch"),
        )
        return observed

    observed["detail"] = f"unsupported check kind for referee: {kind}"
    return observed


def closing_ok(check_results: list[dict]) -> bool:
    """A close requires every closing check to pass, and at least one closing check."""
    closing = [c for c in check_results if c.get("closing") is True]
    if not closing:
        return False
    return all(c.get("ok") is True for c in closing)


def readback_has_external_byte_proof(readback: dict | None) -> bool:
    """Harness EXTERNAL gate: live sha256 evidence present and ok."""
    if not isinstance(readback, dict) or not readback.get("ok"):
        return False
    for check in readback.get("checks") or []:
        if not isinstance(check, dict) or not check.get("ok"):
            continue
        if check.get("closing") is False:
            continue
        digest = check.get("sha256")
        if isinstance(digest, str) and SHA256_RE.fullmatch(digest):
            return True
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "detail": "usage: doctor_external_close.py file_sha256 PATH EXPECTED"}))
        raise SystemExit(2)
    kind, *rest = sys.argv[1], *sys.argv[2:]
    if kind == "file_sha256" and len(rest) == 2:
        print(json.dumps(verify_file_bytes(rest[0], rest[1]), indent=2, sort_keys=True))
        raise SystemExit(0 if verify_file_bytes(rest[0], rest[1]).get("ok") else 1)
    if kind == "http_sha256" and len(rest) == 2:
        out = verify_http_sha256(rest[0], rest[1])
        print(json.dumps(out, indent=2, sort_keys=True))
        raise SystemExit(0 if out.get("ok") else 1)
    print(json.dumps({"ok": False, "detail": "unknown kind"}))
    raise SystemExit(2)
