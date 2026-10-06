"""Frozen, read-only acceptance checks for local task deliverables.

Create with completion_contract="verify:/absolute/manifest.json". The manifest
has schema "hermes.readback/v1", an outcome, and checks with id, path, and kind
(file_sha256 + sha256, http_sha256 + url/path + sha256, or advisory json_equals
+ pointer + expected). Creation freezes the declaration in the board; completion
reads the actual files itself. No worker-supplied receipt, shell command, or
subprocess is executed by this gate.

Closes are decided by the ONE Doctor referee (doctor_external_close): only
file_sha256 or http_sha256 (live-read hash) can close. json_equals is advisory
and never sufficient for ok/close — an agent's own status JSON cannot self-certify.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

PREFIX = "verify-v1:"
MAX_BYTES = 16 * 1024 * 1024
CLOSING_KINDS = frozenset({"file_sha256", "http_sha256"})


def _load_referee():
    try:
        from hermes_cli import doctor_external_close as referee
        return referee
    except ImportError:
        pass
    for cand in (
        Path.home() / "Desktop/solana-agent/.ai-sync/skills/no-loose-ends/scripts/doctor_external_close.py",
        Path(__file__).resolve().parent / "doctor_external_close.py",
    ):
        if cand.is_file():
            import runpy
            return type("R", (), runpy.run_path(str(cand)))
    return None


def freeze_contract(value: str) -> str:
    if value.startswith(PREFIX):
        raw = value[len(PREFIX):]
    else:
        source = Path(value[7:])
        if not source.is_absolute() or not source.is_file():
            raise ValueError("acceptance manifest must be an absolute regular file")
        with source.open() as stream:
            raw = stream.read(65537)
    if len(raw.encode()) > 65536:
        raise ValueError("acceptance manifest exceeds 64 KiB")
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("schema") != "hermes.readback/v1":
        raise ValueError("acceptance manifest requires schema hermes.readback/v1")
    if not isinstance(spec.get("outcome"), str) or not spec["outcome"].strip():
        raise ValueError("acceptance manifest requires the intended outcome")
    checks = spec.get("checks")
    if not isinstance(checks, list) or not 1 <= len(checks) <= 16:
        raise ValueError("acceptance manifest requires 1-16 checks")
    ids = set()
    closing = 0
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("id"), str) or not check["id"].strip() or check["id"] in ids:
            raise ValueError("acceptance check IDs must be nonempty and unique")
        ids.add(check["id"])
        kind = check.get("kind")
        if kind == "file_sha256":
            if not isinstance(check.get("path"), str) or not Path(check["path"]).is_absolute():
                raise ValueError("acceptance paths must be absolute")
            if not isinstance(check.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", check["sha256"]):
                raise ValueError("file_sha256 requires an expected SHA-256")
            closing += 1
        elif kind == "http_sha256":
            url = check.get("url") or check.get("path")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError("http_sha256 requires an http(s) url")
            if not isinstance(check.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", check["sha256"]):
                raise ValueError("http_sha256 requires an expected SHA-256")
            closing += 1
        elif kind == "json_equals":
            if not isinstance(check.get("path"), str) or not Path(check["path"]).is_absolute():
                raise ValueError("acceptance paths must be absolute")
            pointer = check.get("pointer")
            if not isinstance(pointer, list) or any(type(p) not in (str, int) or (type(p) is int and p < 0) for p in pointer) or "expected" not in check:
                raise ValueError("json_equals requires pointer keys/indexes and expected value")
            # Allowed as advisory only; cannot close.
        else:
            raise ValueError("acceptance kind must be file_sha256, http_sha256, or json_equals")
    if closing < 1:
        raise ValueError(
            "closes require at least one external check (file_sha256 or http_sha256); "
            "json_equals alone is self-certifying and cannot close"
        )
    return PREFIX + json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(path: Path) -> list[int]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("acceptance artifact must be a regular file")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def collect_readback(contract: str) -> dict:
    receipt = {"kind": "artifact_acceptance", "ok": False, "classification": "missing",
               "contract_sha256": hashlib.sha256(contract.encode()).hexdigest(), "checks": [],
               "observed_at": datetime.now(timezone.utc).isoformat(),
               "referee": "doctor_external_close",
               "recovery": "Your task and workspace remain available. Correct the failed artifact checks, then retry kanban_complete in this same run. Use kanban_block only for an external dependency you cannot resolve."}
    referee = _load_referee()
    try:
        spec = json.loads(freeze_contract(contract)[len(PREFIX):])
        for check in spec["checks"]:
            kind = check["kind"]
            observed = {"id": check["id"], "path": check.get("path"), "kind": kind,
                        "ok": False, "closing": kind in CLOSING_KINDS}
            receipt["checks"].append(observed)
            try:
                if kind == "http_sha256":
                    if referee is None:
                        raise ValueError("Doctor referee unavailable")
                    url = check.get("url") or check.get("path")
                    result = referee.verify_http_sha256(url, check["sha256"])
                    observed.update(
                        ok=result.get("ok") is True,
                        sha256=result.get("sha256"),
                        url=url,
                        detail=result.get("detail") or ("matched" if result.get("ok") else "live-read differs from frozen acceptance"),
                    )
                    continue

                path = Path(check["path"])
                before = _signature(path)
                with path.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("artifact exceeds the 16 MiB readback limit")
                after = _signature(path)
                if before != after:
                    raise ValueError("artifact changed during readback; retry")
                observed.update(sha256=hashlib.sha256(data).hexdigest(), signature=after)

                if kind == "file_sha256":
                    if referee is not None:
                        result = referee.verify_file_bytes(path, check["sha256"])
                        matched = result.get("ok") is True
                    else:
                        matched = observed["sha256"] == check["sha256"]
                    observed.update(ok=matched, detail="matched" if matched else "artifact differs from frozen acceptance")
                else:
                    # json_equals: advisory only — never closing.
                    observed["closing"] = False
                    if referee is not None:
                        adv = referee.evaluate_manifest_check(check, data=data)
                        observed.update(ok=adv.get("ok") is True, detail=adv.get("detail"), closing=False)
                    else:
                        value = json.loads(data)
                        for part in check["pointer"]:
                            if type(part) is str and not isinstance(value, dict) or type(part) is int and not isinstance(value, list):
                                raise ValueError("JSON pointer does not match artifact structure")
                            value = value[part]
                        matched = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(check["expected"], sort_keys=True, allow_nan=False)
                        observed.update(
                            ok=matched,
                            detail=("advisory matched" if matched else "advisory differed")
                            + " (json_equals cannot close)",
                        )
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                observed["detail"] = "artifact missing, unreadable, changed, or JSON pointer unavailable"
        closing = [c for c in receipt["checks"] if c.get("closing") is True]
        receipt["ok"] = bool(closing) and all(c["ok"] for c in closing)
        if not closing:
            receipt["ok"] = False
            receipt["detail"] = "No external closing check (file_sha256/http_sha256); json_equals cannot close"
            receipt["classification"] = "failure"
        else:
            receipt["classification"] = "success" if receipt["ok"] else "failure"
            receipt["detail"] = (
                "All external closing checks matched."
                if receipt["ok"]
                else "Failed closing checks: " + ", ".join(c["id"] for c in closing if not c["ok"])
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        receipt.update(
            classification="invalid",
            detail="Frozen acceptance declaration is invalid; owner must repair task intake.",
            error=str(exc)[:200],
        )
    return receipt


def readback_unchanged(receipt: dict) -> bool:
    """Refuse stale evidence if an artifact changed before the terminal DB write."""
    try:
        for c in receipt["checks"]:
            if c.get("kind") == "http_sha256":
                continue  # live reads are re-fetched at complete time via collect_readback
            if c.get("closing") is False and "signature" not in c:
                continue
            if "signature" in c and _signature(Path(c["path"])) != c["signature"]:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False
