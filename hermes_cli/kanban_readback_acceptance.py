"""Frozen, read-only acceptance checks for local task deliverables.

Create with completion_contract="verify:/absolute/manifest.json". The manifest
has schema "hermes.readback/v1", an outcome, and checks with id, path, and kind
(file_sha256 + sha256, or json_equals + pointer + expected). Creation freezes
the declaration in the board; completion reads the actual files itself. No
worker-supplied receipt, shell command, or subprocess is executed by this gate.
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
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("id"), str) or not check["id"].strip() or check["id"] in ids:
            raise ValueError("acceptance check IDs must be nonempty and unique")
        ids.add(check["id"])
        if not isinstance(check.get("path"), str) or not Path(check["path"]).is_absolute():
            raise ValueError("acceptance paths must be absolute")
        if check.get("kind") == "file_sha256":
            if not isinstance(check.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", check["sha256"]):
                raise ValueError("file_sha256 requires an expected SHA-256")
        elif check.get("kind") == "json_equals":
            pointer = check.get("pointer")
            if not isinstance(pointer, list) or any(type(p) not in (str, int) or (type(p) is int and p < 0) for p in pointer) or "expected" not in check:
                raise ValueError("json_equals requires pointer keys/indexes and expected value")
        else:
            raise ValueError("acceptance kind must be file_sha256 or json_equals")
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
               "recovery": "Your task and workspace remain available. Correct the failed artifact checks, then retry kanban_complete in this same run. Use kanban_block only for an external dependency you cannot resolve."}
    try:
        spec = json.loads(freeze_contract(contract)[len(PREFIX):])
        for check in spec["checks"]:
            observed = {"id": check["id"], "path": check["path"], "ok": False}
            receipt["checks"].append(observed)
            path = Path(check["path"])
            try:
                before = _signature(path)
                with path.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("artifact exceeds the 16 MiB readback limit")
                after = _signature(path)
                if before != after:
                    raise ValueError("artifact changed during readback; retry")
                observed.update(sha256=hashlib.sha256(data).hexdigest(), signature=after)
                if check["kind"] == "file_sha256":
                    matched = observed["sha256"] == check["sha256"]
                else:
                    value = json.loads(data)
                    for part in check["pointer"]:
                        if type(part) is str and not isinstance(value, dict) or type(part) is int and not isinstance(value, list):
                            raise ValueError("JSON pointer does not match artifact structure")
                        value = value[part]
                    matched = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(check["expected"], sort_keys=True, allow_nan=False)
                observed.update(ok=matched, detail="matched" if matched else "artifact differs from frozen acceptance")
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                # Do not persist file contents or exception strings containing private data.
                observed["detail"] = "artifact missing, unreadable, changed, or JSON pointer unavailable"
        receipt["ok"] = all(c["ok"] for c in receipt["checks"])
        receipt["classification"] = "success" if receipt["ok"] else "failure"
        receipt["detail"] = "All frozen checks matched." if receipt["ok"] else "Failed checks: " + ", ".join(c["id"] for c in receipt["checks"] if not c["ok"])
    except (OSError, ValueError, TypeError, KeyError):
        receipt.update(classification="invalid", detail="Frozen acceptance declaration is invalid; owner must repair task intake.")
    return receipt


def readback_unchanged(receipt: dict) -> bool:
    """Refuse stale evidence if an artifact changed before the terminal DB write."""
    try:
        return all(_signature(Path(c["path"])) == c["signature"] for c in receipt["checks"])
    except (OSError, ValueError, KeyError):
        return False
