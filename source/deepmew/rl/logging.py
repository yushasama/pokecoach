"""Reproducibility helpers: git status summary + content hashing.

Every RL run records enough to reproduce it: config, seed, git status, enabled
modules, reward, opponent sampler, backend, and content hashes. These helpers
degrade gracefully (no exceptions) when git is unavailable, since the RL path
must never crash a run over telemetry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_status_summary() -> dict[str, Any]:
    """Branch, short HEAD sha, and a dirty-file count. Never raises."""
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["rev-parse", "--short", "HEAD"])
    status = _git(["status", "--porcelain"])
    if status is None:
        dirty_files = None
        dirty_count = None
    else:
        lines = [line for line in status.splitlines() if line.strip()]
        dirty_files = lines[:40]
        dirty_count = len(lines)
    return {
        "branch": branch,
        "head_short": head,
        "dirty_count": dirty_count,
        "dirty_files": dirty_files,
        "available": status is not None,
    }
