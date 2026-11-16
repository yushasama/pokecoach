from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_DIR = Path(r"D:\DeepMew_artifacts\checkpoints")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def preserve_checkpoint(
    source_path: Path,
    *,
    mission: str,
    name: str,
    reason: str,
    backup_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> dict[str, Any]:
    source_path = Path(source_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = source_path.suffix or ".pt"
    target_path = backup_dir / f"{mission}-{name}-{timestamp}{suffix}"
    shutil.copy2(source_path, target_path)
    source_sha = sha256_file(source_path)
    target_sha = sha256_file(target_path)

    return {
        "source_path": str(source_path),
        "artifact_path": str(target_path),
        "byte_size": source_path.stat().st_size,
        "sha256": source_sha,
        "artifact_sha256": target_sha,
        "reason": reason,
        "status": "PASS" if source_sha == target_sha else "FAIL",
    }
