"""Schema-versioned manifests for generated DeepMew runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepmew.artifacts.writers import write_json


SCHEMA_VERSION = "deepmew.run_manifest.v1"
REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "kind",
    "created_at_utc",
    "command",
    "status",
    "inputs",
    "outputs",
    "summary",
}


class ManifestError(ValueError):
    """Raised when a manifest is missing required fields or schema support."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_manifest(
    *,
    run_id: str,
    kind: str,
    command: str | list[str],
    status: str = "completed",
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "created_at_utc": created_at_utc or utc_now_iso(),
        "command": command,
        "status": status,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "summary": summary or {},
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        raise ManifestError(f"manifest missing required fields: {', '.join(missing)}")
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema_version: {schema_version!r}")
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    validate_manifest(manifest)
    return write_json(Path(path), manifest)


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    return validate_manifest(data)
