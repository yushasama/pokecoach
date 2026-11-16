"""Schema-versioned manifests for DeepMew submission bundles."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "deepmew.submission_manifest.v1"
REQUIRED_FIELDS = {
    "schema_version",
    "submission_id",
    "created_at",
    "agent_spec",
    "canonical_label",
    "entrypoint",
    "deck_path",
    "deck_hash",
    "git_commit",
    "git_dirty",
    "included_paths",
    "excluded_paths",
    "artifact_hashes",
    "bundle_tar_gz",
    "tar_gz_hash",
    "bundle_zip",
    "zip_hash",
    "validation",
    "notes",
}


class SubmissionManifestError(ValueError):
    """Raised when a submission manifest is missing required schema fields."""


def utc_now_iso(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc).replace(microsecond=0)
    return current.isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        raise SubmissionManifestError(f"submission manifest missing required fields: {', '.join(missing)}")
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SubmissionManifestError(f"unsupported submission manifest schema_version: {schema_version!r}")
    if not isinstance(manifest.get("included_paths"), list):
        raise SubmissionManifestError("included_paths must be a list")
    if not isinstance(manifest.get("artifact_hashes"), dict):
        raise SubmissionManifestError("artifact_hashes must be an object")
    return manifest


def write_submission_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    validate_submission_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_submission_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SubmissionManifestError("submission manifest root must be a JSON object")
    return validate_submission_manifest(data)
