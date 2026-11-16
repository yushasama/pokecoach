"""Build and verify the separate sealed X3 Modal payload.

The payload reuses the already validated X1/X2 runtime closure by content hash,
removes X1/X2 campaign and launcher identities, and adds only the frozen X3
campaign, operational artifact map, cost guard, and separate launcher. Building
or importing this module never invokes Modal.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_MISSION = ROOT / ".research" / "x1_x2_direct_exploiter_first_wave"
BASE_MANIFEST_PATH = BASE_MISSION / "DEEPMEW_MODAL_PAYLOAD_MANIFEST_V2.json"
BASE_STAGING_ROOT = BASE_MISSION / "payload_v2_2_staging_frozen_final"
X3_MISSION = (
    ROOT
    / ".research"
    / "current_rating_bucket_exploiter_league_and_hybrid_v1"
    / "x3"
)
MANIFEST_PATH = X3_MISSION / "X3_MODAL_PAYLOAD_MANIFEST_V2_7.json"
STAGING_ROOT = X3_MISSION / "payload_v2_7_staging_authorized_relaunch"
REMOTE_PAYLOAD_ROOT = PurePosixPath("/opt/deepmew_x3_payload")
MANIFEST_CONTROL_NAME = "X3_MODAL_PAYLOAD_MANIFEST_V2_7.json"
MANIFEST_SCHEMA = "deepmew.x3_modal_payload_manifest.v2_7"
MANIFEST_VERSION = "2.7.0"
BASE_PAYLOAD_IDENTITY_SHA256 = (
    "cd9ee52547b401c03ce3f997ac55d0b45a3858f27c3a15ffbaf61424c971ec9a"
)
FUNCTION_IDENTITY = {
    "modal_app_name": "deepmew-x3-d1-tempo-prize-plan-relaunch-20260721",
    "modal_function_name": "train_x3_d1_tempo_prize_plan_v2",
    "modal_volume_name": "deepmew-x3-d1-tempo-prize-plan-relaunch-20260721",
    "remote_payload_root": str(REMOTE_PAYLOAD_ROOT),
    "remote_output_namespace": "/x3/training_v2",
    "gpu": "L4",
    "gpu_count": 1,
    "max_containers": 1,
    "automatic_retries": 0,
}
FORBIDDEN_BASE_IDENTITY_PATHS = frozenset(
    {
        ".research/x1_x2_direct_exploiter_first_wave/X1_CAMPAIGN_CONFIG.json",
        ".research/x1_x2_direct_exploiter_first_wave/X2_CAMPAIGN_CONFIG.json",
        ".research/x1_x2_direct_exploiter_first_wave/X1_REMOTE_ARTIFACT_MAP_V2.json",
        ".research/x1_x2_direct_exploiter_first_wave/X2_REMOTE_ARTIFACT_MAP_V2.json",
        "scripts/modal_direct_exploiter.py",
        "scripts/modal_payload_v2.py",
    }
)
REPLACED_BASE_CODE_PATHS = frozenset(
    {"deepmew/rl/opponents.py", "scripts/run_terminal_league_ppo.py"}
)
REMOVED_BASE_LOGICAL_PATHS = FORBIDDEN_BASE_IDENTITY_PATHS | REPLACED_BASE_CODE_PATHS


class X3PayloadError(RuntimeError):
    """The local X3 payload seal or identity failed."""


@dataclass(frozen=True)
class AddedSource:
    source_path: Path
    logical_relative_path: str
    payload_role: str
    owning_component: str
    consumer: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise X3PayloadError(f"expected JSON object: {path}")

    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="ascii",
            newline="\n",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _added_sources() -> tuple[AddedSource, ...]:
    return (
        AddedSource(
            ROOT / "deepmew" / "rl" / "opponents.py",
            "deepmew/rl/opponents.py",
            "REQUIRED_CODE",
            "policy/deck opponent materialization",
            "terminal PPO runner",
        ),
        AddedSource(
            ROOT / "scripts" / "run_terminal_league_ppo.py",
            "scripts/run_terminal_league_ppo.py",
            "REQUIRED_CODE",
            "terminal PPO runner",
            "X3 training subprocess",
        ),
        AddedSource(
            X3_MISSION / "X3_CAMPAIGN_CONFIG.json",
            ".research/current_rating_bucket_exploiter_league_and_hybrid_v1/x3/X3_CAMPAIGN_CONFIG.json",
            "REQUIRED_CONFIGURATION",
            "X3 campaign",
            "scripts.run_terminal_league_ppo",
        ),
        AddedSource(
            X3_MISSION / "X3_MODAL_ARTIFACT_MAP.json",
            ".research/current_rating_bucket_exploiter_league_and_hybrid_v1/x3/X3_MODAL_ARTIFACT_MAP.json",
            "REQUIRED_CONFIGURATION",
            "X3 Modal packaging",
            "scripts.run_terminal_league_ppo artifact remapping",
        ),
        AddedSource(
            X3_MISSION / "X3_MODAL_HARDWARE_AND_COST_GUARD.json",
            ".research/current_rating_bucket_exploiter_league_and_hybrid_v1/x3/X3_MODAL_HARDWARE_AND_COST_GUARD.json",
            "REQUIRED_CONFIGURATION",
            "X3 cost guard",
            "scripts.modal_x3_training paid-launch preflight",
        ),
        AddedSource(
            ROOT / "scripts" / "modal_x3_payload.py",
            "scripts/modal_x3_payload.py",
            "REQUIRED_CODE",
            "X3 payload",
            "scripts.modal_x3_training payload verifier",
        ),
        AddedSource(
            ROOT / "scripts" / "modal_x3_training.py",
            "scripts/modal_x3_training.py",
            "REQUIRED_CODE",
            "X3 launcher",
            "separate Modal function source and sealed parity copy",
        ),
    )


def payload_identity(
    entries: Sequence[Mapping[str, Any]],
    function_identity: Mapping[str, Any],
) -> str:
    logical_rows = [
        {
            "logical_relative_path": str(entry["logical_relative_path"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in sorted(entries, key=lambda row: str(row["logical_relative_path"]))
    ]

    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "entries": logical_rows,
                "function_identity": dict(function_identity),
            }
        )
    ).hexdigest()


def base_runtime_entries(remote_payload_root: PurePosixPath) -> list[dict[str, Any]]:
    manifest = _read_json(BASE_MANIFEST_PATH)
    if manifest.get("payload_identity_sha256") != BASE_PAYLOAD_IDENTITY_SHA256:
        raise X3PayloadError("validated X1/X2 base payload identity drifted")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise X3PayloadError("validated X1/X2 base payload entries are missing")
    selected: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise X3PayloadError("base payload entry is malformed")
        logical = str(raw.get("logical_relative_path") or "")
        if logical in REMOVED_BASE_LOGICAL_PATHS:
            continue
        source = BASE_STAGING_ROOT.joinpath(*PurePosixPath(logical).parts)
        if not source.is_file():
            raise X3PayloadError(f"base staged source is missing: {logical}")
        if source.stat().st_size != int(raw["byte_size"]):
            raise X3PayloadError(f"base staged source size drifted: {logical}")
        if sha256_file(source) != raw["sha256"]:
            raise X3PayloadError(f"base staged source hash drifted: {logical}")
        selected.append(
            {
                "logical_relative_path": logical,
                "intended_remote_path": str(remote_payload_root / logical),
                "sha256": str(raw["sha256"]),
                "byte_size": int(raw["byte_size"]),
                "payload_role": str(raw["payload_role"]),
                "owning_package_or_component": str(raw["owning_package_or_component"]),
                "required_import_or_consumer": str(raw["required_import_or_consumer"]),
                "source_identity": f"base_payload:{BASE_PAYLOAD_IDENTITY_SHA256}",
                "manifest_version": MANIFEST_VERSION,
            }
        )

    return selected


def build_manifest() -> dict[str, Any]:
    entries = base_runtime_entries(REMOTE_PAYLOAD_ROOT)
    observed = {str(entry["logical_relative_path"]) for entry in entries}
    for added in _added_sources():
        if not added.source_path.is_file():
            raise X3PayloadError(f"required X3 source is missing: {added.source_path}")
        if added.logical_relative_path in observed:
            raise X3PayloadError(f"duplicate X3 payload path: {added.logical_relative_path}")
        observed.add(added.logical_relative_path)
        entries.append(
            {
                "logical_relative_path": added.logical_relative_path,
                "intended_remote_path": str(
                    REMOTE_PAYLOAD_ROOT / added.logical_relative_path
                ),
                "sha256": sha256_file(added.source_path),
                "byte_size": added.source_path.stat().st_size,
                "payload_role": added.payload_role,
                "owning_package_or_component": added.owning_component,
                "required_import_or_consumer": added.consumer,
                "source_identity": "x3_local_content_addressed",
                "manifest_version": MANIFEST_VERSION,
            }
        )
    entries.sort(key=lambda row: str(row["logical_relative_path"]))
    identity = payload_identity(entries, FUNCTION_IDENTITY)

    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "status": "SEALED_LOCAL_PAID_LAUNCH_READY",
        "base_payload_identity_sha256": BASE_PAYLOAD_IDENTITY_SHA256,
        "sealed_remote_root": str(REMOTE_PAYLOAD_ROOT),
        "manifest_control_remote_path": str(
            REMOTE_PAYLOAD_ROOT / MANIFEST_CONTROL_NAME
        ),
        "identity_algorithm": "sha256(canonical sorted logical path/hash rows plus function identity)",
        "payload_identity_sha256": identity,
        "required_entry_count": len(entries),
        "function_identity": dict(FUNCTION_IDENTITY),
        "entries": entries,
    }


def _source_for_logical(logical: str) -> Path:
    additions = {
        added.logical_relative_path: added.source_path for added in _added_sources()
    }
    if logical in additions:
        return additions[logical]

    return BASE_STAGING_ROOT.joinpath(*PurePosixPath(logical).parts)


def _staging_path(staging_root: Path, entry: Mapping[str, Any]) -> Path:
    intended = PurePosixPath(str(entry["intended_remote_path"]))
    try:
        relative = intended.relative_to(REMOTE_PAYLOAD_ROOT)
    except ValueError as exc:
        raise X3PayloadError(f"entry escapes X3 sealed root: {intended}") from exc

    return staging_root.joinpath(*relative.parts)


def construct_staging_tree(
    staging_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    if staging_root.exists():
        raise X3PayloadError(f"X3 staging root already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    for entry in manifest["entries"]:
        logical = str(entry["logical_relative_path"])
        source = _source_for_logical(logical)
        target = _staging_path(staging_root, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _atomic_json(staging_root / MANIFEST_CONTROL_NAME, manifest)


def verify_staging_tree(
    staging_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise X3PayloadError("unexpected X3 payload schema")
    if manifest.get("function_identity") != FUNCTION_IDENTITY:
        raise X3PayloadError("X3 Modal function identity drifted")
    if manifest.get("base_payload_identity_sha256") != BASE_PAYLOAD_IDENTITY_SHA256:
        raise X3PayloadError("X3 base payload identity drifted")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise X3PayloadError("X3 payload entries are missing")
    observed_identity = payload_identity(entries, FUNCTION_IDENTITY)
    if observed_identity != manifest.get("payload_identity_sha256"):
        raise X3PayloadError("X3 payload identity does not match its entries")
    expected_files = {MANIFEST_CONTROL_NAME}
    for entry in entries:
        path = _staging_path(staging_root, entry)
        relative = path.relative_to(staging_root).as_posix()
        if relative in expected_files:
            raise X3PayloadError(f"duplicate staged X3 path: {relative}")
        expected_files.add(relative)
        if not path.is_file():
            raise X3PayloadError(f"required staged X3 file is missing: {relative}")
        if path.stat().st_size != int(entry["byte_size"]):
            raise X3PayloadError(f"required staged X3 file size mismatch: {relative}")
        if sha256_file(path) != entry["sha256"]:
            raise X3PayloadError(f"required staged X3 file hash mismatch: {relative}")
    observed_files = {
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file()
    }
    undeclared = sorted(observed_files - expected_files)
    missing = sorted(expected_files - observed_files)
    if undeclared:
        raise X3PayloadError(f"undeclared file inside X3 payload: {undeclared[0]}")
    if missing:
        raise X3PayloadError(f"declared X3 payload file is missing: {missing[0]}")
    logical_paths = {str(entry["logical_relative_path"]) for entry in entries}
    if FORBIDDEN_BASE_IDENTITY_PATHS.intersection(logical_paths):
        raise X3PayloadError("X1/X2 campaign or launcher identity entered X3 payload")
    if "deck.csv" in logical_paths:
        raise X3PayloadError("protected root deck.csv entered X3 payload")
    if any(path.startswith("deepmew/submissions/") for path in logical_paths):
        raise X3PayloadError("protected submission surface entered X3 payload")

    return {
        "schema_version": "deepmew.x3_modal_payload_staging_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": observed_identity,
        "required_entry_count": len(entries),
        "observed_declared_file_count": len(observed_files),
        "function_identity": dict(FUNCTION_IDENTITY),
        "undeclared_files": [],
        "missing_files": [],
        "x1_x2_campaign_or_launcher_entries": [],
        "protected_root_deck_packaged": False,
        "protected_submission_surface_packaged": False,
    }


def seal_payload(staging_root: Path = STAGING_ROOT) -> dict[str, Any]:
    manifest = build_manifest()
    _atomic_json(MANIFEST_PATH, manifest)
    construct_staging_tree(staging_root, manifest)
    verification = verify_staging_tree(staging_root, manifest)

    return {
        "schema_version": "deepmew.x3_modal_payload_seal_receipt.v1",
        "status": "SEALED_LOCAL_PAID_LAUNCH_READY",
        "manifest_path": str(MANIFEST_PATH),
        "staging_root": str(staging_root),
        "payload_identity_sha256": manifest["payload_identity_sha256"],
        "function_identity": dict(FUNCTION_IDENTITY),
        "verification": verification,
        "modal_calls": 0,
        "paid_jobs_started": 0,
    }


def main() -> int:
    print(json.dumps(seal_payload(), indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
