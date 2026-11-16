"""Build and verify the isolated, content-addressed H2 Modal payload."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import modal_x3_payload as base


MISSION = ROOT / ".research" / "h2_x3_first_wave_v1" / "h2"
MANIFEST_PATH = MISSION / "H2_MODAL_PAYLOAD_MANIFEST_V1_7.json"
STAGING_ROOT = MISSION / "payload_v1_7_staging_authorized_relaunch"
REMOTE_PAYLOAD_ROOT = PurePosixPath("/opt/deepmew_h2_payload")
MANIFEST_CONTROL_NAME = "H2_MODAL_PAYLOAD_MANIFEST_V1_7.json"
MANIFEST_SCHEMA = "deepmew.h2_modal_payload_manifest.v1_7"
MANIFEST_VERSION = "1.7.0"
FUNCTION_IDENTITY = {
    "modal_app_name": "deepmew-h2-e1-starmie-repair-relaunch-20260721",
    "modal_function_name": "train_h2_e1_starmie_repair_v2",
    "modal_volume_name": "deepmew-h2-e1-starmie-repair-relaunch-20260721",
    "remote_payload_root": str(REMOTE_PAYLOAD_ROOT),
    "remote_output_namespace": "/h2/training_v2",
    "gpu": "L4",
    "gpu_count": 1,
    "max_containers": 1,
    "automatic_retries": 0,
}


class H2PayloadError(RuntimeError):
    """The H2 payload identity or staging closure failed."""


@dataclass(frozen=True)
class AddedSource:
    source_path: Path
    logical_relative_path: str
    payload_role: str
    owning_component: str
    consumer: str


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H2PayloadError(f"expected JSON object: {path}")

    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            encoding="ascii", newline="\n", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(value), stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _added_sources() -> tuple[AddedSource, ...]:
    fixed = (
        AddedSource(ROOT / "deepmew" / "rl" / "opponents.py", "deepmew/rl/opponents.py", "REQUIRED_CODE", "policy/deck opponent materialization", "terminal PPO runner"),
        AddedSource(ROOT / "scripts" / "run_terminal_league_ppo.py", "scripts/run_terminal_league_ppo.py", "REQUIRED_CODE", "terminal PPO runner", "H2 training subprocess"),
        AddedSource(ROOT / "scripts" / "modal_x3_payload.py", "scripts/modal_x3_payload.py", "REQUIRED_CODE", "shared content-addressed base-runtime helper", "H2 payload builder import"),
        AddedSource(ROOT / "scripts" / "modal_h2_payload.py", "scripts/modal_h2_payload.py", "REQUIRED_CODE", "H2 payload", "H2 payload verifier"),
        AddedSource(ROOT / "scripts" / "modal_h2_training.py", "scripts/modal_h2_training.py", "REQUIRED_CODE", "H2 launcher", "separate H2 Modal function"),
        AddedSource(MISSION / "H2_CAMPAIGN_CONFIG.json", ".research/h2_x3_first_wave_v1/h2/H2_CAMPAIGN_CONFIG.json", "REQUIRED_CONFIGURATION", "H2 campaign", "terminal PPO runner"),
        AddedSource(MISSION / "H2_CAMPAIGN_MANIFEST.json", ".research/h2_x3_first_wave_v1/h2/H2_CAMPAIGN_MANIFEST.json", "REQUIRED_CONFIGURATION", "H2 scientific identity", "H2 launcher preflight"),
        AddedSource(MISSION / "H2_DECK_MANIFEST.json", ".research/h2_x3_first_wave_v1/h2/H2_DECK_MANIFEST.json", "REQUIRED_CONFIGURATION", "H2 deck identities", "policy/deck resolver"),
        AddedSource(MISSION / "H2_REMOTE_POLICY_ARTIFACT_MAP.json", ".research/h2_x3_first_wave_v1/h2/H2_REMOTE_POLICY_ARTIFACT_MAP.json", "REQUIRED_CONFIGURATION", "H2 artifact remapping", "terminal PPO runner"),
        AddedSource(MISSION / "H2_MODAL_HARDWARE_AND_COST_GUARD.json", ".research/h2_x3_first_wave_v1/h2/H2_MODAL_HARDWARE_AND_COST_GUARD.json", "REQUIRED_CONFIGURATION", "H2 separate cost guard", "H2 paid preflight"),
        AddedSource(MISSION / "H2_E1_CRITIC_COMPATIBILITY_RECEIPT.json", ".research/h2_x3_first_wave_v1/h2/H2_E1_CRITIC_COMPATIBILITY_RECEIPT.json", "REQUIRED_RECEIPT", "E1 actor/critic compatibility", "H2 launcher preflight"),
        AddedSource(MISSION / "H2_HARDTAIL_IDENTITY_CORRECTION_RECEIPT.json", ".research/h2_x3_first_wave_v1/h2/H2_HARDTAIL_IDENTITY_CORRECTION_RECEIPT.json", "REQUIRED_RECEIPT", "H2 hard-tail identity correction", "H2 scientific audit"),
        AddedSource(Path("D:/DeepMew_artifacts/checkpoints/x1-x2-direct-exploiter-first-wave-x2-final-actor.pt"), ".research/h2_x3_first_wave_v1/assets/x2_actor.pt", "REQUIRED_ARTIFACT", "X2 PPO opponent", "H2 opponent loader"),
        AddedSource(ROOT / ".research/terminal_league_starmie_grimmsnarl_v0/e1_starmie_complete_enhanced_generation_20260714/production_e1_20260716/stage_05_terminal_ppo.pt", ".research/h2_x3_first_wave_v1/assets/e1_actor_critic.pt", "REQUIRED_ARTIFACT", "E1 frozen critic", "H2 critic initializer"),
    )
    reference_decks = tuple(
        AddedSource(path, f".research/h2_x3_first_wave_v1/h2/reference_decks/{path.name}", "REQUIRED_DECK", "H2 frozen reference deck", "policy/deck resolver")
        for path in sorted((MISSION / "reference_decks").glob("*.deck.csv"))
    )

    return fixed + reference_decks


def payload_identity(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {"logical_relative_path": str(row["logical_relative_path"]), "sha256": str(row["sha256"])}
        for row in sorted(entries, key=lambda item: str(item["logical_relative_path"]))
    ]

    return hashlib.sha256(_canonical_json_bytes({"entries": rows, "function_identity": FUNCTION_IDENTITY})).hexdigest()


def build_manifest() -> dict[str, Any]:
    entries = base.base_runtime_entries(REMOTE_PAYLOAD_ROOT)
    observed = {str(entry["logical_relative_path"]) for entry in entries}
    for added in _added_sources():
        if not added.source_path.is_file():
            raise H2PayloadError(f"required H2 source is missing: {added.source_path}")
        if added.logical_relative_path in observed:
            raise H2PayloadError(f"duplicate H2 payload path: {added.logical_relative_path}")
        observed.add(added.logical_relative_path)
        entries.append({
            "logical_relative_path": added.logical_relative_path,
            "intended_remote_path": str(REMOTE_PAYLOAD_ROOT / added.logical_relative_path),
            "sha256": sha256_file(added.source_path),
            "byte_size": added.source_path.stat().st_size,
            "payload_role": added.payload_role,
            "owning_package_or_component": added.owning_component,
            "required_import_or_consumer": added.consumer,
            "source_identity": "h2_local_content_addressed",
            "manifest_version": MANIFEST_VERSION,
        })
    entries.sort(key=lambda row: str(row["logical_relative_path"]))

    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "status": "SEALED_LOCAL_PAID_LAUNCH_READY",
        "base_payload_identity_sha256": base.BASE_PAYLOAD_IDENTITY_SHA256,
        "sealed_remote_root": str(REMOTE_PAYLOAD_ROOT),
        "manifest_control_remote_path": str(REMOTE_PAYLOAD_ROOT / MANIFEST_CONTROL_NAME),
        "payload_identity_sha256": payload_identity(entries),
        "required_entry_count": len(entries),
        "function_identity": dict(FUNCTION_IDENTITY),
        "entries": entries,
    }


def _source_for_logical(logical: str) -> Path:
    additions = {added.logical_relative_path: added.source_path for added in _added_sources()}
    if logical in additions:
        return additions[logical]

    return base.BASE_STAGING_ROOT.joinpath(*PurePosixPath(logical).parts)


def _staging_path(staging_root: Path, entry: Mapping[str, Any]) -> Path:
    intended = PurePosixPath(str(entry["intended_remote_path"]))
    try:
        relative = intended.relative_to(REMOTE_PAYLOAD_ROOT)
    except ValueError as exc:
        raise H2PayloadError(f"entry escapes H2 sealed root: {intended}") from exc

    return staging_root.joinpath(*relative.parts)


def construct_staging_tree(staging_root: Path, manifest: Mapping[str, Any]) -> None:
    if staging_root.exists():
        raise H2PayloadError(f"H2 staging root already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    for entry in manifest["entries"]:
        source = _source_for_logical(str(entry["logical_relative_path"]))
        target = _staging_path(staging_root, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _atomic_json(staging_root / MANIFEST_CONTROL_NAME, manifest)


def verify_staging_tree(staging_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise H2PayloadError("unexpected H2 payload schema")
    if manifest.get("function_identity") != FUNCTION_IDENTITY:
        raise H2PayloadError("H2 function identity drifted")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise H2PayloadError("H2 payload entries are missing")
    identity = payload_identity(entries)
    if identity != manifest.get("payload_identity_sha256"):
        raise H2PayloadError("H2 payload identity disagrees")
    expected_files = {MANIFEST_CONTROL_NAME}
    for entry in entries:
        path = _staging_path(staging_root, entry)
        relative = path.relative_to(staging_root).as_posix()
        if relative in expected_files:
            raise H2PayloadError(f"duplicate staged H2 path: {relative}")
        expected_files.add(relative)
        if not path.is_file() or path.stat().st_size != int(entry["byte_size"]) or sha256_file(path) != entry["sha256"]:
            raise H2PayloadError(f"staged H2 file failed hash/size validation: {relative}")
    observed_files = {path.relative_to(staging_root).as_posix() for path in staging_root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise H2PayloadError("H2 staged closure has missing or undeclared files")
    logical_paths = {str(entry["logical_relative_path"]) for entry in entries}
    if "deck.csv" in logical_paths or any(path.startswith("deepmew/submissions/") for path in logical_paths):
        raise H2PayloadError("protected runtime/submission surface entered H2 payload")

    return {
        "schema_version": "deepmew.h2_modal_payload_staging_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": identity,
        "required_entry_count": len(entries),
        "observed_declared_file_count": len(observed_files),
        "function_identity": dict(FUNCTION_IDENTITY),
        "missing_files": [],
        "undeclared_files": [],
        "protected_root_deck_packaged": False,
        "protected_submission_surface_packaged": False,
    }


def seal_payload(staging_root: Path = STAGING_ROOT) -> dict[str, Any]:
    manifest = build_manifest()
    _atomic_json(MANIFEST_PATH, manifest)
    construct_staging_tree(staging_root, manifest)
    verification = verify_staging_tree(staging_root, manifest)

    return {
        "schema_version": "deepmew.h2_modal_payload_seal_receipt.v1",
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
