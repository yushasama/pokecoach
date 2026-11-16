"""Build and verify the isolated D1-successor Modal payload.

Importing or sealing this module performs local filesystem work only. It never
creates a Modal app, function, container, or paid task.
"""

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


MISSION = ROOT / ".research" / "d1_successor_mixed_league_v1"
MANIFEST_PATH = MISSION / "D1_SUCCESSOR_MODAL_PAYLOAD_MANIFEST_V1_1.json"
STAGING_ROOT = MISSION / "payload_v1_1_staging_authorized"
REMOTE_PAYLOAD_ROOT = PurePosixPath("/opt/deepmew_d1_successor_payload")
MANIFEST_CONTROL_NAME = "D1_SUCCESSOR_MODAL_PAYLOAD_MANIFEST_V1_1.json"
MANIFEST_SCHEMA = "deepmew.d1_successor_modal_payload_manifest.v1_1"
MANIFEST_VERSION = "1.1.0"
FUNCTION_IDENTITY = {
    "modal_app_name": "deepmew-d1-successor-mixed-league-20260722",
    "modal_function_name": "train_d1_successor_mixed_league_v1",
    "modal_volume_name": "deepmew-d1-successor-mixed-league-20260722",
    "remote_payload_root": str(REMOTE_PAYLOAD_ROOT),
    "remote_output_namespace": "/d1_successor/training_v1",
    "gpu": "L4",
    "gpu_count": 1,
    "max_containers": 1,
    "automatic_retries": 0,
}


class D1SuccessorPayloadError(RuntimeError):
    """The D1-successor payload identity or staging closure failed."""


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
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D1SuccessorPayloadError(f"expected JSON object: {path}")

    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
            json.dump(dict(value), stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _added_sources() -> tuple[AddedSource, ...]:
    mission_files = (
        "CAMPAIGN_CONFIG.json",
        "FROZEN_MIXTURE_RECEIPT.json",
        "SUCCESSOR_INITIALIZATION_RECEIPT.json",
        "D1_SUCCESSOR_REMOTE_POLICY_ARTIFACT_MAP.json",
        "D1_SUCCESSOR_MODAL_HARDWARE_AND_COST_GUARD.json",
        "D1_SUCCESSOR_PAID_AUTHORIZATION_RECEIPT.json",
        "D1_SUCCESSOR_LIVE_PRELAUNCH_RECONCILIATION.json",
    )
    fixed = [
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
            "D1-successor training subprocess",
        ),
        AddedSource(
            ROOT / "scripts" / "modal_x3_payload.py",
            "scripts/modal_x3_payload.py",
            "REQUIRED_CODE",
            "content-addressed base-runtime helper",
            "D1-successor payload builder import",
        ),
        AddedSource(
            ROOT / "scripts" / "modal_d1_successor_payload.py",
            "scripts/modal_d1_successor_payload.py",
            "REQUIRED_CODE",
            "D1-successor payload",
            "D1-successor payload verifier",
        ),
        AddedSource(
            ROOT / "scripts" / "modal_d1_successor_training.py",
            "scripts/modal_d1_successor_training.py",
            "REQUIRED_CODE",
            "D1-successor launcher",
            "separate D1-successor Modal function",
        ),
        AddedSource(
            ROOT / ".research" / "h2_x3_first_wave_v1" / "h2" / "H2_DECK_MANIFEST.json",
            ".research/h2_x3_first_wave_v1/h2/H2_DECK_MANIFEST.json",
            "REQUIRED_CONFIGURATION",
            "frozen deck identities",
            "policy/deck resolver",
        ),
    ]
    for name in mission_files:
        fixed.append(
            AddedSource(
                MISSION / name,
                f".research/d1_successor_mixed_league_v1/{name}",
                "REQUIRED_CONFIGURATION",
                "D1-successor frozen contract",
                "D1-successor launcher and runner",
            )
        )
    assets = (
        AddedSource(
            ROOT / ".research/terminal_league_starmie_grimmsnarl_v0/control_closeout_starmie_20260713/remote/final_actor.pt",
            ".research/d1_successor_mixed_league_v1/assets/d1_actor.pt",
            "REQUIRED_ARTIFACT",
            "D1 frozen actor parent",
            "learner initialization and opponent",
        ),
        AddedSource(
            ROOT / ".research/terminal_league_starmie_grimmsnarl_v0/control_closeout_starmie_20260713/remote/final_league_checkpoint.pt",
            ".research/d1_successor_mixed_league_v1/assets/d1_actor_critic.pt",
            "REQUIRED_ARTIFACT",
            "D1 compatible critic",
            "critic initialization",
        ),
        AddedSource(
            ROOT / ".research/terminal_league_starmie_grimmsnarl_v0/control_closeout_starmie_20260713/remote/config_receipt.json",
            ".research/d1_successor_mixed_league_v1/assets/d1_critic_config_receipt.json",
            "REQUIRED_RECEIPT",
            "D1 legacy PPO configuration",
            "critic initialization",
        ),
        AddedSource(
            Path("D:/DeepMew_artifacts/checkpoints/h2-x3-first-wave-h2-final-actor.pt"),
            ".research/d1_successor_mixed_league_v1/assets/h2_actor.pt",
            "REQUIRED_ARTIFACT",
            "H2 qualified repair child",
            "opponent loader",
        ),
        AddedSource(
            ROOT / ".research/terminal_league_starmie_grimmsnarl_v0/e1_starmie_complete_enhanced_generation_20260714/production_e1_20260716/stage_05_terminal_ppo.pt.actor.pt",
            ".research/d1_successor_mixed_league_v1/assets/e1_actor.pt",
            "REQUIRED_ARTIFACT",
            "E1 diversity opponent",
            "opponent loader",
        ),
        AddedSource(
            Path("D:/DeepMew_artifacts/checkpoints/r22-staryu-full-league-a4-8c-r7-BEST.pt"),
            ".research/d1_successor_mixed_league_v1/assets/r7_actor.pt",
            "REQUIRED_ARTIFACT",
            "r7 historical anchor",
            "opponent loader",
        ),
        AddedSource(
            Path("D:/DeepMew_artifacts/checkpoints/x1-x2-direct-exploiter-first-wave-x2-final-actor.pt"),
            ".research/d1_successor_mixed_league_v1/assets/x2_actor.pt",
            "REQUIRED_ARTIFACT",
            "X2 qualified exploiter",
            "opponent loader",
        ),
        AddedSource(
            ROOT / ".research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt",
            ".research/d1_successor_mixed_league_v1/assets/r12_hardtail_actor.pt",
            "REQUIRED_ARTIFACT",
            "r12 supported hard tail",
            "opponent loader",
        ),
    )
    reference_decks = tuple(
        AddedSource(
            path,
            f".research/h2_x3_first_wave_v1/h2/reference_decks/{path.name}",
            "REQUIRED_DECK",
            "frozen reference deck",
            "policy/deck resolver",
        )
        for path in sorted(
            (ROOT / ".research" / "h2_x3_first_wave_v1" / "h2" / "reference_decks").glob("*.deck.csv")
        )
    )

    return tuple(fixed) + assets + reference_decks


def payload_identity(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "logical_relative_path": str(row["logical_relative_path"]),
            "sha256": str(row["sha256"]),
        }
        for row in sorted(entries, key=lambda item: str(item["logical_relative_path"]))
    ]

    return hashlib.sha256(
        _canonical_json_bytes({"entries": rows, "function_identity": FUNCTION_IDENTITY})
    ).hexdigest()


def build_manifest() -> dict[str, Any]:
    entries = base.base_runtime_entries(REMOTE_PAYLOAD_ROOT)
    observed = {str(entry["logical_relative_path"]) for entry in entries}
    for added in _added_sources():
        if not added.source_path.is_file():
            raise D1SuccessorPayloadError(f"required source is missing: {added.source_path}")
        if added.logical_relative_path in observed:
            raise D1SuccessorPayloadError(f"duplicate payload path: {added.logical_relative_path}")
        observed.add(added.logical_relative_path)
        entries.append(
            {
                "logical_relative_path": added.logical_relative_path,
                "intended_remote_path": str(REMOTE_PAYLOAD_ROOT / added.logical_relative_path),
                "sha256": sha256_file(added.source_path),
                "byte_size": added.source_path.stat().st_size,
                "payload_role": added.payload_role,
                "owning_package_or_component": added.owning_component,
                "required_import_or_consumer": added.consumer,
                "source_identity": "d1_successor_local_content_addressed",
                "manifest_version": MANIFEST_VERSION,
            }
        )
    entries.sort(key=lambda row: str(row["logical_relative_path"]))

    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "status": "SEALED_PAID_LAUNCH_READY",
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
        raise D1SuccessorPayloadError(f"entry escapes sealed root: {intended}") from exc

    return staging_root.joinpath(*relative.parts)


def construct_staging_tree(staging_root: Path, manifest: Mapping[str, Any]) -> None:
    if staging_root.exists():
        raise D1SuccessorPayloadError(f"staging root already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    for entry in manifest["entries"]:
        source = _source_for_logical(str(entry["logical_relative_path"]))
        target = _staging_path(staging_root, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _atomic_json(staging_root / MANIFEST_CONTROL_NAME, manifest)


def verify_staging_tree(staging_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise D1SuccessorPayloadError("unexpected payload schema")
    if manifest.get("function_identity") != FUNCTION_IDENTITY:
        raise D1SuccessorPayloadError("function identity drifted")
    if manifest.get("base_payload_identity_sha256") != base.BASE_PAYLOAD_IDENTITY_SHA256:
        raise D1SuccessorPayloadError("base payload identity drifted")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise D1SuccessorPayloadError("payload entries are missing")
    identity = payload_identity(entries)
    if identity != manifest.get("payload_identity_sha256"):
        raise D1SuccessorPayloadError("payload identity disagrees")
    expected_files = {MANIFEST_CONTROL_NAME}
    for entry in entries:
        path = _staging_path(staging_root, entry)
        relative = path.relative_to(staging_root).as_posix()
        if relative in expected_files:
            raise D1SuccessorPayloadError(f"duplicate staged path: {relative}")
        expected_files.add(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["byte_size"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise D1SuccessorPayloadError(f"staged file failed validation: {relative}")
    observed_files = {
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise D1SuccessorPayloadError("staged closure has missing or undeclared files")
    logical_paths = {str(entry["logical_relative_path"]) for entry in entries}
    if "deck.csv" in logical_paths or any(path.startswith("deepmew/submissions/") for path in logical_paths):
        raise D1SuccessorPayloadError("protected runtime/submission surface entered payload")

    return {
        "schema_version": "deepmew.d1_successor_modal_payload_staging_verification.v1",
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
        "schema_version": "deepmew.d1_successor_modal_payload_seal_receipt.v1",
        "status": "SEALED_PAID_LAUNCH_READY",
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
