"""Hash-verified additive runner for completed postmortem artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from deepmew.submission_postmortem.addendum import (
    ADDENDUM_SCHEMA_VERSION,
    DEFAULT_PERMUTATION_SAMPLES,
    DEFAULT_RANDOM_SEED,
    build_statistical_addendum,
    render_statistical_addendum,
)
from deepmew.submission_postmortem.io import write_csv, write_json, write_text
from deepmew.submission_postmortem.pipeline import _privacy_sweep
from deepmew.submission_postmortem.scoring import sha256_file


ADDENDUM_RUN_SCHEMA_VERSION = "deepmew.submission_postmortem.addendum_run.v0"
REQUIRED_BASE_OUTPUTS = (
    "episodes.json",
    "decisions.jsonl",
    "report.md",
    "privacy_sweep.json",
)


class AddendumRunError(RuntimeError):
    """Raised when a base-run or addendum safety gate fails closed."""


@dataclass(frozen=True)
class AddendumRunResult:
    output_dir: Path
    manifest_path: Path
    addendum_path: Path
    payload_path: Path
    payload: Mapping[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AddendumRunError(f"could not read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AddendumRunError(f"expected JSON object: {path}")

    return value


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AddendumRunError(f"could not read JSON array: {path}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise AddendumRunError(f"expected array of objects: {path}")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AddendumRunError(f"expected JSON object at {path}:{line_number}")
                output.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AddendumRunError(f"could not read JSONL: {path}: {exc}") from exc

    return output


def _verify_output_hashes(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AddendumRunError("base manifest has no output hash registry")
    for name in REQUIRED_BASE_OUTPUTS:
        if name not in outputs:
            raise AddendumRunError(f"base manifest does not bind required output: {name}")
    verified = {}
    for name, receipt in outputs.items():
        if not isinstance(receipt, Mapping):
            raise AddendumRunError(f"invalid output receipt for {name}")
        path = run_dir / str(name)
        if not path.is_file():
            raise AddendumRunError(f"base output missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = str(receipt.get("sha256") or "").lower()
        if actual_hash.lower() != expected_hash:
            raise AddendumRunError(
                f"base output hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        expected_size = receipt.get("byte_size")
        if isinstance(expected_size, int) and path.stat().st_size != expected_size:
            raise AddendumRunError(f"base output size mismatch for {name}")
        verified[str(name)] = actual_hash

    return verified


def _output_receipts(output_dir: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": sha256_file(output_dir / name),
            "byte_size": (output_dir / name).stat().st_size,
        }
        for name in names
    }


def run_statistical_addendum(
    base_run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    human_addendum_path: str | Path | None = None,
    authoritative_report_path: str | Path | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
    command: str = "",
) -> AddendumRunResult:
    """Verify a completed base run and write only additive artifacts."""

    base_dir = Path(base_run_dir)
    manifest_path = base_dir / "manifest.json"
    if not base_dir.is_dir() or not manifest_path.is_file():
        raise AddendumRunError(f"completed base run not found: {base_dir}")
    base_manifest = _read_json_object(manifest_path)
    if base_manifest.get("status") != "complete":
        raise AddendumRunError("base postmortem manifest is not complete")
    base_privacy = _read_json_object(base_dir / "privacy_sweep.json")
    if base_privacy.get("passed") is not True:
        raise AddendumRunError("base postmortem privacy sweep did not pass")
    verified_outputs = _verify_output_hashes(base_dir, base_manifest)
    episodes = _read_json_array(base_dir / "episodes.json")
    decisions = _read_jsonl(base_dir / "decisions.jsonl")
    artifacts = build_statistical_addendum(
        episodes,
        decisions,
        random_seed=random_seed,
        permutation_samples=permutation_samples,
    )
    target_dir = Path(output_dir) if output_dir is not None else base_dir / "addendum-v0"
    target_dir.mkdir(parents=True, exist_ok=True)
    payload_path = write_json(target_dir / "statistical_addendum.json", artifacts.payload)
    phase_files = {
        "rating_stability.json": artifacts.payload["phase_a_rating_stability"],
        "rolling_performance.json": artifacts.payload["phase_b_rolling_performance"],
        "matchup_normalization.json": artifacts.payload["phase_c_exposure_normalization"],
        "initiative_effect.json": artifacts.payload["phase_d_first_player_effect"],
        "loss_timeline_summary.json": artifacts.payload["phase_e_loss_timeline"],
        "confidence_diagnostics.json": artifacts.payload["phase_f_confidence_diagnostics"],
        "observational_loss_model.json": artifacts.payload["phase_g_observational_loss_model"],
        "failure_clustering.json": artifacts.payload["phase_h_failure_clustering"],
        "engineering_priority.json": artifacts.payload["phase_i_engineering_priority"],
        "data_value.json": artifacts.payload["phase_j_data_vs_policy"],
        "decision_memo.json": artifacts.payload["phase_k_decision_memo"],
    }
    for name, value in phase_files.items():
        write_json(target_dir / name, value)
    for name, rows in artifacts.tables.items():
        write_csv(target_dir / name, rows)
    addendum_text = render_statistical_addendum(artifacts.payload)
    addendum_path = write_text(target_dir / "ADDENDUM.md", addendum_text)
    decision_memo_text = "# Decision Memo\n\n" + addendum_text.split("## Phase K - Decision memo\n", 1)[-1]
    write_text(target_dir / "decision_memo.md", decision_memo_text)
    if human_addendum_path is not None:
        write_text(human_addendum_path, addendum_text)

    authoritative_report_receipt = None
    if authoritative_report_path is not None:
        authoritative_report = Path(authoritative_report_path)
        if not authoritative_report.is_file():
            raise AddendumRunError(f"authoritative human report not found: {authoritative_report}")
        authoritative_report_receipt = {
            "path": str(authoritative_report),
            "sha256": sha256_file(authoritative_report),
            "byte_size": authoritative_report.stat().st_size,
        }

    privacy = _privacy_sweep(target_dir)
    write_json(target_dir / "privacy_sweep.json", privacy)
    if privacy.get("passed") is not True:
        raise AddendumRunError(f"addendum privacy sweep failed: {privacy.get('violations')}")
    output_names = [
        "statistical_addendum.json",
        *phase_files,
        *artifacts.tables,
        "ADDENDUM.md",
        "decision_memo.md",
        "privacy_sweep.json",
    ]
    manifest = {
        "schema_version": ADDENDUM_RUN_SCHEMA_VERSION,
        "analysis_schema_version": ADDENDUM_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "status": "complete",
        "base_run": {
            "path": str(base_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "submission_id": base_manifest.get("submission_id"),
            "schema_version": base_manifest.get("schema_version"),
            "verified_output_count": len(verified_outputs),
            "report_sha256": verified_outputs["report.md"],
            "episodes_sha256": verified_outputs["episodes.json"],
            "decisions_sha256": verified_outputs["decisions.jsonl"],
            "privacy_passed": True,
            "authoritative_human_report": authoritative_report_receipt,
        },
        "random_seed": random_seed,
        "permutation_samples": permutation_samples,
        "information_boundary": artifacts.payload["source_contract"],
        "outputs": _output_receipts(target_dir, output_names),
        "privacy_sweep": privacy,
    }
    addendum_manifest_path = write_json(target_dir / "manifest.json", manifest)

    return AddendumRunResult(
        output_dir=target_dir,
        manifest_path=addendum_manifest_path,
        addendum_path=addendum_path,
        payload_path=payload_path,
        payload=artifacts.payload,
    )


__all__ = [
    "ADDENDUM_RUN_SCHEMA_VERSION",
    "AddendumRunError",
    "AddendumRunResult",
    "run_statistical_addendum",
]
