"""Admit the frozen CABT 1.32 submission-postmortem corpus canonically.

This command is offline-only.  It consumes the existing replay cache and
postmortem outputs, verifies them in place, and writes compact receipts.  It
does not download, reconstruct, or persist raw observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.replay.contract.causal_alignment import (
    EXPECTED_ACTION_SCHEMA_VERSION,
    KaggleEpisodeRepository,
    align_decision,
    build_admission_transition_registry,
)
from deepmew.replay.contract.identities import sha256_json
from deepmew.replay.contract.schemas import (
    CanonicalDecisionImplementationHashes,
    ValidationOutcome,
)
from deepmew.replay.contract.source_events import (
    compute_source_event_id,
    source_event_from_admission,
)
from deepmew.replay.contract.validation import canonicalize_decision
from deepmew.replay.information_boundary import InformationMode
from deepmew.submission_postmortem.replay import load_replay, project_actor_observation


DEFAULT_SUITE = ROOT / ".research" / "submission_postmortem_suite" / (
    "submission-54788723-full-v0"
)
DEFAULT_OUTPUT = ROOT / ".research" / "cap001_cabt132_replay_admission_v0"
ACTION_SCHEMA_PATHS = (
    "deepmew/replay/contract/schemas.py",
    "deepmew/replay/contract/legal_actions.py",
    "deepmew/replay/contract/chosen_actions.py",
)
CONTRACT_PATHS = (
    "deepmew/replay/contract/causal_alignment.py",
    "deepmew/replay/contract/kaggle_sources.py",
    "deepmew/replay/contract/source_events.py",
    "deepmew/replay/contract/validation.py",
    *ACTION_SCHEMA_PATHS,
    "deepmew/replay/information_boundary.py",
    "deepmew/submission_postmortem/replay.py",
)
FORBIDDEN_FIELDS = frozenset({"logs", "search_begin_input", "looking", "deck"})


class AdmissionHardStop(RuntimeError):
    """Raised whenever the exact canonical-admission contract is incomplete."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _ordered_file_set_sha256(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        normalized = Path(relative).as_posix()
        path = ROOT / relative
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdmissionHardStop(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise AdmissionHardStop(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)

    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _implementation_hashes() -> tuple[CanonicalDecisionImplementationHashes, dict[str, Any]]:
    manifest = {
        relative: _sha256_path(ROOT / relative)
        for relative in CONTRACT_PATHS
    }
    manifest["ordered_action_schema_sha256"] = _ordered_file_set_sha256(
        ACTION_SCHEMA_PATHS
    )
    manifest["contract_file_set_sha256"] = _ordered_file_set_sha256(CONTRACT_PATHS)
    typed = CanonicalDecisionImplementationHashes(
        causal_aligner_implementation_sha256=manifest[
            "deepmew/replay/contract/causal_alignment.py"
        ],
        legality_validator_implementation_sha256=manifest[
            "deepmew/replay/contract/validation.py"
        ],
        action_schema_implementation_sha256=manifest[
            "ordered_action_schema_sha256"
        ],
    )

    return typed, manifest


def _verify_suite(suite: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(suite / "manifest.json")
    binding = _read_json(suite / "binding.json")
    cache = _read_json(suite / "replay_cache.json")
    if not isinstance(manifest, dict) or not isinstance(binding, dict):
        raise AdmissionHardStop("suite manifest or binding is not an object")
    if not isinstance(cache, dict) or cache.get("complete") is not True:
        raise AdmissionHardStop("frozen replay cache is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AdmissionHardStop("suite output hash manifest is absent")
    for name, expected in outputs.items():
        path = suite / str(name)
        if not path.is_file() or not isinstance(expected, Mapping):
            raise AdmissionHardStop(f"suite output is absent: {name}")
        if path.stat().st_size != expected.get("byte_size"):
            raise AdmissionHardStop(f"suite output size mismatch: {name}")
        if _sha256_path(path) != expected.get("sha256"):
            raise AdmissionHardStop(f"suite output hash mismatch: {name}")
    records = cache.get("records")
    if not isinstance(records, list) or not records:
        raise AdmissionHardStop("replay cache has no records")
    normalized = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "cached":
            raise AdmissionHardStop("replay cache contains a non-cached record")
        path = Path(str(record.get("path")))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise AdmissionHardStop(f"cached replay is absent: {path}")
        if path.stat().st_size != record.get("byte_size"):
            raise AdmissionHardStop(f"cached replay size mismatch: {path}")
        if _sha256_path(path) != record.get("sha256"):
            raise AdmissionHardStop(f"cached replay hash mismatch: {path}")
        normalized.append({**record, "path": str(path.resolve())})

    return manifest, binding, normalized


def _raw_episode(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise AdmissionHardStop(f"replay root is not an object: {path}")

    return value


def _actor_outcome(data: Mapping[str, Any], seat: int) -> float:
    rewards = data.get("rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or rewards[seat] not in (-1, 0, 1, -1.0, 0.0, 1.0)
    ):
        raise AdmissionHardStop("terminal actor-relative outcome is unavailable")

    return (float(rewards[seat]) + 1.0) / 2.0


def _anchor_event(
    *,
    admission_identity: str,
    binding_id: str,
    local_row: int,
    global_row: int,
    replay_id: str,
    episode_id: str,
    source_path: Path,
    source_sha256: str,
    data: Mapping[str, Any],
    decision: Any,
    next_decision: Any | None,
):
    steps = data.get("steps")
    if not isinstance(steps, list) or decision.action_step >= len(steps):
        raise AdmissionHardStop(f"result step absent for {decision.decision_id}")
    current_step = steps[decision.action_step]
    if not isinstance(current_step, list) or decision.seat >= len(current_step):
        raise AdmissionHardStop(f"result seat absent for {decision.decision_id}")
    current_agent = current_step[decision.seat]
    if not isinstance(current_agent, Mapping):
        raise AdmissionHardStop(f"result agent invalid for {decision.decision_id}")
    result_observation = current_agent.get("observation")
    if not isinstance(result_observation, Mapping):
        raise AdmissionHardStop(f"result observation absent for {decision.decision_id}")
    projected_result_observation = project_actor_observation(result_observation)
    terminal = next_decision is None
    state_before = {
        "feature_row": decision.action_step,
        "representation": "raw_kaggle_actor_observation.v1",
        "source_observation": {
            "source_sha256": source_sha256,
            "source_path": str(source_path),
            "source_line": decision.action_step,
            "content_sha256": sha256_json(projected_result_observation),
            "source_field": "state_json",
        },
    }
    transition: dict[str, Any] = {
        "schema_version": EXPECTED_ACTION_SCHEMA_VERSION,
        "source_schema": "deepmew.replay_action_row.v1",
        "source_id": "kaggle_ingest",
        "source_game_id": replay_id,
        "canonical_game_id": episode_id,
        "source_hash": source_sha256,
        "actor": decision.seat,
        "global_decision_index": decision.action_step,
        "decision_id": decision.decision_id,
        "state_before": state_before,
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "chosen_action": {
            "indices": list(decision.action),
            "index_space": "option_position",
        },
        "terminal": terminal,
        "terminal_actor_relative_result": _actor_outcome(data, decision.seat),
    }
    if next_decision is not None:
        transition.update({
            "next_same_actor_state": {
                "decision_id": next_decision.decision_id,
                "feature_row": next_decision.action_step,
                "representation": "raw_kaggle_actor_observation.v1",
            },
            "transition_horizon_logical_decisions": (
                next_decision.action_step - decision.action_step
            ),
        })
    source_record = {
        "schema_version": "deepmew.replay_action_row.v1",
        "source_kind": "kaggle_replay",
        "source_path": str(source_path),
        "source_artifact_sha256": source_sha256,
        "game_id": replay_id,
        "player_id": decision.seat,
        "step": decision.action_step,
        "state_json": projected_result_observation,
        "selected": list(decision.action),
    }
    event = source_event_from_admission(
        admission_identity=admission_identity,
        binding_id=binding_id,
        local_row=local_row,
        global_row=global_row,
        transition=transition,
        source_record=source_record,
    )
    if event.source_corrupt:
        raise AdmissionHardStop(f"anchor source event is corrupt: {decision.decision_id}")

    return replace(event, event_id=compute_source_event_id(event))


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_FIELDS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_keys(child))

    return found


def _admit_once(
    *,
    records: list[dict[str, Any]],
    binding: Mapping[str, Any],
    signals: Mapping[str, Mapping[str, Any]],
    implementation_hashes: CanonicalDecisionImplementationHashes,
    admission_identity: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    identity_rows = []
    compatibility_failures = []
    global_row = 0
    submitted_hash = binding.get("submitted_deck_hash")
    if not isinstance(submitted_hash, str) or not submitted_hash:
        raise AdmissionHardStop("submitted deck binding hash is absent")
    for cache_record in records:
        source_path = Path(cache_record["path"])
        episode = load_replay(source_path, submitted_deck_hash=submitted_hash)
        if episode.errors or episode.alignment_status != "structurally_verified_unadmitted":
            raise AdmissionHardStop(f"postmortem extraction rejected {source_path}")
        if episode.source_sha256 != cache_record["sha256"]:
            raise AdmissionHardStop(f"postmortem source hash mismatch: {source_path}")
        data = _raw_episode(source_path)
        decisions = tuple(sorted(
            episode.decisions,
            key=lambda item: (item.action_step, item.seat),
        ))
        next_by_id = {}
        for seat in (0, 1):
            seat_decisions = [item for item in decisions if item.seat == seat]
            for index, item in enumerate(seat_decisions):
                next_by_id[item.decision_id] = (
                    seat_decisions[index + 1]
                    if index + 1 < len(seat_decisions)
                    else None
                )
        episode_binding_id = f"submission-54788723:{episode.episode_id}"
        anchors = tuple(
            _anchor_event(
                admission_identity=admission_identity,
                binding_id=episode_binding_id,
                local_row=index,
                global_row=global_row + index,
                replay_id=episode.replay_id,
                episode_id=episode.episode_id,
                source_path=source_path,
                source_sha256=episode.source_sha256,
                data=data,
                decision=decision,
                next_decision=next_by_id[decision.decision_id],
            )
            for index, decision in enumerate(decisions)
        )
        registry = build_admission_transition_registry(
            anchors,
            admission_identity=admission_identity,
            binding_id=episode_binding_id,
            expected_rows=len(anchors),
        )
        repository = KaggleEpisodeRepository(source_path.parent)
        decision_by_id = {item.decision_id: item for item in decisions}
        for event in anchors:
            decision_id = str(event.transition["decision_id"])
            expected = decision_by_id[decision_id]
            aligned = align_decision(
                event,
                kaggle_repository=repository,
                transition_registry=registry,
            )
            if aligned.failures:
                raise AdmissionHardStop(
                    f"causal alignment failed {decision_id}: "
                    f"{[failure.value for failure in aligned.failures]}"
                )
            proof = aligned.proof
            if not all((
                proof.action_equality_proven,
                proof.actor_seat_proven,
                proof.event_order_proven,
                proof.resulting_transition_proven,
                proof.source_hash_bound,
                proof.deterministic_repair_applied,
            )):
                raise AdmissionHardStop(f"causal proof is incomplete: {decision_id}")
            if _forbidden_keys(aligned.pre_action_event.observation):
                raise AdmissionHardStop(f"visibility projection leaked: {decision_id}")
            if (
                aligned.resulting_event is None
                or _forbidden_keys(aligned.resulting_event.observation)
            ):
                raise AdmissionHardStop(f"result visibility projection leaked: {decision_id}")
            record, legal, chosen = canonicalize_decision(
                aligned,
                implementation_hashes=implementation_hashes,
            )
            if record.validation.outcome not in {
                ValidationOutcome.VALID_ACTOR_AND_CRITIC,
                ValidationOutcome.VALID_CRITIC_ONLY_FORCED,
            }:
                raise AdmissionHardStop(
                    f"canonical validator rejected {decision_id}: "
                    f"{record.validation.outcome.value}"
                )
            if len(legal.stages) != 1 or len(chosen.stages) != 1:
                raise AdmissionHardStop(f"unexpected canonical action shape: {decision_id}")
            stage = legal.stages[0]
            chosen_stage = chosen.stages[0]
            semantic_ok = all((
                stage.minimum_count == expected.min_count,
                stage.maximum_count == expected.max_count,
                len(stage.options) == expected.legal_count,
                chosen_stage.source_selected_values == expected.action,
                chosen.complete,
                chosen.chosen_action_id is not None,
            ))
            if not semantic_ok:
                raise AdmissionHardStop(f"chosen/legal/bounds mismatch: {decision_id}")
            signal = signals.get(decision_id)
            expected_role = "actor" if expected.seat in episode.actor_seats else "opponent"
            compatible = bool(
                signal is not None
                and signal.get("episode_id") == episode.episode_id
                and signal.get("request_step") == expected.request_step
                and signal.get("action_step") == expected.action_step
                and signal.get("seat") == expected.seat
                and signal.get("min_count") == expected.min_count
                and signal.get("max_count") == expected.max_count
                and signal.get("legal_count") == expected.legal_count
                and signal.get("selected_count") == len(expected.action)
                and signal.get("role") == expected_role
            )
            if not compatible:
                compatibility_failures.append(decision_id)
            counts["decision_count"] += 1
            counts[f"role_{expected_role}_count"] += 1
            counts[f"validation_{record.validation.outcome.value}_count"] += 1
            counts["causal_proof_complete_count"] += 1
            counts["semantic_chosen_legal_bounds_count"] += 1
            counts["actor_attribution_exact_count"] += 1
            counts["visibility_clean_count"] += 1
            identity_rows.append({
                "decision_id": decision_id,
                "canonical_decision_id": record.validation.canonical_decision_id,
                "legal_action_set_id": legal.legal_action_set_id,
                "chosen_action_id": chosen.chosen_action_id,
                "resulting_transition_id": record.resulting_transition_id,
            })
        global_row += len(anchors)
        counts["episode_count"] += 1
    if compatibility_failures:
        raise AdmissionHardStop(
            f"postmortem compatibility failed for {len(compatibility_failures)} decisions"
        )
    if len(signals) != counts["decision_count"]:
        raise AdmissionHardStop("postmortem decision inventory is not exact")

    return {
        "counts": dict(sorted(counts.items())),
        "identity_root_sha256": sha256_json(identity_rows),
        "identity_rows": identity_rows,
    }


def run(suite: Path, output: Path) -> dict[str, Any]:
    manifest, binding, records = _verify_suite(suite)
    signal_rows = _read_jsonl(suite / "decisions.jsonl")
    signals = {}
    for row in signal_rows:
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, str) or decision_id in signals:
            raise AdmissionHardStop("postmortem decision identities are invalid or duplicated")
        signals[decision_id] = row
    implementation_hashes, implementation_manifest = _implementation_hashes()
    admission_identity = sha256_json({
        "schema_version": "deepmew.cabt132.canonical_admission_identity.v1",
        "suite_manifest_sha256": _sha256_path(suite / "manifest.json"),
        "replay_cache_sha256": _sha256_path(suite / "replay_cache.json"),
        "binding_sha256": _sha256_path(suite / "binding.json"),
        "contract_file_set_sha256": implementation_manifest[
            "contract_file_set_sha256"
        ],
    })
    first = _admit_once(
        records=records,
        binding=binding,
        signals=signals,
        implementation_hashes=implementation_hashes,
        admission_identity=admission_identity,
    )
    second = _admit_once(
        records=records,
        binding=binding,
        signals=signals,
        implementation_hashes=implementation_hashes,
        admission_identity=admission_identity,
    )
    if first["identity_root_sha256"] != second["identity_root_sha256"]:
        raise AdmissionHardStop("canonical identities are not deterministic")
    counts = first["counts"]
    expected_decisions = manifest.get("summary", {}).get("causal_decision_count")
    if expected_decisions is None:
        summary = _read_json(suite / "summary.json")
        expected_decisions = summary.get("causal_decision_count")
    if counts.get("decision_count") != expected_decisions:
        raise AdmissionHardStop("canonical decision count differs from frozen suite")
    summary = _read_json(suite / "summary.json")
    if (
        counts.get("role_actor_count") != summary.get("actor_decision_count")
        or summary.get("actor_action_agreement_count") != counts.get("role_actor_count")
    ):
        raise AdmissionHardStop("actor attribution or action agreement count differs")
    receipt = {
        "schema_version": "deepmew.cabt132.canonical_admission_receipt.v1",
        "verdict": "RELIABLE",
        "capability": "CAP-001",
        "source_scope": "schema1:cabt1.0.0:module1.32.0",
        "offline_only": True,
        "replay_evidence_rebuilt": False,
        "replay_evidence_redownloaded": False,
        "admission_identity": admission_identity,
        "suite_manifest_sha256": _sha256_path(suite / "manifest.json"),
        "replay_cache_sha256": _sha256_path(suite / "replay_cache.json"),
        "implementation_hashes": implementation_manifest,
        "counts": counts,
        "receipts": {
            "semantic_chosen_legal_bounds": "PASS_EXACT_ALL_DECISIONS",
            "causal_alignment": "PASS_EXACT_ALL_DECISIONS",
            "actor_attribution": "PASS_EXACT_ALL_DECISIONS",
            "visibility": "PASS_ACTOR_VISIBLE_ALL_DECISIONS",
            "deterministic_identity": "PASS_TWO_EXACT_RECONSTRUCTIONS",
            "observation_identity_visibility": "PASS_ACTOR_VISIBLE_CONTENT_ONLY",
            "postmortem_compatibility": "PASS_EXACT_ALL_DECISIONS",
            "source_hash_consistency": "PASS_ALL_CACHED_REPLAYS_AND_SUITE_OUTPUTS",
            "safety": "PASS_ZERO_REJECTIONS_ZERO_LEAKS",
        },
        "identity_root_sha256": first["identity_root_sha256"],
        "repeat_identity_root_sha256": second["identity_root_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "CANONICAL_REPLAY_ADMISSION_RECEIPT.json", receipt)
    _write_json(output / "IMPLEMENTATION_HASHES.json", implementation_manifest)
    compact_index = {
        "schema_version": "deepmew.cabt132.canonical_identity_index.v1",
        "admission_identity": admission_identity,
        "identity_root_sha256": first["identity_root_sha256"],
        "decision_count": len(first["identity_rows"]),
        "rows": first["identity_rows"],
    }
    _write_json(output / "CANONICAL_IDENTITY_INDEX.json", compact_index)

    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run(args.suite.resolve(), args.output.resolve())
    print(json.dumps({
        "verdict": receipt["verdict"],
        "counts": receipt["counts"],
        "identity_root_sha256": receipt["identity_root_sha256"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
