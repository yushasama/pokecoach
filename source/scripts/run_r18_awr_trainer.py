"""R18: Advantage-Weighted Regression (AWR) trainer for the Starmie policy.

The key difference from BC: instead of imitating every teacher action equally
(BC, teacher-bounded) or every WINNING action (filtered BC, still teacher-bounded),
AWR weights each action by its ADVANTAGE = return - baseline, where the baseline is
the per-matchup mean return. So a WIN in a matchup we usually LOSE gets a large
weight (learn this!), a win we always get gets ~1, and losses get down-weighted.
This lets the policy exceed the data-collecting policy (offline RL), unlike BC.

Stability: warm-start from the R4-F prior + a KL-anchor to a FROZEN copy of it
(trust region) so the policy improves without drifting off the good prior
(the DEC-069 collapse guardrail). INVARIANT #2 respected: the reward is the
observed game outcome vs a FIXED opponent snapshot, not a self-play-to-Nash signal.

Example
-------
    python scripts/run_r18_awr_trainer.py \
        --refgames-dir D:/DeepMew_artifacts/datasets/r18/starmie_strongteacher \
        --init-checkpoint <r4f prior.pt> --out-dir .research/r18/starmie_awr \
        --temperature 0.5 --kl-beta 0.05 --max-seconds 180
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_unified_r4_ablation as r4  # noqa: E402
from deepmew.curriculum.awr_ladder_weights import (  # noqa: E402
    compute_awr_weights, load_ladder_weighting_config, outcome_return,
)
from deepmew.curriculum.awr_train_receipt import (  # noqa: E402
    build_strict_trainer_manifest,
    prepare_strict_training_inputs,
)
from deepmew.replay.information_boundary import reject_privileged_rows  # noqa: E402
from run_r9_r4f_retrain import (  # noqa: E402
    R4FIntentAutoregressiveScorer, _copy_checkpoint_weights, encode_refgame_row_r9,
    focal_cross_entropy, save_r4f_checkpoint,
)


CHOSEN_ADVANTAGE_MODES = ("off", "audit", "blend", "replace")
CHOSEN_ADVANTAGE_REPORT_PATH = (
    ROOT / ".research" / "chosen_action_advantage_stage2" / "STAGE2A_TRAINER_CONSUMPTION_AUDIT.md"
)
SAFE_PERSPECTIVE_STATUSES = {"same_actor", "terminal"}
UNSAFE_ADVANTAGE_WARNINGS = {
    "chosen_action_indices_mismatch",
    "chosen_action_semantics_mismatch",
    "chosen_index_not_int",
    "chosen_index_out_of_range",
    "chosen_indices_without_legal_options",
    "legal_options_not_preserved_exactly",
    "raw_observation_not_actor_perspective",
    "terminal_post_result_final_result_mismatch",
}


@dataclass(frozen=True)
class ChosenAdvantageConfig:
    mode: str = "off"
    sidecar_path: Path | None = None
    blend_lambda: float = 0.0
    beta: float = 2.0
    weight_min: float = 0.1
    weight_max: float = 3.0
    normalize_mean: bool = True
    strict_join: bool = True
    allow_privileged: bool = False


class ChosenAdvantageJoinError(ValueError):
    """Raised when strict sidecar identity or action checks fail."""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def _decision_key(row: dict[str, Any]) -> str | None:
    if row.get("decision_id"):
        return str(row["decision_id"])
    if row.get("frame_id"):
        return str(row["frame_id"])
    game_id = row.get("game_id")
    decision_index = row.get("decision_index", row.get("step"))
    if game_id is None or isinstance(decision_index, bool):
        return None
    try:
        index = int(decision_index)
    except (TypeError, ValueError):
        return None

    return f"{game_id}:d{index:04d}"


def _chosen_indices(row: dict[str, Any]) -> list[Any]:
    chosen_action = _as_dict(row.get("chosen_action"))
    indices = chosen_action.get("indices")
    if indices is None:
        indices = row.get("chosen_indices")

    return _as_list(indices)


def _legal_options(row: dict[str, Any]) -> list[Any] | None:
    if isinstance(row.get("legal_set"), list):
        return row["legal_set"]
    if isinstance(row.get("legal_options"), list):
        return row["legal_options"]

    return None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def chosen_advantage_weight(advantage: float, *, beta: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, 1.0 + beta * float(advantage)))


def normalize_advantage_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    mean_weight = sum(weights) / len(weights)
    if mean_weight <= 0.0:
        raise ValueError("eligible chosen-action advantage weights must have positive mean")

    return [weight / mean_weight for weight in weights]


def load_chosen_advantage_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("chosen_action_advantage*.jsonl"))
        paths.extend(sorted(source.rglob("chosen_action_advantage*.jsonl.gz")))
    else:
        raise FileNotFoundError(f"chosen-action advantage sidecar not found: {source}")
    if not paths:
        raise FileNotFoundError(f"no chosen-action advantage JSONL found under: {source}")

    rows: list[dict[str, Any]] = []
    for sidecar_path in paths:
        opener = gzip.open if sidecar_path.suffix == ".gz" else open
        with opener(sidecar_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"sidecar row must be an object: {sidecar_path}:{line_number}")
                rows.append(row)

    return rows


def _duplicate_keys(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_decision_key(row) for row in rows)

    return {str(key): count for key, count in counts.items() if key is not None and count > 1}


def _eligibility_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("value_scale") != "win_probability":
        reasons.append("incompatible_value_scale")
    terminal_scale = row.get("terminal_override_scale")
    if terminal_scale not in {None, "", "win_probability"}:
        reasons.append("incompatible_terminal_override_scale")
    if _finite_float(row.get("advantage")) is None:
        reasons.append("missing_or_nonfinite_advantage")
    if _finite_float(row.get("v_before")) is None:
        reasons.append("missing_or_nonfinite_v_before")
    has_v_after = _finite_float(row.get("v_after")) is not None
    has_terminal_override = row.get("advantage_source") == "terminal_override"
    if not has_v_after and not has_terminal_override:
        reasons.append("missing_v_after_or_terminal_override")
    if row.get("perspective_status") not in SAFE_PERSPECTIVE_STATUSES:
        reasons.append("unsafe_or_unavailable_perspective")
    action_integrity = _as_dict(row.get("action_integrity"))
    if action_integrity.get("chosen_indices_valid") is not True:
        reasons.append("chosen_indices_not_validated")
    if action_integrity.get("chosen_action_indices_match") is False:
        reasons.append("sidecar_chosen_action_mismatch")
    if action_integrity.get("chosen_action_semantics_match") is False:
        reasons.append("sidecar_chosen_semantics_mismatch")
    if action_integrity.get("legal_options_preserved_exact") is False:
        reasons.append("sidecar_legal_options_not_preserved")
    warnings = {str(value) for value in _as_list(row.get("warnings"))}
    if warnings & UNSAFE_ADVANTAGE_WARNINGS:
        reasons.append("unsafe_integrity_warning")

    return reasons


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None}

    return {
        "count": len(finite),
        "min": finite[0],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
        "p50": float(statistics.median(finite)),
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)

    return numerator / denominator if denominator > 0.0 else None


def _audit_example(entry: dict[str, Any]) -> dict[str, Any]:
    trainer = entry["trainer_row"]
    sidecar = entry.get("sidecar_row") or {}
    outcome = trainer.get("outcome")

    return {
        "decision_id": entry["key"],
        "game_id": sidecar.get("game_id", trainer.get("game_id")),
        "episode_id": sidecar.get("episode_id", trainer.get("episode_id")),
        "frame_id": sidecar.get("frame_id", trainer.get("frame_id")),
        "actor": sidecar.get("actor", trainer.get("seat")),
        "turn": sidecar.get("turn", trainer.get("turn")),
        "phase": sidecar.get("phase", trainer.get("phase")),
        "chosen_indices": sidecar.get("chosen_indices", _chosen_indices(trainer)),
        "legal_option_count": _as_dict(sidecar.get("action_integrity")).get("legal_option_count"),
        "outcome": outcome,
        "v_before": sidecar.get("v_before"),
        "v_after": sidecar.get("v_after"),
        "advantage": sidecar.get("advantage"),
        "old_weight": entry["old_weight"],
        "advantage_weight": entry.get("advantage_weight"),
        "final_weight": entry["final_weight"],
        "perspective_status": sidecar.get("perspective_status"),
        "warnings": sidecar.get("warnings"),
        "eligibility_reasons": entry["eligibility_reasons"],
    }


def apply_chosen_advantage_weights(
    trainer_rows: list[dict[str, Any]],
    old_weights: list[float],
    sidecar_rows: list[dict[str, Any]],
    config: ChosenAdvantageConfig,
) -> tuple[list[float], dict[str, Any]]:
    if len(trainer_rows) != len(old_weights):
        raise ValueError("trainer rows and old AWR weights must have equal length")
    if config.mode not in CHOSEN_ADVANTAGE_MODES:
        raise ValueError(f"unsupported chosen-action advantage mode: {config.mode}")

    duplicate_trainer = _duplicate_keys(trainer_rows)
    duplicate_sidecar = _duplicate_keys(sidecar_rows)
    trainer_keys = [_decision_key(row) for row in trainer_rows]
    sidecar_by_key = {_decision_key(row): row for row in sidecar_rows if _decision_key(row) is not None}
    trainer_key_set = {key for key in trainer_keys if key is not None}
    unmatched_sidecar = sorted(str(key) for key in sidecar_by_key if key not in trainer_key_set)
    entries: list[dict[str, Any]] = []
    join_mismatches: list[dict[str, Any]] = []
    for trainer_row, old_weight, key in zip(trainer_rows, old_weights, trainer_keys):
        sidecar_row = sidecar_by_key.get(key)
        reasons: list[str] = []
        if key is None:
            reasons.append("missing_stable_trainer_key")
        if sidecar_row is None:
            reasons.append("missing_sidecar_row")
        else:
            trainer_chosen = _chosen_indices(trainer_row)
            sidecar_chosen = _chosen_indices(sidecar_row)
            if trainer_chosen != sidecar_chosen:
                reasons.append("join_chosen_action_mismatch")
                join_mismatches.append({"decision_id": key, "type": "chosen_action"})
            trainer_legal = _legal_options(trainer_row)
            sidecar_legal = _legal_options(sidecar_row)
            if trainer_legal is not None and sidecar_legal is not None and _canonical(trainer_legal) != _canonical(sidecar_legal):
                reasons.append("join_legal_options_mismatch")
                join_mismatches.append({"decision_id": key, "type": "legal_options"})
            reasons.extend(_eligibility_reasons(sidecar_row))
        entries.append({
            "key": key,
            "trainer_row": trainer_row,
            "sidecar_row": sidecar_row,
            "old_weight": float(old_weight),
            "eligibility_reasons": sorted(set(reasons)),
            "advantage_weight": None,
            "final_weight": float(old_weight),
        })

    unmatched_trainer = [str(entry["key"]) for entry in entries if entry["sidecar_row"] is None]
    strict_errors: list[str] = []
    if duplicate_trainer:
        strict_errors.append(f"duplicate trainer keys: {len(duplicate_trainer)}")
    if duplicate_sidecar:
        strict_errors.append(f"duplicate sidecar keys: {len(duplicate_sidecar)}")
    if unmatched_trainer:
        strict_errors.append(f"unmatched trainer rows: {len(unmatched_trainer)}")
    if unmatched_sidecar:
        strict_errors.append(f"unmatched sidecar rows: {len(unmatched_sidecar)}")
    if join_mismatches:
        strict_errors.append(f"action/legal join mismatches: {len(join_mismatches)}")
    if config.strict_join and strict_errors:
        raise ChosenAdvantageJoinError("; ".join(strict_errors))

    eligible = [entry for entry in entries if not entry["eligibility_reasons"]]
    raw_advantage_weights = [
        chosen_advantage_weight(
            float(entry["sidecar_row"]["advantage"]),
            beta=config.beta,
            minimum=config.weight_min,
            maximum=config.weight_max,
        )
        for entry in eligible
    ]
    normalized_weights = normalize_advantage_weights(raw_advantage_weights) if config.normalize_mean else raw_advantage_weights
    for entry, advantage_weight in zip(eligible, normalized_weights):
        entry["advantage_weight"] = float(advantage_weight)

    for entry in entries:
        advantage_weight = entry["advantage_weight"]
        old_weight = entry["old_weight"]
        if config.mode in {"off", "audit"}:
            final_weight = old_weight
        elif config.mode == "blend" and advantage_weight is not None:
            final_weight = ((1.0 - config.blend_lambda) * old_weight) + (config.blend_lambda * advantage_weight)
        elif config.mode == "replace" and advantage_weight is not None:
            final_weight = advantage_weight
        elif config.mode == "replace":
            final_weight = 1.0
        else:
            final_weight = old_weight
        entry["final_weight"] = float(final_weight)

    eligible_old = [entry["old_weight"] for entry in eligible]
    eligible_adv = [float(entry["advantage_weight"]) for entry in eligible]
    advantages = [float(entry["sidecar_row"]["advantage"]) for entry in eligible]
    reason_counts = Counter(reason for entry in entries for reason in entry["eligibility_reasons"])
    value_scale_counts = Counter(str(row.get("value_scale")) for row in sidecar_rows)
    perspective_counts = Counter(str(row.get("perspective_status")) for row in sidecar_rows)
    information_mode_counts = Counter(str(row.get("information_mode")) for row in sidecar_rows)
    highest = sorted(eligible, key=lambda entry: float(entry["advantage_weight"]), reverse=True)[:5]
    lowest = sorted(eligible, key=lambda entry: float(entry["advantage_weight"]))[:5]
    old_high_adv_low = sorted(eligible, key=lambda entry: entry["old_weight"] - float(entry["advantage_weight"]), reverse=True)[:5]
    old_low_adv_high = sorted(eligible, key=lambda entry: float(entry["advantage_weight"]) - entry["old_weight"], reverse=True)[:5]
    audit = {
        "mode": config.mode,
        "sidecar_path": str(config.sidecar_path) if config.sidecar_path else None,
        "join_key": "decision_id; fallback frame_id; fallback game_id:dNNNN from decision_index",
        "trainer_data_rows": len(trainer_rows),
        "sidecar_rows": len(sidecar_rows),
        "duplicate_trainer_keys": duplicate_trainer,
        "duplicate_sidecar_keys": duplicate_sidecar,
        "matched_rows": sum(entry["sidecar_row"] is not None for entry in entries),
        "unmatched_trainer_rows": len(unmatched_trainer),
        "unmatched_sidecar_rows": len(unmatched_sidecar),
        "unmatched_trainer_examples": unmatched_trainer[:5],
        "unmatched_sidecar_examples": unmatched_sidecar[:5],
        "join_mismatch_count": len(join_mismatches),
        "join_mismatch_examples": join_mismatches[:5],
        "eligible_advantage_rows": len(eligible),
        "ineligible_advantage_rows": len(entries) - len(eligible),
        "fallback_row_count": len(entries) - len(eligible),
        "ineligible_reasons": dict(sorted(reason_counts.items())),
        "value_scale_counts": dict(sorted(value_scale_counts.items())),
        "perspective_status_counts": dict(sorted(perspective_counts.items())),
        "information_mode_counts": dict(sorted(information_mode_counts.items())),
        "advantage_distribution": _distribution(advantages),
        "old_awr_weight_distribution": _distribution([entry["old_weight"] for entry in entries]),
        "advantage_weight_distribution": _distribution(eligible_adv),
        "final_weight_distribution": _distribution([entry["final_weight"] for entry in entries]),
        "old_advantage_weight_correlation": _correlation(eligible_old, eligible_adv),
        "positive_advantage_rows": sum(value > 0.0 for value in advantages),
        "zero_advantage_rows": sum(value == 0.0 for value in advantages),
        "negative_advantage_rows": sum(value < 0.0 for value in advantages),
        "highest_advantage_weight_examples": [_audit_example(entry) for entry in highest],
        "lowest_advantage_weight_examples": [_audit_example(entry) for entry in lowest],
        "old_high_advantage_low_examples": [_audit_example(entry) for entry in old_high_adv_low],
        "old_low_advantage_high_examples": [_audit_example(entry) for entry in old_low_adv_high],
        "beta": config.beta,
        "weight_min": config.weight_min,
        "weight_max": config.weight_max,
        "blend_lambda": config.blend_lambda,
        "normalize_mean": config.normalize_mean,
        "strict_join": config.strict_join,
        "replace_ineligible_fallback": "neutral_weight_1.0",
        "training_status": "not_started",
        "output_checkpoint_path": None,
    }

    return [entry["final_weight"] for entry in entries], audit


def render_chosen_advantage_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage 2A Trainer Consumption Audit",
        "",
        f"training_status={audit['training_status']}",
        f"output_checkpoint_path={audit.get('output_checkpoint_path')}",
        f"mode={audit['mode']} sidecar_path={audit['sidecar_path']}",
        f"join_key={audit['join_key']}",
        f"trainer_data_rows={audit['trainer_data_rows']} sidecar_rows={audit['sidecar_rows']}",
        f"matched_rows={audit['matched_rows']} unmatched_trainer_rows={audit['unmatched_trainer_rows']} "
        f"unmatched_sidecar_rows={audit['unmatched_sidecar_rows']}",
        f"eligible_advantage_rows={audit['eligible_advantage_rows']} "
        f"ineligible_advantage_rows={audit['ineligible_advantage_rows']} fallback_row_count={audit['fallback_row_count']}",
        "",
        "## Join Integrity",
        "",
        f"duplicate_trainer_keys={json.dumps(audit['duplicate_trainer_keys'], sort_keys=True)}",
        f"duplicate_sidecar_keys={json.dumps(audit['duplicate_sidecar_keys'], sort_keys=True)}",
        f"join_mismatch_count={audit['join_mismatch_count']}",
        f"join_mismatch_examples={json.dumps(audit['join_mismatch_examples'], sort_keys=True)}",
        f"unmatched_trainer_examples={json.dumps(audit['unmatched_trainer_examples'], sort_keys=True)}",
        f"unmatched_sidecar_examples={json.dumps(audit['unmatched_sidecar_examples'], sort_keys=True)}",
        "",
        "## Eligibility",
        "",
        f"ineligible_reasons={json.dumps(audit['ineligible_reasons'], sort_keys=True)}",
        f"value_scale_counts={json.dumps(audit['value_scale_counts'], sort_keys=True)}",
        f"perspective_status_counts={json.dumps(audit['perspective_status_counts'], sort_keys=True)}",
        f"information_mode_counts={json.dumps(audit['information_mode_counts'], sort_keys=True)}",
        f"replace_ineligible_fallback={audit['replace_ineligible_fallback']}",
        "",
        "## Weight Distributions",
        "",
        f"advantage={json.dumps(audit['advantage_distribution'], sort_keys=True)}",
        f"old_awr_weight={json.dumps(audit['old_awr_weight_distribution'], sort_keys=True)}",
        f"advantage_weight={json.dumps(audit['advantage_weight_distribution'], sort_keys=True)}",
        f"final_weight={json.dumps(audit['final_weight_distribution'], sort_keys=True)}",
        f"old_advantage_weight_correlation={audit['old_advantage_weight_correlation']}",
        f"positive={audit['positive_advantage_rows']} zero={audit['zero_advantage_rows']} "
        f"negative={audit['negative_advantage_rows']}",
        "",
        "## Configuration",
        "",
        f"beta={audit['beta']} weight_min={audit['weight_min']} weight_max={audit['weight_max']}",
        f"blend_lambda={audit['blend_lambda']} normalize_mean={audit['normalize_mean']} strict_join={audit['strict_join']}",
        "",
        "## Examples",
        "",
        "highest_advantage_weight=" + json.dumps(audit["highest_advantage_weight_examples"], sort_keys=True),
        "lowest_advantage_weight=" + json.dumps(audit["lowest_advantage_weight_examples"], sort_keys=True),
        "old_high_advantage_low=" + json.dumps(audit["old_high_advantage_low_examples"], sort_keys=True),
        "old_low_advantage_high=" + json.dumps(audit["old_low_advantage_high_examples"], sort_keys=True),
    ]

    return "\n".join(lines)


def write_chosen_advantage_audit(audit: dict[str, Any], path: Path = CHOSEN_ADVANTAGE_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_chosen_advantage_audit(audit), encoding="utf-8")


def _return(outcome: dict[str, Any]) -> float:
    return outcome_return(outcome)


def _iter_rows(refgames_dir: Path):
    for shard in sorted((refgames_dir / "shards").glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as h:
            for line in h:
                if line.strip():
                    yield json.loads(line)


def build_awr_samples(refgames_dir: Path, *, temperature: float, w_min: float, w_max: float,
                      limit: int | None, expected_loss_penalty: float = 1.0,
                      ladder_config: dict[str, Any] | None = None,
                      chosen_advantage_config: ChosenAdvantageConfig | None = None) -> tuple[list[Any], dict[str, Any]]:
    rows = list(_iter_rows(refgames_dir))
    if limit:
        rows = rows[:limit]
    row_weights, weight_stats = compute_awr_weights(
        rows, ladder_config or {}, temperature=temperature, w_min=w_min, w_max=w_max,
        expected_loss_penalty=expected_loss_penalty,
    )
    chosen_config = chosen_advantage_config or ChosenAdvantageConfig()
    final_weights = [float(row_weight.w_total) for row_weight in row_weights]
    chosen_audit: dict[str, Any] | None = None
    if chosen_config.mode != "off":
        if chosen_config.sidecar_path is None:
            raise ValueError("--chosen-advantage-sidecar is required when chosen-advantage mode is not off")
        sidecar_rows = load_chosen_advantage_rows(chosen_config.sidecar_path)
        reject_privileged_rows(sidecar_rows, allow_privileged=chosen_config.allow_privileged)
        final_weights, chosen_audit = apply_chosen_advantage_weights(
            rows,
            final_weights,
            sidecar_rows,
            chosen_config,
        )

    samples: list[Any] = []
    adv_sum = w_sum = 0.0
    for r, row_weight, final_weight in zip(rows, row_weights, final_weights):
        encoded = encode_refgame_row_r9(r)
        if encoded is None:
            continue
        for s in r4.make_step_samples(encoded, weighted=False):
            samples.append(dataclasses.replace(s, weight=float(final_weight)))
        adv_sum += row_weight.advantage
        w_sum += final_weight
    stats = {
        "rows": len(rows), "samples": len(samples),
        "matchups": int(weight_stats.get("baseline", {}).get("primary_cells", {}).get("count", 0)),
        "global_return": round(float(weight_stats.get("baseline", {}).get("global_return", 0.0)), 4),
        "mean_adv": round(adv_sum / max(1, len(rows)), 4),
        "mean_weight": round(w_sum / max(1, len(rows)), 4),
        "ladder_weighting": weight_stats,
    }
    if chosen_audit is not None:
        stats["chosen_advantage"] = chosen_audit

    return samples, stats


def train_awr(model: R4FIntentAutoregressiveScorer, ref_model: R4FIntentAutoregressiveScorer,
              samples: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    started = time.perf_counter()
    rng = torch.Generator().manual_seed(int(args.seed))
    n = len(samples)
    batches = kl_total = ce_total = 0
    ce_val = kl_val = 0.0
    while time.perf_counter() - started < float(args.max_seconds):
        order = torch.randperm(n, generator=rng).tolist()
        for start in range(0, n, int(args.batch_size)):
            chunk = [samples[i] for i in order[start:start + int(args.batch_size)]]
            features, option_meta, target, mask, weights = r4.batch_steps(chunk)
            scores = model(features, option_meta).masked_fill(~mask, -1e9)
            ce_by_row = focal_cross_entropy(scores, target, gamma=float(args.focal_gamma))
            ce = (ce_by_row * weights).sum() / weights.sum().clamp_min(1e-6)
            # KL(pi || pi_ref) trust-region anchor over the legal-option distribution
            with torch.no_grad():
                ref_scores = ref_model(features, option_meta).masked_fill(~mask, -1e9)
                ref_logp = torch.log_softmax(ref_scores, dim=1)
            logp = torch.log_softmax(scores, dim=1)
            kl_by_row = (logp.exp() * (logp - ref_logp)).sum(dim=1)
            finite = torch.isfinite(kl_by_row)
            if bool(getattr(args, "weight_kl", False)):
                finite_weights = weights[finite]
                kl = (kl_by_row[finite] * finite_weights).sum() / finite_weights.sum().clamp_min(1e-6)
            else:
                kl = kl_by_row[finite].mean()
            loss = ce + float(args.kl_beta) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches += 1
            ce_val, kl_val = float(ce.item()), float(kl.item())
            ce_total += ce_val
            kl_total += kl_val
            if args.progress_every and batches % int(args.progress_every) == 0:
                print(f"awr batches={batches} ce={ce_val:.4f} kl={kl_val:.4f} "
                      f"elapsed={time.perf_counter() - started:.1f}", file=sys.stderr, flush=True)
            if time.perf_counter() - started >= float(args.max_seconds):
                break
    return {"batches": batches, "final_ce": round(ce_val, 4), "final_kl": round(kl_val, 4),
            "mean_ce": round(ce_total / max(1, batches), 4), "mean_kl": round(kl_total / max(1, batches), 4),
            "elapsed_seconds": round(time.perf_counter() - started, 1)}


def _strict_generation_binding(args: argparse.Namespace) -> dict[str, str] | None:
    fields = {
        "generation_id": str(args.generation_id),
        "snapshot_id": str(args.snapshot_id),
        "champion_member_id": str(args.champion_member_id),
        "candidate_deck_sha256": str(args.candidate_deck_sha256),
        "feature_policy_id": str(args.feature_policy_id),
        "data_manifest": str(args.data_manifest),
    }
    supplied = {name for name, value in fields.items() if value}
    if not supplied:
        return None
    if supplied != set(fields):
        missing = sorted(set(fields) - supplied)
        raise ValueError(
            "strict generation receipt inputs are all-or-none; missing " + ", ".join(missing)
        )

    return {
        key: fields[key]
        for key in (
            "generation_id",
            "snapshot_id",
            "champion_member_id",
            "candidate_deck_sha256",
            "feature_policy_id",
        )
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refgames-dir", required=True)
    ap.add_argument("--init-checkpoint", required=True, help="R4-F prior to warm-start AND anchor to")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--expected-loss-penalty", type=float, default=1.0,
                    help="extra advantage penalty for LOSING a favorable (baseline>0) matchup; 0=off")
    ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--w-min", type=float, default=0.1)
    ap.add_argument("--w-max", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=181818)
    ap.add_argument("--ladder-weighting-config", default="",
                    help="YAML/JSON row-weight config. Empty preserves historical AWR behavior.")
    ap.add_argument("--chosen-advantage-sidecar", default="",
                    help="Stage 1.5 chosen-action advantage JSONL file or containing directory.")
    ap.add_argument("--chosen-advantage-mode", choices=CHOSEN_ADVANTAGE_MODES, default="off")
    ap.add_argument("--chosen-advantage-blend-lambda", type=float, default=0.0)
    ap.add_argument("--chosen-advantage-beta", type=float, default=2.0)
    ap.add_argument("--chosen-advantage-weight-min", type=float, default=0.1)
    ap.add_argument("--chosen-advantage-weight-max", type=float, default=3.0)
    ap.add_argument("--chosen-advantage-normalize-mean", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chosen-advantage-strict-join", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-privileged-chosen-advantage", action="store_true",
                    help="explicit oracle-research override; public trainer rejects privileged sidecars by default")
    ap.add_argument("--generation-id", default="",
                    help="strict league receipt binding; requires all other strict generation inputs")
    ap.add_argument("--snapshot-id", default="")
    ap.add_argument("--champion-member-id", default="")
    ap.add_argument("--candidate-deck-sha256", default="")
    ap.add_argument("--feature-policy-id", default="")
    ap.add_argument("--data-manifest", default="",
                    help="strict deepmew.r19.league_datagen.v2 manifest")
    args = ap.parse_args(argv)

    ladder_config = load_ladder_weighting_config(args.ladder_weighting_config)
    args.weight_kl = bool((ladder_config.get("kl") or {}).get("weight_kl", False))
    if not 0.0 <= float(args.chosen_advantage_blend_lambda) <= 1.0:
        raise ValueError("--chosen-advantage-blend-lambda must be in [0, 1]")
    if float(args.chosen_advantage_weight_min) <= 0.0:
        raise ValueError("--chosen-advantage-weight-min must be positive")
    if float(args.chosen_advantage_weight_max) < float(args.chosen_advantage_weight_min):
        raise ValueError("--chosen-advantage-weight-max must be >= --chosen-advantage-weight-min")
    chosen_config = ChosenAdvantageConfig(
        mode=str(args.chosen_advantage_mode),
        sidecar_path=Path(args.chosen_advantage_sidecar) if args.chosen_advantage_sidecar else None,
        blend_lambda=float(args.chosen_advantage_blend_lambda),
        beta=float(args.chosen_advantage_beta),
        weight_min=float(args.chosen_advantage_weight_min),
        weight_max=float(args.chosen_advantage_weight_max),
        normalize_mean=bool(args.chosen_advantage_normalize_mean),
        strict_join=bool(args.chosen_advantage_strict_join),
        allow_privileged=bool(args.allow_privileged_chosen_advantage),
    )
    strict_binding = _strict_generation_binding(args)
    strict_inputs = None
    if strict_binding is not None:
        if chosen_config.mode != "off":
            raise ValueError("strict generation training requires --chosen-advantage-mode off")
        if chosen_config.allow_privileged:
            raise ValueError("strict generation training forbids privileged chosen advantage")
        strict_inputs = prepare_strict_training_inputs(
            generation_binding=strict_binding,
            data_manifest_path=args.data_manifest,
            refgames_dir=args.refgames_dir,
            init_checkpoint_path=args.init_checkpoint,
        )
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("[awr] building advantage-weighted samples...", flush=True)
    samples, dstats = build_awr_samples(Path(args.refgames_dir), temperature=float(args.temperature),
                                        w_min=float(args.w_min), w_max=float(args.w_max),
                                        limit=int(args.limit) or None,
                                        expected_loss_penalty=float(args.expected_loss_penalty),
                                        ladder_config=ladder_config,
                                        chosen_advantage_config=chosen_config)
    print(f"[awr] {dstats}", flush=True)
    chosen_audit = dstats.get("chosen_advantage")
    if isinstance(chosen_audit, dict):
        chosen_audit["training_status"] = "audit_only_exited" if chosen_config.mode == "audit" else "planned"
        write_chosen_advantage_audit(chosen_audit)
        print(
            "[awr] chosen-advantage "
            f"mode={chosen_config.mode} sidecar={chosen_config.sidecar_path} beta={chosen_config.beta} "
            f"clip=[{chosen_config.weight_min}, {chosen_config.weight_max}] "
            f"blend_lambda={chosen_config.blend_lambda} normalize_mean={chosen_config.normalize_mean} "
            f"matched={chosen_audit['matched_rows']} eligible={chosen_audit['eligible_advantage_rows']} "
            f"fallback={chosen_audit['fallback_row_count']} "
            f"effective_mean={chosen_audit['final_weight_distribution']['mean']} "
            f"effective_max={chosen_audit['final_weight_distribution']['max']}",
            flush=True,
        )
    if chosen_config.mode == "audit":
        print(f"[awr] audit-only report: {CHOSEN_ADVANTAGE_REPORT_PATH}", flush=True)
        return 0
    if not samples:
        raise RuntimeError("no AWR samples built")
    args.input_dim = len(samples[0].features[0])
    torch.manual_seed(int(args.seed))
    model = R4FIntentAutoregressiveScorer(args.input_dim, hidden_dim=args.hidden_dim,
                                          depth=args.depth, embed_dim=args.embed_dim, dropout=args.dropout)
    init = _copy_checkpoint_weights(model, Path(args.init_checkpoint))
    ref_model = R4FIntentAutoregressiveScorer(args.input_dim, hidden_dim=args.hidden_dim,
                                              depth=args.depth, embed_dim=args.embed_dim, dropout=args.dropout)
    _copy_checkpoint_weights(ref_model, Path(args.init_checkpoint))
    ref_model.load_state_dict(model.state_dict())  # anchor = exact warm-start
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    train = train_awr(model, ref_model, samples, args)
    print(f"[awr] train: {train}", flush=True)
    args.weight_decay = 1e-4
    model_path = out_dir / "selected_ar_bc_r4f_candidate.pt"
    save_r4f_checkpoint(model_path, model, args, {"awr": train, "data": dstats}, init)
    if isinstance(chosen_audit, dict):
        chosen_audit["training_status"] = "training_proceeded"
        chosen_audit["output_checkpoint_path"] = str(model_path)
        write_chosen_advantage_audit(chosen_audit)
    legacy_config = {"temperature": args.temperature, "kl_beta": args.kl_beta,
                     "w_min": args.w_min, "w_max": args.w_max, "lr": args.lr,
                     "ladder_weighting_config": args.ladder_weighting_config,
                     "weight_kl": args.weight_kl,
                     "chosen_advantage_mode": chosen_config.mode,
                     "chosen_advantage_sidecar": str(chosen_config.sidecar_path) if chosen_config.sidecar_path else "",
                     "chosen_advantage_beta": chosen_config.beta,
                     "chosen_advantage_weight_min": chosen_config.weight_min,
                     "chosen_advantage_weight_max": chosen_config.weight_max,
                     "chosen_advantage_blend_lambda": chosen_config.blend_lambda,
                     "chosen_advantage_normalize_mean": chosen_config.normalize_mean,
                     "chosen_advantage_strict_join": chosen_config.strict_join,
                     "allow_privileged_chosen_advantage": chosen_config.allow_privileged}
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if strict_inputs is None:
        trainer_manifest = {
            "schema_version": "deepmew.r18.awr.v0",
            "created_at_utc": created_at_utc,
            "data": dstats, "train": train,
            "config": legacy_config,
            "out": str(model_path),
        }
    else:
        full_config = dict(vars(args))
        trainer_manifest = build_strict_trainer_manifest(
            prepared_inputs=strict_inputs,
            output_checkpoint_path=model_path,
            trainer_code_path=Path(__file__),
            full_config=full_config,
            training={"samples": int(dstats["samples"]), "batches": int(train["batches"])},
            metrics={
                name: train[name]
                for name in ("final_ce", "final_kl", "mean_ce", "mean_kl", "elapsed_seconds")
            },
            data_stats=dstats,
            train_stats=train,
            created_at_utc=created_at_utc,
        )
    (out_dir / "awr_manifest.json").write_text(
        json.dumps(trainer_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[awr] saved {model_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
