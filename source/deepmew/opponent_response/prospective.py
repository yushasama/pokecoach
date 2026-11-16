"""Chronological R0/R1 response ablation on actor-visible prospective games.

The fitted count tables are offline diagnostics. They consume the responding
actor's legitimate private observation and therefore cannot be called from a
strict-public branch without a separately validated hidden-state sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from deepmew.artifacts.writers import write_json, write_text
from deepmew.opponent_response.baseline import (
    Candidate,
    MetricAccumulator,
    ResponseCountModel,
    ResponseExample,
    response_state_key,
)
from deepmew.replay.prospective_trace import (
    assert_actor_visible_trace_row_safe,
    iter_trace_rows,
)


SCHEMA_VERSION = "deepmew.prospective_response_ablation.v1"
MANIFEST_VERSION = "deepmew.prospective_response_manifest.v1"
TRAIN_TRANCHE = "T0"
EVALUATION_TRANCHE = "T1"
CANARY_TRANCHE = "S0"
MIN_GROUP_SAMPLES = 50
MIN_MATCHUP_SAMPLES = 100
MIN_R0_NLL_GAIN = 0.01
MIN_R1_NLL_GAIN = 0.005
MAX_AGGREGATE_BRIER_REGRESSION = 0.002
MAX_GROUP_NLL_REGRESSION = 0.02
MAX_SEED_GAIN_STD = 0.02
WRONG_LOCK_CONFIDENCE = 0.70
SUBSAMPLE_FRACTION = 0.80
STABILITY_SEEDS = (11, 29, 47, 71)


@dataclass(frozen=True)
class ProspectiveResponseExample(ResponseExample):
    matchup_cell: str
    policy_id: str
    deck_hash: str
    response_block_id: str
    root_action_family: str


@dataclass(frozen=True)
class LossRecord:
    action_family: str
    policy_id: str
    matchup_cell: str
    deck_hash: str
    losses: dict[str, float]


@dataclass
class PredictorAccumulator:
    metrics: MetricAccumulator
    elapsed_seconds: float = 0.0
    wrong_locks: int = 0

    def add(
        self,
        probabilities: Mapping[Candidate, float],
        example: ProspectiveResponseExample,
        *,
        elapsed_seconds: float,
    ) -> float:
        loss, correct = self.metrics.add(probabilities, example.chosen_candidate)
        self.elapsed_seconds += elapsed_seconds
        confidence = max(probabilities.values()) if probabilities else 0.0
        self.wrong_locks += int(confidence >= WRONG_LOCK_CONFIDENCE and not correct)

        return loss

    def result(self) -> dict[str, Any]:
        result = self.metrics.result(self.elapsed_seconds)
        result["wrong_lock_rate"] = round(self.wrong_locks / self.metrics.samples, 6)
        result["wrong_lock_confidence"] = WRONG_LOCK_CONFIDENCE

        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _tree_receipt(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    files = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for row in files:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    return {"files": files, "file_count": len(files), "tree_sha256": digest.hexdigest()}


def _chosen_action_family(row: dict[str, Any]) -> str:
    chosen = row.get("chosen_action")
    semantics = chosen.get("semantics") if isinstance(chosen, dict) else None
    families = sorted({
        str(option.get("option_type") or "UNKNOWN")
        for option in semantics or []
        if isinstance(option, dict)
    })

    return "+".join(families) if families else "UNKNOWN"


def _build_response_example(
    row: dict[str, Any],
    *,
    matchup_cell: str,
    response_block_id: str,
    root_action_family: str,
    stats: Counter[str],
) -> ProspectiveResponseExample | None:
    raw = row.get("raw_observation")
    legal_set = row.get("legal_set")
    chosen = row.get("chosen_action")
    if not isinstance(raw, dict) or not isinstance(legal_set, list) or not isinstance(chosen, dict):
        stats["skipped_missing_contract"] += 1

        return None
    indices = chosen.get("indices")
    if not isinstance(indices, list) or len(indices) != 1:
        stats["skipped_non_single_choice"] += 1

        return None
    chosen_index = indices[0]
    if isinstance(chosen_index, bool) or not isinstance(chosen_index, int):
        stats["skipped_invalid_chosen_index"] += 1

        return None
    if chosen_index < 0 or chosen_index >= len(legal_set):
        stats["skipped_invalid_chosen_index"] += 1

        return None

    candidates_by_key: dict[str, Candidate] = {}
    for option in legal_set:
        if not isinstance(option, dict):
            stats["skipped_invalid_legal_option"] += 1

            return None
        semantic_key = str(option.get("semantic_key") or "")
        if semantic_key:
            candidates_by_key.setdefault(
                semantic_key,
                (semantic_key, str(option.get("option_type") or "UNKNOWN")),
            )
    chosen_option = legal_set[chosen_index]
    chosen_key = str(chosen_option.get("semantic_key") or "")
    chosen_candidate = candidates_by_key.get(chosen_key)
    candidates = tuple(sorted(candidates_by_key.values()))
    if chosen_candidate is None:
        stats["skipped_missing_chosen_semantic"] += 1

        return None
    if len(candidates) < 2:
        stats["skipped_single_semantic_choice"] += 1

        return None

    first = legal_set[0]
    select_type = str(first.get("select_type") or "UNKNOWN")
    select_context = str(first.get("select_context") or "UNKNOWN")
    policy_id = str(row["policy_id"])
    deck_hash = str(row["deck_hash"])
    state_key = response_state_key(
        raw,
        candidates,
        select_type=select_type,
        select_context=select_context,
    ) + (f"root:{root_action_family}",)

    return ProspectiveResponseExample(
        episode_id=str(row["game_id"]),
        response_split=str(row["tranche_id"]),
        deck_split=deck_hash,
        archetype=policy_id,
        state_key=state_key,
        legal_candidates=candidates,
        chosen_candidate=chosen_candidate,
        revealed_card_ids=(),
        matchup_cell=matchup_cell,
        policy_id=policy_id,
        deck_hash=deck_hash,
        response_block_id=response_block_id,
        root_action_family=root_action_family,
    )


def build_game_response_examples(
    rows: Sequence[dict[str, Any]],
    *,
    matchup_cell: str,
    stats: Counter[str] | None = None,
) -> list[ProspectiveResponseExample]:
    """Select decisions in each actor block after an observed control pass."""
    stats = stats if stats is not None else Counter()
    examples: list[ProspectiveResponseExample] = []
    previous: dict[str, Any] | None = None
    response_block_id = ""
    root_action_family = "UNKNOWN"
    block_index = -1
    for row in rows:
        assert_actor_visible_trace_row_safe(row)
        actor = int(row["actor_seat"])
        if previous is None:
            previous = row
            continue
        previous_actor = int(previous["actor_seat"])
        if actor != previous_actor:
            if previous.get("control_passed") is not True:
                raise ValueError("actor changed without an exact control-pass transition")
            block_index += 1
            response_block_id = f"{row['game_id']}:response:{block_index:04d}"
            root_action_family = _chosen_action_family(previous)
            stats["response_blocks"] += 1
        elif previous.get("control_passed") is True:
            raise ValueError("control_passed row did not transfer the next decision")

        if block_index >= 0:
            stats["response_decisions_seen"] += 1
            example = _build_response_example(
                row,
                matchup_cell=matchup_cell,
                response_block_id=response_block_id,
                root_action_family=root_action_family,
                stats=stats,
            )
            if example is not None:
                examples.append(example)
                stats["response_examples_eligible"] += 1
        previous = row
    if previous is not None and previous.get("control_passed") is True:
        raise ValueError("terminal control-pass row has no responding actor decision")

    return examples


def _verify_collection_manifests(source_root: Path) -> dict[str, Any]:
    plan_paths = sorted(source_root.glob("*/plan.json"))
    if not plan_paths:
        raise ValueError("prospective response source has no plans")
    safety = Counter()
    tranche_counts = Counter()
    league_snapshots: set[str] = set()
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for tranche in plan.get("tranches") or []:
            tranche_id = str(tranche.get("tranche_id") or "")
            if tranche_id not in {CANARY_TRANCHE, TRAIN_TRANCHE, EVALUATION_TRANCHE}:
                raise ValueError(f"unexpected prospective response tranche: {tranche_id}")
            if tranche.get("status") != "completed":
                raise ValueError(f"prospective response tranche is incomplete: {plan_path}:{tranche_id}")
            manifest_path = plan_path.parent / "tranches" / tranche_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("information_mode") != "actor_visible":
                raise ValueError(f"prospective response tranche is not actor-visible: {manifest_path}")
            manifest_safety = manifest.get("safety") or {}
            for field in ("crashes", "timeouts", "illegal"):
                safety[field] += int(manifest_safety.get(field) or 0)
            if any(int(manifest_safety.get(field) or 0) for field in ("crashes", "timeouts", "illegal")):
                raise ValueError(f"prospective response tranche failed safety: {manifest_path}")
            tranche_counts[tranche_id] += int((manifest.get("coverage") or {}).get("games") or 0)
            snapshot = ((manifest.get("contract") or {}).get("frozen_snapshot") or {})
            league_snapshots.add(str(snapshot.get("league_snapshot_id") or "unavailable"))

    return {
        "plans": len(plan_paths),
        "tranche_games": dict(sorted(tranche_counts.items())),
        "safety": dict(safety),
        "league_snapshots": sorted(league_snapshots),
    }


def collect_prospective_response_examples(
    source_root: str | Path,
) -> tuple[list[ProspectiveResponseExample], dict[str, Any], dict[str, Any]]:
    source_root = Path(source_root)
    manifest_receipt = _verify_collection_manifests(source_root)
    shards = sorted(source_root.glob("*/tranches/*/shards/decisions_shard_*.jsonl.gz"))
    if not shards:
        raise ValueError("prospective response source has no decision shards")

    stats: Counter[str] = Counter()
    examples: list[ProspectiveResponseExample] = []
    games: set[str] = set()
    policies: set[str] = set()
    decks: set[str] = set()
    matchups: set[str] = set()
    tranches: Counter[str] = Counter()
    rows_seen = 0
    control_pass_rows = 0
    control_pass_games = 0
    for shard in shards:
        rows = list(iter_trace_rows(shard))
        if not rows:
            raise ValueError(f"empty prospective response shard: {shard}")
        game_ids = {str(row.get("game_id") or "") for row in rows}
        if len(game_ids) != 1 or "" in game_ids:
            raise ValueError(f"prospective response shard does not contain one identified game: {shard}")
        game_id = next(iter(game_ids))
        if game_id in games:
            raise ValueError(f"duplicate prospective response game: {game_id}")
        games.add(game_id)
        matchup_cell = shard.parents[3].name
        matchups.add(matchup_cell)
        rows = sorted(rows, key=lambda row: int(row["decision_index"]))
        rows_seen += len(rows)
        game_control_pass_rows = sum(row.get("control_passed") is True for row in rows)
        control_pass_rows += game_control_pass_rows
        control_pass_games += int(game_control_pass_rows > 0)
        tranche_id = str(rows[0].get("tranche_id") or "")
        if any(str(row.get("tranche_id") or "") != tranche_id for row in rows):
            raise ValueError(f"mixed tranches in prospective response shard: {shard}")
        tranches[tranche_id] += 1
        policies.update(str(row["policy_id"]) for row in rows)
        decks.update(str(row["deck_hash"]) for row in rows)
        examples.extend(build_game_response_examples(rows, matchup_cell=matchup_cell, stats=stats))

    train_games = {example.episode_id for example in examples if example.response_split == TRAIN_TRANCHE}
    evaluation_games = {
        example.episode_id for example in examples if example.response_split == EVALUATION_TRANCHE
    }
    if train_games.intersection(evaluation_games):
        raise ValueError("prospective response train/evaluation games overlap")
    dataset = {
        "information_mode": "actor_visible",
        "rows": rows_seen,
        "games": len(games),
        "matchup_cells": len(matchups),
        "matchup_cell_ids": sorted(matchups),
        "policy_ids": sorted(policies),
        "deck_hashes": sorted(decks),
        "tranche_games": dict(sorted(tranches.items())),
        "train_tranche": TRAIN_TRANCHE,
        "evaluation_tranche": EVALUATION_TRANCHE,
        "canary_tranche_excluded": CANARY_TRANCHE,
        "train_examples": sum(example.response_split == TRAIN_TRANCHE for example in examples),
        "evaluation_examples": sum(
            example.response_split == EVALUATION_TRANCHE for example in examples
        ),
        "eligible_control_pass_rows": control_pass_rows,
        "complete_control_pass_games": control_pass_games,
        "stats": dict(sorted(stats.items())),
        "game_overlap": 0,
        "strength_bucket": "not_instrumented",
    }

    return examples, dataset, {
        "collection_manifests": manifest_receipt,
        "source_shards": _tree_receipt(shards, source_root),
    }


def fit_response_model(examples: Iterable[ProspectiveResponseExample]) -> ResponseCountModel:
    model = ResponseCountModel()
    for example in examples:
        model.observe(example)
    if model.global_cell.total == 0:
        raise ValueError("prospective response model received no training examples")

    return model


def _uniform_probabilities(example: ProspectiveResponseExample) -> dict[Candidate, float]:
    probability = 1.0 / len(example.legal_candidates)

    return {candidate: probability for candidate in example.legal_candidates}


def _group_loss_rows(
    records: Sequence[LossRecord],
    field: str,
    *,
    min_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[LossRecord]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, field))].append(record)
    rows = []
    for value, group in sorted(grouped.items()):
        means = {
            model: statistics.fmean(record.losses[model] for record in group)
            for model in ("uniform", "R0_state", "R1_policy")
        }
        rows.append({
            "value": value,
            "samples": len(group),
            "supported": len(group) >= min_samples,
            "nll": {model: round(metric, 6) for model, metric in means.items()},
            "uniform_minus_r0_nll": round(means["uniform"] - means["R0_state"], 6),
            "r0_minus_r1_nll": round(means["R0_state"] - means["R1_policy"], 6),
        })

    return rows


def evaluate_response_models(
    model: ResponseCountModel,
    examples: Sequence[ProspectiveResponseExample],
) -> tuple[dict[str, Any], list[LossRecord]]:
    if not examples:
        raise ValueError("prospective response evaluation requires examples")
    accumulators = {
        name: PredictorAccumulator(MetricAccumulator())
        for name in ("uniform", "R0_state", "R1_policy")
    }
    records: list[LossRecord] = []
    for example in examples:
        probabilities = {}
        started = time.perf_counter()
        probabilities["uniform"] = _uniform_probabilities(example)
        uniform_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        probabilities["R0_state"] = model.probabilities_state(example)
        r0_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        probabilities["R1_policy"] = model.probabilities_archetype(example, example.policy_id)
        r1_elapsed = time.perf_counter() - started
        elapsed = {
            "uniform": uniform_elapsed,
            "R0_state": r0_elapsed,
            "R1_policy": r1_elapsed,
        }
        losses = {
            name: accumulators[name].add(
                probabilities[name], example, elapsed_seconds=elapsed[name],
            )
            for name in accumulators
        }
        records.append(LossRecord(
            action_family=example.action_family,
            policy_id=example.policy_id,
            matchup_cell=example.matchup_cell,
            deck_hash=example.deck_hash,
            losses=losses,
        ))

    metrics = {name: accumulator.result() for name, accumulator in accumulators.items()}
    metrics["marginal_contribution"] = {
        "uniform_minus_r0_nll": round(
            metrics["uniform"]["nll"] - metrics["R0_state"]["nll"], 6,
        ),
        "r0_minus_r1_nll": round(
            metrics["R0_state"]["nll"] - metrics["R1_policy"]["nll"], 6,
        ),
        "r1_minus_r0_brier": round(
            metrics["R1_policy"]["brier"] - metrics["R0_state"]["brier"], 6,
        ),
    }
    evaluation = {
        "metrics": metrics,
        "slices": {
            "action_family": _group_loss_rows(
                records, "action_family", min_samples=MIN_GROUP_SAMPLES,
            ),
            "policy_id": _group_loss_rows(records, "policy_id", min_samples=MIN_GROUP_SAMPLES),
            "matchup_cell": _group_loss_rows(
                records, "matchup_cell", min_samples=MIN_MATCHUP_SAMPLES,
            ),
            "deck_hash": _group_loss_rows(records, "deck_hash", min_samples=MIN_GROUP_SAMPLES),
        },
    }

    return evaluation, records


def _stable_training_sample(
    examples: Sequence[ProspectiveResponseExample],
    seed: int,
) -> list[ProspectiveResponseExample]:
    selected_games = {
        example.episode_id
        for example in examples
        if int(hashlib.sha256(f"{seed}:{example.episode_id}".encode("utf-8")).hexdigest()[:8], 16)
        / 0xFFFFFFFF
        < SUBSAMPLE_FRACTION
    }

    return [example for example in examples if example.episode_id in selected_games]


def _stability_receipt(
    train: Sequence[ProspectiveResponseExample],
    evaluation: Sequence[ProspectiveResponseExample],
) -> list[dict[str, Any]]:
    rows = []
    for seed in STABILITY_SEEDS:
        sampled = _stable_training_sample(train, seed)
        model = fit_response_model(sampled)
        result, _records = evaluate_response_models(model, evaluation)
        contribution = result["metrics"]["marginal_contribution"]
        rows.append({
            "seed": seed,
            "train_examples": len(sampled),
            "train_games": len({example.episode_id for example in sampled}),
            **contribution,
        })

    return rows


def _holdout_receipt(
    train: Sequence[ProspectiveResponseExample],
    evaluation: Sequence[ProspectiveResponseExample],
    field: str,
) -> list[dict[str, Any]]:
    rows = []
    for value in sorted({str(getattr(example, field)) for example in evaluation}):
        fit_rows = [example for example in train if str(getattr(example, field)) != value]
        score_rows = [example for example in evaluation if str(getattr(example, field)) == value]
        if not fit_rows or not score_rows:
            continue
        model = fit_response_model(fit_rows)
        result, _records = evaluate_response_models(model, score_rows)
        rows.append({
            "held_out": value,
            "train_examples": len(fit_rows),
            "evaluation_examples": len(score_rows),
            **result["metrics"]["marginal_contribution"],
        })

    return rows


def _supported_group_gate(
    rows: Sequence[dict[str, Any]],
    contribution_field: str,
) -> dict[str, Any]:
    supported = [row for row in rows if row["supported"]]
    nonregressed = [row for row in supported if float(row[contribution_field]) >= 0.0]
    worst_regression = max(
        (max(0.0, -float(row[contribution_field])) for row in supported),
        default=0.0,
    )

    return {
        "supported_groups": len(supported),
        "nonregressed_groups": len(nonregressed),
        "nonregressed_fraction": round(len(nonregressed) / len(supported), 6) if supported else 0.0,
        "worst_nll_regression": round(worst_regression, 6),
        "passed": bool(supported)
        and len(nonregressed) * 2 >= len(supported)
        and worst_regression <= MAX_GROUP_NLL_REGRESSION,
    }


def response_feature_gates(
    evaluation: dict[str, Any],
    stability: Sequence[dict[str, Any]],
    held_out_matchups: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    contribution = metrics["marginal_contribution"]
    r0_family = _supported_group_gate(
        evaluation["slices"]["action_family"], "uniform_minus_r0_nll",
    )
    r0_matchup = _supported_group_gate(
        evaluation["slices"]["matchup_cell"], "uniform_minus_r0_nll",
    )
    r1_family = _supported_group_gate(
        evaluation["slices"]["action_family"], "r0_minus_r1_nll",
    )
    r1_matchup = _supported_group_gate(
        evaluation["slices"]["matchup_cell"], "r0_minus_r1_nll",
    )
    seed_gains = [float(row["r0_minus_r1_nll"]) for row in stability]
    seed_mean = statistics.fmean(seed_gains)
    seed_std = statistics.pstdev(seed_gains)
    seed_positive = sum(gain > 0.0 for gain in seed_gains)
    holdout_gains = [float(row["r0_minus_r1_nll"]) for row in held_out_matchups]
    holdout_mean = statistics.fmean(holdout_gains) if holdout_gains else float("-inf")
    holdout_worst_regression = max((max(0.0, -gain) for gain in holdout_gains), default=float("inf"))
    holdout_nonregressed = sum(gain >= 0.0 for gain in holdout_gains)
    r0_checks = {
        "minimum_aggregate_nll_gain": float(contribution["uniform_minus_r0_nll"]) >= MIN_R0_NLL_GAIN,
        "aggregate_brier_nonregression": metrics["R0_state"]["brier"] <= metrics["uniform"]["brier"],
        "action_family_breadth": r0_family["passed"],
        "matchup_breadth": r0_matchup["passed"],
    }
    r1_checks = {
        "minimum_aggregate_nll_gain": float(contribution["r0_minus_r1_nll"]) >= MIN_R1_NLL_GAIN,
        "aggregate_brier_nonregression": (
            float(contribution["r1_minus_r0_brier"]) <= MAX_AGGREGATE_BRIER_REGRESSION
        ),
        "action_family_breadth": r1_family["passed"],
        "matchup_breadth": r1_matchup["passed"],
        "seed_stability": seed_mean >= 0.0
        and seed_positive * 2 >= len(seed_gains)
        and seed_std <= MAX_SEED_GAIN_STD,
        "held_out_matchup_stability": bool(holdout_gains)
        and holdout_mean >= 0.0
        and holdout_nonregressed * 2 >= len(holdout_gains)
        and holdout_worst_regression <= MAX_GROUP_NLL_REGRESSION,
    }

    return {
        "thresholds": {
            "min_r0_nll_gain": MIN_R0_NLL_GAIN,
            "min_r1_nll_gain": MIN_R1_NLL_GAIN,
            "max_aggregate_brier_regression": MAX_AGGREGATE_BRIER_REGRESSION,
            "max_group_nll_regression": MAX_GROUP_NLL_REGRESSION,
            "max_seed_gain_std": MAX_SEED_GAIN_STD,
            "minimum_supported_group_nonregressed_fraction": 0.5,
        },
        "R0_state": {
            "checks": r0_checks,
            "passed": all(r0_checks.values()),
            "action_family_gate": r0_family,
            "matchup_gate": r0_matchup,
        },
        "R1_policy": {
            "checks": r1_checks,
            "passed": all(r1_checks.values()),
            "action_family_gate": r1_family,
            "matchup_gate": r1_matchup,
            "seed_gain_mean": round(seed_mean, 6),
            "seed_gain_std": round(seed_std, 6),
            "seed_positive": seed_positive,
            "held_out_matchup_gain_mean": round(holdout_mean, 6),
            "held_out_matchup_worst_regression": round(holdout_worst_regression, 6),
        },
    }


def render_report(result: dict[str, Any]) -> str:
    dataset = result["dataset"]
    evaluation = result["evaluation"]
    metrics = evaluation["chronological_T1"]["metrics"]
    gates = evaluation["gates"]
    lines = [
        "# Prospective actor-visible response ablation",
        "",
        f"Verdict: **{result['verdict']}**.",
        "",
        "## Evidence",
        "",
        f"- Source: {dataset['games']:,} games / {dataset['rows']:,} actor-visible decisions / "
        f"{dataset['matchup_cells']} matchup cells; safety 0/0/0.",
        f"- Response examples: {dataset['train_examples']:,} T0 train and "
        f"{dataset['evaluation_examples']:,} T1 evaluation; S0 excluded as canary.",
        f"- Exact control-pass support: {dataset['eligible_control_pass_rows']:,} rows across "
        f"{dataset['complete_control_pass_games']:,} complete games.",
        f"- Uniform NLL: {metrics['uniform']['nll']:.6f}.",
        f"- R0 state/legal NLL: {metrics['R0_state']['nll']:.6f}; gain "
        f"{metrics['marginal_contribution']['uniform_minus_r0_nll']:+.6f}.",
        f"- R1 policy-conditioned NLL: {metrics['R1_policy']['nll']:.6f}; marginal gain "
        f"{metrics['marginal_contribution']['r0_minus_r1_nll']:+.6f}.",
        f"- R0 gate: {'PASS' if gates['R0_state']['passed'] else 'FAIL'}; "
        f"R1 gate: {'PASS' if gates['R1_policy']['passed'] else 'FAIL'}.",
        f"- R1 worst held-out-matchup NLL regression: "
        f"{gates['R1_policy']['held_out_matchup_worst_regression']:.6f}.",
        "",
        "## Boundary",
        "",
        "R0/R1 are observational diagnostics fitted from the responding actor's legitimate "
        "actor-visible state. They are not callable from a strict-public opponent state without "
        "a validated hidden-state sampler. No strategy tensor or strength bucket was consumed.",
        "Q count support now passes, but Q remains blocked at the sampler/branch boundary. "
        "Nothing is wired into V, policy, runtime, league, or submissions.",
        "",
    ]

    return "\n".join(lines)


def run_prospective_response_ablation(
    *,
    source_root: str | Path,
    canonical_catalog: str | Path,
    output_dir: str | Path,
    entrypoint: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_path = Path(__file__).with_name("baseline.py")
    validator_path = Path(__file__).resolve().parents[1] / "replay" / "prospective_trace.py"
    examples, dataset, source_receipt = collect_prospective_response_examples(source_root)
    canonical_catalog = Path(canonical_catalog)
    catalog = json.loads(canonical_catalog.read_text(encoding="utf-8"))
    totals = catalog.get("totals") or {}
    if int(totals.get("rows_kept") or 0) != dataset["rows"]:
        raise ValueError("canonical catalog row count does not match prospective response source")
    if int(totals.get("games_kept") or 0) != dataset["games"]:
        raise ValueError("canonical catalog game count does not match prospective response source")
    if int(catalog.get("duplicate_game_seat_blocks_across_units") or 0) != 0:
        raise ValueError("canonical response catalog contains duplicate game-seat blocks")

    train = [example for example in examples if example.response_split == TRAIN_TRANCHE]
    evaluation_examples = [
        example for example in examples if example.response_split == EVALUATION_TRANCHE
    ]
    if not train or not evaluation_examples:
        raise ValueError("prospective response ablation requires non-empty T0 and T1 examples")
    fit_started = time.perf_counter()
    model = fit_response_model(train)
    fit_seconds = time.perf_counter() - fit_started
    chronological, _records = evaluate_response_models(model, evaluation_examples)
    stability = _stability_receipt(train, evaluation_examples)
    held_out_matchups = _holdout_receipt(train, evaluation_examples, "matchup_cell")
    held_out_policies = _holdout_receipt(train, evaluation_examples, "policy_id")
    held_out_decks = _holdout_receipt(train, evaluation_examples, "deck_hash")
    gates = response_feature_gates(chronological, stability, held_out_matchups)
    if gates["R1_policy"]["passed"]:
        verdict = "R1_POLICY_FAMILY_DIAGNOSTIC_PASS"
    elif gates["R0_state"]["passed"]:
        verdict = "R0_DIAGNOSTIC_PASS_R1_REJECTED"
    else:
        verdict = "RESPONSE_BASELINE_REJECTED"
    metrics = chronological["metrics"]
    feature_contribution_audit = {
        "feature": "known responding policy family",
        "baseline": "R0_state",
        "candidate": "R1_policy",
        "equal_budget_contract": {
            "training_examples_equal": True,
            "evaluation_examples_equal": True,
            "chronological_split_equal": True,
            "legal_action_support_equal": True,
        },
        "benefit": {
            "aggregate_nll_gain": metrics["marginal_contribution"]["r0_minus_r1_nll"],
            "aggregate_brier_delta": metrics["marginal_contribution"]["r1_minus_r0_brier"],
            "top1_delta": round(
                metrics["R1_policy"]["top1_accuracy"] - metrics["R0_state"]["top1_accuracy"],
                6,
            ),
        },
        "cost": {
            "r0_inference_ms_per_sample": metrics["R0_state"]["inference_ms_per_sample"],
            "r1_inference_ms_per_sample": metrics["R1_policy"]["inference_ms_per_sample"],
            "known_policy_identity_required": True,
        },
        "damage": {
            "action_family_worst_nll_regression": (
                gates["R1_policy"]["action_family_gate"]["worst_nll_regression"]
            ),
            "matchup_worst_nll_regression": (
                gates["R1_policy"]["matchup_gate"]["worst_nll_regression"]
            ),
            "held_out_matchup_worst_nll_regression": (
                gates["R1_policy"]["held_out_matchup_worst_regression"]
            ),
            "wrong_lock_rate_delta": round(
                metrics["R1_policy"]["wrong_lock_rate"] - metrics["R0_state"]["wrong_lock_rate"],
                6,
            ),
        },
        "uniqueness": (
            "categorical source-policy identity; not a substitute for the rejected strategy tensor, "
            "playstyle, or empirical strength"
        ),
        "disposition": "diagnostic_only" if gates["R1_policy"]["passed"] else "quarantine_rejected",
        "reason": (
            "broad supported groups and held-out matchup checks all passed"
            if gates["R1_policy"]["passed"]
            else "aggregate gain did not survive the predeclared group-damage and held-out matchup gates"
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "deployment_status": "blocked_without_validated_public_hidden_state_sampler",
        "control_authority": "none",
        "information_mode": "actor_visible_training_diagnostic",
        "protected_splits": {"sealed_test_consumed": False},
        "models": {
            "R0_state": "state and legal-action count-table control",
            "R1_policy": "R0 plus known responding policy family",
            "R2_strategy_tensor": "not_run_rejected_full_list_classifier",
            "R3_playstyle": "not_run_no_validated_policy_deconfounded_style",
            "R4_strength": "not_run_strength_not_instrumented",
            "R5_full": "not_run_upstream_features_unavailable",
        },
        "training": {
            "examples": len(train),
            "games": len({example.episode_id for example in train}),
            "seconds": round(fit_seconds, 6),
            "min_cell_support": model.min_support,
            "smoothing_alpha": model.alpha,
            "model_sha256": model.fingerprint(),
        },
        "dataset": dataset,
        "evaluation": {
            "chronological_T1": chronological,
            "stability_subsamples": stability,
            "held_out_matchup_cells": held_out_matchups,
            "held_out_policy_families": held_out_policies,
            "held_out_decks": held_out_decks,
            "gates": gates,
            "gameplay_value": "not_measured",
            "q_ranking_usefulness": "not_measured_sampler_blocked",
        },
        "feature_contribution_audit": feature_contribution_audit,
        "input_identity": {
            "source_root": str(Path(source_root)),
            "canonical_catalog": str(canonical_catalog),
            "canonical_catalog_sha256": _sha256(canonical_catalog),
            "response_baseline_sha256": _sha256(baseline_path),
            "actor_visible_validator_sha256": _sha256(validator_path),
            **source_receipt,
        },
        "artifact_identity": {
            "ontology_version": "deepmew.strategy_tensor_ontology.v1_not_consumed",
            "strategy_classifier": "rejected_not_consumed",
            "deck_hashes": dataset["deck_hashes"],
            "policy_families": dataset["policy_ids"],
            "generation": "prospective_actor_visible_v1",
            "strength_bucket": "not_instrumented",
            "league_snapshot": source_receipt["collection_manifests"]["league_snapshots"],
            "calibration": {"kind": "additive_count_smoothing", "alpha": model.alpha},
            "selection_status": "diagnostic_only_not_selected_for_runtime",
            "known_limitations": [
                "observational response likelihood is not gameplay value",
                "actor-visible response states require a strict-public hidden-state sampler at deployment",
                "multi-select and single-semantic decisions are excluded",
                "no time-valid empirical strength bucket is available",
            ],
        },
        "q_gate": {
            "minimum_rows": 5_000,
            "minimum_games": 50,
            "eligible_control_pass_rows": dataset["eligible_control_pass_rows"],
            "complete_control_pass_games": dataset["complete_control_pass_games"],
            "canonical_rows": dataset["rows"],
            "canonical_games": dataset["games"],
            "count_gate": (
                "pass"
                if dataset["eligible_control_pass_rows"] >= 5_000
                and dataset["complete_control_pass_games"] >= 50
                else "fail"
            ),
            "public_hidden_state_sampler": "missing",
            "verdict": (
                "COUNT_GATE_PASS_SAMPLER_GATE_BLOCKED"
                if dataset["eligible_control_pass_rows"] >= 5_000
                and dataset["complete_control_pass_games"] >= 50
                else "COUNT_GATE_FAIL"
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    write_text(output_dir / "REPORT.md", render_report(result))
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "created_at_utc": utc_now(),
        "status": "completed",
        "verdict": verdict,
        "result_sha256": _sha256(output_dir / "result.json"),
        "report_sha256": _sha256(output_dir / "REPORT.md"),
        "implementation_sha256": _sha256(Path(__file__)),
        "response_baseline_sha256": _sha256(baseline_path),
        "actor_visible_validator_sha256": _sha256(validator_path),
        "entrypoint_sha256": _sha256(entrypoint),
        "model_sha256": model.fingerprint(),
        "source_tree_sha256": source_receipt["source_shards"]["tree_sha256"],
        "canonical_catalog_sha256": _sha256(canonical_catalog),
    }
    write_json(output_dir / "manifest.json", manifest)

    return result
