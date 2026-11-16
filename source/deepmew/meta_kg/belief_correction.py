"""Train/eval-only correction for the strict-public no-staple DSU posterior.

The first DSU evaluation showed that staple attenuation damaged holdout NLL, but
its evidence-scale calibration had only been fit for the staple variant. This
module resolves that mismatch without touching the already-consumed holdout:
the existing eval split is partitioned by episode into calibration and audit
halves, the scale is selected on calibration episodes, and damage is reported
on disjoint eval-audit episodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from deepmew.artifacts.writers import write_json, write_text
from deepmew.imitation.io import (
    ACTION_TABLE_FILENAME,
    iter_action_rows,
    load_episode_seat_decks,
    load_label_map,
    load_split_map,
)
from deepmew.meta_kg.belief_evaluation import (
    DEFAULT_CALIBRATION_SCALES,
    BeliefStatistics,
    RevealDataset,
    RevealExample,
    collect_reveal_dataset,
    evaluate_examples,
    fit_belief_statistics,
    mapping_fingerprint,
)
from deepmew.meta_kg.io import aggregate_unique_decks, load_deck_artifacts
from deepmew.meta_kg.live_belief import LiveArchetypeBelief


SCHEMA_VERSION = "deepmew.dsu_no_staple_correction.v0"
MODEL_SPEC_VERSION = "deepmew.dsu_no_staple_model_spec.v0"
INTERNAL_PARTITION_VERSION = "sha256_episode_parity.v0"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def fingerprint_action_tables(ingest_root: str | Path) -> dict[str, Any]:
    """Content-address every replay action table used by the correction."""
    root = Path(ingest_root)
    rows = []
    total_bytes = 0
    for path in sorted(root.rglob(ACTION_TABLE_FILENAME)):
        size = path.stat().st_size
        total_bytes += size
        rows.append((path.relative_to(root).as_posix(), size, _sha256(path)))
    if not rows:
        raise ValueError(f"no {ACTION_TABLE_FILENAME} inputs under {root}")

    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "aggregate_sha256": _stable_hash(rows),
    }


def fingerprint_unique_decks(unique_decks: dict[str, Any]) -> str:
    """Hash the normalized unique-deck training surface, not filesystem times."""
    rows = []
    for deck_hash, deck in sorted(unique_decks.items()):
        card_counts = getattr(deck, "cards", {})
        rows.append((deck_hash, sorted((str(card_id), int(count)) for card_id, count in card_counts.items())))

    return _stable_hash(rows)


def build_no_staple_model(
    statistics: BeliefStatistics,
    *,
    evidence_scale: float,
    use_evidence: bool = True,
) -> LiveArchetypeBelief:
    scale = evidence_scale if use_evidence else 0.0

    return LiveArchetypeBelief(
        archetype_priors=statistics.priors,
        card_likelihoods=statistics.card_likelihoods,
        global_likelihoods=statistics.global_likelihoods,
        staple_scores={},
        evidence_scale=scale,
    )


def partition_eval_examples(
    examples: Sequence[RevealExample],
) -> tuple[list[RevealExample], list[RevealExample]]:
    """Keep every episode wholly in the calibration or audit half."""
    calibration = []
    audit = []
    for example in examples:
        digest = hashlib.sha256(example.episode_id.encode("utf-8")).digest()
        destination = calibration if digest[0] % 2 == 0 else audit
        destination.append(example)
    calibration_episodes = {example.episode_id for example in calibration}
    audit_episodes = {example.episode_id for example in audit}
    if not calibration or not audit or calibration_episodes.intersection(audit_episodes):
        raise ValueError("eval partition must produce disjoint non-empty episode groups")

    return calibration, audit


def _support_tail(statistics: BeliefStatistics) -> tuple[set[str], int]:
    ordered_counts = sorted(statistics.train_decks_by_archetype.values())
    if not ordered_counts:
        raise ValueError("cannot define support tail without train archetypes")
    quartile_index = max(0, math.ceil(len(ordered_counts) * 0.25) - 1)
    threshold = ordered_counts[quartile_index]
    tail = {
        archetype
        for archetype, count in statistics.train_decks_by_archetype.items()
        if count <= threshold
    }

    return tail, threshold


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "nll": round(candidate["nll"] - baseline["nll"], 6),
        "brier": round(candidate["brier"] - baseline["brier"], 6),
        "top1_accuracy": round(candidate["top1_accuracy"] - baseline["top1_accuracy"], 6),
        "top3_accuracy": round(candidate["top3_accuracy"] - baseline["top3_accuracy"], 6),
        "top_label_ece": round(candidate["top_label_ece"] - baseline["top_label_ece"], 6),
    }


def evaluate_no_staple_correction(
    statistics: BeliefStatistics,
    dataset: RevealDataset,
    *,
    calibration_scales: Sequence[float] = DEFAULT_CALIBRATION_SCALES,
) -> dict[str, Any]:
    eval_examples = [example for example in dataset.finals if example.split == "eval"]
    if len(eval_examples) != len(dataset.finals):
        raise ValueError("no-staple correction accepts eval examples only")
    calibration, audit = partition_eval_examples(eval_examples)

    scale_trials = {}
    for scale in calibration_scales:
        model = build_no_staple_model(statistics, evidence_scale=float(scale))
        scale_trials[str(scale)] = evaluate_examples(model, calibration, statistics.archetypes)
    selected_scale = min(
        (float(scale) for scale in calibration_scales),
        key=lambda scale: (scale_trials[str(scale)]["nll"], abs(scale - 1.0), scale),
    )

    uncalibrated_model = build_no_staple_model(statistics, evidence_scale=1.0)
    calibrated_model = build_no_staple_model(statistics, evidence_scale=selected_scale)
    audit_metrics = {
        "B2_uncalibrated_no_staple": evaluate_examples(
            uncalibrated_model, audit, statistics.archetypes
        ),
        "B2C_eval_calibrated_no_staple": evaluate_examples(
            calibrated_model, audit, statistics.archetypes
        ),
    }
    overall_delta = _metric_delta(
        audit_metrics["B2C_eval_calibrated_no_staple"],
        audit_metrics["B2_uncalibrated_no_staple"],
    )

    per_archetype = {}
    for archetype in statistics.archetypes:
        members = [example for example in audit if example.archetype == archetype]
        if not members:
            continue
        baseline = evaluate_examples(uncalibrated_model, members, statistics.archetypes)
        candidate = evaluate_examples(calibrated_model, members, statistics.archetypes)
        per_archetype[archetype] = {
            "train_decks": statistics.train_decks_by_archetype[archetype],
            "audit_samples": len(members),
            "uncalibrated": baseline,
            "calibrated": candidate,
            "delta": _metric_delta(candidate, baseline),
        }

    tail_archetypes, tail_threshold = _support_tail(statistics)
    support_slices = {}
    for name, selected in (
        ("lowest_support_quartile", tail_archetypes),
        ("remaining_support", set(statistics.archetypes) - tail_archetypes),
    ):
        members = [example for example in audit if example.archetype in selected]
        if not members:
            continue
        baseline = evaluate_examples(uncalibrated_model, members, statistics.archetypes)
        candidate = evaluate_examples(calibrated_model, members, statistics.archetypes)
        support_slices[name] = {
            "archetypes": sorted(selected),
            "audit_samples": len(members),
            "uncalibrated": baseline,
            "calibrated": candidate,
            "delta": _metric_delta(candidate, baseline),
        }

    calibrated_wins = overall_delta["nll"] < 0.0
    support_slice_nll_deltas = {
        name: metrics["delta"]["nll"]
        for name, metrics in support_slices.items()
    }
    support_slices_not_damaged = all(delta <= 0.0 for delta in support_slice_nll_deltas.values())
    selected_variant = (
        "B2C_eval_calibrated_no_staple"
        if calibrated_wins and support_slices_not_damaged
        else "B2_uncalibrated_no_staple"
    )
    calibration_decision = "diagnostic only" if selected_variant.startswith("B2C") else "quarantine"

    model_scale = selected_scale if selected_variant.startswith("B2C") else 1.0
    model_spec = {
        "schema_version": MODEL_SPEC_VERSION,
        "variant": selected_variant,
        "use_staples": False,
        "evidence_scale": model_scale,
        "archetypes": list(statistics.archetypes),
        "priors": statistics.priors,
        "card_likelihoods": sorted(
            (archetype, str(card_id), probability)
            for (archetype, card_id), probability in statistics.card_likelihoods.items()
        ),
        "global_likelihoods": sorted(
            (str(card_id), probability)
            for card_id, probability in statistics.global_likelihoods.items()
        ),
    }

    return {
        "partition": {
            "version": INTERNAL_PARTITION_VERSION,
            "calibration_samples": len(calibration),
            "audit_samples": len(audit),
            "calibration_episodes": len({example.episode_id for example in calibration}),
            "audit_episodes": len({example.episode_id for example in audit}),
            "episode_overlap": 0,
            "holdout_samples_scored": 0,
        },
        "selected_evidence_scale": selected_scale,
        "scale_trials": scale_trials,
        "audit_metrics": audit_metrics,
        "audit_delta_calibrated_vs_uncalibrated": overall_delta,
        "per_archetype": per_archetype,
        "support_tail": {
            "definition": "bottom train-deck support quartile including threshold ties",
            "train_deck_threshold": tail_threshold,
            "slices": support_slices,
        },
        "damage_gate": {
            "overall_nll_improved": calibrated_wins,
            "support_slice_nll_deltas": support_slice_nll_deltas,
            "all_support_slices_not_damaged": support_slices_not_damaged,
            "rule": "universal rescaling requires lower overall NLL and non-positive NLL delta in every broad support slice",
        },
        "selected_variant": selected_variant,
        "selected_model_spec_sha256": _stable_hash(model_spec),
        "calibration_feature_decision": calibration_decision,
        "feature_classification": "diagnostic only",
        "classification_reason": (
            "strict-public no-staple probability correction only; no equal-budget gameplay value measured"
        ),
    }


def render_correction_report(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    partition = evaluation["partition"]
    delta = evaluation["audit_delta_calibrated_vs_uncalibrated"]
    lines = [
        "# Strict-public DSU no-staple correction",
        "",
        f"Verdict: **{evaluation['feature_classification']}**.",
        "",
        "The prior holdout was not rescored. Train deck statistics and the existing eval split only were used.",
        "No posterior is wired into V, Q, policy, runtime, or submissions.",
        "",
        "## Internal eval partition",
        "",
        f"- Partition: `{partition['version']}`; episode overlap `{partition['episode_overlap']}`.",
        f"- Calibration: {partition['calibration_samples']} episode-seats / {partition['calibration_episodes']} episodes.",
        f"- Audit: {partition['audit_samples']} episode-seats / {partition['audit_episodes']} episodes.",
        f"- Holdout samples scored: {partition['holdout_samples_scored']}.",
        "",
        "## Eval-audit correction",
        "",
        "| Variant | NLL | Brier | Top-1 | Top-3 | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in evaluation["audit_metrics"].items():
        lines.append(
            f"| {name} | {metrics['nll']:.4f} | {metrics['brier']:.4f} | "
            f"{metrics['top1_accuracy']:.4f} | {metrics['top3_accuracy']:.4f} | "
            f"{metrics['top_label_ece']:.4f} |"
        )
    lines.extend([
        "",
        f"Calibration-half selected scale: `{evaluation['selected_evidence_scale']}`.",
        f"Audit NLL delta, calibrated minus uncalibrated: `{delta['nll']:+.6f}`.",
        f"Audit Brier delta: `{delta['brier']:+.6f}`; top-1 delta: `{delta['top1_accuracy']:+.6f}`.",
        f"Selected response-diagnostic belief variant: `{evaluation['selected_variant']}`.",
        f"Calibration feature decision: **{evaluation['calibration_feature_decision']}**.",
        f"Broad-slice damage gate passed: `{evaluation['damage_gate']['all_support_slices_not_damaged']}`.",
        "",
        "## Support-tail damage",
        "",
    ])
    for name, metrics in evaluation["support_tail"]["slices"].items():
        lines.append(
            f"- `{name}`: n={metrics['audit_samples']}, NLL delta={metrics['delta']['nll']:+.6f}, "
            f"top-1 delta={metrics['delta']['top1_accuracy']:+.6f}."
        )
    lines.extend([
        "",
        "Per-archetype sample sizes and deltas are recorded in `result.json`; sparse slices remain descriptive.",
        "",
        f"Classification reason: {evaluation['classification_reason']}.",
        "",
    ])

    return "\n".join(lines)


def run_no_staple_correction(
    *,
    meta_kg_dir: str | Path,
    deck_zoo_dir: str | Path,
    ingest_root: str | Path,
    opponent_decks_root: str | Path,
    output_dir: str | Path,
    max_games: int | None = None,
) -> dict[str, Any]:
    labels = load_label_map(meta_kg_dir)
    splits = load_split_map(deck_zoo_dir)
    deck_artifacts = load_deck_artifacts(opponent_decks_root)
    unique_decks, conflicts = aggregate_unique_decks(deck_artifacts)
    statistics = fit_belief_statistics(unique_decks, labels, splits)
    episode_seat_decks = load_episode_seat_decks(opponent_decks_root)
    dataset = collect_reveal_dataset(
        iter_action_rows(ingest_root),
        episode_seat_decks,
        labels,
        splits,
        statistics.archetypes,
        allowed_splits=("eval",),
        max_games=max_games,
    )
    evaluation = evaluate_no_staple_correction(statistics, dataset)
    result = {
        "schema_version": SCHEMA_VERSION,
        "information_mode": "public_belief",
        "feature_schema": "deepmew.public_reveal_prefix.v0",
        "consumed_splits": ["train_deck_statistics", "eval_internal_calibration", "eval_internal_audit"],
        "training": {
            "deck_count": statistics.train_deck_count,
            "archetype_count": len(statistics.archetypes),
            "decks_by_archetype": statistics.train_decks_by_archetype,
            "likelihood_cells": len(statistics.card_likelihoods),
        },
        "dataset": {
            "rows_scanned": dataset.rows_scanned,
            "unique_episodes": dataset.unique_episodes,
            "labeled_episode_seats": dataset.labeled_episode_seats,
            "max_games": max_games,
        },
        "input_identity": {
            "meta_kg_dir": str(Path(meta_kg_dir)),
            "deck_zoo_dir": str(Path(deck_zoo_dir)),
            "ingest_root": str(Path(ingest_root)),
            "opponent_decks_root": str(Path(opponent_decks_root)),
            "label_split_sha256": mapping_fingerprint(labels, splits),
            "unique_decks_sha256": fingerprint_unique_decks(unique_decks),
            "action_tables": fingerprint_action_tables(ingest_root),
            "deck_hash_conflicts": len(conflicts),
            "implementation_sha256": _sha256(Path(__file__)),
        },
        "artifact_identity": {
            "model_spec_sha256": evaluation["selected_model_spec_sha256"],
            "checkpoint": "not_persisted_diagnostic_model_spec_only",
            "generation": "offline_pre_generation_diagnostic",
            "deck_scope": "train-split supported archetypes",
            "policy_family": "external replay actors; not inferred",
            "league_snapshot": "not_applicable_offline_diagnostic",
            "safety_status": "not_instrumented_in_source_replays",
            "selection_status": "diagnostic_only_not_selected_for_runtime",
        },
        "evaluation": evaluation,
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    write_text(output_dir / "REPORT.md", render_correction_report(result))

    return result
