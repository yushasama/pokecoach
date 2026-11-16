"""Strict-public offline evaluation for the revealed-card archetype belief.

This module is research-only. It reprojects actor-visible replay rows through
the public information boundary, then allowlists only the observed player's
public board zones. Full decks are used solely for grouped labels and fitting
statistics; they never enter the reveal features.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from deepmew.artifacts.writers import write_json, write_text
from deepmew.imitation.io import (
    iter_action_rows,
    load_episode_seat_decks,
    load_label_map,
    load_split_map,
)
from deepmew.meta_kg.io import aggregate_unique_decks, load_deck_artifacts
from deepmew.meta_kg.live_belief import LiveArchetypeBelief
from deepmew.meta_kg.schemas import CardId, K2ABuildConfig, UniqueDeck
from deepmew.meta_kg.staples import normalized_entropy, staple_score
from deepmew.replay.prospective_trace import public_trace_observation


SCHEMA_VERSION = "deepmew.dsu_belief_evaluation.v0"
PUBLIC_FEATURE_SCHEMA = "deepmew.public_reveal_prefix.v0"
DEFAULT_CALIBRATION_SCALES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
CONFIDENCE_GATE = 0.80
MIN_TRAIN_DECKS_PER_ARCHETYPE = 5
PUBLIC_CARD_ID_FIELDS = ("id", "cardId")
PUBLIC_NESTED_CARD_FIELDS = ("preEvolution", "energies", "energyCards", "tool", "tools")


@dataclass(frozen=True)
class BeliefStatistics:
    archetypes: tuple[str, ...]
    priors: dict[str, float]
    card_likelihoods: dict[tuple[str, CardId], float]
    global_likelihoods: dict[CardId, float]
    staple_scores: dict[CardId, float]
    train_deck_count: int
    train_decks_by_archetype: dict[str, int]


@dataclass(frozen=True)
class RevealExample:
    episode_id: str
    observed_seat: int
    deck_hash: str
    archetype: str
    split: str
    revealed_card_ids: tuple[CardId, ...]


@dataclass(frozen=True)
class RevealDataset:
    prefixes: tuple[RevealExample, ...]
    finals: tuple[RevealExample, ...]
    rows_scanned: int
    labeled_episode_seats: int
    unique_episodes: int


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _card_ids_from_public_value(value: Any) -> list[CardId]:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, list):
        card_ids: list[CardId] = []
        for item in value:
            card_ids.extend(_card_ids_from_public_value(item))

        return card_ids
    if not isinstance(value, dict):
        return []

    card_ids = []
    for field in PUBLIC_CARD_ID_FIELDS:
        card_id = value.get(field)
        if isinstance(card_id, (str, int)) and not isinstance(card_id, bool):
            card_ids.append(str(card_id))
    for field in PUBLIC_NESTED_CARD_FIELDS:
        if field in value:
            card_ids.extend(_card_ids_from_public_value(value[field]))

    return card_ids


def extract_public_reveals(observation: dict[str, Any], observed_seat: int) -> tuple[CardId, ...]:
    """Return only first-class public board IDs for one observed seat.

    The broad projection runs first. The second stage deliberately ignores
    logs, identities, legal choices, hands, prize cards, deck order, outcomes,
    and every non-board field even if an upstream row happens to contain them.
    """
    public = public_trace_observation(observation)
    players = _as_list(_as_dict(public.get("current")).get("players"))
    if observed_seat < 0 or observed_seat >= len(players):
        return ()
    player = _as_dict(players[observed_seat])
    revealed: list[CardId] = []
    for zone in ("active", "bench", "discard"):
        revealed.extend(_card_ids_from_public_value(player.get(zone)))

    return tuple(dict.fromkeys(revealed))


def fit_belief_statistics(
    unique_decks: dict[str, UniqueDeck],
    labels: dict[str, dict[str, Any]],
    splits: dict[str, str],
    *,
    config: K2ABuildConfig | None = None,
    min_train_decks_per_archetype: int = MIN_TRAIN_DECKS_PER_ARCHETYPE,
) -> BeliefStatistics:
    """Fit priors, likelihoods, and staple attenuation on train decks only."""
    config = config or K2ABuildConfig()
    candidates: list[tuple[str, UniqueDeck, str]] = []
    archetype_counts: Counter[str] = Counter()
    for deck_hash, deck in unique_decks.items():
        label = labels.get(deck_hash, {})
        archetype = str(label.get("archetype") or "")
        if splits.get(deck_hash) != "train" or label.get("classification_status") != "confident" or not archetype:
            continue
        candidates.append((deck_hash, deck, archetype))
        archetype_counts[archetype] += 1

    eligible = tuple(sorted(a for a, count in archetype_counts.items() if count >= min_train_decks_per_archetype))
    if len(eligible) < 2:
        raise ValueError("belief evaluation requires at least two train-supported archetypes")
    eligible_set = set(eligible)
    train = [(deck_hash, deck, archetype) for deck_hash, deck, archetype in candidates if archetype in eligible_set]
    train_counts = Counter(archetype for _, _, archetype in train)
    total_decks = len(train)
    priors = {archetype: train_counts[archetype] / total_decks for archetype in eligible}

    global_counts: Counter[CardId] = Counter()
    conditional_counts: dict[str, Counter[CardId]] = {archetype: Counter() for archetype in eligible}
    for _, deck, archetype in train:
        present = {str(card_id) for card_id in deck.cards}
        global_counts.update(present)
        conditional_counts[archetype].update(present)

    alpha = config.lift_alpha
    global_likelihoods = {
        card_id: (count + alpha) / (total_decks + 2 * alpha)
        for card_id, count in global_counts.items()
    }
    all_conditional: dict[tuple[str, CardId], float] = {}
    lifts_by_card: dict[CardId, list[float]] = defaultdict(list)
    for archetype in eligible:
        archetype_total = train_counts[archetype]
        for card_id, p_global in global_likelihoods.items():
            count = conditional_counts[archetype].get(card_id, 0)
            p_cond = (count + alpha) / (archetype_total + 2 * alpha)
            all_conditional[(archetype, card_id)] = p_cond
            lifts_by_card[card_id].append(p_cond / max(p_global, 1e-12))

    staple_scores: dict[CardId, float] = {}
    for card_id, p_global in global_likelihoods.items():
        if p_global < config.staple_min_global_deck_freq:
            continue
        lifts = lifts_by_card[card_id]
        max_lift = max(lifts, default=1.0)
        if max_lift > config.staple_max_lift:
            continue
        entropy = normalized_entropy(lifts)
        staple_scores[card_id] = staple_score(p_global, max_lift, entropy, config)

    card_likelihoods: dict[tuple[str, CardId], float] = {}
    for (archetype, card_id), p_cond in all_conditional.items():
        archetype_total = train_counts[archetype]
        cond_freq = conditional_counts[archetype].get(card_id, 0) / archetype_total
        lift = p_cond / max(global_likelihoods[card_id], 1e-12)
        if lift >= config.min_report_lift or cond_freq >= config.min_report_conditional_freq:
            card_likelihoods[(archetype, card_id)] = p_cond

    return BeliefStatistics(
        archetypes=eligible,
        priors=priors,
        card_likelihoods=card_likelihoods,
        global_likelihoods=global_likelihoods,
        staple_scores=staple_scores,
        train_deck_count=total_decks,
        train_decks_by_archetype=dict(sorted(train_counts.items())),
    )


def collect_reveal_dataset(
    rows: Iterable[dict[str, Any]],
    episode_seat_decks: dict[str, dict[int, str]],
    labels: dict[str, dict[str, Any]],
    splits: dict[str, str],
    eligible_archetypes: Sequence[str],
    *,
    allowed_splits: Sequence[str] = ("eval", "holdout"),
    max_games: int | None = None,
) -> RevealDataset:
    """Build causal reveal prefixes and one final example per episode-seat."""
    eligible = set(eligible_archetypes)
    accepted_splits = set(allowed_splits)
    if not accepted_splits or not accepted_splits.issubset({"eval", "holdout"}):
        raise ValueError("reveal dataset allowed_splits must be a non-empty eval/holdout subset")
    seen_cards: dict[tuple[str, int], set[CardId]] = defaultdict(set)
    ordered_cards: dict[tuple[str, int], list[CardId]] = defaultdict(list)
    final_by_key: dict[tuple[str, int], RevealExample] = {}
    prefixes: list[RevealExample] = []
    accepted_games: set[str] = set()
    rows_scanned = 0

    for row in rows:
        rows_scanned += 1
        episode_id = str(row.get("game_id") or "")
        if not episode_id:
            continue
        state = row.get("state_json")
        if not isinstance(state, dict):
            continue
        current = _as_dict(state.get("current"))
        try:
            observer_seat = int(current.get("yourIndex"))
        except (TypeError, ValueError):
            continue
        observed_seat = 1 - observer_seat
        deck_hash = episode_seat_decks.get(episode_id, {}).get(observed_seat)
        label = labels.get(str(deck_hash), {})
        archetype = str(label.get("archetype") or "")
        split = splits.get(str(deck_hash), "")
        if (
            deck_hash is None
            or label.get("classification_status") != "confident"
            or archetype not in eligible
            or split not in accepted_splits
        ):
            continue
        if max_games is not None and episode_id not in accepted_games and len(accepted_games) >= max_games:
            continue
        accepted_games.add(episode_id)

        key = (episode_id, observed_seat)
        public_ids = extract_public_reveals(state, observed_seat)
        new_ids = [card_id for card_id in public_ids if card_id not in seen_cards[key]]
        if not new_ids:
            continue
        seen_cards[key].update(new_ids)
        ordered_cards[key].extend(new_ids)
        example = RevealExample(
            episode_id=episode_id,
            observed_seat=observed_seat,
            deck_hash=str(deck_hash),
            archetype=archetype,
            split=split,
            revealed_card_ids=tuple(ordered_cards[key]),
        )
        prefixes.append(example)
        final_by_key[key] = example

    finals = tuple(final_by_key[key] for key in sorted(final_by_key))

    return RevealDataset(
        prefixes=tuple(prefixes),
        finals=finals,
        rows_scanned=rows_scanned,
        labeled_episode_seats=len(finals),
        unique_episodes=len({example.episode_id for example in finals}),
    )


def _build_model(
    statistics: BeliefStatistics,
    *,
    uniform_prior: bool = False,
    use_staples: bool = True,
    evidence_scale: float = 1.0,
) -> LiveArchetypeBelief:
    priors = statistics.priors
    if uniform_prior:
        uniform = 1.0 / len(statistics.archetypes)
        priors = {archetype: uniform for archetype in statistics.archetypes}

    return LiveArchetypeBelief(
        archetype_priors=priors,
        card_likelihoods=statistics.card_likelihoods,
        global_likelihoods=statistics.global_likelihoods,
        staple_scores=statistics.staple_scores if use_staples else {},
        evidence_scale=evidence_scale,
    )


def evaluate_examples(
    model: LiveArchetypeBelief,
    examples: Sequence[RevealExample],
    archetypes: Sequence[str],
    *,
    use_evidence: bool = True,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot evaluate an empty reveal sample")
    classes = tuple(archetypes)
    nll = 0.0
    brier = 0.0
    top1 = 0
    top3 = 0
    entropy = 0.0
    locked = 0
    wrong_locked = 0
    calibration: list[tuple[float, int]] = []
    true_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    started = time.perf_counter()
    for example in examples:
        reveals = list(example.revealed_card_ids) if use_evidence else []
        result = model.update(reveals, top_k=len(classes))
        probabilities = {hypothesis.archetype: hypothesis.probability for hypothesis in result.hypotheses}
        true_probability = max(probabilities.get(example.archetype, 0.0), 1e-12)
        nll -= math.log(true_probability)
        for archetype in classes:
            target = 1.0 if archetype == example.archetype else 0.0
            brier += (probabilities.get(archetype, 0.0) - target) ** 2
        ordered = [hypothesis.archetype for hypothesis in result.hypotheses]
        predicted = ordered[0]
        is_correct = int(predicted == example.archetype)
        top1 += is_correct
        top3 += int(example.archetype in ordered[:3])
        entropy += result.entropy
        calibration.append((result.confidence, is_correct))
        true_counts[example.archetype] += 1
        predicted_counts[predicted] += 1
        true_positive[predicted] += is_correct
        if result.confidence >= CONFIDENCE_GATE:
            locked += 1
            wrong_locked += 1 - is_correct
    elapsed = time.perf_counter() - started
    count = len(examples)

    precisions = []
    recalls = []
    f1s = []
    for archetype in classes:
        precision = true_positive[archetype] / predicted_counts[archetype] if predicted_counts[archetype] else 0.0
        recall = true_positive[archetype] / true_counts[archetype] if true_counts[archetype] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "samples": count,
        "nll": round(nll / count, 6),
        "brier": round(brier / count, 6),
        "top1_accuracy": round(top1 / count, 6),
        "top3_accuracy": round(top3 / count, 6),
        "macro_precision": round(sum(precisions) / len(classes), 6),
        "macro_recall": round(sum(recalls) / len(classes), 6),
        "macro_f1": round(sum(f1s) / len(classes), 6),
        "top_label_ece": round(_expected_calibration_error(calibration), 6),
        "mean_normalized_entropy": round(entropy / count, 6),
        "confidence_gate": CONFIDENCE_GATE,
        "gate_coverage": round(locked / count, 6),
        "gate_accuracy": round((locked - wrong_locked) / locked, 6) if locked else None,
        "high_confidence_wrong_lock_rate": round(wrong_locked / locked, 6) if locked else None,
        "inference_ms_per_sample": round(1000.0 * elapsed / count, 6),
    }


def _expected_calibration_error(calibration: Sequence[tuple[float, int]], bins: int = 10) -> float:
    total = len(calibration)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [row for row in calibration if lower <= row[0] < upper or (bin_index == bins - 1 and row[0] == 1.0)]
        if not members:
            continue
        confidence = sum(row[0] for row in members) / len(members)
        accuracy = sum(row[1] for row in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)

    return error


def _reveal_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"

    return "8+"


def evaluate_belief_variants(
    statistics: BeliefStatistics,
    dataset: RevealDataset,
    *,
    calibration_scales: Sequence[float] = DEFAULT_CALIBRATION_SCALES,
) -> dict[str, Any]:
    eval_examples = [example for example in dataset.finals if example.split == "eval"]
    holdout_examples = [example for example in dataset.finals if example.split == "holdout"]
    if not eval_examples or not holdout_examples:
        raise ValueError("both eval and holdout reveal examples are required")

    scale_trials: dict[str, dict[str, Any]] = {}
    for scale in calibration_scales:
        model = _build_model(statistics, use_staples=True, evidence_scale=float(scale))
        scale_trials[str(scale)] = evaluate_examples(model, eval_examples, statistics.archetypes)
    selected_scale = min(
        (float(scale) for scale in calibration_scales),
        key=lambda scale: (scale_trials[str(scale)]["nll"], abs(scale - 1.0), scale),
    )

    variants = {
        "B0_prior_only": (_build_model(statistics), False),
        "B1_uniform_prior_plus_reveals": (_build_model(statistics, uniform_prior=True, use_staples=False), True),
        "B2_prior_plus_reveals_no_staple": (_build_model(statistics, use_staples=False), True),
        "B3_prior_plus_reveals_staple": (_build_model(statistics, use_staples=True), True),
        "B4_calibrated_staple": (
            _build_model(statistics, use_staples=True, evidence_scale=selected_scale),
            True,
        ),
    }
    holdout_metrics = {
        name: evaluate_examples(model, holdout_examples, statistics.archetypes, use_evidence=use_evidence)
        for name, (model, use_evidence) in variants.items()
    }

    calibrated_model = variants["B4_calibrated_staple"][0]
    prefix_slices: dict[str, dict[str, Any]] = {}
    for bucket in ("1", "2-3", "4-7", "8+"):
        examples = [
            example for example in dataset.prefixes
            if example.split == "holdout" and _reveal_bucket(len(example.revealed_card_ids)) == bucket
        ]
        if examples:
            prefix_slices[bucket] = evaluate_examples(calibrated_model, examples, statistics.archetypes)

    baseline = holdout_metrics["B0_prior_only"]
    calibrated = holdout_metrics["B4_calibrated_staple"]
    contribution = {
        "nll_delta_vs_prior": round(calibrated["nll"] - baseline["nll"], 6),
        "brier_delta_vs_prior": round(calibrated["brier"] - baseline["brier"], 6),
        "top1_delta_vs_prior": round(calibrated["top1_accuracy"] - baseline["top1_accuracy"], 6),
        "top3_delta_vs_prior": round(calibrated["top3_accuracy"] - baseline["top3_accuracy"], 6),
        "nll_delta_staple_vs_no_staple": round(
            holdout_metrics["B3_prior_plus_reveals_staple"]["nll"]
            - holdout_metrics["B2_prior_plus_reveals_no_staple"]["nll"],
            6,
        ),
        "nll_delta_calibrated_vs_uncalibrated": round(
            calibrated["nll"] - holdout_metrics["B3_prior_plus_reveals_staple"]["nll"],
            6,
        ),
    }
    improves_probability_quality = contribution["nll_delta_vs_prior"] < 0 and contribution["brier_delta_vs_prior"] < 0
    classification = "diagnostic only" if improves_probability_quality else "quarantine"

    return {
        "selected_evidence_scale": selected_scale,
        "eval_scale_trials": scale_trials,
        "holdout_metrics": holdout_metrics,
        "holdout_prefix_slices": prefix_slices,
        "marginal_contribution": contribution,
        "feature_classification": classification,
        "classification_reason": (
            "strict-public holdout probability quality improved, but no equal-budget gameplay benefit is measured"
            if classification == "diagnostic only"
            else "strict-public holdout probability quality did not improve over the prior-only baseline"
        ),
    }


def mapping_fingerprint(labels: dict[str, dict[str, Any]], splits: dict[str, str]) -> str:
    rows = [
        (deck_hash, labels[deck_hash].get("archetype"), labels[deck_hash].get("classification_status"), splits.get(deck_hash))
        for deck_hash in sorted(set(labels).intersection(splits))
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def render_report(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    contribution = evaluation["marginal_contribution"]
    lines = [
        "# Strict-public DSU belief baseline",
        "",
        f"Verdict: **{evaluation['feature_classification']}**.",
        "",
        "The revealed-card posterior was evaluated offline only. It is not wired into V, Q, policy, runtime, or submissions.",
        "",
        "## Boundary and split",
        "",
        f"- Features: `{PUBLIC_FEATURE_SCHEMA}`; opponent active, bench, discard, and allowlisted nested public attachments only.",
        "- Excluded: hands, decks/order, prizes, looking/search, legal choices, logs, identities, outcomes, and future state.",
        f"- Train unique decks: {result['training']['deck_count']} across {result['training']['archetype_count']} supported archetypes.",
        f"- Final eval/holdout episode-seats: {result['dataset']['eval_final_samples']} / {result['dataset']['holdout_final_samples']}.",
        "",
        "## Holdout ablation",
        "",
        "| Variant | NLL | Brier | Top-1 | Top-3 | ECE | Wrong lock |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in evaluation["holdout_metrics"].items():
        wrong_lock = metrics["high_confidence_wrong_lock_rate"]
        wrong_text = "n/a" if wrong_lock is None else f"{wrong_lock:.4f}"
        lines.append(
            f"| {name} | {metrics['nll']:.4f} | {metrics['brier']:.4f} | "
            f"{metrics['top1_accuracy']:.4f} | {metrics['top3_accuracy']:.4f} | "
            f"{metrics['top_label_ece']:.4f} | {wrong_text} |"
        )
    lines.extend([
        "",
        f"Eval-selected evidence scale: `{evaluation['selected_evidence_scale']}`.",
        "",
        "## Marginal contribution",
        "",
        f"- NLL delta versus prior-only: {contribution['nll_delta_vs_prior']:+.6f}.",
        f"- Brier delta versus prior-only: {contribution['brier_delta_vs_prior']:+.6f}.",
        f"- Top-1 delta versus prior-only: {contribution['top1_delta_vs_prior']:+.6f}.",
        f"- Staple attenuation NLL delta versus no-staple: {contribution['nll_delta_staple_vs_no_staple']:+.6f}.",
        "",
        f"Classification reason: {evaluation['classification_reason']}.",
        "",
    ])

    return "\n".join(lines)


def run_belief_evaluation(
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
        max_games=max_games,
    )
    evaluation = evaluate_belief_variants(statistics, dataset)
    result = {
        "schema_version": SCHEMA_VERSION,
        "information_mode": "public_belief",
        "feature_schema": PUBLIC_FEATURE_SCHEMA,
        "training": {
            "deck_count": statistics.train_deck_count,
            "archetype_count": len(statistics.archetypes),
            "decks_by_archetype": statistics.train_decks_by_archetype,
            "likelihood_cells": len(statistics.card_likelihoods),
            "staple_cards": len(statistics.staple_scores),
        },
        "dataset": {
            "rows_scanned": dataset.rows_scanned,
            "unique_episodes": dataset.unique_episodes,
            "labeled_episode_seats": dataset.labeled_episode_seats,
            "prefix_samples": len(dataset.prefixes),
            "eval_final_samples": sum(example.split == "eval" for example in dataset.finals),
            "holdout_final_samples": sum(example.split == "holdout" for example in dataset.finals),
            "max_games": max_games,
        },
        "input_identity": {
            "meta_kg_dir": str(Path(meta_kg_dir)),
            "deck_zoo_dir": str(Path(deck_zoo_dir)),
            "ingest_root": str(Path(ingest_root)),
            "opponent_decks_root": str(Path(opponent_decks_root)),
            "label_split_sha256": mapping_fingerprint(labels, splits),
            "deck_hash_conflicts": len(conflicts),
        },
        "evaluation": evaluation,
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    write_text(output_dir / "REPORT.md", render_report(result))

    return result
