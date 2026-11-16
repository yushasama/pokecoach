"""Offline full-list classifier for the diagnostic strategy ontology."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepmew.artifacts.writers import write_json, write_text
from deepmew.deck_profile_inference import DeckList, infer_profile, load_v2_cards


SCHEMA_VERSION = "deepmew.strategy_full_list_classifier.v1"
MANIFEST_VERSION = "deepmew.run_manifest.v1"
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_SEEDS = (915001, 915002, 915003, 915004, 915005)

MIN_GATE_DECKS = 15
MIN_GATE_CLUSTERS = 8
MIN_NONREGRESSED_SLOT_SHARE = 0.50
MAX_GROUP_MEAN_BRIER_REGRESSION = 0.005
MAX_SEED_DISAGREEMENT = 0.10

SEMANTIC_LIST_FIELDS = (
    "main_attackers",
    "setup_attackers",
    "finishers",
    "draw_engine",
    "search_engine",
    "damage_counter_plan",
    "ability_damage_counter_plan",
    "switch_gust_plan",
    "lock_control_plan",
    "healing_recovery_plan",
    "disruption_plan",
    "status_plan",
)
BENCH_PRESSURE_FIELDS = (
    "ordinary_bench_attack_damage",
    "bench_damage_counters",
    "ability_damage_counters",
)
PROTECTION_FIELDS = (
    "counter_placement_prevention",
    "attack_bench_protection",
    "self_bench_protection",
    "ability_prevention",
    "damage_wall",
)
DEPENDENCY_FIELDS = (
    "stadium_dependency",
    "tool_dependency",
    "special_energy_dependency",
    "stage2_dependency",
    "stage1_dependency",
    "ability_engine_dependency",
    "bench_fragility",
    "item_reliance",
    "hand_size_plan",
    "damage_counter_plan",
    "bench_attack_damage_plan",
    "self_ko_liability",
)
CHOKE_POINT_FIELDS = (
    "stage2_evolution_setup_reliance",
    "ability_engine_reliance",
    "special_energy_reliance",
    "high_item_reliance",
    "self_ko_liability",
    "hand_size_dependent",
    "bench_setup_fragility",
    "stadium_reliance",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_fold(value: str, fold_count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big") % fold_count


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

    return rows


def _load_zoo_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            deck_hash = str(row["deck_hash"])
            if deck_hash in rows:
                raise ValueError(f"duplicate deck-zoo hash {deck_hash}")
            rows[deck_hash] = {
                "split": str(row["split"]),
                "cluster_id": str(row["cluster_id"]),
                "archetype": str(row["archetype"]),
                "is_canonical": str(row["is_canonical"]).casefold() == "true",
            }

    return rows


def _load_complete_lists(
    root: str | Path,
    wanted_hashes: set[str],
) -> dict[str, dict[str, int]]:
    lists: dict[str, dict[str, int]] = {}
    for path in Path(root).rglob("deck_summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        deck_hash = str(row.get("deck_hash") or "")
        if deck_hash not in wanted_hashes:
            continue
        counts = {str(key): int(value) for key, value in (row.get("card_counts") or {}).items()}
        if sum(counts.values()) != 60:
            raise ValueError(f"source deck is not 60 cards: {path}")
        prior = lists.setdefault(deck_hash, counts)
        if prior != counts:
            raise ValueError(f"deck hash maps to multiple full lists: {deck_hash}")
    missing = sorted(wanted_hashes - set(lists))
    if missing:
        raise ValueError(f"missing complete lists for {len(missing)} supported decks: {missing[:5]}")

    return lists


def _ref_strength(items: Iterable[Any]) -> float:
    return sum(float(item.count) for item in items) / 4.0


def _semantic_feature_names() -> list[str]:
    names = [f"copies_{field}" for field in SEMANTIC_LIST_FIELDS]
    names += [f"copies_bench_{field}" for field in BENCH_PRESSURE_FIELDS]
    names += [f"copies_protection_{field}" for field in PROTECTION_FIELDS]
    names += [
        "energy_basic_count",
        "energy_special_copies",
        "energy_acceleration_copies",
        "energy_recovery_copies",
        "energy_reposition_copies",
        "evolution_basic_copies",
        "evolution_stage1_copies",
        "evolution_stage2_copies",
        "evolution_rare_candy_copies",
        "evolution_search_copies",
        "max_stage_basic",
        "max_stage_stage1",
        "max_stage_stage2",
    ]
    names += [f"dependency_{field}" for field in DEPENDENCY_FIELDS]
    names += [f"choke_{field}" for field in CHOKE_POINT_FIELDS]
    names += [
        "resolved_card_share",
        "flagged_review_share",
        "profile_confidence_low",
        "profile_confidence_medium",
        "profile_confidence_high",
    ]

    return names


def _semantic_feature_vector(profile: Any) -> list[float]:
    values = [_ref_strength(getattr(profile, field)) for field in SEMANTIC_LIST_FIELDS]
    values += [
        _ref_strength(profile.bench_pressure.get(field, []))
        for field in BENCH_PRESSURE_FIELDS
    ]
    values += [
        _ref_strength(profile.protection_plan.get(field, []))
        for field in PROTECTION_FIELDS
    ]
    energy = profile.energy_plan
    evolution = profile.evolution_plan
    values += [
        float(energy.get("basic_energy_count", 0)) / 4.0,
        _ref_strength(energy.get("special_energy", [])),
        _ref_strength(energy.get("acceleration", [])),
        _ref_strength(energy.get("recovery", [])),
        _ref_strength(energy.get("reposition", [])),
        float(evolution.get("basic_pokemon_count", 0)) / 4.0,
        float(evolution.get("stage1_pokemon_count", 0)) / 4.0,
        float(evolution.get("stage2_pokemon_count", 0)) / 4.0,
        _ref_strength(evolution.get("rare_candy", [])),
        _ref_strength(evolution.get("evolution_search", [])),
        float(evolution.get("max_stage") == "basic"),
        float(evolution.get("max_stage") == "stage1"),
        float(evolution.get("max_stage") == "stage2"),
    ]
    values += [float(profile.dependencies.get(field, 0)) / 4.0 for field in DEPENDENCY_FIELDS]
    values += [float(field in profile.choke_points) for field in CHOKE_POINT_FIELDS]
    values += [
        float(profile.resolved_card_share),
        float(profile.flagged_review_share),
        float(profile.profile_confidence == "low"),
        float(profile.profile_confidence == "medium"),
        float(profile.profile_confidence == "high"),
    ]
    if len(values) != len(_semantic_feature_names()):
        raise AssertionError("semantic deck feature layout mismatch")

    return values


def _encode_feature_matrix(
    *,
    ordered_profiles: Sequence[Mapping[str, Any]],
    complete_lists: Mapping[str, Mapping[str, int]],
    feature_mode: str,
    v2_cards_path: str | Path | None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if feature_mode == "card_counts":
        card_vocab = sorted(
            {card_id for counts in complete_lists.values() for card_id in counts},
            key=int,
        )
        card_index = {card_id: index for index, card_id in enumerate(card_vocab)}
        x = np.zeros((len(ordered_profiles), len(card_vocab)), dtype=np.float64)
        for row_index, row in enumerate(ordered_profiles):
            for card_id, count in complete_lists[str(row["deck_hash"])].items():
                x[row_index, card_index[card_id]] = count / 4.0

        return x, [f"card_count_{value}" for value in card_vocab], {
            "kind": "complete_60_card_count_vector",
            "v2_cards_path": None,
        }
    if feature_mode != "semantic_profile":
        raise ValueError(f"unsupported feature mode: {feature_mode}")
    if v2_cards_path is None:
        raise ValueError("semantic_profile mode requires a v2 card-function artifact")
    v2map = load_v2_cards(v2_cards_path)
    x_rows = []
    for row in ordered_profiles:
        deck_hash = str(row["deck_hash"])
        deck = DeckList(
            deck_hash=deck_hash,
            archetype="not_used",
            split="not_used",
            card_counts=dict(complete_lists[deck_hash]),
        )
        inferred = infer_profile(deck, v2map, deck_id=deck_hash)
        x_rows.append(_semantic_feature_vector(inferred))

    return np.asarray(x_rows, dtype=np.float64), _semantic_feature_names(), {
        "kind": "generalized_card_function_deck_profile",
        "v2_cards_path": str(Path(v2_cards_path)),
        "v2_cards_sha256": _sha256(v2_cards_path),
    }


def _fresh_cluster_splits(
    clusters: Sequence[str],
    salt: str,
) -> tuple[list[str], dict[str, str]]:
    unique = sorted(
        set(clusters),
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest(),
    )
    eval_count = max(MIN_GATE_CLUSTERS, int(round(0.15 * len(unique))))
    holdout_count = max(MIN_GATE_CLUSTERS, int(round(0.15 * len(unique))))
    if eval_count + holdout_count >= len(unique):
        raise ValueError("not enough clusters for fresh train/eval/holdout split")
    train_count = len(unique) - eval_count - holdout_count
    assignment = {}
    for index, cluster in enumerate(unique):
        if index < train_count:
            assignment[cluster] = "train"
        elif index < train_count + eval_count:
            assignment[cluster] = "eval"
        else:
            assignment[cluster] = "holdout"

    return [assignment[value] for value in clusters], assignment


def _cluster_weights(clusters: Sequence[str]) -> np.ndarray:
    counts = Counter(clusters)

    return np.asarray([1.0 / counts[value] for value in clusters], dtype=np.float64)


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, y, sample_weight=_cluster_weights(clusters))

    return model


def _clip_predictions(model: Ridge, x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(model.predict(x), dtype=np.float64), 0.0, 1.0)


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    alphas: Sequence[float],
    fold_count: int = 5,
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    fold_by_cluster = {cluster: _hash_fold(cluster, fold_count) for cluster in set(clusters)}
    alpha_rows = []
    oof_by_alpha = {}
    for alpha in alphas:
        oof = np.zeros_like(y, dtype=np.float64)
        seen = np.zeros(len(y), dtype=np.bool_)
        fold_scores = []
        for fold in range(fold_count):
            valid = np.asarray([fold_by_cluster[value] == fold for value in clusters])
            train = ~valid
            if not valid.any() or not train.any():
                continue
            model = _fit_ridge(x[train], y[train], np.asarray(clusters)[train], alpha)
            oof[valid] = _clip_predictions(model, x[valid])
            seen[valid] = True
            fold_scores.append(float(np.mean((oof[valid] - y[valid]) ** 2)))
        if not seen.all():
            raise ValueError("grouped CV did not assign every training deck")
        mean_brier = float(np.mean((oof - y) ** 2))
        alpha_rows.append({
            "alpha": float(alpha),
            "mean_brier": round(mean_brier, 9),
            "fold_brier": [round(value, 9) for value in fold_scores],
        })
        oof_by_alpha[float(alpha)] = oof
    selected = min(alpha_rows, key=lambda row: (row["mean_brier"], row["alpha"]))

    return float(selected["alpha"]), oof_by_alpha[float(selected["alpha"])], alpha_rows


def _fit_calibrators(
    raw_oof: np.ndarray,
    y: np.ndarray,
) -> list[IsotonicRegression | None]:
    calibrators = []
    for index in range(y.shape[1]):
        if len(np.unique(raw_oof[:, index])) < 2 or len(np.unique(y[:, index])) < 2:
            calibrators.append(None)
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_oof[:, index], y[:, index])
        calibrators.append(calibrator)

    return calibrators


def _apply_calibrators(
    raw: np.ndarray,
    calibrators: Sequence[IsotonicRegression | None],
    fallback: np.ndarray,
) -> np.ndarray:
    calibrated = np.empty_like(raw, dtype=np.float64)
    for index, calibrator in enumerate(calibrators):
        if calibrator is None:
            calibrated[:, index] = fallback[index]
        else:
            calibrated[:, index] = calibrator.predict(raw[:, index])

    return np.clip(calibrated, 0.0, 1.0)


def _ece(y: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (prediction >= lower) & (
            prediction <= upper if index == bins - 1 else prediction < upper
        )
        if mask.any():
            error += (mask.sum() / total) * abs(float(prediction[mask].mean() - y[mask].mean()))

    return error


def _binary_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    binary = y > threshold
    if len(np.unique(binary)) < 2:
        return {"auroc": None, "average_precision": None, "f1": None}
    predicted_binary = prediction > threshold

    return {
        "auroc": round(float(roc_auc_score(binary, prediction)), 9),
        "average_precision": round(float(average_precision_score(binary, prediction)), 9),
        "f1": round(float(f1_score(binary, predicted_binary, zero_division=0)), 9),
    }


def evaluate_predictions(
    *,
    y: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    slots: Sequence[Mapping[str, Any]],
    target_thresholds: np.ndarray,
    deck_count: int,
    cluster_count: int,
    seed_disagreement: np.ndarray,
) -> dict[str, Any]:
    per_slot = []
    baseline_matrix = (
        np.repeat(baseline[None, :], len(y), axis=0)
        if baseline.ndim == 1
        else baseline
    )
    if baseline_matrix.shape != y.shape:
        raise ValueError("baseline predictions must match target shape")
    for index, slot in enumerate(slots):
        brier = float(np.mean((prediction[:, index] - y[:, index]) ** 2))
        baseline_brier = float(np.mean((baseline_matrix[:, index] - y[:, index]) ** 2))
        row = {
            "slot_id": slot["slot_id"],
            "group": slot["group"],
            "brier": round(brier, 9),
            "baseline_brier": round(baseline_brier, 9),
            "brier_delta_vs_constant": round(brier - baseline_brier, 9),
            "ece": round(_ece(y[:, index], prediction[:, index]), 9),
            "target_threshold": round(float(target_thresholds[index]), 9),
            "seed_prediction_std": round(float(seed_disagreement[index]), 9),
        }
        row.update(_binary_metrics(
            y[:, index],
            prediction[:, index],
            float(target_thresholds[index]),
        ))
        per_slot.append(row)
    groups = {}
    for group in ("win", "style", "loss"):
        members = [row for row in per_slot if row["group"] == group]
        groups[group] = {
            "mean_brier": round(float(np.mean([row["brier"] for row in members])), 9),
            "mean_baseline_brier": round(float(np.mean([row["baseline_brier"] for row in members])), 9),
            "mean_brier_delta_vs_constant": round(float(np.mean([row["brier_delta_vs_constant"] for row in members])), 9),
            "regressed_slots": sum(row["brier_delta_vs_constant"] > 0.0 for row in members),
            "nonregressed_slot_share": round(
                sum(row["brier_delta_vs_constant"] <= 0.0 for row in members) / len(members),
                9,
            ),
            "slots": len(members),
        }
    mean_brier = float(np.mean((prediction - y) ** 2))
    mean_baseline = float(np.mean((baseline_matrix - y) ** 2))
    nonregressed = sum(row["brier_delta_vs_constant"] <= 0.0 for row in per_slot)

    return {
        "deck_count": deck_count,
        "cluster_count": cluster_count,
        "mean_brier": round(mean_brier, 9),
        "mean_baseline_brier": round(mean_baseline, 9),
        "mean_brier_delta_vs_constant": round(mean_brier - mean_baseline, 9),
        "nonregressed_slot_share": round(nonregressed / len(per_slot), 9),
        "mean_seed_prediction_std": round(float(np.mean(seed_disagreement)), 9),
        "groups": groups,
        "per_slot": per_slot,
    }


def classifier_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "minimum_decks": int(metrics["deck_count"]) >= MIN_GATE_DECKS,
        "minimum_clusters": int(metrics["cluster_count"]) >= MIN_GATE_CLUSTERS,
        "aggregate_nonregression": float(metrics["mean_brier_delta_vs_constant"]) <= 0.0,
        "slot_nonregression_breadth": float(metrics["nonregressed_slot_share"]) >= MIN_NONREGRESSED_SLOT_SHARE,
        "group_nonregression": all(
            float(group["mean_brier_delta_vs_constant"]) <= MAX_GROUP_MEAN_BRIER_REGRESSION
            for group in metrics["groups"].values()
        ),
        "group_slot_nonregression_breadth": all(
            float(group["nonregressed_slot_share"]) >= MIN_NONREGRESSED_SLOT_SHARE
            for group in metrics["groups"].values()
        ),
        "seed_stability": float(metrics["mean_seed_prediction_std"]) <= MAX_SEED_DISAGREEMENT,
    }

    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "thresholds": {
            "minimum_decks": MIN_GATE_DECKS,
            "minimum_clusters": MIN_GATE_CLUSTERS,
            "minimum_nonregressed_slot_share": MIN_NONREGRESSED_SLOT_SHARE,
            "minimum_nonregressed_slot_share_per_group": MIN_NONREGRESSED_SLOT_SHARE,
            "maximum_group_mean_brier_regression": MAX_GROUP_MEAN_BRIER_REGRESSION,
            "maximum_mean_seed_prediction_std": MAX_SEED_DISAGREEMENT,
        },
    }


def _serialize_calibrator(calibrator: IsotonicRegression | None) -> dict[str, Any]:
    if calibrator is None:
        return {"kind": "constant_train_mean"}

    return {
        "kind": "grouped_oof_isotonic",
        "x_thresholds": [round(float(value), 12) for value in calibrator.X_thresholds_],
        "y_thresholds": [round(float(value), 12) for value in calibrator.y_thresholds_],
    }


def _predict_seed_models(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_clusters: np.ndarray,
    x_target: np.ndarray,
    alpha: float,
    calibrators: Sequence[IsotonicRegression | None],
    fallback: np.ndarray,
    seeds: Sequence[int],
) -> np.ndarray:
    predictions = []
    unique_clusters = np.asarray(sorted(set(train_clusters)))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep_count = max(2, int(round(0.8 * len(unique_clusters))))
        kept = set(rng.choice(unique_clusters, size=keep_count, replace=False).tolist())
        mask = np.asarray([cluster in kept for cluster in train_clusters])
        model = _fit_ridge(x_train[mask], y_train[mask], train_clusters[mask], alpha)
        raw = _clip_predictions(model, x_target)
        predictions.append(_apply_calibrators(raw, calibrators, fallback))

    return np.asarray(predictions, dtype=np.float64)


def _heldout_archetype_cv(
    *,
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    archetypes: np.ndarray,
    alpha: float,
    slots: Sequence[Mapping[str, Any]],
    target_thresholds: np.ndarray,
    seeds: Sequence[int],
    fold_count: int = 5,
) -> dict[str, Any]:
    fold_by_archetype = {
        archetype: _hash_fold(f"archetype:{archetype}", fold_count)
        for archetype in set(archetypes)
    }
    prediction = np.zeros_like(y, dtype=np.float64)
    baseline = np.zeros_like(y, dtype=np.float64)
    seed_std = np.zeros_like(y, dtype=np.float64)
    seen = np.zeros(len(y), dtype=np.bool_)
    fold_rows = []
    for fold in range(fold_count):
        target = np.asarray([fold_by_archetype[value] == fold for value in archetypes])
        train = ~target
        if not target.any() or not train.any():
            continue
        _, inner_oof, _ = _select_alpha(
            x[train],
            y[train],
            clusters[train],
            (alpha,),
        )
        fold_mean = y[train].mean(axis=0)
        calibrators = _fit_calibrators(inner_oof, y[train])
        model = _fit_ridge(x[train], y[train], clusters[train], alpha)
        raw = _clip_predictions(model, x[target])
        prediction[target] = _apply_calibrators(raw, calibrators, fold_mean)
        baseline[target] = fold_mean
        seed_predictions = _predict_seed_models(
            x_train=x[train],
            y_train=y[train],
            train_clusters=clusters[train],
            x_target=x[target],
            alpha=alpha,
            calibrators=calibrators,
            fallback=fold_mean,
            seeds=seeds,
        )
        seed_std[target] = seed_predictions.std(axis=0)
        seen[target] = True
        fold_rows.append({
            "fold": fold,
            "held_out_archetypes": sorted(set(archetypes[target])),
            "decks": int(target.sum()),
            "clusters": len(set(clusters[target])),
        })
    if not seen.all():
        raise ValueError("held-out archetype CV did not cover every training deck")
    metrics = evaluate_predictions(
        y=y,
        prediction=prediction,
        baseline=baseline,
        slots=slots,
        target_thresholds=target_thresholds,
        deck_count=len(y),
        cluster_count=len(set(clusters)),
        seed_disagreement=seed_std.mean(axis=0),
    )
    metrics["folds"] = fold_rows

    return metrics


def render_report(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# Full-list strategy classifier v1",
        "",
        f"Verdict: **{result['verdict']}**.",
        "",
        "The predictor uses only complete 60-card count vectors. Deck hash, archetype, policy identity, outcome, and private state are evaluation/provenance fields, never features.",
        "",
        f"Grouped train selection chose ridge alpha `{result['model']['selected_alpha']}` over {result['model']['training_decks']} decks / {result['model']['training_clusters']} near-duplicate clusters, with grouped out-of-fold isotonic calibration.",
        "",
        "| Split | Decks | Clusters | Mean Brier | Constant | Delta | Non-regressed slots | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name in ("eval", "holdout", "held_out_archetype_cv"):
        row = evaluation.get(split_name)
        if row is None:
            lines.append(f"| {split_name} | - | - | - | - | - | - | not opened |")
            continue
        gate = evaluation[f"{split_name}_gate"]
        lines.append(
            f"| {split_name} | {row['deck_count']} | {row['cluster_count']} | "
            f"{row['mean_brier']:.6f} | {row['mean_baseline_brier']:.6f} | "
            f"{row['mean_brier_delta_vs_constant']:+.6f} | "
            f"{row['nonregressed_slot_share']:.1%} | {'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Feature contribution verdict",
        "",
        f"Classification: `{result['feature_contribution']['classification']}`. {result['feature_contribution']['rationale']}",
        "",
        "## Boundary",
        "",
        "- This evaluates a deck-profile diagnostic, not a policy-deconfounded style model.",
        "- Public-belief inference remains gated on an action-conditioned strict-public posterior and independent prefix evaluation.",
        "- No strategy tensor is authorized for V, response, Q, league, runtime, or submissions unless eval, holdout, and held-out-archetype gates all pass.",
        "",
    ]

    return "\n".join(lines)


def run_full_list_classifier(
    *,
    profiles_path: str | Path,
    ontology_path: str | Path,
    deck_zoo_path: str | Path,
    complete_list_root: str | Path,
    output_dir: str | Path,
    feature_mode: str = "card_counts",
    v2_cards_path: str | Path | None = None,
    fresh_split_salt: str | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    profiles = [
        row
        for row in _load_jsonl(profiles_path)
        if row["support"]["status"] == "supported_diagnostic"
    ]
    ontology = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    slots = ontology["slots"]
    zoo = _load_zoo_metadata(deck_zoo_path)
    wanted_hashes = {str(row["deck_hash"]) for row in profiles}
    if not wanted_hashes <= set(zoo):
        raise ValueError("supported strategy profiles are missing deck-zoo grouping metadata")
    complete_lists = _load_complete_lists(complete_list_root, wanted_hashes)
    ordered = sorted(profiles, key=lambda row: row["deck_hash"])
    x, feature_names, feature_metadata = _encode_feature_matrix(
        ordered_profiles=ordered,
        complete_lists=complete_lists,
        feature_mode=feature_mode,
        v2_cards_path=v2_cards_path,
    )
    y = np.asarray([row["semantic_tensor"] for row in ordered], dtype=np.float64)
    source_splits = []
    clusters = []
    archetypes = []
    for row in ordered:
        deck_hash = str(row["deck_hash"])
        source_splits.append(zoo[deck_hash]["split"])
        clusters.append(zoo[deck_hash]["cluster_id"])
        archetypes.append(zoo[deck_hash]["archetype"])
    if fresh_split_salt:
        splits, split_assignment = _fresh_cluster_splits(clusters, fresh_split_salt)
        split_source = "fresh_hash_balanced_near_duplicate_clusters"
    else:
        splits = source_splits
        split_assignment = None
        split_source = str(Path(deck_zoo_path))
    splits_array = np.asarray(splits)
    clusters_array = np.asarray(clusters)
    train = splits_array == "train"
    eval_mask = splits_array == "eval"
    holdout = splits_array == "holdout"
    if not train.any() or not eval_mask.any() or not holdout.any():
        raise ValueError("classifier requires train, eval, and holdout strategy profiles")
    selected_alpha, raw_oof, alpha_selection = _select_alpha(
        x[train], y[train], clusters_array[train], alphas
    )
    train_mean = y[train].mean(axis=0)
    calibrators = _fit_calibrators(raw_oof, y[train])
    model = _fit_ridge(x[train], y[train], clusters_array[train], selected_alpha)
    target_thresholds = np.asarray([
        0.5 if slot["group"] == "win" else float(np.median(y[train, index]))
        for index, slot in enumerate(slots)
    ])

    def evaluate_mask(mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        raw = _clip_predictions(model, x[mask])
        prediction = _apply_calibrators(raw, calibrators, train_mean)
        seed_predictions = _predict_seed_models(
            x_train=x[train],
            y_train=y[train],
            train_clusters=clusters_array[train],
            x_target=x[mask],
            alpha=selected_alpha,
            calibrators=calibrators,
            fallback=train_mean,
            seeds=seeds,
        )
        seed_disagreement = seed_predictions.std(axis=0).mean(axis=0)
        metrics = evaluate_predictions(
            y=y[mask],
            prediction=prediction,
            baseline=train_mean,
            slots=slots,
            target_thresholds=target_thresholds,
            deck_count=int(mask.sum()),
            cluster_count=len(set(clusters_array[mask])),
            seed_disagreement=seed_disagreement,
        )

        return metrics, prediction

    eval_metrics, _ = evaluate_mask(eval_mask)
    eval_gate = classifier_gate(eval_metrics)
    archetype_metrics = _heldout_archetype_cv(
        x=x[train],
        y=y[train],
        clusters=clusters_array[train],
        archetypes=np.asarray(archetypes)[train],
        alpha=selected_alpha,
        slots=slots,
        target_thresholds=target_thresholds,
        seeds=seeds,
    )
    archetype_gate = classifier_gate(archetype_metrics)
    holdout_metrics = None
    holdout_gate = None
    if eval_gate["passed"]:
        holdout_metrics, _ = evaluate_mask(holdout)
        holdout_gate = classifier_gate(holdout_metrics)
    accepted = bool(
        eval_gate["passed"]
        and holdout_gate
        and holdout_gate["passed"]
        and archetype_gate["passed"]
    )
    verdict = "FULL_LIST_CLASSIFIER_ACCEPTED_DIAGNOSTIC" if accepted else "FULL_LIST_CLASSIFIER_REJECTED"
    regressed_slots = [
        row["slot_id"]
        for row in eval_metrics["per_slot"]
        if row["brier_delta_vs_constant"] > 0.0
    ]
    classification = "diagnostic" if accepted else "quarantine"
    rationale = (
        f"{feature_mode} features beat the equal-budget constant baseline across every gate."
        if accepted
        else "The equal-budget constant baseline was not beaten broadly enough; tensor inference stays out of control paths."
    )

    start = time.perf_counter()
    repeated = np.repeat(x[eval_mask][:1], 10000, axis=0)
    _clip_predictions(model, repeated)
    latency_ms = (time.perf_counter() - start) * 1000.0 / len(repeated)
    classifier_artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only" if accepted else "quarantined",
        "ontology_version": ontology["schema_version"],
        "ontology_sha256": _sha256(ontology_path),
        "feature_contract": feature_metadata,
        "feature_names": feature_names,
        "selected_alpha": selected_alpha,
        "slot_ids": [slot["slot_id"] for slot in slots],
        "coefficients": np.asarray(model.coef_).round(12).tolist(),
        "intercepts": np.asarray(model.intercept_).round(12).tolist(),
        "train_means": train_mean.round(12).tolist(),
        "target_thresholds": target_thresholds.round(12).tolist(),
        "calibrators": [_serialize_calibrator(value) for value in calibrators],
        "uncertainty": "five 80%-cluster subsample models plus feature-support diagnostics",
        "promotion_status": "diagnostic_only_no_control_authority",
    }
    result = {
        "schema_version": "deepmew.strategy_classifier_evaluation.v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "model": {
            "kind": "multi_output_ridge_grouped_oof_isotonic",
            "feature_mode": feature_mode,
            "selected_alpha": selected_alpha,
            "alpha_selection": alpha_selection,
            "training_decks": int(train.sum()),
            "training_clusters": len(set(clusters_array[train])),
            "feature_count": len(feature_names),
            "exact_deck_id_feature": False,
            "archetype_feature": False,
            "inference_latency_ms_per_deck": round(latency_ms, 9),
        },
        "split_contract": {
            "source": split_source,
            "fresh_split_salt": fresh_split_salt,
            "group": "near_duplicate_cluster_id",
            "counts": dict(Counter(splits)),
            "clusters": {
                split: len({clusters[index] for index, value in enumerate(splits) if value == split})
                for split in ("train", "eval", "holdout")
            },
            "cluster_assignments": split_assignment,
        },
        "evaluation": {
            "eval": eval_metrics,
            "eval_gate": eval_gate,
            "holdout": holdout_metrics,
            "holdout_gate": holdout_gate,
            "held_out_archetype_cv": archetype_metrics,
            "held_out_archetype_cv_gate": archetype_gate,
        },
        "feature_contribution": {
            "classification": classification,
            "baseline": "per-slot train mean under identical split and metric budget",
            "benefit": {"eval_mean_brier_delta": eval_metrics["mean_brier_delta_vs_constant"]},
            "damage": {"eval_regressed_slots": regressed_slots},
            "cost": {"features": len(feature_names), "latency_ms_per_deck": round(latency_ms, 9)},
            "uniqueness": f"{feature_metadata['kind']}; deck hash and archetype excluded",
            "rationale": rationale,
        },
        "public_belief_status": "not_evaluated_action_conditioned_posterior_missing",
        "promotion_status": "diagnostic_only_no_control_authority",
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "classifier.json"
    result_path = output_dir / "result.json"
    report_path = output_dir / "REPORT.md"
    write_json(artifact_path, classifier_artifact)
    write_json(result_path, result)
    write_text(report_path, render_report(result))
    outputs = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in (
            ("classifier", artifact_path),
            ("result", result_path),
            ("report", report_path),
        )
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "run_id": f"strategy-full-list-classifier-{feature_mode}-v1-20260711",
        "kind": "strategy_full_list_classifier",
        "status": "completed",
        "created_at_utc": result["created_at_utc"],
        "inputs": {
            "profiles_path": str(Path(profiles_path)),
            "profiles_sha256": _sha256(profiles_path),
            "ontology_path": str(Path(ontology_path)),
            "ontology_sha256": _sha256(ontology_path),
            "deck_zoo_path": str(Path(deck_zoo_path)),
            "deck_zoo_sha256": _sha256(deck_zoo_path),
            "feature_mode": feature_mode,
            "v2_cards_path": str(Path(v2_cards_path)) if v2_cards_path else None,
            "v2_cards_sha256": _sha256(v2_cards_path) if v2_cards_path else None,
            "fresh_split_salt": fresh_split_salt,
            "implementation_sha256": _sha256(Path(__file__)),
            "entrypoint_sha256": _sha256(Path(__file__).resolve().parents[2] / "scripts" / "run_strategy_full_list_classifier.py"),
        },
        "outputs": outputs,
        "summary": {
            "verdict": verdict,
            "classifier_sha256": outputs["classifier"]["sha256"],
            "ontology_version": ontology["schema_version"],
            "eval_gate_passed": eval_gate["passed"],
            "holdout_opened": holdout_metrics is not None,
            "holdout_gate_passed": bool(holdout_gate and holdout_gate["passed"]),
            "held_out_archetype_gate_passed": archetype_gate["passed"],
            "promotion_status": result["promotion_status"],
        },
        "protected_surfaces_changed": [],
    }
    write_json(output_dir / "manifest.json", manifest)

    return result
