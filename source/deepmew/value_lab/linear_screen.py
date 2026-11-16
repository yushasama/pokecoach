"""Bounded training-only linear feature-selection screen for snapshot value."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.value_lab.data import LabData
from deepmew.value_lab.feature_contract import feature_specs, representation_manifest
from deepmew.value_lab.feature_selection import (
    correlation_inventory,
    deterministic_sample,
    feature_moments,
)
from deepmew.value_lab.mc_screen import canonical_game_indices, direct_comparison
from deepmew.value_lab.metrics import brier, ece, full_metrics, log_loss
from deepmew.value_lab.temporal_screen import evaluate_temporal_predictions, temporal_validation_index


LINEAR_SCREEN_SCHEMA_VERSION = "deepmew.value_linear_feature_screen.v1"
STABILITY_FOLD_SCHEMA_VERSION = "deepmew.value_feature_stability_folds.v1"
STABILITY_FOLD_SALT = "deepmew.value_feature_stability_folds.v1"
DEFAULT_SEEDS = (915001, 915002, 915003)
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3)
ELASTIC_ALPHAS = (1e-5, 1e-4, 1e-3)
ELASTIC_L1_RATIOS = (0.25, 0.5, 0.75)
PCA_S1_DIMENSIONS = (8, 24, 40)
PCA_S2_DIMENSIONS = (16, 48, 64)
CORRELATION_PRUNE_THRESHOLD = 0.995
COEFFICIENT_SELECTION_THRESHOLD = 1e-4
COMPACT_NONINFERIORITY_MARGIN = 0.001
MEANINGFUL_FEATURE_REDUCTION = 0.20
SEMANTIC_DROP_MEAN_GAIN = 0.0001
SEMANTIC_DROP_MAX_DAMAGE = 0.0001


@dataclass(frozen=True)
class TransformPlan:
    name: str
    kind: str
    indices: tuple[int, ...]
    components: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None

    @property
    def dimensions(self) -> int:
        if self.kind == "raw":
            return len(self.indices)
        if self.components is None:
            raise ValueError("PCA plan lacks components")

        return int(self.components.shape[0])


@dataclass(frozen=True)
class LinearCandidate:
    name: str
    method: str
    plan: str
    penalty: str
    alpha: float
    l1_ratio: float | None = None
    dimensions: int | None = None


def _sha256(path: str | Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)

        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)

    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)

    return 1.0 / (1.0 + np.exp(-logits))


def source_stratified_game_folds(data: LabData, folds: int = 3) -> dict[str, Any]:
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("stability folds must be an integer of at least two")
    block_game, game_ids = canonical_game_indices(data)
    game_source = np.full(len(game_ids), -1, dtype=np.int32)
    game_split = np.full(len(game_ids), -1, dtype=np.int32)
    for block_id, game_index in enumerate(block_game):
        source = int(data.blocks["source_idx"][block_id])
        split = int(data.blocks["split_idx"][block_id])
        if game_source[game_index] not in (-1, source) or game_split[game_index] not in (-1, split):
            raise ValueError("canonical game source/split is inconsistent across seat blocks")
        game_source[game_index] = source
        game_split[game_index] = split
    train_index = data.split_names.index("train")
    fold_of_game = np.full(len(game_ids), -1, dtype=np.int8)
    receipts = []
    for source in np.unique(game_source[game_split == train_index]):
        selected = np.flatnonzero((game_source == source) & (game_split == train_index))
        ordered = sorted(
            (int(index) for index in selected),
            key=lambda index: (
                hashlib.sha256(
                    f"{STABILITY_FOLD_SALT}\0{data.source_ids[int(source)]}\0{game_ids[index]}".encode("utf-8")
                ).digest(),
                game_ids[index],
            ),
        )
        for rank, game_index in enumerate(ordered):
            fold_of_game[game_index] = rank % folds
        receipts.append({
            "source_id": data.source_ids[int(source)],
            "games": len(ordered),
            "games_by_fold": [sum(1 for index in ordered if int(fold_of_game[index]) == fold) for fold in range(folds)],
        })
    if np.any(fold_of_game[game_split == train_index] < 0):
        raise ValueError("some outer-train games lack a stability fold")
    fold_of_block = fold_of_game[block_game]
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    fold_of_row = np.repeat(fold_of_block, lengths)
    if len(fold_of_row) != len(data.meta["split_idx"]):
        raise ValueError("stability fold row projection does not cover the cache")
    assignment_payload = {
        "schema_version": STABILITY_FOLD_SCHEMA_VERSION,
        "salt": STABILITY_FOLD_SALT,
        "folds": folds,
        "assignment": {game_ids[index]: int(fold_of_game[index]) for index in np.flatnonzero(game_split == train_index)},
    }

    return {
        "block_game": block_game,
        "game_ids": game_ids,
        "fold_of_game": fold_of_game,
        "fold_of_block": fold_of_block,
        "fold_of_row": fold_of_row,
        "source_receipts": receipts,
        "assignment_sha256": _stable_hash(assignment_payload),
        "assignment": assignment_payload,
    }


def correlation_pruned_indices(
    correlation: np.ndarray,
    standard_deviation: np.ndarray,
    indices: Iterable[int],
    *,
    threshold: float = CORRELATION_PRUNE_THRESHOLD,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    selected: list[int] = []
    dropped: list[dict[str, Any]] = []
    for index in (int(value) for value in indices):
        if float(standard_deviation[index]) <= 0.0:
            dropped.append({"index": index, "representative": None, "reason": "zero_variance"})
            continue
        representative = next(
            (other for other in selected if abs(float(correlation[index, other])) >= threshold),
            None,
        )
        if representative is not None:
            dropped.append({
                "index": index,
                "representative": representative,
                "correlation": float(correlation[index, representative]),
                "reason": "correlation_threshold",
            })
            continue
        selected.append(index)
    if not selected:
        raise ValueError("correlation pruning removed every feature")

    return tuple(selected), dropped


def fit_pca_plan(
    standardized_training_sample: np.ndarray,
    indices: Iterable[int],
    *,
    maximum_dimensions: int,
    name: str,
) -> TransformPlan:
    indices = tuple(int(index) for index in indices)
    sample = np.asarray(standardized_training_sample[:, indices], dtype=np.float32)
    sample -= sample.mean(axis=0, keepdims=True)
    covariance = (sample.T @ sample) / max(1, len(sample) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    order = np.argsort(-eigenvalues, kind="mergesort")
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T
    positive = np.flatnonzero(eigenvalues > 1e-10)
    keep = min(maximum_dimensions, len(positive))
    if keep < 1:
        raise ValueError("PCA training surface has no positive-variance component")
    eigenvalues = eigenvalues[:keep]
    components = components[:keep]
    for row in range(keep):
        anchor = int(np.argmax(np.abs(components[row])))
        if components[row, anchor] < 0.0:
            components[row] *= -1.0

    return TransformPlan(
        name=name,
        kind="pca_whitened",
        indices=indices,
        components=np.asarray(components, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
    )


def _apply_plan(
    standardized: np.ndarray,
    plan: TransformPlan,
    dimensions: int | None = None,
) -> np.ndarray:
    selected = np.asarray(standardized[:, plan.indices], dtype=np.float64)
    if plan.kind == "raw":
        return selected
    if plan.kind != "pca_whitened" or plan.components is None or plan.eigenvalues is None:
        raise ValueError(f"unsupported transform plan {plan.kind}")
    take = plan.dimensions if dimensions is None else int(dimensions)
    if take < 1 or take > plan.dimensions:
        raise ValueError("PCA candidate dimensions are outside the fitted plan")

    return (selected @ plan.components[:take].T) / np.sqrt(plan.eigenvalues[:take])


def _expanded_soft_labels(features: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if np.any(~np.isin(outcomes, (0.0, 0.5, 1.0))):
        raise ValueError("linear screen requires actor outcomes in {0,0.5,1}")
    draws = outcomes == 0.5
    if not draws.any():
        return features, outcomes.astype(np.int8), np.ones(len(outcomes), dtype=np.float64)
    decided = ~draws
    expanded = np.concatenate((features[decided], features[draws], features[draws]), axis=0)
    labels = np.concatenate((outcomes[decided].astype(np.int8), np.zeros(int(draws.sum()), dtype=np.int8), np.ones(int(draws.sum()), dtype=np.int8)))
    weights = np.concatenate((np.ones(int(decided.sum())), np.full(int(draws.sum()), 0.5), np.full(int(draws.sum()), 0.5)))

    return expanded, labels, weights


def _new_classifier(candidate: LinearCandidate, seed: int):
    from sklearn.linear_model import SGDClassifier

    kwargs = {
        "loss": "log_loss",
        "penalty": candidate.penalty,
        "alpha": float(candidate.alpha),
        "fit_intercept": True,
        "max_iter": 1,
        "tol": None,
        "shuffle": False,
        "random_state": int(seed),
        "learning_rate": "constant",
        "eta0": 0.01,
        "average": True,
    }
    if candidate.penalty == "elasticnet":
        kwargs["l1_ratio"] = float(candidate.l1_ratio)

    return SGDClassifier(**kwargs)


def _standardization(
    data: LabData,
    rows: np.ndarray,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moments = feature_moments(data.X, rows, batch_rows=batch_rows)
    mean = np.asarray(moments["mean"], dtype=np.float64)
    standard_deviation = np.asarray(moments["std"], dtype=np.float64)
    scale = np.where(standard_deviation > 1e-8, standard_deviation, 1.0)

    return mean, scale, standard_deviation


def _train_candidates(
    data: LabData,
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    plans: dict[str, TransformPlan],
    candidates: list[LinearCandidate],
    *,
    batch_rows: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    classifiers = {candidate.name: _new_classifier(candidate, seed) for candidate in candidates}
    initialized: set[str] = set()
    updates = Counter()
    started = time.perf_counter()
    for start in range(0, len(rows), batch_rows):
        selected = rows[start:start + batch_rows]
        batch = np.asarray(data.X[selected], dtype=np.float64)
        standardized = (batch - mean) / scale
        outcomes = data.meta["z"][selected].astype(np.float64, copy=False)
        transformed: dict[tuple[str, int | None], np.ndarray] = {}
        for candidate in candidates:
            key = (candidate.plan, candidate.dimensions)
            features = transformed.get(key)
            if features is None:
                features = _apply_plan(standardized, plans[candidate.plan], candidate.dimensions)
                if not np.isfinite(features).all():
                    raise ValueError(f"non-finite transformed training features for {candidate.name}")
                transformed[key] = features
            expanded, labels, weights = _expanded_soft_labels(features, outcomes)
            classifier = classifiers[candidate.name]
            if candidate.name not in initialized:
                classifier.partial_fit(expanded, labels, classes=np.asarray([0, 1]), sample_weight=weights)
                initialized.add(candidate.name)
            else:
                classifier.partial_fit(expanded, labels, sample_weight=weights)
            updates[candidate.name] += 1
    elapsed = time.perf_counter() - started
    receipts: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        classifier = classifiers[candidate.name]
        coefficient = np.asarray(classifier.coef_[0], dtype=np.float64)
        intercept = float(classifier.intercept_[0])
        if not np.isfinite(coefficient).all() or not math.isfinite(intercept):
            raise ValueError(f"non-finite classifier parameters for {candidate.name}")
        receipts[candidate.name] = {
            "candidate": asdict(candidate),
            "coefficient": coefficient.tolist(),
            "intercept": intercept,
            "updates": int(updates[candidate.name]),
            "presentations": int(len(rows)),
            "nonfinite_parameters": 0,
        }

    return classifiers, {
        "models": receipts,
        "elapsed_seconds": elapsed,
        "presentations_per_second_all_candidates": len(rows) * len(candidates) / elapsed,
    }


def _predict_candidates(
    data: LabData,
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    plans: dict[str, TransformPlan],
    candidates: list[LinearCandidate],
    classifiers: dict[str, Any],
    *,
    batch_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions = {candidate.name: np.empty(len(rows), dtype=np.float32) for candidate in candidates}
    timing = {candidate.name: 0.0 for candidate in candidates}
    for start in range(0, len(rows), batch_rows):
        stop = min(start + batch_rows, len(rows))
        batch = np.asarray(data.X[rows[start:stop]], dtype=np.float64)
        standardized = (batch - mean) / scale
        transformed: dict[tuple[str, int | None], np.ndarray] = {}
        for candidate in candidates:
            key = (candidate.plan, candidate.dimensions)
            features = transformed.get(key)
            if features is None:
                features = _apply_plan(standardized, plans[candidate.plan], candidate.dimensions)
                transformed[key] = features
            began = time.perf_counter()
            logits = classifiers[candidate.name].decision_function(features)
            timing[candidate.name] += time.perf_counter() - began
            probability = _sigmoid(logits)
            if not np.isfinite(probability).all():
                raise ValueError(f"non-finite validation predictions for {candidate.name}")
            predictions[candidate.name][start:stop] = probability.astype(np.float32)

    return predictions, timing


def _grid_candidates() -> list[LinearCandidate]:
    candidates = []
    for method, plan in (("F1", "s1"), ("F3", "corr_s1"), ("F5", "s2")):
        for alpha in RIDGE_ALPHAS:
            candidates.append(LinearCandidate(f"{method.lower()}_ridge_a{alpha:g}", method, plan, "l2", alpha))
    for method, plan in (("F2", "s1"), ("F6", "s2")):
        for alpha in ELASTIC_ALPHAS:
            for ratio in ELASTIC_L1_RATIOS:
                candidates.append(LinearCandidate(f"{method.lower()}_elastic_a{alpha:g}_l1{ratio:g}", method, plan, "elasticnet", alpha, ratio))
    for method, plan, dimensions in (
        ("F7", "pca_s1", PCA_S1_DIMENSIONS),
        ("F8", "pca_s2", PCA_S2_DIMENSIONS),
    ):
        for dimension in dimensions:
            for alpha in RIDGE_ALPHAS:
                candidates.append(LinearCandidate(f"{method.lower()}_pca{dimension}_a{alpha:g}", method, plan, "l2", alpha, dimensions=dimension))

    return candidates


def _grid_metrics(predictions: dict[str, np.ndarray], outcomes: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {"log_loss": log_loss(probability, outcomes), "brier": brier(probability, outcomes)}
        for name, probability in predictions.items()
    }


def _select_grid(grid_by_fold: list[dict[str, Any]], candidates: list[LinearCandidate]) -> dict[str, Any]:
    by_name = {candidate.name: candidate for candidate in candidates}
    selected = {}
    for method in ("F1", "F2", "F3", "F5", "F6", "F7", "F8"):
        names = [candidate.name for candidate in candidates if candidate.method == method]
        ranking = []
        for name in names:
            losses = [fold["metrics"][name]["log_loss"] for fold in grid_by_fold]
            briers = [fold["metrics"][name]["brier"] for fold in grid_by_fold]
            ranking.append({
                "name": name,
                "candidate": asdict(by_name[name]),
                "mean_log_loss": float(np.mean(losses)),
                "se_log_loss": float(np.std(losses, ddof=1) / math.sqrt(len(losses))),
                "mean_brier": float(np.mean(briers)),
                "per_fold_log_loss": losses,
                "per_fold_brier": briers,
            })
        ranking.sort(key=lambda row: (row["mean_log_loss"], row["mean_brier"], row["name"]))
        selected[method] = {"winner": ranking[0], "ranking": ranking}

    return selected


def _semantic_blocks() -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for spec in feature_specs()[:49]:
        grouped[spec.semantic_block].append(spec.index)

    return {name: tuple(indices) for name, indices in sorted(grouped.items())}


def _selected_candidates(selected_grid: dict[str, Any], semantic_plans: dict[str, TransformPlan]) -> list[LinearCandidate]:
    selected = [LinearCandidate(**selected_grid[method]["winner"]["candidate"]) for method in ("F1", "F2", "F3", "F5", "F6", "F7", "F8")]
    ridge_alpha = float(selected_grid["F1"]["winner"]["candidate"]["alpha"])
    for block in semantic_plans:
        selected.append(LinearCandidate(f"loo_{block}", f"LOO:{block}", f"loo_{block}", "l2", ridge_alpha))

    return selected


def _method_metrics(
    probability: np.ndarray,
    index: dict[str, np.ndarray],
    data: LabData,
    base_rate: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return evaluate_temporal_predictions(
        probability,
        index,
        data,
        base_rate,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def _light_slice_metrics(
    probability: np.ndarray,
    outcomes: np.ndarray,
    keys: np.ndarray,
    names: list[str],
    *,
    min_rows: int,
) -> dict[str, Any]:
    result = {}
    for key in np.unique(keys):
        mask = keys == key
        count = int(mask.sum())
        if count < min_rows:
            continue
        label = names[int(key)] if 0 <= int(key) < len(names) else f"unknown:{int(key)}"
        result[label] = {
            "n": count,
            "log_loss": log_loss(probability[mask], outcomes[mask]),
            "brier": brier(probability[mask], outcomes[mask]),
            "ece": ece(probability[mask], outcomes[mask])["ece"],
        }

    return result


def _semantic_metrics(
    probability: np.ndarray,
    index: dict[str, np.ndarray],
    data: LabData,
) -> dict[str, Any]:
    outcomes = index["z"]
    by_source = _light_slice_metrics(
        probability, outcomes, index["source_idx"], data.source_ids, min_rows=1,
    )
    by_deck = _light_slice_metrics(
        probability, outcomes, index["deck_gidx"], data.deck_vocab, min_rows=500,
    )

    return {
        "overall": {
            "n": int(len(probability)),
            "log_loss": log_loss(probability, outcomes),
            "brier": brier(probability, outcomes),
            "ece": ece(probability, outcomes)["ece"],
        },
        "by_source": by_source,
        "by_deck": by_deck,
        "macro_source": {
            "log_loss": float(np.mean([value["log_loss"] for value in by_source.values()])),
            "brier": float(np.mean([value["brier"] for value in by_source.values()])),
        },
        "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
    }


def _slice_damage(without: dict[str, Any], full: dict[str, Any], field: str) -> dict[str, Any]:
    damage = {}
    for key in sorted(set(without.get(field, {})) & set(full.get(field, {}))):
        damage[key] = {
            "n": int(without[field][key]["n"]),
            "log_loss_damage": float(without[field][key]["log_loss"] - full[field][key]["log_loss"]),
            "brier_damage": float(without[field][key]["brier"] - full[field][key]["brier"]),
        }

    return damage


def _semantic_selection(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = _semantic_blocks()
    result = {}
    dropped = []
    for block in blocks:
        log_damage = []
        brier_damage = []
        ece_damage = []
        macro_damage = []
        inference_damage = []
        for fold in per_fold:
            full = fold["metrics"]["F1"]
            without = fold["loo_metrics"][block]
            log_damage.append(without["overall"]["log_loss"] - full["overall"]["log_loss"])
            brier_damage.append(without["overall"]["brier"] - full["overall"]["brier"])
            ece_damage.append(without["overall"]["ece"] - full["overall"]["ece"])
            macro_damage.append(without["macro_source"]["log_loss"] - full["macro_source"]["log_loss"])
            inference_damage.append(
                fold["loo_inference_seconds"][block] - fold["inference_seconds"]["F1"]
            )
        remove = (
            float(np.mean(log_damage)) <= -SEMANTIC_DROP_MEAN_GAIN
            and sum(value < 0.0 for value in log_damage) >= 2
            and max(log_damage) <= SEMANTIC_DROP_MAX_DAMAGE
            and float(np.mean(brier_damage)) <= 0.0
        )
        if remove:
            dropped.append(block)
        result[block] = {
            "indices": list(blocks[block]),
            "log_loss_damage_per_fold": log_damage,
            "mean_log_loss_damage": float(np.mean(log_damage)),
            "brier_damage_per_fold": brier_damage,
            "mean_brier_damage": float(np.mean(brier_damage)),
            "ece_damage_per_fold": ece_damage,
            "mean_ece_damage": float(np.mean(ece_damage)),
            "macro_source_log_loss_damage_per_fold": macro_damage,
            "inference_seconds_damage_per_fold": inference_damage,
            "mean_inference_seconds_damage": float(np.mean(inference_damage)),
            "drop_under_prespecified_rule": remove,
            "by_source_damage_per_fold": [
                _slice_damage(fold["loo_metrics"][block], fold["metrics"]["F1"], "by_source")
                for fold in per_fold
            ],
            "by_deck_damage_per_fold": [
                _slice_damage(fold["loo_metrics"][block], fold["metrics"]["F1"], "by_deck")
                for fold in per_fold
            ],
            "control_pass_damage": "not_instrumented_cache_has_no_control_pass_row_mask",
        }
    selected_indices = tuple(
        index
        for index in range(49)
        if feature_specs()[index].semantic_block not in set(dropped)
    )

    return {"blocks": result, "dropped_blocks": dropped, "selected_indices": list(selected_indices)}


def _coefficient_stability(
    folds: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    feature_rows = []
    for index, spec in enumerate(feature_specs()):
        ridge_method = "F1" if index < 49 else "F5"
        elastic_method = "F2" if index < 49 else "F6"
        ridge = np.asarray([fold["models"][ridge_method]["coefficient"][index] for fold in folds], dtype=np.float64)
        elastic = np.asarray([fold["models"][elastic_method]["coefficient"][index] for fold in folds], dtype=np.float64)
        ridge_denominator = np.asarray([
            max(1e-12, np.abs(np.asarray(fold["models"][ridge_method]["coefficient"], dtype=np.float64)).sum())
            for fold in folds
        ])
        elastic_denominator = np.asarray([
            max(1e-12, np.abs(np.asarray(fold["models"][elastic_method]["coefficient"], dtype=np.float64)).sum())
            for fold in folds
        ])
        selected = np.abs(elastic) >= COEFFICIENT_SELECTION_THRESHOLD
        signs = np.sign(elastic[selected])
        sign_stability = 0.0 if len(signs) == 0 else max(float((signs > 0).mean()), float((signs < 0).mean()))
        inv = inventory["features"][index]
        feature_rows.append({
            "index": index,
            "name": spec.name,
            "semantic_block": spec.semantic_block,
            "correlation_cluster": inv["correlation_cluster"],
            "elastic_selection_frequency": float(selected.mean()),
            "elastic_sign_stability_when_selected": sign_stability,
            "elastic_coefficients": elastic.tolist(),
            "elastic_normalized_magnitude_mean": float(np.mean(np.abs(elastic) / elastic_denominator)),
            "ridge_coefficients": ridge.tolist(),
            "ridge_sign_stability": max(float((ridge > 0).mean()), float((ridge < 0).mean())),
            "ridge_normalized_magnitude_mean": float(np.mean(np.abs(ridge) / ridge_denominator)),
            "source_shift": inv["source_shift"],
            "deck_shift": inv["deck_shift"],
            "policy_shift": inv["policy_shift"],
        })

    return {
        "coefficient_selection_threshold": COEFFICIENT_SELECTION_THRESHOLD,
        "fold_seed_cells": len(folds),
        "features": feature_rows,
    }


def _global_corr_indices(inventory: dict[str, Any]) -> tuple[int, ...]:
    dropped: set[int] = set()
    for left, right in inventory["exact_duplicate_pairs_on_sample"]:
        if left < 49 and right < 49:
            dropped.add(max(int(left), int(right)))
    for pair in inventory["near_duplicate_pairs_on_sample"]:
        left, right = int(pair["left"]), int(pair["right"])
        if left < 49 and right < 49 and abs(float(pair["correlation"])) >= CORRELATION_PRUNE_THRESHOLD:
            dropped.add(max(left, right))
    dropped.update(
        feature["index"]
        for feature in inventory["features"][:49]
        if feature["constant_on_eligible_train"]
    )

    return tuple(index for index in range(49) if index not in dropped)


def _representation_selection(
    aggregate_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    semantic: dict[str, Any],
    inventory: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    raw = representation_manifest("R0_raw_s1", range(49))
    corr_indices = _global_corr_indices(inventory)
    semantic_indices = tuple(int(index) for index in semantic["selected_indices"])
    compact_options = []
    for method, name, indices in (
        ("F3", "R1_correlation_pruned_s1", corr_indices),
        ("F4", "R1_semantic_block_pruned_s1", semantic_indices),
    ):
        comparison = comparisons[f"{method}_vs_F1"]
        feature_reduction = 1.0 - len(indices) / 49.0
        per_fold_deltas = [
            value[method] - value["F1"]
            for value in aggregate_metrics["per_fold_log_loss"]
        ]
        noninferior = (
            comparison["log_loss"]["ci95"][1] <= COMPACT_NONINFERIORITY_MARGIN
            and comparison["brier"]["delta_candidate_minus_reference"] <= 0.001
            and feature_reduction >= MEANINGFUL_FEATURE_REDUCTION
            and sum(delta <= COMPACT_NONINFERIORITY_MARGIN for delta in per_fold_deltas) >= 2
        )
        compact_options.append({
            "method": method,
            "name": name,
            "indices": list(indices),
            "feature_count": len(indices),
            "feature_reduction": feature_reduction,
            "noninferior_under_prespecified_gate": noninferior,
            "comparison": comparison,
        })
    eligible_compact = [option for option in compact_options if option["noninferior_under_prespecified_gate"]]
    eligible_compact.sort(key=lambda option: (option["feature_count"], option["comparison"]["log_loss"]["delta_candidate_minus_reference"]))
    selected_representations = [raw]
    compact_manifest = None
    if eligible_compact:
        chosen = eligible_compact[0]
        compact_manifest = representation_manifest(chosen["name"], chosen["indices"])
        selected_representations.append(compact_manifest)
    s2_method = min(("F5", "F6"), key=lambda name: aggregate_metrics["ensemble"][name]["overall"]["log_loss"])
    s2_comparison = comparisons[f"{s2_method}_vs_F1"]
    per_fold_s2 = [
        value[s2_method] - value["F1"]
        for value in aggregate_metrics["per_fold_log_loss"]
    ]
    s2_gate = (
        s2_comparison["log_loss"]["ci95"][1] < 0.0
        and s2_comparison["brier"]["delta_candidate_minus_reference"] <= 0.0
        and sum(delta < 0.0 for delta in per_fold_s2) >= 2
        and aggregate_metrics["ensemble"][s2_method]["macro_source"]["log_loss"] <= aggregate_metrics["ensemble"]["F1"]["macro_source"]["log_loss"]
    )
    stable_s2_indices = [
        feature["index"]
        for feature in stability["features"]
        if feature["elastic_selection_frequency"] >= (2.0 / 3.0)
        and feature["elastic_sign_stability_when_selected"] >= (2.0 / 3.0)
        and feature["index"] < 133
    ]
    if s2_gate and any(index >= 49 for index in stable_s2_indices):
        s2_manifest = representation_manifest("R2_stable_regularized_s2_without_hand", stable_s2_indices)
        selected_representations.append(s2_manifest)
    else:
        s2_manifest = None

    return {
        "raw": raw,
        "compact_options": compact_options,
        "selected_compact": compact_manifest,
        "best_s2_linear_method": s2_method,
        "s2_advancement_gate": s2_gate,
        "stable_s2_indices_without_hand": stable_s2_indices,
        "selected_s2": s2_manifest,
        "nonlinear_confirmation_representations": selected_representations,
        "hand_block_selection_status": "quarantine_top_level_own_hand_provenance_not_certified",
    }


def _output_receipts(root: Path) -> list[dict[str, Any]]:
    receipts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            receipts.append({
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })

    return receipts


def run_linear_feature_screen(
    *,
    cache_dir: Path,
    split_manifest_path: Path,
    inventory_path: Path,
    out_dir: Path,
    presentations_per_fold: int,
    batch_rows: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    bootstrap_replicates: int = 500,
    validation_limit: int = 0,
) -> dict[str, Any]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("linear stability screen requires exactly three distinct deterministic seeds")
    if out_dir.exists():
        raise FileExistsError(f"linear feature screen output already exists: {out_dir}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "completed" or len(inventory.get("features", [])) != 144:
        raise ValueError("linear feature screen requires the completed exact feature inventory")
    data = LabData(str(cache_dir), split_manifest_path=str(split_manifest_path))
    fold_contract = source_stratified_game_folds(data, folds=3)
    validation = temporal_validation_index(data)
    if validation_limit:
        if isinstance(validation_limit, bool) or not isinstance(validation_limit, int) or validation_limit < 1:
            raise ValueError("validation_limit must be zero or a positive integer")
        selected_validation = deterministic_sample(
            np.arange(len(validation["row_id"]), dtype=np.int64),
            validation_limit,
            915004,
        )
        validation = {key: value[selected_validation] for key, value in validation.items()}
    validation_rows = validation["row_id"].astype(np.int64, copy=False)
    validation_outcomes = validation["z"].astype(np.float64, copy=False)
    train_index = data.split_names.index("train")
    temporary = out_dir.with_name(f".{out_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        config = {
            "schema_version": LINEAR_SCREEN_SCHEMA_VERSION,
            "cache": str(cache_dir),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": _sha256(split_manifest_path),
            "inventory": str(inventory_path),
            "inventory_file_sha256": _sha256(inventory_path),
            "inventory_scientific_sha256": inventory["inventory_sha256"],
            "presentations_per_stability_fold": int(presentations_per_fold),
            "stability_folds": 3,
            "stability_fold_assignment_sha256": fold_contract["assignment_sha256"],
            "seeds": list(seeds),
            "batch_rows": int(batch_rows),
            "validation_limit": int(validation_limit),
            "validation_rows": int(len(validation_rows)),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "elastic_alphas": list(ELASTIC_ALPHAS),
            "elastic_l1_ratios": list(ELASTIC_L1_RATIOS),
            "pca_s1_dimensions": list(PCA_S1_DIMENSIONS),
            "pca_s2_dimensions": list(PCA_S2_DIMENSIONS),
            "pca": "training_only_standardized_covariance_eigh_whitened_diagnostic",
            "draw_contract": "each z=0.5 row expands to class0/class1 with weights 0.5/0.5",
            "optimizer": "sklearn.SGDClassifier.partial_fit log_loss one pass average=True learning_rate=constant eta0=0.01 shuffle=False",
            "correlation_prune_abs": CORRELATION_PRUNE_THRESHOLD,
            "coefficient_selection_threshold": COEFFICIENT_SELECTION_THRESHOLD,
            "compact_noninferiority_margin_log_loss": COMPACT_NONINFERIORITY_MARGIN,
            "meaningful_feature_reduction": MEANINGFUL_FEATURE_REDUCTION,
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
        }
        config["sha256"] = _stable_hash(config)
        _atomic_json(temporary / "config.json", config)
        _atomic_json(temporary / "stability_folds.json", fold_contract["assignment"])
        _atomic_npz(
            temporary / "validation_index.npz",
            **{key: value for key, value in validation.items() if key != "z"},
        )
        fold_contexts = []
        grid_by_fold = []
        grid_candidates = _grid_candidates()
        for fold, seed in enumerate(seeds):
            eligible = (
                (data.meta["split_idx"] == train_index)
                & data.meta["is_target_eligible"]
                & (fold_contract["fold_of_row"] == fold)
            )
            available = np.flatnonzero(eligible).astype(np.int64, copy=False)
            rows = deterministic_sample(available, presentations_per_fold, seed)
            if len(rows) < presentations_per_fold:
                raise ValueError(f"stability fold {fold} has only {len(rows)} eligible rows")
            mean, scale, fold_standard_deviation = _standardization(data, rows, batch_rows)
            pca_rows = deterministic_sample(rows, min(131_072, len(rows)), seed + 100)
            pca_sample = (np.asarray(data.X[pca_rows], dtype=np.float64) - mean) / scale
            correlation = correlation_inventory(np.asarray(data.X[pca_rows], dtype=np.float32))
            corr_indices, corr_dropped = correlation_pruned_indices(
                np.asarray(correlation["correlation"], dtype=np.float64),
                fold_standard_deviation,
                range(49),
            )
            plans = {
                "s1": TransformPlan("s1", "raw", tuple(range(49))),
                "s2": TransformPlan("s2", "raw", tuple(range(144))),
                "corr_s1": TransformPlan("corr_s1", "raw", corr_indices),
                "pca_s1": fit_pca_plan(pca_sample, range(49), maximum_dimensions=max(PCA_S1_DIMENSIONS), name="pca_s1"),
                "pca_s2": fit_pca_plan(pca_sample, range(144), maximum_dimensions=max(PCA_S2_DIMENSIONS), name="pca_s2"),
            }
            classifiers, train_receipt = _train_candidates(
                data, rows, mean, scale, plans, grid_candidates, batch_rows=batch_rows, seed=seed,
            )
            predictions, inference = _predict_candidates(
                data, validation_rows, mean, scale, plans, grid_candidates, classifiers, batch_rows=batch_rows,
            )
            metrics = _grid_metrics(predictions, validation_outcomes)
            fold_dir = temporary / f"fold_{fold}_seed_{seed}"
            fold_dir.mkdir()
            _atomic_npz(fold_dir / "train_rows.npz", row_id=rows)
            _atomic_npz(
                fold_dir / "transforms.npz",
                mean=mean,
                scale=scale,
                corr_s1_indices=np.asarray(corr_indices, dtype=np.int32),
                pca_s1_components=plans["pca_s1"].components,
                pca_s1_eigenvalues=plans["pca_s1"].eigenvalues,
                pca_s2_components=plans["pca_s2"].components,
                pca_s2_eigenvalues=plans["pca_s2"].eigenvalues,
            )
            grid_result = {
                "fold": fold,
                "seed": seed,
                "available_eligible_rows": int(len(available)),
                "sampled_presentations": int(len(rows)),
                "train_row_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
                "base_rate": float(data.meta["z"][rows].mean()),
                "correlation_pruned_indices": list(corr_indices),
                "correlation_dropped": corr_dropped,
                "metrics": metrics,
                "training": train_receipt,
                "inference_seconds": inference,
            }
            _atomic_json(fold_dir / "grid_result.json", grid_result)
            grid_by_fold.append(grid_result)
            fold_contexts.append({
                "fold": fold,
                "seed": seed,
                "rows": rows,
                "mean": mean,
                "scale": scale,
                "plans": plans,
                "fold_dir": fold_dir,
                "base_rate": grid_result["base_rate"],
            })
            del classifiers, predictions, pca_sample
        selected_grid = _select_grid(grid_by_fold, grid_candidates)
        _atomic_json(temporary / "grid_selection.json", selected_grid)
        per_fold = []
        method_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
        for context in fold_contexts:
            semantic_plans = {}
            for block, indices in _semantic_blocks().items():
                retained = tuple(index for index in range(49) if index not in set(indices))
                semantic_plans[block] = TransformPlan(f"loo_{block}", "raw", retained)
            plans = {**context["plans"], **{f"loo_{name}": plan for name, plan in semantic_plans.items()}}
            candidates = _selected_candidates(selected_grid, semantic_plans)
            classifiers, train_receipt = _train_candidates(
                data,
                context["rows"],
                context["mean"],
                context["scale"],
                plans,
                candidates,
                batch_rows=batch_rows,
                seed=context["seed"],
            )
            predictions, inference = _predict_candidates(
                data,
                validation_rows,
                context["mean"],
                context["scale"],
                plans,
                candidates,
                classifiers,
                batch_rows=batch_rows,
            )
            base = np.full(len(validation_rows), context["base_rate"], dtype=np.float32)
            method_probability = {"F0": base}
            model_receipts = {}
            loo_metrics = {}
            for candidate in candidates:
                receipt = train_receipt["models"][candidate.name]
                if candidate.method.startswith("LOO:"):
                    block = candidate.method.split(":", 1)[1]
                    loo_metrics[block] = _semantic_metrics(predictions[candidate.name], validation, data)
                    continue
                method_probability[candidate.method] = predictions[candidate.name]
                model_receipts[candidate.method] = receipt
            metrics = {
                method: _method_metrics(
                    probability,
                    validation,
                    data,
                    base,
                    bootstrap_replicates=100,
                    bootstrap_seed=context["seed"],
                )
                for method, probability in method_probability.items()
            }
            for method, probability in method_probability.items():
                method_predictions[method].append(probability)
            selected_payload = {
                "fold": context["fold"],
                "seed": context["seed"],
                "metrics": metrics,
                "loo_metrics": loo_metrics,
                "models": model_receipts,
                "training_elapsed_seconds": train_receipt["elapsed_seconds"],
                "inference_seconds": {candidate.method: inference[candidate.name] for candidate in candidates if not candidate.method.startswith("LOO:")},
                "loo_inference_seconds": {
                    candidate.method.split(":", 1)[1]: inference[candidate.name]
                    for candidate in candidates
                    if candidate.method.startswith("LOO:")
                },
            }
            _atomic_json(context["fold_dir"] / "selected_result.json", selected_payload)
            _atomic_json(context["fold_dir"] / "selected_models.json", model_receipts)
            _atomic_npz(
                context["fold_dir"] / "selected_predictions.npz",
                row_id=validation_rows,
                **{method.lower(): probability for method, probability in method_probability.items()},
            )
            per_fold.append(selected_payload)
            del classifiers, predictions
        semantic = _semantic_selection(per_fold)
        _atomic_json(temporary / "semantic_block_ablation.json", semantic)
        f4_indices = tuple(int(index) for index in semantic["selected_indices"])
        f4_predictions = []
        ridge_alpha = float(selected_grid["F1"]["winner"]["candidate"]["alpha"])
        for context, fold_result in zip(fold_contexts, per_fold):
            plan = TransformPlan("f4_semantic", "raw", f4_indices)
            candidate = LinearCandidate("f4_semantic_pruned", "F4", "f4_semantic", "l2", ridge_alpha)
            classifiers, train_receipt = _train_candidates(
                data,
                context["rows"],
                context["mean"],
                context["scale"],
                {"f4_semantic": plan},
                [candidate],
                batch_rows=batch_rows,
                seed=context["seed"],
            )
            predictions, inference = _predict_candidates(
                data,
                validation_rows,
                context["mean"],
                context["scale"],
                {"f4_semantic": plan},
                [candidate],
                classifiers,
                batch_rows=batch_rows,
            )
            probability = predictions[candidate.name]
            base = method_predictions["F0"][context["fold"]]
            metrics = _method_metrics(
                probability,
                validation,
                data,
                base,
                bootstrap_replicates=100,
                bootstrap_seed=context["seed"],
            )
            fold_result["metrics"]["F4"] = metrics
            fold_result["models"]["F4"] = train_receipt["models"][candidate.name]
            fold_result["inference_seconds"]["F4"] = inference[candidate.name]
            method_predictions["F4"].append(probability)
            f4_predictions.append(probability)
            _atomic_npz(context["fold_dir"] / "f4_predictions.npz", row_id=validation_rows, f4=probability)
            _atomic_json(context["fold_dir"] / "f4_model.json", train_receipt["models"][candidate.name])
        ensemble_predictions = {
            method: np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
            for method, values in method_predictions.items()
        }
        ensemble_metrics = {
            method: _method_metrics(
                probability,
                validation,
                data,
                ensemble_predictions["F0"],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=915003,
            )
            for method, probability in ensemble_predictions.items()
        }
        comparisons = {
            f"{method}_vs_F1": direct_comparison(
                ensemble_predictions[method],
                ensemble_predictions["F1"],
                validation,
                candidate=method,
                reference="F1",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=915003,
            )
            for method in ("F2", "F3", "F4", "F5", "F6", "F7", "F8")
        }
        per_fold_log_loss = [
            {method: fold["metrics"][method]["overall"]["log_loss"] for method in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")}
            for fold in per_fold
        ]
        aggregate = {
            "ensemble": ensemble_metrics,
            "comparisons": comparisons,
            "per_fold_log_loss": per_fold_log_loss,
            "per_fold": per_fold,
        }
        stability = _coefficient_stability(per_fold, inventory)
        representation_selection = _representation_selection(aggregate, comparisons, semantic, inventory, stability)
        _atomic_json(temporary / "feature_stability.json", stability)
        representations_dir = temporary / "representations"
        representations_dir.mkdir()
        for manifest in representation_selection["nonlinear_confirmation_representations"]:
            _atomic_json(representations_dir / f"{manifest['name']}.json", manifest)
        selection = {
            "schema_version": LINEAR_SCREEN_SCHEMA_VERSION,
            "status": "completed",
            "selected_grid": selected_grid,
            "semantic_selection": semantic,
            "representation_selection": representation_selection,
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
            "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
        }
        result = {
            "schema_version": LINEAR_SCREEN_SCHEMA_VERSION,
            "status": "completed",
            "config_sha256": config["sha256"],
            "grid": grid_by_fold,
            "aggregate": aggregate,
            "feature_stability": stability,
            "selection": selection,
            "nonfinite_counts": 0,
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
        }
        _atomic_json(temporary / "selection.json", selection)
        _atomic_json(temporary / "result.json", result)
        _atomic_npz(
            temporary / "ensemble_predictions.npz",
            row_id=validation_rows,
            **{method.lower(): probability for method, probability in ensemble_predictions.items()},
        )
        receipts = _output_receipts(temporary)
        manifest = build_manifest(
            run_id=out_dir.name,
            kind="value_linear_feature_screen",
            command=sys.argv,
            inputs={
                "config": config,
                "environment": {
                    "python": sys.version.split()[0],
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "sklearn": __import__("sklearn").__version__,
                },
                "code_sha256": {
                    "linear_screen": _sha256(__file__),
                    "feature_contract": _sha256(Path(__file__).with_name("feature_contract.py")),
                    "feature_selection": _sha256(Path(__file__).with_name("feature_selection.py")),
                },
            },
            outputs={"files": receipts},
            summary={
                "status": "screening_only",
                "selected_representations": [manifest["name"] for manifest in representation_selection["nonlinear_confirmation_representations"]],
                "s2_advancement_gate": representation_selection["s2_advancement_gate"],
                "calibration_targets_consumed": False,
                "test_targets_consumed": False,
                "protected_runtime_surfaces_changed": False,
            },
        )
        write_manifest(temporary / "manifest.json", manifest)
        os.replace(temporary, out_dir)

        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
