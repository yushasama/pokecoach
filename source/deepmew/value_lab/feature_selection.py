"""Training-only feature inventory utilities for the value/Q experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cg.api import all_card_data

from deepmew.value_lab.data import LabData
from deepmew.value_lab.feature_contract import feature_specs, full_feature_manifest


FEATURE_INVENTORY_SCHEMA_VERSION = "deepmew.value_feature_inventory.v1"
DEFAULT_AUDIT_SEED = 915001
DEFAULT_CORRELATION_SAMPLE_ROWS = 131_072
DEFAULT_BATCH_ROWS = 16_384
NEAR_CONSTANT_STD = 1e-4
NEAR_CONSTANT_DOMINANT_FRACTION = 0.999
CORRELATION_CLUSTER_THRESHOLD = 0.98
NEAR_DUPLICATE_THRESHOLD = 0.995


def _sha256(path: str | Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def training_target_rows(data: LabData) -> np.ndarray:
    train_index = data.split_names.index("train")
    rows = np.flatnonzero(
        (data.meta["split_idx"] == train_index) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    if len(rows) == 0:
        raise ValueError("accepted cache has no eligible training targets")

    return rows


def feature_moments(
    matrix: np.ndarray,
    rows: np.ndarray,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, np.ndarray | int]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("feature moments require nonempty one-dimensional rows")
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    dimensions = int(matrix.shape[1])
    minimum = np.full(dimensions, np.inf, dtype=np.float64)
    maximum = np.full(dimensions, -np.inf, dtype=np.float64)
    total = np.zeros(dimensions, dtype=np.float64)
    total_square = np.zeros(dimensions, dtype=np.float64)
    zero_count = np.zeros(dimensions, dtype=np.int64)
    nonfinite_count = np.zeros(dimensions, dtype=np.int64)
    for start in range(0, len(rows), batch_rows):
        selected = rows[start:start + batch_rows]
        batch = np.asarray(matrix[selected], dtype=np.float64)
        finite = np.isfinite(batch)
        nonfinite_count += (~finite).sum(axis=0)
        if not finite.all():
            raise ValueError("canonical feature inventory encountered non-finite values")
        minimum = np.minimum(minimum, batch.min(axis=0))
        maximum = np.maximum(maximum, batch.max(axis=0))
        total += batch.sum(axis=0, dtype=np.float64)
        total_square += np.square(batch).sum(axis=0, dtype=np.float64)
        zero_count += (batch == 0.0).sum(axis=0)
    count = int(len(rows))
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)

    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": np.sqrt(variance),
        "zero_fraction": zero_count.astype(np.float64) / count,
        "nonfinite_count": nonfinite_count,
    }


def deterministic_sample(rows: np.ndarray, size: int, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("sample size must be a positive integer")
    take = min(size, len(rows))
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(rows), size=take, replace=False)

    return np.sort(rows[positions])


def _dominant_fractions(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimensions = sample.shape[1]
    fraction = np.zeros(dimensions, dtype=np.float64)
    value = np.zeros(dimensions, dtype=np.float64)
    for column in range(dimensions):
        values, counts = np.unique(sample[:, column], return_counts=True)
        selected = int(np.argmax(counts))
        value[column] = float(values[selected])
        fraction[column] = float(counts[selected]) / len(sample)

    return value, fraction


def correlation_inventory(sample: np.ndarray) -> dict[str, Any]:
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[0] < 2:
        raise ValueError("correlation sample must be a two-dimensional matrix with multiple rows")
    dimensions = sample.shape[1]
    std = sample.std(axis=0)
    correlation = np.eye(dimensions, dtype=np.float64)
    variable = std > 0.0
    if int(variable.sum()) > 1:
        sub = np.corrcoef(sample[:, variable], rowvar=False)
        indices = np.flatnonzero(variable)
        correlation[np.ix_(indices, indices)] = sub
    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)
    dominant_value, dominant_fraction = _dominant_fractions(sample)
    parent = np.arange(dimensions, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])

        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    exact_pairs: list[list[int]] = []
    near_pairs: list[dict[str, float | int]] = []
    for left in range(dimensions):
        for right in range(left + 1, dimensions):
            absolute = abs(float(correlation[left, right]))
            if absolute >= CORRELATION_CLUSTER_THRESHOLD:
                union(left, right)
            if np.array_equal(sample[:, left], sample[:, right]):
                exact_pairs.append([left, right])
            elif absolute >= NEAR_DUPLICATE_THRESHOLD:
                near_pairs.append({"left": left, "right": right, "correlation": float(correlation[left, right])})
    groups: dict[int, list[int]] = {}
    for column in range(dimensions):
        groups.setdefault(find(column), []).append(column)
    clusters = [values for values in groups.values() if len(values) > 1]
    clusters.sort(key=lambda values: values[0])
    cluster_of = {column: index for index, values in enumerate(clusters) for column in values}
    top: list[list[dict[str, float | int]]] = []
    for column in range(dimensions):
        order = np.argsort(-np.abs(correlation[column]))
        top.append([
            {"index": int(other), "correlation": float(correlation[column, other])}
            for other in order
            if int(other) != column
        ][:3])

    return {
        "correlation": correlation,
        "dominant_value": dominant_value,
        "dominant_fraction": dominant_fraction,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "clusters": clusters,
        "cluster_of": cluster_of,
        "top_correlations": top,
    }


def _row_block_ids(data: LabData, rows: np.ndarray) -> np.ndarray:
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_ids = np.searchsorted(starts, rows, side="right") - 1
    if np.any(block_ids < 0) or np.any(rows >= starts[block_ids] + lengths[block_ids]):
        raise ValueError("feature inventory rows do not map to canonical blocks")

    return block_ids.astype(np.int64, copy=False)


def _major_slice_shift(
    sample: np.ndarray,
    keys: np.ndarray,
    labels: list[str],
    *,
    min_rows: int,
) -> list[dict[str, Any]]:
    global_mean = sample.mean(axis=0)
    global_std = sample.std(axis=0)
    best_delta = np.zeros(sample.shape[1], dtype=np.float64)
    best_label = np.full(sample.shape[1], "none", dtype=object)
    best_rows = np.zeros(sample.shape[1], dtype=np.int64)
    for key in np.unique(keys):
        mask = keys == key
        count = int(mask.sum())
        if count < min_rows:
            continue
        mean = sample[mask].mean(axis=0)
        delta = np.divide(np.abs(mean - global_mean), global_std, out=np.zeros_like(mean), where=global_std > 0)
        replace = delta > best_delta
        best_delta[replace] = delta[replace]
        label = labels[int(key)] if 0 <= int(key) < len(labels) else f"unknown:{int(key)}"
        best_label[replace] = label
        best_rows[replace] = count

    return [
        {"max_abs_mean_shift_std": float(best_delta[index]), "group": str(best_label[index]), "rows": int(best_rows[index])}
        for index in range(sample.shape[1])
    ]


def _card_catalog_hash() -> str:
    rows = [
        (
            int(card.cardId),
            int(card.cardType),
            bool(card.basic),
            bool(card.stage1),
            bool(card.stage2),
            bool(card.megaEx),
            bool(card.ex),
            int(card.energyType),
        )
        for card in all_card_data()
        if card is not None
    ]

    return _stable_hash(rows)


def build_feature_inventory(
    data: LabData,
    *,
    cache_dir: Path,
    split_manifest_path: Path,
    correlation_sample_rows: int = DEFAULT_CORRELATION_SAMPLE_ROWS,
    seed: int = DEFAULT_AUDIT_SEED,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = training_target_rows(data)
    moments = feature_moments(data.X, rows, batch_rows=batch_rows)
    sample_rows = deterministic_sample(rows, correlation_sample_rows, seed)
    sample = np.asarray(data.X[sample_rows], dtype=np.float32)
    correlations = correlation_inventory(sample)
    block_ids = _row_block_ids(data, sample_rows)
    source_shift = _major_slice_shift(sample, data.meta["source_idx"][sample_rows], data.source_ids, min_rows=500)
    policy_shift = _major_slice_shift(sample, data.blocks["policy_gidx"][block_ids], data.policy_vocab, min_rows=500)
    deck_shift = _major_slice_shift(sample, data.blocks["deck_gidx"][block_ids], data.deck_vocab, min_rows=500)
    specs = feature_specs()
    features = []
    for spec in specs:
        index = spec.index
        top = correlations["top_correlations"][index]
        features.append({
            **spec.__dict__,
            "constant_on_eligible_train": bool(float(moments["min"][index]) == float(moments["max"][index])),
            "near_constant_on_eligible_train": bool(
                float(moments["std"][index]) < NEAR_CONSTANT_STD
                or float(correlations["dominant_fraction"][index]) >= NEAR_CONSTANT_DOMINANT_FRACTION
            ),
            "observed_min": float(moments["min"][index]),
            "observed_max": float(moments["max"][index]),
            "observed_mean": float(moments["mean"][index]),
            "observed_std": float(moments["std"][index]),
            "zero_fraction": float(moments["zero_fraction"][index]),
            "dominant_sample_value": float(correlations["dominant_value"][index]),
            "dominant_sample_fraction": float(correlations["dominant_fraction"][index]),
            "correlation_cluster": correlations["cluster_of"].get(index),
            "top_correlations": top,
            "source_shift": source_shift[index],
            "policy_shift": policy_shift[index],
            "deck_shift": deck_shift[index],
        })
    root = Path(__file__).resolve().parents[2]
    producer_paths = [
        root / "scripts" / "run_unified_r3_value.py",
        root / "scripts" / "run_unified_r4_ablation.py",
        root / "scripts" / "build_value_q_canonical.py",
        root / "deepmew" / "replay" / "canonical_actor_dataset.py",
        root / "deepmew" / "second_order" / "reader.py",
        root / "cg" / "api.py",
    ]
    result = {
        "schema_version": FEATURE_INVENTORY_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "python_executable": sys.executable,
        },
        "bindings": {
            "cache": str(cache_dir),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": _sha256(split_manifest_path),
            "x_sha256_from_cache_manifest": data.manifest["output_files"]["X.f32.bin"]["sha256"],
            "producer_sha256": {str(path.relative_to(root)).replace("\\", "/"): _sha256(path) for path in producer_paths},
            "card_catalog_relevant_fields_sha256": _card_catalog_hash(),
        },
        "feature_contract": full_feature_manifest(),
        "eligible_train_rows": int(len(rows)),
        "correlation_sample": {
            "rows": int(len(sample_rows)),
            "seed": int(seed),
            "row_ids_sha256": hashlib.sha256(np.ascontiguousarray(sample_rows, dtype=np.int64).tobytes()).hexdigest(),
            "sampling": "uniform_without_replacement_over_eligible_outer_train_rows_then_sorted_for_io",
            "row_weighting_caveat": "long games contribute more rows; predictive comparisons use game-clustered uncertainty",
        },
        "thresholds": {
            "near_constant_std": NEAR_CONSTANT_STD,
            "near_constant_dominant_fraction": NEAR_CONSTANT_DOMINANT_FRACTION,
            "correlation_cluster_abs": CORRELATION_CLUSTER_THRESHOLD,
            "near_duplicate_abs": NEAR_DUPLICATE_THRESHOLD,
            "major_slice_sample_min_rows": 500,
        },
        "features": features,
        "correlation_clusters": correlations["clusters"],
        "exact_duplicate_pairs_on_sample": correlations["exact_duplicate_pairs"],
        "near_duplicate_pairs_on_sample": correlations["near_duplicate_pairs"],
        "nonfinite_values": int(np.asarray(moments["nonfinite_count"]).sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "correlations and dominant-value frequencies use the declared deterministic row sample",
            "cache manifests bind feature bytes but do not independently bind every imported producer or the live card catalog",
            "top-level source-row own_hand is consulted before the projected observation hand; producer absence was not proven over external corpora",
            "control_passed is not present in the cache and select_context is not a valid substitute",
        ],
    }
    result["inventory_sha256"] = _stable_hash(result)

    return result


def write_feature_inventory(path: Path, inventory: dict[str, Any]) -> None:
    _atomic_json(path, inventory)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_feature_audit(inventory: dict[str, Any]) -> str:
    features = inventory["features"]
    exact_pairs = inventory["exact_duplicate_pairs_on_sample"]
    near_pairs = inventory["near_duplicate_pairs_on_sample"]
    constants = [feature["index"] for feature in features if feature["constant_on_eligible_train"]]
    near_constants = [feature["index"] for feature in features if feature["near_constant_on_eligible_train"]]
    lines = [
        "# Feature Selection Audit",
        "",
        f"Status: Phase F0 complete. Inventory SHA-256 `{inventory['inventory_sha256']}`.",
        "",
        "The accepted cache was read without modification. Canonical X is exactly `r15_state_v2[49] + encode_base[84] + hand_composition[11]`; policy IDs, deck hashes, source IDs, outcomes, R4 bench-danger fields, and option-specific features are not model inputs.",
        "",
        "## Bindings and measured scope",
        "",
        f"- Eligible outer-train targets scanned exactly: {inventory['eligible_train_rows']:,}.",
        f"- Correlation/dominant-value sample: {inventory['correlation_sample']['rows']:,} rows, seed `{inventory['correlation_sample']['seed']}`, row-id SHA `{inventory['correlation_sample']['row_ids_sha256']}`.",
        f"- Cache manifest SHA: `{inventory['bindings']['cache_manifest_sha256']}`; split SHA: `{inventory['bindings']['split_manifest_sha256']}`; X SHA: `{inventory['bindings']['x_sha256_from_cache_manifest']}`.",
        f"- Constant columns: `{constants}`. Near-constant columns under the prespecified thresholds: `{near_constants}`.",
        f"- Exact duplicate pairs on the deterministic sample: `{exact_pairs}`.",
        f"- Additional |correlation| >= {NEAR_DUPLICATE_THRESHOLD} pairs: `{near_pairs}`.",
        "- Cache values are raw float32 and are not standardized. Every later transform must fit outer-train data only.",
        "",
        "## Critical provenance and visibility findings",
        "",
        "- Every field is intended to be actor-visible. Opponent inputs are public board/count fields only; no opponent hidden hand/deck/prize identities enter X.",
        "- S1 indices 32-38 are named-card/deck-specific Water, Surfing Beach (ID 1262), and Mega Abomasnow (ID 723) proxies. They are actor-observable but not deck-neutral.",
        "- S2 indices 49-132 can fingerprint source schema, phase, or policy through legal-context/missingness patterns; indices 133-143 are strong deck proxies through the actor's legitimate own hand.",
        "- `_hand_ids` consults a top-level source-row `own_hand` before the certified projected observation. No repository producer was found, but external-corpus absence is unproven; the hand block stays diagnostic/quarantined for representation selection unless separately certified.",
        "- The accepted feature bytes are immutable and hashed, but their unit manifests do not bind every imported R3/R4/second-order/card-catalog producer. This inventory binds the current live producer hashes; it does not rewrite Phase 2.",
        "- `control_passed` is not instrumented in the cache. `select_context` is not used as a substitute.",
        "",
        "## Exact per-column contract and measured statistics",
        "",
        "`source/deck/policy shift` is the largest absolute group-mean deviation in sample standard deviations among groups with at least 500 sampled rows. It indicates proxy concentration, not direct metadata input or causal importance.",
        "",
        "| idx | exact name | exact formula / source | semantic block | dtype; declared/observed range; missing/default; standardized | constant / near | correlation cluster and top correlations | actor observable | specificity and measured proxy shifts | duplicates / caveat |",
        "|---:|---|---|---|---|---|---|---|---|---|",
    ]
    names = {feature["index"]: feature["name"] for feature in features}
    for feature in features:
        top = ", ".join(
            f"{item['index']}:{names[item['index']]}={item['correlation']:+.4f}"
            for item in feature["top_correlations"]
        )
        shifts = (
            f"declared deck={feature['deck_specificity']}; policy={feature['policy_specificity']}; "
            f"source={feature['source_specificity']}; turn={feature['turn_specificity']}; phase={feature['phase_specificity']}; "
            f"measured source={feature['source_shift']['max_abs_mean_shift_std']:.2f}σ ({feature['source_shift']['group']}), "
            f"deck={feature['deck_shift']['max_abs_mean_shift_std']:.2f}σ ({feature['deck_shift']['group']}), "
            f"policy={feature['policy_shift']['max_abs_mean_shift_std']:.2f}σ ({feature['policy_shift']['group']})"
        )
        duplicates = f"declared {list(feature['duplicate_candidates'])}"
        if feature["caveat"]:
            duplicates += f"; {feature['caveat']}"
        lines.append(
            "| " + " | ".join(_cell(value) for value in (
                feature["index"],
                feature["name"],
                f"`{feature['formula']}`; `{feature['source_symbol']}`",
                feature["semantic_block"],
                f"{feature['dtype']}; {feature['declared_range']}; observed [{feature['observed_min']:.6g},{feature['observed_max']:.6g}], mean {feature['observed_mean']:.6g}, std {feature['observed_std']:.6g}; {feature['missing_default']}; cache standardized={feature['standardized_in_cache']}",
                f"{feature['constant_on_eligible_train']} / {feature['near_constant_on_eligible_train']} (dominant {feature['dominant_sample_value']:.6g} @ {feature['dominant_sample_fraction']:.4%})",
                f"cluster {feature['correlation_cluster']}; {top}",
                feature["directly_actor_observable"],
                shifts,
                duplicates,
            )) + " |"
        )
    lines.extend([
        "",
        "## Phase F0 verdict",
        "",
        "- S1 is the only currently selected nonlinear core representation. Its named-card proxies are interpretable but distribution-specific and must earn retention through stability/block ablations.",
        "- Raw S2 remains diagnostic. Its strongest concerns are heavy algebraic duplication, legal-context/source proxying, and the hand-block provenance bypass. A positive linear result cannot override the negative Phase 4 dense-S2 MLP result without matched nonlinear confirmation.",
        "- Correlated substitutes will be reported as groups. Correlation pruning protects the simplest lower-index S1 representative and never uses validation/calibration/test statistics.",
        "- This inventory is descriptive only; it does not claim that any smaller representation is better.",
        "",
    ])

    return "\n".join(lines)


def write_feature_audit(path: Path, inventory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(render_feature_audit(inventory), encoding="utf-8")
    os.replace(temporary, path)
