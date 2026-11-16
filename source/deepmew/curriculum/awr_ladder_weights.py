"""Configurable AWR baselines and ladder row weights.

This module is training-only. Disabled config reproduces the historical AWR
shape: per-opponent-deck baseline, no metadata multipliers, unweighted KL.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from deepmew.curriculum.ladder_metadata import UNKNOWN, enrich_refgame_row

GLOBAL_KEY = "__global__"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "baseline": {
        "key_fields": ["opponent_deck_hash"],
        "fallback_key_fields": [["global"]],
        "min_cell_n": 32,
        "shrink_to_global": False,
        "shrink_k": 64,
    },
    "multipliers": {},
    "total_weight": {
        "min": 0.0,
        "max": 1.0e18,
        "normalize_mean": None,
        "normalize_by": "global",
    },
    "kl": {"weight_kl": False},
}


@dataclass(frozen=True)
class RowWeight:
    ret: float
    baseline: float
    advantage: float
    w_awr: float
    w_meta: float
    w_total: float
    key: str


def outcome_return(outcome: dict[str, Any]) -> float:
    if outcome.get("won"):
        return 1.0
    if outcome.get("draw"):
        return 0.0

    return -1.0


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw or {}
    if "ladder_weighting" in data:
        data = data.get("ladder_weighting") or {}
    config = _deep_merge(DEFAULT_CONFIG, data)
    baseline = config.setdefault("baseline", {})
    baseline.setdefault("key_fields", ["opponent_deck_hash"])
    baseline.setdefault("fallback_key_fields", [["global"]])
    baseline.setdefault("min_cell_n", 32)
    baseline.setdefault("shrink_to_global", False)
    baseline.setdefault("shrink_k", 64)
    config.setdefault("multipliers", {})
    config.setdefault("total_weight", {})
    config["total_weight"].setdefault("min", 0.0)
    config["total_weight"].setdefault("max", 1.0e18)
    config["total_weight"].setdefault("normalize_mean", None)
    config["total_weight"].setdefault("normalize_by", "global")
    config.setdefault("kl", {})
    config["kl"].setdefault("weight_kl", False)

    return config


def load_ladder_weighting_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return normalize_config(None)
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.suffix.lower() in {".yaml", ".yml"} else json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"ladder weighting config must be a mapping: {p}")

    return normalize_config(raw)


def build_group_key(row: dict[str, Any], fields: list[str] | tuple[str, ...]) -> str:
    if list(fields) == ["global"]:
        return GLOBAL_KEY
    parts = []
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            value = UNKNOWN
        parts.append(f"{field}={value}")

    return "|".join(parts)


def _groups(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[build_group_key(row, fields)].append(outcome_return(row.get("outcome") or {}))

    return grouped


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _cell_summary(groups: dict[str, list[float]], limit: int = 8) -> dict[str, Any]:
    rows = [{"key": key, "n": len(values), "mean_return": round(_mean(values), 4)}
            for key, values in groups.items()]
    rows.sort(key=lambda item: (-int(item["n"]), str(item["key"])))

    return {
        "count": len(rows),
        "top_by_count": rows[:limit],
        "bottom_by_count": list(reversed(rows[-limit:])) if rows else [],
    }


class BaselineModel:
    def __init__(self, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
        baseline_cfg = config["baseline"]
        self.key_fields = [str(v) for v in baseline_cfg.get("key_fields") or ["opponent_deck_hash"]]
        self.fallback_key_fields = [[str(v) for v in fields] for fields in baseline_cfg.get("fallback_key_fields") or [["global"]]]
        self.min_cell_n = int(baseline_cfg.get("min_cell_n", 32))
        self.shrink_to_global = bool(baseline_cfg.get("shrink_to_global", False))
        self.shrink_k = float(baseline_cfg.get("shrink_k", 64))
        self.global_return = _mean([outcome_return(row.get("outcome") or {}) for row in rows])
        self.primary = _groups(rows, self.key_fields)
        self.fallbacks = {tuple(fields): _groups(rows, fields) for fields in self.fallback_key_fields if fields != ["global"]}

    def _shrunk(self, mean_value: float, n: int) -> float:
        if not self.shrink_to_global:
            return mean_value

        return ((n * mean_value) + (self.shrink_k * self.global_return)) / max(1.0, n + self.shrink_k)

    def baseline_for(self, row: dict[str, Any]) -> tuple[float, str]:
        primary_key = build_group_key(row, self.key_fields)
        primary_values = self.primary.get(primary_key, [])
        if primary_values and (len(primary_values) >= self.min_cell_n or self.shrink_to_global):
            return self._shrunk(_mean(primary_values), len(primary_values)), primary_key
        for fields in self.fallback_key_fields:
            if fields == ["global"]:
                return self.global_return, GLOBAL_KEY
            key = build_group_key(row, fields)
            values = self.fallbacks.get(tuple(fields), {}).get(key, [])
            if values and len(values) >= self.min_cell_n:
                return self._shrunk(_mean(values), len(values)), key
        if primary_values:
            return self._shrunk(_mean(primary_values), len(primary_values)), primary_key

        return self.global_return, GLOBAL_KEY

    def summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        resolved = Counter(self.baseline_for(row)[1] for row in rows)

        return {
            "key_fields": list(self.key_fields),
            "fallback_key_fields": [list(fields) for fields in self.fallback_key_fields],
            "min_cell_n": self.min_cell_n,
            "shrink_to_global": self.shrink_to_global,
            "shrink_k": self.shrink_k,
            "global_return": round(self.global_return, 4),
            "primary_cells": _cell_summary(self.primary),
            "resolved_cells": [{"key": key, "rows": n} for key, n in resolved.most_common(12)],
        }


def _multiplier(mapping: dict[str, Any], value: Any) -> float:
    label = str(value or UNKNOWN)
    default = float(mapping.get("default", 1.0))

    return float(mapping.get(label, default))


def metadata_multiplier(row: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    if not config.get("enabled", False):
        return 1.0, {}
    multipliers = config.get("multipliers") or {}
    parts = {
        "phase": _multiplier(multipliers.get("phase") or {}, row.get("training_phase")),
        "bucket": _multiplier(multipliers.get("bucket") or {}, row.get("opponent_bucket")),
        "opponent_shell": _multiplier(multipliers.get("opponent_shell") or {}, row.get("opponent_shell")),
        "opponent_pilot": _multiplier(multipliers.get("opponent_pilot") or {}, row.get("opponent_pilot")),
    }
    product = 1.0
    for value in parts.values():
        product *= value

    return product, parts


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": round(sum(values) / len(values), 6),
        "median": round(float(statistics.median(values)), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _effective_sample_size(values: list[float]) -> float:
    denom = sum(v * v for v in values)
    if denom <= 0.0:
        return 0.0

    return (sum(values) ** 2) / denom


def _normalize_clipped_weights(values: list[float], *, target: float,
                               minimum: float, maximum: float) -> list[float]:
    if not values:
        return []
    adjusted = list(values)
    for _ in range(6):
        mean_value = sum(adjusted) / max(1, len(adjusted))
        if mean_value <= 0.0:
            break
        scale = target / mean_value
        adjusted = [max(minimum, min(maximum, value * scale)) for value in adjusted]

    return adjusted


def _contribution(rows: list[dict[str, Any]], weights: list[float], field: str) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row, weight in zip(rows, weights):
        key = str(row.get(field) or UNKNOWN)
        totals[key] += float(weight)
        counts[key] += 1
    total_weight = sum(totals.values()) or 1.0
    out = [
        {"value": key, "rows": counts[key], "weight_sum": round(value, 4), "weight_share": round(value / total_weight, 4)}
        for key, value in totals.items()
    ]
    out.sort(key=lambda item: (-float(item["weight_sum"]), str(item["value"])))

    return out


def compute_awr_weights(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    temperature: float,
    w_min: float,
    w_max: float,
    expected_loss_penalty: float,
) -> tuple[list[RowWeight], dict[str, Any]]:
    config = normalize_config(config)
    enriched = [enrich_refgame_row(dict(row)) for row in rows]
    baseline_model = BaselineModel(enriched, config)
    total_cfg = config.get("total_weight") or {}
    total_min = float(total_cfg.get("min", 0.0))
    total_max = float(total_cfg.get("max", 1.0e18))
    weights: list[RowWeight] = []
    meta_parts: list[dict[str, float]] = []
    for row in enriched:
        ret = outcome_return(row.get("outcome") or {})
        baseline, key = baseline_model.baseline_for(row)
        advantage = ret - baseline
        if ret < 0.0 and baseline > 0.0:
            advantage -= expected_loss_penalty * baseline
        w_awr = max(w_min, min(w_max, math.exp(advantage / max(1e-6, temperature))))
        w_meta, parts = metadata_multiplier(row, config)
        w_total = max(total_min, min(total_max, w_awr * w_meta))
        weights.append(RowWeight(ret, baseline, advantage, w_awr, w_meta, w_total, key))
        meta_parts.append(parts)

    normalize_target = total_cfg.get("normalize_mean")
    if config.get("enabled", False) and normalize_target is not None:
        target = float(normalize_target)
        adjusted = _normalize_clipped_weights(
            [w.w_total for w in weights],
            target=target,
            minimum=total_min,
            maximum=total_max,
        )
        weights = [
            replace(w, w_total=float(value))
            for w, value in zip(weights, adjusted)
        ]

    warnings = _warnings(enriched, weights, config)
    outcomes = Counter("win" if w.ret > 0 else "draw" if w.ret == 0 else "loss" for w in weights)
    total_values = [w.w_total for w in weights]
    stats = {
        "enabled": bool(config.get("enabled", False)),
        "rows": len(rows),
        "outcome_distribution": dict(outcomes),
        "baseline": baseline_model.summary(enriched),
        "weights": {
            "w_awr": _stats([w.w_awr for w in weights]),
            "w_meta": _stats([w.w_meta for w in weights]),
            "w_total": _stats(total_values),
            "effective_sample_size": round(_effective_sample_size(total_values), 4),
        },
        "weight_contribution": {
            "training_phase": _contribution(enriched, total_values, "training_phase"),
            "opponent_bucket": _contribution(enriched, total_values, "opponent_bucket"),
            "opponent_shell": _contribution(enriched, total_values, "opponent_shell"),
            "opponent_pilot": _contribution(enriched, total_values, "opponent_pilot"),
        },
        "kl": {"weight_kl": bool((config.get("kl") or {}).get("weight_kl", False))},
        "warnings": warnings,
    }

    return weights, stats


def _warnings(rows: list[dict[str, Any]], weights: list[RowWeight], config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not rows:
        return ["no rows"]
    for field in ("training_phase", "opponent_bucket", "opponent_shell", "opponent_pilot"):
        unknown = sum(1 for row in rows if str(row.get(field) or UNKNOWN) == UNKNOWN)
        frac = unknown / len(rows)
        if frac > 0.25:
            warnings.append(f"metadata field {field} is {frac:.1%} unknown")
    total_values = [w.w_total for w in weights]
    if any(not math.isfinite(v) for v in total_values):
        warnings.append("non-finite total weight detected")
    ess = _effective_sample_size(total_values)
    if weights and ess < max(8.0, 0.2 * len(weights)):
        warnings.append(f"low effective sample size: {ess:.2f}/{len(weights)}")
    by_key: dict[str, float] = defaultdict(float)
    for w in weights:
        by_key[w.key] += w.w_total
    total = sum(by_key.values()) or 1.0
    if by_key and max(by_key.values()) / total > 0.5:
        warnings.append("one baseline cell contributes more than 50% of total weight")
    target = (config.get("total_weight") or {}).get("normalize_mean")
    if target is not None and total_values:
        mean_value = sum(total_values) / len(total_values)
        if abs(mean_value - float(target)) > 1e-4:
            warnings.append(f"total weights mean {mean_value:.6f} != target {float(target):.6f}")

    return warnings
