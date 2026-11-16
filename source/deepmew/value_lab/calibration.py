"""Simple JSON-serializable probability calibrators for value models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from deepmew.value_lab.metrics import EPS, log_loss


CALIBRATOR_SCHEMA_VERSION = "deepmew.value_probability_calibrator.v1"
CALIBRATOR_METHODS = ("identity", "temperature", "isotonic")


def _validated_arrays(p: np.ndarray, z: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p must be a nonempty one-dimensional array")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must contain finite probabilities in [0, 1]")
    if z is None:
        return p, None
    z = np.asarray(z, dtype=np.float64)
    if z.shape != p.shape or z.ndim != 1:
        raise ValueError("z must be one-dimensional and aligned with p")
    if not np.isfinite(z).all() or np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("z must contain finite actor-relative outcomes in [0, 1]")

    return p, z


def _logits(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPS, 1.0 - EPS)

    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -40.0, 40.0)

    return 1.0 / (1.0 + np.exp(-logits))


def temperature_transform(p: np.ndarray, temperature: float) -> np.ndarray:
    p, _ = _validated_arrays(p)
    if isinstance(temperature, bool):
        raise ValueError("temperature must be a finite positive float")
    try:
        value = float(temperature)
    except (TypeError, ValueError) as error:
        raise ValueError("temperature must be a finite positive float") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be a finite positive float")

    return _sigmoid(_logits(p) / value)


def fit_temperature(p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Fit one positive temperature by deterministic one-dimensional search."""
    p, z = _validated_arrays(p, z)
    assert z is not None
    logits = _logits(p)

    def objective(log_temperature: float) -> float:
        calibrated = _sigmoid(logits / math.exp(log_temperature))

        return log_loss(calibrated, z)

    if float(np.ptp(logits)) < 1e-12:
        best_log_temperature = 0.0
    else:
        grid = np.linspace(-4.0, 4.0, 161)
        values = np.asarray([objective(float(point)) for point in grid])
        best_index = int(np.argmin(values))
        lo = float(grid[max(0, best_index - 1)])
        hi = float(grid[min(len(grid) - 1, best_index + 1)])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left = hi - ratio * (hi - lo)
        right = lo + ratio * (hi - lo)
        left_value = objective(left)
        right_value = objective(right)
        for _ in range(64):
            if left_value <= right_value:
                hi = right
                right = left
                right_value = left_value
                left = hi - ratio * (hi - lo)
                left_value = objective(left)
            else:
                lo = left
                left = right
                left_value = right_value
                right = lo + ratio * (hi - lo)
                right_value = objective(right)
        best_log_temperature = 0.5 * (lo + hi)
    temperature = math.exp(best_log_temperature)
    fitted = temperature_transform(p, temperature)

    return {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "method": "temperature",
        "temperature": float(temperature),
        "fit_rows": int(len(p)),
        "input_log_loss": log_loss(p, z),
        "fitted_log_loss": log_loss(fitted, z),
    }


def fit_isotonic(p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Fit nondecreasing probabilities with the pool-adjacent-violators algorithm."""
    p, z = _validated_arrays(p, z)
    assert z is not None
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_z = z[order]
    thresholds, first, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_z, first)
    blocks: list[list[float | int]] = []
    for index, (weight, outcome_sum) in enumerate(zip(counts, sums)):
        blocks.append([index, index, int(weight), float(outcome_sum)])
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous[3]) / int(previous[2])
            current_mean = float(current[3]) / int(current[2])
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [[
                int(previous[0]),
                int(current[1]),
                int(previous[2]) + int(current[2]),
                float(previous[3]) + float(current[3]),
            ]]
    fitted_unique = np.empty(len(thresholds), dtype=np.float64)
    for start, stop, weight, outcome_sum in blocks:
        fitted_unique[int(start):int(stop) + 1] = float(outcome_sum) / int(weight)
    fitted = np.interp(p, thresholds, fitted_unique)

    return {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "method": "isotonic",
        "thresholds": thresholds.tolist(),
        "values": fitted_unique.tolist(),
        "fit_rows": int(len(p)),
        "fit_unique_probabilities": int(len(thresholds)),
        "fit_blocks": int(len(blocks)),
        "input_log_loss": log_loss(p, z),
        "fitted_log_loss": log_loss(fitted, z),
    }


def identity_calibrator() -> dict[str, Any]:
    return {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "method": "identity",
    }


def apply_calibrator(p: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    p, _ = _validated_arrays(p)
    if calibrator.get("schema_version") != CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("calibrator schema is missing or stale")
    method = calibrator.get("method")
    if method == "identity":
        return p.copy()
    if method == "temperature":
        return temperature_transform(p, calibrator.get("temperature"))
    if method == "isotonic":
        thresholds = np.asarray(calibrator.get("thresholds"), dtype=np.float64)
        values = np.asarray(calibrator.get("values"), dtype=np.float64)
        if (
            thresholds.ndim != 1
            or values.shape != thresholds.shape
            or len(thresholds) == 0
            or not np.isfinite(thresholds).all()
            or not np.isfinite(values).all()
            or np.any(np.diff(thresholds) <= 0.0)
            or np.any(np.diff(values) < 0.0)
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise ValueError("isotonic calibrator thresholds/values are invalid")

        return np.interp(p, thresholds, values)
    raise ValueError(f"unknown calibrator method {method}")
