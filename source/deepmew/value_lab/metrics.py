"""Validated probabilistic metrics for actor-relative value models."""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-6


def _validated_probabilities(p: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if p.ndim != 1 or z.ndim != 1 or p.shape != z.shape:
        raise ValueError("p and z must be aligned one-dimensional arrays")
    if len(p) == 0:
        raise ValueError("probability metrics require at least one row")
    if not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError("p and z must contain only finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be on the probability scale [0, 1]")
    if np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("z must be on the actor-relative outcome scale [0, 1]")

    return p, z


def _loss_rows(p: np.ndarray, z: np.ndarray, metric: str) -> np.ndarray:
    p, z = _validated_probabilities(p, z)
    if metric == "log_loss":
        clipped = np.clip(p, EPS, 1.0 - EPS)

        return -(z * np.log(clipped) + (1.0 - z) * np.log(1.0 - clipped))
    if metric == "brier":
        return (p - z) ** 2
    raise ValueError(f"unsupported bootstrap metric {metric}")


def log_loss(p: np.ndarray, z: np.ndarray) -> float:
    p, z = _validated_probabilities(p, z)
    p = np.clip(p, EPS, 1.0 - EPS)

    return float(-(z * np.log(p) + (1.0 - z) * np.log(1.0 - p)).mean())


def brier(p: np.ndarray, z: np.ndarray) -> float:
    p, z = _validated_probabilities(p, z)

    return float(((p - z) ** 2).mean())


def auc(p: np.ndarray, z: np.ndarray) -> float:
    """Rank-based AUC over decided outcomes (draws excluded)."""
    p, z = _validated_probabilities(p, z)
    decided = (z <= 0.0) | (z >= 1.0)
    p, z = p[decided], z[decided]
    pos = z >= 1.0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties
    sorted_p = p[order]
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    tie_start = 0
    for i in range(1, len(p) + 1):
        if i == len(p) or sorted_p[i] != sorted_p[tie_start]:
            if i - tie_start > 1:
                mean_rank = 0.5 * (tie_start + 1 + i)
                ranks[order[tie_start:i]] = mean_rank
            tie_start = i

    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(p: np.ndarray, z: np.ndarray, bins: int = 15) -> dict:
    p, z = _validated_probabilities(p, z)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = len(p)
    value = 0.0
    reliability = []
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(z[mask].mean())
        value += (count / total) * abs(conf - acc)
        reliability.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                            "count": count, "mean_p": conf, "mean_z": acc})

    return {"ece": float(value), "bins": reliability}


def calibration_slope_intercept(p: np.ndarray, z: np.ndarray) -> dict:
    """Logistic recalibration fit: z ~ sigmoid(a * logit(p) + b)."""
    p, z = _validated_probabilities(p, z)
    decided = (z <= 0.0) | (z >= 1.0)
    p_d = np.clip(p[decided].astype(np.float64), EPS, 1.0 - EPS)
    z_d = (z[decided] >= 1.0).astype(np.float64)
    if len(p_d) < 100 or z_d.min() == z_d.max():
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "calibration_fit_status": "insufficient_rows_or_classes",
            "calibration_fit_iterations": 0,
            "calibration_decided_n": int(len(p_d)),
            "calibration_draw_n": int(len(p) - len(p_d)),
        }
    x = np.log(p_d / (1.0 - p_d))
    if float(np.ptp(x)) < 1e-12:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "calibration_fit_status": "unidentifiable_constant_prediction",
            "calibration_fit_iterations": 0,
            "calibration_decided_n": int(len(p_d)),
            "calibration_draw_n": int(len(p) - len(p_d)),
        }
    a, b = 1.0, 0.0
    status = "max_iterations"
    iterations = 0
    for iteration in range(1, 51):
        iterations = iteration
        eta = np.clip(a * x + b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        grad_a = ((mu - z_d) * x).mean()
        grad_b = (mu - z_d).mean()
        h_aa = (w * x * x).mean()
        h_ab = (w * x).mean()
        h_bb = w.mean()
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            status = "singular_design"
            break
        da = (h_bb * grad_a - h_ab * grad_b) / det
        db = (h_aa * grad_b - h_ab * grad_a) / det
        a, b = a - da, b - db
        if not math.isfinite(a) or not math.isfinite(b):
            status = "numerical_failure"
            break
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            status = "converged"
            break

    if status != "converged":
        a = b = float("nan")

    return {
        "slope": float(a),
        "intercept": float(b),
        "calibration_fit_status": status,
        "calibration_fit_iterations": iterations,
        "calibration_decided_n": int(len(p_d)),
        "calibration_draw_n": int(len(p) - len(p_d)),
    }


def full_metrics(p: np.ndarray, z: np.ndarray, *, bins: int = 15) -> dict:
    p, z = _validated_probabilities(p, z)
    cal = ece(p, z, bins=bins)

    return {
        "n": int(len(p)),
        "log_loss": log_loss(p, z),
        "brier": brier(p, z),
        "auc": auc(p, z),
        "ece": cal["ece"],
        "reliability": cal["bins"],
        **calibration_slope_intercept(p, z),
        "mean_p": float(p.mean()) if len(p) else float("nan"),
        "mean_z": float(z.mean()) if len(z) else float("nan"),
    }


def slice_metrics(p: np.ndarray, z: np.ndarray, keys: np.ndarray, *, min_n: int = 500) -> dict:
    """Metrics per unique key (keys: int or str array aligned with p/z)."""
    p, z = _validated_probabilities(p, z)
    keys = np.asarray(keys)
    if keys.ndim != 1 or keys.shape != p.shape:
        raise ValueError("keys must be one-dimensional and aligned with p/z")
    if isinstance(min_n, bool) or not isinstance(min_n, int) or min_n < 1:
        raise ValueError("min_n must be a positive integer")
    out = {}
    for key in np.unique(keys):
        mask = keys == key
        if int(mask.sum()) < min_n:
            continue
        sub_p, sub_z = p[mask], z[mask]
        out[str(key)] = {
            "n": int(mask.sum()),
            "log_loss": log_loss(sub_p, sub_z),
            "brier": brier(sub_p, sub_z),
            "auc": auc(sub_p, sub_z),
            "ece": ece(sub_p, sub_z)["ece"],
            "mean_z": float(sub_z.mean()),
        }

    return out


def game_cluster_bootstrap(
    p: np.ndarray,
    z: np.ndarray,
    game_ids: np.ndarray,
    *,
    metric: str,
    replicates: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap a row-weighted metric by resampling whole games.

    Outcomes repeat within a game, so rows are not independent sampling units.
    Each bootstrap replicate samples games with replacement and then aggregates
    all eligible rows belonging to each sampled game.
    """
    losses = _loss_rows(p, z, metric)

    return _cluster_bootstrap_from_losses(losses, game_ids, replicates=replicates, seed=seed)


def paired_game_cluster_bootstrap(
    candidate_p: np.ndarray,
    reference_p: np.ndarray,
    z: np.ndarray,
    game_ids: np.ndarray,
    *,
    metric: str,
    replicates: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired whole-game bootstrap of candidate minus reference loss."""
    candidate_loss = _loss_rows(candidate_p, z, metric)
    reference_loss = _loss_rows(reference_p, z, metric)
    if candidate_loss.shape != reference_loss.shape:
        raise ValueError("candidate and reference predictions must be aligned")
    result = _cluster_bootstrap_from_losses(
        candidate_loss - reference_loss,
        game_ids,
        replicates=replicates,
        seed=seed,
    )
    result["delta_candidate_minus_reference"] = result.pop("estimate")

    return result


def _cluster_bootstrap_from_losses(
    losses: np.ndarray,
    game_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    game_ids = np.asarray(game_ids)
    if game_ids.ndim != 1 or game_ids.shape != losses.shape:
        raise ValueError("game_ids must be one-dimensional and aligned with predictions")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer of at least two")
    _, inverse = np.unique(game_ids, return_inverse=True)
    game_count = int(inverse.max()) + 1
    if game_count < 2:
        raise ValueError("game-cluster bootstrap requires at least two games")
    loss_sums = np.bincount(inverse, weights=losses, minlength=game_count)
    row_counts = np.bincount(inverse, minlength=game_count).astype(np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    chunk_size = max(1, min(128, replicates))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        sampled_games = rng.integers(0, game_count, size=(stop - start, game_count))
        numerators = loss_sums[sampled_games].sum(axis=1)
        denominators = row_counts[sampled_games].sum(axis=1)
        samples[start:stop] = numerators / denominators
    lower, upper = np.quantile(samples, [0.025, 0.975])

    return {
        "estimate": float(losses.mean()),
        "standard_error": float(samples.std(ddof=1)),
        "ci95": [float(lower), float(upper)],
        "games": game_count,
        "rows": int(len(losses)),
        "replicates": replicates,
        "seed": int(seed),
    }
