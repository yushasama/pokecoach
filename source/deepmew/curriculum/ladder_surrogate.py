"""Eval-only ladder trajectory surrogate.

This is not Kaggle ladder truth. It converts local bucket-arena matchup rows into
a coarse trajectory stress test so candidates can be compared on volatility and
wall-crossing risk before any real ladder submission.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatchupRate:
    bucket: str
    key: str
    win: float
    draw: float
    loss: float


def load_matchup_rates(path: str | Path) -> list[MatchupRate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("matchup input must be a bucket-arena results.json or a list of rows")
    rates: list[MatchupRate] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        wins = float(row.get("wins", 0) or 0)
        losses = float(row.get("losses", 0) or 0)
        draws = float(row.get("draws", 0) or 0)
        total = wins + losses + draws
        if total <= 0.0 and "winrate" in row:
            win = max(0.0, min(1.0, float(row["winrate"])))
            draw = 0.0
            loss = 1.0 - win
        elif total > 0.0:
            win = wins / total
            draw = draws / total
            loss = losses / total
        else:
            continue
        key = "|".join(str(row.get(k) or "") for k in ("bucket", "shell", "pilot", "deck_hash")) or f"row-{i}"
        rates.append(MatchupRate(bucket=str(row.get("bucket") or "unknown"), key=key, win=win, draw=draw, loss=loss))
    if not rates:
        raise ValueError(f"no usable matchup rows in {path}")

    return rates


def _bucket_for_mu(mu: float) -> str:
    if mu < 800:
        return "600-800"
    if mu < 900:
        return "800-900"
    if mu < 1000:
        return "900-1000"

    return "1000-1100"


def _sample_rate(rng: random.Random, rates: list[MatchupRate], mu: float) -> MatchupRate:
    bucket = _bucket_for_mu(mu)
    candidates = [rate for rate in rates if rate.bucket == bucket]
    if not candidates and bucket == "1000-1100":
        candidates = [rate for rate in rates if rate.bucket in {"1000-1100", "1100-1200", "1000_plus"}]
    if not candidates:
        candidates = rates

    return rng.choice(candidates)


def _sample_result(rng: random.Random, rate: MatchupRate) -> str:
    roll = rng.random()
    if roll < rate.win:
        return "win"
    if roll < rate.win + rate.draw:
        return "draw"

    return "loss"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None

    return round(sum(values) / len(values), 4)


def simulate_ladder_surrogate(
    rates: list[MatchupRate],
    *,
    episodes: int = 5000,
    games: int = 64,
    start_mu: float = 600.0,
    start_sigma: float = 120.0,
    min_sigma: float = 35.0,
    sigma_decay: float = 0.985,
    base_k: float = 28.0,
    seed: int = 1901,
) -> dict[str, Any]:
    rng = random.Random(seed)
    finals: list[float] = []
    mu8: list[float] = []
    mu32: list[float] = []
    reaches = {800: 0, 900: 0, 1000: 0, 1100: 0}
    zero_two = 0
    two_six_or_worse = 0
    draw_count = 0
    total_games = 0
    collapse_after_900 = 0
    reached_900_count = 0
    for _ in range(int(episodes)):
        mu = float(start_mu)
        sigma = float(start_sigma)
        first_two_losses = 0
        first8_wins = 0
        reached_900 = False
        reached_flags = {threshold: False for threshold in reaches}
        for game_index in range(1, int(games) + 1):
            rate = _sample_rate(rng, rates, mu)
            result = _sample_result(rng, rate)
            scale = max(0.15, sigma / max(1e-6, start_sigma))
            delta = base_k * scale
            if result == "win":
                mu += delta
                if game_index <= 8:
                    first8_wins += 1
            elif result == "loss":
                mu -= delta
                if game_index <= 2:
                    first_two_losses += 1
            else:
                draw_count += 1
                mu += 0.10 * delta * (800.0 - mu) / 200.0
            sigma = max(float(min_sigma), sigma * float(sigma_decay))
            total_games += 1
            if game_index == 8:
                mu8.append(mu)
            if game_index == 32:
                mu32.append(mu)
            if mu >= 900.0:
                reached_900 = True
            for threshold in reached_flags:
                if mu >= threshold:
                    reached_flags[threshold] = True
        finals.append(mu)
        if first_two_losses == 2:
            zero_two += 1
        if first8_wins <= 2:
            two_six_or_worse += 1
        for threshold, was_reached in reached_flags.items():
            reaches[threshold] += int(was_reached)
        if reached_900:
            reached_900_count += 1
            if mu < 900.0:
                collapse_after_900 += 1

    finals_sorted = sorted(finals)
    p50 = finals_sorted[len(finals_sorted) // 2] if finals_sorted else start_mu

    return {
        "schema_version": "deepmew.ladder_surrogate.v0",
        "note": "Eval-only surrogate. Not Kaggle ladder truth. Score margin ignored.",
        "config": {
            "episodes": int(episodes),
            "games": int(games),
            "start_mu": float(start_mu),
            "start_sigma": float(start_sigma),
            "min_sigma": float(min_sigma),
            "sigma_decay": float(sigma_decay),
            "base_k": float(base_k),
            "seed": int(seed),
        },
        "matchup_cells": len(rates),
        "expected_final_mu": round(sum(finals) / max(1, len(finals)), 4),
        "median_final_mu": round(p50, 4),
        "mu_after_8": _mean_or_none(mu8),
        "mu_after_32": _mean_or_none(mu32),
        "final_mu_se": round(statistics.pstdev(finals) / max(1, len(finals)) ** 0.5, 4) if len(finals) > 1 else 0.0,
        "P_reach_800": round(reaches[800] / max(1, episodes), 6),
        "P_reach_900": round(reaches[900] / max(1, episodes), 6),
        "P_reach_1000": round(reaches[1000] / max(1, episodes), 6),
        "P_reach_1100": round(reaches[1100] / max(1, episodes), 6),
        "P_0_2_start": round(zero_two / max(1, episodes), 6),
        "P_2_6_or_worse_first_8": round(two_six_or_worse / max(1, episodes), 6),
        "draw_rate": round(draw_count / max(1, total_games), 6),
        "collapse_rate_after_900": round(collapse_after_900 / max(1, reached_900_count), 6),
    }
