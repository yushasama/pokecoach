"""Progressive curriculum stages over ELO-bucket deck pools.

Stage k targets one bucket and, per the R17 curriculum contract, carries the
most-played decks of every earlier bucket too ("with each bucket we pass,
throw in previous buckets"). Opponent weights follow game share within their
bucket; a PFSP reweighting hook (AlphaStar-style: prefer opponents that beat
the candidate) is applied by the runner AFTER a measurement pass, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepmew.bucket_arena.pool import DeckPoolEntry

STAGE_ORDER = ("600-800", "800-900", "900-1000", "1000-1100", "1100-1200", "1200-1300")


@dataclass(frozen=True)
class StageOpponent:
    bucket: str
    deck_hash: str
    card_ids: tuple[int, ...]
    games_in_bucket: int
    bucket_win_rate: float
    carried: bool  # True if pulled in from an earlier (already-passed) bucket

    def to_json(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "deck_hash": self.deck_hash,
            "card_ids": list(self.card_ids),
            "games_in_bucket": self.games_in_bucket,
            "bucket_win_rate": round(self.bucket_win_rate, 4),
            "carried": self.carried,
        }


def build_stage(
    pools: dict[str, list[DeckPoolEntry]],
    target_bucket: str,
    *,
    top_n: int = 8,
    carry_n: int = 4,
    include_previous: bool = True,
    min_games: int = 2,
) -> list[StageOpponent]:
    """Top-`top_n` decks of the target bucket plus top-`carry_n` of each earlier one."""
    if target_bucket not in STAGE_ORDER:
        raise ValueError(f"unknown bucket {target_bucket!r}; expected one of {STAGE_ORDER}")
    target_index = STAGE_ORDER.index(target_bucket)
    opponents: list[StageOpponent] = []
    seen: set[str] = set()
    for index in range(target_index + 1):
        bucket = STAGE_ORDER[index]
        if bucket != target_bucket and not include_previous:
            continue
        limit = top_n if bucket == target_bucket else carry_n
        entries = [e for e in pools.get(bucket, []) if e.games >= min_games]
        for entry in entries[:limit]:
            if entry.deck_hash in seen:
                continue
            seen.add(entry.deck_hash)
            opponents.append(
                StageOpponent(
                    bucket=bucket,
                    deck_hash=entry.deck_hash,
                    card_ids=tuple(entry.card_ids),
                    games_in_bucket=entry.games,
                    bucket_win_rate=entry.win_rate,
                    carried=bucket != target_bucket,
                )
            )
    return opponents


def pfsp_weights(win_rates: dict[str, float], *, floor: float = 0.1, power: float = 1.0) -> dict[str, float]:
    """AlphaStar-style prioritized fictitious self-play weights: opponents we
    LOSE to get proportionally more play. `win_rates` maps deck_hash -> our
    winrate vs that opponent from the last measurement pass."""
    raw = {key: max(floor, (1.0 - wr) ** power) for key, wr in win_rates.items()}
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}
