"""Adversarial best-response: pick the strongest exploiter of a FROZEN candidate.

The curriculum's `adversarial` tier is an opponent that best-responds to *our current
candidate* -- the AlphaStar main-exploiter idea. This module holds the pure
ranking/selection logic (no engine): given per-(brain, deck) measurements of how
often an exploiter BEAT the frozen candidate, it ranks brains by mean win-vs-candidate,
selects the strongest as the adversarial opponent, and surfaces the single most-
exploiting (brain, deck) combos as structural counters.

Two ways an adversarial exploiter is produced, both INVARIANT #2-safe because the
target is a FIXED candidate snapshot (best-response, never co-adapting to equilibrium):
  - SELECT: argmax over an existing frozen brain pool (checkpoints + heuristics).
    Cheap, works today, and is how AWR's output gets chosen once it exists.
  - TRAIN:  a gradient best-response (AWR) over a corpus generated vs the frozen
    candidate (see scripts/run_r19_best_response_datagen.py). This module ranks/
    registers whatever checkpoint that produces.

Offline/eval-only (INVARIANT #8): never imported by `main.py`/`cg`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExploiterResult:
    """How one exploiter brain, piloting one deck, fared vs the frozen candidate."""

    brain: str            # display label (checkpoint stem or spec, e.g. "greedy")
    ref: str              # checkpoint path or spec text, for league registration/wiring
    deck_hash: str
    shell: str
    bucket: str
    games: int
    exploiter_wins: int   # games the exploiter BEAT the candidate (the objective)
    candidate_wins: int

    @property
    def decided(self) -> int:
        return self.exploiter_wins + self.candidate_wins

    @property
    def win_vs_candidate(self) -> float:
        return self.exploiter_wins / self.decided if self.decided else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "brain": self.brain, "ref": self.ref, "deck_hash": self.deck_hash,
            "shell": self.shell, "bucket": self.bucket, "games": self.games,
            "exploiter_wins": self.exploiter_wins, "candidate_wins": self.candidate_wins,
            "win_vs_candidate": round(self.win_vs_candidate, 4),
        }


@dataclass(frozen=True)
class BrainRanking:
    """A brain's aggregate exploitation of the candidate, across decks."""

    brain: str
    ref: str
    decks: int
    decided: int
    exploiter_wins: int
    mean_win_vs_candidate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "brain": self.brain, "ref": self.ref, "decks": self.decks,
            "decided": self.decided, "exploiter_wins": self.exploiter_wins,
            "mean_win_vs_candidate": round(self.mean_win_vs_candidate, 4),
        }


def rank_brains(results: list[ExploiterResult]) -> list[BrainRanking]:
    """Aggregate results per brain, ranked by pooled win-vs-candidate (desc).

    Pooled (game-weighted) rather than mean-of-means so a brain measured on more
    decks isn't diluted by a single tiny-sample deck."""
    grouped: dict[tuple[str, str], list[ExploiterResult]] = defaultdict(list)
    for r in results:
        grouped[(r.brain, r.ref)].append(r)
    rankings: list[BrainRanking] = []
    for (brain, ref), group in grouped.items():
        decided = sum(r.decided for r in group)
        wins = sum(r.exploiter_wins for r in group)
        rankings.append(BrainRanking(
            brain=brain, ref=ref, decks=len(group), decided=decided,
            exploiter_wins=wins,
            mean_win_vs_candidate=(wins / decided) if decided else 0.0,
        ))
    rankings.sort(key=lambda b: (-b.mean_win_vs_candidate, -b.decided, b.brain))
    return rankings


def select_adversarial(results: list[ExploiterResult], *, min_decided: int = 1) -> BrainRanking | None:
    """The strongest brain that beat the candidate on enough games. None if no brain
    clears `min_decided` (don't crown an exploiter on noise)."""
    for ranking in rank_brains(results):
        if ranking.decided >= min_decided:
            return ranking
    return None


def top_counters(results: list[ExploiterResult], *, k: int = 3, min_decided: int = 1) -> list[ExploiterResult]:
    """The single most-exploiting (brain, deck) combos -- structural counters to add
    to the curriculum's counter list (a specific deck a specific brain crushes us with)."""
    eligible = [r for r in results if r.decided >= min_decided]
    eligible.sort(key=lambda r: (-r.win_vs_candidate, -r.decided, r.deck_hash))
    return eligible[:k]


def best_response_summary(results: list[ExploiterResult], *, min_decided: int = 1) -> dict[str, Any]:
    rankings = rank_brains(results)
    best = select_adversarial(results, min_decided=min_decided)
    return {
        "brain_rankings": [b.to_json() for b in rankings],
        "adversarial": best.to_json() if best else None,
        "top_counters": [r.to_json() for r in top_counters(results, min_decided=min_decided)],
        "n_results": len(results),
    }
