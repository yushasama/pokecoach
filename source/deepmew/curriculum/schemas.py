"""Curriculum-arena schemas: difficulty rungs, config, per-rung results, league.

A *curriculum* here is an ordered ladder of opponent-difficulty rungs. Each rung
composes the four independent "harder opponent" levers:

  1. DECK BAND   -- which ELO bucket(s) the opponent decks come from (harder decks)
  2. PILOT TIERS -- which brains pilot those decks, ascending in strength:
                    first-legal < v2.4 < replay < exploiter < adversarial
  3. COUNTERS    -- structural hard-counter shells injected as extra opponents
  4. GATE        -- the target-band win rate needed to advance to the next rung

As the candidate passes rung K's gate, rung K+1 unlocks harder decks + stronger
pilots + more counters. The league (see league.py) grows across ROUNDS by freezing
the passing candidate + any freshly trained exploiter as new frozen opponents.

INVARIANT #2 (load-bearing): every opponent is a FROZEN snapshot. Exploiters are
best-response to a fixed candidate snapshot, then frozen; the league only grows by
ADDING frozen members. No opponent co-adapts continuously with the candidate and
no value/policy is trained toward a self-play equilibrium -- the population is a
generator only. The ladder module enforces this; do not weaken it.

Offline/eval-only (INVARIANT #8): never imported by `main.py`/`cg`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Pilot brains in ascending difficulty. A rung enables a subset (usually a
# suffix window) of these. Kept as an explicit tuple so difficulty is orderable.
PILOT_TIERS: tuple[str, ...] = ("first-legal", "v2.4", "replay", "exploiter", "adversarial")

# League member kinds that are legal frozen opponents under INVARIANT #2.
FROZEN_KINDS: frozenset[str] = frozenset({"mined_deck", "candidate_snapshot", "exploiter", "counter_deck"})


def pilot_rank(tier: str) -> int:
    """Difficulty rank of a pilot tier (higher = harder). Unknown tiers sort last-but-known-invalid."""
    try:
        return PILOT_TIERS.index(tier)
    except ValueError:
        return -1


@dataclass(frozen=True)
class CurriculumRung:
    """One difficulty step. Ascends in band + pilot strength + counters."""

    rung_id: int
    name: str
    target_bucket: str
    pilot_tiers: tuple[str, ...]
    gate: float
    top_n: int = 8
    carry_n: int = 4
    exploiter_top: int = 8
    counter_shells: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.pilot_tiers:
            raise ValueError(f"rung {self.rung_id} has no pilot tiers")
        for tier in self.pilot_tiers:
            if tier not in PILOT_TIERS:
                raise ValueError(f"rung {self.rung_id}: unknown pilot tier {tier!r}; expected {PILOT_TIERS}")
        if not 0.0 <= self.gate <= 1.0:
            raise ValueError(f"rung {self.rung_id}: gate {self.gate} out of [0,1]")

    @property
    def hardest_tier(self) -> str:
        return max(self.pilot_tiers, key=pilot_rank)

    @property
    def difficulty(self) -> int:
        """A coarse ordinal: hardest pilot rank dominates, counters break ties."""
        return pilot_rank(self.hardest_tier) * 10 + min(len(self.counter_shells), 9)

    def to_json(self) -> dict[str, Any]:
        return {
            "rung_id": self.rung_id,
            "name": self.name,
            "target_bucket": self.target_bucket,
            "pilot_tiers": list(self.pilot_tiers),
            "gate": self.gate,
            "top_n": self.top_n,
            "carry_n": self.carry_n,
            "exploiter_top": self.exploiter_top,
            "counter_shells": list(self.counter_shells),
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CurriculumRung":
        return cls(
            rung_id=int(data["rung_id"]),
            name=str(data["name"]),
            target_bucket=str(data["target_bucket"]),
            pilot_tiers=tuple(data["pilot_tiers"]),
            gate=float(data["gate"]),
            top_n=int(data.get("top_n", 8)),
            carry_n=int(data.get("carry_n", 4)),
            exploiter_top=int(data.get("exploiter_top", 8)),
            counter_shells=tuple(data.get("counter_shells", ())),
        )


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered rungs. Rung ids must be contiguous 0..n-1 and non-decreasing difficulty."""

    rungs: tuple[CurriculumRung, ...]

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("curriculum has no rungs")
        for expected, rung in enumerate(self.rungs):
            if rung.rung_id != expected:
                raise ValueError(f"rung ids must be contiguous from 0; got {rung.rung_id} at position {expected}")
        # Difficulty must be monotonic non-decreasing so "as the policy improves,
        # add more/harder opponents" holds by construction.
        for prev, cur in zip(self.rungs, self.rungs[1:]):
            if cur.difficulty < prev.difficulty:
                raise ValueError(
                    f"rung {cur.rung_id} is easier than {prev.rung_id} "
                    f"({cur.difficulty} < {prev.difficulty}); curriculum must not regress difficulty"
                )

    def rung(self, rung_id: int) -> CurriculumRung:
        return self.rungs[rung_id]

    def next_rung(self, rung_id: int) -> CurriculumRung | None:
        nxt = rung_id + 1
        return self.rungs[nxt] if 0 <= nxt < len(self.rungs) else None

    def to_json(self) -> dict[str, Any]:
        return {"rungs": [r.to_json() for r in self.rungs]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CurriculumConfig":
        return cls(rungs=tuple(CurriculumRung.from_json(r) for r in data["rungs"]))


@dataclass(frozen=True)
class RungResult:
    """Outcome of measuring a candidate against one rung."""

    rung_id: int
    target_bucket: str
    target_winrate: float
    overall_winrate: float
    gate: float
    passed: bool
    safety: dict[str, int]
    # deck_hash:pilot -> our winrate, for PFSP reweighting of the next round.
    per_opponent_winrate: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "rung_id": self.rung_id,
            "target_bucket": self.target_bucket,
            "target_winrate": round(self.target_winrate, 4),
            "overall_winrate": round(self.overall_winrate, 4),
            "gate": self.gate,
            "passed": self.passed,
            "safety": dict(self.safety),
            "per_opponent_winrate": {k: round(v, 4) for k, v in self.per_opponent_winrate.items()},
        }


@dataclass(frozen=True)
class LeagueMember:
    """A frozen opponent in the league. Frozen is the whole point (INVARIANT #2)."""

    member_id: str
    kind: str            # one of FROZEN_KINDS
    round_added: int     # the training round at which this snapshot was frozen
    ref: str             # checkpoint path / bundle dir / deck_hash / shell
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FROZEN_KINDS:
            raise ValueError(
                f"league member {self.member_id!r} kind {self.kind!r} not in FROZEN_KINDS "
                f"{sorted(FROZEN_KINDS)} -- a non-frozen (co-adapting) opponent violates INVARIANT #2"
            )
        if self.round_added < 0:
            raise ValueError(f"league member {self.member_id!r} has negative round_added")

    def to_json(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "kind": self.kind,
            "round_added": self.round_added,
            "ref": self.ref,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LeagueMember":
        return cls(
            member_id=str(data["member_id"]),
            kind=str(data["kind"]),
            round_added=int(data["round_added"]),
            ref=str(data["ref"]),
            note=str(data.get("note", "")),
        )
