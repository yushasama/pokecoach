"""League registry: the mechanism by which opponents get *harder over rounds*.

This is the AlphaStar-league / PFSP frame the R17 arena already adopts. The league
is a growing set of FROZEN opponents. Each training round:

  1. The candidate is measured against the current league (fixed).
  2. If it passes a rung, the passing candidate is FROZEN and added as a new
     `candidate_snapshot` opponent -- so next round's candidate must beat a copy of
     this round's best. This is how the population "further improves" without any
     equilibrium training (INVARIANT #2): we accumulate frozen best-responses.
  3. Optionally a fresh `exploiter` (best-response trained vs the frozen candidate)
     is added the same way.

PFSP weights (opponents we lose to get more play) come from the existing
`deepmew.bucket_arena.pfsp_weights` so the trainer samples the hard opponents more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepmew.bucket_arena import pfsp_weights
from deepmew.curriculum.ladder import guard_invariant_2
from deepmew.curriculum.schemas import LeagueMember


@dataclass
class League:
    """A growing set of frozen opponents plus the current round counter."""

    members: list[LeagueMember] = field(default_factory=list)
    round: int = 0

    def _ids(self) -> set[str]:
        return {m.member_id for m in self.members}

    def add(self, member: LeagueMember) -> LeagueMember:
        if member.member_id in self._ids():
            raise ValueError(f"duplicate league member id {member.member_id!r}")
        self.members.append(member)
        guard_invariant_2(self.members, self.round)
        return member

    def freeze_candidate(self, ref: str, *, note: str = "") -> LeagueMember:
        """Freeze the current-round candidate as a permanent opponent snapshot."""
        member = LeagueMember(
            member_id=f"candidate-r{self.round}",
            kind="candidate_snapshot",
            round_added=self.round,
            ref=ref,
            note=note or f"frozen passing candidate from round {self.round}",
        )
        return self.add(member)

    def add_exploiter(self, ref: str, *, note: str = "") -> LeagueMember:
        """Add a best-response exploiter trained vs a frozen candidate snapshot."""
        member = LeagueMember(
            member_id=f"exploiter-r{self.round}",
            kind="exploiter",
            round_added=self.round,
            ref=ref,
            note=note or f"best-response exploiter frozen at round {self.round}",
        )
        return self.add(member)

    def start_next_round(self) -> int:
        """Advance the round counter. New members added after this are round+1."""
        self.round += 1
        return self.round

    def members_of_kind(self, kind: str) -> list[LeagueMember]:
        return [m for m in self.members if m.kind == kind]

    def pfsp(self, win_rates: dict[str, float], *, floor: float = 0.1, power: float = 1.0) -> dict[str, float]:
        """Delegate to the shared PFSP weighting (opponents we lose to get more play)."""
        return pfsp_weights(win_rates, floor=floor, power=power)

    def to_json(self) -> dict[str, Any]:
        return {"round": self.round, "members": [m.to_json() for m in self.members]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "League":
        league = cls(round=int(data.get("round", 0)))
        league.members = [LeagueMember.from_json(m) for m in data.get("members", [])]
        guard_invariant_2(league.members, league.round)
        return league
