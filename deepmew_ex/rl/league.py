"""Prioritized fictitious self-play over fixed outside opponents plus capped own snapshots.

DeepMew invariant #2 rejects equilibrium/self-play-to-Nash as a ground-truth
training signal, so the league is anchored on frozen *outside* opponents
(D1, the reference notebooks, the downloaded 7th-place agent): they always
carry most of the sampling mass, and our own past snapshots are capped at
``snapshot_share`` of each iteration's games. Weights follow PFSP,
w = floor + (1 - win_rate)^2, with win rates tracked as an exponential moving
average from our policy's own games.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Member:
    identity: str
    kind: str  # 'outside' | 'snapshot'
    win_rate: float = 0.5
    games: int = 0
    max_share: float | None = None

    def weight(self, floor: float) -> float:
        return floor + (1.0 - self.win_rate) ** 2


@dataclass
class League:
    members: dict[str, Member] = field(default_factory=dict)
    floor: float = 0.15
    ema: float = 0.1
    snapshot_share: float = 0.2
    max_snapshots: int = 4

    def add_outside(self, identity: str, max_share: float | None = None) -> None:
        self.members[identity] = Member(identity, 'outside', max_share=max_share)

    def add_snapshot(self, tag: str, win_rate: float = 0.5) -> list[str]:
        """Add a snapshot; returns the tags evicted (oldest first) beyond ``max_snapshots``."""
        self.members[f'snapshot:{tag}'] = Member(f'snapshot:{tag}', 'snapshot', win_rate)
        snapshots = [key for key, member in self.members.items() if member.kind == 'snapshot']
        evicted = snapshots[:max(0, len(snapshots) - self.max_snapshots)]
        for key in evicted:
            del self.members[key]
        return [key.removeprefix('snapshot:') for key in evicted]

    def shares(self) -> dict[str, float]:
        """Per-member probability: PFSP within each kind, snapshots capped, per-member caps honoured."""
        outside = {key: member.weight(self.floor) for key, member in self.members.items() if member.kind == 'outside'}
        snapshots = {key: member.weight(self.floor) for key, member in self.members.items() if member.kind == 'snapshot'}
        if not outside:
            raise ValueError('league needs at least one outside opponent')
        snapshot_mass = self.snapshot_share if snapshots else 0.0
        shares = {key: (1 - snapshot_mass) * value / sum(outside.values()) for key, value in outside.items()}
        shares.update({key: snapshot_mass * value / sum(snapshots.values()) for key, value in snapshots.items()})
        # Clip capped members and give the excess to uncapped outside members, proportionally.
        for _ in range(len(shares)):
            capped = {key for key, member in self.members.items() if member.max_share is not None and shares[key] > member.max_share + 1e-12}
            if not capped:
                break
            excess = sum(shares[key] - self.members[key].max_share for key in capped)
            for key in capped:
                shares[key] = self.members[key].max_share
            free = [key for key in outside if self.members[key].max_share is None or shares[key] < self.members[key].max_share]
            total = sum(shares[key] for key in free)
            for key in free:
                shares[key] += excess * shares[key] / total
        return shares

    def schedule(self, games: int, rng: np.random.Generator) -> list[str]:
        """Exactly ``games`` opponent identities with counts proportional to ``shares`` (largest remainder)."""
        shares = self.shares()
        keys = sorted(shares)
        quotas = np.array([shares[key] * games for key in keys])
        counts = np.floor(quotas).astype(int)
        for index in np.argsort(-(quotas - counts), kind='stable')[:games - counts.sum()]:
            counts[index] += 1
        result = [key for key, count in zip(keys, counts) for _ in range(count)]
        rng.shuffle(result)
        return result

    def update(self, identity: str, score: float) -> None:
        member = self.members.get(identity)
        if member is None:
            return
        member.games += 1
        member.win_rate += self.ema * (score - member.win_rate)

    def state(self) -> dict:
        return {'floor': self.floor, 'ema': self.ema, 'snapshot_share': self.snapshot_share, 'max_snapshots': self.max_snapshots, 'members': {key: {'kind': member.kind, 'win_rate': member.win_rate, 'games': member.games, 'max_share': member.max_share} for key, member in self.members.items()}, 'shares': self.shares()}

    @classmethod
    def from_state(cls, state: dict) -> 'League':
        league = cls(floor=state['floor'], ema=state['ema'], snapshot_share=state['snapshot_share'], max_snapshots=state['max_snapshots'])
        for key, value in state['members'].items():
            league.members[key] = Member(key, value['kind'], value['win_rate'], value['games'], value['max_share'])
        return league
