"""YAML-driven league configuration for curriculum data generation.

Replaces long `--opp-pilots a,b,c` CLI incantations with a declarative roster: who
the sparring partners are, what brain each uses, and which deck each pilots (with a
per-opponent override). Two opponent shapes:

  - BAND opponents pilot the curriculum's per-ELO-band decks (`decks: band`): the
    diverse ladder field (greedy/v2.4/replay/exploiter over the R16 bucket decks).
  - FIXED-deck sparring partners pilot their own deck(s): the ~600 ELO reference
    agents, bd27 Mega Lucario, and our current Starmie policy on its coherent deck.

PFSP (AlphaStar prioritized fictitious self-play): given the prior round's per-
opponent win rates, allocate more games to the opponents that BEAT us.

INVARIANT #2: the roster is a FIXED opponent population (frozen bundles/checkpoints
+ mined decks); reward is the observed outcome vs those snapshots, never an
equilibrium signal. Offline/training-only (INVARIANT #8): never imported by main.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BAND = "band"
# Pilot brains that are deck-agnostic heuristics (no deck binding / checkpoint needed).
HEURISTIC_PILOTS = frozenset({"first-legal", "random", "greedy", "legacy-v2", "v2.4", "v4"})


@dataclass(frozen=True)
class OurSide:
    """The policy being trained (its decisions are recorded for the trainer)."""

    pilot: str          # spec | bundle:<dir> | checkpoint:<path> | exploiter
    deck: str           # bundle:<dir> | deck:<path> | root
    label: str = "our"
    feature_policy_id: str = ""
    league_member_id: str = ""

    @property
    def effective_feature_policy_id(self) -> str:
        return self.feature_policy_id or self.label


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    pilot: str          # heuristic | replay | exploiter | bundle:<dir> | checkpoint:<path>
    decks: Any = BAND   # "band" or a list of deck refs (bundle:<dir> | deck:<path> | root)
    enabled: bool = True
    top_n: int | None = None        # band opponents: override how many band decks
    exploiter_top: int | None = None
    games: int | None = None        # per-cell games override
    feature_policy_id: str = ""
    league_member_id: str = ""

    @property
    def is_band(self) -> bool:
        return isinstance(self.decks, str) and self.decks == BAND

    @property
    def deck_refs(self) -> list[str]:
        if self.is_band:
            return []
        if isinstance(self.decks, str):
            return [self.decks]
        return [str(d) for d in self.decks]


@dataclass(frozen=True)
class PfspSpec:
    enabled: bool = False
    weights_path: str = ""   # a curriculum/arena results.json carrying `pfsp_weights`
    floor: float = 0.1
    power: float = 1.0
    min_games: int = 2


@dataclass(frozen=True)
class LeagueConfig:
    our: OurSide
    opponents: tuple[OpponentSpec, ...]
    stages: tuple[str, ...] = ("600-800", "800-900", "900-1000", "1000-1100")
    games: int = 8
    top_n: int = 8
    carry_n: int = 4
    exploiter_top: int = 4
    seed: int = 181818
    champion: str = ""       # checkpoint for the `exploiter` pilot
    replay_tables: str = ""  # band_replay_tables.json for the `replay` pilot
    pfsp: PfspSpec = field(default_factory=PfspSpec)

    def enabled_opponents(self) -> list[OpponentSpec]:
        return [o for o in self.opponents if o.enabled]

    def needs_replay(self) -> bool:
        return any(o.pilot == "replay" for o in self.enabled_opponents())

    def needs_champion(self) -> bool:
        return any(o.pilot == "exploiter" for o in self.enabled_opponents()) or self.our.pilot == "exploiter"


def load_league_config(path: str | Path) -> LeagueConfig:
    return parse_league_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def parse_league_config(data: dict[str, Any]) -> LeagueConfig:
    """Build a LeagueConfig from an already-parsed mapping (pure; no file IO)."""
    our_raw = data.get("our") or {}
    if "pilot" not in our_raw or "deck" not in our_raw:
        raise ValueError("league config `our` must set both `pilot` and `deck`")
    our = OurSide(
        pilot=str(our_raw["pilot"]),
        deck=str(our_raw["deck"]),
        label=str(our_raw.get("label", "our")),
        feature_policy_id=str(our_raw.get("feature_policy_id", "")),
        league_member_id=str(our_raw.get("league_member_id", "")),
    )

    opponents: list[OpponentSpec] = []
    for raw in data.get("opponents") or []:
        if "name" not in raw or "pilot" not in raw:
            raise ValueError(f"opponent entry needs `name` and `pilot`: {raw!r}")
        opponents.append(OpponentSpec(
            name=str(raw["name"]), pilot=str(raw["pilot"]),
            decks=raw.get("decks", BAND), enabled=bool(raw.get("enabled", True)),
            top_n=raw.get("top_n"), exploiter_top=raw.get("exploiter_top"),
            games=raw.get("games"),
            feature_policy_id=str(raw.get("feature_policy_id", "")),
            league_member_id=str(raw.get("league_member_id", "")),
        ))
    if not opponents:
        raise ValueError("league config has no opponents")

    pfsp_raw = data.get("pfsp") or {}
    pfsp = PfspSpec(
        enabled=bool(pfsp_raw.get("enabled", False)),
        weights_path=str(pfsp_raw.get("weights_path", "")),
        floor=float(pfsp_raw.get("floor", 0.1)),
        power=float(pfsp_raw.get("power", 1.0)),
        min_games=int(pfsp_raw.get("min_games", 2)),
    )
    return LeagueConfig(
        our=our, opponents=tuple(opponents),
        stages=tuple(data.get("stages", LeagueConfig.stages)),
        games=int(data.get("games", 8)), top_n=int(data.get("top_n", 8)),
        carry_n=int(data.get("carry_n", 4)), exploiter_top=int(data.get("exploiter_top", 4)),
        seed=int(data.get("seed", 181818)),
        champion=str(data.get("champion", "")), replay_tables=str(data.get("replay_tables", "")),
        pfsp=pfsp,
    )


def pfsp_weights_from_data(data: dict[str, Any]) -> dict[str, float]:
    """Extract `pfsp_weights` (deck_hash:pilot -> weight) from a results mapping (pure)."""
    weights = data.get("pfsp_weights")
    if not isinstance(weights, dict):
        raise ValueError("results has no `pfsp_weights` dict")
    return {str(k): float(v) for k, v in weights.items()}


def load_pfsp_weights(results_path: str | Path) -> dict[str, float]:
    """Extract `pfsp_weights` from a prior run's results.json."""
    return pfsp_weights_from_data(json.loads(Path(results_path).read_text(encoding="utf-8")))


def pfsp_game_counts(cell_keys: list[str], weights: dict[str, float], base_games: int,
                     *, min_games: int = 2) -> dict[str, int]:
    """Allocate a fixed budget (`base_games * n_cells`) across cells proportional to
    PFSP weight, flooring each at `min_games`. Cells with no weight get `min_games`.

    A cell we LOSE to (high weight) gets more games; a cell we crush gets `min_games`.
    With no usable weights, falls back to uniform `base_games` per cell."""
    if not cell_keys:
        return {}
    budget = base_games * len(cell_keys)
    w = {k: max(0.0, weights.get(k, 0.0)) for k in cell_keys}
    total = sum(w.values())
    if total <= 0:
        return {k: base_games for k in cell_keys}
    return {k: max(min_games, round(budget * w[k] / total)) for k in cell_keys}
