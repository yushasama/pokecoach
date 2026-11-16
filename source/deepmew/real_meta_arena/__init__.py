"""K2B Real-Meta Arena + ReplayPolicy opponents (eval-only).

Instantiates K3 Deck Zoo decklists in the engine (`battle_start`), validates
them, and measures DeepMew (root `deck.csv` under v2.4 / first-legal) against the
real-meta bucket opponent decks, plus a ReplayPolicy opponent built from the K4
train action-category distributions. Nothing here is imported by the submission
runtime; root `main.py` / `deck.csv` / `cg/` / default `v2.4` stay frozen.
"""

from __future__ import annotations

from deepmew.real_meta_arena.build import RealMetaArenaConfig, RealMetaArenaResult, build_real_meta_arena
from deepmew.real_meta_arena.decklists import ZooDeck, load_zoo_decklists, validate_decklist
from deepmew.real_meta_arena.replay_policy import ReplayAgent, ReplayPolicyTable, build_replay_policy_table
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup

__all__ = [
    "RealMetaArenaConfig",
    "RealMetaArenaResult",
    "ReplayAgent",
    "ReplayPolicyTable",
    "SpecAgent",
    "ZooDeck",
    "build_real_meta_arena",
    "build_replay_policy_table",
    "load_zoo_decklists",
    "run_real_meta_matchup",
    "validate_decklist",
]
