"""Futile-attack guard (runtime-safe, self-contained core).

`core.py` is written to be copied verbatim into a submission bundle as a flat
`futility_guard.py` module (stdlib only; no deepmew/cg imports).
"""

from deepmew.futility_guard.core import (
    BASIC_WATER_ENERGY_ID,
    EX_CARD_IDS,
    EX_WALL_CARD_IDS,
    RIPTIDE_ATTACK_ID,
    SELF_MILL_ATTACK_IDS,
    FutilityGuard,
)

__all__ = [
    "BASIC_WATER_ENERGY_ID",
    "EX_CARD_IDS",
    "EX_WALL_CARD_IDS",
    "RIPTIDE_ATTACK_ID",
    "SELF_MILL_ATTACK_IDS",
    "FutilityGuard",
]
