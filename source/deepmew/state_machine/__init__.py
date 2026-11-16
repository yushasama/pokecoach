"""Board-state machine public API."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import tables as _tables
from .effects import attachment_effect
from .readiness import _energy_count, _own_player, _own_pokemon, _readiness_for_pokemon
from .rendering import describe_plan
from .state import AttachmentEffect, BoardPlan, PokemonReadiness, TurnState
from .tables import _load_card_tables
from .transition import analyze_board


_CACHE_ATTRS = {"_ATTACKS_BY_ID", "_CARDS_BY_ID"}


class _StateMachineModule(ModuleType):
    """Delegate legacy cache attributes to `tables` for compatibility."""

    def __getattr__(self, name: str) -> Any:
        if name in _CACHE_ATTRS:
            return getattr(_tables, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _CACHE_ATTRS:
            setattr(_tables, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _StateMachineModule

__all__ = [
    "AttachmentEffect",
    "BoardPlan",
    "PokemonReadiness",
    "TurnState",
    "analyze_board",
    "attachment_effect",
    "describe_plan",
]
