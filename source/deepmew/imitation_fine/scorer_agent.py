"""Eval-only FineBC pilot agent for the K2B real-meta arena.

Ranks the engine's legal options by a K5-lite ``FineBCTable``: each legal option
is resolved to its fine-target key (PLAY/EVOLVE card id, ATTACH target area, else
option type) via ``deepmew.options.identity`` and scored by the table cell for
``(select_context, archetype)`` with backoff. The highest-weighted legal option is
chosen; anything uncertain falls back to first-legal, so the agent is always legal
(INVARIANT #1). This agent is used only by local arena scripts / .research runs --
it is never imported by the submission runtime.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from typing import Callable, Iterable

from cg.api import OptionType, SelectContext
from deepmew.imitation_fine.action_value import ConservativeActionValueTable
from deepmew.imitation_fine.bc import FineBCTable
from deepmew.options.identity import resolve_option_identity
from deepmew.policy import guard_selection


def _context_name(select: dict[str, Any]) -> str:
    try:
        return SelectContext(select.get("context")).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{select.get('context')}"


def _option_type_name(option: dict[str, Any]) -> str:
    try:
        return OptionType(option.get("type")).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{option.get('type')}"


def fine_target_for_option(obs: dict[str, Any], option: dict[str, Any]) -> str:
    """Fine-target key for one legal option (mirrors FineExample.fine_target_key)."""
    option_type = _option_type_name(option)
    identity = resolve_option_identity(obs, option)
    if option_type in ("PLAY", "EVOLVE") and identity.card_id is not None:
        return f"{option_type}:{identity.card_id}"
    if option_type == "ATTACH" and identity.target_area:
        return f"ATTACH:{identity.target_area}"
    return option_type


def _select_by_target(
    obs: dict[str, Any],
    rng: random.Random,
    choose: Callable[[str, Iterable[str]], str | None],
) -> list[int]:
    """Resolve legal options to fine targets, pick the best per `choose`, stay legal.

    `choose(context, candidate_keys) -> best_key | None` is the only policy-specific
    step; everything else (legality, first-legal fallback, multi-select fill) is
    shared so every scorer obeys INVARIANT #1 identically.
    """
    select = obs.get("select")
    if not isinstance(select, dict):
        import main

        return main.read_deck_csv()
    options = select.get("option") or []
    max_count = min(int(select.get("maxCount") or 0), len(options))
    first_legal = guard_selection(list(range(max_count)), select)
    if not options or max_count == 0:
        return first_legal

    context = _context_name(select)
    candidates: dict[str, list[int]] = {}
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        key = fine_target_for_option(obs, option)
        candidates.setdefault(key, []).append(index)
    best_key = choose(context, candidates.keys())
    if best_key is None or best_key not in candidates:
        return first_legal

    matching = candidates[best_key]
    if max_count == 1:
        chosen = [matching[rng.randrange(len(matching))]]
    else:
        chosen = list(matching[:max_count])
        for index in range(len(options)):
            if len(chosen) >= max_count:
                break
            if index not in chosen:
                chosen.append(index)
    guarded = guard_selection(chosen[:max_count], select)
    return guarded if guarded else first_legal


@dataclass
class FineBCAgent:
    """A FineBC pilot bound to one archetype and a fine BC table."""

    archetype: str
    table: FineBCTable
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"finebc[{self.table.method}|{self.archetype}]"

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return _select_by_target(
            obs, rng,
            lambda context, keys: self.table.predict_best(context, self.archetype, keys),
        )


@dataclass
class ConservativeValueAgent:
    """A K5 conservative action-value pilot bound to an archetype + value table.

    Ranks legal options by conservative estimated value over *behavior-supported*
    fine targets only (never boosting OOD/unsupported actions); if no legal action
    is supported, defers to first-legal. This is value ranking, not BC argmax.
    """

    archetype: str
    table: ConservativeActionValueTable
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"value[{self.archetype}]"

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return _select_by_target(
            obs, rng,
            lambda context, keys: self.table.rank_best(context, self.archetype, keys),
        )
