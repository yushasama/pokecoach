from __future__ import annotations

from typing import Any

from deepmew.policy.models import Candidate
from deepmew.policy.scoring import score_candidates, score_candidates_v2
from deepmew.policy.selection import guard_selection, selection_bounds


def _choose_from_candidates(obs: dict[str, Any], candidates: list[Candidate]) -> list[int]:
    select = obs.get("select")
    if select is None:
        raise ValueError("deck selection is handled by main.read_deck_csv")

    min_count, max_count = selection_bounds(select, len(candidates))
    if max_count == 0:
        return guard_selection([], select)

    ordered = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.index))
    if min_count == max_count:
        selection = [candidate.index for candidate in ordered[:max_count]]
    else:
        positive = [candidate.index for candidate in ordered if candidate.score > 0]
        desired = max(min_count, min(max_count, len(positive)))
        selection = positive[:desired]

    return guard_selection(selection, select)


def choose_action_v2(obs: dict[str, Any]) -> list[int]:
    return _choose_from_candidates(obs, score_candidates_v2(obs))


def choose_action(
    obs: dict[str, Any],
    use_card_semantics: bool = True,
    attach_timing: str = "old",
    evolve_priority: str = "old",
) -> list[int]:
    return _choose_from_candidates(
        obs,
        score_candidates(
            obs,
            use_card_semantics=use_card_semantics,
            attach_timing=attach_timing,
            evolve_priority=evolve_priority,
        ),
    )
