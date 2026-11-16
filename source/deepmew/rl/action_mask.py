"""Legal-action masking for the RL policy interface.

The Pokemon engine only ever presents legal options, so the legal-action set is
exactly ``select["option"]`` under the inclusive count bound ``[minCount,
maxCount]``. This module turns that into (a) a fixed-width boolean mask over a
padded action dimension and (b) a priority->selection projection that always
returns a selection accepted by ``valid_selection`` (INVARIANT #1: every return
is validated; illegal actions are impossible at the policy interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from deepmew.policy.selection import guard_selection, selection_bounds, valid_selection


@dataclass(frozen=True)
class ActionMask:
    """A legal mask over a fixed action dimension for one decision."""

    legal: tuple[bool, ...]
    n_options: int
    min_count: int
    max_count: int

    @property
    def dim(self) -> int:
        return len(self.legal)

    def legal_indices(self) -> list[int]:
        return [index for index in range(self.n_options) if self.legal[index]]


class ActionMasker:
    """Builds masks and projects a preference ordering onto a legal selection."""

    def __init__(self, max_options: int = 64):
        if max_options <= 0:
            raise ValueError("max_options must be positive")
        self.max_options = max_options

    def build_mask(self, select: dict[str, Any] | None) -> ActionMask:
        select = select or {}
        options = select.get("option") or []
        n_options = min(len(options), self.max_options)
        min_count, max_count = selection_bounds(select, n_options)
        legal = tuple(index < n_options for index in range(self.max_options))
        return ActionMask(legal=legal, n_options=n_options, min_count=min_count, max_count=max_count)

    def is_legal(self, selection: Sequence[int], select: dict[str, Any]) -> bool:
        return valid_selection(list(selection), select)

    def sanitize(self, selection: Sequence[int], select: dict[str, Any]) -> list[int]:
        """Return ``selection`` if legal, else the engine-safe fallback."""
        return guard_selection(list(selection), select)

    def select_from_priority(
        self, priority: Sequence[int], select: dict[str, Any]
    ) -> list[int]:
        """Project a preference ordering of option indices onto a legal pick.

        ``priority`` is option indices best-first (e.g. an argsort of masked
        policy logits). Picks up to ``maxCount`` legal options (never fewer than
        ``minCount``), then validates. Out-of-range or duplicate priority
        entries are dropped before validation; the result always passes
        ``valid_selection``.
        """
        options = select.get("option") or []
        n_options = len(options)
        min_count, max_count = selection_bounds(select, n_options)
        if max_count == 0:
            return guard_selection([], select)

        chosen: list[int] = []
        seen: set[int] = set()
        for index in priority:
            if 0 <= index < n_options and index not in seen:
                chosen.append(index)
                seen.add(index)
            if len(chosen) >= max_count:
                break
        # Top up from natural order if priority under-filled the required count.
        if len(chosen) < min_count:
            for index in range(n_options):
                if index not in seen:
                    chosen.append(index)
                    seen.add(index)
                if len(chosen) >= min_count:
                    break
        return guard_selection(chosen, select)
