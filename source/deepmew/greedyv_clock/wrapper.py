"""Clock-budgeted wrapper around a GreedyV-style search agent (BUG-003).

The wrapped agent must expose:
  - ``config`` with mutable ``per_decision_budget_ms`` / ``worlds`` / ``shortlist_k``
  - ``select(obs) -> list[int]`` and/or ``decide(obs) -> decision`` (``.selection``)
  - ``reset(seed)`` and ``deck`` (proxied through)
  - optionally ``shortlist_policy.select(obs) -> list[int]`` (the depth-0 prior)

Budget rule (per decision, from the whole-game bank ``remainingOverageTime``):
  spendable = max(0, remaining_s - reserve_s)
  budget_ms = clamp(spendable * 1000 / est_decisions_remaining, 0, max_budget_ms)
If ``budget_ms < min_budget_ms`` the wrapper skips search entirely and defers to
the depth-0 prior (instant, guaranteed no overrun). When the bank is in its lower
band the world count is clamped so even a single decision stays cheap.

The wrapper never raises (INVARIANT #1): on any internal error it falls back to
the prior's selection, then to a guarded first-legal choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Engine/whole-game constants (see cg config: actTimeout=0, 600 s overage bank).
_TOTAL_BANK_S = 600.0


@dataclass
class ClockBudgetConfig:
    total_bank_s: float = _TOTAL_BANK_S
    # Never spend the bank below this floor (endgame margin + fallback safety).
    reserve_s: float = 60.0
    # Conservative estimate of OUR remaining decisions in a game. Larger => more
    # cautious per-decision spend. Chosen so early-game budget stays modest.
    est_decisions_remaining: int = 80
    # Below this per-decision budget, don't search at all -> depth-0 prior.
    min_budget_ms: float = 25.0
    # Hard per-decision ceiling regardless of how much bank remains.
    max_budget_ms: float = 4000.0
    # When (remaining - reserve) drops below this fraction of the bank, clamp
    # the world count to keep a single decision cheap.
    low_bank_fraction: float = 0.25
    low_bank_worlds: int = 1


def _guard_first_legal(select: dict[str, Any]) -> list[int]:
    """A minimally-valid selection for a MAIN/other single-select context."""
    try:
        min_count = int(select.get("minCount") or 1)
    except (TypeError, ValueError):
        min_count = 1
    options = select.get("option") or []
    n = len(options)
    if n <= 0:
        return []
    return list(range(min(max(min_count, 1), n)))


class ClockBudgetedAgent:
    def __init__(self, agent: Any, config: ClockBudgetConfig | None = None):
        self.agent = agent
        self.cfg = config or ClockBudgetConfig()
        cfg = getattr(agent, "config", None)
        self._base_worlds = int(getattr(cfg, "worlds", 1) or 1)
        self.label = f"ClockBudgeted({getattr(agent, 'label', 'greedyv')})"

    # --- proxied attributes -------------------------------------------------
    @property
    def deck(self) -> Any:
        return getattr(self.agent, "deck", None)

    @property
    def deck_hash(self) -> Any:
        return getattr(self.agent, "deck_hash", None)

    @property
    def stats(self) -> Any:
        return getattr(self.agent, "stats", None)

    def reset(self, seed: int) -> None:
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset(seed)

    def bind_timer(self, timer: Any | None) -> None:
        bind = getattr(self.agent, "bind_timer", None)
        if callable(bind):
            bind(timer)

    # --- budgeting ----------------------------------------------------------
    def budget_ms(self, remaining_s: float) -> float:
        spendable = max(0.0, float(remaining_s) - self.cfg.reserve_s)
        per = spendable * 1000.0 / max(1, self.cfg.est_decisions_remaining)
        return max(0.0, min(self.cfg.max_budget_ms, per))

    def _remaining_s(self, obs: dict[str, Any]) -> float:
        rot = obs.get("remainingOverageTime") if isinstance(obs, dict) else None
        if isinstance(rot, (int, float)) and rot >= 0:
            return float(rot)
        return self.cfg.total_bank_s

    def _apply_budget(self, obs: dict[str, Any]) -> bool:
        """Set the agent's per-decision budget/worlds from the bank.

        Returns True if search is affordable, False if we should go depth-0.
        """
        remaining = self._remaining_s(obs)
        budget = self.budget_ms(remaining)
        cfg = getattr(self.agent, "config", None)
        if cfg is None or budget < self.cfg.min_budget_ms:
            return False
        cfg.per_decision_budget_ms = float(budget)
        spendable = remaining - self.cfg.reserve_s
        if spendable < self.cfg.low_bank_fraction * self.cfg.total_bank_s:
            cfg.worlds = min(self.cfg.low_bank_worlds, self._base_worlds)
        else:
            cfg.worlds = self._base_worlds
        return True

    def _prior_select(self, obs: dict[str, Any]) -> list[int]:
        prior = getattr(self.agent, "shortlist_policy", None)
        sel = getattr(prior, "select", None)
        if callable(sel):
            return list(sel(obs))
        # last resort: guarded first-legal
        select = obs.get("select") if isinstance(obs, dict) else None
        return _guard_first_legal(select or {})

    # --- decision entrypoints ----------------------------------------------
    def select(self, obs: dict[str, Any]) -> list[int]:
        select = obs.get("select") if isinstance(obs, dict) else None
        if not isinstance(select, dict):
            # deck-selection phase or malformed: defer to the agent/deck.
            return list(getattr(self.agent, "deck", None) or [])
        try:
            if not self._apply_budget(obs):
                return self._prior_select(obs)
            return list(self.agent.select(obs))
        except Exception:
            try:
                return self._prior_select(obs)
            except Exception:
                return _guard_first_legal(select)

    def decide(self, obs: dict[str, Any]) -> Any:
        """Return the underlying agent's decision object when searching, else a
        light record wrapping the depth-0 prior selection."""
        select = obs.get("select") if isinstance(obs, dict) else None
        if isinstance(select, dict) and self._apply_budget(obs):
            return self.agent.decide(obs)
        return _DepthZeroDecision(self._prior_select(obs))


@dataclass
class _DepthZeroDecision:
    selection: list[int]
    source: str = "clock_depth0_prior"
