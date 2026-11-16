"""K5 conservative offline action-value scorer over K4.1 labels (eval-only).

This is the deliberate successor to naive fine BC argmax (which mode-collapsed to
the most frequent atomic action in K2B). Instead of imitating the most-imitated
action, it estimates a conservative *value* per ``(select_context, archetype,
fine_target)`` cell from:

- outcome advantage: smoothed cell win rate minus the context baseline win rate
  (whole-game win is a crude credit-assignment label, not a true value model -- so
  it is only one term, blended with K4.1 label shaping, never the whole signal);
- goal progress: fraction of the cell's rows whose goal_of_play advances the deck's
  goal_of_deck;
- punishment avoidance: penalty for the cell's share of unjustified punished lines;
- last-stand correctness: small bonus for the cell's share of justified last-stand
  comeback risk (so the scorer does not learn blanket risk aversion).

Conservatism (offline-RL flavored, CQL-style):
- behavior-support gate: a cell below ``min_support`` is treated as UNSUPPORTED and
  is never assigned a boostable value -- the scorer never extrapolates value to
  out-of-distribution / unsupported legal actions;
- LCB pessimism: low-count supported cells are shrunk toward the baseline.

At decision time only behavior-supported legal actions are ranked; if none of the
legal actions is supported, the agent defers to first-legal. Offline/eval-only:
never imported by the submission runtime.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

from deepmew.imitation_fine.schemas import FineExample
from deepmew.imitation_fine.teacher import goal_progress

TABLE_SCHEMA_VERSION = "deepmew.action_value_table.v1"
UNSUPPORTED = float("-inf")


def _key(context: str, archetype: str) -> str:
    return f"{context}||{archetype}"


@dataclass
class ActionValueConfig:
    """Weights and conservatism knobs for the action-value estimate."""

    min_support: int = 12          # behavior-support gate: cells below this are OOD
    alpha: float = 20.0            # win-rate smoothing toward the context baseline
    w_advantage: float = 1.0       # outcome advantage weight (primary, but blended)
    w_goal: float = 0.3            # goal-progress weight
    w_punish: float = 0.6          # unjustified-punishment penalty weight
    w_last_stand: float = 0.2      # justified last-stand comeback bonus weight
    pessimism: float = 0.4         # LCB coefficient; shrink = pessimism / sqrt(count)


@dataclass
class ActionStat:
    """Raw aggregates for one (context, archetype, fine_target) cell."""

    count: int = 0
    wins: int = 0
    draws: int = 0
    goal_progress: int = 0
    punished_unjustified: int = 0
    last_stand: int = 0
    last_stand_justified: int = 0

    def add(self, ex: FineExample) -> None:
        self.count += 1
        if ex.is_draw:
            self.draws += 1
        elif ex.won:
            self.wins += 1
        if goal_progress(ex.goal_of_deck, ex.goal_of_play):
            self.goal_progress += 1
        risk_justified = ex.risk_justification.startswith("risk_justified")
        if ex.punishment_labels and not (risk_justified or ex.last_stand_triggered):
            self.punished_unjustified += 1
        if ex.last_stand_triggered:
            self.last_stand += 1
            if risk_justified:
                self.last_stand_justified += 1


def _value(stat: ActionStat, baseline: float, cfg: ActionValueConfig) -> float:
    """Conservative value for a supported cell (caller enforces the support gate)."""
    win_rate = (stat.wins + cfg.alpha * baseline) / (stat.count + cfg.alpha)
    advantage = win_rate - baseline
    goal_rate = stat.goal_progress / stat.count
    punish_rate = stat.punished_unjustified / stat.count
    last_stand_share = stat.last_stand_justified / stat.count
    pessimism = cfg.pessimism / math.sqrt(stat.count)
    return (
        cfg.w_advantage * advantage
        + cfg.w_goal * goal_rate
        - cfg.w_punish * punish_rate
        + cfg.w_last_stand * last_stand_share
        - pessimism
    )


@dataclass
class ConservativeActionValueTable:
    """Conservative offline action-value table with a behavior-support gate."""

    method: str = "conservative_action_value"
    config: ActionValueConfig = field(default_factory=ActionValueConfig)
    ctx_arch: dict[str, dict[str, ActionStat]] = field(default_factory=dict)
    ctx: dict[str, dict[str, ActionStat]] = field(default_factory=dict)
    context_baseline: dict[str, float] = field(default_factory=dict)
    global_baseline: float = 0.0
    schema_version: str = TABLE_SCHEMA_VERSION
    train_examples: int = 0

    def _stat(self, context: str, archetype: str, target: str) -> tuple[ActionStat | None, float]:
        """Most specific *supported* stat for (context, archetype, target) + baseline."""
        baseline = self.context_baseline.get(context, self.global_baseline)
        cell = self.ctx_arch.get(_key(context, archetype))
        stat = cell.get(target) if cell else None
        if stat and stat.count >= self.config.min_support:
            return stat, baseline
        coarse = self.ctx.get(context)
        stat = coarse.get(target) if coarse else None
        if stat and stat.count >= self.config.min_support:
            return stat, baseline
        return None, baseline

    def value(self, context: str, archetype: str, target: str) -> float:
        """Conservative value; UNSUPPORTED (-inf) for OOD/low-support actions."""
        stat, baseline = self._stat(context, archetype, target)
        if stat is None:
            return UNSUPPORTED
        return _value(stat, baseline, self.config)

    def is_supported(self, context: str, archetype: str, target: str) -> bool:
        stat, _ = self._stat(context, archetype, target)
        return stat is not None

    def rank_best(self, context: str, archetype: str, candidates: Iterable[str]) -> str | None:
        """Highest-value behavior-supported candidate, or None if none supported."""
        scored = [(c, self.value(context, archetype, c)) for c in candidates]
        scored = [(c, v) for c, v in scored if v != UNSUPPORTED]
        if not scored:
            return None
        # Deterministic: highest value, then lexical key.
        return max(scored, key=lambda cv: (cv[1], cv[0]))[0]

    def to_dict(self) -> dict[str, object]:
        def dump(cells: dict[str, dict[str, ActionStat]]) -> dict[str, dict[str, dict]]:
            return {k: {t: asdict(s) for t, s in v.items()} for k, v in cells.items()}

        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "config": asdict(self.config),
            "train_examples": self.train_examples,
            "global_baseline": round(self.global_baseline, 6),
            "context_baseline": {k: round(v, 6) for k, v in self.context_baseline.items()},
            "ctx_arch": dump(self.ctx_arch),
            "ctx": dump(self.ctx),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConservativeActionValueTable":
        def load(cells: dict) -> dict[str, dict[str, ActionStat]]:
            return {k: {t: ActionStat(**s) for t, s in v.items()} for k, v in (cells or {}).items()}

        cfg_data = data.get("config") or {}
        return cls(
            method=str(data.get("method", "conservative_action_value")),
            config=ActionValueConfig(**cfg_data),  # type: ignore[arg-type]
            ctx_arch=load(data.get("ctx_arch")),  # type: ignore[arg-type]
            ctx=load(data.get("ctx")),  # type: ignore[arg-type]
            context_baseline=dict(data.get("context_baseline") or {}),  # type: ignore[arg-type]
            global_baseline=float(data.get("global_baseline", 0.0)),  # type: ignore[arg-type]
            schema_version=str(data.get("schema_version", TABLE_SCHEMA_VERSION)),
            train_examples=int(data.get("train_examples", 0)),  # type: ignore[arg-type]
        )


def _decided_win_rate(examples: list[FineExample]) -> float:
    decided = [e for e in examples if not e.is_draw and e.result >= 0]
    return sum(1 for e in decided if e.won) / len(decided) if decided else 0.0


def build_action_value_table(
    examples: Iterable[FineExample],
    config: ActionValueConfig | None = None,
    *,
    split: str | None = "train",
) -> ConservativeActionValueTable:
    """Aggregate K4.1 rows into a conservative (context, archetype, target) value table."""
    cfg = config or ActionValueConfig()
    rows = [e for e in examples if split is None or e.split == split]

    ctx_arch: dict[str, dict[str, ActionStat]] = defaultdict(lambda: defaultdict(ActionStat))
    ctx: dict[str, dict[str, ActionStat]] = defaultdict(lambda: defaultdict(ActionStat))
    by_context: dict[str, list[FineExample]] = defaultdict(list)
    for ex in rows:
        target = ex.fine_target_key()
        ctx_arch[_key(ex.select_context, ex.archetype)][target].add(ex)
        ctx[ex.select_context][target].add(ex)
        by_context[ex.select_context].append(ex)

    context_baseline = {c: round(_decided_win_rate(group), 6) for c, group in by_context.items()}
    return ConservativeActionValueTable(
        config=cfg,
        ctx_arch={k: dict(v) for k, v in ctx_arch.items()},
        ctx={k: dict(v) for k, v in ctx.items()},
        context_baseline=context_baseline,
        global_baseline=round(_decided_win_rate(rows), 6),
        train_examples=len(rows),
    )


@dataclass
class ActionValueEval:
    scope: str
    n_test: int
    coverage: float            # fraction of test rows with a supported best action
    agree_rate: float          # fraction where chosen target == value-best target
    agreed_win_rate: float     # realized win rate when the player agreed with the scorer
    disagreed_win_rate: float  # realized win rate when the player disagreed
    lift: float                # agreed_win_rate - disagreed_win_rate


def evaluate_action_value(
    table: ConservativeActionValueTable,
    test: list[FineExample],
    *,
    scope: str = "ALL",
) -> ActionValueEval:
    """Directional offline sanity: do the scorer's high-value picks win more?

    Open-set over the supported targets known per cell (legal sets are unavailable
    offline). For each test row we compare the player's actual chosen target to the
    value-best supported target for that (context, archetype), then compare realized
    win rates of agreeing vs disagreeing rows. The K2B arena remains the real test.
    """
    answered = agreed = 0
    agreed_wins = disagreed = disagreed_wins = 0
    for ex in test:
        cell = table.ctx_arch.get(_key(ex.select_context, ex.archetype)) or {}
        supported = [t for t, s in cell.items() if s.count >= table.config.min_support]
        if not supported:
            cell = table.ctx.get(ex.select_context) or {}
            supported = [t for t, s in cell.items() if s.count >= table.config.min_support]
        if not supported:
            continue
        answered += 1
        best = table.rank_best(ex.select_context, ex.archetype, supported)
        won = (not ex.is_draw) and ex.won
        if best == ex.fine_target_key():
            agreed += 1
            agreed_wins += int(won)
        else:
            disagreed += 1
            disagreed_wins += int(won)
    n = len(test)
    agreed_wr = round(agreed_wins / agreed, 4) if agreed else 0.0
    disagreed_wr = round(disagreed_wins / disagreed, 4) if disagreed else 0.0
    return ActionValueEval(
        scope=scope,
        n_test=n,
        coverage=round(answered / n, 4) if n else 0.0,
        agree_rate=round(agreed / answered, 4) if answered else 0.0,
        agreed_win_rate=agreed_wr,
        disagreed_win_rate=disagreed_wr,
        lift=round(agreed_wr - disagreed_wr, 4),
    )
