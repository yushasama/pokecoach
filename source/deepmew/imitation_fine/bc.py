"""K5-lite behavior cloning: filtered BC and reward-weighted BC over fine targets.

Both methods build a transparent weighted count table keyed by
``(select_context, archetype)`` over the fine-grained imitation target
(``FineExample.fine_target_key``: PLAY/EVOLVE card id, ATTACH target area, else
bare option type), with backoff to ``(select_context)`` then global. No torch /
lightgbm: this is a pure-Python count scorer that can rank legal options offline
and, if it ever earns promotion, be distilled to a runtime-safe form. Eval-only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from deepmew.imitation_fine.schemas import FineConfig, FineExample
from deepmew.imitation_fine.teacher import is_trusted_teacher, reward_weight

TABLE_SCHEMA_VERSION = "deepmew.fine_bc_table.v1"


def _key(context: str, archetype: str) -> str:
    return f"{context}||{archetype}"


@dataclass
class FineBCTable:
    """Weighted count tables over fine targets with deterministic backoff."""

    method: str
    min_cell_count: float
    ctx_arch: dict[str, dict[str, float]] = field(default_factory=dict)
    ctx: dict[str, dict[str, float]] = field(default_factory=dict)
    global_counts: dict[str, float] = field(default_factory=dict)
    schema_version: str = TABLE_SCHEMA_VERSION
    train_examples: int = 0
    train_weight: float = 0.0

    def _cell(self, context: str, archetype: str) -> dict[str, float] | None:
        """Most specific cell meeting the support floor, else coarser, else global."""
        cell = self.ctx_arch.get(_key(context, archetype))
        if cell and sum(cell.values()) >= self.min_cell_count:
            return cell
        cell = self.ctx.get(context)
        if cell and sum(cell.values()) >= self.min_cell_count:
            return cell
        return self.global_counts or None

    def predict_best(self, context: str, archetype: str, candidates: Iterable[str]) -> str | None:
        """Return the highest-weighted candidate fine target, or None if unseen."""
        cell = self._cell(context, archetype)
        if not cell:
            return None
        scored = [(c, cell.get(c, 0.0)) for c in candidates]
        scored = [(c, w) for c, w in scored if w > 0.0]
        if not scored:
            return None
        # Deterministic: highest weight, then lexical key.
        return max(scored, key=lambda cw: (cw[1], cw[0]))[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "min_cell_count": self.min_cell_count,
            "train_examples": self.train_examples,
            "train_weight": round(self.train_weight, 4),
            "ctx_arch": self.ctx_arch,
            "ctx": self.ctx,
            "global": self.global_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FineBCTable":
        return cls(
            method=str(data.get("method", "")),
            min_cell_count=float(data.get("min_cell_count", 1)),  # type: ignore[arg-type]
            ctx_arch={k: dict(v) for k, v in (data.get("ctx_arch") or {}).items()},  # type: ignore[union-attr]
            ctx={k: dict(v) for k, v in (data.get("ctx") or {}).items()},  # type: ignore[union-attr]
            global_counts=dict(data.get("global") or {}),  # type: ignore[arg-type]
            schema_version=str(data.get("schema_version", TABLE_SCHEMA_VERSION)),
            train_examples=int(data.get("train_examples", 0)),  # type: ignore[arg-type]
            train_weight=float(data.get("train_weight", 0.0)),  # type: ignore[arg-type]
        )


def _build_table(
    method: str,
    weighted_rows: Iterable[tuple[FineExample, float]],
    min_cell_count: float,
) -> FineBCTable:
    ctx_arch: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ctx: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_counts: dict[str, float] = defaultdict(float)
    n = 0
    total_weight = 0.0
    for ex, weight in weighted_rows:
        if weight <= 0.0:
            continue
        target = ex.fine_target_key()
        ctx_arch[_key(ex.select_context, ex.archetype)][target] += weight
        ctx[ex.select_context][target] += weight
        global_counts[target] += weight
        n += 1
        total_weight += weight
    return FineBCTable(
        method=method,
        min_cell_count=min_cell_count,
        ctx_arch={k: dict(v) for k, v in ctx_arch.items()},
        ctx={k: dict(v) for k, v in ctx.items()},
        global_counts=dict(global_counts),
        train_examples=n,
        train_weight=total_weight,
    )


def build_filtered_bc_table(
    examples: Iterable[FineExample],
    cfg: FineConfig,
    *,
    split: str | None = "train",
) -> FineBCTable:
    """Filtered BC: unit-weight, trusted-teacher rows only."""
    rows = (
        (ex, 1.0)
        for ex in examples
        if (split is None or ex.split == split) and is_trusted_teacher(ex, cfg)
    )
    return _build_table("filtered_bc", rows, cfg.min_cell_count)


def build_reward_weighted_bc_table(
    examples: Iterable[FineExample],
    cfg: FineConfig,
    *,
    split: str | None = "train",
) -> FineBCTable:
    """Reward-weighted BC: all rows, scaled by the transparent reward weight."""
    rows = (
        (ex, reward_weight(ex, cfg))
        for ex in examples
        if split is None or ex.split == split
    )
    return _build_table("reward_weighted_bc", rows, cfg.min_cell_count)


def build_context_baseline_table(
    examples: Iterable[FineExample],
    cfg: FineConfig,
    *,
    split: str | None = "train",
) -> FineBCTable:
    """Unweighted by-context fine-target table (the K4 by_context analog)."""
    rows = ((ex, 1.0) for ex in examples if split is None or ex.split == split)
    return _build_table("context_baseline", rows, cfg.min_cell_count)


@dataclass
class FineBCEval:
    method: str
    accuracy: float
    coverage: float
    n_test: int
    n_train: int


def evaluate_fine_bc(
    table: FineBCTable,
    test: list[FineExample],
) -> FineBCEval:
    """Top-1 accuracy of predicting the real fine target on held-out rows.

    The candidate set per row is the actual chosen fine target plus the row's
    legal-option fine targets are unavailable offline, so we score recovery of the
    chosen target against the full vocabulary seen in training (open-set top-1):
    a prediction is correct when the table's argmax over all known targets in the
    relevant cell equals the row's chosen target.
    """
    if not test:
        return FineBCEval(table.method, 0.0, 0.0, 0, table.train_examples)
    correct = 0
    answered = 0
    for ex in test:
        cell = table._cell(ex.select_context, ex.archetype)
        if not cell:
            continue
        answered += 1
        pred = max(cell.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if pred == ex.fine_target_key():
            correct += 1
    n = len(test)
    return FineBCEval(
        method=table.method,
        accuracy=round(correct / n, 4),
        coverage=round(answered / n, 4),
        n_test=n,
        n_train=table.train_examples,
    )
