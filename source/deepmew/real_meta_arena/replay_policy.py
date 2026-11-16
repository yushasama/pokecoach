"""ReplayPolicy opponent: sample action categories from K4 replay distributions.

The table is keyed by (select_context, archetype) -> action-category counts, with
backoff to (select_context) then global, built from the K4 imitation examples
(split=train by default). At decision time the opponent samples a legal action
category from the distribution and selects a legal option of that category;
anything uncertain falls back to first-legal, so the opponent is always legal.

Honest limitation: K4 broad action-category BC did not improve with archetype
labels, so this is useful as an *opponent distribution*, not as our own policy. It
shapes which action category the opponent takes, not the exact card/target.

Eval-only and gated: nothing here is imported by the submission runtime.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cg.api import OptionType, SelectContext
from deepmew.policy import guard_selection

EXAMPLES_FILENAME = "imitation_examples.jsonl"
TABLE_SCHEMA_VERSION = "deepmew.replay_policy_table.v1"


def _option_type_name(option: dict[str, Any]) -> str:
    try:
        return OptionType(option.get("type")).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{option.get('type')}"


def _context_name(select: dict[str, Any]) -> str:
    try:
        return SelectContext(select.get("context")).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{select.get('context')}"


@dataclass
class ReplayPolicyTable:
    schema_version: str
    split: str
    ctx_arch: dict[str, dict[str, int]]  # "context||archetype" -> {action_type: count}
    ctx: dict[str, dict[str, int]]        # "context" -> {action_type: count}
    global_counts: dict[str, int]
    train_examples: int

    @staticmethod
    def _key(context: str, archetype: str) -> str:
        return f"{context}||{archetype}"

    def sample(self, context: str, archetype: str, legal_types: set[str], rng: random.Random) -> str | None:
        for table, key in (
            (self.ctx_arch, self._key(context, archetype)),
            (self.ctx, context),
            (self.global_counts, None),
        ):
            counts = table if key is None else table.get(key)
            choice = _weighted_choice(counts or {}, legal_types, rng)
            if choice is not None:
                return choice
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "train_examples": self.train_examples,
            "ctx_arch": self.ctx_arch,
            "ctx": self.ctx,
            "global": self.global_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayPolicyTable":
        return cls(
            schema_version=str(data.get("schema_version", TABLE_SCHEMA_VERSION)),
            split=str(data.get("split", "train")),
            ctx_arch={k: dict(v) for k, v in (data.get("ctx_arch") or {}).items()},
            ctx={k: dict(v) for k, v in (data.get("ctx") or {}).items()},
            global_counts=dict(data.get("global") or {}),
            train_examples=int(data.get("train_examples", 0)),
        )


def _weighted_choice(counts: dict[str, int], legal_types: set[str], rng: random.Random) -> str | None:
    items = sorted((t, c) for t, c in counts.items() if t in legal_types and c > 0)
    if not items:
        return None
    total = sum(c for _, c in items)
    threshold = rng.random() * total
    acc = 0.0
    for action_type, count in items:
        acc += count
        if threshold <= acc:
            return action_type
    return items[-1][0]


def build_replay_policy_table(imitation_dir: str | Path, split: str = "train") -> ReplayPolicyTable:
    """Build the table from a K4 imitation run's examples (split=train by default)."""
    path = Path(imitation_dir) / EXAMPLES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"imitation examples not found: {path}")
    ctx_arch: dict[str, Counter] = defaultdict(Counter)
    ctx: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            context = str(row.get("select_context"))
            archetype = str(row.get("archetype"))
            action = str(row.get("chosen_option_type"))
            ctx_arch[ReplayPolicyTable._key(context, archetype)][action] += 1
            ctx[context][action] += 1
            global_counts[action] += 1
            n += 1
    return ReplayPolicyTable(
        schema_version=TABLE_SCHEMA_VERSION,
        split=split,
        ctx_arch={k: dict(v) for k, v in ctx_arch.items()},
        ctx={k: dict(v) for k, v in ctx.items()},
        global_counts=dict(global_counts),
        train_examples=n,
    )


@dataclass
class ReplayAgent:
    """A ReplayPolicy opponent bound to one archetype and a replay table."""

    archetype: str
    table: ReplayPolicyTable
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"replay[{self.archetype}]"

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        select = obs.get("select")
        if not isinstance(select, dict):
            import main

            return main.read_deck_csv()
        options = select.get("option") or []
        min_count = max(0, int(select.get("minCount") or 0))
        max_count = min(int(select.get("maxCount") or 0), len(options))
        first_legal = guard_selection(list(range(max_count)), select)
        if not options or max_count == 0:
            return first_legal

        context = _context_name(select)
        legal_types = {_option_type_name(o) for o in options if isinstance(o, dict)}
        desired = self.table.sample(context, self.archetype, legal_types, rng)
        if desired is None:
            return first_legal
        matching = [i for i, o in enumerate(options) if isinstance(o, dict) and _option_type_name(o) == desired]
        if not matching:
            return first_legal
        # Single-select: pick a sampled option of the desired category. Multi-select:
        # take desired-type options first, then fill to minCount with first-legal.
        if max_count == 1:
            chosen = [matching[rng.randrange(len(matching))]]
        else:
            chosen = list(matching[:max_count])
            for i in range(len(options)):
                if len(chosen) >= max_count:
                    break
                if i not in chosen:
                    chosen.append(i)
            if len(chosen) < min_count:
                return first_legal
        guarded = guard_selection(chosen[:max_count], select)
        return guarded if guarded else first_legal
