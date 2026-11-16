"""Bounded action-space audit for the RL policy contract.

The PPO backend needs to know whether an engine decision is a single categorical
choice or a bounded set selection. This module measures the live simulator
surface without importing any learning backend.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cg.api import OptionType, SelectContext
from deepmew.policy.selection import selection_bounds, valid_selection
from deepmew.rl.env_adapter import EnvAdapter
from deepmew.rl.opponents import OpponentAgent


def _enum_name(enum_type: Any, value: Any) -> str:
    try:
        return enum_type(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _option_type(option: dict[str, Any]) -> str:
    return _enum_name(OptionType, option.get("type"))


def _context_name(select: dict[str, Any]) -> str:
    return _enum_name(SelectContext, select.get("context"))


def _option_identity(option: dict[str, Any]) -> tuple[Any, ...]:
    """Stable-ish public identity used only for audit grouping."""
    return (
        option.get("type"),
        option.get("area"),
        option.get("index"),
        option.get("cardId"),
        option.get("inPlayArea"),
        option.get("inPlayIndex"),
        option.get("playerIndex"),
        option.get("attackId"),
        option.get("number"),
    )


def _option_order_signature(select: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_option_identity(option) for option in select.get("option") or [])


def _source_key(option: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (option.get("area"), option.get("index"), option.get("cardId"))


def _has_target_specific_duplicate_source(options: list[dict[str, Any]]) -> bool:
    by_source: dict[tuple[Any, Any, Any], set[tuple[Any, Any]]] = defaultdict(set)
    for option in options:
        source = _source_key(option)
        if source == (None, None, None):
            continue
        by_source[source].add((option.get("inPlayArea"), option.get("inPlayIndex")))
    return any(len(targets) > 1 for targets in by_source.values())


def _first_legal(select: dict[str, Any]) -> list[int]:
    options = select.get("option") or []
    min_count, _ = selection_bounds(select, len(options))
    return list(range(min_count))


@dataclass
class ActionSpaceAudit:
    decisions: int = 0
    episodes: int = 0
    crashes: int = 0
    timeouts: int = 0
    max_option_count: int = 0
    option_count_distribution: Counter[int] = field(default_factory=Counter)
    min_count_distribution: Counter[int] = field(default_factory=Counter)
    max_count_distribution: Counter[int] = field(default_factory=Counter)
    count_pair_distribution: Counter[str] = field(default_factory=Counter)
    context_distribution: Counter[str] = field(default_factory=Counter)
    option_type_distribution: Counter[str] = field(default_factory=Counter)
    multiselect_contexts: Counter[str] = field(default_factory=Counter)
    multiselect_option_types: Counter[str] = field(default_factory=Counter)
    zero_select_decisions: int = 0
    single_select_decisions: int = 0
    multiselect_decisions: int = 0
    duplicate_selection_rejected: bool = True
    option_order_signatures_seen: int = 0
    option_order_repeated_decision_shapes: int = 0
    option_order_changed_for_repeated_shape: int = 0
    same_source_multi_target_decisions: int = 0
    sample_multiselects: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, obs: dict[str, Any]) -> None:
        select = obs.get("select") or {}
        options = select.get("option") or []
        n_options = len(options)
        min_count, max_count = selection_bounds(select, n_options)
        context = _context_name(select)

        self.decisions += 1
        self.max_option_count = max(self.max_option_count, n_options)
        self.option_count_distribution[n_options] += 1
        self.min_count_distribution[min_count] += 1
        self.max_count_distribution[max_count] += 1
        self.count_pair_distribution[f"{min_count}:{max_count}"] += 1
        self.context_distribution[context] += 1

        for option in options:
            self.option_type_distribution[_option_type(option)] += 1

        if min_count == 0:
            self.zero_select_decisions += 1
        if min_count == max_count == 1:
            self.single_select_decisions += 1
        if max_count > 1 or min_count > 1:
            self.multiselect_decisions += 1
            self.multiselect_contexts[context] += 1
            for option in options:
                self.multiselect_option_types[_option_type(option)] += 1
            if len(self.sample_multiselects) < 12:
                self.sample_multiselects.append(
                    {
                        "context": context,
                        "n_options": n_options,
                        "min_count": min_count,
                        "max_count": max_count,
                        "option_types": [_option_type(option) for option in options],
                    }
                )

        if n_options >= 1 and valid_selection([0, 0], select):
            self.duplicate_selection_rejected = False

        if _has_target_specific_duplicate_source(options):
            self.same_source_multi_target_decisions += 1

    def to_json(self) -> dict[str, Any]:
        denom = max(1, self.decisions)
        return {
            "schema_version": "deepmew.rl_action_space_audit.v0",
            "method": "bounded simulator exploration with first-legal agent selections and fixed opponents",
            "decisions": self.decisions,
            "episodes": self.episodes,
            "crashes": self.crashes,
            "timeouts": self.timeouts,
            "max_option_count": self.max_option_count,
            "option_count_distribution": dict(sorted(self.option_count_distribution.items())),
            "min_count_distribution": dict(sorted(self.min_count_distribution.items())),
            "max_count_distribution": dict(sorted(self.max_count_distribution.items())),
            "min_max_pair_distribution": dict(sorted(self.count_pair_distribution.items())),
            "context_distribution": dict(sorted(self.context_distribution.items())),
            "option_type_distribution": dict(sorted(self.option_type_distribution.items())),
            "zero_select_fraction": round(self.zero_select_decisions / denom, 6),
            "single_select_fraction": round(self.single_select_decisions / denom, 6),
            "multiselect_fraction": round(self.multiselect_decisions / denom, 6),
            "zero_select_decisions": self.zero_select_decisions,
            "single_select_decisions": self.single_select_decisions,
            "multiselect_decisions": self.multiselect_decisions,
            "multiselect_contexts": dict(sorted(self.multiselect_contexts.items())),
            "multiselect_option_types": dict(sorted(self.multiselect_option_types.items())),
            "duplicate_selections_valid": not self.duplicate_selection_rejected,
            "selection_order_matters": "not_proven_by_api_validation; PPO preserves sampled order",
            "option_order_stability": {
                "unique_signatures": self.option_order_signatures_seen,
                "repeated_decision_shapes": self.option_order_repeated_decision_shapes,
                "changed_for_repeated_shape": self.option_order_changed_for_repeated_shape,
                "interpretation": (
                    "bounded exploration only; option order is treated as part of the action contract"
                ),
            },
            "same_source_card_multiple_target_specific_options": {
                "decisions": self.same_source_multi_target_decisions,
                "fraction": round(self.same_source_multi_target_decisions / denom, 6),
            },
            "sample_multiselects": self.sample_multiselects,
        }


def run_action_space_audit(
    *,
    decisions: int,
    seed: int = 0,
    max_steps_per_game: int = 2000,
    opponents: tuple[str, ...] = ("first-legal", "v2.4"),
) -> ActionSpaceAudit:
    rng = random.Random(seed)
    audit = ActionSpaceAudit()
    signatures_by_shape: dict[tuple[str, int, int, int], tuple[tuple[Any, ...], ...]] = {}

    while audit.decisions < decisions:
        opponent = OpponentAgent.from_str(opponents[audit.episodes % len(opponents)])
        env = EnvAdapter(
            opponent=opponent,
            our_seat=audit.episodes % 2,
            max_steps=max_steps_per_game,
            opponent_seed=rng.randint(0, 2**31 - 1),
        )
        step = env.reset()
        audit.episodes += 1
        while not step.done and audit.decisions < decisions:
            obs = step.obs or {}
            select = obs.get("select") or {}
            options = select.get("option") or []
            min_count, max_count = selection_bounds(select, len(options))
            context = _context_name(select)
            signature = _option_order_signature(select)
            shape = (context, len(options), min_count, max_count)
            old_signature = signatures_by_shape.get(shape)
            if old_signature is None:
                signatures_by_shape[shape] = signature
            else:
                audit.option_order_repeated_decision_shapes += 1
                if old_signature != signature:
                    audit.option_order_changed_for_repeated_shape += 1

            audit.observe(obs)
            step = env.step(_first_legal(select))
        if step.crashed:
            audit.crashes += 1
        if step.timed_out:
            audit.timeouts += 1
        env.close()

    audit.option_order_signatures_seen = len(set(signatures_by_shape.values()))
    return audit


def write_action_space_audit(audit: ActionSpaceAudit, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = audit.to_json()
    json_path = out / "action_space_audit_v0.json"
    md_path = out / "action_space_audit_v0.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Action Space Audit v0",
        "",
        f"- decisions: {payload['decisions']}",
        f"- episodes: {payload['episodes']}",
        f"- crashes/timeouts: {payload['crashes']} / {payload['timeouts']}",
        f"- max option count: {payload['max_option_count']}",
        f"- zero-select fraction: {payload['zero_select_fraction']}",
        f"- single-select fraction: {payload['single_select_fraction']}",
        f"- multiselect fraction: {payload['multiselect_fraction']}",
        f"- duplicate selections valid: {payload['duplicate_selections_valid']}",
        f"- same source card as multiple target-specific options: "
        f"{payload['same_source_card_multiple_target_specific_options']['decisions']} decisions",
        "",
        "## Count Distributions",
        "",
        f"- option counts: `{payload['option_count_distribution']}`",
        f"- minCount: `{payload['min_count_distribution']}`",
        f"- maxCount: `{payload['max_count_distribution']}`",
        f"- min/max pairs: `{payload['min_max_pair_distribution']}`",
        "",
        "## Multiselect Contexts",
        "",
        f"- contexts: `{payload['multiselect_contexts']}`",
        f"- option types inside multiselect decisions: `{payload['multiselect_option_types']}`",
        "",
        "## Contract Notes",
        "",
        "- A single categorical action is insufficient when any `maxCount > 1` decision appears.",
        "- Duplicate selections are rejected by the live selection validator.",
        "- PPO must score the full ordered submitted selection. The reference backend uses an autoregressive masked selection distribution and preserves sampled order.",
        "- Option order is treated as part of the simulator contract; bounded exploration can detect repeated-shape changes but cannot prove global stability.",
        "",
        "## Sample Multiselect Decisions",
        "",
    ]
    for sample in payload["sample_multiselects"]:
        lines.append(f"- `{sample}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
