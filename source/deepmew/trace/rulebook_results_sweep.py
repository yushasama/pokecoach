"""V3.10 multi-seed rulebook results sweep and bucket-collapse audit."""

from __future__ import annotations

import csv
import io
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from cg.api import OptionType

from deepmew.arena import MatrixSummary, build_matrix, rank_agents, run_pair
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id, safe_slug
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.card_semantics import effective_attack_semantics, visible_basic_water_in_discard
from deepmew.deck_profiles import (
    CURRENT_WATER_MEGA_ABOMASNOW,
    CYRANO_CARD_ID,
    HAMMER_LANCHE_ATTACK_ID,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    RIPTIDE_ATTACK_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.policy.attacks import attack_damage_for_option, attack_pressure, opponent_active_hp
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.trace.record import TraceRecorder
from deepmew.trace.rulebook_audit import (
    DEFAULT_POLICIES,
    DEFAULT_POLICY,
    build_rulebook_delta_event,
)


RUN_KIND = "babymew_rulebook_results_sweep"
HANDOFF_KIND = "babymew_rulebook_results_review"
SUMMARY_SCHEMA_VERSION = "deepmew.rulebook_results_sweep.v1"
COLLAPSED_EVENT_SCHEMA_VERSION = "deepmew.collapsed_rulebook_delta.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "deepmew.rulebook_results_snapshot.v1"

DEFAULT_AGENTS = DEFAULT_POLICIES
KEY_PAIR_SPECS = (
    ("v2.4", "first-legal"),
    ("v2.4[evolve_priority=new]", "v2.4"),
    ("v2.4[attach_timing=new]", "v2.4"),
    ("v2.4[semantics=off]", "v2.4"),
)

DRAW_SEARCH_CARD_IDS = frozenset(
    {
        LILLIES_DETERMINATION_CARD_ID,
        WAITRESS_CARD_ID,
        CYRANO_CARD_ID,
        MEGA_SIGNAL_CARD_ID,
    }
)
TARGET_CREATING_CARD_IDS = frozenset({SNOVER_CARD_ID, KYOGRE_CARD_ID})
NONESSENTIAL_PLAY_CARD_IDS = frozenset({MAXIMUM_BELT_CARD_ID})

REQUIRED_RUN_FILES = [
    "manifest.json",
    "report.md",
    "multi_seed_matrix_summary.json",
    "pair_sweep_summary.json",
    "attack_pressure_collapse.json",
    "unknown_collapse.json",
    "delta_family_summary.json",
    "collapsed_delta_events.jsonl",
    "collapsed_delta_events.csv",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "multi_seed_matrix_summary.json",
    "pair_sweep_summary.json",
    "attack_pressure_collapse.json",
    "unknown_collapse.json",
    "delta_family_summary.json",
    "collapsed_delta_events.csv",
    "collapsed_delta_events.jsonl",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
    "relevant_paths.txt",
]


class AttackPressureLabel(str, Enum):
    WEAK_ATTACK_OVER_DEVELOPMENT = "WEAK_ATTACK_OVER_DEVELOPMENT"
    WEAK_ATTACK_BUT_NO_BETTER_DEVELOPMENT = "WEAK_ATTACK_BUT_NO_BETTER_DEVELOPMENT"
    MEANINGFUL_ATTACK_CORRECT = "MEANINGFUL_ATTACK_CORRECT"
    KO_ATTACK_CORRECT = "KO_ATTACK_CORRECT"
    PRIZE_PRESSURE_ATTACK_CORRECT = "PRIZE_PRESSURE_ATTACK_CORRECT"
    HAMMER_LANCHE_PRESSURE = "HAMMER_LANCHE_PRESSURE"
    HAMMER_LANCHE_LOW_VALUE = "HAMMER_LANCHE_LOW_VALUE"
    RIPTIDE_LOW_VALUE = "RIPTIDE_LOW_VALUE"
    RIPTIDE_GOOD_RECYCLE_OR_PRESSURE = "RIPTIDE_GOOD_RECYCLE_OR_PRESSURE"
    ATTACK_OVER_EVOLVE = "ATTACK_OVER_EVOLVE"
    ATTACK_OVER_ATTACH = "ATTACK_OVER_ATTACH"
    ATTACK_OVER_SETUP_PLAY = "ATTACK_OVER_SETUP_PLAY"
    DEVELOPMENT_OVER_WEAK_ATTACK = "DEVELOPMENT_OVER_WEAK_ATTACK"
    UNKNOWN_ATTACK_VALUE = "UNKNOWN_ATTACK_VALUE"


class UnknownLabel(str, Enum):
    UNKNOWN_DUE_TO_MISSING_SCORE_REASON = "UNKNOWN_DUE_TO_MISSING_SCORE_REASON"
    UNKNOWN_DUE_TO_MISSING_OPTION_IDENTITY = "UNKNOWN_DUE_TO_MISSING_OPTION_IDENTITY"
    UNKNOWN_DUE_TO_MISSING_STATE_SNAPSHOT = "UNKNOWN_DUE_TO_MISSING_STATE_SNAPSHOT"
    UNKNOWN_DUE_TO_MISSING_FOLLOWTHROUGH = "UNKNOWN_DUE_TO_MISSING_FOLLOWTHROUGH"
    UNKNOWN_DUE_TO_UNMODELED_TRAINER_EFFECT = "UNKNOWN_DUE_TO_UNMODELED_TRAINER_EFFECT"
    UNKNOWN_DUE_TO_UNMODELED_ATTACK_EFFECT = "UNKNOWN_DUE_TO_UNMODELED_ATTACK_EFFECT"
    UNKNOWN_DUE_TO_POLICY_SCORE_TIE = "UNKNOWN_DUE_TO_POLICY_SCORE_TIE"
    UNKNOWN_DUE_TO_MULTI_ACTION_SEQUENCE = "UNKNOWN_DUE_TO_MULTI_ACTION_SEQUENCE"
    UNKNOWN_BUT_LOW_IMPACT = "UNKNOWN_BUT_LOW_IMPACT"
    UNKNOWN_HIGH_IMPACT = "UNKNOWN_HIGH_IMPACT"


class TrainerSequencingLabel(str, Enum):
    DRAW_SEARCH_BEFORE_DEVELOPMENT = "DRAW_SEARCH_BEFORE_DEVELOPMENT"
    DRAW_SEARCH_BEFORE_ATTACK = "DRAW_SEARCH_BEFORE_ATTACK"
    SETUP_PLAY_BEFORE_ATTACH = "SETUP_PLAY_BEFORE_ATTACH"
    SETUP_PLAY_BEFORE_EVOLVE = "SETUP_PLAY_BEFORE_EVOLVE"
    TARGET_CREATING_PLAY_CORRECT = "TARGET_CREATING_PLAY_CORRECT"
    NONESSENTIAL_PLAY_OVER_DEVELOPMENT = "NONESSENTIAL_PLAY_OVER_DEVELOPMENT"
    TRAINER_SEQUENCE_UNKNOWN = "TRAINER_SEQUENCE_UNKNOWN"


class ReadinessLabel(str, Enum):
    ATTACK_READY_WEAK_ONLY = "ATTACK_READY_WEAK_ONLY"
    ATTACK_READY_MEANINGFUL = "ATTACK_READY_MEANINGFUL"
    ATTACK_READY_KO_PRESSURE = "ATTACK_READY_KO_PRESSURE"
    ATTACKER_NEAR_READY_CORRECT = "ATTACKER_NEAR_READY_CORRECT"
    NO_ATTACKER_READY_CORRECT = "NO_ATTACKER_READY_CORRECT"
    STATE_READINESS_CONTRADICTS_ATTACK_PRESSURE = "STATE_READINESS_CONTRADICTS_ATTACK_PRESSURE"
    STATE_READINESS_UNKNOWN = "STATE_READINESS_UNKNOWN"


@dataclass(frozen=True)
class AttackPressureFacts:
    default_action: str | None
    first_action: str | None
    best_pressure: str | None
    best_attack_id: int | None
    best_effective_damage: int | None
    development_available: bool
    important_evolve_legal: bool
    attach_legal: bool
    setup_play_available: bool


@dataclass(frozen=True)
class UnknownFacts:
    default_action: str | None
    default_card_id: int | None
    default_score_reasons: tuple[str, ...]
    board_plan_missing: bool
    later_same_turn_actions: int
    any_attack_unknown: bool
    any_play_unmodeled: bool
    score_tie: bool
    source_result: str | None


@dataclass(frozen=True)
class TrainerFacts:
    default_action: str | None
    first_action: str | None
    card_id: int | None
    attach_legal: bool
    important_evolve_legal: bool


@dataclass(frozen=True)
class ReadinessFacts:
    state: str | None
    best_pressure: str | None
    default_action: str | None


@dataclass(frozen=True)
class CollapseRule:
    name: str
    guard: Callable[[Any], bool]
    label: str
    confidence: str = "medium"


@dataclass(frozen=True)
class RulebookResultsSweepResult:
    run_dir: Path
    manifest_path: Path
    summary: dict[str, Any]
    bundle_dir: Path | None = None


def _attack_is_ko(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.best_pressure == "ko"


def _hammer_lanche_pressure(facts: AttackPressureFacts) -> bool:
    return facts.best_attack_id == HAMMER_LANCHE_ATTACK_ID and facts.best_pressure in {"ko", "meaningful"}


def _hammer_lanche_low(facts: AttackPressureFacts) -> bool:
    return facts.best_attack_id == HAMMER_LANCHE_ATTACK_ID and facts.best_pressure == "weak"


def _riptide_low(facts: AttackPressureFacts) -> bool:
    return facts.best_attack_id == RIPTIDE_ATTACK_ID and facts.best_pressure == "weak"


def _riptide_good(facts: AttackPressureFacts) -> bool:
    return facts.best_attack_id == RIPTIDE_ATTACK_ID and facts.best_pressure in {"ko", "meaningful"}


def _attack_meaningful(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.best_pressure == "meaningful"


def _weak_attack_over_development(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.best_pressure == "weak" and facts.development_available


def _weak_attack_no_better_development(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.best_pressure == "weak" and not facts.development_available


def _attack_over_evolve(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.important_evolve_legal


def _attack_over_attach(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.attach_legal


def _attack_over_setup(facts: AttackPressureFacts) -> bool:
    return facts.default_action == "ATTACK" and facts.setup_play_available


def _development_over_weak_attack(facts: AttackPressureFacts) -> bool:
    return facts.default_action in {"ATTACH", "PLAY", "EVOLVE"} and facts.best_pressure == "weak"


def _always(_: Any) -> bool:
    return True


ATTACK_PRESSURE_RULES: tuple[CollapseRule, ...] = (
    CollapseRule("ko-attack-correct", _attack_is_ko, AttackPressureLabel.KO_ATTACK_CORRECT.value, "high"),
    CollapseRule("hammer-lanche-pressure", _hammer_lanche_pressure, AttackPressureLabel.HAMMER_LANCHE_PRESSURE.value, "medium"),
    CollapseRule("hammer-lanche-low-value", _hammer_lanche_low, AttackPressureLabel.HAMMER_LANCHE_LOW_VALUE.value, "medium"),
    CollapseRule("riptide-low-value", _riptide_low, AttackPressureLabel.RIPTIDE_LOW_VALUE.value, "medium"),
    CollapseRule("riptide-good-recycle-or-pressure", _riptide_good, AttackPressureLabel.RIPTIDE_GOOD_RECYCLE_OR_PRESSURE.value, "medium"),
    CollapseRule("meaningful-attack-correct", _attack_meaningful, AttackPressureLabel.MEANINGFUL_ATTACK_CORRECT.value, "medium"),
    CollapseRule("weak-attack-over-development", _weak_attack_over_development, AttackPressureLabel.WEAK_ATTACK_OVER_DEVELOPMENT.value, "medium"),
    CollapseRule("weak-attack-no-better-development", _weak_attack_no_better_development, AttackPressureLabel.WEAK_ATTACK_BUT_NO_BETTER_DEVELOPMENT.value, "low"),
    CollapseRule("attack-over-evolve", _attack_over_evolve, AttackPressureLabel.ATTACK_OVER_EVOLVE.value, "medium"),
    CollapseRule("attack-over-attach", _attack_over_attach, AttackPressureLabel.ATTACK_OVER_ATTACH.value, "medium"),
    CollapseRule("attack-over-setup-play", _attack_over_setup, AttackPressureLabel.ATTACK_OVER_SETUP_PLAY.value, "medium"),
    CollapseRule("development-over-weak-attack", _development_over_weak_attack, AttackPressureLabel.DEVELOPMENT_OVER_WEAK_ATTACK.value, "low"),
    CollapseRule("unknown-attack-value", _always, AttackPressureLabel.UNKNOWN_ATTACK_VALUE.value, "low"),
)


def _missing_score_reason(facts: UnknownFacts) -> bool:
    return facts.default_action not in {None, "NO", "YES"} and not facts.default_score_reasons


def _missing_option_identity(facts: UnknownFacts) -> bool:
    return facts.default_action == "PLAY" and facts.default_card_id is None


def _missing_state_snapshot(facts: UnknownFacts) -> bool:
    return facts.board_plan_missing


def _missing_followthrough(facts: UnknownFacts) -> bool:
    return facts.later_same_turn_actions == 0


def _unmodeled_trainer(facts: UnknownFacts) -> bool:
    return facts.any_play_unmodeled


def _unmodeled_attack(facts: UnknownFacts) -> bool:
    return facts.any_attack_unknown


def _score_tie(facts: UnknownFacts) -> bool:
    return facts.score_tie


def _multi_action_sequence(facts: UnknownFacts) -> bool:
    return facts.later_same_turn_actions >= 2


def _low_impact(facts: UnknownFacts) -> bool:
    return facts.source_result == "source_win"


UNKNOWN_RULES: tuple[CollapseRule, ...] = (
    CollapseRule("missing-option-identity", _missing_option_identity, UnknownLabel.UNKNOWN_DUE_TO_MISSING_OPTION_IDENTITY.value, "high"),
    CollapseRule("missing-state-snapshot", _missing_state_snapshot, UnknownLabel.UNKNOWN_DUE_TO_MISSING_STATE_SNAPSHOT.value, "high"),
    CollapseRule("missing-score-reason", _missing_score_reason, UnknownLabel.UNKNOWN_DUE_TO_MISSING_SCORE_REASON.value, "medium"),
    CollapseRule("unmodeled-trainer-effect", _unmodeled_trainer, UnknownLabel.UNKNOWN_DUE_TO_UNMODELED_TRAINER_EFFECT.value, "medium"),
    CollapseRule("unmodeled-attack-effect", _unmodeled_attack, UnknownLabel.UNKNOWN_DUE_TO_UNMODELED_ATTACK_EFFECT.value, "medium"),
    CollapseRule("policy-score-tie", _score_tie, UnknownLabel.UNKNOWN_DUE_TO_POLICY_SCORE_TIE.value, "low"),
    CollapseRule("multi-action-sequence", _multi_action_sequence, UnknownLabel.UNKNOWN_DUE_TO_MULTI_ACTION_SEQUENCE.value, "low"),
    CollapseRule("missing-followthrough", _missing_followthrough, UnknownLabel.UNKNOWN_DUE_TO_MISSING_FOLLOWTHROUGH.value, "low"),
    CollapseRule("low-impact", _low_impact, UnknownLabel.UNKNOWN_BUT_LOW_IMPACT.value, "low"),
    CollapseRule("high-impact", _always, UnknownLabel.UNKNOWN_HIGH_IMPACT.value, "low"),
)


def _draw_search_before_attack(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.first_action == "ATTACK" and facts.card_id in DRAW_SEARCH_CARD_IDS


def _draw_search_before_development(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.card_id in DRAW_SEARCH_CARD_IDS


def _setup_before_attach(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.attach_legal and facts.card_id in {MEGA_SIGNAL_CARD_ID, WAITRESS_CARD_ID}


def _setup_before_evolve(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.important_evolve_legal and facts.card_id in {MEGA_SIGNAL_CARD_ID, WAITRESS_CARD_ID}


def _target_creating_play(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.card_id in TARGET_CREATING_CARD_IDS


def _nonessential_play(facts: TrainerFacts) -> bool:
    return facts.default_action == "PLAY" and facts.card_id in NONESSENTIAL_PLAY_CARD_IDS


TRAINER_RULES: tuple[CollapseRule, ...] = (
    CollapseRule("target-creating-play-correct", _target_creating_play, TrainerSequencingLabel.TARGET_CREATING_PLAY_CORRECT.value, "medium"),
    CollapseRule("draw-search-before-attack", _draw_search_before_attack, TrainerSequencingLabel.DRAW_SEARCH_BEFORE_ATTACK.value, "medium"),
    CollapseRule("setup-play-before-attach", _setup_before_attach, TrainerSequencingLabel.SETUP_PLAY_BEFORE_ATTACH.value, "medium"),
    CollapseRule("setup-play-before-evolve", _setup_before_evolve, TrainerSequencingLabel.SETUP_PLAY_BEFORE_EVOLVE.value, "medium"),
    CollapseRule("draw-search-before-development", _draw_search_before_development, TrainerSequencingLabel.DRAW_SEARCH_BEFORE_DEVELOPMENT.value, "low"),
    CollapseRule("nonessential-play-over-development", _nonessential_play, TrainerSequencingLabel.NONESSENTIAL_PLAY_OVER_DEVELOPMENT.value, "medium"),
    CollapseRule("trainer-sequence-unknown", _always, TrainerSequencingLabel.TRAINER_SEQUENCE_UNKNOWN.value, "low"),
)


def _ready_ko(facts: ReadinessFacts) -> bool:
    return facts.state == "ATTACK_READY" and facts.best_pressure == "ko"


def _ready_meaningful(facts: ReadinessFacts) -> bool:
    return facts.state == "ATTACK_READY" and facts.best_pressure == "meaningful"


def _ready_weak(facts: ReadinessFacts) -> bool:
    return facts.state == "ATTACK_READY" and facts.best_pressure == "weak"


def _near_ready(facts: ReadinessFacts) -> bool:
    return facts.state == "ATTACKER_NEAR_READY"


def _no_attacker(facts: ReadinessFacts) -> bool:
    return facts.state == "NO_ATTACKER_READY"


def _readiness_contradicts(facts: ReadinessFacts) -> bool:
    return facts.state == "ATTACK_READY" and facts.best_pressure in {None, "unknown"}


READINESS_RULES: tuple[CollapseRule, ...] = (
    CollapseRule("attack-ready-ko-pressure", _ready_ko, ReadinessLabel.ATTACK_READY_KO_PRESSURE.value, "high"),
    CollapseRule("attack-ready-meaningful", _ready_meaningful, ReadinessLabel.ATTACK_READY_MEANINGFUL.value, "medium"),
    CollapseRule("attack-ready-weak-only", _ready_weak, ReadinessLabel.ATTACK_READY_WEAK_ONLY.value, "medium"),
    CollapseRule("attacker-near-ready-correct", _near_ready, ReadinessLabel.ATTACKER_NEAR_READY_CORRECT.value, "medium"),
    CollapseRule("no-attacker-ready-correct", _no_attacker, ReadinessLabel.NO_ATTACKER_READY_CORRECT.value, "medium"),
    CollapseRule("readiness-contradicts-pressure", _readiness_contradicts, ReadinessLabel.STATE_READINESS_CONTRADICTS_ATTACK_PRESSURE.value, "medium"),
    CollapseRule("readiness-unknown", _always, ReadinessLabel.STATE_READINESS_UNKNOWN.value, "low"),
)


def classify_attack_pressure(facts: AttackPressureFacts) -> dict[str, str]:
    for rule in ATTACK_PRESSURE_RULES:
        if rule.guard(facts):
            return {"label": rule.label, "reason": rule.name, "confidence": rule.confidence}
    raise AssertionError("unreachable attack-pressure classifier")


def classify_unknown(facts: UnknownFacts) -> dict[str, str]:
    for rule in UNKNOWN_RULES:
        if rule.guard(facts):
            return {"label": rule.label, "reason": rule.name, "confidence": rule.confidence}
    raise AssertionError("unreachable unknown classifier")


def classify_trainer(facts: TrainerFacts) -> dict[str, str]:
    for rule in TRAINER_RULES:
        if rule.guard(facts):
            return {"label": rule.label, "reason": rule.name, "confidence": rule.confidence}
    raise AssertionError("unreachable trainer classifier")


def classify_readiness(facts: ReadinessFacts) -> dict[str, str]:
    for rule in READINESS_RULES:
        if rule.guard(facts):
            return {"label": rule.label, "reason": rule.name, "confidence": rule.confidence}
    raise AssertionError("unreachable readiness classifier")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_handoff_dir(*, out_root: Path | str = ".research/handoff_bundles", now: datetime | None = None) -> Path:
    created = now or _utc_now()
    bundle_dir = Path(out_root) / created.strftime("%Y%m%d") / make_run_id(HANDOFF_KIND, now=created)
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def _pair_to_dict(pair: Any) -> dict[str, Any]:
    return {
        "agent_a": pair.agent_a,
        "agent_b": pair.agent_b,
        "games": pair.games,
        "completed": pair.completed,
        "a_wins": pair.a_wins,
        "b_wins": pair.b_wins,
        "draws": pair.draws,
        "crashes": pair.crashes,
        "timeouts": pair.timeouts,
        "avg_turns": pair.avg_turns,
        "elapsed_seconds": pair.elapsed_seconds,
    }


def _run_matrix_with_unique_trace(
    agents: list[str],
    *,
    games_per_pair: int,
    max_steps: int,
    seed: int,
    trace_recorder: TraceRecorder,
) -> MatrixSummary:
    pairs = []
    for left_index, left in enumerate(agents):
        for right_index, right in enumerate(agents):
            if right_index <= left_index:
                continue
            pair_seed = seed + left_index * 1009 + right_index * 9173
            pairs.append(
                run_pair(
                    left,
                    right,
                    games=games_per_pair,
                    max_steps=max_steps,
                    seed=pair_seed,
                    trace_recorder=trace_recorder,
                    game_id_prefix=f"s{seed}-{left_index:02d}-{right_index:02d}",
                )
            )
    labels = []
    for pair in pairs:
        for label in (pair.agent_a, pair.agent_b):
            if label not in labels:
                labels.append(label)
    matrix = build_matrix(pairs, labels)
    return MatrixSummary(labels, games_per_pair, seed, False, pairs, matrix, rank_agents(matrix))


def collect_results_sweep(
    *,
    run_id: str,
    agents: Iterable[str],
    seeds: Iterable[int],
    games_per_pair: int,
    max_steps: int,
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[MatrixSummary], dict[str, Any]]:
    trace_recorder = TraceRecorder(run_id=run_id, sample_limit=limit_frames)
    agent_list = list(agents)
    matrices = [
        _run_matrix_with_unique_trace(
            agent_list,
            games_per_pair=games_per_pair,
            max_steps=max_steps,
            seed=seed,
            trace_recorder=trace_recorder,
        )
        for seed in seeds
    ]
    frames = []
    for frame in trace_recorder.frames:
        payload = frame.to_jsonable()
        payload["source_agent"] = payload.get("agent_label")
        payload["source_matchup"] = str(payload.get("game_id", "")).split("-000", 1)[0]
        frames.append(payload)
    return frames, matrices, trace_recorder.summary()


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def build_multi_seed_matrix_summary(matrices: list[MatrixSummary]) -> dict[str, Any]:
    all_pairs = [pair for matrix in matrices for pair in matrix.pairs]
    agents = matrices[0].agents if matrices else list(DEFAULT_AGENTS)
    aggregate = build_matrix(all_pairs, agents)
    aggregate_rankings = rank_agents(aggregate)
    per_seed = []
    per_agent_rates: dict[str, list[float]] = defaultdict(list)
    for matrix in matrices:
        ranking_by_agent = {row["agent"]: row for row in matrix.rankings}
        for agent in agents:
            per_agent_rates[agent].append(float(ranking_by_agent.get(agent, {}).get("point_rate", 0.0)))
        per_seed.append(
            {
                "seed": matrix.seed,
                "matrix": matrix.matrix,
                "rankings": matrix.rankings,
                "pairs": [_pair_to_dict(pair) for pair in matrix.pairs],
            }
        )
    agent_stats = {}
    for agent, values in per_agent_rates.items():
        agent_stats[agent] = {
            "point_rates": values,
            "mean_point_rate": sum(values) / len(values) if values else 0.0,
            "variance": _variance(values),
            "spread": max(values) - min(values) if values else 0.0,
        }
    default_mean = agent_stats.get(DEFAULT_POLICY, {}).get("mean_point_rate", 0.0)
    experimental_status = {}
    for agent in agents:
        if agent == DEFAULT_POLICY or "[" not in agent:
            continue
        rates = agent_stats.get(agent, {}).get("point_rates", [])
        default_rates = agent_stats.get(DEFAULT_POLICY, {}).get("point_rates", [])
        consistently_better = bool(rates and default_rates and all(left > right for left, right in zip(rates, default_rates)))
        experimental_status[agent] = {
            "mean_delta_vs_default": agent_stats.get(agent, {}).get("mean_point_rate", 0.0) - default_mean,
            "consistently_better_than_default": consistently_better,
            "recommendation": "expand eval" if consistently_better else "keep experimental",
        }
    max_spread = max((stats["spread"] for stats in agent_stats.values()), default=0.0)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "seeds": [matrix.seed for matrix in matrices],
        "games_per_pair": matrices[0].games_per_pair if matrices else 0,
        "agents": agents,
        "per_seed": per_seed,
        "aggregate_matrix": aggregate,
        "aggregate_rankings": aggregate_rankings,
        "agent_point_rate_stats": agent_stats,
        "crashes": sum(pair.crashes for pair in all_pairs),
        "timeouts": sum(pair.timeouts for pair in all_pairs),
        "seed_variance_too_high": max_spread >= 0.15,
        "max_point_rate_spread": max_spread,
        "experimental_status": experimental_status,
    }


def _pair_key(agent_a: str, agent_b: str) -> tuple[str, str]:
    return tuple(sorted((agent_a, agent_b)))


def build_pair_sweep_summary(matrices: list[MatrixSummary], key_pairs: Iterable[tuple[str, str]] = KEY_PAIR_SPECS) -> dict[str, Any]:
    wanted = {_pair_key(left, right): (left, right) for left, right in key_pairs}
    rows: dict[str, Any] = {}
    for key, display in wanted.items():
        left, right = display
        rows[f"{left} vs {right}"] = {
            "agent_a": left,
            "agent_b": right,
            "per_seed": [],
            "aggregate": {"a_wins": 0, "b_wins": 0, "draws": 0, "completed": 0, "crashes": 0, "timeouts": 0},
        }
    for matrix in matrices:
        for pair in matrix.pairs:
            key = _pair_key(pair.agent_a, pair.agent_b)
            if key not in wanted:
                continue
            left, right = wanted[key]
            row = rows[f"{left} vs {right}"]
            if pair.agent_a == left:
                left_wins, right_wins = pair.a_wins, pair.b_wins
            else:
                left_wins, right_wins = pair.b_wins, pair.a_wins
            seed_row = {
                "seed": matrix.seed,
                "a_wins": left_wins,
                "b_wins": right_wins,
                "draws": pair.draws,
                "completed": pair.completed,
                "crashes": pair.crashes,
                "timeouts": pair.timeouts,
            }
            row["per_seed"].append(seed_row)
            row["aggregate"]["a_wins"] += left_wins
            row["aggregate"]["b_wins"] += right_wins
            row["aggregate"]["draws"] += pair.draws
            row["aggregate"]["completed"] += pair.completed
            row["aggregate"]["crashes"] += pair.crashes
            row["aggregate"]["timeouts"] += pair.timeouts
    for row in rows.values():
        agg = row["aggregate"]
        completed = agg["completed"]
        agg["a_point_rate"] = (agg["a_wins"] + agg["draws"] * 0.5) / completed if completed else None
        agg["b_point_rate"] = (agg["b_wins"] + agg["draws"] * 0.5) / completed if completed else None
    return {"schema_version": SUMMARY_SCHEMA_VERSION, "pairs": rows}


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("raw_observation")
    return raw if isinstance(raw, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return [option for option in ((_raw_observation(frame).get("select") or {}).get("option") or []) if isinstance(option, dict)]


def _option_type(option: dict[str, Any]) -> str | None:
    try:
        return OptionType(option.get("type")).name
    except (TypeError, ValueError):
        return None


def _attack_name_from_event(event: dict[str, Any], attack_id: int | None) -> str | None:
    for option in event.get("legal_options") or []:
        if _int_or_none(option.get("attack_id")) == attack_id:
            return option.get("attack_name")
    return None


def _attack_facts(frame: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame)
    opponent_hp = opponent_active_hp(raw) if raw else None
    attacks = []
    for index, option in enumerate(_raw_options(frame)):
        if _option_type(option) != "ATTACK":
            continue
        semantics = effective_attack_semantics(option, raw) if raw else None
        damage = attack_damage_for_option(option, raw, True) if raw else None
        pressure = attack_pressure(option, raw, True) if raw else "unknown"
        attack_id = _int_or_none(option.get("attackId"))
        attacks.append(
            {
                "index": index,
                "attack_id": attack_id,
                "attack_name": _attack_name_from_event(event, attack_id),
                "effective_damage": damage,
                "opponent_active_hp": opponent_hp,
                "pressure_tier": pressure,
                "ko_available": pressure == "ko",
                "prize_pressure_available": pressure in {"ko", "meaningful"},
                "meaningful_attack_available": pressure == "meaningful",
                "weak_attack_only": pressure == "weak",
                "hammer_lanche_pressure": attack_id == HAMMER_LANCHE_ATTACK_ID,
                "riptide_pressure": attack_id == RIPTIDE_ATTACK_ID,
                "riptide_water_count": visible_basic_water_in_discard(raw) if attack_id == RIPTIDE_ATTACK_ID else None,
                "semantic_reason": getattr(semantics, "reason", None),
            }
        )
    best = _best_attack(attacks)
    return {
        "attacks": attacks,
        "best_attack": best,
        "ko_available": any(item["ko_available"] for item in attacks),
        "prize_pressure_available": any(item["prize_pressure_available"] for item in attacks),
        "meaningful_attack_available": any(item["meaningful_attack_available"] for item in attacks),
        "weak_attack_only": bool(attacks) and all(item["pressure_tier"] == "weak" for item in attacks),
    }


def _best_attack(attacks: list[dict[str, Any]]) -> dict[str, Any] | None:
    order = {"ko": 4, "meaningful": 3, "weak": 2, "unknown": 1}
    return max(attacks, key=lambda item: (order.get(str(item.get("pressure_tier")), 0), item.get("effective_damage") or -1), default=None)


def _choice(event: dict[str, Any], policy: str) -> dict[str, Any]:
    return (event.get("policy_choices") or {}).get(policy) or {}


def _default_choice(event: dict[str, Any]) -> dict[str, Any]:
    return _choice(event, DEFAULT_POLICY)


def _score_reasons(choice: dict[str, Any]) -> tuple[str, ...]:
    reasons = choice.get("reasons")
    return tuple(str(item) for item in reasons or ())


def _score_tie(event: dict[str, Any]) -> bool:
    by_index = (((event.get("score_payloads") or {}).get(DEFAULT_POLICY) or {}).get("by_index") or {})
    scores = [payload.get("score") for payload in by_index.values() if isinstance(payload, dict)]
    top = max(scores, default=None)
    return top is not None and sum(1 for score in scores if score == top) > 1


def _hand_summary(event: dict[str, Any]) -> list[dict[str, Any]]:
    return event.get("hand_visible_play_cards") or []


def _bench_summary(event: dict[str, Any]) -> Any:
    return (event.get("board_summary") or {}).get("own_bench") or []


def _active_summary(event: dict[str, Any]) -> dict[str, Any]:
    active = (event.get("board_summary") or {}).get("own_active")
    return active if isinstance(active, dict) else {}


def _development_facts(event: dict[str, Any]) -> dict[str, Any]:
    best_attach = event.get("best_attach_target") or {}
    return {
        "important_evolve_legal": bool(event.get("important_evolve_legal")),
        "attach_legal": bool(event.get("attach_legal")),
        "manual_energy_already_attached": bool(event.get("energy_attached")),
        "best_attach_target": best_attach,
        "target_is_already_ready": bool(best_attach.get("already_ready")),
        "target_improves_readiness": bool(best_attach.get("improves_readiness")),
    }


def _trainer_facts(event: dict[str, Any]) -> dict[str, Any]:
    default = _default_choice(event)
    card_id = _int_or_none(default.get("card_id"))
    return {
        "chosen_play_card_id": card_id if default.get("option_type") == "PLAY" else None,
        "chosen_play_card_name": default.get("card_name") if default.get("option_type") == "PLAY" else None,
        "setup_or_search_draw_card": card_id in DRAW_SEARCH_CARD_IDS,
        "target_creating_body_card": card_id in TARGET_CREATING_CARD_IDS,
        "nonessential_play": card_id in NONESSENTIAL_PLAY_CARD_IDS,
    }


def _followthrough(event: dict[str, Any]) -> dict[str, Any]:
    later = event.get("later_same_turn_actions") or []
    action_types = [((item.get("chosen") or {}).get("option_type")) for item in later]
    return {
        "later_same_turn_attach": "ATTACH" in action_types,
        "later_same_turn_evolve": "EVOLVE" in action_types,
        "later_same_turn_attack": "ATTACK" in action_types,
        "later_same_turn_actions": later,
        "final_source_result": event.get("source_trajectory_final_result"),
    }


def _collapse_label(frame: dict[str, Any], event: dict[str, Any], attack_info: dict[str, Any]) -> dict[str, str]:
    old_family = str(event.get("likely_rule_family") or "unknown")
    default = _default_choice(event)
    first = _choice(event, "first-legal")
    best = attack_info.get("best_attack") or {}
    if old_family == "attack_pressure":
        return classify_attack_pressure(
            AttackPressureFacts(
                default_action=default.get("option_type"),
                first_action=first.get("option_type"),
                best_pressure=best.get("pressure_tier"),
                best_attack_id=_int_or_none(best.get("attack_id")),
                best_effective_damage=_int_or_none(best.get("effective_damage")),
                development_available=bool(event.get("attach_legal") or event.get("important_evolve_legal") or _hand_summary(event)),
                important_evolve_legal=bool(event.get("important_evolve_legal")),
                attach_legal=bool(event.get("attach_legal")),
                setup_play_available=bool(_hand_summary(event)),
            )
        )
    if old_family == "unknown":
        return classify_unknown(
            UnknownFacts(
                default_action=default.get("option_type"),
                default_card_id=_int_or_none(default.get("card_id")),
                default_score_reasons=_score_reasons(default),
                board_plan_missing=not bool(event.get("board_plan")),
                later_same_turn_actions=len(event.get("later_same_turn_actions") or []),
                any_attack_unknown=any(item.get("pressure_tier") == "unknown" for item in attack_info.get("attacks") or []),
                any_play_unmodeled=any(item.get("card_id") in DRAW_SEARCH_CARD_IDS for item in _hand_summary(event)),
                score_tie=_score_tie(event),
                source_result=event.get("source_trajectory_final_result"),
            )
        )
    if old_family == "trainer_sequencing":
        return classify_trainer(
            TrainerFacts(
                default_action=default.get("option_type"),
                first_action=first.get("option_type"),
                card_id=_int_or_none(default.get("card_id")),
                attach_legal=bool(event.get("attach_legal")),
                important_evolve_legal=bool(event.get("important_evolve_legal")),
            )
        )
    if old_family == "state_machine_readiness":
        return classify_readiness(
            ReadinessFacts(
                state=(event.get("board_plan") or {}).get("state"),
                best_pressure=best.get("pressure_tier"),
                default_action=default.get("option_type"),
            )
        )
    if old_family == "card_semantics":
        if best.get("attack_id") == HAMMER_LANCHE_ATTACK_ID and best.get("pressure_tier") in {"ko", "meaningful"}:
            return {"label": AttackPressureLabel.HAMMER_LANCHE_PRESSURE.value, "reason": "card-semantics-hammer-lanche-pressure", "confidence": "medium"}
        if best.get("attack_id") == RIPTIDE_ATTACK_ID and best.get("pressure_tier") == "weak":
            return {"label": AttackPressureLabel.RIPTIDE_LOW_VALUE.value, "reason": "card-semantics-riptide-low", "confidence": "medium"}
    return {"label": f"{old_family.upper()}_UNCOLLAPSED", "reason": "family-not-targeted", "confidence": "low"}


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int]:
    result_weight = 0 if event.get("source_trajectory_final_result") == "source_loss" else 1
    turn = _int_or_none(event.get("turn")) or 999
    return result_weight, turn


def _sample_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_family[str(event.get("likely_rule_family") or "unknown")].append(event)
    selected: list[dict[str, Any]] = []
    for family, limit in (
        ("attack_pressure", 200),
        ("unknown", 200),
        ("state_machine_readiness", 160),
        ("trainer_sequencing", 160),
        ("card_semantics", 160),
    ):
        selected.extend(sorted(by_family.get(family, []), key=_event_sort_key)[:limit])
    seen = set()
    unique = []
    for event in selected:
        key = event.get("frame_id"), event.get("likely_rule_family")
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def collapse_rulebook_events(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames_by_id = {str(frame.get("frame_id")): frame for frame in frames}
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        key = (
            str(frame.get("source_agent") or frame.get("agent_label") or ""),
            str(frame.get("game_id") or ""),
            _int_or_none(frame.get("seat")),
        )
        groups[key].append(frame)
    for group in groups.values():
        group.sort(key=lambda item: _int_or_none(item.get("decision_index")) or -1)

    all_events = []
    snapshots = []
    for frame in frames:
        key = (
            str(frame.get("source_agent") or frame.get("agent_label") or ""),
            str(frame.get("game_id") or ""),
            _int_or_none(frame.get("seat")),
        )
        built = build_rulebook_delta_event(frame, group=groups.get(key, []))
        if built is None:
            continue
        event, snapshot = built
        snapshots.append({"schema_version": STATE_SNAPSHOT_SCHEMA_VERSION, **snapshot})
        if event.get("is_delta_event"):
            all_events.append(event)

    collapsed = []
    for event in _sample_events(all_events):
        frame = frames_by_id.get(str(event.get("frame_id")), {})
        attack_info = _attack_facts(frame, event)
        label = _collapse_label(frame, event, attack_info)
        default = _default_choice(event)
        row = {
            "schema_version": COLLAPSED_EVENT_SCHEMA_VERSION,
            "frame_id": event.get("frame_id"),
            "source_agent": event.get("source_agent"),
            "source_matchup": frame.get("source_matchup") or event.get("game_id"),
            "turn": event.get("turn"),
            "decision_index": event.get("decision_index"),
            "board_plan_state": (event.get("board_plan") or {}).get("state"),
            "active": _active_summary(event),
            "active_card_id": _active_summary(event).get("card_id"),
            "active_card_name": _active_summary(event).get("name"),
            "active_energy_count": _active_summary(event).get("energy_count"),
            "bench_summary": _bench_summary(event),
            "hand_summary": _hand_summary(event),
            "legal_options": event.get("legal_options") or [],
            "policy_choices": event.get("policy_choices") or {},
            "default_score": default.get("score"),
            "default_score_reasons": default.get("reasons") or [],
            "attack_pressure_facts": attack_info,
            "development_facts": _development_facts(event),
            "trainer_sequencing_facts": _trainer_facts(event),
            "followthrough_facts": _followthrough(event),
            "old_rulebook_family": event.get("likely_rule_family"),
            "old_rulebook_label": event.get("classification"),
            "new_collapsed_label": label["label"],
            "collapse_reason": label["reason"],
            "confidence": label["confidence"],
            "tags": sorted(set(event.get("tags") or []) | {str(event.get("likely_rule_family")), label["label"]}),
        }
        collapsed.append(row)
    return collapsed, snapshots


def _collapse_counter(events: list[dict[str, Any]], family: str | None = None) -> Counter[str]:
    return Counter(
        str(event.get("new_collapsed_label"))
        for event in events
        if family is None or event.get("old_rulebook_family") == family
    )


def build_delta_family_summary(collapsed_events: list[dict[str, Any]]) -> dict[str, Any]:
    old_counts = Counter(str(event.get("old_rulebook_family")) for event in collapsed_events)
    new_counts = Counter(str(event.get("new_collapsed_label")) for event in collapsed_events)
    concrete_modes = Counter(
        f"{event.get('old_rulebook_family')}::{event.get('new_collapsed_label')}"
        for event in collapsed_events
    )
    result_split: dict[str, Counter[str]] = defaultdict(Counter)
    for event in collapsed_events:
        result_split[str(event.get("new_collapsed_label"))][str((event.get("followthrough_facts") or {}).get("final_source_result"))] += 1
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sampled_collapsed_events": len(collapsed_events),
        "old_family_counts": dict(old_counts),
        "new_collapsed_family_counts": dict(new_counts),
        "top_20_repeated_concrete_failure_modes": dict(concrete_modes.most_common(20)),
        "source_result_split_by_collapsed_label": {key: dict(value) for key, value in result_split.items()},
    }


def build_attack_pressure_collapse(collapsed_events: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for event in collapsed_events if event.get("old_rulebook_family") == "attack_pressure"]
    counts = _collapse_counter(events)
    remaining = counts.get(AttackPressureLabel.UNKNOWN_ATTACK_VALUE.value, 0)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sampled_events": len(events),
        "collapse_counts": dict(counts),
        "remaining_unknown_attack_value_count": remaining,
        "interpretation": _attack_interpretation(counts),
    }


def _attack_interpretation(counts: Counter[str]) -> str:
    if counts.get(AttackPressureLabel.WEAK_ATTACK_OVER_DEVELOPMENT.value, 0):
        return "weak attacks over development remain the highest-value attack-pressure candidate"
    if counts.get(AttackPressureLabel.KO_ATTACK_CORRECT.value, 0) or counts.get(AttackPressureLabel.MEANINGFUL_ATTACK_CORRECT.value, 0):
        return "many attack-pressure rows appear to preserve real damage pressure"
    return "attack-pressure rows still need richer attack/effect instrumentation"


def build_unknown_collapse(collapsed_events: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for event in collapsed_events if event.get("old_rulebook_family") == "unknown"]
    counts = _collapse_counter(events)
    terminal_unknowns = counts.get(UnknownLabel.UNKNOWN_HIGH_IMPACT.value, 0) + counts.get(UnknownLabel.UNKNOWN_BUT_LOW_IMPACT.value, 0)
    reclassified = len(events) - terminal_unknowns
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sampled_events": len(events),
        "collapse_counts": dict(counts),
        "reclassified_unknown_count": reclassified,
        "remaining_unknown_count": terminal_unknowns,
        "needs_instrumentation_or_semantics": terminal_unknowns > max(10, len(events) // 4),
        "interpretation": _unknown_interpretation(counts, terminal_unknowns),
    }


def _unknown_interpretation(counts: Counter[str], remaining: int) -> str:
    if counts.get(UnknownLabel.UNKNOWN_DUE_TO_UNMODELED_TRAINER_EFFECT.value, 0):
        return "unknowns mostly collapse toward unmodeled trainer effects and sequencing context"
    if remaining:
        return "remaining unknowns are still large enough to justify instrumentation before score tuning"
    return "unknown bucket collapsed cleanly in this sample"


def _csv_text(events: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "source_agent",
        "source_matchup",
        "turn",
        "decision_index",
        "old_rulebook_family",
        "old_rulebook_label",
        "new_collapsed_label",
        "confidence",
        "board_plan_state",
        "active_card_name",
        "active_energy_count",
        "default_score",
        "tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for event in events:
        writer.writerow({field: ";".join(event.get(field) or []) if field == "tags" else event.get(field) for field in fields})
    return stream.getvalue()


def _ranking_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in summary.get("aggregate_rankings") or []:
        stats = (summary.get("agent_point_rate_stats") or {}).get(item["agent"], {})
        rows.append(
            [
                item["agent"],
                f"{item['point_rate']:.3f}",
                f"{stats.get('mean_point_rate', 0.0):.3f}",
                f"{stats.get('spread', 0.0):.3f}",
                item["wins"],
                item["losses"],
                item["draws"],
            ]
        )
    return rows


def _pair_rows(pair_summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for name, row in (pair_summary.get("pairs") or {}).items():
        agg = row["aggregate"]
        rows.append([name, agg["a_wins"], agg["b_wins"], agg["draws"], agg["completed"], agg["crashes"], agg["timeouts"]])
    return rows


def build_report(
    *,
    multi_seed: dict[str, Any],
    pair_sweep: dict[str, Any],
    attack_collapse: dict[str, Any],
    unknown_collapse: dict[str, Any],
    delta_family: dict[str, Any],
) -> str:
    lines = [title("DeepMew V3.10 Rulebook Results Sweep")]
    lines.append(section("Executive Summary"))
    lines.append(
        bullets(
            [
                "Diagnostic-only pass: no gameplay/default behavior changed.",
                f"Seeds: {multi_seed.get('seeds')} games_per_pair={multi_seed.get('games_per_pair')}.",
                f"Crashes={multi_seed.get('crashes')} timeouts={multi_seed.get('timeouts')}.",
                f"Seed variance too high: {multi_seed.get('seed_variance_too_high')} (max spread={multi_seed.get('max_point_rate_spread')}).",
                _highest_value_next(attack_collapse, unknown_collapse, multi_seed),
            ]
        )
    )
    lines.append(section("Multi-Seed Real-Results Sweep"))
    lines.append(table(["Agent", "Aggregate", "Mean", "Spread", "Wins", "Losses", "Draws"], _ranking_rows(multi_seed)))
    lines.append(section("Pairwise Summary"))
    lines.append(table(["Pair", "A Wins", "B Wins", "Draws", "Completed", "Crashes", "Timeouts"], _pair_rows(pair_sweep)))
    lines.append(section("Rulebook Family Deltas"))
    lines.append(table(["Old Family", "Count"], Counter(delta_family.get("old_family_counts") or {}).most_common()))
    lines.append(section("Attack-Pressure Collapse"))
    lines.append(table(["Label", "Count"], Counter(attack_collapse.get("collapse_counts") or {}).most_common()))
    lines.append(section("Unknown Collapse"))
    lines.append(table(["Label", "Count"], Counter(unknown_collapse.get("collapse_counts") or {}).most_common()))
    lines.append(section("Corrected First-Legal Gap"))
    lines.append(
        bullets(
            [
                _first_legal_summary(pair_sweep),
                "Delta families are reported from same-observation samples; do not infer full-game counterfactual causality after divergence.",
            ]
        )
    )
    lines.append(section("Experimental Ablation Status"))
    lines.append(table(["Agent", "Mean Delta vs Default", "Consistent", "Recommendation"], _experimental_rows(multi_seed)))
    lines.append(section("Highest-Value Next Candidate"))
    lines.append(bullets([_highest_value_next(attack_collapse, unknown_collapse, multi_seed)]))
    lines.append(section("What Not To Change"))
    lines.append(
        bullets(
            [
                "Do not promote `v2.4[attach_timing=new]`.",
                "Do not promote `v2.4[evolve_priority=new]`.",
                "Do not change deck.csv, cg/, root main.py, broad parser behavior, trainer semantics, ML/search/belief/replay, or default v2.4 from this audit.",
            ]
        )
    )
    lines.append(section("Repro Commands"))
    lines.append(
        "```powershell\n"
        "python -m compileall deepmew scripts tests\n"
        "python -m pytest tests -q\n"
        "python scripts\\audit_babymew_rulebook_results.py --seeds 20240720 20240721 20240722 --games-per-pair 20 --export-handoff-bundle\n"
        "```\n"
    )
    return "\n".join(lines)


def _first_legal_summary(pair_sweep: dict[str, Any]) -> str:
    row = (pair_sweep.get("pairs") or {}).get("v2.4 vs first-legal") or {}
    agg = row.get("aggregate") or {}
    a_wins = agg.get("a_wins", 0)
    b_wins = agg.get("b_wins", 0)
    if a_wins == b_wins:
        return f"Default vs first-legal is tied in the aggregate ({a_wins}-{b_wins}); gap looks seed-sensitive/noisy."
    leader = "default v2.4" if a_wins > b_wins else "first-legal"
    return f"{leader} leads aggregate default-vs-first-legal {a_wins}-{b_wins}; inspect family deltas before assigning cause."


def _experimental_rows(multi_seed: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for agent, status in (multi_seed.get("experimental_status") or {}).items():
        rows.append(
            [
                agent,
                f"{status.get('mean_delta_vs_default', 0.0):.3f}",
                status.get("consistently_better_than_default"),
                status.get("recommendation"),
            ]
        )
    return rows or [["none", "0.000", False, "none"]]


def _highest_value_next(attack_collapse: dict[str, Any], unknown_collapse: dict[str, Any], multi_seed: dict[str, Any]) -> str:
    if multi_seed.get("seed_variance_too_high"):
        return "run larger evaluation because seed variance dominates"
    unknown_remaining = unknown_collapse.get("remaining_unknown_count", 0)
    if unknown_remaining and unknown_collapse.get("needs_instrumentation_or_semantics"):
        return "improve instrumentation because unknowns remain too large"
    attack_counts = Counter(attack_collapse.get("collapse_counts") or {})
    if attack_counts.get(AttackPressureLabel.WEAK_ATTACK_OVER_DEVELOPMENT.value, 0):
        return "implement a specific narrow weak-attack-over-development ablation next"
    return "freeze current experimental bundles and expand evaluation before default changes"


def _example_frames_markdown(events: list[dict[str, Any]]) -> str:
    lines = ["# Example Frames", "", "Collapsed same-observation delta examples; source results are weak trajectory context only.", ""]
    for event in events[:40]:
        default = (event.get("policy_choices") or {}).get(DEFAULT_POLICY) or {}
        lines.extend(
            [
                f"## {event.get('frame_id')}",
                "",
                f"- old family: {event.get('old_rulebook_family')}",
                f"- collapsed label: {event.get('new_collapsed_label')} confidence={event.get('confidence')}",
                f"- default: {default.get('option_type')} {default.get('card_name') or default.get('attack_name') or ''}",
                f"- active: {event.get('active_card_name')} energy={event.get('active_energy_count')}",
                f"- attack facts: {event.get('attack_pressure_facts', {}).get('best_attack')}",
                f"- followthrough: {event.get('followthrough_facts')}",
                "",
            ]
        )
    return "\n".join(lines)


def _readme(command: str) -> str:
    return "\n".join(
        [
            "# BabyMew Rulebook Results Review Bundle",
            "",
            "This bundle contains the V3.10 diagnostic multi-seed results sweep and bucket-collapse audit.",
            "",
            f"- generated by: `{command}`",
            "- default behavior is unchanged.",
            "- same-observation deltas are not full-game counterfactual proof.",
            "",
            "Inspect first: `executive_summary.md`, `multi_seed_matrix_summary.json`, `attack_pressure_collapse.json`, and `unknown_collapse.json`.",
            "",
        ]
    )


def _executive_summary(multi_seed: dict[str, Any], attack: dict[str, Any], unknown: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Executive Summary",
            "",
            "This is a diagnostic pass, not a behavior patch.",
            "",
            f"- seeds: {multi_seed.get('seeds')}",
            f"- games per pair: {multi_seed.get('games_per_pair')}",
            f"- aggregate rankings: {multi_seed.get('aggregate_rankings')}",
            f"- attack-pressure collapse: {attack.get('collapse_counts')}",
            f"- unknown collapse: {unknown.get('collapse_counts')}",
            f"- next recommendation: {_highest_value_next(attack, unknown, multi_seed)}",
            "",
        ]
    )


def _relevant_paths(run_dir: Path, bundle_dir: Path | None) -> str:
    paths = [
        run_dir / "report.md",
        run_dir / "multi_seed_matrix_summary.json",
        run_dir / "attack_pressure_collapse.json",
        run_dir / "unknown_collapse.json",
        run_dir / "collapsed_delta_events.jsonl",
        bundle_dir or "",
        Path("deepmew/trace/rulebook_results_sweep.py"),
        Path("scripts/audit_babymew_rulebook_results.py"),
        Path("tests/test_rulebook_results_sweep.py"),
        Path("deepmew/trace/rulebook_audit.py"),
        Path(".research/STATE.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def export_results_bundle(
    *,
    run_dir: Path,
    command: str,
    multi_seed: dict[str, Any],
    pair_sweep: dict[str, Any],
    attack: dict[str, Any],
    unknown: dict[str, Any],
    delta_family: dict[str, Any],
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=out_root)
    write_markdown(bundle_dir / "README.md", _readme(command))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(multi_seed, attack, unknown))
    write_json(bundle_dir / "multi_seed_matrix_summary.json", multi_seed)
    write_json(bundle_dir / "pair_sweep_summary.json", pair_sweep)
    write_json(bundle_dir / "attack_pressure_collapse.json", attack)
    write_json(bundle_dir / "unknown_collapse.json", unknown)
    write_json(bundle_dir / "delta_family_summary.json", delta_family)
    write_text(bundle_dir / "collapsed_delta_events.csv", _csv_text(events))
    write_jsonl(bundle_dir / "collapsed_delta_events.jsonl", events)
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(bundle_dir / "example_frames.md", _example_frames_markdown(events))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(run_dir, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=HANDOFF_KIND,
        command=command,
        inputs={"results_sweep_run": str(run_dir)},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary={
            "seeds": multi_seed.get("seeds"),
            "attack_pressure_sampled": attack.get("sampled_events"),
            "unknown_sampled": unknown.get("sampled_events"),
            "next": _highest_value_next(attack, unknown, multi_seed),
        },
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def write_results_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    frames: list[dict[str, Any]],
    matrices: list[MatrixSummary],
    inputs: dict[str, Any] | None = None,
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> RulebookResultsSweepResult:
    multi_seed = build_multi_seed_matrix_summary(matrices)
    pair_sweep = build_pair_sweep_summary(matrices)
    collapsed_events, snapshots = collapse_rulebook_events(frames)
    attack = build_attack_pressure_collapse(collapsed_events)
    unknown = build_unknown_collapse(collapsed_events)
    delta_family = build_delta_family_summary(collapsed_events)
    run_path = Path(run_dir)
    write_json(run_path / "multi_seed_matrix_summary.json", multi_seed)
    write_json(run_path / "pair_sweep_summary.json", pair_sweep)
    write_json(run_path / "attack_pressure_collapse.json", attack)
    write_json(run_path / "unknown_collapse.json", unknown)
    write_json(run_path / "delta_family_summary.json", delta_family)
    write_jsonl(run_path / "collapsed_delta_events.jsonl", collapsed_events)
    write_text(run_path / "collapsed_delta_events.csv", _csv_text(collapsed_events))
    write_jsonl(run_path / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(run_path / "example_frames.md", _example_frames_markdown(collapsed_events))
    write_markdown(
        run_path / "report.md",
        build_report(
            multi_seed=multi_seed,
            pair_sweep=pair_sweep,
            attack_collapse=attack,
            unknown_collapse=unknown,
            delta_family=delta_family,
        ),
    )
    outputs = {
        "report_markdown": str(run_path / "report.md"),
        "multi_seed_matrix_summary_json": str(run_path / "multi_seed_matrix_summary.json"),
        "pair_sweep_summary_json": str(run_path / "pair_sweep_summary.json"),
        "attack_pressure_collapse_json": str(run_path / "attack_pressure_collapse.json"),
        "unknown_collapse_json": str(run_path / "unknown_collapse.json"),
        "delta_family_summary_json": str(run_path / "delta_family_summary.json"),
        "collapsed_delta_events_jsonl": str(run_path / "collapsed_delta_events.jsonl"),
        "collapsed_delta_events_csv": str(run_path / "collapsed_delta_events.csv"),
        "state_machine_snapshots_jsonl": str(run_path / "state_machine_snapshots.jsonl"),
        "example_frames_markdown": str(run_path / "example_frames.md"),
    }
    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_results_bundle(
            run_dir=run_path,
            command=command,
            multi_seed=multi_seed,
            pair_sweep=pair_sweep,
            attack=attack,
            unknown=unknown,
            delta_family=delta_family,
            events=collapsed_events,
            snapshots=snapshots,
            out_root=handoff_root,
        )
        outputs["handoff_bundle"] = str(bundle_dir)
    manifest = build_manifest(
        run_id=run_id,
        kind=RUN_KIND,
        command=command,
        inputs=inputs or {},
        outputs=outputs,
        summary={
            "seeds": multi_seed.get("seeds"),
            "crashes": multi_seed.get("crashes"),
            "timeouts": multi_seed.get("timeouts"),
            "attack_pressure_sampled": attack.get("sampled_events"),
            "unknown_sampled": unknown.get("sampled_events"),
            "highest_value_next": _highest_value_next(attack, unknown, multi_seed),
        },
    )
    write_manifest(run_path / "manifest.json", manifest)
    return RulebookResultsSweepResult(
        run_dir=run_path,
        manifest_path=run_path / "manifest.json",
        summary={"multi_seed": multi_seed, "attack": attack, "unknown": unknown, "delta_family": delta_family},
        bundle_dir=bundle_dir,
    )
