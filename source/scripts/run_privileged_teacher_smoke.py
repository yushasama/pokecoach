"""Local safety smoke for the isolated fresh-simulator Q teacher.

The parent process launches a short-lived worker.  Exact simulator state exists
only in that worker and only behind a zero-argument SearchBegin closure.  The
worker writes aggregate supervision and boolean/count receipts; the parent
verifies the worker exited, rescans the artifact, and checks protected surfaces.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for candidate in (ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cg.api import (  # noqa: E402
    SelectContext,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from deepmew.deck import read_deck  # noqa: E402
from deepmew.q_guided.branching import (  # noqa: E402
    LegalSelectionCapExceeded,
    assert_private_content_absent,
    enumerate_legal_selections,
    select_epsilon_mixture,
    student_input_bytes as branching_student_input_bytes,
)
from deepmew.q_guided.forked_simulator_teacher import (  # noqa: E402
    ForkedSimulatorTeacher,
    ForkedSimulatorTeacherSafetyError,
)
from deepmew.q_guided.simulator_teacher import (  # noqa: E402
    PrivilegedSimulatorTeacher,
    SimulatorTeacherSafetyError,
)
from deepmew.q_guided.teacher_contract import (  # noqa: E402
    BranchRollout,
    RootTeacherEvidence,
    aggregate_teacher_target,
    assert_student_input_byte_identity,
    validate_safe_shard,
)
from deepmew.real_meta_arena.runner import SpecAgent  # noqa: E402
from deepmew.replay.contract.preflight import (  # noqa: E402
    require_accepted_canonical_decision_receipt,
)
from deepmew.replay.information_boundary import InformationBoundary  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402


SCHEMA_VERSION = "deepmew.privileged_simulator_teacher_smoke.v1"
FAILURE_SCHEMA_VERSION = "deepmew.privileged_simulator_teacher_failure.v1"
DEFAULT_DECK = ROOT / ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
DEFAULT_V_CHECKPOINT = Path(
    r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs"
    r"\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt"
)
DEFAULT_CALIBRATOR_DIR = (
    ROOT / ".research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
)
DEFAULT_V_CHECKPOINT_SHA256 = "cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9"
DEFAULT_V_CALIBRATOR_SHA256 = "6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398"
ORDER_SENSITIVE_CONTEXTS = frozenset({int(SelectContext.SKILL_ORDER)})
PROTECTED_PATHS = (
    ROOT / "main.py",
    ROOT / "deck.csv",
    ROOT / "cg",
    ROOT / "deepmew/submissions",
    ROOT / "deepmew/arena",
    ROOT / "deepmew/policy.py",
)


class SmokeSafetyError(RuntimeError):
    """Hard-stop failure with no private payload in its message."""


@dataclass(frozen=True)
class SmokeTeacherReceipt:
    isolated_mode_active: bool
    fork_clone_active: bool
    actor_visible_input_stable: bool
    reconstructed_roots_match: bool
    live_root_unchanged: bool
    begin_search_count: int
    search_end_count: int
    fork_count: int
    rollout_count: int
    failed_rollout_count: int
    timeout_count: int
    payload_rejection_count: int


def _combine_teacher_receipts(receipts: list[Any]) -> SmokeTeacherReceipt:
    if not receipts:
        raise SmokeSafetyError("teacher produced no world receipts")

    return SmokeTeacherReceipt(
        isolated_mode_active=all(receipt.isolated_mode_active for receipt in receipts),
        fork_clone_active=all(getattr(receipt, "fork_clone_active", False) for receipt in receipts),
        actor_visible_input_stable=all(receipt.actor_visible_input_stable for receipt in receipts),
        reconstructed_roots_match=all(receipt.reconstructed_roots_match for receipt in receipts),
        live_root_unchanged=all(receipt.live_root_unchanged for receipt in receipts),
        begin_search_count=sum(receipt.begin_search_count for receipt in receipts),
        search_end_count=sum(receipt.search_end_count for receipt in receipts),
        fork_count=sum(getattr(receipt, "fork_count", 0) for receipt in receipts),
        rollout_count=sum(receipt.rollout_count for receipt in receipts),
        failed_rollout_count=sum(getattr(receipt, "failed_rollout_count", 0) for receipt in receipts),
        timeout_count=sum(getattr(receipt, "timeout_count", 0) for receipt in receipts),
        payload_rejection_count=sum(getattr(receipt, "payload_rejection_count", 0) for receipt in receipts),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _tree_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    for path in sorted(files, key=lambda item: str(item.resolve()).casefold()):
        relative = str(path.resolve().relative_to(ROOT)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))

    return digest.hexdigest()


def _action_id(selection: list[int]) -> str:
    return json.dumps(selection, separators=(",", ":"))


def _order_sensitive(select: dict[str, Any]) -> bool:
    context = select.get("context")

    return isinstance(context, int) and not isinstance(context, bool) and context in ORDER_SENSITIVE_CONTEXTS


def _latest_visualizer() -> dict[str, Any]:
    try:
        decoded = json.loads(visualize_data())
    except Exception:
        raise SmokeSafetyError("live simulator visualizer could not be read") from None
    if not isinstance(decoded, list) or not decoded or not isinstance(decoded[-1], dict):
        raise SmokeSafetyError("live simulator visualizer has an invalid shape")

    return decoded[-1]


def _card_ids(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise SmokeSafetyError("private simulator zone has an invalid shape")
    result = []
    for value in values:
        if not isinstance(value, dict) or isinstance(value.get("id"), bool) or not isinstance(value.get("id"), int):
            raise SmokeSafetyError("private simulator zone contains an invalid card")
        result.append(int(value["id"]))

    return result


def _scrub(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _scrub(item)
        value.clear()
    elif isinstance(value, dict):
        for item in value.values():
            _scrub(item)
        value.clear()


class _LiveRootSearchFactory:
    """Private worker-only bridge from a live battle to CABT SearchBegin."""

    def __init__(self, actor_observation: dict[str, Any]) -> None:
        self._root = copy.deepcopy(actor_observation)
        self._root_digest = self._digest_live_root()
        self._closed = False

    def __repr__(self) -> str:
        return f"_LiveRootSearchFactory(closed={self._closed})"

    def _digest_live_root(self) -> str:
        latest = _latest_visualizer()
        payload = {
            "opaque_search_state": self._root.get("search_begin_input"),
            "engine_state": latest,
        }
        digest = _sha256_bytes(_canonical_bytes(payload))
        _scrub(payload)

        return digest

    def verify_unchanged(self) -> bool:
        if self._closed:
            return False

        return self._digest_live_root() == self._root_digest

    def begin(self) -> Any:
        if self._closed or not self.verify_unchanged():
            raise SmokeSafetyError("live simulator root changed before SearchBegin")
        latest = _latest_visualizer()
        current = self._root.get("current")
        players = current.get("players") if isinstance(current, dict) else None
        private_current = latest.get("current")
        private_players = private_current.get("players") if isinstance(private_current, dict) else None
        actor = current.get("yourIndex") if isinstance(current, dict) else None
        if (
            not isinstance(players, list)
            or not isinstance(private_players, list)
            or len(players) != 2
            or len(private_players) != 2
            or actor not in (0, 1)
        ):
            _scrub(latest)
            raise SmokeSafetyError("live simulator root cannot be reconstructed")
        own = private_players[actor]
        opponent = private_players[1 - actor]
        if not isinstance(own, dict) or not isinstance(opponent, dict):
            _scrub(latest)
            raise SmokeSafetyError("live simulator private players are invalid")
        own_deck = _card_ids(own.get("deck"))
        own_prize = _card_ids(own.get("prize"))
        opponent_deck = _card_ids(opponent.get("deck"))
        opponent_prize = _card_ids(opponent.get("prize"))
        opponent_hand = _card_ids(opponent.get("hand"))
        visible_opponent_active = players[1 - actor].get("active") if isinstance(players[1 - actor], dict) else None
        opponent_active = _card_ids(opponent.get("active")) if visible_opponent_active == [None] else []
        select = self._root.get("select")
        select_deck = select.get("deck") if isinstance(select, dict) else None
        expected = {
            "own_deck": 0 if select_deck is not None else int(players[actor].get("deckCount") or 0),
            "own_prize": len(players[actor].get("prize") or []),
            "opponent_deck": int(players[1 - actor].get("deckCount") or 0),
            "opponent_prize": len(players[1 - actor].get("prize") or []),
            "opponent_hand": int(players[1 - actor].get("handCount") or 0),
            "opponent_active": 1 if visible_opponent_active == [None] else 0,
        }
        if select_deck is not None:
            own_deck.clear()
        actual = {
            "own_deck": len(own_deck),
            "own_prize": len(own_prize),
            "opponent_deck": len(opponent_deck),
            "opponent_prize": len(opponent_prize),
            "opponent_hand": len(opponent_hand),
            "opponent_active": len(opponent_active),
        }
        if actual != expected:
            for zone in (own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active):
                zone.clear()
            _scrub(latest)
            raise SmokeSafetyError("private simulator zone counts do not match the actor root")
        try:
            state = search_begin(
                to_observation_class(copy.deepcopy(self._root)),
                own_deck,
                own_prize,
                opponent_deck,
                opponent_prize,
                opponent_hand,
                opponent_active,
                manual_coin=True,
            )
        except Exception:
            raise SmokeSafetyError("CABT SearchBegin rejected the fresh live root") from None
        finally:
            for zone in (own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active):
                zone.clear()
            _scrub(latest)

        return state

    @staticmethod
    def step(state: Any, selection: list[int]) -> Any:
        try:
            return search_step(int(state.searchId), selection)
        except Exception:
            raise SmokeSafetyError("CABT SearchStep rejected a teacher selection") from None

    @staticmethod
    def end() -> None:
        try:
            search_end()
        except Exception:
            raise SmokeSafetyError("CABT SearchEnd failed") from None

    def serialized_private_content_absent(self, serialized: bytes) -> bool:
        token = self._root.get("search_begin_input")
        if isinstance(token, str) and token.encode("ascii") in serialized:
            return False

        return b"search_begin_input" not in serialized

    def close(self) -> None:
        self._root.clear()
        self._root_digest = ""
        self._closed = True


def _load_v0(args: argparse.Namespace) -> Any:
    from deepmew.value_lab.frozen_v0_endpoint import FrozenV0Endpoint, FrozenV0Spec

    result = json.loads((Path(args.v_calibrator_dir) / "result.json").read_text(encoding="utf-8"))
    receipt = result.get("final_calibrator")
    if not isinstance(receipt, dict):
        raise SmokeSafetyError("frozen V0 calibrator receipt is unavailable")
    spec = FrozenV0Spec(
        checkpoint_sha256=args.v_checkpoint_sha256,
        calibrator_bundle_sha256=args.v_calibrator_sha256,
    )
    try:
        return FrozenV0Endpoint.load(
            Path(args.v_checkpoint),
            Path(args.v_calibrator_dir),
            receipt,
            spec,
        )
    except Exception:
        raise SmokeSafetyError("frozen V0 endpoint failed its identity gate") from None


def _score_v0_root(args: argparse.Namespace, actor_visible_root: dict[str, Any]) -> float:
    import numpy as np
    import run_unified_r3_value as value_source

    v0 = _load_v0(args)
    features = np.asarray(
        value_source._features(
            {"raw_observation": actor_visible_root},
            value_source.FEATURE_CONFIG_V2,
        ),
        dtype=np.float32,
    )

    return float(v0.score(features))


def _create_teacher_executor(
    root_observation: dict[str, Any],
    factory: _LiveRootSearchFactory,
    args: argparse.Namespace,
) -> PrivilegedSimulatorTeacher | ForkedSimulatorTeacher:
    common = {
        "begin_search": factory.begin,
        "step_search": factory.step,
        "end_search": factory.end,
        "verify_live_root_unchanged": factory.verify_unchanged,
    }
    if args.execution_mode == "fork":
        return ForkedSimulatorTeacher(
            root_observation,
            child_timeout_seconds=args.branch_child_timeout_seconds,
            **common,
        )

    return PrivilegedSimulatorTeacher(root_observation, **common)


def _continuation_selector(
    agents: dict[int, Any],
    *,
    epsilon: float,
    root_cap: int,
) -> Any:
    def select(observation: dict[str, Any], rng: random.Random) -> list[int]:
        current = observation.get("current")
        actor = current.get("yourIndex") if isinstance(current, dict) else None
        select_data = observation.get("select")
        if actor not in (0, 1) or not isinstance(select_data, dict):
            raise SimulatorTeacherSafetyError("continuation observation is invalid")

        return select_epsilon_mixture(
            select_data,
            order_sensitive=_order_sensitive(select_data),
            rng=rng,
            epsilon=epsilon,
            base_selector=lambda _select, local_rng: agents[int(actor)].select(observation, local_rng),
            root_cap=root_cap,
        )

    return select


def _complete_action_table(
    actor_root: dict[str, Any],
    selections: list[list[int]],
) -> list[dict[str, Any]]:
    select = actor_root.get("select")
    if not isinstance(select, dict):
        raise SmokeSafetyError("actor root lacks a legal selection")
    option_semantics = legal_action_table(actor_root, select)

    return [
        {
            "selection": list(selection),
            "semantics": [copy.deepcopy(option_semantics[index]) for index in selection],
        }
        for selection in selections
    ]


def _actor_visible_signature(root: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes({"root": root, "actions": actions}))


def _config_sha256(args: argparse.Namespace) -> str:
    config = {
        "schema_version": SCHEMA_VERSION,
        "source_seed": args.source_seed,
        "minimum_source_turn": args.minimum_source_turn,
        "maximum_source_turn": args.maximum_source_turn,
        "branch_seed": args.branch_seed,
        "execution_mode": args.execution_mode,
        "branch_child_timeout_seconds": args.branch_child_timeout_seconds,
        "world_samples": args.world_samples,
        "rollouts_per_action": args.rollouts_per_action,
        "root_action_cap": args.root_action_cap,
        "continuation_action_cap": args.continuation_action_cap,
        "continuation_epsilon": args.continuation_epsilon,
        "branch_max_steps": args.branch_max_steps,
        "source_policy_bucket": args.source_policy_bucket,
        "opponent_strength_bucket": args.opponent_strength_bucket,
        "v_checkpoint_sha256": args.v_checkpoint_sha256,
        "v_calibrator_sha256": args.v_calibrator_sha256,
        "skip_v0_score_after_validated_parent": args.skip_v0_score_after_validated_parent,
    }

    return _sha256_bytes(_canonical_bytes(config))


def _run_root(
    root_observation: dict[str, Any],
    behavior_selection: list[int],
    agents: dict[int, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    select = root_observation.get("select")
    if not isinstance(select, dict):
        raise SmokeSafetyError("sampled root has no legal selection")
    order_sensitive = _order_sensitive(select)
    selections = enumerate_legal_selections(
        select,
        order_sensitive=order_sensitive,
        root_cap=args.root_action_cap,
    )
    behavior = list(behavior_selection)
    if not order_sensitive:
        behavior.sort()
    if behavior not in selections:
        raise SmokeSafetyError("source policy action is not in the complete legal root")
    factory = _LiveRootSearchFactory(root_observation)
    continuation = _continuation_selector(
        agents,
        epsilon=args.continuation_epsilon,
        root_cap=args.continuation_action_cap,
    )
    reproducible = True
    reproducibility_components = {
        "terminal_result": True,
        "terminal_payoff": True,
        "logical_decision_steps": True,
        "public_trajectory": True,
    }
    config_sha = _config_sha256(args)
    evidence_rows: list[RootTeacherEvidence] = []
    teacher_receipts: list[Any] = []
    actor_root: dict[str, Any] | None = None
    actions: list[dict[str, Any]] | None = None
    action_ids = [_action_id(selection) for selection in selections]
    for world_sample_group in range(args.world_samples):
        rollouts: list[BranchRollout] = []
        teacher = _create_teacher_executor(root_observation, factory, args)
        with teacher:
            world_actor_root = teacher.actor_visible_root
            world_actions = _complete_action_table(world_actor_root, selections)
            if actor_root is None:
                actor_root = world_actor_root
                actions = world_actions
            elif (
                branching_student_input_bytes(actor_root)
                != branching_student_input_bytes(world_actor_root)
                or _canonical_bytes(actions) != _canonical_bytes(world_actions)
            ):
                raise SmokeSafetyError("compatible world actor-visible roots do not match")
            for action_index, selection in enumerate(selections):
                action_id = action_ids[action_index]
                for rollout_index in range(args.rollouts_per_action):
                    seed = args.branch_seed + rollout_index
                    first = teacher.rollout_terminal(
                        selection,
                        seed=seed,
                        continuation_selector=continuation,
                        max_steps=args.branch_max_steps,
                    )
                    repeated = teacher.rollout_terminal(
                        selection,
                        seed=seed,
                        continuation_selector=continuation,
                        max_steps=args.branch_max_steps,
                    )
                    reproducibility_components["terminal_result"] &= (
                        first.terminal_result == repeated.terminal_result
                    )
                    reproducibility_components["terminal_payoff"] &= (
                        first.terminal_payoff == repeated.terminal_payoff
                    )
                    reproducibility_components["logical_decision_steps"] &= (
                        first.logical_decision_steps == repeated.logical_decision_steps
                    )
                    reproducibility_components["public_trajectory"] &= (
                        first.public_trajectory_sha256 == repeated.public_trajectory_sha256
                    )
                    reproducible &= first == repeated
                    rollouts.append(
                        BranchRollout(
                            action_id=action_id,
                            seed=seed,
                            value=first.terminal_payoff,
                            terminal=True,
                        )
                    )
        teacher_receipts.append(teacher.receipt())
        if actor_root is None or actions is None:
            raise SmokeSafetyError("teacher actor-visible root is unavailable")
        evidence_rows.append(
            RootTeacherEvidence(
                root_id=f"modal-pilot-root-world-{world_sample_group}",
                game_id="modal-pilot-game",
                split="smoke",
                actor_visible_signature=_actor_visible_signature(actor_root, actions),
                actor_visible_root_observation=actor_root,
                actor_visible_legal_options=actions,
                legal_action_ids=action_ids,
                behavior_action_id=_action_id(behavior),
                world_sample_group=world_sample_group,
                continuation_policy_identity=args.source_policy_bucket,
                opponent_strength_bucket=args.opponent_strength_bucket,
                v_checkpoint_sha256=args.v_checkpoint_sha256,
                v_calibrator_sha256=args.v_calibrator_sha256,
                config_sha256=config_sha,
                rollouts=rollouts,
            )
        )
    teacher_receipt = _combine_teacher_receipts(teacher_receipts)
    if actor_root is None or actions is None:
        raise SmokeSafetyError("teacher actor-visible root is unavailable")
    aggregation = aggregate_teacher_target(tuple(evidence_rows))
    shard = dict(aggregation.safe_shard)
    validate_safe_shard(shard)
    without_teacher = {
        "actor_visible_root_observation": shard["actor_visible_root_observation"],
        "actor_visible_legal_options": shard["actor_visible_legal_options"],
    }
    assert_student_input_byte_identity(shard, without_teacher)
    synthetic_private_payload = {"opponent_hidden_hand": [987654321]}
    student_with_teacher = branching_student_input_bytes(
        actor_root,
        teacher_payload=synthetic_private_payload,
    )
    student_without_teacher = branching_student_input_bytes(actor_root)
    student_identical = student_with_teacher == student_without_teacher
    serialized = _canonical_bytes(shard)
    assert_private_content_absent(shard)
    private_absent = factory.serialized_private_content_absent(serialized)
    factory.close()
    checks = {
        "exact_branch_reproducibility": reproducible,
        "siblings_begin_from_same_root": teacher_receipt.reconstructed_roots_match,
        "root_simulator_state_unchanged": teacher_receipt.live_root_unchanged,
        "persisted_private_content_scan": private_absent,
        "search_begin_input_absent": b"search_begin_input" not in serialized,
        "student_actor_visible_only": student_identical,
        "execution_mode_active": (
            teacher_receipt.fork_clone_active
            if args.execution_mode == "fork"
            else isinstance(teacher, PrivilegedSimulatorTeacher)
        ),
        "branch_executor_safety_clean": all(
            getattr(teacher_receipt, name, 0) == 0
            for name in (
                "failed_rollout_count",
                "timeout_count",
                "payload_rejection_count",
            )
        ),
        "teacher_memory_released": (
            teacher_receipt.begin_search_count
            == teacher_receipt.search_end_count
            == args.world_samples
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        failed.extend(
            f"reproducibility_{name}"
            for name, passed in reproducibility_components.items()
            if not passed
        )
        root_select = actor_root.get("select") if isinstance(actor_root, dict) else None
        root_current = actor_root.get("current") if isinstance(actor_root, dict) else None
        if isinstance(root_select, dict) and isinstance(root_select.get("context"), int):
            failed.append(f"root_context_{root_select['context']}")
        if isinstance(root_current, dict) and isinstance(root_current.get("turn"), int):
            failed.append(f"root_turn_{root_current['turn']}")
        raise SmokeSafetyError(f"local privileged-teacher safety checks failed: {','.join(failed)}")
    diagnostics = {
        "teacher_receipt": asdict(teacher_receipt),
        "aggregation_status": aggregation.status,
        "aggregation_quarantine_reason": aggregation.quarantine_reason,
        "policy_control_eligible": aggregation.policy_control_eligible,
        "hidden_world_sensitivity": aggregation.hidden_world_sensitivity,
        "sampled_complete_actions": len(selections),
        "paired_rollouts_per_action": args.rollouts_per_action,
        "executed_searches": teacher_receipt.rollout_count,
        "world_sample_count": args.world_samples,
        "checks": checks,
        "reproducibility_components": reproducibility_components,
    }

    return shard, diagnostics


def _find_root_and_run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    decks = {0: read_deck(Path(args.deck0)), 1: read_deck(Path(args.deck1))}
    agents = {0: SpecAgent.from_str(args.policy0), 1: SpecAgent.from_str(args.policy1)}
    boundary = InformationBoundary.for_isolated_simulator_teacher()
    source_decisions = 0
    for game_index in range(args.source_games):
        observation = None
        try:
            observation, start = battle_start(decks[0], decks[1])
            if observation is None:
                raise SmokeSafetyError("source BattleStart failed")
            rng = random.Random(args.source_seed + game_index)
            for _ in range(args.source_max_steps):
                current = observation.get("current")
                if not isinstance(current, dict) or current.get("result") != -1:
                    break
                actor = current.get("yourIndex")
                if actor not in (0, 1):
                    raise SmokeSafetyError("source game actor is invalid")
                try:
                    selected = list(agents[int(actor)].select(observation, rng))
                except Exception:
                    raise SmokeSafetyError("source policy failed") from None
                select = observation.get("select")
                candidate = False
                turn = current.get("turn")
                battle_is_materialized = (
                    not isinstance(turn, bool)
                    and isinstance(turn, int)
                    and turn >= args.minimum_source_turn
                    and (
                        args.maximum_source_turn == 0
                        or turn <= args.maximum_source_turn
                    )
                )
                if isinstance(select, dict) and battle_is_materialized:
                    try:
                        safe_root = boundary.artifact_observation(observation)
                        legal = enumerate_legal_selections(
                            select,
                            order_sensitive=_order_sensitive(select),
                            root_cap=args.root_action_cap,
                        )
                        normalized = list(selected)
                        if not _order_sensitive(select):
                            normalized.sort()
                        candidate = len(legal) >= 2 and normalized in legal and safe_root is not None
                    except (ValueError, LegalSelectionCapExceeded):
                        candidate = False
                if candidate:
                    shard, diagnostics = _run_root(observation, selected, agents, args)
                    diagnostics["source_decisions_before_root"] = source_decisions

                    return shard, diagnostics
                try:
                    observation = battle_select(selected)
                except Exception:
                    raise SmokeSafetyError("source game selection failed") from None
                source_decisions += 1
            else:
                raise SmokeSafetyError("source game exceeded the logical-decision limit")
        finally:
            if observation is not None:
                battle_finish()

    raise SmokeSafetyError("no bounded actor-visible multi-action root was found")


def _write_worker_artifacts(worker_dir: Path, shard: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    worker_dir.mkdir(parents=True, exist_ok=False)
    shard_path = worker_dir / "branch_targets.jsonl"
    shard_path.write_bytes(_canonical_bytes(shard) + b"\n")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "worker_completed_safe",
        "teacher_process_id": os.getpid(),
        "safety": {"illegal_actions": 0, "crashes": 0, "timeouts": 0},
        **diagnostics,
    }
    (worker_dir / "worker_receipt.json").write_bytes(_canonical_bytes(receipt) + b"\n")


def _safe_failure_category(error: BaseException) -> str:
    if isinstance(error, (
        SmokeSafetyError,
        SimulatorTeacherSafetyError,
        ForkedSimulatorTeacherSafetyError,
    )):
        return str(error)
    if isinstance(error, LegalSelectionCapExceeded):
        return "legal selection cap exceeded"

    return f"unexpected {type(error).__name__}"


def _write_failure_receipt(path: Path, error: BaseException) -> None:
    receipt = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "worker_failed_safe",
        "failure_category": _safe_failure_category(error),
        "exception_type": type(error).__name__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(receipt) + b"\n")


def _write_fixed_failure_receipt(path: Path, category: str, exception_type: str) -> None:
    receipt = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "worker_failed_safe",
        "failure_category": category,
        "exception_type": exception_type,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(receipt) + b"\n")


def _run_worker(args: argparse.Namespace) -> int:
    try:
        shard, diagnostics = _find_root_and_run(args)
        _write_worker_artifacts(Path(args.worker_dir), shard, diagnostics)
    except Exception as error:
        if args.failure_receipt:
            _write_failure_receipt(Path(args.failure_receipt), error)

        return 2

    return 0


def _read_jsonl_one(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise SmokeSafetyError("aggregate branch shard must contain exactly one smoke row")
    value = json.loads(lines[0])
    validate_safe_shard(value)

    return value


def _run_parent(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    worker_dir = out_dir.with_name(f".{out_dir.name}.worker")
    failure_path = out_dir.with_name(f".{out_dir.name}.failure.json")
    if out_dir.exists() or worker_dir.exists() or failure_path.exists():
        raise SmokeSafetyError("smoke output path already exists")
    protected_before = _tree_sha256(PROTECTED_PATHS)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--canonical-decision-receipt",
        args.canonical_decision_receipt,
        "--worker-dir",
        str(worker_dir),
        "--failure-receipt",
        str(failure_path),
        "--execution-mode",
        args.execution_mode,
        "--branch-child-timeout-seconds",
        str(args.branch_child_timeout_seconds),
        "--world-samples",
        str(args.world_samples),
        "--deck0",
        args.deck0,
        "--deck1",
        args.deck1,
        "--policy0",
        args.policy0,
        "--policy1",
        args.policy1,
        "--source-games",
        str(args.source_games),
        "--source-seed",
        str(args.source_seed),
        "--source-max-steps",
        str(args.source_max_steps),
        "--minimum-source-turn",
        str(args.minimum_source_turn),
        "--maximum-source-turn",
        str(args.maximum_source_turn),
        "--branch-seed",
        str(args.branch_seed),
        "--rollouts-per-action",
        str(args.rollouts_per_action),
        "--root-action-cap",
        str(args.root_action_cap),
        "--continuation-action-cap",
        str(args.continuation_action_cap),
        "--continuation-epsilon",
        str(args.continuation_epsilon),
        "--branch-max-steps",
        str(args.branch_max_steps),
        "--source-policy-bucket",
        args.source_policy_bucket,
        "--opponent-strength-bucket",
        args.opponent_strength_bucket,
        "--v-checkpoint",
        args.v_checkpoint,
        "--v-calibrator-dir",
        args.v_calibrator_dir,
        "--v-checkpoint-sha256",
        args.v_checkpoint_sha256,
        "--v-calibrator-sha256",
        args.v_calibrator_sha256,
    ]
    if args.skip_v0_score_after_validated_parent:
        command.append("--skip-v0-score-after-validated-parent")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=args.worker_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _write_fixed_failure_receipt(
            failure_path,
            "isolated teacher worker timed out",
            "TimeoutExpired",
        )
        raise SmokeSafetyError("isolated teacher worker timed out") from None
    if completed.returncode != 0:
        category = "failure receipt unavailable"
        if failure_path.is_file():
            failure_receipt = json.loads(failure_path.read_text(encoding="utf-8"))
            category = str(failure_receipt.get("failure_category") or category)
        else:
            category = f"isolated teacher worker exit code {completed.returncode}"
            _write_fixed_failure_receipt(
                failure_path,
                category,
                "WorkerProcessExit",
            )
        raise SmokeSafetyError(f"isolated teacher worker failed: {category}")
    if completed.stdout.strip() or completed.stderr.strip():
        raise SmokeSafetyError("isolated teacher worker emitted unexpected output")
    shard_path = worker_dir / "branch_targets.jsonl"
    receipt_path = worker_dir / "worker_receipt.json"
    shard = _read_jsonl_one(shard_path)
    worker_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert_private_content_absent(shard)
    if args.skip_v0_score_after_validated_parent:
        root_v0_probability = None
        v0_score_status = "delegated_to_validated_miner_parent"
    else:
        root_v0_probability = _score_v0_root(args, shard["actor_visible_root_observation"])
        v0_score_status = "validated_in_parent"
    protected_after = _tree_sha256(PROTECTED_PATHS)
    checks = dict(worker_receipt.get("checks") or {})
    checks.update(
        {
            "protected_runtime_submission_unchanged": protected_before == protected_after,
            "teacher_process_exited": True,
            "aggregate_readable_after_teacher_exit": bool(shard),
        }
    )
    required = (
        "exact_branch_reproducibility",
        "siblings_begin_from_same_root",
        "root_simulator_state_unchanged",
        "persisted_private_content_scan",
        "search_begin_input_absent",
        "student_actor_visible_only",
        "execution_mode_active",
        "branch_executor_safety_clean",
        "protected_runtime_submission_unchanged",
        "teacher_memory_released",
        "teacher_process_exited",
        "aggregate_readable_after_teacher_exit",
    )
    if any(checks.get(name) is not True for name in required):
        raise SmokeSafetyError("parent safety gate rejected the teacher artifact")
    final_receipt = dict(worker_receipt)
    final_receipt.update(
        {
            "status": "passed_end_to_end_smoke",
            "execution_mode": args.execution_mode,
            "execution_platform": platform.system().lower(),
            "checks": checks,
            "branch_shard_sha256": _sha256_file(shard_path),
            "root_v0_probability": root_v0_probability,
            "v0_score_status": v0_score_status,
        }
    )
    final_receipt.pop("teacher_process_id", None)
    receipt_path.unlink()
    failure_path.unlink(missing_ok=True)
    (worker_dir / "smoke_receipt.json").write_bytes(_canonical_bytes(final_receipt) + b"\n")
    os.replace(worker_dir, out_dir)
    print(json.dumps({
        "status": final_receipt["status"],
        "out_dir": str(out_dir),
        "checks": checks,
    }, indent=2, sort_keys=True))

    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--out-dir", default=str(ROOT / ".research/q_guided_starmie/privileged_teacher_smoke_20260712"))
    parser.add_argument("--worker-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--failure-receipt", default="", help=argparse.SUPPRESS)
    parser.add_argument("--execution-mode", choices=("persistent", "fork"), default="persistent")
    parser.add_argument("--deck0", default=str(DEFAULT_DECK))
    parser.add_argument("--deck1", default=str(DEFAULT_DECK))
    parser.add_argument("--policy0", default="v2.4")
    parser.add_argument("--policy1", default="v2.4")
    parser.add_argument("--source-games", type=int, default=4)
    parser.add_argument("--source-seed", type=int, default=7_121_201)
    parser.add_argument("--source-max-steps", type=int, default=5_000)
    parser.add_argument("--minimum-source-turn", type=int, default=1)
    parser.add_argument("--maximum-source-turn", type=int, default=0)
    parser.add_argument("--branch-seed", type=int, default=7_122_201)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--world-samples", type=int, default=1)
    parser.add_argument("--root-action-cap", type=int, default=6)
    parser.add_argument("--continuation-action-cap", type=int, default=4_096)
    parser.add_argument("--continuation-epsilon", type=float, default=0.20)
    parser.add_argument("--branch-max-steps", type=int, default=5_000)
    parser.add_argument("--branch-child-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--source-policy-bucket", default="v2.4_epsilon_uniform_0.20")
    parser.add_argument("--opponent-strength-bucket", default="reference_v2.4")
    parser.add_argument("--v-checkpoint", default=str(DEFAULT_V_CHECKPOINT))
    parser.add_argument("--v-calibrator-dir", default=str(DEFAULT_CALIBRATOR_DIR))
    parser.add_argument("--v-checkpoint-sha256", default=DEFAULT_V_CHECKPOINT_SHA256)
    parser.add_argument("--v-calibrator-sha256", default=DEFAULT_V_CALIBRATOR_SHA256)
    parser.add_argument(
        "--skip-v0-score-after-validated-parent",
        action="store_true",
        help="reuse a V0 identity validation already completed by the invoking miner",
    )
    parser.add_argument("--worker-timeout-seconds", type=int, default=600)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    if args.world_samples < 1:
        raise SmokeSafetyError("world_samples must be positive")
    if args.minimum_source_turn < 1:
        raise SmokeSafetyError("minimum_source_turn must be positive")
    if args.maximum_source_turn < 0:
        raise SmokeSafetyError("maximum_source_turn must be non-negative")
    if args.maximum_source_turn and args.maximum_source_turn < args.minimum_source_turn:
        raise SmokeSafetyError("maximum_source_turn must be zero or at least the minimum")
    if args.worker_dir:
        return _run_worker(args)

    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
