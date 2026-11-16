"""Train one research-only Starmie or Grimmsnarl terminal-reward league learner.

The runner is deliberately outside the protected runtime/submission path. It
uses complete real games, actor-visible R4F features, exact +1/0/-1 terminal
returns, immutable policy/deck opponent pairs, and fail-closed full-state
checkpoints. A safety failure exits with code 86 so the Modal wrapper can stop
without retrying the failed lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import numpy as np
import torch

import run_r17_bucket_arena as arena
import run_unified_r4_ablation as r4
import deepmew.rl.opponents as opponent_registry
from deepmew.policy.selection import selection_bounds, valid_selection
from deepmew.rl.algorithms.r4f_ppo import (
    FrozenR4FState,
    R4FNativePPO,
    R4FPPOConfig,
    R4FPPOValidationError,
    R4FRolloutStep,
    R4FTerminalEpisode,
)
from deepmew.rl.deck_identity import (
    DeckIdentity,
    deck_identity_from_deck,
    r4f_feature_deck_hash,
    read_deck_any,
    resolve_repo_path,
)
from deepmew.rl.env_adapter import EnvAdapter, StepResult
from deepmew.rl.league_checkpoint import (
    LeagueCheckpointError,
    LeagueCheckpointBindings,
    LeagueCheckpointLoadResult,
    LeagueTrainingProgress,
    StateHooks,
    atomic_save_league_checkpoint,
    checkpoint_latest_ready_path,
    load_league_checkpoint,
    prove_resume_counter_advance,
    snapshot_league_checkpoint,
)
from deepmew.rl.opponents import (
    OpponentSampler,
    PolicyDeckBinding,
    make_opponent_sampler,
    register_policy_deck_binding,
    register_policy_spec_loader,
    remap_policy_artifact_paths,
    resolve_policy_deck_binding,
)
from deepmew.rl.perf.phase_timer import PhaseTimer
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.registry import resolve_experiment
from deepmew.rl1_teacher_bc.semantics import legal_action_table
from deepmew.second_order.reader import augment_option_features


SCHEMA_VERSION = "deepmew.terminal_league_runner.v1"
STATUS_SCHEMA_VERSION = "deepmew.terminal_league_status.v1"
FAILURE_SCHEMA_VERSION = "deepmew.terminal_league_safety_failure.v1"
SAFETY_HARD_STOP_EXIT_CODE = 86
TRAINING_HARD_STOP_EXIT_CODE = 87
SAFETY_KEYS = ("illegal_actions", "crashes", "timeouts")


@dataclass(frozen=True)
class LaneSpec:
    lane: str
    learner_id: str
    feature_policy_id: str
    health_experiment: str
    league_experiment: str


@dataclass(frozen=True)
class FrozenCampaign:
    """Validated config-only specialization of the terminal-PPO runner."""

    path: Path
    file_sha256: str
    lane: LaneSpec
    experiment: RLExperimentConfig
    mixture_name: str
    mixture_entries: tuple[tuple[str, float], ...]

    def make_sampler(self, *, seed: int) -> OpponentSampler:
        return OpponentSampler(self.mixture_name, list(self.mixture_entries), seed=seed)


LANES: dict[str, LaneSpec] = {
    "starmie": LaneSpec(
        lane="starmie",
        learner_id="terminal-starmie-g1",
        feature_policy_id="starmie",
        health_experiment="RL0_starmie_health_v0",
        league_experiment="RL0_starmie_league_terminal_v0",
    ),
    "grimmsnarl": LaneSpec(
        lane="grimmsnarl",
        learner_id="terminal-grimmsnarl-d2-g1",
        feature_policy_id="grimmsnarl",
        health_experiment="RL0_grimmsnarl_d2_health_v0",
        league_experiment="RL0_grimmsnarl_d2_terminal_v0",
    ),
}


class SafetyHardStop(RuntimeError):
    """A non-retriable illegal action, crash, or timeout was observed."""

    def __init__(self, category: str, detail: str, context: Mapping[str, Any] | None = None):
        if category not in SAFETY_KEYS and category != "prior_safety_failure":
            raise ValueError(f"unknown safety category {category!r}")
        super().__init__(detail)
        self.category = category
        self.detail = detail
        self.context = dict(context or {})


class TrainingHardStop(RuntimeError):
    """A deterministic learner/checkpoint invariant failed and must not retry."""

    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category = category
        self.detail = detail


@dataclass
class EnvironmentCursor:
    """Serializable between-game cursor; never contains an engine/private state."""

    rng: random.Random
    next_episode_index: int = 0
    active: bool = False

    def state_dict(self) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("environment cursor may be checkpointed only between games")

        return {
            "next_episode_index": self.next_episode_index,
            "rng_state": self.rng.getstate(),
            "engine_episode_active": False,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_fields = {"next_episode_index", "rng_state", "engine_episode_active"}
        if not isinstance(state, Mapping) or set(state) != expected_fields:
            raise ValueError("environment cursor state is incomplete or unexpected")
        if state["engine_episode_active"] is not False:
            raise ValueError("checkpoint contains an in-progress engine episode")
        episode_index = state["next_episode_index"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("environment episode cursor is invalid")
        self.rng.setstate(state["rng_state"])
        self.next_episode_index = episode_index
        self.active = False


@dataclass(frozen=True)
class EncodedLiveDecision:
    features: torch.Tensor
    min_count: int
    max_count: int
    select: dict[str, Any]


@dataclass(frozen=True)
class CollectedEpisode:
    episode: R4FTerminalEpisode
    opponent_index: int
    opponent_label: str
    pair_identity: Mapping[str, Any]
    our_seat: int
    result: int
    environment_decisions: int


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _load_frozen_campaign(path: str | Path) -> FrozenCampaign:
    campaign_path = Path(path).resolve(strict=True)
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("campaign config must be a JSON object")
    if payload.get("schema_version") != "deepmew.direct_exploiter_campaign.v1":
        raise ValueError("unsupported direct-exploiter campaign schema")
    required_text = (
        "campaign_id",
        "learner_id",
        "feature_policy_id",
        "learner_deck_id",
        "learner_deck_path",
        "learner_deck_sha256",
    )
    values = {key: str(payload.get(key) or "").strip() for key in required_text}
    if any(not value for value in values.values()):
        raise ValueError("campaign config is missing a required identity field")
    learner_deck_path = resolve_repo_path(values["learner_deck_path"]).resolve(strict=True)
    if _sha256_file(learner_deck_path) != values["learner_deck_sha256"].lower():
        raise ValueError("campaign learner deck hash drifted")
    raw_bindings = payload.get("policy_deck_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("campaign config needs policy_deck_bindings")
    binding_fields = {field.name for field in fields(PolicyDeckBinding)}
    registered_ids: set[str] = set()
    for raw_binding in raw_bindings:
        if not isinstance(raw_binding, Mapping):
            raise ValueError("campaign policy/deck binding must be an object")
        unknown = set(raw_binding) - binding_fields
        if unknown:
            raise ValueError(f"campaign policy/deck binding has unknown fields: {sorted(unknown)}")
        binding = PolicyDeckBinding(**dict(raw_binding))
        if binding.binding_id in registered_ids:
            raise ValueError(f"duplicate campaign binding: {binding.binding_id}")
        register_policy_deck_binding(binding)
        registered_ids.add(binding.binding_id)
    mixture = payload.get("opponent_mixture")
    if not isinstance(mixture, Mapping):
        raise ValueError("campaign config needs an opponent_mixture object")
    mixture_name = str(mixture.get("mixture_id") or "").strip()
    raw_entries = mixture.get("entries")
    if not mixture_name or not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("campaign opponent mixture is incomplete")
    mixture_entries: list[tuple[str, float]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("campaign mixture entry must be an object")
        binding_id = str(raw_entry.get("binding_id") or "").strip()
        weight = float(raw_entry.get("weight") or 0.0)
        if binding_id not in registered_ids or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("campaign mixture entry has an unknown binding or invalid weight")
        mixture_entries.append((f"binding:{binding_id}", weight))
    if not math.isclose(sum(weight for _, weight in mixture_entries), 1.0, abs_tol=1e-9):
        raise ValueError("campaign mixture weights must sum to exactly 1.0")
    experiment = RLExperimentConfig(
        experiment_id=f"RL0_{values['campaign_id']}",
        observation="base",
        action_mask="legal_only",
        reward="terminal_win_loss",
        opponent_sampler=mixture_name,
        feature_modules=(),
        seed=int(payload.get("seed") or 0),
        max_steps_per_game=int(payload.get("max_steps_per_game") or 5000),
        trainer_backend="r4f_native_ppo",
        agent_deck_id=values["learner_deck_id"],
        agent_deck_path=str(learner_deck_path),
        opponent_deck_id="per_binding_manifest",
        opponent_deck_path="",
        notes=str(payload.get("notes") or "frozen direct-exploiter terminal-PPO campaign"),
    ).validate()
    lane = LaneSpec(
        lane=values["campaign_id"],
        learner_id=values["learner_id"],
        feature_policy_id=values["feature_policy_id"],
        health_experiment=experiment.experiment_id,
        league_experiment=experiment.experiment_id,
    )

    return FrozenCampaign(
        path=campaign_path,
        file_sha256=_sha256_file(campaign_path),
        lane=lane,
        experiment=experiment,
        mixture_name=mixture_name,
        mixture_entries=tuple(mixture_entries),
    )


def preflight_campaign_opponents(
    campaign_config: str | Path,
    artifact_map_path: str | Path = "",
    *,
    seed: int,
) -> dict[str, Any]:
    """Materialize every frozen opponent through the production runner path."""

    binding_snapshot = dict(opponent_registry.POLICY_DECK_BINDINGS)
    try:
        campaign = _load_frozen_campaign(campaign_config)
        artifact_map = load_artifact_map(artifact_map_path)
        if artifact_map:
            remap_policy_artifact_paths(artifact_map)
        _register_artifact_loaders()
        sampler = campaign.make_sampler(seed=seed + 1)
        sampler.materialize_all()

        return {
            "schema_version": "deepmew.terminal_league_opponent_materialization_preflight.v1",
            "status": "PASS",
            "campaign_config_sha256": campaign.file_sha256,
            "mixture_name": campaign.mixture_name,
            "opponent_count": len(sampler.agents),
            "labels": list(sampler.labels),
            "pair_identities": [agent.pair_identity for agent in sampler.agents],
            "sampler_cursor_advanced": False,
            "registry_restored_after_preflight": True,
        }
    finally:
        opponent_registry.POLICY_DECK_BINDINGS.clear()
        opponent_registry.POLICY_DECK_BINDINGS.update(binding_snapshot)
        opponent_registry.resolve_policy_deck_binding.cache_clear()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_artifact_map(path: str | Path | None) -> dict[str, str]:
    if path is None or str(path) == "":
        return {}
    artifact_map_path = Path(path).resolve(strict=True)
    payload = json.loads(artifact_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact map must be a JSON object")
    normalized: dict[str, str] = {}
    for binding_id, artifact_path in payload.items():
        if not isinstance(binding_id, str) or not binding_id:
            raise ValueError("artifact map binding IDs must be non-empty strings")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"artifact map path for {binding_id!r} must be non-empty")
        normalized[binding_id] = artifact_path

    return normalized


def _register_artifact_loaders() -> None:
    def load_checkpoint(
        binding: PolicyDeckBinding,
        deck: tuple[int, ...],
        identity: DeckIdentity,
    ) -> Any:
        resolved = resolve_policy_deck_binding(binding)
        if resolved.policy_artifact_path is None:
            raise ValueError(f"{binding.binding_id}: checkpoint artifact is unresolved")

        return arena.CheckpointAgent(
            checkpoint=resolved.policy_artifact_path,
            label=binding.label,
            deck_policy=binding.deck_policy_id,
            deck_binding={
                "deck": list(deck),
                "deck_hash": binding.feature_deck_hash,
                "policy_id": binding.feature_policy_id or binding.binding_id,
            },
            feature_policy_id=binding.feature_policy_id or binding.binding_id,
        )

    def load_bundle(
        binding: PolicyDeckBinding,
        deck: tuple[int, ...],
        _identity: DeckIdentity,
    ) -> Any:
        resolved = resolve_policy_deck_binding(binding)
        if resolved.policy_artifact_path is None:
            raise ValueError(f"{binding.binding_id}: bundle artifact is unresolved")

        return arena.BundleAgent(
            bundle_dir=resolved.policy_artifact_path.parent,
            label=binding.label,
            feature_policy_id=binding.feature_policy_id,
            engine_deck=list(deck),
        )

    register_policy_spec_loader("checkpoint", load_checkpoint, replace=True)
    register_policy_spec_loader("bundle", load_bundle, replace=True)


def _frozen_equivalence_panel(input_dim: int) -> tuple[FrozenR4FState, ...]:
    return (
        FrozenR4FState(
            torch.linspace(-1.0, 1.0, steps=3 * input_dim).reshape(3, input_dim),
            min_count=0,
            max_count=2,
        ),
        FrozenR4FState(
            torch.linspace(0.75, -0.5, steps=5 * input_dim).reshape(5, input_dim),
            min_count=1,
            max_count=3,
            selected=(2,),
        ),
        FrozenR4FState(
            torch.linspace(-0.25, 0.9, steps=7 * input_dim).reshape(7, input_dim),
            min_count=2,
            max_count=4,
            selected=(1, 3),
        ),
    )


def encode_actor_visible_decision(
    obs: dict[str, Any],
    *,
    feature_policy_id: str,
    deck_hash: str,
    expected_input_dim: int,
) -> EncodedLiveDecision:
    select = obs.get("select")
    if not isinstance(select, dict):
        raise ValueError("actor-visible observation is missing select")
    raw_options = select.get("option")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("actor-visible selection has no legal options")
    table = legal_action_table(obs, select)
    if len(table) != len(raw_options):
        raise ValueError("legal action table does not match the engine option count")
    encoded = r4.build_live_features(
        obs,
        table,
        policy_id=feature_policy_id,
        deck_hash=deck_hash,
    )
    if encoded is None or encoded.legal_count != len(table):
        raise ValueError("R4F live feature encoder rejected a legal decision")
    features = augment_option_features(
        obs,
        table[: encoded.legal_count],
        encoded.features,
    )
    tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if tensor.ndim != 2 or tensor.shape != (len(table), expected_input_dim):
        raise ValueError(
            "R4F live feature shape mismatch: "
            f"expected {(len(table), expected_input_dim)}, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("R4F live features contain NaN/Inf")
    min_count, max_count = selection_bounds(select, len(table))
    if max_count < min_count or max_count <= 0:
        raise ValueError(
            f"invalid engine selection bounds [{min_count}, {max_count}] for {len(table)} options"
        )

    return EncodedLiveDecision(tensor, min_count, max_count, select)


def _raise_step_safety(
    step: StepResult,
    *,
    counters: dict[str, int],
    context: Mapping[str, Any],
) -> None:
    if step.crashed:
        counters["crashes"] += 1
        raise SafetyHardStop("crashes", str(step.info.get("error") or step.info), context)
    if step.timed_out:
        counters["timeouts"] += 1
        raise SafetyHardStop("timeouts", str(step.info.get("reason") or step.info), context)


def _terminal_reward(result: int, our_seat: int) -> float:
    if result == 2:
        return 0.0
    if result not in (0, 1):
        raise ValueError(f"terminal result must be winner seat 0/1 or draw 2, got {result!r}")

    return 1.0 if result == our_seat else -1.0


def collect_terminal_episode(
    *,
    learner: R4FNativePPO,
    sampler: Any,
    environment_cursor: EnvironmentCursor,
    learner_deck: Sequence[int],
    learner_deck_hash: str,
    feature_policy_id: str,
    max_steps: int,
    counters: dict[str, int],
    phase_timer: PhaseTimer | None = None,
) -> CollectedEpisode:
    sampled = sampler.sample()
    episode_index = environment_cursor.next_episode_index
    our_seat = episode_index % 2
    opponent_deck = sampled.agent.deck
    if opponent_deck is None or len(opponent_deck) != 60:
        raise ValueError(f"{sampled.agent.label}: opponent did not expose a 60-card deck")
    context = {
        "episode_index": episode_index,
        "our_seat": our_seat,
        "opponent_index": sampled.index,
        "opponent_label": sampled.agent.label,
        "pair_identity": sampled.pair_identity,
    }
    env = EnvAdapter(
        opponent=sampled.agent,
        our_seat=our_seat,
        our_deck=list(learner_deck),
        opp_deck=list(opponent_deck),
        max_steps=max_steps,
        opponent_seed=environment_cursor.rng.randrange(0, 2**31),
        phase_timer=phase_timer,
    )
    environment_cursor.active = True
    rollout_steps: list[R4FRolloutStep] = []
    try:
        with phase_timer.phase("env_reset") if phase_timer is not None else nullcontext():
            step = env.reset()
        _raise_step_safety(step, counters=counters, context=context)
        if step.done:
            raise ValueError("real environment ended before the learner received a decision")
        while not step.done:
            obs = step.obs
            if not isinstance(obs, dict):
                raise ValueError("real environment returned no actor-visible observation")
            with phase_timer.phase("observation_encode") if phase_timer is not None else nullcontext():
                encoded = encode_actor_visible_decision(
                    obs,
                    feature_policy_id=feature_policy_id,
                    deck_hash=learner_deck_hash,
                    expected_input_dim=learner.model.actor.config.input_dim,
                )
            with phase_timer.phase("policy_forward") if phase_timer is not None else nullcontext():
                action = learner.sample(
                    encoded.features,
                    encoded.min_count,
                    encoded.max_count,
                )
            selection = list(action.selection)
            if not valid_selection(selection, encoded.select):
                counters["illegal_actions"] += 1
                raise SafetyHardStop(
                    "illegal_actions",
                    f"learner selection {selection} violated the live legal mask",
                    context,
                )
            with phase_timer.phase("engine_step") if phase_timer is not None else nullcontext():
                next_step = env.step(selection)
            _raise_step_safety(next_step, counters=counters, context=context)
            with phase_timer.phase("rollout_record_pack") if phase_timer is not None else nullcontext():
                rollout_steps.append(
                    R4FRolloutStep(
                        option_features=encoded.features,
                        min_count=encoded.min_count,
                        max_count=encoded.max_count,
                        selection=action.selection,
                        old_log_prob=float(action.log_prob.detach().cpu()),
                        old_value=float(action.value.detach().cpu()),
                    )
                )
            step = next_step
        if not rollout_steps:
            raise ValueError("completed environment episode contains no learner decisions")
        if step.result is None:
            raise ValueError("terminal environment step is missing its result")
        reward = _terminal_reward(int(step.result), our_seat)
    finally:
        env.close()
        environment_cursor.active = False
    environment_cursor.next_episode_index += 1

    return CollectedEpisode(
        episode=R4FTerminalEpisode(tuple(rollout_steps), reward),
        opponent_index=sampled.index,
        opponent_label=sampled.agent.label,
        pair_identity=sampled.pair_identity,
        our_seat=our_seat,
        result=int(step.result),
        environment_decisions=env.selection_steps,
    )


def _initial_counters(opponent_count: int) -> dict[str, int]:
    counters = {
        "terminal_episodes": 0,
        "environment_decisions": 0,
        "non_forced_decisions": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "illegal_actions": 0,
        "crashes": 0,
        "timeouts": 0,
    }
    counters.update({f"opponent_index_{index:02d}": 0 for index in range(opponent_count)})

    return counters


def _record_episode(counters: dict[str, int], collected: CollectedEpisode) -> None:
    counters["terminal_episodes"] += 1
    counters["environment_decisions"] += collected.environment_decisions
    counters["non_forced_decisions"] += len(collected.episode.steps)
    counters[f"opponent_index_{collected.opponent_index:02d}"] += 1
    reward = collected.episode.terminal_reward
    if reward == 1.0:
        counters["wins"] += 1
    elif reward == -1.0:
        counters["losses"] += 1
    else:
        counters["draws"] += 1


def _require_healthy_update(stats: Mapping[str, Any]) -> None:
    for name, value in stats.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingHardStop("non_finite_update", f"update metric {name} is non-finite")
    if float(stats.get("actor_parameter_delta_norm") or 0.0) <= 0.0:
        raise TrainingHardStop("optimizer_not_updating", "actor parameter delta is zero")
    if float(stats.get("critic_parameter_delta_norm") or 0.0) <= 0.0:
        raise TrainingHardStop("optimizer_not_updating", "critic parameter delta is zero")


def _mixture_receipt(sampler: Any) -> dict[str, Any]:
    entries = sampler.entries

    return {
        "schema_version": "deepmew.terminal_league_mixture.v1",
        "sampler_name": sampler.name,
        "entries": [
            {
                "raw_spec": raw_spec,
                "weight": float(weight),
                "normalized_weight": float(sampler.normalized[index]),
                "pair_identity": sampler.agents[index].pair_identity,
            }
            for index, (raw_spec, weight) in enumerate(entries)
        ],
    }


def _build_bindings(
    *,
    lane: LaneSpec,
    experiment: Any,
    learner_deck_identity: DeckIdentity,
    learner_feature_deck_hash: str,
    learner_deck_file_sha256: str,
    sampler: Any,
    source_checkpoint_sha256: str,
    source_payload: Mapping[str, Any],
    ppo_config: R4FPPOConfig,
    precision: str,
    rollout_decisions: int,
) -> LeagueCheckpointBindings:
    mixture_receipt = _mixture_receipt(sampler)
    training_config = {
        "schema_version": SCHEMA_VERSION,
        "lane": lane.lane,
        "experiment": experiment.to_dict(),
        "feature_policy_id": lane.feature_policy_id,
        "precision": precision,
        "rollout_decisions": rollout_decisions,
        "ppo": asdict(ppo_config),
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    observation_schema = {
        "schema_version": "deepmew.terminal_league_actor_visible_r4f.v1",
        "input_dim": int(source_payload["input_dim"]),
        "feature_mode": source_payload.get("feature_mode"),
        "state_schema": list(source_payload.get("state_schema") or []),
        "option_schema": list(source_payload.get("option_schema") or []),
        "feature_policy_id": lane.feature_policy_id,
        "deck_hash": learner_deck_identity.deck_hash,
        "feature_deck_hash": learner_feature_deck_hash,
        "hidden_state_allowed": False,
    }
    action_schema = {
        "schema_version": "deepmew.terminal_league_r4f_action.v1",
        "semantics": "ordered_unique_live_option_indices_plus_stop",
        "mask": "engine_live_legal_options_and_inclusive_count_bounds",
        "learner_sanitization": False,
        "terminal_rewards": [-1, 0, 1],
    }
    league_snapshot = {
        "schema_version": "deepmew.terminal_league_snapshot.v1",
        "generation": 0,
        "frozen": True,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "members": [agent.pair_identity for agent in sampler.agents],
    }
    deck_receipt = {
        "schema_version": "deepmew.terminal_league_deck_binding.v1",
        "identity": learner_deck_identity.to_dict(),
        "feature_deck_hash": learner_feature_deck_hash,
        "deck_file_sha256": learner_deck_file_sha256,
    }

    return LeagueCheckpointBindings(
        learner_id=lane.learner_id,
        training_config_sha256=_sha256_json(training_config),
        opponent_mixture_receipt=mixture_receipt,
        league_snapshot_receipt=league_snapshot,
        deck_binding_receipt=deck_receipt,
        observation_schema_sha256=_sha256_json(observation_schema),
        action_schema_sha256=_sha256_json(action_schema),
    )


def _disabled_fp32_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def _parse_milestone_fractions(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return ()
    fractions = tuple(float(value) for value in raw.split(","))
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("milestone fractions must be finite values in (0,1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("milestone fractions must be unique and strictly increasing")

    return fractions


def _preserve_reached_milestones(
    *,
    run_dir: Path,
    progress: LeagueTrainingProgress,
    target_seconds: float,
    fractions: Sequence[float],
) -> list[str]:
    if target_seconds <= 0.0:
        return []
    preserved: list[str] = []
    for fraction in fractions:
        if progress.cumulative_training_seconds < target_seconds * fraction:
            continue
        label = f"{round(fraction * 100):03d}pct"
        milestone_dir = run_dir / "milestones" / label
        receipt_path = milestone_dir / "receipt.json"
        if receipt_path.exists():
            continue
        if milestone_dir.exists():
            raise ValueError(f"incomplete milestone directory already exists: {milestone_dir}")
        staging_dir = milestone_dir.with_name(f".{label}.{uuid.uuid4().hex}.tmp")
        staging_dir.mkdir(parents=True, exist_ok=False)
        try:
            checkpoint_snapshot = snapshot_league_checkpoint(
                run_dir / "league_checkpoint.pt",
                staging_dir / "league_checkpoint.pt",
            )
            copied: dict[str, dict[str, Any]] = dict(checkpoint_snapshot["files"])
            for name in ("actor_latest.pt", "status.json"):
                source = run_dir / name
                if not source.is_file():
                    raise ValueError(f"milestone source is missing: {source}")
                destination = staging_dir / name
                shutil.copy2(source, destination)
                copied[name] = {
                    "path": name,
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            _atomic_write_json(
                staging_dir / "receipt.json",
                {
                    "schema_version": "deepmew.direct_exploiter_milestone.v2",
                    "fraction": fraction,
                    "target_seconds": target_seconds,
                    "progress": progress.to_dict(),
                    "checkpoint": {
                        key: value
                        for key, value in checkpoint_snapshot.items()
                        if key != "files"
                    },
                    "files": copied,
                },
            )
            os.replace(staging_dir, milestone_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        preserved.append(label)

    return preserved


def _progress(
    *,
    learner: R4FNativePPO,
    base_seconds: float,
    session_started: float,
    counters: Mapping[str, int],
) -> LeagueTrainingProgress:
    if learner.state.episode_count != counters["terminal_episodes"]:
        raise ValueError("learner and environment episode counters disagree")

    return LeagueTrainingProgress(
        environment_decisions=learner.state.decision_count,
        update_count=learner.state.update_count,
        cumulative_training_seconds=base_seconds + max(0.0, time.perf_counter() - session_started),
        counters=counters,
    )


def _validation_receipt(
    *,
    learner: R4FNativePPO,
    latest_update: Mapping[str, Any] | None,
    counters: Mapping[str, int],
) -> dict[str, Any]:
    safety = {name: int(counters[name]) for name in SAFETY_KEYS}
    if any(safety.values()):
        raise SafetyHardStop(
            next(name for name, value in safety.items() if value),
            f"refusing checkpoint with non-zero safety: {safety}",
        )

    return {
        "schema_version": "deepmew.terminal_league_validation.v1",
        "safety": safety,
        "actor_equivalence": asdict(learner.proof),
        "latest_update": dict(latest_update or {"status": "not_yet_updated"}),
        "learner_output_validation": "hard_fail_no_sanitization",
        "opponent_fallback_monitor": "not_instrumented",
        "runtime_contamination": False,
        "private_state_persisted": False,
    }


def _save_training_state(
    *,
    run_dir: Path,
    lane: LaneSpec,
    learner: R4FNativePPO,
    scheduler: Any,
    scaler: Any,
    environment_cursor: EnvironmentCursor,
    sampler: Any,
    bindings: LeagueCheckpointBindings,
    progress: LeagueTrainingProgress,
    latest_update: Mapping[str, Any] | None,
    minimum_cumulative_seconds: float,
    experiment_id: str,
    last_episode: CollectedEpisode | None,
) -> str:
    checkpoint_path = run_dir / "league_checkpoint.pt"
    validation = _validation_receipt(
        learner=learner,
        latest_update=latest_update,
        counters=progress.counters,
    )
    checkpoint_sha256 = atomic_save_league_checkpoint(
        checkpoint_path,
        actor=learner.model.actor,
        critic=learner.model.critic,
        optimizer=learner.optimizer,
        scheduler=scheduler,
        scaler=scaler,
        environment_state=StateHooks(
            environment_cursor.state_dict,
            environment_cursor.load_state_dict,
            "between-game environment cursor",
        ),
        sampler_state=StateHooks(
            sampler.state_dict,
            sampler.load_state_dict,
            "opponent sampler cursor",
        ),
        progress=progress,
        bindings=bindings,
        validation_receipt=validation,
    )
    learner.export_actor(
        run_dir / "actor_latest.pt",
        training_metadata={
            "lane": lane.lane,
            "experiment_id": experiment_id,
            "progress": progress.to_dict(),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": (
            "minimum_training_floor_reached"
            if minimum_cumulative_seconds > 0
            and progress.cumulative_training_seconds >= minimum_cumulative_seconds
            else "training"
        ),
        "lane": lane.lane,
        "experiment_id": experiment_id,
        "minimum_cumulative_seconds": minimum_cumulative_seconds,
        "progress": progress.to_dict(),
        "checkpoint_sha256": checkpoint_sha256,
        "actor_checkpoint": str(run_dir / "actor_latest.pt"),
        "bindings": bindings.to_dict(),
        "validation": validation,
        "last_episode": (
            {
                "opponent_index": last_episode.opponent_index,
                "opponent_label": last_episode.opponent_label,
                "pair_identity": dict(last_episode.pair_identity),
                "our_seat": last_episode.our_seat,
                "result": last_episode.result,
                "decisions": len(last_episode.episode.steps),
                "terminal_reward": last_episode.episode.terminal_reward,
            }
            if last_episode is not None
            else None
        ),
    }
    _atomic_write_json(run_dir / "status.json", status)

    return checkpoint_sha256


def _load_source_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("initial R4F checkpoint payload must be a mapping")

    return payload


def initialize_frozen_critic(
    learner: R4FNativePPO,
    actor_source_payload: Mapping[str, Any],
    critic_checkpoint: Path,
    critic_config_receipt: Path | None = None,
) -> dict[str, Any]:
    """Load a critic only when its frozen actor is tensor-identical to the learner."""

    payload = torch.load(critic_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("critic initialization checkpoint must be a mapping")
    actor_state = payload.get("actor_state")
    critic_state = payload.get("critic_state")
    ppo_config = payload.get("ppo_config")
    if not isinstance(actor_state, Mapping) or not isinstance(critic_state, Mapping):
        raise ValueError("critic initialization checkpoint lacks actor or critic state")
    config_receipt_sha256 = None
    ppo_config_source = "embedded_checkpoint"
    if not isinstance(ppo_config, Mapping):
        if critic_config_receipt is None:
            raise ValueError("critic initialization checkpoint lacks PPO configuration")
        config_payload = json.loads(critic_config_receipt.read_text(encoding="utf-8"))
        config = config_payload.get("config")
        ppo_config = (
            config.get("ppo")
            if isinstance(config, Mapping)
            else config_payload.get("ppo")
        )
        if not isinstance(ppo_config, Mapping):
            raise ValueError("critic config receipt lacks PPO configuration")
        config_receipt_sha256 = _sha256_file(critic_config_receipt)
        ppo_config_source = "bound_legacy_config_receipt"
    source_actor_state = actor_source_payload.get("model_state")
    if not isinstance(source_actor_state, Mapping):
        raise ValueError("actor source checkpoint lacks model state")
    if actor_state.keys() != source_actor_state.keys():
        raise ValueError("critic checkpoint actor keys disagree with actor source")
    for name, expected in source_actor_state.items():
        actual = actor_state[name]
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise ValueError(f"actor tensor {name} is malformed")
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise ValueError(f"critic checkpoint actor tensor {name} disagrees with actor source")
    expected_hidden_dim = int(ppo_config.get("critic_hidden_dim", -1))
    if expected_hidden_dim != learner.config.critic_hidden_dim:
        raise ValueError("critic hidden dimension disagrees with frozen PPO configuration")
    current_critic = learner.model.critic.state_dict()
    if critic_state.keys() != current_critic.keys():
        raise ValueError("critic state keys disagree with terminal-league critic")
    for name, value in critic_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"critic tensor {name} is malformed")
        if value.shape != current_critic[name].shape:
            raise ValueError(f"critic tensor {name} has an incompatible shape")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"critic tensor {name} contains NaN/Inf")
    learner.model.critic.load_state_dict(dict(critic_state), strict=True)

    return {
        "mode": "exact_frozen_actor_and_critic_fresh_optimizer",
        "checkpoint": str(critic_checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(critic_checkpoint.resolve()),
        "actor_source_match": True,
        "actor_tensors_checked": len(actor_state),
        "critic_tensors_loaded": len(critic_state),
        "critic_hidden_dim": expected_hidden_dim,
        "ppo_config_source": ppo_config_source,
        "critic_config_receipt": (
            str(critic_config_receipt.resolve()) if critic_config_receipt is not None else None
        ),
        "critic_config_receipt_sha256": config_receipt_sha256,
        "optimizer_state_loaded": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANES))
    parser.add_argument("--campaign-config", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--init-critic-checkpoint", default="")
    parser.add_argument("--init-critic-config-receipt", default="")
    parser.add_argument("--artifact-map", default="")
    parser.add_argument("--minimum-cumulative-seconds", type=float, required=True)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=600.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--seed", type=int, default=9271301)
    parser.add_argument("--rollout-decisions", type=int, default=128)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--milestone-fractions", default="")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if bool(args.lane) == bool(args.campaign_config):
        raise ValueError("pass exactly one of --lane or --campaign-config")
    if args.campaign_config and args.health:
        raise ValueError("campaign configs do not support --health")
    if not math.isfinite(args.minimum_cumulative_seconds) or args.minimum_cumulative_seconds < 0:
        raise ValueError("minimum cumulative seconds must be finite and non-negative")
    if not math.isfinite(args.checkpoint_interval_seconds) or args.checkpoint_interval_seconds <= 0:
        raise ValueError("checkpoint interval seconds must be finite and positive")
    if args.rollout_decisions <= 0:
        raise ValueError("rollout decisions must be positive")
    if args.max_decisions < 0:
        raise ValueError("max decisions must be non-negative")
    if args.minimum_cumulative_seconds == 0 and args.max_decisions == 0:
        raise ValueError("a run needs a positive cumulative-time floor or max-decisions bound")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    milestone_fractions = _parse_milestone_fractions(args.milestone_fractions)
    if milestone_fractions and args.minimum_cumulative_seconds <= 0.0:
        raise ValueError("milestone fractions require a cumulative-time floor")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    milestone_fractions = _parse_milestone_fractions(args.milestone_fractions)
    campaign = _load_frozen_campaign(args.campaign_config) if args.campaign_config else None
    if campaign is None:
        lane = LANES[args.lane]
        experiment_id = lane.health_experiment if args.health else lane.league_experiment
        experiment = resolve_experiment(experiment_id)
    else:
        lane = campaign.lane
        experiment = campaign.experiment
        experiment_id = experiment.experiment_id
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    phase_timer = (
        PhaseTimer(
            run_id=f"{lane.lane}-{args.seed}",
            out_dir=run_dir / "perf",
            config={"campaign": lane.lane, "seed": args.seed, "device": args.device},
        )
        if args.profile
        else None
    )
    prior_failure_path = run_dir / "failure.json"
    if prior_failure_path.exists():
        raise SafetyHardStop(
            "prior_safety_failure",
            "run directory already contains a safety hard-stop receipt",
            {"failure_path": str(prior_failure_path)},
        )
    prior_training_failure_path = run_dir / "training_failure.json"
    if prior_training_failure_path.exists():
        raise TrainingHardStop(
            "prior_training_failure",
            "run directory already contains a non-retriable training failure receipt",
        )
    checkpoint_path = run_dir / "league_checkpoint.pt"
    checkpoint_available = bool(
        checkpoint_path.exists() or checkpoint_latest_ready_path(checkpoint_path).exists()
    )
    if checkpoint_available and not args.resume:
        raise ValueError("full-state checkpoint exists; pass --resume or use a new run directory")

    artifact_map = load_artifact_map(args.artifact_map)
    if artifact_map:
        remap_policy_artifact_paths(artifact_map)
    _register_artifact_loaders()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    init_checkpoint = Path(args.init_checkpoint).resolve(strict=True)
    with phase_timer.phase("checkpoint_load") if phase_timer is not None else nullcontext():
        source_payload = _load_source_payload(init_checkpoint)
    input_dim = int(source_payload["input_dim"])
    ppo_config = R4FPPOConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=1.0,
        gae_lambda=0.95,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        critic_hidden_dim=args.critic_hidden_dim,
        device=args.device,
    )
    with phase_timer.phase("model_initialization") if phase_timer is not None else nullcontext():
        learner = R4FNativePPO.from_r4f_checkpoint(
            init_checkpoint,
            _frozen_equivalence_panel(input_dim),
            ppo_config,
        )
    critic_initialization = None
    if args.init_critic_checkpoint:
        critic_initialization = initialize_frozen_critic(
            learner,
            source_payload,
            Path(args.init_critic_checkpoint).resolve(strict=True),
            (
                Path(args.init_critic_config_receipt).resolve(strict=True)
                if args.init_critic_config_receipt
                else None
            ),
        )
    elif args.init_critic_config_receipt:
        raise ValueError("--init-critic-config-receipt requires --init-critic-checkpoint")
    if not learner.proof.passed:
        raise ValueError("r7 actor equivalence proof failed")
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner.optimizer, lambda _step: 1.0)
    scaler = _disabled_fp32_scaler()

    learner_deck_path = resolve_repo_path(experiment.agent_deck_path).resolve(strict=True)
    learner_deck = read_deck_any(learner_deck_path)
    if len(learner_deck) != 60:
        raise ValueError(f"learner deck must contain 60 cards, got {len(learner_deck)}")
    learner_deck_identity = deck_identity_from_deck(
        policy_id=lane.learner_id,
        policy_role="research_terminal_league_learner",
        deck_id=experiment.agent_deck_id,
        deck_role="frozen_training_deck",
        deck=learner_deck,
        deck_status="runnable_full_deck",
        pairing_role=f"{lane.lane}_terminal_league_training",
        deck_path=learner_deck_path,
        deck_source_id=_sha256_file(learner_deck_path),
        notes="Research-only learner deck; no protected runtime or submission binding.",
    )
    learner_feature_deck_hash = r4f_feature_deck_hash(learner_deck)
    sampler = (
        make_opponent_sampler(experiment.opponent_sampler, seed=args.seed + 1)
        if campaign is None
        else campaign.make_sampler(seed=args.seed + 1)
    )
    with phase_timer.phase("opponent_binding") if phase_timer is not None else nullcontext():
        sampler.materialize_all()
    environment_cursor = EnvironmentCursor(random.Random(args.seed + 2))
    counters = _initial_counters(len(sampler.agents))
    bindings = _build_bindings(
        lane=lane,
        experiment=experiment,
        learner_deck_identity=learner_deck_identity,
        learner_feature_deck_hash=learner_feature_deck_hash,
        learner_deck_file_sha256=_sha256_file(learner_deck_path),
        sampler=sampler,
        source_checkpoint_sha256=learner.proof.source_checkpoint_sha256,
        source_payload=source_payload,
        ppo_config=ppo_config,
        precision=args.precision,
        rollout_decisions=args.rollout_decisions,
    )
    config_receipt = {
        "schema_version": SCHEMA_VERSION,
        "lane": lane.lane,
        "experiment": experiment.to_dict(),
        "source_checkpoint": str(init_checkpoint),
        "source_checkpoint_sha256": learner.proof.source_checkpoint_sha256,
        "critic_initialization": critic_initialization,
        "learner_feature_deck_hash": learner_feature_deck_hash,
        "artifact_map": artifact_map,
        "ppo": asdict(ppo_config),
        "precision": args.precision,
        "minimum_cumulative_seconds": args.minimum_cumulative_seconds,
        "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
        "rollout_decisions": args.rollout_decisions,
        "max_decisions": args.max_decisions,
        "milestone_fractions": list(milestone_fractions),
        "seed": args.seed,
        "bindings": bindings.to_dict(),
        "campaign_config": (
            {
                "path": str(campaign.path),
                "sha256": campaign.file_sha256,
            }
            if campaign is not None
            else None
        ),
    }
    config_receipt_path = run_dir / "config_receipt.json"
    if config_receipt_path.exists():
        existing_config_receipt = json.loads(config_receipt_path.read_text(encoding="ascii"))
        scheduling_fields = {
            "minimum_cumulative_seconds",
            "checkpoint_interval_seconds",
            "max_decisions",
        }
        existing_immutable = {
            key: value
            for key, value in existing_config_receipt.items()
            if key not in scheduling_fields
        }
        current_immutable = {
            key: value
            for key, value in config_receipt.items()
            if key not in scheduling_fields
        }
        if existing_immutable != current_immutable:
            raise ValueError("run configuration disagrees with the immutable config receipt")
    else:
        _atomic_write_json(config_receipt_path, config_receipt)

    restored: LeagueCheckpointLoadResult | None = None
    base_seconds = 0.0
    if checkpoint_available:
        restored = load_league_checkpoint(
            checkpoint_path,
            actor=learner.model.actor,
            critic=learner.model.critic,
            optimizer=learner.optimizer,
            scheduler=scheduler,
            scaler=scaler,
            environment_state=StateHooks(
                environment_cursor.state_dict,
                environment_cursor.load_state_dict,
                "between-game environment cursor",
            ),
            sampler_state=StateHooks(
                sampler.state_dict,
                sampler.load_state_dict,
                "opponent sampler cursor",
            ),
            expected_bindings=bindings,
            map_location=args.device,
        )
        counters = dict(restored.progress.counters)
        learner.state.update_count = restored.progress.update_count
        learner.state.episode_count = counters["terminal_episodes"]
        learner.state.decision_count = restored.progress.environment_decisions
        if environment_cursor.next_episode_index != learner.state.episode_count:
            raise ValueError("restored environment cursor disagrees with learner episodes")
        base_seconds = restored.progress.cumulative_training_seconds
        _atomic_write_json(run_dir / "restore_receipt.json", dict(restored.restore_receipt))

    session_started = time.perf_counter()
    latest_update: Mapping[str, Any] | None = None
    if restored is not None:
        restored_latest_update = restored.validation_receipt.get("latest_update")
        if not isinstance(restored_latest_update, Mapping):
            raise LeagueCheckpointError(
                "restored validation receipt lacks latest update state"
            )
        latest_update = dict(restored_latest_update)
    last_episode: CollectedEpisode | None = None
    pending: list[R4FTerminalEpisode] = []
    pending_decisions = 0
    last_checkpoint_wall = session_started

    while True:
        current_seconds = base_seconds + max(0.0, time.perf_counter() - session_started)
        floor_reached = bool(
            args.minimum_cumulative_seconds > 0
            and current_seconds >= args.minimum_cumulative_seconds
        )
        max_reached = bool(
            args.max_decisions
            and learner.state.decision_count + pending_decisions >= args.max_decisions
        )
        if (floor_reached or max_reached) and not pending:
            break
        collected = collect_terminal_episode(
            learner=learner,
            sampler=sampler,
            environment_cursor=environment_cursor,
            learner_deck=learner_deck,
            learner_deck_hash=learner_feature_deck_hash,
            feature_policy_id=lane.feature_policy_id,
            max_steps=experiment.max_steps_per_game,
            counters=counters,
            phase_timer=phase_timer,
        )
        last_episode = collected
        pending.append(collected.episode)
        pending_decisions += len(collected.episode.steps)
        _record_episode(counters, collected)

        checkpoint_due = (
            time.perf_counter() - last_checkpoint_wall >= args.checkpoint_interval_seconds
        )
        floor_reached = bool(
            args.minimum_cumulative_seconds > 0
            and base_seconds + max(0.0, time.perf_counter() - session_started)
            >= args.minimum_cumulative_seconds
        )
        max_reached = bool(
            args.max_decisions
            and learner.state.decision_count + pending_decisions >= args.max_decisions
        )
        update_due = pending_decisions >= args.rollout_decisions
        if not (checkpoint_due or floor_reached or max_reached or update_due):
            continue

        with phase_timer.phase("optimizer_step") if phase_timer is not None else nullcontext():
            update_stats = learner.update(tuple(pending))
        scheduler.step()
        latest_update = asdict(update_stats)
        _require_healthy_update(latest_update)
        pending.clear()
        pending_decisions = 0
        print(
            json.dumps(
                {
                    "event": "update",
                    "lane": lane.lane,
                    "episodes": learner.state.episode_count,
                    "decisions": learner.state.decision_count,
                    "updates": learner.state.update_count,
                    "cumulative_seconds": round(
                        base_seconds + time.perf_counter() - session_started,
                        3,
                    ),
                    "safety": {name: counters[name] for name in SAFETY_KEYS},
                    "stats": latest_update,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if checkpoint_due or floor_reached or max_reached:
            progress = _progress(
                learner=learner,
                base_seconds=base_seconds,
                session_started=session_started,
                counters=counters,
            )
            with phase_timer.phase("checkpoint_write") if phase_timer is not None else nullcontext():
                checkpoint_sha256 = _save_training_state(
                    run_dir=run_dir,
                    lane=lane,
                    learner=learner,
                    scheduler=scheduler,
                    scaler=scaler,
                    environment_cursor=environment_cursor,
                    sampler=sampler,
                    bindings=bindings,
                    progress=progress,
                    latest_update=latest_update,
                    minimum_cumulative_seconds=args.minimum_cumulative_seconds,
                    experiment_id=experiment_id,
                    last_episode=last_episode,
                )
            _preserve_reached_milestones(
                run_dir=run_dir,
                progress=progress,
                target_seconds=args.minimum_cumulative_seconds,
                fractions=milestone_fractions,
            )
            last_checkpoint_wall = time.perf_counter()
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "lane": lane.lane,
                        "checkpoint_sha256": checkpoint_sha256,
                        "progress": progress.to_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_progress = _progress(
        learner=learner,
        base_seconds=base_seconds,
        session_started=session_started,
        counters=counters,
    )
    with phase_timer.phase("checkpoint_write") if phase_timer is not None else nullcontext():
        final_checkpoint_sha256 = _save_training_state(
            run_dir=run_dir,
            lane=lane,
            learner=learner,
            scheduler=scheduler,
            scaler=scaler,
            environment_cursor=environment_cursor,
            sampler=sampler,
            bindings=bindings,
            progress=final_progress,
            latest_update=latest_update,
            minimum_cumulative_seconds=args.minimum_cumulative_seconds,
            experiment_id=experiment_id,
            last_episode=last_episode,
        )
    preserved_milestones = _preserve_reached_milestones(
        run_dir=run_dir,
        progress=final_progress,
        target_seconds=args.minimum_cumulative_seconds,
        fractions=milestone_fractions,
    )
    resume_proof = None
    if restored is not None and (
        final_progress.environment_decisions > restored.progress.environment_decisions
    ):
        resume_proof = prove_resume_counter_advance(
            restored=restored,
            current=final_progress,
        )
        _atomic_write_json(run_dir / "resume_proof.json", resume_proof)
    floor_reached = bool(
        args.minimum_cumulative_seconds > 0
        and final_progress.cumulative_training_seconds >= args.minimum_cumulative_seconds
    )
    bounded_stop = bool(
        args.max_decisions
        and final_progress.environment_decisions >= args.max_decisions
        and not floor_reached
    )
    profile_outputs = None
    if phase_timer is not None:
        for key, value in counters.items():
            phase_timer.set_counter(key, value)
        phase_timer.set_counter("optimizer_updates", learner.state.update_count)
        phase_timer.set_counter("learner_decisions", learner.state.decision_count)
        profile_outputs = phase_timer.write_outputs()
    summary = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": (
            "minimum_training_floor_reached"
            if floor_reached
            else "bounded_smoke_complete"
            if bounded_stop
            else "stopped_before_floor"
        ),
        "lane": lane.lane,
        "experiment_id": experiment_id,
        "progress": final_progress.to_dict(),
        "minimum_cumulative_seconds": args.minimum_cumulative_seconds,
        "checkpoint_sha256": final_checkpoint_sha256,
        "actor_checkpoint_sha256": _sha256_file(run_dir / "actor_latest.pt"),
        "resume_proof": resume_proof,
        "safety": {name: counters[name] for name in SAFETY_KEYS},
        "opponent_fallback_monitor": "not_instrumented",
        "protected_surfaces_mutated": False,
        "profile_outputs": profile_outputs,
        "newly_preserved_milestones": preserved_milestones,
    }
    _atomic_write_json(run_dir / "summary.json", summary)

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    try:
        summary = run(args)
    except SafetyHardStop as exc:
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "hard_stop_safety_failure",
            "lane": args.lane,
            "category": exc.category,
            "detail": exc.detail,
            "context": exc.context,
            "retriable": False,
            "protected_surfaces_mutated": False,
        }
        _atomic_write_json(run_dir / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)

        return SAFETY_HARD_STOP_EXIT_CODE
    except (
        TrainingHardStop,
        R4FPPOValidationError,
        LeagueCheckpointError,
        ValueError,
        FileNotFoundError,
        ImportError,
    ) as exc:
        category = getattr(exc, "category", type(exc).__name__)
        failure = {
            "schema_version": "deepmew.terminal_league_training_failure.v1",
            "status": "hard_stop_training_invariant",
            "lane": args.lane,
            "category": str(category),
            "detail": str(exc),
            "retriable": False,
            "protected_surfaces_mutated": False,
        }
        _atomic_write_json(run_dir / "training_failure.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)

        return TRAINING_HARD_STOP_EXIT_CODE
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
