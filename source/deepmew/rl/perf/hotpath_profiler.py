"""DeepMew RL hot-path profiler.

The profiler is intentionally external to the training loop. It samples the
same env/encoder/mask/policy path used by RL0, records coarse phase timings, and
writes a reusable report bundle under `.research/rl_perf/<run_id>/`.
"""

from __future__ import annotations

import math
import random
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepmew.rl.algorithms.masked_ppo import PPOPolicy, build_model_from_checkpoint
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.env_adapter import EnvAdapter, StepResult
from deepmew.rl.metadata import build_run_metadata, checkpoint_metadata
from deepmew.rl.perf.phase_timer import PhaseTimer
from deepmew.rl.registry import build_encoder, build_masker, build_opponent_sampler, build_reward, resolve_experiment
from deepmew.rl.rewards import RewardContext
from deepmew.rl.trainer import MaskedRandomPolicy


@dataclass(frozen=True)
class HotpathProfileConfig:
    experiment: str = "RL0_pure_masked_ppo"
    profile_mode: str = "rollout_single_worker"
    decisions: int = 2048
    warmup_decisions: int = 128
    seed: int = 0
    out_dir: Path = Path(".research/rl_perf/hotpath_profile_v0")
    run_id: str | None = None
    checkpoint: Path | None = None
    device: str = "cpu"
    max_steps_per_game: int | None = None
    backend: str = "masked_ppo"
    advantage_backend: str = "torch"
    measure_allocations: bool = True


class _ProfileEnv:
    def __init__(
        self,
        *,
        config: RLExperimentConfig,
        seed: int,
        max_steps_per_game: int | None,
        phase_timer: PhaseTimer | None = None,
    ):
        self.config = config
        self.seed = seed
        self.max_steps_per_game = max_steps_per_game or config.max_steps_per_game
        self.sampler = build_opponent_sampler(config, seed=seed)
        self.rng = random.Random(seed + 1701)
        self.episode_index = 0
        self.env: EnvAdapter | None = None
        self.opponent_label = "unknown"
        self.our_seat = 0
        self.phase_timer = phase_timer

    def reset(self) -> StepResult:
        self.close()
        sampled = self.sampler.sample()
        self.opponent_label = sampled.agent.label
        self.our_seat = self.episode_index % 2
        self.episode_index += 1
        self.env = EnvAdapter(
            opponent=sampled.agent,
            our_seat=self.our_seat,
            max_steps=self.max_steps_per_game,
            opponent_seed=self.rng.randint(0, 2**31 - 1),
            phase_timer=self.phase_timer,
        )
        return self.env.reset()

    @property
    def select(self) -> dict[str, Any] | None:
        if self.env is None:
            return None
        return self.env.select

    def step(self, selection: list[int]) -> StepResult:
        if self.env is None:
            return StepResult(obs=None, done=True, result=None, crashed=True, info={"error": "step before reset"})
        return self.env.step(selection)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None


def _json_ready_config(config: HotpathProfileConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["out_dir"] = str(config.out_dir)
    payload["checkpoint"] = str(config.checkpoint) if config.checkpoint is not None else None
    return payload


def _is_finite_vector(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _record_record_bytes(timer: PhaseTimer, features: list[float], mask_dim: int, selection: list[int], options: list[Any]) -> None:
    obs_bytes = len(features) * 4
    mask_bytes = mask_dim
    action_bytes = len(selection) * 4
    option_bytes = len(options) * 32
    timer.increment("obs_bytes_total", obs_bytes)
    timer.increment("mask_bytes_total", mask_bytes)
    timer.increment("action_bytes_total", action_bytes)
    timer.increment("action_table_bytes_est_total", option_bytes)
    timer.observe("obs_bytes_per_decision", obs_bytes)
    timer.observe("mask_bytes_per_decision", mask_bytes)
    timer.observe("action_bytes_per_decision", action_bytes)
    timer.observe("action_table_bytes_est_per_decision", option_bytes)


def _build_policy(profile: HotpathProfileConfig):
    if profile.checkpoint is None:
        return MaskedRandomPolicy(rng=random.Random(profile.seed + 7)), None, None
    model, algo_config, payload = build_model_from_checkpoint(profile.checkpoint, device=profile.device)
    return PPOPolicy(model, algo_config, deterministic=True), algo_config, payload


def _metadata(
    *,
    profile: HotpathProfileConfig,
    config: RLExperimentConfig,
    schema_names: tuple[str, ...],
    max_options: int,
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_algo_config: Any,
) -> dict[str, Any]:
    from_checkpoint = checkpoint_metadata(checkpoint_payload) if checkpoint_payload is not None else None
    if from_checkpoint is not None:
        return from_checkpoint
    advantage_backend = (
        checkpoint_algo_config.advantage_backend
        if checkpoint_algo_config is not None
        else profile.advantage_backend
    )
    puffer_status = "not_attempted"
    if advantage_backend == "puffer_cpu":
        puffer_status = "not_vectorized_puffer_cpu_advantage_only"
    return build_run_metadata(
        config=config,
        schema_names=schema_names,
        max_options=max_options,
        trainer_backend="masked_ppo_torch" if profile.backend != "puffer_vectorized" else "puffer_vectorized",
        advantage_backend=advantage_backend,
        vector_backend="single_process_env",
        puffer_vectorized_status=puffer_status,
    )


def _select_action(
    *,
    timer: PhaseTimer | None,
    policy: Any,
    features: list[float],
    mask: Any,
    select: dict[str, Any],
    masker: Any,
) -> list[int]:
    direct_select = getattr(policy, "select", None)
    if callable(direct_select):
        if timer is None:
            return masker.sanitize(direct_select(features, mask, select), select)
        with timer.phase("policy_forward"):
            selection = direct_select(features, mask, select)
        with timer.phase("action_sample"):
            return masker.sanitize(selection, select)
    if timer is None:
        priority = policy.act(features, mask)
        return masker.select_from_priority(priority, select)
    with timer.phase("policy_forward"):
        priority = policy.act(features, mask)
    with timer.phase("action_sample"):
        return masker.select_from_priority(priority, select)


def _run_rollout_decisions(
    *,
    profile: HotpathProfileConfig,
    timer: PhaseTimer | None,
    config: RLExperimentConfig,
    policy: Any,
    decisions: int,
) -> None:
    encoder = build_encoder(config)
    masker = build_masker(config)
    reward_fn = build_reward(config)
    env = _ProfileEnv(
        config=config,
        seed=profile.seed,
        max_steps_per_game=profile.max_steps_per_game,
        phase_timer=timer,
    )
    step: StepResult
    if timer is None:
        step = env.reset()
    else:
        step = env.reset()
    prev_obs = step.obs
    episode_decisions = 0

    try:
        measured = 0
        while measured < decisions:
            if step.done:
                if step.crashed and timer is not None:
                    timer.increment("crashes")
                if step.timed_out and timer is not None:
                    timer.increment("timeouts")
                if timer is None:
                    step = env.reset()
                else:
                    step = env.reset()
                prev_obs = step.obs
                episode_decisions = 0
                continue

            obs = step.obs or {}
            select = env.select or {}
            if timer is None:
                options = list(select.get("option") or [])
                features = encoder.encode(obs)
                mask = masker.build_mask(select)
                selection = _select_action(timer=None, policy=policy, features=features, mask=mask, select=select, masker=masker)
                next_step = env.step(selection)
                illegal = 0 if masker.is_legal(selection, select) else 1
                _ = reward_fn(
                    RewardContext(
                        prev_obs=prev_obs,
                        next_obs=next_step.obs if next_step.obs is not None else obs,
                        our_seat=env.our_seat,
                        done=next_step.done,
                        result=next_step.result,
                        crashed=next_step.crashed,
                        timed_out=next_step.timed_out,
                        illegal_attempts=illegal,
                    )
                )
                prev_obs = obs
                step = next_step
                measured += 1
                episode_decisions += 1
                continue

            with timer.phase("complete_action_table_build"):
                options = list(select.get("option") or [])
            timer.observe("option_count", len(options))
            with timer.phase("legal_action_enumeration"):
                _legal_indices = list(range(min(len(options), masker.max_options)))
            with timer.phase("action_mask_build"):
                mask = masker.build_mask(select)
            if mask.max_count <= 0:
                timer.increment("zero_select_decisions")
            elif mask.max_count == 1:
                timer.increment("single_select_decisions")
            else:
                timer.increment("multiselect_decisions")
            with timer.phase("observation_encode"):
                features = encoder.encode(obs)
            if not _is_finite_vector(features):
                timer.increment("nan_inf_count")

            selection = _select_action(
                timer=timer,
                policy=policy,
                features=features,
                mask=mask,
                select=select,
                masker=masker,
            )
            illegal = 0 if masker.is_legal(selection, select) else 1
            if illegal:
                timer.increment("illegal_attempts")
            next_step = env.step(selection)
            with timer.phase("reward_done_extract"):
                _ = reward_fn(
                    RewardContext(
                        prev_obs=prev_obs,
                        next_obs=next_step.obs if next_step.obs is not None else obs,
                        our_seat=env.our_seat,
                        done=next_step.done,
                        result=next_step.result,
                        crashed=next_step.crashed,
                        timed_out=next_step.timed_out,
                        illegal_attempts=illegal,
                    )
                )
            with timer.phase("rollout_record_pack"):
                _record_record_bytes(timer, features, mask.dim, selection, options)
            if next_step.crashed:
                timer.increment("crashes")
            if next_step.timed_out:
                timer.increment("timeouts")
            if next_step.done:
                if next_step.result == env.our_seat:
                    timer.increment("wins")
                elif next_step.result == 2:
                    timer.increment("draws")
                elif next_step.result is not None:
                    timer.increment("losses")
            timer.increment("decisions")
            measured += 1
            episode_decisions += 1
            timer.set_counter("last_episode_decisions", episode_decisions)
            prev_obs = obs
            step = next_step
    finally:
        env.close()


def run_hotpath_profile(profile: HotpathProfileConfig) -> dict[str, str]:
    """Run the requested profile mode and write profiler outputs."""
    config = resolve_experiment(profile.experiment).validate()
    if profile.max_steps_per_game is not None:
        from dataclasses import replace

        config = replace(config, max_steps_per_game=profile.max_steps_per_game)
    encoder = build_encoder(config)
    masker = build_masker(config)
    policy, algo_config, checkpoint_payload = _build_policy(profile)
    run_id = profile.run_id or f"{profile.profile_mode}_{int(time.time())}"
    out_dir = Path(profile.out_dir)
    metadata = _metadata(
        profile=profile,
        config=config,
        schema_names=encoder.schema.names,
        max_options=masker.max_options,
        checkpoint_payload=checkpoint_payload,
        checkpoint_algo_config=algo_config,
    )
    timer = PhaseTimer(
        run_id=run_id,
        out_dir=out_dir,
        config={
            **_json_ready_config(profile),
            "experiment_id": config.experiment_id,
            "observation_dim": encoder.schema.dim,
            "max_options": masker.max_options,
            "run_metadata": metadata,
        },
    )
    timer.set_counter("advantage_backend", metadata["advantage_backend"])
    timer.set_counter("trainer_backend", metadata["trainer_backend"])
    timer.set_counter("vector_backend", metadata["vector_backend"])
    timer.set_counter("puffer_vectorized_status", metadata["puffer_vectorized_status"])
    timer.set_counter("engine_step_replaced_by_subphases", True)
    timer.set_counter(
        "engine_rules_effect_resolution_status",
        "not_independently_measurable_without_engine_instrumentation",
    )

    if profile.profile_mode not in {"rollout_single_worker", "full_train_short_profile"}:
        raise ValueError(
            "supported profile modes for this harness are rollout_single_worker and "
            "full_train_short_profile (rollout-only learner phases remain zero)"
        )

    if profile.warmup_decisions:
        _run_rollout_decisions(
            profile=profile,
            timer=None,
            config=config,
            policy=policy,
            decisions=profile.warmup_decisions,
        )
    if profile.measure_allocations:
        tracemalloc.start()
    _run_rollout_decisions(
        profile=profile,
        timer=timer,
        config=config,
        policy=policy,
        decisions=profile.decisions,
    )
    if profile.measure_allocations:
        with timer.phase("allocation_snapshot"):
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("filename")
            top_stats = [
                {"file": str(stat.traceback[0].filename), "size_bytes": stat.size, "count": stat.count}
                for stat in stats[:10]
            ]
            timer.set_counter("tracemalloc_current_bytes", current_bytes)
            timer.set_counter("tracemalloc_peak_bytes", peak_bytes)
            timer.set_counter("allocation_stat_count", len(stats))
            timer.set_counter("allocation_top_files", top_stats)
            tracemalloc.stop()
    with timer.phase("serialization_probe"):
        payload = {
            "run_id": run_id,
            "config": _json_ready_config(profile),
            "counters": timer.counters,
            "event_count": len(timer.events),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        timer.set_counter("serialization_probe_bytes", len(encoded))
    return timer.write_outputs()
