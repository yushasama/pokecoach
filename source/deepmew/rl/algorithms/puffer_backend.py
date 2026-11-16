"""Puffer-related DeepMew backends.

`masked_ppo_puffer_advantage` is Torch masked PPO with PufferLib's CPU
advantage helper. It is not full Puffer training.

`puffer_vectorized` is reserved for real Puffer/vectorized env training. On the
current Windows checkout it blocks until DeepMew has a native/Puffer-compatible
env binding for the masked multiselect action space.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from deepmew.rl.algorithms.masked_ppo import MaskedPPOConfig, SampledBattleEnv, train_masked_ppo
from deepmew.rl.algorithms.puffer_support import PufferSupportError, require_puffer_cpu_advantage
from deepmew.rl.metadata import build_run_metadata
from deepmew.rl.puffer_env import check_puffer_compatibility
from deepmew.rl.trainer import RLComponents, RLRunResult


class PufferAdvantageBackend:
    name = "masked_ppo_puffer_advantage"

    def __init__(self, components: RLComponents, *, seed: int = 0, **kwargs: Any):
        self.components = components
        self.seed = seed
        self.device = str(kwargs.get("device", "cpu"))
        self.checkpoint_path = kwargs.get("checkpoint_path")
        self.resume = bool(kwargs.get("resume", False))
        self.rollout_steps = int(kwargs.get("rollout_steps", 256))
        self.num_envs = max(1, int(kwargs.get("num_envs", 1)))
        self.env_vectorization = str(kwargs.get("env_vectorization", "local"))
        self.max_steps_per_game = kwargs.get("max_steps_per_game") or components.config.max_steps_per_game
        self.requested_backend = str(kwargs.get("requested_backend", self.name))
        self.compatibility = check_puffer_compatibility()

    def _empty_result(self, *, dry_run: bool) -> RLRunResult:
        result = RLRunResult(
            experiment_id=self.components.config.experiment_id,
            backend=self.name,
            dry_run=dry_run,
            observation_dim=self.components.encoder.schema.dim,
        )
        result.notes.append(f"puffer_compatibility={self.compatibility.to_dict()}")
        if self.requested_backend not in {self.name, "puffer_advantage_cpu"}:
            result.notes.append(
                f"compatibility alias: requested_backend={self.requested_backend!r}; "
                f"canonical_backend={self.name!r}; this is not puffer_vectorized"
            )
        result.training_metrics.update(
            {
                "trainer_backend": "masked_ppo_torch",
                "advantage_backend": "puffer_cpu",
                "vector_backend": self._vector_backend_label(),
                "puffer_vectorized_status": "not_vectorized_puffer_cpu_advantage_only",
                "advantage_helper_active": True,
                "vectorized_env_active": self.env_vectorization == "subprocess",
                "puffer_vectorized_active": False,
                "num_envs_supported": "1 local; N subprocess-isolated via Torch rollout loop",
                "bottleneck": "Python battle env stepping; PufferLib only computes CPU GAE",
                "puffer_integration": "puffer_cpu_advantage_helper_only",
                "puffer_backend_status": self.compatibility.adapter_status,
            }
        )
        return result

    def _blocked(self, message: str, *, dry_run: bool) -> RLRunResult:
        result = self._empty_result(dry_run=dry_run)
        result.notes.append(f"BLOCKER: {message}")
        result.training_metrics["puffer_integration"] = "blocked"
        return result

    def _env_fns(self) -> list[Any]:
        if self.env_vectorization == "subprocess":
            from deepmew.rl.vector_env import SubprocessEnvProxy, sampled_battle_env_spec

            return [
                lambda offset=offset: SubprocessEnvProxy(
                    offset,
                    sampled_battle_env_spec(
                        experiment=self.components.config.experiment_id,
                        seed=self.seed,
                        max_steps_per_game=self.max_steps_per_game,
                        env_id=offset,
                    ),
                )
                for offset in range(self.num_envs)
            ]
        return [
            lambda offset=offset: SampledBattleEnv(
                self.components,
                seed=self.seed + 11 + offset,
                max_steps_per_game=self.max_steps_per_game,
            )
            for offset in range(self.num_envs)
        ]

    def _vector_backend_label(self) -> str:
        if self.env_vectorization == "subprocess":
            return "subprocess_vector_env"
        return "single_process_env"

    def _run_metadata(self) -> dict[str, Any]:
        return build_run_metadata(
            config=self.components.config,
            schema_names=self.components.encoder.schema.names,
            max_options=self.components.masker.max_options,
            trainer_backend="masked_ppo_torch",
            advantage_backend="puffer_cpu",
            vector_backend=self._vector_backend_label(),
            puffer_vectorized_status="not_vectorized_puffer_cpu_advantage_only",
        )

    def run(self, *, total_steps: int, dry_run: bool) -> RLRunResult:
        result = self._empty_result(dry_run=dry_run)
        if dry_run:
            result.notes.append("dry-run: Torch masked PPO + Puffer CPU advantage helper assembled; no rollout/update")
            return result
        if not self.compatibility.available or not self.compatibility.native_available:
            return self._blocked(str(self.compatibility.blocker), dry_run=dry_run)
        if self.device != "cpu":
            return self._blocked(
                "Puffer CPU advantage helper is CPU-only; full GPU/vectorized Puffer belongs to puffer_vectorized",
                dry_run=dry_run,
            )
        if self.num_envs != 1 and self.env_vectorization != "subprocess":
            return self._blocked(
                "same-process SampledBattleEnv num_envs>1 is blocked on Windows global battle-engine state; "
                "use --env-vectorization subprocess for isolated worker processes",
                dry_run=dry_run,
            )
        try:
            require_puffer_cpu_advantage()
        except PufferSupportError as exc:
            return self._blocked(str(exc), dry_run=dry_run)

        self.components.opponent_sampler.reset(self.seed)
        algo_config = MaskedPPOConfig(
            obs_dim=self.components.encoder.schema.dim,
            action_dim=self.components.masker.max_options,
            seed=self.seed,
            device="cpu",
            rollout_steps=self.rollout_steps,
            advantage_backend="puffer_cpu",
        )
        run_metadata = self._run_metadata()
        trained, model, optimizer, state = train_masked_ppo(
            env_fns=self._env_fns(),
            encoder=self.components.encoder,
            masker=self.components.masker,
            reward_fn=self.components.reward_fn,
            algo_config=algo_config,
            total_steps=total_steps,
            experiment_config=self.components.config,
            checkpoint_path=self.checkpoint_path,
            resume=self.resume,
            checkpoint_extra={
                "run_metadata": run_metadata,
                "backend_metadata": {
                    "trainer_backend": run_metadata["trainer_backend"],
                    "advantage_backend": run_metadata["advantage_backend"],
                    "vector_backend": run_metadata["vector_backend"],
                    "puffer_vectorized_status": run_metadata["puffer_vectorized_status"],
                },
            },
        )
        trained.backend = self.name
        trained.training_metrics.update(result.training_metrics)
        trained.training_metrics["env_vectorization"] = self.env_vectorization
        trained.training_metrics["num_envs_requested"] = self.num_envs
        trained.training_metrics.update(
            {
                "trainer_backend": run_metadata["trainer_backend"],
                "advantage_backend": run_metadata["advantage_backend"],
                "vector_backend": run_metadata["vector_backend"],
                "puffer_vectorized_status": run_metadata["puffer_vectorized_status"],
                "agent_deck_hash": run_metadata["agent_deck_hash"],
                "opponent_deck_hash": run_metadata["opponent_deck_hash"],
                "obs_schema_hash": run_metadata["obs_schema_hash"],
                "action_schema_hash": run_metadata["action_schema_hash"],
            }
        )
        trained.notes.insert(0, f"puffer_compatibility={self.compatibility.to_dict()}")
        trained.notes.insert(
            1,
            "puffer_integration=puffer_cpu_advantage_helper_only; "
            "this is Torch masked PPO plus Puffer CPU GAE, not puffer_vectorized",
        )
        if self.requested_backend not in {self.name, "puffer_advantage_cpu"}:
            trained.notes.insert(
                2,
                f"compatibility alias: requested_backend={self.requested_backend!r}; "
                f"canonical_backend={self.name!r}",
            )
        self.model = model
        self.optimizer = optimizer
        self.state = state
        return trained


class PufferVectorizedBackend:
    name = "puffer_vectorized"

    def __init__(self, components: RLComponents, *, seed: int = 0, **kwargs: Any):
        self.components = components
        self.seed = seed
        self.num_envs = max(1, int(kwargs.get("num_envs", 1)))
        self.compatibility = check_puffer_compatibility()

    def run(self, *, total_steps: int, dry_run: bool) -> RLRunResult:
        result = RLRunResult(
            experiment_id=self.components.config.experiment_id,
            backend=self.name,
            dry_run=dry_run,
            observation_dim=self.components.encoder.schema.dim,
        )
        vector_module = importlib.util.find_spec("pufferlib.vector")
        emulation_module = importlib.util.find_spec("pufferlib.emulation")
        result.training_metrics.update(
            {
                "trainer_backend": "blocked",
                "advantage_backend": "none",
                "vector_backend": "blocked_no_deepmew_native_vec_binding",
                "puffer_vectorized_status": "blocked_no_deepmew_native_vec_binding",
                "advantage_helper_active": False,
                "vectorized_env_active": False,
                "puffer_vectorized_active": False,
                "num_envs_requested": self.num_envs,
                "num_envs_supported": 0,
                "puffer_backend_status": self.compatibility.adapter_status,
                "puffer_integration": "blocked_no_deepmew_native_vec_binding",
                "pufferlib_vector_module": bool(vector_module),
                "pufferlib_emulation_module": bool(emulation_module),
                "bottleneck": "missing DeepMew native/Puffer-compatible vector env and masked multiselect adapter",
            }
        )
        result.notes.append(f"puffer_compatibility={self.compatibility.to_dict()}")
        result.notes.append(
            "BLOCKER: puffer_vectorized is reserved for real Puffer/vectorized training. "
            "The installed Windows PufferLib exposes _C.create_vec for compiled Ocean envs, "
            "but this checkout has no DeepMew native env binding and no pufferlib.vector/emulation "
            "Python wrapper module to host the autoregressive masked multiselect action space."
        )
        result.notes.append(
            "Use subprocess vector env + Torch PPO for local Windows scale experiments, "
            "or run the native binding work on WSL/Linux/Modal."
        )
        return result


# Compatibility alias for older imports. The canonical backend label is now
# `masked_ppo_puffer_advantage`.
PufferBackend = PufferAdvantageBackend
