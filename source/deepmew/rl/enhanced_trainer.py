"""Research-only staged trainer for the enhanced terminal-league generation.

The plain terminal-PPO control runner intentionally does not import this module.
This trainer keeps one r7-initialized actor, one critic, and one quarantine-grade
Q head alive across V0 distillation, replay warm start, branch-Q supervision,
and terminal-reward PPO.  Every stage fails closed when its declared signal did
not update the parameters it is meant to supervise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from deepmew.q_guided.training import QDistributionHead, masked_gaussian_q_loss
from deepmew.rl.algorithms.r4f_ppo import (
    FrozenR4FState,
    R4FNativePPO,
    R4FPPOConfig,
    R4FPPOState,
    R4FRolloutStep,
    R4FTerminalEpisode,
)


CHECKPOINT_SCHEMA_VERSION = "deepmew.enhanced_league_trainer_checkpoint.v1"
CHECKPOINT_RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_league_trainer_checkpoint_receipt.v1"
STAGE_CHECKPOINT_RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_stage_checkpoint_receipt.v1"
STAGE_RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_league_stage_receipt.v1"
STAGE_ORDER = (
    "v0_distillation",
    "replay_warm_start",
    "branch_q",
    "curriculum_terminal_ppo",
)
STAGE_BOUNDARY_ORDER = (
    "v0_initialization_or_distillation",
    "replay_warm_start",
    "branch_q_supervision",
    "curriculum_activation",
    "terminal_ppo",
)
_STAGE_BOUNDARY_RECEIPT_COUNTS = (1, 2, 3, 3, 4)
ZERO_SAFETY = {"illegal_actions": 0, "crashes": 0, "timeouts": 0}


class EnhancedTrainerHardStop(RuntimeError):
    """A mandatory enhanced-training signal was absent, unsafe, or non-finite."""


@dataclass(frozen=True)
class ReplayTrainerExample:
    row_id: str
    option_features: torch.Tensor
    selection: tuple[int, ...]
    min_count: int
    max_count: int
    v0_probability: float
    legal_action_count: int | None = None

    def validate(self, *, input_dim: int) -> None:
        if not isinstance(self.row_id, str) or not self.row_id:
            raise EnhancedTrainerHardStop("replay row_id must be non-empty")
        if self.option_features.ndim != 2 or self.option_features.shape[1] != input_dim:
            raise EnhancedTrainerHardStop("replay option feature shape does not match r7")
        if self.option_features.shape[0] <= 0 or not bool(torch.isfinite(self.option_features).all().item()):
            raise EnhancedTrainerHardStop("replay option features are empty or non-finite")
        count = int(self.option_features.shape[0])
        if not 0 <= self.min_count <= self.max_count <= count:
            raise EnhancedTrainerHardStop("replay selection bounds are invalid")
        if len(self.selection) != len(set(self.selection)):
            raise EnhancedTrainerHardStop("replay chosen selection contains duplicates")
        if any(index < 0 or index >= count for index in self.selection):
            raise EnhancedTrainerHardStop("replay chosen selection is outside the legal set")
        if not self.min_count <= len(self.selection) <= self.max_count:
            raise EnhancedTrainerHardStop("replay chosen selection violates count bounds")
        if not math.isfinite(self.v0_probability) or not 0.0 <= self.v0_probability <= 1.0:
            raise EnhancedTrainerHardStop("replay V0 target must be finite in [0, 1]")
        legal_action_count = self.legal_action_count
        if legal_action_count is None:
            legal_action_count = count
        if isinstance(legal_action_count, bool) or legal_action_count < 1:
            raise EnhancedTrainerHardStop("replay legal_action_count must be positive")

    @property
    def is_forced_decision(self) -> bool:
        legal_action_count = self.legal_action_count
        if legal_action_count is None:
            legal_action_count = int(self.option_features.shape[0])

        return legal_action_count == 1


@dataclass(frozen=True)
class BranchTrainerBatch:
    row_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    state_features: torch.Tensor
    action_features: torch.Tensor
    target_mean: torch.Tensor
    target_variance: torch.Tensor
    action_mask: torch.Tensor
    policy_examples: tuple[Any, ...]
    is_training: bool

    def validate(self, *, state_dim: int, action_dim: int) -> None:
        batch_size = len(self.row_ids)
        if batch_size <= 0 or len(self.shard_ids) != batch_size:
            raise EnhancedTrainerHardStop("branch batch identities are empty or misaligned")
        if len(set(self.row_ids)) != batch_size:
            raise EnhancedTrainerHardStop("one branch batch contains duplicate row identities")
        if any(not isinstance(value, str) or not value for value in (*self.row_ids, *self.shard_ids)):
            raise EnhancedTrainerHardStop("branch row and shard identities must be non-empty")
        if tuple(self.state_features.shape) != (batch_size, state_dim):
            raise EnhancedTrainerHardStop("branch state tensor shape mismatch")
        if (
            self.action_features.ndim != 3
            or self.action_features.shape[0] != batch_size
            or self.action_features.shape[2] != action_dim
        ):
            raise EnhancedTrainerHardStop("branch action tensor shape mismatch")
        expected = tuple(self.action_features.shape[:2])
        if tuple(self.target_mean.shape) != expected or tuple(self.target_variance.shape) != expected:
            raise EnhancedTrainerHardStop("branch target tensor shape mismatch")
        if self.action_mask.dtype is not torch.bool or tuple(self.action_mask.shape) != expected:
            raise EnhancedTrainerHardStop("branch action mask shape or dtype mismatch")
        if not bool(self.action_mask.any(dim=1).all().item()):
            raise EnhancedTrainerHardStop("every branch row must expose a sibling action")
        tensors = (
            self.state_features,
            self.action_features,
            self.target_mean,
            self.target_variance,
        )
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise EnhancedTrainerHardStop("branch tensors contain NaN or Inf")
        if bool(((self.target_mean < 0.0) | (self.target_mean > 1.0)).any().item()):
            raise EnhancedTrainerHardStop("branch Q means must be in [0, 1]")
        if bool((self.target_variance < 0.0).any().item()):
            raise EnhancedTrainerHardStop("branch Q variances must be non-negative")
        if self.is_training and len(self.policy_examples) != batch_size:
            raise EnhancedTrainerHardStop("training branch rows require aligned policy examples")
        if not self.is_training and self.policy_examples:
            raise EnhancedTrainerHardStop("held-out branch rows must not carry gradient policy examples")


@dataclass(frozen=True)
class StageReceipt:
    schema_version: str
    trainer_lineage_id: str
    stage: str
    input_component_hashes: Mapping[str, str]
    output_component_hashes: Mapping[str, str]
    actor_parameter_delta_norm: float
    critic_parameter_delta_norm: float
    q_parameter_delta_norm: float
    optimizer_steps: int
    consumed: Mapping[str, Any]
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))

    return digest.hexdigest()


def _flat_parameters(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    if not values:
        raise EnhancedTrainerHardStop("enhanced trainer module has no parameters")

    return torch.cat(values)


def _parameter_delta(before: torch.Tensor, module: nn.Module) -> float:
    after = _flat_parameters(module)
    if before.shape != after.shape:
        raise EnhancedTrainerHardStop("parameter shape drifted during enhanced stage")
    value = float(torch.linalg.vector_norm(after - before).item())
    if not math.isfinite(value):
        raise EnhancedTrainerHardStop("enhanced stage produced a non-finite parameter delta")

    return value


def _pairwise_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    material_epsilon: float,
) -> torch.Tensor:
    predicted_delta = predicted[:, :, None] - predicted[:, None, :]
    target_delta = target[:, :, None] - target[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :] & (target_delta.abs() > material_epsilon)
    if not bool(pair_mask.any().item()):
        return predicted.sum() * 0.0
    signs = torch.sign(target_delta[pair_mask])

    return torch.nn.functional.softplus(-signs * predicted_delta[pair_mask]).mean()


GuidanceLoss = Callable[[nn.Module, Sequence[Any], torch.device], torch.Tensor]


class EnhancedLeagueTrainer:
    """One fail-closed model lineage spanning every enhanced training stage."""

    def __init__(
        self,
        ppo: R4FNativePPO,
        q_head: QDistributionHead,
        *,
        source_checkpoint: Path,
        lineage_id: str | None = None,
    ) -> None:
        self.ppo = ppo
        self.q_head = q_head.to(torch.device(ppo.config.device))
        self.source_checkpoint = Path(source_checkpoint).resolve()
        if _sha256_file(self.source_checkpoint) != ppo.proof.source_checkpoint_sha256:
            raise EnhancedTrainerHardStop("enhanced trainer source checkpoint hash drifted")
        self.anchor_actor = copy.deepcopy(ppo.model.actor).to(torch.device(ppo.config.device)).eval()
        for parameter in self.anchor_actor.parameters():
            parameter.requires_grad_(False)
        self.ppo.optimizer.add_param_group({"params": list(self.q_head.parameters())})
        self.lineage_id = lineage_id or uuid.uuid4().hex
        self.stage_receipts: list[StageReceipt] = []
        self.consumption_state: dict[str, Any] = {}
        self._replay_session: dict[str, Any] | None = None
        self._terminal_session: dict[str, Any] | None = None

    @classmethod
    def from_r7_checkpoint(
        cls,
        checkpoint: Path | str,
        frozen_states: Sequence[FrozenR4FState],
        *,
        q_state_dim: int,
        q_action_dim: int,
        ppo_config: R4FPPOConfig | None = None,
        lineage_id: str | None = None,
    ) -> "EnhancedLeagueTrainer":
        path = Path(checkpoint).resolve(strict=True)
        ppo = R4FNativePPO.from_r4f_checkpoint(path, frozen_states, ppo_config)
        q_head = QDistributionHead(q_state_dim, q_action_dim)

        return cls(ppo, q_head, source_checkpoint=path, lineage_id=lineage_id)

    @property
    def device(self) -> torch.device:
        return torch.device(self.ppo.config.device)

    def component_hashes(self) -> dict[str, str]:
        return {
            "actor": _module_hash(self.ppo.model.actor),
            "critic": _module_hash(self.ppo.model.critic),
            "q_head": _module_hash(self.q_head),
        }

    def _require_next_stage(self, stage: str) -> None:
        index = len(self.stage_receipts)
        if index >= len(STAGE_ORDER) or STAGE_ORDER[index] != stage:
            expected = STAGE_ORDER[index] if index < len(STAGE_ORDER) else "none"
            raise EnhancedTrainerHardStop(f"enhanced stage order mismatch: expected {expected}, got {stage}")

    def _step_loss(self, loss: torch.Tensor) -> float:
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise EnhancedTrainerHardStop("enhanced optimizer loss is non-finite or non-scalar")
        self.ppo.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.ppo.model.parameters()) + list(self.q_head.parameters())
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise EnhancedTrainerHardStop("enhanced optimizer gradient contains NaN or Inf")
        gradient_norm = nn.utils.clip_grad_norm_(parameters, self.ppo.config.max_grad_norm)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise EnhancedTrainerHardStop("enhanced optimizer gradient norm is non-finite")
        self.ppo.optimizer.step()
        modules = (self.ppo.model.actor, self.ppo.model.critic, self.q_head)
        if any(any(not bool(torch.isfinite(value).all().item()) for value in module.state_dict().values()) for module in modules):
            raise EnhancedTrainerHardStop("enhanced optimizer produced non-finite parameters")

        return float(gradient_norm.detach().cpu())

    def _record_stage(
        self,
        *,
        stage: str,
        input_hashes: Mapping[str, str],
        before_actor: torch.Tensor,
        before_critic: torch.Tensor,
        before_q: torch.Tensor,
        optimizer_steps: int,
        consumed: Mapping[str, Any],
        metrics: Mapping[str, float],
        require_actor_update: bool,
        require_critic_update: bool,
        require_q_update: bool,
    ) -> StageReceipt:
        actor_delta = _parameter_delta(before_actor, self.ppo.model.actor)
        critic_delta = _parameter_delta(before_critic, self.ppo.model.critic)
        q_delta = _parameter_delta(before_q, self.q_head)
        required = (
            (require_actor_update, actor_delta, "actor"),
            (require_critic_update, critic_delta, "critic"),
            (require_q_update, q_delta, "Q head"),
        )
        for is_required, delta, name in required:
            if is_required and delta <= 0.0:
                raise EnhancedTrainerHardStop(f"{stage} did not update the required {name} parameters")
        output_hashes = self.component_hashes()
        receipt = StageReceipt(
            schema_version=STAGE_RECEIPT_SCHEMA_VERSION,
            trainer_lineage_id=self.lineage_id,
            stage=stage,
            input_component_hashes=dict(input_hashes),
            output_component_hashes=output_hashes,
            actor_parameter_delta_norm=actor_delta,
            critic_parameter_delta_norm=critic_delta,
            q_parameter_delta_norm=q_delta,
            optimizer_steps=optimizer_steps,
            consumed=dict(consumed),
            metrics={key: float(value) for key, value in metrics.items()},
        )
        if self.stage_receipts and self.stage_receipts[-1].output_component_hashes != receipt.input_component_hashes:
            raise EnhancedTrainerHardStop("enhanced stage input hash does not match the prior stage output")
        self.stage_receipts.append(receipt)

        return receipt

    def distill_v0(self, examples: Sequence[ReplayTrainerExample]) -> StageReceipt:
        stage = "v0_distillation"
        self._require_next_stage(stage)
        if not examples:
            raise EnhancedTrainerHardStop("V0 distillation requires replay examples")
        input_dim = self.ppo.model.actor.config.input_dim
        row_ids = [example.row_id for example in examples]
        if len(set(row_ids)) != len(row_ids):
            raise EnhancedTrainerHardStop("V0 distillation rows are not unique")
        for example in examples:
            example.validate(input_dim=input_dim)
        input_hashes = self.component_hashes()
        before_actor = _flat_parameters(self.ppo.model.actor)
        before_critic = _flat_parameters(self.ppo.model.critic)
        before_q = _flat_parameters(self.q_head)
        losses = []
        for example in examples:
            features = example.option_features.to(self.device, dtype=torch.float32)
            with torch.no_grad():
                embeddings = self.ppo.model.encode_decision(features).detach()
            predicted = self.ppo.model.value_from_embeddings(embeddings)
            target = torch.tensor(2.0 * example.v0_probability - 1.0, device=self.device)
            losses.append((predicted - target).square())
        loss = torch.stack(losses).mean()
        gradient_norm = self._step_loss(loss)
        self.consumption_state[stage] = {"row_ids": tuple(sorted(row_ids))}

        return self._record_stage(
            stage=stage,
            input_hashes=input_hashes,
            before_actor=before_actor,
            before_critic=before_critic,
            before_q=before_q,
            optimizer_steps=1,
            consumed={"v0_weights_or_distillation_consumed": True, "unique_rows": len(row_ids)},
            metrics={"distillation_mse": float(loss.detach().cpu()), "gradient_norm": gradient_norm},
            require_actor_update=False,
            require_critic_update=True,
            require_q_update=False,
        )

    def replay_warm_start(
        self,
        examples: Sequence[ReplayTrainerExample],
        *,
        value_weight: float = 0.5,
        r7_retention_weight: float = 1.0e-5,
    ) -> StageReceipt:
        stage = "replay_warm_start"
        self._require_next_stage(stage)
        if not examples:
            raise EnhancedTrainerHardStop("replay warm start requires at least one row")
        if value_weight <= 0.0 or r7_retention_weight <= 0.0:
            raise EnhancedTrainerHardStop("replay warm-start weights must be positive")
        input_dim = self.ppo.model.actor.config.input_dim
        row_ids = [example.row_id for example in examples]
        if len(set(row_ids)) != len(row_ids):
            raise EnhancedTrainerHardStop("replay warm-start rows are not unique")
        for example in examples:
            example.validate(input_dim=input_dim)
        input_hashes = self.component_hashes()
        before_actor = _flat_parameters(self.ppo.model.actor)
        before_critic = _flat_parameters(self.ppo.model.critic)
        before_q = _flat_parameters(self.q_head)
        action_losses = []
        value_losses = []
        forced_rows = 0
        for example in examples:
            step = R4FRolloutStep(
                option_features=example.option_features,
                min_count=example.min_count,
                max_count=example.max_count,
                selection=example.selection,
                old_log_prob=0.0,
                old_value=0.0,
            )
            log_probability, _entropy, value = self.ppo.distribution.evaluate(step)
            if example.is_forced_decision:
                forced_rows += 1
            else:
                action_losses.append(-log_probability)
            target = torch.tensor(2.0 * example.v0_probability - 1.0, device=self.device)
            value_losses.append((value - target).square())
        retention = torch.zeros((), device=self.device)
        for current, anchor in zip(
            self.ppo.model.actor.parameters(),
            self.anchor_actor.parameters(),
            strict=True,
        ):
            retention = retention + (current - anchor).square().sum()
        action_loss = (
            torch.stack(action_losses).sum() / len(action_losses)
            if action_losses
            else torch.zeros((), device=self.device)
        )
        value_loss = torch.stack(value_losses).mean()
        total_loss = action_loss + value_weight * value_loss + r7_retention_weight * retention
        gradient_norm = self._step_loss(total_loss)
        self.consumption_state[stage] = {"row_ids": tuple(sorted(row_ids))}

        return self._record_stage(
            stage=stage,
            input_hashes=input_hashes,
            before_actor=before_actor,
            before_critic=before_critic,
            before_q=before_q,
            optimizer_steps=1,
            consumed={
                "unique_replay_rows_consumed": len(row_ids),
                "forced_rows": forced_rows,
                "nonforced_rows": len(action_losses),
                "actor_weight_sum": float(len(action_losses)),
                "all_forced_actor_skipped": int(not action_losses),
                "forced_actor_loss_weight": 0.0,
                "nonforced_actor_loss_weight": 1.0,
            },
            metrics={
                "action_nll": float(action_loss.detach().cpu()),
                "value_mse": float(value_loss.detach().cpu()),
                "r7_parameter_retention": float(retention.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": gradient_norm,
            },
            require_actor_update=True,
            require_critic_update=True,
            require_q_update=False,
        )

    def begin_replay_warm_start(self) -> None:
        """Begin the complete streaming replay stage after V0."""
        self._require_next_stage("replay_warm_start")
        if self._replay_session is not None:
            raise EnhancedTrainerHardStop("streaming replay session is already active")
        self._replay_session = {
            "input_hashes": self.component_hashes(),
            "before_actor": _flat_parameters(self.ppo.model.actor),
            "before_critic": _flat_parameters(self.ppo.model.critic),
            "before_q": _flat_parameters(self.q_head),
            "row_count": 0,
            "forced_rows": 0,
            "nonforced_rows": 0,
            "all_forced_actor_skipped": 0,
            "optimizer_steps": 0,
            "action_loss_weighted_sum": 0.0,
            "value_loss_weighted_sum": 0.0,
            "first_row_id": None,
            "final_row_id": None,
        }

    def replay_warm_start_batch(
        self,
        examples: Sequence[ReplayTrainerExample],
        *,
        value_weight: float = 0.5,
        r7_retention_weight: float = 1.0e-5,
    ) -> Mapping[str, Any]:
        """Consume one deterministic bounded batch, retaining forced rows for V."""
        session = self._replay_session
        if session is None:
            raise EnhancedTrainerHardStop("streaming replay session is not active")
        if not examples:
            raise EnhancedTrainerHardStop("streaming replay batch is empty")
        if value_weight <= 0.0 or r7_retention_weight <= 0.0:
            raise EnhancedTrainerHardStop("replay warm-start weights must be positive")
        input_dim = self.ppo.model.actor.config.input_dim
        row_ids = [example.row_id for example in examples]
        if len(set(row_ids)) != len(row_ids):
            raise EnhancedTrainerHardStop("streaming replay batch row identities are duplicated")
        for example in examples:
            example.validate(input_dim=input_dim)
        action_losses = []
        value_losses = []
        for example in examples:
            step = R4FRolloutStep(
                option_features=example.option_features,
                min_count=example.min_count,
                max_count=example.max_count,
                selection=example.selection,
                old_log_prob=0.0,
                old_value=0.0,
            )
            log_probability, _entropy, value = self.ppo.distribution.evaluate(step)
            if not example.is_forced_decision:
                action_losses.append(-log_probability)
            target = torch.tensor(2.0 * example.v0_probability - 1.0, device=self.device)
            value_losses.append((value - target).square())
        action_loss = (
            torch.stack(action_losses).sum() / len(action_losses)
            if action_losses
            else torch.zeros((), device=self.device)
        )
        value_loss = torch.stack(value_losses).mean()
        retention = torch.zeros((), device=self.device)
        for current, anchor in zip(self.ppo.model.actor.parameters(), self.anchor_actor.parameters(), strict=True):
            retention = retention + (current - anchor).square().sum()
        total_loss = action_loss + value_weight * value_loss + r7_retention_weight * retention
        gradient_norm = self._step_loss(total_loss)
        forced_rows = len(examples) - len(action_losses)
        session["row_count"] += len(examples)
        session["forced_rows"] += forced_rows
        session["nonforced_rows"] += len(action_losses)
        session["all_forced_actor_skipped"] += int(not action_losses)
        session["optimizer_steps"] += 1
        session["action_loss_weighted_sum"] += float(action_loss.detach().cpu()) * len(action_losses)
        session["value_loss_weighted_sum"] += float(value_loss.detach().cpu()) * len(examples)
        session["first_row_id"] = session["first_row_id"] or row_ids[0]
        session["final_row_id"] = row_ids[-1]

        return {
            "rows": len(examples),
            "forced_rows": forced_rows,
            "nonforced_rows": len(action_losses),
            "actor_weight_sum": float(len(action_losses)),
            "all_forced_actor_skipped": int(not action_losses),
            "actor_loss": float(action_loss.detach().cpu()),
            "critic_value_loss": float(value_loss.detach().cpu()),
            "gradient_norm": gradient_norm,
        }

    def finish_replay_warm_start(self, *, expected_unique_rows: int) -> StageReceipt:
        session = self._replay_session
        if session is None:
            raise EnhancedTrainerHardStop("streaming replay session is not active")
        if session["row_count"] != expected_unique_rows:
            raise EnhancedTrainerHardStop(
                f"complete replay consumption mismatch: {session['row_count']}/{expected_unique_rows}"
            )
        nonforced = session["nonforced_rows"]
        rows = session["row_count"]
        receipt = self._record_stage(
            stage="replay_warm_start",
            input_hashes=session["input_hashes"],
            before_actor=session["before_actor"],
            before_critic=session["before_critic"],
            before_q=session["before_q"],
            optimizer_steps=session["optimizer_steps"],
            consumed={
                "unique_replay_rows_consumed": rows,
                "forced_rows": session["forced_rows"],
                "nonforced_rows": nonforced,
                "actor_weight_sum": float(nonforced),
                "all_forced_actor_skipped": session["all_forced_actor_skipped"],
                "forced_actor_loss_weight": 0.0,
                "nonforced_actor_loss_weight": 1.0,
                "first_row_id": session["first_row_id"],
                "final_row_id": session["final_row_id"],
            },
            metrics={
                "action_nll": session["action_loss_weighted_sum"] / nonforced if nonforced else 0.0,
                "value_mse": session["value_loss_weighted_sum"] / rows,
            },
            require_actor_update=True,
            require_critic_update=True,
            require_q_update=False,
        )
        self._replay_session = None

        return receipt

    def branch_q_stage(
        self,
        batches: Iterable[BranchTrainerBatch],
        *,
        expected_unique_rows: int,
        expected_unique_shards: int,
        guidance_loss: GuidanceLoss,
        q_weight: float = 1.0,
        ranking_weight: float = 0.25,
        guidance_weight: float = 0.25,
        material_epsilon: float = 0.02,
    ) -> StageReceipt:
        stage = "branch_q"
        self._require_next_stage(stage)
        if expected_unique_rows <= 0 or expected_unique_shards <= 0:
            raise EnhancedTrainerHardStop("branch expectations must be positive")
        if min(q_weight, ranking_weight, guidance_weight) <= 0.0:
            raise EnhancedTrainerHardStop("branch loss weights must be positive")
        input_hashes = self.component_hashes()
        before_actor = _flat_parameters(self.ppo.model.actor)
        before_critic = _flat_parameters(self.ppo.model.critic)
        before_q = _flat_parameters(self.q_head)
        seen_rows: set[str] = set()
        seen_shards: set[str] = set()
        train_rows = 0
        heldout_rows = 0
        optimizer_steps = 0
        metric_totals = {"q_regression_loss": 0.0, "ranking_loss": 0.0, "guidance_loss": 0.0}
        evaluated_batches = 0
        for batch in batches:
            batch.validate(state_dim=self.q_head.state_dim, action_dim=self.q_head.action_dim)
            overlap = seen_rows.intersection(batch.row_ids)
            if overlap:
                raise EnhancedTrainerHardStop(f"branch rows were consumed twice: {sorted(overlap)[:3]}")
            seen_rows.update(batch.row_ids)
            seen_shards.update(batch.shard_ids)
            state = batch.state_features.to(self.device)
            actions = batch.action_features.to(self.device)
            target_mean = batch.target_mean.to(self.device)
            target_variance = batch.target_variance.to(self.device)
            mask = batch.action_mask.to(self.device)
            if batch.is_training:
                distribution = self.q_head(state, actions, mask)
                q_loss = masked_gaussian_q_loss(distribution, target_mean, target_variance, mask)
                ranking_loss = _pairwise_ranking_loss(
                    distribution.mean,
                    target_mean,
                    mask,
                    material_epsilon=material_epsilon,
                )
                actor_guidance = guidance_loss(self.ppo.model.actor, batch.policy_examples, self.device)
                if actor_guidance.ndim != 0 or not bool(torch.isfinite(actor_guidance).item()):
                    raise EnhancedTrainerHardStop("branch actor-guidance loss is non-finite or non-scalar")
                total = q_weight * q_loss + ranking_weight * ranking_loss + guidance_weight * actor_guidance
                self._step_loss(total)
                optimizer_steps += 1
                train_rows += len(batch.row_ids)
            else:
                with torch.no_grad():
                    distribution = self.q_head(state, actions, mask)
                    q_loss = masked_gaussian_q_loss(distribution, target_mean, target_variance, mask)
                    ranking_loss = _pairwise_ranking_loss(
                        distribution.mean,
                        target_mean,
                        mask,
                        material_epsilon=material_epsilon,
                    )
                actor_guidance = torch.zeros((), device=self.device)
                heldout_rows += len(batch.row_ids)
            metric_totals["q_regression_loss"] += float(q_loss.detach().cpu())
            metric_totals["ranking_loss"] += float(ranking_loss.detach().cpu())
            metric_totals["guidance_loss"] += float(actor_guidance.detach().cpu())
            evaluated_batches += 1
        if len(seen_rows) != expected_unique_rows or len(seen_shards) != expected_unique_shards:
            raise EnhancedTrainerHardStop(
                "branch corpus consumption mismatch: "
                f"rows={len(seen_rows)}/{expected_unique_rows}, shards={len(seen_shards)}/{expected_unique_shards}"
            )
        if optimizer_steps <= 0 or train_rows <= 0 or heldout_rows <= 0:
            raise EnhancedTrainerHardStop("branch stage requires loss-bearing train rows and explicit held-out evaluation")
        metrics = {key: value / evaluated_batches for key, value in metric_totals.items()}
        self.consumption_state[stage] = {
            "row_ids": tuple(sorted(seen_rows)),
            "shard_ids": tuple(sorted(seen_shards)),
            "train_rows": train_rows,
            "heldout_rows": heldout_rows,
        }

        return self._record_stage(
            stage=stage,
            input_hashes=input_hashes,
            before_actor=before_actor,
            before_critic=before_critic,
            before_q=before_q,
            optimizer_steps=optimizer_steps,
            consumed={
                "unique_branch_rows_consumed": len(seen_rows),
                "unique_branch_shards_consumed": len(seen_shards),
                "train_rows": train_rows,
                "heldout_rows": heldout_rows,
            },
            metrics=metrics,
            require_actor_update=True,
            require_critic_update=False,
            require_q_update=True,
        )

    def terminal_ppo_stage(
        self,
        episodes: Sequence[R4FTerminalEpisode],
        *,
        curriculum_receipt: Mapping[str, Any],
        safety: Mapping[str, int],
    ) -> StageReceipt:
        stage = "curriculum_terminal_ppo"
        self._require_next_stage(stage)
        if not episodes:
            raise EnhancedTrainerHardStop("terminal PPO smoke requires a completed real episode")
        if dict(safety) != ZERO_SAFETY:
            raise EnhancedTrainerHardStop(f"terminal PPO safety hard stop: {dict(safety)}")
        required_curriculum = {
            "opponent_deck_strength_metadata_consumed_by_scheduler",
            "sampled_policy_id",
            "sampled_deck_hash",
            "sampled_strength_bucket",
        }
        missing = required_curriculum - set(curriculum_receipt)
        if missing or curriculum_receipt.get("opponent_deck_strength_metadata_consumed_by_scheduler") is not True:
            raise EnhancedTrainerHardStop(f"curriculum receipt is missing mandatory fields: {sorted(missing)}")
        input_hashes = self.component_hashes()
        before_actor = _flat_parameters(self.ppo.model.actor)
        before_critic = _flat_parameters(self.ppo.model.critic)
        before_q = _flat_parameters(self.q_head)
        stats = self.ppo.update(episodes)
        self.consumption_state[stage] = {
            "episodes": len(episodes),
            "curriculum_receipt": dict(curriculum_receipt),
            "safety": dict(safety),
        }

        return self._record_stage(
            stage=stage,
            input_hashes=input_hashes,
            before_actor=before_actor,
            before_critic=before_critic,
            before_q=before_q,
            optimizer_steps=1,
            consumed={
                "terminal_reward_league_ppo_consumed": True,
                "episodes": len(episodes),
                "decisions": stats.decisions,
                "opponent_deck_strength_metadata_consumed_by_scheduler": True,
            },
            metrics={
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
                "approximate_kl": stats.approximate_kl,
                "gradient_norm": stats.gradient_norm,
            },
            require_actor_update=True,
            require_critic_update=True,
            require_q_update=False,
        )

    @staticmethod
    def _validate_curriculum_sample(receipt: Mapping[str, Any]) -> None:
        required = {
            "opponent_deck_strength_metadata_consumed_by_scheduler",
            "sampled_policy_id",
            "sampled_deck_hash",
            "sampled_strength_bucket",
            "sampled_weight",
        }
        missing = required - set(receipt)
        if (
            missing
            or receipt.get("opponent_deck_strength_metadata_consumed_by_scheduler") is not True
        ):
            raise EnhancedTrainerHardStop(
                f"terminal curriculum sample is missing mandatory fields: {sorted(missing)}"
            )
        weight = receipt.get("sampled_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise EnhancedTrainerHardStop("terminal curriculum sample weight is invalid")
        if not math.isfinite(float(weight)) or float(weight) <= 0.0:
            raise EnhancedTrainerHardStop("terminal curriculum sample weight must be positive")
        for field in ("sampled_policy_id", "sampled_deck_hash", "sampled_strength_bucket"):
            if not isinstance(receipt.get(field), str) or not receipt[field]:
                raise EnhancedTrainerHardStop(f"terminal curriculum sample {field} is invalid")

    def begin_terminal_ppo(self, *, curriculum_identity: Mapping[str, Any]) -> None:
        """Begin the resumable terminal-PPO stage after curriculum activation."""

        self._require_next_stage("curriculum_terminal_ppo")
        if self._terminal_session is not None:
            raise EnhancedTrainerHardStop("terminal PPO session is already active")
        if not isinstance(curriculum_identity, Mapping) or not curriculum_identity:
            raise EnhancedTrainerHardStop("terminal PPO requires a frozen curriculum identity")
        self._terminal_session = {
            "input_hashes": self.component_hashes(),
            "before_actor": _flat_parameters(self.ppo.model.actor),
            "before_critic": _flat_parameters(self.ppo.model.critic),
            "before_q": _flat_parameters(self.q_head),
            "curriculum_identity": copy.deepcopy(dict(curriculum_identity)),
            "optimizer_steps": 0,
            "episodes": 0,
            "decisions": 0,
            "cumulative_seconds": 0.0,
            "metric_sums": {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approximate_kl": 0.0,
                "gradient_norm": 0.0,
            },
            "curriculum_sample_counts": {},
            "safety": copy.deepcopy(ZERO_SAFETY),
        }

    def terminal_ppo_update(
        self,
        episodes: Sequence[R4FTerminalEpisode],
        *,
        curriculum_samples: Sequence[Mapping[str, Any]],
        safety: Mapping[str, int],
        cumulative_seconds: float,
    ) -> Mapping[str, float]:
        """Commit one complete rollout update inside the terminal-PPO stage."""

        session = self._terminal_session
        if session is None:
            raise EnhancedTrainerHardStop("terminal PPO session is not active")
        if not episodes or len(episodes) != len(curriculum_samples):
            raise EnhancedTrainerHardStop(
                "terminal PPO episodes and curriculum samples must be non-empty and aligned"
            )
        if dict(safety) != ZERO_SAFETY:
            raise EnhancedTrainerHardStop(f"terminal PPO safety hard stop: {dict(safety)}")
        if not math.isfinite(cumulative_seconds) or cumulative_seconds < session["cumulative_seconds"]:
            raise EnhancedTrainerHardStop("terminal PPO cumulative time regressed or is non-finite")
        for receipt in curriculum_samples:
            self._validate_curriculum_sample(receipt)
        stats = self.ppo.update(episodes)
        metrics = {
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approximate_kl": stats.approximate_kl,
            "gradient_norm": stats.gradient_norm,
        }
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            raise EnhancedTrainerHardStop("terminal PPO update produced non-finite metrics")
        session["optimizer_steps"] += 1
        session["episodes"] += len(episodes)
        session["decisions"] += sum(len(episode.steps) for episode in episodes)
        session["cumulative_seconds"] = float(cumulative_seconds)
        session["safety"] = copy.deepcopy(dict(safety))
        for name, value in metrics.items():
            session["metric_sums"][name] += float(value)
        for receipt in curriculum_samples:
            key = "|".join((
                str(receipt["sampled_policy_id"]),
                str(receipt["sampled_deck_hash"]),
                str(receipt["sampled_strength_bucket"]),
            ))
            counts = session["curriculum_sample_counts"]
            counts[key] = int(counts.get(key, 0)) + 1

        return metrics

    def update_terminal_elapsed(self, cumulative_seconds: float) -> None:
        """Record elapsed trainer time at an interruption-safe update boundary."""

        session = self._terminal_session
        if session is None:
            raise EnhancedTrainerHardStop("terminal PPO session is not active")
        if not math.isfinite(cumulative_seconds) or cumulative_seconds < session["cumulative_seconds"]:
            raise EnhancedTrainerHardStop("terminal PPO cumulative time regressed or is non-finite")
        session["cumulative_seconds"] = float(cumulative_seconds)

    def finish_terminal_ppo(
        self,
        *,
        minimum_cumulative_seconds: float,
        safety: Mapping[str, int],
    ) -> StageReceipt:
        session = self._terminal_session
        if session is None:
            raise EnhancedTrainerHardStop("terminal PPO session is not active")
        if dict(safety) != ZERO_SAFETY or session["safety"] != ZERO_SAFETY:
            raise EnhancedTrainerHardStop(f"terminal PPO safety hard stop: {dict(safety)}")
        if (
            not math.isfinite(minimum_cumulative_seconds)
            or minimum_cumulative_seconds <= 0.0
            or session["cumulative_seconds"] < minimum_cumulative_seconds
        ):
            raise EnhancedTrainerHardStop("terminal PPO stopped before its cumulative time floor")
        updates = int(session["optimizer_steps"])
        if updates <= 0 or session["episodes"] <= 0 or session["decisions"] <= 0:
            raise EnhancedTrainerHardStop("terminal PPO completed without loss-bearing updates")
        receipt = self._record_stage(
            stage="curriculum_terminal_ppo",
            input_hashes=session["input_hashes"],
            before_actor=session["before_actor"],
            before_critic=session["before_critic"],
            before_q=session["before_q"],
            optimizer_steps=updates,
            consumed={
                "terminal_reward_league_ppo_consumed": True,
                "episodes": session["episodes"],
                "decisions": session["decisions"],
                "cumulative_training_seconds": session["cumulative_seconds"],
                "minimum_cumulative_seconds": minimum_cumulative_seconds,
                "opponent_deck_strength_metadata_consumed_by_scheduler": True,
                "curriculum_identity": session["curriculum_identity"],
                "curriculum_sample_counts": dict(sorted(session["curriculum_sample_counts"].items())),
                "safety": copy.deepcopy(dict(safety)),
            },
            metrics={
                name: total / updates
                for name, total in session["metric_sums"].items()
            },
            require_actor_update=True,
            require_critic_update=True,
            require_q_update=False,
        )
        self.consumption_state["curriculum_terminal_ppo"] = {
            "episodes": session["episodes"],
            "decisions": session["decisions"],
            "cumulative_seconds": session["cumulative_seconds"],
            "curriculum_identity": session["curriculum_identity"],
            "curriculum_sample_counts": dict(sorted(session["curriculum_sample_counts"].items())),
            "safety": copy.deepcopy(dict(safety)),
        }
        self._terminal_session = None

        return receipt

    def save_checkpoint(self, path: Path | str, *, extra_state: Mapping[str, Any]) -> str:
        if len(self.stage_receipts) != len(STAGE_ORDER):
            raise EnhancedTrainerHardStop("enhanced checkpoint requires every mandatory training stage")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "trainer_lineage_id": self.lineage_id,
            "source_checkpoint": str(self.source_checkpoint),
            "source_checkpoint_sha256": self.ppo.proof.source_checkpoint_sha256,
            "actor_config": asdict(self.ppo.model.actor.config),
            "ppo_config": asdict(self.ppo.config),
            "initialization_proof": asdict(self.ppo.proof),
            "actor_state": self.ppo.model.actor.state_dict(),
            "critic_state": self.ppo.model.critic.state_dict(),
            "q_head_config": {
                "state_dim": self.q_head.state_dim,
                "action_dim": self.q_head.action_dim,
                "state_hidden_dims": list(self.q_head.state_hidden_dims),
            },
            "q_head_state": self.q_head.state_dict(),
            "optimizer_state": self.ppo.optimizer.state_dict(),
            "ppo_state": asdict(self.ppo.state),
            "stage_receipts": [receipt.to_dict() for receipt in self.stage_receipts],
            "consumption_state": copy.deepcopy(self.consumption_state),
            "terminal_session": copy.deepcopy(self._terminal_session),
            "extra_state": copy.deepcopy(dict(extra_state)),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "component_hashes": self.component_hashes(),
        }
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        checkpoint_sha256 = _sha256_file(target)
        receipt = {
            "schema_version": CHECKPOINT_RECEIPT_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "trainer_lineage_id": self.lineage_id,
            "component_hashes": self.component_hashes(),
        }
        receipt_path = target.with_suffix(target.suffix + ".receipt.json")
        temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.{uuid.uuid4().hex}.tmp")
        temporary_receipt.write_bytes(_canonical_json_bytes(receipt) + b"\n")
        os.replace(temporary_receipt, receipt_path)

        return checkpoint_sha256

    def save_stage_checkpoint(
        self,
        path: Path | str,
        *,
        boundary_stage: str,
        frozen_metadata: Mapping[str, Any],
        extra_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Freeze one immutable stage boundary without requiring later stages."""
        is_replay_resume = boundary_stage == "replay_warm_start_in_progress"
        is_terminal_resume = boundary_stage == "terminal_ppo_in_progress"
        if (
            boundary_stage not in STAGE_BOUNDARY_ORDER
            and not is_replay_resume
            and not is_terminal_resume
        ):
            raise EnhancedTrainerHardStop("unknown enhanced stage boundary")
        boundary_index = (
            1
            if is_replay_resume
            else 4
            if is_terminal_resume
            else STAGE_BOUNDARY_ORDER.index(boundary_stage)
        )
        expected_receipts = (
            1
            if is_replay_resume
            else 3
            if is_terminal_resume
            else _STAGE_BOUNDARY_RECEIPT_COUNTS[boundary_index]
        )
        if len(self.stage_receipts) != expected_receipts:
            raise EnhancedTrainerHardStop("stage boundary does not match completed trainer stages")
        target = Path(path)
        receipt_path = target.with_suffix(target.suffix + ".receipt.json")
        actor_path = target.with_suffix(target.suffix + ".actor.pt")
        if target.exists() or receipt_path.exists() or actor_path.exists():
            raise EnhancedTrainerHardStop("immutable stage checkpoint target already exists")
        required_metadata = {
            "configuration_sha256",
            "action_schema_identity",
            "observation_schema_identity",
            "deck_identities",
            "input_hashes",
            "stage_counters",
            "data_cursors",
            "curriculum_state",
            "elapsed_seconds",
            "code_identity",
            "next_operation",
            "continuous_stage_chain_sha256",
        }
        if not required_metadata.issubset(frozen_metadata):
            missing = sorted(required_metadata - set(frozen_metadata))
            raise EnhancedTrainerHardStop(f"stage checkpoint metadata is incomplete: {missing}")
        metadata = copy.deepcopy(dict(frozen_metadata))
        metadata_sha256 = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "boundary_stage": boundary_stage,
            "boundary_index": boundary_index,
            "frozen_metadata": metadata,
            "frozen_metadata_sha256": metadata_sha256,
            "trainer_lineage_id": self.lineage_id,
            "source_checkpoint": str(self.source_checkpoint),
            "source_checkpoint_sha256": self.ppo.proof.source_checkpoint_sha256,
            "actor_config": asdict(self.ppo.model.actor.config),
            "ppo_config": asdict(self.ppo.config),
            "initialization_proof": asdict(self.ppo.proof),
            "actor_state": self.ppo.model.actor.state_dict(),
            "critic_state": self.ppo.model.critic.state_dict(),
            "q_head_config": {
                "state_dim": self.q_head.state_dim,
                "action_dim": self.q_head.action_dim,
                "state_hidden_dims": list(self.q_head.state_hidden_dims),
            },
            "q_head_state": self.q_head.state_dict(),
            "optimizer_state": self.ppo.optimizer.state_dict(),
            "ppo_state": asdict(self.ppo.state),
            "stage_receipts": [receipt.to_dict() for receipt in self.stage_receipts],
            "consumption_state": copy.deepcopy(self.consumption_state),
            "replay_session": copy.deepcopy(self._replay_session),
            "terminal_session": copy.deepcopy(self._terminal_session),
            "extra_state": copy.deepcopy(dict(extra_state)),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "component_hashes": self.component_hashes(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.ppo.export_actor(
            actor_path,
            training_metadata={
                "kind": "enhanced_stage_boundary",
                "boundary_stage": boundary_stage,
                "trainer_lineage_id": self.lineage_id,
                "frozen_metadata_sha256": metadata_sha256,
            },
        )
        checkpoint_sha256 = _sha256_file(target)
        actor_sha256 = _sha256_file(actor_path)
        receipt = {
            "schema_version": STAGE_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "actor_export_sha256": actor_sha256,
            "actor_state_sha256": self.component_hashes()["actor"],
            "boundary_stage": boundary_stage,
            "boundary_index": boundary_index,
            "trainer_lineage_id": self.lineage_id,
            "component_hashes": self.component_hashes(),
            "frozen_metadata_sha256": metadata_sha256,
            "continuous_stage_chain_sha256": metadata["continuous_stage_chain_sha256"],
            "next_operation": metadata["next_operation"],
        }
        temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.{uuid.uuid4().hex}.tmp")
        temporary_receipt.write_bytes(_canonical_json_bytes(receipt) + b"\n")
        os.replace(temporary_receipt, receipt_path)

        return receipt

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        frozen_states: Sequence[FrozenR4FState],
        *,
        map_location: str | torch.device,
    ) -> tuple["EnhancedLeagueTrainer", dict[str, Any]]:
        checkpoint_path = Path(path).resolve(strict=True)
        receipt_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        expected_receipt_fields = {
            "schema_version",
            "checkpoint_sha256",
            "trainer_lineage_id",
            "component_hashes",
        }
        if set(receipt) != expected_receipt_fields or receipt["schema_version"] != CHECKPOINT_RECEIPT_SCHEMA_VERSION:
            raise EnhancedTrainerHardStop("enhanced checkpoint receipt schema mismatch")
        if _sha256_file(checkpoint_path) != receipt["checkpoint_sha256"]:
            raise EnhancedTrainerHardStop("enhanced checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise EnhancedTrainerHardStop("enhanced checkpoint schema mismatch")
        ppo_config = R4FPPOConfig(**payload["ppo_config"])
        ppo_config = R4FPPOConfig(**{**asdict(ppo_config), "device": str(map_location)})
        q_config = payload["q_head_config"]
        trainer = cls.from_r7_checkpoint(
            payload["source_checkpoint"],
            frozen_states,
            q_state_dim=int(q_config["state_dim"]),
            q_action_dim=int(q_config["action_dim"]),
            ppo_config=ppo_config,
            lineage_id=payload["trainer_lineage_id"],
        )
        if trainer.ppo.proof.source_checkpoint_sha256 != payload["source_checkpoint_sha256"]:
            raise EnhancedTrainerHardStop("enhanced checkpoint source identity mismatch")
        trainer.ppo.model.actor.load_state_dict(payload["actor_state"], strict=True)
        trainer.ppo.model.critic.load_state_dict(payload["critic_state"], strict=True)
        trainer.q_head.load_state_dict(payload["q_head_state"], strict=True)
        trainer.ppo.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.ppo.state = R4FPPOState(**payload["ppo_state"])
        trainer.stage_receipts = [StageReceipt(**value) for value in payload["stage_receipts"]]
        trainer.consumption_state = copy.deepcopy(payload["consumption_state"])
        trainer._replay_session = copy.deepcopy(payload.get("replay_session"))
        trainer._terminal_session = copy.deepcopy(payload.get("terminal_session"))
        if trainer.component_hashes() != payload["component_hashes"] or payload["component_hashes"] != receipt["component_hashes"]:
            raise EnhancedTrainerHardStop("enhanced checkpoint component hash mismatch")
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        cuda_states = payload["torch_cuda_rng_states"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise EnhancedTrainerHardStop("enhanced checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])

        return trainer, copy.deepcopy(payload["extra_state"])

    @classmethod
    def load_stage_checkpoint(
        cls,
        path: Path | str,
        frozen_states: Sequence[FrozenR4FState],
        *,
        map_location: str | torch.device,
        expected_frozen_metadata: Mapping[str, Any],
    ) -> tuple["EnhancedLeagueTrainer", dict[str, Any], dict[str, Any]]:
        checkpoint_path = Path(path).resolve(strict=True)
        receipt_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        if receipt.get("schema_version") != STAGE_CHECKPOINT_RECEIPT_SCHEMA_VERSION:
            raise EnhancedTrainerHardStop("stage checkpoint receipt schema mismatch")
        if _sha256_file(checkpoint_path) != receipt.get("checkpoint_sha256"):
            raise EnhancedTrainerHardStop("stage checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        expected_metadata_sha256 = hashlib.sha256(
            _canonical_json_bytes(dict(expected_frozen_metadata))
        ).hexdigest()
        if payload.get("frozen_metadata_sha256") != expected_metadata_sha256:
            raise EnhancedTrainerHardStop("stage checkpoint data or configuration binding drifted")
        boundary_stage = payload.get("boundary_stage")
        if boundary_stage != receipt.get("boundary_stage"):
            raise EnhancedTrainerHardStop("stage checkpoint boundary identity mismatch")
        ppo_config = R4FPPOConfig(**payload["ppo_config"])
        ppo_config = R4FPPOConfig(**{**asdict(ppo_config), "device": str(map_location)})
        q_config = payload["q_head_config"]
        trainer = cls.from_r7_checkpoint(
            payload["source_checkpoint"],
            frozen_states,
            q_state_dim=int(q_config["state_dim"]),
            q_action_dim=int(q_config["action_dim"]),
            ppo_config=ppo_config,
            lineage_id=payload["trainer_lineage_id"],
        )
        trainer.ppo.model.actor.load_state_dict(payload["actor_state"], strict=True)
        trainer.ppo.model.critic.load_state_dict(payload["critic_state"], strict=True)
        trainer.q_head.load_state_dict(payload["q_head_state"], strict=True)
        trainer.ppo.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.ppo.state = R4FPPOState(**payload["ppo_state"])
        trainer.stage_receipts = [StageReceipt(**value) for value in payload["stage_receipts"]]
        trainer.consumption_state = copy.deepcopy(payload["consumption_state"])
        trainer._replay_session = copy.deepcopy(payload.get("replay_session"))
        trainer._terminal_session = copy.deepcopy(payload.get("terminal_session"))
        if trainer.component_hashes() != payload["component_hashes"]:
            raise EnhancedTrainerHardStop("stage checkpoint component hash mismatch")
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        cuda_states = payload["torch_cuda_rng_states"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise EnhancedTrainerHardStop(
                    "stage checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])

        return trainer, copy.deepcopy(payload["extra_state"]), copy.deepcopy(payload["frozen_metadata"])


__all__ = [
    "BranchTrainerBatch",
    "CHECKPOINT_SCHEMA_VERSION",
    "EnhancedLeagueTrainer",
    "EnhancedTrainerHardStop",
    "ReplayTrainerExample",
    "STAGE_ORDER",
    "STAGE_BOUNDARY_ORDER",
    "StageReceipt",
    "ZERO_SAFETY",
]
