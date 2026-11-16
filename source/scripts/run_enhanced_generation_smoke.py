"""Run the single fail-closed local smoke for the enhanced league trainer.

This runner is research-only and local-only.  It binds the accepted replay
corpus, validated V0 endpoint, frozen full branch-Q corpus, an isolated
identity/strength curriculum, and one terminal-reward real-game PPO update to
one trainer lineage.  It never calls Modal and never writes protected runtime
or submission surfaces.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import numpy as np
import torch

import run_q_guided_gpu_train as q_gpu
import run_terminal_league_ppo as terminal
import run_unified_r3_value as value_source
from deepmew.q_guided.training import ShardBinding
from deepmew.q_guided.teacher_contract import validate_safe_shard
from deepmew.replay.information_boundary import (
    InformationMode,
    assert_deployable_observation,
)
from deepmew.rl.algorithms.r4f_ppo import R4FPPOConfig, R4FRolloutStep
from deepmew.rl.deck_identity import (
    r4f_feature_deck_hash,
    read_deck_any,
    resolve_repo_path,
)
from deepmew.rl.enhanced_branch import (
    FULL_BRANCH_CORPUS_ROWS,
    FULL_BRANCH_CORPUS_SHARDS,
    BranchPolicyPayload,
    build_branch_trainer_batches,
    load_enhanced_branch_plan,
)
from deepmew.rl.enhanced_curriculum import (
    CurriculumCell,
    CurriculumCellKey,
    EnhancedCurriculumScheduler,
)
from deepmew.rl.enhanced_replay import (
    BoundedReplayDataset,
    ReplayRecord,
    load_bounded_replay_dataset,
)
from deepmew.rl.enhanced_replay_derived import DerivedReplayIndex
from deepmew.rl.enhanced_smoke_receipt import (
    REQUIRED_STAGE_NAMES,
    SMOKE_CLASSIFICATION,
    SMOKE_RECEIPT_SCHEMA_VERSION,
    STAGE_CHAIN_GENESIS_SHA256,
    build_stage_record,
    compute_stage_chain_sha256,
    validate_enhanced_smoke_receipt,
)
from deepmew.rl.enhanced_trainer import (
    EnhancedLeagueTrainer,
    ReplayTrainerExample,
    StageReceipt,
    ZERO_SAFETY,
)
from deepmew.rl.opponents import OPPONENT_SETS, OpponentAgent, SampledOpponent
from deepmew.rl.registry import resolve_experiment
from deepmew.value_lab.frozen_v0_endpoint import FrozenV0Endpoint, FrozenV0Spec


SCHEMA_VERSION = "deepmew.enhanced_generation_local_smoke_runner.v1"
FAILURE_SCHEMA_VERSION = "deepmew.enhanced_generation_local_smoke_failure.v1"
DEFAULT_SEED = 20_260_713
DEFAULT_REPLAY_ROWS = 32
DEFAULT_BRANCH_BATCH_SIZE = 64
DEFAULT_CURRICULUM_BINDINGS = (
    "starmie-r7-d1",
    "starmie-r23-d1",
)
DEFAULT_STRENGTH_MULTIPLIERS = (
    "local_frontier=1.25",
    "historical_intermediate=0.75",
)
DEFAULT_D2_DECK_IDENTITY = (
    "sha256:242086768949fcab0101e4975db53fc11629702e212537ef5505037c6d482aa3"
)
FORBIDDEN_PERSISTED_KEYS = frozenset({
    "search_begin_input",
    "private_state",
    "private_hash",
    "hidden_state",
    "opponent_hand",
    "opponent_deck",
    "opponent_prize",
})


class EnhancedSmokeHardStop(RuntimeError):
    """The local smoke could not prove every mandatory consumption signal."""


class CurriculumOpponentAdapter:
    """Expose an enhanced scheduler through the real-game sampler interface."""

    def __init__(
        self,
        scheduler: EnhancedCurriculumScheduler,
        agents_by_key: Mapping[CurriculumCellKey, tuple[int, OpponentAgent]],
    ) -> None:
        self.scheduler = scheduler
        self.agents_by_key = dict(agents_by_key)
        if set(self.agents_by_key) != {cell.key for cell in self.scheduler.cells}:
            raise EnhancedSmokeHardStop("curriculum agent identities do not match scheduler cells")
        for _, agent in self.agents_by_key.values():
            agent.materialize()
        self.last_sample = None

    def sample(self) -> SampledOpponent:
        sampled = self.scheduler.sample()
        binding = self.agents_by_key.get(sampled.cell.key)
        if binding is None:
            raise EnhancedSmokeHardStop("scheduler selected an unbound policy/deck/strength cell")
        index, agent = binding
        pair_identity = agent.pair_identity
        expected = sampled.cell.key.to_dict()
        observed = {
            "policy_identity": pair_identity.get("binding_id"),
            "deck_hash": str(pair_identity.get("deck_hash") or "").removeprefix("sha256:"),
            "strength_bucket": pair_identity.get("strength_bucket"),
        }
        if observed != expected:
            raise EnhancedSmokeHardStop("sampled opponent identity drifted after scheduling")
        self.last_sample = sampled

        return SampledOpponent(index=index, agent=agent, weight=sampled.normalized_weight)


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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(dict(payload)) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(path)


def _evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value["evidence_sha256"] = _sha256_json(value)

    return value


def _component_model_sha256(component_hashes: Mapping[str, str]) -> str:
    expected = {"actor", "critic", "q_head"}
    if set(component_hashes) != expected:
        raise EnhancedSmokeHardStop("trainer component hashes are incomplete")

    return _sha256_json(dict(sorted(component_hashes.items())))


def _flat_parameters(module: torch.nn.Module) -> torch.Tensor:
    values = [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    if not values:
        raise EnhancedSmokeHardStop("trainer component has no parameters")

    return torch.cat(values)


def _parameter_delta(before: torch.Tensor, module: torch.nn.Module) -> float:
    after = _flat_parameters(module)
    if after.shape != before.shape:
        raise EnhancedSmokeHardStop("trainer parameter shape changed during the smoke")
    delta = float(torch.linalg.vector_norm(after - before).item())
    if not math.isfinite(delta) or delta <= 0.0:
        raise EnhancedSmokeHardStop("required trainer parameter delta was not positive")

    return delta


def _digest_update(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))

        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))

        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _digest_update(digest, str(key))
            _digest_update(digest, value[key])

        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _digest_update(digest, item)

        return
    digest.update(b"scalar")
    digest.update(_canonical_json_bytes(value))


def _digest_value(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_update(digest, value)

    return digest.hexdigest()


def _trainer_state_sha256(
    trainer: EnhancedLeagueTrainer,
    extra_state: Mapping[str, Any],
) -> str:
    payload = {
        "component_hashes": trainer.component_hashes(),
        "optimizer_state": trainer.ppo.optimizer.state_dict(),
        "ppo_state": asdict(trainer.ppo.state),
        "stage_receipts": [receipt.to_dict() for receipt in trainer.stage_receipts],
        "consumption_state": trainer.consumption_state,
        "extra_state": dict(extra_state),
    }

    return _digest_value(payload)


def _parse_strength_multipliers(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw in values:
        bucket, separator, raw_multiplier = raw.partition("=")
        if not separator or not bucket or bucket != bucket.strip():
            raise EnhancedSmokeHardStop("strength multipliers require bucket=positive_float")
        try:
            multiplier = float(raw_multiplier)
        except ValueError as exc:
            raise EnhancedSmokeHardStop("strength multiplier is not numeric") from exc
        if not math.isfinite(multiplier) or multiplier <= 0.0 or bucket in result:
            raise EnhancedSmokeHardStop("strength multipliers must be unique, finite, and positive")
        result[bucket] = multiplier

    return result


def _load_v0(args: argparse.Namespace) -> FrozenV0Endpoint:
    calibrator_root = Path(args.v0_calibrator_root).resolve(strict=True)
    result = json.loads((calibrator_root / "result.json").read_text(encoding="utf-8"))
    receipt = result.get("final_calibrator")
    if not isinstance(receipt, Mapping):
        raise EnhancedSmokeHardStop("V0 final calibrator receipt is missing")
    spec = FrozenV0Spec(
        checkpoint_sha256=args.v0_checkpoint_sha256,
        calibrator_bundle_sha256=args.v0_calibrator_sha256,
    )

    return FrozenV0Endpoint.load(
        Path(args.v0_checkpoint).resolve(strict=True),
        calibrator_root,
        receipt,
        spec,
    )


def _encode_replay_record(
    record: ReplayRecord,
    *,
    v0: FrozenV0Endpoint,
    input_dim: int,
    v0_probability: float | None = None,
) -> ReplayTrainerExample:
    observation = copy.deepcopy(record.actor_visible_observation)
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise EnhancedSmokeHardStop("replay source observation lacks a selection contract")
    canonical_legal_table = terminal.legal_action_table(observation, dict(select))
    if len(canonical_legal_table) != len(record.legal_set):
        raise EnhancedSmokeHardStop(
            "replay sidecar and canonical R4F legal table disagree on option-position coverage"
        )
    if not record.policy_id or not record.deck_identity:
        raise EnhancedSmokeHardStop("replay row lacks exact actor policy/deck identity")
    chosen = record.chosen_selection
    if not isinstance(chosen, Mapping):
        raise EnhancedSmokeHardStop("replay chosen action contract is missing")
    indices = chosen.get("indices")
    if (
        not isinstance(indices, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
    ):
        raise EnhancedSmokeHardStop("replay chosen action indices are malformed")
    if chosen.get("index_space") == "option_position":
        positions = tuple(indices)
    else:
        position_by_source_index: dict[int, int] = {}
        for position, option in enumerate(record.legal_set):
            source_index = option.get("index", position)
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index in position_by_source_index
            ):
                raise EnhancedSmokeHardStop(
                    "replay source-index legal action mapping is missing or ambiguous"
                )
            position_by_source_index[source_index] = position
        try:
            positions = tuple(position_by_source_index[index] for index in indices)
        except KeyError as exc:
            raise EnhancedSmokeHardStop(
                "replay chosen source index is absent from its exact legal-action sidecar"
            ) from exc
    if len(positions) != len(set(positions)):
        raise EnhancedSmokeHardStop("replay chosen action maps to duplicate option positions")
    encoded = terminal.encode_actor_visible_decision(
        observation,
        feature_policy_id=record.policy_id,
        deck_hash=record.deck_identity,
        expected_input_dim=input_dim,
    )
    probability = (
        v0.score(np.array(record.s1, dtype=np.float32, copy=True))
        if v0_probability is None
        else float(v0_probability)
    )

    return ReplayTrainerExample(
        row_id=record.decision_id,
        option_features=encoded.features,
        selection=positions,
        min_count=encoded.min_count,
        max_count=encoded.max_count,
        v0_probability=float(probability),
        legal_action_count=len(record.legal_set),
    )


def _load_replay_examples(
    args: argparse.Namespace,
    *,
    v0: FrozenV0Endpoint,
    input_dim: int,
) -> tuple[BoundedReplayDataset, tuple[ReplayTrainerExample, ...]]:
    audit_rows = 1 if args.replay_derived_manifest else args.replay_rows
    dataset = load_bounded_replay_dataset(
        args.replay_cache,
        split_manifest_path=args.replay_split_manifest,
        max_rows=audit_rows,
        seed=args.seed,
        exact_d2_deck_identity=args.d2_deck_identity,
    )
    if args.replay_derived_manifest:
        derived = DerivedReplayIndex(args.replay_derived_manifest)
        rows = derived.global_rows()[:args.replay_rows]
        if len(rows) != args.replay_rows:
            raise EnhancedSmokeHardStop("derived replay smoke row count was silently clamped")
        examples = derived.load(rows)

        return dataset, examples
    s1_batch = np.stack(
        [np.array(record.s1, dtype=np.float32, copy=True) for record in dataset.records],
        axis=0,
    )
    probabilities = np.asarray(v0.score(s1_batch), dtype=np.float64)
    examples = tuple(
        _encode_replay_record(
            record,
            v0=v0,
            input_dim=input_dim,
            v0_probability=float(probability),
        )
        for record, probability in zip(dataset.records, probabilities, strict=True)
    )
    if len(examples) != args.replay_rows:
        raise EnhancedSmokeHardStop("replay smoke row count was silently clamped")

    return dataset, examples


def _branch_state_encoder(observation: Mapping[str, Any]) -> np.ndarray:
    features = np.asarray(
        value_source._features(
            {"raw_observation": copy.deepcopy(dict(observation))},
            value_source.FEATURE_CONFIG_V2,
        ),
        dtype=np.float32,
    )
    if features.shape != (value_source.FEATURE_DIM_V2,) or not np.isfinite(features).all():
        raise EnhancedSmokeHardStop("branch state encoder produced a malformed V0 feature row")

    return features


def _branch_policy_example(payload: BranchPolicyPayload, input_dim: int) -> Any:
    loaded_row = SimpleNamespace(
        row_sha256=payload.row_id,
        row={
            "actor_visible_root_observation": copy.deepcopy(
                dict(payload.actor_visible_root_observation)
            ),
            "actor_visible_legal_options": copy.deepcopy(
                list(payload.actor_visible_legal_options)
            ),
            "selected_root_action": copy.deepcopy(payload.selected_root_action),
            "aggregate_q_mean": copy.deepcopy(dict(payload.aggregate_q_mean)),
            "q_variance": copy.deepcopy(dict(payload.q_variance)),
            "continuation_policy_identity": payload.continuation_policy_identity,
        },
    )
    example = q_gpu._policy_example(loaded_row, input_dim)
    if not np.allclose(
        np.asarray(example.q_mean, dtype=np.float64),
        np.asarray(payload.ordered_q_mean, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise EnhancedSmokeHardStop("branch policy adapter reordered Q means")
    if not np.allclose(
        np.asarray(example.q_variance, dtype=np.float64),
        np.asarray(payload.ordered_q_variance, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise EnhancedSmokeHardStop("branch policy adapter reordered Q variances")

    return example


def _q_guided_actor_loss(
    actor: torch.nn.Module,
    examples: Sequence[Any],
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for example in examples:
        q_mean = torch.as_tensor(example.q_mean, dtype=torch.float32, device=device)
        target_index = int(q_mean.argmax().item())
        target_selection = example.sibling_selections[target_index]
        if example.order_sensitive:
            log_probability = q_gpu._ordered_sequence_log_probability(
                actor,
                example,
                target_selection,
                device,
            )
        else:
            log_probability = q_gpu._unordered_sequence_log_probability(
                actor,
                example,
                target_selection,
                device,
            )
        losses.append(-log_probability)
    if not losses:
        raise EnhancedSmokeHardStop("branch-Q actor guidance received no examples")

    return torch.stack(losses).mean()


def _load_branch_batches(args: argparse.Namespace, *, input_dim: int) -> tuple[Any, Any]:
    compatibility_config = json.loads(
        Path(args.branch_compatibility_config).resolve(strict=True).read_text(encoding="utf-8")
    )
    compatibility = compatibility_config.get("trainer_compatibility")
    if not isinstance(compatibility, Mapping):
        raise EnhancedSmokeHardStop("branch trainer compatibility receipt is missing")
    corpus_root = Path(args.branch_corpus_root).resolve(strict=True)
    plan = load_enhanced_branch_plan(
        Path(args.branch_manifest).resolve(strict=True),
        corpus_root=corpus_root,
        historical_tranche_binding=ShardBinding(
            corpus_root / "tranche" / "tranche.jsonl",
            corpus_root / "tranche" / "tranche.manifest.json",
        ),
        expected_trainer_compatibility=compatibility,
    )
    batches = build_branch_trainer_batches(
        plan,
        state_encoder=_branch_state_encoder,
        policy_example_factory=lambda payload: _branch_policy_example(payload, input_dim),
        batch_size=args.branch_batch_size,
        expected_state_dim=value_source.FEATURE_DIM_V2,
    )
    if (
        batches.receipt.unique_branch_rows_consumed != FULL_BRANCH_CORPUS_ROWS
        or batches.receipt.unique_branch_shards_consumed != FULL_BRANCH_CORPUS_SHARDS
    ):
        raise EnhancedSmokeHardStop("branch adapter did not emit the complete frozen corpus")

    return plan, batches


def _build_curriculum(
    args: argparse.Namespace,
    *,
    opponent_sampler_name: str,
) -> tuple[EnhancedCurriculumScheduler, dict[CurriculumCellKey, tuple[int, OpponentAgent]]]:
    terminal._register_artifact_loaders()
    entries = OPPONENT_SETS.get(opponent_sampler_name)
    if not entries:
        raise EnhancedSmokeHardStop("declared opponent mixture is unavailable")
    raw_weights = {raw: float(weight) for raw, weight in entries}
    multipliers = _parse_strength_multipliers(args.strength_multiplier)
    cells = []
    agents_by_key = {}
    for index, binding_id in enumerate(args.curriculum_binding):
        raw = f"binding:{binding_id}"
        if raw not in raw_weights:
            raise EnhancedSmokeHardStop("smoke curriculum binding is not in the declared mixture")
        agent = OpponentAgent.from_str(raw)
        identity = agent.pair_identity
        key = CurriculumCellKey(
            policy_identity=str(identity.get("binding_id") or ""),
            deck_hash=str(identity.get("deck_hash") or ""),
            strength_bucket=str(identity.get("strength_bucket") or ""),
        )
        cells.append(CurriculumCell(key=key, base_weight=raw_weights[raw]))
        agents_by_key[key] = (index, agent)
    scheduler = EnhancedCurriculumScheduler(
        cells,
        strength_bucket_multipliers=multipliers,
        seed=args.seed + 101,
    )
    curriculum_receipt = scheduler.receipt()
    if not curriculum_receipt["strength_bucket_multipliers_changed_effective_distribution"]:
        raise EnhancedSmokeHardStop("strength metadata did not affect the isolated curriculum")

    return scheduler, agents_by_key


def _scheduler_from_state(
    scheduler: EnhancedCurriculumScheduler,
    state: Mapping[str, Any],
) -> EnhancedCurriculumScheduler:
    restored = EnhancedCurriculumScheduler(
        scheduler.cells,
        strength_bucket_multipliers=scheduler.strength_bucket_multipliers,
        seed=scheduler.seed,
    )
    restored.load_state_dict(state)

    return restored


def _probe_trainer(trainer: EnhancedLeagueTrainer, example: ReplayTrainerExample) -> dict[str, float]:
    step = R4FRolloutStep(
        option_features=example.option_features,
        min_count=example.min_count,
        max_count=example.max_count,
        selection=example.selection,
        old_log_prob=0.0,
        old_value=0.0,
    )
    with torch.no_grad():
        log_probability, entropy, value = trainer.ppo.distribution.evaluate(step)

    return {
        "log_probability": float(log_probability.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "value": float(value.detach().cpu()),
    }


def _expected_rng_draws() -> dict[str, Any]:
    python_rng = random.Random()
    python_rng.setstate(random.getstate())
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(np.random.get_state())
    torch_rng = torch.Generator(device="cpu")
    torch_rng.set_state(torch.random.get_rng_state())

    return {
        "python": python_rng.random(),
        "numpy": float(numpy_rng.random_sample()),
        "torch": torch.rand(4, generator=torch_rng).tolist(),
    }


def _actual_rng_draws() -> dict[str, Any]:
    return {
        "python": random.random(),
        "numpy": float(np.random.random_sample()),
        "torch": torch.rand(4).tolist(),
    }


def _next_step_payload(
    *,
    trainer_probe: Mapping[str, float],
    scheduler: EnhancedCurriculumScheduler,
    environment_cursor: terminal.EnvironmentCursor,
    rng_draws: Mapping[str, Any],
) -> dict[str, Any]:
    sampled = scheduler.sample()

    return {
        "trainer_probe": dict(trainer_probe),
        "next_curriculum_sample": sampled.to_dict(),
        "next_environment_episode_index": environment_cursor.next_episode_index,
        "next_environment_seed": environment_cursor.rng.randrange(0, 2**31),
        "global_rng_draws": dict(rng_draws),
    }


def _checkpoint_resume_proof(
    *,
    trainer: EnhancedLeagueTrainer,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    frozen_states: Sequence[Any],
    extra_state: Mapping[str, Any],
    scheduler: EnhancedCurriculumScheduler,
    replay_probe: ReplayTrainerExample,
) -> tuple[dict[str, Any], EnhancedLeagueTrainer]:
    saved_state_sha256 = _trainer_state_sha256(trainer, extra_state)
    expected_scheduler = _scheduler_from_state(scheduler, extra_state["curriculum_state"])
    expected_cursor = terminal.EnvironmentCursor(random.Random(0))
    expected_cursor.load_state_dict(extra_state["environment_state"])
    expected_rng = _expected_rng_draws()
    expected_next = _next_step_payload(
        trainer_probe=_probe_trainer(trainer, replay_probe),
        scheduler=expected_scheduler,
        environment_cursor=expected_cursor,
        rng_draws=expected_rng,
    )
    saved_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    resumed, restored_extra = EnhancedLeagueTrainer.load_checkpoint(
        checkpoint_path,
        frozen_states,
        map_location=trainer.device,
    )
    restored_state_sha256 = _trainer_state_sha256(resumed, restored_extra)
    resumed_scheduler = _scheduler_from_state(scheduler, restored_extra["curriculum_state"])
    resumed_cursor = terminal.EnvironmentCursor(random.Random(0))
    resumed_cursor.load_state_dict(restored_extra["environment_state"])
    resumed_next = _next_step_payload(
        trainer_probe=_probe_trainer(resumed, replay_probe),
        scheduler=resumed_scheduler,
        environment_cursor=resumed_cursor,
        rng_draws=_actual_rng_draws(),
    )
    cuda_rng_equal = len(saved_cuda_rng) == len(torch.cuda.get_rng_state_all()) and all(
        torch.equal(left, right)
        for left, right in zip(saved_cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
    )
    expected_next_sha256 = _sha256_json(expected_next)
    resumed_next_sha256 = _sha256_json(resumed_next)
    parity = bool(
        saved_state_sha256 == restored_state_sha256
        and expected_next_sha256 == resumed_next_sha256
        and cuda_rng_equal
        and checkpoint_sha256 == _sha256_file(checkpoint_path)
    )
    if not parity:
        raise EnhancedSmokeHardStop("checkpoint/resume parity proof failed")
    evidence = _evidence({
        "parity": True,
        "checkpoint_sha256": checkpoint_sha256,
        "saved_state_sha256": saved_state_sha256,
        "restored_state_sha256": restored_state_sha256,
        "next_step_expected_sha256": expected_next_sha256,
        "next_step_resumed_sha256": resumed_next_sha256,
    })

    return evidence, resumed


def _contains_forbidden_persisted_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_PERSISTED_KEYS:
                return True
            if _contains_forbidden_persisted_key(item):
                return True

        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_persisted_key(item) for item in value)

    return False


def _leakage_evidence(
    *,
    replay: BoundedReplayDataset,
    branch_plan: Any,
    checkpoint_path: Path,
) -> dict[str, Any]:
    for record in replay.records:
        assert_deployable_observation(
            record.actor_visible_observation,
            expected_mode=InformationMode.ACTOR_VISIBLE.value,
        )
    negative = copy.deepcopy(replay.records[0].actor_visible_observation)
    current = negative.get("current")
    if not isinstance(current, dict):
        raise EnhancedSmokeHardStop("replay negative-control observation lacks current state")
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise EnhancedSmokeHardStop("replay negative-control observation lacks two players")
    actor = int(current.get("yourIndex", replay.records[0].actor))
    opponent = 1 - actor
    if not isinstance(players[opponent], dict):
        raise EnhancedSmokeHardStop("replay negative-control opponent state is malformed")
    players[opponent]["hand"] = [{"cardId": 999_999}]
    negative_rejected = False
    try:
        assert_deployable_observation(negative)
    except ValueError:
        negative_rejected = True
    if not negative_rejected:
        raise EnhancedSmokeHardStop("hidden-information negative control was not rejected")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _contains_forbidden_persisted_key(checkpoint_payload):
        raise EnhancedSmokeHardStop("enhanced checkpoint contains a forbidden private-state key")
    try:
        for row in branch_plan.rows:
            validate_safe_shard(row.loaded_row.row)
    except (TypeError, ValueError) as exc:
        raise EnhancedSmokeHardStop(
            "branch leakage scan rejected a frozen actor-visible teacher row"
        ) from exc

    return _evidence({
        "zero_hidden_information_leakage": True,
        "violation_count": 0,
        "scanned_artifact_count": len(replay.records) + len(branch_plan.rows) + 1,
        "negative_control_rejected": True,
    })


def _stage_chain(
    *,
    trainer_identity: str,
    v0_receipt: StageReceipt,
    replay_receipt: StageReceipt,
    branch_receipt: StageReceipt,
    terminal_receipt: StageReceipt,
    evidence_sha256s: Sequence[str],
) -> list[dict[str, Any]]:
    model_receipts = (v0_receipt, replay_receipt, branch_receipt)
    stages = []
    previous_stage_sha256 = STAGE_CHAIN_GENESIS_SHA256
    for index, (name, receipt, evidence_sha256) in enumerate(
        zip(REQUIRED_STAGE_NAMES[:3], model_receipts, evidence_sha256s[:3], strict=True)
    ):
        stage = build_stage_record(
            stage_index=index,
            stage_name=name,
            trainer_identity=trainer_identity,
            input_model_sha256=_component_model_sha256(receipt.input_component_hashes),
            output_model_sha256=_component_model_sha256(receipt.output_component_hashes),
            signal_consumed=True,
            signal_evidence_sha256=evidence_sha256,
            previous_stage_sha256=previous_stage_sha256,
        )
        stages.append(stage)
        previous_stage_sha256 = stage["stage_sha256"]
    branch_output = _component_model_sha256(branch_receipt.output_component_hashes)
    curriculum_stage = build_stage_record(
        stage_index=3,
        stage_name=REQUIRED_STAGE_NAMES[3],
        trainer_identity=trainer_identity,
        input_model_sha256=branch_output,
        output_model_sha256=branch_output,
        signal_consumed=True,
        signal_evidence_sha256=evidence_sha256s[3],
        previous_stage_sha256=previous_stage_sha256,
    )
    stages.append(curriculum_stage)
    terminal_stage = build_stage_record(
        stage_index=4,
        stage_name=REQUIRED_STAGE_NAMES[4],
        trainer_identity=trainer_identity,
        input_model_sha256=_component_model_sha256(terminal_receipt.input_component_hashes),
        output_model_sha256=_component_model_sha256(terminal_receipt.output_component_hashes),
        signal_consumed=True,
        signal_evidence_sha256=evidence_sha256s[4],
        previous_stage_sha256=curriculum_stage["stage_sha256"],
    )
    stages.append(terminal_stage)

    return stages


def _build_receipt(
    *,
    args: argparse.Namespace,
    trainer: EnhancedLeagueTrainer,
    initial_hashes: Mapping[str, str],
    initial_actor: torch.Tensor,
    initial_critic: torch.Tensor,
    v0_receipt: StageReceipt,
    replay_receipt: StageReceipt,
    branch_receipt: StageReceipt,
    terminal_receipt: StageReceipt,
    replay: BoundedReplayDataset,
    branch_plan: Any,
    branch_batches: Any,
    curriculum: EnhancedCurriculumScheduler,
    checkpoint_resume: Mapping[str, Any],
    leakage_scan: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    final_hashes = trainer.component_hashes()
    initial_model_sha256 = _component_model_sha256(initial_hashes)
    final_model_sha256 = _component_model_sha256(final_hashes)
    v0_evidence = _evidence({
        "consumed": True,
        "mode": "distillation",
        "weights_consumed": False,
        "distillation_consumed": True,
        "source_checkpoint_sha256": args.v0_checkpoint_sha256,
    })
    replay_evidence = _evidence({
        "unique_rows_consumed": replay_receipt.consumed["unique_replay_rows_consumed"],
        "row_identity_set_sha256": _sha256_json(sorted(
            record.decision_id for record in replay.records
        )),
        "source_catalog_sha256": _sha256_file(
            Path(args.replay_catalog).resolve(strict=True)
        ),
    })
    branch_evidence = _evidence({
        "unique_rows_consumed": branch_receipt.consumed["unique_branch_rows_consumed"],
        "unique_shards_consumed": branch_receipt.consumed["unique_branch_shards_consumed"],
        "row_identity_set_sha256": _sha256_json(sorted(
            row.row_id for row in branch_plan.rows
        )),
        "shard_identity_set_sha256": _sha256_json(sorted(branch_plan.shard_ids)),
        "root_manifest_sha256": branch_batches.receipt.root_manifest_sha256,
    })
    curriculum_receipt = curriculum.receipt()
    curriculum_evidence = _evidence({
        "consumed": True,
        "deck_identity_participated": curriculum_receipt[
            "deck_identity_participated_in_scheduler_weighting"
        ],
        "anchored_strength_bucket_participated": curriculum_receipt[
            "strength_bucket_participated_in_scheduler_weighting"
        ],
        "all_strength_buckets_anchored": curriculum_receipt["proofs"][
            "anchored_strength_bucket_in_every_weighting_key"
        ],
        "effective_distribution_l1_delta_from_base": curriculum_receipt[
            "effective_distribution_l1_delta_from_base"
        ],
        "scheduler_config_sha256": curriculum.config_sha256,
        "scheduler_receipt_sha256": _sha256_json(curriculum_receipt),
    })
    actor_delta = _parameter_delta(initial_actor, trainer.ppo.model.actor)
    critic_delta = _parameter_delta(initial_critic, trainer.ppo.model.critic)
    parameter_updates = _evidence({
        "actor_updated": True,
        "critic_updated": True,
        "actor_delta_norm": actor_delta,
        "critic_delta_norm": critic_delta,
        "actor_before_sha256": initial_hashes["actor"],
        "actor_after_sha256": final_hashes["actor"],
        "critic_before_sha256": initial_hashes["critic"],
        "critic_after_sha256": final_hashes["critic"],
        "before_model_sha256": initial_model_sha256,
        "after_model_sha256": final_model_sha256,
    })
    terminal_evidence = _evidence({
        "consumed": terminal_receipt.consumed["terminal_reward_league_ppo_consumed"],
        "terminal_episodes": terminal_receipt.consumed["episodes"],
        "optimizer_updates": terminal_receipt.optimizer_steps,
        "illegal_actions": 0,
        "crashes": 0,
        "timeouts": 0,
    })
    evidence_sha256s = (
        v0_evidence["evidence_sha256"],
        replay_evidence["evidence_sha256"],
        branch_evidence["evidence_sha256"],
        curriculum_evidence["evidence_sha256"],
        terminal_evidence["evidence_sha256"],
    )
    stage_chain = _stage_chain(
        trainer_identity=trainer.lineage_id,
        v0_receipt=v0_receipt,
        replay_receipt=replay_receipt,
        branch_receipt=branch_receipt,
        terminal_receipt=terminal_receipt,
        evidence_sha256s=evidence_sha256s,
    )
    receipt = {
        "schema_version": SMOKE_RECEIPT_SCHEMA_VERSION,
        "classification": SMOKE_CLASSIFICATION,
        "status": "pass",
        "trainer_identity": trainer.lineage_id,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "v0_weights_or_distillation_consumed": True,
        "unique_replay_rows_consumed": replay_receipt.consumed[
            "unique_replay_rows_consumed"
        ],
        "unique_branch_rows_consumed": branch_receipt.consumed[
            "unique_branch_rows_consumed"
        ],
        "unique_branch_shards_consumed": branch_receipt.consumed[
            "unique_branch_shards_consumed"
        ],
        "opponent_deck_strength_metadata_consumed_by_scheduler": True,
        "actor_parameters_updated": True,
        "critic_parameters_updated": True,
        "checkpoint_resume_parity": checkpoint_resume["parity"],
        "zero_hidden_information_leakage": leakage_scan[
            "zero_hidden_information_leakage"
        ],
        "same_trainer_identity_stage_hash_chain": True,
        "v0_consumption": v0_evidence,
        "replay_consumption": replay_evidence,
        "branch_consumption": branch_evidence,
        "curriculum_consumption": curriculum_evidence,
        "parameter_updates": parameter_updates,
        "checkpoint_resume": dict(checkpoint_resume),
        "leakage_scan": dict(leakage_scan),
        "terminal_ppo": terminal_evidence,
        "stage_chain": stage_chain,
        "stage_chain_sha256": compute_stage_chain_sha256(stage_chain),
        "external_launch_performed": False,
        "protected_surfaces_mutated": False,
    }
    if checkpoint_resume["checkpoint_sha256"] != checkpoint_sha256:
        raise EnhancedSmokeHardStop("checkpoint receipt identity drifted")

    return validate_enhanced_smoke_receipt(receipt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay_rows <= 0 or args.branch_batch_size <= 0:
        raise EnhancedSmokeHardStop("replay rows and branch batch size must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_receipt_path = output_dir / "smoke_receipt.json"
    if pass_receipt_path.exists():
        raise EnhancedSmokeHardStop("passed smoke receipt already exists; use a new output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise EnhancedSmokeHardStop("CUDA was requested but is unavailable")
    source_checkpoint = Path(args.r7_checkpoint).resolve(strict=True)
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    input_dim = int(source_payload["input_dim"])
    v0 = _load_v0(args)
    replay, replay_examples = _load_replay_examples(
        args,
        v0=v0,
        input_dim=input_dim,
    )
    branch_plan, branch_batches = _load_branch_batches(args, input_dim=input_dim)
    ppo_config = R4FPPOConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        gamma=1.0,
        gae_lambda=0.95,
        clip_coef=0.15,
        value_coef=0.5,
        entropy_coef=0.005,
        max_grad_norm=0.5,
        update_epochs=1,
        minibatch_size=64,
        critic_hidden_dim=128,
        device=device,
    )
    frozen_states = terminal._frozen_equivalence_panel(input_dim)
    trainer = EnhancedLeagueTrainer.from_r7_checkpoint(
        source_checkpoint,
        frozen_states,
        q_state_dim=branch_batches.state_dim,
        q_action_dim=branch_batches.action_dim,
        ppo_config=ppo_config,
    )
    initial_hashes = trainer.component_hashes()
    initial_actor = _flat_parameters(trainer.ppo.model.actor)
    initial_critic = _flat_parameters(trainer.ppo.model.critic)
    v0_receipt = trainer.distill_v0(replay_examples)
    stage_metadata = {
        "configuration_sha256": _sha256_json({
            "replay_rows": args.replay_rows,
            "replay_batch_size": args.replay_batch_size,
            "branch_batch_size": args.branch_batch_size,
            "seed": args.seed,
        }),
        "action_schema_identity": "deepmew.actor_transition.v1",
        "observation_schema_identity": "deepmew.replay_action_row.v1:state_json actor_visible",
        "deck_identities": [args.d2_deck_identity],
        "input_hashes": trainer.component_hashes(),
        "stage_counters": {"v0_distillation": 1, "replay_warm_start": 0},
        "data_cursors": {"replay_next_row": 0, "branch_next_row": 0},
        "curriculum_state": {"status": "not_started"},
        "elapsed_seconds": 0.0,
        "code_identity": _sha256_file(Path(__file__)),
        "next_operation": "replay_warm_start",
        "continuous_stage_chain_sha256": _sha256_json(asdict(v0_receipt)),
    }
    trainer.save_stage_checkpoint(
        output_dir / "v0_distillation_stage.pt",
        boundary_stage="v0_initialization_or_distillation",
        frozen_metadata=stage_metadata,
        extra_state={"smoke": True},
    )
    trainer.begin_replay_warm_start()
    for offset in range(0, len(replay_examples), args.replay_batch_size):
        trainer.replay_warm_start_batch(replay_examples[offset:offset + args.replay_batch_size])
    replay_receipt = trainer.finish_replay_warm_start(expected_unique_rows=len(replay_examples))
    branch_receipt = trainer.branch_q_stage(
        branch_batches,
        expected_unique_rows=FULL_BRANCH_CORPUS_ROWS,
        expected_unique_shards=FULL_BRANCH_CORPUS_SHARDS,
        guidance_loss=_q_guided_actor_loss,
    )
    lane = terminal.LANES["starmie"]
    experiment = resolve_experiment(lane.league_experiment)
    scheduler, agents_by_key = _build_curriculum(
        args,
        opponent_sampler_name=experiment.opponent_sampler,
    )
    opponent_adapter = CurriculumOpponentAdapter(scheduler, agents_by_key)
    learner_deck_path = resolve_repo_path(experiment.agent_deck_path).resolve(strict=True)
    learner_deck = read_deck_any(learner_deck_path)
    if len(learner_deck) != 60:
        raise EnhancedSmokeHardStop("local learner deck is not a complete 60-card deck")
    learner_deck_hash = r4f_feature_deck_hash(learner_deck)
    environment_cursor = terminal.EnvironmentCursor(random.Random(args.seed + 202))
    safety_counters = dict(ZERO_SAFETY)
    collected = terminal.collect_terminal_episode(
        learner=trainer.ppo,
        sampler=opponent_adapter,
        environment_cursor=environment_cursor,
        learner_deck=learner_deck,
        learner_deck_hash=learner_deck_hash,
        feature_policy_id=lane.feature_policy_id,
        max_steps=experiment.max_steps_per_game,
        counters=safety_counters,
    )
    if opponent_adapter.last_sample is None:
        raise EnhancedSmokeHardStop("real episode did not consume the enhanced curriculum")
    sampled_key = opponent_adapter.last_sample.cell.key
    terminal_receipt = trainer.terminal_ppo_stage(
        (collected.episode,),
        curriculum_receipt={
            "opponent_deck_strength_metadata_consumed_by_scheduler": True,
            "sampled_policy_id": sampled_key.policy_identity,
            "sampled_deck_hash": sampled_key.deck_hash,
            "sampled_strength_bucket": sampled_key.strength_bucket,
        },
        safety=safety_counters,
    )
    extra_state = {
        "schema_version": SCHEMA_VERSION,
        "curriculum_state": scheduler.state_dict(),
        "environment_state": environment_cursor.state_dict(),
        "safety": safety_counters,
        "sampled_curriculum_key": sampled_key.to_dict(),
        "terminal_result": collected.result,
        "terminal_reward": collected.episode.terminal_reward,
        "replay_cache_manifest_sha256": replay.cache_manifest_sha256,
        "replay_split_manifest_sha256": replay.split_manifest_sha256,
        "branch_root_manifest_sha256": branch_batches.receipt.root_manifest_sha256,
    }
    checkpoint_path = output_dir / "enhanced_smoke_checkpoint.pt"
    checkpoint_sha256 = trainer.save_checkpoint(checkpoint_path, extra_state=extra_state)
    checkpoint_resume, resumed_trainer = _checkpoint_resume_proof(
        trainer=trainer,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        frozen_states=frozen_states,
        extra_state=extra_state,
        scheduler=scheduler,
        replay_probe=replay_examples[0],
    )
    leakage_scan = _leakage_evidence(
        replay=replay,
        branch_plan=branch_plan,
        checkpoint_path=checkpoint_path,
    )
    receipt = _build_receipt(
        args=args,
        trainer=resumed_trainer,
        initial_hashes=initial_hashes,
        initial_actor=initial_actor,
        initial_critic=initial_critic,
        v0_receipt=v0_receipt,
        replay_receipt=replay_receipt,
        branch_receipt=branch_receipt,
        terminal_receipt=terminal_receipt,
        replay=replay,
        branch_plan=branch_plan,
        branch_batches=branch_batches,
        curriculum=scheduler,
        checkpoint_resume=checkpoint_resume,
        leakage_scan=leakage_scan,
        checkpoint_sha256=checkpoint_sha256,
    )
    _atomic_json(pass_receipt_path, receipt)
    _atomic_json(output_dir / "run_summary.json", {
        "schema_version": SCHEMA_VERSION,
        "classification": SMOKE_CLASSIFICATION,
        "status": "pass",
        "trainer_identity": trainer.lineage_id,
        "smoke_receipt_sha256": _sha256_file(pass_receipt_path),
        "checkpoint_sha256": checkpoint_sha256,
        "terminal_result": collected.result,
        "terminal_reward": collected.episode.terminal_reward,
        "terminal_decisions": len(collected.episode.steps),
        "sampled_curriculum_key": sampled_key.to_dict(),
        "control_jobs_modified": False,
        "external_launch_performed": False,
        "protected_surfaces_mutated": False,
    })

    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / ".research"
            / "terminal_league_starmie_grimmsnarl_v0"
            / "enhanced_generation_local_smoke_v0"
        ),
    )
    parser.add_argument(
        "--r7-checkpoint",
        default=r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt",
    )
    parser.add_argument(
        "--v0-checkpoint",
        default=(
            r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs"
            r"\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt"
        ),
    )
    parser.add_argument(
        "--v0-checkpoint-sha256",
        default="cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9",
    )
    parser.add_argument(
        "--v0-calibrator-root",
        default=str(
            ROOT
            / ".research"
            / "value_q_experiment"
            / "artifacts"
            / "phasef5_grouped_calibration_raw_s1_s915002_v3"
        ),
    )
    parser.add_argument(
        "--v0-calibrator-sha256",
        default="6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398",
    )
    parser.add_argument(
        "--replay-cache",
        default=(
            r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2"
            r"\7459a801-77afa18a"
        ),
    )
    parser.add_argument(
        "--replay-split-manifest",
        default=r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json",
    )
    parser.add_argument(
        "--replay-catalog",
        default=r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\DATASET_CATALOG.json",
    )
    parser.add_argument("--replay-rows", type=int, default=DEFAULT_REPLAY_ROWS)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--replay-derived-manifest")
    parser.add_argument("--d2-deck-identity", default=DEFAULT_D2_DECK_IDENTITY)
    parser.add_argument(
        "--branch-manifest",
        default=str(
            ROOT
            / ".research"
            / "q_guided_starmie"
            / "modal_first_wave_20260712"
            / "corpus_audit_final_20260713"
            / "final_corpus.manifest.json"
        ),
    )
    parser.add_argument(
        "--branch-corpus-root",
        default=str(ROOT / ".research" / "local" / "deepmew-final-corpus-20260713-1004"),
    )
    parser.add_argument(
        "--branch-compatibility-config",
        default=str(
            ROOT
            / ".research"
            / "q_guided_starmie"
            / "modal_first_wave_20260712"
            / "m3_streaming_v2_migration_config.json"
        ),
    )
    parser.add_argument("--branch-batch-size", type=int, default=DEFAULT_BRANCH_BATCH_SIZE)
    parser.add_argument(
        "--curriculum-binding",
        action="append",
        default=list(DEFAULT_CURRICULUM_BINDINGS),
    )
    parser.add_argument(
        "--strength-multiplier",
        action="append",
        default=list(DEFAULT_STRENGTH_MULTIPLIERS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    try:
        receipt = run(args)
    except Exception as exc:
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "hard_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "external_launch_performed": False,
            "protected_surfaces_mutated": False,
        }
        _atomic_json(output_dir / "failure_receipt.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)

        return 1
    print(json.dumps({
        "status": receipt["status"],
        "trainer_identity": receipt["trainer_identity"],
        "unique_replay_rows_consumed": receipt["unique_replay_rows_consumed"],
        "unique_branch_rows_consumed": receipt["unique_branch_rows_consumed"],
        "unique_branch_shards_consumed": receipt["unique_branch_shards_consumed"],
        "checkpoint_resume_parity": receipt["checkpoint_resume_parity"],
        "zero_hidden_information_leakage": receipt["zero_hidden_information_leakage"],
    }, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
