"""Fail-closed adapter for the frozen full branch-Q research corpus.

The adapter binds the exact audited 9,184-row corpus, applies its frozen
row-hash split, and produces :class:`BranchTrainerBatch` values without
depending on script-layer encoders or an M3 model checkpoint.  The historical
64-row tranche binding is data-lineage evidence required by the existing
corpus loader; it is never used as a source of model weights.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from deepmew.q_guided.teacher_contract import validate_safe_shard
from deepmew.q_guided.training import (
    EncodedQExample,
    LoadedTeacherRow,
    ManifestBoundCorpus,
    RowSplitPolicy,
    ShardBinding,
    encode_actor_visible_actions,
    encode_teacher_row,
    load_manifest_bound_corpus,
)
from deepmew.rl.enhanced_trainer import BranchTrainerBatch


BRANCH_CONSUMPTION_RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_branch_consumption_receipt.v1"

# Immutable identity and coverage of the audited frozen-final corpus.
FULL_BRANCH_CORPUS_MANIFEST_SHA256 = "be941e15b309fbdbd4eccfdbd41a012b59ae971fccdf96977988caebf4d33afc"
FULL_BRANCH_CORPUS_LINEAGE_ID = "qcl1-2c3aabe5d9f03e9779a848c1ee3455f6"
FULL_BRANCH_CORPUS_GENERATION = "q_guided_first_wave_20260712"
FULL_BRANCH_CORPUS_ORDERED_SHARDS_SHA256 = (
    "4becec693b7dac4abe5f71b0d11702fd0579521f8e2bf65a3a03d2d384b7c635"
)
FULL_BRANCH_CORPUS_ROWS = 9_184
FULL_BRANCH_CORPUS_SHARDS = 288
FULL_BRANCH_TRAIN_ROWS = 7_383
FULL_BRANCH_HELDOUT_ROWS = 1_801

# The aggregate labels in the frozen corpus are bound to this validated V0.
FULL_BRANCH_LABEL_V0_CHECKPOINT_SHA256 = (
    "cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9"
)
FULL_BRANCH_LABEL_V0_CALIBRATOR_SHA256 = (
    "6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398"
)

FROZEN_BRANCH_SPLIT_POLICY = RowSplitPolicy(modulus=5, heldout_residues=(0,))


class EnhancedBranchContractError(RuntimeError):
    """The full-corpus branch adapter encountered a hard contract failure."""


@dataclass(frozen=True)
class BranchPolicyPayload:
    """Safe, aligned inputs for a caller-owned Q-guided policy encoder.

    The legal sibling records contain only actor-visible selections and
    semantics.  Aggregate Q targets are permitted privileged-teacher labels;
    no hidden-world identity or simulator state is exposed.
    """

    row_id: str
    actor_visible_root_observation: Mapping[str, Any]
    actor_visible_legal_options: Sequence[Mapping[str, Any]]
    legal_sibling_action_ids: tuple[str, ...]
    selected_root_action: Any
    aggregate_q_mean: Mapping[str, float]
    q_variance: Mapping[str, float]
    continuation_policy_identity: str

    @property
    def ordered_q_mean(self) -> tuple[float, ...]:

        return tuple(float(self.aggregate_q_mean[action_id]) for action_id in self.legal_sibling_action_ids)

    @property
    def ordered_q_variance(self) -> tuple[float, ...]:

        return tuple(float(self.q_variance[action_id]) for action_id in self.legal_sibling_action_ids)


@dataclass(frozen=True)
class EnhancedBranchRow:
    loaded_row: LoadedTeacherRow
    shard_id: str
    split: str

    @property
    def row_id(self) -> str:

        return self.loaded_row.row_sha256


@dataclass(frozen=True)
class EnhancedBranchPlan:
    corpus: ManifestBoundCorpus
    rows: tuple[EnhancedBranchRow, ...]
    shard_ids: tuple[str, ...]

    @property
    def training_rows(self) -> tuple[EnhancedBranchRow, ...]:

        return tuple(row for row in self.rows if row.split == "train")

    @property
    def heldout_rows(self) -> tuple[EnhancedBranchRow, ...]:

        return tuple(row for row in self.rows if row.split == "heldout")


@dataclass(frozen=True)
class EnhancedBranchConsumptionReceipt:
    schema_version: str
    root_manifest_sha256: str
    corpus_lineage_id: str
    ordered_shard_identity_sha256: str
    split_policy_sha256: str
    unique_branch_rows_consumed: int
    unique_branch_shards_consumed: int
    training_rows_consumed: int
    heldout_rows_consumed: int
    policy_examples_built: int
    training_batches: int
    heldout_batches: int
    all_rows_emitted_once: bool
    all_shards_covered: bool
    zero_hidden_information_leakage: bool
    m3_checkpoint_dependency: bool

    def to_dict(self) -> dict[str, Any]:

        return asdict(self)


@dataclass(frozen=True)
class EnhancedBranchBatchSet:
    batches: tuple[BranchTrainerBatch, ...]
    state_dim: int
    action_dim: int
    receipt: EnhancedBranchConsumptionReceipt

    def __iter__(self) -> Iterator[BranchTrainerBatch]:

        return iter(self.batches)

    def __len__(self) -> int:

        return len(self.batches)


def _require_exact_corpus_identity(corpus: ManifestBoundCorpus) -> None:
    expected_identity = {
        "root manifest": FULL_BRANCH_CORPUS_MANIFEST_SHA256,
        "corpus lineage": FULL_BRANCH_CORPUS_LINEAGE_ID,
        "corpus generation": FULL_BRANCH_CORPUS_GENERATION,
        "ordered shard identity": FULL_BRANCH_CORPUS_ORDERED_SHARDS_SHA256,
    }
    observed_identity = {
        "root manifest": corpus.root_manifest_sha256,
        "corpus lineage": corpus.corpus_lineage_id,
        "corpus generation": corpus.generation,
        "ordered shard identity": corpus.ordered_shard_identity_sha256,
    }
    for name, expected in expected_identity.items():
        if observed_identity[name] != expected:
            raise EnhancedBranchContractError(f"full branch corpus {name} mismatch")
    if len(corpus.shards) != FULL_BRANCH_CORPUS_SHARDS:
        raise EnhancedBranchContractError(
            f"full branch corpus requires exactly {FULL_BRANCH_CORPUS_SHARDS} shards"
        )
    if corpus.total_rows != FULL_BRANCH_CORPUS_ROWS:
        raise EnhancedBranchContractError(
            f"full branch corpus requires exactly {FULL_BRANCH_CORPUS_ROWS} rows"
        )
    if len(corpus.dataset.rows) != FULL_BRANCH_CORPUS_ROWS:
        raise EnhancedBranchContractError("full branch corpus loaded-row coverage is incomplete")
    if corpus.trainer_compatibility.get("split_policy_sha256") != FROZEN_BRANCH_SPLIT_POLICY.sha256:
        raise EnhancedBranchContractError("trainer compatibility does not bind the frozen branch split")
    source_compatibility = corpus.source_compatibility
    if source_compatibility.get("label_v_checkpoint_sha256s") != [
        FULL_BRANCH_LABEL_V0_CHECKPOINT_SHA256
    ]:
        raise EnhancedBranchContractError("full branch corpus V0 label checkpoint identity mismatch")
    if source_compatibility.get("label_v_calibrator_sha256s") != [
        FULL_BRANCH_LABEL_V0_CALIBRATOR_SHA256
    ]:
        raise EnhancedBranchContractError("full branch corpus V0 calibrator identity mismatch")
    if source_compatibility.get("split_values") != ["train"]:
        raise EnhancedBranchContractError("full branch corpus source split inventory mismatch")


def _bind_and_validate_rows(corpus: ManifestBoundCorpus) -> EnhancedBranchPlan:
    expected_shard_ids = tuple(shard.identity.shard_sha256 for shard in corpus.shards)
    expected_manifest_ids = tuple(shard.identity.manifest_sha256 for shard in corpus.shards)
    if len(set(expected_shard_ids)) != FULL_BRANCH_CORPUS_SHARDS:
        raise EnhancedBranchContractError("full branch corpus contains duplicate shard identities")
    if corpus.dataset.shard_sha256s != expected_shard_ids:
        raise EnhancedBranchContractError("dataset shard order disagrees with frozen corpus order")
    if corpus.dataset.manifest_sha256s != expected_manifest_ids:
        raise EnhancedBranchContractError("dataset manifest order disagrees with frozen corpus order")
    rows: list[EnhancedBranchRow] = []
    row_offset = 0
    for expected_ordinal, shard in enumerate(corpus.shards):
        shard.identity.validate()
        if shard.identity.ordinal != expected_ordinal:
            raise EnhancedBranchContractError("full branch corpus shard ordinals are not contiguous")
        next_offset = row_offset + shard.identity.row_count
        loaded_rows = corpus.dataset.rows[row_offset:next_offset]
        loaded_hashes = tuple(row.row_sha256 for row in loaded_rows)
        if loaded_hashes != shard.row_sha256s:
            raise EnhancedBranchContractError("full branch corpus row-to-shard alignment mismatch")
        if len(loaded_rows) != shard.identity.row_count:
            raise EnhancedBranchContractError("full branch corpus shard row count mismatch")
        for loaded_row in loaded_rows:
            try:
                validate_safe_shard(loaded_row.row)
            except (TypeError, ValueError) as exc:
                raise EnhancedBranchContractError(
                    f"branch row {loaded_row.row_sha256} failed the leakage-safe shard contract"
                ) from exc
            rows.append(
                EnhancedBranchRow(
                    loaded_row=loaded_row,
                    shard_id=shard.identity.shard_sha256,
                    split=FROZEN_BRANCH_SPLIT_POLICY.split_for_row(loaded_row.row_sha256),
                )
            )
        row_offset = next_offset
    if row_offset != len(corpus.dataset.rows):
        raise EnhancedBranchContractError("full branch corpus has unbound trailing rows")
    row_ids = tuple(row.row_id for row in rows)
    if len(row_ids) != FULL_BRANCH_CORPUS_ROWS or len(set(row_ids)) != FULL_BRANCH_CORPUS_ROWS:
        raise EnhancedBranchContractError("full branch corpus row identities are incomplete or duplicated")
    training_rows = sum(row.split == "train" for row in rows)
    heldout_rows = sum(row.split == "heldout" for row in rows)
    if training_rows != FULL_BRANCH_TRAIN_ROWS or heldout_rows != FULL_BRANCH_HELDOUT_ROWS:
        raise EnhancedBranchContractError(
            "frozen branch split must contain exactly "
            f"{FULL_BRANCH_TRAIN_ROWS:,} train and {FULL_BRANCH_HELDOUT_ROWS:,} held-out rows"
        )

    return EnhancedBranchPlan(
        corpus=corpus,
        rows=tuple(rows),
        shard_ids=expected_shard_ids,
    )


def load_enhanced_branch_plan(
    root_manifest_path: Path,
    *,
    corpus_root: Path,
    historical_tranche_binding: ShardBinding,
    expected_trainer_compatibility: Mapping[str, Any],
) -> EnhancedBranchPlan:
    """Load and revalidate the exact frozen-final branch corpus.

    ``historical_tranche_binding`` proves the root manifest's recorded source
    mapping.  This API accepts no M3 checkpoint, optimizer, or model state.
    """
    corpus = load_manifest_bound_corpus(
        Path(root_manifest_path),
        corpus_root=Path(corpus_root),
        m3_tranche_binding=historical_tranche_binding,
        expected_trainer_compatibility=expected_trainer_compatibility,
    )
    _require_exact_corpus_identity(corpus)

    return _bind_and_validate_rows(corpus)


def _validate_positive_int(value: int | None, name: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:

        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnhancedBranchContractError(f"{name} must be a positive integer")


def _encoder_compatible_legal_options(
    legal_options: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    encoded_options = copy.deepcopy(list(legal_options))
    for option in encoded_options:
        if option.get("selection") == [] and option.get("semantics") == []:
            option["semantics"] = [{
                "option_type": "STOP",
                "select_type": "",
                "select_context": "",
                "semantic_key": "STOP",
            }]

    return encoded_options


def _encoder_compatible_loaded_row(loaded_row: LoadedTeacherRow) -> LoadedTeacherRow:
    row = copy.deepcopy(loaded_row.row)
    row["actor_visible_legal_options"] = _encoder_compatible_legal_options(
        row["actor_visible_legal_options"]
    )

    return LoadedTeacherRow(
        split=loaded_row.split,
        row=row,
        row_sha256=loaded_row.row_sha256,
        actor_visible_signature_sha256=loaded_row.actor_visible_signature_sha256,
        actor_visible_source_sha256=loaded_row.actor_visible_source_sha256,
    )


def _policy_payload(row: EnhancedBranchRow) -> BranchPolicyPayload:
    raw = row.loaded_row.row
    legal_options = copy.deepcopy(raw["actor_visible_legal_options"])
    encoded_options = _encoder_compatible_legal_options(legal_options)
    _, action_ids = encode_actor_visible_actions(encoded_options)
    q_mean = copy.deepcopy(raw["aggregate_q_mean"])
    q_variance = copy.deepcopy(raw["q_variance"])
    if set(action_ids) != set(q_mean) or set(action_ids) != set(q_variance):
        raise EnhancedBranchContractError("policy payload sibling identities do not align with Q targets")

    return BranchPolicyPayload(
        row_id=row.row_id,
        actor_visible_root_observation=copy.deepcopy(raw["actor_visible_root_observation"]),
        actor_visible_legal_options=legal_options,
        legal_sibling_action_ids=action_ids,
        selected_root_action=copy.deepcopy(raw["selected_root_action"]),
        aggregate_q_mean=q_mean,
        q_variance=q_variance,
        continuation_policy_identity=str(raw["continuation_policy_identity"]),
    )


def _validate_encoded_dimensions(
    example: EncodedQExample,
    *,
    state_dim: int | None,
    action_dim: int | None,
) -> tuple[int, int]:
    if example.state_features.ndim != 1 or example.state_features.numel() == 0:
        raise EnhancedBranchContractError("branch state encoding must be one non-empty vector")
    if (
        example.action_features.ndim != 2
        or example.action_features.shape[0] <= 0
        or example.action_features.shape[1] <= 0
    ):
        raise EnhancedBranchContractError("branch action encoding must be one non-empty matrix")
    observed_state_dim = int(example.state_features.shape[0])
    observed_action_dim = int(example.action_features.shape[1])
    if state_dim is not None and observed_state_dim != state_dim:
        raise EnhancedBranchContractError("branch state feature dimension drifted within the corpus")
    if action_dim is not None and observed_action_dim != action_dim:
        raise EnhancedBranchContractError("branch action feature dimension drifted within the corpus")

    return observed_state_dim, observed_action_dim


def _build_batch(
    rows: Sequence[EnhancedBranchRow],
    *,
    is_training: bool,
    state_encoder: Callable[[Mapping[str, Any]], Any],
    policy_example_factory: Callable[[BranchPolicyPayload], Any],
    state_dim: int | None,
    action_dim: int | None,
) -> tuple[BranchTrainerBatch, int, int]:
    encoded_rows: list[EncodedQExample] = []
    observed_state_dim = state_dim
    observed_action_dim = action_dim
    for row in rows:
        example = encode_teacher_row(
            _encoder_compatible_loaded_row(row.loaded_row),
            state_encoder,
        )
        if example.row_sha256 != row.row_id:
            raise EnhancedBranchContractError("branch encoder changed the row identity")
        next_state_dim, next_action_dim = _validate_encoded_dimensions(
            example,
            state_dim=observed_state_dim,
            action_dim=observed_action_dim,
        )
        if observed_state_dim is None:
            observed_state_dim = next_state_dim
        if observed_action_dim is None:
            observed_action_dim = next_action_dim
        encoded_rows.append(example)
    if observed_state_dim is None or observed_action_dim is None:
        raise EnhancedBranchContractError("branch batch cannot be empty")
    batch_size = len(encoded_rows)
    max_actions = max(int(example.action_features.shape[0]) for example in encoded_rows)
    state_features = torch.stack([example.state_features for example in encoded_rows])
    action_features = torch.zeros((batch_size, max_actions, observed_action_dim), dtype=torch.float32)
    target_mean = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_variance = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    for index, example in enumerate(encoded_rows):
        action_count = int(example.action_features.shape[0])
        action_features[index, :action_count] = example.action_features
        target_mean[index, :action_count] = example.q_mean
        target_variance[index, :action_count] = example.q_variance
        action_mask[index, :action_count] = example.action_mask
    policy_examples: tuple[Any, ...] = ()
    if is_training:
        built_examples = tuple(policy_example_factory(_policy_payload(row)) for row in rows)
        if any(example is None for example in built_examples):
            raise EnhancedBranchContractError("policy example factory returned a missing training example")
        policy_examples = built_examples
    batch = BranchTrainerBatch(
        row_ids=tuple(row.row_id for row in rows),
        shard_ids=tuple(row.shard_id for row in rows),
        state_features=state_features,
        action_features=action_features,
        target_mean=target_mean,
        target_variance=target_variance,
        action_mask=action_mask,
        policy_examples=policy_examples,
        is_training=is_training,
    )
    batch.validate(state_dim=observed_state_dim, action_dim=observed_action_dim)

    return batch, observed_state_dim, observed_action_dim


def build_branch_trainer_batches(
    plan: EnhancedBranchPlan,
    *,
    state_encoder: Callable[[Mapping[str, Any]], Any],
    policy_example_factory: Callable[[BranchPolicyPayload], Any],
    batch_size: int = 32,
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> EnhancedBranchBatchSet:
    """Materialize complete train and held-out batches with exact coverage.

    Training batches are emitted first, followed by held-out batches.  Original
    corpus order is preserved within each split, and no row enters both splits.
    """
    if not isinstance(plan, EnhancedBranchPlan):
        raise TypeError("plan must be an EnhancedBranchPlan")
    if not callable(state_encoder) or not callable(policy_example_factory):
        raise TypeError("state_encoder and policy_example_factory must be callable")
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(expected_state_dim, "expected_state_dim", allow_none=True)
    _validate_positive_int(expected_action_dim, "expected_action_dim", allow_none=True)
    training_rows = plan.training_rows
    heldout_rows = plan.heldout_rows
    if len(training_rows) != FULL_BRANCH_TRAIN_ROWS or len(heldout_rows) != FULL_BRANCH_HELDOUT_ROWS:
        raise EnhancedBranchContractError("branch plan split coverage changed after validation")
    batches: list[BranchTrainerBatch] = []
    observed_state_dim = expected_state_dim
    observed_action_dim = expected_action_dim
    split_groups = ((training_rows, True), (heldout_rows, False))
    for split_rows, is_training in split_groups:
        for offset in range(0, len(split_rows), batch_size):
            batch, observed_state_dim, observed_action_dim = _build_batch(
                split_rows[offset:offset + batch_size],
                is_training=is_training,
                state_encoder=state_encoder,
                policy_example_factory=policy_example_factory,
                state_dim=observed_state_dim,
                action_dim=observed_action_dim,
            )
            batches.append(batch)
    if observed_state_dim is None or observed_action_dim is None:
        raise EnhancedBranchContractError("full branch corpus produced no batches")
    emitted_row_ids = tuple(row_id for batch in batches for row_id in batch.row_ids)
    expected_row_ids = tuple(row.row_id for row in (*training_rows, *heldout_rows))
    if emitted_row_ids != expected_row_ids or len(set(emitted_row_ids)) != FULL_BRANCH_CORPUS_ROWS:
        raise EnhancedBranchContractError("every unique branch row must be emitted exactly once")
    emitted_shard_ids = tuple(shard_id for batch in batches for shard_id in batch.shard_ids)
    unique_emitted_shards = set(emitted_shard_ids)
    if unique_emitted_shards != set(plan.shard_ids) or len(plan.shard_ids) != FULL_BRANCH_CORPUS_SHARDS:
        raise EnhancedBranchContractError("every frozen branch shard must have emitted row coverage")
    training_batches = sum(batch.is_training for batch in batches)
    heldout_batches = len(batches) - training_batches
    receipt = EnhancedBranchConsumptionReceipt(
        schema_version=BRANCH_CONSUMPTION_RECEIPT_SCHEMA_VERSION,
        root_manifest_sha256=plan.corpus.root_manifest_sha256,
        corpus_lineage_id=plan.corpus.corpus_lineage_id,
        ordered_shard_identity_sha256=plan.corpus.ordered_shard_identity_sha256,
        split_policy_sha256=FROZEN_BRANCH_SPLIT_POLICY.sha256,
        unique_branch_rows_consumed=len(set(emitted_row_ids)),
        unique_branch_shards_consumed=len(unique_emitted_shards),
        training_rows_consumed=len(training_rows),
        heldout_rows_consumed=len(heldout_rows),
        policy_examples_built=sum(len(batch.policy_examples) for batch in batches),
        training_batches=training_batches,
        heldout_batches=heldout_batches,
        all_rows_emitted_once=True,
        all_shards_covered=True,
        zero_hidden_information_leakage=True,
        m3_checkpoint_dependency=False,
    )

    return EnhancedBranchBatchSet(
        batches=tuple(batches),
        state_dim=observed_state_dim,
        action_dim=observed_action_dim,
        receipt=receipt,
    )
