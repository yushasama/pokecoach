"""Bounded full-membership training with durable updates and exact row ledgers."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from deepmew_ex.datasets.v2.learning.full_corpus import FullCorpus
from deepmew_ex.datasets.v2.learning.history import LEGACY_INPUT_CONTRACT
from deepmew_ex.datasets.v2.learning.prepared import PreparedCorpus
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence
from deepmew_ex.experiments.config import PROJECT_ROOT, OutputPaths, current_output_bytes
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.receipts import blank_receipt
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash, verify_run
from deepmew_ex.training.bitmap_ledger import EMPTY_CHAIN, LEDGER_FILES, BitmapLedger, ExpectedIndex, expected_rows_from_sequences
from deepmew_ex.training.batching import temporal_batches
from deepmew_ex.training.schedule import DecisionSchedule
from deepmew_ex.training.recovery import TrainingLease


@dataclass(frozen=True)
class FullTrainingConfig:
    corpus: str
    output_root: str = 'outputs'
    device: str = 'cuda'
    seed: int = 42
    hidden: int = 128
    embedding: int = 24
    layers: int = 2
    heads: int = 4
    epochs: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    snapshot_batch: int = 32
    checkpoint_updates: int = 500
    max_wall_seconds: int = 86400
    max_output_bytes: int = 64 * 1024**3
    minimum_free_bytes: int = 10 * 1024**3
    gpu_memory_fraction: float = 0.6
    cpu_threads: int = 2
    temporal_batch_size: int = 1
    batch_leaves: int = 100000
    batch_attention_cells: int = 4000000
    prepared_corpus: str | None = None
    shuffle_train_sequences: bool = False
    loader_workers: int = 0

    def validate(self) -> None:
        if self.prepared_corpus is not None and (not isinstance(self.prepared_corpus, str) or not self.prepared_corpus):
            raise ValueError('invalid prepared corpus path')
        if type(self.shuffle_train_sequences) is not bool:
            raise ValueError('invalid shuffle_train_sequences')
        if self.shuffle_train_sequences and self.prepared_corpus is None:
            raise ValueError('seeded training order requires a sealed prepared corpus')
        if type(self.loader_workers) is not int or not 0 <= self.loader_workers <= 32:
            raise ValueError('invalid loader_workers')
        if self.loader_workers and self.prepared_corpus is None:
            raise ValueError('parallel sequence loading requires a sealed prepared corpus')
        if type(self.heads) is not int or self.heads < 1 or self.device not in ('cpu', 'cuda') or self.hidden % self.heads or self.hidden % 2:
            raise ValueError('invalid device or attention dimensions')
        for field in ('hidden', 'embedding', 'layers', 'heads', 'epochs', 'snapshot_batch', 'checkpoint_updates', 'max_wall_seconds', 'max_output_bytes', 'minimum_free_bytes', 'cpu_threads', 'temporal_batch_size', 'batch_leaves', 'batch_attention_cells'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f'invalid {field}')
        if not 0 < self.gpu_memory_fraction <= 0.8 or not 0 < self.learning_rate < 1 or not 0 < self.gradient_clip or self.weight_decay < 0:
            raise ValueError('invalid resource or optimizer bounds')
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.gradient_clip, self.gpu_memory_fraction)):
            raise ValueError('nonfinite configuration')
        OutputPaths(self.resolve(self.output_root)).validate()

    def resolve(self, path: str) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def all_finite(tensors: list[torch.Tensor]) -> bool:
    """Exact elementwise finiteness of every tensor with one device sync instead of one per tensor."""
    return bool(torch.isfinite(torch.cat([tensor.detach().reshape(-1) for tensor in tensors])).all())


def batches(rows: Iterator[ActorSequence], count: int) -> Iterator[list[ActorSequence]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == count:
            yield batch
            batch = []
    if batch:
        yield batch


def train(config: FullTrainingConfig, run_id: str, resume: Path | None = None) -> Path:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('CUDA was initialized before the required deterministic CuBLAS workspace; restart this training process with CUBLAS_WORKSPACE_CONFIG=:4096:8')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; no silent CPU substitution')
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
        torch.cuda.manual_seed_all(config.seed)
    numerical_policy = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if config.device == 'cuda' else None, 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'matmul_precision': torch.get_float32_matmul_precision(), 'device': torch.cuda.get_device_name() if config.device == 'cuda' else 'cpu'}
    store = RunStore.allocate(OutputPaths(config.resolve(config.output_root)), 'deepmew_ex', 'v2_full_temporal_v1', run_id)
    started, elapsed_before = time.monotonic(), 0.0
    deadline_epoch = time.time() + config.max_wall_seconds
    restoring = resume is not None
    updates, presentations, phase, epoch, cursor = 0, 0, 'raw', 0, 0
    code = code_identity(PROJECT_ROOT)
    config_sha = digest(canonical(asdict(config)))
    original_corpus_sha = file_hash(config.resolve(config.corpus) / 'manifest.json')
    corpus_sha = file_hash(config.resolve(config.prepared_corpus or config.corpus) / 'manifest.json')
    model_config = {key: getattr(config, key) for key in ('hidden', 'embedding', 'layers', 'heads')}
    model = TemporalPolicy(**model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler: DecisionSchedule | None = None
    saved_schedule: dict[str, object] | None = None
    best: dict[str, object] = {'loss': None, 'epoch': None, 'checkpoint': None, 'sha256': None}
    # Exact consumption proof: in-memory per-epoch bitmap plus an append-only
    # binary log (training/bitmap_ledger.py); no per-decision database I/O.
    ledger: BitmapLedger | None = None
    ledger_digest = EMPTY_CHAIN
    progress_path = store.path / 'progress.jsonl'
    last_checkpoint, last_report, in_update = 0, 0.0, False

    def elapsed() -> float:
        return elapsed_before + time.monotonic() - started

    def check_resources() -> None:
        if (store.path / 'STOP').exists():
            raise InterruptedError('owner or usage checkpoint pause requested')
        if elapsed() >= config.max_wall_seconds or time.time() >= deadline_epoch:
            raise TimeoutError('cumulative training/preparation wall budget exhausted')
        if shutil.disk_usage(store.path).free < config.minimum_free_bytes:
            raise RuntimeError('minimum free disk reserve reached')
        if current_output_bytes(store.path) > config.max_output_bytes:
            raise RuntimeError('output byte budget reached')

    def progress(event: str, fields: dict[str, object]) -> None:
        check_resources()
        record = {'event': event, 'elapsed_seconds': elapsed(), 'phase': phase, 'epoch': epoch, 'cursor': cursor, 'updates': updates, 'presentations': presentations, **fields}
        with progress_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
        print(json.dumps(record, allow_nan=False), flush=True)

    def checkpoint(label: str) -> str:
        nonlocal last_checkpoint
        if in_update or restoring:
            raise RuntimeError('mid-update state cannot be published as resumable')
        if ledger is not None:
            ledger.sync()
        checkpoint_value = {'architecture': 'full_prefix_byte_semantic_v1', 'model_config': model_config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if config.device == 'cuda' else [], 'python_rng': random.getstate(), 'numpy_rng': list(np.random.get_state()[:1]) + [np.random.get_state()[1].tolist(), *np.random.get_state()[2:]], 'updates': updates, 'presentations': presentations, 'phase': phase, 'epoch': epoch, 'cursor': cursor, 'elapsed_seconds': elapsed(), 'config_sha256': config_sha, 'code_sha256': code['source_manifest_sha256'], 'corpus_sha256': corpus_sha, 'parser_hashes': corpus.parser_hashes}
        checkpoint_value.update({'training_state_version': 5, 'input_contract': input_contract, 'numerical_policy': numerical_policy, 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'best_validation': dict(best), 'ledger_digest': ledger_digest, 'deadline_epoch': deadline_epoch})
        buffer = io.BytesIO()
        torch.save(checkpoint_value, buffer)
        name = f'checkpoints/{label}-e{epoch}-{phase}-{cursor}-u{updates}.pt'
        sha = store.write_bytes(name, buffer.getvalue())
        store.write(name + '.json', {'sha256': sha, 'updates': updates, 'presentations': presentations, 'phase': phase, 'epoch': epoch, 'cursor': cursor})
        last_checkpoint = updates
        return name

    corpus = PreparedCorpus(config.resolve(config.prepared_corpus)) if config.prepared_corpus is not None else FullCorpus(config.resolve(config.corpus), store.path, progress)
    input_contract = corpus.input_contract if isinstance(corpus, PreparedCorpus) else LEGACY_INPUT_CONTRACT
    sequences = partial(corpus.sequences, workers=config.loader_workers) if isinstance(corpus, PreparedCorpus) else corpus.raw_sequences
    corpus_report = corpus.manifest['source_coverage'] if isinstance(corpus, PreparedCorpus) else corpus.report
    corpus_counts = corpus.manifest['membership']['reasons'] if isinstance(corpus, PreparedCorpus) else corpus.counts
    lease = TrainingLease(store.path)
    parent_lease: TrainingLease | None = None
    try:
        if isinstance(corpus, PreparedCorpus) and corpus.manifest.get('original_corpus_manifest_sha256') != original_corpus_sha:
            raise ValueError('prepared corpus belongs to a different historical corpus')
        store.write('config.json', asdict(config))
        store.write('input-contract.json', {'input_contract': input_contract})
        store.write('code.json', code)
        store.write('environment.json', {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': config.device, 'gpu': torch.cuda.get_device_name() if config.device == 'cuda' else None, 'platform': platform.platform(), 'pid': os.getpid(), 'cpu_threads': config.cpu_threads, 'numerical_policy': numerical_policy})
        store.write('parameters.json', {'total': sum(parameter.numel() for parameter in model.parameters()), 'subsystems': {name: sum(parameter.numel() for parameter in child.parameters()) for name, child in model.named_children()}})
        store.write('authorization.json', {'user_request': 'full-scale temporal agent training; approved 8.09M architecture', 'scope': 'local full-membership training; no paid services or promotion', 'arbitrary_decision_cap': None, 'source_universe': {'corpus_manifest_sha256': corpus_sha, 'original_corpus_manifest_sha256': original_corpus_sha, 'canonical_references': corpus_report['catalog_rows'], 'prepared_temporal': isinstance(corpus, PreparedCorpus)}, 'limits': asdict(config)})
        resume_record: dict[str, object] | None = None
        if resume is not None:
            parent_root = resume.parent.parent
            unsealed_parent = not (parent_root / 'manifest.json').is_file()
            if unsealed_parent:
                parent_lease = TrainingLease(parent_root, require_existing=True)
            else:
                parent_manifest = verify_run(parent_root)
                if parent_manifest['status'] not in ('completed', 'failed'):
                    raise ValueError('resume requires a terminated parent attempt, never an active writer')
            expected = json.loads(resume.with_suffix(resume.suffix + '.json').read_text())['sha256']
            if file_hash(resume) != expected:
                raise ValueError('resume checkpoint hash mismatch')
            # Load on CPU first: AdamW keeps non-capturable step counters on CPU
            # while load_state_dict places moments beside their model parameters.
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if saved.get('numerical_policy') != numerical_policy:
                raise ValueError('resume numerical runtime policy changed')
            if saved.get('input_contract') != input_contract:
                raise ValueError('resume input contract drift')
            if saved['config_sha256'] != config_sha or saved['corpus_sha256'] != corpus_sha or saved['code_sha256'] != code['source_manifest_sha256'] or saved['parser_hashes'] != corpus.parser_hashes:
                raise ValueError('resume config/code/data drift')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if config.device == 'cuda':
                torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng']])
            random.setstate(saved['python_rng'])
            rng = saved['numpy_rng']
            np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
            updates, presentations, phase, epoch, cursor, elapsed_before = (saved[key] for key in ('updates', 'presentations', 'phase', 'epoch', 'cursor', 'elapsed_seconds'))
            if saved.get('training_state_version') != 5 or updates and saved.get('scheduler') is None:
                raise ValueError('resume requires decision-scheduler and committed-ledger state')
            saved_schedule, best = saved['scheduler'], saved['best_validation']
            ledger_digest = saved['ledger_digest']
            deadline_epoch = saved['deadline_epoch']
            if type(deadline_epoch) not in (int, float) or not math.isfinite(deadline_epoch):
                raise ValueError('invalid original training deadline')
            if best['checkpoint'] is not None and file_hash(Path(best['checkpoint'])) != best['sha256']:
                raise ValueError('selected validation checkpoint changed')
            failure_path = resume.parent.parent / 'failure.json'
            if failure_path.exists():
                elapsed_before = max(elapsed_before, json.loads(failure_path.read_text())['elapsed_seconds'])
            if any(not (parent_root / name).is_file() for name in LEDGER_FILES):
                raise ValueError('resume parent has no bitmap coverage ledger')
            # The committed ledger prefix is replayed once the independently
            # rebuilt expected membership is proven identical to the parent's.
            resume_record = {'parent_checkpoint': str(resume.resolve()), 'sha256': expected, 'restored_optimizer_rng': True, 'restored_updates': updates, 'cumulative_elapsed_seconds': elapsed_before, 'unsealed_parent': unsealed_parent, 'committed_ledger_verified': True, 'original_deadline_epoch': deadline_epoch}
        progress('starting', {'training_started': False, 'arbitrary_decision_cap': None})
        if isinstance(corpus, PreparedCorpus):
            store.write('prepared-admission.json', {'path': str(corpus.root), 'manifest_sha256': corpus_sha, 'source_coverage': corpus_report, 'membership': corpus.manifest['membership']})
        else:
            corpus.index_snapshots()
            store.write('snapshot-scope.json', dict(corpus.counts))
            corpus.audit_raw()
            corpus.audit_reserved_snapshots()
        store.write('raw-audit.json', dict(corpus_counts))
        store.write('parser-identity.json', corpus.parser_hashes)
        # A separate full pass fixes eligibility before optimization. Building
        # expected rows from trained batches would conceal a dropped sequence.
        decoded: dict[str, int] = {}
        expected_rows = corpus.expected_rows('train', check_resources) if isinstance(corpus, PreparedCorpus) else expected_rows_from_sequences(sequences('train'), check_resources, decoded)
        index = ExpectedIndex.build(expected_rows, store.path / 'expected-ids.txt.zst')
        if decoded:
            index.sequences = decoded['actor_sequences']
        index.save(store.path)
        membership = index.summary(config.epochs)
        ledger = BitmapLedger(store.path, index, config.epochs)
        if resume_record is not None:
            parent_index = ExpectedIndex.load(resume.parent.parent)
            if not all(np.array_equal(getattr(index, name), getattr(parent_index, name)) for name in ('high', 'low', 'games', 'families')) or (index.game_names, index.family_names) != (parent_index.game_names, parent_index.family_names):
                raise ValueError('resume expected membership differs from the parent run')
            ledger.replay(resume.parent.parent, updates, ledger_digest)
            if ledger.presentations() != presentations:
                raise ValueError('resume ledger does not match checkpoint presentations/updates')
            store.write('resume.json', resume_record)
        store.write('expected-membership.json', membership)
        scheduler = DecisionSchedule(optimizer, membership['expected_presentations'], base_lrs=[config.learning_rate])
        if saved_schedule is not None:
            scheduler.load_state_dict(saved_schedule)
            if scheduler.presentations != presentations:
                raise ValueError('scheduler and consumed-decision clocks disagree')
        if phase != 'raw':
            raise ValueError('snapshot-phase resume is outside temporal training scope')
        restoring = False
        start_epoch, start_phase, start_cursor = epoch, phase, cursor
        for epoch in range(start_epoch, config.epochs):
            for phase in ('raw',):
                # Seeded per-epoch permutation: identical admitted population,
                # deterministic order for exact cursor resume, no source blocks.
                train_rows = sequences('train', order_seed=config.seed * 1_000_003 + epoch) if config.shuffle_train_sequences else sequences('train')
                iterator = temporal_batches(train_rows, config.temporal_batch_size, config.batch_leaves, config.batch_attention_cells)
                skip = start_cursor if epoch == start_epoch and phase == start_phase else 0
                cursor = 0
                for ordinal, group in enumerate(iterator, 1):
                    if ordinal <= skip:
                        cursor = ordinal
                        continue
                    check_resources()
                    model.train()
                    losses, ids = model.batch_losses(group)
                    expected_ids = [prediction.row_id for sequence in group for prediction in sequence.predictions if prediction.loss_eligible]
                    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
                        raise ValueError('loss output omitted or duplicated eligible targets')
                    if len(ids):
                        admitted = ledger.admit(epoch, group)
                        loss = losses.mean()
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                        if not gradients or not all_finite(gradients):
                            raise ValueError('missing or nonfinite active gradients')
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                        in_update = True
                        optimizer.step()
                        if not all_finite(list(model.parameters())):
                            raise ValueError('nonfinite model parameters')
                        updates += 1
                        presentations += len(ids)
                        scheduler.step(presentations)
                        ledger_digest = ledger.record(epoch, updates, group, admitted)
                    cursor = ordinal
                    in_update = False
                    if updates == 1 or updates - last_checkpoint >= config.checkpoint_updates:
                        checkpoint('periodic')
                    if time.monotonic() - last_report > 20:
                        progress('training', {'loss': float(loss.detach()) if len(ids) else None, 'gradient_norm': float(norm) if len(ids) else None, 'learning_rate': optimizer.param_groups[0]['lr'], 'batch_targets': len(ids), 'batch_sequences': len(group), 'full_corpus_complete': False})
                        last_report = time.monotonic()
                phase_checkpoint = checkpoint('phase-end')
                # Every available validation row is visited; no validation cap.
                model.eval()
                loss_sum, count = 0.0, 0
                validation = temporal_batches(sequences('validation'), config.temporal_batch_size, config.batch_leaves, config.batch_attention_cells)
                with torch.no_grad():
                    for group in validation:
                        check_resources()
                        loss, ids = model.batch_losses(group)
                        if not torch.isfinite(loss).all():
                            raise ValueError('nonfinite validation')
                        loss_sum += float(loss.sum())
                        count += len(ids)
                if count == 0:
                    raise ValueError('validation selection requires eligible validation decisions')
                if best['loss'] is None or loss_sum / count < best['loss']:
                    best = {'loss': loss_sum / count, 'epoch': epoch, 'checkpoint': str(store.path / phase_checkpoint), 'sha256': file_hash(store.path / phase_checkpoint)}
                store.write(f'evaluations/epoch-{epoch}-{phase}.json', {'checkpoint_sha256': file_hash(store.path / phase_checkpoint), 'rows': count, 'loss_sum': loss_sum, 'loss': loss_sum / count, 'selection': 'minimum full-validation decision-weighted NLL; earliest epoch breaks ties', 'gameplay': 'not measured'})
                progress('phase_completed', {'validation_rows': count, 'validation_loss': loss_sum / count if count else None})
        selected_source = Path(best['checkpoint'])
        if selected_source.parent.resolve() != (store.path / 'checkpoints').resolve():
            check_resources()
            if file_hash(selected_source) != best['sha256']:
                raise ValueError('inherited selected checkpoint identity changed')
            selected_name = 'checkpoints/inherited-selected.pt'
            copied_sha = store.write_bytes(selected_name, selected_source.read_bytes())
            if copied_sha != best['sha256']:
                raise ValueError('inherited selected checkpoint copy identity changed')
            store.write_bytes(selected_name + '.json', selected_source.with_suffix('.pt.json').read_bytes())
            store.write('selected-origin.json', {'path': str(selected_source), 'sha256': copied_sha, 'relation': 'byte-identical selected checkpoint from prior attempt'})
            best = {**best, 'checkpoint': str(store.path / selected_name)}
            check_resources()
        final = checkpoint('final')
        final_sha = file_hash(store.path / final)
        store.write('selection.json', {'rule': 'minimum full-validation decision-weighted NLL; earliest epoch breaks ties', 'selected': best, 'last_training_checkpoint': final, 'last_training_checkpoint_sha256': final_sha, 'promotion': False})
        reloaded = TemporalPolicy(**model_config).to(config.device)
        reloaded.load_state_dict(torch.load(store.path / final, map_location=config.device, weights_only=True)['model'])
        del model
        model = reloaded.eval()
        for population in ('raw',):
            validation = temporal_batches(sequences('validation'), config.temporal_batch_size, config.batch_leaves, config.batch_attention_cells)
            total, count = 0.0, 0
            with torch.no_grad():
                for group in validation:
                    check_resources()
                    loss, ids = model.batch_losses(group)
                    if not torch.isfinite(loss).all():
                        raise ValueError('nonfinite final reload validation')
                    total += float(loss.sum())
                    count += len(ids)
            store.write(f'evaluations/final-{population}.json', {'checkpoint_sha256': final_sha, 'rows': count, 'loss_sum': total, 'loss': total / count if count else None, 'reloaded_from_checkpoint': True, 'gameplay': 'not measured'})
        selected_path = Path(best['checkpoint'])
        if file_hash(selected_path) != best['sha256']:
            raise ValueError('selected checkpoint identity changed before reload')
        model.load_state_dict(torch.load(selected_path, map_location=config.device, weights_only=True)['model'])
        selected_total, selected_count = 0.0, 0
        with torch.no_grad():
            for group in temporal_batches(sequences('validation'), config.temporal_batch_size, config.batch_leaves, config.batch_attention_cells):
                check_resources()
                values, identities = model.batch_losses(group)
                if not torch.isfinite(values).all():
                    raise ValueError('nonfinite selected-checkpoint validation')
                selected_total += float(values.sum())
                selected_count += len(identities)
        if not selected_count or not math.isclose(selected_total / selected_count, best['loss'], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('reloaded selected checkpoint does not reproduce its validation loss')
        store.write('evaluations/selected-reload.json', {'checkpoint_sha256': best['sha256'], 'rows': selected_count, 'loss': selected_total / selected_count, 'reloaded_from_checkpoint': True, 'gameplay': 'not measured'})
        coverage_proof = ledger.verify_complete()
        receipt = blank_receipt()
        receipt.update({'receipt_version': 1, 'receipt_id': f'receipt:{run_id}', 'artifact_id': f'sha256:{file_hash(store.path / final)}', 'experiment_id': run_id, 'artifact_kind': 'full_prefix_temporal_bc', 'git_commit': code['git_commit'], 'code_version': code, 'config_version_or_hash': config_sha, 'dirty_patch_hash': code['dirty_patch_hash'], 'dataset_hash_or_version': corpus_sha, 'dataset_manifest_location': str(corpus.root / 'manifest.json'), 'exact_trainable_parameter_count': sum(parameter.numel() for parameter in model.parameters()), 'input_schema': 'all UTF-8 semantic path/value bytes and numeric channel', 'temporal_semantics': 'all same-actor request/action events with causal attention; post-state inputs only at observed actor request', 'training_recurrence_and_state_propagation': 'complete actor episode prefix, gradients through entire sequence, reset at each actor/game boundary', 'sequence_length': 'variable complete actor episode; no truncation', 'objective_or_loss': 'exact unordered subset likelihood and ordered autoregressive likelihood; forced context retained', 'optimizer': 'AdamW', 'random_seeds': [config.seed], 'training_steps_or_updates': updates, 'unique_examples_actually_consumed': sum(count for _, _, count in ledger.family_counts()), 'consumed_example_and_game_manifest': 'ledger.bin + expected-index.npz + expected-ids.txt.zst (training/bitmap_ledger.py)', 'coverage_proof': coverage_proof, 'compute_and_wall_time': {'seconds': elapsed(), 'device': config.device}, 'checkpoint_hash': file_hash(store.path / final), 'final_trained_artifact_hash': file_hash(store.path / final), 'decision': 'UNKNOWN; no gameplay promotion decision', 'what_this_establishes': 'actual updates and declared source coverage, subject to coverage receipt', 'what_this_does_not_establish': 'useful history, gameplay strength, unrecorded simulator events, validity of quarantined archives', 'field_evidence': {'known': ['config.json', 'code.json', 'parameters.json', *LEDGER_FILES, 'evaluations/'], 'unknown': 'Unmeasured fields explicitly remain UNKNOWN.'}})
        store.write('lineage-receipt.json', receipt)
        store.write('coverage.json', {'counts': dict(corpus_counts), 'expected_membership': membership, 'consumed': ledger.family_counts(), 'arbitrary_decision_cap': None, 'quarantined_original_sources': corpus_report.get('raw_quarantine_reasons', {}), 'source_coverage': corpus_report if isinstance(corpus, PreparedCorpus) else None, 'all_historical_sources_validated': False, 'scope': 'all admitted prepared temporal targets with independently enumerated consumption' if isinstance(corpus, PreparedCorpus) else 'eligible raw temporal targets only; snapshot references guard splits but are never trained'})
        ledger.close()
        corpus.close()
        store.seal({'kind': 'v2_full_temporal_training', 'updates': updates, 'checkpoint': final, 'promotion': False})
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, Exception)):
            try:
                if not in_update:
                    checkpoint('interrupted')
            except Exception:
                pass
            if ledger is not None:
                ledger.close()
            corpus.close()
            store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'updates': updates, 'presentations': presentations, 'elapsed_seconds': elapsed(), 'full_corpus_complete': False, 'interrupted_mid_update': in_update, 'recovery': 'Resume only a hash-verified fully committed checkpoint; incomplete optimizer effects are not published as resumable.'})
            store.seal({'kind': 'v2_full_temporal_training', 'updates': updates}, 'failed')
        raise
    finally:
        lease.close()
        if parent_lease is not None:
            parent_lease.close()
    return store.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = FullTrainingConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    print(train(config, args.run_id, args.resume), flush=True)


if __name__ == '__main__':
    main()
