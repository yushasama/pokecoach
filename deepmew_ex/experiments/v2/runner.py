"""Execute a real-data BC condition using the shared scorer, trainer and artifact store."""

from __future__ import annotations

import copy
import io
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import orjson
import torch

from deepmew_ex.datasets.v2.learning.cache import CachedDataset
from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.experiments.v2.config import V2ExperimentConfig
from deepmew_ex.models.semantic import SemanticBC
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.receipts import blank_receipt
from deepmew_ex.registry.records import Artifact, Evaluation, Family, Model, Run, SemanticModelSpec, Variant
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash
from deepmew_ex.training.bc import set_seed
from deepmew_ex.training.semantic import evaluate_semantic, make_batch, train_batch


def tensor_identity(model: torch.nn.Module) -> str:
    import hashlib
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def load_semantic_checkpoint(path: Path, expected_sha256: str | None = None) -> tuple[SemanticBC, dict[str, object]]:
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError('checkpoint artifact identity mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != 'semantic_shared_option_v1':
        raise ValueError('unsupported semantic checkpoint architecture')
    model = SemanticBC(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint


def run_v2(config: V2ExperimentConfig, run_id: str, progress: Callable[[int, int, float], None] | None = None) -> Path:
    config.validate()
    resolved = config.resolved()
    dataset = CachedDataset(Path(resolved['cache']))
    train, validation = dataset.examples['train'], dataset.examples['validation']
    if not train or not validation or config.presentations % len(train):
        raise ValueError('positive splits and whole training presentations required')
    if {row['game_id'] for row in train} & {row['game_id'] for row in validation} or {row['signature'] for row in train} & {row['signature'] for row in validation}:
        raise ValueError('semantic data split contamination')
    model_config = {'paths': len(dataset.vocabulary.paths) + 2, 'values': len(dataset.vocabulary.values) + 2, 'hidden': config.hidden_dim, 'embedding': config.embedding_dim, 'history': config.history}
    vocabulary_sha = file_hash(dataset.path / 'vocabulary.json')
    spec_fields = {'architecture': config.architecture, 'hidden_dim': config.hidden_dim, 'option_dim': config.hidden_dim, 'state_dim': config.hidden_dim if config.history == 'snapshot' else config.hidden_dim * 2, 'recurrence': 'none' if config.history == 'snapshot' else f'GRU{config.hidden_dim}_window{dataset.window}', 'reset': 'h0_zero_each_prediction; actor/game-local eight-request input window', 'input_schema': config.encoder, 'action_schema': 'required_single_selection_with_declared_live_fallback', 'vocabulary': vocabulary_sha}
    spec_fields.update({'reset': 'not_applicable_snapshot' if config.history == 'snapshot' else f'h0_zero_each_prediction; actor/game-local {dataset.window}-request input window', 'semantic_embedding_dim': config.embedding_dim, 'history_window': 0 if config.history == 'snapshot' else dataset.window, 'semantic_leaf_limit': dataset.vocabulary.max_leaves})
    spec = SemanticModelSpec('spec:sha256:' + digest(canonical(spec_fields)), **spec_fields)
    store = RunStore.allocate(config.paths(), config.family, config.variant, run_id, specification=asdict(spec))
    started = time.perf_counter()
    config_hash = digest(canonical(resolved))
    run_fields = {'run_id': run_id, 'family_id': config.family, 'variant_id': config.variant, 'model_spec_id': spec.model_spec_id, 'model_id': f'model:{run_id}', 'config_sha256': config_hash}
    def check_time() -> None:
        if time.perf_counter() - started > config.max_wall_seconds:
            raise TimeoutError('frozen V2 training wall budget exceeded')
    try:
        store.write('config.json', resolved)
        code = code_identity(PROJECT_ROOT)
        store.write('code.json', code)
        cache_sha = file_hash(dataset.path / 'manifest.json')
        store.write('dataset.json', {'cache': str(dataset.path.resolve()), 'cache_manifest_sha256': cache_sha, 'vocabulary_sha256': vocabulary_sha, 'split_hashes': {split: digest(canonical(rows)) for split, rows in dataset.examples.items()}, 'train_rows': len(train), 'validation_rows': len(validation), 'holdout_training_reads': 0, 'inputs': 'semantic path/value/numeric leaves; current inputs exclude current chosen action and outcome', 'history': 'own request public snapshots and past completed actions only'})
        set_seed(config.seed)
        model = SemanticBC(**model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        initial_hash = tensor_identity(model)
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        counts = {name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad) for name, module in model.named_children()}
        store.write('parameters.json', {'total': sum(counts.values()), 'by_subsystem': counts, 'initial_tensor_sha256': initial_hash})
        baseline = evaluate_semantic(model, dataset, validation, config.seed, config.batch_size, check_time)
        best_loss, best, final = float('inf'), None, None
        history, consumed, gradient_norms = [], [], []
        rng = np.random.default_rng(config.seed)
        order = np.concatenate([rng.permutation(len(train)) for _ in range(config.presentations // len(train))])
        updates = config.presentations // config.batch_size
        for update in range(1, updates + 1):
            check_time()
            indices = order[(update - 1) * config.batch_size:update * config.batch_size]
            rows = [train[int(index)] for index in indices]
            batch = make_batch(dataset, rows, config.history, config.seed)
            loss, norm = train_batch(model, batch, optimizer, config.gradient_clip)
            consumed.extend(row['row_id'] for row in rows)
            gradient_norms.append(norm)
            if update % config.evaluate_every and update != updates:
                continue
            evaluated = evaluate_semantic(model, dataset, validation, config.seed, config.batch_size, check_time)
            metrics = {key: value for key, value in evaluated.items() if key != 'predictions'}
            history.append({'update': update, 'presentations': len(consumed), 'last_batch_loss': loss, 'gradient_norm': norm, 'validation': metrics})
            final = {'architecture': config.architecture, 'model_config': model_config, 'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()), 'torch_rng_state': torch.get_rng_state(), 'cache_manifest_sha256': cache_sha, 'vocabulary_sha256': vocabulary_sha, 'history_window': dataset.window, 'seed': config.seed, 'update': update, 'validation': metrics, 'model_spec_id': spec.model_spec_id, 'run_id': run_id}
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best = copy.deepcopy(final)
            if progress:
                progress(update, updates, metrics['loss'])
        if best is None or final is None or tensor_identity(model) == initial_hash:
            raise ValueError('no updated semantic model/checkpoint')
        artifacts = []
        for role, checkpoint in (('best', best), ('final', final)):
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            relative = f'checkpoints/{role}.pt'
            sha = store.write_bytes(relative, buffer.getvalue())
            artifacts.append(Artifact(f'sha256:{sha}', run_id, relative, sha, role))
        model.load_state_dict(best['model'])
        reloaded, _ = load_semantic_checkpoint(store.path / artifacts[0].path, artifacts[0].sha256)
        probe = make_batch(dataset, validation[:config.batch_size], config.history, config.seed)
        with torch.no_grad():
            if not torch.equal(probe.logits(model), probe.logits(reloaded)):
                raise ValueError('semantic checkpoint reload logits differ')
        best_eval = evaluate_semantic(reloaded, dataset, validation, config.seed, config.batch_size, check_time)
        if {key: value for key, value in best_eval.items() if key != 'predictions'} != best['validation']:
            raise ValueError('semantic checkpoint validation parity failed')
        changed = [name for name, value in final['model'].items() if not torch.equal(initial[name], value)]
        evaluation = Evaluation(f'eval:{run_id}:reload', artifacts[0].artifact_id, run_id, digest(canonical(validation)), config.evaluator, best['validation'], uncertainty='single training seed; game-grouped descriptive metrics, no strength claim')
        store.write('evaluations/best-reload.json', {'evaluation': asdict(evaluation), 'predictions': best_eval['predictions']})
        store.write('metrics.json', {'initial_validation': {key: value for key, value in baseline.items() if key != 'predictions'}, 'history': history, 'best': best['validation'], 'selected_update': best['update'], 'final': final['validation'], 'gradients_finite_every_update': True, 'gradient_norm_range': [min(gradient_norms), max(gradient_norms)], 'changed_parameter_tensors': changed, 'checkpoint_reload_exact': True, 'gameplay_evaluation': 'not_run'})
        consumption = {'presentations': consumed, 'presentation_count': len(consumed), 'unique_rows': len(set(consumed)), 'unique_games': sorted({row['game_id'] for row in train}), 'epochs': config.presentations // len(train), 'optimizer_updates': updates, 'decisions_per_update': config.batch_size, 'holdout_fitting_or_selection_reads': 0, 'recurrent_context': 'all prior forced/multi-select requests inside bounded windows; none is silently made a single-selection target'}
        store.write('consumption.json', consumption)
        used_history = config.history in ('ordered', 'shuffled')
        store.write('history-consumption.json', {'treatment': config.history, 'prior_request_presentations': sum(len(row['history_records']) for row in train) * consumption['epochs'] if used_history else 0, 'unique_prior_request_records': len({(row['cache_file'], index) for row in train for index in row['history_records']}) if used_history else 0, 'window_limit': dataset.window, 'current_target_in_actor_inputs': False, 'inactive_seat_only_log_snapshots_consumed': 0})
        environment = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': 1, 'deterministic_algorithms': True, 'paid_compute_usd': 0}
        store.write('environment.json', environment)
        elapsed = time.perf_counter() - started
        receipt = training_receipt(config, run_id, code, cache_sha, dataset, artifacts, evaluation, counts, elapsed, store.path, environment)
        store.write('lineage-receipt.json', receipt)
        store.write('lineage.json', {'nodes': [{'node_id': run_id, 'kind': 'run'}, {'node_id': f'sha256:{cache_sha}', 'kind': 'dataset'}, *[{'node_id': artifact.artifact_id, 'kind': 'artifact'} for artifact in artifacts], {'node_id': evaluation.evaluation_id, 'kind': 'evaluation'}], 'edges': [{'source': run_id, 'target': f'sha256:{cache_sha}', 'relation': 'uses_dataset'}, {'source': artifacts[0].artifact_id, 'target': evaluation.evaluation_id, 'relation': 'evaluated_in'}]})
        store.seal({'kind': 'v2_bc_training', 'run': asdict(Run(**run_fields, status='completed')), 'model': asdict(Model(run_fields['model_id'], spec.model_spec_id, (run_id,))), 'family': asdict(Family(config.family, 'DeepMew EX research models')), 'variant': asdict(Variant(config.variant, config.family, spec.model_spec_id, 'v2-bc-20261025-v1')), 'artifacts': [asdict(artifact) for artifact in artifacts], 'evaluation': asdict(evaluation), 'gameplay_evaluation': 'not_run'})
    except Exception as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.perf_counter() - started, 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_bc_training', 'run': {**run_fields, 'status': 'failed'}}, 'failed')
        raise
    return store.path


def training_receipt(config: V2ExperimentConfig, run_id: str, code: dict[str, object], cache_sha: str, dataset: CachedDataset, artifacts: list[Artifact], evaluation: Evaluation, counts: dict[str, int], elapsed: float, path: Path, environment: dict[str, object]) -> dict[str, object]:
    receipt = blank_receipt()
    for key in ('calibration_metrics', 'arena_or_task_metrics', 'external_competition_or_production_metric', 'quantization', 'pruning', 'compilation', 'conversion', 'distillation', 'deployed_artifact_hash', 'submission_or_production_run_id', 'runtime_transformation_chain'):
        receipt[key] = 'NOT_APPLICABLE; offline checkpoint, league evaluation has a separate receipt'
    receipt.update({'receipt_version': 1, 'receipt_id': f'receipt:{run_id}', 'artifact_id': artifacts[0].artifact_id, 'experiment_id': run_id, 'artifact_kind': 'semantic_v2_bc_checkpoint', 'aliases': ['best'], 'git_commit': code['git_commit'], 'code_version': code, 'config_version_or_hash': digest(canonical(config.resolved())), 'dirty_patch_hash': code['dirty_patch_hash'], 'environment_lock_or_container_hash': file_hash(path / 'environment.json'), 'reconstruction_command': f'python -m deepmew_ex.cli experiment run experiments/deepmew_ex/v2-{config.history}.json --run-id <NEW_ID>', 'artifact_location_and_retention': {'path': str(path.resolve()), 'retention': 'immutable local gitignored run; checkpoint bytes retained'}, 'parent_artifacts_and_checkpoints': 'NOT_APPLICABLE; fresh seed42 initialization', 'dataset_id': f'sha256:{cache_sha}', 'dataset_hash_or_version': cache_sha, 'dataset_manifest_location': str(dataset.path / 'manifest.json'), 'raw_example_count': {key: len(rows) for key, rows in dataset.examples.items()}, 'unique_episode_or_game_count': {key: len({row['game_id'] for row in rows}) for key, rows in dataset.examples.items()}, 'train_split_definition_and_manifest': digest(canonical(dataset.examples['train'])), 'validation_split_definition_and_manifest': digest(canonical(dataset.examples['validation'])), 'test_split_definition_and_manifest': 'NOT_APPLICABLE; reserved test/holdout not read for training or checkpoint selection', 'leakage_constraints': 'native game, full game content and current semantic signature admission; private channels removed; current targets kept separate', 'architecture': config.architecture, 'exact_trainable_parameter_count': sum(counts.values()), 'parameter_count_by_subsystem': counts, 'parameter_count_method_and_evidence': 'sum(p.numel() for p in model.named_children() if p.requires_grad); parameters.json', 'input_schema': config.encoder, 'output_or_action_schema': 'required single legal-option index; unsupported live prompts use frozen declared fallback', 'temporal_semantics': config.history, 'training_recurrence_and_state_propagation': 'none' if config.history == 'snapshot' else f'GRU over up to {dataset.window} same-actor prior requests/actions/log snapshots; gradients through entire window', 'inference_recurrence_and_state_propagation': 'same reconstruction from bounded game/seat-local history; no hidden-state carry outside window', 'sequence_length': 1 if config.history == 'snapshot' else dataset.window, 'burn_in': 'NOT_APPLICABLE; full window has gradients, h0 is zero', 'recurrent_reset_boundary': 'each prediction starts h0=0; game reset clears actor history', 'learning_or_decision_clock': f'{config.batch_size} target-bearing decisions per optimizer update; context requests do not add loss/updates', 'snapshot_or_sequence_examples': config.history, 'objective_or_loss': 'unweighted mean categorical cross-entropy over supported demonstrations, no outcome filtering/weights', 'optimizer': 'AdamW betas=(0.9,0.999), eps=1e-8', 'learning_rate_and_scheduler': {'learning_rate': config.learning_rate, 'scheduler': 'none'}, 'material_hyperparameters': asdict(config), 'random_seeds': [config.seed], 'training_steps_or_updates': config.presentations // config.batch_size, 'unique_examples_actually_consumed': len(dataset.examples['train']), 'consumed_example_and_game_manifest': 'consumption.json; exact repeated row presentations and game membership', 'compute_and_wall_time': {'seconds': elapsed, 'environment': environment}, 'checkpoint_hash': artifacts[0].sha256, 'final_trained_artifact_hash': artifacts[1].sha256, 'evaluation_receipt_id_and_location': {'id': evaluation.evaluation_id, 'path': 'evaluations/best-reload.json'}, 'evaluation_protocol_and_code_config': config.evaluator, 'evaluation_dataset_opponents_and_seeds': {'validation_sha256': evaluation.split_sha256, 'opponents': 'NOT_APPLICABLE offline', 'seed': config.seed}, 'matched_baseline_artifact_and_receipt': 'UNKNOWN; sibling conditions linked by final comparison receipt', 'held_out_metrics': evaluation.metrics, 'metric_sample_counts_and_uncertainty': {'validation_rows': evaluation.metrics['rows'], 'validation_games': evaluation.metrics['games'], 'scope': 'single-seed exploratory diagnostic; correlated decisions are not independent games'}, 'export_format': 'pytorch state_dict with semantic model config/vocabulary/cache identity', 'decision': 'DIAGNOSTIC', 'decision_reason': 'first approved real V2 retraining comparison; no champion/model-strength promotion', 'decision_timestamp_and_authority': {'timestamp': datetime.now(timezone.utc).isoformat(), 'authority': 'explicit user approval on 2026-10-25'}, 'supersedes_receipt_id': 'NOT_APPLICABLE; initial receipt for this new artifact', 'what_this_establishes': 'actual semantic/history consumption, finite BC updates and exact checkpoint reload on admitted real replays', 'what_this_does_not_establish': 'full-corpus utilization, useful memory, capacity saturation, gameplay strength or statistical confirmation', 'field_evidence': {'level': 'VERIFIED_ARTIFACT', 'sources': ['config.json', 'dataset.json', 'parameters.json', 'consumption.json', 'metrics.json', 'environment.json', 'checkpoint bytes'], 'missing_comparison': 'sibling artifact IDs become available only in a separate append-only comparison receipt', 'not_applicable': 'each sentinel states scope/reason'}, 'reproduction_status_and_receipt': 'best checkpoint reloaded and independently rescored on complete frozen validation membership'})
    return receipt
