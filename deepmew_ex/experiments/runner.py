"""The S0 vertical slice: one pilot, one output owner, no gameplay."""

from __future__ import annotations

import copy
import io
import json
import platform
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from deepmew_ex.data import Decision
from deepmew_ex.experiments.config import ExperimentConfig, PROJECT_ROOT
from deepmew_ex.registry.records import Artifact, Evaluation, Evidence, Family, Lineage, LineageEdge, Model, ModelSpec, Node, Run, Variant, artifact_nodes
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash
from deepmew_ex.registry.provenance import code_identity


def row_manifest(rows: list[Decision]) -> list[dict[str, object]]:
    return [{'row_id': row.row_id, 'game_id': row.game_id, 'trajectory_id': row.trajectory_id, 'split': row.split, 'options_sha256': digest(row.option_features.tobytes()), 'values_sha256': digest(row.value_features.tobytes()), 'selection': row.selection.tolist(), 'min_count': row.min_count, 'max_count': row.max_count, 'legal_count': row.complete_legal_action_count, 'weight': row.actor_loss_weight} for row in rows]


def run_s0(config: ExperimentConfig, run_id: str | None = None, project: Path = PROJECT_ROOT) -> Path:
    config.validate()
    paths = config.paths(project)
    # Capability/scope admission above precedes dataset/model imports, reads and allocation.
    import numpy as np
    import torch
    from deepmew_ex.evaluation.offline import evaluate, load_checkpoint
    from deepmew_ex.experiments.synthetic import SyntheticDataset
    from deepmew_ex.models.mlp import DeepMewEX
    from deepmew_ex.training.bc import admit, check_isolation, encode, set_seed, train_epoch

    identity = run_id or f's0-{uuid.uuid4().hex}'
    resolved = config.resolved(project)
    config_hash = digest(canonical(resolved))
    code = code_identity(project)
    spec_fields = {'architecture': config.architecture, 'hidden_dim': config.hidden_dim, 'option_dim': 304, 'state_dim': 144, 'recurrence': 'none', 'reset': 'not_applicable_snapshot', 'input_schema': config.encoder, 'action_schema': 'complete_legal_options_single_selection', 'vocabulary': 'not_applicable_fixed_features'}
    spec_id = f'spec:sha256:{digest(canonical(spec_fields))}'
    spec = ModelSpec(spec_id, **spec_fields)
    store = RunStore.allocate(paths, config.family, config.variant, identity, specification=asdict(spec))
    started = time.perf_counter()
    run_fields: dict[str, object] = {'run_id': identity, 'family_id': config.family, 'variant_id': config.variant, 'model_spec_id': spec_id, 'model_id': f'model:{identity}', 'config_sha256': config_hash}
    try:
        store.write('config.json', resolved)
        store.write('code.json', code)
        inputs_path = project / 'docs/receipts/implementation-inputs.json'
        if inputs_path.is_file():
            store.write('implementation-inputs.json', json.loads(inputs_path.read_text(encoding='utf-8')))
        from deepmew_ex.registry.legacy import import_legacy
        store.write('legacy-import.json', asdict(import_legacy(project)))
        set_seed(config.seed)
        dataset = SyntheticDataset(config.seed, config.train_decisions, config.validation_decisions)
        raw_train = list(dataset.decisions('train'))
        raw_validation = list(dataset.decisions('validation'))
        check_isolation(raw_train, raw_validation)
        train, train_support = admit(raw_train, 'train')
        validation, validation_support = admit(raw_validation, 'validation')
        data = {'generator': dataset.version, 'seed': config.seed, 'train': row_manifest(raw_train), 'validation': row_manifest(raw_validation), 'holdout': 'not_read', 'source_population': 'synthetic_only', 'visibility': 'only_generated_fixed_features_no_privileged_inputs'}
        data_hash = store.write('dataset.json', data)
        validation_hash = digest(canonical(data['validation']))
        train_hash = digest(canonical(data['train']))
        support = {'train': asdict(train_support), 'validation': asdict(validation_support), 'exclusion_precedence': ['forced', 'zero_actor_weight', 'optional_empty', 'multi_select', 'unsupported_selection']}
        store.write('support.json', support)
        model = DeepMewEX(hidden_dim=config.hidden_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        baseline = evaluate(model, validation).report()
        history: list[dict[str, object]] = []
        consumed: list[str] = []
        best_accuracy = -1.0
        best: dict[str, object] | None = None
        best_step = 0
        final: dict[str, object] = {}
        for step in range(1, config.updates + 1):
            row = train[(step - 1) % len(train)]
            training = train_epoch(model, [row], optimizer, config.gradient_clip)
            consumed.append(row.row_id)
            if step % config.evaluate_every != 0 and step != config.updates:
                continue
            metrics = evaluate(model, validation).report()
            history.append({'update': step, 'last_update_train_metrics': training.report(), 'validation': metrics})
            final = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()), 'torch_rng_state': torch.get_rng_state(), 'architecture': config.architecture, 'hidden_dim': config.hidden_dim, 'seed': config.seed, 'update': step, 'validation': metrics, 'model_spec_id': spec_id, 'run_id': identity, 'config_sha256': config_hash}
            if float(metrics['top1']) > best_accuracy:
                best_accuracy = float(metrics['top1'])
                best_step = step
                best = copy.deepcopy(final)
        if best is None:
            raise ValueError('no eligible validation checkpoint')
        changed = [name for name, value in model.state_dict().items() if not torch.equal(initial[name], value)]
        if not changed:
            raise ValueError('optimizer did not change parameters')
        artifacts: list[Artifact] = []
        for role, checkpoint in [('best', best), ('final', final)]:
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            relative = f'checkpoints/{role}.pt'
            sha = store.write_bytes(relative, buffer.getvalue())
            artifacts.append(Artifact(f'sha256:{sha}', identity, relative, sha, role))
        reloaded, _ = load_checkpoint(store.path / 'checkpoints/best.pt')
        model.load_state_dict(best['model'])
        parity = all(torch.equal(model(*encode(row)), reloaded(*encode(row))) for row in validation)
        if not parity:
            raise ValueError('checkpoint reload logits differ')
        reevaluation = evaluate(reloaded, validation).report()
        if reevaluation != best['validation']:
            raise ValueError('checkpoint reevaluation differs')
        evaluation = Evaluation(f'eval:{identity}:best-reload', artifacts[0].artifact_id, identity, validation_hash, config.evaluator, reevaluation)
        store.write('evaluations/best-reload.json', asdict(evaluation))
        ledger = {'presentations': consumed, 'unique_rows': sorted(set(consumed)), 'unique_games': sorted({row.game_id for row in train if row.row_id in consumed}), 'updates': config.updates, 'holdout_reads': 0, 'dataset_reads': dataset.reads}
        store.write('consumption.json', ledger)
        metrics_report = {'baseline_validation': baseline, 'history': history, 'selected_update': best_step, 'selection_rule': config.checkpoint_selection, 'best': reevaluation, 'fitting_loss': 'actor_weight_times_CE_per_decision; no weight normalization', 'evaluation': 'both unweighted and weight-normalized CE/top1; selection uses unweighted top1', 'support': support, 'gradients_finite_every_update': True, 'changed_parameter_tensors': changed, 'checkpoint_reload_exact': parity, 'gameplay_evaluation': 'not_run'}
        store.write('metrics.json', metrics_report)
        parameter_counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in model.named_children()}
        environment = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads(), 'deterministic_algorithms': True, 'dependencies_sha256': file_hash(project / 'pyproject.toml')}
        environment_hash = store.write('environment.json', environment)
        comparison_ids = {'dataset': data_hash, 'train_split': train_hash, 'validation_split': validation_hash, 'seed': config.seed, 'budget': [config.train_decisions, config.validation_decisions, config.updates], 'checkpoint_selection': config.checkpoint_selection, 'evaluator': config.evaluator, 'source_population': 'synthetic_only', 'legal_action_handling': config.objective, 'device': 'cpu'}
        store.write('comparison-identities.json', comparison_ids)
        model_record = Model(f'model:{identity}', spec_id, (identity,))
        run = Run(**run_fields, status='completed')
        family = Family(config.family, 'Synthetic S0 snapshot option BC; no gameplay claim')
        variant = Variant(config.variant, family.family_id, spec_id, 's0-v1')
        nodes = [Node(identity, 'run'), Node(data_hash, 'dataset'), Node(spec_id, 'model_spec'), Node(evaluation.evaluation_id, 'evaluation'), *artifact_nodes(artifacts)]
        evidence = Evidence('config.json', config_hash, '', 'verified', 'executed synthetic S0 composition')
        edges = [LineageEdge(identity, data_hash, 'uses_dataset', evidence), LineageEdge(artifacts[0].artifact_id, evaluation.evaluation_id, 'evaluated_in', evidence)]
        lineage = Lineage(nodes, edges)
        store.write('lineage.json', {'nodes': [asdict(n) for n in lineage.nodes.values()], 'edges': [asdict(e) for e in edges], 'artifacts': [asdict(a) for a in artifacts]})
        store.write('lineage-receipt.json', lineage_receipt(identity, config, code, config_hash, environment_hash, data_hash, train_hash, validation_hash, parameter_counts, ledger, artifacts, evaluation, time.perf_counter() - started, store.path))
        gates = {'finite_loss_gradients_updates': {'status': 'passed', 'scope': 'all S0 optimizer updates'}, 'checkpoint_parity': {'status': 'passed', 'scope': 'exact logits on all eligible synthetic validation rows'}, 'holdout_isolation': {'status': 'passed', 'scope': 'synthetic game/trajectory/row IDs; holdout reader not called'}, 'E04': {'status': 'not_run', 'scope': 'separate executed test receipt covers permutation fixture'}, 'E05': {'status': 'not_run', 'scope': 'separate test covers fixed input allowlist; upstream raw-source privacy unverified'}, 'E01_D1': {'status': 'not_applicable', 'reason': 'S0 synthetic candidate only'}, 'E02_E03_E06_E09_E23_gameplay': {'status': 'not_applicable', 'reason': 'no engine, temporal, multi-select, TD or gameplay consumer'}}
        store.seal({'schema_version': 1, 'run': asdict(run), 'model': asdict(model_record), 'family': asdict(family), 'variant': asdict(variant), 'artifacts': [asdict(a) for a in artifacts], 'evaluation': asdict(evaluation), 'gates': gates, 'gameplay_evaluation': 'not_run', 'what_this_establishes': 'synthetic BC wiring only'})
        return store.path
    except BaseException as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.perf_counter() - started})
        store.seal({'schema_version': 1, 'run': {**run_fields, 'status': 'failed'}, 'gameplay_evaluation': 'not_run'}, status='failed')
        raise


def lineage_receipt(identity: str, config: ExperimentConfig, code: dict[str, object], config_hash: str, environment_hash: str, data_hash: str, train_hash: str, validation_hash: str, parameters: dict[str, int], ledger: dict[str, object], artifacts: list[Artifact], evaluation: Evaluation, elapsed: float, path: Path) -> dict[str, object]:
    # Complete receipt vocabulary: unknowns are explicit rather than silently omitted.
    from deepmew_ex.registry.receipts import blank_receipt
    receipt = blank_receipt()
    receipt.update({'receipt_version': 1, 'receipt_id': f'receipt:{identity}', 'artifact_id': artifacts[0].artifact_id, 'experiment_id': identity, 'artifact_kind': 'synthetic_bc_checkpoint', 'aliases': ['best'], 'git_commit': code['git_commit'], 'code_version': code['source_manifest_sha256'], 'config_version_or_hash': config_hash, 'dirty_patch_hash': code['dirty_patch_hash'], 'environment_lock_or_container_hash': environment_hash, 'reconstruction_command': 'python -m deepmew_ex.cli experiment run experiments/deepmew_ex/s0.json --run-id <NEW_ID>', 'artifact_location_and_retention': {'path': str(path), 'retention': 'local gitignored outputs; no remote checkpoint storage claimed'}, 'parent_artifacts_and_checkpoints': 'NOT_APPLICABLE random initialization', 'dataset_id': 'synthetic_v1_ragged_v1', 'dataset_hash_or_version': data_hash, 'dataset_manifest_location': 'dataset.json', 'raw_example_count': config.train_decisions + config.validation_decisions, 'unique_episode_or_game_count': 'see dataset.json exact game IDs', 'train_split_definition_and_manifest': {'sha256': train_hash, 'manifest': 'dataset.json#train'}, 'validation_split_definition_and_manifest': {'sha256': validation_hash, 'manifest': 'dataset.json#validation'}, 'test_split_definition_and_manifest': 'NOT_APPLICABLE holdout not read', 'leakage_constraints': 'game, trajectory and row disjoint', 'architecture': config.architecture, 'exact_trainable_parameter_count': sum(parameters.values()), 'parameter_count_by_subsystem': parameters, 'parameter_count_method_and_evidence': 'sum(p.numel() for p in module.parameters() if p.requires_grad)', 'input_schema': config.encoder, 'output_or_action_schema': 'single legal-option index', 'temporal_semantics': 'snapshot only', 'training_recurrence_and_state_propagation': 'none', 'inference_recurrence_and_state_propagation': 'none', 'sequence_length': 'NOT_APPLICABLE snapshot', 'burn_in': 'NOT_APPLICABLE snapshot', 'recurrent_reset_boundary': 'NOT_APPLICABLE snapshot', 'learning_or_decision_clock': 'one optimizer update per eligible decision presentation', 'snapshot_or_sequence_examples': 'snapshot', 'objective_or_loss': 'actor_weight * categorical cross_entropy', 'optimizer': 'AdamW betas=(0.9,0.999), eps=1e-8, amsgrad=False', 'learning_rate_and_scheduler': {'learning_rate': config.learning_rate, 'scheduler': 'none'}, 'material_hyperparameters': config.resolved(), 'random_seeds': [config.seed], 'training_steps_or_updates': config.updates, 'unique_examples_actually_consumed': len(ledger['unique_rows']), 'consumed_example_and_game_manifest': 'consumption.json', 'compute_and_wall_time': {'device': 'cpu', 'seconds': elapsed}, 'checkpoint_hash': artifacts[0].sha256, 'final_trained_artifact_hash': artifacts[1].sha256, 'evaluation_receipt_id_and_location': {'id': evaluation.evaluation_id, 'path': 'evaluations/best-reload.json'}, 'evaluation_protocol_and_code_config': config.evaluator, 'evaluation_dataset_opponents_and_seeds': {'validation_sha256': validation_hash, 'seed': config.seed, 'opponents': 'NOT_APPLICABLE offline'}, 'matched_baseline_artifact_and_receipt': 'NOT_APPLICABLE wiring only; untrained metrics in metrics.json', 'held_out_metrics': evaluation.metrics, 'metric_sample_counts_and_uncertainty': {'rows': evaluation.metrics['rows'], 'uncertainty': evaluation.uncertainty}, 'runtime_transformation_chain': 'NOT_APPLICABLE no deployed derivative', 'export_format': 'pytorch_state_dict_v1; not Dataset V2 export', 'decision': 'DIAGNOSTIC', 'decision_reason': 'bounded synthetic plumbing only', 'decision_timestamp_and_authority': {'timestamp': datetime.now(timezone.utc).isoformat(), 'authority': 'explicit user S0 request'}, 'supersedes_receipt_id': 'NOT_APPLICABLE initial receipt', 'what_this_establishes': 'reused snapshot BC can update/save/reload on synthetic CPU rows', 'what_this_does_not_establish': 'real corpus safety, gameplay, strength, recurrence, critic or D1 reproduction', 'reproduction_status_and_receipt': 'checkpoint reevaluated; metrics.json and evaluations/best-reload.json'})
    for field in ('calibration_metrics', 'arena_or_task_metrics', 'external_competition_or_production_metric', 'quantization', 'pruning', 'compilation', 'conversion', 'distillation', 'deployed_artifact_hash', 'submission_or_production_run_id'):
        receipt[field] = 'NOT_APPLICABLE synthetic offline checkpoint only'
    receipt['field_evidence'] = {'measured_fields': 'config.json, dataset.json, consumption.json, environment.json, metrics.json, checkpoint bytes and sealed manifest digests', 'level': 'VERIFIED_ARTIFACT for local checkpoint; no scientific strength claim', 'UNKNOWN': 'not measured; no inference permitted', 'NOT_APPLICABLE': 'reason included in each sentinel', 'hash_algorithm': 'sha256'}
    return receipt
