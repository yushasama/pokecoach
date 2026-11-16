"""Frozen checkpoint routing and unpaired, seat-balanced native league games."""

from __future__ import annotations

import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import orjson
import torch

from deepmew_ex.datasets.v2.learning.semantic import Vocabulary
from deepmew_ex.evaluation.league.config import LeagueConfig, OpponentBinding, population
from deepmew_ex.evaluation.league.policies import IndependentSeat, ReferenceSeatPolicy, SeatPolicy, SemanticLivePolicy, legacy_checkpoint_policy, reference_directory
from deepmew_ex.evaluation.league.temporal import ActorNotifyingSeat, FullPrefixLivePolicy, load_full_prefix_checkpoint
from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.experiments.v2.runner import load_semantic_checkpoint
from deepmew_ex.experiments.v2.comparison import validate_conditions
from deepmew_ex.league import bind_legacy_source
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash, verify_run
from deepmew_ex.registry.receipts import blank_receipt
from deepmew_ex.training.bc import set_seed


def read_deck(path: Path) -> list[int]:
    cards = [int(item) for item in path.read_text().split()]
    if len(cards) != 60:
        raise ValueError('frozen deck must have sixty cards')
    return cards


def wilson(wins: int, count: int) -> list[float]:
    if count == 0:
        raise ValueError('empty matchup')
    z = 1.959963984540054
    probability = wins / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    half = z * ((probability * (1 - probability) / count + z * z / (4 * count * count)) ** .5) / denominator
    return [max(0., center - half), min(1., center + half)]


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[row['participant'], row['opponent']].append(row)
    matchups = []
    by_participant: dict[str, list[float]] = defaultdict(list)
    for (participant, opponent), rows in sorted(groups.items()):
        wins = sum(row['outcome'] == 'win' for row in rows)
        draws = sum(row['outcome'] == 'draw' for row in rows)
        score = (wins + draws / 2) / len(rows)
        matchups.append({'participant': participant, 'opponent': opponent, 'games': len(rows), 'wins': wins, 'draws': draws, 'losses': len(rows) - wins - draws, 'win_rate': wins / len(rows), 'score_rate': score, 'win_rate_wilson95': wilson(wins, len(rows))})
        by_participant[participant].append(score)
    aggregate = {}
    rng = np.random.default_rng(42)
    for participant, values in sorted(by_participant.items()):
        scores = np.array(values)
        boot = scores[rng.integers(0, len(scores), size=(10000, len(scores)))].mean(axis=1)
        aggregate[participant] = {'equal_opponent_score': float(scores.mean()), 'opponents': len(scores), 'opponent_cluster_bootstrap95': np.quantile(boot, [.025, .975]).tolist(), 'uncertainty_scope': 'exploratory across frozen opponent strata, one training seed, unpaired engine chance'}
    return {'matchups': matchups, 'aggregate': aggregate}


def run_league(config: LeagueConfig, progress: Callable[[int, int], None] | None = None) -> Path:
    config.validate()
    bindings = population(config)  # Hash-check before any checkpoint deserialization.
    d1 = next(row for row in bindings if row.identity == 'd1')
    own_deck = read_deck(d1.deck)
    trained = []
    comparison = None if config.canary or not config.runs else validate_conditions(config.runs)
    for path in config.runs:
        manifest = verify_run(path)
        if manifest['status'] != 'completed' or manifest.get('kind') != 'v2_bc_training':
            raise ValueError('league requires a completed verified V2 training run')
        artifact = next(item for item in manifest['artifacts'] if item['role'] == 'best')
        trained.append((path, manifest, artifact))
    store = RunStore.allocate(config.paths, 'deepmew_ex', 'v2_league_canary' if config.canary else 'v2_league', config.run_id)
    (store.path / 'games').mkdir()
    started = time.perf_counter()
    results = []
    def check_time(observation: object = None) -> None:
        if time.perf_counter() - started > config.max_wall_seconds:
            raise TimeoutError('frozen league wall budget exceeded')
    try:
        sys.dont_write_bytecode = True
        set_seed(42)
        resolved_config = {'training_runs': [str(path.resolve()) for path in config.runs], 'registry': str(config.registry.resolve()), 'registry_sha256': file_hash(config.registry), 'bindings': [asdict(row) | {'checkpoint': str(row.checkpoint) if row.checkpoint else None, 'deck': str(row.deck)} for row in bindings], 'policy_rng_seeds': config.seeds, 'repetitions': config.repetitions, 'seat_schedule': [0, 1], 'max_selections': config.max_selections, 'max_wall_seconds': config.max_wall_seconds, 'max_decision_seconds': config.max_decision_seconds, 'engine_rng': 'unpaired; no seed argument exposed by consumed native battle_start interface', 'policy_input': 'common actor-visible projection and conservative public logs; simulator buffer never reaches any policy', 'fallback': 'preserved greedy policy for unsupported or forced native inputs; counted separately', 'temporal_checkpoint': str(config.temporal_checkpoint.resolve()) if config.temporal_checkpoint else None, 'temporal_checkpoint_sha256': config.temporal_checkpoint_sha256, 'temporal_device': config.temporal_device, 'authority': 'explicit user approval for free local temporal-checkpoint evaluation 2026-11-07', 'promotion': False}
        store.write('config.json', resolved_config)
        evaluation_code = code_identity(PROJECT_ROOT)
        store.write('code.json', evaluation_code)
        if comparison is not None:
            store.write('offline-comparison.json', comparison)
        store.write('purpose.json', {'canary': config.canary, 'opponent_ids': config.opponent_ids, 'scope': 'routing/safety canary only' if config.canary else 'frozen exploratory retraining comparison', 'reference_input_contract': 'common actor projection; explicit absent deck/looking keys; restricted-zone option requests use shared greedy fallback; unexpected notebook execution errors stop the run'})
        bind_legacy_source(PROJECT_ROOT)
        from deepmew.real_meta_arena.runner import run_real_meta_game
        from deepmew.rl1_teacher_bc.teachers import make_teacher
        from run_r9_r4f_retrain import load_r4f_model
        legacy_models = {row.identity: load_r4f_model(row.checkpoint) for row in bindings if row.checkpoint is not None}
        for model, _ in legacy_models.values():
            model.requires_grad_(False)
        reference_paths = {}
        for row in bindings:
            if row.reference_teacher is None:
                continue
            directory = store.path / 'reference-runtime' / row.identity
            directory.mkdir(parents=True)
            store.write_bytes(f'reference-runtime/{row.identity}/deck.csv', row.deck.read_bytes())
            reference_paths[row.identity] = directory
        def historical(row: OpponentBinding) -> SeatPolicy:
            if row.checkpoint is not None:
                model, metadata = legacy_models[row.identity]
                return legacy_checkpoint_policy(row.label, row.checkpoint, model, metadata, read_deck(row.deck), row.feature_policy_id)
            with reference_directory(reference_paths[row.identity]):
                teacher = make_teacher(row.reference_teacher)
            if type(teacher).__name__ != 'NotebookTeacherAgent' or teacher._module is None:
                raise ValueError('reference notebook unavailable; proxy substitution rejected')
            if Counter(teacher.deck) != Counter(read_deck(row.deck)):
                raise ValueError('reference policy/deck binding mismatch')
            return ReferenceSeatPolicy(teacher, reference_paths[row.identity])
        participants: dict[str, Callable[[int], SeatPolicy]] = {'D1_anchor': lambda seed: historical(d1)}
        candidate_evidence = {}
        for path, manifest, artifact in trained:
            model, checkpoint = load_semantic_checkpoint(path / artifact['path'], artifact['sha256'])
            model.requires_grad_(False)
            cache = Path(orjson.loads((path / 'config.json').read_bytes())['cache'])
            if file_hash(cache / 'vocabulary.json') != checkpoint['vocabulary_sha256'] or file_hash(cache / 'manifest.json') != checkpoint['cache_manifest_sha256']:
                raise ValueError('live vocabulary/cache binding mismatch')
            vocab = Vocabulary(**orjson.loads((cache / 'vocabulary.json').read_bytes()))
            label, window = model.history, checkpoint['history_window']
            if label in participants:
                raise ValueError('duplicate candidate history treatment')
            participants[label] = lambda seed, m=model, v=vocab, w=window, name=label: SemanticLivePolicy(name, m, v, w, seed)
            candidate_evidence[label] = {'run_manifest_sha256': file_hash(path / 'manifest.json'), 'checkpoint_sha256': artifact['sha256'], 'selected_update': checkpoint['update'], 'parameters': sum(parameter.numel() for parameter in model.parameters())}
        if config.temporal_checkpoint is not None and config.temporal_checkpoint_sha256 is not None:
            temporal_model, temporal_evidence = load_full_prefix_checkpoint(config.temporal_checkpoint, config.temporal_checkpoint_sha256, config.temporal_device)
            participants['full_prefix_temporal'] = lambda seed, model=temporal_model, mode=temporal_evidence['public_event_mode']: FullPrefixLivePolicy(model, public_event_mode=mode)
            candidate_evidence['full_prefix_temporal'] = temporal_evidence
        engine = PROJECT_ROOT / 'source/cg/cg.dll'
        store.write('assets.json', {'engine': {'path': str(engine), 'sha256': file_hash(engine)}, 'candidates': candidate_evidence, 'historical_checkpoints': {row.identity: row.checkpoint_sha256 for row in bindings if row.checkpoint}, 'native_api_docs': 'https://matsuoinstitute.github.io/cabt/#game-api', 'api_pairing_contract': 'separate policy RNG seeds, unpaired native chance; no bit-reproducible-game claim', 'legacy_loader': 'preserved R4FLivePolicy.select with explicit model/deck/feature-policy bindings', 'all_policies_frozen': True})
        jobs = [(participant, opponent.identity, seed, seat, repetition) for participant in sorted(participants) for opponent in bindings for seed in config.seeds for seat in (0, 1) for repetition in range(config.repetitions)]
        random.Random(20261025).shuffle(jobs)
        store.write('schedule.json', {'jobs': jobs, 'statistical_unit': 'independent native game within opponent/seat/seed stratum; native chance unpaired', 'stopping': 'exact frozen schedule or first safety/resource failure; no early stopping on scores'})
        by_id = {row.identity: row for row in bindings}
        for index, (participant, opponent, seed, seat, repetition) in enumerate(jobs):
            check_time()
            our = IndependentSeat(participants[participant](seed), seed + seat * 1000003, config.max_decision_seconds)
            other = IndependentSeat(historical(by_id[opponent]), seed + (1 - seat) * 1000003, config.max_decision_seconds)
            observers = tuple(policy for policy in (our.policy, other.policy) if isinstance(policy, FullPrefixLivePolicy))
            our_agent = ActorNotifyingSeat(our, observers) if observers else our
            other_agent = ActorNotifyingSeat(other, observers) if observers else other
            record = run_real_meta_game(our_agent, other_agent, own_deck, read_deck(by_id[opponent].deck), seat_our=seat, max_steps=config.max_selections, rng=random.Random(seed), game_index=index, rng_seed=seed, on_observation=check_time)
            row = {'participant': participant, 'opponent': opponent, 'policy_seed': seed, 'seat': seat, 'repetition': repetition, 'record': asdict(record) | {'error': record.error.split(':', 1)[0][:80] if record.error else None}, 'outcome': record.outcome, 'candidate_calls': our.calls, 'opponent_calls': other.calls, 'candidate_decisions': dict(getattr(our.policy, 'counts', {})), 'reference_fallback_errors': getattr(other.policy, '_fallback_errors', 0)}
            row['reference_fallback_errors'] = int(getattr(other.policy, 'fallback_errors', 0))
            row['opponent_decisions'] = dict(getattr(other.policy, 'counts', {}))
            row['candidate_latency_ms'] = {'mean': float(np.mean(our.latencies) * 1000), 'p99': float(np.quantile(our.latencies, .99) * 1000), 'max': float(max(our.latencies) * 1000)} if our.latencies else {'status': 'not_measured_no_candidate_calls'}
            store.write(f'games/{index:06d}.json', row)
            results.append(row)
            if record.illegal or record.crashed or record.timed_out or record.result not in (0, 1, 2):
                raise RuntimeError('league safety stop; exact game and identities preserved')
            if progress:
                progress(index + 1, len(jobs))
        routes = Counter()
        for row in results:
            if row['participant'] != 'D1_anchor':
                routes[row['participant']] += row['candidate_decisions'].get('model_choices', 0)
        if any(routes[participant] == 0 for participant in participants if participant != 'D1_anchor'):
            raise ValueError('learned policy never reached; fallback-only league result rejected')
        source_hashes = {str(Path(module.__file__).resolve().relative_to(PROJECT_ROOT)): file_hash(Path(module.__file__)) for module in list(sys.modules.values()) if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT / 'source')}
        store.write('source-runtime.json', {'imported_source_hashes': source_hashes, 'bytecode_writes_disabled': True, 'reference_notebooks': {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in (PROJECT_ROOT / 'league/teachers').glob('*.ipynb')}})
        report = {'records': results, **summarize(results), 'safety': {'illegal': 0, 'crashes': 0, 'timeouts': 0}, 'model_route_counts': dict(routes), 'elapsed_seconds': time.perf_counter() - started, 'interpretation': 'single-training-seed exploratory unpaired league diagnostic, no promotion or champion change', 'paid_compute_usd': 0}
        store.write('results.json', report)
        if config.temporal_checkpoint is not None:
            temporal = candidate_evidence['full_prefix_temporal']
            receipt = blank_receipt()
            receipt.update({'receipt_version': 1, 'receipt_id': f"receipt:{config.run_id}", 'artifact_id': f"sha256:{temporal['sha256']}", 'experiment_id': config.run_id, 'artifact_kind': 'full_prefix_temporal_native_d1_evaluation', 'aliases': ['full_prefix_temporal'], 'git_commit': evaluation_code['git_commit'], 'code_version': evaluation_code, 'config_version_or_hash': digest(canonical(resolved_config)), 'dirty_patch_hash': evaluation_code['dirty_patch_hash'], 'environment_lock_or_container_hash': 'UNKNOWN; local Python/Torch/CABT versions are recorded by source and checkpoint receipts but no lock hash exists', 'reconstruction_command': f"python -m deepmew_ex.cli league temporal-d1 --run-id {config.run_id} --checkpoint {config.temporal_checkpoint} --checkpoint-sha256 {config.temporal_checkpoint_sha256} --device {config.temporal_device}", 'artifact_location_and_retention': {'path': temporal['path'], 'retention': 'recovered immutable-by-policy local checkpoint; original interrupted run preserved'}, 'parent_artifacts_and_checkpoints': temporal, 'dataset_id': 'NOT_APPLICABLE; native gameplay evaluation', 'dataset_hash_or_version': 'NOT_APPLICABLE; native gameplay evaluation', 'dataset_manifest_location': 'NOT_APPLICABLE; native gameplay evaluation', 'raw_example_count': temporal['presentations'], 'unique_episode_or_game_count': len(results), 'train_split_definition_and_manifest': 'Recovered training consumption.sqlite3; not modified by evaluation', 'validation_split_definition_and_manifest': 'Recovered evaluations/epoch-0-raw.json; not reused for gameplay selection', 'test_split_definition_and_manifest': {'schedule': 'schedule.json', 'opponent_registry_sha256': resolved_config['registry_sha256']}, 'leakage_constraints': 'Actor-visible projection and public logs only; no simulator buffer, hidden hand/deck/prize, or training mutation', 'architecture': temporal['architecture'], 'exact_trainable_parameter_count': temporal['parameters'], 'parameter_count_by_subsystem': 'Recovered parent parameters.json', 'parameter_count_method_and_evidence': 'sum(parameter.numel()) after exact checkpoint load; all parameters frozen for evaluation', 'input_schema': 'actor-visible current state, own hand, exact legal option payloads, request context, public log snapshots, selected past actions, and directly observed same-seat continuation states', 'output_or_action_schema': 'exact native CABT selection positions reconstructed through bound legal contract', 'temporal_semantics': 'complete per-game actor prefix; intermediate result event only for immediately consecutive same-seat native prompts', 'training_recurrence_and_state_propagation': 'Recovered full-prefix causal Transformer checkpoint; no training performed here', 'inference_recurrence_and_state_propagation': 'Prefix rebuilt from actor-visible events each request; new policy instance resets every game and seat', 'sequence_length': 'complete actor prefix within one native game', 'burn_in': 'NOT_APPLICABLE; full prefix evaluated', 'recurrent_reset_boundary': 'new policy/history object for every game; seat identity fixed within game', 'learning_or_decision_clock': 'NOT_APPLICABLE; zero optimizer updates', 'snapshot_or_sequence_examples': 'native causal sequences', 'objective_or_loss': 'NOT_APPLICABLE; gameplay evaluation only', 'optimizer': 'NOT_APPLICABLE; zero optimizer updates', 'learning_rate_and_scheduler': 'NOT_APPLICABLE; zero optimizer updates', 'material_hyperparameters': resolved_config, 'random_seeds': list(config.seeds), 'training_steps_or_updates': 0, 'unique_examples_actually_consumed': len(results), 'consumed_example_and_game_manifest': 'schedule.json and games/*.json', 'compute_and_wall_time': {'device': config.temporal_device, 'seconds': report['elapsed_seconds'], 'paid_compute_usd': 0}, 'checkpoint_hash': temporal['sha256'], 'final_trained_artifact_hash': temporal['sha256'], 'evaluation_receipt_id_and_location': {'id': f"receipt:{config.run_id}", 'path': 'lineage-receipt.json'}, 'evaluation_protocol_and_code_config': 'config.json, schedule.json, code.json, source-runtime.json', 'evaluation_dataset_opponents_and_seeds': {'opponents': [row.identity for row in bindings], 'seeds': list(config.seeds), 'seats': [0, 1], 'repetitions': config.repetitions}, 'matched_baseline_artifact_and_receipt': {'baseline': 'D1_anchor self-play against the same frozen D1 checkpoint/deck', 'checkpoint_sha256': d1.checkpoint_sha256, 'deck_sha256': d1.deck_sha256}, 'held_out_metrics': 'NOT_APPLICABLE; native gameplay schedule', 'calibration_metrics': 'UNKNOWN; no probability calibration evaluated', 'arena_or_task_metrics': {'matchups': report['matchups'], 'safety': report['safety'], 'routes': report['model_route_counts'], 'runtime_seconds': report['elapsed_seconds']}, 'external_competition_or_production_metric': 'NOT_APPLICABLE; local evaluation only', 'metric_sample_counts_and_uncertainty': {'games': len(results), 'matchup_intervals': 'Wilson 95% for win rate; D1-only candidate and D1 self-play control', 'engine_chance': 'unpaired'}, 'runtime_transformation_chain': [{'operation': 'load exact checkpoint and attach nonlearned causal inference adapter', 'parent_sha256': temporal['sha256'], 'output_checkpoint_sha256': temporal['sha256'], 'validation': 'tests plus this native league receipt'}], 'quantization': 'NOT_APPLICABLE; none', 'pruning': 'NOT_APPLICABLE; none', 'compilation': 'NOT_APPLICABLE; eager PyTorch', 'conversion': 'NOT_APPLICABLE; checkpoint weights unchanged', 'distillation': 'NOT_APPLICABLE; none', 'export_format': 'PyTorch checkpoint loaded weights_only=True', 'deployed_artifact_hash': temporal['sha256'], 'submission_or_production_run_id': 'NOT_APPLICABLE; local evaluation only', 'decision': 'DIAGNOSTIC', 'decision_reason': 'Playable-adapter and frozen-D1 evidence only; promotion prohibited', 'decision_timestamp_and_authority': {'date': '2026-11-07', 'authority': 'explicit user request'}, 'supersedes_receipt_id': 'NOT_APPLICABLE; first native evaluation of recovered checkpoint', 'what_this_establishes': 'Exact hash-bound checkpoint can or cannot execute legally through the causal actor-visible adapter on the frozen D1 schedule', 'what_this_does_not_establish': 'Promotion, external competition strength, paired engine-chance superiority, temporal-memory causality, or benefit from more training', 'field_evidence': {'checkpoint': 'VERIFIED_ARTIFACT via SHA-256 and sidecar', 'training_identity': 'VERIFIED_ARTIFACT via recovered code/config/corpus/parser identities', 'gameplay': 'VERIFIED_ARTIFACT via sealed per-game records and result receipt'}, 'reproduction_status_and_receipt': 'Executed locally; sealed run manifest and source hashes'})
            store.write('lineage-receipt.json', receipt)
        store.seal({'kind': 'v2_league_evaluation', 'games': len(results), 'promotion': False})
    except Exception as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'completed_games': len(results), 'elapsed_seconds': time.perf_counter() - started, 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_league_evaluation', 'promotion': False}, 'failed')
        raise
    return store.path
