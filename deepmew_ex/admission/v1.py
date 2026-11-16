"""Hash-bound streaming corpus audit. No trainer, model or engine imports."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import time
import platform
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from deepmew_ex.admission.config import AuditConfig
from deepmew_ex.admission.decisions import EXCLUSION_PRECEDENCE, exclusion_reason, validate_decision
from deepmew_ex.admission.sources import inspect_sources
from deepmew_ex.data import Decision, Split
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash
from deepmew_ex.registry.provenance import code_identity

SCHEMA = 'deepmew.ex.portable_actor_corpus.v1'
SPLITS: tuple[Split, ...] = ('train', 'validation', 'holdout')
ARRAY_DTYPES = {'option_features': 'float32', 'value_features': 'float32', 'option_offsets': 'int64', 'selection_offsets': 'int64', 'selection_indices': 'int32', 'min_count': 'int32', 'max_count': 'int32', 'complete_legal_action_count': 'int64', 'terminal_outcome': 'float32', 'actor_loss_weight': 'float32'}


def json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path.name}')
    return value


def validate_manifest(corpus: Path, trusted: Path) -> dict[str, object]:
    expected = json_object(trusted)
    actual = json_object(corpus / 'EXPORT_MANIFEST.json')
    if actual != expected:
        raise ValueError('corpus manifest does not match trusted dataset manifest')
    if actual.get('schema_version') != SCHEMA or actual.get('actor_feature_dim') != 304 or actual.get('value_feature_dim') != 144:
        raise ValueError('unsupported dataset schema or feature dimensions')
    shards = actual.get('shards')
    if not isinstance(shards, list) or not shards:
        raise ValueError('empty shard manifest')
    for ordinal, shard in enumerate(shards):
        if shard['ordinal'] != ordinal or type(shard['rows']) is not int or shard['rows'] < 1:
            raise ValueError('shard order/row count is invalid')
        names = {f'shard_{ordinal:05d}.npz', f'shard_{ordinal:05d}.rows.jsonl.gz'}
        if len(shard['files']) != 2 or {entry['path'] for entry in shard['files']} != names:
            raise ValueError('unexpected or unsafe shard filename')
        for entry in shard['files']:
            validate_hash(entry['sha256'])
            if type(entry['bytes']) is not int or entry['bytes'] < 1:
                raise ValueError('invalid shard byte count')
    return actual


def load_arrays(path: Path, row_count: int) -> dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_DTYPES):
            raise ValueError('unexpected/missing NPZ arrays')
        arrays = {key: archive[key] for key in archive.files}
    for name, dtype in ARRAY_DTYPES.items():
        if str(arrays[name].dtype) != dtype:
            raise ValueError(f'invalid array dtype: {name}')
    for name in ('option_offsets', 'selection_offsets'):
        offsets = arrays[name]
        if offsets.shape != (row_count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError(f'invalid ragged offsets: {name}')
    if arrays['option_features'].shape != (int(arrays['option_offsets'][-1]), 304) or arrays['value_features'].shape != (row_count, 144):
        raise ValueError('invalid feature array shape or final option offset')
    if arrays['selection_indices'].shape != (int(arrays['selection_offsets'][-1]),):
        raise ValueError('invalid selection array shape or final selection offset')
    for name in ('min_count', 'max_count', 'complete_legal_action_count', 'terminal_outcome', 'actor_loss_weight'):
        if arrays[name].shape != (row_count,):
            raise ValueError(f'invalid row array shape: {name}')
    return arrays


def shard_rows(corpus: Path, ordinal: int, arrays: dict[str, NDArray[np.generic]], row_count: int) -> Iterator[tuple[Decision, dict[str, object]]]:
    with gzip.open(corpus / f'shard_{ordinal:05d}.rows.jsonl.gz', 'rt', encoding='utf-8') as handle:
        count = 0
        for index, line in enumerate(handle):
            if index >= row_count:
                raise ValueError('extra metadata rows')
            metadata = json.loads(line)
            for field in ('stable_row_id', 'source_game_id', 'trajectory_id', 'source_family', 'continuity'):
                if not isinstance(metadata.get(field), str) or not metadata[field]:
                    raise ValueError(f'invalid metadata identity: {field}')
            split = metadata.get('split')
            if split not in SPLITS or type(metadata.get('decision_index')) is not int or metadata['decision_index'] < 0:
                raise ValueError('invalid split or decision coordinate')
            if type(metadata.get('actor_seat')) is not int or metadata['actor_seat'] not in (0, 1) or metadata.get('seed') in (40009, 50021, 60013):
                raise ValueError('invalid actor seat or frozen evaluation seed')
            start, stop = arrays['option_offsets'][index:index + 2]
            left, right = arrays['selection_offsets'][index:index + 2]
            row = Decision(metadata['stable_row_id'], metadata['source_game_id'], metadata['trajectory_id'], split, arrays['option_features'][start:stop], arrays['value_features'][index], arrays['selection_indices'][left:right], int(arrays['min_count'][index]), int(arrays['max_count'][index]), int(arrays['complete_legal_action_count'][index]), float(arrays['terminal_outcome'][index]), float(arrays['actor_loss_weight'][index]), metadata['continuity'], metadata['decision_index'])
            count += 1
            yield row, metadata
        if count != row_count:
            raise ValueError('missing metadata rows')


def audit_v1(config: AuditConfig, progress: Callable[[int, int], None] | None = None) -> Path:
    config.validate()
    store = RunStore.allocate(config.output_paths, 'deepmew_ex', 'v1_admission', config.audit_id)
    started = time.perf_counter()
    try:
        store.write('config.json', {'root': str(config.root.resolve()), 'trusted_manifest': str(config.expected_manifest.resolve()), 'validation_receipt': str(config.validation_receipt.resolve()), 'audit_id': config.audit_id, 'selection_seed': config.selection_seed, 'selection_limits': {'train': config.train_selection_limit, 'validation': config.validation_selection_limit}, 'scope': 'read_only_integrity_and_admission; no fitting/tuning or model loading'})
        manifest = validate_manifest(config.corpus, config.expected_manifest)
        expected = json_object(config.validation_receipt)
        store.write('code.json', code_identity(config.project))
        store.write('environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(), 'execution': 'CPU NumPy integrity checks only; no model/trainer/engine imports'})
        store.write('dataset-manifest.json', manifest)
        store.write('historical-validation-receipt.json', expected)
        ids: set[str] = set()
        game_splits: dict[str, str] = {}
        trajectory_splits: dict[str, str] = {}
        support = {split: Counter() for split in SPLITS}
        source_counts: Counter[str] = Counter()
        source_support: dict[str, Counter[str]] = {}
        continuity: Counter[str] = Counter()
        split_digests = {split: hashlib.sha256() for split in SPLITS}
        selected: dict[str, list[tuple[int, str, str, str]]] = {'train': [], 'validation': []}
        hashes: dict[str, str] = {'EXPORT_MANIFEST.json': file_hash(config.corpus / 'EXPORT_MANIFEST.json')}
        complete_count_differs = 0
        total_actor_rows = 0
        for ordinal, shard in enumerate(manifest['shards']):
            for entry in shard['files']:
                path = config.corpus / entry['path']
                actual = file_hash(path)
                if path.stat().st_size != entry['bytes'] or actual != entry['sha256']:
                    raise ValueError(f'shard size/hash mismatch: {entry["path"]}')
                hashes[entry['path']] = actual
            receipt = config.corpus / f'shard_{ordinal:05d}.receipt.json'
            if receipt.exists():
                if json_object(receipt) != shard:
                    raise ValueError('export shard receipt differs from trusted manifest')
                hashes[receipt.name] = file_hash(receipt)
            arrays = load_arrays(config.corpus / f'shard_{ordinal:05d}.npz', shard['rows'])
            per_split: Counter[str] = Counter()
            per_source: Counter[str] = Counter()
            games: set[str] = set()
            actor_rows = 0
            for row, metadata in shard_rows(config.corpus, ordinal, arrays, shard['rows']):
                if row.row_id in ids:
                    raise ValueError('duplicate stable row identity')
                ids.add(row.row_id)
                for mapping, identity in ((game_splits, row.game_id), (trajectory_splits, row.trajectory_id)):
                    if mapping.setdefault(identity, row.split) != row.split:
                        raise ValueError('game/trajectory crosses frozen splits')
                validate_decision(row)
                if not np.isfinite(row.terminal_outcome) or not 0 <= row.terminal_outcome <= 1:
                    raise ValueError('invalid actor-relative terminal outcome')
                if not row.min_count <= len(row.selection) <= row.max_count:
                    raise ValueError('chosen selection cardinality outside bounds')
                complete_count_differs += int(row.complete_legal_action_count != len(row.option_features))
                reason = exclusion_reason(row)
                split_support = support[row.split]
                split_support['raw'] += 1
                split_support[reason or 'eligible'] += 1
                if reason is None:
                    split_support['eligible_weight_sum'] += row.actor_loss_weight
                family = str(metadata['source_family'])
                source_support.setdefault(family, Counter())[f'{row.split}:{reason or "eligible"}'] += 1
                source_counts[family] += 1
                per_split[row.split] += 1
                per_source[family] += 1
                continuity[row.continuity] += 1
                games.add(row.game_id)
                actor_rows += int(row.actor_loss_weight > 0)
                membership = {'row_id': row.row_id, 'game_id': row.game_id, 'trajectory_id': row.trajectory_id, 'source_family': family, 'eligible': reason is None, 'exclusion': reason}
                split_digests[row.split].update(canonical(membership))
                if reason is None and row.split in selected:
                    limit = config.train_selection_limit if row.split == 'train' else config.validation_selection_limit
                    rank = int(digest(f'{config.selection_seed}:{row.row_id}'.encode()), 16)
                    item = (-rank, row.row_id, row.game_id, family)
                    heap = selected[row.split]
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            if dict(per_split) != shard['split_counts'] or dict(per_source) != shard['source_counts'] or len(games) != shard['games_in_shard'] or actor_rows != shard['actor_rows']:
                raise ValueError(f'shard support differs from frozen receipt: {ordinal}')
            total_actor_rows += actor_rows
            del arrays
            if progress:
                progress(ordinal + 1, len(manifest['shards']))
        totals = {split: int(support[split]['raw']) for split in SPLITS}
        if len(ids) != manifest['rows'] or len(game_splits) != manifest['games'] or len(ids) != expected['rows'] or len(game_splits) != expected['games'] or total_actor_rows != expected['actor_rows'] or len(ids) - total_actor_rows != expected['forced_rows'] or totals != expected['splits'] or dict(source_counts) != expected['sources'] or dict(continuity) != expected['continuity']:
            raise ValueError('corpus totals differ from frozen validation receipt')
        if any(support[split]['eligible'] == 0 for split in ('train', 'validation')):
            raise ValueError('empty eligible training or validation support')
        for relative, sha in hashes.items():
            if file_hash(config.corpus / relative) != sha:
                raise ValueError('source corpus changed during read-only audit')
        sources = inspect_sources(config.project)
        store.write('source-evidence.json', sources)
        selected_manifests: dict[str, object] = {}
        for split, heap in selected.items():
            rows = [{'rank_sha256': f'{-rank:064x}', 'row_id': row_id, 'game_id': game_id, 'source_family': family} for rank, row_id, game_id, family in sorted(heap, reverse=True)]
            sha = store.write(f'proposed-{split}-membership.json', rows)
            selected_manifests[split] = {'path': f'proposed-{split}-membership.json', 'sha256': sha, 'rows': len(rows), 'games': len({row['game_id'] for row in rows}), 'source_counts': dict(Counter(row['source_family'] for row in rows))}
        support_report = {split: {'raw': int(counts['raw']), 'eligible': int(counts['eligible']), 'eligible_weight_sum': counts['eligible_weight_sum'], 'excluded': {reason: int(counts[reason]) for reason in EXCLUSION_PRECEDENCE}, 'membership_sha256': split_digests[split].hexdigest()} for split, counts in support.items()}
        result = {'audit_id': config.audit_id, 'status': 'passed', 'readiness': 'ready_for_snapshot_bc_review' if not sources['missing'] else 'blocked_missing_source_evidence', 'training_execution': 'blocked; S1 trainer/scope must be separately implemented and approved', 'scope': 'all corpus bytes/metadata audited; no examples consumed by fitting or tuning', 'schema': SCHEMA, 'dataset_manifest_sha256': digest(canonical(manifest)), 'input_files_sha256': hashes, 'rows': len(ids), 'games': len(game_splits), 'shards': len(manifest['shards']), 'splits': support_report, 'source_counts': dict(source_counts), 'source_support': {key: dict(value) for key, value in source_support.items()}, 'continuity': dict(continuity), 'complete_action_count_differs_from_candidate_count': complete_count_differs, 'game_split_sha256': digest(canonical(game_splits)), 'capabilities': {'snapshot_bc': True, 'semantic_v2': False, 'temporal_bc': False, 'td_transition': False}, 'visibility': sources['status'], 'selection_recipe': 'lowest SHA256(seed:stable_row_id) among eligible rows of each frozen split; no label-based selection', 'proposed_membership': selected_manifests, 'holdout_use': 'integrity/metadata/eligibility audit only; not selected for model scoring, fitting or tuning', 'model_imported': False, 'optimizer_updates': 0, 'gameplay_evaluation': 'not_run', 'seconds': time.perf_counter() - started}
        store.write('audit.json', result)
        store.seal({'schema_version': 1, 'kind': 'dataset_admission', 'audit_id': config.audit_id, 'readiness': result['readiness'], 'optimizer_updates': 0})
        return store.path
    except BaseException as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'seconds': time.perf_counter() - started, 'optimizer_updates': 0})
        store.seal({'schema_version': 1, 'kind': 'dataset_admission', 'audit_id': config.audit_id, 'optimizer_updates': 0}, status='failed')
        raise
