"""Independent immutable part/episode writers; no models or engine initialization."""

from __future__ import annotations

import gzip
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.corpus.records import normalize_reference
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import RunStore, file_hash

_INVENTORY: Payload = {}
_SOURCES: dict[str, Payload] = {}
_ROOT: Path | None = None


def initialize(inventory: str, root: str) -> None:
    global _INVENTORY, _SOURCES, _ROOT
    _INVENTORY = orjson.loads(Path(inventory).read_bytes())
    _SOURCES = {item['path']: item for item in _INVENTORY['sources']}
    _ROOT = Path(root)


def child_store(kind: str, name: str) -> RunStore:
    if _ROOT is None:
        raise ValueError('worker is not initialized')
    path = _ROOT / kind / name
    path.parent.mkdir(exist_ok=True)
    path.mkdir()
    return RunStore(path)


def rows(path: Path) -> Iterator[Payload]:
    with gzip.open(path, 'rb') as handle:
        for line in handle:
            row = orjson.loads(line)
            if not isinstance(row, dict):
                raise ValueError('invalid JSON row')
            yield row


def verify_file(entry: Payload) -> None:
    path = Path(entry['path'])
    if path.stat().st_size != entry['byte_size'] or file_hash(path) != entry['sha256']:
        raise ValueError('source_hash_or_size_mismatch')


def raw_rejection(error: Exception) -> str:
    message = str(error)
    exact = {'unsupported replay source contract': 'unsupported_source_contract', 'source hash mismatch': 'source_hash_mismatch', 'oversized_raw_episode': 'oversized_raw_episode', 'active actor request missing; temporal history gap': 'missing_actor_request', 'action without an active prior request': 'unbound_source_action', 'missing seed or frozen evaluation seed': 'missing_or_frozen_seed', 'terminal summary conflicts with final actor evidence': 'terminal_summary_conflict', 'incomplete game, unsafe final status or unsupported outcome': 'incomplete_or_unsafe_episode', 'source safety/status failure': 'unsafe_source_status', 'intermediate terminal/reward evidence is unsupported': 'intermediate_reward_or_terminal_conflict'}
    if message in exact:
        return exact[message]
    if message.startswith('legal/chosen action contract rejection:'):
        return 'legal_chosen_or_order_rejection'
    if message.startswith('observation classified'):
        return 'non_actor_visible_observation'
    if message.startswith('unclassified'):
        return 'unclassified_semantic_extension'
    return 'malformed_or_unsupported_replay'


def canonical_part(part: Payload) -> Payload:
    store = child_store('canonical', part['part_id'])
    counts, reasons = Counter(), Counter()
    try:
        for entry in [*part['files'].values(), part['target_mask']]:
            verify_file(entry)
        name = part['name']
        transitions = Path(part['files'][name + '.transitions.jsonl.gz']['path'])
        sequences = Path(part['files'][name + '.sequences.jsonl.gz']['path'])
        with np.load(part['target_mask']['path'], allow_pickle=False) as masks:
            mask = masks['is_target_eligible']
        if mask.dtype != np.bool_ or mask.shape != (part['rows'],) or int((~mask).sum()) != part['excluded_target_rows']:
            raise ValueError('invalid_target_mask')
        # Game metadata is small and source-bound; its rows also preserve policy/deck
        # identity without exposing those audit labels as model inputs.
        games = list(rows(Path(part['files'][name + '.games.jsonl.gz']['path'])))
        if any(game['information_mode'] != 'actor_visible' or game['game_completion_status'] != 'completed' or game['safety_status'] != 'clean' for game in games):
            raise ValueError('unsafe_game_metadata')
        store.write('games.json', games)
        output = store.path / 'frames.jsonl.zst'
        semantic_root = hashlib.sha256()
        with output.open('xb') as handle:
            with zstandard.ZstdCompressor(level=1).stream_writer(handle, closefd=False) as writer:
                sequence_iter = iter(rows(sequences))
                for index, transition in enumerate(rows(transitions)):
                    sequence = next(sequence_iter, None)
                    if sequence is None or index >= len(mask):
                        raise ValueError('sidecar_length_mismatch')
                    counts['historical_target_mask_excluded'] += not bool(mask[index])
                    game = transition['canonical_game_id']
                    try:
                        normalized = normalize_reference(transition, sequence, _SOURCES, _INVENTORY['source_assignments'][game], _INVENTORY['effective_assignments'][game], bool(mask[index]), part['unit_id'])
                        counts['semantic_references'] += 1
                        counts['optional_empty'] += not normalized['chosen_action']['indices']
                        counts['multi_select'] += len(normalized['chosen_action']['indices']) > 1
                        counts['enriched_action_envelopes'] += normalized['chosen_action']['source_complete_action_id'] is not None
                    except (KeyError, TypeError, ValueError) as error:
                        reason = str(error) if type(error) is ValueError else 'malformed_source_metadata'
                        if len(reason) > 80 or not all(char.islower() or char == '_' for char in reason):
                            reason = 'invalid_action_or_metadata'
                        reasons[reason] += 1
                        # Quarantine keeps exact source coordinates, never an unsafe payload.
                        normalized = {'schema': 'deepmew.ex.quarantined_reference.v2', 'source_family': part['source_id'], 'part_id': part['part_id'], 'row': index, 'source_transition_sha256': part['files'][name + '.transitions.jsonl.gz']['sha256'], 'reason': reason, 'capabilities': {'snapshot_bc': False, 'temporal_bc': False, 'td_transition': False}}
                    encoded = orjson.dumps(normalized, option=orjson.OPT_SORT_KEYS) + b'\n'
                    semantic_root.update(encoded)
                    writer.write(encoded)
                    counts['rows'] += 1
                if next(sequence_iter, None) is not None or counts['rows'] != part['rows']:
                    raise ValueError('sidecar_row_conservation_failure')
        store.write('result.json', {'part_id': part['part_id'], 'source_family': part['source_id'], 'counts': dict(counts), 'quarantine_reasons': dict(reasons), 'semantic_root_sha256': semantic_root.hexdigest(), 'input': part})
        store.seal({'kind': 'v2_canonical_reference_part', 'optimizer_updates': 0})
        return {'kind': 'canonical', 'part_id': part['part_id'], 'source_family': part['source_id'], 'path': str(store.path), 'manifest_sha256': file_hash(store.path / 'manifest.json'), 'semantic_root_sha256': semantic_root.hexdigest(), 'counts': dict(counts), 'quarantine_reasons': dict(reasons)}
    except Exception as error:
        store.write('failure.json', {'error_type': type(error).__name__, 'rows_completed': counts['rows'], 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_canonical_reference_part'}, 'failed')
        raise


def raw_episode(source: Payload, max_bytes: int, max_steps: int) -> Payload:
    store = child_store('raw', source['sha256'])
    candidate_requests: int | None = None
    # A changed supposedly immutable source is a run-integrity failure, not an
    # eligibility exclusion that can be hidden in a completed port.
    verify_file(source)
    try:
        if source['byte_size'] > max_bytes:
            raise ValueError('oversized_raw_episode')
        raw = Path(source['path']).read_bytes()
        data = orjson.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('malformed_replay_root')
        steps = data.get('steps')
        if isinstance(steps, list):
            candidate_requests = sum(isinstance(agent, dict) and agent.get('status') == 'ACTIVE' and isinstance(agent.get('observation'), dict) and isinstance(agent['observation'].get('select'), dict) for step in steps[:-1] if isinstance(step, list) for agent in step)
        contract = load_contract()
        native = data.get('id')
        module = data.get('module_version')
        frozen = _INVENTORY['raw_splits'].get(native)
        if frozen is None:
            number = int(hashlib.sha256(str(native).encode()).hexdigest(), 16) % 10
            frozen = 'train' if number < 8 else 'validation' if number == 8 else 'holdout'
        # The golden converter's deterministic fixture partition remains intact.
        # Production effective assignment is a separate, more restrictive ledger.
        from deepmew_ex.datasets.v2.conversion import game_split
        metadata = {'sha256': source['sha256'], 'native_game_id': native, 'module_version': module, 'split': game_split(str(native)), 'family': 'kaggle_cabt_' + str(module), 'asset_id': Path(source['path']).name}
        episode = convert_episode(raw, metadata, contract, max_steps)
        with (store.path / 'episode.json.zst').open('xb') as handle:
            # Compact JSON is a storage view; the semantic object remains unchanged.
            handle.write(zstandard.ZstdCompressor(level=1).compress(episode.to_bytes()))
        result = {'source': source, 'native_game_id': native, 'effective_split': frozen, 'frames': len(episode.frames), 'episode_id': episode.episode_id, 'status': 'admitted_actor_visible_episode', 'source_unchanged': file_hash(Path(source['path'])) == source['sha256']}
        if not result['source_unchanged']:
            raise RuntimeError('raw_source_changed_during_conversion')
        store.write('result.json', result)
        store.seal({'kind': 'v2_raw_episode', 'optimizer_updates': 0})
        return {'kind': 'raw', 'path': str(store.path), 'manifest_sha256': file_hash(store.path / 'manifest.json'), **result}
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        reason = raw_rejection(error) if not isinstance(error, OSError) else 'raw_episode_unavailable'
        store.write('quarantine.json', {'source': source, 'reason': reason, 'candidate_request_events': candidate_requests, 'error_type': type(error).__name__, 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_raw_episode_quarantine', 'optimizer_updates': 0})
        return {'kind': 'raw', 'path': str(store.path), 'source': source, 'manifest_sha256': file_hash(store.path / 'manifest.json'), 'status': 'quarantined', 'reason': reason, 'candidate_request_events': candidate_requests, 'frames': 0}
