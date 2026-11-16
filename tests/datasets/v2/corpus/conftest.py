from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepmew_ex.datasets.v2.contract import binding_hashes, load_contract
from deepmew_ex.registry.store import canonical, file_hash


@pytest.fixture
def corpus_inventory(tmp_path: Path) -> Path:
    source = tmp_path / 'source.jsonl'
    contract = load_contract()
    observations = []
    for index in range(2):
        observations.append({'current': {'yourIndex': 0, 'players': [{'hand': [{'id': 123}], 'prize': [None], 'active': [], 'bench': [], 'discard': []}, {'hand': None, 'prize': [None], 'active': [], 'bench': [], 'discard': []}], 'turn': index}, 'select': {'type': 9, 'context': 41, 'minCount': 1, 'maxCount': 1, 'option': [{'type': 1}, {'type': 2}]}, 'logs': [], 'search_begin_input': 'PRIVATE-SENTINEL'})
    source_rows = [{'schema_version': 'deepmew.refgames_factory.decision.v0', 'game_id': 'game', 'game_index': 0, 'seat': 0, 'decision_index': index, 'raw_observation': observation, 'chosen_action': {'indices': [index]}} for index, observation in enumerate(observations)]
    source.write_bytes(b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in source_rows))
    source_sha = file_hash(source)
    transitions, sequences = [], []
    for index, observation in enumerate(observations):
        semantic = list(contract.semantic_options(contract.project_actor(observation)))
        state_ref = {'source_path': str(source), 'source_line': index + 1, 'source_field': 'raw_observation', 'content_sha256': hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()}
        common = {'decision_id': f'game:d{index}', 'canonical_game_id': 'unified_r2:unit:game', 'actor': 0, 'actor_decision_index': index, 'global_decision_index': index, 'source_hash': source_sha, 'source_schema': 'deepmew.refgames_factory.decision.v0', 'state_before': {'source_observation': state_ref, 'feature_dim': 144, 'feature_row': index, 'representation': 'canonical_feature_vector.v1'}, 'information_mode': 'actor_visible'}
        complete = {'schema': 'deepmew.complete_legal_option_selection.v0', 'indices': [index], 'semantics': [{'index': index, 'option_type': 'YES' if index == 0 else 'NO', 'select_context': 'IS_FIRST', 'semantic_key': 'YES' if index == 0 else 'NO'}]}
        transitions.append({**common, 'source_id': 'unified_r2', 'terminal_actor_relative_result': 1.0, 'legal_action_set': semantic, 'chosen_action': {'indices': [index], 'semantics': [semantic[index]], 'complete_action_id': json.dumps(complete)}, 'state_after': None, 'state_after_status': 'unavailable_missing_post_observation', 'next_same_actor_state': {'decision_id': 'game:d1'} if index == 0 else None, 'terminal': index == 1, 'transition_horizon_logical_decisions': 1 if index == 0 else None})
        sequences.append({**common, 'source_game_id': 'game', 'source_order': 0, 'control_owner': 0, 'public_event_history': {'status': 'unavailable'}})
    names = {'part_0000.transitions.jsonl.gz': transitions, 'part_0000.sequences.jsonl.gz': sequences, 'part_0000.games.jsonl.gz': [{'information_mode': 'actor_visible', 'game_completion_status': 'completed', 'safety_status': 'clean'}], 'part_0000.actions.jsonl.gz': []}
    files = {}
    for name, rows in names.items():
        path = tmp_path / name
        with gzip.open(path, 'wb') as handle:
            for row in rows:
                handle.write(json.dumps(row).encode() + b'\n')
        files[name] = {'path': str(path), 'sha256': file_hash(path), 'byte_size': path.stat().st_size}
    mask = tmp_path / 'mask.npz'
    np.savez(mask, is_target_eligible=np.array([True, False]))
    part = {'part_id': 'synthetic-part', 'name': 'part_0000', 'source_id': 'unified_r2', 'unit_id': 'unit', 'rows': 2, 'files': files, 'target_mask': {'path': str(mask), 'sha256': file_hash(mask), 'byte_size': mask.stat().st_size}, 'excluded_target_rows': 1}
    inventory = {'schema': 'deepmew.ex.full_corpus_inventory.v1', 'parser_hashes': binding_hashes(), 'parts': [part], 'sources': [{'path': str(source), 'sha256': source_sha, 'byte_size': source.stat().st_size}], 'evidence': [], 'source_assignments': {'unified_r2:unit:game': 'train'}, 'effective_assignments': {'unified_r2:unit:game': 'validation'}, 'raw_splits': {}, 'replays': [], 'family_counts': {'unified_r2': 2}, 'catalog_counts': {'rows_kept': 2, 'games_kept': 1}, 'legacy_excluded_target_rows': 1}
    path = tmp_path / 'inventory.json'
    path.write_bytes(canonical(inventory))
    return path
