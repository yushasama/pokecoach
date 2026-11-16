from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.learning.legacy_contract import recorded_legacy_contract
from deepmew_ex.datasets.v2.learning.raw_recovery import inherit_raw_masks, recover_raw_histories
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence
from deepmew_ex.datasets.v2.schema import Episode, Payload
from deepmew_ex.registry.store import canonical, digest
from tests.datasets.v2.conftest import source_row
from tests.datasets.v2.learning.test_legacy_contract import legacy_replay


@pytest.fixture
def mask_inputs() -> tuple[Episode, bytes, tuple[ActorSequence, ...], list[Payload], dict[str, bytes]]:
    replay = legacy_replay()
    raw = canonical(replay)
    episode = convert_episode(raw, source_row(raw, Path('unused'), replay['id'], '1.30.1'), recorded_legacy_contract())
    sequences = recover_raw_histories(episode, raw, 'train')
    rows, references = [], []
    for frame in episode.frames:
        action = replay['steps'][frame.action_step][frame.actor_seat]
        row = {'schema_version': 'deepmew.replay_action_row.v1', 'game_id': episode.source_native_game_id, 'seed': episode.seed, 'step': frame.action_step, 'player_id': frame.actor_seat, 'state_json': action['observation'], 'chosen_indices': action['action'], 'legal_options_json': action['observation']['select']['option']}
        rows.append(row)
        observation_hash = digest(json.dumps(row['state_json'], sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())
        references.append({'frame_id': 'legacy:' + frame.frame_id, 'source_schema': row['schema_version'], 'native_game_binding': {'raw_game_id': episode.source_native_game_id}, 'global_order': frame.action_step, 'actor_seat': frame.actor_seat, 'state_before': {'source_path': 'table.jsonl', 'source_line': len(rows), 'source_field': 'raw_observation', 'content_sha256': observation_hash}, 'chosen_action': {'indices': row['chosen_indices']}, 'legal_actions': row['legal_options_json'], 'target_mask': {'source_target_eligible': True}, 'effective_split': 'train'})
    encoded = b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in rows)
    for reference in references:
        reference['state_before']['source_sha256'] = digest(encoded)
    return episode, raw, sequences, references, {'table.jsonl': encoded}


def test_legacy_mask_join_preserves_full_context_and_protects_reserved_game(mask_inputs: tuple) -> None:
    episode, raw, sequences, references, sources = mask_inputs
    references[0]['target_mask']['source_target_eligible'] = False
    references[0]['effective_split'] = 'holdout'
    restored, receipt = inherit_raw_masks(episode, raw, sequences, references, sources)
    assert receipt['matched_native_actions'] == len(episode.frames) and receipt['masked_native_actions'] == 1
    assert all(sequence.split == 'holdout' for sequence in restored)
    assert all(first.records == second.records and first.events == second.events and first.preserved_transition_ids == second.preserved_transition_ids for first, second in zip(sequences, restored, strict=True))
    target = episode.frames[0].frame_id
    assert not next(row for sequence in restored for row in sequence.predictions if row.row_id == target).loss_eligible
    assert not receipt['input_records_changed'] and not receipt['training_admitted']


def test_duplicate_formats_cannot_reenable_one_physical_action(mask_inputs: tuple) -> None:
    episode, raw, sequences, references, sources = mask_inputs
    masked = copy.deepcopy(references[0])
    masked['frame_id'] = 'other-format'
    masked['target_mask']['source_target_eligible'] = False
    restored, receipt = inherit_raw_masks(episode, raw, sequences, [masked, *references], sources)
    assert receipt['references'] == len(episode.frames) + 1 and receipt['masked_native_actions'] == 1
    assert not restored[0].predictions[0].loss_eligible


def test_unreferenced_new_native_targets_keep_existing_masks(mask_inputs: tuple) -> None:
    episode, raw, sequences, _, _ = mask_inputs
    masked = replace(sequences[0], predictions=tuple(replace(row, loss_eligible=False) for row in sequences[0].predictions))
    sequences = masked, *sequences[1:]
    restored, receipt = inherit_raw_masks(episode, raw, sequences, [], {})
    assert restored == sequences and receipt['matched_native_actions'] == 0


def test_swapped_same_actor_coordinates_cannot_move_a_historical_mask(mask_inputs: tuple) -> None:
    episode, raw, sequences, references, sources = mask_inputs
    first, second, *remaining = sequences[0].predictions
    changed = replace(sequences[0], predictions=(replace(first, row_id=second.row_id), replace(second, row_id=first.row_id), *remaining))
    references[0]['target_mask']['source_target_eligible'] = False
    with pytest.raises(ValueError, match='actor mismatch'):
        inherit_raw_masks(episode, raw, (changed, *sequences[1:]), references, sources)


@pytest.mark.parametrize('defect', ['hash', 'native', 'coordinate', 'observation', 'action', 'legal', 'mask', 'split', 'repeated_reference'])
def test_uncertain_historical_join_cannot_publish_masks(mask_inputs: tuple, defect: str) -> None:
    episode, raw, sequences, references, sources = mask_inputs
    if defect == 'hash':
        sources['table.jsonl'] += b'\n'
    elif defect == 'native':
        references[0]['native_game_binding']['raw_game_id'] = 'different-game'
    elif defect == 'coordinate':
        references[0]['global_order'] += 1
    elif defect == 'observation':
        references[0]['state_before']['content_sha256'] = '0' * 64
    elif defect == 'action':
        references[0]['chosen_action']['indices'] = [999]
    elif defect == 'legal':
        references[0]['legal_actions'] = []
    elif defect == 'mask':
        references[0]['target_mask']['source_target_eligible'] = None
    elif defect == 'split':
        references[0]['effective_split'] = 'unknown'
    else:
        references.append(references[0])
    with pytest.raises(ValueError):
        inherit_raw_masks(episode, raw, sequences, references, sources)


def test_post_action_evidence_is_not_accepted_as_prior_request_evidence(mask_inputs: tuple) -> None:
    episode, raw, sequences, references, sources = mask_inputs
    replay = json.loads(raw)
    rows = [json.loads(line) for line in sources['table.jsonl'].splitlines()]
    frame = episode.frames[0]
    rows[0]['state_json'] = replay['steps'][frame.request_step][frame.actor_seat]['observation']
    encoded = b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in rows)
    sources['table.jsonl'] = encoded
    for reference in references:
        reference['state_before']['source_sha256'] = digest(encoded)
    references[0]['state_before']['content_sha256'] = digest(json.dumps(rows[0]['state_json'], sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())
    with pytest.raises(ValueError, match='post-observation'):
        inherit_raw_masks(episode, raw, sequences, references, sources)
