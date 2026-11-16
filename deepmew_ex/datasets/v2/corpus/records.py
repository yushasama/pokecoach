"""Validate enriched historical action envelopes and retain semantic source identity."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import orjson

from deepmew_ex.datasets.v2.schema import Payload

REFERENCE_SCHEMA = 'deepmew.ex.semantic_reference.v2'
ACTION_FIELDS = frozenset({'indices', 'index_space', 'semantics', 'complete_action_id', 'allows_empty_choice'})
SEMANTIC_FIELDS = frozenset({'index', 'option_type', 'select_context', 'select_type', 'semantic_key', 'source_area', 'source_index', 'source_card_id', 'source_card_name', 'target_area', 'target_index', 'target_card_id', 'target_card_name', 'attack_id'})


def native_game_binding(family: str, canonical_native: str, source_order: int | None) -> Payload:
    # This is the exact r21 disambiguation format emitted by the hash-bound
    # build_value_q_canonical owner (lines 935–946), not a prefix/adjacency join.
    original = canonical_native
    if family == 'r21_staryu' and ':game_index=' in canonical_native:
        original, _, index = canonical_native.rpartition(':game_index=')
        if not original or type(source_order) is not int or index != str(source_order):
            raise ValueError('native_game_disambiguation_conflict')
    return {'canonical_native_game_id': canonical_native, 'raw_game_id': original, 'source_game_index': source_order, 'disambiguation': 'exact_r21_game_index_suffix' if original != canonical_native else 'exact_native_game_id'}


def action_envelope(legal: list[Payload], chosen: Payload) -> Payload:
    if set(chosen) - ACTION_FIELDS:
        raise ValueError('unknown_action_envelope')
    indices = chosen.get('indices')
    if not isinstance(indices, list) or any(type(value) is not int or value < 0 or value >= len(legal) for value in indices) or len(set(indices)) != len(indices):
        raise ValueError('invalid_chosen_positions')
    if chosen.get('index_space', 'option_position') != 'option_position':
        raise ValueError('unsupported_index_space')
    semantics = chosen.get('semantics')
    if semantics is not None:
        if not isinstance(semantics, list) or len(semantics) != len(indices):
            raise ValueError('chosen_semantic_cardinality')
        for position, semantic in zip(indices, semantics):
            if not isinstance(semantic, dict) or any(key not in legal[position] or legal[position][key] != value for key, value in semantic.items()):
                raise ValueError('chosen_semantic_mismatch')
    complete = chosen.get('complete_action_id')
    if complete is not None:
        if not isinstance(complete, str):
            raise ValueError('invalid_complete_action_identity')
        decoded = orjson.loads(complete)
        if not isinstance(decoded, dict) or set(decoded) != {'indices', 'schema', 'semantics'} or decoded['schema'] != 'deepmew.complete_legal_option_selection.v0' or decoded['indices'] != indices:
            raise ValueError('complete_action_identity_mismatch')
        if not isinstance(decoded['semantics'], list) or len(decoded['semantics']) != len(indices):
            raise ValueError('complete_action_semantic_cardinality')
        for position, semantic in zip(indices, decoded['semantics']):
            if not isinstance(semantic, dict) or any(legal[position].get(key) != value for key, value in semantic.items()):
                raise ValueError('complete_action_semantic_mismatch')
    optional = chosen.get('allows_empty_choice')
    if optional is not None and (type(optional) is not bool or not indices and not optional):
        raise ValueError('optional_empty_metadata_conflict')
    return {'indices': indices, 'index_space': 'option_position', 'semantics': semantics, 'source_complete_action_id': complete, 'allows_empty_choice': optional}


def normalize_reference(transition: Payload, sequence: Payload, source_files: Mapping[str, Payload], source_split: str, effective_split: str, mask: bool, unit_id: str) -> Payload:
    for name in ('decision_id', 'canonical_game_id', 'actor', 'actor_decision_index', 'global_decision_index', 'source_hash', 'source_schema', 'state_before'):
        if transition.get(name) != sequence.get(name):
            raise ValueError('sequence_transition_mismatch')
    actor = transition['actor']
    native = sequence['source_game_id']
    if not isinstance(native, str) or transition['canonical_game_id'] != f"{transition['source_id']}:{unit_id}:{native}":
        raise ValueError('canonical_native_game_binding_mismatch')
    game_binding = native_game_binding(transition['source_id'], native, sequence.get('source_order'))
    if type(actor) is not int or actor not in (0, 1) or sequence['control_owner'] != actor or transition['information_mode'] != 'actor_visible' or sequence['information_mode'] != 'actor_visible':
        raise ValueError('invalid_actor_visibility_metadata')
    outcome = transition['terminal_actor_relative_result']
    if type(outcome) not in (int, float) or not math.isfinite(outcome) or not 0 <= outcome <= 1:
        raise ValueError('invalid_terminal_target')
    reference = transition['state_before']['source_observation']
    if not isinstance(reference, dict) or reference.get('source_field') not in ('raw_observation', 'state_json') or type(reference.get('source_line')) is not int or reference['source_line'] < 1:
        raise ValueError('invalid_observation_reference')
    source = source_files[reference['source_path']]
    if source['sha256'] != transition['source_hash']:
        raise ValueError('observation_source_binding_mismatch')
    legal = transition['legal_action_set']
    if not isinstance(legal, list) or not legal or any(not isinstance(option, dict) for option in legal):
        raise ValueError('missing_semantic_legal_set')
    semantic = transition['source_schema'] != 'deepmew.replay_action_row.v1'
    if semantic and any(set(option) - SEMANTIC_FIELDS for option in legal):
        raise ValueError('unknown_semantic_option_payload')
    chosen = action_envelope(legal, transition['chosen_action'])
    reason = 'raw_kaggle_prior_request_repair_required' if not semantic else 'observation_visibility_and_bounds_verified_on_resolution'
    key = (transition['source_hash'], reference['source_line'], actor, transition['global_decision_index'])
    frame_id = 'frame2ref:sha256:' + hashlib.sha256(orjson.dumps(key)).hexdigest()
    after = transition.get('state_after')
    after_reference = after.get('source_observation') if isinstance(after, dict) else None
    return {'schema': REFERENCE_SCHEMA, 'frame_id': frame_id, 'episode_id': transition['canonical_game_id'], 'source_family': transition['source_id'], 'source_unit_id': unit_id, 'native_game_binding': game_binding, 'source_schema': transition['source_schema'], 'source_decision_id': transition['decision_id'], 'source_split': source_split, 'effective_split': effective_split, 'actor_seat': actor, 'actor_order': transition['actor_decision_index'], 'global_order': transition['global_decision_index'], 'state_before': {**reference, 'source_sha256': transition['source_hash']}, 'legal_actions': legal, 'chosen_action': chosen, 'action_result': {'state_after_reference': after_reference, 'state_after_status': transition['state_after_status'], 'next_same_actor_source_reference': transition['next_same_actor_state'], 'terminal': transition['terminal'], 'logical_horizon': transition['transition_horizon_logical_decisions'], 'immediate_reward': None, 'td_transition': False}, 'terminal_target': outcome, 'public_history': sequence['public_event_history'], 'capabilities': {'semantic_action_metadata': True, 'snapshot_bc': False, 'temporal_bc': False, 'td_transition': False, 'requires_source_resolution': True, 'reason': reason}, 'target_mask': {'source_target_eligible': mask, 'reason': None if mask else 'historical_cross_split_state_action_duplicate', 'model_training_authorized': False}, 'encoding_reference': {'feature_dim': transition['state_before']['feature_dim'], 'feature_row': transition['state_before']['feature_row'], 'representation': transition['state_before']['representation']}, 'v1_link': {'status': 'unavailable', 'reason': 'no exact observation/legal/chosen identity join performed'}}
