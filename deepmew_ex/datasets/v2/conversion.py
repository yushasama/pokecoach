"""Hash-bound prior-request/current-action conversion, one raw episode at a time."""

from __future__ import annotations

import copy
import json
import platform
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from deepmew_ex.datasets.v2.config import GoldenConfig
from deepmew_ex.datasets.v2.contract import BoundContract, binding_hashes, load_contract
from deepmew_ex.datasets.v2.schema import ActionResult, DecisionFrame, Episode, GameState, LegalActionSet, Payload, PublicEvent, SCHEMA, identity, payload
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash
from deepmew_ex.experiments.config import PROJECT_ROOT


def game_split(native_game_id: str) -> str:
    # Golden partitions are regression fixtures, never authorization to fit a model.
    return 'golden_validation' if int(digest(native_game_id.encode()), 16) % 5 == 0 else 'golden_development'


def classified_fields(value: Payload, schema: str, contract: BoundContract) -> None:
    if set(value) - contract.observation_fields[schema]:
        raise ValueError(f'unclassified {schema} extension')
    containers = {'State': {'players', 'stadium', 'looking'}, 'PlayerState': {'active', 'bench', 'discard', 'prize', 'hand'}, 'Pokemon': {'energies', 'energyCards', 'tools', 'preEvolution'}, 'Card': set(), 'SelectData': {'option', 'contextCard', 'effect', 'deck'}}
    for field, item in value.items():
        if field not in containers[schema] and item is not None and type(item) not in (int, bool):
            raise ValueError(f'unclassified {schema} scalar payload')


def check_cards(values: object, schema: str, contract: BoundContract) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise ValueError('malformed semantic card list')
    for item in values:
        if item is None:
            continue
        card = payload(item)
        classified_fields(card, schema, contract)
        if 'energies' in card and (not isinstance(card['energies'], list) or any(type(energy) is not int for energy in card['energies'])):
            raise ValueError('malformed energy identities')
        for field in ('energyCards', 'tools', 'preEvolution'):
            if field in card:
                check_cards(card[field], 'Card', contract)


def state_view(observation: object, seat: int, step: int, contract: BoundContract) -> GameState:
    projected = payload(contract.project_actor(observation))
    if set(projected) - {'current', 'select', 'step', 'remainingOverageTime'}:
        raise ValueError('unclassified observation extension')
    current = payload(projected['current'])
    classified_fields(current, 'State', contract)
    if type(current['yourIndex']) is not int or current['yourIndex'] != seat:
        raise ValueError('actor seat identity mismatch')
    public = copy.deepcopy(projected)
    public.pop('select', None)
    players = public['current']['players']
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError('two actor-perspective player states required')
    check_cards(current.get('stadium'), 'Card', contract)
    for player in players:
        classified_fields(player, 'PlayerState', contract)
        for field in ('hand', 'discard', 'prize'):
            check_cards(player.get(field), 'Card', contract)
        for field in ('active', 'bench'):
            check_cards(player.get(field), 'Pokemon', contract)
    if isinstance(projected.get('select'), dict):
        classified_fields(projected['select'], 'SelectData', contract)
        for field in ('contextCard', 'effect'):
            if projected['select'].get(field) is not None:
                classified_fields(payload(projected['select'][field]), 'Card', contract)
    own_hand = players[seat].pop('hand')
    for player in players:
        player.pop('hand', None)
    private = {'hand': own_hand}
    fields = {'shared_public': public, 'actor_private': private, 'actor_seat': seat, 'observation_step': step}
    return GameState(identity('state2', fields), public, private, seat, step)


def public_events(raw_observation: object, seat: int, step: int, episode_id: str, contract: BoundContract) -> tuple[PublicEvent, ...]:
    observation = payload(raw_observation)
    logs = observation.get('logs')
    if not isinstance(logs, list) or any(not isinstance(log, dict) for log in logs):
        raise ValueError('public log snapshot missing or malformed')
    result = []
    for index, log in enumerate(logs):
        if type(log.get('type')) is not int or log.get('playerIndex') is not None and (type(log['playerIndex']) is not int or log['playerIndex'] not in (0, 1)):
            raise ValueError('malformed public event envelope')
        projected = contract.public_logs([log])
        if len(projected) != 1:
            raise ValueError('log event has no classifiable public envelope')
        if any(type(value) not in (int, bool) for value in projected[0].values()):
            raise ValueError('unclassified public event payload')
        fields = {'episode_id': episode_id, 'seat': seat, 'observation_step': step, 'log_index': index, 'payload': projected[0]}
        result.append(PublicEvent(identity('event2', fields), seat, step, index, contract.log_names.get(log.get('type'), 'unknown'), payload(projected[0])))
    return tuple(result)


def convert_episode(raw: bytes, source: Payload, contract: BoundContract, max_steps: int = 2000) -> Episode:
    if digest(raw) != source['sha256']:
        raise ValueError('source hash mismatch')
    data = payload(json.loads(raw))
    source_contract = contract.source_contract(data)
    if source_contract is None or source_contract.module_version != source['module_version']:
        raise ValueError('unsupported replay source contract')
    if data['id'] != source['native_game_id'] or source['split'] != game_split(data['id']):
        raise ValueError('native game or deterministic split mismatch')
    if data.get('statuses') != ['DONE', 'DONE'] or data.get('rewards') not in ([-1, 1], [1, -1], [0, 0]):
        raise ValueError('incomplete game, unsafe final status or unsupported outcome')
    seed = payload(data['configuration'])['seed']
    if type(seed) is not int or seed in (40009, 50021, 60013):
        raise ValueError('missing seed or frozen evaluation seed')
    steps = data['steps']
    if not isinstance(steps, list) or not 3 <= len(steps) <= max_steps:
        raise ValueError('empty/oversized episode')
    for step in steps:
        if not isinstance(step, list) or len(step) != 2:
            raise ValueError('missing seat or source history gap')
        for agent in step:
            if not isinstance(agent, dict) or agent.get('status') not in ('ACTIVE', 'INACTIVE', 'DONE') or agent.get('info') not in ({}, None):
                raise ValueError('source safety/status failure')
    for seat in (0, 1):
        final = steps[-1][seat]
        if final['status'] != 'DONE' or type(final.get('reward')) not in (int, float) or final['reward'] != data['rewards'][seat]:
            raise ValueError('terminal summary conflicts with final actor evidence')
    for step in steps[1:-1]:
        if any(agent['status'] == 'DONE' or type(agent.get('reward')) not in (int, float) or agent['reward'] != 0 for agent in step):
            raise ValueError('intermediate terminal/reward evidence is unsupported')
    episode_id = identity('episode2', {'source_sha256': source['sha256'], 'native_game_id': data['id'], 'schema': SCHEMA})
    capability = {'snapshot_actor_state': True, 'snapshot_bc': True, 'semantic_legal_actions': True, 'exact_chosen_action': True, 'ordered_public_history': True, 'temporal_bc': True, 'exact_post_action_state': True, 'exact_next_event': True, 'exact_next_same_actor': True, 'terminal_outcome': True, 'actor_deck_identity': False, 'opponent_deck_identity': False, 'v1_join': False, 'td_transition': False, 'response_supervision': False, 'privileged_teacher_label': False, 'public_history_semantics': 'ordered per-seat observation log snapshots; repeated batches retained, never deduced global event deltas', 'history_scope': 'complete same-actor request sequence after deck setup; no opponent-private frames', 'usage': 'golden_validation_only'}
    receipt_id = identity('capability2', {'episode_id': episode_id, 'capabilities': capability, 'parser_hashes': contract.hashes})
    states, events, frames = {}, {}, []
    # Every recorded log snapshot is retained, including repeated inactive-seat
    # batches. Views enforce the request-time cutoff and focal-seat boundary.
    for observation_step, step in enumerate(steps):
        for seat in (0, 1):
            for item in public_events(step[seat]['observation'], seat, observation_step, episode_id, contract):
                events[item.event_id] = item
    actor_orders = [0, 0]
    for action_step in range(1, len(steps)):
        for seat in (0, 1):
            previous, resulting = steps[action_step - 1][seat], steps[action_step][seat]
            before = payload(previous['observation'])
            select = before.get('select')
            action = resulting.get('action')
            if previous['status'] != 'ACTIVE' or not isinstance(select, dict):
                # CABT's initial 60-card setup is not a legal-option request.
                if action_step == 1 and select is None and isinstance(action, list) and len(action) == 60 and all(type(item) is int for item in action):
                    continue
                if previous['status'] == 'ACTIVE':
                    raise ValueError('active actor request missing; temporal history gap')
                if action not in ([], None):
                    raise ValueError('action without an active prior request')
                continue
            if not isinstance(action, list) or any(type(item) is not int for item in action):
                raise ValueError('missing or malformed action')
            state = state_view(before, seat, action_step - 1, contract)
            after = state_view(resulting['observation'], seat, action_step, contract)
            # Legal identities are built from the pre-action actor projection.
            projected_select = payload(contract.project_actor(before))['select']
            legal = contract.build_legal(state.state_id, projected_select)
            chosen = contract.reconstruct_chosen(legal, [action])
            if not chosen.complete or chosen.failures or legal.failures:
                raise ValueError('legal/chosen action contract rejection: ' + ','.join(failure.value for failure in chosen.failures))
            state_events = public_events(before, seat, action_step - 1, episode_id, contract)
            states[state.state_id], states[after.state_id] = state, after
            frame_id = identity('frame2', {'episode_id': episode_id, 'action_step': action_step, 'actor_seat': seat, 'state_before_id': state.state_id, 'chosen_action_id': chosen.chosen_action_id})
            proof = {'source_sha256': source['sha256'], 'request': {'step': action_step - 1, 'seat': seat, 'field': 'observation'}, 'action': {'step': action_step, 'seat': seat, 'field': 'action'}, 'result': {'step': action_step, 'seat': seat, 'field': 'observation'}, 'alignment_rule': source_contract.alignment_rule_id, 'legal_set_from': 'prior_request', 'parser_hashes': contract.hashes, 'raw_private_payloads_persisted': False}
            eligibility = {'snapshot_bc': True, 'temporal_bc': True, 'actor_loss': legal.complete_action_count != 1, 'actor_loss_exclusion': 'forced' if legal.complete_action_count == 1 else None, 'td_transition': False, 'td_reason': 'reward/perspective/logical-transition contract not implemented'}
            link = {'status': 'unavailable', 'stable_row_id': None, 'v1_manifest_sha256': None, 'proof_receipt_id': None, 'reason': 'no exact V1 source/game/seat/coordinate/legal/chosen identity join performed'}
            frames.append(DecisionFrame(frame_id, episode_id, len(frames), actor_orders[seat], seat, seat, action_step - 1, action_step, state.state_id, LegalActionSet(legal.legal_action_set_id, payload(projected_select), payload(asdict(legal))), payload(asdict(chosen)), tuple(event.event_id for event in state_events), ActionResult(after.state_id, action_step, None, None), proof, receipt_id, eligibility, link))
            actor_orders[seat] += 1
    if not frames or any(count == 0 for count in actor_orders):
        raise ValueError('empty eligible actor request history')
    for index, frame in enumerate(frames):
        next_actor = next((item.frame_id for item in frames[index + 1:] if item.actor_seat == frame.actor_seat), None)
        frames[index] = replace(frame, action_result=replace(frame.action_result, next_event_frame_id=frames[index + 1].frame_id if index + 1 < len(frames) else None, next_same_actor_frame_id=next_actor))
    return Episode(SCHEMA, episode_id, source['family'], source_contract.source_schema, source['sha256'], data['id'], source['asset_id'], source_contract.module_version, source['split'], seed, tuple(float(item) for item in data['rewards']), 'complete_after_deck_setup', {**capability, 'receipt_id': receipt_id}, tuple(states.values()), tuple(events.values()), tuple(frames))


def run_golden(config: GoldenConfig, progress: Callable[[int, int], None] | None = None) -> Path:
    inventory = config.load()
    contract = load_contract()
    if inventory['parser_hashes'] != contract.hashes:
        raise ValueError('parser binding changed; source review required')
    store = RunStore.allocate(config.paths, 'deepmew_ex', 'v2_golden', config.run_id)
    started = time.perf_counter()
    consumed, episode_files, counts, exclusions = [], [], Counter(), Counter()
    try:
        store.write('config.json', {'inventory': str(config.inventory.resolve()), 'inventory_sha256': file_hash(config.inventory), 'limits': {name: getattr(config, name) for name in ('max_episodes', 'max_episode_bytes', 'max_source_bytes', 'max_output_bytes', 'max_steps')}, 'usage': 'golden_validation_only', 'optimizer_updates': 0})
        store.write('code.json', code_identity(PROJECT_ROOT))
        store.write('environment.json', {'python': platform.python_version(), 'platform': platform.platform(), 'engine_imported': False, 'torch_imported': False})
        store.write('inventory.json', inventory)
        for evidence in inventory['evidence']:
            if file_hash(Path(evidence['path'])) != evidence['sha256']:
                raise ValueError('upstream acceptance evidence changed')
        source_bytes, output_bytes = 0, 0
        games = set()
        for ordinal, source in enumerate(inventory['sources']):
            path = Path(source['path'])
            size = path.stat().st_size
            source_bytes += size
            if size != source['byte_size'] or size > config.max_episode_bytes or source_bytes > config.max_source_bytes:
                raise ValueError('source size mismatch or bound exceeded')
            raw = path.read_bytes()
            episode = convert_episode(raw, source, contract, config.max_steps)
            repeated = convert_episode(raw, source, contract, config.max_steps)
            encoded = episode.to_bytes()
            if encoded != repeated.to_bytes() or Episode.from_bytes(encoded).to_bytes() != encoded:
                raise ValueError('deterministic identity or round-trip failure')
            from deepmew_ex.datasets.v2.reader import validate_episode
            validate_episode(episode)
            if episode.source_native_game_id in games:
                raise ValueError('game split leakage')
            games.add(episode.source_native_game_id)
            output_bytes += len(encoded)
            if output_bytes > config.max_output_bytes:
                raise ValueError('output size bound exceeded')
            name = f'episode-{ordinal:03d}.json'
            output_hash = store.write_bytes(name, encoded)
            episode_files.append({'path': name, 'sha256': output_hash, 'episode_id': episode.episode_id, 'split': episode.split, 'frame_count': len(episode.frames), 'source_sha256': source['sha256']})
            consumed.append({'path': str(path.resolve()), 'sha256': source['sha256'], 'byte_size': size})
            counts[f"{source['module_version']}_episodes"] += 1
            counts[f"{source['module_version']}_frames"] += len(episode.frames)
            counts[f'{episode.split}_episodes'] += 1
            counts['states'] += len(episode.states)
            counts['public_event_observations'] += len(episode.public_events)
            for frame in episode.frames:
                counts['forced'] += not frame.eligibility['actor_loss']
                counts['optional_empty'] += not frame.chosen_action['stages'][0]['source_selected_values']
                counts['multi_select'] += len(frame.chosen_action['stages'][0]['source_selected_values']) > 1
                if frame.eligibility['actor_loss_exclusion']:
                    exclusions[frame.eligibility['actor_loss_exclusion']] += 1
            if progress:
                progress(ordinal + 1, len(inventory['sources']))
        if binding_hashes() != contract.hashes or any(file_hash(Path(item['path'])) != item['sha256'] for item in [*consumed, *inventory['evidence']]):
            raise ValueError('source or parser mutation during conversion')
        if inventory.get('expected_frames_by_version'):
            for version, expected in inventory['expected_frames_by_version'].items():
                if counts[f'{version}_frames'] != expected:
                    raise ValueError('golden acceptance count mismatch')
        root = digest(canonical(episode_files))
        report = {'schema': SCHEMA, 'identity_root_sha256': root, 'repeat_identity_root_sha256': root, 'two_pass_determinism': True, 'counts': dict(counts), 'actor_loss_exclusions': dict(exclusions), 'source_bytes': source_bytes, 'output_bytes': output_bytes, 'episodes': episode_files, 'source_files': consumed, 'optimizer_updates': 0, 'source_mutation': False, 'elapsed_seconds': time.perf_counter() - started, 'v1_joins': 'unavailable; no heuristic links', 'td_transition': False, 'limitations': ['golden fixtures only; no training admission', 'public logs are conservative ordered observation snapshots, not deduplicated omniscient event history', 'Unified R2 not converted; no immediate post-state reconstruction', 'no model/gameplay/strength evidence']}
        store.write('conversion.json', report)
        store.seal({'kind': 'v2_golden_conversion', 'schema': SCHEMA, 'identity_root_sha256': root, 'optimizer_updates': 0})
    except Exception as error:
        # Exception messages may contain source payloads from third-party parsers.
        # Failure receipts intentionally retain only the exception class and ordinal.
        store.write('failure.json', {'error_type': type(error).__name__, 'completed_episodes': len(episode_files), 'optimizer_updates': 0, 'details': 'conversion rejected; source payloads suppressed'})
        store.seal({'kind': 'v2_golden_conversion', 'optimizer_updates': 0}, 'failed')
        raise
    return store.path
