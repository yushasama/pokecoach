"""Inspect complete V2 references or resolve bounded actor-visible snapshots."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import state_view
from deepmew_ex.datasets.v2.corpus.records import REFERENCE_SCHEMA
from deepmew_ex.datasets.v2.schema import Episode, GameState, Payload, payload
from deepmew_ex.registry.store import file_hash, verify_run


@dataclass(frozen=True)
class ResolvedSnapshot:
    frame_id: str
    episode_id: str
    effective_split: str
    state: GameState
    legal_actions: Payload
    target_action: Payload
    source_target_eligible: bool | None
    temporal_bc: bool = False
    td_transition: bool = False


@dataclass(frozen=True)
class CorpusEpisode:
    episode: Episode
    effective_split: str
    model_training_authorized: bool = False


def compressed_rows(path: Path) -> Iterator[Payload]:
    with path.open('rb') as handle:
        with zstandard.ZstdDecompressor().stream_reader(handle) as stream:
            for line in io.BufferedReader(stream):
                yield orjson.loads(line)


def reference_rows(path: Path, family: str | None = None, objective: str = 'inspect') -> Iterator[Payload]:
    if objective != 'inspect':
        raise ValueError('reference records require explicit snapshot resolution; temporal/TD objectives are unsupported')
    manifest = verify_run(path)
    if manifest['status'] != 'completed' or manifest['kind'] not in ('v2_full_corpus_port', 'v2_snapshot_supplement'):
        raise ValueError('incomplete full V2 port')
    if manifest['kind'] == 'v2_snapshot_supplement':
        supplement = orjson.loads((path / 'supplement.json').read_bytes())
        for result in supplement['results']:
            if family is None or family == result['source_family']:
                yield from compressed_rows(path / f"{result['source_sha256']}.frames.jsonl.zst")
        return
    report = orjson.loads((path / 'port.json').read_bytes())
    inventory = orjson.loads((path / 'inventory.json').read_bytes())
    by_id = {result['part_id']: result for result in report['results'] if result['kind'] == 'canonical'}
    for part in inventory['parts']:
        if family is not None and part['source_id'] != family:
            continue
        child = Path(by_id[part['part_id']]['path'])
        if not child.resolve().is_relative_to(path.resolve()):
            raise ValueError('part path escapes run')
        yield from compressed_rows(child / 'frames.jsonl.zst')


def resolve_row(frame: Payload, source_row: Payload) -> ResolvedSnapshot:
    if frame['schema'] != REFERENCE_SCHEMA or frame['source_schema'] == 'deepmew.replay_action_row.v1':
        raise ValueError('quarantined/legacy Kaggle reference requires raw prior-request repair')
    reference = frame['state_before']
    binding = frame['native_game_binding']
    if source_row.get('game_id') != binding['raw_game_id'] or binding['source_game_index'] is not None and source_row.get('game_index') != binding['source_game_index']:
        raise ValueError('source native game identity mismatch')
    observation = source_row.get(reference['source_field'])
    if not isinstance(observation, dict):
        raise ValueError('referenced semantic observation missing')
    actual = hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if actual != reference['content_sha256']:
        raise ValueError('observation content hash mismatch')
    if source_row.get('seat') != frame['actor_seat'] or source_row.get('decision_index') != frame['global_order']:
        raise ValueError('source actor/decision coordinate mismatch')
    contract = load_contract()
    mode = source_row.get('information_mode')
    if mode is not None and mode not in contract.deployable_modes:
        raise ValueError('source information mode is not deployable')
    projected = payload(contract.project_actor(observation))
    semantic = contract.semantic_options(projected)
    # Card-name annotations are audit metadata, not canonical action semantics.
    clean = lambda value: {key: item for key, item in value.items() if key not in ('source_card_name', 'target_card_name')}
    if [clean(value) for value in semantic] != [clean(value) for value in frame['legal_actions']]:
        raise ValueError('source semantic legal set mismatch')
    state = state_view(observation, frame['actor_seat'], frame['global_order'], contract)
    legal = contract.build_legal(state.state_id, projected['select'])
    chosen = contract.reconstruct_chosen(legal, [frame['chosen_action']['indices']])
    if not chosen.complete or chosen.failures or legal.failures:
        raise ValueError('source bounds/order/chosen rejection')
    source_choice = source_row.get('chosen_action')
    if not isinstance(source_choice, dict) or source_choice.get('indices') != frame['chosen_action']['indices']:
        raise ValueError('source choice mismatch')
    optional = frame['chosen_action']['allows_empty_choice']
    if optional is not None and optional != (projected['select']['minCount'] == 0):
        raise ValueError('optional-empty bound conflict')
    return ResolvedSnapshot(frame['frame_id'], frame['episode_id'], frame['effective_split'], state, payload(asdict(legal)), payload(asdict(chosen)), frame['target_mask']['source_target_eligible'])


def resolve_references(frames: Iterable[Payload], max_references: int = 1024) -> Iterator[ResolvedSnapshot]:
    if type(max_references) is not int or not 1 <= max_references <= 4096:
        raise ValueError('snapshot resolution batch must be in [1, 4096]')
    groups: dict[str, dict[int, list[Payload]]] = {}
    sources: dict[str, str] = {}
    for ordinal, frame in enumerate(frames):
        if ordinal >= max_references:
            raise ValueError('snapshot resolution batch exceeded before source read')
        if frame['schema'] != REFERENCE_SCHEMA or frame['source_schema'] == 'deepmew.replay_action_row.v1':
            raise ValueError('unsupported reference before source read')
        reference = frame['state_before']
        path = reference['source_path']
        if path in sources and sources[path] != reference['source_sha256']:
            raise ValueError('conflicting source identity')
        sources[path] = reference['source_sha256']
        groups.setdefault(path, {}).setdefault(reference['source_line'], []).append(frame)
    for location, targets in groups.items():
        path = Path(location)
        if file_hash(path) != sources[location]:
            raise ValueError('source file hash mismatch')
        opener = gzip.open if path.suffix == '.gz' else Path.open
        with opener(path, 'rb') as handle:
            for ordinal, line in enumerate(handle, 1):
                if ordinal in targets:
                    source = orjson.loads(line)
                    for frame in targets.pop(ordinal):
                        yield resolve_row(frame, source)
                if not targets:
                    break
        if targets or file_hash(path) != sources[location]:
            raise ValueError('missing reference or source mutation')


def raw_episodes(path: Path, objective: str = 'temporal_bc', split: str | None = None) -> Iterator[CorpusEpisode]:
    if objective not in ('snapshot_bc', 'temporal_bc'):
        raise ValueError('unsupported raw V2 objective before dataset read')
    manifest = verify_run(path)
    if manifest['status'] != 'completed' or manifest['kind'] != 'v2_full_corpus_port':
        raise ValueError('incomplete full corpus port')
    report = orjson.loads((path / 'port.json').read_bytes())
    from deepmew_ex.datasets.v2.reader import validate_episode
    for result in report['results']:
        if result['kind'] != 'raw' or result['status'] != 'admitted_actor_visible_episode' or result['native_game_id'] in report['raw_native_game_conflicts']:
            continue
        if split is not None and split != result['effective_split']:
            continue
        child = Path(result['path'])
        if not child.resolve().is_relative_to(path.resolve()):
            raise ValueError('raw episode path escapes run')
        encoded = zstandard.ZstdDecompressor().decompress((child / 'episode.json.zst').read_bytes())
        episode = Episode.from_bytes(encoded)
        validate_episode(episode)
        if episode.source_native_game_id != result['native_game_id'] or episode.source_sha256 != result['source']['sha256']:
            raise ValueError('raw episode manifest binding mismatch')
        yield CorpusEpisode(episode, result['effective_split'])
