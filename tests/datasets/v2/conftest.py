from __future__ import annotations

from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.conversion import game_split
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, digest


def synthetic_replay(game_id: str = 'synthetic-golden-0', version: str = '1.32.0') -> Payload:
    def observation(seat: int, step: int) -> Payload:
        players = [{'hand': [{'id': 100 + index}] if index == seat else None, 'prize': [None], 'deckCount': 40, 'active': [], 'discard': []} for index in (0, 1)]
        select = {'type': 0, 'context': 0, 'minCount': 1, 'maxCount': 1, 'option': [{'type': 14}, {'type': 13, 'attackId': 7}], 'effect': None, 'deck': 'PRIVATE-DECK-SENTINEL'}
        if seat == 1:
            select = {'type': 1, 'context': 8, 'minCount': 0, 'maxCount': 2, 'option': [{'type': 3, 'area': 3, 'index': index, 'playerIndex': seat} for index in (0, 1)], 'effect': None}
        return {'current': {'yourIndex': seat, 'players': players, 'turn': step, 'looking': 'PRIVATE-LOOKING-SENTINEL'}, 'select': select, 'step': step, 'logs': [{'type': 4, 'playerIndex': seat, 'cardId': 'PRIVATE-LOG-SENTINEL', 'serial': 999}, {'type': 987, 'playerIndex': seat, 'futureField': 'PRIVATE-UNKNOWN-SENTINEL'}], 'search_begin_input': 'PRIVATE-SIMULATOR-SENTINEL'}
    steps = []
    actions = {(1, 0): [1], (2, 1): [], (3, 0): [0], (4, 1): [0, 1]}
    for step in range(5):
        active = step % 2
        steps.append([{'status': 'DONE' if step == 4 else 'ACTIVE' if seat == active else 'INACTIVE', 'observation': observation(seat, step), 'action': actions.get((step, seat), []), 'reward': (1 if seat else -1) if step == 4 else 0, 'info': {}} for seat in (0, 1)])
    return {'schema_version': 1, 'name': 'cabt', 'version': '1.0.0', 'module_version': version, 'id': game_id, 'configuration': {'seed': 42}, 'statuses': ['DONE', 'DONE'], 'rewards': [-1, 1], 'steps': steps}


def source_row(raw: bytes, path: Path, game_id: str, version: str = '1.32.0') -> Payload:
    return {'family': f'kaggle_cabt_{version}', 'module_version': version, 'path': str(path), 'sha256': digest(raw), 'byte_size': len(raw), 'native_game_id': game_id, 'split': game_split(game_id), 'asset_id': path.name}


@pytest.fixture
def golden_inventory(tmp_path: Path) -> Path:
    sources = []
    for index in range(2):
        game_id = f'synthetic-golden-{index}'
        data = synthetic_replay(game_id, '1.30.2' if index else '1.32.0')
        raw = canonical(data)
        path = tmp_path / f'input-{index}.json'
        path.write_bytes(raw)
        sources.append(source_row(raw, path, game_id, data['module_version']))
    value = {'schema': 'deepmew.ex.golden_inventory.v1', 'purpose': 'golden_validation_only', 'sources': sources, 'evidence': [], 'parser_hashes': binding_hashes()}
    path = tmp_path / 'inventory.json'
    path.write_bytes(canonical(value))
    return path
