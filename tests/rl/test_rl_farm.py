"""End-to-end farm smoke: real native games, worker-kind routing, lean workers stay torch-free."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.rl.farm import Farm, FarmSettings, InferenceServer, Net
from deepmew_ex.rl.ppo import ValueHead

DECK = PROJECT_ROOT / 'source/.research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv'
NOTEBOOKS = [PROJECT_ROOT / f'league/teachers/teacher_{name}.ipynb' for name in ('abomasnow', 'iono', 'dragapult', 'lucario')]


@pytest.mark.skipif(not (PROJECT_ROOT / 'source/cg/api.py').is_file() or not all(path.is_file() for path in NOTEBOOKS), reason='needs the competition engine and the Kaggle sample notebooks (not redistributable; see CREDITS.md)')
def test_farm_routes_checkpoint_opponents_to_torch_workers_and_keeps_lean_workers_torch_free(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TemporalPolicy(hidden=32, embedding=8, layers=1, heads=4).eval()
    server = InferenceServer({'current': Net(model, ValueHead(32).eval())}, seed=0)
    deck = tuple(int(card) for card in DECK.read_text().split())
    farm = Farm(FarmSettings(str(PROJECT_ROOT), str(tmp_path / 'runtime'), deck, None), workers=3, external_workers=0, torch_workers=1)
    try:
        jobs = [{'opponent': opponent, 'seat': index % 2, 'seed': 100 + index, 'tag': 'current', 'public_event_mode': True} for index, opponent in enumerate(['d1', 'ref_abomasnow', 'ref_iono', 'd1', 'ref_lucario', 'ref_dragapult'])]
        results, stats = farm.run(jobs, server)
    finally:
        farm.close()
    assert stats['games'] == len(jobs)
    assert all(result['outcome'] in ('win', 'loss', 'draw') for result in results), [result['error'] for result in results]
    assert all(result['trajectory'] is not None and result['trajectory'].decisions for result in results)
    # D1 needs torch; reference notebooks ran in workers that never imported it.
    assert all(result['torch_loaded'] for result in results if result['job']['opponent'] == 'd1')
    assert any(not result['torch_loaded'] for result in results if result['job']['opponent'] != 'd1')
    assert not server.pools['current'].slots  # every finished game released its cache slot
