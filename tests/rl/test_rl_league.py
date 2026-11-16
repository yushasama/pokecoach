from __future__ import annotations

import numpy as np
import pytest

from deepmew_ex.rl.league import League
from deepmew_ex.rl.train import RLConfig


def test_pfsp_weights_caps_and_snapshot_share() -> None:
    league = League(floor=0.15, snapshot_share=0.2, max_snapshots=2)
    for identity in ('d1', 'ref', 'ext'):
        league.add_outside(identity, 0.1 if identity == 'ext' else None)
    league.members['d1'].win_rate, league.members['ref'].win_rate, league.members['ext'].win_rate = 0.8, 0.3, 0.0
    shares = league.shares()
    assert shares['ext'] == pytest.approx(0.1)  # PFSP would give it the most mass; the cap binds
    assert shares['ref'] / shares['d1'] == pytest.approx((0.15 + 0.7 ** 2) / (0.15 + 0.2 ** 2))
    assert sum(shares.values()) == pytest.approx(1.0)
    assert league.add_snapshot('a') == [] and league.add_snapshot('b') == [] and league.add_snapshot('c') == ['a']
    shares = league.shares()
    assert sum(value for key, value in shares.items() if key.startswith('snapshot:')) == pytest.approx(0.2)
    assert sum(shares.values()) == pytest.approx(1.0)
    schedule = league.schedule(100, np.random.default_rng(0))
    assert len(schedule) == 100 and schedule.count('ext') == 10
    restored = League.from_state(league.state())
    assert restored.shares() == pytest.approx(shares)


def test_league_ema_moves_toward_results_and_config_rejects_snapshot_dominance() -> None:
    league = League(ema=0.5)
    league.add_outside('d1')
    league.update('d1', 1.0)
    assert league.members['d1'].win_rate == pytest.approx(0.75) and league.members['d1'].games == 1
    config = RLConfig.load(__import__('pathlib').Path('experiments/deepmew_ex/rl-league-pilot.json'))
    config.validate()
    with pytest.raises(ValueError):
        type(config)(**{**config.__dict__, 'league': type(config.league)(snapshot_share=0.6)}).validate()
