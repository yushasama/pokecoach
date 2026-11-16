from pathlib import Path
import sqlite3

import pytest

from deepmew_ex.experiments.v2.boundary import acquire_boundary


def test_boundary_preserves_index_and_other_training_files(tmp_path: Path) -> None:
    index = tmp_path / 'snapshot-index.sqlite3'
    owner = sqlite3.connect(index, timeout=0.05)
    owner.executescript('PRAGMA journal_mode=OFF; CREATE TABLE sources(path TEXT); INSERT INTO sources VALUES ("original");')
    before = index.read_bytes()
    barrier = acquire_boundary(index)
    ledger = sqlite3.connect(tmp_path / 'consumption.sqlite3')
    ledger.executescript('CREATE TABLE consumed(id INTEGER); INSERT INTO consumed VALUES (1);')
    (tmp_path / 'temporal-checkpoint.pt').write_bytes(b'checkpoint')
    assert ledger.execute('SELECT COUNT(*) FROM consumed').fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError, match='locked'):
        owner.execute('SELECT path FROM sources').fetchall()
    barrier.rollback()
    barrier.close()
    assert owner.execute('SELECT path FROM sources').fetchall() == [('original',)]
    owner.close()
    ledger.close()
    assert index.read_bytes() == before
