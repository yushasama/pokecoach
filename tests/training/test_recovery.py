from __future__ import annotations

from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from deepmew_ex.training.recovery import EMPTY_LEDGER_DIGEST, TrainingLease, extend_ledger_digest, verify_ledger_prefix
from deepmew_ex.experiments.config import PROJECT_ROOT


def test_live_writer_lease_is_exclusive_and_released(tmp_path: Path) -> None:
    run = tmp_path / 'run'
    first = TrainingLease(run)
    try:
        with pytest.raises(ValueError, match='still active'):
            TrainingLease(run, require_existing=True)
    finally:
        first.close()
    TrainingLease(run, require_existing=True).close()


def test_unsealed_recovery_requires_original_lease(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='recorded training lease'):
        TrainingLease(tmp_path / 'absent', require_existing=True)


def test_abrupt_process_exit_releases_writer_lease(tmp_path: Path) -> None:
    run = tmp_path / 'abrupt'
    code = 'import os,sys; from pathlib import Path; from deepmew_ex.training.recovery import TrainingLease; lease=TrainingLease(Path(sys.argv[1])); os._exit(17)'
    result = subprocess.run([sys.executable, '-B', '-c', code, str(run)], cwd=PROJECT_ROOT, capture_output=True, timeout=30)
    assert result.returncode == 17, result.stderr.decode()
    TrainingLease(run, require_existing=True).close()


def test_checkpoint_digest_authenticates_both_targets_and_context() -> None:
    connection = sqlite3.connect(':memory:')
    connection.executescript('CREATE TABLE consumed(epoch,phase,id,game,family,update_id); CREATE TABLE context(epoch,phase,game,actor,requests,events,post_states,unavailable_post_states,update_id); INSERT INTO consumed VALUES (0,"raw","d1","g","f",1); INSERT INTO context VALUES (0,"raw","g",0,1,2,0,1,1);')
    expected = extend_ledger_digest(connection, EMPTY_LEDGER_DIGEST, 1)
    verify_ledger_prefix(connection, 1, expected)
    connection.execute('UPDATE context SET events=1')
    with pytest.raises(ValueError, match='checkpoint identity'):
        verify_ledger_prefix(connection, 1, expected)
    connection.execute('DELETE FROM consumed')
    with pytest.raises(ValueError, match='missing optimizer update'):
        verify_ledger_prefix(connection, 1, expected)
