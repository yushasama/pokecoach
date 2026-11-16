from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from deepmew_ex.experiments.config import current_output_bytes


def entry_error(monkeypatch: pytest.MonkeyPatch, target: Path, error: type[OSError]) -> None:
    original = os.scandir

    @contextmanager
    def scan(directory: Path | str) -> Iterator[Iterator[os.DirEntry[str]]]:
        with original(directory) as entries:
            def injected() -> Iterator[os.DirEntry[str]]:
                for entry in entries:
                    if Path(entry.path) == target:
                        proxy = Mock(wraps=entry)
                        proxy.stat.side_effect = error('injected filesystem race or denial')
                        yield cast(os.DirEntry[str], proxy)
                    else:
                        yield entry
            yield injected()

    monkeypatch.setattr(os, 'scandir', scan)


def test_other_worker_removing_temp_file_does_not_abort_storage_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stable, temporary = tmp_path / 'checkpoint', tmp_path / 'temporary'
    stable.write_bytes(b'accepted')
    temporary.write_bytes(b'gone')
    entry_error(monkeypatch, temporary, FileNotFoundError)
    assert current_output_bytes(tmp_path) == len(b'accepted')


def test_permission_error_is_not_treated_as_zero_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = tmp_path / 'candidate'
    blocked.write_bytes(b'data')
    entry_error(monkeypatch, blocked, PermissionError)
    with pytest.raises(PermissionError):
        current_output_bytes(tmp_path)


def test_missing_owner_is_not_accepted_as_empty_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        current_output_bytes(tmp_path / 'missing')


def test_nested_output_is_counted_and_fresh_scan_observes_growth(tmp_path: Path) -> None:
    nested = tmp_path / 'nested'
    nested.mkdir()
    (tmp_path / 'first').write_bytes(b'abc')
    child = nested / 'second'
    child.write_bytes(b'defgh')
    assert current_output_bytes(tmp_path) == 8
    child.write_bytes(b'longer-file')
    assert current_output_bytes(tmp_path) == 14


def test_owner_disappearing_during_scan_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.stat
    calls = 0

    def vanished(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        nonlocal calls
        if path == tmp_path:
            calls += 1
            if calls == 2:
                raise FileNotFoundError('output owner vanished')
        return original(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, 'stat', vanished)
    with pytest.raises(FileNotFoundError, match='owner vanished'):
        current_output_bytes(tmp_path)
