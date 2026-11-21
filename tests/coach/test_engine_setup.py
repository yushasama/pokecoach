import hashlib
import io
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from deepmew_ex.coach import engine_setup


def test_server_with_missing_engine_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_setup, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(engine_setup, 'native_library', lambda system, machine: 'cg.dll')
    with patch.object(engine_setup, 'engine_api_path', side_effect=ModuleNotFoundError), patch.object(engine_setup, 'download') as fetch:
        with pytest.raises(ModuleNotFoundError, match='python -m deepmew_ex.coach.engine_setup'):
            runpy.run_module('deepmew_ex.coach.server', run_name='__main__')
        fetch.assert_not_called()
    assert not (tmp_path / 'outputs').exists()


def test_first_install_and_offline_restart(tmp_path: Path) -> None:
    source = b'# test SDK'
    binary = b'test DLL'
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('cg/api.py', source)
    with patch.object(engine_setup, 'native_library', return_value='cg.dll'), patch.object(engine_setup, 'PYTHON_HASHES', {'api.py': hashlib.sha256(source).hexdigest()}), patch.dict(engine_setup.NATIVE_HASHES, {'cg.dll': hashlib.sha256(binary).hexdigest()}), patch.object(engine_setup, 'download', side_effect=[buffer.getvalue(), b'damaged', buffer.getvalue(), binary]) as fetch:
        with pytest.raises(ValueError, match='checksum mismatch'):
            engine_setup.coach_sdk(tmp_path, install=True)
        assert not (tmp_path / 'v8/cg.dll/cg').exists()
        root = engine_setup.coach_sdk(tmp_path, install=True)
        assert (root / 'cg/cg.dll').read_bytes() == binary
        assert engine_setup.coach_sdk(tmp_path, install=True) == root
        assert fetch.call_count == 4
        (root / 'cg/api.py').write_bytes(b'changed')
        with pytest.raises(ValueError, match='checksum mismatch'):
            engine_setup.coach_sdk(tmp_path, install=False)
