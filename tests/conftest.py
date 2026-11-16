from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from deepmew_ex.experiments.config import ExperimentConfig, OutputPaths, PROJECT_ROOT
from deepmew_ex.experiments.runner import run_s0
from deepmew_ex.registry.store import verify_run


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--s0-run-path', type=Path, default=None, help='Verify an existing bounded S0 run instead of training a test fixture')


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        paths = OutputPaths(PROJECT_ROOT / 'outputs')
        config.option.basetemp = str(paths.test_results / 'temporary' / uuid.uuid4().hex)
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope='session')
def s0_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    existing = request.config.getoption('--s0-run-path')
    if existing is not None:
        verify_run(existing)
        return existing
    config = ExperimentConfig(output_root=str(tmp_path_factory.mktemp('s0') / 'outputs'))
    return run_s0(config, 'test-s0')
