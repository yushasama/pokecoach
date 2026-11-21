"""Install the public CABT v8 SDK for the coach during setup."""

import hashlib
import io
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from deepmew_ex.datasets.v2.contract import engine_api_path
from deepmew_ex.experiments.config import PROJECT_ROOT

SDK_URL = 'https://api.kaggle.com/v1/datasets/download/kiyotah/cg-lib?datasetVersionNumber=8'
NATIVE_URL = 'https://raw.githubusercontent.com/Kaggle/kaggle-environments/03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303/kaggle_environments/envs/cabt/cg'
PYTHON_HASHES = {
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'api.py': '593f1298e52a635f90f8f505a52113e9af114f444c293404e37906f18ee06ced',
    'game.py': '3bd3d4f4a369a11e6d2f5da9094cf15ebc410a2221835e6417b7cff4883f1fc2',
    'sim.py': '1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655',
    'utils.py': '60f29665cee0a88525d6f0383bc45959a6262d16fe35ef380aece1e0ea13c49b',
}
NATIVE_HASHES = {
    'cg.dll': 'eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771',
    'libcg.so': 'd16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7',
    'libcg-arm64.so': '1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2',
    'libcg.dylib': '7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30',
}


def native_library(system: str, machine: str) -> str:
    arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine.lower(), machine.lower())
    libraries = {('Windows', 'x86_64'): 'cg.dll', ('Linux', 'x86_64'): 'libcg.so', ('Linux', 'aarch64'): 'libcg-arm64.so', ('Darwin', 'aarch64'): 'libcg.dylib'}
    if (system, arch) not in libraries:
        raise RuntimeError(f'No CABT binary in this release for {system} {machine}. Use Windows x64, Linux x86-64/ARM64, or Apple Silicon Python.')
    return libraries[system, arch]


def download(url: str) -> bytes:
    limit = 8 * 1024 * 1024
    with urlopen(url, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('CABT download exceeds expected size')
    return data


def verify(package: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'CABT checksum mismatch for {package / name}')


def coach_sdk(cache: Path, *, install: bool) -> Path:
    library = native_library(platform.system(), platform.machine())
    root = cache / 'v8' / library
    package = root / 'cg'
    hashes = {**PYTHON_HASHES, library: NATIVE_HASHES[library]}
    if not package.exists() and install:
        print('Downloading the public CABT game engine for first use...', file=sys.stderr, flush=True)
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as temp:
            staged = Path(temp) / 'cg'
            staged.mkdir()
            with ZipFile(io.BytesIO(download(SDK_URL))) as sdk:
                for name in hashes:
                    if name == library and library != 'libcg.so':
                        data = download(f'{NATIVE_URL}/{library}')
                    else:
                        member = sdk.getinfo(f'cg/{name}')
                        if member.file_size > 4 * 1024 * 1024:
                            raise ValueError(f'Oversized CABT SDK member {name}')
                        data = sdk.read(member)
                    (staged / name).write_bytes(data)
            verify(staged, hashes)
            try:
                staged.rename(package)
            except OSError:
                if not package.is_dir():
                    raise
    if not package.is_dir():
        raise ModuleNotFoundError('Install CABT first: python -m deepmew_ex.coach.engine_setup')
    verify(package, hashes)
    return root


def activate_engine(*, install: bool) -> Path:
    sys.path.insert(0, str(PROJECT_ROOT / 'source'))
    try:
        return engine_api_path().resolve().parents[1]
    except ModuleNotFoundError:
        root = coach_sdk(PROJECT_ROOT / 'outputs/cabt-sdk', install=install)
        sys.path.insert(0, str(root))
        return root


if __name__ == '__main__':
    print(f'CABT ready in {activate_engine(install=True)}')
