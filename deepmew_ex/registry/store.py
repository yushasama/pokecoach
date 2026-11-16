"""Single local output owner: exclusive allocation, write-once files and seals."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.records import validate_hash


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_new(path: Path, content: bytes) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    lock = path.with_name(f'.{path.name}.lock')
    # Exclusive cooperative publisher lock works on Windows without hard-link privileges.
    handle_lock = lock.open('xb')
    try:
        with handle_lock:
            if path.exists():
                raise FileExistsError(path)
            with temporary.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunStore:
    path: Path

    @classmethod
    def continue_attempt(cls, paths: OutputPaths, family: str, variant: str, run_id: str, parent: Path) -> RunStore:
        manifest = verify_run(parent)
        if manifest['status'] != 'completed':
            raise ValueError('resume requires a completed verified parent')
        store = cls.allocate(paths, family, variant, run_id, str(parent.resolve()))
        store.write('resume.json', {'relation': 'continued_training_from', 'parent_run_id': manifest['run']['run_id'], 'parent_manifest_sha256': file_hash(parent / 'manifest.json'), 'parent_artifacts': manifest['artifacts'], 'optimizer_rng_restored': False, 'scope': 'new attempt allocation only; training resume is unsupported in S0'})
        return store

    @classmethod
    def allocate(cls, paths: OutputPaths, family: str, variant: str, run_id: str, continued_from: str | None = None, specification: Mapping[str, object] | None = None) -> RunStore:
        path = paths.run(family, variant, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.parent / '.allocation.lock'
        handle = lock.open('xb')
        try:
            with handle:
                if specification is not None:
                    for existing in path.parent.glob('*/model-spec.json'):
                        if json.loads(existing.read_text(encoding='utf-8')) != dict(specification):
                            raise ValueError('variant is already bound to another model specification; use a new variant/version')
                path.mkdir()  # Collision is never overwrite or resume.
                for directory in ('checkpoints', 'evaluations'):
                    (path / directory).mkdir()
                store = cls(path)
                store.write('allocation.json', {'run_id': run_id, 'continued_training_from': continued_from, 'status': 'allocated', 'optimizer_rng_restored': False})
                if specification is not None:
                    store.write('model-spec.json', dict(specification))
                return store
        finally:
            lock.unlink(missing_ok=True)

    def write_bytes(self, relative: str, data: bytes) -> str:
        if (self.path / 'manifest.json').exists():
            raise ValueError('sealed run is immutable')
        path = (self.path / relative).resolve()
        if not path.is_relative_to(self.path.resolve()) or path == self.path.resolve():
            raise ValueError('output path traversal')
        atomic_new(path, data)
        return digest(data)

    def write(self, relative: str, value: object) -> str:
        return self.write_bytes(relative, canonical(value))

    def seal(self, fields: Mapping[str, object], status: str = 'completed') -> dict[str, object]:
        if status not in ('completed', 'failed'):
            raise ValueError('invalid terminal status')
        files = {file.relative_to(self.path).as_posix(): file_hash(file) for file in sorted(self.path.rglob('*')) if file.is_file() and file != self.path / 'manifest.json'}
        manifest = {**fields, 'status': status, 'files_sha256': files}
        self.write('manifest.json', manifest)
        return manifest


def verify_run(path: Path) -> dict[str, object]:
    if not (path / 'manifest.json').is_file():
        raise ValueError(f'interrupted/incomplete run: {path}')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    expected = manifest['files_sha256']
    actual = {file.relative_to(path).as_posix(): file_hash(file) for file in path.rglob('*') if file.is_file() and file != path / 'manifest.json'}
    for value in expected.values():
        validate_hash(value)
    if expected != actual:
        raise ValueError('run integrity mismatch: missing, changed or unexpected files')
    return manifest
