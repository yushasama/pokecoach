"""Rebuildable references; sealed run manifests remain authoritative."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.store import canonical, file_hash, verify_run


def lookup_run(output_root: Path, identity: str) -> Path:
    matches: list[Path] = []
    for path in sorted(output_root.glob('*/*/*')):
        if path.is_relative_to(OutputPaths(output_root).test_results):
            continue
        if not (path / 'manifest.json').is_file():
            continue
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest.get('run'), dict):
            continue
        run = manifest.get('run', {})
        ids = {run.get('run_id'), run.get('model_id'), run.get('model_spec_id'), manifest.get('evaluation', {}).get('evaluation_id')}
        for artifact in manifest.get('artifacts', []):
            ids.update((artifact['artifact_id'], artifact['sha256']))
        if identity in ids:
            verify_run(path)
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f'unknown or ambiguous registered identity: {identity}; matches={[str(p) for p in matches]}')
    return matches[0]


def rebuild(output_root: Path) -> dict[str, list[dict[str, object]]]:
    families: dict[str, dict[str, object]] = {}
    models: dict[str, dict[str, object]] = {}
    runs: list[dict[str, object]] = []
    for path in sorted(output_root.glob('*/*/*')):
        if path.is_relative_to(OutputPaths(output_root).test_results):
            continue
        if not path.is_dir():
            continue
        if not (path / 'manifest.json').is_file():
            runs.append({'run_id': path.name, 'status': 'incomplete', 'manifest': None, 'allocation': str(path / 'allocation.json')})
            continue
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        run = manifest.get('run')
        if not isinstance(run, dict):
            continue
        manifest = verify_run(path)
        runs.append({**run, 'manifest': str(path / 'manifest.json'), 'manifest_sha256': file_hash(path / 'manifest.json')})
        families[str(run['family_id'])] = {'family_id': run['family_id'], 'source_manifest': str(path / 'manifest.json')}
        if manifest['status'] == 'completed':
            model = manifest['model']
            if model['model_id'] in models:
                raise ValueError('duplicate logical model ID')
            models[model['model_id']] = {**model, 'specification': str(path / 'model-spec.json'), 'source_manifest': str(path / 'manifest.json')}
    return {'families': list(families.values()), 'models': list(models.values()), 'runs': runs}


def publish_indexes(output_root: Path, registry_root: Path) -> dict[str, list[dict[str, object]]]:
    indexes = rebuild(output_root)
    registry_root.mkdir(parents=True, exist_ok=True)
    for name, records in indexes.items():
        temporary = registry_root / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            temporary.write_bytes(canonical({'schema_version': 1, 'rebuild_command': 'python -m deepmew_ex.cli registry rebuild', 'records': records}))
            os.replace(temporary, registry_root / f'{name}.json')
        finally:
            temporary.unlink(missing_ok=True)
    return indexes
