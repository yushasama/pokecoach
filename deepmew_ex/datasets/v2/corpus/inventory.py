"""Freeze catalog, sidecars, masks and every available named raw replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import atomic_new, canonical, digest, file_hash


def bound(path: Path, expected: str | None = None) -> Payload:
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError(f'input manifest hash mismatch: {path.name}')
    return {'path': str(path.resolve()), 'sha256': actual, 'byte_size': path.stat().st_size}


def freeze_inventory(catalog: Path, raw_registry: Path, replay_roots: Iterable[Path], v1_assignments: Path, golden_inventory: Path, output: Path) -> Payload:
    root = catalog.resolve().parent
    catalog_data = json.loads(catalog.read_bytes())
    splits = root / 'SPLIT_MANIFEST.json'
    split_data = json.loads(splits.read_bytes())
    if split_data['catalog_sha256'] != file_hash(catalog) or split_data['status'] != 'accepted':
        raise ValueError('canonical split manifest is not bound/accepted')
    exclusions_path = Path(split_data['row_exclusion_manifest_path'])
    exclusions = json.loads(exclusions_path.read_bytes())
    masks = {(part['source_id'], part['unit_id'], part['part']): part for part in exclusions['parts']}
    with gzip.open(v1_assignments, 'rt', encoding='utf-8') as handle:
        v1 = {row['canonical_game_id']: row['split'] for row in json.load(handle)}
    golden = json.loads(golden_inventory.read_bytes())
    golden_games = {row['native_game_id'] for row in golden['sources']}
    effective = dict(split_data['assignment'])
    raw_splits: dict[str, str] = {}
    for game, original in list(effective.items()):
        frozen = v1.get(game)
        # A pre-existing confirmation/test/calibration assignment never becomes train.
        ranks = {'train': 0, 'validation': 1, 'calibration': 2, 'test': 3, 'holdout': 4, 'golden_validation': 5}
        if frozen is not None and ranks[frozen] > ranks[original]:
            effective[game] = frozen
        if game.startswith('kaggle_ingest:'):
            native = game.rsplit(':', 1)[-1]
            if native in golden_games:
                effective[game] = 'golden_validation'
            previous = raw_splits.get(native)
            raw_splits[native] = max((effective[game], previous or 'train'), key=ranks.__getitem__)
    raw_splits.update({game: 'golden_validation' for game in golden_games})
    sources, parts, evidence = {}, [], []
    for unit in catalog_data['included_units']:
        path = root / unit['manifest_path']
        evidence.append(bound(path, unit['manifest_sha256']))
        manifest = json.loads(path.read_bytes())
        if manifest['status'] != 'completed' or manifest['information_mode'] != 'actor_visible':
            raise ValueError('unsupported canonical conversion unit')
        for source in manifest['source_fingerprint']:
            location = str(Path(source['path']).resolve())
            entry = {'path': location, 'sha256': source['sha256'], 'byte_size': source['size']}
            if location in sources and sources[location] != entry:
                raise ValueError('conflicting source file bindings')
            sources[location] = entry
        for part in manifest['parts']:
            mask = masks[(unit['source_id'], unit['unit_id'], part['name'])]
            if mask['input_transitions_sha256'] != part['files_sha256'][part['name'] + '.transitions.jsonl.gz']:
                raise ValueError('target mask does not bind exact transition bytes')
            files = {name: {'path': str((path.parent / name).resolve()), 'sha256': sha, 'byte_size': (path.parent / name).stat().st_size} for name, sha in part['files_sha256'].items()}
            target_mask = exclusions_path.parent / mask['mask_path']
            parts.append({'part_id': digest(f"{unit['source_id']}:{unit['unit_id']}:{part['name']}".encode())[:24], 'source_id': unit['source_id'], 'unit_id': unit['unit_id'], 'name': part['name'], 'rows': part['rows'], 'files': files, 'target_mask': bound(target_mask, mask['mask_sha256']), 'excluded_target_rows': mask['excluded_rows']})
    # The registry is a precise historical binding; discover additional later replay
    # files only inside the explicitly configured replay shelves.
    registry = json.loads(raw_registry.read_bytes())
    replay_paths = {str(Path(row['path']).resolve()): row['sha256'] for row in registry['episodes']}
    for replay_root in replay_roots:
        for path in replay_root.glob('**/episode-*-replay.json'):
            replay_paths.setdefault(str(path.resolve()), None)
    replays, seen = [], {}
    for location, expected in sorted(replay_paths.items()):
        path = Path(location)
        if not path.is_file():
            replays.append({'path': location, 'sha256': expected, 'status': 'unavailable'})
            continue
        row = bound(path, expected)
        row['status'] = 'available'
        if row['sha256'] in seen:
            row['alias_of'] = seen[row['sha256']]
        else:
            seen[row['sha256']] = location
        replays.append(row)
    evidence += [bound(catalog), bound(splits), bound(exclusions_path, split_data['row_exclusion_manifest_sha256']), bound(raw_registry), bound(v1_assignments), bound(golden_inventory)]
    result = {'schema': 'deepmew.ex.full_corpus_inventory.v1', 'source_mode': 'immutable_reference_backed', 'catalog': bound(catalog), 'catalog_counts': catalog_data['totals'], 'family_counts': {family: value['stats']['rows_kept'] for family, value in catalog_data['sources'].items()}, 'catalog_exclusions': catalog_data['unit_exclusions'], 'source_assignments': split_data['assignment'], 'effective_assignments': effective, 'raw_splits': raw_splits, 'sources': list(sources.values()), 'parts': parts, 'replays': replays, 'evidence': evidence, 'parser_hashes': binding_hashes(), 'legacy_excluded_target_rows': exclusions['excluded_target_rows'], 'limitations': ['reference-backed observations remain in read-only source storage', 'canonical Kaggle state references require prior-request repair; never admitted by adjacency', 'whole-corpus temporal/TD training is not authorized or claimed']}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_new(output, canonical(result))
    return result
