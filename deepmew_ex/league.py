"""Bind legacy research code to the explicit portable asset inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict


class LeagueAsset(TypedDict):
    alias: str
    checkpoint: str | None
    bundle: str | None
    decks: list[str]
    status: str


def bind_legacy_source(root: Path) -> Path:
    base = root.resolve(strict=True)
    source = base / 'source'
    sys.path[:0] = [str(source), str(source / 'scripts')]
    from deepmew.rl1_teacher_bc import teachers
    teacher_manifest = json.loads((base / 'league/teachers/manifest.json').read_text(encoding='utf-8'))
    for teacher in teacher_manifest:
        path = base / teacher['portable_notebook']
        if not path.is_file():
            raise FileNotFoundError(path)
        teachers.NOTEBOOK_POLICY_PATHS[teacher['teacher_id']] = path
    return base


def configure_legacy_assets(root: Path) -> list[LeagueAsset]:
    base = bind_legacy_source(root)
    registry = json.loads((base / 'league/registry.json').read_text(encoding='utf-8'))
    assets: list[LeagueAsset] = []
    for member in registry['p0_members']:
        checkpoint = member.get('portable_checkpoint')
        decks = member.get('portable_decks', [member['portable_deck']] if member.get('portable_deck') else [])
        bundle = member.get('portable_bundle')
        assets.append({'alias': member['alias'], 'checkpoint': str(base / checkpoint) if checkpoint else None, 'bundle': str(base / bundle) if bundle else None, 'decks': [str(base / deck) for deck in decks], 'status': member['status']})
    for member in registry['latest_training_population']:
        checkpoint = member.get('portable_checkpoint')
        assets.append({'alias': member['label'], 'checkpoint': str(base / checkpoint) if checkpoint else None, 'bundle': None, 'decks': [str(base / member['portable_deck'])], 'status': member['role']})
    successor = registry['successor']
    learner = next(member for member in registry['latest_training_population'] if member['id'] == 'd1')
    assets.append({'alias': 'D1_SUCCESSOR', 'checkpoint': str(base / successor['actor']), 'bundle': None, 'decks': [str(base / learner['portable_deck'])], 'status': successor['status']})
    for asset in assets:
        for path in ([asset['checkpoint']] if asset['checkpoint'] else []) + asset['decks']:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if asset['bundle'] and not Path(asset['bundle']).is_dir():
            raise FileNotFoundError(asset['bundle'])

    return assets
