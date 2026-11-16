"""Read only the explicitly bundled metadata; never open checkpoint payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from deepmew_ex.registry.records import Evidence, Lineage, LineageEdge, Node, validate_hash
from deepmew_ex.registry.store import file_hash

SOURCES = (
    'models/MODEL_MANIFEST.json',
    'league/registry.json',
    'league/MODEL_CARDS.json',
    'source/.research/value_q_experiment/MODEL_REGISTRY.json',
    'source/.research/league_temporal_rl/EXPERIMENT_MANIFEST.json',
    'source/.research/league_temporal_rl/league_registry.v1.json',
    'source/.research/d1_successor_mixed_league_v1/D1_SUCCESSOR_TRAINING_RECEIPT.json',
    'source/.research/d1_successor_mixed_league_v1/closeout/LINEAGE_GRAPH_UPDATE_QUALIFICATION_COMPLETE.json',
)
EXTERNAL = (
    '.research/value_q_experiment/EXPERIMENT_MANIFEST.json',
    '.research/terminal_value_rl_v1/production_e1_20260705T062726Z/E1_FINAL_PROTECTED_RECEIPT.json',
)


@dataclass(frozen=True)
class LegacyRecord:
    record_id: str
    aliases: tuple[str, ...]
    evidence: Evidence
    fields: dict[str, object]
    asset_presence: str
    training_reproduced: bool = False
    unknown_training_fields: str = 'UNKNOWN unless explicitly present in source fields'


@dataclass(frozen=True)
class LegacyImport:
    records: list[LegacyRecord]
    sources: dict[str, object]
    unresolved: list[str]
    edges: list[LineageEdge]

    def lookup(self, alias: str) -> dict[str, object]:
        records = [record for record in self.records if record.record_id == alias or alias in record.aliases]
        if not records:
            raise ValueError(f'unknown legacy identity: {alias}')
        return {'resolution': 'exact' if len(records) == 1 else 'ambiguous_preserved', 'records': [asdict(record) for record in records]}

    def lineage(self) -> Lineage:
        nodes = [Node(record.record_id, 'historical_evidence', record.aliases) for record in self.records]
        return Lineage(nodes, self.edges)


def import_legacy(root: Path) -> LegacyImport:
    records: list[LegacyRecord] = []
    sources: dict[str, object] = {}
    unresolved = list(EXTERNAL)
    root = root.resolve()
    for relative in SOURCES:
        path = root / relative
        if not path.is_file():
            unresolved.append(relative)
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        sha = file_hash(path)
        # Keep all source-level fields, including unknowns, Windows paths and conflicting claims.
        sources[relative] = {'sha256': sha, 'fields': data, 'scope': 'VERIFIED_RECEIPT; payload bytes and training not verified'}
        groups: list[tuple[str, dict[str, object]]] = []
        if isinstance(data, list):
            groups.extend((f'/{i}', row) for i, row in enumerate(data))
        else:
            groups.append(('', data))
            for key in ('models', 'p0_members', 'latest_training_population', 'members'):
                groups.extend((f'/{key}/{i}', row) for i, row in enumerate(data.get(key, [])))
            if relative == 'league/registry.json':
                groups.append(('/successor', {**data['successor'], 'alias': 'D1_SUCCESSOR'}))
        for pointer, fields in groups:
            aliases = list(fields.get('aliases', []))
            for key in ('alias', 'label', 'name'):
                if isinstance(fields.get(key), str):
                    aliases.append(fields[key])
            for key in ('sha256', 'checkpoint_sha256', 'artifact_sha256', 'policy_checkpoint_sha256'):
                value = fields.get(key)
                if value is not None:
                    validate_hash(value)
            asset = fields.get('portable_checkpoint', fields.get('path') if relative == 'models/MODEL_MANIFEST.json' else None)
            presence = 'UNKNOWN_metadata_only'
            if isinstance(asset, str):
                local = (root / asset).resolve()
                presence = 'present_not_hashed_or_loaded' if local.is_relative_to(root) and local.is_file() else 'missing'
            evidence = Evidence(relative, sha, pointer, 'verified', 'source metadata bytes only; historical assertions not re-executed')
            records.append(LegacyRecord(f'legacy:{relative}#{pointer}', tuple(aliases), evidence, fields, presence))
    edges: list[LineageEdge] = []
    successor_source = SOURCES[-1]
    if successor_source in sources:
        successor = next(record for record in records if record.evidence.source == successor_source)
        d1 = next((record for record in records if record.evidence.source == 'league/registry.json' and record.evidence.pointer.startswith('/latest_training_population/') and 'D1' in record.aliases), None)
        if d1 is not None:
            edges.append(LineageEdge(successor.record_id, d1.record_id, 'initialized_from', successor.evidence))
    result = LegacyImport(records, sources, unresolved, edges)
    result.lineage()
    return result
