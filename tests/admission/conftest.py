from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from deepmew_ex.admission.config import AuditConfig
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.experiments.synthetic import SyntheticDataset
from deepmew_ex.registry.store import canonical, file_hash


@dataclass(frozen=True)
class CorpusFixture:
    config: AuditConfig

    def metadata(self) -> list[dict[str, object]]:
        with gzip.open(self.config.corpus / 'shard_00000.rows.jsonl.gz', 'rt') as handle:
            return [json.loads(line) for line in handle]

    def write_metadata(self, rows: list[dict[str, object]]) -> None:
        with gzip.open(self.config.corpus / 'shard_00000.rows.jsonl.gz', 'wt') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')

    def arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.config.corpus / 'shard_00000.npz', allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def write_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        np.savez(self.config.corpus / 'shard_00000.npz', **arrays)

    def rebind(self) -> None:
        manifest = json.loads(self.config.expected_manifest.read_text())
        for entry in manifest['shards'][0]['files']:
            path = self.config.corpus / entry['path']
            entry.update(bytes=path.stat().st_size, sha256=file_hash(path))
        self.config.expected_manifest.write_bytes(canonical(manifest))
        (self.config.corpus / 'EXPORT_MANIFEST.json').write_bytes(canonical(manifest))
        (self.config.corpus / 'shard_00000.receipt.json').write_bytes(canonical(manifest['shards'][0]))


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusFixture:
    root = tmp_path / 'input'
    directory = root / 'data/clean_replay'
    directory.mkdir(parents=True)
    rows = []
    for split in ('train', 'validation', 'holdout'):
        source_split = 'train' if split == 'train' else 'validation'
        for index, row in enumerate(SyntheticDataset(42, 8, 8).decisions(source_split)):
            if index == 2:
                row = replace(row, complete_legal_action_count=10)
            if index == 3:
                row = replace(row, complete_legal_action_count=len(row.option_features) + 1)
            rows.append(replace(row, row_id=f'{split}-row-{index}', game_id=f'{split}-game-{index // 4}', trajectory_id=f'{split}-trajectory-{index // 4}', split=split))
    arrays = {'option_features': np.concatenate([row.option_features for row in rows]), 'value_features': np.stack([row.value_features for row in rows]), 'option_offsets': np.array([0, *np.cumsum([len(row.option_features) for row in rows])], dtype=np.int64), 'selection_offsets': np.array([0, *np.cumsum([len(row.selection) for row in rows])], dtype=np.int64), 'selection_indices': np.concatenate([row.selection for row in rows]), 'min_count': np.array([row.min_count for row in rows], dtype=np.int32), 'max_count': np.array([row.max_count for row in rows], dtype=np.int32), 'complete_legal_action_count': np.array([row.complete_legal_action_count for row in rows], dtype=np.int64), 'terminal_outcome': np.array([row.terminal_outcome for row in rows], dtype=np.float32), 'actor_loss_weight': np.array([row.actor_loss_weight for row in rows], dtype=np.float32)}
    config = AuditConfig(root, 'fixture-audit', OutputPaths(tmp_path / 'outputs'), tmp_path / 'trusted.json', tmp_path / 'validation.json', train_selection_limit=3, validation_selection_limit=2)
    fixture = CorpusFixture(config)
    fixture.write_arrays(arrays)
    fixture.write_metadata([{'stable_row_id': row.row_id, 'source_game_id': row.game_id, 'trajectory_id': row.trajectory_id, 'split': row.split, 'source_family': 'fixture', 'continuity': row.continuity, 'decision_index': row.decision_index, 'actor_seat': 0, 'seed': 42} for row in rows])
    shard = {'ordinal': 0, 'rows': len(rows), 'games_in_shard': 6, 'split_counts': {'train': 8, 'validation': 8, 'holdout': 8}, 'source_counts': {'fixture': 24}, 'actor_rows': 18, 'files': [{'path': f'shard_00000{suffix}', 'bytes': 1, 'sha256': 'a' * 64} for suffix in ('.npz', '.rows.jsonl.gz')]}
    config.expected_manifest.write_bytes(canonical({'schema_version': 'deepmew.ex.portable_actor_corpus.v1', 'actor_feature_dim': 304, 'value_feature_dim': 144, 'rows': 24, 'games': 6, 'shards': [shard]}))
    config.validation_receipt.write_bytes(canonical({'rows': 24, 'games': 6, 'actor_rows': 18, 'forced_rows': 6, 'splits': shard['split_counts'], 'sources': {'fixture': 24}, 'continuity': {'segmented': 24}}))
    fixture.rebind()
    return fixture
