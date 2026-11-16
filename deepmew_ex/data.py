"""Load standalone ragged actor examples without the original replay database."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray


Split = Literal['train', 'validation', 'holdout']


@dataclass(frozen=True)
class Decision:
    row_id: str
    game_id: str
    trajectory_id: str
    split: Split
    option_features: NDArray[np.float32]
    value_features: NDArray[np.float32]
    selection: NDArray[np.int32]
    min_count: int
    max_count: int
    complete_legal_action_count: int
    terminal_outcome: float
    actor_loss_weight: float
    continuity: str
    decision_index: int


def iter_decisions(root: Path, split: Split = 'train') -> Iterator[Decision]:
    directory = root.resolve(strict=True) / 'data' / 'clean_replay'
    manifest = json.loads((directory / 'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 'deepmew.ex.portable_actor_corpus.v1':
        raise ValueError('unsupported corpus schema')
    for shard in manifest['shards']:
        ordinal = int(shard['ordinal'])
        with np.load(directory / f'shard_{ordinal:05d}.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
            with gzip.open(directory / f'shard_{ordinal:05d}.rows.jsonl.gz', 'rt', encoding='utf-8') as handle:
                for index, line in enumerate(handle):
                    metadata = json.loads(line)
                    if metadata['split'] != split:
                        continue
                    start, stop = arrays['option_offsets'][index:index + 2]
                    selected_start, selected_stop = arrays['selection_offsets'][index:index + 2]
                    yield Decision(metadata['stable_row_id'], metadata['source_game_id'], metadata['trajectory_id'], split, arrays['option_features'][start:stop].copy(), arrays['value_features'][index].copy(), arrays['selection_indices'][selected_start:selected_stop].copy(), int(arrays['min_count'][index]), int(arrays['max_count'][index]), int(arrays['complete_legal_action_count'][index]), float(arrays['terminal_outcome'][index]), float(arrays['actor_loss_weight'][index]), str(metadata['continuity']), int(metadata['decision_index']))
