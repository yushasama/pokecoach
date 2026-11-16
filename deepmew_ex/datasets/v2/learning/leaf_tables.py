"""Torch-free encoder input tables for one actor sequence's semantic records.

The byte-leaf encoder needs, for every record, the identity of each leaf
(path string, value string, float32 numeric bits), which records are exact
duplicates, and which leaves inside the distinct records are exact
duplicates. Building that is pure Python/numpy work; loader workers compute it
per sequence so the training process only merges small arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepmew_ex.datasets.v2.learning.semantic import Leaf

TABLES_ATTRIBUTE = '_encoder_tables'


@dataclass(frozen=True)
class LeafTables:
    strings: tuple[str, ...]
    record_map: np.ndarray
    distinct_counts: np.ndarray
    unique_rows: np.ndarray
    inverse: np.ndarray

    @property
    def distinct_leaves(self) -> int:
        return int(self.distinct_counts.sum())


def leaf_tables(records: tuple[list[Leaf], ...]) -> LeafTables:
    """Exact identity tables: rows are (path id, value id, float32 bit pattern) over sorted strings."""
    if not records or any(not record for record in records):
        raise ValueError('empty semantic record')
    strings = sorted({text for record in records for path, value, _ in record for text in (path, value)})
    lookup = {text: index for index, text in enumerate(strings)}
    counts = np.fromiter(map(len, records), dtype=np.int64, count=len(records))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    paths, values, numeric = zip(*(leaf for record in records for leaf in record))
    bits = np.asarray(numeric, dtype=np.float64).astype(np.float32).view(np.int32).astype(np.int64)
    rows = np.stack((np.fromiter(map(lookup.__getitem__, paths), dtype=np.int64, count=len(paths)), np.fromiter(map(lookup.__getitem__, values), dtype=np.int64, count=len(values)), bits), axis=1)
    identities: dict[bytes, int] = {}
    record_map = np.fromiter((identities.setdefault(rows[offsets[index]:offsets[index + 1]].tobytes(), len(identities)) for index in range(len(records))), dtype=np.int64, count=len(records))
    distinct = np.unique(record_map, return_index=True)[1]
    distinct_counts = counts[distinct]
    distinct_offsets = np.concatenate(([0], np.cumsum(distinct_counts)))
    distinct_rows = rows[np.repeat(offsets[distinct] - distinct_offsets[:-1], distinct_counts) + np.arange(distinct_offsets[-1])]
    pair_id = np.unique(distinct_rows[:, 0] * len(strings) + distinct_rows[:, 1], return_inverse=True)[1].reshape(-1).astype(np.int64)
    _, first, inverse = np.unique((pair_id << 32) | (distinct_rows[:, 2] & 0xFFFFFFFF), return_index=True, return_inverse=True)
    return LeafTables(tuple(strings), record_map, distinct_counts, distinct_rows[first], inverse.reshape(-1).astype(np.int64))


def attach_tables(sequence: object) -> object:
    """Cache derived tables on a frozen sequence; dataclass equality ignores the attribute."""
    object.__setattr__(sequence, TABLES_ATTRIBUTE, leaf_tables(sequence.records))
    return sequence
