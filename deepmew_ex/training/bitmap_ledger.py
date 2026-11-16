"""Exact per-epoch consumption proof in memory with an append-only binary log.

The SQLite ledger answered two per-decision point lookups against a
multi-gigabyte random-key table on every optimizer update. This ledger keeps
the same guarantees with arrays:

* expected membership is frozen before optimization, one row per admitted
  decision, identified by a 128-bit BLAKE2b key of its decision id;
* an update is admitted only if each decision is expected, belongs to the
  sequence's game and source family, and is not yet consumed in that epoch;
* an actor context is recorded at most once per epoch;
* each update appends one binary record; the SHA-256 chain over records is the
  resumable ledger identity bound into checkpoints;
* completion requires every expected decision consumed exactly once per epoch.

Decision id strings are written once to ``expected-ids.txt.zst`` (row order)
for offline readers; they are never held in memory during training.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import zstandard

from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence

RECORD_MAGIC = b'DMXL'
RECORD_HEADER = struct.Struct('<4sqiii')
CONTEXT_FIELDS = 6
EMPTY_CHAIN = hashlib.sha256(b'').hexdigest()
PHASE = 'raw'
LEDGER_FILES = ('expected-index.npz', 'expected-names.txt.zst', 'expected-ids.txt.zst', 'ledger.bin')


def keys_of(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """128-bit keys as (high, low) uint64 arrays."""
    digests = b''.join(hashlib.blake2b(identity.encode('utf-8'), digest_size=16).digest() for identity in ids)
    pairs = np.frombuffer(digests, dtype='<u8').reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


class ExpectedIndex:
    """Admitted decisions in row order with a (high, low)-sorted lookup view."""

    def __init__(self, high: np.ndarray, low: np.ndarray, games: np.ndarray, families: np.ndarray, game_names: list[str], family_names: list[str], sequences: int) -> None:
        if not len(high):
            raise ValueError('empty eligible temporal training membership')
        self.high, self.low, self.games, self.families = high, low, games, families
        self.game_names, self.family_names, self.sequences = game_names, family_names, sequences
        self.game_codes = {name: code for code, name in enumerate(game_names)}
        self.family_codes = {name: code for code, name in enumerate(family_names)}
        self.order = np.lexsort((low, high))
        self.sorted_high, self.sorted_low = high[self.order], low[self.order]
        if np.any((self.sorted_high[1:] == self.sorted_high[:-1]) & (self.sorted_low[1:] == self.sorted_low[:-1])):
            raise ValueError('duplicate expected decision identity')

    def __len__(self) -> int:
        return len(self.high)

    @classmethod
    def build(cls, rows: Iterable[tuple[str, str, str, int]], ids_path: Path | None = None) -> ExpectedIndex:
        """Stream (decision id, game, source family, sequence ordinal) rows into compact arrays."""
        high, low, games, families = array.array('Q'), array.array('Q'), array.array('i'), array.array('h')
        game_codes: dict[str, int] = {}
        family_codes: dict[str, int] = {}
        sequences: set[int] = set()
        writer = zstandard.ZstdCompressor(level=3).stream_writer(ids_path.open('wb')) if ids_path is not None else None
        batch: list[str] = []

        def flush() -> None:
            if batch:
                pair_high, pair_low = keys_of(batch)
                high.extend(pair_high.tolist())
                low.extend(pair_low.tolist())
                if writer is not None:
                    writer.write(('\n'.join(batch) + '\n').encode('utf-8'))
                batch.clear()
        try:
            for identity, game, family, ordinal in rows:
                batch.append(identity)
                games.append(game_codes.setdefault(game, len(game_codes)))
                families.append(family_codes.setdefault(family, len(family_codes)))
                sequences.add(ordinal)
                if len(batch) >= 65536:
                    flush()
            flush()
        finally:
            if writer is not None:
                writer.close()
        return cls(np.frombuffer(high, dtype=np.uint64).copy(), np.frombuffer(low, dtype=np.uint64).copy(), np.frombuffer(games, dtype=np.int32).copy(), np.frombuffer(families, dtype=np.int16).copy(), list(game_codes), list(family_codes), len(sequences))

    def lookup(self, ids: list[str]) -> np.ndarray:
        """Row indices of ``ids``; any id outside the expected set is a membership violation."""
        high, low = keys_of(ids)
        position = np.minimum(np.searchsorted(self.sorted_high, high, side='left'), len(self.sorted_high) - 1)
        found = (self.sorted_high[position] == high) & (self.sorted_low[position] == low)
        for slot in np.flatnonzero(~found).tolist():
            # Equal high words are possible; scan that short run for the low word.
            probe = int(position[slot])
            while probe < len(self.sorted_high) and self.sorted_high[probe] == high[slot] and self.sorted_low[probe] != low[slot]:
                probe += 1
            if probe == len(self.sorted_high) or self.sorted_high[probe] != high[slot]:
                raise ValueError('update membership differs from independent preflight')
            position[slot] = probe
        return self.order[position]

    def summary(self, epochs: int) -> dict[str, int]:
        return {'actor_sequences': self.sequences, 'games': len(self.game_names), 'unique_eligible_decisions': len(self), 'epochs': epochs, 'expected_presentations': len(self) * epochs}

    def save(self, root: Path) -> None:
        np.savez(root / 'expected-index.npz', high=self.high, low=self.low, games=self.games, families=self.families, sequences=np.array([self.sequences], dtype=np.int64))
        (root / 'expected-names.txt.zst').write_bytes(zstandard.ZstdCompressor().compress(('\x1e'.join(self.game_names) + '\n' + '\x1e'.join(self.family_names)).encode('utf-8')))

    @classmethod
    def load(cls, root: Path) -> ExpectedIndex:
        with np.load(root / 'expected-index.npz') as arrays:
            values = {name: arrays[name] for name in ('high', 'low', 'games', 'families', 'sequences')}
        games, families = zstandard.ZstdDecompressor().decompress((root / 'expected-names.txt.zst').read_bytes()).decode('utf-8').split('\n')
        return cls(values['high'], values['low'], values['games'], values['families'], games.split('\x1e'), families.split('\x1e'), int(values['sequences'][0]))


def expected_rows_from_sequences(sequences: Iterable[ActorSequence], check_resources: Callable[[], None], counts: dict[str, int]) -> Iterator[tuple[str, str, str, int]]:
    """Decode-pass membership with the historical preflight checks; ``counts`` receives the sequence total."""
    seen: set[tuple[str, int]] = set()
    counts['actor_sequences'] = 0
    for ordinal, sequence in enumerate(sequences):
        check_resources()
        if sequence.split != 'train' or not sequence.history_available:
            raise ValueError('temporal training requires train-split causal histories')
        identity = (sequence.game_id, sequence.actor)
        if identity in seen:
            raise ValueError('duplicate actor sequence in expected membership')
        seen.add(identity)
        counts['actor_sequences'] += 1
        for prediction in sequence.predictions:
            if prediction.loss_eligible:
                yield prediction.row_id, sequence.game_id, sequence.source_family, ordinal


def read_records(path: Path) -> Iterable[tuple[int, int, np.ndarray, np.ndarray, bytes]]:
    """(update, epoch, decision row indices, context rows, raw record bytes) in log order."""
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        if offset + RECORD_HEADER.size > len(data):
            raise ValueError('truncated coverage ledger record')
        magic, update, epoch, count, contexts = RECORD_HEADER.unpack_from(data, offset)
        size = RECORD_HEADER.size + 4 * count + 4 * CONTEXT_FIELDS * contexts
        if magic != RECORD_MAGIC or offset + size > len(data):
            raise ValueError('truncated coverage ledger record')
        payload = data[offset:offset + size]
        indices = np.frombuffer(payload, dtype=np.int32, count=count, offset=RECORD_HEADER.size)
        rows = np.frombuffer(payload, dtype=np.int32, count=CONTEXT_FIELDS * contexts, offset=RECORD_HEADER.size + 4 * count).reshape(-1, CONTEXT_FIELDS)
        yield update, epoch, indices, rows, payload
        offset += size


class BitmapLedger:
    def __init__(self, root: Path, index: ExpectedIndex, epochs: int) -> None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError('positive epoch count required')
        self.root, self.index, self.epochs = root, index, epochs
        self.update_of = np.zeros((epochs, len(index)), dtype=np.int32)
        self.contexts: list[set[tuple[int, int]]] = [set() for _ in range(epochs)]
        self.chain, self.updates = EMPTY_CHAIN, 0
        self.log = (root / 'ledger.bin').open('ab')

    def admit(self, epoch: int, group: list[ActorSequence]) -> list[np.ndarray]:
        """Validate one update before its optimizer step; returns per-sequence row indices."""
        if not 0 <= epoch < self.epochs:
            raise ValueError('update is outside temporal training scope')
        consumed: list[np.ndarray] = []
        actors: set[tuple[int, int]] = set()
        for sequence in group:
            if sequence.split != 'train' or not sequence.history_available:
                raise ValueError('update is outside temporal training scope')
            ids = [row.row_id for row in sequence.predictions if row.loss_eligible]
            index = self.index.lookup(ids) if ids else np.zeros(0, dtype=np.int64)
            game, family = self.index.game_codes.get(sequence.game_id), self.index.family_codes.get(sequence.source_family)
            if game is None or family is None or np.any(self.index.games[index] != game) or np.any(self.index.families[index] != family) or np.any(self.update_of[epoch, index] != 0):
                raise ValueError('update membership differs from independent preflight')
            actor = (game, sequence.actor)
            if actor in self.contexts[epoch] or actor in actors:
                raise ValueError('actor context recorded twice in one epoch')
            actors.add(actor)
            consumed.append(index)
        merged = np.concatenate(consumed) if consumed else np.zeros(0, dtype=np.int64)
        if len(np.unique(merged)) != len(merged):
            raise ValueError('update membership differs from independent preflight')
        return consumed

    def record(self, epoch: int, update: int, group: list[ActorSequence], consumed: list[np.ndarray]) -> str:
        """Commit one optimizer update after its step; returns the new chain identity."""
        indices = np.concatenate(consumed).astype(np.int32) if consumed else np.zeros(0, dtype=np.int32)
        if update != self.updates + 1 or not len(indices):
            raise ValueError('coverage ledger has a missing optimizer update')
        contexts = np.array([(self.index.game_codes[sequence.game_id], sequence.actor, len(sequence.predictions), len(sequence.events), len(sequence.preserved_transition_ids), len(sequence.unavailable_post_states)) for sequence in group], dtype=np.int32).reshape(-1, CONTEXT_FIELDS)
        payload = RECORD_HEADER.pack(RECORD_MAGIC, update, epoch, len(indices), len(contexts)) + indices.tobytes() + contexts.tobytes()
        self.apply(epoch, update, indices, contexts)
        self.log.write(payload)
        self.log.flush()
        self.chain = hashlib.sha256(self.chain.encode('ascii') + payload).hexdigest()
        return self.chain

    def apply(self, epoch: int, update: int, indices: np.ndarray, contexts: np.ndarray) -> None:
        self.update_of[epoch, indices] = update
        for row in contexts.tolist():
            self.contexts[epoch].add((row[0], row[1]))
        self.updates = update

    def sync(self) -> None:
        self.log.flush()
        os.fsync(self.log.fileno())

    def replay(self, parent: Path, updates: int, expected_chain: str) -> None:
        """Restore a committed prefix from a parent run's log and copy it into this run's log."""
        for update, epoch, indices, rows, payload in read_records(parent / 'ledger.bin'):
            if update > updates:
                break
            if update != self.updates + 1 or not 0 <= epoch < self.epochs or np.any(self.update_of[epoch, indices] != 0):
                raise ValueError('coverage prefix differs from committed checkpoint identity')
            self.apply(epoch, update, indices, rows)
            self.log.write(payload)
            self.chain = hashlib.sha256(self.chain.encode('ascii') + payload).hexdigest()
        if self.updates != updates or self.chain != expected_chain:
            raise ValueError('coverage prefix differs from committed checkpoint identity')
        self.sync()

    def presentations(self) -> int:
        return int(np.count_nonzero(self.update_of))

    def verify_complete(self) -> dict[str, int]:
        missing = int(np.count_nonzero(self.update_of == 0))
        if missing:
            raise ValueError(f'training coverage mismatch: missing={missing}, unexpected=0')
        return {'missing': 0, 'unexpected': 0, 'presentations': self.presentations()}

    def family_counts(self) -> list[tuple[str, str, int]]:
        any_epoch = np.any(self.update_of != 0, axis=0)
        return [(PHASE, name, int(np.count_nonzero(any_epoch & (self.index.families == code)))) for code, name in enumerate(self.index.family_names)]

    def close(self) -> None:
        if not self.log.closed:
            self.sync()
            self.log.close()


def verify_run_ledger(run: Path) -> dict[str, object]:
    """Independent offline check of a run's binary ledger: exact per-epoch coverage and train games."""
    index = ExpectedIndex.load(run)
    epochs = int(json.loads((run / 'expected-membership.json').read_text())['epochs'])
    update_of = np.zeros((epochs, len(index)), dtype=np.int32)
    chain, last = EMPTY_CHAIN, 0
    context = np.zeros(5, dtype=np.int64)
    for update, epoch, indices, rows, payload in read_records(run / 'ledger.bin'):
        if update != last + 1 or np.any(update_of[epoch, indices] != 0):
            raise ValueError('coverage ledger is not a contiguous exactly-once log')
        update_of[epoch, indices] = update
        context += [len(rows), *rows[:, 2:].sum(axis=0).tolist()] if len(rows) else 0
        chain, last = hashlib.sha256(chain.encode('ascii') + payload).hexdigest(), update
    missing = int(np.count_nonzero(update_of == 0))
    consumed_rows = np.any(update_of != 0, axis=0)
    games = {index.game_names[code] for code in np.unique(index.games[consumed_rows]).tolist()}
    coverage = [(name, int(np.count_nonzero(update_of[:, index.families == code])), int(np.count_nonzero(consumed_rows & (index.families == code))), len({index.game_names[g] for g in np.unique(index.games[consumed_rows & (index.families == code)]).tolist()})) for code, name in enumerate(index.family_names)]
    return {'missing': missing, 'unexpected': 0, 'updates': last, 'chain': chain, 'presentations': int(np.count_nonzero(update_of)), 'per_epoch': [int(np.count_nonzero(row)) for row in update_of], 'train_games': games, 'coverage': coverage, 'context': tuple(int(value) for value in context)}
