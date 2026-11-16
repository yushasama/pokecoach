"""Cached per-game inference for the causal ``TemporalPolicy``.

A live game only ever appends semantic records and causal events, and every
record is encoded independently of the others, so a record's encoding never
changes once computed. The context transformer is causal (norm-first, no
dropout), so earlier positions' keys/values and outputs never change either.
``CachePool`` keeps record encodings, per-layer keys/values and context rows
for every live game in slot-indexed GPU buffers; ``extend`` encodes just the
new records and runs all games' new event tokens through each layer in one
padded, masked attention call against the cached keys/values. The result
equals ``TemporalPolicy.encode`` on the full prefix up to float rounding
(tests/rl/test_rl_incremental.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from deepmew_ex.datasets.v2.learning.leaf_tables import LeafTables, leaf_tables
from deepmew_ex.datasets.v2.learning.semantic import Leaf
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.rl.records import tables_of  # re-exported


def positional(positions: torch.Tensor, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    """The model's sinusoidal table evaluated at absolute event positions."""
    position = positions.to(dtype)[:, None]
    frequency = torch.exp(torch.arange(0, hidden, 2, device=positions.device, dtype=dtype) * (-math.log(10000.0) / hidden))
    table = position.new_zeros((len(positions), hidden))
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


def check_model(model: TemporalPolicy) -> None:
    if model.context.norm is not None:
        raise ValueError('incremental context assumes no final encoder norm')
    for layer in model.context.layers:
        if not layer.norm_first or layer.self_attn.batch_first is not True or layer.self_attn.in_proj_weight is None:
            raise ValueError('incremental context assumes norm-first packed-projection layers')
        if layer.dropout.p or layer.dropout1.p or layer.dropout2.p:
            raise ValueError('incremental context requires dropout-free layers')


class Upload:
    """Host-to-device copies of int64 index arrays through a reused pinned buffer.

    On CUDA the copy is non-blocking, so a batch needs no host/device sync
    until its final device-to-host read. Before the buffer is rewritten, the
    previous copy is awaited (by then it has long finished when serving).
    """

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.buffer: torch.Tensor | None = None
        self.done: torch.cuda.Event | None = None

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        array = np.ascontiguousarray(array, dtype=np.int64)
        if self.device.type != 'cuda':
            return torch.from_numpy(array)
        if self.done is not None:
            self.done.synchronize()
        if self.buffer is None or self.buffer.numel() < array.size:
            self.buffer = torch.empty(max(array.size, 1 << 16, 2 * (0 if self.buffer is None else self.buffer.numel())), dtype=torch.int64, pin_memory=True)
        view = self.buffer[:array.size]
        view.numpy()[:] = array
        result = view.to(self.device, non_blocking=True)
        self.done = torch.cuda.Event()
        self.done.record()
        return result


ENCODER_PARTS = 8  # arrays returned by RecordEncoder.prepare


class Rows:
    """Append-only row buffer on a device, grown by doubling (no per-append copy of old rows)."""

    def __init__(self, width: int, device: torch.device, dtype: torch.dtype, capacity: int = 4096) -> None:
        self.data = torch.zeros((capacity, width), device=device, dtype=dtype)
        self.count = 0

    def reserve(self, rows: int) -> int:
        """Reserve ``rows`` new rows; returns the first index."""
        start = self.count
        if start + rows > self.data.shape[0]:
            grown = self.data.new_zeros((max(2 * self.data.shape[0], start + rows), self.data.shape[1]))
            grown[:start] = self.data[:start]
            self.data = grown
        self.count += rows
        return start


class RecordEncoder:
    """Exact ``ByteLeafEncoder`` over ``LeafTables`` with weight-version caches.

    Everything the encoder computes is a pure function of content, so three
    exact caches are kept until the weights change (build a new instance):
    each string's GRU vector, and each record's pooled encoding keyed by the
    bytes of its leaf rows in global string ids (exact, no hashing). Only
    never-seen records are encoded, their distinct leaves once each. (A
    per-leaf cache was measured slower: missed records are large state
    records whose per-leaf dictionary lookups cost more than the matmul.)
    ``max_rows`` bounds both caches; on overflow they restart empty.
    """

    def __init__(self, model: TemporalPolicy, max_rows: int = 500_000) -> None:
        self.encoder = model.encoder
        self.max_rows = max_rows
        weight = self.encoder.bytes.weight
        self.device, self.dtype = weight.device, weight.dtype
        self.hidden = self.encoder.leaf[0].out_features
        self.reset()

    def reset(self) -> None:
        self.index: dict[str, int] = {}
        self.vectors: torch.Tensor | None = None
        self.record_index: dict[bytes, int] = {}
        self.records = Rows(self.hidden, self.device, self.dtype)
        self.stats = {'records': 0, 'record_misses': 0, 'leaf_misses': 0, 'resets': 0}

    def strings(self, texts: list[str]) -> None:
        encoder = self.encoder
        blobs = [text.encode('utf-8') for text in texts]
        raw = np.fromiter(map(len, blobs), dtype=np.int64, count=len(blobs))
        lengths = np.maximum(raw, 1)
        tokens = np.zeros((len(blobs), int(lengths.max())), dtype=np.int64)
        flat = np.frombuffer(b''.join(blobs), dtype=np.uint8).astype(np.int64) + 1
        tokens[np.repeat(np.arange(len(blobs)), raw), np.arange(len(flat)) - np.repeat(np.cumsum(raw) - raw, raw)] = flat
        tokens[raw == 0, 0] = 257
        packed = pack_padded_sequence(encoder.bytes(torch.from_numpy(tokens).to(self.device)), lengths.tolist(), batch_first=True, enforce_sorted=False)
        vectors = encoder.string(packed)[1][0]
        start = len(self.index)
        for offset, text in enumerate(texts):
            self.index[text] = start + offset
        self.vectors = vectors if self.vectors is None else torch.cat((self.vectors, vectors))

    def prepare(self, tables: list[LeafTables]) -> list[np.ndarray]:
        """Host-side int64 arrays for ``run``; registers strings, reserves rows for new records."""
        incoming = sum(len(table.strings) for table in tables)
        records_in = sum(len(table.distinct_counts) for table in tables)
        if max(len(self.index) + incoming, self.records.count + records_in) > self.max_rows:
            stats = self.stats
            self.reset()
            self.stats = stats | {'resets': stats['resets'] + 1}
        missing = sorted({text for table in tables for text in table.strings} - self.index.keys())
        if missing:
            self.strings(missing)
        pairs, bits, occurrence_leaves, positions, sizes, owners, destinations, gathers = [], [], [], [], [], [], [], []
        record_base, new_records, leaf_base = self.records.count, 0, 0
        for table in tables:
            remap = np.fromiter(map(self.index.__getitem__, table.strings), dtype=np.int64, count=len(table.strings))
            rows = np.column_stack((remap[table.unique_rows[:, 0]], remap[table.unique_rows[:, 1]], table.unique_rows[:, 2]))
            bounds = np.concatenate(([0], np.cumsum(table.distinct_counts)))
            # Exact record identity: the bytes of its leaf rows in global ids, sliced from one buffer.
            occurrence = rows[table.inverse]
            buffer, width = occurrence.tobytes(), occurrence.strides[0]
            keys = [buffer[low * width:high * width] for low, high in zip(bounds[:-1].tolist(), bounds[1:].tolist())]
            distinct_rows = np.fromiter((self.record_index.get(key, -1) for key in keys), dtype=np.int64, count=len(keys))
            fresh = []
            for record in np.flatnonzero(distinct_rows < 0).tolist():
                row = self.record_index.get(keys[record])  # an earlier record of this batch may have added it
                if row is None:
                    row = record_base + new_records
                    self.record_index[keys[record]] = row
                    new_records += 1
                    fresh.append(record)
                    destinations.append(row)
                distinct_rows[record] = row
            if fresh:
                # Leaves of this table's new records, each distinct leaf computed once.
                members = np.concatenate([table.inverse[bounds[record]:bounds[record + 1]] for record in fresh])
                used, local = np.unique(members, return_inverse=True)
                pairs.append(rows[used, :2].reshape(-1))
                bits.append(rows[used, 2])
                occurrence_leaves.append(local.reshape(-1) + leaf_base)
                counts = table.distinct_counts[fresh]
                positions.append(np.arange(len(members), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts))
                sizes.append(counts)
                owners.append(np.repeat(np.arange(len(destinations) - len(fresh), len(destinations), dtype=np.int64), counts))
                leaf_base += len(used)
            gathers.append(distinct_rows[table.record_map])
        self.records.reserve(new_records)
        self.stats['records'] += sum(len(table.record_map) for table in tables)
        self.stats['record_misses'] += new_records
        self.stats['leaf_misses'] += leaf_base
        join = lambda parts: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        sizes = join(sizes)
        self.longest = int(sizes.max()) if len(sizes) else 0  # host-side, so ``run`` never syncs to read it
        return [join(pairs), join(bits), join(occurrence_leaves), join(positions), sizes, np.array(destinations, dtype=np.int64), join(gathers), join(owners)]

    def run(self, parts: list[torch.Tensor]) -> torch.Tensor:
        pairs, bits, occurrence_leaves, positions, sizes, destinations, gathers, record_of = parts
        encoder = self.encoder
        if len(destinations):
            # The numeric channel travels as its float32 bit pattern (sign-extended int64).
            numbers = bits.to(torch.int32).view(torch.float32).unsqueeze(1)
            pair = pairs.view(-1, 2)
            leaves = encoder.leaf(torch.cat((self.vectors[pair[:, 0]], self.vectors[pair[:, 1]], numbers), dim=1))
            # Exact mean/max pool over each new record's leaf occurrences via one padded gather.
            count, longest = len(destinations), self.longest
            padded = leaves.new_full((count, longest, self.hidden), -torch.inf)
            padded[record_of, positions] = leaves[occurrence_leaves]
            valid = (torch.arange(longest, device=sizes.device)[None, :] < sizes[:, None]).unsqueeze(2)
            mean = torch.where(valid, padded, 0.0).sum(dim=1) / sizes.unsqueeze(1).to(padded.dtype)
            self.records.data[destinations] = encoder.pool(torch.cat((mean, padded.amax(dim=1)), dim=1))
        return self.records.data[gathers]

    def __call__(self, tables: list[LeafTables]) -> torch.Tensor:
        """Encodings of every record of every table, concatenated in table order."""
        parts = self.prepare(tables)
        flat = torch.from_numpy(np.concatenate(parts)).to(self.device)  # standalone use; CachePool batches its own upload
        return self.run(list(flat.split([len(part) for part in parts])))


class CachePool:
    """Per-game caches in slot-indexed GPU buffers (grown by doubling when a game outgrows them)."""

    def __init__(self, model: TemporalPolicy, slots: int = 16, events: int = 256, records: int = 1024, encoder: RecordEncoder | None = None) -> None:
        check_model(model)
        self.model = model
        self.encoder = encoder or RecordEncoder(model)
        weight = model.reset_kind.weight
        self.device, self.dtype, self.hidden = weight.device, weight.dtype, model.hidden
        self.layers = len(model.context.layers)
        self.heads = model.context.layers[0].self_attn.num_heads
        self.kv = weight.new_zeros((slots, self.layers, 2, events, self.hidden))
        self.context = weight.new_zeros((slots, events, self.hidden))
        self.records = weight.new_zeros((slots, records, self.hidden))
        self.positions = positional(torch.arange(events, device=self.device), self.hidden, self.dtype)
        self.upload, self.server_upload = Upload(self.device), Upload(self.device)
        self.slots: dict[tuple, int] = {}
        self.free = list(range(slots))
        self.record_counts = np.zeros(slots, dtype=np.int64)
        self.event_counts = np.zeros(slots, dtype=np.int64)

    def slot(self, key: tuple) -> int:
        if key not in self.slots:
            if not self.free:
                self.grow(slots=2 * len(self.record_counts))
            slot = self.free.pop(0)
            self.slots[key] = slot
            self.record_counts[slot] = self.event_counts[slot] = 0
        return self.slots[key]

    def release(self, key: tuple) -> None:
        slot = self.slots.pop(key, None)
        if slot is not None:
            self.free.append(slot)

    def grow(self, slots: int | None = None, events: int | None = None, records: int | None = None) -> None:
        old_slots, old_events, old_records = len(self.record_counts), self.kv.shape[3], self.records.shape[1]
        slots, events, records = max(slots or 0, old_slots), max(events or 0, old_events), max(records or 0, old_records)
        kv = self.kv.new_zeros((slots, self.layers, 2, events, self.hidden))
        kv[:old_slots, :, :, :old_events] = self.kv
        context = self.context.new_zeros((slots, events, self.hidden))
        context[:old_slots, :old_events] = self.context
        stored = self.records.new_zeros((slots, records, self.hidden))
        stored[:old_slots, :old_records] = self.records
        self.kv, self.context, self.records = kv, context, stored
        if events > old_events:
            self.positions = positional(torch.arange(events, device=self.device), self.hidden, self.dtype)
        self.free += list(range(old_slots, slots))
        self.record_counts = np.concatenate((self.record_counts, np.zeros(slots - old_slots, dtype=np.int64)))
        self.event_counts = np.concatenate((self.event_counts, np.zeros(slots - old_slots, dtype=np.int64)))

    def game_records(self, key: tuple) -> torch.Tensor:
        slot = self.slots[tuple(key)]
        return self.records[slot, :self.record_counts[slot]]

    def game_context(self, key: tuple) -> torch.Tensor:
        slot = self.slots[tuple(key)]
        return self.context[slot, :self.event_counts[slot]]

    def event_count(self, key: tuple) -> int:
        return int(self.event_counts[self.slots[tuple(key)]])

    @torch.no_grad()
    def extend(self, keys: list[tuple], new_tables: list[LeafTables | None], new_events: list[list[int]]) -> None:
        """Append each game's new records (``leaf_tables`` or ``None``) and causal events.

        ``new_events`` are absolute record indices (into all of the game's
        records so far, including the new ones), in causal order.
        """
        keys = [tuple(key) for key in keys]
        if not len(keys) == len(new_tables) == len(new_events) or len(set(keys)) != len(keys):
            raise ValueError('cache batch arguments differ in length or repeat a game')
        slots = np.array([self.slot(key) for key in keys], dtype=np.int64)
        added = np.array([0 if table is None else len(table.record_map) for table in new_tables], dtype=np.int64)
        counts = np.array([len(events) for events in new_events], dtype=np.int64)
        need_records, need_events = int((self.record_counts[slots] + added).max()), int((self.event_counts[slots] + counts).max())
        if need_records > self.records.shape[1] or need_events > self.kv.shape[3]:
            self.grow(events=1 << max(need_events - 1, 1).bit_length(), records=1 << max(need_records - 1, 1).bit_length())
        for slot, events, extra in zip(slots, new_events, added, strict=True):
            limit = self.record_counts[slot] + extra
            if events and (min(events) < 0 or max(events) >= limit or any(a >= b for a, b in zip(events, events[1:]))):
                raise IndexError('new causal events are outside the cached records or out of order')
        present = [position for position, table in enumerate(new_tables) if table is not None]
        active = np.flatnonzero(counts)
        # Every host-side index array goes to the device in a single copy.
        host = []
        if present:
            host += self.encoder.prepare([new_tables[position] for position in present])
            rows = np.repeat(slots[present], added[present])
            columns = np.concatenate([self.record_counts[slots[position]] + np.arange(added[position]) for position in present])
            host += [rows, columns]
        if len(active):
            before, sizes = self.event_counts[slots[active]], counts[active]
            batch, width, total = len(active), int(sizes.max()), int((before + sizes).max())
            owner = np.repeat(np.arange(batch), sizes)
            offset = np.concatenate([np.arange(size) for size in sizes])
            record = np.concatenate([new_events[index] for index in active])
            host += [slots[active][owner], before[owner] + offset, record, owner, offset, slots[active], before, sizes]
        if not host:
            return
        flat = self.upload(np.concatenate(host))
        parts = list(flat.split([len(part) for part in host]))
        if present:
            encoded = self.encoder.run(parts[:ENCODER_PARTS])
            self.records[parts[ENCODER_PARTS], parts[ENCODER_PARTS + 1]] = encoded
            self.record_counts[slots] += added
            parts = parts[ENCODER_PARTS + 2:]
        if not len(active):
            return
        slot_t, position_t, record_t, owner_t, offset_t, game_slots, before_t, sizes_t = parts
        x = self.records[slot_t, record_t] + self.positions[position_t] + self.model.reset_kind.weight[1]
        # Query i of game b sits at absolute position before[b] + i and sees keys up to it.
        # Padding queries see key 0 only (finite softmax; their outputs are discarded).
        columns = torch.arange(total, device=self.device)
        rows = torch.arange(width, device=self.device)
        allowed = columns[None, None, :] <= (before_t[:, None] + rows[None, :])[:, :, None]
        allowed = (allowed & (rows[None, :] < sizes_t[:, None])[:, :, None]) | (columns == 0)[None, None, :]
        size = self.hidden // self.heads
        for depth, layer in enumerate(self.model.context.layers):
            attention = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
            self.kv[slot_t, depth, 0, position_t] = k
            self.kv[slot_t, depth, 1, position_t] = v
            padded = x.new_zeros((batch, width, self.hidden))
            padded[owner_t, offset_t] = q
            query = padded.view(batch, width, self.heads, size).transpose(1, 2)
            cached = self.kv[game_slots, depth, :, :total]
            key = cached[:, 0].reshape(batch, total, self.heads, size).transpose(1, 2)
            value = cached[:, 1].reshape(batch, total, self.heads, size).transpose(1, 2)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed[:, None])
            x = x + attention.out_proj(out.transpose(1, 2).reshape(batch, width, self.hidden)[owner_t, offset_t])
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        self.context[slot_t, position_t] = x
        self.event_counts[slots[active]] += sizes
