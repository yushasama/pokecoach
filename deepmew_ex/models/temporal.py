"""UTF-8 semantic leaves, complete causal event attention and exact action losses."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.checkpoint import checkpoint

from deepmew_ex.datasets.v2.learning.leaf_tables import TABLES_ATTRIBUTE, LeafTables, leaf_tables
from deepmew_ex.datasets.v2.learning.semantic import Leaf
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, StageInput, StageTarget
from deepmew_ex.models.mlp import DeepMewEX

ACTIVATION_LEAF_BUDGET = 65536


class ByteLeafEncoder(nn.Module):
    """Every path/value byte is consumed; unseen cards do not collapse to UNK."""

    def __init__(self, hidden: int, embedding: int = 24) -> None:
        super().__init__()
        self.bytes = nn.Embedding(258, embedding, padding_idx=0)
        self.string = nn.GRU(embedding, embedding, batch_first=True)
        self.leaf = nn.Sequential(nn.Linear(embedding * 2 + 1, hidden), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, records: tuple[list[Leaf], ...], tables: list[LeafTables] | None = None) -> torch.Tensor:
        """Encode records; ``tables`` (one per consecutive record group) skips in-process table building."""
        device = self.bytes.weight.device
        if tables is None:
            tables = [leaf_tables(records)]
        if sum(len(table.record_map) for table in tables) != len(records):
            raise ValueError('encoder tables do not match records')
        strings = sorted(set().union(*(table.strings for table in tables))) if len(tables) > 1 else list(tables[0].strings)
        lookup = {text: index for index, text in enumerate(strings)}
        remaps = [np.fromiter(map(lookup.__getitem__, table.strings), dtype=np.int64, count=len(table.strings)) for table in tables]
        # Bytes are shifted by one (0 is padding); an empty string is the single token 257.
        blobs = [text.encode('utf-8') for text in strings]
        raw = np.fromiter(map(len, blobs), dtype=np.int64, count=len(blobs))
        lengths = np.maximum(raw, 1)
        tokens = np.zeros((len(blobs), int(lengths.max())), dtype=np.int64)
        flat = np.frombuffer(b''.join(blobs), dtype=np.uint8).astype(np.int64) + 1
        tokens[np.repeat(np.arange(len(blobs)), raw), np.arange(len(flat)) - np.repeat(np.cumsum(raw) - raw, raw)] = flat
        tokens[raw == 0, 0] = 257
        packed = pack_padded_sequence(self.bytes(torch.from_numpy(tokens).to(device)), lengths.tolist(), batch_first=True, enforce_sorted=False)
        _, vectors = self.string(packed)
        vectors = vectors[0]
        # A game repeats most records verbatim between requests: each distinct
        # record is pooled once and gathered back for every occurrence. Chunks
        # cut the global distinct-record order under the activation budget.
        sizes = np.concatenate([table.distinct_counts for table in tables])
        owners = np.concatenate([np.full(len(table.distinct_counts), position, dtype=np.int64) for position, table in enumerate(tables)])
        firsts = np.concatenate(([0], np.cumsum([len(table.distinct_counts) for table in tables])))
        chunks: list[tuple[int, int]] = []
        start, leaves = 0, 0
        for index, size in enumerate(sizes.tolist()):
            if index > start and leaves + size > ACTIVATION_LEAF_BUDGET:
                chunks.append((start, index))
                start, leaves = index, 0
            leaves += size
        chunks.append((start, len(sizes)))
        pooled = []
        for start, end in chunks:
            rows, inverses, unique_count = [], [], 0
            for position in np.unique(owners[start:end]).tolist():
                table = tables[position]
                low, high = max(start, int(firsts[position])) - int(firsts[position]), min(end, int(firsts[position + 1])) - int(firsts[position])
                bounds = np.concatenate(([0], np.cumsum(table.distinct_counts)))
                if low == 0 and high == len(table.distinct_counts):
                    used, local = table.unique_rows, table.inverse
                else:
                    # Identical leaves inside the chunk share one leaf activation.
                    selected, local = np.unique(table.inverse[bounds[low]:bounds[high]], return_inverse=True)
                    used, local = table.unique_rows[selected], local.reshape(-1)
                rows.append(np.column_stack((remaps[position][used[:, 0]], remaps[position][used[:, 1]], used[:, 2])))
                inverses.append(local + unique_count)
                unique_count += len(used)
            unique = np.concatenate(rows)
            pair = torch.from_numpy(np.ascontiguousarray(unique[:, :2])).to(device)
            numbers = torch.from_numpy(unique[:, 2].astype(np.int32).view(np.float32)).to(device).unsqueeze(1)
            leaf_index = torch.from_numpy(np.concatenate(inverses)).to(device)
            owner = torch.from_numpy(np.repeat(np.arange(end - start, dtype=np.int64), sizes[start:end])).to(device)
            owned = torch.from_numpy(sizes[start:end].copy()).to(device).unsqueeze(1)
            # Each record's exact mean/max pool is independent. Retain every
            # leaf and share the same byte vectors; recompute only activations.
            if len(chunks) > 1 and torch.is_grad_enabled() and vectors.requires_grad:
                pooled.append(checkpoint(self.pool_records, vectors, pair, numbers, leaf_index, owner, owned, use_reentrant=False))
            else:
                pooled.append(self.pool_records(vectors, pair, numbers, leaf_index, owner, owned))
        record_index = np.concatenate([table.record_map + int(firsts[position]) for position, table in enumerate(tables)])
        return torch.cat(pooled)[torch.from_numpy(record_index).to(device)]

    def pool_records(self, vectors: torch.Tensor, pair: torch.Tensor, numbers: torch.Tensor, leaf_index: torch.Tensor, owner: torch.Tensor, owned: torch.Tensor) -> torch.Tensor:
        features = self.leaf(torch.cat((vectors[pair[:, 0]], vectors[pair[:, 1]], numbers), dim=1))[leaf_index]
        sums = features.new_zeros((owned.shape[0], features.shape[1])).index_add(0, owner, features)
        maximum = features.new_full(sums.shape, -torch.inf).scatter_reduce(0, owner[:, None].expand_as(features), features, reduce='amax', include_self=False)
        return self.pool(torch.cat((sums / owned, maximum), dim=1))


def subset_nll(logits: torch.Tensor, selected: tuple[int, ...], minimum: int, maximum: int, count_scores: torch.Tensor) -> torch.Tensor:
    """Exact normalized unordered no-replacement subset likelihood, O(n*k)."""
    if not 0 <= minimum <= len(selected) <= maximum <= len(logits) or len(set(selected)) != len(selected):
        raise ValueError('invalid subset bounds/target')
    coefficients = [logits.sum() * 0]
    for logit in logits:
        previous = coefficients
        coefficients = [previous[0]]
        for count in range(1, min(maximum, len(previous)) + 1):
            included = previous[count - 1] + logit
            coefficients.append(torch.logaddexp(previous[count], included) if count < len(previous) else included)
    normalizer = torch.logsumexp(torch.stack([coefficients[count] + count_scores[count] for count in range(minimum, maximum + 1)]), dim=0)
    target_score = logits[list(selected)].sum() + count_scores[len(selected)]
    return normalizer - target_score


def option_matrix(options: list[tuple[int, ...]], bases: list[int], counts: list[int], pad: int) -> np.ndarray:
    """Padded flat-record indices for each decision's options; padding is ``pad``."""
    index = np.full((len(options), max(1, max(map(len, options)))), pad, dtype=np.int64)
    for row, (values, base, count) in enumerate(zip(options, bases, counts, strict=True)):
        if values:
            array = np.asarray(values, dtype=np.int64)
            if array.min() < 0 or array.max() >= count:
                raise IndexError('temporal option outside its game records')
            index[row, :len(values)] = base + array
    return index


class TemporalPolicy(nn.Module):
    def __init__(self, hidden: int = 128, embedding: int = 24, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.encoder = ByteLeafEncoder(hidden, embedding)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.scorer = DeepMewEX(hidden, hidden, hidden)
        self.count = nn.Linear(hidden, 3)
        self.stop = nn.Linear(hidden, 1)
        self.decoder = nn.GRUCell(hidden, hidden)
        self.reset_kind = nn.Embedding(2, hidden)

    def encode(self, sequence: ActorSequence) -> tuple[torch.Tensor, torch.Tensor]:
        records = self.encoder(sequence.records)
        events = records[list(sequence.events)]
        position = torch.arange(len(events), device=events.device, dtype=events.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.hidden, 2, device=events.device, dtype=events.dtype) * (-math.log(10000.0) / self.hidden))
        positional = torch.zeros_like(events)
        positional[:, 0::2] = torch.sin(position * frequency)
        positional[:, 1::2] = torch.cos(position * frequency)
        # Snapshot-only records explicitly have unknown history; never infer a
        # trajectory by source-file adjacency or reuse another game's state.
        kind = int(sequence.history_available)
        events = events + positional + self.reset_kind.weight[kind]
        mask = torch.triu(torch.ones((len(events), len(events)), device=events.device, dtype=torch.bool), diagonal=1)
        context = self.context(events.unsqueeze(0), mask=mask)[0]
        return records, context

    def stage_loss(self, state: torch.Tensor, options: torch.Tensor, stage: StageTarget) -> torch.Tensor:
        logits = self.scorer(options, state)
        if not stage.ordered:
            counts = torch.arange(stage.maximum + 1, device=state.device, dtype=state.dtype)
            scale = max(len(options), 1)
            features = torch.stack((counts / scale, (counts / scale).square(), torch.log1p(counts)), dim=1)
            return subset_nll(logits, stage.selected, stage.minimum, stage.maximum, features @ self.count(state))
        hidden = state
        selected = set()
        total = state.sum() * 0
        for step in range(len(stage.selected) + 1):
            if step == stage.maximum:
                break
            scores = self.scorer(options, hidden)
            allowed = torch.ones(len(options) + 1, device=state.device, dtype=torch.bool)
            if not stage.repeated and selected:
                allowed[list(selected)] = False
            allowed[-1] = step >= stage.minimum
            scores = torch.cat((scores, self.stop(hidden))).masked_fill(~allowed, -torch.inf)
            target = stage.selected[step] if step < len(stage.selected) else len(options)
            total = total + F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=state.device))
            if target == len(options):
                break
            selected.add(target)
            hidden = self.decoder(options[target], hidden)
        return total

    def decode_stage(self, state: torch.Tensor, options: torch.Tensor, stage: StageInput) -> tuple[tuple[int, ...], torch.Tensor]:
        if not stage.ordered:
            logits = self.scorer(options, state)
            counts = torch.arange(stage.maximum + 1, device=state.device, dtype=state.dtype)
            scale = max(len(options), 1)
            features = torch.stack((counts / scale, (counts / scale).square(), torch.log1p(counts)), dim=1)
            count_scores = features @ self.count(state)
            ranked = torch.argsort(logits, descending=True, stable=True)
            scores = [count_scores[count] + logits[ranked[:count]].sum() for count in range(stage.minimum, stage.maximum + 1)]
            count = stage.minimum + int(torch.stack(scores).argmax())
            selected = tuple(sorted(int(value) for value in ranked[:count]))
            return selected, state
        hidden = state
        selected = []
        for step in range(stage.maximum):
            scores = self.scorer(options, hidden)
            allowed = torch.ones(len(options) + 1, device=state.device, dtype=torch.bool)
            if not stage.repeated and selected:
                allowed[selected] = False
            allowed[-1] = step >= stage.minimum
            scores = torch.cat((scores, self.stop(hidden))).masked_fill(~allowed, -torch.inf)
            target = int(scores.argmax())
            if target == len(options):
                break
            selected.append(target)
            hidden = self.decoder(options[target], hidden)
        if not stage.minimum <= len(selected) <= stage.maximum:
            raise ValueError('decoded structured action violates count bounds')
        return tuple(selected), hidden

    def decode(self, sequence: ActorSequence, event_index: int, stages: tuple[StageInput, ...]) -> tuple[tuple[int, ...], ...]:
        records, context = self.encode(sequence)
        if not 0 <= event_index < len(context):
            raise ValueError('live prediction event is outside causal prefix')
        state = context[event_index]
        result = []
        for stage in stages:
            options = records[list(stage.options)]
            selected, ordered_state = self.decode_stage(state, options, stage)
            result.append(selected)
            if selected:
                state = ordered_state if stage.ordered else self.decoder(options[list(selected)].mean(dim=0), state)
        return tuple(result)

    def losses(self, sequence: ActorSequence) -> tuple[torch.Tensor, list[str]]:
        records, context = self.encode(sequence)
        losses, ids = [], []
        for prediction in sequence.predictions:
            if not prediction.loss_eligible:
                continue
            state = context[prediction.event_index]
            terms = []
            for stage in prediction.stages:
                options = records[list(stage.options)]
                terms.append(self.stage_loss(state, options, stage))
                if stage.selected:
                    state = self.advance_target(state, options, stage)
            losses.append(torch.stack(terms).sum())
            ids.append(prediction.row_id)
        return (torch.stack(losses) if losses else records.new_empty(0)), ids

    def advance_target(self, state: torch.Tensor, options: torch.Tensor, stage: StageTarget) -> torch.Tensor:
        if stage.ordered:
            for index in stage.selected:
                state = self.decoder(options[index], state)
            return state
        return self.decoder(options[list(stage.selected)].mean(dim=0), state)

    def batched_stage_losses(self, table: torch.Tensor, states: torch.Tensor, bases: list[int], counts: list[int], stages: list[tuple[StageTarget, ...]]) -> torch.Tensor:
        """``stage_loss``/``advance_target`` for many decisions at once.

        Decisions advance together stage by stage. Within a stage the exact
        unordered subset recursion runs once per option position, and ordered
        picks once per autoregressive step, each across the whole group.
        ``table`` is every record encoding plus a trailing zero padding row.
        """
        if any(not stage for stage in stages):
            raise ValueError('temporal target without stages')
        device, pad = table.device, table.shape[0] - 1
        current, totals = states, states.new_zeros(len(stages))
        for depth in range(max(map(len, stages))):
            for ordered in (False, True):
                rows = [row for row, stage in enumerate(stages) if len(stage) > depth and stage[depth].ordered == ordered]
                if not rows:
                    continue
                specs = [stages[row][depth] for row in rows]
                for spec in specs:
                    if any(not 0 <= value < len(spec.options) for value in spec.selected):
                        raise IndexError('stage target outside its options')
                gather = torch.from_numpy(option_matrix([spec.options for spec in specs], [bases[row] for row in rows], [counts[row] for row in rows], pad)).to(device)
                index = torch.tensor(rows, device=device)
                options, state = table[gather], current[index]
                if ordered:
                    terms, after = self.ordered_group(options, state, specs)
                else:
                    terms, after = self.subset_group(options, state, specs)
                totals = totals.index_add(0, index, terms)
                current = current.index_copy(0, index, after)
        return totals

    def subset_group(self, options: torch.Tensor, state: torch.Tensor, specs: list[StageTarget]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched ``subset_nll`` plus the mean-selected-option state advance."""
        for spec in specs:
            if not 0 <= spec.minimum <= len(spec.selected) <= spec.maximum <= len(spec.options) or len(set(spec.selected)) != len(spec.selected):
                raise ValueError('invalid subset bounds/target')
        device, rows, width = state.device, len(specs), options.shape[1]
        logits = self.scorer(options, state)
        top = max(spec.maximum for spec in specs)
        lengths = torch.tensor([len(spec.options) for spec in specs], device=device)
        # Entry c holds log of the elementary symmetric sum of degree c over the
        # options seen so far. Rows stop updating past their own option count;
        # entries a row never reaches are never read, so zeros cannot leak.
        coefficients = logits.new_zeros((rows, top + 1))
        for position in range(width if top else 0):
            reach = min(position + 1, top)
            included = coefficients[:, :reach] + logits[:, position:position + 1]
            existing = min(position, top)
            parts = [coefficients[:, :1], torch.logaddexp(coefficients[:, 1:existing + 1], included[:, :existing])]
            if reach > existing:
                parts.append(included[:, existing:reach])
            parts.append(coefficients[:, reach + 1:])
            coefficients = torch.where((position < lengths)[:, None], torch.cat(parts, dim=1), coefficients)
        counts = torch.arange(top + 1, device=device, dtype=state.dtype)
        scale = torch.tensor([max(len(spec.options), 1) for spec in specs], device=device, dtype=state.dtype)[:, None]
        ratio = counts[None, :] / scale
        features = torch.stack((ratio, ratio.square(), torch.log1p(counts).expand_as(ratio)), dim=2)
        count_scores = (features @ self.count(state).unsqueeze(2)).squeeze(2)
        minimum = torch.tensor([spec.minimum for spec in specs], device=device)[:, None]
        maximum = torch.tensor([spec.maximum for spec in specs], device=device)[:, None]
        valid = (counts[None, :] >= minimum) & (counts[None, :] <= maximum)
        normalizer = torch.logsumexp((coefficients + count_scores).masked_fill(~valid, -torch.inf), dim=1)
        chosen = torch.from_numpy(np.array([list(spec.selected) + [0] * (max(1, max(len(item.selected) for item in specs)) - len(spec.selected)) for spec in specs], dtype=np.int64)).to(device)
        picked = torch.tensor([[column < len(spec.selected) for column in range(chosen.shape[1])] for spec in specs], device=device)
        target_logits = logits.gather(1, chosen).masked_fill(~picked, 0.0).sum(dim=1)
        target_counts = count_scores.gather(1, torch.tensor([[len(spec.selected)] for spec in specs], device=device)).squeeze(1)
        terms = normalizer - (target_logits + target_counts)
        moving = [row for row, spec in enumerate(specs) if spec.selected]
        after = state
        if moving:
            index = torch.tensor(moving, device=device)
            vectors = options[index[:, None], chosen[index]].masked_fill(~picked[index][:, :, None], 0.0)
            means = vectors.sum(dim=1) / picked[index].sum(dim=1, keepdim=True).to(state.dtype)
            after = state.index_copy(0, index, self.decoder(means, state[index]))
        return terms, after

    def ordered_group(self, options: torch.Tensor, state: torch.Tensor, specs: list[StageTarget]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched ordered autoregressive picks with a stop action at column ``width``."""
        device, width = state.device, options.shape[1]
        steps = [min(len(spec.selected) + 1, spec.maximum) for spec in specs]
        terms, hidden = state.new_zeros(len(specs)), state
        chosen: list[set[int]] = [set() for _ in specs]
        for step in range(max(steps)):
            active = [row for row, count in enumerate(steps) if count > step]
            index = torch.tensor(active, device=device)
            current = hidden[index]
            scores = torch.cat((self.scorer(options[index], current), self.stop(current)), dim=1)
            allowed = np.zeros((len(active), width + 1), dtype=bool)
            targets = []
            for slot, row in enumerate(active):
                spec = specs[row]
                allowed[slot, :len(spec.options)] = True
                if not spec.repeated and chosen[row]:
                    allowed[slot, list(chosen[row])] = False
                allowed[slot, width] = step >= spec.minimum
                targets.append(spec.selected[step] if step < len(spec.selected) else width)
            mask = torch.from_numpy(allowed).to(device)
            terms = terms.index_add(0, index, F.cross_entropy(scores.masked_fill(~mask, -torch.inf), torch.tensor(targets, device=device), reduction='none'))
            moving = [(slot, row, target) for slot, (row, target) in enumerate(zip(active, targets, strict=True)) if target != width]
            if moving:
                rows = torch.tensor([row for _, row, _ in moving], device=device)
                picks = options[rows, torch.tensor([target for _, _, target in moving], device=device)]
                hidden = hidden.index_copy(0, rows, self.decoder(picks, hidden[rows]))
                for _, row, target in moving:
                    chosen[row].add(target)
        return terms, hidden

    def encode_batch(self, sequences: list[ActorSequence]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pad complete games with independent causal masks; share only leaf encoding."""
        encoded, context, lengths = self.encode_padded(sequences)
        records = encoded.split([len(sequence.records) for sequence in sequences])
        return [(values, context[index, :lengths[index]]) for index, values in enumerate(records)]

    def encode_padded(self, sequences: list[ActorSequence]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Concatenated record encodings plus the padded per-game causal context."""
        if not sequences or any(not sequence.events or not sequence.history_available for sequence in sequences):
            raise ValueError('temporal batches require nonempty causal histories')
        tables = [getattr(sequence, TABLES_ATTRIBUTE, None) for sequence in sequences]
        encoded = self.encoder(tuple(record for sequence in sequences for record in sequence.records), tables if all(table is not None for table in tables) else None)
        records = list(encoded.split([len(sequence.records) for sequence in sequences]))
        lengths = [len(sequence.events) for sequence in sequences]
        width = max(lengths)
        events = encoded.new_zeros((len(sequences), width, self.hidden))
        padding = torch.ones((len(sequences), width), dtype=torch.bool, device=encoded.device)
        for index, (sequence, values) in enumerate(zip(sequences, records, strict=True)):
            events[index, :lengths[index]] = values[list(sequence.events)]
            padding[index, :lengths[index]] = False
        position = torch.arange(width, device=encoded.device, dtype=encoded.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.hidden, 2, device=encoded.device, dtype=encoded.dtype) * (-math.log(10000.0) / self.hidden))
        positional = torch.zeros_like(events[0])
        positional[:, 0::2] = torch.sin(position * frequency)
        positional[:, 1::2] = torch.cos(position * frequency)
        mask = torch.triu(torch.ones((width, width), device=encoded.device, dtype=torch.bool), diagonal=1)
        context = self.context(events + positional + self.reset_kind.weight[1], mask=mask, src_key_padding_mask=padding)
        return encoded, context, lengths

    def batch_losses(self, sequences: list[ActorSequence]) -> tuple[torch.Tensor, list[str]]:
        flat, padded_context, lengths = self.encode_padded(sequences)
        offsets = np.concatenate(([0], np.cumsum([len(sequence.records) for sequence in sequences])))
        losses: list[torch.Tensor | None] = []
        identities, simple, complex_rows = [], [], []
        for game, sequence in enumerate(sequences):
            for prediction in sequence.predictions:
                if not prediction.loss_eligible:
                    continue
                identity = len(identities)
                identities.append(prediction.row_id)
                losses.append(None)
                if len(prediction.stages) == 1 and prediction.stages[0].minimum == prediction.stages[0].maximum == 1:
                    stage = prediction.stages[0]
                    if not 0 <= prediction.event_index < lengths[game]:
                        raise IndexError('temporal target event outside its game history')
                    simple.append((identity, game, prediction.event_index, stage.options, stage.selected[0]))
                    continue
                if not 0 <= prediction.event_index < lengths[game]:
                    raise IndexError('temporal target event outside its game history')
                complex_rows.append((identity, game, prediction.event_index, prediction.stages))
        device, pad = flat.device, flat.shape[0]
        # Every record encoding plus one trailing zero row: padding gathers the
        # exact zeros the per-decision scorer saw before batching.
        table = torch.cat((flat, flat.new_zeros((1, flat.shape[1])))) if simple or complex_rows else flat
        if simple:
            gather = torch.from_numpy(option_matrix([row[3] for row in simple], [int(offsets[row[1]]) for row in simple], [int(offsets[row[1] + 1] - offsets[row[1]]) for row in simple], pad)).to(device)
            options = table[gather]
            mask = gather != pad
            states = padded_context[torch.tensor([row[1] for row in simple], device=device), torch.tensor([row[2] for row in simple], device=device)]
            logits = self.scorer(options, states).masked_fill(~mask, -torch.inf)
            values = F.cross_entropy(logits, torch.tensor([row[4] for row in simple], device=device), reduction='none')
            for row, value in zip(simple, values.unbind(), strict=True):
                losses[row[0]] = value
        if complex_rows:
            states = padded_context[torch.tensor([row[1] for row in complex_rows], device=device), torch.tensor([row[2] for row in complex_rows], device=device)]
            values = self.batched_stage_losses(table, states, [int(offsets[row[1]]) for row in complex_rows], [int(offsets[row[1] + 1] - offsets[row[1]]) for row in complex_rows], [row[3] for row in complex_rows])
            for row, value in zip(complex_rows, values.unbind(), strict=True):
                losses[row[0]] = value
        if any(value is None for value in losses):
            raise RuntimeError('batch loss omitted a temporal target')
        return (torch.stack(losses) if losses else flat.new_empty(0)), identities

    def snapshot_losses(self, sequences: list[ActorSequence]) -> tuple[torch.Tensor, list[str]]:
        """Batch the dominant single-choice snapshot path; preserve other targets."""
        simple, complex_rows = [], []
        for sequence in sequences:
            prediction = sequence.predictions[0]
            if not prediction.loss_eligible:
                continue
            if len(prediction.stages) == 1 and prediction.stages[0].minimum == prediction.stages[0].maximum == 1:
                simple.append(sequence)
            else:
                complex_rows.append(sequence)
        losses, ids = [], []
        if simple:
            all_records, offsets = [], []
            for sequence in simple:
                offsets.append(len(all_records))
                all_records.extend(sequence.records)
            encoded = self.encoder(tuple(all_records))
            current = encoded[[offset + sequence.events[0] for offset, sequence in zip(offsets, simple, strict=True)]]
            positional = torch.zeros_like(current)
            positional[:, 1::2] = 1
            state = self.context((current + positional + self.reset_kind.weight[0]).unsqueeze(1))[:, 0]
            counts = [len(sequence.predictions[0].stages[0].options) for sequence in simple]
            options = encoded.new_zeros((len(simple), max(counts), self.hidden))
            mask = torch.zeros(options.shape[:2], dtype=torch.bool, device=encoded.device)
            targets = []
            for row, (offset, sequence) in enumerate(zip(offsets, simple, strict=True)):
                stage = sequence.predictions[0].stages[0]
                options[row, :len(stage.options)] = encoded[[offset + value for value in stage.options]]
                mask[row, :len(stage.options)] = True
                targets.append(stage.selected[0])
                ids.append(sequence.predictions[0].row_id)
            logits = self.scorer(options, state).masked_fill(~mask, -torch.inf)
            losses.extend(F.cross_entropy(logits, torch.tensor(targets, device=encoded.device), reduction='none').unbind())
        for sequence in complex_rows:
            loss, identities = self.losses(sequence)
            losses.extend(loss.unbind())
            ids.extend(identities)
        device = self.reset_kind.weight.device
        return (torch.stack(losses) if losses else torch.empty(0, device=device)), ids
