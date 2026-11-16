"""Train-owned vocabulary and immutable per-game ragged token caches."""

from __future__ import annotations

import io
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import orjson
import zstandard

from deepmew_ex.datasets.v2.learning.semantic import Leaf, Vocabulary, action_options, frame_inputs, leaves, request_logs, selected_payload
from deepmew_ex.datasets.v2.schema import Episode
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, digest, file_hash, verify_run


@dataclass(frozen=True)
class GameRecords:
    records: list[list[Leaf]]
    examples: list[dict[str, object]]


def game_records(episode: Episode, selected: list[dict[str, object]], window: int) -> GameRecords:
    wanted = {row['frame_id']: row for row in selected}
    states = {state.state_id: state for state in episode.states}
    logs = request_logs(episode)
    history: list[list[str]] = [[], []]
    frames = {frame.frame_id: frame for frame in episode.frames}
    examples, needed = [], set()
    for frame in episode.frames:
        if frame.frame_id in wanted:
            options, target, reason = action_options(frame)
            if reason or target is None:
                raise ValueError('admitted target no longer eligible')
            prior = history[frame.actor_seat][-window:]
            examples.append({**wanted[frame.frame_id], 'history_frame_ids': list(prior), 'target': target, 'options': options})
            needed.update(prior)
            needed.add(frame.frame_id)
        history[frame.actor_seat].append(frame.frame_id)
    if len(examples) != len(wanted):
        raise ValueError('admitted frame absent from episode')
    records, current_records, history_records = [], {}, {}
    for identity in sorted(needed):
        frame = frames[identity]
        current = frame_inputs(states[frame.state_before_id], frame, logs[identity])
        current_records[identity] = len(records)
        records.append(list(leaves(current)))
        history_records[identity] = len(records)
        records.append(list(leaves({**current, 'past_action': selected_payload(frame)})))
    for example in examples:
        example['state_record'] = current_records[example['frame_id']]
        example['history_records'] = [history_records[identity] for identity in example.pop('history_frame_ids')]
        example['option_records'] = []
        for option in example.pop('options'):
            example['option_records'].append(len(records))
            records.append(list(leaves(option)))
    return GameRecords(records, examples)


def source_episode(rows: list[dict[str, object]], corpus: Path, manifest: dict[str, object]) -> Episode:
    child = Path(rows[0]['episode_path']).resolve()
    if not child.is_relative_to(corpus.resolve()):
        raise ValueError('episode path outside corpus')
    path = child / 'episode.json.zst'
    if file_hash(path) != manifest['files_sha256'][path.relative_to(corpus).as_posix()]:
        raise ValueError('episode cache input changed')
    episode = Episode.from_bytes(zstandard.ZstdDecompressor().decompress(path.read_bytes()))
    if any(row['game_id'] != episode.source_native_game_id or row['source_sha256'] != episode.source_sha256 for row in rows):
        raise ValueError('admission source binding changed')
    return episode


def build_cache(admission: Path, paths: OutputPaths, run_id: str, progress: Callable[[int, int], None] | None = None) -> Path:
    admitted = verify_run(admission)
    if admitted['status'] != 'completed' or admitted['kind'] != 'v2_bc_admission':
        raise ValueError('unverified training admission')
    config = orjson.loads((admission / 'config.json').read_bytes())
    membership = orjson.loads((admission / 'membership.json').read_bytes())
    corpus = Path(config['corpus'])
    manifest = verify_run(corpus)
    report = orjson.loads((admission / 'admission.json').read_bytes())
    if file_hash(corpus / 'manifest.json') != report['corpus_manifest_sha256']:
        raise ValueError('corpus identity changed since admission')
    store = RunStore.allocate(paths, 'deepmew_ex', 'v2_learning_cache', run_id)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rows in membership.values():
        for row in rows:
            groups[row['game_id']].append(row)
    try:
        store.write('code.json', code_identity(Path(__file__).resolve().parents[4]))
        store.write('config.json', {'admission': str(admission.resolve()), 'admission_manifest_sha256': file_hash(admission / 'manifest.json'), 'window': config['context_window'], 'max_leaves': config['max_leaves'], 'vocabulary_population': 'train membership and its causal context only', 'records': 'semantic path/value/numeric triples; no fixed 49/144/304 feature projection'})
        paths_seen, values_seen = set(), set()
        for game, rows in sorted(groups.items()):
            if rows[0]['split'] != 'train':
                continue
            data = game_records(source_episode(rows, corpus, manifest), rows, config['context_window'])
            for record in data.records:
                if len(record) > config['max_leaves']:
                    raise ValueError('admission missed semantic record budget')
                for path, value, _ in record:
                    paths_seen.add(path)
                    values_seen.add(value)
        vocab = Vocabulary({value: index + 2 for index, value in enumerate(sorted(paths_seen))}, {value: index + 2 for index, value in enumerate(sorted(values_seen))}, config['max_leaves'])
        store.write('vocabulary.json', asdict(vocab))
        examples: dict[str, list[dict[str, object]]] = {'train': [], 'validation': []}
        unknown: dict[str, Counter[str]] = {'train': Counter(), 'validation': Counter()}
        for index, (game, rows) in enumerate(sorted(groups.items()), 1):
            split = rows[0]['split']
            data = game_records(source_episode(rows, corpus, manifest), rows, config['context_window'])
            encoded = [vocab.encode(record, unknown[split]) for record in data.records]
            offsets = np.concatenate(([0], np.cumsum([len(record) for record in encoded]))).astype(np.int64)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, tokens=np.concatenate(encoded), offsets=offsets)
            relative = f'game-{digest(game.encode())}.npz'
            sha = store.write_bytes(relative, buffer.getvalue())
            examples[split].extend({**row, 'cache_file': relative, 'cache_sha256': sha} for row in data.examples)
            if progress and index % 100 == 0:
                progress(index, len(groups))
        for split, rows in examples.items():
            rows.sort(key=lambda row: row['selection_hash'])
            if len(rows) != len(membership[split]):
                raise ValueError('cache lost admitted examples')
        store.write('examples.json', examples)
        store.write('support.json', {'examples': {key: len(value) for key, value in examples.items()}, 'vocabulary': {'paths': len(vocab.paths) + 2, 'values': len(vocab.values) + 2}, 'unknown_leaves': {key: dict(value) for key, value in unknown.items()}, 'optimizer_updates': 0})
        store.seal({'kind': 'v2_learning_cache', 'optimizer_updates': 0})
    except Exception as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_learning_cache'}, 'failed')
        raise
    return store.path


class CachedDataset:
    def __init__(self, path: Path, max_games: int = 128) -> None:
        manifest = verify_run(path)
        if manifest['status'] != 'completed' or manifest['kind'] != 'v2_learning_cache':
            raise ValueError('unverified learning cache')
        self.path, self.max_games = path, max_games
        self.examples = orjson.loads((path / 'examples.json').read_bytes())
        self.vocabulary = Vocabulary(**orjson.loads((path / 'vocabulary.json').read_bytes()))
        self.window = orjson.loads((path / 'config.json').read_bytes())['window']
        self.loaded: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def record(self, row: dict[str, object], index: int) -> np.ndarray:
        name = row['cache_file']
        if name not in self.loaded:
            if file_hash(self.path / name) != row['cache_sha256']:
                raise ValueError('cached game changed')
            with np.load(self.path / name, allow_pickle=False) as data:
                self.loaded[name] = data['tokens'], data['offsets']
            while len(self.loaded) > self.max_games:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(name)
        tokens, offsets = self.loaded[name]
        return tokens[offsets[index]:offsets[index + 1]]
