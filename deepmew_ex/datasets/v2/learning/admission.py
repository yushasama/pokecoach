"""Conserve replay decisions, audit split signatures and freeze fitting membership."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import orjson

from deepmew_ex.datasets.v2.corpus.reader import raw_episodes
from deepmew_ex.datasets.v2.learning.semantic import action_options, actor_state, frame_inputs, input_signature, leaves, request_context, request_logs, selected_payload
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash

SPLIT_PRIORITY = {'train': 0, 'validation': 1, 'calibration': 2, 'test': 3, 'holdout': 4, 'golden_validation': 5}
FROZEN_EVALUATION_SEEDS = frozenset({40009, 50021, 60013})


@dataclass(frozen=True)
class AdmissionConfig:
    corpus: Path
    paths: OutputPaths
    run_id: str
    train_decisions: int = 16384
    validation_decisions: int = 2048
    seed: int = 42
    max_leaves: int = 1024
    context_window: int = 8
    objective: str = 'single_selection_causal_v2_bc'

    def validate(self) -> None:
        if self.objective != 'single_selection_causal_v2_bc':
            raise ValueError('unsupported objective before data loading')
        if type(self.train_decisions) is not int or type(self.validation_decisions) is not int or min(self.train_decisions, self.validation_decisions) < 1:
            raise ValueError('positive training and validation limits required')
        if not 1 <= self.max_leaves <= 4096:
            raise ValueError('invalid semantic leaf limit')
        if type(self.context_window) is not int or not 1 <= self.context_window <= 32:
            raise ValueError('invalid context window')
        self.paths.validate()


def admit_replays(config: AdmissionConfig, progress: Callable[[int, int], None] | None = None) -> Path:
    config.validate()
    store = RunStore.allocate(config.paths, 'deepmew_ex', 'v2_admission', config.run_id)
    started = time.perf_counter()
    counters: dict[str, Counter[str]] = {}
    signatures: dict[str, tuple[int, str]] = {}
    games: dict[str, str] = {}
    game_contents: dict[str, str] = {}
    content_priority: dict[str, int] = {}
    selections: dict[str, list[tuple[str, dict[str, object]]]] = {'train': [], 'validation': []}
    candidates = []
    try:
        store.write('config.json', {'corpus': str(config.corpus.resolve()), 'train_decisions': config.train_decisions, 'validation_decisions': config.validation_decisions, 'seed': config.seed, 'max_leaves': config.max_leaves, 'context_window': config.context_window, 'objective': config.objective, 'authorization': 'user approved V2 retraining and local league evaluation on 2026-10-25'})
        store.write('code.json', code_identity(Path(__file__).resolve().parents[4]))
        report = orjson.loads((config.corpus / 'port.json').read_bytes())
        sources = {item['native_game_id']: item for item in report['results'] if item.get('status') == 'admitted_actor_visible_episode'}
        for index, wrapped in enumerate(raw_episodes(config.corpus), 1):
            episode, split = wrapped.episode, wrapped.effective_split
            if split not in SPLIT_PRIORITY:
                raise ValueError('unknown effective split')
            native = episode.source_native_game_id
            if native in games:
                raise ValueError('native game repeated or crossing splits')
            games[native] = split
            counts = counters.setdefault(split, Counter())
            counts['games'] += 1
            states = {state.state_id: state for state in episode.states}
            logs = request_logs(episode)  # Verify live-reachable own-request public snapshots.
            past_budgets: list[list[bool]] = [[], []]
            content = []
            for frame in episode.frames:
                counts['raw_decisions'] += 1
                options, target, reason = action_options(frame)
                if episode.seed in FROZEN_EVALUATION_SEEDS:
                    reason = 'frozen_evaluation_seed'
                state, context = actor_state(states[frame.state_before_id]), request_context(frame)
                current = frame_inputs(states[frame.state_before_id], frame, logs[frame.frame_id])
                content.append({'actor': frame.actor_seat, 'order': frame.actor_order, 'input': current, 'chosen': selected_payload(frame)})
                history_bad = len(list(leaves({**current, 'past_action': selected_payload(frame)}))) > config.max_leaves
                previous_bad = any(past_budgets[frame.actor_seat][-config.context_window:])
                past_budgets[frame.actor_seat].append(history_bad)
                if reason is None and (len(list(leaves(current))) > config.max_leaves or any(len(list(leaves(option))) > config.max_leaves for option in options)):
                    reason = 'semantic_leaf_budget'
                if reason is None and previous_bad:
                    reason = 'history_leaf_budget'
                if reason:
                    counts['excluded:' + reason] += 1
                    continue
                signature = input_signature(state, context, options)
                rank = SPLIT_PRIORITY[split]
                if signature not in signatures or rank > signatures[signature][0]:
                    signatures[signature] = rank, split
                counts['supported_decisions'] += 1
                # Membership uses native game/seat/request, never chosen target or outcome.
                coordinate = f'{native}:seat={frame.actor_seat}:request={frame.request_step}'
                order = digest(f'{config.seed}:{coordinate}'.encode())
                candidates.append({'row_id': coordinate, 'frame_id': frame.frame_id, 'game_id': native, 'actor': frame.actor_seat, 'actor_order': frame.actor_order, 'split': split, 'signature': signature, 'selection_hash': order, 'episode_path': sources[native]['path'], 'source_sha256': episode.source_sha256})
            content_sha = digest(canonical(content))
            game_contents[native] = content_sha
            content_priority[content_sha] = max(content_priority.get(content_sha, -1), SPLIT_PRIORITY[split])
            if progress and index % 50 == 0:
                progress(index, len(sources))
        seen: dict[str, set[str]] = {'train': set(), 'validation': set()}
        for row in sorted(candidates, key=lambda item: item['selection_hash']):
            split, signature = row['split'], row['signature']
            counts = counters[split]
            if content_priority[game_contents[row['game_id']]] > SPLIT_PRIORITY[split]:
                counts['excluded:cross_split_game_content'] += 1
                continue
            if signatures[signature][1] != split:
                counts['excluded:cross_split_semantic_signature'] += 1
                continue
            if split not in seen:
                continue
            if signature in seen[split]:
                counts['excluded:within_split_duplicate_signature'] += 1
                continue
            seen[split].add(signature)
            counts['eligible_unique_signatures'] += 1
            cap = config.train_decisions if split == 'train' else config.validation_decisions
            if len(selections[split]) < cap:
                selections[split].append((row['selection_hash'], row))
        membership = {split: [row for _, row in rows] for split, rows in selections.items()}
        if any(not rows for rows in membership.values()):
            raise ValueError('empty eligible training or validation membership')
        if {row['game_id'] for row in membership['train']} & {row['game_id'] for row in membership['validation']}:
            raise ValueError('selected game split contamination')
        if {row['signature'] for row in membership['train']} & {row['signature'] for row in membership['validation']}:
            raise ValueError('selected semantic split contamination')
        store.write('membership.json', membership)
        store.write('admission.json', {'scope': 'all admitted causal raw replays; canonical reference-only rows remain outside fitting admission', 'corpus_manifest_sha256': file_hash(config.corpus / 'manifest.json'), 'games': games, 'game_content_sha256': game_contents, 'counts': {key: dict(value) for key, value in counters.items()}, 'selected': {key: {'rows': len(rows), 'games': len({row['game_id'] for row in rows}), 'sha256': digest(canonical(rows))} for key, rows in membership.items()}, 'split_priority': SPLIT_PRIORITY, 'selection': 'lowest SHA256(seed:native-game:actor:request), target/outcome independent', 'historical_masks': 'not reused as new admission; semantic signatures recomputed conservatively for all reserved splits', 'history_stream': 'same-actor request snapshots and actions only; no inactive-seat-only public snapshots', 'raw_steps_are_not_internal_simulator_ticks': True, 'canonical_reference_rows_not_admitted': report['catalog_rows'], 'optimizer_updates': 0, 'elapsed_seconds': time.perf_counter() - started})
        store.seal({'kind': 'v2_bc_admission', 'optimizer_updates': 0})
    except Exception as error:
        store.write('failure.json', {'type': type(error).__name__, 'message': str(error), 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_bc_admission'}, 'failed')
        raise
    return store.path
