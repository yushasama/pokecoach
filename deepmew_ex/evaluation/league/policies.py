"""Deployable semantic history, explicit fallback and independently seeded seat adapters."""

from __future__ import annotations

import random
import time
import os
from contextlib import contextmanager
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import torch

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import state_view
from deepmew_ex.datasets.v2.learning.semantic import Vocabulary, action_payload, actor_state, leaves, relative_players, selection_options
from deepmew_ex.datasets.v2.schema import Payload, payload
from deepmew_ex.models.semantic import SemanticBC
from deepmew_ex.evaluation.league.seats import IndependentSeat, ReferenceSeatPolicy, SeatPolicy, greedy_fallback, reference_directory  # re-exported


class SemanticLivePolicy:
    def __init__(self, label: str, model: SemanticBC, vocabulary: Vocabulary, window: int, seed: int) -> None:
        self.label, self.model, self.vocabulary = label, model, vocabulary
        self.window, self.seed = window, seed
        self.previous: deque[Payload] = deque(maxlen=window)
        self.counts: Counter[str] = Counter()
        self.actor_seat: int | None = None
        self.actor_order = 0

    def score(self, current: Payload, options: list[Payload]) -> torch.Tensor:
        now = self.vocabulary.encode(list(leaves(current)))
        candidates = [self.vocabulary.encode(list(leaves(option))) for option in options]
        prior = list(self.previous)
        if self.model.history == 'shuffled':
            np.random.default_rng(self.seed + self.actor_order).shuffle(prior)
        histories = [self.vocabulary.encode(list(leaves(record))) for record in prior] if self.model.history in ('ordered', 'shuffled') else []
        opts = np.zeros((1, len(candidates), max(map(len, candidates)), 3), dtype=np.float32)
        history = np.zeros((1, self.window, max(map(len, histories), default=1), 3), dtype=np.float32)
        for index, tokens in enumerate(candidates):
            opts[0, index, :len(tokens)] = tokens
        for index, tokens in enumerate(histories):
            history[0, self.window - len(histories) + index, :len(tokens)] = tokens
        with torch.no_grad():
            return self.model(torch.from_numpy(now[None]), torch.from_numpy(opts), torch.from_numpy(history), torch.ones((1, len(candidates)), dtype=torch.bool))[0]

    def select(self, observation: Payload, rng: random.Random) -> list[int]:
        contract = load_contract()
        projected = payload(contract.project_actor(observation))
        seat = projected['current']['yourIndex']
        if self.actor_seat is not None and self.actor_seat != seat:
            raise ValueError('live actor seat changed within game')
        self.actor_seat = seat
        state = state_view(observation, seat, self.actor_order, contract)
        source_select = projected['select']
        legal = contract.build_legal(state.state_id, source_select)
        if legal.failures:
            raise ValueError('native legal contract unsupported')
        canonical_contract = payload(asdict(legal))
        options, ids, reason = selection_options(canonical_contract, seat)
        logs = [relative_players(event, seat) for event in contract.public_logs(observation.get('logs', []))]
        context = relative_players({key: value for key, value in source_select.items() if key != 'option'}, seat)
        current = {'state': actor_state(state), 'request': context, 'public_log_snapshot': logs}
        if reason is None:
            for option in canonical_contract['stages'][0]['options']:
                decoded = contract.reconstruct_chosen(legal, [[int(option['position'])]])
                if not decoded.complete or decoded.failures or payload(asdict(decoded))['stages'][0]['selected_option_ids'] != [option['option_id']]:
                    reason = 'non_index_decoder'
                    break
        if reason is None:
            try:
                logits = self.score(current, options)
            except ValueError as error:
                if str(error) != 'semantic_leaf_budget_exceeded':
                    raise
                reason = 'semantic_leaf_budget'
            else:
                if logits.shape != (len(options),) or not torch.isfinite(logits).all():
                    raise ValueError('nonfinite or malformed live logits')
                position = int(logits.argmax())
                option = canonical_contract['stages'][0]['options'][position]
                selected = [int(option['position'])]
                self.counts['model_choices'] += 1
        if reason is not None:
            selected = greedy_fallback(projected)
            self.counts['fallback:' + reason] += 1
        chosen = contract.reconstruct_chosen(legal, [selected])
        if not chosen.complete or chosen.failures:
            raise ValueError('live choice contract reconstruction failed')
        self.previous.append({**current, 'past_action': action_payload(canonical_contract, payload(asdict(chosen)), seat)})
        self.actor_order += 1
        return selected


def legacy_checkpoint_policy(label: str, checkpoint: Path, model: torch.nn.Module, metadata: dict[str, object], deck: list[int], feature_policy_id: str) -> SeatPolicy:
    from run_r9_r4f_retrain import R4FLivePolicy
    from deepmew.rl.metadata import normalized_deck_hash
    class BoundCheckpoint(R4FLivePolicy):
        # Override archival asset resolution while retaining its exact select path.
        def __init__(self) -> None:
            self.label, self.checkpoint = label, checkpoint
            self.model, self.payload = model, metadata
            self.deck, self.policy_id = list(deck), feature_policy_id
            self.deck_hash = normalized_deck_hash(deck)
    class SeatAdapter:
        def __init__(self) -> None:
            self.label, self.policy = label, BoundCheckpoint()

        def select(self, observation: Payload, rng: random.Random) -> list[int]:
            # Legacy extractor gets actor projection, never opaque engine buffers.
            projected = payload(load_contract().project_actor(observation))
            projected['logs'] = load_contract().public_logs(observation.get('logs', []))
            return self.policy.select(projected)
    return SeatAdapter()
