"""Decision makers for the coach: BC temporal model (plays and advises), BabyMew rules, greedy.

Every brain sees every native observation through ``observe`` (the model's
history needs both seats' prompts to place turn boundaries) and chooses only
for its own seat through ``act``. A model brain that hits an unsupported input
degrades to the greedy fallback for the rest of the game instead of failing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from deepmew_ex.datasets.v2.learning.history import QUERY_INPUT_CONTRACT, FullPrefixHistory, LiveDecision
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.registry.store import file_hash

Payload = dict[str, Any]


class Brain(Protocol):
    label: str

    def observe(self, observation: Payload) -> None: ...

    def act(self, observation: Payload) -> list[int]: ...


def greedy(observation: Payload) -> list[int]:
    from deepmew.policy.greedy import choose_greedy_action
    return choose_greedy_action(observation)


class GreedyBrain:
    label = 'Greedy baseline'

    def observe(self, observation: Payload) -> None:
        pass

    def act(self, observation: Payload) -> list[int]:
        return greedy(observation)


class RulesBrain:
    """BabyMew v2.4 hand-written strategy (the frozen DeepMew submission policy)."""
    label = 'BabyMew rules (v2.4)'

    def observe(self, observation: Payload) -> None:
        pass

    def act(self, observation: Payload) -> list[int]:
        from deepmew.policy import choose_action
        try:
            return list(choose_action(observation))
        except Exception:
            return greedy(observation)


@dataclass(frozen=True)
class CoachModel:
    model: TemporalPolicy
    public_event_mode: bool
    checkpoint: Path
    updates: int
    presentations: int


def load_coach_model(checkpoint: Path, threads: int = 2) -> CoachModel:
    """Load a hash-verified training checkpoint for CPU inference (never touches the GPU)."""
    sidecar = json.loads(checkpoint.with_suffix(checkpoint.suffix + '.json').read_text(encoding='utf-8'))
    if file_hash(checkpoint) != sidecar.get('sha256'):
        raise ValueError(f'checkpoint hash does not match its sidecar: {checkpoint}')
    torch.set_num_threads(threads)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = TemporalPolicy(**saved['model_config'])
    model.load_state_dict(saved['model'])
    model.requires_grad_(False).eval()
    return CoachModel(model, saved.get('input_contract') == QUERY_INPUT_CONTRACT, checkpoint, int(saved['updates']), int(saved['presentations']))


def find_checkpoints(roots: list[Path]) -> list[Path]:
    """Hash-sidecar checkpoints under training run roots, most-trained first."""
    found = []
    for root in roots:
        for path in root.glob('**/checkpoints/*.pt'):
            sidecar = path.with_suffix('.pt.json')
            if sidecar.is_file():
                found.append((json.loads(sidecar.read_text(encoding='utf-8')).get('presentations', 0), path))
    return [path for _, path in sorted(found, key=lambda row: row[0], reverse=True)]


@dataclass
class Advice:
    ranked: list[tuple[tuple[int, ...], float]]  # complete selections (option positions) with probability
    pick: list[int]
    source: str


class ModelBrain:
    def __init__(self, coach: CoachModel, label: str = 'BC model') -> None:
        self.coach, self.label = coach, label
        self.history = FullPrefixHistory(coach.public_event_mode)
        self.pending: LiveDecision | None = None
        self.advice: Advice | None = None
        self.broken: str | None = None

    def observe(self, observation: Payload) -> None:
        if self.broken is None:
            try:
                self.history.observe(observation)
            except Exception as error:
                self.broken = f'history: {error}'

    def advise(self, observation: Payload) -> Advice:
        """Rank this seat's legal options; keeps the decision pending until ``commit``."""
        if self.advice is not None:
            return self.advice
        if self.broken is not None:
            self.advice = Advice([], greedy(observation), f'greedy fallback ({self.broken})')
            return self.advice
        try:
            decision = self.history.prepare(observation)
            self.pending = decision
            if decision.fallback_reason is not None:
                pick = greedy(observation)
                self.advice = Advice([(tuple(pick), 1.0)], pick, 'only legal choice' if decision.fallback_reason == 'forced' else 'greedy fallback (unsupported prompt)')
                return self.advice
            stage = decision.stages[0]
            positions = [int(option['position']) for option in decision.contract['stages'][0]['options']]
            with torch.no_grad():
                records, context = self.coach.model.encode(decision.sequence)
                logits = self.coach.model.scorer(records[list(stage.options)], context[decision.event_index])
                probabilities = torch.softmax(logits, dim=0).tolist()
                decoded = self.coach.model.decode(decision.sequence, decision.event_index, decision.stages)
            if len(decoded) != 1:
                raise ValueError('native selection requires one flattened stage')
            pick = [positions[index] for index in decoded[0]]
            ranked = sorted((((position,), probability) for position, probability in zip(positions, probabilities)), key=lambda row: row[1], reverse=True)
            self.advice = Advice(ranked, pick, 'model')
        except Exception as error:
            self.broken = f'model: {error}'
            self.advice = Advice([], greedy(observation), f'greedy fallback ({self.broken})')
        return self.advice

    def commit(self, selection: list[int]) -> None:
        """Record what this seat actually played (the model's pick or the human's)."""
        decision, self.pending, self.advice = self.pending, None, None
        if decision is not None and self.broken is None:
            try:
                self.history.commit(decision, [list(selection)])
            except Exception as error:
                self.broken = f'commit: {error}'

    def act(self, observation: Payload) -> list[int]:
        pick = self.advise(observation).pick
        self.commit(pick)
        return pick


class ExternalBrain:
    """A downloaded Kaggle submission (``main.agent``) run in an isolated ``python -I`` subprocess."""

    def __init__(self, submission: Path, engine_source: Path, label: str) -> None:
        self.label = label
        worker = Path(__file__).with_name('external_worker.py')
        # Kaggle agents run on CPU; an empty CUDA_VISIBLE_DEVICES (used to keep the coach
        # off the GPU) crashes torch's cuDNN probe on import, so the worker drops it.
        environment = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
        self.proc = subprocess.Popen([sys.executable, '-I', str(worker), str(engine_source)], cwd=submission, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def ask(self, observation: Payload) -> list[int]:
        return self.ask_full(observation)['ok']

    def ask_full(self, observation: Payload) -> Payload:
        self.proc.stdin.write(json.dumps(observation) + '\n')
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline() or '{"error": "worker exited"}')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def reset(self) -> None:
        """Kaggle starts every game with a deck request (select=None); agents reset per-game state on it."""
        self.ask({'select': None, 'logs': [], 'current': None})

    def observe(self, observation: Payload) -> None:
        pass

    def act(self, observation: Payload) -> list[int]:
        try:
            return self.ask(observation)
        except Exception:
            return greedy(observation)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


class ExternalCoach(ExternalBrain):
    """Coach backed by an external agent: its pick plus a softmax over its scored complete actions."""

    def __init__(self, submission: Path, engine_source: Path, label: str) -> None:
        super().__init__(submission, engine_source, label)
        self.advice: Advice | None = None

    def advise(self, observation: Payload) -> Advice:
        if self.advice is not None:
            return self.advice
        try:
            reply = self.ask_full(observation)
            pick = list(reply['ok'])
            scores = reply.get('scores')
            if scores and len(scores['actions']) > 1:
                logits = torch.tensor(scores['scores'], dtype=torch.float64)
                probabilities = torch.softmax(logits - logits.max(), dim=0).tolist()
                ranked = sorted(((tuple(int(v) for v in action), probability) for action, probability in zip(scores['actions'], probabilities)), key=lambda row: row[1], reverse=True)
                source = 'lethal search: this line wins the game this turn' if scores.get('override') else self.label
            else:
                ranked, source = [(tuple(pick), 1.0)], 'only sensible choice'
            self.advice = Advice(ranked, pick, source)
        except Exception as error:
            self.advice = Advice([], greedy(observation), f'greedy fallback ({error})')
        return self.advice

    def commit(self, selection: list[int]) -> None:
        self.advice = None
