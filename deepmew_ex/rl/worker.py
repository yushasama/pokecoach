"""Torch-free farm worker: one native game at a time per process.

A worker builds actor-visible histories, runs frozen outside opponents
locally (reference notebooks; D1 on CPU in torch-capable workers only; the
external Kaggle agent in its own isolated subprocess) and steps the engine.
Every decision of a learned seat goes to the main-process server as only the
records/events added since that seat last asked. Lean workers never import
torch (about 70 MB instead of 520 MB each), so many more games can be in
flight; ``TorchBlocker`` turns an accidental import into a loud error.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from deepmew_ex.rl.records import Decision, Trajectory, tables_of

EXTERNAL = 'z3rox_7th'


class TorchBlocker:
    def find_spec(self, name: str, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('torch import in a lean farm worker (job routed to the wrong worker kind)')
        return None


def read_deck(path: Path) -> list[int]:
    cards = [int(item) for item in path.read_text().split()]
    if len(cards) != 60:
        raise ValueError('frozen deck must have sixty cards')
    return cards


def torch_opponents(project_root: str) -> set[str]:
    """League members whose policy is a checkpoint (they need torch in the worker)."""
    import json
    registry = json.loads((Path(project_root) / 'league/registry.json').read_text(encoding='utf-8'))
    return {member['id'] for member in registry['latest_training_population'] if member.get('checkpoint_path')}


@dataclass(frozen=True)
class FarmSettings:
    project_root: str
    runtime_dir: str  # reference notebooks run inside per-opponent copies of their deck here
    our_deck: tuple[int, ...]
    external_submission: str | None
    seat_seconds: float = 60.0
    max_selections: int = 4000


class RemoteSeat:
    """Actor-visible full-prefix history whose decisions are made by the main-process server."""

    def __init__(self, key: tuple, tag: str, greedy: bool, inbox: mp.Queue, outbox: mp.Queue, worker: int, public_event_mode: bool) -> None:
        from deepmew_ex.datasets.v2.learning.history import FullPrefixHistory
        self.key, self.tag, self.greedy = key, tag, greedy
        self.inbox, self.outbox, self.worker = inbox, outbox, worker
        self.label = f'learned:{tag}'
        self.history = FullPrefixHistory(public_event_mode)
        self.sent_records = self.sent_events = 0
        self.decisions: list[Decision] = []
        self.counts: Counter[str] = Counter()
        self.wait_seconds = 0.0

    def observe_actor(self, actor: int) -> None:
        self.history.observe_actor(actor)

    def observe(self, observation: dict) -> None:
        self.history.observe(observation)

    def select(self, observation: dict, rng: random.Random) -> list[int]:
        from deepmew_ex.datasets.v2.learning.trajectory import StageTarget
        from deepmew_ex.evaluation.league.seats import greedy_fallback
        decision = self.history.prepare(observation)
        if decision.fallback_reason is None:
            records, events = decision.sequence.records, decision.sequence.events
            started = time.perf_counter()
            self.inbox.put(('act', self.worker, self.key, self.tag, tables_of(list(records[self.sent_records:])), list(events[self.sent_events:]), decision.event_index, decision.stages, self.greedy))
            self.sent_records, self.sent_events = len(records), len(events)
            message = self.outbox.get()
            self.wait_seconds += time.perf_counter() - started
            if message[0] != 'reply':
                raise RuntimeError(f'unexpected server message {message[0]!r}')
            _, chosen, logp, value, entropy = message
            selected = [[int(decision.contract['stages'][stage]['options'][index]['position']) for index in indices] for stage, indices in enumerate(chosen)]
            targets = tuple(StageTarget(stage.options, tuple(indices), stage.minimum, stage.maximum, stage.ordered, stage.repeated) for stage, indices in zip(decision.stages, chosen, strict=True))
            self.decisions.append(Decision(decision.event_index, targets, float(logp), float(value), entropy))
            self.counts['model_choices'] += 1
        else:
            selected = [greedy_fallback(observation)]
            self.counts['fallback:' + decision.fallback_reason] += 1
        self.history.commit(decision, selected)
        if len(selected) != 1:
            raise ValueError('native CABT selection requires one flattened stage result')
        return selected[0]

    def trajectory(self, job: dict, reward: float) -> Trajectory | None:
        from deepmew_ex.datasets.v2.learning.leaf_tables import leaf_tables
        if not self.decisions:
            return None
        records = tuple(self.history.records)
        return Trajectory(job, leaf_tables(records), len(records), tuple(self.history.events), tuple(self.decisions), reward, {'counts': dict(self.counts)})


class ExternalSeat:
    """A downloaded Kaggle agent in an isolated ``python -I`` subprocess (raw observations, per-game reset)."""

    def __init__(self, submission: Path, engine_source: Path) -> None:
        import subprocess
        worker = Path(__file__).resolve().parents[1] / 'coach' / 'external_worker.py'
        environment = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
        environment.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        self.proc = subprocess.Popen([sys.executable, '-I', str(worker), str(engine_source)], cwd=submission, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.label, self.errors, self.latencies = EXTERNAL, 0, []

    def ask(self, observation: dict) -> list[int]:
        import json
        self.proc.stdin.write(json.dumps(observation) + '\n')
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline() or '{"error": "external worker exited"}')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply['ok']

    def reset(self) -> None:
        self.ask({'select': None, 'logs': [], 'current': None})

    def select(self, observation: dict, rng: random.Random) -> list[int]:
        started = time.perf_counter()
        try:
            return self.ask(observation)
        finally:
            self.latencies.append(time.perf_counter() - started)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


class Opponents:
    """Lazily built frozen outside opponents for one worker (league construction pattern)."""

    def __init__(self, settings: FarmSettings, worker: int = 0) -> None:
        from deepmew_ex.evaluation.league.config import LeagueConfig, population
        self.worker = worker
        from deepmew_ex.experiments.config import OutputPaths
        self.settings = settings
        root = Path(settings.project_root)
        config = LeagueConfig((), OutputPaths(root / 'outputs'), 'rl-farm', canary=True)
        self.bindings = {row.identity: row for row in population(config)}
        self.models: dict[str, tuple] = {}
        self.external: ExternalSeat | None = None

    def deck(self, identity: str) -> list[int]:
        if identity == EXTERNAL:
            return read_deck(Path(self.settings.external_submission) / 'deck.csv')
        return read_deck(self.bindings[identity].deck)

    def seat(self, identity: str):
        from deepmew_ex.evaluation.league.seats import ReferenceSeatPolicy, reference_directory
        if identity == EXTERNAL:
            if self.external is None:
                self.external = ExternalSeat(Path(self.settings.external_submission), Path(self.settings.project_root) / 'source')
            self.external.reset()
            return self.external
        row = self.bindings[identity]
        if row.checkpoint is not None:
            from deepmew_ex.evaluation.league.policies import legacy_checkpoint_policy  # torch: full workers only
            if identity not in self.models:
                import torch
                torch.set_num_threads(1)
                from run_r9_r4f_retrain import load_r4f_model
                model, metadata = load_r4f_model(row.checkpoint)
                model.requires_grad_(False)
                self.models[identity] = (model, metadata)
            model, metadata = self.models[identity]
            return legacy_checkpoint_policy(row.label, row.checkpoint, model, metadata, read_deck(row.deck), row.feature_policy_id)
        from deepmew.rl1_teacher_bc.teachers import make_teacher
        directory = Path(self.settings.runtime_dir) / f'worker-{self.worker:02d}' / identity
        if not (directory / 'deck.csv').is_file():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'deck.csv').write_bytes(row.deck.read_bytes())
        with reference_directory(directory):
            teacher = make_teacher(row.reference_teacher)
        if type(teacher).__name__ != 'NotebookTeacherAgent' or teacher._module is None:
            raise ValueError('reference notebook unavailable; proxy substitution rejected')
        return ReferenceSeatPolicy(teacher, directory)

    def close(self) -> None:
        if self.external is not None:
            self.external.close()


def play(job: dict, settings: FarmSettings, opponents: Opponents, inbox: mp.Queue, outbox: mp.Queue, worker: int, game: int) -> dict:
    from deepmew.real_meta_arena.runner import run_real_meta_game
    from deepmew_ex.evaluation.league.seats import ActorNotifyingSeat, IndependentSeat
    seat, seed = int(job['seat']), int(job['seed'])
    ours = RemoteSeat((worker, game, 'ours'), job['tag'], bool(job.get('greedy')), inbox, outbox, worker, bool(job['public_event_mode']))
    opponent_id = job['opponent']
    if opponent_id.startswith('snapshot:'):
        policy = RemoteSeat((worker, game, 'opponent'), opponent_id.removeprefix('snapshot:'), bool(job.get('greedy')), inbox, outbox, worker, bool(job['public_event_mode']))
        opponent_deck = list(settings.our_deck)
    else:
        policy = opponents.seat(opponent_id)
        opponent_deck = opponents.deck(opponent_id)
    observers = tuple(item for item in (ours, policy) if isinstance(item, RemoteSeat))
    our_seat = IndependentSeat(ours, seed + seat * 1000003, settings.seat_seconds)
    other_seat = policy if isinstance(policy, ExternalSeat) else IndependentSeat(policy, seed + (1 - seat) * 1000003, settings.seat_seconds)
    started = time.perf_counter()
    record = run_real_meta_game(ActorNotifyingSeat(our_seat, observers), ActorNotifyingSeat(other_seat, observers), list(settings.our_deck), opponent_deck, seat_our=seat, max_steps=settings.max_selections, rng=random.Random(seed), game_index=game, rng_seed=seed)
    elapsed = time.perf_counter() - started
    outcome = record.outcome
    reward = {'win': 1.0, 'loss': -1.0, 'draw': 0.0}.get(outcome)
    result = {'job': job, 'outcome': outcome, 'turns': record.turns, 'error': record.error, 'elapsed': elapsed, 'server_wait': ours.wait_seconds + (policy.wait_seconds if isinstance(policy, RemoteSeat) else 0.0),
              'our_calls': our_seat.calls, 'our_latency_max': max(our_seat.latencies, default=0.0), 'our_counts': dict(ours.counts), 'opponent_errors': int(getattr(getattr(other_seat, 'policy', other_seat), 'fallback_errors', 0) or getattr(other_seat, 'errors', 0)),
              'keys': [item.key for item in observers], 'torch_loaded': 'torch' in sys.modules}
    result['trajectory'] = ours.trajectory(job, reward) if reward is not None and job.get('train', True) else None
    return result


def worker_main(worker: int, settings: FarmSettings, inbox: mp.Queue, outbox: mp.Queue, torch_capable: bool = True) -> None:
    try:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        if not torch_capable:
            sys.meta_path.insert(0, TorchBlocker())  # lean worker: any torch import is a routing bug
        sys.dont_write_bytecode = True
        from deepmew_ex.league import bind_legacy_source
        bind_legacy_source(Path(settings.project_root))
        opponents = Opponents(settings, worker)
        inbox.put(('ready', worker))
        game = 0
        while True:
            message = outbox.get()
            if message[0] == 'stop':
                break
            if message[0] != 'job':
                raise RuntimeError(f'unexpected message {message[0]!r} while idle')
            game += 1
            inbox.put(('done', worker, play(message[1], settings, opponents, inbox, outbox, worker, game)))
        opponents.close()
    except BaseException:
        inbox.put(('error', worker, traceback.format_exc()))
