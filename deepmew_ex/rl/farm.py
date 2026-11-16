"""Parallel native game farm with one batched inference server.

Worker processes (``deepmew_ex.rl.worker``) each play one game at a time and
send every learned-seat decision here. ``InferenceServer`` keeps per-game
caches (``CachePool``) and answers all pending requests of all workers in one
batch. ``Farm`` routes jobs by worker kind: checkpoint opponents (D1) need a
torch-capable worker, the external agent a worker that hosts it.

Messages, worker -> main (one shared queue):
  ('ready', worker), ('act', worker, key, tag, new-record tables, events, event_index, stages, greedy),
  ('done', worker, result), ('error', worker, traceback)
main -> worker (one queue each): ('job', job), ('reply', selected, logp, value, entropy), ('stop',)
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from deepmew_ex.rl.incremental import CachePool
from deepmew_ex.rl.ppo import ValueHead
from deepmew_ex.rl.sampling import draw, greedy_subset, sample_subset, subset_logp
from deepmew_ex.rl.worker import EXTERNAL, ExternalSeat, FarmSettings, Opponents, RemoteSeat, play, torch_opponents, worker_main  # re-exported


@dataclass
class Net:
    model: torch.nn.Module
    value: ValueHead


class InferenceServer:
    """Answers every pending learned-seat decision of every worker in one batch per net."""

    def __init__(self, nets: dict[str, Net], seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.stats: Counter[str] = Counter()
        self.nets: dict[str, Net] = {}
        self.pools: dict[str, CachePool] = {}
        self.refresh(nets)

    def refresh(self, nets: dict[str, Net] | None = None) -> None:
        """Weights changed (or nets were added/removed): every cached encoding is stale."""
        if any(pool.slots for pool in self.pools.values()):
            raise RuntimeError('cannot refresh the server while games are in flight')
        self.nets = nets if nets is not None else self.nets
        self.pools = {tag: CachePool(net.model) for tag, net in self.nets.items()}

    def drop(self, keys: list[tuple]) -> None:
        for pool in self.pools.values():
            for key in keys:
                pool.release(tuple(key))

    @torch.no_grad()
    def serve(self, requests: list[tuple]) -> list[tuple]:
        """Each request: (key, tag, new-record leaf tables, events, event_index, stages, greedy) -> (selected, logp, value, entropy)."""
        replies: list[tuple | None] = [None] * len(requests)
        groups: dict[str, list[int]] = defaultdict(list)
        for position, request in enumerate(requests):
            groups[request[1]].append(position)
        for tag, positions in groups.items():
            net, pool = self.nets[tag], self.pools[tag]
            model = net.model
            keys = [tuple(requests[position][0]) for position in positions]
            pool.extend(keys, [requests[position][2] for position in positions], [requests[position][3] for position in positions])
            slots = np.array([pool.slots[key] for key in keys], dtype=np.int64)
            events = np.array([requests[position][4] for position in positions], dtype=np.int64)
            if (events != pool.event_counts[slots] - 1).any():
                raise ValueError('live decision is not the newest causal event')
            device = pool.device
            # Every single-stage unordered decision (pick-one included) in one scorer call.
            single = [row for row, position in enumerate(positions) if len(requests[position][5]) == 1 and not requests[position][5][0].ordered]
            host_index = [slots, events]
            if single:
                stages = [requests[positions[row]][5][0] for row in single]
                width = max(1, max(len(stage.options) for stage in stages))
                index = np.zeros((len(single), width), dtype=np.int64)
                for line, stage in enumerate(stages):
                    index[line, :len(stage.options)] = stage.options
                lengths = np.array([len(stage.options) for stage in stages], dtype=np.int64)
                top = max(stage.maximum for stage in stages)
                host_index += [np.array(single, dtype=np.int64), index.reshape(-1), lengths]
            flat = pool.server_upload(np.concatenate(host_index))
            parts = flat.split([len(part) for part in host_index])
            slot_t = parts[0]
            states = pool.context[slot_t, parts[1]]
            values = net.value(states)
            host = values.float()
            if single:
                single_t, gather, lengths_t = parts[2], parts[3].view(index.shape), parts[4]
                chosen_states = states[single_t]
                logits = model.scorer(pool.records[slot_t[single_t][:, None], gather], chosen_states)
                counts = torch.arange(top + 1, device=device, dtype=logits.dtype)
                ratio = counts[None, :] / lengths_t.clamp(min=1).to(logits.dtype)[:, None]
                features = torch.stack((ratio, ratio.square(), torch.log1p(counts).expand_as(ratio)), dim=2)
                count_scores = (features @ model.count(chosen_states).unsqueeze(2)).squeeze(2)
                host = torch.cat((host, logits.reshape(-1).float(), count_scores.reshape(-1).float()))
            host = host.double().cpu().numpy()
            value_host = host[:len(positions)]
            if single:
                logit_host = host[len(positions):len(positions) + index.size].reshape(index.shape)
                count_host = host[len(positions) + index.size:].reshape(len(single), top + 1)
                for line, row in enumerate(single):
                    stage, greedy = stages[line], requests[positions[row]][6]
                    scores = logit_host[line, :lengths[line]]
                    if not np.isfinite(scores).all() or not np.isfinite(count_host[line]).all():
                        raise FloatingPointError('nonfinite live action scores')
                    entropy = None
                    if stage.minimum == stage.maximum == 1:
                        logs = scores - np.logaddexp.reduce(scores)
                        probabilities = np.exp(logs)
                        choice = int(np.argmax(scores)) if greedy else int(self.rng.choice(len(scores), p=probabilities / probabilities.sum()))
                        selected, logp = (choice,), float(logs[choice])
                        entropy = float(-(probabilities * logs).sum())
                    else:
                        counts_row = count_host[line, :stage.maximum + 1]
                        selected = greedy_subset(scores, counts_row, stage.minimum, stage.maximum) if greedy else sample_subset(scores, counts_row, stage.minimum, stage.maximum, self.rng)
                        logp = subset_logp(scores, counts_row, selected, stage.minimum, stage.maximum)
                        self.stats['subset'] += 1
                    replies[positions[row]] = ((selected,), logp, float(value_host[row]), entropy)
            for row, position in enumerate(positions):
                if replies[position] is None:
                    result = draw(model, states[row], pool.game_records(keys[row]), requests[position][5], self.rng, greedy=requests[position][6])
                    replies[position] = (result.selected, result.logp, float(value_host[row]), result.entropy)
                    self.stats['structured'] += 1
            self.stats['decisions'] += len(positions)
        self.stats['batches'] += 1
        return replies


class Farm:
    """Worker kinds: indices [0, torch_workers) can run checkpoint opponents (D1);
    the last ``external_workers`` host the external agent; the rest are lean."""

    def __init__(self, settings: FarmSettings, workers: int, external_workers: int, torch_workers: int | None = None) -> None:
        context = mp.get_context('spawn')
        torch_workers = workers if torch_workers is None else min(torch_workers, workers)
        self.inbox = context.Queue()
        self.outboxes = [context.Queue() for _ in range(workers)]
        self.torch_capable = set(range(torch_workers))
        self.external_capable = set(range(workers - min(external_workers, workers), workers))
        self.needs_torch = torch_opponents(settings.project_root)
        self.processes = [context.Process(target=worker_main, args=(index, settings, self.inbox, self.outboxes[index], index in self.torch_capable), daemon=True) for index in range(workers)]
        for process in self.processes:
            process.start()
        ready = set()
        while len(ready) < workers:
            message = self.inbox.get(timeout=600)
            if message[0] == 'error':
                raise RuntimeError(f'farm worker {message[1]} failed to start:\n{message[2]}')
            ready.add(message[1])

    def can_run(self, worker: int, job: dict) -> bool:
        if job['opponent'] == EXTERNAL:
            return worker in self.external_capable
        if job['opponent'] in self.needs_torch:
            return worker in self.torch_capable
        return True

    def rarity(self, job: dict) -> int:
        """Jobs only a few workers can run are dispatched first."""
        return 0 if job['opponent'] == EXTERNAL or job['opponent'] in self.needs_torch else 1

    def run(self, jobs: list[dict], server: InferenceServer, on_result=None) -> tuple[list[dict], dict]:
        pending = list(jobs)
        idle = set(range(len(self.processes)))
        busy = 0
        results, batch_sizes = [], []
        started = time.perf_counter()
        serve_seconds = 0.0

        def dispatch() -> None:
            nonlocal busy
            for worker in sorted(idle):
                choices = [index for index, job in enumerate(pending) if self.can_run(worker, job)]
                if not choices:
                    continue
                choice = min(choices, key=lambda index: self.rarity(pending[index]))
                self.outboxes[worker].put(('job', pending.pop(choice)))
                idle.discard(worker)
                busy += 1

        dispatch()
        while busy:
            messages = [self.inbox.get(timeout=900)]
            while True:
                try:
                    messages.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            acts = []
            for message in messages:
                if message[0] == 'act':
                    acts.append(message)
                elif message[0] == 'done':
                    result = message[2]
                    server.drop(result['keys'])
                    results.append(result)
                    idle.add(message[1])
                    busy -= 1
                    if on_result:
                        on_result(result)
                elif message[0] == 'error':
                    raise RuntimeError(f'farm worker {message[1]} failed:\n{message[2]}')
            if acts:
                began = time.perf_counter()
                replies = server.serve([message[2:] for message in acts])
                serve_seconds += time.perf_counter() - began
                batch_sizes.append(len(acts))
                for message, reply in zip(acts, replies, strict=True):
                    self.outboxes[message[1]].put(('reply', *reply))
            dispatch()
        if pending:
            raise RuntimeError('farm jobs left undispatched (no capable worker)')
        elapsed = time.perf_counter() - started
        return results, {'games': len(results), 'seconds': elapsed, 'games_per_second': len(results) / max(elapsed, 1e-9), 'serve_seconds': serve_seconds, 'mean_batch': float(np.mean(batch_sizes)) if batch_sizes else 0.0}

    def close(self) -> None:
        for outbox in self.outboxes:
            outbox.put(('stop',))
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
