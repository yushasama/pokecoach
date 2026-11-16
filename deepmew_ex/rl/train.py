"""League PPO fine-tuning of the behaviour-cloned temporal actor.

    python -m deepmew_ex.rl.train --config experiments/deepmew_ex/rl-league-pilot.json --run-id <id>
    python -m deepmew_ex.rl.train --config ... --run-id <id> --resume <run>/checkpoints/iter-000040.pt

Each iteration: the farm plays ``games_per_iteration`` games of the current
policy (sampling) against PFSP-chosen league members, then PPO (GAE(lambda),
clipped ratio, entropy bonus, optional UPGO term) updates policy + value head
for a few epochs with a KL early stop. Synchronous and on-policy: workers idle
while the GPU trains. Outputs (``output_root/<run_id>/``): config.json,
code.json, progress.jsonl (one row per iteration), games.jsonl (one row per
game), league.json, checkpoints/iter-*.pt with SHA-256 sidecars. Touch
``STOP`` in the run directory for a graceful checkpoint-and-exit.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.store import canonical, digest, file_hash
from deepmew_ex.rl.farm import EXTERNAL, Farm, FarmSettings, InferenceServer, Net
from deepmew_ex.rl.league import League
from deepmew_ex.rl.ppo import Trajectory, ValueHead, gae, policy_terms, ppo_policy_loss, upgo

CHECKPOINT_KIND = 'deepmew_ex_rl_league_v1'


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 2
    clip: float = 0.2
    lam: float = 0.95
    gamma: float = 1.0
    entropy: float = 0.003
    value_coef: float = 1.0
    learning_rate: float = 2e-5
    value_learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    target_kl: float = 0.03
    minibatch_games: int = 8
    minibatch_leaves: int = 150_000
    upgo: float = 0.0
    value_warmup_iterations: int = 3
    value_trunk_gradient: bool = False


@dataclass(frozen=True)
class LeagueSettings:
    floor: float = 0.15
    ema: float = 0.1
    snapshot_share: float = 0.2
    max_snapshots: int = 4
    snapshot_every: int = 10


@dataclass(frozen=True)
class RLConfig:
    bc_checkpoint: str
    bc_sha256: str
    deck: str
    outside_opponents: dict[str, float | None]
    output_root: str = 'D:/deepmew_ex/outputs/rl-runs'
    external_submission: str | None = None
    workers: int = 9
    external_workers: int = 2
    torch_workers: int | None = None  # workers able to run checkpoint opponents (D1); None = all
    games_per_iteration: int = 192
    iterations: int = 100000
    max_wall_seconds: float = 4 * 3600
    checkpoint_every: int = 10
    seed: int = 20261115
    minimum_free_bytes: int = 10 * 2**30
    system_minimum_free_bytes: int = 4 * 2**30
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueSettings = field(default_factory=LeagueSettings)

    @classmethod
    def load(cls, path: Path) -> 'RLConfig':
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw['ppo'] = PPOConfig(**raw.get('ppo', {}))
        raw['league'] = LeagueSettings(**raw.get('league', {}))
        return cls(**raw)

    def validate(self) -> None:
        if not self.outside_opponents or self.games_per_iteration < 1 or self.workers < 1:
            raise ValueError('RL config needs outside opponents, games and workers')
        if EXTERNAL in self.outside_opponents and (not self.external_submission or self.external_workers < 1):
            raise ValueError('the external agent needs a submission path and at least one capable worker')
        if not 0 <= self.league.snapshot_share < 0.5:
            raise ValueError('snapshots may not dominate the league (invariant #2: outside opponents anchor it)')


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_policy(config: RLConfig, device: str) -> tuple[torch.nn.Module, dict]:
    from deepmew_ex.evaluation.league.temporal import load_full_prefix_checkpoint
    model, evidence = load_full_prefix_checkpoint(resolve(config.bc_checkpoint), config.bc_sha256, device)
    model.requires_grad_(True)
    return model, evidence


def compact(model: torch.nn.Module) -> torch.nn.Module:
    """Copied/loaded GRU weights lose cuDNN's single flat buffer; restore it."""
    model.encoder.string.flatten_parameters()
    return model


def frozen_copy(model: torch.nn.Module, value: ValueHead) -> Net:
    snapshot, head = compact(copy.deepcopy(model)).eval(), copy.deepcopy(value).eval()
    snapshot.requires_grad_(False)
    head.requires_grad_(False)
    return Net(snapshot, head)


def save_checkpoint(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    path.with_suffix(path.suffix + '.json').write_text(json.dumps({'sha256': file_hash(path), 'kind': CHECKPOINT_KIND, 'iteration': payload['iteration'], 'games': payload['games'], 'parent_sha256': payload['parent']['sha256']}, indent=1), encoding='utf-8')
    return path


def load_rl_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, ValueHead, dict]:
    """An RL checkpoint's policy + value head (sidecar hash verified)."""
    from deepmew_ex.models.temporal import TemporalPolicy
    sidecar = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf-8'))
    if file_hash(path) != sidecar['sha256']:
        raise ValueError(f'RL checkpoint hash does not match its sidecar: {path}')
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('kind') != CHECKPOINT_KIND:
        raise ValueError('not a DeepMew EX RL league checkpoint')
    model = TemporalPolicy(**saved['model_config']).to(device)
    model.load_state_dict(saved['model'])
    compact(model)
    head = ValueHead(model.hidden).to(device)
    head.load_state_dict(saved['value_head'])
    return model, head, saved


def minibatches(order: np.ndarray, trajectories: list[Trajectory], games: int, leaves: int) -> list[list[int]]:
    batches, current, budget = [], [], 0
    for index in order.tolist():
        size = trajectories[index].tables.distinct_leaves
        if current and (len(current) >= games or budget + size > leaves):
            batches.append(current)
            current, budget = [], 0
        current.append(index)
        budget += size
    if current:
        batches.append(current)
    return batches


def advantages(trajectories: list[Trajectory], ppo: PPOConfig) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    adv, ret, up = [], [], []
    for trajectory in trajectories:
        values = np.array([decision.value for decision in trajectory.decisions])
        rewards = np.zeros(len(values))
        rewards[-1] = trajectory.reward
        a, r = gae(rewards, values, ppo.lam, ppo.gamma)
        adv.append(a)
        ret.append(r)
        up.append(upgo(rewards, values, ppo.gamma) - values)
    return adv, ret, up


def update(model: torch.nn.Module, value: ValueHead, optimizer: torch.optim.Optimizer, trajectories: list[Trajectory], ppo: PPOConfig, policy_on: bool, rng: np.random.Generator) -> dict:
    adv, ret, up = advantages(trajectories, ppo)
    flat = np.concatenate(adv)
    mean, std = float(flat.mean()), float(flat.std() + 1e-8)
    up_flat = np.concatenate(up)
    up_scale = float(np.abs(up_flat).mean() + 1e-8)
    old = [np.array([decision.logp for decision in trajectory.decisions]) for trajectory in trajectories]
    device = next(model.parameters()).device
    model.train()
    stats: dict[str, list[float]] = defaultdict(list)
    epochs_run = 0
    for epoch in range(ppo.epochs):
        epochs_run += 1
        kls = []
        for batch in minibatches(rng.permutation(len(trajectories)), trajectories, ppo.minibatch_games, ppo.minibatch_leaves):
            sequences = [trajectories[index].sequence() for index in batch]
            logp, entropy, values = policy_terms(model, value, sequences, detach_value=not ppo.value_trunk_gradient)
            to = lambda arrays: torch.from_numpy(np.concatenate([arrays[index] for index in batch])).to(device, torch.float32)
            advantage = (to(adv) - mean) / std
            target, old_logp = to(ret), to(old)
            policy_loss, policy_stats = ppo_policy_loss(logp, old_logp, advantage, ppo.clip)
            value_loss = 0.5 * (values - target).square().mean()
            known = ~torch.isnan(entropy)
            entropy_mean = entropy[known].mean() if known.any() else logp.new_zeros(())
            loss = ppo.value_coef * value_loss
            if policy_on:
                loss = loss + policy_loss - ppo.entropy * entropy_mean
                if ppo.upgo:
                    ratio = torch.exp(logp - old_logp).detach().clamp(max=1.0)
                    loss = loss - ppo.upgo * (ratio * to(up) / up_scale * logp).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *value.parameters()], ppo.gradient_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite PPO gradient')
            optimizer.step()
            kls.append(policy_stats['approx_kl'])
            for key, number in (('policy_loss', policy_loss.item()), ('value_loss', value_loss.item()), ('entropy', entropy_mean.item()), ('grad_norm', norm.item()), *policy_stats.items()):
                stats[key].append(float(number))
        if policy_on and np.mean(kls) > ppo.target_kl:
            break
    model.eval()
    values = np.concatenate([[decision.value for decision in trajectory.decisions] for trajectory in trajectories])
    returns = np.concatenate(ret)
    explained = 1 - float(np.var(returns - values) / (np.var(returns) + 1e-8))
    return {key: float(np.mean(numbers)) for key, numbers in stats.items()} | {'epochs': epochs_run, 'decisions': int(len(flat)), 'advantage_mean': mean, 'advantage_std': std, 'explained_variance': explained, 'policy_updates': policy_on}


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def train(config: RLConfig, run_id: str, resume: Path | None = None, iterations: int | None = None) -> Path:
    config.validate()
    device = 'cuda'
    torch.set_num_threads(2)
    run = Path(config.output_root) / run_id
    (run / 'checkpoints').mkdir(parents=True, exist_ok=resume is not None)
    resolved = asdict(config)
    if not resume:
        (run / 'config.json').write_text(json.dumps(resolved, indent=1), encoding='utf-8')
        from deepmew_ex.registry.provenance import code_identity
        (run / 'code.json').write_text(json.dumps(code_identity(PROJECT_ROOT), indent=1), encoding='utf-8')
    model, evidence = load_policy(config, device)
    value = ValueHead(model.hidden).to(device)
    optimizer = torch.optim.Adam([{'params': list(model.parameters()), 'lr': config.ppo.learning_rate}, {'params': list(value.parameters()), 'lr': config.ppo.value_learning_rate}])
    league = League(floor=config.league.floor, ema=config.league.ema, snapshot_share=config.league.snapshot_share, max_snapshots=config.league.max_snapshots)
    for identity, share in config.outside_opponents.items():
        league.add_outside(identity, share)
    snapshots: dict[str, Net] = {}
    iteration, games_total, elapsed_before = 0, 0, 0.0
    rng = np.random.default_rng(config.seed)
    if resume is not None:
        model, value, saved = load_rl_checkpoint(resume, device)
        model.requires_grad_(True)
        optimizer = torch.optim.Adam([{'params': list(model.parameters()), 'lr': config.ppo.learning_rate}, {'params': list(value.parameters()), 'lr': config.ppo.value_learning_rate}])
        optimizer.load_state_dict(saved['optimizer'])
        league = League.from_state(saved['league'])
        iteration, games_total, elapsed_before = saved['iteration'], saved['games'], saved.get('elapsed_seconds', 0.0)
        for tag in saved.get('snapshot_tags', []):
            path = run / 'checkpoints' / f'snapshot-{tag}.pt'
            snap_model, snap_value, _ = load_rl_checkpoint(path, device)
            snapshots[tag] = frozen_copy(snap_model, snap_value)
        rng = np.random.default_rng([config.seed, iteration])
    model.eval()
    deck = tuple(int(card) for card in resolve(config.deck).read_text().split())
    if len(deck) != 60:
        raise ValueError('specialization deck must have sixty cards')
    settings = FarmSettings(str(PROJECT_ROOT), str(run / 'reference-runtime'), deck, str(resolve(config.external_submission)) if config.external_submission else None)
    parent = {'path': str(resolve(config.bc_checkpoint)), 'sha256': config.bc_sha256}
    started = time.perf_counter()
    farm = Farm(settings, config.workers, config.external_workers, config.torch_workers)
    server = InferenceServer({'current': Net(model, value), **snapshots}, seed=int(rng.integers(2**31)))
    limit = iterations if iterations is not None else config.iterations
    stop_reason = 'iterations'

    def checkpoint(name: str) -> Path:
        return save_checkpoint(run / 'checkpoints', name, {'kind': CHECKPOINT_KIND, 'architecture': evidence['architecture'], 'model_config': evidence['model_config'], 'input_contract': evidence['input_contract'], 'public_event_mode': evidence['public_event_mode'], 'model': model.state_dict(), 'value_head': value.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration, 'games': games_total, 'elapsed_seconds': elapsed_before + time.perf_counter() - started, 'parent': parent, 'league': league.state(), 'snapshot_tags': sorted(snapshots), 'config_sha256': digest(canonical(resolved)), 'deck': list(deck)})

    try:
        with (run / 'progress.jsonl').open('a', encoding='utf-8') as progress, (run / 'games.jsonl').open('a', encoding='utf-8') as game_log:
            while iteration < limit:
                if (run / 'STOP').exists():
                    stop_reason = 'stop_file'
                    break
                if elapsed_before + time.perf_counter() - started > config.max_wall_seconds:
                    stop_reason = 'wall_budget'
                    break
                if free_bytes(Path(config.output_root)) < config.minimum_free_bytes or free_bytes(Path('C:/')) < config.system_minimum_free_bytes:
                    stop_reason = 'disk_reserve'
                    break
                iteration += 1
                shares = league.shares()
                opponents = league.schedule(config.games_per_iteration, rng)
                jobs = [{'iteration': iteration, 'index': index, 'opponent': opponent, 'seat': index % 2, 'seed': int(rng.integers(2**31)), 'tag': 'current', 'public_event_mode': evidence['public_event_mode']} for index, opponent in enumerate(opponents)]
                results, farm_stats = farm.run(jobs, server)
                outcomes = Counter()
                per_opponent: dict[str, Counter] = defaultdict(Counter)
                for result in results:
                    opponent, outcome = result['job']['opponent'], result['outcome']
                    outcomes[outcome] += 1
                    per_opponent[opponent][outcome] += 1
                    if outcome in ('win', 'loss', 'draw'):
                        league.update(opponent, {'win': 1.0, 'draw': 0.5, 'loss': 0.0}[outcome])
                    game_log.write(json.dumps({'iteration': iteration, 'opponent': opponent, 'seat': result['job']['seat'], 'seed': result['job']['seed'], 'outcome': outcome, 'turns': result['turns'], 'decisions': len(result['trajectory'].decisions) if result['trajectory'] else 0, 'elapsed': round(result['elapsed'], 4), 'counts': result['our_counts'], 'error': result['error']}) + '\n')
                unsafe = sum(outcomes[kind] for kind in ('crash', 'illegal', 'timeout', 'incomplete'))
                if unsafe > max(2, 0.02 * len(results)):
                    raise RuntimeError(f'too many unsafe training games: {dict(outcomes)}')
                trajectories = [result['trajectory'] for result in results if result['trajectory'] is not None]
                began = time.perf_counter()
                stats = update(model, value, optimizer, trajectories, config.ppo, iteration > config.ppo.value_warmup_iterations, rng)
                train_seconds = time.perf_counter() - began
                games_total += len(results)
                if iteration % config.league.snapshot_every == 0 and iteration > config.ppo.value_warmup_iterations:
                    tag = f'it{iteration:05d}'
                    snapshots[tag] = frozen_copy(model, value)
                    save_checkpoint(run / 'checkpoints', f'snapshot-{tag}.pt', {'kind': CHECKPOINT_KIND, 'architecture': evidence['architecture'], 'model_config': evidence['model_config'], 'input_contract': evidence['input_contract'], 'public_event_mode': evidence['public_event_mode'], 'model': snapshots[tag].model.state_dict(), 'value_head': snapshots[tag].value.state_dict(), 'optimizer': {}, 'iteration': iteration, 'games': games_total, 'parent': parent, 'league': {}, 'snapshot_tags': [], 'config_sha256': digest(canonical(resolved)), 'deck': list(deck)})
                    for evicted in league.add_snapshot(tag, win_rate=0.5):
                        snapshots.pop(evicted, None)
                server.refresh({'current': Net(model, value), **snapshots})
                row = {'iteration': iteration, 'games_total': games_total, 'wall_seconds': round(elapsed_before + time.perf_counter() - started, 1), 'farm': farm_stats, 'train_seconds': train_seconds, 'outcomes': dict(outcomes), 'score_by_opponent': {key: (value_counts['win'] + 0.5 * value_counts['draw']) / max(1, sum(value_counts.values())) for key, value_counts in sorted(per_opponent.items())}, 'games_by_opponent': {key: sum(value_counts.values()) for key, value_counts in sorted(per_opponent.items())}, 'league_shares': shares, 'league_win_rates': {key: member.win_rate for key, member in league.members.items()}, 'update': stats, 'server': dict(server.stats), 'free_gb': {'output': free_bytes(Path(config.output_root)) / 2**30, 'C': free_bytes(Path('C:/')) / 2**30}}
                progress.write(json.dumps(row) + '\n')
                progress.flush()
                game_log.flush()
                score = (outcomes['win'] + 0.5 * outcomes['draw']) / max(1, len(results))
                print(f"it {iteration} games {games_total} score {score:.3f} {farm_stats['games_per_second']:.1f} g/s train {train_seconds:.0f}s kl {stats.get('approx_kl', 0):.4f} clip {stats.get('clip_fraction', 0):.3f} ent {stats.get('entropy', 0):.3f} vloss {stats['value_loss']:.3f} ev {stats['explained_variance']:.2f}", flush=True)
                if iteration % config.checkpoint_every == 0:
                    checkpoint(f'iter-{iteration:06d}.pt')
        final = checkpoint(f'iter-{iteration:06d}.pt')
        (run / 'league.json').write_text(json.dumps(league.state(), indent=1), encoding='utf-8')
        (run / 'summary.json').write_text(json.dumps({'status': 'stopped', 'reason': stop_reason, 'iterations': iteration, 'games': games_total, 'final_checkpoint': str(final), 'final_sha256': file_hash(final), 'wall_seconds': elapsed_before + time.perf_counter() - started, 'parent': parent, 'promotion': False}, indent=1), encoding='utf-8')
        return final
    except BaseException as error:
        (run / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)[:4000], 'iteration': iteration, 'games': games_total}, indent=1), encoding='utf-8')
        raise
    finally:
        farm.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--iterations', type=int)
    arguments = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    path = train(RLConfig.load(arguments.config), arguments.run_id, arguments.resume, arguments.iterations)
    print(json.dumps({'final_checkpoint': str(path)}), flush=True)


if __name__ == '__main__':
    main()
