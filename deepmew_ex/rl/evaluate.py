"""Frozen league evaluation of BC / RL checkpoints with the existing native-game harness.

    python -m deepmew_ex.rl.evaluate --run-id <id> --deck <deck.csv> \
        --participant bc=<bc.pt>@<sha256> --participant rl=<rl iter-*.pt> \
        --opponents d1,ref_abomasnow,ref_lucario,ref_iono,ref_dragapult,z3rox_7th

Same pattern as ``evaluation.league.runner.run_league``: each participant
pilots ``--deck`` through the unchanged ``FullPrefixLivePolicy`` (argmax
decode of the full-prefix forward, not the RL farm's cached path) inside
``IndependentSeat`` (5 s per-decision budget); both seats notify the
participant's actor-visible history; every opponent is frozen and rebuilt per
game (the 7th-place agent is reset per game in its isolated subprocess);
``run_real_meta_game`` plays each game. Schedule per opponent: seeds x seats
(0, 1) x repetitions. Engine chance is unpaired. Writes
``output_root/<run_id>/{config.json, games.jsonl, results.json}``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.store import file_hash
from deepmew_ex.rl.farm import EXTERNAL, ExternalSeat, FarmSettings, Opponents


def wilson(wins: float, count: int) -> list[float]:
    from deepmew_ex.evaluation.league.runner import wilson as league_wilson
    return league_wilson(int(round(wins)), count)


def load_participant(spec: str, device: str) -> tuple[torch.nn.Module, bool, dict]:
    """``path@sha256`` = BC training checkpoint (hash-checked run evidence); ``path`` = RL checkpoint."""
    if '@' in spec:
        from deepmew_ex.evaluation.league.temporal import load_full_prefix_checkpoint
        path, sha = spec.rsplit('@', 1)
        model, evidence = load_full_prefix_checkpoint(Path(path), sha, device)
        return model, bool(evidence['public_event_mode']), {'path': str(Path(path).resolve()), 'sha256': sha, 'kind': 'bc', 'updates': evidence['updates']}
    from deepmew_ex.rl.train import load_rl_checkpoint
    model, _, saved = load_rl_checkpoint(Path(spec), device)
    model.requires_grad_(False).eval()
    return model, bool(saved['public_event_mode']), {'path': str(Path(spec).resolve()), 'sha256': file_hash(Path(spec)), 'kind': 'rl', 'iteration': saved['iteration'], 'games': saved['games'], 'parent_sha256': saved['parent']['sha256']}


def evaluate(run_id: str, deck_path: Path, participants: dict[str, str], opponents: list[str], seeds: list[int], repetitions: int, device: str, output_root: Path, external_submission: Path | None) -> dict:
    from deepmew_ex.evaluation.league.policies import IndependentSeat
    from deepmew_ex.evaluation.league.runner import read_deck
    from deepmew_ex.evaluation.league.temporal import ActorNotifyingSeat, FullPrefixLivePolicy
    from deepmew_ex.league import bind_legacy_source
    run = output_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    bind_legacy_source(PROJECT_ROOT)
    from deepmew.real_meta_arena.runner import run_real_meta_game
    deck = read_deck(deck_path)
    loaded = {label: load_participant(spec, device) for label, spec in participants.items()}
    settings = FarmSettings(str(PROJECT_ROOT), str(run / 'reference-runtime'), tuple(deck), str(external_submission) if external_submission else None)
    pool = Opponents(settings)
    config = {'run_id': run_id, 'deck': str(deck_path.resolve()), 'deck_sha256': file_hash(deck_path), 'participants': {label: evidence for label, (_, _, evidence) in loaded.items()}, 'opponents': opponents, 'seeds': seeds, 'repetitions': repetitions, 'seat_schedule': [0, 1], 'device': device, 'decode': 'argmax via FullPrefixLivePolicy (unchanged full-prefix forward)', 'max_decision_seconds': 5.0, 'max_selections': 4000, 'engine_rng': 'unpaired', 'promotion': False}
    (run / 'config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
    records = []
    started = time.perf_counter()
    try:
        with (run / 'games.jsonl').open('w', encoding='utf-8') as log:
            for label, (model, public, _) in loaded.items():
                for opponent in opponents:
                    opponent_deck = pool.deck(opponent)
                    for seed in seeds:
                        for seat in (0, 1):
                            for repetition in range(repetitions):
                                policy = FullPrefixLivePolicy(model, public_event_mode=public)
                                ours = IndependentSeat(policy, seed + seat * 1000003, 5.0)
                                raw = pool.seat(opponent)
                                other = raw if isinstance(raw, ExternalSeat) else IndependentSeat(raw, seed + (1 - seat) * 1000003, 5.0)
                                record = run_real_meta_game(ActorNotifyingSeat(ours, (policy,)), ActorNotifyingSeat(other, (policy,)), deck, opponent_deck, seat_our=seat, max_steps=4000, rng=random.Random(seed * 7 + repetition), game_index=len(records), rng_seed=seed)
                                row = {'participant': label, 'opponent': opponent, 'seed': seed, 'seat': seat, 'repetition': repetition, 'outcome': record.outcome, 'turns': record.turns, 'error': record.error, 'elapsed': record.elapsed_seconds, 'calls': ours.calls, 'latency_ms': [round(value * 1000, 3) for value in ours.latencies], 'counts': dict(policy.counts)}
                                log.write(json.dumps(row) + '\n')
                                log.flush()
                                records.append(row)
                                print(f"{label} vs {opponent} seed {seed} seat {seat} rep {repetition}: {record.outcome} ({record.turns} turns)", flush=True)
    finally:
        pool.close()
    results = summarize(records)
    results['elapsed_seconds'] = time.perf_counter() - started
    (run / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    return results


def summarize(records: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[row['participant'], row['opponent']].append(row)
    matchups, per_participant = [], defaultdict(list)
    for (participant, opponent), rows in sorted(groups.items()):
        outcomes = Counter(row['outcome'] for row in rows)
        score = (outcomes['win'] + 0.5 * outcomes['draw']) / len(rows)
        matchups.append({'participant': participant, 'opponent': opponent, 'games': len(rows), 'wins': outcomes['win'], 'losses': outcomes['loss'], 'draws': outcomes['draw'], 'win_rate': outcomes['win'] / len(rows), 'score_rate': score, 'win_rate_wilson95': wilson(outcomes['win'], len(rows))})
        per_participant[participant].append(score)
    aggregate = {}
    rng = np.random.default_rng(42)
    for participant, values in sorted(per_participant.items()):
        rows = [row for row in records if row['participant'] == participant]
        outcomes = Counter(row['outcome'] for row in rows)
        latency = np.array([value for row in rows for value in row['latency_ms']]) if rows else np.zeros(0)
        scores = np.array(values)
        boot = scores[rng.integers(0, len(scores), size=(10000, len(scores)))].mean(axis=1)
        wins = outcomes['win'] + 0.5 * outcomes['draw']
        aggregate[participant] = {'games': len(rows), 'wins': outcomes['win'], 'losses': outcomes['loss'], 'draws': outcomes['draw'], 'equal_opponent_score': float(scores.mean()), 'opponent_cluster_bootstrap95': np.quantile(boot, [.025, .975]).tolist(), 'pooled_win_rate_wilson95': wilson(wins, len(rows)),
                                  'safety': {'crashes': outcomes['crash'], 'illegal': outcomes['illegal'], 'timeouts': outcomes['timeout'], 'incomplete': outcomes['incomplete']},
                                  'decision_latency_ms': {'decisions': int(latency.size), 'mean': float(latency.mean()) if latency.size else None, 'median': float(np.median(latency)) if latency.size else None, 'p99': float(np.quantile(latency, .99)) if latency.size else None, 'max': float(latency.max()) if latency.size else None},
                                  'fallbacks': dict(sum((Counter({key: value for key, value in row['counts'].items() if key.startswith('fallback')}) for row in rows), Counter()))}
    return {'matchups': matchups, 'aggregate': aggregate, 'interpretation': 'exploratory frozen-league diagnostic; unpaired engine chance; no promotion'}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--participant', action='append', required=True, help='label=path@sha256 (BC) or label=path (RL)')
    parser.add_argument('--opponents', default='d1,ref_abomasnow,ref_lucario,ref_iono,ref_dragapult,z3rox_7th')
    parser.add_argument('--seeds', default='40009,50021,60013')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs/rl-evals')
    parser.add_argument('--external-submission', type=Path, default=PROJECT_ROOT / 'outputs/external_agents/z3rox_7th_1dde9eca/kaggle_submission')
    arguments = parser.parse_args(argv)
    participants = dict(item.split('=', 1) for item in arguments.participant)
    results = evaluate(arguments.run_id, arguments.deck, participants, arguments.opponents.split(','), [int(seed) for seed in arguments.seeds.split(',')], arguments.repetitions, arguments.device, arguments.output_root, arguments.external_submission)
    print(json.dumps(results['aggregate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
