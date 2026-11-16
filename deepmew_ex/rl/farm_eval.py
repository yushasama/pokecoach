"""High-volume greedy evaluation on the RL farm (cached inference path), for tight intervals.

    python -m deepmew_ex.rl.farm_eval --run-id <id> --deck <deck.csv> --games-per-opponent 200 \
        --participant bc=<bc.pt>@<sha256> --participant rl=<rl iter-*.pt>

Complements ``deepmew_ex.rl.evaluate`` (the league-harness pattern with the
unchanged full-prefix policy): same frozen opponents and argmax decoding, but
games run in parallel worker processes with batched cached inference
(``CachePool``, parity-tested against the full-prefix forward). Seats
alternate; engine chance is unpaired.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.store import file_hash
from deepmew_ex.rl.evaluate import load_participant, wilson
from deepmew_ex.rl.farm import EXTERNAL, Farm, FarmSettings, InferenceServer, Net
from deepmew_ex.rl.ppo import ValueHead


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--participant', action='append', required=True)
    parser.add_argument('--opponents', default='d1,ref_abomasnow,ref_lucario,ref_iono,ref_dragapult,z3rox_7th')
    parser.add_argument('--games-per-opponent', type=int, default=200)
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--external-workers', type=int, default=2)
    parser.add_argument('--torch-workers', type=int, default=None, help='workers able to run D1 (others stay torch-free)')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs/rl-evals')
    parser.add_argument('--external-submission', type=Path, default=PROJECT_ROOT / 'outputs/external_agents/z3rox_7th_1dde9eca/kaggle_submission')
    arguments = parser.parse_args(argv)
    run = arguments.output_root / arguments.run_id
    run.mkdir(parents=True, exist_ok=False)
    deck = tuple(int(card) for card in arguments.deck.read_text().split())
    participants = dict(item.split('=', 1) for item in arguments.participant)
    nets, evidence, public = {}, {}, set()
    for label, spec in participants.items():
        model, mode, info = load_participant(spec, 'cuda')
        nets[label] = Net(model, ValueHead(model.hidden).cuda().eval())
        evidence[label] = info
        public.add(mode)
    if len(public) != 1:
        raise ValueError('participants disagree on the input contract')
    public_event_mode = next(iter(public))
    opponents = arguments.opponents.split(',')
    rng = np.random.default_rng(arguments.seed)
    jobs = [{'opponent': opponent, 'seat': index % 2, 'seed': int(rng.integers(2**31)), 'tag': label, 'greedy': True, 'train': False, 'public_event_mode': public_event_mode} for label in participants for opponent in opponents for index in range(arguments.games_per_opponent)]
    (run / 'config.json').write_text(json.dumps({'participants': evidence, 'deck': str(arguments.deck.resolve()), 'deck_sha256': file_hash(arguments.deck), 'opponents': opponents, 'games_per_opponent': arguments.games_per_opponent, 'seed': arguments.seed, 'decode': 'argmax on cached incremental inference', 'promotion': False}, indent=1), encoding='utf-8')
    settings = FarmSettings(str(PROJECT_ROOT), str(run / 'reference-runtime'), deck, str(arguments.external_submission), seat_seconds=60.0)
    farm = Farm(settings, arguments.workers, arguments.external_workers if EXTERNAL in opponents else 0, arguments.torch_workers)
    started = time.perf_counter()
    try:
        results, stats = farm.run(jobs, InferenceServer(nets, seed=arguments.seed))
    finally:
        farm.close()
    with (run / 'games.jsonl').open('w', encoding='utf-8') as log:
        for result in results:
            log.write(json.dumps({'participant': result['job']['tag'], 'opponent': result['job']['opponent'], 'seat': result['job']['seat'], 'seed': result['job']['seed'], 'outcome': result['outcome'], 'turns': result['turns'], 'error': result['error'], 'counts': result['our_counts']}) + '\n')
    matchups, aggregate = [], {}
    for label in participants:
        scores, safety = [], Counter()
        for opponent in opponents:
            outcomes = Counter(result['outcome'] for result in results if result['job']['tag'] == label and result['job']['opponent'] == opponent)
            games = sum(outcomes.values())
            safety.update({key: outcomes[key] for key in ('crash', 'illegal', 'timeout', 'incomplete')})
            score = (outcomes['win'] + 0.5 * outcomes['draw']) / games
            scores.append(score)
            matchups.append({'participant': label, 'opponent': opponent, 'games': games, 'wins': outcomes['win'], 'losses': outcomes['loss'], 'draws': outcomes['draw'], 'score_rate': score, 'win_rate_wilson95': wilson(outcomes['win'], games)})
        aggregate[label] = {'equal_opponent_score': float(np.mean(scores)), 'safety': dict(safety)}
    summary = {'matchups': matchups, 'aggregate': aggregate, 'farm': stats, 'elapsed_seconds': time.perf_counter() - started, 'interpretation': 'exploratory; unpaired engine chance; no promotion'}
    (run / 'results.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
