"""Independently reload a completed temporal checkpoint without further training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import torch

from deepmew_ex.training.bitmap_ledger import verify_run_ledger
from deepmew_ex.datasets.v2.learning.full_corpus import FullCorpus
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, actor_sequences
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, file_hash


def score(model: TemporalPolicy, sequence: ActorSequence) -> tuple[float, float, int, list[str]]:
    losses, ids = model.losses(sequence)
    if not torch.isfinite(losses).all():
        raise ValueError('nonfinite independent evaluation loss')
    uniform = 0.0
    for prediction in sequence.predictions:
        if not prediction.loss_eligible:
            continue
        for stage in prediction.stages:
            n = len(stage.options)
            if stage.ordered:
                count = sum(n ** k if stage.repeated else math.perm(n, k) for k in range(stage.minimum, stage.maximum + 1))
            else:
                count = sum(math.comb(n, k) for k in range(stage.minimum, stage.maximum + 1))
            uniform += math.log(count)
    return float(losses.sum()), uniform, len(ids), ids


def evaluate(parent: Path, output: Path, run_id: str) -> Path:
    if not (parent / 'manifest.json').is_file():
        raise ValueError('parent training process must finish first')
    checkpoints = list((parent / 'checkpoints').glob('phase-end-*-raw-*.pt'))
    if len(checkpoints) != 1 or not (parent / 'evaluations/epoch-0-raw.json').is_file():
        raise ValueError('completed temporal phase and validation are required')
    checkpoint = checkpoints[0]
    sha = file_hash(checkpoint)
    if sha != json.loads(checkpoint.with_suffix('.pt.json').read_text())['sha256']:
        raise ValueError('temporal checkpoint hash mismatch')
    if (parent / 'ledger.bin').is_file():
        # Bitmap ledger runs record only raw-phase updates; replay proves exact coverage.
        proof = verify_run_ledger(parent)
        if proof['missing'] or proof['unexpected']:
            raise ValueError('temporal consumption ledger mismatch')
        train_games, coverage, context = proof['train_games'], proof['coverage'], proof['context']
    else:
        db = sqlite3.connect(f'file:{parent.as_posix()}/consumption.sqlite3?mode=ro', uri=True)
        if db.execute("SELECT COUNT(*) FROM consumed WHERE phase!='raw'").fetchone()[0]:
            raise ValueError('snapshot training occurred')
        missing = db.execute('SELECT COUNT(*) FROM (SELECT epoch,phase,id FROM expected EXCEPT SELECT epoch,phase,id FROM consumed)').fetchone()[0]
        unexpected = db.execute('SELECT COUNT(*) FROM (SELECT epoch,phase,id FROM consumed EXCEPT SELECT epoch,phase,id FROM expected)').fetchone()[0]
        if missing or unexpected:
            raise ValueError('temporal consumption ledger mismatch')
        train_games = {row[0] for row in db.execute('SELECT DISTINCT game FROM consumed')}
        coverage = db.execute('SELECT family,COUNT(*),COUNT(DISTINCT id),COUNT(DISTINCT game) FROM consumed GROUP BY family').fetchall()
        context = db.execute('SELECT COUNT(*),SUM(requests),SUM(events),SUM(post_states),SUM(unavailable_post_states) FROM context').fetchone()
        db.close()
    config = json.loads((parent / 'config.json').read_text())
    torch.set_num_threads(config['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    saved = torch.load(checkpoint, map_location=config['device'], weights_only=True)
    model = TemporalPolicy(**saved['model_config']).to(config['device']).eval()
    model.load_state_dict(saved['model'])
    store = RunStore.allocate(OutputPaths(output), 'deepmew_ex', 'v2_temporal_evaluation_v1', run_id)
    store.write('code.json', code_identity(Path(__file__).resolve().parents[3]))
    store.write('temporal-completion.json', {'parent_run': str(parent), 'parent_manifest_sha256': file_hash(parent / 'manifest.json'), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha, 'updates': saved['updates'], 'presentations': saved['presentations'], 'phase': saved['phase'], 'cursor': saved['cursor'], 'optimizer_preserved': 'optimizer' in saved, 'rng_preserved': all(key in saved for key in ('torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')), 'snapshot_updates': 0, 'coverage_by_family': coverage, 'context_totals': context, 'missing_expected': missing, 'unexpected_consumed': unexpected, 'parent_ledger_sha256': file_hash(parent / 'consumption.sqlite3'), 'training_code_sha256': saved['code_sha256'], 'training_config_sha256': saved['config_sha256'], 'corpus_sha256': saved['corpus_sha256'], 'parser_hashes': saved['parser_hashes'], 'scope': 'raw temporal phase completed; original full-run failure is retained as the user-directed snapshot-boundary stop', 'all_historical_corpus_trained': False})
    corpus = FullCorpus(Path(config['corpus']).resolve(), store.path, lambda event, fields: None)
    if file_hash(corpus.root / 'manifest.json') != saved['corpus_sha256'] or corpus.parser_hashes != saved['parser_hashes']:
        raise ValueError('evaluation corpus/parser drift')
    inventory = json.loads((corpus.root / 'inventory.json').read_text())
    total, uniform, count, games = 0.0, 0.0, 0, 0
    identities = hashlib.sha256()
    # Independent raw-validation cohort, not the prior snapshot-reserved filter.
    # Original training excludes raw validation signatures before optimization.
    with torch.no_grad():
        for entry in corpus.raw:
            native = entry['native_game_id']
            if entry['effective_split'] != 'validation' or inventory['raw_splits'].get(native) != 'validation' or native in train_games:
                continue
            episode = corpus.episode(entry)
            for sequence in actor_sequences(episode, 'validation'):
                loss, baseline, size, ids = score(model, sequence)
                total += loss
                uniform += baseline
                count += size
                for identity in ids:
                    identities.update((identity + '\n').encode())
            games += 1
            if games % 25 == 0:
                print(json.dumps({'event': 'independent_validation', 'games': games, 'decisions': count}), flush=True)
    if not count:
        raise ValueError('empty independent validation cohort')
    store.write('evaluation.json', {'checkpoint_sha256': sha, 'reloaded_in_separate_process': True, 'optimizer_updates': 0, 'games': games, 'decisions': count, 'ordered_decision_ids_sha256': identities.hexdigest(), 'mean_nll': total / count, 'uniform_complete_action_mean_nll': uniform / count, 'cohort': 'all nonforced decisions in raw port validation games also assigned validation by frozen raw split inventory; excludes consumed training games; does not apply legacy snapshot-reserved signature exclusions', 'comparison_with_parent_validation': 'different declared eligibility; do not treat counts or metrics as exact parity', 'limits': ['offline supervised likelihood only', 'uniform legal-action baseline, not a trained comparator', 'no useful-memory ablation or gameplay strength established', 'complete intermediate simulator effects beyond recorded actor evidence not established']})
    corpus.close()
    store.seal({'kind': 'independent_temporal_validation', 'checkpoint_sha256': sha, 'optimizer_updates': 0})
    return store.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(evaluate(args.parent.resolve(), args.output.resolve(), args.run_id), flush=True)


if __name__ == '__main__':
    main()
