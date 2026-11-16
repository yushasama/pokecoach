"""Thin command adapter; only `experiment run` imports the trainer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from deepmew_ex.experiments.config import PROJECT_ROOT, load_config
from deepmew_ex.registry.indexes import lookup_run, publish_indexes, rebuild
from deepmew_ex.registry.legacy import import_legacy
from deepmew_ex.registry.store import verify_run


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='DeepMew EX research datasets, BC training and local league evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('experiment').add_subparsers(dest='action', required=True)
    for name in ('validate', 'dry-run', 'run'):
        action = experiment.add_parser(name)
        action.add_argument('config', type=Path)
        if name == 'run':
            action.add_argument('--run-id')
    report = experiment.add_parser('report')
    report.add_argument('--run-id', required=True)
    report.add_argument('--outputs', type=Path, default=PROJECT_ROOT / 'outputs')
    registry = commands.add_parser('registry').add_subparsers(dest='action', required=True)
    registry.add_parser('list')
    registry.add_parser('rebuild')
    for name in ('show', 'lineage'):
        action = registry.add_parser(name)
        action.add_argument('identity')
        if name == 'lineage':
            action.add_argument('--format', choices=['json'], default='json')
    registry.add_parser('import-legacy').add_argument('--dry-run', action='store_true')
    comparison = commands.add_parser('ablation').add_subparsers(dest='action', required=True).add_parser('validate')
    comparison.add_argument('config', type=Path)
    datasets = commands.add_parser('dataset').add_subparsers(dest='action', required=True)
    audit = datasets.add_parser('audit-v1', help='Read-only corpus integrity/eligibility checks; no model imports')
    audit.add_argument('--root', type=Path, required=True)
    audit.add_argument('--audit-id', required=True)
    audit.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    audit.add_argument('--manifest', type=Path, default=PROJECT_ROOT / 'DATASET_MANIFEST.json')
    audit.add_argument('--validation-receipt', type=Path, default=PROJECT_ROOT / 'VALIDATION_RECEIPT.json')
    golden = datasets.add_parser('convert-v2-golden', help='Bounded semantic golden fixtures; no training or engine imports')
    golden.add_argument('--inventory', type=Path, required=True)
    golden.add_argument('--run-id', required=True)
    golden.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    inspect_v2 = datasets.add_parser('verify-v2-golden')
    inspect_v2.add_argument('--path', type=Path, required=True)
    inspect_v2.add_argument('--objective', default='temporal_bc')
    corpus = datasets.add_parser('port-v2-corpus', help='Port the complete frozen catalog as source-bound V2 records; no training')
    corpus.add_argument('--inventory', type=Path, required=True)
    corpus.add_argument('--run-id', required=True)
    corpus.add_argument('--workers', type=int, default=4)
    corpus.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    freeze = datasets.add_parser('freeze-v2-corpus', help='Bind catalog, split/mask metadata and available replay files without training')
    freeze.add_argument('--catalog', type=Path, required=True)
    freeze.add_argument('--raw-registry', type=Path, required=True)
    freeze.add_argument('--replay-root', type=Path, action='append', required=True)
    freeze.add_argument('--v1-assignments', type=Path, required=True)
    freeze.add_argument('--golden-inventory', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    snapshots = datasets.add_parser('port-v2-snapshots', help='Port uncataloged structured snapshots with unknown training admission')
    snapshots.add_argument('--source', type=Path, action='append', required=True)
    snapshots.add_argument('--run-id', required=True)
    snapshots.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    admit_v2 = datasets.add_parser('admit-v2', help='Audit causal raw replay fitting support and freeze membership')
    admit_v2.add_argument('--corpus', type=Path, required=True)
    admit_v2.add_argument('--run-id', required=True)
    admit_v2.add_argument('--train-decisions', type=int, default=16384)
    admit_v2.add_argument('--validation-decisions', type=int, default=2048)
    admit_v2.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    cache_v2 = datasets.add_parser('cache-v2', help='Encode admitted semantic inputs with a train-owned vocabulary')
    cache_v2.add_argument('--admission', type=Path, required=True)
    cache_v2.add_argument('--run-id', required=True)
    cache_v2.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    datasets.add_parser('report').add_argument('--audit-path', type=Path, required=True)
    leagues = commands.add_parser('league').add_subparsers(dest='action', required=True)
    for name in ('canary', 'evaluate', 'temporal-d1'):
        action = leagues.add_parser(name)
        action.add_argument('--run-id', required=True)
        action.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
        if name == 'evaluate':
            action.add_argument('--run', type=Path, action='append', required=True)
        elif name == 'temporal-d1':
            action.add_argument('--checkpoint', type=Path, required=True)
            action.add_argument('--checkpoint-sha256', required=True)
            action.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
            action.add_argument('--canary', action='store_true')
        else:
            action.add_argument('--opponent', action='append', help='Explicit frozen canary opponent; defaults to d1')
    args = parser.parse_args(arguments)
    result: object
    if args.command == 'league':
        import sys
        from deepmew_ex.evaluation.league.config import LeagueConfig
        from deepmew_ex.evaluation.league.runner import run_league
        from deepmew_ex.experiments.config import OutputPaths
        def played(done: int, total: int) -> None:
            print(f'league: {done}/{total}', file=sys.stderr, flush=True)
        canary = args.action == 'canary' or args.action == 'temporal-d1' and args.canary
        temporal = args.action == 'temporal-d1'
        config = LeagueConfig(tuple(path.resolve() for path in getattr(args, 'run', [])), OutputPaths(args.output_root.resolve()), args.run_id, canary=canary, opponent_ids=('d1',) if temporal else tuple(args.opponent or ['d1']) if canary else (), seeds=(40009,) if canary else (40009, 50021, 60013), repetitions=1 if canary else 2, max_wall_seconds=300 if canary else 2400, temporal_checkpoint=args.checkpoint.resolve() if temporal else None, temporal_checkpoint_sha256=args.checkpoint_sha256 if temporal else None, temporal_device=args.device if temporal else 'cpu')
        path = run_league(config, played)
        print(json.dumps({'path': str(path.resolve()), 'manifest': verify_run(path)}, indent=2))
        return
    if args.command == 'dataset':
        if args.action in ('admit-v2', 'cache-v2'):
            import sys
            from deepmew_ex.experiments.config import OutputPaths
            def learned_input(done: int, total: int) -> None:
                print(f'{args.action}: {done}/{total}', file=sys.stderr, flush=True)
            if args.action == 'admit-v2':
                from deepmew_ex.datasets.v2.learning.admission import AdmissionConfig, admit_replays
                path = admit_replays(AdmissionConfig(args.corpus.resolve(), OutputPaths(args.output_root.resolve()), args.run_id, args.train_decisions, args.validation_decisions), learned_input)
            else:
                from deepmew_ex.datasets.v2.learning.cache import build_cache
                path = build_cache(args.admission.resolve(), OutputPaths(args.output_root.resolve()), args.run_id, learned_input)
            print(json.dumps({'path': str(path.resolve()), 'manifest': verify_run(path)}, indent=2))
            return
        if args.action == 'port-v2-snapshots':
            from deepmew_ex.datasets.v2.corpus.snapshots import port_snapshot_sources
            from deepmew_ex.experiments.config import OutputPaths
            path = port_snapshot_sources(args.source, OutputPaths(args.output_root.resolve()), args.run_id)
            print(json.dumps({'path': str(path.resolve()), 'manifest': verify_run(path)}, indent=2))
            return
        if args.action == 'freeze-v2-corpus':
            from deepmew_ex.datasets.v2.corpus.inventory import freeze_inventory
            from deepmew_ex.registry.store import file_hash
            inventory = freeze_inventory(args.catalog, args.raw_registry, args.replay_root, args.v1_assignments, args.golden_inventory, args.output)
            print(json.dumps({'inventory': str(args.output.resolve()), 'sha256': file_hash(args.output), 'canonical_parts': len(inventory['parts']), 'replay_paths': len(inventory['replays'])}, indent=2))
            return
        if args.action == 'port-v2-corpus':
            import sys
            from deepmew_ex.datasets.v2.corpus.config import CorpusConfig
            from deepmew_ex.datasets.v2.corpus.runner import run_corpus
            from deepmew_ex.experiments.config import OutputPaths
            def ported(phase: str, done: int, total: int) -> None:
                print(f'{phase}: {done}/{total}', file=sys.stderr, flush=True)
            path = run_corpus(CorpusConfig(args.inventory.resolve(), args.run_id, OutputPaths(args.output_root.resolve()), workers=args.workers), ported)
            result = {'path': str(path.resolve()), 'manifest': verify_run(path)}
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.action in ('convert-v2-golden', 'verify-v2-golden'):
            from deepmew_ex.datasets.v2.reader import iter_episodes
            if args.action == 'convert-v2-golden':
                import sys
                from deepmew_ex.datasets.v2.config import GoldenConfig
                from deepmew_ex.datasets.v2.conversion import run_golden
                from deepmew_ex.experiments.config import OutputPaths
                def converted(done: int, total: int) -> None:
                    print(f'Converted {done}/{total} golden episodes twice', file=sys.stderr, flush=True)
                path = run_golden(GoldenConfig(args.inventory.resolve(), args.run_id, OutputPaths(args.output_root.resolve())), converted)
            else:
                path = args.path
            count = sum(len(episode.frames) for episode in iter_episodes(path, getattr(args, 'objective', 'temporal_bc')))
            result = {'path': str(path.resolve()), 'verified_frames': count, 'optimizer_updates': 0, 'manifest': verify_run(path)}
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.action == 'audit-v1':
            import sys
            from deepmew_ex.admission.config import AuditConfig
            from deepmew_ex.admission.v1 import audit_v1
            from deepmew_ex.experiments.config import OutputPaths
            def progress(done: int, total: int) -> None:
                if done % 8 == 0 or done == total:
                    print(f'Verified {done}/{total} shards', file=sys.stderr, flush=True)
            path = audit_v1(AuditConfig(args.root, args.audit_id, OutputPaths(args.output_root.resolve()), args.manifest, args.validation_receipt), progress)
        else:
            path = args.audit_path
        manifest = verify_run(path)
        reports = {'dataset_admission': 'audit.json', 'v2_golden_conversion': 'conversion.json', 'v2_full_corpus_port': 'port.json', 'v2_snapshot_supplement': 'supplement.json', 'v2_bc_admission': 'admission.json', 'v2_learning_cache': 'support.json'}
        report_file = reports[manifest['kind']] if manifest['status'] == 'completed' else 'failure.json'
        result = {'audit': str(path.resolve()), 'manifest': manifest, 'result': json.loads((path / report_file).read_text(encoding='utf-8'))}
    elif args.command == 'experiment' and args.action != 'report':
        config = load_config(args.config.resolve())
        if args.action == 'run':
            if config.stage == 'S2':
                import sys
                import uuid
                from deepmew_ex.experiments.v2.runner import run_v2
                def trained(done: int, total: int, loss: float) -> None:
                    print(f'V2 {config.history}: update {done}/{total}, validation CE={loss:.6f}', file=sys.stderr, flush=True)
                path = run_v2(config, args.run_id or 'v2-' + uuid.uuid4().hex, trained)
            else:
                from deepmew_ex.experiments.runner import run_s0
                path = run_s0(config, args.run_id)
            result = {'run': str(path), 'manifest': verify_run(path)}
        else:
            result = {'status': 'passed', 'scope': 'validation only; no data reads or run allocation', 'resolved_configuration': config.resolved()}
    elif args.command == 'experiment':
        from deepmew_ex.experiments.config import component
        component(args.run_id)
        matches = list(args.outputs.glob(f'*/*/{args.run_id}'))
        if len(matches) != 1:
            raise ValueError('run ID is missing or ambiguous')
        result = verify_run(matches[0])
    elif args.command == 'ablation':
        from deepmew_ex.evaluation.comparisons import Comparison
        raw = json.loads(args.config.read_text(encoding='utf-8'))
        result = Comparison(**raw).validate()
    elif args.action == 'rebuild':
        result = publish_indexes(PROJECT_ROOT / 'outputs', PROJECT_ROOT / 'registry')
    else:
        legacy = import_legacy(PROJECT_ROOT)
        if args.action == 'show':
            try:
                result = legacy.lookup(args.identity)
            except ValueError:
                result = verify_run(lookup_run(PROJECT_ROOT / 'outputs', args.identity))
        elif args.action == 'lineage':
            try:
                records = legacy.lookup(args.identity)['records']
                ids = {record['record_id'] for record in records}
                result = {'records': records, 'edges': [asdict(edge) for edge in legacy.edges if edge.source in ids or edge.target in ids], 'unresolved': legacy.unresolved}
            except ValueError:
                path = lookup_run(PROJECT_ROOT / 'outputs', args.identity)
                verify_run(path)
                result = json.loads((path / 'lineage.json').read_text(encoding='utf-8'))
        else:
            result = {'mode': 'read_only_metadata', 'records': len(legacy.records), 'sources': {key: value['sha256'] for key, value in legacy.sources.items()}, 'unresolved': legacy.unresolved, 'new_runs': rebuild(PROJECT_ROOT / 'outputs')['runs'], 'training_reproduced': False, 'checkpoint_payloads_opened': 0}
            if args.action == 'list':
                result['entries'] = [{'record_id': record.record_id, 'aliases': record.aliases, 'source': record.evidence.source, 'asset_presence': record.asset_presence} for record in legacy.records]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
