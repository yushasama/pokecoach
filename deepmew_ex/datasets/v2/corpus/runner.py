"""Execute the complete frozen inventory and preserve every row disposition."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import orjson

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.corpus.config import CorpusConfig
from deepmew_ex.datasets.v2.corpus.workers import canonical_part, initialize, raw_episode, verify_file
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash


def run_corpus(config: CorpusConfig, progress: Callable[[str, int, int], None] | None = None) -> Path:
    config.validate()
    inventory_hash = file_hash(config.inventory)
    inventory = orjson.loads(config.inventory.read_bytes())
    if inventory['schema'] != 'deepmew.ex.full_corpus_inventory.v1' or inventory['parser_hashes'] != binding_hashes():
        raise ValueError('unsupported inventory or parser drift')
    if shutil.disk_usage(config.paths.root.resolve().anchor).free < config.min_free_bytes:
        raise ValueError('insufficient free storage')
    store = RunStore.allocate(config.paths, 'deepmew_ex', 'v2_corpus', config.run_id)
    started = time.perf_counter()
    results: list[Payload] = []
    try:
        store.write('config.json', {'inventory': str(config.inventory.resolve()), 'inventory_sha256': inventory_hash, 'workers': config.workers, 'max_episode_bytes': config.max_episode_bytes, 'max_steps': config.max_steps, 'min_free_bytes': config.min_free_bytes, 'optimizer_updates': 0})
        store.write('code.json', code_identity(PROJECT_ROOT))
        store.write('environment.json', {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'orjson', 'zstandard')}, 'paid_compute': 0, 'worker_count': config.workers})
        # Keep one immutable inventory copy inside the run. Child processes share
        # only source identities and explicit configuration, never live parser state.
        store.write_bytes('inventory.json', config.inventory.read_bytes())
        old_threads = os.environ.get('OPENBLAS_NUM_THREADS')
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        try:
            with ProcessPoolExecutor(max_workers=config.workers, initializer=initialize, initargs=(str(store.path / 'inventory.json'), str(store.path))) as pool:
                checks = [pool.submit(verify_file, item) for item in [*inventory['sources'], *inventory['evidence']]]
                for index, future in enumerate(as_completed(checks), 1):
                    future.result()
                    if progress and (index % 100 == 0 or index == len(checks)):
                        progress('source_hashes', index, len(checks))
                parts = [pool.submit(canonical_part, part) for part in inventory['parts']]
                for index, future in enumerate(as_completed(parts), 1):
                    results.append(future.result())
                    if progress:
                        progress('canonical_parts', index, len(parts))
                raw_sources = [source for source in inventory['replays'] if source['status'] == 'available' and 'alias_of' not in source]
                episodes = [pool.submit(raw_episode, source, config.max_episode_bytes, config.max_steps) for source in raw_sources]
                for index, future in enumerate(as_completed(episodes), 1):
                    results.append(future.result())
                    if progress and (index % 25 == 0 or index == len(episodes)):
                        progress('raw_episodes', index, len(episodes))
        finally:
            if old_threads is None:
                os.environ.pop('OPENBLAS_NUM_THREADS', None)
            else:
                os.environ['OPENBLAS_NUM_THREADS'] = old_threads
        totals, families, exclusions = Counter(), {}, Counter()
        raw_games: dict[str, list[Payload]] = {}
        for result in results:
            if result['kind'] == 'canonical':
                counts = result['counts']
                totals.update(counts)
                family = families.setdefault(result['source_family'], Counter())
                family.update(counts)
                exclusions.update(result['quarantine_reasons'])
            elif result['status'] == 'admitted_actor_visible_episode':
                raw_games.setdefault(result['native_game_id'], []).append(result)
        for family, expected in inventory['family_counts'].items():
            if families.get(family, {}).get('rows', 0) != expected:
                raise ValueError('source_family_row_conservation_failure')
        if totals['rows'] != inventory['catalog_counts']['rows_kept'] or totals['historical_target_mask_excluded'] != inventory['legacy_excluded_target_rows']:
            raise ValueError('catalog_or_target_mask_conservation_failure')
        # Distinct raw bytes for the same game are retained as conflicting variants,
        # never automatically pooled or counted as independent training examples.
        conflicts = {game: [item['source']['sha256'] for item in variants] for game, variants in raw_games.items() if len(variants) > 1}
        raw_count = sum(len(value) for value in raw_games.values())
        raw_frames = sum(item['frames'] for values in raw_games.values() for item in values)
        raw_quarantine = sum(item['kind'] == 'raw' and item['status'] == 'quarantined' for item in results)
        raw_reasons: dict[str, dict[str, int]] = {}
        for item in results:
            if item['kind'] == 'raw' and item['status'] == 'quarantined':
                entry = raw_reasons.setdefault(item['reason'], {'episodes': 0, 'observed_candidate_requests': 0, 'episodes_with_unknown_request_count': 0})
                entry['episodes'] += 1
                entry['observed_candidate_requests'] += item['candidate_request_events'] or 0
                entry['episodes_with_unknown_request_count'] += item['candidate_request_events'] is None
        if file_hash(config.inventory) != inventory_hash or binding_hashes() != inventory['parser_hashes']:
            raise ValueError('inventory_or_parser_changed_during_port')
        results.sort(key=lambda item: item['path'])
        report = {'schema': 'deepmew.ex.full_corpus_port.v2', 'reference_backed': True, 'catalog_rows': totals['rows'], 'catalog_games': inventory['catalog_counts']['games_kept'], 'family_counts': {key: dict(value) for key, value in families.items()}, 'totals': dict(totals), 'quarantine_reasons': dict(exclusions), 'canonical_parts': len(inventory['parts']), 'raw_admitted_episodes': raw_count, 'raw_admitted_frames': raw_frames, 'raw_quarantined_episodes': raw_quarantine, 'raw_path_aliases': sum('alias_of' in row for row in inventory['replays']), 'raw_unavailable_paths': sum(row['status'] == 'unavailable' for row in inventory['replays']), 'raw_native_game_conflicts': conflicts, 'identity_root_sha256': digest(canonical(results)), 'results': results, 'elapsed_seconds': time.perf_counter() - started, 'optimizer_updates': 0, 'source_payloads_copied': False, 'limitations': ['canonical observations are immutable source references, not copied feature vectors', 'snapshot visibility/bounds require source resolution; catalog rows are not uniformly training-admitted', 'canonical Kaggle references retain known causal-alignment restriction; correct raw episodes are separate repair evidence', 'raw repair frames overlap the canonical population; counts are not additive independent examples', 'no Unified R2 immediate post-observation, TD or response-model admission', 'historical target masks and more restrictive existing splits retained', 'no model training or gameplay']}
        report['raw_quarantine_reasons'] = raw_reasons
        store.write('port.json', report)
        store.seal({'kind': 'v2_full_corpus_port', 'identity_root_sha256': report['identity_root_sha256'], 'optimizer_updates': 0})
    except Exception as error:
        store.write('failure.json', {'error_type': type(error).__name__, 'completed_jobs': len(results), 'source_payloads_suppressed': True, 'optimizer_updates': 0})
        store.seal({'kind': 'v2_full_corpus_port', 'optimizer_updates': 0}, 'failed')
        raise
    return store.path
