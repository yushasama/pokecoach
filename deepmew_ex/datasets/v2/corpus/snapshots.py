"""Port uncataloged structured snapshots without inventing split/target admission."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import orjson
import zstandard

from deepmew_ex.datasets.v2.corpus.reader import resolve_row
from deepmew_ex.datasets.v2.corpus.records import normalize_reference
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.experiments.config import OutputPaths, PROJECT_ROOT
from deepmew_ex.registry.provenance import code_identity
from deepmew_ex.registry.store import RunStore, file_hash


def label_helpers() -> tuple[Callable[[object], float | None], Callable[[object], bool], str]:
    path = PROJECT_ROOT / 'source/deepmew/replay/canonical_actor_dataset.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'refgame_actor_label', 'refgame_safety_flagged'}
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]]
    scope: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(ast.Module(selected, type_ignores=[])), str(path), 'exec'), scope)
    return scope['refgame_actor_label'], scope['refgame_safety_flagged'], file_hash(path)


def snapshot_reference(row: Payload, family: str, source: Path, source_sha: str, ordinal: int, actor_order: int, label: float) -> Payload:
    native, seat, index = row['game_id'], row['seat'], row['decision_index']
    unit = f"{source_sha}:{row['game_index']}"
    game = f'{family}:{unit}:{native}'
    observation = row['raw_observation']
    content_hash = hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    reference = {'source_path': str(source.resolve()), 'source_field': 'raw_observation', 'source_line': ordinal, 'content_sha256': content_hash}
    before = {'source_observation': reference, 'feature_dim': None, 'feature_row': None, 'representation': 'semantic_source_without_materialized_encoding'}
    common = {'decision_id': f'{game}:d{index}:s{seat}', 'canonical_game_id': game, 'actor': seat, 'actor_decision_index': actor_order, 'global_decision_index': index, 'source_hash': source_sha, 'source_schema': row['schema_version'], 'state_before': before, 'information_mode': 'actor_visible'}
    transition = {**common, 'source_id': family, 'terminal_actor_relative_result': label, 'legal_action_set': row['legal_set'], 'chosen_action': row['chosen_action'], 'state_after': None, 'state_after_status': 'unavailable_missing_post_observation', 'next_same_actor_state': None, 'terminal': None, 'transition_horizon_logical_decisions': None}
    sequence = {**common, 'source_game_id': native, 'source_order': row['game_index'], 'control_owner': seat, 'public_event_history': {'status': 'not_admitted_source_logs', 'source_row_reference': reference}}
    frame = normalize_reference(transition, sequence, {str(source.resolve()): {'sha256': source_sha}}, 'uncataloged_filtered', 'quarantine_pending_dedup_and_split_admission', False, unit)
    frame['target_mask'] = {'source_target_eligible': None, 'reason': 'uncataloged_filtered_population_has_no_admitted_target_mask', 'model_training_authorized': False}
    # Check the actual semantic observation, source identity and action crosswalk
    # before persisting this new source. No encoder/model or native engine executes.
    resolve_row(frame, row)
    return frame


def port_snapshot_sources(sources: Iterable[Path], paths: OutputPaths, run_id: str) -> Path:
    paths.validate()
    store = RunStore.allocate(paths, 'deepmew_ex', 'v2_snapshot_supplement', run_id)
    label, unsafe, helper_hash = label_helpers()
    results = []
    try:
        store.write('code.json', code_identity(PROJECT_ROOT))
        store.write('contract.json', {'label_owner_sha256': helper_hash, 'usage': 'inspect_only; unknown split/mask admission, outcome-selected source population'})
        for source in sources:
            source_sha = file_hash(source)
            family = 'uncataloged_' + source.parent.parent.name
            counts, reasons, orders = Counter(), Counter(), Counter()
            with (store.path / f'{source_sha}.frames.jsonl.zst').open('xb') as output:
                with zstandard.ZstdCompressor(level=1).stream_writer(output, closefd=False) as writer:
                    with gzip.open(source, 'rb') as handle:
                        for ordinal, line in enumerate(handle, 1):
                            row = orjson.loads(line)
                            try:
                                target = label(row.get('outcome'))
                                if target is None or unsafe(row.get('outcome')):
                                    raise ValueError('invalid_or_unsafe_outcome')
                                key = (row['game_id'], row['game_index'], row['seat'])
                                frame = snapshot_reference(row, family, source, source_sha, ordinal, orders[key], target)
                                orders[key] += 1
                                counts['resolved_semantic_snapshots'] += 1
                            except (KeyError, TypeError, ValueError) as error:
                                reason = str(error) if type(error) is ValueError and str(error).replace('_', '').islower() and len(str(error)) < 80 else 'snapshot_contract_rejection'
                                reasons[reason] += 1
                                frame = {'schema': 'deepmew.ex.quarantined_reference.v2', 'source_sha256': source_sha, 'source_line': ordinal, 'reason': reason}
                            writer.write(orjson.dumps(frame, option=orjson.OPT_SORT_KEYS) + b'\n')
                            counts['rows'] += 1
            if file_hash(source) != source_sha:
                raise ValueError('source_changed_during_port')
            results.append({'source_path': str(source.resolve()), 'source_sha256': source_sha, 'source_bytes': source.stat().st_size, 'source_family': family, 'counts': dict(counts), 'quarantine_reasons': dict(reasons), 'selection_warning': 'outcome-filtered source; not independent unbiased samples; overlap with catalog is not admitted/deduplicated'})
        store.write('supplement.json', {'results': results, 'optimizer_updates': 0, 'training_authorized': False})
        store.seal({'kind': 'v2_snapshot_supplement', 'optimizer_updates': 0})
    except Exception as error:
        store.write('failure.json', {'error_type': type(error).__name__, 'source_payloads_suppressed': True})
        store.seal({'kind': 'v2_snapshot_supplement'}, 'failed')
        raise
    return store.path
