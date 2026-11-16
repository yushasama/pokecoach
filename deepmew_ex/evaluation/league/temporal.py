"""Hash-bound full-prefix temporal inference for actor-visible native games."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Protocol

import torch

from deepmew_ex.training.bitmap_ledger import LEDGER_FILES
from deepmew_ex.datasets.v2.contract import binding_hashes, load_contract
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.datasets.v2.learning.history import FullPrefixHistory, LiveDecision, INPUT_CONTRACTS, LEGACY_INPUT_CONTRACT, QUERY_INPUT_CONTRACT
from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.evaluation.league.seats import ActorNotifyingSeat  # re-exported
from deepmew_ex.registry.store import canonical, digest, file_hash, verify_run


EXPECTED_ARCHITECTURE = 'full_prefix_byte_semantic_v1'


class SelectPolicy(Protocol):
    label: str

    def select(self, observation: Payload, rng: random.Random) -> list[int]: ...


class FullPrefixLivePolicy:
    def __init__(self, model: TemporalPolicy, label: str = 'full_prefix_temporal', public_event_mode: bool = False) -> None:
        self.model = model
        self.label = label
        self.history = FullPrefixHistory(public_event_mode)
        self.counts: Counter[str] = Counter()

    def observe_actor(self, actor: int) -> None:
        self.history.observe_actor(actor)

    def observe(self, observation: Payload) -> None:
        self.history.observe(observation)

    def select(self, observation: Payload, rng: random.Random) -> list[int]:
        from deepmew_ex.evaluation.league.policies import greedy_fallback
        decision = self.history.prepare(observation)
        reason = decision.fallback_reason
        selected: list[list[int]]
        if reason is None:
            with torch.no_grad():
                decoded = self.model.decode(decision.sequence, decision.event_index, decision.stages)
            selected = [[int(decision.contract['stages'][stage]['options'][index]['position']) for index in indices] for stage, indices in enumerate(decoded)]
            self.counts['model_choices'] += 1
            self.counts['model_stages'] += len(selected)
        else:
            selected = [greedy_fallback(observation)]
            self.counts['fallback:' + reason] += 1
        self.history.commit(decision, selected)
        if len(selected) != 1:
            raise ValueError('native CABT selection requires one flattened stage result')
        return selected[0]


def load_full_prefix_checkpoint(checkpoint: Path, expected_sha256: str, device: str) -> tuple[TemporalPolicy, dict[str, object]]:
    checkpoint = checkpoint.resolve()
    if device not in ('cpu', 'cuda') or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested evaluation device is unavailable')
    actual = file_hash(checkpoint)
    if actual != expected_sha256:
        raise ValueError('temporal checkpoint hash mismatch')
    sidecar = json.loads(checkpoint.with_suffix(checkpoint.suffix + '.json').read_text(encoding='utf-8'))
    if sidecar.get('sha256') != actual:
        raise ValueError('temporal checkpoint sidecar mismatch')
    parent = checkpoint.parent.parent
    completed = (parent / 'manifest.json').is_file()
    if completed:
        manifest = verify_run(parent)
        if manifest.get('kind') != 'v2_full_temporal_training' or manifest.get('status') != 'completed':
            raise ValueError('evaluation requires a completed temporal training run')
    required = ('allocation.json', 'authorization.json', 'code.json', 'config.json', 'parser-identity.json', 'progress.jsonl', 'input-contract.json' if completed else 'failure.json')
    ledger = ('consumption.sqlite3',) if (parent / 'consumption.sqlite3').is_file() else LEDGER_FILES
    if any(not (parent / name).is_file() for name in (*required, *ledger)):
        raise ValueError('recovered temporal run evidence is incomplete')
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    expected_keys = {'architecture', 'model_config', 'model', 'optimizer', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'updates', 'presentations', 'phase', 'epoch', 'cursor', 'elapsed_seconds', 'config_sha256', 'code_sha256', 'corpus_sha256', 'parser_hashes'}
    input_contract = LEGACY_INPUT_CONTRACT
    if completed:
        expected_keys |= {'training_state_version', 'input_contract', 'scheduler', 'best_validation', 'ledger_digest', 'deadline_epoch'}
        input_contract = saved.get('input_contract')
        if saved.get('training_state_version') not in (3, 4, 5) or input_contract not in INPUT_CONTRACTS or json.loads((parent / 'input-contract.json').read_text()) != {'input_contract': input_contract}:
            raise ValueError('unsupported or inconsistent temporal input contract')
        if saved['training_state_version'] in (4, 5):
            expected_keys.add('numerical_policy')
            environment = parent / 'environment.json'
            if not environment.is_file() or not isinstance(saved.get('numerical_policy'), dict) or saved['numerical_policy'] != json.loads(environment.read_text()).get('numerical_policy'):
                raise ValueError('temporal checkpoint numerical policy lacks bound run evidence')
    if set(saved) != expected_keys or saved['architecture'] != EXPECTED_ARCHITECTURE:
        raise ValueError('unsupported temporal checkpoint contract')
    config = json.loads((parent / 'config.json').read_text(encoding='utf-8'))
    code = json.loads((parent / 'code.json').read_text(encoding='utf-8'))
    if saved['config_sha256'] != digest(canonical(config)) or saved['code_sha256'] != code['source_manifest_sha256']:
        raise ValueError('temporal checkpoint code/config identity mismatch')
    corpus = (PROJECT_ROOT / (config.get('prepared_corpus') or config['corpus'])).resolve()
    if saved['corpus_sha256'] != file_hash(corpus / 'manifest.json') or saved['parser_hashes'] != binding_hashes():
        raise ValueError('temporal checkpoint corpus/parser identity mismatch')
    if config.get('prepared_corpus'):
        corpus_manifest = json.loads((corpus / 'manifest.json').read_text())
        if corpus_manifest.get('input_contract') != input_contract or corpus_manifest.get('training_admitted') is not True or corpus_manifest.get('kind') != 'v2_prepared_temporal_corpus' or corpus_manifest.get('status') != 'completed':
            raise ValueError('checkpoint input contract differs from admitted corpus')
    elif input_contract != LEGACY_INPUT_CONTRACT:
        raise ValueError('query-aware checkpoint requires prepared temporal corpus')
    if saved['phase'] != 'raw' or saved['updates'] != sidecar['updates'] or saved['presentations'] != sidecar['presentations'] or saved['cursor'] != sidecar['cursor']:
        raise ValueError('temporal checkpoint cursor sidecar mismatch')
    model = TemporalPolicy(**saved['model_config']).to(device)
    model.load_state_dict(saved['model'])
    model.requires_grad_(False).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    recorded_parameters = json.loads((parent / 'parameters.json').read_text(encoding='utf-8'))['total']
    if parameters != recorded_parameters:
        raise ValueError('temporal checkpoint parameter identity mismatch')
    evidence = {'path': str(checkpoint), 'sha256': actual, 'parent_run': str(parent), 'architecture': saved['architecture'], 'model_config': saved['model_config'], 'parameters': parameters, 'updates': saved['updates'], 'presentations': saved['presentations'], 'phase': saved['phase'], 'epoch': saved['epoch'], 'cursor': saved['cursor'], 'training_code_sha256': saved['code_sha256'], 'training_config_sha256': saved['config_sha256'], 'corpus_sha256': saved['corpus_sha256'], 'parser_hashes': saved['parser_hashes'], 'optimizer_state_present': bool(saved['optimizer']), 'rng_state_present': all(key in saved for key in ('torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')), 'device': device}
    evidence.update({'input_contract': input_contract, 'public_event_mode': input_contract == QUERY_INPUT_CONTRACT, 'completed_training_run': completed, 'training_numerical_policy': saved.get('numerical_policy')})
    return model, evidence
