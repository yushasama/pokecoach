"""Reuse hash-bound pure archived parsers without importing archived package initializers.

Only import names are relocated into a private namespace. Source bytes stay intact;
no engine, replay runner, source writer or archived experiment entrypoint executes.
"""

from __future__ import annotations

import ast
import copy
import sys
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Protocol, Sequence

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.store import file_hash

NAMESPACE = 'deepmew_ex.datasets.v2._bound_contract'
LEAVES = ('information_boundary', 'contract.schemas', 'contract.identities', 'contract.kaggle_sources', 'contract.legal_actions', 'contract.chosen_actions', 'contract.source_events')
STDLIB = {'__future__', 'copy', 'json', 'dataclasses', 'enum', 'pathlib', 'typing', 'hashlib', 'collections', 'itertools', 'math'}


class Relocate(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module and node.module.startswith('deepmew.replay.'):
            node.module = node.module.replace('deepmew.replay', NAMESPACE, 1)
        elif not node.module or node.module.split('.')[0] not in STDLIB:
            raise ValueError(f'unapproved parser dependency: {node.module}')
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        if any(alias.name.split('.')[0] not in STDLIB for alias in node.names):
            raise ValueError('unapproved parser import')
        return node


class SourceContract(Protocol):
    module_version: str
    source_schema: str
    alignment_rule_id: str


class ContractFailure(Protocol):
    value: str


class LegalContract(Protocol):
    legal_action_set_id: str
    complete_action_count: int | None
    failures: tuple[ContractFailure, ...]


class ChosenContract(Protocol):
    chosen_action_id: str | None
    complete: bool
    failures: tuple[ContractFailure, ...]


class VisibleGate(Protocol):
    def __call__(self, observation: object, *, expected_mode: str) -> str: ...


@dataclass(frozen=True)
class BoundContract:
    project_actor: Callable[[object], Mapping[str, object]]
    assert_visible: VisibleGate
    build_legal: Callable[[str, Mapping[str, object]], LegalContract]
    reconstruct_chosen: Callable[[LegalContract, Sequence[Sequence[int]]], ChosenContract]
    source_contract: Callable[[Mapping[str, object]], SourceContract | None]
    public_logs: Callable[[object], list[dict[str, object]]]
    log_names: dict[int, str]
    observation_fields: dict[str, frozenset[str]]
    hashes: dict[str, str]
    semantic_options: Callable[[Mapping[str, object]], tuple[dict[str, object], ...]]
    sha_json: Callable[[object], str]
    deployable_modes: frozenset[str]


def engine_api_path() -> Path:
    local = PROJECT_ROOT / 'source/cg/api.py'
    if local.is_file():
        return local
    # Locate the installed SDK without running its native package initializer.
    package = PathFinder.find_spec('cg')
    if package and package.submodule_search_locations:
        api = PathFinder.find_spec('cg.api', package.submodule_search_locations)
        if api and api.origin and Path(api.origin).is_file():
            return Path(api.origin)
    raise ModuleNotFoundError('CABT cg.api is missing. Set PYTHONPATH to the directory containing the competition cg package (see source/cg/README.md).')


def binding_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / 'source'
    paths = [root / 'deepmew' / 'replay' / f"{leaf.replace('.', '/')}.py" for leaf in LEAVES]
    paths += [root / 'deepmew/replay/prospective_trace.py', root / 'deepmew/replay/contract/causal_alignment.py']
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): file_hash(path) for path in paths}
    hashes['source/cg/api.py'] = file_hash(engine_api_path())
    return hashes


@lru_cache(maxsize=1)
def load_contract() -> BoundContract:
    hashes = binding_hashes()
    for name in (NAMESPACE, f'{NAMESPACE}.contract'):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    for leaf in LEAVES:
        path = PROJECT_ROOT / 'source/deepmew/replay' / f"{leaf.replace('.', '/')}.py"
        tree = ast.fix_missing_locations(Relocate().visit(ast.parse(path.read_text(encoding='utf-8'))))
        module = ModuleType(f'{NAMESPACE}.{leaf}')
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    # Reuse precisely the established conservative public projection. The native
    # cg initializer is bypassed: only its declarative LogType enum is evaluated.
    api = ast.parse(engine_api_path().read_text(encoding='utf-8-sig'))
    log_enum = next(node for node in api.body if isinstance(node, ast.ClassDef) and node.name == 'LogType')
    fields = {node.name: frozenset(field.target.id for field in node.body if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)) for node in api.body if isinstance(node, ast.ClassDef) and node.name in ('Card', 'Pokemon', 'PlayerState', 'State', 'SelectData')}
    # Raw 1.30.2/1.32 Pokemon objects also carry Card.playerIndex (public owner),
    # omitted from the bundled Pokemon declaration. Preserve the shared card fields.
    fields['Pokemon'] |= fields['Card']
    public = {'IntEnum': IntEnum, 'copy': copy}
    exec(compile(ast.Module([log_enum], type_ignores=[]), '<bound LogType>', 'exec'), public)
    names = {'PUBLIC_LOG_BASE_FIELDS', 'PUBLIC_LOG_FIELDS_BY_TYPE', '_as_dict', '_as_list', '_as_int', '_public_log_fields', 'public_trace_logs'}
    tree = ast.parse((PROJECT_ROOT / 'source/deepmew/replay/prospective_trace.py').read_text(encoding='utf-8'))
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else node.target.id if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else ''
        if name in names:
            selected.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(selected, type_ignores=[])), '<bound public projection>', 'exec'), public)
    modules = {leaf: sys.modules[f'{NAMESPACE}.{leaf}'] for leaf in LEAVES}
    return BoundContract(modules['contract.source_events'].project_actor_observation, modules['information_boundary'].assert_deployable_observation, modules['contract.legal_actions'].build_cabt_legal_action_set, modules['contract.chosen_actions'].reconstruct_chosen_action, modules['contract.kaggle_sources'].kaggle_source_contract_from_metadata, public['public_trace_logs'], {int(item): item.name.lower() for item in public['LogType']}, fields, hashes, modules['contract.legal_actions'].build_refgame_semantic_legal_options, modules['contract.identities'].sha256_json, modules['information_boundary'].DEPLOYABLE_MODES)
