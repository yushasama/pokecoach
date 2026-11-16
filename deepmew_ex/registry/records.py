"""Identity contracts and lineage validation; no model/runtime imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

EvidenceLevel = Literal['verified', 'inferred', 'unknown', 'conflicted']
ANCESTRY = frozenset({'architecture_derived_from', 'initialized_from', 'continued_training_from', 'distilled_from', 'fine_tuned_from'})
RELATIONS = ANCESTRY | {'trained_against', 'evaluated_against', 'uses_dataset', 'evaluated_in', 'member_of_league'}


def validate_hash(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'invalid sha256: {value!r}')
    return value


@dataclass(frozen=True)
class Evidence:
    source: str
    sha256: str
    pointer: str
    level: EvidenceLevel
    scope: str

    def __post_init__(self) -> None:
        validate_hash(self.sha256)


@dataclass(frozen=True)
class Family:
    family_id: str
    description: str


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family_id: str
    model_spec_id: str
    hypothesis_version: str


@dataclass(frozen=True)
class ModelSpec:
    model_spec_id: str
    architecture: str
    hidden_dim: int
    option_dim: int = 304
    state_dim: int = 144
    recurrence: str = 'none'
    reset: str = 'not_applicable_snapshot'
    input_schema: str = 'fixed_304_144_v1'
    action_schema: str = 'complete_legal_options_single_selection'
    vocabulary: str = 'not_applicable_fixed_features'


@dataclass(frozen=True)
class SemanticModelSpec(ModelSpec):
    semantic_embedding_dim: int = 24
    history_window: int = 8
    semantic_leaf_limit: int = 1024


@dataclass(frozen=True)
class Model:
    model_id: str
    model_spec_id: str
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    run_id: str
    path: str
    sha256: str
    role: str
    format: str = 'pytorch_state_dict_v1'

    def __post_init__(self) -> None:
        validate_hash(self.sha256)
        if self.artifact_id != f'sha256:{self.sha256}':
            raise ValueError('artifact ID must address its bytes')


@dataclass(frozen=True)
class Run:
    run_id: str
    family_id: str
    variant_id: str
    model_spec_id: str
    model_id: str
    config_sha256: str
    status: Literal['completed', 'failed']


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str
    artifact_id: str
    run_id: str
    split_sha256: str
    protocol: str
    metrics: dict[str, object]
    gameplay_evaluation: str = 'not_run'
    uncertainty: str = 'not_applicable_synthetic_plumbing_no_population_claim'


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineageEdge:
    source: str
    target: str
    relation: str
    evidence: Evidence


def artifact_nodes(artifacts: list[Artifact]) -> list[Node]:
    """Best and final roles may legitimately name the same checkpoint bytes."""
    return [Node(identity, 'artifact') for identity in dict.fromkeys(artifact.artifact_id for artifact in artifacts)]


class Lineage:
    def __init__(self, nodes: list[Node], edges: list[LineageEdge]) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = edges
        if len(self.nodes) != len(nodes):
            raise ValueError('duplicate node ID')
        for edge in edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError('missing lineage endpoint')
            if edge.relation not in RELATIONS:
                raise ValueError('unsupported lineage relation')
        for node_id in self.nodes:
            self._ancestors(node_id, set())

    def resolve(self, identity: str) -> Node:
        if identity in self.nodes:
            return self.nodes[identity]
        matches = [node for node in self.nodes.values() if identity in node.aliases]
        if len(matches) != 1:
            raise ValueError(f'unknown or ambiguous alias: {identity}; candidates={[n.node_id for n in matches]}')
        return matches[0]

    def _ancestors(self, identity: str, active: set[str]) -> set[str]:
        if identity in active:
            raise ValueError('ancestry cycle')
        result: set[str] = set()
        for edge in self.edges:
            if edge.source == identity and edge.relation in ANCESTRY:
                result.add(edge.target)
                result.update(self._ancestors(edge.target, active | {identity}))
        return result

    def ancestors(self, identity: str) -> list[str]:
        return sorted(self._ancestors(self.resolve(identity).node_id, set()))

    def descendants(self, identity: str) -> list[str]:
        target = self.resolve(identity).node_id
        return sorted(node for node in self.nodes if target in self._ancestors(node, set()))

    def links(self, identity: str) -> list[LineageEdge]:
        target = self.resolve(identity).node_id
        return [edge for edge in self.edges if target in (edge.source, edge.target)]
