from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.legacy import SOURCES, import_legacy
from deepmew_ex.registry.records import Artifact, Evidence, Lineage, LineageEdge, Node, artifact_nodes
from deepmew_ex.registry.store import file_hash


def evidence() -> Evidence:
    return Evidence('synthetic', 'a' * 64, '/fixture', 'verified', 'fixture only')


def test_identity_and_lineage_errors() -> None:
    nodes = [Node('a', 'model', ('shared',)), Node('b', 'artifact', ('shared',))]
    graph = Lineage(nodes, [LineageEdge('b', 'a', 'initialized_from', evidence())])
    assert graph.ancestors('b') == ['a'] and graph.descendants('a') == ['b']
    with pytest.raises(ValueError, match='ambiguous'):
        graph.resolve('shared')
    with pytest.raises(ValueError, match='duplicate'):
        Lineage(nodes + [nodes[0]], [])
    with pytest.raises(ValueError, match='endpoint'):
        Lineage(nodes, [LineageEdge('a', 'missing', 'initialized_from', evidence())])
    with pytest.raises(ValueError, match='cycle'):
        Lineage(nodes, [LineageEdge('a', 'b', 'initialized_from', evidence()), LineageEdge('b', 'a', 'fine_tuned_from', evidence())])
    # Opponent cycles have no weight-ancestry meaning.
    Lineage(nodes, [LineageEdge('a', 'b', 'trained_against', evidence()), LineageEdge('b', 'a', 'evaluated_against', evidence())])
    with pytest.raises(ValueError, match='relation'):
        Lineage(nodes, [LineageEdge('a', 'b', 'guessed_parent', evidence())])


def test_best_final_can_share_content_identity() -> None:
    sha = 'a' * 64
    artifacts = [Artifact(f'sha256:{sha}', 'run', f'checkpoints/{role}.pt', sha, role) for role in ('best', 'final')]
    graph = Lineage(artifact_nodes(artifacts), [])
    assert len(graph.nodes) == 1
    assert {artifact.role for artifact in artifacts} == {'best', 'final'}


@pytest.mark.parametrize('sha', ['bad', 'a' * 63, 'A' * 64])
def test_bad_hashes(sha: str) -> None:
    with pytest.raises(ValueError, match='sha256'):
        Artifact(f'sha256:{sha}', 'run', 'missing.pt', sha, 'best')


def test_metadata_import_preserves_sources_aliases_status_and_missing_assets() -> None:
    before = {source: file_hash(PROJECT_ROOT / source) for source in SOURCES}
    with patch('torch.load', side_effect=AssertionError('checkpoint deserialized')):
        imported = import_legacy(PROJECT_ROOT)
    assert before == {source: file_hash(PROJECT_ROOT / source) for source in SOURCES}
    assert len(imported.sources) == 8 and len(imported.records) == 332
    for alias in ('D1', 'E1', 'r7', 'X2', 'H2', 'D1_SUCCESSOR'):
        assert imported.lookup(alias)['records']
    assert imported.lookup('D1')['resolution'] == 'ambiguous_preserved'
    successor = imported.lookup('D1_SUCCESSOR')['records'][0]
    assert successor['fields']['status'] == 'QUALIFIED_RESEARCH_LEAGUE_D1_SUCCESSOR_NOT_PROMOTED'
    assert all(not record.training_reproduced for record in imported.records)
    assert any(record.asset_presence == 'missing' for record in imported.records)
    assert len(imported.unresolved) == 2
    assert len(imported.edges) == 1 and imported.edges[0].relation == 'initialized_from'


def test_unavailable_metadata_is_explicit(tmp_path: Path) -> None:
    imported = import_legacy(tmp_path)
    assert not imported.records and len(imported.unresolved) == len(SOURCES) + 2
