"""Source-bound static evidence without importing archived execution code."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

from deepmew_ex.registry.store import file_hash


@dataclass(frozen=True)
class SourceEvidence:
    path: str
    sha256: str
    symbol: str
    first_line: int
    last_line: int
    claim: str
    level: str = 'VERIFIED_STATIC_SOURCE; behavior not re-executed'


def inspect_sources(project: Path) -> dict[str, object]:
    declarations = (
        ('tools/reconstruction/export_corpus.py', 'encode_reference', 'Frozen legal/chosen/pre-action identities checked; actor projection and 304-feature encoder are called.'),
        ('tools/reconstruction/export_corpus.py', 'export_shard', '144-feature corrected observations loaded from reference index; exported metadata and zero forced actor weights preserved.'),
        ('source/deepmew/replay/contract/source_events.py', 'project_actor_observation', 'Drops opaque logs/search state, private looking buffer and select.deck, then asserts actor-visible projection.'),
        ('source/deepmew/replay/canonical_actor_dataset.py', 'canonical_actor_feature_observation', 'Defines the actor feature observation projection consumed by export.'),
        ('source/deepmew/rl/enhanced_replay_derived.py', 'load_reference_observations', 'Defines corrected 144-state feature retrieval and its reference binding checks.'),
        ('source/scripts/run_terminal_league_ppo.py', 'encode_actor_visible_decision', 'Defines exact legal-option order and finite 304-dimensional actor feature tensors.'),
        ('source/deepmew/replay/information_boundary.py', 'assert_deployable_observation', 'Defines actor-visible input classification and deployment rejection checks.'),
        ('source/deepmew/replay/contract/legal_actions.py', 'complete_action_count', 'Complete action cardinality is distinct from stage candidate count for multi-selection.'),
    )
    evidence: list[SourceEvidence] = []
    missing: list[str] = []
    for relative, symbol, claim in declarations:
        path = project / relative
        if not path.is_file():
            missing.append(relative)
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        matches = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol]
        if len(matches) != 1:
            missing.append(f'{relative}#{symbol}')
            continue
        node = matches[0]
        evidence.append(SourceEvidence(relative, file_hash(path), symbol, node.lineno, node.end_lineno or node.lineno, claim))
    return {'status': 'receipt_supported_not_reproduced' if not missing else 'blocked_missing_source_binding', 'bindings': [asdict(item) for item in evidence], 'missing': missing, 'claims_not_established': ['upstream raw-source privacy negative controls were not rerun', 'export was not regenerated from historical raw sources', 'feature width does not certify source visibility', 'no semantic or temporal information was reconstructed from fixed features']}
