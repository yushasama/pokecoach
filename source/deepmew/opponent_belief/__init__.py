"""OpponentBeliefState v0 - offline, deterministic public-information belief (DEC-038).

Builds a hierarchical opponent belief (archetype / list variant / known deck hash /
structural playstyle / UNKNOWN) from public/revealed card events using the existing
K2A/DSU/deck-zoo knowledge concepts (log-lift diagnosticity, staple/splash
downweighting, support-chain/package combination evidence). It is research-only:

  * public/revealed information only (no future replay rows, no private hand,
    no unrevealed prize/deck, no final deck identity during update generation);
  * deterministic and regenerable; ground-truth deck identity is joined only by the
    evaluation module, never by the update logic;
  * never imports the submission runtime (`main`), the battle engine (`cg`), the
    active policy, or the evaluation path (INVARIANT #7/#8);
  * trains no model and changes no runtime/eval policy.

This package extends - it does not replace - ``deepmew.meta_kg.live_belief``
(``LiveArchetypeBelief``), which remains the flat archetype-only log-lift belief.
"""

from deepmew.opponent_belief.belief import OpponentBeliefState
from deepmew.opponent_belief.knowledge import KnowledgeBase, build_fixture_knowledge_base
from deepmew.opponent_belief.schemas import (
    ARTIFACT_VERSION,
    BELIEF_KIND,
    BeliefConfig,
    BeliefSnapshot,
    EvidenceEvent,
    UpdateTrace,
)

__all__ = [
    "ARTIFACT_VERSION",
    "BELIEF_KIND",
    "BeliefConfig",
    "BeliefSnapshot",
    "EvidenceEvent",
    "KnowledgeBase",
    "OpponentBeliefState",
    "UpdateTrace",
    "build_fixture_knowledge_base",
]
