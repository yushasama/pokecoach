"""Materialise candidate actions from a deck's own answer/attacker/engine cards.

This does NOT fabricate live game state. For a given deck we read its
DeckProfileInference v0 profile (main attackers, draw/search engine) and its
MatchupThreatInference v0 threat surface (coverage items, fragile-bench evidence),
and turn each into an abstract :class:`ActionCandidate`. The scorer then judges each
candidate against an opponent threat surface.
"""

from __future__ import annotations

from deepmew.deck_profile_inference.schemas import DeckProfile, V2Card
from deepmew.matchup_threat_inference.schemas import DeckThreatProfile
from deepmew.strategic_action_scoring.schemas import (
    ActionCandidate,
    KIND_ATTACK_ADVANCE,
    KIND_BENCH_PLAY,
    KIND_PLAY_ANSWER,
    KIND_SETUP_ENGINE,
    ScorerConfig,
)


def _conf(v2map: dict[str, V2Card], card_id: str) -> str:
    card = v2map.get(card_id)
    return card.confidence if card else "low"


def build_candidates(
    deck_profile: DeckProfile,
    threat_profile: DeckThreatProfile,
    v2map: dict[str, V2Card],
    config: ScorerConfig | None = None,
) -> list[ActionCandidate]:
    config = config or ScorerConfig()
    prefix = threat_profile.deck_id
    by_key: dict[str, ActionCandidate] = {}

    # --- play_answer candidates from coverage items (merge per card) ----------
    coverage_by_card: dict[str, dict] = {}
    for item in threat_profile.coverage:
        for ref in item.evidence:
            entry = coverage_by_card.setdefault(
                ref.card_id,
                {"name": ref.name, "copies": ref.count, "is_core": ref.is_core, "covs": set()},
            )
            entry["covs"].add(item.category)
            entry["copies"] = max(entry["copies"], ref.count)
            entry["is_core"] = entry["is_core"] or ref.is_core
    for card_id, entry in coverage_by_card.items():
        cid = f"{prefix}:answer:{card_id}"
        by_key[cid] = ActionCandidate(
            candidate_id=cid,
            action_kind=KIND_PLAY_ANSWER,
            card_id=card_id,
            card_name=entry["name"],
            copies=entry["copies"],
            is_core=entry["is_core"],
            provides_coverage=tuple(sorted(entry["covs"])),
            card_confidence=_conf(v2map, card_id),
        )

    # --- attacker candidates (advance the main attacker) ----------------------
    for ref in deck_profile.main_attackers:
        cid = f"{prefix}:attack:{ref.card_id}"
        by_key[cid] = ActionCandidate(
            candidate_id=cid,
            action_kind=KIND_ATTACK_ADVANCE,
            card_id=ref.card_id,
            card_name=ref.name,
            copies=ref.count,
            is_core=ref.is_core,
            advances_main_attacker=True,
            card_confidence=_conf(v2map, ref.card_id),
        )

    # --- engine candidates (set up draw/search) -------------------------------
    seen_engine: set[str] = set()
    for ref in deck_profile.draw_engine + deck_profile.search_engine:
        if ref.card_id in seen_engine:
            continue
        seen_engine.add(ref.card_id)
        cid = f"{prefix}:setup:{ref.card_id}"
        by_key[cid] = ActionCandidate(
            candidate_id=cid,
            action_kind=KIND_SETUP_ENGINE,
            card_id=ref.card_id,
            card_name=ref.name,
            copies=ref.count,
            is_core=ref.is_core,
            sets_up_engine=True,
            card_confidence=_conf(v2map, ref.card_id),
        )

    # --- fragile bench play candidates (over-bench risk) ----------------------
    for item in threat_profile.exposes:
        if item.category != "fragile_bench_liability":
            continue
        for ref in item.evidence:
            cid = f"{prefix}:bench:{ref.card_id}"
            if cid in by_key:
                continue
            by_key[cid] = ActionCandidate(
                candidate_id=cid,
                action_kind=KIND_BENCH_PLAY,
                card_id=ref.card_id,
                card_name=ref.name,
                copies=ref.count,
                is_core=ref.is_core,
                benches_fragile=True,
                card_confidence=_conf(v2map, ref.card_id),
            )

    # Deterministic order: answers, then attackers, then engine, then bench; by id.
    kind_rank = {
        KIND_PLAY_ANSWER: 0,
        KIND_ATTACK_ADVANCE: 1,
        KIND_SETUP_ENGINE: 2,
        KIND_BENCH_PLAY: 3,
    }
    ordered = sorted(
        by_key.values(),
        key=lambda c: (kind_rank.get(c.action_kind, 9), -c.copies, c.card_name, c.card_id),
    )
    return ordered[: config.max_candidates_per_deck]
