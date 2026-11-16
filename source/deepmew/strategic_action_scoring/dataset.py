"""Assemble a bounded, inspectable strategic-action-scoring dataset.

For each subject deck we build candidate actions from its own cards and score them
against a small set of meta-representative opponent threat surfaces (deduped by
primary-threat signature, then capped to the most frequent). Every row is a
transparent :class:`ActionScore` plus display labels. Deterministic and offline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from deepmew.deck_profile_inference.inference import infer_profile
from deepmew.deck_profile_inference.schemas import DeckList, ProfileConfig, V2Card
from deepmew.matchup_threat_inference.inference import build_all_profiles
from deepmew.matchup_threat_inference.schemas import (
    DeckThreatProfile,
    MatchupConfig,
)
from deepmew.strategic_action_scoring.candidates import build_candidates
from deepmew.strategic_action_scoring.scorer import score_action
from deepmew.strategic_action_scoring.schemas import (
    ActionCandidate,
    MatchupContext,
    ScorerConfig,
)


def _signature(profile: DeckThreatProfile) -> tuple:
    return tuple(sorted(t.category for t in profile.all_threats() if t.is_primary))


# Threats we want at least one representative presenter for, so the dataset
# exercises every answer reason tag (label-free; these are categories, not names).
_KEY_THREATS: tuple[str, ...] = (
    "bench_attack_damage_threat",
    "bench_damage_counter_threat",
    "ability_damage_counter_threat",
    "hand_size_scaling_threat",
    "tool_dependency_threat",
    "stadium_dependency_threat",
    "special_energy_dependency_threat",
)


def _primary(profile: DeckThreatProfile) -> set[str]:
    return {t.category for t in profile.all_threats() if t.is_primary}


def select_representative_opponents(
    profiles: list[DeckThreatProfile], max_opponents: int
) -> list[DeckThreatProfile]:
    """A bounded, label-free, deterministic opponent set. First guarantee one
    representative presenter per key threat (so every answer reason tag is exercised),
    then fill remaining slots with the most frequent primary-threat signatures."""
    freq: Counter = Counter(_signature(p) for p in profiles)
    rep: dict[tuple, DeckThreatProfile] = {}
    for profile in sorted(profiles, key=lambda p: p.deck_id):
        sig = _signature(profile)
        if sig:  # skip the empty-threat signature (a poor opponent)
            rep.setdefault(sig, profile)
    ranked = sorted(rep.values(), key=lambda p: (-freq[_signature(p)], p.deck_id))

    chosen: list[DeckThreatProfile] = []
    chosen_ids: set[str] = set()

    def add(profile: DeckThreatProfile) -> None:
        if profile.deck_id not in chosen_ids:
            chosen.append(profile)
            chosen_ids.add(profile.deck_id)

    for threat in _KEY_THREATS:
        for profile in ranked:
            if threat in _primary(profile):
                add(profile)
                break
    for profile in ranked:
        if len(chosen) >= max_opponents:
            break
        add(profile)
    return sorted(chosen, key=lambda p: p.deck_id)


def build_score_rows(
    decks: list[DeckList],
    v2map: dict[str, V2Card],
    config: ScorerConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or ScorerConfig()
    threat_profiles = build_all_profiles(decks, v2map, MatchupConfig())
    by_id = {p.deck_id: p for p in threat_profiles}

    # Re-derive Phase 2 profiles with matching deck_ids (sorted by deck_hash).
    ordered = sorted(decks, key=lambda d: d.deck_hash)
    deck_profiles = {
        f"deck_{i:04d}": infer_profile(deck, v2map, ProfileConfig(), deck_id=f"deck_{i:04d}")
        for i, deck in enumerate(ordered)
    }

    opponents = select_representative_opponents(threat_profiles, config.max_opponents)

    rows: list[dict[str, Any]] = []
    for subject in sorted(threat_profiles, key=lambda p: p.deck_id):
        candidates: list[ActionCandidate] = build_candidates(
            deck_profiles[subject.deck_id], subject, v2map, config
        )
        for opponent in opponents:
            if opponent.deck_id == subject.deck_id:
                continue
            context = MatchupContext(subject=subject, opponent=opponent)
            for candidate in candidates:
                score = score_action(candidate, context, config)
                row = score.to_dict()
                # Display-only labels (never used as scoring logic).
                row["subject_label"] = subject.archetype_label
                row["opponent_label"] = opponent.archetype_label
                row["provides_coverage"] = list(candidate.provides_coverage)
                row["is_core"] = candidate.is_core
                rows.append(row)
    rows.sort(key=lambda r: (r["subject_deck_id"], r["opponent_deck_id"], r["candidate_id"]))
    _ = by_id  # threat profiles already indexed via subject loop
    return rows
