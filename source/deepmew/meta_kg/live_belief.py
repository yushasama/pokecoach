"""Live opponent archetype belief from revealed cards (standalone utility).

Bayesian/log-lift update with per-card contribution caps and staple downweight,
so a single noisy tech or a generic staple cannot dominate the belief. This is a
reusable offline utility; it is NOT wired into the battle policy in this phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from deepmew.meta_kg.schemas import CardId


@dataclass
class ArchetypeHypothesis:
    archetype: str
    probability: float
    log_score: float
    evidence_card_ids: list[CardId] = field(default_factory=list)
    evidence_card_names: list[str] = field(default_factory=list)


@dataclass
class ArchetypeBeliefResult:
    top_archetype: str
    confidence: float
    entropy: float
    hypotheses: list[ArchetypeHypothesis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_archetype": self.top_archetype,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "hypotheses": [
                {
                    "archetype": h.archetype,
                    "probability": round(h.probability, 4),
                    "evidence_cards": h.evidence_card_ids,
                    "evidence_card_names": h.evidence_card_names,
                }
                for h in self.hypotheses
            ],
        }


class LiveArchetypeBelief:
    def __init__(
        self,
        archetype_priors: dict[str, float],
        card_likelihoods: dict[tuple[str, CardId], float],
        global_likelihoods: dict[CardId, float],
        staple_scores: dict[CardId, float],
        card_names: dict[CardId, str] | None = None,
        max_card_log_contribution: float = 2.0,
        evidence_scale: float = 1.0,
    ):
        if not math.isfinite(evidence_scale) or evidence_scale < 0.0:
            raise ValueError("evidence_scale must be finite and non-negative")
        self.archetype_priors = archetype_priors
        self.card_likelihoods = card_likelihoods
        self.global_likelihoods = global_likelihoods
        self.staple_scores = staple_scores
        self.card_names = card_names or {}
        self.max_card_log_contribution = max_card_log_contribution
        self.evidence_scale = evidence_scale

    def update(self, revealed_card_ids: list[CardId], top_k: int = 5) -> ArchetypeBeliefResult:
        # The likelihood tables model deck presence, not copy count. Normalizing
        # and deduplicating here also makes integer CABT IDs match persisted IDs.
        unique_reveals = list(dict.fromkeys(str(cid) for cid in revealed_card_ids if cid is not None))
        log_scores: dict[str, float] = {}
        evidence: dict[str, list[CardId]] = {}
        for archetype, prior in self.archetype_priors.items():
            score = math.log(max(prior, 1e-12))
            ev: list[CardId] = []
            for cid in unique_reveals:
                staple = self.staple_scores.get(cid, 0.0)
                p_global = self.global_likelihoods.get(cid, 1e-6)
                p_cond = self.card_likelihoods.get((archetype, cid), p_global)
                contrib = math.log(p_cond + 1e-12) - math.log(p_global + 1e-12)
                contrib *= max(0.0, 1.0 - staple)
                contrib *= self.evidence_scale
                contrib = max(-self.max_card_log_contribution, min(self.max_card_log_contribution, contrib))
                score += contrib
                if abs(contrib) > 0.1:
                    ev.append(cid)
            log_scores[archetype] = score
            evidence[archetype] = ev
        return self._normalize(log_scores, evidence, top_k)

    def _normalize(
        self,
        log_scores: dict[str, float],
        evidence: dict[str, list[CardId]],
        top_k: int,
    ) -> ArchetypeBeliefResult:
        if not log_scores:
            return ArchetypeBeliefResult("Unknown", 0.0, 0.0, [])
        max_log = max(log_scores.values())
        exp_scores = {a: math.exp(s - max_log) for a, s in log_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probs = {a: v / total for a, v in exp_scores.items()}

        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log(p)
        entropy_norm = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0

        ordered = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        hypotheses = [
            ArchetypeHypothesis(
                archetype=archetype,
                probability=prob,
                log_score=log_scores[archetype],
                evidence_card_ids=evidence.get(archetype, []),
                evidence_card_names=[self.card_names.get(cid, f"Unknown Card {cid}") for cid in evidence.get(archetype, [])],
            )
            for archetype, prob in ordered[:top_k]
        ]
        top = ordered[0]
        return ArchetypeBeliefResult(
            top_archetype=top[0],
            confidence=top[1],
            entropy=round(entropy_norm, 4),
            hypotheses=hypotheses,
        )


def build_live_belief_payload(
    archetype_priors: dict[str, float],
    lift_rows: list[dict[str, Any]],
    global_likelihoods: dict[CardId, float],
    staple_scores: dict[CardId, float],
) -> dict[str, Any]:
    """Serialize the statistics a LiveArchetypeBelief needs into a JSON payload."""
    card_likelihoods: dict[str, dict[str, float]] = {}
    for row in lift_rows:
        card_likelihoods.setdefault(row["archetype"], {})[row["card_id"]] = float(row["p_card_given_archetype"])
    return {
        "schema_version": "deepmew.meta_kg.live_belief.v1",
        "archetype_priors": {a: round(p, 6) for a, p in sorted(archetype_priors.items())},
        "card_likelihoods": {a: dict(sorted(cards.items())) for a, cards in sorted(card_likelihoods.items())},
        "global_likelihoods": {cid: round(p, 6) for cid, p in sorted(global_likelihoods.items())},
        "staple_scores": {cid: round(s, 6) for cid, s in sorted(staple_scores.items())},
    }


def live_belief_from_payload(payload: dict[str, Any], card_names: dict[CardId, str] | None = None) -> LiveArchetypeBelief:
    card_likelihoods: dict[tuple[str, CardId], float] = {}
    for archetype, cards in payload.get("card_likelihoods", {}).items():
        for cid, p in cards.items():
            card_likelihoods[(archetype, cid)] = float(p)
    return LiveArchetypeBelief(
        archetype_priors={a: float(p) for a, p in payload.get("archetype_priors", {}).items()},
        card_likelihoods=card_likelihoods,
        global_likelihoods={cid: float(p) for cid, p in payload.get("global_likelihoods", {}).items()},
        staple_scores={cid: float(s) for cid, s in payload.get("staple_scores", {}).items()},
        card_names=card_names,
    )
