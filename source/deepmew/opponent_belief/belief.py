"""OpponentBeliefState v0: hierarchical, public-information belief updater.

Maintains a leaf distribution over:
  * one leaf per known deck hash (LEVEL_KNOWN_HASH);
  * one ``unknown_within_archetype`` leaf per archetype;
  * one global ``GLOBAL_UNKNOWN`` leaf (flat baseline, never pushed to zero).

Each public evidence event contributes, transparently:
  * diagnosticity (log-lift x role x (1 - staple)), applied at archetype granularity;
  * a correlation-safe DSU/package chain bonus (sub-modular increment + completion),
    applied at archetype granularity, counting each card once per component;
  * a hash-level contradiction penalty (foreign signature / over-copy), which shifts
    mass from an exact hash to ``unknown_within_archetype`` without erasing the
    archetype.

A configurable UNKNOWN floor guarantees unknown mass never collapses. The update
logic reads ONLY public ``EvidenceEvent`` fields - ground truth is never visible
here. See :mod:`deepmew.opponent_belief.evaluation` for the (separate) scoring path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deepmew.opponent_belief.knowledge import (
    KnowledgeBase,
    KnownDeck,
    ROLE_ENERGY,
    ROLE_LINE,
    ROLE_SPLASH,
    ROLE_STAPLE,
    ROLE_SUPPORT,
    ROLE_TECH,
    ROLE_TERMINAL,
    inverse_prevalence_weight,
)
from deepmew.opponent_belief.schemas import (
    BeliefConfig,
    BeliefSnapshot,
    CardId,
    EvidenceContribution,
    EvidenceEvent,
    GLOBAL_UNKNOWN_ID,
    LEVEL_GLOBAL_UNKNOWN,
    LEVEL_KNOWN_HASH,
    LEVEL_UNKNOWN_VARIANT,
    STATUS_AMBIGUOUS,
    STATUS_CONCENTRATED,
    STATUS_EXACT_HASH_CANDIDATE,
    STATUS_LIKELY,
    UNKNOWN_ARCHETYPE,
    UNKNOWN_VARIANT,
    UpdateTrace,
)

_SIGNATURE_LIFT = 5.0  # a card with in-archetype lift >= this is a "signature"
_IDENTITY_ROLES = frozenset({ROLE_TERMINAL, ROLE_LINE, ROLE_SUPPORT, ROLE_TECH})


@dataclass(frozen=True)
class Leaf:
    leaf_id: str
    level: str
    archetype: str
    variant_label: str
    deck: KnownDeck | None


def _role_factor(role: str, cfg: BeliefConfig) -> float:
    return {
        ROLE_TERMINAL: cfg.role_factor_terminal,
        ROLE_LINE: cfg.role_factor_line,
        ROLE_SUPPORT: cfg.role_factor_support,
        ROLE_ENERGY: cfg.role_factor_energy,
        ROLE_TECH: cfg.role_factor_tech,
        ROLE_STAPLE: cfg.role_factor_staple,
        ROLE_SPLASH: cfg.role_factor_splash,
    }.get(role, cfg.role_factor_support)


class OpponentBeliefState:
    """Deterministic hierarchical belief; one instance per modeled opponent/game."""

    def __init__(self, kb: KnowledgeBase, config: BeliefConfig | None = None, modeled_actor: str = "opponent"):
        self.kb = kb
        self.cfg = config or BeliefConfig()
        self.modeled_actor = modeled_actor
        self.leaves: list[Leaf] = self._build_leaves()
        self.leaf_by_id = {leaf.leaf_id: leaf for leaf in self.leaves}
        self.log_score: dict[str, float] = {}
        # Chain bookkeeping: per (archetype, component_id) -> set of observed members.
        self._component_observed: dict[tuple[str, str], set[CardId]] = {}
        self._component_score: dict[tuple[str, str], float] = {}
        # Per-card public copy count seen so far (must be monotonic non-decreasing).
        self._seen_copies: dict[CardId, int] = {}
        self._seen_cards: set[CardId] = set()
        # Idempotency + ordering bookkeeping.
        self._applied_event_ids: set[str] = set()
        self._last_applied_seq: int = -1
        self.sequence_index = 0
        self._init_priors()

    # -- construction ----------------------------------------------------------
    def _build_leaves(self) -> list[Leaf]:
        leaves: list[Leaf] = []
        for deck in sorted(self.kb.decks, key=lambda d: d.deck_hash):
            leaves.append(
                Leaf(
                    leaf_id=f"hash:{deck.deck_hash}",
                    level=LEVEL_KNOWN_HASH,
                    archetype=deck.archetype,
                    variant_label=deck.variant_label,
                    deck=deck,
                )
            )
        for archetype in self.kb.archetype_names():
            leaves.append(
                Leaf(
                    leaf_id=f"unkarch:{archetype}",
                    level=LEVEL_UNKNOWN_VARIANT,
                    archetype=archetype,
                    variant_label=UNKNOWN_VARIANT,
                    deck=None,
                )
            )
        leaves.append(
            Leaf(
                leaf_id=GLOBAL_UNKNOWN_ID,
                level=LEVEL_GLOBAL_UNKNOWN,
                archetype=UNKNOWN_ARCHETYPE,
                variant_label=UNKNOWN_VARIANT,
                deck=None,
            )
        )
        return leaves

    def _init_priors(self) -> None:
        non_global = [leaf for leaf in self.leaves if leaf.level != LEVEL_GLOBAL_UNKNOWN]
        g = self.cfg.global_unknown_prior
        share = (1.0 - g) / max(1, len(non_global))
        for leaf in self.leaves:
            prior = g if leaf.level == LEVEL_GLOBAL_UNKNOWN else share
            self.log_score[leaf.leaf_id] = math.log(max(prior, 1e-12))

    # -- diagnosticity ---------------------------------------------------------
    def _single_card_evidence(self, card_id: CardId, archetype: str) -> float:
        """Log-lift x role x (1 - staple) x inverse-prevalence, capped. >= 0 in-arch."""
        card = self.kb.card(card_id)
        if card is None:
            return 0.0
        lift = card.lift_by_archetype.get(archetype, 1.0)
        base = math.log(max(lift, self.cfg.lift_floor))
        rf = _role_factor(card.role, self.cfg)
        inv = 0.5 + 0.5 * (inverse_prevalence_weight(card.global_prevalence) / 3.0)
        contrib = base * rf * (1.0 - card.staple_score) * inv
        return _clamp(contrib, -self.cfg.max_card_log_contribution, self.cfg.max_card_log_contribution)

    def _diagnosticity_for_leaf(self, leaf: Leaf, card_id: CardId) -> float:
        card = self.kb.card(card_id)
        if card is None or leaf.level == LEVEL_GLOBAL_UNKNOWN:
            return 0.0  # global UNKNOWN is a flat baseline; never moved by diagnosticity
        if leaf.archetype in card.lift_by_archetype and card.lift_by_archetype[leaf.archetype] > 1.0:
            return self._single_card_evidence(card_id, leaf.archetype)
        # Foreign card. Only reliable signatures of *other* archetypes push a leaf down.
        max_lift = max(card.lift_by_archetype.values(), default=1.0)
        if card.reliable and card.role in _IDENTITY_ROLES and max_lift >= _SIGNATURE_LIFT:
            rf = _role_factor(card.role, self.cfg)
            penalty = -0.4 * math.log(max_lift) * rf
            return _clamp(penalty, -self.cfg.max_card_log_contribution, 0.0)
        return 0.0

    # -- chain bonus (correlation-safe, archetype granularity) -----------------
    def _chain_delta_by_archetype(self, card_id: CardId) -> dict[str, float]:
        """Recompute component scores touched by this card; return per-archetype delta.

        Each component scores sub-modularly: increment for distinct members + a
        completion bonus. The strongest single member is already credited by
        diagnosticity, so only the *combination* part is added here, and each card is
        counted at most once per component (tracked in ``_component_observed``).
        """
        deltas: dict[str, float] = {}
        if not self.cfg.use_chain:
            return deltas
        for archetype in self.kb.archetype_names():
            for comp in self.kb.components(archetype):
                if card_id not in comp.members:
                    continue
                key = (archetype, comp.component_id)
                observed = self._component_observed.setdefault(key, set())
                if card_id in observed:
                    continue
                observed.add(card_id)
                n = len(observed)
                increment = min(self.cfg.chain_member_cap, self.cfg.chain_gamma * math.log(1 + max(0, n - 1)))
                completed = (
                    (comp.prereq is None or comp.prereq in observed)
                    and (comp.support is None or comp.support in observed)
                    and (comp.payoff is None or comp.payoff in observed)
                    and n >= 2
                )
                completion = self.cfg.chain_completion_bonus if completed else 0.0
                new_score = increment + completion
                old_score = self._component_score.get(key, 0.0)
                self._component_score[key] = new_score
                # Variant-specific components route only to matching-variant hashes (handled
                # at apply time); record the archetype delta and let apply gate it.
                deltas[archetype] = deltas.get(archetype, 0.0) + (new_score - old_score)
        return deltas

    def _variant_specific_archetypes(self, card_id: CardId) -> dict[str, str]:
        """component archetype -> variant_label for variant-specific components hit."""
        out: dict[str, str] = {}
        for archetype in self.kb.archetype_names():
            for comp in self.kb.components(archetype):
                if comp.variant_specific and card_id in comp.members:
                    out[comp.component_id] = comp.variant_specific
        return out

    # -- hash multiset compatibility (positive coverage, hash level only) ------
    def _coverage_for_hash(self, leaf: Leaf, card_id: CardId) -> float:
        """Reward a known hash when a revealed card is in its list.

        Staples appear in every list, so coverage is scaled by ``(1 - staple_score)``
        and they barely distinguish hashes; a signature/support card in this exact
        list lifts the hash above its archetype's ``unknown_within_archetype`` leaf.
        """
        if not self.cfg.use_contradiction or leaf.deck is None:
            return 0.0
        if card_id not in leaf.deck.card_counts:
            return 0.0
        card = self.kb.card(card_id)
        staple = card.staple_score if card else 1.0
        return self.cfg.hash_coverage_weight * (1.0 - staple)

    # -- contradiction (hash level only) ---------------------------------------
    def _contradiction_for_hash(self, leaf: Leaf, card_id: CardId, count: int) -> float:
        if not self.cfg.use_contradiction or leaf.deck is None:
            return 0.0
        deck = leaf.deck
        d_count = deck.card_counts.get(card_id, 0)
        if d_count > 0:
            return self.cfg.overcopy_penalty if count > d_count else 0.0
        card = self.kb.card(card_id)
        if card is None or not card.reliable or card.role not in _IDENTITY_ROLES:
            return 0.0  # staples / splash / energy not in the list are not contradictions
        if leaf.archetype in card.lift_by_archetype:
            return 0.5 * self.cfg.contradiction_penalty  # same-archetype tech absent from this exact list
        max_lift = max(card.lift_by_archetype.values(), default=1.0)
        if max_lift >= _SIGNATURE_LIFT:
            return self.cfg.contradiction_penalty  # foreign signature: this deck cannot be D
        return 0.25 * self.cfg.contradiction_penalty

    # -- update ----------------------------------------------------------------
    def _noop_trace(self, event: EvidenceEvent, skip_reason: str, reliability: float = 1.0) -> UpdateTrace:
        """A no-op trace: the belief is unchanged. Used for duplicate / zero-reliability events."""
        snap = self.snapshot()
        unknown = self._unknown_mass(self._normalized_probs())
        return UpdateTrace(
            sequence_index=event.sequence_index,
            event=event,
            diagnosticity_total=0.0,
            chain_total=0.0,
            coverage_total=0.0,
            contradiction_total=0.0,
            unknown_mass_before=unknown,
            unknown_mass_after=unknown,
            top_before=snap.top_hypothesis,
            top_after=snap.top_hypothesis,
            top_probability_after=snap.top_probability,
            applied=False,
            skip_reason=skip_reason,
            reliability_applied=reliability,
            contributions=[],
        )

    def update(self, event: EvidenceEvent) -> UpdateTrace:
        # 1. Leakage boundary: only public evidence may ever enter the update logic.
        if not event.is_public():
            raise ValueError(
                f"OpponentBeliefState consumes public evidence only; got scope="
                f"{event.evidence_scope!r} type={event.event_type!r}"
            )
        # 2. Idempotency FIRST: replaying an already-applied event_id is a no-op (so it
        #    is never penalized by the later monotonic-sequence guard).
        if event.event_id in self._applied_event_ids:
            return self._noop_trace(event, "duplicate_event_id")
        # 3. Actor boundary: this state models exactly one opponent.
        if event.actor != self.modeled_actor:
            raise ValueError(
                f"OpponentBeliefState models actor={self.modeled_actor!r}; refusing event from actor={event.actor!r}"
            )
        # 4. Count integrity: positive integer, monotonic non-decreasing per card.
        if isinstance(event.count, bool) or not isinstance(event.count, int) or event.count < 1:
            raise ValueError(f"EvidenceEvent.count must be a positive integer; got {event.count!r}")
        prev_copies = self._seen_copies.get(event.card_id, 0)
        if event.count < prev_copies:
            raise ValueError(
                f"public copy count for card {event.card_id} decreased from {prev_copies} to {event.count}"
            )
        # 5. Sequence integrity: non-negative integer, strictly increasing for applied events.
        if isinstance(event.sequence_index, bool) or not isinstance(event.sequence_index, int) or event.sequence_index < 0:
            raise ValueError(f"sequence_index must be a non-negative integer; got {event.sequence_index!r}")
        if event.sequence_index <= self._last_applied_seq:
            raise ValueError(
                f"non-monotonic sequence_index {event.sequence_index} (last applied {self._last_applied_seq}); "
                f"duplicate or out-of-order distinct events are rejected"
            )
        # 6. Reliability: validate, and treat 0 as an explicit non-applied (no-evidence)
        #    trace that mutates NO bookkeeping, so later reliable evidence still counts.
        reliability = event.reliability
        if not (0.0 <= reliability <= 1.0):
            raise ValueError(f"reliability must be in [0,1]; got {reliability!r}")
        if reliability == 0.0:
            return self._noop_trace(event, "zero_reliability", reliability=0.0)

        self.sequence_index = event.sequence_index
        before_snapshot = self.snapshot()
        before_probs = self._normalized_probs()
        unknown_before = self._unknown_mass(before_probs)

        scale = self.cfg.temperature * reliability
        chain_deltas = self._chain_delta_by_archetype(event.card_id)
        variant_hits = self._variant_specific_archetypes(event.card_id)
        variant_labels = set(variant_hits.values())

        contributions: list[EvidenceContribution] = []
        diag_total = chain_total = coverage_total = contra_total = 0.0
        for leaf in self.leaves:
            diag = self._diagnosticity_for_leaf(leaf, event.card_id) * scale
            chain = 0.0
            if leaf.level != LEVEL_GLOBAL_UNKNOWN and leaf.archetype in chain_deltas:
                # Variant-specific chain evidence only strengthens matching-variant hashes.
                if variant_labels and leaf.level == LEVEL_KNOWN_HASH and leaf.variant_label not in variant_labels:
                    chain = 0.0
                else:
                    chain = chain_deltas[leaf.archetype] * scale
            coverage = 0.0
            contra = 0.0
            if leaf.level == LEVEL_KNOWN_HASH:
                coverage = self._coverage_for_hash(leaf, event.card_id) * scale
                contra = -self._contradiction_for_hash(leaf, event.card_id, event.count) * scale
            self.log_score[leaf.leaf_id] += diag + chain + coverage + contra
            diag_total += diag
            chain_total += chain
            coverage_total += coverage
            contra_total += contra
            if abs(diag) > 1e-9 or abs(chain) > 1e-9 or abs(coverage) > 1e-9 or abs(contra) > 1e-9:
                contributions.append(
                    EvidenceContribution(
                        hypothesis_id=leaf.leaf_id,
                        posterior_before=before_probs.get(leaf.leaf_id, 0.0),
                        diagnosticity=diag,
                        chain_bonus=chain,
                        coverage=coverage,
                        contradiction=contra,
                        log_score_after=self.log_score[leaf.leaf_id],
                    )
                )
        self._seen_cards.add(event.card_id)
        self._seen_copies[event.card_id] = event.count
        self._applied_event_ids.add(event.event_id)
        self._last_applied_seq = event.sequence_index

        after_probs = self._normalized_probs()
        unknown_after = self._unknown_mass(after_probs)
        after_snapshot = self.snapshot()
        return UpdateTrace(
            sequence_index=event.sequence_index,
            event=event,
            diagnosticity_total=diag_total,
            chain_total=chain_total,
            coverage_total=coverage_total,
            contradiction_total=contra_total,
            unknown_mass_before=unknown_before,
            unknown_mass_after=unknown_after,
            top_before=before_snapshot.top_hypothesis,
            top_after=after_snapshot.top_hypothesis,
            top_probability_after=after_snapshot.top_probability,
            applied=True,
            skip_reason="",
            reliability_applied=reliability,
            contributions=sorted(contributions, key=lambda c: (-abs(c.diagnosticity + c.chain_bonus + c.coverage + c.contradiction), c.hypothesis_id)),
        )

    def update_sequence(self, events: list[EvidenceEvent]) -> list[UpdateTrace]:
        """Apply events in deterministic sequence order.

        Input may be unsorted; events are sorted by ``(sequence_index, event_id)`` so a
        shuffled but uniquely-indexed stream yields the same result. Duplicate sequence
        indexes across distinct events are then rejected by the monotonic guard in
        :meth:`update` (no silent reordering).
        """
        ordered = sorted(events, key=lambda e: (e.sequence_index, e.event_id))
        return [self.update(event) for event in ordered]

    # -- normalization ---------------------------------------------------------
    def _raw_softmax(self) -> dict[str, float]:
        max_log = max(self.log_score.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in self.log_score.items()}
        total = sum(exp_scores.values()) or 1.0
        return {k: v / total for k, v in exp_scores.items()}

    def _normalized_probs(self) -> dict[str, float]:
        if self.cfg.hard_top1:
            return self._hard_top1_probs()
        if self.cfg.uniform_over_matches:
            return self._uniform_match_probs()
        probs = self._raw_softmax()
        if self.cfg.use_unknown_floor:
            probs = self._apply_unknown_floor(probs)
        return probs

    def _hard_top1_probs(self) -> dict[str, float]:
        ordered = sorted(self.log_score.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ordered[0][0]
        eps = 1e-6
        n = len(self.log_score)
        return {k: (1.0 - eps) if k == top else eps / max(1, n - 1) for k in self.log_score}

    def _uniform_match_probs(self) -> dict[str, float]:
        prior_log = math.log(max((1.0 - self.cfg.global_unknown_prior) / max(1, len(self.leaves) - 1), 1e-12))
        matched = [k for k, v in self.log_score.items() if v > prior_log + 1e-9 and k != GLOBAL_UNKNOWN_ID]
        if not matched:
            return self._raw_softmax()
        eps = 1e-6
        p = (1.0 - eps) / len(matched)
        out = {k: eps / max(1, len(self.log_score) - len(matched)) for k in self.log_score}
        for k in matched:
            out[k] = p
        return out

    def _apply_unknown_floor(self, probs: dict[str, float]) -> dict[str, float]:
        unknown_ids = [leaf.leaf_id for leaf in self.leaves if leaf.level != LEVEL_KNOWN_HASH]
        unk = sum(probs[i] for i in unknown_ids)
        floor = self.cfg.unknown_floor
        if unk >= floor or unk >= 1.0:
            return probs
        out = dict(probs)
        scale = (1.0 - floor) / max(1.0 - unk, 1e-9)
        for k in out:
            if k not in unknown_ids:
                out[k] *= scale
        add = floor - unk
        for i in unknown_ids:
            share = (probs[i] / unk) if unk > 1e-12 else (1.0 / len(unknown_ids))
            out[i] = probs[i] + add * share
        return out

    def _unknown_mass(self, probs: dict[str, float]) -> float:
        return sum(probs[leaf.leaf_id] for leaf in self.leaves if leaf.level != LEVEL_KNOWN_HASH)

    # -- snapshot --------------------------------------------------------------
    def snapshot(self) -> BeliefSnapshot:
        probs = self._normalized_probs()
        archetype: dict[str, float] = {}
        variant: dict[str, float] = {}
        known_hash: dict[str, float] = {}
        playstyle: dict[str, float] = {}
        for leaf in self.leaves:
            p = probs[leaf.leaf_id]
            archetype[leaf.archetype] = archetype.get(leaf.archetype, 0.0) + p
            vkey = f"{leaf.archetype} :: {leaf.variant_label}"
            variant[vkey] = variant.get(vkey, 0.0) + p
            if leaf.level == LEVEL_KNOWN_HASH:
                known_hash[leaf.deck.deck_hash] = known_hash.get(leaf.deck.deck_hash, 0.0) + p
            pstyle = self.kb.playstyle(leaf.archetype) if leaf.archetype != UNKNOWN_ARCHETYPE else "unknown"
            playstyle[pstyle] = playstyle.get(pstyle, 0.0) + p

        ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        top_id, top_p = ordered[0]
        runner = ordered[1][1] if len(ordered) > 1 else 0.0
        entropy = _normalized_entropy(list(probs.values()))
        credible, credible_mass = _credible_set(ordered, self.cfg.credible_mass)
        global_unknown = probs[GLOBAL_UNKNOWN_ID]
        within = sum(probs[leaf.leaf_id] for leaf in self.leaves if leaf.level == LEVEL_UNKNOWN_VARIANT)
        top_leaf = self.leaf_by_id[top_id]

        return BeliefSnapshot(
            sequence_index=self.sequence_index,
            archetype_posterior=archetype,
            variant_posterior=variant,
            known_hash_posterior=known_hash,
            structural_playstyle_posterior=playstyle,
            unknown_mass=global_unknown + within,
            unknown_within_archetype_mass=within,
            global_unknown_mass=global_unknown,
            top_hypothesis=top_id,
            top_archetype=top_leaf.archetype,
            top_probability=top_p,
            runner_up_probability=runner,
            margin=top_p - runner,
            normalized_entropy=entropy,
            credible_set=credible,
            credible_set_mass=credible_mass,
            belief_status=self._status(top_leaf, top_p, top_p - runner, entropy),
        )

    def _status(self, top_leaf: Leaf, top_p: float, margin: float, entropy: float) -> str:
        cfg = self.cfg
        if (
            top_leaf.level == LEVEL_KNOWN_HASH
            and top_p >= cfg.exact_hash_min_prob
            and margin >= cfg.exact_hash_min_margin
        ):
            return STATUS_EXACT_HASH_CANDIDATE
        if top_p >= cfg.concentrated_top_prob and entropy <= cfg.concentrated_max_entropy:
            return STATUS_CONCENTRATED
        if top_p >= cfg.likely_top_prob:
            return STATUS_LIKELY
        return STATUS_AMBIGUOUS


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalized_entropy(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if len(positive) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in positive)
    return h / math.log(len(positive))


def _credible_set(ordered: list[tuple[str, float]], mass: float) -> tuple[list[str], float]:
    chosen: list[str] = []
    acc = 0.0
    for leaf_id, p in ordered:
        chosen.append(leaf_id)
        acc += p
        if acc >= mass:
            break
    return chosen, acc
