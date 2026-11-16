"""Leakage and import-safety boundaries for the ablation-driven RL scaffold.

This module centralizes the *hard rules* the rest of `deepmew.rl` enforces so a
later experiment cannot silently smuggle a blocked label, a private/future
opponent field, or a runtime/submission edit into a training run:

- RL0 (`pure_masked_ppo`) consumes nothing but the legal-action mask, public
  current-observation features, and a terminal win/loss reward.
- Raw Phase 5 `label_candidates_v0` and the offline belief/prior packages are
  never imported on the RL training path. Later experiments may pull
  belief/support/prior signal in **only** through a registered `FeatureModule`
  whose schema and leakage boundary are declared and gated by experiment id.
- No component may read future/private opponent information.

Nothing here changes runtime/submission behavior; it is an offline guardrail.
"""

from __future__ import annotations

from typing import Iterable

# Offline label / belief / prior packages that must NOT be imported at module
# import time on the RL training path. Later experiments reach belief/support/
# prior signal only via a gated FeatureModule that imports lazily inside its
# own compute(), never at scaffold import time.
FORBIDDEN_OFFLINE_LABEL_MODULES: frozenset[str] = frozenset(
    {
        "deepmew.offline_label_draft",
        "deepmew.matchup_card_priors",
        "deepmew.opponent_belief",
        "deepmew.belief_aware_action_selector",
    }
)

# Raw Phase 5 artifact files that are matchup priors with known semantic
# inversions, never trusted state-action labels / Q-values (DEC-038/DEC-039).
UNSAFE_PHASE5_LABEL_ARTIFACTS: frozenset[str] = frozenset(
    {
        "label_candidates_v0.csv",
        "label_candidates_v0.jsonl",
    }
)

# Observation feature names that would constitute future/private opponent
# information leakage. The public base encoder never emits these, and a feature
# module that declares any of them as an input is rejected.
FORBIDDEN_OBSERVATION_INPUTS: frozenset[str] = frozenset(
    {
        "opponent_hand_cards",
        "opponent_hand_identities",
        "opponent_deck_order",
        "opponent_deck_cards",
        "opponent_prize_cards",
        "future_draw",
        "future_cards",
        "future_opponent_action",
        "private_opponent_state",
        "ground_truth_deck",
        "ground_truth_archetype",
        "final_result_lookahead",
        "hidden_opponent_state",
    }
)

# Feature-module / reward names that are forbidden in RL0 regardless of how an
# experiment is assembled.
RL0_FORBIDDEN_FEATURE_MODULES: frozenset[str] = frozenset(
    {"support_chain", "dsu_belief", "robust_posterior", "static_priors"}
)


class RLSafetyError(ValueError):
    """Raised when an experiment/feature configuration violates a hard rule."""


def assert_no_forbidden_inputs(name: str, inputs: Iterable[str]) -> None:
    """Reject any input that constitutes future/private opponent leakage."""
    bad = sorted(set(inputs) & FORBIDDEN_OBSERVATION_INPUTS)
    if bad:
        raise RLSafetyError(
            f"feature/component {name!r} declares forbidden private/future inputs: {bad}"
        )


def assert_rl0_is_pure(
    *,
    feature_modules: Iterable[str],
    dsu: bool,
    belief: bool,
    support_features: bool,
    static_priors: bool,
    reward: str,
    bc_init: bool,
) -> None:
    """RL0 is the masked-PPO control: no DSU/belief/support/static priors, no
    shaping beyond terminal win/loss + illegal/crash/timeout penalties, no warm
    start. Any deviation means the caller mislabeled a richer experiment as RL0.
    """
    enabled = sorted(set(feature_modules))
    if enabled:
        raise RLSafetyError(f"RL0 must enable zero feature modules; got {enabled}")
    flags = {
        "dsu": dsu,
        "belief": belief,
        "support_features": support_features,
        "static_priors": static_priors,
        "bc_init": bc_init,
    }
    on = sorted(name for name, value in flags.items() if value)
    if on:
        raise RLSafetyError(f"RL0 must disable {on}")
    if reward != "terminal_win_loss":
        raise RLSafetyError(
            f"RL0 reward must be terminal_win_loss (+ illegal/crash/timeout); got {reward!r}"
        )
