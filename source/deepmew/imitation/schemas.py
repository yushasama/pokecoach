"""K4 imitation dataset schemas and config.

Builds offline behavior-cloning examples from real ladder action rows joined to
K2A archetype labels and K3 deck-zoo splits. The chosen action *category*
(OptionType) is the supervised label. Engine enum maps are embedded statically so
this layer works without importing the engine, and is never on the eval path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_VERSION = "deepmew.imitation_dataset.v1"
IMITATION_KIND = "imitation_dataset"

OUTPUT_FILES = {
    "examples": "imitation_examples.jsonl",
    "dataset_summary": "imitation_dataset_summary.csv",
    "bucket_summary": "imitation_bucket_summary.csv",
    "baseline_eval": "imitation_baseline_eval.csv",
    "report": "imitation_dataset_report.md",
}

# cg.api.OptionType (value -> name); embedded so the dataset is auditable offline.
OPTION_TYPE_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD",
    6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}

# cg.api.SelectContext (value -> name).
SELECT_CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE_POKEMON", 2: "SETUP_BENCH_POKEMON", 3: "SWITCH",
    4: "TO_ACTIVE", 5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD",
    9: "TO_DECK", 10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE",
    13: "DAMAGE_COUNTER", 14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER", 17: "HEAL", 18: "EVOLVES_FROM", 19: "EVOLVES_TO",
    20: "DEVOLVE", 21: "ATTACH_FROM", 22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK",
    25: "EFFECT_TARGET", 26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD", 29: "DISCARD_CARD_OR_ATTACHED_CARD", 30: "DISCARD_ENERGY",
    35: "ATTACK", 37: "EVOLVE", 41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
}

# The strategic, policy-relevant decision context: the MAIN turn phase.
MAIN_CONTEXT_ID = 0


def option_type_name(value: object) -> str:
    try:
        return OPTION_TYPE_NAMES.get(int(value), f"TYPE_{value}")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "TYPE_UNKNOWN"


def select_context_name(value: object) -> str:
    try:
        return SELECT_CONTEXT_NAMES.get(int(value), f"CTX_{value}")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "CTX_UNKNOWN"


@dataclass
class ImitationConfig:
    # Behavior-cloning baseline trains on this context only (the strategic phase).
    main_context_only_for_baseline: bool = True
    # Minimum observations for a conditioned cell to be trusted (else back off).
    min_cell_count: int = 20
    # Confidence buckets for labels.
    confidence_high: float = 0.66
    confidence_low: float = 0.33
    # Cap representative example ids retained per bucket in summaries.
    bucket_example_cap: int = 5


@dataclass
class ImitationExample:
    episode_id: str
    seat: int
    turn: int
    step: int
    deck_hash: str
    archetype: str
    variant_label: str
    confidence: float
    confidence_bucket: str
    classification_status: str
    opponent_deck_hash: str
    opponent_archetype: str
    split: str
    select_context_id: int
    select_context: str
    n_legal_options: int
    legal_option_types: list[str]
    chosen_option_type: str
    chosen_option_types: list[str]
    result: int
    won: bool
    is_draw: bool
    source_run_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id, "seat": self.seat, "turn": self.turn, "step": self.step,
            "deck_hash": self.deck_hash, "archetype": self.archetype, "variant_label": self.variant_label,
            "confidence": self.confidence, "confidence_bucket": self.confidence_bucket,
            "classification_status": self.classification_status,
            "opponent_deck_hash": self.opponent_deck_hash, "opponent_archetype": self.opponent_archetype,
            "split": self.split, "select_context_id": self.select_context_id, "select_context": self.select_context,
            "n_legal_options": self.n_legal_options, "legal_option_types": self.legal_option_types,
            "chosen_option_type": self.chosen_option_type, "chosen_option_types": self.chosen_option_types,
            "result": self.result, "won": self.won, "is_draw": self.is_draw,
            "source_run_id": self.source_run_id,
        }
