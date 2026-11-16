"""Typed domain objects and build configuration for the K2A meta knowledge graph.

K2A converts mined Kaggle ladder deck artifacts into a machine-readable meta map.
Card IDs are canonical (strings here for CSV/JSON stability); names are display
enrichment only. This module is offline/training-only and must never import the
battle engine (``cg``) or the submission entrypoint (``main``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CardId = str
DeckHash = str

ARTIFACT_VERSION = "deepmew.meta_kg.v2"
META_KG_KIND = "meta_kg"

# Output filenames (stable contract consumed by K2B/K3/K4/K5/K6).
OUTPUT_FILES = {
    "card_catalog": "card_catalog.csv",
    "deck_archetypes": "deck_archetypes.csv",
    "archetype_summary": "archetype_summary.csv",
    "support_packages": "support_packages.csv",
    "staples": "staples.csv",
    "card_lift_by_archetype": "card_lift_by_archetype.csv",
    "deck_similarity": "deck_similarity.csv",
    "deck_meta_report": "deck_meta_report.md",
    "live_belief": "live_belief.json",
}

UNKNOWN_ARCHETYPE = "Unknown"
MIXED_ARCHETYPE = "Mixed"

# Classification status vocabulary (kept stable for downstream consumers).
STATUS_CONFIDENT = "confident"
STATUS_MIXED = "mixed"
STATUS_ENGINE_FIRST = "engine_first"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeckArtifact:
    """One observed deck payload from one side of one replay episode."""

    episode_id: str
    cards: dict[CardId, int]
    names: dict[CardId, str] = field(default_factory=dict)
    deck_hash: DeckHash | None = None
    player_id: str | None = None
    team_id: str | None = None
    side: str | None = None
    rank: int | None = None
    rank_bucket: str | None = None
    source_path: str | None = None


@dataclass
class UniqueDeck:
    """Aggregated representative deck for one deck hash."""

    deck_hash: DeckHash
    cards: dict[CardId, int]
    names: dict[CardId, str]
    artifact_count: int
    episode_ids: list[str]
    player_ids: list[str]
    team_ids: list[str]
    rank_buckets: dict[str, int] = field(default_factory=dict)

    @property
    def episode_count(self) -> int:
        return len(set(self.episode_ids))


@dataclass
class CardCatalogRow:
    card_id: CardId
    card_name: str
    copies_total: int
    deck_count: int
    artifact_count: int
    episode_count: int
    player_team_count: int
    global_deck_freq: float
    global_artifact_freq: float
    avg_copies_when_present: float
    name_missing: bool
    metadata_status: str = "payload_only"
    type_guess: str = "unknown"
    stage_guess: str = "unknown"


@dataclass(frozen=True)
class AnchorDefinition:
    anchor_id: str
    label: str
    anchor_type: str  # pokemon_core, engine, manual, auto, unknown
    terminal_card_ids: tuple[CardId, ...] = ()
    line_card_ids: tuple[CardId, ...] = ()
    support_card_ids: tuple[CardId, ...] = ()
    engine_card_ids: tuple[CardId, ...] = ()
    shared_line_group_id: str | None = None
    branch_id: str | None = None
    priority: int = 0
    source: str = "config"
    # Win-condition grade: ex / Mega / Stage2 terminals get full terminal weight;
    # non-ex Stage1/Basic support Pokemon (Hariyama, Dudunsparce) are reduced so
    # they cannot out-score or tie a real core. Manual anchors default to True.
    terminal_is_wincon: bool = True

    @property
    def is_engine(self) -> bool:
        return self.anchor_type == "engine"


@dataclass
class AnchorScore:
    anchor_id: str
    label: str
    anchor_type: str
    score: float
    terminal_score: float
    line_score: float
    support_score: float
    copy_score: float
    staple_penalty: float
    evidence_card_ids: list[CardId]
    priority: int = 0
    terminal_is_wincon: bool = True


@dataclass
class ArchetypeAssignment:
    deck_hash: DeckHash
    primary_archetype: str
    secondary_package: str | None
    variant_label: str
    confidence: float
    classification_status: str
    core_cards: list[CardId]
    package_cards: list[CardId]
    staple_cards: list[CardId]
    tech_cards: list[CardId]
    representative_team_ids: list[str]
    representative_player_ids: list[str]
    representative_episode_ids: list[str]
    artifact_count: int
    episode_count: int
    rank_bucket_summary: dict[str, int]
    anchor_scores: dict[str, float]
    notes: str


@dataclass
class BuildWarning:
    code: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)

    def bump(self, example: str | None = None) -> None:
        self.count += 1
        if example is not None and len(self.examples) < 10 and example not in self.examples:
            self.examples.append(example)


@dataclass
class K2ABuildConfig:
    # Classification weights / thresholds.
    primary_threshold: float = 3.0
    engine_primary_threshold: float = 3.25
    secondary_threshold: float = 1.8
    secondary_ratio: float = 0.45
    mixed_ratio: float = 0.85
    terminal_weight: float = 5.0
    line_weight: float = 1.0
    support_weight: float = 1.25
    engine_weight: float = 5.0
    copy_weight: float = 0.25
    max_copy_bonus: float = 1.0
    confident_confidence_threshold: float = 0.50
    # Terminal-weight multiplier for non-win-condition terminals (non-ex Stage1/Basic
    # support Pokemon). Keeps support attackers from tying real cores.
    nonwincon_terminal_scale: float = 0.5

    # Staple detection.
    staple_min_global_deck_freq: float = 0.18
    staple_max_lift: float = 1.75
    staple_min_entropy_norm: float = 0.55
    staple_downweight_min: float = 0.05
    staple_downweight_max: float = 0.35
    staple_terminal_exclusion: bool = True

    # Lift / packages.
    lift_alpha: float = 0.5
    min_report_lift: float = 1.25
    min_report_conditional_freq: float = 0.10
    tech_max_conditional_freq: float = 0.35
    tech_min_lift: float = 2.0
    signature_min_conditional_freq: float = 0.45
    signature_min_lift: float = 1.5
    package_min_lift: float = 1.5
    package_min_conditional_freq: float = 0.25
    package_max_cards: int = 12

    # Similarity.
    similarity_top_k: int = 20
    similarity_include_raw_jaccard: bool = True
    similarity_staple_downweight: bool = True

    # Auto anchors.
    auto_anchor_min_deck_count: int = 5
    auto_anchor_enabled: bool = True
    fallback_anchor_min_deck_count: int = 5
    # A non-win-condition Pokemon that appears as a card in this many or more
    # distinct reliable-core archetypes is a cross-deck engine/splash (e.g.
    # Dudunsparce, Fezandipiti ex) and is never treated as an archetype identity.
    splash_min_archetypes: int = 4

    # Optional engine card-metadata enrichment (Pokemon/Trainer/Energy, stage,
    # evolution). Degrades to name heuristics if the engine is unavailable.
    use_engine_card_metadata: bool = True

    # Write-time caps.
    representative_cap: int = 10

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "K2ABuildConfig":
        """Build a config from a (possibly nested) YAML mapping, ignoring unknown keys."""
        if not data:
            return cls()
        flat: dict[str, Any] = {}
        classification = data.get("classification", {}) or {}
        staples = data.get("staples", {}) or {}
        lift = data.get("lift", {}) or {}
        similarity = data.get("similarity", {}) or {}
        auto = data.get("auto_anchors", {}) or {}

        for key in (
            "primary_threshold", "engine_primary_threshold", "secondary_threshold",
            "secondary_ratio", "mixed_ratio", "terminal_weight", "line_weight",
            "support_weight", "engine_weight", "copy_weight", "max_copy_bonus",
            "confident_confidence_threshold", "nonwincon_terminal_scale",
        ):
            if key in classification:
                flat[key] = classification[key]

        staple_map = {
            "min_global_deck_freq": "staple_min_global_deck_freq",
            "max_lift_for_staple": "staple_max_lift",
            "min_archetype_entropy_norm": "staple_min_entropy_norm",
            "downweight_min": "staple_downweight_min",
            "downweight_max": "staple_downweight_max",
            "terminal_card_exclusion": "staple_terminal_exclusion",
        }
        for src, dst in staple_map.items():
            if src in staples:
                flat[dst] = staples[src]

        lift_map = {
            "alpha": "lift_alpha",
            "min_report_lift": "min_report_lift",
            "min_report_conditional_freq": "min_report_conditional_freq",
            "tech_max_conditional_freq": "tech_max_conditional_freq",
            "tech_min_lift": "tech_min_lift",
            "signature_min_conditional_freq": "signature_min_conditional_freq",
            "signature_min_lift": "signature_min_lift",
            "package_min_lift": "package_min_lift",
            "package_min_conditional_freq": "package_min_conditional_freq",
            "package_max_cards": "package_max_cards",
        }
        for src, dst in lift_map.items():
            if src in lift:
                flat[dst] = lift[src]

        sim_map = {
            "top_k": "similarity_top_k",
            "include_raw_jaccard": "similarity_include_raw_jaccard",
            "staple_downweight": "similarity_staple_downweight",
        }
        for src, dst in sim_map.items():
            if src in similarity:
                flat[dst] = similarity[src]

        if "min_deck_count" in auto:
            flat["auto_anchor_min_deck_count"] = auto["min_deck_count"]
        if "enabled" in auto:
            flat["auto_anchor_enabled"] = auto["enabled"]
        if "fallback_min_deck_count" in auto:
            flat["fallback_anchor_min_deck_count"] = auto["fallback_min_deck_count"]
        if "splash_min_archetypes" in auto:
            flat["splash_min_archetypes"] = auto["splash_min_archetypes"]
        if "representative_cap" in data:
            flat["representative_cap"] = data["representative_cap"]
        if "use_engine_card_metadata" in data:
            flat["use_engine_card_metadata"] = data["use_engine_card_metadata"]

        return cls(**flat)
