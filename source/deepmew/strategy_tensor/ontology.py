"""Evidence-backed diagnostic ontology for compact deck strategy profiles.

Labels are offline, same-frame pseudo-label aggregates. They describe observed
deck-plus-pilot behavior and are not policy-deconfounded traits. Exact deck
hashes are grouping/provenance keys only and never tensor dimensions.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from deepmew.artifacts.writers import write_json, write_jsonl, write_text


ONTOLOGY_SCHEMA = "deepmew.strategy_tensor_ontology.v1"
PROFILE_SCHEMA = "deepmew.strategy_deck_profile.v1"
MANIFEST_SCHEMA = "deepmew.run_manifest.v1"
MAIN_CONTEXT = "MAIN"
MIN_MAIN_DECISIONS = 50
MIN_EPISODE_SEATS = 5
MIN_POSITIVE_SLOT_ROWS = 5

WIN_GOAL_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("win_setup_sweeper", ("setup_sweeper",)),
    ("win_spread_damage", ("spread_damage",)),
    ("win_tempo_pressure", ("tempo_pressure",)),
    ("win_other_or_unknown", ("engine_setup", "unknown")),
)

STYLE_LABELS: tuple[str, ...] = (
    "consistency_find_piece",
    "board_development",
    "attacker_readiness",
    "damage_setup",
    "resource_preservation",
    "deny_punish",
    "stall_extend_game",
    "reduce_punishability",
    "lethal_or_forced_win",
    "disruption",
    "prize_conversion",
    "forced_defense",
)

LOSS_LABELS: tuple[str, ...] = (
    "clock_or_sequence_punish",
    "prize_punish",
    "deck_plan_punish",
    "tempo_lock_punish",
)


@dataclass
class EpisodeSeatProfile:
    deck_hash: str
    goal_of_deck: str
    main_decisions: int = 0
    play_counts: Counter[str] = field(default_factory=Counter)
    punishment_counts: Counter[str] = field(default_factory=Counter)
    known_pilot_rows: int = 0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _slot_templates() -> list[dict[str, Any]]:
    slots = []
    for slot_id, source_labels in WIN_GOAL_SLOTS:
        slots.append({
            "slot_id": slot_id,
            "group": "win",
            "source_labels": list(source_labels),
            "definition": "One-hot coarse deck-plan context from the deterministic goal_of_deck mapping.",
            "label_source": {
                "field": "goal_of_deck",
                "producer": "deepmew/imitation_fine/archetype_goals.py",
                "aggregation": "constant per deck; one-hot",
            },
            "ambiguous_cases": [
                "engine_setup and unknown are pooled in win_other_or_unknown",
                "the source is archetype logic rather than observed causal wins",
            ],
            "overlap_with": [],
            "expected_strategic_meaning": "Coarse context for how the list is expected to establish pressure.",
            "known_limitations": ["not a causal win-condition label", "not policy-deconfounded"],
        })
    for label in STYLE_LABELS:
        slots.append({
            "slot_id": f"style_{label}",
            "group": "style",
            "source_labels": [label],
            "definition": "Mean episode-seat fraction of MAIN decisions assigned this action-intent pseudo-label.",
            "label_source": {
                "field": "goal_of_play",
                "producer": "deepmew/imitation_fine/labelers.py:derive_goal_of_play",
                "aggregation": "MAIN-only; normalize within episode-seat, then equal-weight episode-seats",
            },
            "ambiguous_cases": [
                "unknown intent mass is excluded and reported as uncertainty",
                "reached states and opponent mixture affect the observed rate",
            ],
            "overlap_with": [],
            "expected_strategic_meaning": "Observed tendency to spend decisions on this tactical intent.",
            "known_limitations": [
                "same-frame deterministic pseudo-label",
                "deck, pilot, matchup, and reached-state effects are confounded",
            ],
        })
    for label in LOSS_LABELS:
        slots.append({
            "slot_id": f"loss_{label.removesuffix('_punish')}_exposure",
            "group": "loss",
            "source_labels": [label],
            "definition": "Mean episode-seat fraction of MAIN decisions exposing this punishment surface.",
            "label_source": {
                "field": "punishment_labels",
                "producer": "deepmew/imitation_fine/labelers.py:derive_punishment",
                "aggregation": "MAIN-only; normalize within episode-seat, then equal-weight episode-seats",
            },
            "ambiguous_cases": [
                "exposure is observational and is not a proven cause of loss",
                "multiple punishment labels may be present on one decision",
            ],
            "overlap_with": [],
            "expected_strategic_meaning": "Observed vulnerability to this loss-of-agency surface.",
            "known_limitations": ["not keyed to final outcomes", "not policy-deconfounded"],
        })
    if len(slots) != 20:
        raise AssertionError(f"expected 20 semantic slots, found {len(slots)}")
    for index, slot in enumerate(slots):
        slot["index"] = index

    return slots


def _semantic_values(
    *,
    goal_of_deck: str,
    episode_profiles: list[EpisodeSeatProfile],
    slots: list[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    values: dict[str, float] = {}
    occurrences: dict[str, int] = {}
    for slot_id, goals in WIN_GOAL_SLOTS:
        values[slot_id] = float(goal_of_deck in goals)
        occurrences[slot_id] = (
            sum(profile.main_decisions for profile in episode_profiles)
            if values[slot_id]
            else 0
        )
    for label in STYLE_LABELS:
        slot_id = f"style_{label}"
        rates = [profile.play_counts[label] / profile.main_decisions for profile in episode_profiles]
        values[slot_id] = float(np.mean(rates))
        occurrences[slot_id] = sum(profile.play_counts[label] for profile in episode_profiles)
    for label in LOSS_LABELS:
        slot_id = f"loss_{label.removesuffix('_punish')}_exposure"
        rates = [profile.punishment_counts[label] / profile.main_decisions for profile in episode_profiles]
        values[slot_id] = float(np.mean(rates))
        occurrences[slot_id] = sum(profile.punishment_counts[label] for profile in episode_profiles)

    episode_vectors = []
    for profile in episode_profiles:
        vector = [profile.play_counts[label] / profile.main_decisions for label in STYLE_LABELS]
        vector += [profile.punishment_counts[label] / profile.main_decisions for label in LOSS_LABELS]
        episode_vectors.append(vector)
    episode_variance = np.var(np.asarray(episode_vectors, dtype=np.float64), axis=0)
    unknown_rates = [profile.play_counts["unknown"] / profile.main_decisions for profile in episode_profiles]
    known_pilot_rows = sum(profile.known_pilot_rows for profile in episode_profiles)
    main_decisions = sum(profile.main_decisions for profile in episode_profiles)
    uncertainty = {
        "unknown_intent_mass": float(np.mean(unknown_rates)),
        "episode_seat_disagreement": float(np.mean(np.clip(4.0 * episode_variance, 0.0, 1.0))),
        "pilot_attribution_unknown_mass": 1.0 - (known_pilot_rows / main_decisions),
    }
    if set(values) != {slot["slot_id"] for slot in slots}:
        raise AssertionError("semantic profile does not match ontology")

    return values, occurrences, uncertainty


def aggregate_strategy_profiles(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slots = _slot_templates()
    episode_seats: dict[tuple[str, int], EpisodeSeatProfile] = {}
    all_decks: set[str] = set()
    pilot_ids_by_deck: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    main_rows = 0
    known_pilot_rows = 0
    for row in rows:
        total_rows += 1
        deck_hash = str(row.get("deck_hash") or "")
        if not deck_hash:
            raise ValueError("strategy profile row is missing deck_hash")
        all_decks.add(deck_hash)
        pilot_id = str(row.get("pilot_team_id") or "")
        if pilot_id:
            known_pilot_rows += 1
            pilot_ids_by_deck[deck_hash].add(pilot_id)
        if row.get("select_context") != MAIN_CONTEXT:
            continue
        main_rows += 1
        key = (str(row.get("episode_id") or ""), int(row.get("seat", -1)))
        if not key[0] or key[1] not in {0, 1}:
            raise ValueError("MAIN strategy row has invalid episode-seat identity")
        goal = str(row.get("goal_of_deck") or "unknown")
        profile = episode_seats.get(key)
        if profile is None:
            profile = EpisodeSeatProfile(deck_hash=deck_hash, goal_of_deck=goal)
            episode_seats[key] = profile
        if profile.deck_hash != deck_hash or profile.goal_of_deck != goal:
            raise ValueError(f"episode-seat identity drift for {key}")
        profile.main_decisions += 1
        profile.play_counts[str(row.get("goal_of_play") or "unknown")] += 1
        for label in set(str(value) for value in row.get("punishment_labels") or []):
            profile.punishment_counts[label] += 1
        if pilot_id:
            profile.known_pilot_rows += 1

    episodes_by_deck: dict[str, list[EpisodeSeatProfile]] = defaultdict(list)
    for profile in episode_seats.values():
        episodes_by_deck[profile.deck_hash].append(profile)

    profiles = []
    for deck_hash in sorted(all_decks):
        episode_profiles = episodes_by_deck.get(deck_hash, [])
        if not episode_profiles:
            profiles.append({
                "schema_version": PROFILE_SCHEMA,
                "deck_hash": deck_hash,
                "support": {
                    "status": "unknown_low_support",
                    "main_decisions": 0,
                    "episode_seats": 0,
                    "support_mask": [0] * len(slots),
                },
                "semantic_tensor": None,
                "diagnostic_tensor": None,
                "uncertainty": {
                    "unknown_intent_mass": 1.0,
                    "episode_seat_disagreement": 1.0,
                    "pilot_attribution_unknown_mass": 1.0,
                },
            })
            continue
        goals = {profile.goal_of_deck for profile in episode_profiles}
        if len(goals) != 1:
            raise ValueError(f"goal_of_deck is not constant for {deck_hash}: {sorted(goals)}")
        main_decisions = sum(profile.main_decisions for profile in episode_profiles)
        is_supported = (
            main_decisions >= MIN_MAIN_DECISIONS
            and len(episode_profiles) >= MIN_EPISODE_SEATS
        )
        goal = next(iter(goals))
        values, occurrences, uncertainty = _semantic_values(
            goal_of_deck=goal,
            episode_profiles=episode_profiles,
            slots=slots,
        )
        vector = [round(values[slot["slot_id"]], 9) for slot in slots]
        profiles.append({
            "schema_version": PROFILE_SCHEMA,
            "deck_hash": deck_hash,
            "goal_of_deck": goal,
            "support": {
                "status": "supported_diagnostic" if is_supported else "unknown_low_support",
                "main_decisions": main_decisions,
                "episode_seats": len(episode_profiles),
                "support_mask": [1 if is_supported else 0] * len(slots),
            },
            "semantic_tensor": vector if is_supported else None,
            "diagnostic_tensor": vector,
            "slot_occurrences": occurrences,
            "uncertainty": {key: round(value, 9) for key, value in uncertainty.items()},
        })

    supported = [p for p in profiles if p["support"]["status"] == "supported_diagnostic"]
    summary = {
        "total_rows": total_rows,
        "main_rows": main_rows,
        "decks": len(all_decks),
        "episode_seats": len(episode_seats),
        "supported_decks": len(supported),
        "supported_deck_share": round(len(supported) / len(all_decks), 9),
        "support_floor": {
            "minimum_main_decisions": MIN_MAIN_DECISIONS,
            "minimum_episode_seats": MIN_EPISODE_SEATS,
        },
        "pilot_identity": {
            "known_rows": known_pilot_rows,
            "known_row_share": round(known_pilot_rows / total_rows, 9),
            "decks_with_identified_pilot": sum(bool(ids) for ids in pilot_ids_by_deck.values()),
            "decks_with_at_least_two_identified_pilots": sum(len(ids) >= 2 for ids in pilot_ids_by_deck.values()),
            "decks_with_at_least_three_identified_pilots": sum(len(ids) >= 3 for ids in pilot_ids_by_deck.values()),
        },
    }

    return profiles, summary


def _ontology_with_support(
    profiles: list[Mapping[str, Any]],
    input_path: str | Path,
) -> dict[str, Any]:
    slots = _slot_templates()
    supported = [p for p in profiles if p["support"]["status"] == "supported_diagnostic"]
    for slot in slots:
        index = int(slot["index"])
        values = [float(profile["semantic_tensor"][index]) for profile in supported]
        positive_support = sum(
            int(profile.get("slot_occurrences", {}).get(slot["slot_id"], 0))
            >= MIN_POSITIVE_SLOT_ROWS
            for profile in supported
        )
        slot["support"] = {
            "eligible_decks": len(supported),
            "decks_with_at_least_five_positive_rows": positive_support,
            "mean_deck_intensity": round(float(np.mean(values)), 9),
            "median_deck_intensity": round(float(np.median(values)), 9),
            "nonzero_decks": sum(value > 0.0 for value in values),
        }

    return {
        "schema_version": ONTOLOGY_SCHEMA,
        "status": "diagnostic_only",
        "semantic_slot_count": len(slots),
        "tensor_layout": {
            "z_win": [0, 4],
            "z_style": [4, 16],
            "z_loss": [16, 20],
            "uncertainty": [
                "unknown_intent_mass",
                "episode_seat_disagreement",
                "pilot_attribution_unknown_mass",
                "classifier_seed_fold_disagreement",
                "public_posterior_entropy",
                "public_unknown_mass",
            ],
        },
        "aggregation_contract": {
            "row_filter": "select_context == MAIN",
            "within_episode_seat": "normalize label counts by MAIN decision count",
            "across_episode_seats": "equal-weight mean; long games do not dominate",
            "support_floor": {
                "minimum_main_decisions": MIN_MAIN_DECISIONS,
                "minimum_episode_seats": MIN_EPISODE_SEATS,
            },
            "low_support_emission": "semantic_tensor=null and all-zero support_mask",
            "forbidden_inputs": [
                "won",
                "result",
                "teacher_weight",
                "future action rows",
                "true opponent deck/archetype",
                "opponent hand/prizes/deck order",
            ],
        },
        "label_artifact": str(Path(input_path)),
        "slots": slots,
        "known_limitations": [
            "No policy identifier exists in the K4.1 example schema.",
            "Deck rates mix deck, pilot, matchup, and reached-state effects.",
            "Loss slots are observational punishment exposure, not causal loss labels.",
            "Public action history is not yet consumed by the DSU posterior.",
        ],
    }


def implementation_map() -> list[dict[str, Any]]:
    return [
        {"surface": "DSU card-family components", "status": "implemented", "evidence": ["deepmew/meta_kg/anchors.py", "deepmew/meta_kg/classify.py", ".research/DSU_REGISTRY.json"], "boundary": "offline/training-only"},
        {"surface": "full-deck archetype classifier", "status": "implemented", "evidence": ["deepmew/meta_kg/classify.py", ".research/runs/meta_kg/20260707/20260707T054219Z-meta_kg-r16/deck_archetypes.csv"], "boundary": "classification metadata; not a strategy tensor"},
        {"surface": "support-package classifier", "status": "partial", "evidence": ["deepmew/meta_kg/packages.py"], "boundary": "three heuristic types; no calibrated posterior, conflict, or unknown mass"},
        {"surface": "win-condition labels", "status": "partial", "evidence": ["deepmew/imitation_fine/archetype_goals.py"], "boundary": "four coarse deterministic archetype goals; not causal outcomes"},
        {"surface": "loss-condition labels", "status": "partial", "evidence": ["deepmew/imitation_fine/labelers.py:derive_punishment"], "boundary": "same-frame observational exposure; two labels are too sparse for core slots"},
        {"surface": "persistent playstyle labels", "status": "missing", "evidence": ["deepmew/imitation_fine/labelers.py:derive_goal_of_play"], "boundary": "action-intent rates exist; validated agent traits do not"},
        {"surface": "strict-public posterior updates", "status": "partial", "evidence": ["deepmew/meta_kg/live_belief.py:LiveArchetypeBelief.update", "deepmew/meta_kg/belief_evaluation.py:extract_public_reveals"], "boundary": "revealed cards only; public actions are not posterior evidence"},
        {"surface": "public reveal and action streams", "status": "partial", "evidence": ["deepmew/meta_kg/belief_evaluation.py:extract_public_reveals", "deepmew/replay/prospective_trace.py"], "boundary": "strict-public card prefixes exist; action-conditioned belief adapter is missing"},
        {"surface": "replay calibration", "status": "implemented", "evidence": ["deepmew/meta_kg/belief_evaluation.py", ".research/dsu_belief_no_staple_correction_v0/result.json"], "boundary": "diagnostic-only; universal 0.75 no-staple scale is quarantined"},
        {"surface": "policy-family metadata", "status": "partial", "evidence": ["deepmew/curriculum/league_registry.py", "deepmew/imitation_fine/schemas.py:FineExample"], "boundary": "controlled league metadata exists; K4.1 examples carry no policy identifier"},
        {"surface": "empirical strength buckets", "status": "partial", "evidence": ["deepmew/curriculum/ladder_metadata.py"], "boundary": "controlled/replay metadata only; no validated inference from behavior"},
        {"surface": "true opponent hidden-state strategy inference", "status": "invalid_private_information_dependent", "evidence": ["AGENTS.md protected actor-visibility boundary"], "boundary": "opponent hand, prizes, deck order, and private legal options are forbidden"},
    ]


def _load_full_list_contract(path: str | Path) -> tuple[set[str], dict[str, int]]:
    hashes = set()
    splits = Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            card_counts = {str(key): int(value) for key, value in row["card_counts"].items()}
            if sum(card_counts.values()) != 60:
                raise ValueError(f"full-list row is not 60 cards: {row.get('deck_hash')}")
            deck_hash = str(row["deck_hash"])
            if deck_hash in hashes:
                raise ValueError(f"duplicate canonical full-list hash: {deck_hash}")
            hashes.add(deck_hash)
            splits[str(row["split"])] += 1

    return hashes, dict(sorted(splits.items()))


def render_report(result: Mapping[str, Any], ontology: Mapping[str, Any]) -> str:
    summary = result["profile_support"]
    join = result["full_list_join"]
    lines = [
        "# Strategy tensor ontology v1",
        "",
        "Verdict: **DIAGNOSTIC_ONTOLOGY_DEFINED; POLICY_STYLE_UNRESOLVED**.",
        "",
        f"The live K4.1 artifact contains {summary['total_rows']:,} decisions, {summary['main_rows']:,} MAIN decisions, and {summary['decks']:,} complete deck identities. Only {summary['supported_decks']:,} decks ({summary['supported_deck_share']:.1%}) meet the {MIN_MAIN_DECISIONS}-MAIN-decision/{MIN_EPISODE_SEATS}-episode-seat floor.",
        "",
        f"The pinned full-list contract has {join['full_list_decks']:,} canonical 60-card lists; {join['supported_profile_overlap']:,} supported behavior profiles have a canonical full list.",
        "",
        "## Ontology",
        "",
        "The tensor has 20 semantic slots: 4 win-context, 12 observed action-intent, and 4 punishment-exposure slots. Uncertainty is carried separately.",
        "",
        "| # | Slot | Group | Positive-support decks | Mean intensity |",
        "|---:|---|---|---:|---:|",
    ]
    for slot in ontology["slots"]:
        support = slot["support"]
        lines.append(f"| {slot['index']} | `{slot['slot_id']}` | {slot['group']} | {support['decks_with_at_least_five_positive_rows']:,} | {support['mean_deck_intensity']:.4f} |")
    lines += [
        "",
        "## Scientific boundary",
        "",
        "- MAIN sub-selections are excluded; labels are normalized inside each episode-seat before deck aggregation.",
        "- Low-support decks emit a null semantic tensor and an all-zero support mask.",
        "- Outcome, teacher weight, future actions, true opponent deck, and private zones are forbidden inputs.",
        "- K4.1 has no policy identifier. These rates are deck-plus-pilot-plus-matchup diagnostics, not persistent agent style.",
        "- The next gate is independent full-list classifier evaluation. No tensor is authorized for V, response, Q, league, runtime, or submissions yet.",
        "",
        "## Implementation map",
        "",
        "| Surface | Status | Boundary |",
        "|---|---|---|",
    ]
    for row in result["implementation_map"]:
        lines.append(f"| {row['surface']} | `{row['status']}` | {row['boundary']} |")
    lines.append("")

    return "\n".join(lines)


def build_strategy_ontology(
    *,
    examples_path: str | Path,
    full_lists_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    examples_path = Path(examples_path)
    full_lists_path = Path(full_lists_path)
    with examples_path.open("rb") as handle:
        profiles, support_summary = aggregate_strategy_profiles(
            json.loads(raw_line) for raw_line in handle
        )
    ontology = _ontology_with_support(profiles, examples_path)
    full_list_hashes, full_list_splits = _load_full_list_contract(full_lists_path)
    supported_hashes = {
        p["deck_hash"] for p in profiles if p["support"]["status"] == "supported_diagnostic"
    }
    result = {
        "schema_version": "deepmew.strategy_ontology_build.v1",
        "status": "completed_diagnostic_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "DIAGNOSTIC_ONTOLOGY_DEFINED_POLICY_STYLE_UNRESOLVED",
        "profile_support": support_summary,
        "full_list_join": {
            "full_list_decks": len(full_list_hashes),
            "full_list_splits": full_list_splits,
            "supported_profile_overlap": len(supported_hashes & full_list_hashes),
            "supported_profiles_without_canonical_full_list": len(supported_hashes - full_list_hashes),
        },
        "implementation_map": implementation_map(),
        "information_mode": "offline_actor_visible_labels; strict-public opponent inference not yet built",
        "promotion_status": "diagnostic_only_no_control_authority",
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profiles": output_dir / "deck_profiles.jsonl",
        "ontology": output_dir / "ontology.json",
        "implementation_map": output_dir / "implementation_map.json",
        "result": output_dir / "result.json",
        "report": output_dir / "REPORT.md",
    }
    write_jsonl(paths["profiles"], profiles)
    write_json(paths["ontology"], ontology)
    write_json(paths["implementation_map"], result["implementation_map"])
    write_json(paths["result"], result)
    write_text(paths["report"], render_report(result, ontology))
    outputs = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": "strategy-ontology-v1-20260711",
        "kind": "strategy_tensor_ontology",
        "status": "completed_diagnostic_only",
        "created_at_utc": result["created_at_utc"],
        "inputs": {
            "examples_path": str(examples_path),
            "examples_sha256": _sha256(examples_path),
            "full_lists_path": str(full_lists_path),
            "full_lists_sha256": _sha256(full_lists_path),
            "implementation_sha256": _sha256(Path(__file__)),
            "entrypoint_sha256": _sha256(Path(__file__).resolve().parents[2] / "scripts" / "build_strategy_ontology.py"),
        },
        "outputs": outputs,
        "summary": {
            "ontology_version": ONTOLOGY_SCHEMA,
            "ontology_sha256": outputs["ontology"]["sha256"],
            "semantic_slots": len(ontology["slots"]),
            "supported_decks": support_summary["supported_decks"],
            "full_list_overlap": result["full_list_join"]["supported_profile_overlap"],
            "promotion_status": result["promotion_status"],
        },
        "protected_surfaces_changed": [],
    }
    write_json(output_dir / "manifest.json", manifest)

    return result
