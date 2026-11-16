"""K2A meta knowledge graph builder orchestration + CLI.

Two-pass classification: anchors + preliminary staples, then lift-derived final
staples and re-classification, support packages, variants, similarity, live
belief, report, and a versioned run manifest. Deterministic for fixed
input/config.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_markdown
from deepmew.meta_kg import anchors as anchors_mod
from deepmew.meta_kg import card_metadata
from deepmew.meta_kg import catalog as catalog_mod
from deepmew.meta_kg import classify as classify_mod
from deepmew.meta_kg import features as features_mod
from deepmew.meta_kg import io as io_mod
from deepmew.meta_kg import lift as lift_mod
from deepmew.meta_kg import packages as packages_mod
from deepmew.meta_kg import similarity as similarity_mod
from deepmew.meta_kg import staples as staples_mod
from deepmew.meta_kg.live_belief import build_live_belief_payload
from deepmew.meta_kg.report import render_report
from deepmew.meta_kg.schemas import (
    ARTIFACT_VERSION,
    ArchetypeAssignment,
    BuildWarning,
    CardId,
    K2ABuildConfig,
    META_KG_KIND,
    OUTPUT_FILES,
    STATUS_ENGINE_FIRST,
    STATUS_LOW_CONFIDENCE,
    STATUS_MIXED,
    STATUS_UNKNOWN,
    UniqueDeck,
)
from deepmew.meta_kg.catalog import CATALOG_FIELDS, catalog_row_to_dict

DECK_ARCHETYPE_FIELDS = [
    "deck_hash", "primary_archetype", "secondary_package", "variant_label", "confidence",
    "classification_status", "core_cards", "package_cards", "staple_cards", "tech_cards",
    "representative_team_ids", "representative_player_ids", "representative_episode_ids",
    "artifact_count", "episode_count", "rank_bucket_summary", "anchor_scores", "notes",
]
ARCHETYPE_SUMMARY_FIELDS = [
    "archetype", "deck_count", "artifact_count", "episode_count", "player_team_count",
    "meta_share_by_deck", "meta_share_by_artifact", "avg_confidence", "top_core_cards",
    "top_support_cards", "top_variants", "top_staples", "top_tech_cards",
    "representative_deck_hashes", "representative_episode_ids",
]
SUPPORT_PACKAGE_FIELDS = [
    "package_id", "package_name", "owner_archetype", "package_type", "card_ids", "card_names",
    "deck_count", "artifact_count", "conditional_freq", "global_freq", "lift", "cohesion_score",
    "package_score", "representative_deck_hashes", "notes",
]
STAPLE_FIELDS = [
    "card_id", "card_name", "global_deck_freq", "global_artifact_freq", "max_archetype_lift",
    "archetype_entropy", "staple_score", "downweight", "present_in_archetypes", "notes",
]
LIFT_FIELDS = [
    "archetype", "card_id", "card_name", "deck_count_in_archetype", "archetype_deck_count",
    "global_deck_count", "unique_deck_count", "p_card_given_archetype", "p_card_overall",
    "lift", "log_lift", "role_guess", "staple_score", "is_signature", "is_tech_candidate",
]
SIMILARITY_FIELDS = [
    "deck_hash", "neighbor_deck_hash", "weighted_similarity", "raw_jaccard", "shared_core_cards",
    "shared_package_cards", "shared_staples", "different_signature_cards", "same_primary_archetype",
    "same_variant_label", "rank",
]


@dataclass
class K2ABuildResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]
    warnings: dict[str, BuildWarning]


def _assignment_to_row(a: ArchetypeAssignment) -> dict[str, Any]:
    return {
        "deck_hash": a.deck_hash,
        "primary_archetype": a.primary_archetype,
        "secondary_package": a.secondary_package or "",
        "variant_label": a.variant_label,
        "confidence": a.confidence,
        "classification_status": a.classification_status,
        "core_cards": io_mod.json_cell(a.core_cards),
        "package_cards": io_mod.json_cell(a.package_cards),
        "staple_cards": io_mod.json_cell(a.staple_cards),
        "tech_cards": io_mod.json_cell(a.tech_cards),
        "representative_team_ids": io_mod.json_cell(a.representative_team_ids),
        "representative_player_ids": io_mod.json_cell(a.representative_player_ids),
        "representative_episode_ids": io_mod.json_cell(a.representative_episode_ids),
        "artifact_count": a.artifact_count,
        "episode_count": a.episode_count,
        "rank_bucket_summary": io_mod.json_cell(a.rank_bucket_summary),
        "anchor_scores": io_mod.json_cell(a.anchor_scores),
        "notes": a.notes,
    }


def build_archetype_summary(
    assignments: list[ArchetypeAssignment],
    unique_decks: dict[str, UniqueDeck],
    lift_rows: list[dict[str, Any]],
    staple_rows: list[dict[str, Any]],
    card_names: dict[CardId, str],
) -> list[dict[str, Any]]:
    total_decks = max(1, len(unique_decks))
    total_artifacts = max(1, sum(d.artifact_count for d in unique_decks.values()))
    staple_set = {r["card_id"] for r in staple_rows}

    by_archetype: dict[str, list[ArchetypeAssignment]] = defaultdict(list)
    for a in assignments:
        by_archetype[a.primary_archetype].append(a)

    core_lift: dict[str, list[tuple[float, CardId]]] = defaultdict(list)
    support_lift: dict[str, list[tuple[float, CardId]]] = defaultdict(list)
    tech_lift: dict[str, list[tuple[float, CardId]]] = defaultdict(list)
    for row in lift_rows:
        archetype = row["archetype"]
        pair = (float(row["lift"]), row["card_id"])
        if row["role_guess"] == "core" or row.get("is_signature"):
            core_lift[archetype].append(pair)
        elif row["role_guess"] == "support":
            support_lift[archetype].append(pair)
        elif row["role_guess"] == "tech":
            tech_lift[archetype].append(pair)

    def top_ids(pairs: list[tuple[float, CardId]], n: int = 8) -> list[str]:
        return [cid for _, cid in sorted(pairs, key=lambda p: (-p[0], p[1]))[:n]]

    rows: list[dict[str, Any]] = []
    for archetype, members in by_archetype.items():
        deck_count = len(members)
        artifact_count = sum(m.artifact_count for m in members)
        episode_count = len({eid for m in members for eid in m.representative_episode_ids})
        players = {pid for m in members for pid in m.representative_player_ids}
        confidences = [m.confidence for m in members]
        variant_counts = defaultdict(int)
        for m in members:
            variant_counts[m.variant_label] += 1
        staple_present = defaultdict(int)
        for m in members:
            for cid in m.staple_cards:
                staple_present[cid] += 1
        rows.append(
            {
                "archetype": archetype,
                "deck_count": deck_count,
                "artifact_count": artifact_count,
                "episode_count": episode_count,
                "player_team_count": len(players),
                "meta_share_by_deck": round(deck_count / total_decks, 4),
                "meta_share_by_artifact": round(artifact_count / total_artifacts, 4),
                "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "top_core_cards": io_mod.json_cell(top_ids(core_lift.get(archetype, []))),
                "top_support_cards": io_mod.json_cell(top_ids(support_lift.get(archetype, []))),
                "top_variants": io_mod.json_cell(
                    [v for v, _ in sorted(variant_counts.items(), key=lambda p: (-p[1], p[0]))[:5]]
                ),
                "top_staples": io_mod.json_cell(
                    [cid for cid, _ in sorted(staple_present.items(), key=lambda p: (-p[1], p[0]))[:8]]
                ),
                "top_tech_cards": io_mod.json_cell(top_ids(tech_lift.get(archetype, []))),
                "representative_deck_hashes": io_mod.json_cell(
                    [m.deck_hash for m in sorted(members, key=lambda x: (-x.artifact_count, x.deck_hash))[:8]]
                ),
                "representative_episode_ids": io_mod.json_cell(
                    sorted({eid for m in members for eid in m.representative_episode_ids})[:8]
                ),
            }
        )
    rows.sort(key=lambda r: (-r["artifact_count"], r["archetype"]))
    return rows


def _attach_names(rows: list[dict[str, Any]], card_names: dict[CardId, str]) -> None:
    for row in rows:
        row["card_name"] = card_names.get(row["card_id"], f"Unknown Card {row['card_id']}")


def build_meta_kg(
    input_dir: str | Path,
    out_root: str | Path = ".research/runs",
    config_path: str | Path | None = None,
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> K2ABuildResult:
    warnings: dict[str, BuildWarning] = {}
    config, raw_anchors, raw_engine_packages = io_mod.load_config(config_path)

    artifacts = io_mod.load_deck_artifacts(input_dir)
    if not artifacts:
        raise ValueError(f"no deck artifacts found under {input_dir}")
    unique_decks, conflicts = io_mod.aggregate_unique_decks(artifacts)
    if conflicts:
        w = warnings.setdefault("deck_hash_conflict", BuildWarning("deck_hash_conflict", "deck hash with multiple multisets"))
        for deck_hash, _detail in conflicts:
            w.bump(deck_hash)

    card_meta = card_metadata.load_card_metadata(config.use_engine_card_metadata)
    catalog_rows, _alternates = catalog_mod.build_card_catalog(artifacts, unique_decks, warnings, card_meta)
    card_names = {row.card_id: row.card_name for row in catalog_rows}

    manual_anchors = anchors_mod.resolve_anchor_config(raw_anchors, raw_engine_packages, catalog_rows, warnings)

    # Stage 1: classify with reliable cores only (manual + ex/Mega/Stage2 autos)
    # to measure each card's spread across real archetypes; broadly-spread
    # non-core Pokemon are cross-deck engine/splash cards, not identities.
    reliable_auto = anchors_mod.generate_auto_anchors(
        catalog_rows, config, existing=manual_anchors, card_meta=card_meta, reliable_only=True
    )
    reliable_anchors = anchors_mod.merge_anchors(manual_anchors, reliable_auto)
    reliable_terminal_ids = {cid for anchor in reliable_anchors for cid in anchor.terminal_card_ids}
    spread_staples = staples_mod.detect_preliminary_staples(catalog_rows, config, terminal_card_ids=reliable_terminal_ids)
    spread_assignments = classify_mod.classify_decks(unique_decks, reliable_anchors, spread_staples, config)
    splash_ids = anchors_mod.splash_card_ids(spread_assignments, unique_decks, config, card_meta=card_meta)
    # Exempt only curated manual cores; auto basic-ex splashes (Fezandipiti ex,
    # Meowth ex) stay eligible for splash demotion.
    manual_terminal_ids = {cid for anchor in manual_anchors for cid in anchor.terminal_card_ids}
    splash_ids -= manual_terminal_ids

    # Stage 2: full auto anchors (incl. non-ex Stage1/Stage2 terminals) minus splash.
    auto_anchors = anchors_mod.generate_auto_anchors(
        catalog_rows, config, existing=manual_anchors, card_meta=card_meta, exclude_ids=splash_ids
    )
    named_anchors = anchors_mod.merge_anchors(manual_anchors, auto_anchors)

    prelim_staples = spread_staples
    prelim_assignments = classify_mod.classify_decks(unique_decks, named_anchors, prelim_staples, config)
    prelim_lift = lift_mod.compute_card_lift(prelim_assignments, unique_decks, config, staple_scores=prelim_staples)
    staple_rows = staples_mod.detect_final_staples(catalog_rows, prelim_lift, named_anchors, config)
    staple_scores = {r["card_id"]: r["staple_score"] for r in staple_rows}
    staple_set: set[CardId] = set(staple_scores)

    # Fallback anchors: discover recurring non-ex Pokemon families among the
    # decks the named anchors left Unknown, so big real archetypes get labels.
    # Splash/engine cards are excluded so they cannot become labels.
    unknown_hashes = {a.deck_hash for a in prelim_assignments if a.classification_status == STATUS_UNKNOWN}
    unknown_decks = {h: unique_decks[h] for h in unknown_hashes}
    fallback_anchors = anchors_mod.generate_fallback_anchors(
        unknown_decks, catalog_rows, config, existing=named_anchors,
        staple_card_ids=staple_set | splash_ids, card_meta=card_meta,
    )
    all_anchors = anchors_mod.merge_anchors(named_anchors, fallback_anchors)

    assignments = classify_mod.classify_decks(unique_decks, all_anchors, staple_scores, config)
    lift_rows = lift_mod.compute_card_lift(assignments, unique_decks, config, staple_scores=staple_scores)
    support_packages = packages_mod.discover_support_packages(
        assignments, unique_decks, lift_rows, staple_set, card_names, config
    )
    classify_mod.enrich_card_roles(assignments, unique_decks, lift_rows, staple_set, config)
    packages_mod.assign_variants(assignments, support_packages, card_names, config)

    archetype_summary = build_archetype_summary(assignments, unique_decks, lift_rows, staple_rows, card_names)
    card_weights = features_mod.build_card_weights(catalog_rows, staple_scores, all_anchors, lift_rows, config)
    similarity_rows = similarity_mod.build_deck_similarity(unique_decks, card_weights, assignments, staple_set, config)

    # Live belief payload.
    archetype_priors = _archetype_priors(archetype_summary)
    global_likelihoods = {
        row.card_id: lift_mod.smoothed_global_prob(row.deck_count, len(unique_decks), config.lift_alpha)
        for row in catalog_rows
    }
    live_payload = build_live_belief_payload(archetype_priors, lift_rows, global_likelihoods, staple_scores)

    # Attach card names to lift & staple rows for output.
    _attach_names(lift_rows, card_names)

    # Write outputs.
    run_dir = create_run_dir(kind=META_KG_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    catalog_dicts = [catalog_row_to_dict(r) for r in catalog_rows]
    assignment_rows = sorted(
        (_assignment_to_row(a) for a in assignments),
        key=lambda r: (-r["artifact_count"], r["deck_hash"]),
    )

    output_paths: dict[str, str] = {}
    output_paths["card_catalog"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["card_catalog"]), CATALOG_FIELDS, catalog_dicts))
    output_paths["deck_archetypes"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["deck_archetypes"]), DECK_ARCHETYPE_FIELDS, assignment_rows))
    output_paths["archetype_summary"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["archetype_summary"]), ARCHETYPE_SUMMARY_FIELDS, archetype_summary))
    output_paths["support_packages"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["support_packages"]), SUPPORT_PACKAGE_FIELDS, [_pkg_row(p) for p in support_packages]))
    output_paths["staples"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["staples"]), STAPLE_FIELDS, [_staple_row(r) for r in staple_rows]))
    output_paths["card_lift_by_archetype"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["card_lift_by_archetype"]), LIFT_FIELDS, lift_rows))
    output_paths["deck_similarity"] = str(io_mod.write_csv(run_dir.output_path(OUTPUT_FILES["deck_similarity"]), SIMILARITY_FIELDS, [_sim_row(r) for r in similarity_rows]))
    output_paths["live_belief"] = str(write_json(run_dir.output_path(OUTPUT_FILES["live_belief"]), live_payload))

    counts = _build_counts(artifacts, unique_decks, catalog_rows, assignments, staple_rows, support_packages, lift_rows, similarity_rows)

    report_ctx = {
        "built_at_utc": _utc(),
        "artifact_version": ARTIFACT_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(run_dir.path),
        "counts": counts,
        "config": config,
        "catalog_rows": catalog_dicts,
        "staple_rows": staple_rows,
        "archetype_summary": archetype_summary,
        "assignment_rows": assignment_rows,
        "support_packages": support_packages,
        "warnings": warnings,
        "card_names": card_names,
        "output_paths": output_paths,
    }
    output_paths["deck_meta_report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["deck_meta_report"]), render_report(report_ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=META_KG_KIND,
        command=command,
        inputs={"input_dir": str(input_dir), "config_path": str(config_path) if config_path else None},
        outputs=output_paths,
        summary={
            "artifact_version": ARTIFACT_VERSION,
            **counts,
            "warnings": {code: w.count for code, w in warnings.items()},
        },
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)

    return K2ABuildResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        output_paths=output_paths,
        counts=counts,
        warnings=warnings,
    )


def _archetype_priors(archetype_summary: list[dict[str, Any]]) -> dict[str, float]:
    total = sum(r["deck_count"] for r in archetype_summary) or 1
    return {r["archetype"]: r["deck_count"] / total for r in archetype_summary}


def _build_counts(
    artifacts: list,
    unique_decks: dict[str, UniqueDeck],
    catalog_rows: list,
    assignments: list[ArchetypeAssignment],
    staple_rows: list,
    support_packages: list,
    lift_rows: list,
    similarity_rows: list,
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    for a in assignments:
        status_counts[a.classification_status] += 1
    return {
        "deck_artifact_count": len(artifacts),
        "unique_deck_count": len(unique_decks),
        "unique_card_count": len(catalog_rows),
        "episode_count": len({a.episode_id for a in artifacts}),
        "player_team_count": len({a.player_id for a in artifacts if a.player_id}),
        "archetype_count": len({a.primary_archetype for a in assignments}),
        "staple_count": len(staple_rows),
        "support_package_count": len(support_packages),
        "lift_row_count": len(lift_rows),
        "similarity_row_count": len(similarity_rows),
        "status_confident": status_counts.get("confident", 0),
        "status_mixed": status_counts.get(STATUS_MIXED, 0),
        "status_engine_first": status_counts.get(STATUS_ENGINE_FIRST, 0),
        "status_low_confidence": status_counts.get(STATUS_LOW_CONFIDENCE, 0),
        "status_unknown": status_counts.get(STATUS_UNKNOWN, 0),
    }


def _pkg_row(p: dict[str, Any]) -> dict[str, Any]:
    out = dict(p)
    out["card_ids"] = io_mod.json_cell(p["card_ids"])
    out["card_names"] = io_mod.json_cell(p["card_names"])
    out["representative_deck_hashes"] = io_mod.json_cell(p["representative_deck_hashes"])
    return out


def _staple_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    out["present_in_archetypes"] = io_mod.json_cell(r["present_in_archetypes"])
    return out


def _sim_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    for key in ("shared_core_cards", "shared_package_cards", "shared_staples", "different_signature_cards"):
        out[key] = io_mod.json_cell(r[key])
    return out


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K2A meta knowledge graph from mined deck artifacts.")
    parser.add_argument("--input", default=io_mod.DEFAULT_INPUT_DIR, help="dir of mined deck_summary.json artifacts")
    parser.add_argument("--out-root", default=".research/runs", help="run-dir root for outputs")
    parser.add_argument("--config", default=None, help="optional YAML config (anchors, thresholds)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    result = build_meta_kg(
        input_dir=args.input,
        out_root=args.out_root,
        config_path=args.config,
        command="python -m deepmew.meta_kg.build " + " ".join(argv or []),
        run_id=args.run_id,
        suffix=args.suffix,
    )
    print(f"meta_kg build complete: {result.run_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    if result.warnings:
        print("  warnings: " + ", ".join(f"{code}={w.count}" for code, w in result.warnings.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
