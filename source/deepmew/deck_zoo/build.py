"""K3 Deck Zoo builder: curate representative decks + arena opponent sets from K2A.

Reads a K2A meta-kg run directory, selects representative decks per archetype,
flags near-duplicates via deck similarity, assigns stratified train/eval/holdout
splits, and groups canonical decks into named arena opponent sets. Outputs a
machine-readable zoo plus a human report. Offline/training-only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_jsonl, write_markdown
from deepmew.deck_zoo import io as zoo_io
from deepmew.deck_zoo.report import render_report
from deepmew.deck_zoo.schemas import (
    ARTIFACT_VERSION,
    DECK_ZOO_KIND,
    DeckZooConfig,
    DeckZooEntry,
    OUTPUT_FILES,
    SET_ANTI_OVERFIT_HOLDOUT,
    SET_ENGINE_FIRST,
    SET_TOP_META,
    SET_UNKNOWN_MIXED,
    SET_VARIANT_DIVERSITY,
    SPLIT_EVAL,
    SPLIT_HOLDOUT,
    SPLIT_TRAIN,
)
from deepmew.meta_kg.io import json_cell, write_csv

DECK_ZOO_FIELDS = [
    "deck_hash", "archetype", "variant_label", "classification_status", "confidence",
    "representative_score", "artifact_count", "episode_count", "deck_count_in_cluster",
    "is_canonical", "is_near_duplicate", "cluster_id", "split", "card_count",
    "core_cards", "package_cards", "staple_cards", "tech_cards",
    "representative_player_ids", "representative_episode_ids", "reason_selected",
]
DECK_ZOO_SUMMARY_FIELDS = [
    "archetype", "deck_count", "canonical_count", "near_duplicate_count",
    "artifact_count", "episode_count", "meta_share_by_artifact", "avg_confidence",
    "train_count", "eval_count", "holdout_count", "representative_deck_hashes",
]
OPPONENT_SET_FIELDS = [
    "set_name", "rank_in_set", "deck_hash", "archetype", "variant_label",
    "classification_status", "confidence", "weight", "split", "reason",
]
MATCHUP_FIELDS = [
    "matchup_id", "set_name", "our_deck", "opponent_deck_hash", "opponent_archetype",
    "opponent_variant", "weight", "suggested_games",
]


@dataclass
class DeckZooResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]


def _stable_unit(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _representative_score(deck: dict[str, Any], max_log_art: float, max_log_ep: float, cfg: DeckZooConfig) -> float:
    art = math.log1p(deck["artifact_count"]) / max_log_art if max_log_art else 0.0
    ep = math.log1p(deck["episode_count"]) / max_log_ep if max_log_ep else 0.0
    return round(cfg.weight_confidence * deck["confidence"] + cfg.weight_artifact * art + cfg.weight_episode * ep, 6)


def _cluster_within_archetype(
    decks: list[dict[str, Any]],
    similarity: dict[str, list[tuple[str, float, bool]]],
    cfg: DeckZooConfig,
) -> dict[str, str]:
    """Union-find near-duplicate clustering within each archetype. Returns deck_hash -> cluster_id."""
    by_hash = {d["deck_hash"]: d for d in decks}
    parent = {h: h for h in by_hash}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for deck_hash, deck in by_hash.items():
        for neighbor, sim, same_arch in similarity.get(deck_hash, []):
            if not same_arch or neighbor not in by_hash:
                continue
            if by_hash[neighbor]["primary_archetype"] != deck["primary_archetype"]:
                continue
            if sim >= cfg.dup_similarity_threshold:
                union(deck_hash, neighbor)
    return {h: find(h) for h in by_hash}


def _assign_splits(canonical_by_archetype: dict[str, list[dict[str, Any]]], cfg: DeckZooConfig) -> dict[str, str]:
    """Per-archetype stratified split; the best exemplar always trains."""
    split: dict[str, str] = {}
    for archetype, reps in canonical_by_archetype.items():
        ordered = sorted(reps, key=lambda d: (-d["representative_score"], d["deck_hash"]))
        for i, deck in enumerate(ordered):
            if i == 0:
                split[deck["deck_hash"]] = SPLIT_TRAIN
                continue
            u = _stable_unit(deck["deck_hash"])
            if u < cfg.holdout_fraction:
                split[deck["deck_hash"]] = SPLIT_HOLDOUT
            elif u < cfg.holdout_fraction + cfg.eval_fraction:
                split[deck["deck_hash"]] = SPLIT_EVAL
            else:
                split[deck["deck_hash"]] = SPLIT_TRAIN
    return split


def build_deck_zoo(
    meta_kg_dir: str | Path,
    input_dir: str | Path,
    out_root: str | Path = ".research/runs",
    config: DeckZooConfig | None = None,
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> DeckZooResult:
    cfg = config or DeckZooConfig()
    meta = zoo_io.load_meta_kg(meta_kg_dir)
    decklists = zoo_io.load_full_decklists(input_dir)
    decks = meta["decks"]
    similarity = meta["similarity"]
    summary_by_archetype = {row["archetype"]: row for row in meta["archetype_summary"]}

    max_log_art = max((math.log1p(d["artifact_count"]) for d in decks), default=1.0) or 1.0
    max_log_ep = max((math.log1p(d["episode_count"]) for d in decks), default=1.0) or 1.0
    for d in decks:
        d["representative_score"] = _representative_score(d, max_log_art, max_log_ep, cfg)

    # Cluster per archetype.
    decks_by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decks:
        decks_by_archetype[d["primary_archetype"]].append(d)
    cluster_of: dict[str, str] = {}
    cluster_size: dict[str, int] = defaultdict(int)
    for archetype, group in decks_by_archetype.items():
        sub = _cluster_within_archetype(group, similarity, cfg)
        cluster_of.update(sub)
    for cid in cluster_of.values():
        cluster_size[cid] += 1
    # Canonical = highest rep_score within each cluster.
    best_in_cluster: dict[str, dict[str, Any]] = {}
    for d in decks:
        cid = cluster_of[d["deck_hash"]]
        cur = best_in_cluster.get(cid)
        if cur is None or (d["representative_score"], d["deck_hash"]) > (cur["representative_score"], cur["deck_hash"]):
            best_in_cluster[cid] = d
    canonical_hashes = {d["deck_hash"] for d in best_in_cluster.values()}

    canonical_by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decks:
        if d["deck_hash"] in canonical_hashes:
            canonical_by_archetype[d["primary_archetype"]].append(d)
    split_of = _assign_splits(canonical_by_archetype, cfg)
    # Near-duplicates inherit their canonical's split (avoids imitation leakage).
    for d in decks:
        if d["deck_hash"] not in canonical_hashes:
            canonical = best_in_cluster[cluster_of[d["deck_hash"]]]
            split_of[d["deck_hash"]] = split_of.get(canonical["deck_hash"], SPLIT_TRAIN)

    entries = _build_entries(decks, cluster_of, cluster_size, canonical_hashes, split_of, decklists)
    zoo_summary = _build_summary(decks, canonical_hashes, cluster_of, split_of, summary_by_archetype, canonical_by_archetype, cfg)
    opponent_sets = _build_opponent_sets(canonical_by_archetype, summary_by_archetype, split_of, cfg)
    matchups = _build_matchups(opponent_sets, cfg)

    # Write.
    run_dir = create_run_dir(kind=DECK_ZOO_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    output_paths: dict[str, str] = {}
    output_paths["deck_zoo"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["deck_zoo"]), DECK_ZOO_FIELDS, [_entry_row(e) for e in entries]))
    output_paths["deck_zoo_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["deck_zoo_summary"]), DECK_ZOO_SUMMARY_FIELDS, zoo_summary))
    output_paths["arena_opponent_sets"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["arena_opponent_sets"]), OPPONENT_SET_FIELDS, opponent_sets))
    output_paths["arena_matchups"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["arena_matchups"]), MATCHUP_FIELDS, matchups))
    output_paths["deck_zoo_decklists"] = str(write_jsonl(
        run_dir.output_path(OUTPUT_FILES["deck_zoo_decklists"]),
        [{"deck_hash": e.deck_hash, "archetype": e.archetype, "split": e.split, "is_canonical": e.is_canonical,
          "card_counts": decklists.get(e.deck_hash, {})} for e in entries if e.is_canonical],
    ))

    counts = _build_counts(entries, opponent_sets, matchups, zoo_summary)
    report_ctx = {
        "built_at_utc": _utc(),
        "artifact_version": ARTIFACT_VERSION,
        "meta_kg_dir": str(meta_kg_dir),
        "output_dir": str(run_dir.path),
        "counts": counts,
        "config": cfg,
        "zoo_summary": zoo_summary,
        "opponent_sets": opponent_sets,
        "card_names": meta["card_names"],
        "output_paths": output_paths,
    }
    output_paths["deck_zoo_report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["deck_zoo_report"]), render_report(report_ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id, kind=DECK_ZOO_KIND, command=command,
        inputs={"meta_kg_dir": str(meta_kg_dir), "input_dir": str(input_dir)},
        outputs=output_paths,
        summary={"artifact_version": ARTIFACT_VERSION, **counts},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return DeckZooResult(run_dir.path, manifest_path, output_paths, counts)


def _build_entries(decks, cluster_of, cluster_size, canonical_hashes, split_of, decklists) -> list[DeckZooEntry]:
    entries: list[DeckZooEntry] = []
    for d in decks:
        h = d["deck_hash"]
        is_canon = h in canonical_hashes
        cid = cluster_of[h]
        reason = "canonical archetype representative" if is_canon else f"near-duplicate of {cid}"
        entries.append(
            DeckZooEntry(
                deck_hash=h,
                archetype=d["primary_archetype"],
                variant_label=d["variant_label"],
                classification_status=d["classification_status"],
                confidence=d["confidence"],
                representative_score=d["representative_score"],
                artifact_count=d["artifact_count"],
                episode_count=d["episode_count"],
                deck_count_in_cluster=cluster_size[cid],
                is_canonical=is_canon,
                is_near_duplicate=not is_canon,
                cluster_id=cid,
                split=split_of.get(h, SPLIT_TRAIN),
                core_cards=d["core_cards"],
                package_cards=d["package_cards"],
                staple_cards=d["staple_cards"],
                tech_cards=d["tech_cards"],
                representative_player_ids=d["representative_player_ids"],
                representative_episode_ids=d["representative_episode_ids"],
                card_count=sum(decklists.get(h, {}).values()),
                reason_selected=reason,
            )
        )
    entries.sort(key=lambda e: (-e.representative_score, e.deck_hash))
    return entries


def _entry_row(e: DeckZooEntry) -> dict[str, Any]:
    return {
        "deck_hash": e.deck_hash, "archetype": e.archetype, "variant_label": e.variant_label,
        "classification_status": e.classification_status, "confidence": e.confidence,
        "representative_score": e.representative_score, "artifact_count": e.artifact_count,
        "episode_count": e.episode_count, "deck_count_in_cluster": e.deck_count_in_cluster,
        "is_canonical": e.is_canonical, "is_near_duplicate": e.is_near_duplicate,
        "cluster_id": e.cluster_id, "split": e.split, "card_count": e.card_count,
        "core_cards": json_cell(e.core_cards), "package_cards": json_cell(e.package_cards),
        "staple_cards": json_cell(e.staple_cards), "tech_cards": json_cell(e.tech_cards),
        "representative_player_ids": json_cell(e.representative_player_ids),
        "representative_episode_ids": json_cell(e.representative_episode_ids),
        "reason_selected": e.reason_selected,
    }


def _build_summary(decks, canonical_hashes, cluster_of, split_of, summary_by_archetype, canonical_by_archetype, cfg) -> list[dict[str, Any]]:
    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decks:
        by_archetype[d["primary_archetype"]].append(d)
    rows: list[dict[str, Any]] = []
    for archetype, group in by_archetype.items():
        canon = [d for d in group if d["deck_hash"] in canonical_hashes]
        splits = [split_of.get(d["deck_hash"], SPLIT_TRAIN) for d in canon]
        meta_row = summary_by_archetype.get(archetype, {})
        rows.append(
            {
                "archetype": archetype,
                "deck_count": len(group),
                "canonical_count": len(canon),
                "near_duplicate_count": len(group) - len(canon),
                "artifact_count": sum(d["artifact_count"] for d in group),
                "episode_count": sum(d["episode_count"] for d in group),
                "meta_share_by_artifact": meta_row.get("meta_share_by_artifact", 0.0),
                "avg_confidence": round(sum(d["confidence"] for d in group) / len(group), 4) if group else 0.0,
                "train_count": splits.count(SPLIT_TRAIN),
                "eval_count": splits.count(SPLIT_EVAL),
                "holdout_count": splits.count(SPLIT_HOLDOUT),
                "representative_deck_hashes": json_cell(
                    [d["deck_hash"] for d in sorted(canon, key=lambda x: (-x["representative_score"], x["deck_hash"]))[:cfg.max_reps_per_archetype]]
                ),
            }
        )
    rows.sort(key=lambda r: (-r["artifact_count"], r["archetype"]))
    return rows


def _top_reps(reps: list[dict[str, Any]], cfg: DeckZooConfig) -> list[dict[str, Any]]:
    return sorted(reps, key=lambda d: (-d["representative_score"], d["deck_hash"]))[: cfg.max_reps_per_archetype]


def _build_opponent_sets(canonical_by_archetype, summary_by_archetype, split_of, cfg) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # top_meta: cover cumulative artifact share with the best confident rep per archetype.
    ranked_archetypes = sorted(
        (a for a in canonical_by_archetype if a not in ("Unknown", "Mixed")),
        key=lambda a: -summary_by_archetype.get(a, {}).get("meta_share_by_artifact", 0.0),
    )
    cumulative = 0.0
    rank = 0
    for archetype in ranked_archetypes:
        share = summary_by_archetype.get(archetype, {}).get("meta_share_by_artifact", 0.0)
        reps = [d for d in _top_reps(canonical_by_archetype[archetype], cfg) if d["confidence"] >= cfg.top_meta_min_confidence]
        if not reps:
            continue
        rank += 1
        best = reps[0]
        rows.append(_set_row(SET_TOP_META, rank, best, share, split_of, "top archetype by artifact share"))
        cumulative += share
        if cumulative >= cfg.top_meta_cumulative_share:
            break

    # variant_diversity: distinct variants within common archetypes.
    rank = 0
    for archetype, reps in canonical_by_archetype.items():
        if summary_by_archetype.get(archetype, {}).get("deck_count", 0) < cfg.variant_min_archetype_decks:
            continue
        seen_variants: set[str] = set()
        for d in _top_reps(reps, cfg):
            if d["variant_label"] in seen_variants:
                continue
            seen_variants.add(d["variant_label"])
            rank += 1
            rows.append(_set_row(SET_VARIANT_DIVERSITY, rank, d, d["confidence"], split_of, "distinct variant exemplar"))

    # anti_overfit_holdout: canonical reps held out from imitation.
    rank = 0
    holdout = [d for reps in canonical_by_archetype.values() for d in reps if split_of.get(d["deck_hash"]) == SPLIT_HOLDOUT]
    for d in sorted(holdout, key=lambda d: (-d["representative_score"], d["deck_hash"])):
        rank += 1
        rows.append(_set_row(SET_ANTI_OVERFIT_HOLDOUT, rank, d, d["confidence"], split_of, "held out from imitation"))

    # unknown_mixed: robustness / chaos bucket.
    rank = 0
    um = [d for reps in canonical_by_archetype.values() for d in reps if d["classification_status"] in ("unknown", "mixed")]
    for d in sorted(um, key=lambda d: (-d["representative_score"], d["deck_hash"])):
        rank += 1
        rows.append(_set_row(SET_UNKNOWN_MIXED, rank, d, d["confidence"], split_of, "low-confidence/mixed robustness bucket"))

    # engine_first: decks anchored by a non-Pokemon engine (may be empty in v0).
    rank = 0
    ef = [d for reps in canonical_by_archetype.values() for d in reps if d["classification_status"] == "engine_first"]
    for d in sorted(ef, key=lambda d: (-d["representative_score"], d["deck_hash"])):
        rank += 1
        rows.append(_set_row(SET_ENGINE_FIRST, rank, d, d["confidence"], split_of, "engine-first deck"))

    return rows


def _set_row(set_name, rank, deck, weight, split_of, reason) -> dict[str, Any]:
    return {
        "set_name": set_name,
        "rank_in_set": rank,
        "deck_hash": deck["deck_hash"],
        "archetype": deck["primary_archetype"],
        "variant_label": deck["variant_label"],
        "classification_status": deck["classification_status"],
        "confidence": deck["confidence"],
        "weight": round(weight, 6),
        "split": split_of.get(deck["deck_hash"], SPLIT_TRAIN),
        "reason": reason,
    }


def _build_matchups(opponent_sets: list[dict[str, Any]], cfg: DeckZooConfig) -> list[dict[str, Any]]:
    """Candidate arena matchups: DeepMew baseline vs each top_meta / holdout opponent."""
    rows: list[dict[str, Any]] = []
    idx = 0
    for set_name in (SET_TOP_META, SET_ANTI_OVERFIT_HOLDOUT, SET_UNKNOWN_MIXED, SET_ENGINE_FIRST):
        for row in [r for r in opponent_sets if r["set_name"] == set_name]:
            idx += 1
            rows.append(
                {
                    "matchup_id": f"m{idx:04d}",
                    "set_name": set_name,
                    "our_deck": "deepmew_current",
                    "opponent_deck_hash": row["deck_hash"],
                    "opponent_archetype": row["archetype"],
                    "opponent_variant": row["variant_label"],
                    "weight": row["weight"],
                    "suggested_games": cfg.suggested_games_per_matchup,
                }
            )
    return rows


def _build_counts(entries, opponent_sets, matchups, zoo_summary) -> dict[str, Any]:
    set_counts: dict[str, int] = defaultdict(int)
    for row in opponent_sets:
        set_counts[row["set_name"]] += 1
    split_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        if e.is_canonical:
            split_counts[e.split] += 1
    return {
        "deck_count": len(entries),
        "canonical_count": sum(1 for e in entries if e.is_canonical),
        "near_duplicate_count": sum(1 for e in entries if e.is_near_duplicate),
        "archetype_count": len(zoo_summary),
        "opponent_set_rows": len(opponent_sets),
        "matchup_count": len(matchups),
        "set_top_meta": set_counts.get(SET_TOP_META, 0),
        "set_variant_diversity": set_counts.get(SET_VARIANT_DIVERSITY, 0),
        "set_anti_overfit_holdout": set_counts.get(SET_ANTI_OVERFIT_HOLDOUT, 0),
        "set_unknown_mixed": set_counts.get(SET_UNKNOWN_MIXED, 0),
        "set_engine_first": set_counts.get(SET_ENGINE_FIRST, 0),
        "canonical_train": split_counts.get(SPLIT_TRAIN, 0),
        "canonical_eval": split_counts.get(SPLIT_EVAL, 0),
        "canonical_holdout": split_counts.get(SPLIT_HOLDOUT, 0),
    }


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K3 Deck Zoo from a K2A meta-kg run.")
    parser.add_argument("--meta-kg", default=None, help="K2A run dir (default: latest under .research/runs/meta_kg)")
    parser.add_argument("--input", default=".research/opponent_decks", help="mined deck artifacts dir (for full decklists)")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    meta_kg_dir = args.meta_kg or zoo_io.find_latest_meta_kg()
    result = build_deck_zoo(
        meta_kg_dir=meta_kg_dir, input_dir=args.input, out_root=args.out_root,
        command="python -m deepmew.deck_zoo.build " + " ".join(argv or []),
        run_id=args.run_id, suffix=args.suffix,
    )
    print(f"deck_zoo build complete: {result.run_dir}")
    print(f"  meta_kg: {meta_kg_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
