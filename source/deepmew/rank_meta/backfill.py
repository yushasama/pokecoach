"""K1.7 rank-metadata backfill: materialize scout ranks into generated tables.

Joins scout leaderboard ranks onto mined deck artifacts (by team_name) and the K2A
archetype labels (by deck_hash) to produce backfill / coverage / per-deck / per-
archetype rank tables and a report. It NEVER rewrites the original
``deck_summary.json`` files -- everything is new generated evidence under
``.research/runs/rank_metadata/``. Rank-weighted meta is a rank-known scout-subset
view (~45% coverage), not full ladder truth; raw frequency meta stays valid.

Conventions (documented because the columns mix two granularities):
- ``*_artifact_*`` columns are artifact-level (one mined deck payload = one row);
- ``best/mean/median_rank``, ``top*_count``, and ``unique_ranked_team_count`` are
  team-level (deduplicated by team_name), matching the directional archetype table.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown
from deepmew.imitation.io import load_label_map
from deepmew.meta_kg.io import json_cell, write_csv
from deepmew.rank_meta.loader import (
    DEFAULT_OPPONENT_DECKS,
    DEFAULT_SCOUT_ROOT,
    iter_deck_summaries,
    load_team_rank_map,
    rank_info_for,
)
from deepmew.rank_meta.schemas import (
    ARTIFACT_VERSION,
    RANK_BUCKET_ORDER,
    RANK_MISSING,
    RANK_META_KIND,
    rank_bucket,
)

RANK_BUCKETS = ("top10", "top25", "top50", "top100", "top300")

OUTPUT_FILES = {
    "backfill": "rank_metadata_backfill.csv",
    "coverage": "rank_metadata_coverage.csv",
    "ranked_deck": "ranked_deck_summary.csv",
    "archetype": "raw_vs_rank_weighted_archetype_summary.csv",
    "divergence": "frequency_vs_rank_divergence.csv",
    "top_player_decks": "top_player_decks.csv",
    "report": "rank_metadata_report.md",
}

BACKFILL_FIELDS = [
    "episode_id", "deck_hash", "team_name", "player_index", "leaderboard_rank_at_download",
    "rank_bucket", "leaderboard_rank_weight", "leaderboard_score", "team_id",
    "scout_source_manifest", "rank_missing", "rank_ambiguous", "source_deck_summary_path",
]
COVERAGE_FIELDS = ["metric", "value"]
RANKED_DECK_FIELDS = [
    "deck_hash", "artifact_count_raw", "artifact_count_rank_known", "rank_known_share",
    "best_rank", "mean_rank", "median_rank", "top10_count", "top25_count", "top50_count",
    "top100_count", "top300_count", "unique_ranked_team_count", "rank_weighted_artifact_count",
    "rank_weighted_artifact_share", "rank_missing_count",
]
ARCHETYPE_FIELDS = [
    "archetype", "raw_artifact_count", "raw_artifact_share", "rank_known_artifact_count",
    "rank_known_share", "rank_weighted_artifact_count", "rank_weighted_artifact_share",
    "best_rank", "mean_rank", "median_rank", "top10_count", "top25_count", "top50_count",
    "top100_count", "unique_ranked_team_count", "divergence_rank_minus_raw",
]
DIVERGENCE_FIELDS = ["archetype", "raw_artifact_share", "rank_weighted_artifact_share", "divergence_rank_minus_raw", "unique_ranked_team_count"]
TOP_PLAYER_FIELDS = ["deck_hash", "archetype", "best_rank", "best_team_name", "ranked_team_count", "ranked_teams", "rank_weighted_artifact_count"]


@dataclass
class RankMetaResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]


@dataclass
class _ArtifactRow:
    episode_id: str
    deck_hash: str
    team_name: str
    player_index: int | None
    rank: int | None
    bucket: str
    weight: float
    score: str | None
    team_id: str | None
    manifest: str
    missing: bool
    ambiguous: bool
    source_path: str
    archetype: str


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def _team_rank_stats(team_ranks: dict[str, int]) -> dict[str, Any]:
    """Team-level rank distribution from {team_name -> rank} (rank-known only)."""
    ranks = sorted(team_ranks.values())
    if not ranks:
        return {"best_rank": "", "mean_rank": "", "median_rank": "", "unique_ranked_team_count": 0,
                **{f"{b}_count": 0 for b in RANK_BUCKETS}}
    bucket_counts = {f"{b}_count": 0 for b in RANK_BUCKETS}
    for r in ranks:
        bucket_counts[f"{rank_bucket(r)}_count"] += 1
    return {
        "best_rank": ranks[0],
        "mean_rank": _round(statistics.fmean(ranks), 2),
        "median_rank": _round(statistics.median(ranks), 1),
        "unique_ranked_team_count": len(ranks),
        **bucket_counts,
    }


def build_rank_metadata(
    scout_root: str | Path = DEFAULT_SCOUT_ROOT,
    opponent_decks_root: str | Path = DEFAULT_OPPONENT_DECKS,
    meta_kg_dir: str | Path | None = None,
    out_root: str | Path = ".research/runs",
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> RankMetaResult:
    team_rank_map, warnings = load_team_rank_map(scout_root)
    label_map = load_label_map(meta_kg_dir) if meta_kg_dir else {}

    rows: list[_ArtifactRow] = []
    for record in iter_deck_summaries(opponent_decks_root):
        info = rank_info_for(team_rank_map, record["team_name"])
        archetype = (label_map.get(record["deck_hash"]) or {}).get("archetype", "Unknown")
        rows.append(_ArtifactRow(
            episode_id=record["episode_id"], deck_hash=record["deck_hash"],
            team_name=str(record["team_name"] or ""), player_index=record["player_index"],
            rank=info.rank, bucket=info.bucket, weight=info.leaderboard_weight, score=info.score,
            team_id=info.team_id, manifest=team_rank_map.get(str(record["team_name"]), None).scout_source_manifest if str(record["team_name"]) in team_rank_map else "",
            missing=info.missing, ambiguous=info.ambiguous, source_path=record["source_path"],
            archetype=archetype,
        ))

    backfill_rows = [_backfill_row(r) for r in rows]
    coverage_rows, coverage_counts = _coverage(rows, team_rank_map)
    ranked_deck_rows = _ranked_deck_summary(rows)
    archetype_rows = _archetype_summary(rows)
    divergence_rows = _divergence(archetype_rows)
    top_player_rows = _top_player_decks(rows)

    run_dir = create_run_dir(kind=RANK_META_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    out: dict[str, str] = {}
    out["backfill"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["backfill"]), BACKFILL_FIELDS, backfill_rows))
    out["coverage"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["coverage"]), COVERAGE_FIELDS, coverage_rows))
    out["ranked_deck"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["ranked_deck"]), RANKED_DECK_FIELDS, ranked_deck_rows))
    out["archetype"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["archetype"]), ARCHETYPE_FIELDS, archetype_rows))
    out["divergence"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["divergence"]), DIVERGENCE_FIELDS, divergence_rows))
    out["top_player_decks"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["top_player_decks"]), TOP_PLAYER_FIELDS, top_player_rows))

    report_ctx = {
        "built_at_utc": _utc(), "scout_root": str(scout_root), "meta_kg_dir": str(meta_kg_dir),
        "output_dir": str(run_dir.path), "counts": coverage_counts, "warnings": warnings,
        "archetype_rows": archetype_rows, "divergence_rows": divergence_rows,
    }
    out["report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["report"]), _render_report(report_ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id, kind=RANK_META_KIND, command=command,
        inputs={"scout_root": str(scout_root), "opponent_decks_root": str(opponent_decks_root), "meta_kg_dir": str(meta_kg_dir)},
        outputs=out, summary={"artifact_version": ARTIFACT_VERSION, **coverage_counts, "warnings": len(warnings)},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return RankMetaResult(run_dir.path, manifest_path, out, coverage_counts)


def _backfill_row(r: _ArtifactRow) -> dict[str, Any]:
    return {
        "episode_id": r.episode_id, "deck_hash": r.deck_hash, "team_name": r.team_name,
        "player_index": r.player_index if r.player_index is not None else "",
        "leaderboard_rank_at_download": r.rank if r.rank is not None else "",
        "rank_bucket": r.bucket, "leaderboard_rank_weight": r.weight,
        "leaderboard_score": r.score or "", "team_id": r.team_id or "",
        "scout_source_manifest": r.manifest, "rank_missing": r.missing, "rank_ambiguous": r.ambiguous,
        "source_deck_summary_path": r.source_path,
    }


def _coverage(rows: list[_ArtifactRow], team_rank_map: dict) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = len(rows)
    known = [r for r in rows if not r.missing]
    teams_with_rank = sum(1 for t in team_rank_map.values() if t.rank is not None)
    teams_ambiguous = sum(1 for t in team_rank_map.values() if t.ambiguous)
    bucket_counts = {b: sum(1 for r in rows if r.bucket == b) for b in (*RANK_BUCKETS, RANK_MISSING)}
    counts = {
        "selected_teams_total": len(team_rank_map),
        "teams_with_rank": teams_with_rank,
        "teams_ambiguous": teams_ambiguous,
        "artifacts_total": total,
        "artifacts_rank_known": len(known),
        "artifacts_rank_missing": total - len(known),
        "rank_known_share": _round(len(known) / total) if total else 0.0,
        "unique_teams_total": len({r.team_name for r in rows}),
        "unique_teams_rank_known": len({r.team_name for r in known}),
        "unique_deck_hashes_total": len({r.deck_hash for r in rows}),
        "unique_deck_hashes_rank_known": len({r.deck_hash for r in known}),
        **{f"artifacts_{b}": bucket_counts[b] for b in (*RANK_BUCKETS, RANK_MISSING)},
    }
    coverage_rows = [{"metric": k, "value": v} for k, v in counts.items()]
    return coverage_rows, counts


def _ranked_deck_summary(rows: list[_ArtifactRow]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[_ArtifactRow]] = defaultdict(list)
    for r in rows:
        by_hash[r.deck_hash].append(r)
    total_weight = sum(r.weight for r in rows) or 1.0
    out: list[dict[str, Any]] = []
    for deck_hash, group in by_hash.items():
        known = [r for r in group if not r.missing]
        team_ranks = {r.team_name: r.rank for r in known if r.rank is not None}
        weighted = sum(r.weight for r in group)
        stats = _team_rank_stats(team_ranks)
        out.append({
            "deck_hash": deck_hash,
            "artifact_count_raw": len(group),
            "artifact_count_rank_known": len(known),
            "rank_known_share": _round(len(known) / len(group)) if group else 0.0,
            **stats,
            "rank_weighted_artifact_count": _round(weighted, 2),
            "rank_weighted_artifact_share": _round(weighted / total_weight),
            "rank_missing_count": len(group) - len(known),
        })
    out.sort(key=lambda d: (d["best_rank"] if isinstance(d["best_rank"], int) else 1 << 30, -d["artifact_count_raw"]))
    return out


def _archetype_summary(rows: list[_ArtifactRow]) -> list[dict[str, Any]]:
    by_arch: dict[str, list[_ArtifactRow]] = defaultdict(list)
    for r in rows:
        by_arch[r.archetype].append(r)
    total_raw = len(rows) or 1
    total_weight = sum(r.weight for r in rows) or 1.0
    out: list[dict[str, Any]] = []
    for archetype, group in by_arch.items():
        known = [r for r in group if not r.missing]
        team_ranks = {r.team_name: r.rank for r in known if r.rank is not None}
        weighted = sum(r.weight for r in group)
        stats = _team_rank_stats(team_ranks)
        raw_share = len(group) / total_raw
        weighted_share = weighted / total_weight
        out.append({
            "archetype": archetype,
            "raw_artifact_count": len(group),
            "raw_artifact_share": _round(raw_share),
            "rank_known_artifact_count": len(known),
            "rank_known_share": _round(len(known) / len(group)) if group else 0.0,
            "rank_weighted_artifact_count": _round(weighted, 2),
            "rank_weighted_artifact_share": _round(weighted_share),
            "best_rank": stats["best_rank"], "mean_rank": stats["mean_rank"], "median_rank": stats["median_rank"],
            "top10_count": stats["top10_count"], "top25_count": stats["top25_count"],
            "top50_count": stats["top50_count"], "top100_count": stats["top100_count"],
            "unique_ranked_team_count": stats["unique_ranked_team_count"],
            "divergence_rank_minus_raw": _round(weighted_share - raw_share),
        })
    out.sort(key=lambda d: -d["rank_weighted_artifact_share"])
    return out


def _divergence(archetype_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "archetype": r["archetype"], "raw_artifact_share": r["raw_artifact_share"],
            "rank_weighted_artifact_share": r["rank_weighted_artifact_share"],
            "divergence_rank_minus_raw": r["divergence_rank_minus_raw"],
            "unique_ranked_team_count": r["unique_ranked_team_count"],
        }
        for r in archetype_rows
    ]
    rows.sort(key=lambda d: d["divergence_rank_minus_raw"], reverse=True)
    return rows


def _top_player_decks(rows: list[_ArtifactRow], rank_cutoff: int = 50) -> list[dict[str, Any]]:
    by_hash: dict[str, list[_ArtifactRow]] = defaultdict(list)
    for r in rows:
        by_hash[r.deck_hash].append(r)
    out: list[dict[str, Any]] = []
    for deck_hash, group in by_hash.items():
        known = [r for r in group if not r.missing and r.rank is not None]
        if not known:
            continue
        best = min(known, key=lambda r: r.rank)
        if best.rank > rank_cutoff:
            continue
        team_ranks = sorted({(r.team_name, r.rank) for r in known}, key=lambda tr: tr[1])
        out.append({
            "deck_hash": deck_hash, "archetype": group[0].archetype, "best_rank": best.rank,
            "best_team_name": best.team_name, "ranked_team_count": len(team_ranks),
            "ranked_teams": json_cell([{"team": t, "rank": rk} for t, rk in team_ranks]),
            "rank_weighted_artifact_count": _round(sum(r.weight for r in group), 2),
        })
    out.sort(key=lambda d: (d["best_rank"], d["deck_hash"]))
    return out


def _render_report(ctx: dict[str, Any]) -> str:
    c = ctx["counts"]
    lines = ["# K1.7 Rank Metadata Backfill", "",
             f"- Built: {ctx['built_at_utc']}  ·  artifact: `{ARTIFACT_VERSION}`",
             f"- Scout root: `{ctx['scout_root']}`  ·  K2A: `{ctx['meta_kg_dir']}`",
             f"- Output: `{ctx['output_dir']}`", "",
             "Rank = team **leaderboard standing at scout time (1-300)**, not per-game ELO/MMR "
             "and not a per-decision rating. Rank-weighted meta below is a **rank-known scout-subset "
             "view (~45% of artifacts)**, not full ladder truth; raw K2A/K3 frequency meta stays valid. "
             "Original `deck_summary.json` files were not modified.", "",
             "## Coverage", ""]
    lines += ["| metric | value |", "| --- | --- |"]
    for k in ("selected_teams_total", "teams_with_rank", "teams_ambiguous", "artifacts_total",
              "artifacts_rank_known", "artifacts_rank_missing", "rank_known_share",
              "unique_teams_total", "unique_teams_rank_known", "unique_deck_hashes_total",
              "unique_deck_hashes_rank_known"):
        lines.append(f"| {k} | {c.get(k)} |")
    if ctx["warnings"]:
        lines += ["", "## Warnings (ambiguous / conflicting team names)", ""]
        lines += [f"- {w}" for w in ctx["warnings"][:20]]
    lines += ["", "## Raw vs rank-weighted archetype share (directional; small samples)", "",
              "| archetype | raw_share | rank_wt_share | divergence | teams | best | median |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in ctx["archetype_rows"][:14]:
        lines.append(f"| {r['archetype']} | {r['raw_artifact_share']} | {r['rank_weighted_artifact_share']} | "
                     f"{r['divergence_rank_minus_raw']} | {r['unique_ranked_team_count']} | {r['best_rank']} | {r['median_rank']} |")
    lines += ["", "## Biggest raw->rank divergences (who rises/falls under rank weighting)", "",
              "| archetype | divergence_rank_minus_raw | raw_share | rank_wt_share |", "| --- | --- | --- | --- |"]
    top = ctx["divergence_rows"][:5]
    bottom = ctx["divergence_rows"][-5:]
    for r in [*top, *bottom]:
        lines.append(f"| {r['archetype']} | {r['divergence_rank_minus_raw']} | {r['raw_artifact_share']} | {r['rank_weighted_artifact_share']} |")
    lines += ["", "Directional only: rank-known coverage is partial and per-archetype team counts are small. "
              "Counterdeck/top-player claims must use rank-weighted meta where available, or explicitly "
              "state they are raw-frequency-only.", ""]
    return "\n".join(lines)


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize scout rank metadata onto mined deck artifacts (K1.7).")
    parser.add_argument("--scout-root", default=DEFAULT_SCOUT_ROOT)
    parser.add_argument("--opponent-decks", default=DEFAULT_OPPONENT_DECKS)
    parser.add_argument("--meta-kg", default=None, help="K2A run dir for archetype labels (default: latest)")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    meta_kg_dir = args.meta_kg
    if meta_kg_dir is None:
        from deepmew.imitation_fine.io import find_latest_meta_kg

        try:
            meta_kg_dir = str(find_latest_meta_kg())
        except FileNotFoundError:
            meta_kg_dir = None

    result = build_rank_metadata(
        scout_root=args.scout_root, opponent_decks_root=args.opponent_decks, meta_kg_dir=meta_kg_dir,
        out_root=args.out_root, command="python -m deepmew.rank_meta.backfill " + " ".join(argv or []),
        suffix=args.suffix,
    )
    print(f"rank_metadata build complete: {result.run_dir}")
    print(f"  meta_kg: {meta_kg_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
