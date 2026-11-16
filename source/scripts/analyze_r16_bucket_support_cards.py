"""R16: per-ELO-bucket support-card analysis over the K2A (DSU) classifier.

For every deck piloted by an R16 bucket-sampled team, splits the 60 cards into
the K2A classifier's per-deck **core** (anchor-family) cards, global **staples**,
and the remaining **support** cards, then reports per bucket x archetype which
support cards carry the core — and how support choices shift across score bands
(what higher-rated pilots of the same archetype add or drop).

Counting grain is (team, deck_hash): a player running a card counts once per
deck they pilot, so grinders do not drown out the composition signal; artifact
counts are carried as secondary evidence. Offline/research-only.

Examples
--------
    python scripts/analyze_r16_bucket_support_cards.py --section r16-elo-001
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_markdown  # noqa: E402
from deepmew.imitation_fine import io as fio  # noqa: E402
from deepmew.kaggle_replays.elo_buckets import find_latest_bucket_scout  # noqa: E402
from deepmew.meta_kg.io import write_csv  # noqa: E402
from deepmew.rank_meta.loader import DEFAULT_OPPONENT_DECKS  # noqa: E402

BUCKET_SUPPORT_KIND = "elo_bucket_support_cards"

SUPPORT_FIELDS = ["bucket", "archetype", "card_id", "card_name", "role", "n_players", "n_decks",
                  "deck_share", "avg_copies", "n_artifacts"]
SHIFT_FIELDS = ["archetype", "card_id", "card_name", "role", "low_share", "high_share", "shift",
                "low_decks", "high_decks"]

LOW_BUCKETS = {"600-800", "800-900"}
HIGH_BUCKETS = {"1000-1100", "1100-1200", "1200-1300"}


def _json_list(cell: str) -> list[str]:
    try:
        value = json.loads(cell) if cell else []
    except json.JSONDecodeError:
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _load_deck_assignments(meta_kg_dir: Path) -> dict[str, dict[str, Any]]:
    """deck_hash -> {archetype, core: set, staple: set} from the K2A classifier."""
    out: dict[str, dict[str, Any]] = {}
    with (meta_kg_dir / "deck_archetypes.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["deck_hash"]] = {
                "archetype": row["primary_archetype"],
                "core": set(_json_list(row.get("core_cards", ""))),
                "staple": set(_json_list(row.get("staple_cards", ""))),
            }
    return out


def _load_card_names(meta_kg_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with (meta_kg_dir / "card_catalog.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names[str(row["card_id"])] = row.get("card_name") or f"Unknown Card {row['card_id']}"
    return names


def _load_bucket_team_names(metadata_path: Path) -> dict[str, str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    teams: dict[str, str] = {}
    for record in metadata.get("sampled_teams") or []:
        name = record.get("team_name")
        if name:
            teams[str(name)] = str(record.get("bucket"))
    return teams


def _iter_bucket_decks(opponent_decks_root: Path, teams: dict[str, str]):
    """Yield (bucket, team_name, deck_hash, card_counts) per deck artifact of a bucketed team."""
    for path in sorted(Path(opponent_decks_root).glob("**/deck_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bucket = teams.get(str(data.get("team_name") or ""))
        if bucket is None or not data.get("deck_hash"):
            continue
        counts = data.get("card_counts") or {}
        if isinstance(counts, dict) and counts:
            yield bucket, str(data["team_name"]), str(data["deck_hash"]), {str(k): int(v) for k, v in counts.items()}


def analyze(args: argparse.Namespace) -> int:
    metadata_path = Path(args.bucket_scout) if args.bucket_scout else find_latest_bucket_scout()
    meta_kg_dir = Path(args.meta_kg) if args.meta_kg else Path(fio.find_latest_meta_kg())
    teams = _load_bucket_team_names(metadata_path)
    assignments = _load_deck_assignments(meta_kg_dir)
    card_names = _load_card_names(meta_kg_dir)

    # (team, deck_hash) grain with artifact counts.
    deck_records: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket, team, deck_hash, counts in _iter_bucket_decks(Path(args.opponent_decks), teams):
        record = deck_records.setdefault((team, deck_hash), {
            "bucket": bucket, "team": team, "deck_hash": deck_hash, "counts": counts, "artifacts": 0,
        })
        record["artifacts"] += 1
    if not deck_records:
        print("no bucketed decks found", file=sys.stderr)
        return 2

    # bucket x archetype x card aggregation.
    group_decks: dict[tuple[str, str], int] = defaultdict(int)
    card_agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in deck_records.values():
        assignment = assignments.get(record["deck_hash"])
        if assignment is None:
            continue
        bucket, archetype = record["bucket"], assignment["archetype"]
        group_decks[(bucket, archetype)] += 1
        for card_id, copies in record["counts"].items():
            role = "core" if card_id in assignment["core"] else ("staple" if card_id in assignment["staple"] else "support")
            agg = card_agg.setdefault((bucket, archetype, card_id), {
                "role": role, "players": set(), "decks": 0, "copies": 0, "artifacts": 0,
            })
            agg["players"].add(record["team"])
            agg["decks"] += 1
            agg["copies"] += copies
            agg["artifacts"] += record["artifacts"]

    support_rows: list[dict[str, Any]] = []
    for (bucket, archetype, card_id), agg in card_agg.items():
        n_group = group_decks[(bucket, archetype)]
        support_rows.append({
            "bucket": bucket, "archetype": archetype, "card_id": card_id,
            "card_name": card_names.get(card_id, f"Unknown Card {card_id}"), "role": agg["role"],
            "n_players": len(agg["players"]), "n_decks": agg["decks"],
            "deck_share": round(agg["decks"] / n_group, 4),
            "avg_copies": round(agg["copies"] / agg["decks"], 2),
            "n_artifacts": agg["artifacts"],
        })
    support_rows.sort(key=lambda r: (r["bucket"], r["archetype"], -r["n_decks"], r["card_id"]))

    # Low-band vs high-band support shift per archetype (support role only).
    def _band_shares(buckets: set[str]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], dict[str, int]]:
        deck_totals: dict[str, int] = defaultdict(int)
        card_decks: dict[tuple[str, str], int] = defaultdict(int)
        roles: dict[tuple[str, str], str] = {}
        for (bucket, archetype, card_id), agg in card_agg.items():
            if bucket not in buckets:
                continue
            card_decks[(archetype, card_id)] += agg["decks"]
            roles[(archetype, card_id)] = agg["role"]
        for (bucket, archetype), n in group_decks.items():
            if bucket in buckets:
                deck_totals[archetype] += n
        shares = {key: n / deck_totals[key[0]] for key, n in card_decks.items() if deck_totals.get(key[0])}
        return shares, dict(card_decks), dict(deck_totals)

    low_shares, low_decks, low_totals = _band_shares(LOW_BUCKETS)
    high_shares, high_decks, high_totals = _band_shares(HIGH_BUCKETS)
    shift_rows: list[dict[str, Any]] = []
    for key in sorted(set(low_shares) | set(high_shares)):
        archetype, card_id = key
        if low_totals.get(archetype, 0) < args.min_band_decks or high_totals.get(archetype, 0) < args.min_band_decks:
            continue
        low = low_shares.get(key, 0.0)
        high = high_shares.get(key, 0.0)
        if max(low, high) < 0.15:
            continue
        role = next((agg["role"] for (b, a, c), agg in card_agg.items() if a == archetype and c == card_id), "support")
        shift_rows.append({
            "archetype": archetype, "card_id": card_id,
            "card_name": card_names.get(card_id, f"Unknown Card {card_id}"), "role": role,
            "low_share": round(low, 4), "high_share": round(high, 4), "shift": round(high - low, 4),
            "low_decks": low_decks.get(key, 0), "high_decks": high_decks.get(key, 0),
        })
    shift_rows.sort(key=lambda r: (r["archetype"], -abs(r["shift"])))

    run_dir = create_run_dir(kind=BUCKET_SUPPORT_KIND, out_root=args.out_root, suffix=args.section)
    out = {
        "bucket_support_cards": str(write_csv(run_dir.output_path("bucket_support_cards.csv"), SUPPORT_FIELDS, support_rows)),
        "support_shift_low_vs_high": str(write_csv(run_dir.output_path("support_shift_low_vs_high.csv"), SHIFT_FIELDS, shift_rows)),
    }
    report = _report(support_rows, shift_rows, group_decks, metadata_path, meta_kg_dir)
    out["report"] = str(write_markdown(run_dir.output_path("report.md"), report))
    summary = {
        "bucketed_unique_decks": len(deck_records),
        "bucket_archetype_groups": len(group_decks),
        "support_rows": len(support_rows),
        "shift_rows": len(shift_rows),
    }
    write_manifest(run_dir.manifest_path, build_manifest(
        run_id=run_dir.run_id, kind=BUCKET_SUPPORT_KIND,
        command="python scripts/analyze_r16_bucket_support_cards.py " + " ".join(sys.argv[1:]),
        inputs={"bucket_scout": str(metadata_path), "meta_kg": str(meta_kg_dir),
                "opponent_decks": str(args.opponent_decks)},
        outputs=out, summary=summary,
    ))
    print(f"run_dir={run_dir.path}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


def _report(support_rows, shift_rows, group_decks, metadata_path, meta_kg_dir) -> str:
    lines = [
        "# R16 Bucket Support-Card Analysis (K2A/DSU classifier slice)",
        "",
        f"bucket_scout={metadata_path}",
        f"meta_kg={meta_kg_dir}",
        "",
        "Roles come from the K2A classifier: `core` = anchor-family cards, `staple` = global",
        "staples, `support` = everything else that carries the core.",
        "",
    ]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in support_rows:
        by_group[(row["bucket"], row["archetype"])].append(row)
    for (bucket, archetype), n in sorted(group_decks.items(), key=lambda kv: (kv[0][0], -kv[1])):
        if n < 3 or archetype == "Unknown":
            continue
        rows = [r for r in by_group[(bucket, archetype)] if r["role"] == "support" and r["deck_share"] >= 0.4]
        if not rows:
            continue
        lines.append(f"## {bucket} — {archetype} ({n} unique decks)")
        lines.append("")
        for r in sorted(rows, key=lambda r: -r["deck_share"])[:10]:
            lines.append(f"- {r['card_name']} (id {r['card_id']}): {r['deck_share']:.0%} of decks, "
                         f"avg {r['avg_copies']} copies, {r['n_players']} players")
        lines.append("")
    lines += ["## Support shifts: high band (1000+) vs low band (600-900)", ""]
    strong = [r for r in shift_rows if abs(r["shift"]) >= 0.25 and r["role"] != "core"]
    if strong:
        for r in strong[:30]:
            direction = "ADDED at high" if r["shift"] > 0 else "DROPPED at high"
            lines.append(f"- {r['archetype']}: {r['card_name']} {direction} "
                         f"(low {r['low_share']:.0%} -> high {r['high_share']:.0%})")
    else:
        lines.append("- No support card shifts >= 25 points with current coverage.")
    lines += [
        "",
        "## Caveats",
        "",
        "- Counting grain is (team, deck_hash); a card counts once per player-deck, not per game.",
        "- Bucket = leaderboard score at scout time. Small archetype x bucket cells are noise; the",
        "  CSVs carry n_decks/n_players so downstream consumers can threshold.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R16 per-bucket support-card analysis.")
    parser.add_argument("--section", required=True)
    parser.add_argument("--bucket-scout", default=None, help="bucket_teams.json (default: latest).")
    parser.add_argument("--meta-kg", default=None, help="K2A run dir (default: latest).")
    parser.add_argument("--opponent-decks", default=DEFAULT_OPPONENT_DECKS)
    parser.add_argument("--min-band-decks", type=int, default=8,
                        help="Min unique decks per band for an archetype to enter the shift table.")
    parser.add_argument("--out-root", default=".research/runs")
    return analyze(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
