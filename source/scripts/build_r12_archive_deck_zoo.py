from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.deck.tables import card_table


SCHEMA_VERSION = "deepmew.r12.archive_deck_zoo.v0"
DEFAULT_GROUPS = ("dragapult", "starmie_staryu", "trevenant", "latias", "lucario", "abomasnow")
GROUP_RULES = {
    "dragapult": ("Dragapult ex",),
    "starmie_staryu": ("Mega Starmie ex", "Staryu"),
    "trevenant": ("Hop's Trevenant",),
    "latias": ("Latias ex",),
    "lucario": ("Mega Lucario ex",),
    "abomasnow": ("Mega Abomasnow ex",),
}
ARCHETYPE_LABELS = {
    "dragapult": "Dragapult",
    "starmie_staryu": "Mega Starmie ex",
    "trevenant": "Hop's Trevenant",
    "latias": "Latias ex",
    "lucario": "Mega Lucario ex",
    "abomasnow": "Mega Abomasnow ex",
}
GROUP_PRIORITY = ("dragapult", "starmie_staryu", "trevenant", "lucario", "abomasnow", "latias")
FEATURE_NAMES = (
    "Mega Lucario ex",
    "Dragapult ex",
    "Latias ex",
    "Mega Starmie ex",
    "Staryu",
    "Hop's Trevenant",
    "Mega Abomasnow ex",
    "Marnie's Grimmsnarl ex",
    "Alakazam",
    "Dudunsparce",
)
DECK_ZOO_FIELDS = [
    "deck_hash",
    "archetype",
    "variant_label",
    "classification_status",
    "confidence",
    "representative_score",
    "artifact_count",
    "episode_count",
    "deck_count_in_cluster",
    "is_canonical",
    "is_near_duplicate",
    "cluster_id",
    "split",
    "card_count",
    "core_cards",
    "package_cards",
    "staple_cards",
    "tech_cards",
    "representative_player_ids",
    "representative_episode_ids",
    "reason_selected",
    "source_groups",
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    columns = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _card_name_counts(card_counts: dict[str, int]) -> dict[str, int]:
    table = card_table()
    names: dict[str, int] = {}
    for raw_id, raw_count in card_counts.items():
        card_id = int(raw_id)
        card = table.get(card_id)
        name = card.name if card is not None else str(card_id)
        names[name] = names.get(name, 0) + int(raw_count)

    return names


def _groups_for_names(name_counts: dict[str, int], wanted: set[str]) -> list[str]:
    groups: list[str] = []
    for group in DEFAULT_GROUPS:
        if group not in wanted:
            continue
        needles = GROUP_RULES[group]
        if any(name in name_counts for name in needles):
            groups.append(group)

    return groups


def _primary_group(groups: list[str]) -> str | None:
    for group in GROUP_PRIORITY:
        if group in groups:
            return group

    return groups[0] if groups else None


def _variant_label(name_counts: dict[str, int], fallback: str) -> str:
    present = [name for name in FEATURE_NAMES if name_counts.get(name)]
    if present:
        return " / ".join(present[:4])

    return fallback or "Unknown"


def _merge_entry(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    existing["artifact_count"] = max(int(existing.get("artifact_count") or 0), int(update.get("artifact_count") or 0))
    existing["episode_count"] = max(int(existing.get("episode_count") or 0), int(update.get("episode_count") or 0))
    existing["source_groups"] = sorted(set(existing.get("source_groups") or []) | set(update.get("source_groups") or []))
    for key in ("representative_player_ids", "representative_episode_ids"):
        values = list(dict.fromkeys(list(existing.get(key) or []) + list(update.get(key) or [])))
        existing[key] = values[:12]
    if existing.get("archetype") == "Unknown" and update.get("archetype") != "Unknown":
        existing["archetype"] = update["archetype"]
    if existing.get("variant_label") == "Unknown" and update.get("variant_label") != "Unknown":
        existing["variant_label"] = update["variant_label"]

    return existing


def _base_entries(deck_zoo_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    meta = {row["deck_hash"]: row for row in _read_csv(deck_zoo_dir / "deck_zoo.csv")}
    entries: dict[str, dict[str, Any]] = {}
    decklists_path = deck_zoo_dir / "deck_zoo_decklists.jsonl"
    for line in decklists_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        deck_hash = str(row["deck_hash"])
        card_counts = {str(card_id): int(count) for card_id, count in (row.get("card_counts") or {}).items()}
        name_counts = _card_name_counts(card_counts)
        groups = _groups_for_names(name_counts, wanted)
        primary = _primary_group(groups)
        info = meta.get(deck_hash, {})
        archetype = ARCHETYPE_LABELS.get(primary or "", str(row.get("archetype") or "Unknown"))
        variant = _variant_label(name_counts, info.get("variant_label") or archetype)
        entries[deck_hash] = {
            "deck_hash": deck_hash,
            "archetype": archetype,
            "variant_label": variant,
            "classification_status": "fresh_top_archive",
            "confidence": float(info.get("confidence") or 1.0),
            "representative_score": float(info.get("representative_score") or info.get("artifact_count") or 1.0),
            "artifact_count": int(info.get("artifact_count") or 1),
            "episode_count": int(info.get("episode_count") or 1),
            "split": str(row.get("split") or info.get("split") or "train"),
            "card_counts": card_counts,
            "name_counts": name_counts,
            "source_groups": sorted(set(["fresh_top"] + groups)),
            "representative_player_ids": _load_json_cell(info.get("representative_player_ids")),
            "representative_episode_ids": _load_json_cell(info.get("representative_episode_ids")),
            "reason_selected": "fresh top-player deck zoo row",
        }

    return entries


def _load_json_cell(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []

    return [str(item) for item in value]


def _scan_mined(opponent_decks: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    mined: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, int] = defaultdict(int)
    episodes: dict[str, set[str]] = defaultdict(set)
    players: dict[str, list[str]] = defaultdict(list)
    episode_ids: dict[str, list[str]] = defaultdict(list)
    for manifest_path in opponent_decks.rglob("deck_manifest.json"):
        summary_path = manifest_path.parent / "deck_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        deck_hash = str(summary.get("deck_hash") or "")
        card_counts = {str(card_id): int(count) for card_id, count in (summary.get("card_counts") or {}).items()}
        name_counts = {str(name): int(count) for name, count in (summary.get("name_counts") or {}).items()}
        groups = _groups_for_names(name_counts, wanted)
        if not deck_hash or not card_counts or not groups:
            continue
        artifacts[deck_hash] += 1
        episode_id = str(summary.get("episode_id") or "")
        if episode_id:
            episodes[deck_hash].add(episode_id)
            if episode_id not in episode_ids[deck_hash]:
                episode_ids[deck_hash].append(episode_id)
        team_name = str(summary.get("team_name") or "")
        if team_name and team_name not in players[deck_hash]:
            players[deck_hash].append(team_name)
        primary = _primary_group(groups)
        entry = {
            "deck_hash": deck_hash,
            "archetype": ARCHETYPE_LABELS.get(primary or "", "Unknown"),
            "variant_label": _variant_label(name_counts, ARCHETYPE_LABELS.get(primary or "", "Unknown")),
            "classification_status": "mined_archive",
            "confidence": 1.0,
            "representative_score": 1.0,
            "artifact_count": 1,
            "episode_count": 1,
            "split": "archive",
            "card_counts": card_counts,
            "name_counts": name_counts,
            "source_groups": sorted(groups),
            "representative_player_ids": [],
            "representative_episode_ids": [],
            "reason_selected": "local mined deck_manifest archetype match",
        }
        if deck_hash not in mined:
            mined[deck_hash] = entry
        else:
            mined[deck_hash] = _merge_entry(mined[deck_hash], entry)
    for deck_hash, entry in mined.items():
        entry["artifact_count"] = artifacts[deck_hash]
        entry["episode_count"] = len(episodes[deck_hash])
        entry["representative_score"] = artifacts[deck_hash] + 0.01 * len(episodes[deck_hash])
        entry["representative_player_ids"] = players[deck_hash][:12]
        entry["representative_episode_ids"] = episode_ids[deck_hash][:12]

    return mined


def _select_mined(mined: dict[str, dict[str, Any]], max_per_group: int) -> dict[str, dict[str, Any]]:
    if max_per_group <= 0:
        return mined
    selected: dict[str, dict[str, Any]] = {}
    for group in DEFAULT_GROUPS:
        group_rows = [
            row for row in mined.values()
            if group in set(row.get("source_groups") or [])
        ]
        group_rows.sort(key=lambda row: (-int(row["artifact_count"]), -int(row["episode_count"]), row["deck_hash"]))
        for row in group_rows[:max_per_group]:
            selected[row["deck_hash"]] = row

    return selected


def _deck_zoo_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in sorted(entries.values(), key=lambda item: (item["archetype"], item["variant_label"], item["deck_hash"])):
        rows.append(
            {
                "deck_hash": entry["deck_hash"],
                "archetype": entry["archetype"],
                "variant_label": entry["variant_label"],
                "classification_status": entry["classification_status"],
                "confidence": entry["confidence"],
                "representative_score": entry["representative_score"],
                "artifact_count": entry["artifact_count"],
                "episode_count": entry["episode_count"],
                "deck_count_in_cluster": 1,
                "is_canonical": True,
                "is_near_duplicate": False,
                "cluster_id": entry["deck_hash"],
                "split": entry["split"],
                "card_count": sum(int(count) for count in entry["card_counts"].values()),
                "core_cards": "[]",
                "package_cards": "[]",
                "staple_cards": "[]",
                "tech_cards": "[]",
                "representative_player_ids": _json_cell(entry.get("representative_player_ids") or []),
                "representative_episode_ids": _json_cell(entry.get("representative_episode_ids") or []),
                "reason_selected": entry["reason_selected"],
                "source_groups": _json_cell(entry.get("source_groups") or []),
            }
        )

    return rows


def _decklist_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in sorted(entries.values(), key=lambda item: (item["archetype"], item["deck_hash"])):
        rows.append(
            {
                "deck_hash": entry["deck_hash"],
                "archetype": entry["archetype"],
                "split": entry["split"],
                "is_canonical": True,
                "card_counts": entry["card_counts"],
            }
        )

    return rows


def _summary_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries.values():
        by_archetype[entry["archetype"]].append(entry)
    rows: list[dict[str, Any]] = []
    for archetype, group in by_archetype.items():
        rows.append(
            {
                "archetype": archetype,
                "deck_count": len(group),
                "canonical_count": len(group),
                "artifact_count": sum(int(row["artifact_count"]) for row in group),
                "episode_count": sum(int(row["episode_count"]) for row in group),
                "representative_deck_hashes": _json_cell([row["deck_hash"] for row in sorted(group, key=lambda item: (-int(item["artifact_count"]), item["deck_hash"]))[:12]]),
            }
        )
    rows.sort(key=lambda row: (-int(row["artifact_count"]), row["archetype"]))

    return rows


def _reference_rows(entries: dict[str, dict[str, Any]], wanted: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in DEFAULT_GROUPS:
        if group not in wanted:
            continue
        candidates = [
            row for row in entries.values()
            if group in set(row.get("source_groups") or [])
        ]
        if group == "latias":
            clean_candidates = [
                row for row in candidates
                if set(row.get("source_groups") or []) <= {group, "fresh_top"}
            ]
            if clean_candidates:
                candidates = clean_candidates
        candidates.sort(key=lambda row: (-int(row["artifact_count"]), -int(row["episode_count"]), row["deck_hash"]))
        if not candidates:
            continue
        best = candidates[0]
        rows.append(
            {
                "reference_group": group,
                "deck_hash": best["deck_hash"],
                "archetype": best["archetype"],
                "variant_label": best["variant_label"],
                "artifact_count": best["artifact_count"],
                "episode_count": best["episode_count"],
                "source_groups": _json_cell(best.get("source_groups") or []),
            }
        )

    return rows


def _render_report(payload: dict[str, Any], references: list[dict[str, Any]]) -> str:
    lines = [
        "# R12 Archive Deck Zoo",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- base_deck_zoo: `{payload['inputs']['base_deck_zoo']}`",
        f"- opponent_decks: `{payload['inputs']['opponent_decks']}`",
        f"- decks: `{payload['counts']['decks']}`",
        "",
        "## Reference Opponent Decks",
        "",
        "| group | deck_hash | archetype | variant | artifacts | episodes |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in references:
        lines.append(
            f"| {row['reference_group']} | `{row['deck_hash']}` | {row['archetype']} | "
            f"{row['variant_label']} | {row['artifact_count']} | {row['episode_count']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is deck-zoo-compatible glue over local mined `deck_summary.json` and the fresh top-player deck zoo.",
            "- It does not mutate root `deck.csv`, root `main.py`, `cg/`, `deepmew/submissions/`, or default `v2.4`.",
            "- Reference rows are deck references, not distinct live opponent-policy implementations.",
        ]
    )

    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {str(group) for group in args.group}
    if unknown := sorted(wanted - set(DEFAULT_GROUPS)):
        raise ValueError(f"unknown group(s): {', '.join(unknown)}")
    entries = _base_entries(Path(args.base_deck_zoo), wanted) if args.include_base else {}
    mined = _select_mined(_scan_mined(Path(args.opponent_decks), wanted), int(args.max_per_group))
    for deck_hash, entry in mined.items():
        if deck_hash in entries:
            entries[deck_hash] = _merge_entry(entries[deck_hash], entry)
        else:
            entries[deck_hash] = entry
    deck_zoo_rows = _deck_zoo_rows(entries)
    decklist_rows = _decklist_rows(entries)
    summary_rows = _summary_rows(entries)
    reference_rows = _reference_rows(entries, wanted)
    _write_csv(out_dir / "deck_zoo.csv", deck_zoo_rows, DECK_ZOO_FIELDS)
    _write_csv(out_dir / "deck_zoo_summary.csv", summary_rows)
    _write_csv(out_dir / "reference_opponents.csv", reference_rows)
    _write_jsonl(out_dir / "deck_zoo_decklists.jsonl", decklist_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "base_deck_zoo": str(args.base_deck_zoo),
            "opponent_decks": str(args.opponent_decks),
        },
        "config": {
            "groups": list(args.group),
            "max_per_group": int(args.max_per_group),
            "include_base": bool(args.include_base),
        },
        "counts": {
            "decks": len(entries),
            "base_decks": len(_base_entries(Path(args.base_deck_zoo), wanted)) if args.include_base else 0,
            "mined_decks_selected": len(mined),
            "reference_opponents": len(reference_rows),
            "by_archetype": dict(Counter(row["archetype"] for row in deck_zoo_rows)),
        },
        "reference_opponents": reference_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "deck_zoo_report.md").write_text(_render_report(payload, reference_rows), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": payload["counts"], "reference_opponents": reference_rows}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an R12 deck-zoo-compatible archive for named archetype experiments.")
    parser.add_argument("--base-deck-zoo", required=True)
    parser.add_argument("--opponent-decks", default=".research/opponent_decks")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group", action="append", choices=DEFAULT_GROUPS, default=list(DEFAULT_GROUPS))
    parser.add_argument("--max-per-group", type=int, default=0, help="0 keeps all unique mined hashes per group.")
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    build(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
