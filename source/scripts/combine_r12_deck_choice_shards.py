from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "deepmew.r12.deck_choice_shard_combine.v0"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
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


def _as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row.get(key) or 0))


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key) or 0.0)


def _winrate_se(wins: int, losses: int) -> float | None:
    decided = wins + losses
    if decided <= 0:
        return None
    p = wins / decided

    return math.sqrt(p * (1.0 - p) / decided)


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key) or "") for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in grouped.items():
        games = sum(_as_int(row, "games") for row in group)
        completed = sum(_as_int(row, "completed") for row in group)
        wins = sum(_as_int(row, "wins") for row in group)
        losses = sum(_as_int(row, "losses") for row in group)
        draws = sum(_as_int(row, "draws") for row in group)
        crashes = sum(_as_int(row, "crashes") for row in group)
        timeouts = sum(_as_int(row, "timeouts") for row in group)
        avg_turns_num = sum(_as_float(row, "avg_turns") * _as_int(row, "completed") for row in group)
        decided = wins + losses
        first = group[0]
        payload = {name: value for name, value in zip(group_keys, key, strict=True)}
        for field in (
            "pilot_label",
            "pilot_deck_hash",
            "pilot_policy_id",
            "pilot_archetype",
            "pilot_variant_label",
            "opponent_policy",
            "opponent_deck_hash",
            "opponent_archetype",
            "opponent_variant_label",
        ):
            if field not in payload and field in first:
                payload[field] = first[field]
        payload.update(
            {
                "pilot_decks": len({row["pilot_deck_hash"] for row in group}),
                "opponent_decks": len({row["opponent_deck_hash"] for row in group}),
                "games": games,
                "completed": completed,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winrate": round(wins / decided, 6) if decided else None,
                "winrate_se": round(_winrate_se(wins, losses) or 0.0, 6) if decided else None,
                "crashes": crashes,
                "timeouts": timeouts,
                "avg_turns": round(avg_turns_num / completed, 3) if completed else None,
            }
        )
        out.append(payload)

    return out


def _reference_delta(by_pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = next((row for row in by_pilot if row["pilot_label"] == "reference_abomasnow"), None)
    if reference is None or reference["winrate"] is None:
        return []
    reference_wr = float(reference["winrate"])
    rows: list[dict[str, Any]] = []
    for row in by_pilot:
        if row["winrate"] is None:
            continue
        rows.append(
            {
                "pilot_label": row["pilot_label"],
                "pilot_deck_hash": row["pilot_deck_hash"],
                "pilot_archetype": row["pilot_archetype"],
                "pilot_variant_label": row["pilot_variant_label"],
                "winrate": row["winrate"],
                "reference_abomasnow_winrate": reference_wr,
                "delta_vs_reference_abomasnow": round(float(row["winrate"]) - reference_wr, 6),
                "wins_losses_draws": f"{row['wins']}-{row['losses']}-{row['draws']}",
            }
        )
    rows.sort(key=lambda row: (-float(row["delta_vs_reference_abomasnow"]), row["pilot_label"]))

    return rows


def _sort_by_winrate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-(float(row["winrate"]) if row["winrate"] is not None else -1.0), row.get("pilot_label", ""), row.get("pilot_archetype", "")))


def _render_report(payload: dict[str, Any], by_pilot: list[dict[str, Any]], by_archetype: list[dict[str, Any]], by_opponent: list[dict[str, Any]]) -> str:
    top = _sort_by_winrate(by_pilot)[:20]
    weak = list(reversed(_sort_by_winrate(by_pilot)[-12:]))
    archetypes = _sort_by_winrate(by_archetype)
    lines = [
        "# R12 Reference Deck Panel",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- shard_root: `{payload['inputs']['shard_root']}`",
        f"- matchup_rows: `{payload['counts']['matchup_rows']}` / `{payload['counts']['expected_matchup_rows']}`",
        f"- games: `{payload['counts']['games']}`",
        f"- safety: crashes `{payload['safety']['crashes']}`, timeouts `{payload['safety']['timeouts']}`, draws `{payload['safety']['draws']}`",
        "",
        "## Read This Correctly",
        "",
        "- R4-F is the pilot for every candidate deck; only the pilot deck changes.",
        "- Opponent pilot is local `first-legal` on a fixed reference deck panel.",
        "- Reference rows are deck references; rebuilt `reference_*` labels are not distinct live opponent bots.",
        "",
        "## Top Pilot Decks",
        "",
        "| rank | pilot | archetype | winrate | SE | wins-losses-draws |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(top, start=1):
        lines.append(
            f"| {index} | `{row['pilot_label']}` | {row['pilot_archetype']} | {row['winrate']} | "
            f"{row['winrate_se']} | {row['wins']}-{row['losses']}-{row['draws']} |"
        )
    lines.extend(
        [
            "",
            "## By Archetype",
            "",
            "| archetype | decks | winrate | SE | wins-losses-draws |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in archetypes:
        lines.append(
            f"| {row['pilot_archetype']} | {row['pilot_decks']} | {row['winrate']} | {row['winrate_se']} | "
            f"{row['wins']}-{row['losses']}-{row['draws']} |"
        )
    lines.extend(
        [
            "",
            "## Opponent Panel",
            "",
            "| opponent deck | archetype | pilot winrate | wins-losses-draws |",
            "|---|---|---:|---:|",
        ]
    )
    for row in sorted(by_opponent, key=lambda item: item["opponent_deck_hash"]):
        lines.append(
            f"| `{row['opponent_deck_hash']}` | {row['opponent_archetype']} | {row['winrate']} | "
            f"{row['wins']}-{row['losses']}-{row['draws']} |"
        )
    lines.extend(
        [
            "",
            "## Weakest Pilot Decks",
            "",
            "| pilot | archetype | winrate | wins-losses-draws |",
            "|---|---|---:|---:|",
        ]
    )
    for row in weak:
        lines.append(
            f"| `{row['pilot_label']}` | {row['pilot_archetype']} | {row['winrate']} | "
            f"{row['wins']}-{row['losses']}-{row['draws']} |"
        )

    return "\n".join(lines) + "\n"


def combine(args: argparse.Namespace) -> dict[str, Any]:
    shard_root = Path(args.shard_root)
    out_dir = Path(args.out_dir or shard_root / "combined")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard in sorted(shard_root.glob("shard_*")):
        matchup_path = shard / "by_matchup.csv"
        summary_path = shard / "summary.json"
        if matchup_path.exists():
            rows.extend(_read_csv(matchup_path))
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    by_pilot = _sort_by_winrate(_aggregate(rows, ["pilot_label"]))
    by_archetype = _sort_by_winrate(_aggregate(rows, ["pilot_archetype"]))
    by_opponent = _aggregate(rows, ["opponent_deck_hash"])
    by_archetype_opponent = _aggregate(rows, ["pilot_archetype", "opponent_deck_hash"])
    deltas = _reference_delta(by_pilot)
    safety = {
        "crashes": sum(_as_int(row, "crashes") for row in rows),
        "timeouts": sum(_as_int(row, "timeouts") for row in rows),
        "draws": sum(_as_int(row, "draws") for row in rows),
    }
    expected = sum(int((summary.get("counts") or {}).get("expected_matchup_rows") or 0) for summary in summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {"shard_root": str(shard_root)},
        "counts": {
            "shards": len(summaries),
            "matchup_rows": len(rows),
            "expected_matchup_rows": expected,
            "complete": len(rows) == expected,
            "pilot_decks": len({row["pilot_deck_hash"] for row in rows}),
            "opponent_decks": len({row["opponent_deck_hash"] for row in rows}),
            "games": sum(_as_int(row, "games") for row in rows),
            "completed": sum(_as_int(row, "completed") for row in rows),
        },
        "safety": safety,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(out_dir / "by_matchup.csv", rows)
    _write_csv(out_dir / "by_pilot.csv", by_pilot)
    _write_csv(out_dir / "by_archetype.csv", by_archetype)
    _write_csv(out_dir / "by_opponent.csv", by_opponent)
    _write_csv(out_dir / "by_archetype_opponent.csv", by_archetype_opponent)
    _write_csv(out_dir / "delta_vs_reference_abomasnow.csv", deltas)
    (out_dir / "report.md").write_text(_render_report(payload, by_pilot, by_archetype, by_opponent), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": payload["counts"], "safety": safety, "top5": by_pilot[:5]}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine sharded R12 deck-choice panel outputs.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    combine(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
