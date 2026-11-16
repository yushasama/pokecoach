from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".research" / "r12" / "lane3_rating"
SCHEMA_VERSION = "deepmew.ladder_rating_log.rebuilt.v1"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("submissions") or []
            return [row for row in rows if isinstance(row, dict)]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    return []


def _read_csv_text(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    return list(csv.DictReader(text.splitlines()))


def _match_submission(rows: list[dict[str, Any]], submission_id: str) -> dict[str, Any] | None:
    for row in rows:
        values = {str(value) for value in row.values()}
        if submission_id in values:
            return row

    return None


def _run_live_kaggle(competition: str) -> tuple[bool, str, str]:
    command = ["kaggle", "competitions", "submissions", "-c", competition, "--csv"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=60)

    return proc.returncode == 0, proc.stdout, proc.stderr


def _slot_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = payload.get("slots") if isinstance(payload, dict) else None
    if not isinstance(slots, list):
        raise ValueError(f"slots config must contain a slots list: {path}")

    return [slot for slot in slots if isinstance(slot, dict)]


def render_report(payload: dict[str, Any]) -> str:
    rating = payload.get("rating") or {}
    lines = [
        "# Ladder Rating Log",
        "",
        f"- status: `{payload['status']}`",
        f"- submission_id: `{payload['submission_id']}`",
        f"- source: `{payload.get('source')}`",
        f"- rating: `{rating.get('rating')}`",
        f"- episodes: `{rating.get('episodes')}`",
        f"- message: {payload.get('message', '')}",
        "",
    ]

    return "\n".join(lines)


def render_multi_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Ladder Rating Log - R13 Slots",
        "",
        f"- status: `{payload['status']}`",
        f"- source: `{payload.get('source')}`",
        f"- competition: `{payload['competition']}`",
        "",
        "| slot | submission_id | label | status | publicScore | episodes |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for row in payload.get("rows", []):
        rating = row.get("rating") or {}
        lines.append(
            f"| {row.get('slot')} | `{row.get('submission_id')}` | {row.get('label')} | "
            f"`{rating.get('status')}` | {rating.get('publicScore')} | {rating.get('episodes')} |"
        )
    lines.append("")

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    rows = _read_rows(Path(args.source)) if args.source else []
    source = str(args.source) if args.source else "none"
    live_stdout = ""
    live_stderr = ""
    live_ok = None
    if args.live:
        live_ok, live_stdout, live_stderr = _run_live_kaggle(args.competition)
        source = f"kaggle:{args.competition}"
        if live_ok:
            rows = _read_csv_text(live_stdout)
    matched = _match_submission(rows, str(args.submission_id)) if rows else None
    rating = {}
    status = "not_available"
    message = "No cached rating source supplied; use --source with a CSV/JSON export or --live for Kaggle CLI."
    if matched:
        status = "found"
        rating = {
            "rating": matched.get("rating") or matched.get("score") or matched.get("publicScore"),
            "episodes": matched.get("episodes") or matched.get("episode_count") or matched.get("numEpisodes"),
            "raw": matched,
        }
        message = "Submission row found in supplied source."
    elif args.live:
        status = "not_found_in_live_query" if live_ok else "live_query_failed"
        message = "Submission id not found in live Kaggle CSV." if live_ok else "Live Kaggle command failed."
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "submission_id": str(args.submission_id),
        "competition": args.competition,
        "source": source,
        "status": status,
        "rating": rating,
        "message": message,
        "live": {
            "attempted": bool(args.live),
            "ok": live_ok,
            "stdout_prefix": live_stdout[:1000],
            "stderr_prefix": live_stderr[:1000],
        },
    }
    _write_json(out_dir / f"ladder_rating_{args.submission_id}.json", payload)
    _write_text(out_dir / f"ladder_rating_{args.submission_id}.md", render_report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    return payload


def run_multi(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    slots = _slot_rows(Path(args.slots_config))
    rows = _read_rows(Path(args.source)) if args.source else []
    source = str(args.source) if args.source else "none"
    live_stdout = ""
    live_stderr = ""
    live_ok = None
    if args.live:
        live_ok, live_stdout, live_stderr = _run_live_kaggle(args.competition)
        source = f"kaggle:{args.competition}"
        if live_ok:
            rows = _read_csv_text(live_stdout)
    out_rows = []
    for slot in slots:
        submission_id = str(slot.get("submission_id") or "")
        matched = _match_submission(rows, submission_id) if submission_id else None
        rating = {
            "status": matched.get("status") if matched else None,
            "publicScore": matched.get("publicScore") if matched else None,
            "privateScore": matched.get("privateScore") if matched else None,
            "episodes": matched.get("episodes") or matched.get("episode_count") or matched.get("numEpisodes") if matched else None,
            "raw": matched,
        }
        out_rows.append(
            {
                "slot": slot.get("slot"),
                "label": slot.get("label"),
                "submission_id": submission_id,
                "tar_gz": slot.get("tar_gz"),
                "rating": rating,
                "found": matched is not None,
            }
        )
    found = sum(1 for row in out_rows if row["found"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "competition": args.competition,
        "source": source,
        "status": "found_all" if found == len(out_rows) else "partial" if found else "not_found",
        "rows": out_rows,
        "live": {
            "attempted": bool(args.live),
            "ok": live_ok,
            "stdout_prefix": live_stdout[:1000],
            "stderr_prefix": live_stderr[:1000],
        },
    }
    _write_json(out_dir / "ladder_rating_r13_slots.json", payload)
    _write_text(out_dir / "ladder_rating_r13_slots.md", render_multi_report(payload))
    with (out_dir / "ladder_rating_r13_slots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slot", "submission_id", "label", "status", "publicScore", "privateScore", "episodes", "found"])
        writer.writeheader()
        for row in out_rows:
            rating = row["rating"]
            writer.writerow(
                {
                    "slot": row.get("slot"),
                    "submission_id": row.get("submission_id"),
                    "label": row.get("label"),
                    "status": rating.get("status"),
                    "publicScore": rating.get("publicScore"),
                    "privateScore": rating.get("privateScore"),
                    "episodes": rating.get("episodes"),
                    "found": row.get("found"),
                }
            )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log Kaggle ladder rating/episode count for a submission id.")
    parser.add_argument("--submission-id", default="54318639")
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--source", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--slots-config", default="")
    args = parser.parse_args(argv)
    if args.slots_config:
        run_multi(args)
    else:
        run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
