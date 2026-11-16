"""Report on downloaded/ingested Kaggle replay corpora.

K1.6 reporting reads only offline artifacts: replay JSON filenames, K1 ingest
manifests, and extracted opponent deck summaries.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown, write_text
from deepmew.kaggle_replays.episode_sources import REPLAY_FILENAME_RE, unique_first_seen

KAGGLE_CORPUS_KIND = "kaggle_corpus"
DEFAULT_OUT_ROOT = ".research"
DEFAULT_REPLAY_ROOT = ".research/kaggle_replays"
DEFAULT_INGEST_ROOT = ".research/kaggle_ingest"
DEFAULT_OPPONENT_DECKS_ROOT = ".research/opponent_decks"
ARCHETYPE_KEYWORDS = ("Lucario", "Starmie", "Dragapult", "Froslass", "Cinderace", "Dusknoir")


@dataclass(frozen=True)
class CorpusOptions:
    replay_root: Path = Path(DEFAULT_REPLAY_ROOT)
    ingest_root: Path = Path(DEFAULT_INGEST_ROOT)
    opponent_decks_root: Path = Path(DEFAULT_OPPONENT_DECKS_ROOT)
    out_root: Path = Path(DEFAULT_OUT_ROOT)
    section: str = "corpus"
    command: str = ""
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class CorpusResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    deck_hash_summary_path: Path
    player_summary_path: Path
    archetype_hint_summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _episode_id_from_replay(path: Path) -> str | None:
    match = REPLAY_FILENAME_RE.match(path.name)
    return match.group(1) if match else None


def scan_replay_files(root: str | Path) -> list[dict[str, str]]:
    replay_root = Path(root)
    if not replay_root.exists():
        return []
    records: list[dict[str, str]] = []
    for path in sorted(replay_root.rglob("episode-*-replay.json")):
        episode_id = _episode_id_from_replay(path)
        if episode_id:
            records.append({"episode_id": episode_id, "path": str(path)})
    return records


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def scan_ingest_manifests(root: str | Path) -> list[dict[str, Any]]:
    ingest_root = Path(root)
    if not ingest_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(ingest_root.rglob("manifest.json")):
        data = _load_json(path)
        if not data or data.get("kind") != "kaggle_ingest":
            continue
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
        episode_id = str(summary.get("episode_id") or inputs.get("episode_id") or "").strip()
        replay_episode_id = None
        episode_json = inputs.get("episode_json")
        if isinstance(episode_json, str):
            replay_episode_id = _episode_id_from_replay(Path(episode_json))
        if not episode_id:
            continue
        records.append(
            {
                "episode_id": episode_id,
                "replay_episode_id": replay_episode_id or episode_id,
                "path": str(path),
                "manifest": data,
            }
        )
    return records


def scan_deck_summaries(root: str | Path) -> list[dict[str, Any]]:
    deck_root = Path(root)
    if not deck_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(deck_root.rglob("deck_summary.json")):
        data = _load_json(path)
        if not data:
            continue
        data = dict(data)
        data["_path"] = str(path)
        records.append(data)
    return records


def archetype_hint(name_counts: dict[str, Any] | None) -> str:
    if not isinstance(name_counts, dict):
        return "unknown"
    names = " ".join(str(name) for name in name_counts)
    hits = [keyword for keyword in ARCHETYPE_KEYWORDS if keyword.lower() in names.lower()]
    return "/".join(hits) if hits else "unknown"


def _result_by_episode(ingest_records: list[dict[str, Any]]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for record in ingest_records:
        summary = record["manifest"].get("summary", {})
        result = summary.get("result") if isinstance(summary, dict) else None
        out[record["episode_id"]] = result if isinstance(result, int) and not isinstance(result, bool) else None
        out[record["replay_episode_id"]] = out[record["episode_id"]]
    return out


def _deck_hash_rows(deck_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for deck in deck_summaries:
        grouped[str(deck.get("deck_hash") or "unknown")].append(deck)
    rows: list[dict[str, Any]] = []
    for deck_hash, decks in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        hints = Counter(archetype_hint(deck.get("name_counts")) for deck in decks)
        rows.append(
            {
                "deck_hash": deck_hash,
                "deck_count": len(decks),
                "episode_count": len({str(deck.get("episode_id")) for deck in decks}),
                "player_count": len({str(deck.get("team_name")) for deck in decks}),
                "archetype_hint": hints.most_common(1)[0][0] if hints else "unknown",
                "team_names": ";".join(sorted({str(deck.get("team_name") or "") for deck in decks if deck.get("team_name")})),
            }
        )
    return rows


def _player_rows(deck_summaries: list[dict[str, Any]], results_by_episode: dict[str, int | None]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for deck in deck_summaries:
        grouped[str(deck.get("team_name") or f"player_{deck.get('player_index', 'unknown')}")].append(deck)
    rows: list[dict[str, Any]] = []
    for team_name, decks in sorted(grouped.items()):
        wins = losses = draws = 0
        for deck in decks:
            result = results_by_episode.get(str(deck.get("episode_id")))
            player_index = deck.get("player_index")
            if result == 2:
                draws += 1
            elif isinstance(result, int) and player_index == result:
                wins += 1
            elif isinstance(result, int):
                losses += 1
        rows.append(
            {
                "team_name": team_name,
                "deck_count": len(decks),
                "episode_count": len({str(deck.get("episode_id")) for deck in decks}),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "deck_hashes": ";".join(sorted({str(deck.get("deck_hash") or "unknown") for deck in decks})),
                "archetype_hints": ";".join(sorted({archetype_hint(deck.get("name_counts")) for deck in decks})),
            }
        )
    return rows


def _archetype_rows(deck_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for deck in deck_summaries:
        grouped[archetype_hint(deck.get("name_counts"))].append(deck)
    return [
        {
            "archetype_hint": hint,
            "deck_count": len(decks),
            "deck_hash_count": len({str(deck.get("deck_hash") or "unknown") for deck in decks}),
            "episode_count": len({str(deck.get("episode_id")) for deck in decks}),
        }
        for hint, decks in sorted(grouped.items())
    ]


def _action_stats(ingest_records: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"rows": 0, "malformed": 0, "skipped_no_chosen_indices": 0, "select_decisions": 0, "steps_total": 0}
    for record in ingest_records:
        summary = record["manifest"].get("summary", {})
        action_rows = summary.get("action_rows") if isinstance(summary, dict) else None
        if not isinstance(action_rows, dict):
            continue
        for key in stats:
            value = action_rows.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                stats[key] += value
    return stats


def _csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return handle.getvalue()


def report_kaggle_replay_corpus(options: CorpusOptions) -> CorpusResult:
    options = CorpusOptions(
        replay_root=Path(options.replay_root),
        ingest_root=Path(options.ingest_root),
        opponent_decks_root=Path(options.opponent_decks_root),
        out_root=Path(options.out_root),
        section=options.section,
        command=options.command,
        args=options.args,
    )
    run_dir = create_run_dir(kind=KAGGLE_CORPUS_KIND, out_root=options.out_root, suffix=options.section)
    replay_records = scan_replay_files(options.replay_root)
    ingest_records = scan_ingest_manifests(options.ingest_root)
    deck_summaries = scan_deck_summaries(options.opponent_decks_root)

    replay_episode_ids = unique_first_seen(record["episode_id"] for record in replay_records)
    ingested_episode_ids = unique_first_seen(record["replay_episode_id"] for record in ingest_records)
    ingested_episode_id_set = set(ingested_episode_ids)
    missing_ingest_ids = [episode_id for episode_id in replay_episode_ids if episode_id not in ingested_episode_id_set]
    results_by_episode = _result_by_episode(ingest_records)
    deck_hash_rows = _deck_hash_rows(deck_summaries)
    player_rows = _player_rows(deck_summaries, results_by_episode)
    archetype_rows = _archetype_rows(deck_summaries)
    action_stats = _action_stats(ingest_records)

    deck_hash_summary_path = write_text(
        run_dir.output_path("deck_hash_summary.csv"),
        _csv_text(["deck_hash", "deck_count", "episode_count", "player_count", "archetype_hint", "team_names"], deck_hash_rows),
    )
    player_summary_path = write_text(
        run_dir.output_path("player_summary.csv"),
        _csv_text(["team_name", "deck_count", "episode_count", "wins", "losses", "draws", "deck_hashes", "archetype_hints"], player_rows),
    )
    archetype_hint_summary_path = write_text(
        run_dir.output_path("archetype_hint_summary.csv"),
        _csv_text(["archetype_hint", "deck_count", "deck_hash_count", "episode_count"], archetype_rows),
    )

    summary = {
        "replay_file_count": len(replay_records),
        "unique_replay_episode_count": len(replay_episode_ids),
        "ingested_episode_count": len(ingested_episode_ids),
        "missing_ingest_count": len(missing_ingest_ids),
        "deck_artifact_count": len(deck_summaries),
        "unique_deck_hash_count": len({str(deck.get("deck_hash") or "unknown") for deck in deck_summaries}),
        "player_count": len(player_rows),
        "action_rows": action_stats["rows"],
        "action_rows_malformed": action_stats["malformed"],
        "action_rows_skipped_no_chosen_indices": action_stats["skipped_no_chosen_indices"],
    }
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_CORPUS_KIND,
        command=options.command,
        inputs={
            "args": options.args or {},
            "replay_root": str(options.replay_root),
            "ingest_root": str(options.ingest_root),
            "opponent_decks_root": str(options.opponent_decks_root),
            "section": options.section,
        },
        outputs={
            "corpus_report": str(run_dir.output_path("corpus_report.md")),
            "deck_hash_summary": str(deck_hash_summary_path),
            "player_summary": str(player_summary_path),
            "archetype_hint_summary": str(archetype_hint_summary_path),
        },
        summary=summary,
    )
    manifest["replays"] = replay_records
    manifest["ingest_manifests"] = [
        {"episode_id": record["episode_id"], "replay_episode_id": record["replay_episode_id"], "path": record["path"]}
        for record in ingest_records
    ]
    manifest["missing_ingest_episode_ids"] = missing_ingest_ids
    manifest["deck_hash_summary"] = deck_hash_rows
    manifest["player_summary"] = player_rows
    manifest["archetype_hint_summary"] = archetype_rows
    manifest["action_stats"] = action_stats

    report_path = write_markdown(
        run_dir.output_path("corpus_report.md"),
        _report(options, summary, missing_ingest_ids, deck_hash_rows, player_rows, archetype_rows),
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return CorpusResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        deck_hash_summary_path=deck_hash_summary_path,
        player_summary_path=player_summary_path,
        archetype_hint_summary_path=archetype_hint_summary_path,
        manifest=manifest,
        summary=summary,
    )


def _report(
    options: CorpusOptions,
    summary: dict[str, Any],
    missing_ingest_ids: list[str],
    deck_hash_rows: list[dict[str, Any]],
    player_rows: list[dict[str, Any]],
    archetype_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Kaggle Replay Corpus Report",
        "",
        f"replay_root={options.replay_root}",
        f"ingest_root={options.ingest_root}",
        f"opponent_decks_root={options.opponent_decks_root}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    lines += ["", "## Missing Ingest Coverage", ""]
    lines.extend(f"- {episode_id}" for episode_id in missing_ingest_ids) if missing_ingest_ids else lines.append("- None.")
    lines += ["", "## Top Deck Hashes", "", "| deck_hash | count | hint | teams |", "|---|---:|---|---|"]
    for row in deck_hash_rows[:20]:
        lines.append(f"| {row['deck_hash']} | {row['deck_count']} | {row['archetype_hint']} | {row['team_names']} |")
    lines += ["", "## Players", "", "| team | decks | wins | losses | hints |", "|---|---:|---:|---:|---|"]
    for row in player_rows[:50]:
        lines.append(f"| {row['team_name']} | {row['deck_count']} | {row['wins']} | {row['losses']} | {row['archetype_hints']} |")
    lines += ["", "## Archetype Hints", "", "| hint | decks | deck_hashes | episodes |", "|---|---:|---:|---:|"]
    for row in archetype_rows:
        lines.append(f"| {row['archetype_hint']} | {row['deck_count']} | {row['deck_hash_count']} | {row['episode_count']} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- Corpus reporting only; no policy/runtime/gameplay changes and no replay-prior online integration.",
    ]
    return "\n".join(lines)
