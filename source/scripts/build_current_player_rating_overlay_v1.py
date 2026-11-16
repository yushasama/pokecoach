"""Build the current external-rating replay overlay without touching replay assets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from shutil import which
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.current_rating import (
    CurrentRatingError,
    ReplayIdentityEvidence,
    assign_bucket,
    behavior_weighting_spec,
    canonical_json_sha256,
    compare_overlays,
    freeze_rating_buckets,
    resolve_replay_identity,
    summarize_rating_history,
    utc_now_text,
    validate_leaderboard,
)
from deepmew.kaggle_replays.download import (
    DEFAULT_COMPETITION,
    _default_runner,
    ensure_kaggle_cli,
    kaggle_config_path,
    resolve_kaggle_executable,
)
from deepmew.kaggle_replays.scout import LeaderboardRow, _fetch_leaderboard

RESEARCH_ROOT = Path(".research")
MISSION_ROOT = RESEARCH_ROOT / "current_rating_bucket_exploiter_league_and_hybrid_v1" / "rating"
CANONICAL_ROOT = RESEARCH_ROOT / "centralized_causal_replay_contract_and_derived_v1_20260714"
FULL_AUDIT_GAME_STATS = CANONICAL_ROOT / "FULL_AUDIT_GAME_STATS.csv.gz"
DERIVED_MANIFEST = CANONICAL_ROOT / "DEEPMEW_DERIVED_REPLAY_V1_MANIFEST.json"
FULL_BOARD_MAX_RANK = 10**9
EPISODE_ID_RE = re.compile(r"episode-(\d+)-replay\.json$", re.IGNORECASE)
TEAM_NAMES_KEY_RE = re.compile(r'"TeamNames"\s*:\s*')
EPISODE_TIME_RE = re.compile(r"^\s*(\d{7,})\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)", re.MULTILINE)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_leaderboard(path: Path) -> list[LeaderboardRow]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                rows.append(
                    LeaderboardRow(
                        rank=int(record["rank"]),
                        row_index=int(record.get("row_index", record["rank"])),
                        team_id=str(record["team_id"]),
                        team_name=str(record["team_name"]),
                        score=str(record["score"]),
                        submission_date=str(record["submission_date"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CurrentRatingError(f"invalid leaderboard JSONL line {line_number}: {exc}") from exc

    return rows


def _fetch_current_leaderboard(kaggle_exe: str | None) -> tuple[list[LeaderboardRow], list[dict[str, Any]]]:
    kaggle = resolve_kaggle_executable(
        kaggle_exe,
        dry_run=False,
        path_lookup=which,
        config_path=kaggle_config_path(RESEARCH_ROOT),
    )
    ensure_kaggle_cli(kaggle, runner=_default_runner)
    rows, pages, failure = _fetch_leaderboard(
        kaggle,
        DEFAULT_COMPETITION,
        runner=_default_runner,
        max_rank=FULL_BOARD_MAX_RANK,
        max_pages=100,
    )
    if failure is not None:
        raise CurrentRatingError(f"full leaderboard fetch failed: return code {failure.returncode}")

    return rows, pages


def _episode_times(stdout: str) -> dict[str, str]:
    return {episode_id: timestamp.replace(" ", "T") + "Z" for episode_id, timestamp in EPISODE_TIME_RE.findall(stdout)}


def _add_history(
    histories: dict[str, list[dict[str, Any]]],
    records: Iterable[dict[str, Any]],
    timestamp: str | None,
) -> None:
    if not timestamp:
        return
    for record in records:
        team_id = str(record.get("team_id") or "").strip()
        try:
            score = float(record["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if team_id:
            histories[team_id].append({"snapshot_timestamp_utc": timestamp, "score": score})


def scan_identity_provenance() -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    aliases: dict[str, set[str]] = defaultdict(set)
    episode_team_ids: dict[str, set[str]] = defaultdict(set)
    replay_timestamps: dict[str, str] = {}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: list[str] = []
    for path in sorted((RESEARCH_ROOT / "kaggle_scout").glob("**/manifest.json")):
        manifest = _read_json(path)
        sources.append(str(path))
        timestamp = manifest.get("created_at_utc")
        selected_teams = manifest.get("selected_teams", [])
        _add_history(histories, selected_teams, timestamp)
        for record in selected_teams:
            team_id = str(record.get("team_id") or "")
            team_name = str(record.get("team_name") or "").strip()
            if team_id and team_name:
                aliases[team_id].add(team_name)
        submission_to_teams: dict[str, set[str]] = defaultdict(set)
        for record in manifest.get("team_submissions", []):
            team_id = str(record.get("team_id") or "")
            team_name = str(record.get("team_name") or "").strip()
            if team_id and team_name:
                aliases[team_id].add(team_name)
            for submission_id in record.get("submission_ids", []):
                submission_to_teams[str(submission_id)].add(team_id)
        for record in manifest.get("submission_episode_records", []):
            submission_id = str(record.get("submission_id") or "")
            times = _episode_times(str((record.get("result") or {}).get("stdout") or ""))
            for episode_id in record.get("episode_ids", []):
                episode_id = str(episode_id)
                episode_team_ids[episode_id].update(submission_to_teams.get(submission_id, set()))
                if episode_id in times:
                    replay_timestamps[episode_id] = max(replay_timestamps.get(episode_id, ""), times[episode_id])
    for path in sorted((RESEARCH_ROOT / "elo_bucket_scout").glob("**/bucket_teams.json")):
        metadata = _read_json(path)
        sources.append(str(path))
        timestamp = metadata.get("created_at_utc")
        sampled_teams = metadata.get("sampled_teams", [])
        _add_history(histories, sampled_teams, timestamp)
        for record in sampled_teams:
            team_id = str(record.get("team_id") or "")
            team_name = str(record.get("team_name") or "").strip()
            if team_id and team_name:
                aliases[team_id].add(team_name)
            for episode_id in record.get("episode_ids", []):
                episode_team_ids[str(episode_id)].add(team_id)

    return aliases, episode_team_ids, replay_timestamps, histories, sources


def clean_kaggle_games() -> dict[str, Path]:
    games: dict[str, Path] = {}
    with gzip.open(FULL_AUDIT_GAME_STATS, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_id"] != "kaggle_ingest":
                continue
            if int(row["rejected"]) or int(row["unresolved_critic_only"]):
                continue
            if int(row["decisions"]) != int(row["valid_actor_and_critic"]) + int(row["forced_critic_only"]):
                continue
            games[row["canonical_game_id"]] = Path(row["source_artifact_path"])

    return games


def derived_decision_counts(game_ids: set[str]) -> Counter[tuple[str, int]]:
    manifest = _read_json(DERIVED_MANIFEST)
    counts: Counter[tuple[str, int]] = Counter()
    for shard_path in sorted(Path(manifest["shards_root"]).glob("*.jsonl.gz")):
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                game_id = str(row["source_game_id"])
                if game_id in game_ids:
                    counts[(game_id, int(row["actor_seat"]))] += 1

    return counts


def replay_team_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(131_072)
    match = TEAM_NAMES_KEY_RE.search(prefix)
    if not match:
        raise CurrentRatingError(f"TeamNames not found in replay prefix: {path}")
    try:
        names, _end_index = json.JSONDecoder().raw_decode(prefix, match.end())
    except json.JSONDecodeError as exc:
        raise CurrentRatingError(f"TeamNames is not valid JSON in replay prefix: {path}") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise CurrentRatingError(f"TeamNames is not a string list: {path}")

    return names


def build_identity_evidence(
    games: dict[str, Path],
    decision_counts: Counter[tuple[str, int]],
    episode_team_ids: dict[str, set[str]],
    replay_timestamps: dict[str, str],
) -> list[ReplayIdentityEvidence]:
    evidence = []
    for game_id, source_path in sorted(games.items()):
        match = EPISODE_ID_RE.search(source_path.name)
        if not match:
            raise CurrentRatingError(f"canonical Kaggle source path has no episode id: {source_path}")
        episode_id = match.group(1)
        names = replay_team_names(source_path)
        for actor_seat, name in enumerate(names):
            evidence.append(
                ReplayIdentityEvidence(
                    source_game_id=game_id,
                    actor_seat=actor_seat,
                    replay_team_name=name,
                    candidate_team_ids=tuple(sorted(episode_team_ids.get(episode_id, set()), key=int)),
                    decision_count=decision_counts.get((game_id, actor_seat), 0),
                    replay_timestamp_utc=replay_timestamps.get(episode_id),
                )
            )

    return evidence


def aggregate_players(
    resolved_rows: list[dict[str, Any]],
    leaderboard_by_team: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    snapshot_timestamp: str,
) -> list[dict[str, Any]]:
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved_rows:
        if row["team_id"] and row["team_id"] in leaderboard_by_team:
            by_team[row["team_id"]].append(row)
    players = []
    for team_id, rows in by_team.items():
        ladder = leaderboard_by_team[team_id]
        observations = list(histories.get(team_id, []))
        observations.append({"snapshot_timestamp_utc": snapshot_timestamp, "score": ladder["score"]})
        replay_timestamps = [row["replay_timestamp_utc"] for row in rows if row["replay_timestamp_utc"]]
        players.append({
            "canonical_player_id": f"kaggle-team:{team_id}",
            "team_id": team_id,
            "canonical_player_name": ladder["team_name"],
            "source_identity": "KAGGLE_COMPETITION_TEAM_ID",
            "mapping_confidence": 1.0,
            "current_rating_classification": "CURRENT_RATING_RESOLVED",
            "mapping_rule": "DIRECT_EPISODE_SUBMISSION_TEAM_ID_AND_EXACT_REPLAY_SEAT_NAME",
            "current_external_rating": ladder["score"],
            "current_rank": ladder["rank"],
            "current_bucket": None,
            "rating_uncertainty": "EXACT_FULL_BOARD_SNAPSHOT_NO_PER_TEAM_INTERVAL",
            "rating_staleness": {"snapshot_timestamp_utc": snapshot_timestamp, "age_seconds_at_build": 0},
            "rating_history": summarize_rating_history(observations),
            "replay_count": len({row["source_game_id"] for row in rows}),
            "decision_count": sum(int(row["decision_count"]) for row in rows),
            "most_recent_replay_timestamp_utc": max(replay_timestamps) if replay_timestamps else None,
            "most_recent_replay_timestamp_status": "MEASURED" if replay_timestamps else "NOT_INSTRUMENTED",
            "identity_provenance": {
                "source_games": sorted({row["source_game_id"] for row in rows}),
                "observed_replay_names": sorted({row["replay_team_name"] for row in rows}),
            },
        })
    players.sort(key=lambda player: (int(player["current_rank"]), int(player["team_id"])))

    return players


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_overlay(args: argparse.Namespace) -> dict[str, Any]:
    if args.leaderboard_jsonl:
        if not args.snapshot_timestamp_utc:
            raise CurrentRatingError("--snapshot-timestamp-utc is required with --leaderboard-jsonl")
        leaderboard_rows = _read_jsonl_leaderboard(args.leaderboard_jsonl)
        page_receipts: list[dict[str, Any]] = []
        snapshot_timestamp = args.snapshot_timestamp_utc
        acquisition = "PROVIDED_IMMUTABLE_JSONL"
    else:
        leaderboard_rows, page_receipts = _fetch_current_leaderboard(args.kaggle_exe)
        snapshot_timestamp = utc_now_text()
        acquisition = "LIVE_KAGGLE_CLI_FULL_BOARD"
    validated = validate_leaderboard(leaderboard_rows)
    full_ladder = [row.to_dict() for row in validated]
    leaderboard_by_team = {row["team_id"]: row for row in full_ladder}
    aliases, episode_team_ids, replay_timestamps, histories, identity_sources = scan_identity_provenance()
    for row in full_ladder:
        aliases[row["team_id"]].add(row["team_name"])
    games = clean_kaggle_games()
    decision_counts = derived_decision_counts(set(games))
    evidence = build_identity_evidence(games, decision_counts, episode_team_ids, replay_timestamps)
    resolved = [resolve_replay_identity(row, aliases) for row in evidence]
    players = aggregate_players(resolved, leaderboard_by_team, histories, snapshot_timestamp)
    bucket_contract = freeze_rating_buckets(players, full_ladder)
    for player in players:
        player["current_bucket"] = assign_bucket(
            float(player["current_external_rating"]),
            bucket_contract["thresholds_ascending"],
        )
    previous = _read_json(args.previous_overlay) if args.previous_overlay else None
    refresh = compare_overlays(previous, players)
    refresh.update({
        "schema_version": "deepmew.current_rating_refresh_contract.v1",
        "snapshot_retention": "APPEND_ONLY; never overwrite prior snapshots",
        "material_change_review_thresholds": {
            "bucket_migration_share": 0.10,
            "bucket_support_player_change_share": 0.10,
            "weighted_effective_sample_size_change_share": 0.15,
            "bucket_support_failure": "ANY",
        },
        "automatic_retraining": "FORBIDDEN",
    })
    resolution_counts = Counter(row["status"] for row in resolved)
    overlay = {
        "schema_version": "deepmew.current_player_rating_overlay.v1",
        "overlay_version": snapshot_timestamp,
        "snapshot_timestamp_utc": snapshot_timestamp,
        "external_rating_scale_boundary": bucket_contract["rating_scale_boundary"],
        "players": players,
        "identity_resolution": {
            "counts": dict(sorted(resolution_counts.items())),
            "unresolved_or_ambiguous": [row for row in resolved if not row["team_id"]],
            "name_only_matching": "FORBIDDEN_FOR_CANONICAL_INCLUSION",
        },
        "provenance": {
            "full_audit_game_stats": str(FULL_AUDIT_GAME_STATS),
            "derived_replay_manifest": str(DERIVED_MANIFEST),
            "identity_sources": identity_sources,
        },
    }
    overlay["overlay_sha256"] = canonical_json_sha256(overlay)
    snapshot_receipt = {
        "schema_version": "deepmew.current_external_rating_snapshot_receipt.v1",
        "snapshot_timestamp_utc": snapshot_timestamp,
        "acquisition": acquisition,
        "competition": DEFAULT_COMPETITION,
        "row_count": len(full_ladder),
        "canonical_rows_sha256": canonical_json_sha256(full_ladder),
        "strict_validation": "PASS",
        "page_receipts": page_receipts,
    }
    weighting = behavior_weighting_spec()
    output_root = args.output_root / snapshot_timestamp.replace(":", "-")
    if output_root.exists():
        raise CurrentRatingError(f"append-only output directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    snapshot_path = output_root / "CURRENT_EXTERNAL_RATING_SNAPSHOT_V1.jsonl"
    _write_jsonl(snapshot_path, full_ladder)
    snapshot_receipt["snapshot_jsonl_file_sha256"] = _file_sha256(snapshot_path)
    _write_json(output_root / "CURRENT_EXTERNAL_RATING_SNAPSHOT_RECEIPT_V1.json", snapshot_receipt)
    _write_json(output_root / "CURRENT_PLAYER_RATING_OVERLAY_V1.json", overlay)
    _write_json(output_root / "CURRENT_EXTERNAL_BUCKET_BOUNDARIES_V1.json", bucket_contract)
    _write_json(output_root / "CURRENT_BEHAVIOR_WEIGHTING_SPEC_V1.json", weighting)
    _write_json(output_root / "CURRENT_RATING_REFRESH_CONTRACT_V1.json", refresh)
    report = (
        "# Current rating identity and provenance\n\n"
        f"- Status: COMPLETE_PREREQUISITES_ONLY_NO_TRAINING\n"
        f"- Snapshot: `{snapshot_timestamp}` ({len(full_ladder)} full-board teams)\n"
        f"- Replay-linked canonical players: {len(players)}\n"
        f"- Resolution counts: `{dict(sorted(resolution_counts.items()))}`\n"
        f"- Buckets: `{bucket_contract['replay_linked_support']}`\n"
        "- Identity rule: direct episode/submission team ID plus exact replay seat name; name-only matches excluded.\n"
        "- Rating boundary: external Kaggle ratings are separate from internal panel Bradley-Terry values.\n"
        "- Replay/observation assets were read-only; no episodes were fetched and no replay was rewritten.\n"
        "- Training status: not run; weighting spec is frozen prerequisite only.\n"
    )
    (output_root / "RATING_IDENTITY_AND_PROVENANCE_REPORT.md").write_text(report, encoding="utf-8")

    return {"output_root": str(output_root), "players": len(players), "full_ladder_rows": len(full_ladder)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-exe")
    parser.add_argument("--leaderboard-jsonl", type=Path)
    parser.add_argument("--snapshot-timestamp-utc")
    parser.add_argument("--previous-overlay", type=Path)
    parser.add_argument("--output-root", type=Path, default=MISSION_ROOT)

    return parser.parse_args()


def main() -> None:
    result = build_overlay(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
