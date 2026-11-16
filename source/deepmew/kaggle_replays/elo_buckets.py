"""Score-bucket (ladder ELO) player sampling scout for R16 meta mining.

Samples random players from Kaggle leaderboard *score* buckets (e.g. 600-800,
800-900, ...), collects their public episode ids through the official CLI flow
(leaderboard -> team-submissions -> episodes), and writes per-bucket episode
files for the throttle-aware mining queue plus a versioned bucket-metadata
artifact for downstream analysis.

Artifacts live under ``.research/elo_bucket_scout/`` — deliberately NOT under
``.research/kaggle_scout/`` — because the K1.7 rank join scans every scout
manifest and marks team names with conflicting ranks as AMBIGUOUS; a fresh
full-board snapshot re-ranks previously scouted teams and would strip their
ranks from the existing join. The metadata key is ``sampled_teams`` (not
``selected_teams``) for the same reason.

Offline infrastructure only: no gameplay, policy, deck, or runtime change; the
artifacts are never loaded by the eval path.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Callable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.kaggle_replays.download import (
    DEFAULT_COMPETITION,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPLAY_ROOT,
    KaggleExecutable,
    PathLookup,
    Runner,
    _run_command,
    ensure_kaggle_cli,
    kaggle_config_path,
    resolve_kaggle_executable,
)
from deepmew.kaggle_replays.episode_sources import unique_first_seen
from deepmew.kaggle_replays.scout import (
    LeaderboardRow,
    _default_runner,
    _episodes_command,
    _fetch_leaderboard,
    _team_submissions_command,
    parse_episode_ids_from_episodes_output,
    parse_submission_ids,
)

ELO_BUCKET_SCOUT_KIND = "elo_bucket_scout"
ELO_BUCKET_SCOUT_VERSION = "deepmew.elo_bucket_scout.v1"
DEFAULT_ELO_BUCKET_ROOT = ".research/elo_bucket_scout"
DEFAULT_EPISODE_IDS_ROOT = ".research/kaggle_episode_ids"
DEFAULT_BUCKET_SPECS = ["600-800", "800-900", "900-1000", "1000-1100", "1100-1200", "1200-1300"]

# Practically unbounded: _fetch_leaderboard stops on its own when a page has no
# next-page token, so this only guards against a pathological infinite board.
FULL_BOARD_MAX_RANK = 10**9

_REPLAY_FILE_RE = re.compile(r"^episode-(\d+)-replay\.json$")


class EloBucketScoutError(ValueError):
    """Raised when score-bucket scouting cannot continue."""


@dataclass(frozen=True)
class BucketSpec:
    """Half-open score interval [low, high)."""

    low: float
    high: float
    label: str

    def contains(self, score: float) -> bool:
        return self.low <= score < self.high


@dataclass(frozen=True)
class EloBucketScoutOptions:
    section: str
    competition: str = DEFAULT_COMPETITION
    bucket_specs: list[str] | None = None
    players_per_bucket: int = 60
    seed: int = 20260706
    max_submissions_per_team: int = 1
    max_episodes_per_submission: int = 6
    sleep_seconds: float = 0.0
    kaggle_exe: str | None = None
    out_root: Path = Path(DEFAULT_OUT_ROOT)
    scout_root: Path = Path(DEFAULT_ELO_BUCKET_ROOT)
    episode_ids_root: Path = Path(DEFAULT_EPISODE_IDS_ROOT)
    replay_root: Path = Path(DEFAULT_REPLAY_ROOT)
    dry_run: bool = False
    command: str = ""
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class EloBucketScoutResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    metadata_path: Path
    episode_files: dict[str, Path]
    summary: dict[str, Any]
    metadata: dict[str, Any]


def parse_score_buckets(specs: list[str] | None) -> list[BucketSpec]:
    """Parse ``600-800`` style specs into half-open [low, high) buckets."""
    out: list[BucketSpec] = []
    for raw in specs or DEFAULT_BUCKET_SPECS:
        value = str(raw).strip()
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", value)
        if not match:
            raise EloBucketScoutError(f"score bucket must look like 600-800, got {value!r}")
        low, high = float(match.group(1)), float(match.group(2))
        if high <= low:
            raise EloBucketScoutError(f"score bucket high must exceed low: {value!r}")
        out.append(BucketSpec(low=low, high=high, label=value))
    if not out:
        raise EloBucketScoutError("at least one score bucket is required")
    return out


def _score_of(row: LeaderboardRow) -> float | None:
    try:
        return float(str(row.score).strip())
    except (TypeError, ValueError):
        return None


def bucket_leaderboard(rows: list[LeaderboardRow], buckets: list[BucketSpec]) -> dict[str, list[LeaderboardRow]]:
    """Assign each scored leaderboard row to the first bucket containing its score."""
    out: dict[str, list[LeaderboardRow]] = {bucket.label: [] for bucket in buckets}
    for row in rows:
        score = _score_of(row)
        if score is None:
            continue
        for bucket in buckets:
            if bucket.contains(score):
                out[bucket.label].append(row)
                break
    return out


def sample_bucket_teams(
    bucketed: dict[str, list[LeaderboardRow]],
    *,
    players_per_bucket: int,
    seed: int,
) -> dict[str, list[LeaderboardRow]]:
    """Deterministic per-bucket random sample; buckets at/below the limit are taken whole."""
    if players_per_bucket <= 0:
        raise EloBucketScoutError("players_per_bucket must be positive")
    sampled: dict[str, list[LeaderboardRow]] = {}
    for label in sorted(bucketed):
        rows = sorted(bucketed[label], key=lambda row: (row.rank, row.team_id))
        if len(rows) <= players_per_bucket:
            sampled[label] = rows
            continue
        rng = random.Random(f"{seed}:{label}")
        chosen = rng.sample(rows, players_per_bucket)
        sampled[label] = sorted(chosen, key=lambda row: (row.rank, row.team_id))
    return sampled


def downloaded_episode_ids(replay_root: str | Path = DEFAULT_REPLAY_ROOT) -> set[str]:
    """Episode ids already downloaded in ANY section (cross-section dedupe)."""
    root = Path(replay_root)
    if not root.exists():
        return set()
    found: set[str] = set()
    for path in root.glob("*/episode-*-replay.json"):
        match = _REPLAY_FILE_RE.match(path.name)
        if match:
            found.add(match.group(1))
    return found


def assign_pending_episodes(
    team_records: list[dict[str, Any]],
    *,
    already_downloaded: set[str],
) -> dict[str, list[str]]:
    """First-seen bucket assignment of not-yet-downloaded episode ids.

    An episode is a game between two teams and can be discovered from both
    sides (possibly in different buckets); it is downloaded once, under the
    bucket that saw it first. The team->episode links in the metadata keep the
    full picture regardless of which bucket downloads the file.
    """
    pending: dict[str, list[str]] = {}
    claimed: set[str] = set(already_downloaded)
    for record in team_records:
        bucket = record["bucket"]
        pending.setdefault(bucket, [])
        for episode_id in record["episode_ids"]:
            if episode_id in claimed:
                continue
            claimed.add(episode_id)
            pending[bucket].append(episode_id)
    return pending


def bucket_section_name(section: str, bucket_label: str) -> str:
    return f"{section}-{bucket_label}"


def _collect_team_episodes(
    kaggle: KaggleExecutable,
    options: EloBucketScoutOptions,
    sampled: dict[str, list[LeaderboardRow]],
    *,
    runner: Runner,
    sleep_fn: Callable[[float], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    team_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label in sorted(sampled):
        for row in sampled[label]:
            record: dict[str, Any] = {
                "bucket": label,
                "rank": row.rank,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "score": row.score,
                "submission_ids": [],
                "episode_ids": [],
                "status": "ok",
            }
            result = _run_command(_team_submissions_command(kaggle, options.competition, row.team_id), runner=runner)
            if result.returncode != 0:
                record["status"] = "team_submissions_failed"
                failures.append({"stage": "team_submissions", "team_id": row.team_id, "returncode": result.returncode})
                team_records.append(record)
                continue
            submission_ids = parse_submission_ids(result.stdout)[: options.max_submissions_per_team]
            record["submission_ids"] = submission_ids
            if options.sleep_seconds:
                sleep_fn(options.sleep_seconds)
            episode_ids: list[str] = []
            for submission_id in submission_ids:
                ep_result = _run_command(_episodes_command(kaggle, options.competition, submission_id), runner=runner)
                if ep_result.returncode != 0:
                    record["status"] = "episodes_failed"
                    failures.append({"stage": "episodes", "team_id": row.team_id, "submission_id": submission_id,
                                     "returncode": ep_result.returncode})
                    continue
                episode_ids.extend(parse_episode_ids_from_episodes_output(ep_result.stdout)[: options.max_episodes_per_submission])
                if options.sleep_seconds:
                    sleep_fn(options.sleep_seconds)
            record["episode_ids"] = unique_first_seen(episode_ids)
            if not record["episode_ids"] and record["status"] == "ok":
                record["status"] = "no_public_episodes"
            team_records.append(record)
    return team_records, failures


def scout_elo_buckets(
    options: EloBucketScoutOptions,
    *,
    runner: Runner | None = None,
    path_lookup: PathLookup = which,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> EloBucketScoutResult:
    if not str(options.section).strip():
        raise EloBucketScoutError("section is required")
    buckets = parse_score_buckets(options.bucket_specs)
    runner = runner or _default_runner
    kaggle = resolve_kaggle_executable(
        options.kaggle_exe,
        dry_run=options.dry_run,
        path_lookup=path_lookup,
        config_path=kaggle_config_path(options.out_root),
    )
    run_dir = create_run_dir(kind=ELO_BUCKET_SCOUT_KIND, out_root=options.scout_root, suffix=options.section)

    leaderboard_rows: list[LeaderboardRow] = []
    failures: list[dict[str, Any]] = []
    if not options.dry_run:
        ensure_kaggle_cli(kaggle, runner=runner)
        leaderboard_rows, _pages, lb_failure = _fetch_leaderboard(
            kaggle,
            options.competition,
            runner=runner,
            max_rank=FULL_BOARD_MAX_RANK,
            max_pages=100,
        )
        if lb_failure is not None and not leaderboard_rows:
            raise EloBucketScoutError(f"leaderboard fetch failed with return code {lb_failure.returncode}")

    bucketed = bucket_leaderboard(leaderboard_rows, buckets)
    sampled = sample_bucket_teams(bucketed, players_per_bucket=options.players_per_bucket, seed=options.seed)

    team_records: list[dict[str, Any]] = []
    if not options.dry_run:
        team_records, failures = _collect_team_episodes(kaggle, options, sampled, runner=runner, sleep_fn=sleep_fn)

    already = downloaded_episode_ids(options.replay_root)
    pending = assign_pending_episodes(team_records, already_downloaded=already)

    episode_files: dict[str, Path] = {}
    for label, episode_ids in pending.items():
        path = Path(options.episode_ids_root) / f"{bucket_section_name(options.section, label)}.txt"
        episode_files[label] = write_text(path, "\n".join(episode_ids) + ("\n" if episode_ids else ""))

    all_episode_ids = unique_first_seen(eid for record in team_records for eid in record["episode_ids"])
    summary = {
        "section": options.section,
        "schema_version": ELO_BUCKET_SCOUT_VERSION,
        "leaderboard_row_count": len(leaderboard_rows),
        "bucket_populations": {label: len(rows) for label, rows in bucketed.items()},
        "bucket_sampled": {label: len(rows) for label, rows in sampled.items()},
        "team_count": len(team_records),
        "teams_with_episodes": sum(1 for record in team_records if record["episode_ids"]),
        "unique_episode_count": len(all_episode_ids),
        "already_downloaded_count": sum(1 for eid in all_episode_ids if eid in already),
        "pending_by_bucket": {label: len(ids) for label, ids in pending.items()},
        "failure_count": len(failures),
        "seed": options.seed,
        "players_per_bucket": options.players_per_bucket,
        "dry_run": options.dry_run,
    }
    metadata = {
        "schema_version": ELO_BUCKET_SCOUT_VERSION,
        "section": options.section,
        "competition": options.competition,
        "buckets": [{"label": bucket.label, "low": bucket.low, "high": bucket.high} for bucket in buckets],
        "sampled_teams": team_records,
        "pending_episode_ids_by_bucket": pending,
        "episode_files": {label: str(path) for label, path in episode_files.items()},
        "summary": summary,
    }

    metadata_path = write_json(run_dir.output_path("bucket_teams.json"), metadata)
    write_jsonl(run_dir.output_path("leaderboard_snapshot.jsonl"), (row.to_dict() for row in leaderboard_rows))
    report_path = write_markdown(run_dir.output_path("report.md"), _report(options, buckets, summary, team_records, failures))
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=ELO_BUCKET_SCOUT_KIND,
        command=options.command,
        status="completed" if not failures else "completed_with_failures",
        inputs={"args": options.args or {}, "competition": options.competition, "section": options.section,
                "bucket_specs": [bucket.label for bucket in buckets], "players_per_bucket": options.players_per_bucket,
                "seed": options.seed, "max_submissions_per_team": options.max_submissions_per_team,
                "max_episodes_per_submission": options.max_episodes_per_submission,
                **kaggle.to_manifest_fields()},
        outputs={"bucket_teams": str(metadata_path), "report": str(report_path),
                 "episode_files": {label: str(path) for label, path in episode_files.items()}},
        summary=summary,
    )
    manifest["failures"] = failures
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return EloBucketScoutResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        metadata_path=metadata_path,
        episode_files=episode_files,
        summary=summary,
        metadata=metadata,
    )


def rescout_failed_teams(
    metadata_path: Path,
    options: EloBucketScoutOptions,
    *,
    runner: Runner | None = None,
    path_lookup: PathLookup = which,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> EloBucketScoutResult:
    """Retry episode collection for teams that failed in a previous scout run.

    Loads an existing ``bucket_teams.json``, retries only teams whose status is a
    failure (quota-wall casualties), merges the recovered records, and writes a
    fresh run dir + refreshed per-bucket episode files. Successful teams are
    never re-queried, so a rescout after the quota window resets is cheap.
    """
    previous = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = list(previous.get("sampled_teams") or [])
    retry = [r for r in records if str(r.get("status", "")).endswith("_failed")]
    if not retry:
        raise EloBucketScoutError(f"no failed teams to rescout in {metadata_path}")
    buckets = parse_score_buckets([b["label"] for b in previous.get("buckets") or []] or options.bucket_specs)
    runner = runner or _default_runner
    kaggle = resolve_kaggle_executable(
        options.kaggle_exe,
        dry_run=options.dry_run,
        path_lookup=path_lookup,
        config_path=kaggle_config_path(options.out_root),
    )
    run_dir = create_run_dir(kind=ELO_BUCKET_SCOUT_KIND, out_root=options.scout_root, suffix=f"{options.section}-rescout")

    retry_rows: dict[str, list[LeaderboardRow]] = {}
    for record in retry:
        retry_rows.setdefault(record["bucket"], []).append(
            LeaderboardRow(
                rank=int(record.get("rank") or 0), row_index=int(record.get("rank") or 0),
                team_id=str(record["team_id"]), team_name=str(record.get("team_name") or ""),
                score=record.get("score"),
            )
        )
    retried_records, failures = _collect_team_episodes(kaggle, options, retry_rows, runner=runner, sleep_fn=sleep_fn)

    # The retry attempt is always the newest truth for a retried team; a repeat
    # failure keeps its *_failed status and stays eligible for the next rescout.
    by_team_id = {str(r["team_id"]): r for r in retried_records}
    merged: list[dict[str, Any]] = [by_team_id.get(str(record["team_id"]), record) for record in records]

    already = downloaded_episode_ids(options.replay_root)
    pending = assign_pending_episodes(merged, already_downloaded=already)
    episode_files: dict[str, Path] = {}
    for label, episode_ids in pending.items():
        path = Path(options.episode_ids_root) / f"{bucket_section_name(options.section, label)}.txt"
        episode_files[label] = write_text(path, "\n".join(episode_ids) + ("\n" if episode_ids else ""))

    all_episode_ids = unique_first_seen(eid for record in merged for eid in record["episode_ids"])
    still_failed = sum(1 for r in merged if str(r.get("status", "")).endswith("_failed"))
    summary = {
        "section": options.section,
        "schema_version": ELO_BUCKET_SCOUT_VERSION,
        "rescout_of": str(metadata_path),
        "leaderboard_row_count": int((previous.get("summary") or {}).get("leaderboard_row_count") or 0),
        "bucket_populations": (previous.get("summary") or {}).get("bucket_populations") or {},
        "bucket_sampled": (previous.get("summary") or {}).get("bucket_sampled") or {},
        "retried_teams": len(retry),
        "recovered_teams": len(retry) - still_failed,
        "still_failed_teams": still_failed,
        "team_count": len(merged),
        "teams_with_episodes": sum(1 for record in merged if record["episode_ids"]),
        "unique_episode_count": len(all_episode_ids),
        "already_downloaded_count": sum(1 for eid in all_episode_ids if eid in already),
        "pending_by_bucket": {label: len(ids) for label, ids in pending.items()},
        "failure_count": len(failures),
        "seed": options.seed,
        "players_per_bucket": options.players_per_bucket,
        "dry_run": options.dry_run,
    }
    metadata = {
        "schema_version": ELO_BUCKET_SCOUT_VERSION,
        "section": options.section,
        "competition": options.competition,
        "buckets": [{"label": bucket.label, "low": bucket.low, "high": bucket.high} for bucket in buckets],
        "sampled_teams": merged,
        "pending_episode_ids_by_bucket": pending,
        "episode_files": {label: str(path) for label, path in episode_files.items()},
        "summary": summary,
    }
    metadata_out = write_json(run_dir.output_path("bucket_teams.json"), metadata)
    report_path = write_markdown(run_dir.output_path("report.md"), _report(options, buckets, summary, merged, failures))
    manifest = build_manifest(
        run_id=run_dir.run_id, kind=ELO_BUCKET_SCOUT_KIND, command=options.command,
        status="completed" if not failures else "completed_with_failures",
        inputs={"args": options.args or {}, "rescout_of": str(metadata_path), "section": options.section,
                **kaggle.to_manifest_fields()},
        outputs={"bucket_teams": str(metadata_out), "report": str(report_path),
                 "episode_files": {label: str(path) for label, path in episode_files.items()}},
        summary=summary,
    )
    manifest["failures"] = failures
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return EloBucketScoutResult(
        run_dir=run_dir.path, manifest_path=manifest_path, report_path=report_path,
        metadata_path=metadata_out, episode_files=episode_files, summary=summary, metadata=metadata,
    )


def find_latest_bucket_scout(scout_root: str | Path = DEFAULT_ELO_BUCKET_ROOT) -> Path:
    """Latest bucket_teams.json under the elo-bucket scout root."""
    root = Path(scout_root)
    candidates = sorted(root.glob("*/*/*/bucket_teams.json")) if root.exists() else []
    if not candidates:
        raise EloBucketScoutError(f"no elo bucket scout metadata found under {root}")
    return candidates[-1]


def _report(
    options: EloBucketScoutOptions,
    buckets: list[BucketSpec],
    summary: dict[str, Any],
    team_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# ELO Score-Bucket Scout",
        "",
        f"section={options.section}",
        f"seed={options.seed} players_per_bucket={options.players_per_bucket}",
        "",
        "## Buckets",
        "",
        "| bucket | population | sampled | teams w/ episodes | pending episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in buckets:
        label = bucket.label
        with_eps = sum(1 for record in team_records if record["bucket"] == label and record["episode_ids"])
        lines.append(
            f"| {label} | {summary['bucket_populations'].get(label, 0)} | {summary['bucket_sampled'].get(label, 0)} "
            f"| {with_eps} | {summary['pending_by_bucket'].get(label, 0)} |"
        )
    lines += [
        "",
        f"- leaderboard rows: {summary['leaderboard_row_count']}",
        f"- unique episodes discovered: {summary['unique_episode_count']} "
        f"(already downloaded: {summary['already_downloaded_count']})",
        f"- failures: {len(failures)}",
        "",
        "## Boundary",
        "",
        "- Offline mining infrastructure only; no gameplay/policy/deck/runtime change.",
        "- Bucket = leaderboard score at scout time, not per-game rating.",
    ]
    return "\n".join(lines)
