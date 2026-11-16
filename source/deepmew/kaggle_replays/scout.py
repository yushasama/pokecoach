"""Official Kaggle episode scouting for K1.6 replay corpus building.

This module shells out to the public Kaggle CLI flow only:

* competitions leaderboard
* competitions team-submissions
* competitions episodes

It never scrapes web pages and never imports DeepMew gameplay/runtime code.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import which
from typing import Any, Callable, Iterable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown, write_text
from deepmew.kaggle_replays.corpus import CorpusOptions, CorpusResult, report_kaggle_replay_corpus
from deepmew.kaggle_replays.download import (
    DEFAULT_COMPETITION,
    DEFAULT_OUT_ROOT,
    DownloadOptions,
    KaggleExecutable,
    KaggleDownloadResult,
    PathLookup,
    Runner,
    _run_command,
    download_kaggle_replays,
    ensure_kaggle_cli,
    kaggle_config_path,
    resolve_kaggle_executable,
)
from deepmew.kaggle_replays.episode_sources import extract_episode_ids_from_text, unique_first_seen

LEADERBOARD_PAGE_SIZE = 200
MAX_LEADERBOARD_PAGES = 12

KAGGLE_SCOUT_KIND = "kaggle_scout"
DEFAULT_EPISODE_IDS_ROOT = ".research/kaggle_episode_ids"


class KaggleScoutError(ValueError):
    """Raised when strict K1.6 scouting cannot continue."""


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    row_index: int
    team_id: str
    team_name: str
    score: str | None = None
    submission_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "row_index": self.row_index,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "score": self.score,
            "submission_date": self.submission_date,
        }


@dataclass(frozen=True)
class ScoutOptions:
    section: str
    competition: str = DEFAULT_COMPETITION
    rank_ranges: list[str] | None = None
    teams_per_range: int = 10
    all_teams_in_range: bool = False
    team_ids: list[str] | None = None
    team_name_contains: list[str] | None = None
    neighbor_window: int | None = None
    submission_ids: list[str] | None = None
    max_submissions_per_team: int = 2
    max_episodes_per_submission: int = 20
    download_after: bool = False
    ingest_after: bool = False
    corpus_after: bool = True
    dry_run: bool = False
    strict: bool = False
    sleep_seconds: float = 0.0
    kaggle_exe: str | None = None
    out_root: Path = Path(DEFAULT_OUT_ROOT)
    episode_ids_root: Path = Path(DEFAULT_EPISODE_IDS_ROOT)
    command: str = ""
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScoutResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    episode_file: Path
    manifest: dict[str, Any]
    episode_ids: list[str]
    summary: dict[str, Any]
    download_after_result: KaggleDownloadResult | None
    corpus_after_result: CorpusResult | None = None


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _get(row: dict[str, str], *names: str) -> str | None:
    norm = {_norm_key(key): value for key, value in row.items() if key is not None}
    for name in names:
        value = norm.get(_norm_key(name))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


_NEXT_PAGE_TOKEN_RE = re.compile(r"next\s+page\s+token\s*=\s*(\S+)", re.IGNORECASE)
_COLUMN_SPLIT_RE = re.compile(r"\s{2,}|\t+")
_LEADERBOARD_TABLE_ROW_RE = re.compile(
    r"^\s*(?P<team_id>\d+)\s+(?P<team_name>.*?)\s+"
    r"(?P<submission_date>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


def _next_page_token(text: str) -> str | None:
    """Extract the `Next Page Token = ...` value emitted by the Kaggle CLI, if any."""
    for line in text.splitlines():
        match = _NEXT_PAGE_TOKEN_RE.search(line)
        if match:
            token = match.group(1).strip()
            return token or None
    return None


def _strip_bytes_wrapper(line: str) -> str:
    """Decode a Python ``b'...'`` repr the Kaggle CLI prints for non-ASCII names on Windows."""
    stripped = line.strip()
    if len(stripped) >= 3 and stripped[0] == "b" and stripped[1] in "'\"" and stripped[-1] == stripped[1]:
        try:
            decoded = ast.literal_eval(stripped)
            if isinstance(decoded, bytes):
                return decoded.decode("utf-8").strip()
        except (SyntaxError, UnicodeDecodeError, ValueError):
            return stripped[2:-1].strip()
    return stripped


def _split_columns(line: str) -> list[str]:
    return [part.strip() for part in _COLUMN_SPLIT_RE.split(line.strip()) if part.strip()]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", " "}


def parse_kaggle_table(text: str) -> list[dict[str, str]]:
    """Parse a plain Kaggle CLI table (whitespace-aligned columns) into header-keyed rows.

    Skips the ``Next Page Token`` preamble and the ``------`` separator, treats the first
    remaining line as the header, and splits each row on runs of 2+ spaces so that values
    that contain single spaces (team names, ``date time`` stamps) stay intact.
    """
    headers: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or _NEXT_PAGE_TOKEN_RE.search(stripped) or _is_separator(stripped):
            continue
        cells = _split_columns(_strip_bytes_wrapper(stripped))
        if not cells:
            continue
        if headers is None:
            headers = cells
            continue
        rows.append({headers[index]: cells[index] for index in range(min(len(headers), len(cells)))})
    return rows


def parse_leaderboard(text: str) -> list[LeaderboardRow]:
    """Parse Kaggle CLI leaderboard output from CSV-ish or plain-table text."""

    rows = _parse_leaderboard_csv(text)
    if rows:
        return rows
    return _parse_leaderboard_table(text)


def _parse_leaderboard_csv(text: str) -> list[LeaderboardRow]:
    sample = "\n".join(line for line in text.splitlines() if line.strip())
    if "," not in sample:
        return []
    try:
        reader = csv.DictReader(io.StringIO(sample))
    except csv.Error:
        return []
    if not reader.fieldnames:
        return []
    keys = {_norm_key(key) for key in reader.fieldnames}
    if not ({"teamid", "teamname"} & keys or {"teamid", "name"} <= keys):
        return []

    out: list[LeaderboardRow] = []
    for index, row in enumerate(reader, start=1):
        team_id = _get(row, "teamId", "team id", "id")
        team_name = _get(row, "teamName", "team name", "name") or ""
        if not team_id:
            continue
        rank = _int_or_none(_get(row, "rank", "position")) or index
        out.append(
            LeaderboardRow(
                rank=rank,
                row_index=index,
                team_id=team_id,
                team_name=team_name,
                score=_get(row, "score"),
                submission_date=_get(row, "submissionDate", "submission date", "date"),
            )
        )
    return out


def _parse_leaderboard_table(text: str) -> list[LeaderboardRow]:
    """Parse the real ``leaderboard -s`` plain table: ``teamId teamName submissionDate score``.

    The leaderboard has no rank column; rank is the (1-based) row order on the
    score-sorted board. Team names may contain spaces and non-ASCII characters.
    """
    out: list[LeaderboardRow] = []
    has_full_schema = "submissionDate" in text and re.search(r"\bscore\b", text, re.IGNORECASE) is not None
    for raw_line in text.splitlines():
        stripped = _strip_bytes_wrapper(raw_line)
        match = _LEADERBOARD_TABLE_ROW_RE.fullmatch(stripped)
        if not match:
            if has_full_schema and re.match(r"^\d+\s+", stripped):
                raise KaggleScoutError(f"unparseable leaderboard data row: {stripped!r}")
            continue
        out.append(
            LeaderboardRow(
                rank=len(out) + 1,
                row_index=len(out) + 1,
                team_id=match.group("team_id"),
                team_name=match.group("team_name").strip(),
                score=match.group("score"),
                submission_date=match.group("submission_date"),
            )
        )
    if out:
        return out

    for record in parse_kaggle_table(text):
        team_id = _get(record, "teamId", "team id", "id")
        if not team_id or not team_id.isdigit():
            continue
        out.append(
            LeaderboardRow(
                rank=len(out) + 1,
                row_index=len(out) + 1,
                team_id=team_id,
                team_name=_get(record, "teamName", "team name", "name") or "",
                score=_get(record, "score", "publicScore"),
                submission_date=_get(record, "submissionDate", "submission date", "date"),
            )
        )
    return out


def parse_rank_ranges(values: Iterable[str]) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for raw_value in values:
        value = str(raw_value).strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
        if not match:
            raise KaggleScoutError(f"rank range must look like 1-10 or 25, got {value!r}")
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if start <= 0 or end < start:
            raise KaggleScoutError(f"invalid rank range: {value!r}")
        ranges.append((start, end, value))
    return ranges


def _max_requested_rank(options: ScoutOptions) -> int | None:
    ranges = parse_rank_ranges(options.rank_ranges or [])
    if not ranges:
        return None
    return max(end for _start, end, _label in ranges)


def select_rows_evenly(rows: list[LeaderboardRow], *, limit: int, all_rows: bool = False) -> list[LeaderboardRow]:
    if all_rows or len(rows) <= limit:
        return rows
    if limit <= 0:
        return []
    if limit == 1:
        return [rows[0]]
    chosen_indices: list[int] = []
    for offset in range(limit):
        index = round(offset * (len(rows) - 1) / (limit - 1))
        if index not in chosen_indices:
            chosen_indices.append(index)
    for index in range(len(rows)):
        if len(chosen_indices) >= limit:
            break
        if index not in chosen_indices:
            chosen_indices.append(index)
    return [rows[index] for index in sorted(chosen_indices[:limit])]


def _table_ids(text: str, *id_fields: str) -> list[str]:
    ids: list[str] = []
    for record in parse_kaggle_table(text):
        value = _get(record, *id_fields)
        if value and value.isdigit():
            ids.append(value)
    return unique_first_seen(ids)


def parse_submission_ids(text: str) -> list[str]:
    """Parse `team-submissions` output (real plain table has an ``id`` first column)."""
    rows = _parse_id_csv(text, "submissionId", "submission id", "id")
    if rows:
        return rows
    table_ids = _table_ids(text, "id", "submissionId", "submission id")
    if table_ids:
        return table_ids
    return unique_first_seen(extract_episode_ids_from_text(text))


def parse_episode_ids_from_episodes_output(text: str) -> list[str]:
    """Parse `episodes` output, preferring public episodes from the real plain table.

    The real table is ``id createTime endTime state type``; validation episodes are
    dropped so the replay corpus stays on public games. CSV/URL inputs still work via
    the legacy fallbacks, which keeps the per-row microsecond noise out of the IDs.
    """
    rows = _parse_id_csv(text, "episodeId", "episode id", "id")
    if rows:
        return rows
    table_ids: list[str] = []
    for record in parse_kaggle_table(text):
        value = _get(record, "id", "episodeId", "episode id")
        if not value or not value.isdigit():
            continue
        episode_type = (_get(record, "type") or "").lower()
        if "validation" in episode_type:
            continue
        table_ids.append(value)
    if table_ids:
        return unique_first_seen(table_ids)
    # Free-text fallback (URLs / bare id lists) only. Require >= 7 digits so a degenerate,
    # non-tabular CLI response can never leak a 6-digit microsecond field (e.g. ".541000")
    # as an episode id; real episode ids on this competition are 8 digits.
    return unique_first_seen(value for value in extract_episode_ids_from_text(text) if len(value) >= 7)


def _parse_id_csv(text: str, *id_fields: str) -> list[str]:
    sample = "\n".join(line for line in text.splitlines() if line.strip())
    if "," not in sample:
        return []
    try:
        reader = csv.DictReader(io.StringIO(sample))
    except csv.Error:
        return []
    if not reader.fieldnames:
        return []
    ids: list[str] = []
    for row in reader:
        value = _get(row, *id_fields)
        if value and value.isdigit():
            ids.append(value)
    return unique_first_seen(ids)


def _leaderboard_command(
    kaggle: KaggleExecutable,
    competition: str,
    *,
    page_size: int | None = None,
    page_token: str | None = None,
) -> list[str]:
    command = [kaggle.command, "competitions", "leaderboard", competition, "-s"]
    if page_size is not None:
        command += ["--page-size", str(page_size)]
    if page_token:
        command += ["--page-token", page_token]
    return command


def _team_submissions_command(kaggle: KaggleExecutable, competition: str, team_id: str) -> list[str]:
    del competition
    return [kaggle.command, "competitions", "team-submissions", str(team_id)]


def _episodes_command(kaggle: KaggleExecutable, competition: str, submission_id: str) -> list[str]:
    # Plain table only. The `-v`/`--csv` path triggers an interactive auth error on the
    # simulations CLI and returned zero episodes; the default table is what works manually.
    del competition
    return [kaggle.command, "competitions", "episodes", str(submission_id)]


def _fetch_leaderboard(
    kaggle: KaggleExecutable,
    competition: str,
    *,
    runner: Runner,
    max_rank: int | None,
    page_size: int = LEADERBOARD_PAGE_SIZE,
    max_pages: int = MAX_LEADERBOARD_PAGES,
) -> tuple[list[LeaderboardRow], list[dict[str, Any]], Any | None]:
    """Page through `leaderboard -s` until the deepest requested rank is covered.

    Returns the globally re-ranked rows, a per-page record list for the manifest, and the
    last failing command result (or None). Rows are de-duplicated by team id across pages.
    """
    target = max_rank if max_rank is not None else page_size
    collected: list[LeaderboardRow] = []
    page_records: list[dict[str, Any]] = []
    failure: Any | None = None
    token: str | None = None
    for page_number in range(1, max_pages + 1):
        command = _leaderboard_command(kaggle, competition, page_size=page_size, page_token=token)
        result = _run_command(command, runner=runner)
        page_records.append(
            {"page": page_number, "page_token": token, "command": command, "returncode": result.returncode}
        )
        if result.returncode != 0:
            failure = result
            break
        page_rows = parse_leaderboard(result.stdout)
        collected.extend(page_rows)
        token = _next_page_token(result.stdout)
        if not page_rows or token is None or len(collected) >= target:
            break

    seen: set[str] = set()
    ranked: list[LeaderboardRow] = []
    for row in collected:
        if row.team_id in seen:
            continue
        seen.add(row.team_id)
        ranked.append(replace(row, rank=len(ranked) + 1, row_index=len(ranked) + 1))
    return ranked, page_records, failure


def _record_failure(failures: list[dict[str, Any]], *, strict: bool, stage: str, message: str, result: Any = None) -> None:
    record: dict[str, Any] = {"stage": stage, "error": message}
    if result is not None:
        record["result"] = result.to_dict()
    failures.append(record)
    if strict:
        raise KaggleScoutError(message)


def _team_row(row: LeaderboardRow, *, source_kind: str, source_range: str | None = None, source_query: str | None = None) -> dict[str, Any]:
    return {
        **row.to_dict(),
        "source_kind": source_kind,
        "source_range": source_range,
        "source_query": source_query,
    }


def _dedupe_teams(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        team_id = str(record["team_id"])
        if team_id in seen:
            continue
        seen.add(team_id)
        out.append(record)
    return out


def _select_teams(options: ScoutOptions, leaderboard: list[LeaderboardRow]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"rank_ranges": [], "unmatched_team_name_contains": [], "own_rank": None}

    for start, end, label in parse_rank_ranges(options.rank_ranges or []):
        in_range = [row for row in leaderboard if start <= row.rank <= end]
        selected = select_rows_evenly(in_range, limit=options.teams_per_range, all_rows=options.all_teams_in_range)
        detail["rank_ranges"].append(
            {"range": label, "available": len(in_range), "selected": len(selected), "team_ids": [row.team_id for row in selected]}
        )
        records.extend(_team_row(row, source_kind="rank_range", source_range=label) for row in selected)

    by_id = {row.team_id: row for row in leaderboard}
    for team_id in options.team_ids or []:
        row = by_id.get(str(team_id))
        if row is None:
            records.append(
                {
                    "rank": None,
                    "row_index": None,
                    "team_id": str(team_id),
                    "team_name": None,
                    "score": None,
                    "submission_date": None,
                    "source_kind": "team_id",
                    "source_range": None,
                    "source_query": str(team_id),
                }
            )
        else:
            records.append(_team_row(row, source_kind="team_id", source_query=str(team_id)))

    for query in options.team_name_contains or []:
        needle = query.lower()
        matches = [row for row in leaderboard if needle in row.team_name.lower()]
        if not matches:
            detail["unmatched_team_name_contains"].append(query)
            continue
        if options.neighbor_window is None:
            records.extend(_team_row(row, source_kind="team_name_contains", source_query=query) for row in matches)
            continue
        own = matches[0]
        detail["own_rank"] = own.rank
        start = max(1, own.rank - options.neighbor_window)
        end = own.rank + options.neighbor_window
        neighbors = [row for row in leaderboard if start <= row.rank <= end]
        label = f"{start}-{end}"
        records.extend(_team_row(row, source_kind="neighbor_window", source_range=label, source_query=query) for row in neighbors)

    return _dedupe_teams(records), detail


def _coerce_options(options: ScoutOptions) -> ScoutOptions:
    if not str(options.section).strip():
        raise KaggleScoutError("section is required")
    if options.teams_per_range < 0:
        raise KaggleScoutError("teams_per_range must be non-negative")
    if options.max_submissions_per_team < 0:
        raise KaggleScoutError("max_submissions_per_team must be non-negative")
    if options.max_episodes_per_submission < 0:
        raise KaggleScoutError("max_episodes_per_submission must be non-negative")
    if options.neighbor_window is not None and options.neighbor_window < 0:
        raise KaggleScoutError("neighbor_window must be non-negative")
    if options.sleep_seconds < 0:
        raise KaggleScoutError("sleep_seconds must be non-negative")
    return ScoutOptions(
        section=str(options.section).strip(),
        competition=options.competition or DEFAULT_COMPETITION,
        rank_ranges=list(options.rank_ranges or []),
        teams_per_range=int(options.teams_per_range),
        all_teams_in_range=bool(options.all_teams_in_range),
        team_ids=[str(value).strip() for value in options.team_ids or [] if str(value).strip()],
        team_name_contains=[str(value).strip() for value in options.team_name_contains or [] if str(value).strip()],
        neighbor_window=options.neighbor_window,
        submission_ids=[str(value).strip() for value in options.submission_ids or [] if str(value).strip()],
        max_submissions_per_team=int(options.max_submissions_per_team),
        max_episodes_per_submission=int(options.max_episodes_per_submission),
        download_after=options.download_after,
        ingest_after=options.ingest_after,
        corpus_after=options.corpus_after,
        dry_run=options.dry_run,
        strict=options.strict,
        sleep_seconds=float(options.sleep_seconds),
        kaggle_exe=str(options.kaggle_exe) if options.kaggle_exe else None,
        out_root=Path(options.out_root),
        episode_ids_root=Path(options.episode_ids_root),
        command=options.command,
        args=options.args,
    )


def scout_kaggle_episodes(
    options: ScoutOptions,
    *,
    runner: Runner | None = None,
    path_lookup: PathLookup = which,
    download_fn: Callable[..., KaggleDownloadResult] = download_kaggle_replays,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ScoutResult:
    options = _coerce_options(options)
    runner = runner or _default_runner
    kaggle = resolve_kaggle_executable(
        options.kaggle_exe,
        dry_run=options.dry_run,
        path_lookup=path_lookup,
        config_path=kaggle_config_path(options.out_root),
    )
    run_dir = create_run_dir(kind=KAGGLE_SCOUT_KIND, out_root=options.out_root, suffix=options.section)

    cli_check: dict[str, Any] | None = None
    leaderboard_pages: list[dict[str, Any]] = []
    leaderboard_rows: list[LeaderboardRow] = []
    selected_teams: list[dict[str, Any]] = []
    selection_detail: dict[str, Any] = {"rank_ranges": [], "unmatched_team_name_contains": [], "own_rank": None}
    team_submission_records: list[dict[str, Any]] = []
    submission_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    needs_leaderboard = bool(options.rank_ranges or options.team_name_contains)

    if not options.dry_run:
        cli_check = ensure_kaggle_cli(kaggle, runner=runner).to_dict()
        if needs_leaderboard:
            leaderboard_rows, leaderboard_pages, leaderboard_failure = _fetch_leaderboard(
                kaggle,
                options.competition,
                runner=runner,
                max_rank=_max_requested_rank(options),
            )
            if leaderboard_failure is not None and not leaderboard_rows:
                _record_failure(
                    failures,
                    strict=options.strict,
                    stage="leaderboard",
                    message=f"leaderboard command failed with return code {leaderboard_failure.returncode}",
                    result=leaderboard_failure,
                )
            elif not leaderboard_rows:
                _record_failure(
                    failures,
                    strict=options.strict,
                    stage="leaderboard_parse",
                    message="leaderboard command returned no parseable rows",
                )
        selected_teams, selection_detail = _select_teams(options, leaderboard_rows)
        if options.team_ids and not needs_leaderboard:
            selected_teams = _dedupe_teams(
                {
                    "rank": None,
                    "row_index": None,
                    "team_id": team_id,
                    "team_name": None,
                    "score": None,
                    "submission_date": None,
                    "source_kind": "team_id",
                    "source_range": None,
                    "source_query": team_id,
                }
                for team_id in options.team_ids
            )
    else:
        selected_teams = _dedupe_teams(
            {
                "rank": None,
                "row_index": None,
                "team_id": team_id,
                "team_name": None,
                "score": None,
                "submission_date": None,
                "source_kind": "team_id",
                "source_range": None,
                "source_query": team_id,
            }
            for team_id in options.team_ids or []
        )

    selected_submission_ids: list[str] = list(options.submission_ids or [])

    if not options.dry_run:
        for team in selected_teams:
            command = _team_submissions_command(kaggle, options.competition, str(team["team_id"]))
            result = _run_command(command, runner=runner)
            record = {"team_id": team["team_id"], "team_name": team.get("team_name"), "command": command, "result": result.to_dict()}
            if result.returncode != 0:
                record["status"] = "failed"
                team_submission_records.append(record)
                _record_failure(
                    failures,
                    strict=options.strict,
                    stage="team_submissions",
                    message=f"team-submissions failed for team {team['team_id']} with return code {result.returncode}",
                    result=result,
                )
                continue
            submission_ids = parse_submission_ids(result.stdout)[: options.max_submissions_per_team]
            record["status"] = "ok"
            record["submission_ids"] = submission_ids
            team_submission_records.append(record)
            for submission_id in submission_ids:
                selected_submission_ids.append(submission_id)
            if options.sleep_seconds:
                sleep_fn(options.sleep_seconds)

    selected_submission_ids = unique_first_seen(selected_submission_ids)

    if not options.dry_run:
        for submission_id in selected_submission_ids:
            command = _episodes_command(kaggle, options.competition, submission_id)
            result = _run_command(command, runner=runner)
            record = {"submission_id": submission_id, "command": command, "result": result.to_dict()}
            if result.returncode != 0:
                record["status"] = "failed"
                submission_records.append(record)
                _record_failure(
                    failures,
                    strict=options.strict,
                    stage="episodes",
                    message=f"episodes failed for submission {submission_id} with return code {result.returncode}",
                    result=result,
                )
                continue
            episode_ids = parse_episode_ids_from_episodes_output(result.stdout)[: options.max_episodes_per_submission]
            record["status"] = "ok"
            record["episode_ids"] = episode_ids
            submission_records.append(record)
            for episode_id in episode_ids:
                episode_records.append({"episode_id": episode_id, "submission_id": submission_id})
            if options.sleep_seconds:
                sleep_fn(options.sleep_seconds)

    episode_ids = unique_first_seen(record["episode_id"] for record in episode_records)
    episode_file = Path(options.episode_ids_root) / f"{options.section}.txt"
    episode_file = write_text(episode_file, "\n".join(episode_ids) + ("\n" if episode_ids else ""))

    download_result: KaggleDownloadResult | None = None
    download_after_summary: dict[str, Any] | None = None
    if options.download_after and episode_ids and not options.dry_run:
        download_result = download_fn(
            DownloadOptions(
                episode_ids=episode_ids,
                section=options.section,
                competition=options.competition,
                dry_run=False,
                ingest_after=options.ingest_after,
                sleep_seconds=options.sleep_seconds,
                out_root=options.out_root,
                kaggle_exe=options.kaggle_exe,
                command=options.command,
                args={**(options.args or {}), "episode_file": str(episode_file), "source": "kaggle_scout"},
            ),
            runner=runner,
            path_lookup=path_lookup,
            sleep_fn=sleep_fn,
        )
        download_after_summary = {
            "run_dir": str(download_result.run_dir),
            "manifest_path": str(download_result.manifest_path),
            "replay_dir": str(download_result.replay_dir),
            "summary": download_result.summary,
        }

    corpus_result: CorpusResult | None = None
    corpus_after_summary: dict[str, Any] | None = None
    if options.corpus_after and download_result is not None and not options.dry_run:
        corpus_result = report_kaggle_replay_corpus(
            CorpusOptions(
                replay_root=options.out_root / "kaggle_replays",
                ingest_root=options.out_root / "kaggle_ingest",
                opponent_decks_root=options.out_root / "opponent_decks",
                out_root=options.out_root,
                section=options.section,
                command=options.command,
                args={**(options.args or {}), "source": "kaggle_scout"},
            )
        )
        corpus_after_summary = {
            "run_dir": str(corpus_result.run_dir),
            "manifest_path": str(corpus_result.manifest_path),
            "report_path": str(corpus_result.report_path),
            "summary": corpus_result.summary,
        }

    duplicate_count = len(episode_records) - len(episode_ids)
    summary = {
        "section": options.section,
        "competition": options.competition,
        "leaderboard_row_count": len(leaderboard_rows),
        "selected_team_count": len(selected_teams),
        "selected_submission_count": len(selected_submission_ids),
        "raw_episode_id_count": len(episode_records),
        "unique_episode_id_count": len(episode_ids),
        "duplicate_episode_id_count": duplicate_count,
        "failure_count": len(failures),
        "download_after": bool(download_after_summary),
        "corpus_after": bool(corpus_after_summary),
        "leaderboard_page_count": len(leaderboard_pages),
        "dry_run": options.dry_run,
    }
    status = "completed" if not failures else "completed_with_failures"
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_SCOUT_KIND,
        command=options.command,
        status=status,
        inputs={
            "args": options.args or {},
            "competition": options.competition,
            "section": options.section,
            "rank_ranges": options.rank_ranges,
            "teams_per_range": options.teams_per_range,
            "all_teams_in_range": options.all_teams_in_range,
            "team_ids": options.team_ids,
            "team_name_contains": options.team_name_contains,
            "neighbor_window": options.neighbor_window,
            "submission_ids": options.submission_ids,
            "max_submissions_per_team": options.max_submissions_per_team,
            "max_episodes_per_submission": options.max_episodes_per_submission,
            "dry_run": options.dry_run,
            "strict": options.strict,
            "sleep_seconds": options.sleep_seconds,
            "kaggle_cli_check": cli_check,
            **kaggle.to_manifest_fields(),
        },
        outputs={
            "episode_file": str(episode_file),
            "scout_report": str(run_dir.output_path("scout_report.md")),
            "download_after": download_after_summary,
            "corpus_after": corpus_after_summary,
        },
        summary=summary,
    )
    manifest["leaderboard"] = {
        "command": None if options.dry_run else _leaderboard_command(kaggle, options.competition, page_size=LEADERBOARD_PAGE_SIZE),
        "pages": leaderboard_pages,
        "page_count": len(leaderboard_pages),
        "row_count": len(leaderboard_rows),
    }
    manifest["selection"] = selection_detail
    manifest["selected_teams"] = selected_teams
    manifest["team_submissions"] = team_submission_records
    manifest["selected_submissions"] = selected_submission_ids
    manifest["submission_episode_records"] = submission_records
    manifest["discovered_episodes"] = episode_records
    manifest["unique_episode_ids"] = episode_ids
    manifest["duplicate_episode_ids_skipped"] = duplicate_count
    manifest["failures"] = failures
    manifest["download_after_result"] = download_after_summary
    manifest["corpus_after_result"] = corpus_after_summary

    report_path = write_markdown(run_dir.output_path("scout_report.md"), _report(options, summary, episode_file, selected_teams, episode_ids, failures))
    manifest_path = write_manifest(run_dir.manifest_path, manifest)

    return ScoutResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        episode_file=episode_file,
        manifest=manifest,
        episode_ids=episode_ids,
        summary=summary,
        download_after_result=download_result,
        corpus_after_result=corpus_result,
    )


def _report(
    options: ScoutOptions,
    summary: dict[str, Any],
    episode_file: Path,
    selected_teams: list[dict[str, Any]],
    episode_ids: list[str],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# Kaggle Episode Scout",
        "",
        f"competition={options.competition}",
        f"section={options.section}",
        f"episode_file={episode_file}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    lines += ["", "## Selected Teams", "", "| rank | team_id | team_name | source | range |", "|---:|---|---|---|---|"]
    for team in selected_teams:
        lines.append(
            f"| {team.get('rank') or ''} | {team['team_id']} | {team.get('team_name') or ''} | "
            f"{team.get('source_kind') or ''} | {team.get('source_range') or ''} |"
        )
    lines += ["", "## Episodes", ""]
    if episode_ids:
        lines.extend(f"- {episode_id}" for episode_id in episode_ids)
    else:
        lines.append("- None discovered.")
    lines += ["", "## Failures", ""]
    if failures:
        lines.extend(f"- {failure['stage']}: {failure['error']}" for failure in failures)
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Boundary",
        "",
        "- K1.6 infrastructure/reporting only.",
        "- No gameplay, scoring, policy, agents, turn-plan, replay-prior online integration, value model, RL, belief model, or search change.",
    ]
    return "\n".join(lines)


def _default_runner(command: Any, **kwargs: Any) -> Any:
    import subprocess

    return subprocess.run(command, **kwargs)
