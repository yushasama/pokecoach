"""Kaggle submission catalogs and validated public replay caching.

The preferred catalog path uses the official Kaggle Python SDK because its
``ApiEpisode`` objects retain per-seat submission/team metadata that the CLI's
CSV display omits.  The SDK is invoked with the Python interpreter adjacent to
the configured ``kaggle`` executable so the repository does not accidentally
use a different Kaggle installation.

CLI and offline fallbacks are deliberately labelled ``metadata_limited``.
They remain useful for exact episode/range filtering and already-downloaded
replays, but callers must not mistake them for full opponent identity data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from shutil import which
from typing import Any, Callable, Iterable, Mapping, Sequence

from deepmew.kaggle_replays.download import (
    KAGGLE_EXE_ENV_VAR,
    kaggle_config_path,
    load_configured_kaggle_exe,
)
from deepmew.submission_postmortem.models import CatalogAgent, CatalogEpisode, CatalogQuery


CATALOG_SOURCE_SDK = "full_sdk"
CATALOG_SOURCE_CLI = "metadata_limited_cli"
CATALOG_SOURCE_OFFLINE = "metadata_limited_offline"
SDK_OUTPUT_MARKER = "DEEPMEW_SUBMISSION_CATALOG_JSON="

DEFAULT_OUT_ROOT = Path(".research")
DEFAULT_REPLAY_CACHE = Path(".research/submission_postmortem/cache/replays")

Runner = Callable[..., Any]
SdkFetcher = Callable[[str], Iterable[Any]]
PathLookup = Callable[[str], str | None]


class SubmissionCatalogError(ValueError):
    """Raised when a submission catalog or replay cache cannot be validated."""


@dataclass(frozen=True)
class CatalogFetchResult:
    """Normalized episode catalog plus explicit metadata provenance."""

    submission_id: str
    metadata_status: str
    episodes: tuple[CatalogEpisode, ...]
    source_command: tuple[str, ...] = ()
    kaggle_executable: str | None = None
    python_executable: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_metadata_limited(self) -> bool:
        return self.metadata_status.startswith("metadata_limited")

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "metadata_status": self.metadata_status,
            "is_metadata_limited": self.is_metadata_limited,
            "episode_count": len(self.episodes),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "source_command": list(self.source_command),
            "kaggle_executable": self.kaggle_executable,
            "python_executable": self.python_executable,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ReplayCacheRecord:
    """Integrity receipt for one replay cache target."""

    episode_id: str
    path: str
    status: str
    sha256: str | None = None
    byte_size: int | None = None
    source_path: str | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status in {"cached", "downloaded", "offline_imported"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "path": self.path,
            "status": self.status,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "source_path": self.source_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class ReplayCacheResult:
    """Bounded concurrent replay-cache operation."""

    records: tuple[ReplayCacheRecord, ...]

    @property
    def complete(self) -> bool:
        return all(record.is_valid for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": len(self.records),
            "valid": sum(record.is_valid for record in self.records),
            "failed": sum(not record.is_valid for record in self.records),
            "complete": self.complete,
            "records": [record.to_dict() for record in self.records],
        }


def _read_field(value: Any, *names: str, default: Any = None) -> Any:
    """Read a Kaggle SDK attribute or any common dict spelling."""

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]

        normalized = {str(key).replace("_", "").lower(): item for key, item in value.items()}
        for name in names:
            key = name.replace("_", "").lower()
            if key in normalized:
                return normalized[key]

        return default

    for name in names:
        if hasattr(value, name):
            return getattr(value, name)

    return default


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(value).strip()

    return rendered or None


def _identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    rendered = str(value).strip()

    return rendered or None


def _integer(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reward(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None

    return number


def _agent_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [item for _, item in sorted(value.items(), key=lambda pair: str(pair[0])) if item is not None]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if item is not None]

    return []


def normalize_catalog_agent(value: Any, *, fallback_index: int) -> CatalogAgent:
    """Normalize either an ``ApiEpisodeAgent`` or a JSON/dict equivalent."""

    return CatalogAgent(
        index=_integer(_read_field(value, "index", "agent_index", "agentIndex"), default=fallback_index),
        submission_id=_identifier(_read_field(value, "submission_id", "submissionId")),
        team_id=_identifier(_read_field(value, "team_id", "teamId")),
        team_name=_text(_read_field(value, "team_name", "teamName", "name")),
        reward=_reward(_read_field(value, "reward")),
        state=_text(_read_field(value, "state", "status")),
    )


def normalize_catalog_episode(value: Any) -> CatalogEpisode:
    """Normalize a full SDK episode without requiring SDK-specific dict methods."""

    episode_id = _identifier(_read_field(value, "episode_id", "episodeId", "EpisodeId", "id"))
    if not episode_id:
        raise SubmissionCatalogError("catalog episode is missing an episode id")
    raw_type = _text(_read_field(value, "type", "episode_type", "episodeType")) or ""
    explicit_validation = _read_field(value, "is_validation", "isValidation")
    is_validation = bool(explicit_validation) if isinstance(explicit_validation, bool) else "validation" in raw_type.lower()
    raw_agents = _agent_values(_read_field(value, "agents", default=[]))
    agents = tuple(normalize_catalog_agent(agent, fallback_index=index) for index, agent in enumerate(raw_agents))

    return CatalogEpisode(
        episode_id=episode_id,
        create_time=_text(_read_field(value, "create_time", "createTime")),
        end_time=_text(_read_field(value, "end_time", "endTime")),
        state=_text(_read_field(value, "state", "status")),
        is_validation=is_validation,
        agents=tuple(sorted(agents, key=lambda agent: agent.index)),
    )


def _episode_id_key(episode_id: str) -> tuple[int, int | str]:
    if episode_id.isdigit():
        return 0, int(episode_id)

    return 1, episode_id


def stable_sort_episodes(episodes: Iterable[CatalogEpisode]) -> tuple[CatalogEpisode, ...]:
    """Sort by Kaggle creation time, then numeric episode identity."""

    return tuple(
        sorted(
            episodes,
            key=lambda episode: (
                episode.create_time is None,
                episode.create_time or "",
                _episode_id_key(episode.episode_id),
            ),
        )
    )


def _validate_query(query: CatalogQuery) -> None:
    if query.episode_min is not None and query.episode_max is not None and query.episode_min > query.episode_max:
        raise SubmissionCatalogError("episode_min must not exceed episode_max")
    if query.ordinal_min is not None and query.ordinal_min < 1:
        raise SubmissionCatalogError("ordinal_min is 1-based and must be positive")
    if query.ordinal_max is not None and query.ordinal_max < 1:
        raise SubmissionCatalogError("ordinal_max is 1-based and must be positive")
    if query.ordinal_min is not None and query.ordinal_max is not None and query.ordinal_min > query.ordinal_max:
        raise SubmissionCatalogError("ordinal_min must not exceed ordinal_max")


def opponent_agents(episode: CatalogEpisode, *, submission_id: str) -> tuple[CatalogAgent, ...]:
    """Return seats not owned by the requested submission.

    Missing submission IDs remain candidates.  Full SDK catalogs provide the
    IDs; metadata-limited callers can still perform best-effort name filters,
    but the enclosing catalog result remains explicitly marked limited.
    """

    target = str(submission_id)

    return tuple(agent for agent in episode.agents if agent.submission_id != target)


def _matches_opponent_filters(episode: CatalogEpisode, *, submission_id: str, query: CatalogQuery) -> bool:
    if not (query.opponent_terms or query.opponent_team_ids or query.opponent_submission_ids):
        return True
    candidates = opponent_agents(episode, submission_id=submission_id)
    if not candidates:
        return False

    if query.opponent_terms:
        terms = tuple(term.casefold() for term in query.opponent_terms if term)
        if not any(agent.team_name and any(term in agent.team_name.casefold() for term in terms) for agent in candidates):
            return False
    if query.opponent_team_ids:
        if not any(agent.team_id in query.opponent_team_ids for agent in candidates):
            return False
    if query.opponent_submission_ids:
        if not any(agent.submission_id in query.opponent_submission_ids for agent in candidates):
            return False

    return True


def filter_catalog_episodes(
    episodes: Iterable[CatalogEpisode],
    *,
    submission_id: str,
    query: CatalogQuery | None = None,
) -> tuple[CatalogEpisode, ...]:
    """Apply composable filters to a stably ordered submission catalog.

    Filter families combine with AND.  Repeated values inside exact-ID and
    opponent families combine with OR.  Ordinals refer to the full catalog
    after stable ``(create_time, episode_id)`` ordering and are inclusive,
    1-based bounds.
    """

    query = query or CatalogQuery()
    _validate_query(query)
    selected: list[CatalogEpisode] = []
    for ordinal, episode in enumerate(stable_sort_episodes(episodes), start=1):
        if not query.include_validation and episode.is_validation:
            continue
        if query.episode_ids and episode.episode_id not in query.episode_ids:
            continue
        numeric_id = int(episode.episode_id) if episode.episode_id.isdigit() else None
        if query.episode_min is not None and (numeric_id is None or numeric_id < query.episode_min):
            continue
        if query.episode_max is not None and (numeric_id is None or numeric_id > query.episode_max):
            continue
        if query.ordinal_min is not None and ordinal < query.ordinal_min:
            continue
        if query.ordinal_max is not None and ordinal > query.ordinal_max:
            continue
        if not _matches_opponent_filters(episode, submission_id=str(submission_id), query=query):
            continue
        selected.append(episode)

    return tuple(selected)


def resolve_kaggle_executable_path(
    kaggle_executable: str | Path | None,
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    env: Mapping[str, str] | None = None,
    path_lookup: PathLookup = which,
) -> str:
    """Resolve Kaggle CLI without reading or returning authentication material."""

    if kaggle_executable is not None and str(kaggle_executable).strip():
        return str(kaggle_executable).strip()
    environ = os.environ if env is None else env
    configured_env = environ.get(KAGGLE_EXE_ENV_VAR)
    if configured_env and configured_env.strip():
        return configured_env.strip()
    configured_file = load_configured_kaggle_exe(kaggle_config_path(out_root))
    if configured_file:
        return configured_file
    discovered = path_lookup("kaggle")
    if discovered:
        return discovered

    raise SubmissionCatalogError("Kaggle executable not found; pass --kaggle-exe or configure .research/local/kaggle_cli.json")


def resolve_adjacent_python(
    kaggle_executable: str | Path,
    *,
    path_lookup: PathLookup = which,
) -> str:
    """Resolve the Python runtime belonging to the configured Kaggle CLI."""

    rendered = str(kaggle_executable)
    resolved = path_lookup(rendered) if not Path(rendered).is_absolute() else rendered
    executable_path = Path(resolved or rendered)
    candidates = (
        executable_path.parent.parent / "python.exe",
        executable_path.parent / "python.exe",
        executable_path.parent / "python",
        executable_path.parent / "python3",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise SubmissionCatalogError(
        f"could not resolve the Python interpreter adjacent to Kaggle executable {kaggle_executable}"
    )


def _sdk_script() -> str:
    """Return the isolated official-SDK catalog exporter source."""

    return """
import json
import sys
from datetime import date, datetime
from enum import Enum
from kaggle import api

def field(value, *names, default=None):
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        normalized = {str(key).replace('_', '').lower(): item for key, item in value.items()}
        for name in names:
            key = name.replace('_', '').lower()
            if key in normalized:
                return normalized[key]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default

def scalar(value):
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    name = getattr(value, 'name', None)
    if isinstance(name, str) and name:
        return name
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

episodes = api.competition_list_episodes(int(sys.argv[1])) or []
payload = []
for episode in episodes:
    agents = []
    raw_agents = field(episode, 'agents', default=[]) or []
    if isinstance(raw_agents, dict):
        raw_agents = [item for _, item in sorted(raw_agents.items(), key=lambda pair: str(pair[0])) if item is not None]
    else:
        raw_agents = [item for item in raw_agents if item is not None]
    for index, agent in enumerate(raw_agents):
        agents.append({
            'index': scalar(field(agent, 'index', 'agent_index', 'agentIndex', default=index)),
            'submission_id': scalar(field(agent, 'submission_id', 'submissionId')),
            'team_id': scalar(field(agent, 'team_id', 'teamId')),
            'team_name': scalar(field(agent, 'team_name', 'teamName', 'name')),
            'reward': scalar(field(agent, 'reward')),
            'state': scalar(field(agent, 'state', 'status')),
        })
    payload.append({
        'episode_id': scalar(field(episode, 'episode_id', 'episodeId', 'EpisodeId', 'id')),
        'create_time': scalar(field(episode, 'create_time', 'createTime')),
        'end_time': scalar(field(episode, 'end_time', 'endTime')),
        'state': scalar(field(episode, 'state', 'status')),
        'type': scalar(field(episode, 'type', 'episode_type', 'episodeType')),
        'agents': agents,
    })
print('DEEPMEW_SUBMISSION_CATALOG_JSON=' + json.dumps(payload, separators=(',', ':')))
""".strip()


def sdk_catalog_command(python_executable: str | Path, submission_id: str | int) -> list[str]:
    return [str(python_executable), "-c", _sdk_script(), str(submission_id)]


def _default_runner(command: Sequence[str], **kwargs: Any) -> Any:
    return subprocess.run(list(command), **kwargs)


def _run(command: Sequence[str], *, runner: Runner) -> Any:
    try:
        return runner(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise SubmissionCatalogError(f"command executable not found: {command[0]}") from exc


def _returncode(result: Any) -> int:
    try:
        return int(getattr(result, "returncode"))
    except (TypeError, ValueError):
        return 1


def _stdout(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _stderr(result: Any) -> str:
    return str(getattr(result, "stderr", "") or "")


def _parse_sdk_output(text: str) -> list[Any]:
    for line in reversed(text.splitlines()):
        if line.startswith(SDK_OUTPUT_MARKER):
            try:
                payload = json.loads(line[len(SDK_OUTPUT_MARKER) :])
            except json.JSONDecodeError as exc:
                raise SubmissionCatalogError(f"Kaggle SDK returned invalid catalog JSON: {exc}") from exc
            if not isinstance(payload, list):
                raise SubmissionCatalogError("Kaggle SDK catalog payload must be a JSON list")

            return payload

    raise SubmissionCatalogError("Kaggle SDK catalog output marker was not found")


def _parse_cli_catalog(text: str) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error as exc:
        raise SubmissionCatalogError(f"Kaggle CLI catalog CSV could not be parsed: {exc}") from exc
    records: list[dict[str, Any]] = []
    for row in reader:
        episode_id = _identifier(_read_field(row, "id", "episode_id", "episodeId"))
        if not episode_id:
            continue
        records.append(
            {
                "episode_id": episode_id,
                "create_time": _read_field(row, "createTime", "create_time"),
                "end_time": _read_field(row, "endTime", "end_time"),
                "state": _read_field(row, "state"),
                "type": _read_field(row, "type"),
                "agents": [],
            }
        )

    return records


def _normalize_episode_collection(values: Iterable[Any]) -> tuple[CatalogEpisode, ...]:
    by_id: dict[str, CatalogEpisode] = {}
    for value in values:
        episode = normalize_catalog_episode(value)
        by_id.setdefault(episode.episode_id, episode)

    return stable_sort_episodes(by_id.values())


def fetch_submission_catalog(
    submission_id: str | int,
    *,
    kaggle_executable: str | Path | None = None,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    allow_cli_fallback: bool = False,
    offline_replay_dir: str | Path | None = None,
    offline_only: bool = False,
    runner: Runner | None = None,
    sdk_fetcher: SdkFetcher | None = None,
    path_lookup: PathLookup = which,
    env: Mapping[str, str] | None = None,
) -> CatalogFetchResult:
    """Fetch all episodes for one submission, including validation episodes.

    ``sdk_fetcher`` is an injectable official-SDK adapter for tests and embedded
    callers.  Without it, the function invokes the SDK through the Python
    interpreter adjacent to the configured Kaggle executable.
    """

    target = _identifier(submission_id)
    if not target or not target.isdigit():
        raise SubmissionCatalogError(f"submission_id must contain digits, got {submission_id!r}")
    if offline_only:
        if offline_replay_dir is None:
            raise SubmissionCatalogError("offline_only requires offline_replay_dir")

        return catalog_from_replay_dir(offline_replay_dir, submission_id=target)

    if sdk_fetcher is not None:
        episodes = _normalize_episode_collection(sdk_fetcher(target))

        return CatalogFetchResult(
            submission_id=target,
            metadata_status=CATALOG_SOURCE_SDK,
            episodes=episodes,
            source_command=("in_process_sdk_fetcher", target),
        )

    runner = runner or _default_runner
    try:
        kaggle = resolve_kaggle_executable_path(
            kaggle_executable,
            out_root=out_root,
            env=env,
            path_lookup=path_lookup,
        )
    except SubmissionCatalogError as exc:
        if offline_replay_dir is None:
            raise
        offline = catalog_from_replay_dir(offline_replay_dir, submission_id=target)

        return CatalogFetchResult(
            submission_id=offline.submission_id,
            metadata_status=offline.metadata_status,
            episodes=offline.episodes,
            source_command=offline.source_command,
            warnings=(f"online catalog unavailable: {exc}",),
        )
    sdk_error: str | None = None
    python_executable: str | None = None
    try:
        python_executable = resolve_adjacent_python(kaggle, path_lookup=path_lookup)
        command = sdk_catalog_command(python_executable, target)
        result = _run(command, runner=runner)
        if _returncode(result) != 0:
            raise SubmissionCatalogError(
                f"Kaggle SDK catalog command failed with return code {_returncode(result)}: {_stderr(result) or _stdout(result)}"
            )
        episodes = _normalize_episode_collection(_parse_sdk_output(_stdout(result)))

        return CatalogFetchResult(
            submission_id=target,
            metadata_status=CATALOG_SOURCE_SDK,
            episodes=episodes,
            source_command=(python_executable, "-c", "<sdk_catalog_exporter>", target),
            kaggle_executable=kaggle,
            python_executable=python_executable,
        )
    except SubmissionCatalogError as exc:
        sdk_error = str(exc)

    if allow_cli_fallback:
        command = [kaggle, "competitions", "episodes", target, "--csv", "--quiet"]
        try:
            result = _run(command, runner=runner)
        except SubmissionCatalogError as exc:
            sdk_error += f"; CLI fallback failed: {exc}"
        else:
            if _returncode(result) == 0:
                episodes = _normalize_episode_collection(_parse_cli_catalog(_stdout(result)))

                return CatalogFetchResult(
                    submission_id=target,
                    metadata_status=CATALOG_SOURCE_CLI,
                    episodes=episodes,
                    source_command=tuple(command),
                    kaggle_executable=kaggle,
                    python_executable=python_executable,
                    warnings=(f"full SDK catalog unavailable: {sdk_error}",),
                )
            sdk_error += f"; CLI fallback failed: {_stderr(result) or _stdout(result)}"

    if offline_replay_dir is not None:
        offline = catalog_from_replay_dir(offline_replay_dir, submission_id=target)

        return CatalogFetchResult(
            submission_id=offline.submission_id,
            metadata_status=offline.metadata_status,
            episodes=offline.episodes,
            source_command=offline.source_command,
            kaggle_executable=kaggle,
            python_executable=python_executable,
            warnings=(f"online catalog unavailable: {sdk_error}",),
        )

    raise SubmissionCatalogError(sdk_error or "Kaggle SDK catalog fetch failed")


def _episode_id_from_filename(path: Path) -> str | None:
    name = path.name
    prefix = "episode-"
    suffix = "-replay.json"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    value = name[len(prefix) : -len(suffix)]

    return value if value.isdigit() else None


def _embedded_episode_id(data: Mapping[str, Any]) -> str | None:
    info = _read_field(data, "info", default={})
    from_info = _identifier(_read_field(info, "episode_id", "episodeId", "EpisodeId"))
    if from_info:
        return from_info
    direct = _identifier(_read_field(data, "episode_id", "episodeId", "EpisodeId"))
    if direct:
        return direct
    generic_id = _identifier(_read_field(data, "id"))

    return generic_id if generic_id and generic_id.isdigit() else None


def _offline_episode(path: Path) -> CatalogEpisode:
    expected = _episode_id_from_filename(path)
    if expected is None:
        raise SubmissionCatalogError(f"offline replay filename is not canonical: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionCatalogError(f"offline replay is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SubmissionCatalogError(f"offline replay root must be a JSON object: {path}")
    embedded = _embedded_episode_id(data)
    if embedded != expected:
        raise SubmissionCatalogError(f"offline replay EpisodeId mismatch: expected {expected}, got {embedded}")

    info = _read_field(data, "info", default={})
    names = _read_field(info, "TeamNames", "teamNames", "team_names", default=[])
    rewards = _read_field(data, "rewards", default=[])
    statuses = _read_field(data, "statuses", default=[])
    names = list(names) if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)) else []
    rewards = list(rewards) if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)) else []
    statuses = list(statuses) if isinstance(statuses, Sequence) and not isinstance(statuses, (str, bytes, bytearray)) else []
    count = max(len(names), len(rewards), len(statuses))
    agents = tuple(
        CatalogAgent(
            index=index,
            team_name=_text(names[index]) if index < len(names) else None,
            reward=_reward(rewards[index]) if index < len(rewards) else None,
            state=_text(statuses[index]) if index < len(statuses) else None,
        )
        for index in range(count)
    )

    return CatalogEpisode(episode_id=expected, agents=agents)


def catalog_from_replay_dir(replay_dir: str | Path, *, submission_id: str | int) -> CatalogFetchResult:
    """Build an explicitly limited catalog from already-downloaded replays."""

    root = Path(replay_dir)
    if not root.is_dir():
        raise SubmissionCatalogError(f"offline replay directory not found: {root}")
    episodes = [_offline_episode(path) for path in sorted(root.rglob("episode-*-replay.json"))]
    normalized = _normalize_episode_collection(episodes)

    return CatalogFetchResult(
        submission_id=str(submission_id),
        metadata_status=CATALOG_SOURCE_OFFLINE,
        episodes=normalized,
        source_command=("offline_replay_dir", str(root)),
        warnings=("offline replay metadata has no authoritative submission/team identity or episode type",),
    )


def validate_replay_file(path: str | Path, *, expected_episode_id: str | int) -> ReplayCacheRecord:
    """Validate JSON shape and embedded EpisodeId, then produce an integrity receipt."""

    replay_path = Path(path)
    expected = str(expected_episode_id)
    try:
        raw = replay_path.read_bytes()
        byte_size = len(raw)
        if byte_size <= 0:
            raise SubmissionCatalogError("replay file is empty")
        data = json.loads(raw)
        if not isinstance(data, Mapping):
            raise SubmissionCatalogError("replay root must be a JSON object")
        embedded = _embedded_episode_id(data)
        if embedded != expected:
            raise SubmissionCatalogError(f"EpisodeId mismatch: expected {expected}, got {embedded}")
    except (OSError, json.JSONDecodeError, SubmissionCatalogError) as exc:
        return ReplayCacheRecord(
            episode_id=expected,
            path=str(replay_path),
            status="invalid",
            error=str(exc),
        )

    return ReplayCacheRecord(
        episode_id=expected,
        path=str(replay_path),
        status="cached",
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=byte_size,
    )


def _offline_candidate(root: Path | None, episode_id: str) -> Path | None:
    if root is None:
        return None
    direct = root / f"episode-{episode_id}-replay.json"
    if direct.is_file():
        return direct
    matches = list(root.rglob(f"episode-{episode_id}-replay.json"))

    return matches[0] if len(matches) == 1 else None


def _atomic_import(source: Path, target: Path, *, episode_id: str) -> ReplayCacheRecord:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        validated = validate_replay_file(temp_path, expected_episode_id=episode_id)
        if not validated.is_valid:
            return ReplayCacheRecord(
                episode_id=episode_id,
                path=str(target),
                status="failed",
                source_path=str(source),
                error=f"offline replay failed validation: {validated.error}",
            )
        os.replace(temp_path, target)

        return ReplayCacheRecord(
            episode_id=episode_id,
            path=str(target),
            status="offline_imported",
            sha256=validated.sha256,
            byte_size=validated.byte_size,
            source_path=str(source),
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _download_one(
    episode_id: str,
    *,
    target: Path,
    kaggle_executable: str,
    runner: Runner,
) -> ReplayCacheRecord:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"episode-{episode_id}-", dir=target.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        command = [kaggle_executable, "competitions", "replay", episode_id, "-p", str(temp_dir), "--quiet"]
        result = _run(command, runner=runner)
        if _returncode(result) != 0:
            return ReplayCacheRecord(
                episode_id=episode_id,
                path=str(target),
                status="failed",
                error=f"Kaggle replay command failed with return code {_returncode(result)}: {_stderr(result) or _stdout(result)}",
            )
        downloaded = temp_dir / f"episode-{episode_id}-replay.json"
        validated = validate_replay_file(downloaded, expected_episode_id=episode_id)
        if not validated.is_valid:
            return ReplayCacheRecord(
                episode_id=episode_id,
                path=str(target),
                status="failed",
                error=f"downloaded replay failed validation: {validated.error}",
            )
        os.replace(downloaded, target)

        return ReplayCacheRecord(
            episode_id=episode_id,
            path=str(target),
            status="downloaded",
            sha256=validated.sha256,
            byte_size=validated.byte_size,
        )


def _cache_one(
    episode_id: str,
    *,
    cache_dir: Path,
    offline_replay_dir: Path | None,
    kaggle_executable: str | None,
    force: bool,
    runner: Runner,
) -> ReplayCacheRecord:
    target = cache_dir / f"episode-{episode_id}-replay.json"
    invalid_existing: str | None = None
    if target.is_file() and not force:
        existing = validate_replay_file(target, expected_episode_id=episode_id)
        if existing.is_valid:
            return existing
        invalid_existing = existing.error

    offline = _offline_candidate(offline_replay_dir, episode_id)
    if offline is not None:
        if offline.resolve() == target.resolve():
            same_file = validate_replay_file(target, expected_episode_id=episode_id)
            if same_file.is_valid:
                return same_file
        else:
            imported = _atomic_import(offline, target, episode_id=episode_id)
            if imported.is_valid:
                return imported
            if kaggle_executable is None:
                return imported

    if kaggle_executable is None:
        detail = f"; invalid existing cache: {invalid_existing}" if invalid_existing else ""

        return ReplayCacheRecord(
            episode_id=episode_id,
            path=str(target),
            status="failed",
            error=f"no valid offline replay and Kaggle download is disabled{detail}",
        )

    downloaded = _download_one(
        episode_id,
        target=target,
        kaggle_executable=kaggle_executable,
        runner=runner,
    )
    if not downloaded.is_valid and invalid_existing:
        return ReplayCacheRecord(
            episode_id=downloaded.episode_id,
            path=downloaded.path,
            status=downloaded.status,
            error=f"{downloaded.error}; invalid existing cache: {invalid_existing}",
        )

    return downloaded


def cache_submission_replays(
    episodes: Iterable[CatalogEpisode | str | int],
    *,
    cache_dir: str | Path = DEFAULT_REPLAY_CACHE,
    offline_replay_dir: str | Path | None = None,
    kaggle_executable: str | Path | None = None,
    workers: int = 4,
    force: bool = False,
    runner: Runner | None = None,
) -> ReplayCacheResult:
    """Validate/cache selected replays with bounded concurrent downloads."""

    if workers <= 0:
        raise SubmissionCatalogError("workers must be positive")
    ids: list[str] = []
    seen: set[str] = set()
    for value in episodes:
        episode_id = value.episode_id if isinstance(value, CatalogEpisode) else str(value)
        if not episode_id.isdigit():
            raise SubmissionCatalogError(f"episode id must contain digits, got {episode_id!r}")
        if episode_id not in seen:
            seen.add(episode_id)
            ids.append(episode_id)
    if not ids:
        return ReplayCacheResult(records=())

    root = Path(cache_dir)
    offline_root = Path(offline_replay_dir) if offline_replay_dir is not None else None
    if offline_root is not None and not offline_root.is_dir():
        raise SubmissionCatalogError(f"offline replay directory not found: {offline_root}")
    runner = runner or _default_runner
    kaggle = str(kaggle_executable) if kaggle_executable is not None else None

    records_by_id: dict[str, ReplayCacheRecord] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(ids)), thread_name_prefix="kaggle-replay") as executor:
        futures = {
            executor.submit(
                _cache_one,
                episode_id,
                cache_dir=root,
                offline_replay_dir=offline_root,
                kaggle_executable=kaggle,
                force=force,
                runner=runner,
            ): episode_id
            for episode_id in ids
        }
        for future in as_completed(futures):
            episode_id = futures[future]
            try:
                records_by_id[episode_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - convert worker failures into explicit receipts
                records_by_id[episode_id] = ReplayCacheRecord(
                    episode_id=episode_id,
                    path=str(root / f"episode-{episode_id}-replay.json"),
                    status="failed",
                    error=str(exc),
                )

    return ReplayCacheResult(records=tuple(records_by_id[episode_id] for episode_id in ids))


def catalog_sha256(episodes: Iterable[CatalogEpisode]) -> str:
    payload = [episode.to_dict() for episode in stable_sort_episodes(episodes)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()
