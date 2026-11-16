"""Download public Kaggle replay JSONs for offline DeepMew mining.

This is K1.5 infrastructure only. It shells out to the Kaggle CLI for public
episode replays and optional agent logs, records a manifest/report, and can
hand the replay directory to the accepted K1 ingest CLI. It never imports the
runtime policy path or touches root submission files.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Callable, Mapping, Sequence

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown

DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_REPLAY_ROOT = ".research/kaggle_replays"
DEFAULT_LOG_ROOT = ".research/kaggle_logs"
DEFAULT_OUT_ROOT = ".research"
KAGGLE_DOWNLOAD_KIND = "kaggle_downloads"

KAGGLE_EXE_ENV_VAR = "DEEPMEW_KAGGLE_EXE"
KAGGLE_CLI_CONFIG_NAME = "kaggle_cli.json"

REPLAY_DOWNLOADED = "downloaded"
REPLAY_SKIPPED_EXISTING = "skipped_existing"
REPLAY_FAILED = "failed"
REPLAY_THROTTLED = "throttled"
REPLAY_DRY_RUN = "dry_run"

LOG_DOWNLOADED = "downloaded"
LOG_SKIPPED_EXISTING = "skipped_existing"
LOG_FORBIDDEN = "forbidden"
LOG_FAILED = "failed"
LOG_DRY_RUN = "dry_run"

Runner = Callable[..., Any]
PathLookup = Callable[[str], str | None]

_KAGGLE_SCRIPT_DIR_HINT = 'py -c "import sysconfig; print(sysconfig.get_path(\'scripts\'))"'


class KaggleDownloadError(ValueError):
    """Raised when K1.5 cannot start or validate the requested download."""


@dataclass(frozen=True)
class KaggleExecutable:
    command: str
    explicit: bool
    resolution: str

    def to_manifest_fields(self) -> dict[str, Any]:
        return {
            "kaggle_executable": self.command,
            "kaggle_executable_explicit": self.explicit,
            "kaggle_executable_resolution": self.resolution,
        }


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @classmethod
    def from_completed(cls, completed: Any, fallback_args: Sequence[str]) -> "CommandResult":
        args = list(getattr(completed, "args", fallback_args) or fallback_args)
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        return cls(args=args, returncode=int(getattr(completed, "returncode", 1)), stdout=str(stdout), stderr=str(stderr))

    def to_dict(self) -> dict[str, Any]:
        return {
            "args": self.args,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class DownloadOptions:
    episode_ids: list[str]
    section: str
    competition: str = DEFAULT_COMPETITION
    logs: list[int] | None = None
    force: bool = False
    dry_run: bool = False
    ingest_after: bool = False
    sleep_seconds: float = 0.0
    replay_root: Path = Path(DEFAULT_REPLAY_ROOT)
    log_root: Path = Path(DEFAULT_LOG_ROOT)
    out_root: Path = Path(DEFAULT_OUT_ROOT)
    kaggle_exe: str | None = None
    command: str = ""
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class KaggleDownloadResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    replay_dir: Path
    log_dir: Path
    manifest: dict[str, Any]
    replay_records: list[dict[str, Any]]
    log_records: list[dict[str, Any]]
    ingest_after_result: dict[str, Any] | None
    summary: dict[str, Any]


def parse_episode_file(path: str | Path) -> list[str]:
    episode_file = Path(path)
    if not episode_file.is_file():
        raise KaggleDownloadError(f"episodes file not found: {episode_file}")
    episode_ids: list[str] = []
    for line_number, raw_line in enumerate(episode_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not line.isdigit():
            raise KaggleDownloadError(f"{episode_file}:{line_number}: episode id must be digits, got {line!r}")
        episode_ids.append(line)
    return episode_ids


def replay_output_path(section: str, episode_id: str | int, replay_root: str | Path = DEFAULT_REPLAY_ROOT) -> Path:
    return Path(replay_root) / section / f"episode-{episode_id}-replay.json"


def log_output_path(
    section: str,
    episode_id: str | int,
    agent_index: int,
    log_root: str | Path = DEFAULT_LOG_ROOT,
) -> Path:
    return Path(log_root) / section / f"episode-{episode_id}-agent-{agent_index}-logs.json"


def replay_command(episode_id: str | int, replay_dir: str | Path, *, kaggle_exe: str = "kaggle") -> list[str]:
    return [kaggle_exe, "competitions", "replay", str(episode_id), "-p", str(replay_dir)]


def log_command(episode_id: str | int, agent_index: int, log_dir: str | Path, *, kaggle_exe: str = "kaggle") -> list[str]:
    return [kaggle_exe, "competitions", "logs", str(episode_id), str(agent_index), "-p", str(log_dir)]


def ingest_after_command(replay_dir: str | Path) -> list[str]:
    return [sys.executable, str(Path("scripts") / "ingest_kaggle_replay.py"), "--input-dir", str(replay_dir)]


def _default_runner(command: Sequence[str], **kwargs: Any) -> Any:
    import subprocess

    return subprocess.run(list(command), **kwargs)


def _run_command(command: Sequence[str], *, runner: Runner) -> CommandResult:
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return CommandResult(args=list(command), returncode=127, stderr=str(exc))
    return CommandResult.from_completed(completed, command)


def kaggle_config_path(out_root: str | Path = DEFAULT_OUT_ROOT) -> Path:
    """Return the durable Kaggle-CLI config location under the research root."""
    return Path(out_root) / "local" / KAGGLE_CLI_CONFIG_NAME


def load_configured_kaggle_exe(config_path: str | Path | None) -> str | None:
    """Read ``kaggle_exe`` from a ``.research/local/kaggle_cli.json`` config, if present."""
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("kaggle_exe") if isinstance(data, dict) else None
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def resolve_kaggle_executable(
    kaggle_exe: str | Path | None,
    *,
    dry_run: bool = False,
    path_lookup: PathLookup = which,
    env: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> KaggleExecutable:
    """Resolve the Kaggle CLI by precedence: explicit arg, env var, config file, PATH.

    The config file (``.research/local/kaggle_cli.json``, written by
    ``scripts/configure_kaggle_cli.py``) lets future agents skip manual PATH debugging.
    """
    if kaggle_exe is not None and str(kaggle_exe).strip():
        return KaggleExecutable(command=str(kaggle_exe).strip(), explicit=True, resolution="explicit")
    environ = os.environ if env is None else env
    env_exe = environ.get(KAGGLE_EXE_ENV_VAR)
    if env_exe and env_exe.strip():
        return KaggleExecutable(command=env_exe.strip(), explicit=False, resolution="env")
    configured = load_configured_kaggle_exe(config_path)
    if configured:
        return KaggleExecutable(command=configured, explicit=False, resolution="config")
    if dry_run:
        return KaggleExecutable(command="kaggle", explicit=False, resolution="path")
    if path_lookup("kaggle"):
        return KaggleExecutable(command="kaggle", explicit=False, resolution="path")
    raise KaggleDownloadError(
        "Kaggle CLI executable not found on PATH. Rerun with `--kaggle-exe <path>`, set the "
        f"{KAGGLE_EXE_ENV_VAR} environment variable, run "
        "`python scripts/configure_kaggle_cli.py --kaggle-exe <path>`, or add Kaggle to this shell's "
        "PATH. To locate Python script installs in PowerShell, run: "
        f"{_KAGGLE_SCRIPT_DIR_HINT}"
    )


def ensure_kaggle_cli(kaggle: KaggleExecutable, *, runner: Runner) -> CommandResult:
    result = _run_command([kaggle.command, "--help"], runner=runner)
    if result.returncode == 127:
        raise KaggleDownloadError(
            f"Kaggle CLI executable not found: {kaggle.command}. Rerun with `--kaggle-exe <path>` "
            "or add Kaggle to this shell's PATH. To locate Python script installs in PowerShell, run: "
            f"{_KAGGLE_SCRIPT_DIR_HINT}"
        )
    if result.returncode != 0:
        raise KaggleDownloadError(
            f"Kaggle CLI health check failed (`{kaggle.command} --help` returned "
            f"{result.returncode}): {result.stderr or result.stdout}"
        )
    return result


def _is_forbidden(result: CommandResult) -> bool:
    text = f"{result.stdout}\n{result.stderr}".lower()
    return "403" in text or "forbidden" in text


def _is_throttled(result: CommandResult) -> bool:
    """Detect Kaggle's per-account window quota wall.

    Once the simulations replay quota is exhausted the CLI stops returning data and
    prints "Authentication required to call the Kaggle API" (even for metadata calls);
    classic HTTP 429/rate-limit phrasing is matched too.
    """
    text = f"{result.stdout}\n{result.stderr}".lower()
    return (
        "authentication required" in text
        or "too many requests" in text
        or "rate limit" in text
        or "429" in text
    )


def _download_replay(
    episode_id: str,
    options: DownloadOptions,
    *,
    replay_dir: Path,
    kaggle: KaggleExecutable,
    runner: Runner,
) -> dict[str, Any]:
    target = replay_output_path(options.section, episode_id, options.replay_root)
    command = replay_command(episode_id, replay_dir, kaggle_exe=kaggle.command)
    base = {"episode_id": episode_id, "path": str(target), "command": command}

    if target.is_file() and not options.force:
        return {**base, "status": REPLAY_SKIPPED_EXISTING}
    if options.dry_run:
        return {**base, "status": REPLAY_DRY_RUN}

    target.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(command, runner=runner)
    if result.returncode == 0 and target.is_file():
        return {**base, "status": REPLAY_DOWNLOADED, "result": result.to_dict()}
    if _is_throttled(result):
        return {**base, "status": REPLAY_THROTTLED, "error": "kaggle API throttled (per-account window quota exhausted)", "result": result.to_dict()}
    error = "kaggle replay command failed" if result.returncode != 0 else "expected replay file not found after command"
    return {**base, "status": REPLAY_FAILED, "error": error, "result": result.to_dict()}


def _download_log(
    episode_id: str,
    agent_index: int,
    options: DownloadOptions,
    *,
    log_dir: Path,
    kaggle: KaggleExecutable,
    runner: Runner,
) -> dict[str, Any]:
    target = log_output_path(options.section, episode_id, agent_index, options.log_root)
    command = log_command(episode_id, agent_index, log_dir, kaggle_exe=kaggle.command)
    base = {"episode_id": episode_id, "agent_index": agent_index, "path": str(target), "command": command}

    if target.is_file() and not options.force:
        return {**base, "status": LOG_SKIPPED_EXISTING}
    if options.dry_run:
        return {**base, "status": LOG_DRY_RUN}

    target.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(command, runner=runner)
    if result.returncode == 0 and target.is_file():
        return {**base, "status": LOG_DOWNLOADED, "result": result.to_dict()}
    if _is_forbidden(result):
        return {**base, "status": LOG_FORBIDDEN, "error": "opponent private log returned 403/forbidden", "result": result.to_dict()}
    error = "kaggle logs command failed" if result.returncode != 0 else "expected log file not found after command"
    return {**base, "status": LOG_FAILED, "error": error, "result": result.to_dict()}


def _count(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("status") == status)


def _summary(
    *,
    options: DownloadOptions,
    replay_records: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
    ingest_after_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "section": options.section,
        "competition": options.competition,
        "requested_episode_count": len(options.episode_ids),
        "downloaded_replays": _count(replay_records, REPLAY_DOWNLOADED),
        "skipped_replays": _count(replay_records, REPLAY_SKIPPED_EXISTING),
        "failed_replays": _count(replay_records, REPLAY_FAILED),
        "throttled_replays": _count(replay_records, REPLAY_THROTTLED),
        "dry_run_replays": _count(replay_records, REPLAY_DRY_RUN),
        "downloaded_logs": _count(log_records, LOG_DOWNLOADED),
        "skipped_logs": _count(log_records, LOG_SKIPPED_EXISTING),
        "forbidden_logs": _count(log_records, LOG_FORBIDDEN),
        "failed_logs": _count(log_records, LOG_FAILED),
        "dry_run_logs": _count(log_records, LOG_DRY_RUN),
        "ingest_after_returncode": None if ingest_after_result is None else ingest_after_result.get("returncode"),
    }


def _report(
    *,
    options: DownloadOptions,
    replay_dir: Path,
    log_dir: Path,
    replay_records: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
    ingest_after_result: dict[str, Any] | None,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Kaggle Replay Download",
        "",
        f"competition={options.competition}",
        f"section={options.section}",
        f"replay_dir={replay_dir}",
        f"log_dir={log_dir}",
        f"dry_run={options.dry_run} force={options.force} logs={options.logs if options.logs is not None else 'none'}",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    lines += ["", "## Replays", "", "| episode | status | path |", "|---|---|---|"]
    for record in replay_records:
        lines.append(f"| {record['episode_id']} | {record['status']} | {record['path']} |")
        if record.get("error"):
            lines.append(f"| {record['episode_id']} | error | {record['error']} |")

    lines += ["", "## Agent Logs", ""]
    if not log_records:
        lines.append("- Not requested. Opponent private logs are optional; public replay JSON is the source of truth.")
    else:
        lines += ["| episode | agent | status | path |", "|---|---:|---|---|"]
        for record in log_records:
            lines.append(f"| {record['episode_id']} | {record['agent_index']} | {record['status']} | {record['path']} |")
            if record.get("error"):
                lines.append(f"| {record['episode_id']} | {record['agent_index']} | error | {record['error']} |")

    lines += ["", "## Ingest After", ""]
    if ingest_after_result is None:
        lines.append("- Not requested or skipped for dry-run.")
    else:
        lines.append(f"- command: {' '.join(ingest_after_result['args'])}")
        lines.append(f"- returncode: {ingest_after_result['returncode']}")

    lines += [
        "",
        "## Deferred Discovery",
        "",
        "- K1.5 intentionally consumes explicit episode IDs or an episode-ID file.",
        "- Leaderboard/team/submission discovery and replay farming queues remain future K1.6/K1.7 work.",
        "",
        "## Boundary",
        "",
        "- Infrastructure only: no gameplay, scorer, policy, agent, turn-plan, default `v2.4`, root `main.py`, root `deck.csv`, or `cg/` change.",
        "- No replay-prior online integration, value model, RL, PPO, Deep RL, belief model, or search is started here.",
    ]
    return "\n".join(lines)


def _coerce_options(options: DownloadOptions) -> DownloadOptions:
    episode_ids = [str(episode_id).strip() for episode_id in options.episode_ids if str(episode_id).strip()]
    if not episode_ids:
        raise KaggleDownloadError("at least one episode id is required")
    bad = [episode_id for episode_id in episode_ids if not episode_id.isdigit()]
    if bad:
        raise KaggleDownloadError(f"episode ids must be digits: {bad}")
    logs = None if options.logs is None else [int(agent) for agent in options.logs]
    if logs is not None and any(agent < 0 for agent in logs):
        raise KaggleDownloadError(f"log agent indices must be non-negative: {logs}")
    if options.sleep_seconds < 0:
        raise KaggleDownloadError("sleep_seconds must be non-negative")
    return DownloadOptions(
        episode_ids=episode_ids,
        section=options.section,
        competition=options.competition or DEFAULT_COMPETITION,
        logs=logs,
        force=options.force,
        dry_run=options.dry_run,
        ingest_after=options.ingest_after,
        sleep_seconds=float(options.sleep_seconds),
        replay_root=Path(options.replay_root),
        log_root=Path(options.log_root),
        out_root=Path(options.out_root),
        kaggle_exe=str(options.kaggle_exe) if options.kaggle_exe is not None else None,
        command=options.command,
        args=options.args,
    )


def download_kaggle_replays(
    options: DownloadOptions,
    *,
    runner: Runner | None = None,
    ingest_runner: Runner | None = None,
    path_lookup: PathLookup = which,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> KaggleDownloadResult:
    options = _coerce_options(options)
    runner = runner or _default_runner
    ingest_runner = ingest_runner or runner

    replay_dir = Path(options.replay_root) / options.section
    log_dir = Path(options.log_root) / options.section
    kaggle = resolve_kaggle_executable(
        options.kaggle_exe,
        dry_run=options.dry_run,
        path_lookup=path_lookup,
        config_path=kaggle_config_path(options.out_root),
    )

    cli_check: dict[str, Any] | None = None
    if not options.dry_run:
        cli_check = ensure_kaggle_cli(kaggle, runner=runner).to_dict()

    run_dir = create_run_dir(kind=KAGGLE_DOWNLOAD_KIND, out_root=options.out_root, suffix=options.section)

    replay_records: list[dict[str, Any]] = []
    for episode_id in options.episode_ids:
        record = _download_replay(episode_id, options, replay_dir=replay_dir, kaggle=kaggle, runner=runner)
        replay_records.append(record)
        if not options.dry_run and options.sleep_seconds > 0:
            sleep_fn(options.sleep_seconds)

    log_records: list[dict[str, Any]] = []
    if options.logs is not None:
        for episode_id in options.episode_ids:
            for agent_index in options.logs:
                record = _download_log(episode_id, agent_index, options, log_dir=log_dir, kaggle=kaggle, runner=runner)
                log_records.append(record)
                if not options.dry_run and options.sleep_seconds > 0:
                    sleep_fn(options.sleep_seconds)

    ingest_after_result: dict[str, Any] | None = None
    if options.ingest_after and not options.dry_run:
        command = ingest_after_command(replay_dir)
        result = _run_command(command, runner=ingest_runner)
        ingest_after_result = result.to_dict()

    summary = _summary(
        options=options,
        replay_records=replay_records,
        log_records=log_records,
        ingest_after_result=ingest_after_result,
    )
    outputs = {
        "replay_dir": str(replay_dir),
        "log_dir": str(log_dir),
        "download_report": str(run_dir.output_path("download_report.md")),
    }
    if ingest_after_result is not None:
        outputs["ingest_after_replay_dir"] = str(replay_dir)

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_DOWNLOAD_KIND,
        command=options.command,
        status="completed" if (summary["failed_replays"] == 0 and summary["throttled_replays"] == 0) else "completed_with_replay_failures",
        inputs={
            "args": options.args or {},
            "competition": options.competition,
            "section": options.section,
            "requested_episode_ids": options.episode_ids,
            "logs": options.logs,
            "force": options.force,
            "dry_run": options.dry_run,
            "sleep_seconds": options.sleep_seconds,
            "kaggle_cli_check": cli_check,
            **kaggle.to_manifest_fields(),
        },
        outputs=outputs,
        summary=summary,
    )
    manifest["args"] = options.args or {}
    manifest["replays"] = replay_records
    manifest["logs"] = log_records
    manifest["ingest_after"] = ingest_after_result

    report_text = _report(
        options=options,
        replay_dir=replay_dir,
        log_dir=log_dir,
        replay_records=replay_records,
        log_records=log_records,
        ingest_after_result=ingest_after_result,
        summary=summary,
    )
    report_path = write_markdown(run_dir.output_path("download_report.md"), report_text)
    manifest_path = write_manifest(run_dir.manifest_path, manifest)

    return KaggleDownloadResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        replay_dir=replay_dir,
        log_dir=log_dir,
        manifest=manifest,
        replay_records=replay_records,
        log_records=log_records,
        ingest_after_result=ingest_after_result,
        summary=summary,
    )
