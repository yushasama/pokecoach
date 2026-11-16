"""Throttle-aware, resumable replay-mining queue for K1.6.

Downloads public Kaggle replays in fixed-size batches and stops the moment the Kaggle
API throttles. The simulations replay endpoint enforces a per-account window quota;
once it is exhausted the CLI stops returning data and prints "Authentication required
to call the Kaggle API" for every call. Rather than burning thousands of doomed
requests, this queue detects the wall, stops, ingests what it just fetched, refreshes
the corpus report, and reports how many episodes remain. Re-running the same command
later resumes: already-downloaded replays are skipped, so only missing episodes are
attempted.

Infrastructure only. No gameplay/scoring/policy/agents/turn-plan/default-`v2.4` change;
never touches root `main.py`/`deck.csv`/`cg/`; no ReplayPolicy / replay-prior online
integration / value model / RL / belief model / search is started here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Callable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown
from deepmew.kaggle_replays.corpus import CorpusOptions, CorpusResult, report_kaggle_replay_corpus
from deepmew.kaggle_replays.download import (
    DEFAULT_COMPETITION,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPLAY_ROOT,
    REPLAY_DOWNLOADED,
    DownloadOptions,
    KaggleDownloadError,
    KaggleDownloadResult,
    PathLookup,
    Runner,
    download_kaggle_replays,
    replay_output_path,
)
from deepmew.kaggle_replays.episode_sources import unique_first_seen
from deepmew.kaggle_replays.ingest import ingest_episode

KAGGLE_QUEUE_KIND = "kaggle_queue"


class ReplayQueueError(ValueError):
    """Raised when the queue cannot start with the requested inputs."""


@dataclass(frozen=True)
class QueueOptions:
    episode_ids: list[str]
    section: str
    competition: str = DEFAULT_COMPETITION
    batch_size: int = 50
    max_batches: int | None = None
    stop_on_throttle: bool = True
    ingest_after: bool = True
    corpus_after: bool = True
    force: bool = False
    sleep_seconds: float = 0.0
    replay_root: Path = Path(DEFAULT_REPLAY_ROOT)
    out_root: Path = Path(DEFAULT_OUT_ROOT)
    kaggle_exe: str | None = None
    command: str = ""
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueueResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    batches: list[dict[str, Any]]
    ingested_episode_ids: list[str]
    corpus_result: CorpusResult | None
    summary: dict[str, Any]


def _coerce_options(options: QueueOptions) -> QueueOptions:
    if not str(options.section).strip():
        raise ReplayQueueError("section is required")
    episode_ids = unique_first_seen(str(value).strip() for value in options.episode_ids)
    bad = [value for value in episode_ids if not value.isdigit()]
    if bad:
        raise ReplayQueueError(f"episode ids must be digits: {bad[:5]}")
    if options.batch_size <= 0:
        raise ReplayQueueError("batch_size must be positive")
    if options.max_batches is not None and options.max_batches < 0:
        raise ReplayQueueError("max_batches must be non-negative")
    if options.sleep_seconds < 0:
        raise ReplayQueueError("sleep_seconds must be non-negative")
    return QueueOptions(
        episode_ids=episode_ids,
        section=str(options.section).strip(),
        competition=options.competition or DEFAULT_COMPETITION,
        batch_size=int(options.batch_size),
        max_batches=options.max_batches,
        stop_on_throttle=bool(options.stop_on_throttle),
        ingest_after=bool(options.ingest_after),
        corpus_after=bool(options.corpus_after),
        force=bool(options.force),
        sleep_seconds=float(options.sleep_seconds),
        replay_root=Path(options.replay_root),
        out_root=Path(options.out_root),
        kaggle_exe=str(options.kaggle_exe) if options.kaggle_exe else None,
        command=options.command,
        args=options.args,
    )


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def mine_replay_queue(
    options: QueueOptions,
    *,
    runner: Runner | None = None,
    download_fn: Callable[..., KaggleDownloadResult] = download_kaggle_replays,
    ingest_fn: Callable[..., Any] = ingest_episode,
    corpus_fn: Callable[..., CorpusResult] = report_kaggle_replay_corpus,
    path_lookup: PathLookup = which,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> QueueResult:
    options = _coerce_options(options)
    run_dir = create_run_dir(kind=KAGGLE_QUEUE_KIND, out_root=options.out_root, suffix=options.section)
    replay_root = Path(options.replay_root)

    # Resume: only attempt episodes whose replay JSON is not already on disk.
    pending: list[str] = []
    already = 0
    for episode_id in options.episode_ids:
        if not options.force and replay_output_path(options.section, episode_id, replay_root).is_file():
            already += 1
        else:
            pending.append(episode_id)

    batches: list[dict[str, Any]] = []
    downloaded_ids: list[str] = []
    throttled_stop = False
    for batch_index, batch in enumerate(_chunks(pending, options.batch_size), start=1):
        if options.max_batches is not None and batch_index > options.max_batches:
            break
        result = download_fn(
            DownloadOptions(
                episode_ids=batch,
                section=options.section,
                competition=options.competition,
                force=options.force,
                ingest_after=False,
                sleep_seconds=options.sleep_seconds,
                replay_root=replay_root,
                out_root=options.out_root,
                kaggle_exe=options.kaggle_exe,
                command=options.command,
                args={**(options.args or {}), "source": "kaggle_queue", "batch": batch_index},
            ),
            runner=runner,
            path_lookup=path_lookup,
            sleep_fn=sleep_fn,
        )
        batch_downloaded = [record["episode_id"] for record in result.replay_records if record["status"] == REPLAY_DOWNLOADED]
        downloaded_ids.extend(batch_downloaded)
        batch_throttled = int(result.summary.get("throttled_replays", 0) or 0)
        batches.append(
            {
                "batch": batch_index,
                "requested": len(batch),
                "downloaded": len(batch_downloaded),
                "skipped": int(result.summary.get("skipped_replays", 0) or 0),
                "failed": int(result.summary.get("failed_replays", 0) or 0),
                "throttled": batch_throttled,
                "download_manifest": str(result.manifest_path),
            }
        )
        if batch_throttled and options.stop_on_throttle:
            throttled_stop = True
            break
        if options.sleep_seconds:
            sleep_fn(options.sleep_seconds)

    # Ingest only what this run downloaded (keeps resumed runs cheap).
    ingested_ids: list[str] = []
    ingest_errors: list[dict[str, str]] = []
    if options.ingest_after:
        for episode_id in downloaded_ids:
            try:
                ingest_fn(
                    replay_output_path(options.section, episode_id, replay_root),
                    out_root=str(options.out_root),
                    opponent_decks_root=str(Path(options.out_root) / "opponent_decks"),
                    command=options.command,
                )
                ingested_ids.append(episode_id)
            except Exception as exc:  # noqa: BLE001 - record, never abort the queue on one bad replay
                ingest_errors.append({"episode_id": episode_id, "error": str(exc)})

    corpus_result: CorpusResult | None = None
    if options.corpus_after:
        corpus_result = corpus_fn(
            CorpusOptions(
                replay_root=replay_root,
                ingest_root=Path(options.out_root) / "kaggle_ingest",
                opponent_decks_root=Path(options.out_root) / "opponent_decks",
                out_root=options.out_root,
                section=options.section,
                command=options.command,
                args={**(options.args or {}), "source": "kaggle_queue"},
            )
        )

    remaining = len(pending) - len(downloaded_ids)
    summary = {
        "section": options.section,
        "requested_episode_count": len(options.episode_ids),
        "already_downloaded": already,
        "pending_at_start": len(pending),
        "batches_run": len(batches),
        "batch_size": options.batch_size,
        "downloaded": len(downloaded_ids),
        "ingested": len(ingested_ids),
        "ingest_errors": len(ingest_errors),
        "throttled_stop": throttled_stop,
        "remaining": remaining,
        "complete": remaining == 0,
        "corpus_after": corpus_result is not None,
    }
    if corpus_result is not None:
        summary["corpus_unique_episode_count"] = corpus_result.summary.get("unique_replay_episode_count")
        summary["corpus_deck_artifact_count"] = corpus_result.summary.get("deck_artifact_count")
        summary["corpus_unique_deck_hash_count"] = corpus_result.summary.get("unique_deck_hash_count")

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_QUEUE_KIND,
        command=options.command,
        status="completed" if not throttled_stop else "stopped_on_throttle",
        inputs={
            "args": options.args or {},
            "competition": options.competition,
            "section": options.section,
            "batch_size": options.batch_size,
            "max_batches": options.max_batches,
            "stop_on_throttle": options.stop_on_throttle,
            "ingest_after": options.ingest_after,
            "corpus_after": options.corpus_after,
            "force": options.force,
            "sleep_seconds": options.sleep_seconds,
            "requested_episode_count": len(options.episode_ids),
        },
        outputs={
            "queue_report": str(run_dir.output_path("queue_report.md")),
            "corpus_report": None if corpus_result is None else str(corpus_result.report_path),
            "replay_dir": str(replay_root / options.section),
        },
        summary=summary,
    )
    manifest["batches"] = batches
    manifest["ingested_episode_ids"] = ingested_ids
    manifest["ingest_errors"] = ingest_errors

    report_path = write_markdown(run_dir.output_path("queue_report.md"), _report(options, summary, batches, ingest_errors))
    manifest_path = write_manifest(run_dir.manifest_path, manifest)

    return QueueResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        batches=batches,
        ingested_episode_ids=ingested_ids,
        corpus_result=corpus_result,
        summary=summary,
    )


def _report(
    options: QueueOptions,
    summary: dict[str, Any],
    batches: list[dict[str, Any]],
    ingest_errors: list[dict[str, str]],
) -> str:
    lines = [
        "# Kaggle Replay Mining Queue",
        "",
        f"section={options.section}",
        f"competition={options.competition}",
        f"replay_dir={Path(options.replay_root) / options.section}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    if summary["throttled_stop"]:
        lines += [
            "",
            "> Stopped on Kaggle API throttle (per-account window quota). "
            f"{summary['remaining']} episode(s) remain. Re-run the same command after the "
            "quota window resets; already-downloaded replays are skipped.",
        ]
    lines += ["", "## Batches", "", "| batch | requested | downloaded | failed | throttled |", "|---:|---:|---:|---:|---:|"]
    for batch in batches:
        lines.append(
            f"| {batch['batch']} | {batch['requested']} | {batch['downloaded']} | {batch['failed']} | {batch['throttled']} |"
        )
    lines += ["", "## Ingest Errors", ""]
    if ingest_errors:
        lines.extend(f"- {error['episode_id']}: {error['error']}" for error in ingest_errors[:20])
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Boundary",
        "",
        "- K1.6 replay infrastructure only: batched, resumable download + ingest + corpus refresh.",
        "- No gameplay/scoring/policy/agents/turn-plan/default-`v2.4` change; root `main.py`/`deck.csv`/`cg/` untouched.",
        "- No ReplayPolicy, replay-prior online integration, value model, RL, belief model, or search.",
    ]
    return "\n".join(lines)
