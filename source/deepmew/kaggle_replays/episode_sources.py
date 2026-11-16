"""Normalize Kaggle episode-ID sources for replay farming.

K1.6 keeps episode discovery as data plumbing: collect public episode IDs from
text files, pasted URLs, or existing replay filenames, preserve first-seen
order, and write a reproducible source file plus manifest/report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown, write_text

KAGGLE_EPISODE_SOURCE_KIND = "kaggle_episode_sources"
DEFAULT_OUT_ROOT = ".research"
EPISODE_ID_RE = re.compile(r"\b\d{6,}\b")
REPLAY_FILENAME_RE = re.compile(r"episode-(\d+)-replay\.json$")


class EpisodeSourceError(ValueError):
    """Raised when a requested episode source cannot be read."""


@dataclass(frozen=True)
class EpisodeSourceResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    episode_file: Path
    episode_ids: list[str]
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def extract_episode_ids_from_text(text: str) -> list[str]:
    """Return digit episode IDs from text, skipping blank/comment lines."""

    ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0]
        ids.extend(match.group(0) for match in EPISODE_ID_RE.finditer(line))
    return ids


def unique_first_seen(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def scan_replay_dir(path: str | Path) -> list[dict[str, str]]:
    root = Path(path)
    if not root.exists():
        raise EpisodeSourceError(f"replay scan directory not found: {root}")
    records: list[dict[str, str]] = []
    for replay_path in sorted(root.rglob("episode-*-replay.json")):
        match = REPLAY_FILENAME_RE.match(replay_path.name)
        if match:
            records.append({"episode_id": match.group(1), "source": str(replay_path), "source_kind": "replay_file"})
    return records


def _records_from_input(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EpisodeSourceError(f"episode input file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return [
        {"episode_id": episode_id, "source": str(path), "source_kind": "input_file"}
        for episode_id in extract_episode_ids_from_text(text)
    ]


def _dedupe_records(records: list[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]]]:
    seen: set[str] = set()
    episode_ids: list[str] = []
    annotated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        episode_id = record["episode_id"]
        duplicate = episode_id in seen
        if not duplicate:
            seen.add(episode_id)
            episode_ids.append(episode_id)
        annotated.append({**record, "input_index": index, "duplicate": duplicate})
    return episode_ids, annotated


def build_episode_source(
    *,
    input_paths: list[str | Path] | None = None,
    scan_replay_dirs: list[str | Path] | None = None,
    out_path: str | Path,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    command: str | list[str] = "",
    args: dict[str, Any] | None = None,
) -> EpisodeSourceResult:
    records: list[dict[str, str]] = []
    for input_path in input_paths or []:
        records.extend(_records_from_input(Path(input_path)))
    for replay_dir in scan_replay_dirs or []:
        records.extend(scan_replay_dir(replay_dir))

    episode_ids, annotated_records = _dedupe_records(records)
    out_file = write_text(Path(out_path), "\n".join(episode_ids) + ("\n" if episode_ids else ""))
    run_dir = create_run_dir(kind=KAGGLE_EPISODE_SOURCE_KIND, out_root=out_root, suffix=Path(out_path).stem)

    summary = {
        "input_file_count": len(input_paths or []),
        "scan_replay_dir_count": len(scan_replay_dirs or []),
        "raw_episode_id_count": len(records),
        "unique_episode_id_count": len(episode_ids),
        "duplicate_episode_id_count": len(records) - len(episode_ids),
    }
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_EPISODE_SOURCE_KIND,
        command=command,
        inputs={
            "args": args or {},
            "input_paths": [str(path) for path in input_paths or []],
            "scan_replay_dirs": [str(path) for path in scan_replay_dirs or []],
        },
        outputs={
            "episode_file": str(out_file),
            "source_report": str(run_dir.output_path("source_report.md")),
        },
        summary=summary,
    )
    manifest["records"] = annotated_records

    lines = [
        "# Kaggle Episode Source",
        "",
        f"episode_file={out_file}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    lines += ["", "## Episodes", "", "| episode | source_kind | duplicate | source |", "|---|---|---|---|"]
    for record in annotated_records:
        lines.append(
            f"| {record['episode_id']} | {record['source_kind']} | {record['duplicate']} | {record['source']} |"
        )

    report_path = write_markdown(run_dir.output_path("source_report.md"), "\n".join(lines))
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return EpisodeSourceResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        episode_file=out_file,
        episode_ids=episode_ids,
        records=annotated_records,
        summary=summary,
    )
