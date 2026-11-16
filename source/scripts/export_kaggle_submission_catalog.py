"""Export and optionally cache a complete Kaggle submission episode catalog."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.submission_postmortem.catalog import (  # noqa: E402
    DEFAULT_REPLAY_CACHE,
    SubmissionCatalogError,
    cache_submission_replays,
    catalog_sha256,
    fetch_submission_catalog,
    filter_catalog_episodes,
    resolve_kaggle_executable_path,
)
from deepmew.submission_postmortem.models import SCHEMA_VERSION, CatalogQuery  # noqa: E402


DEFAULT_CATALOG_ROOT = Path(".research/submission_postmortem/catalogs")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch every Kaggle episode for a submission, filter it, and optionally cache validated replays."
    )
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--episode-id", dest="episode_ids", action="append", default=[])
    parser.add_argument("--episode-min", type=int)
    parser.add_argument("--episode-max", type=int)
    parser.add_argument("--ordinal-min", type=int, help="Inclusive 1-based ordinal after (create_time, episode_id) sorting.")
    parser.add_argument("--ordinal-max", type=int, help="Inclusive 1-based ordinal after (create_time, episode_id) sorting.")
    parser.add_argument("--opponent", dest="opponent_terms", action="append", default=[], help="Case-insensitive team-name substring.")
    parser.add_argument("--opponent-team-id", action="append", default=[])
    parser.add_argument("--opponent-submission-id", action="append", default=[])
    parser.add_argument("--exclude-validation", action="store_true", help="Exclude validation episodes; included by default.")
    parser.add_argument("--kaggle-exe")
    parser.add_argument(
        "--allow-metadata-limited-cli-fallback",
        action="store_true",
        help="Explicitly allow CLI catalog fallback without per-seat submission/team metadata.",
    )
    parser.add_argument("--offline-replay-dir", help="Use already-downloaded episode-*-replay.json files before network download.")
    parser.add_argument("--offline-only", action="store_true", help="Build a metadata-limited catalog entirely from --offline-replay-dir.")
    parser.add_argument("--download-replays", action="store_true", help="Cache every selected replay after exporting the catalog.")
    parser.add_argument("--workers", type=int, default=4, help="Bounded concurrent replay downloads.")
    parser.add_argument("--force", action="store_true", help="Replace valid cached replays too.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_REPLAY_CACHE))
    parser.add_argument("--out-dir", help="Default: .research/submission_postmortem/catalogs/submission-<id>.")
    parser.add_argument("--out-root", default=".research", help="Root containing local Kaggle CLI configuration.")

    return parser.parse_args(argv)


def _query(args: argparse.Namespace) -> CatalogQuery:
    return CatalogQuery(
        episode_ids=frozenset(str(value) for value in args.episode_ids),
        episode_min=args.episode_min,
        episode_max=args.episode_max,
        ordinal_min=args.ordinal_min,
        ordinal_max=args.ordinal_max,
        opponent_terms=tuple(str(value) for value in args.opponent_terms),
        opponent_team_ids=frozenset(str(value) for value in args.opponent_team_id),
        opponent_submission_ids=frozenset(str(value) for value in args.opponent_submission_id),
        include_validation=not args.exclude_validation,
    )


def _write_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return path


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _jsonl_text(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n" for value in values)


def export_submission_catalog(args: argparse.Namespace) -> dict[str, Any]:
    query = _query(args)
    catalog = fetch_submission_catalog(
        args.submission_id,
        kaggle_executable=args.kaggle_exe,
        out_root=args.out_root,
        allow_cli_fallback=args.allow_metadata_limited_cli_fallback,
        offline_replay_dir=args.offline_replay_dir,
        offline_only=args.offline_only,
    )
    selected = filter_catalog_episodes(
        catalog.episodes,
        submission_id=catalog.submission_id,
        query=query,
    )
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_CATALOG_ROOT / f"submission-{catalog.submission_id}"
    all_rows = [episode.to_dict() for episode in catalog.episodes]
    selected_rows = [episode.to_dict() for episode in selected]
    _write_atomic(out_dir / "episodes.jsonl", _jsonl_text(all_rows))
    _write_atomic(out_dir / "selected_episodes.jsonl", _jsonl_text(selected_rows))

    replay_cache = None
    if args.download_replays:
        kaggle_executable = None
        if not args.offline_only:
            kaggle_executable = resolve_kaggle_executable_path(
                args.kaggle_exe,
                out_root=args.out_root,
            )
        replay_cache = cache_submission_replays(
            selected,
            cache_dir=args.cache_dir,
            offline_replay_dir=args.offline_replay_dir,
            kaggle_executable=kaggle_executable,
            workers=args.workers,
            force=args.force,
        )
        _write_atomic(
            out_dir / "replay_cache.jsonl",
            _jsonl_text([record.to_dict() for record in replay_cache.records]),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "submission_postmortem_catalog",
        "submission_id": catalog.submission_id,
        "metadata_status": catalog.metadata_status,
        "is_metadata_limited": catalog.is_metadata_limited,
        "query": query.to_dict(),
        "catalog_sha256": catalog_sha256(catalog.episodes),
        "selected_catalog_sha256": catalog_sha256(selected),
        "summary": {
            "listed": len(catalog.episodes),
            "selected": len(selected),
            "excluded_by_filter": len(catalog.episodes) - len(selected),
            "validation_listed": sum(episode.is_validation for episode in catalog.episodes),
            "validation_selected": sum(episode.is_validation for episode in selected),
            "replay_cache_requested": 0 if replay_cache is None else len(replay_cache.records),
            "replay_cache_valid": 0 if replay_cache is None else sum(record.is_valid for record in replay_cache.records),
            "replay_cache_complete": None if replay_cache is None else replay_cache.complete,
        },
        "outputs": {
            "manifest": str(out_dir / "manifest.json"),
            "episodes": str(out_dir / "episodes.jsonl"),
            "selected_episodes": str(out_dir / "selected_episodes.jsonl"),
            "replay_cache": None if replay_cache is None else str(out_dir / "replay_cache.jsonl"),
            "replay_cache_dir": str(Path(args.cache_dir)),
        },
        "catalog_source": {
            "source_command": list(catalog.source_command),
            "kaggle_executable": catalog.kaggle_executable,
            "python_executable": catalog.python_executable,
            "warnings": list(catalog.warnings),
        },
    }
    _write_atomic(out_dir / "manifest.json", _json_text(manifest))

    return manifest


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    try:
        manifest = export_submission_catalog(args)
    except SubmissionCatalogError as exc:
        print(f"catalog export failed: {exc}", file=sys.stderr)
        return 1

    summary = manifest["summary"]
    print(f"submission_id={manifest['submission_id']} metadata_status={manifest['metadata_status']}")
    print(
        f"listed={summary['listed']} selected={summary['selected']} "
        f"validation_selected={summary['validation_selected']}"
    )
    if summary["replay_cache_complete"] is not None:
        print(
            f"replay_cache_valid={summary['replay_cache_valid']}/"
            f"{summary['replay_cache_requested']} complete={summary['replay_cache_complete']}"
        )
    print(f"manifest={manifest['outputs']['manifest']}")

    return 0 if summary["replay_cache_complete"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
