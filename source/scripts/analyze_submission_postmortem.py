#!/usr/bin/env python
"""Pull, cache, rescore, and analyze every replay for a Kaggle submission."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.submission_postmortem.addendum_runner import run_statistical_addendum  # noqa: E402
from deepmew.submission_postmortem.models import CatalogQuery  # noqa: E402
from deepmew.submission_postmortem.pipeline import PostmortemOptions, run_postmortem  # noqa: E402


def _inclusive_range(value: str, *, label: str) -> tuple[int | None, int | None]:
    parts = value.split(":", 1)
    if len(parts) != 2 or (not parts[0].strip() and not parts[1].strip()):
        raise argparse.ArgumentTypeError(f"{label} must be START:END with at least one inclusive bound")
    try:
        start = int(parts[0]) if parts[0].strip() else None
        end = int(parts[1]) if parts[1].strip() else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} bounds must be integers") from exc
    if start is not None and end is not None and start > end:
        raise argparse.ArgumentTypeError(f"{label} start must not exceed end")

    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze all public/validation replay episodes for one Kaggle submission. "
            "Exact checkpoint ratings are emitted only when a hash-bound submitted bundle resolves."
        )
    )
    parser.add_argument("--submission-id", required=True, help="external Kaggle submission ID")
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--replay-dir", type=Path, help="existing replay directory; defaults to offline catalog mode")
    parser.add_argument("--online-catalog", action="store_true", help="fetch authoritative SDK metadata even with --replay-dir")
    parser.add_argument("--no-download", action="store_true", help="require every selected replay to exist locally")
    parser.add_argument("--allow-metadata-limited-cli-fallback", action="store_true")
    parser.add_argument("--kaggle-exe", help="explicit Kaggle executable")

    parser.add_argument("--episode-id", action="append", default=[], help="exact episode ID; repeat for OR")
    parser.add_argument("--episode-range", type=lambda value: _inclusive_range(value, label="episode range"), help="inclusive numeric ID range START:END")
    parser.add_argument("--ordinal-range", type=lambda value: _inclusive_range(value, label="ordinal range"), help="inclusive 1-based chronological range START:END")
    parser.add_argument("--opponent", action="append", default=[], help="case-insensitive opponent team substring; repeat for OR")
    parser.add_argument("--opponent-team-id", action="append", default=[])
    parser.add_argument("--opponent-submission-id", action="append", default=[])
    parser.add_argument("--exclude-validation", action="store_true", help="exclude validation episodes (included by default)")
    parser.add_argument("--result", choices=("all", "win", "loss", "draw"), default="all")

    parser.add_argument("--bundle-dir", type=Path, help="explicit exact submitted bundle")
    parser.add_argument(
        "--bindings",
        type=Path,
        default=Path("configs/submission_postmortem/submission_bindings.json"),
        help="external-submission to hash-bound local-bundle registry",
    )
    parser.add_argument("--skip-scoring", action="store_true", help="extract observational signals without actor logits")
    parser.add_argument("--encoder-batch-size", type=int, default=8192)

    parser.add_argument("--out-dir", type=Path, help="exact output directory (reusable with content caches)")
    parser.add_argument("--out-root", type=Path, default=Path(".research/submission_postmortems"))
    parser.add_argument("--cache-root", type=Path, default=Path(".research/submission_postmortem/cache"))
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--parse-workers", type=int, default=4)
    parser.add_argument("--refresh-downloads", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument(
        "--skip-statistical-addendum",
        action="store_true",
        help="skip additive phases A-K after the base run completes",
    )
    parser.add_argument("--statistical-permutations", type=int, default=40_000)

    return parser


def _query(args: argparse.Namespace) -> CatalogQuery:
    episode_min, episode_max = args.episode_range or (None, None)
    ordinal_min, ordinal_max = args.ordinal_range or (None, None)
    bad_ids = [value for value in args.episode_id if not str(value).isdigit()]
    if bad_ids:
        raise ValueError(f"episode IDs must contain digits: {bad_ids}")

    return CatalogQuery(
        episode_ids=frozenset(str(value) for value in args.episode_id),
        episode_min=episode_min,
        episode_max=episode_max,
        ordinal_min=ordinal_min,
        ordinal_max=ordinal_max,
        opponent_terms=tuple(args.opponent),
        opponent_team_ids=frozenset(str(value) for value in args.opponent_team_id),
        opponent_submission_ids=frozenset(str(value) for value in args.opponent_submission_id),
        include_validation=not args.exclude_validation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(shlex.quote(value) for value in ([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]))
    if args.statistical_permutations <= 0:
        parser.error("--statistical-permutations must be positive")
    try:
        options = PostmortemOptions(
            submission_id=str(args.submission_id),
            competition=str(args.competition),
            query=_query(args),
            replay_dir=args.replay_dir,
            online_catalog=bool(args.online_catalog),
            download_missing=not args.no_download,
            allow_cli_catalog_fallback=bool(args.allow_metadata_limited_cli_fallback),
            kaggle_executable=args.kaggle_exe,
            bundle_dir=args.bundle_dir,
            bindings_path=args.bindings,
            output_dir=args.out_dir,
            output_root=args.out_root,
            cache_root=args.cache_root,
            download_workers=args.download_workers,
            parse_workers=args.parse_workers,
            encoder_batch_size=args.encoder_batch_size,
            refresh_downloads=bool(args.refresh_downloads),
            refresh_scores=bool(args.refresh_scores),
            skip_scoring=bool(args.skip_scoring),
            result_filter=args.result,
            command=command,
        )
        result = run_postmortem(options)
        addendum_result = None
        decisive_games = int(result.summary.get("wins") or 0) + int(result.summary.get("losses") or 0)
        if not args.skip_statistical_addendum and decisive_games > 0:
            addendum_result = run_statistical_addendum(
                result.output_dir,
                permutation_samples=args.statistical_permutations,
                command=command,
            )
    except Exception as exc:  # noqa: BLE001 - CLI converts fail-closed gate errors into one receipt
        print(f"submission postmortem failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        return 2

    response = {
        "status": "complete",
        "output_dir": str(result.output_dir.resolve()),
        "manifest": str(result.manifest_path.resolve()),
        "report": str(result.report_path.resolve()),
        "summary": result.summary,
        "statistical_addendum": None,
    }
    if addendum_result is not None:
        response["statistical_addendum"] = {
            "output_dir": str(addendum_result.output_dir.resolve()),
            "manifest": str(addendum_result.manifest_path.resolve()),
            "addendum": str(addendum_result.addendum_path.resolve()),
        }
    print(
        json.dumps(
            response,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
