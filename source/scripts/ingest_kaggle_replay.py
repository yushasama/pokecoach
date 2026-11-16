"""Ingest a downloaded Kaggle episode JSON into DeepMew artifacts.

Extracts public opponent decklists from the initial visualize payload and (unless
`--extract-decks-only`) maps replay steps into `deepmew.replay_action_row.v1` rows.
Offline only; never touches root `main.py`/`deck.csv`/`cg/` or the eval path.

Examples
--------
    python scripts/ingest_kaggle_replay.py --episode-json <path-to-81454940.json>
    python scripts/ingest_kaggle_replay.py --input-dir .research/kaggle_replays/manual
    python scripts/ingest_kaggle_replay.py --episode-json <path> --extract-decks-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.decks import DeckExtractionError  # noqa: E402
from deepmew.kaggle_replays.episode import KaggleEpisodeError  # noqa: E402
from deepmew.kaggle_replays.ingest import ingest_episode  # noqa: E402


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def discover_episode_replays(input_dir: str | Path) -> list[Path]:
    replay_dir = Path(input_dir)
    if not replay_dir.is_dir():
        raise KaggleEpisodeError(f"input directory not found: {replay_dir}")
    return sorted(path for path in replay_dir.glob("episode-*-replay.json") if path.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a Kaggle episode JSON into DeepMew artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode-json", help="Path to a downloaded Kaggle episode JSON.")
    source.add_argument("--input-dir", help="Directory containing episode-*-replay.json files.")
    parser.add_argument("--extract-decks-only", action="store_true", help="Only extract opponent decklists.")
    parser.add_argument("--out-root", default=".research", help="Run-directory root (kaggle_ingest/<date>/<run_id>).")
    parser.add_argument("--opponent-decks-root", default=".research/opponent_decks", help="Opponent deck output root.")
    return parser.parse_args(argv)


def _print_result(result) -> None:
    s = result.summary
    print(f"episode_id={s['episode_id']} teams={s['team_names']} rewards={s['rewards']} result={s['result']}")
    for deck in s["decks"]:
        print(
            f"  player {deck['player_index']} ({deck['team_name']}): cards={deck['card_count']} "
            f"standard60={deck['is_standard_size']} hash={deck['deck_hash']} warnings={deck['warnings'] or 'none'}"
        )
    if s["action_rows"] is None:
        print("action_rows=skipped (--extract-decks-only)")
    else:
        a = s["action_rows"]
        print(f"action_rows={a['rows']} (select_decisions={a['select_decisions']}, malformed={a['malformed']})")
    print(f"run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.report_path}")
    if result.action_table_path is not None:
        print(f"replay_action_table={result.action_table_path}")
    for deck in result.deck_results:
        print(f"deck_dir={deck['out_dir']}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    effective_argv = argv if argv is not None else sys.argv[1:]
    command = "python scripts/ingest_kaggle_replay.py " + " ".join(effective_argv)

    try:
        episode_paths = [Path(args.episode_json)] if args.episode_json else discover_episode_replays(args.input_dir)
    except KaggleEpisodeError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    if not episode_paths:
        print(f"ingest failed: no episode-*-replay.json files found in {args.input_dir}", file=sys.stderr)
        return 1

    failures = 0
    results = []
    for episode_path in episode_paths:
        try:
            result = ingest_episode(
                episode_path,
                extract_decks_only=args.extract_decks_only,
                out_root=args.out_root,
                opponent_decks_root=args.opponent_decks_root,
                command=command,
            )
        except (KaggleEpisodeError, DeckExtractionError) as exc:
            print(f"ingest failed for {episode_path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        results.append(result)
        _print_result(result)

    if args.input_dir:
        deck_count = sum(len(result.deck_results) for result in results)
        print(f"ingested_replay_files={len(results)} failed={failures} deck_artifacts={deck_count}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
