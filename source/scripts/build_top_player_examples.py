"""Rebuild top-player BC examples from downloaded ladder replays (R14 Lane 2).

Produces the JSONL format consumed by ``run_r9_r4f_retrain.encode_top_player_row_r9``:
one row per top-ranked player's decision with the raw observation, the derived
legal action table, the selected indices, the action mask, and the final result.

The original ``deepmew/rl1_top_player_bc`` generator source was lost (pycache
only); this is a fresh implementation against the documented consumer contract.

Example
-------
    uv run python scripts/build_top_player_examples.py \
        --replay-dirs .research/kaggle_replays/20260706-top20-r4f-arena \
        --scout-manifest ".research/kaggle_scout/20260706/20260706T063704Z-kaggle_scout-20260706-top20-r4f-arena/manifest.json" \
        --max-rank 20 --exclude-team DeepMew \
        --out .research/r14/lane2_top_player_bc/replay_label_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_r14_ladder_replays import DEFAULT_CARD_CSV, load_card_names, shell_label  # noqa: E402
from deepmew.kaggle_replays.decks import extract_decklists  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402


SCHEMA_VERSION = "deepmew.r14.top_player_examples.v1"
RANK_BUCKETS = ((10, "top10"), (25, "top25"), (50, "top50"), (100, "top100"), (300, "top300"))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return ""
    for limit, bucket in RANK_BUCKETS:
        if rank <= limit:
            return bucket

    return ""


def team_ranks(manifest_paths: list[Path]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for team in _as_list(manifest.get("selected_teams")):
            record = _as_dict(team)
            name = str(record.get("team_name") or "")
            rank = record.get("rank")
            if name and isinstance(rank, int):
                ranks[name] = min(rank, ranks.get(name, rank))

    return ranks


def emit_episode_rows(
    data: dict[str, Any],
    *,
    episode_id: str,
    ranks: dict[str, int],
    max_rank: int,
    exclude_teams: set[str],
    cards: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    info = _as_dict(data.get("info"))
    team_names = [str(name) for name in _as_list(info.get("TeamNames"))]
    rewards = _as_list(data.get("rewards"))
    if len(team_names) != 2 or len(rewards) != 2:
        return []
    decks = extract_decklists(data)
    rows: list[dict[str, Any]] = []
    for seat, team in enumerate(team_names):
        rank = ranks.get(team)
        if team in exclude_teams or rank is None or rank > max_rank:
            continue
        reward = rewards[seat]
        won = isinstance(reward, (int, float)) and reward > 0
        deck_hash = normalized_deck_hash(sorted(decks[seat].card_ids)) if len(decks) == 2 else ""
        shell = shell_label(list(decks[seat].card_ids), cards) if len(decks) == 2 else "Unknown"
        for step_index, step in enumerate(_as_list(data.get("steps"))):
            agents = _as_list(step)
            if len(agents) != 2:
                continue
            agent = _as_dict(agents[seat])
            obs = _as_dict(agent.get("observation"))
            select = _as_dict(obs.get("select"))
            options = _as_list(select.get("option"))
            action = agent.get("action")
            if not (obs and options and isinstance(action, list) and action):
                continue
            selected = [index for index in action if isinstance(index, int) and 0 <= index < len(options)]
            if not selected:
                continue
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "step": step_index,
                    "seat": seat,
                    "team_name": team,
                    "rank": rank,
                    "rank_bucket": rank_bucket(rank),
                    "deck_hash": deck_hash,
                    "shell": shell,
                    "reward": 1 if won else 0,
                    "observation": obs,
                    "legal_action_table": legal_action_table(obs, select),
                    "selected_action_indices": selected,
                    "action_mask": {
                        "min_count": select.get("minCount"),
                        "max_count": select.get("maxCount"),
                    },
                }
            )

    return rows


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build top-player BC example rows from ladder replays.")
    parser.add_argument("--replay-dirs", nargs="+", required=True)
    parser.add_argument("--scout-manifest", nargs="+", required=True)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--exclude-team", nargs="*", default=["DeepMew"])
    parser.add_argument("--card-csv", default=str(DEFAULT_CARD_CSV))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ranks = team_ranks([Path(path) for path in args.scout_manifest])
    cards = load_card_names(Path(args.card_csv))
    exclude_teams = set(args.exclude_team)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    episode_count = 0
    seen_episodes: set[str] = set()
    by_team: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    by_shell: Counter[str] = Counter()
    win_rows = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for replay_dir in args.replay_dirs:
            for path in sorted(Path(replay_dir).glob("episode-*-replay.json")):
                episode_id = path.stem.replace("episode-", "").replace("-replay", "")
                if episode_id in seen_episodes:
                    continue
                seen_episodes.add(episode_id)
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = emit_episode_rows(
                    data,
                    episode_id=episode_id,
                    ranks=ranks,
                    max_rank=args.max_rank,
                    exclude_teams=exclude_teams,
                    cards=cards,
                )
                if rows:
                    episode_count += 1
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
                    row_count += 1
                    by_team[row["team_name"]] += 1
                    by_bucket[row["rank_bucket"]] += 1
                    by_shell[row["shell"]] += 1
                    win_rows += row["reward"]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "replay_dirs": list(args.replay_dirs),
            "scout_manifest": args.scout_manifest,
            "max_rank": args.max_rank,
            "exclude_teams": sorted(exclude_teams),
        },
        "rows": row_count,
        "episodes_with_rows": episode_count,
        "win_rows": win_rows,
        "rows_by_rank_bucket": dict(by_bucket),
        "rows_by_team": dict(by_team.most_common()),
        "rows_by_shell": dict(by_shell.most_common()),
        "out": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "episodes_with_rows", "win_rows", "rows_by_rank_bucket", "rows_by_shell")}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
