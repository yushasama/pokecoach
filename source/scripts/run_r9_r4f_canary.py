from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r9_r4f_retrain as r4f
import run_unified_r4_closeout_delta as closeout
from deepmew.rl.decks import read_config_deck
from deepmew.rl.opponents import OpponentAgent
from deepmew.rl.perf.phase_timer import PhaseTimer
from deepmew.rl.registry import resolve_experiment


SCHEMA_VERSION = "deepmew.r9.r4f_canary.v0"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane3_canary"
DEFAULT_R4F = ROOT / ".research" / "r9" / "lane2_r4f_equal_epoch" / "selected_ar_bc_r4f_candidate.pt"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _wilson_ci(wins: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom

    return max(0.0, center - half), min(1.0, center + half)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# R9 R4-F Canary vs Random",
        "",
        f"- schema: `{payload['schema_version']}`",
        f"- games_per_policy: `{payload['config']['games']}`",
        f"- seed: `{payload['config']['seed']}`",
        "",
        "| policy | winrate | wins | losses | draws | illegal | crashes | timeouts | Wilson 95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        ci = row["wilson95"]
        lines.append(
            f"| `{row['policy']}` | {row['winrate']} | {row['wins']} | {row['losses']} | {row['draws']} | "
            f"{row['illegal']} | {row['crashes']} | {row['timeouts']} | [{ci[0]}, {ci[1]}] |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            f"- R4-F minus R4-A: `{payload['deltas']['r4f_minus_r4a']}`",
            f"- R4-F minus v1b: `{payload['deltas']['r4f_minus_v1b']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _policies(args: argparse.Namespace) -> list[Any]:
    return [
        closeout.R4ObsPolicy("R4-A_incumbent", Path(args.r4a), args.deck_policy),
        closeout.R4ObsPolicy("v1b_incumbent", Path(args.v1b), args.deck_policy),
        r4f.R4FLivePolicy("R4-F_candidate", Path(args.r4f), args.deck_policy),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_experiment("RL4_arena_opponent_mix_canary").validate()
    deck = read_config_deck(config, role="agent")
    opponent = OpponentAgent.from_str("random")
    max_steps = config.max_steps_per_game
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for policy in _policies(args):
        rng = random.Random(args.seed)
        timer = PhaseTimer(
            run_id=f"r9_canary_{policy.label}",
            out_dir=out_dir / "perf" / policy.label,
            config={"schema_version": SCHEMA_VERSION},
        )
        row = {
            "policy": policy.label,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "illegal": 0,
            "crashes": 0,
            "timeouts": 0,
        }
        started = time.perf_counter()
        for game_index in range(args.games):
            our_seat = game_index % 2
            episode = closeout._play_obs_episode(
                policy,
                opponent=opponent,
                our_seat=our_seat,
                our_deck=deck,
                max_steps=max_steps,
                opponent_seed=rng.randint(0, 2**31 - 1),
                policy_seed=args.seed * 1_000_003 + game_index,
                phase_timer=timer,
            )
            row["games"] += 1
            row["illegal"] += int(episode.illegal)
            if episode.crashed:
                row["crashes"] += 1
            elif episode.timed_out:
                row["timeouts"] += 1
            elif episode.is_win(our_seat):
                row["wins"] += 1
            elif episode.is_loss(our_seat):
                row["losses"] += 1
            elif episode.is_draw():
                row["draws"] += 1
        completed = row["wins"] + row["losses"] + row["draws"]
        row["winrate"] = round(row["wins"] / completed, 6) if completed else None
        row["wilson95"] = [round(value, 6) for value in _wilson_ci(row["wins"], completed)]
        row["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"[{policy.label}] winrate={row['winrate']} safety={row['illegal']}/{row['crashes']}/{row['timeouts']}", flush=True)
    by_policy = {row["policy"]: row for row in rows}
    r4f_wr = by_policy["R4-F_candidate"]["winrate"] or 0.0
    r4a_wr = by_policy["R4-A_incumbent"]["winrate"] or 0.0
    v1b_wr = by_policy["v1b_incumbent"]["winrate"] or 0.0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "config": {
            "games": args.games,
            "seed": args.seed,
            "opponent": "random",
            "both_seats": True,
            "r4f": str(args.r4f),
        },
        "rows": rows,
        "deltas": {
            "r4f_minus_r4a": round(r4f_wr - r4a_wr, 6),
            "r4f_minus_v1b": round(r4f_wr - v1b_wr, 6),
        },
        "total_wall_seconds": round(time.perf_counter() - started_all, 1),
    }
    _write_json(out_dir / "canary_summary.json", payload)
    _write_md(out_dir / "canary_summary.md", payload)
    print(json.dumps({"deltas": payload["deltas"], "rows": rows}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R9 R4-F canary vs random for R4-A/v1b/R4-F.")
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--r4a", default=str(DEFAULT_R4A))
    parser.add_argument("--v1b", default=str(DEFAULT_V1B))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=927000)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
