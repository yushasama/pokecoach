from __future__ import annotations

import argparse
import json
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
from deepmew.rl.opponents import make_eval_suite
from deepmew.rl.perf.phase_timer import PhaseTimer
from deepmew.rl.registry import resolve_experiment


SCHEMA_VERSION = "deepmew.r9.r4f_panel.v0"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane3_panel"
DEFAULT_R4F = ROOT / ".research" / "r9" / "lane2_r4f_conservative" / "selected_ar_bc_r4f_candidate.pt"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"
FRESH_SEEDS = [70207, 80317, 90439]
SLICE_NAMES = ["overall", "trivial", "v2_4_legacy", "references", "iono"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join("" if row.get(col) is None else str(row.get(col)).replace(",", ";") for col in columns) + "\n")


def _mean_se(values: list[float]) -> tuple[float, float]:
    import math
    import statistics

    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0

    return mean, se


def build_policies(args: argparse.Namespace) -> list[Any]:
    return [
        closeout.R4ObsPolicy("R4-A_incumbent", Path(args.r4a), args.deck_policy),
        closeout.R4ObsPolicy("v1b_incumbent", Path(args.v1b), args.deck_policy),
        r4f.R4FLivePolicy("R4-F_candidate", Path(args.r4f), args.deck_policy),
    ]


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = make_eval_suite(json.loads(closeout.PANEL_PATH.read_text(encoding="utf-8"))["eval_suite"])
    config = resolve_experiment("RL4_arena_opponent_mix_canary").validate()
    deck = read_config_deck(config, role="agent")
    max_steps = config.max_steps_per_game
    policies = build_policies(args)
    seeds = FRESH_SEEDS[: args.seed_count]
    aggregate_rows: list[dict[str, Any]] = []
    by_opponent_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for policy in policies:
        timer = PhaseTimer(run_id=f"r9_{policy.label}", out_dir=out_dir / "perf" / policy.label, config={"schema_version": SCHEMA_VERSION})
        pooled = {name: [0, 0] for name in SLICE_NAMES}
        per_seed_overall: list[float] = []
        safety = {"illegal": 0, "crashes": 0, "timeouts": 0, "games": 0}
        started = time.perf_counter()
        for seed in seeds:
            rng = random.Random(seed)
            per: dict[str, dict[str, int]] = {}
            for opponent in suite:
                bucket = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
                for game_index in range(args.games):
                    our_seat = game_index % 2
                    episode = closeout._play_obs_episode(
                        policy,
                        opponent=opponent,
                        our_seat=our_seat,
                        our_deck=deck,
                        max_steps=max_steps,
                        opponent_seed=rng.randint(0, 2**31 - 1),
                        policy_seed=seed * 1_000_003 + game_index,
                        phase_timer=timer,
                    )
                    bucket["games"] += 1
                    safety["games"] += 1
                    safety["illegal"] += int(episode.illegal)
                    if episode.crashed:
                        safety["crashes"] += 1
                    elif episode.timed_out:
                        safety["timeouts"] += 1
                    elif episode.is_win(our_seat):
                        bucket["wins"] += 1
                    elif episode.is_loss(our_seat):
                        bucket["losses"] += 1
                    elif episode.is_draw():
                        bucket["draws"] += 1
                per[opponent.label] = bucket
                by_opponent_rows.append({"policy": policy.label, "seed": seed, "opponent": opponent.label, **bucket, "winrate": bucket["wins"] / (bucket["wins"] + bucket["losses"]) if (bucket["wins"] + bucket["losses"]) else None})
            for name in SLICE_NAMES:
                wins, completed, _wr = closeout._slice_winrate(per, closeout._slice_keys(per, name))
                pooled[name][0] += wins
                pooled[name][1] += completed
            _wins, _completed, overall_wr = closeout._slice_winrate(per, closeout._slice_keys(per, "overall"))
            per_seed_overall.append(overall_wr or 0.0)
            print(f"[{policy.label}] seed {seed} overall={overall_wr}", flush=True)
        mean, se = _mean_se(per_seed_overall)
        row = {
            "policy": policy.label,
            "overall_pooled": round(pooled["overall"][0] / pooled["overall"][1], 6) if pooled["overall"][1] else None,
            "overall_seed_mean": round(mean, 6),
            "overall_seed_se": round(se, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            **{f"{name}_pooled": (round(pooled[name][0] / pooled[name][1], 6) if pooled[name][1] else None) for name in SLICE_NAMES},
            **safety,
        }
        aggregate_rows.append(row)
        print(f"[{policy.label}] overall_pooled={row['overall_pooled']} safety={safety}", flush=True)
    by_policy = {row["policy"]: row for row in aggregate_rows}
    r4f_overall = by_policy.get("R4-F_candidate", {}).get("overall_pooled")
    r4a_overall = by_policy.get("R4-A_incumbent", {}).get("overall_pooled")
    v1b_overall = by_policy.get("v1b_incumbent", {}).get("overall_pooled")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "config": {"games_per_opp_per_seed": args.games, "seeds": seeds, "both_seats": True, "r4f": str(args.r4f)},
        "aggregate_rows": aggregate_rows,
        "deltas": {
            "r4f_minus_r4a": round((r4f_overall or 0) - (r4a_overall or 0), 6),
            "r4f_minus_v1b": round((r4f_overall or 0) - (v1b_overall or 0), 6),
        },
        "total_wall_seconds": round(time.perf_counter() - started_all, 1),
    }
    (out_dir / "panel_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(out_dir / "aggregate.csv", aggregate_rows)
    _write_csv(out_dir / "by_opponent.csv", by_opponent_rows)
    print(json.dumps({"deltas": summary["deltas"], "aggregate_rows": aggregate_rows}, indent=2, sort_keys=True))

    return summary


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R9 R4-F panel vs R4-A and v1b.")
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--r4a", default=str(DEFAULT_R4A))
    parser.add_argument("--v1b", default=str(DEFAULT_V1B))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    run_panel(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
