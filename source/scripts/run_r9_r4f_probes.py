from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r8_greedyv_probes as r8_probes
import run_r9_r4f_retrain as r4f
import run_unified_r4_closeout_delta as closeout
from deepmew.rl.decks import read_config_deck
from deepmew.rl.opponents import OpponentAgent
from deepmew.rl.registry import resolve_experiment


SCHEMA_VERSION = "deepmew.r9.r4f_probes.v0"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane3_probes"
DEFAULT_R4F = ROOT / ".research" / "r9" / "lane2_r4f_conservative" / "selected_ar_bc_r4f_candidate.pt"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_experiment("RL4_arena_opponent_mix_canary").validate()
    deck = read_config_deck(config, role="agent")
    max_steps = config.max_steps_per_game
    opponents = [OpponentAgent.from_str(name) for name in ("random", "reference_lucario", "reference_dragapult")]
    seeds = [131313, 242424][: args.seed_count]
    policies = [
        closeout.R4ObsPolicy("R4-A", Path(args.r4a), args.deck_policy),
        closeout.R4ObsPolicy("v1b", Path(args.v1b), args.deck_policy),
        r4f.R4FLivePolicy("R4-F", Path(args.r4f), args.deck_policy),
    ]
    rows: dict[str, Any] = {}
    for policy in policies:
        print(f"[{policy.label}] playing probe games...", flush=True)
        probe, safety = r8_probes.play_games(policy, opponents, deck, args.games, seeds, max_steps)
        rows[policy.label] = {"probes": probe.summary(), "safety": safety}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "config": {"games_per_opp": args.games, "opponents": [opponent.label for opponent in opponents], "seeds": seeds},
        "r7_ladder_baseline": {"P1_riptide_whiff_rate": 0.5455, "P2_mega_latency_mean": 6.93, "P3_hammer_low_deck_rate": 0.2571, "P4_surfing_activation_rate": 0.0283},
        "policies": rows,
    }
    (out_dir / "r4f_probes.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    return summary


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R9 R4-F P1-P4 probes.")
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--r4a", default=str(DEFAULT_R4A))
    parser.add_argument("--v1b", default=str(DEFAULT_V1B))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
