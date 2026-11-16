"""R15: GreedyV-on-R4F - run the recovered GreedyV v3 bytecode with the R4-F prior.

The original `deepmew/greedyv` implementation survives as cpython-312 bytecode
(imported sourceless; original .pyc preserved under D:/DeepMew_artifacts/greedyv_pyc_backup).
Two adaptations, made WITHOUT touching the recovered bytecode:

1. `run_unified_r4_ablation.build_live_features` is patched process-locally to emit
   second-order-augmented (304-dim) features, because the R4-F prior expects them
   where the original R4-A prior used base features. Research runner only.
2. `GreedyVAgent.shortlist_policy` is replaced post-construction with an R4-F adapter
   exposing the same surface (.model/.select/.deck/.deck_hash/.policy_id).

The value model is the rebuilt `selected_value_source_v0` (run_unified_r3_value).
Probes compare GreedyV-on-R4F against the plain R4-F prior (= the depth-0/BUG-004
guard baseline) on P1-P4 plus safety and clock stats.

Example
-------
    uv run python scripts/run_r15_greedyv_on_r4f.py --games 40 --seed-count 1 \
        --out-dir .research/r15/lane5_greedyv_on_r4f
"""

from __future__ import annotations

import argparse
import json
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

import main  # noqa: E402
import run_unified_r4_ablation as r4  # noqa: E402
import run_r9_r4f_retrain as r4f  # noqa: E402
import run_r8_greedyv_probes as r8_probes  # noqa: E402
from deepmew.policy.selection import guard_selection  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl.opponents import OpponentAgent  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402
from deepmew.second_order.reader import augment_option_features  # noqa: E402


SCHEMA_VERSION = "deepmew.r15.greedyv_on_r4f.v0"
DEFAULT_R4F = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
DEFAULT_VALUE = ROOT / ".research" / "unified_r3" / "value" / "selected_value_source_v0.pt"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_BASE_BUILD_LIVE_FEATURES = r4.build_live_features


def _r4f_build_live_features(obs: dict[str, Any], legal_set: list[dict[str, Any]], *, policy_id: str, deck_hash: str) -> Any:
    encoded = _BASE_BUILD_LIVE_FEATURES(obs, legal_set, policy_id=policy_id, deck_hash=deck_hash)
    if encoded is None:
        return None
    features = augment_option_features(obs, legal_set[: encoded.legal_count], encoded.features)

    return r4.EncodedDecision(
        features=features,
        target_index=encoded.target_index,
        legal_count=encoded.legal_count,
        metadata=encoded.metadata,
    )


class R4FShortlistAdapter:
    """Duck-typed replacement for the R4-A shortlist policy inside GreedyVAgent."""

    def __init__(self, checkpoint: Path, deck_policy: str):
        self.model, self.payload = r4f.load_r4f_model(checkpoint)
        agent = OpponentAgent.from_str(deck_policy)
        self.deck = list(agent.deck or main.read_deck_csv())
        self.deck_hash = getattr(agent.deck_identity, "deck_hash", None) or normalized_deck_hash(self.deck)
        self.policy_id = deck_policy
        self.label = "R4-F_shortlist"

    def reset(self, seed: int) -> None:
        _ = seed

    def bind_timer(self, timer: Any | None) -> None:
        _ = timer

    def select(self, obs: dict[str, Any]) -> list[int]:
        select = _as_dict(obs.get("select"))
        if not select:
            return list(self.deck)
        table = legal_action_table(obs, select)
        encoded = r4.build_live_features(obs, table, policy_id=self.policy_id, deck_hash=self.deck_hash)
        if encoded is None:
            return guard_selection([0], select)
        min_count = _as_int(select.get("minCount"))
        max_count = _as_int(select.get("maxCount"), default=encoded.legal_count)
        chosen = r4f.select_r4f_from_features(self.model, encoded.features, encoded.legal_count, min_count, max_count)

        return guard_selection(chosen, select)


def build_greedyv_on_r4f(args: argparse.Namespace) -> Any:
    r4.build_live_features = _r4f_build_live_features
    from deepmew.greedyv.agent import GreedyVAgent, GreedyVConfig

    config = GreedyVConfig(
        worlds=int(args.worlds),
        shortlist_k=int(args.shortlist_k),
        per_decision_budget_ms=float(args.budget_ms),
        override_margin=float(args.override_margin),
        rng_seed=int(args.seed),
        value_checkpoint=Path(args.value_checkpoint),
        r4a_checkpoint=Path(args.construction_r4a_checkpoint),
        deck_policy=str(args.deck_policy),
    )
    adapter = R4FShortlistAdapter(Path(args.r4f_checkpoint), str(args.deck_policy))
    agent = GreedyVAgent(config)
    agent.shortlist_policy = adapter
    agent.deck = list(adapter.deck)
    agent.deck_hash = adapter.deck_hash
    agent.policy_id = adapter.policy_id
    agent.label = "GreedyV_R4F"

    return agent


def _play_winrate_game(policy: Any, opponent: Any, *, our_seat: int, max_steps: int) -> tuple[int | None, dict[str, int]]:
    from cg.game import battle_finish, battle_select, battle_start
    import run_unified_r4_closeout_delta as closeout

    safety = {"illegal": 0, "crashes": 0, "timeouts": 0}
    obs = None
    try:
        our_deck = list(policy.deck)
        opp_deck = list(getattr(opponent, "deck", None) or main.read_deck_csv())
        deck0 = our_deck if our_seat == 0 else opp_deck
        deck1 = opp_deck if our_seat == 0 else our_deck
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(f"BattleStart failed: {start_data.errorType}")
        for _ in range(max_steps):
            current = _as_dict(obs.get("current"))
            result = _as_int(current.get("result"), -1)
            if result != -1:
                return result, safety
            actor = policy if _as_int(current.get("yourIndex")) == our_seat else opponent
            selected = closeout._select_from_policy(actor, obs)
            legal_count = len((_as_dict(obs.get("select")).get("option") or []))
            if any(index < 0 or index >= legal_count for index in selected):
                safety["illegal"] += 1

                return None, safety
            obs = battle_select(selected)
        safety["timeouts"] += 1

        return None, safety
    except Exception:
        safety["crashes"] += 1

        return None, safety
    finally:
        if obs is not None:
            battle_finish()


def compare_winrates(policies: list[tuple[str, Any]], *, games_per_opponent: int, seed: int) -> dict[str, Any]:
    opponent_names = ("reference_abomasnow", "reference_iono", "reference_dragapult", "reference_lucario")
    out: dict[str, Any] = {}
    for label, policy in policies:
        wins = 0
        losses = 0
        draws = 0
        safety = {"illegal": 0, "crashes": 0, "timeouts": 0}
        for opponent_name in opponent_names:
            opponent = OpponentAgent.from_str(opponent_name)
            for game_index in range(games_per_opponent):
                policy.reset(seed + game_index)
                opponent.reset(seed + game_index)
                our_seat = game_index % 2
                result, game_safety = _play_winrate_game(policy, opponent, our_seat=our_seat, max_steps=5000)
                for key in safety:
                    safety[key] += game_safety[key]
                if result is None:
                    continue
                if result == our_seat:
                    wins += 1
                elif result in (0, 1):
                    losses += 1
                else:
                    draws += 1
        total = max(1, wins + losses + draws)
        out[label] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": round(wins / total, 6),
            "safety": safety,
            "opponents": list(opponent_names),
            "games_per_opponent": games_per_opponent,
        }
        print(f"[winrate] {label}: {wins}-{losses}-{draws} ({wins / total:.4f}) safety={safety}", flush=True)

    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    greedyv = build_greedyv_on_r4f(args)
    prior = greedyv.shortlist_policy
    deck = list(greedyv.deck)
    opponents = [OpponentAgent.from_str(name) for name in ("random", "reference_lucario", "reference_dragapult")]
    seeds = [131313, 242424][: int(args.seed_count)]
    max_steps = 5000
    results: dict[str, Any] = {}
    for label, policy in (("GreedyV_on_R4F", greedyv), ("R4F_prior_depth0", prior)):
        if int(args.games) > 0:
            print(f"[{label}] playing probe games...", flush=True)
            probe, safety = r8_probes.play_games(policy, opponents, deck, int(args.games), seeds, max_steps)
            results[label] = {"probes": probe.summary(), "safety": safety}
        else:
            results[label] = {}
        stats = getattr(policy, "stats", None)
        if stats is not None and hasattr(stats, "as_dict"):
            results[label]["greedyv_stats"] = stats.as_dict()
    winrates: dict[str, Any] = {}
    if int(args.winrate_games) > 0:
        winrates = compare_winrates(
            [("GreedyV_on_R4F", greedyv), ("R4F_prior_depth0", prior)],
            games_per_opponent=int(args.winrate_games),
            seed=int(args.seed) + 500,
        )
        stats = getattr(greedyv, "stats", None)
        if stats is not None and hasattr(stats, "as_dict"):
            winrates["greedyv_stats_cumulative"] = stats.as_dict()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "config": {
            "r4f_checkpoint": str(args.r4f_checkpoint),
            "value_checkpoint": str(args.value_checkpoint),
            "worlds": int(args.worlds),
            "shortlist_k": int(args.shortlist_k),
            "budget_ms": float(args.budget_ms),
            "override_margin": float(args.override_margin),
            "games_per_opp": int(args.games),
            "seeds": seeds,
            "prior_swap": "R4FShortlistAdapter + augmented build_live_features patch (runner-local)",
        },
        "depth0_guard_note": "R4F_prior_depth0 row is the same prior with no search (BUG-004 ablation baseline)",
        "policies": results,
        "winrates": winrates,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "greedyv_on_r4f_probes.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:4000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GreedyV with the R4-F prior and rebuilt value model; probe P1-P4 vs the depth-0 prior.")
    parser.add_argument("--r4f-checkpoint", default=str(DEFAULT_R4F))
    parser.add_argument(
        "--construction-r4a-checkpoint",
        default=str(ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"),
        help="Architecturally valid R4-A checkpoint used only to satisfy GreedyVAgent construction; the prior is swapped to R4-F immediately after.",
    )
    parser.add_argument("--value-checkpoint", default=str(DEFAULT_VALUE))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--shortlist-k", type=int, default=4)
    parser.add_argument("--budget-ms", type=float, default=250.0)
    parser.add_argument("--override-margin", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--winrate-games", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=915002)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
