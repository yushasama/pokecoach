import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from cg.game import battle_finish, battle_select, battle_start
from deepmew.inspection import print_observation
from deepmew.metrics import PolicyMetrics, update_metrics
from deepmew.policy import choose_action_v2, guard_selection
from deepmew.state_machine import analyze_board, describe_plan


POLICIES = ("v2", "legacy-v2", "random", "first")


@dataclass
class GameResult:
    result: int
    turns: int
    p0_metrics: PolicyMetrics
    p1_metrics: PolicyMetrics


def _initial_observation() -> dict:
    return {"select": None, "logs": [], "current": None, "search_begin_input": None}


def _select_with_policy(policy: str, obs: dict) -> list[int]:
    select = obs.get("select")
    if select is None:
        return main.read_deck_csv()

    options = select.get("option") or []
    if policy == "v2":
        return main.agent(obs)
    if policy == "legacy-v2":
        return choose_action_v2(obs)
    if policy == "random":
        max_count = int(select.get("maxCount") or 0)
        max_count = min(max_count, len(options))
        return guard_selection(random.sample(list(range(len(options))), max_count), select)
    if policy == "first":
        max_count = int(select.get("maxCount") or 0)
        max_count = min(max_count, len(options))
        return guard_selection(list(range(max_count)), select)
    raise ValueError(f"unknown policy: {policy}")


def run_game(
    max_steps: int,
    verbose: bool = False,
    option_limit: int = 12,
    policy0: str = "v2",
    policy1: str = "v2",
) -> GameResult:
    deck0 = main.read_deck_csv()
    deck1 = main.read_deck_csv()
    metrics = [PolicyMetrics(), PolicyMetrics()]
    if verbose:
        print("initial deck request: obs.select=None, obs.current=None, config=None in local harness")
        print(f"deck0_len={len(deck0)} deck1_len={len(deck1)}")
        print(f"policies: player0={policy0} player1={policy1}")
    obs, start_data = battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(
            f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
            f"errorType={start_data.errorType}"
        )

    try:
        for step in range(max_steps):
            current = obs.get("current")
            if current is not None and current.get("result", -1) != -1:
                if verbose:
                    print_observation(obs, step, option_limit)
                return GameResult(current["result"], int(current.get("turn") or 0), metrics[0], metrics[1])
            if verbose:
                print_observation(obs, step, option_limit)
                if obs.get("select") is not None:
                    print(f"  board_plan: {describe_plan(analyze_board(obs))}")
            player_index = int((current or {}).get("yourIndex") or 0)
            policy = policy0 if player_index == 0 else policy1
            selected = _select_with_policy(policy, obs)
            analysis = update_metrics(metrics[player_index], obs, selected)
            if verbose:
                print(f"  policy={policy} selected_indices={selected} chosen_types={analysis['chosen_types']}")
                if policy == "v2":
                    legacy_selected = _select_with_policy("legacy-v2", obs)
                    legacy_analysis = update_metrics(PolicyMetrics(), obs, legacy_selected)
                    if legacy_selected != selected:
                        print(
                            "  differs_from_legacy_v2: "
                            f"legacy_indices={legacy_selected} "
                            f"legacy_types={legacy_analysis['chosen_types']}"
                        )
                print(
                    "  probes: "
                    f"end_with_useful={analysis['end_with_useful']} "
                    f"attack_skipped={analysis['attack_skipped']} "
                    f"unknown_action={analysis['unknown_action']} "
                    f"risky_over_safer={analysis['risky_over_safer']}"
                )
            obs = battle_select(selected)

        raise TimeoutError(f"game exceeded {max_steps} selections")
    finally:
        battle_finish()


def prove_imports() -> None:
    importlib.import_module("main")
    importlib.import_module("cg.api")
    importlib.import_module("cg.game")
    importlib.import_module("deepmew.policy")


def _print_metrics(label: str, metrics: PolicyMetrics) -> None:
    print(
        f"{label}: decisions={metrics.decisions} "
        f"end_with_useful={metrics.end_with_useful_options} "
        f"attack_skipped={metrics.attack_available_but_not_picked} "
        f"unknown_action={metrics.unknown_action} "
        f"risky_over_safer={metrics.risky_over_safer}"
    )


def run_series(
    games: int,
    max_steps: int,
    policy0: str,
    policy1: str,
    verbose: bool,
    option_limit: int,
    fail_on_error: bool = True,
) -> None:
    results = {0: 0, 1: 0, 2: 0}
    crashes = 0
    timeouts = 0
    turns = 0
    p0_metrics = PolicyMetrics()
    p1_metrics = PolicyMetrics()
    for game in range(1, games + 1):
        try:
            game_result = run_game(
                max_steps,
                verbose=verbose,
                option_limit=option_limit,
                policy0=policy0,
                policy1=policy1,
            )
        except TimeoutError:
            timeouts += 1
            if verbose:
                raise
            continue
        except Exception:
            crashes += 1
            if verbose:
                raise
            continue

        results[game_result.result] = results.get(game_result.result, 0) + 1
        turns += game_result.turns
        p0_metrics.add(game_result.p0_metrics)
        p1_metrics.add(game_result.p1_metrics)
        print(f"game {game}: result={game_result.result}")

    completed = sum(results.values())
    avg_turns = turns / completed if completed else 0.0
    print(f"completed {completed}/{games} game(s): {results}")
    print(f"crashes={crashes} timeouts={timeouts} avg_turns={avg_turns:.2f}")
    _print_metrics(f"player0 {policy0}", p0_metrics)
    _print_metrics(f"player1 {policy1}", p1_metrics)
    if fail_on_error and (crashes or timeouts):
        raise RuntimeError(f"self-play failed: crashes={crashes} timeouts={timeouts}")


def run_comparison(games: int, max_steps: int) -> None:
    for policy0, policy1 in (
        ("v2", "v2"),
        ("v2", "legacy-v2"),
        ("legacy-v2", "v2"),
        ("v2", "random"),
        ("random", "v2"),
        ("v2", "first"),
        ("first", "v2"),
    ):
        print(f"matchup: player0={policy0} player1={policy1}")
        run_series(games, max_steps, policy0, policy1, verbose=False, option_limit=0, fail_on_error=False)


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--option-limit", type=int, default=12)
    parser.add_argument("--policy", choices=POLICIES, default="v2")
    parser.add_argument("--opponent-policy", choices=POLICIES)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.verbose and args.games != 1:
        raise ValueError("--verbose is intended for one inspection game; use --games 1")

    prove_imports()
    if args.compare:
        run_comparison(args.games, args.max_steps)
        return

    opponent_policy = args.opponent_policy or args.policy
    run_series(
        args.games,
        args.max_steps,
        args.policy,
        opponent_policy,
        verbose=args.verbose,
        option_limit=args.option_limit,
        fail_on_error=True,
    )


if __name__ == "__main__":
    main_cli()
