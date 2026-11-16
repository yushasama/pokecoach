from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from cg.api import OptionType, SelectContext
from cg.game import battle_finish, battle_select, battle_start
from deepmew.metrics import PolicyMetrics, update_metrics
from deepmew.policy import guard_selection, score_candidates
from scripts.self_play import _select_with_policy


@dataclass
class Divergence:
    game: int
    step: int
    player: int
    actual_policy: str
    context: str
    available: tuple[str, ...]
    v2_indices: tuple[int, ...]
    first_indices: tuple[int, ...]
    v2_types: tuple[str, ...]
    first_types: tuple[str, ...]
    top_scores: tuple[str, ...]


def _enum_name(enum_cls: Any, value: Any) -> str:
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _types_for(select: dict[str, Any], indices: list[int]) -> tuple[str, ...]:
    options = select.get("option") or []
    return tuple(
        _enum_name(OptionType, options[index].get("type"))
        for index in indices
        if 0 <= index < len(options)
    )


def _first_legal(obs: dict[str, Any]) -> list[int]:
    select = obs["select"]
    option_count = len(select.get("option") or [])
    max_count = min(int(select.get("maxCount") or 0), option_count)
    return guard_selection(list(range(max_count)), select)


def _pattern(divergence: Divergence) -> str:
    v2 = "+".join(divergence.v2_types) or "NONE"
    first = "+".join(divergence.first_types) or "NONE"
    return f"{divergence.context}: v2={v2} first={first}"


def _available_summary(select: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_enum_name(OptionType, option.get("type")) for option in select.get("option") or [])


def _top_score_summary(obs: dict[str, Any], limit: int = 5) -> tuple[str, ...]:
    candidates = sorted(score_candidates(obs), key=lambda candidate: (-candidate.score, candidate.index))
    return tuple(
        f"{candidate.index}:{candidate.option_type}:{candidate.score}:{'/'.join(candidate.reasons)}"
        for candidate in candidates[:limit]
    )


def _run_game(game: int, policy0: str, policy1: str, max_steps: int) -> tuple[int, int, list[Divergence], list[PolicyMetrics]]:
    deck = main.read_deck_csv()
    obs, start_data = battle_start(deck, deck)
    if obs is None:
        raise RuntimeError(
            f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
            f"errorType={start_data.errorType}"
        )

    divergences: list[Divergence] = []
    metrics = [PolicyMetrics(), PolicyMetrics()]
    try:
        for step in range(max_steps):
            current = obs.get("current")
            if current is not None and current.get("result", -1) != -1:
                return current["result"], int(current.get("turn") or 0), divergences, metrics

            player = int((current or {}).get("yourIndex") or 0)
            actual_policy = policy0 if player == 0 else policy1
            actual = _select_with_policy(actual_policy, obs)
            update_metrics(metrics[player], obs, actual)

            select = obs.get("select")
            if select is not None:
                v2 = _select_with_policy("v2", obs)
                first = _first_legal(obs)
                if v2 != first:
                    divergences.append(
                        Divergence(
                            game=game,
                            step=step,
                            player=player,
                            actual_policy=actual_policy,
                            context=_enum_name(SelectContext, select.get("context")),
                            available=_available_summary(select),
                            v2_indices=tuple(v2),
                            first_indices=tuple(first),
                            v2_types=_types_for(select, v2),
                            first_types=_types_for(select, first),
                            top_scores=_top_score_summary(obs),
                        )
                    )

            obs = battle_select(actual)

        raise TimeoutError(f"game exceeded {max_steps} selections")
    finally:
        battle_finish()


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    results = {0: 0, 1: 0, 2: 0}
    crashes = 0
    timeouts = 0
    turns = 0
    all_divergences: list[Divergence] = []
    aggregate_metrics = defaultdict(PolicyMetrics)

    for game in range(1, args.games + 1):
        policy0, policy1 = ("v2", "first") if game % 2 else ("first", "v2")
        try:
            result, game_turns, divergences, metrics = _run_game(game, policy0, policy1, args.max_steps)
        except TimeoutError:
            timeouts += 1
            continue
        except Exception as exc:
            crashes += 1
            print(f"game {game}: crash={type(exc).__name__}: {exc}")
            continue

        results[result] = results.get(result, 0) + 1
        turns += game_turns
        all_divergences.extend(divergences)
        aggregate_metrics[policy0].add(metrics[0])
        aggregate_metrics[policy1].add(metrics[1])
        print(
            f"game {game}: p0={policy0} p1={policy1} result={result} "
            f"turns={game_turns} divergences={len(divergences)}"
        )

    completed = sum(results.values())
    avg_turns = turns / completed if completed else 0.0
    print("")
    print("summary")
    print(f"completed={completed}/{args.games} results={results} crashes={crashes} timeouts={timeouts} avg_turns={avg_turns:.2f}")
    for policy, metrics in sorted(aggregate_metrics.items()):
        print(
            f"{policy}: decisions={metrics.decisions} "
            f"end_with_useful={metrics.end_with_useful_options} "
            f"attack_skipped={metrics.attack_available_but_not_picked} "
            f"unknown_action={metrics.unknown_action} "
            f"risky_over_safer={metrics.risky_over_safer}"
        )

    pattern_counts = Counter(_pattern(divergence) for divergence in all_divergences)
    print("")
    print("top divergence patterns")
    examples_by_pattern: dict[str, list[Divergence]] = defaultdict(list)
    for divergence in all_divergences:
        examples_by_pattern[_pattern(divergence)].append(divergence)

    for pattern, count in pattern_counts.most_common(5):
        print(f"- {pattern}: {count}")
        for example in examples_by_pattern[pattern][: args.examples]:
            print(
                f"  game={example.game} step={example.step} player={example.player} "
                f"actual={example.actual_policy} v2={list(example.v2_indices)} "
                f"first={list(example.first_indices)} available={dict(Counter(example.available))}"
            )
            print(f"  top_scores={list(example.top_scores)}")


if __name__ == "__main__":
    main_cli()
