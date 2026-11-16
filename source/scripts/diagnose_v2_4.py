from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from cg.api import OptionType, SelectContext, SelectType, all_attack, all_card_data
from cg.game import battle_finish, battle_select, battle_start
from deepmew.policy import choose_action, choose_action_v2, guard_selection, score_candidates
from deepmew.state_machine import analyze_board, attachment_effect


ATTACK_DAMAGE = {attack.attackId: attack.damage for attack in all_attack()}
CARD_DATA = {card.cardId: card for card in all_card_data()}


@dataclass
class TraceSummary:
    games: int = 0
    completed: int = 0
    crashes: int = 0
    timeouts: int = 0
    v23_wins: int = 0
    baseline_wins: int = 0
    draws: int = 0
    v23_decisions: int = 0
    divergences: int = 0
    categories: Counter[str] = None
    patterns: Counter[str] = None
    examples: dict[str, list[str]] = None
    ko_visible: int = 0
    ko_missed: int = 0
    attack_available: int = 0
    attack_skipped: int = 0

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = Counter()
        if self.patterns is None:
            self.patterns = Counter()
        if self.examples is None:
            self.examples = defaultdict(list)


def _enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "None"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _select_with_policy(policy: str, obs: dict[str, Any]) -> list[int]:
    select = obs.get("select")
    if select is None:
        return main.read_deck_csv()
    options = select.get("option") or []
    if policy == "v2":
        return choose_action(obs)
    if policy == "legacy-v2":
        return choose_action_v2(obs)
    if policy == "first":
        max_count = min(int(select.get("maxCount") or 0), len(options))
        return guard_selection(list(range(max_count)), select)
    raise ValueError(f"unknown policy: {policy}")


def _selected_types(obs: dict[str, Any], selected: list[int]) -> tuple[str, ...]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    return tuple(_enum_name(OptionType, options[index].get("type")) for index in selected if 0 <= index < len(options))


def _best_attack_damage(obs: dict[str, Any]) -> int | None:
    select = obs.get("select") or {}
    damages = [
        ATTACK_DAMAGE.get(int(option.get("attackId")))
        for option in select.get("option") or []
        if option.get("type") == OptionType.ATTACK.value and option.get("attackId") is not None
    ]
    known = [damage for damage in damages if damage is not None]
    return max(known) if known else None


def _opponent_active(obs: dict[str, Any]) -> dict[str, Any] | None:
    current = obs.get("current") or {}
    your_index = int(current.get("yourIndex") or 0)
    players = current.get("players") or []
    opponent_index = 1 - your_index
    if opponent_index >= len(players):
        return None
    active = (players[opponent_index] or {}).get("active") or []
    return active[0] if active and active[0] is not None else None


def _own_prizes_remaining(obs: dict[str, Any]) -> int | None:
    current = obs.get("current") or {}
    your_index = int(current.get("yourIndex") or 0)
    players = current.get("players") or []
    if your_index >= len(players):
        return None
    return len((players[your_index] or {}).get("prize") or [])


def _visible_ko_damage(obs: dict[str, Any]) -> bool:
    active = _opponent_active(obs)
    damage = _best_attack_damage(obs)
    if active is None or damage is None:
        return False
    return damage >= int(active.get("hp") or 0)


def _opponent_active_prize_value(obs: dict[str, Any]) -> int | None:
    active = _opponent_active(obs)
    if active is None:
        return None
    card = CARD_DATA.get(int(active.get("id") or 0))
    if card is None:
        return None
    if getattr(card, "megaEx", False):
        return 3
    if getattr(card, "ex", False):
        return 2
    return 1


def _attachment_improves(obs: dict[str, Any], selected: list[int]) -> bool:
    plan = analyze_board(obs)
    options = (obs.get("select") or {}).get("option") or []
    for index in selected:
        if 0 <= index < len(options):
            effect = attachment_effect(plan, options[index])
            if effect.improves_readiness:
                return True
    return False


def _classify(obs: dict[str, Any], v23: list[int], baseline: list[int]) -> tuple[str, ...]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    v23_types = set(_selected_types(obs, v23))
    baseline_types = set(_selected_types(obs, baseline))
    available_types = {_enum_name(OptionType, option.get("type")) for option in options}
    plan = analyze_board(obs)
    best_damage = _best_attack_damage(obs)
    visible_ko = _visible_ko_damage(obs)
    categories: list[str] = []

    if visible_ko and "ATTACK" not in v23_types:
        categories.append("missed_ko")
        categories.append("delayed_prize")

    if "ATTACK" in v23_types and "ATTACK" not in baseline_types:
        if not visible_ko and (best_damage is None or best_damage < 60):
            categories.append("attacked_too_early")
        else:
            categories.append("sequencing_mistake")

    if "ATTACK" in available_types and "ATTACK" not in v23_types:
        if visible_ko or (best_damage is not None and best_damage >= 100):
            categories.append("developed_too_long")
        else:
            categories.append("sequencing_mistake")

    if "ATTACH" in v23_types:
        if not _attachment_improves(obs, v23) and plan.state.name == "ATTACK_READY":
            categories.append("wasted_attachment")
        elif "ATTACK" in available_types and "ATTACK" not in v23_types:
            categories.append("resource_sequencing")

    if "EVOLVE" in v23_types and "ATTACK" in available_types:
        categories.append("wasted_evolution")

    if not categories:
        categories.append("unknown")

    return tuple(dict.fromkeys(categories))


def _format_decision(obs: dict[str, Any], v23: list[int], baseline: list[int], baseline_name: str) -> str:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    plan = analyze_board(obs)
    return (
        f"turn={current.get('turn')} player={current.get('yourIndex')} "
        f"context={_enum_name(SelectContext, select.get('context'))} "
        f"state={plan.state.value} best_damage={_best_attack_damage(obs)} "
        f"visible_ko={_visible_ko_damage(obs)} prize_value={_opponent_active_prize_value(obs)} "
        f"prizes_left={_own_prizes_remaining(obs)} "
        f"v23={v23}/{_selected_types(obs, v23)} "
        f"{baseline_name}={baseline}/{_selected_types(obs, baseline)}"
    )


def _record(summary: TraceSummary, obs: dict[str, Any], baseline_name: str) -> None:
    select = obs.get("select")
    if select is None:
        return
    v23 = _select_with_policy("v2", obs)
    baseline = _select_with_policy(baseline_name, obs)
    summary.v23_decisions += 1

    available_types = {_enum_name(OptionType, option.get("type")) for option in select.get("option") or []}
    if "ATTACK" in available_types:
        summary.attack_available += 1
        if "ATTACK" not in _selected_types(obs, v23):
            summary.attack_skipped += 1
    if _visible_ko_damage(obs):
        summary.ko_visible += 1
        if "ATTACK" not in _selected_types(obs, v23):
            summary.ko_missed += 1

    if v23 == baseline:
        return

    summary.divergences += 1
    v23_types = "+".join(_selected_types(obs, v23)) or "NONE"
    base_types = "+".join(_selected_types(obs, baseline)) or "NONE"
    context = _enum_name(SelectContext, select.get("context"))
    pattern = f"{context}: v23={v23_types} {baseline_name}={base_types}"
    summary.patterns[pattern] += 1

    for category in _classify(obs, v23, baseline):
        summary.categories[category] += 1
        if len(summary.examples[category]) < 5:
            summary.examples[category].append(_format_decision(obs, v23, baseline, baseline_name))


def run_matchup(games: int, max_steps: int, baseline_name: str) -> TraceSummary:
    summary = TraceSummary(games=games)
    for game in range(games):
        v23_player = game % 2
        policy0 = "v2" if v23_player == 0 else baseline_name
        policy1 = baseline_name if v23_player == 0 else "v2"
        deck0 = main.read_deck_csv()
        deck1 = main.read_deck_csv()
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            summary.crashes += 1
            continue

        try:
            for _ in range(max_steps):
                current = obs.get("current") or {}
                if current.get("result", -1) != -1:
                    result = int(current.get("result"))
                    summary.completed += 1
                    if result == 2:
                        summary.draws += 1
                    elif result == v23_player:
                        summary.v23_wins += 1
                    else:
                        summary.baseline_wins += 1
                    break

                if int(current.get("yourIndex") or 0) == v23_player:
                    _record(summary, obs, baseline_name)

                policy = policy0 if int(current.get("yourIndex") or 0) == 0 else policy1
                obs = battle_select(_select_with_policy(policy, obs))
            else:
                summary.timeouts += 1
        except Exception:
            summary.crashes += 1
        finally:
            battle_finish()
    return summary


def print_summary(name: str, summary: TraceSummary) -> None:
    print(f"## V2.3b vs {name}")
    print(f"completed={summary.completed}/{summary.games} crashes={summary.crashes} timeouts={summary.timeouts}")
    print(f"v23_wins={summary.v23_wins} {name}_wins={summary.baseline_wins} draws={summary.draws}")
    print(f"v23_decisions={summary.v23_decisions} divergences={summary.divergences}")
    print(f"attack_available={summary.attack_available} attack_skipped={summary.attack_skipped}")
    print(f"visible_ko={summary.ko_visible} missed_ko={summary.ko_missed}")
    print("categories:")
    for category, count in summary.categories.most_common():
        print(f"  {category}: {count}")
    print("patterns:")
    for pattern, count in summary.patterns.most_common(10):
        print(f"  {pattern}: {count}")
    print("examples:")
    for category, examples in summary.examples.items():
        print(f"  {category}:")
        for example in examples:
            print(f"    - {example}")
    print()


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    importlib.import_module("deepmew.policy")
    for baseline in ("legacy-v2", "first"):
        print_summary(baseline, run_matchup(args.games, args.max_steps, baseline))


if __name__ == "__main__":
    main_cli()
