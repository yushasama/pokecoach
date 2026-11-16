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
from cg.api import OptionType, SelectContext, all_attack
from cg.game import battle_finish, battle_select, battle_start
from deepmew.policy import choose_action, guard_selection, score_candidates
from deepmew.state_machine import analyze_board, attachment_effect


ATTACK_DAMAGE = {attack.attackId: attack.damage for attack in all_attack()}
SETUP_TYPES = {"PLAY", "EVOLVE", "ATTACH"}


@dataclass
class Summary:
    games: int = 0
    completed: int = 0
    crashes: int = 0
    timeouts: int = 0
    v24_wins: int = 0
    first_wins: int = 0
    draws: int = 0
    decisions: int = 0
    divergences: int = 0
    categories: Counter[str] = None
    patterns: Counter[str] = None
    examples: dict[str, list[str]] = None
    visible_ko: int = 0
    missed_visible_ko: int = 0

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


def _option_types(obs: dict[str, Any], selected: list[int]) -> tuple[str, ...]:
    options = (obs.get("select") or {}).get("option") or []
    return tuple(_enum_name(OptionType, options[index].get("type")) for index in selected if 0 <= index < len(options))


def _first_legal(obs: dict[str, Any]) -> list[int]:
    select = obs.get("select")
    if select is None:
        return main.read_deck_csv()
    options = select.get("option") or []
    max_count = min(int(select.get("maxCount") or 0), len(options))
    return guard_selection(list(range(max_count)), select)


def _select(policy: str, obs: dict[str, Any]) -> list[int]:
    if obs.get("select") is None:
        return main.read_deck_csv()
    if policy == "v2":
        return choose_action(obs)
    if policy == "first":
        return _first_legal(obs)
    raise ValueError(policy)


def _best_attack_damage(obs: dict[str, Any], selected: list[int] | None = None) -> int | None:
    options = (obs.get("select") or {}).get("option") or []
    indices = selected if selected is not None else range(len(options))
    damages: list[int] = []
    for index in indices:
        if 0 <= index < len(options) and options[index].get("type") == OptionType.ATTACK.value:
            damage = ATTACK_DAMAGE.get(int(options[index].get("attackId") or -1))
            if damage is not None:
                damages.append(damage)
    return max(damages) if damages else None


def _opponent_active_hp(obs: dict[str, Any]) -> int | None:
    current = obs.get("current") or {}
    your_index = int(current.get("yourIndex") or 0)
    players = current.get("players") or []
    opponent_index = 1 - your_index
    if opponent_index >= len(players):
        return None
    active = (players[opponent_index] or {}).get("active") or []
    if not active or active[0] is None:
        return None
    hp = active[0].get("hp")
    return int(hp) if hp is not None else None


def _visible_ko(obs: dict[str, Any], selected: list[int] | None = None) -> bool:
    hp = _opponent_active_hp(obs)
    damage = _best_attack_damage(obs, selected)
    return hp is not None and damage is not None and damage >= hp


def _attachment_target_key(obs: dict[str, Any], selected: list[int]) -> tuple[Any, ...]:
    options = (obs.get("select") or {}).get("option") or []
    keys = []
    for index in selected:
        if 0 <= index < len(options):
            option = options[index]
            if option.get("type") == OptionType.ATTACH.value:
                keys.append((option.get("inPlayArea"), option.get("inPlayIndex"), option.get("area"), option.get("index")))
    return tuple(keys)


def _attachment_improves(obs: dict[str, Any], selected: list[int]) -> bool:
    plan = analyze_board(obs)
    options = (obs.get("select") or {}).get("option") or []
    for index in selected:
        if 0 <= index < len(options) and attachment_effect(plan, options[index]).improves_readiness:
            return True
    return False


def _candidate_reasons(obs: dict[str, Any], selected: list[int]) -> tuple[str, ...]:
    by_index = {candidate.index: candidate for candidate in score_candidates(obs)}
    reasons: list[str] = []
    for index in selected:
        candidate = by_index.get(index)
        if candidate is not None:
            reasons.extend(candidate.reasons)
    return tuple(reasons)


def _classify(obs: dict[str, Any], v24: list[int], first: list[int]) -> tuple[str, ...]:
    select = obs.get("select") or {}
    context = _enum_name(SelectContext, select.get("context"))
    options = select.get("option") or []
    available = {_enum_name(OptionType, option.get("type")) for option in options}
    v24_types = set(_option_types(obs, v24))
    first_types = set(_option_types(obs, first))
    v24_damage = _best_attack_damage(obs, v24)
    best_damage = _best_attack_damage(obs)
    hp = _opponent_active_hp(obs)
    categories: list[str] = []

    if context == "IS_FIRST" and "NO" in v24_types and "YES" in first_types:
        categories.append("bad_first_player_choice")

    if _visible_ko(obs) and "ATTACK" not in v24_types:
        categories.append("missed_visible_ko")

    if "ATTACK" in v24_types:
        if not _visible_ko(obs, v24) and (v24_damage is None or v24_damage < 60):
            categories.append("weak_attack_chosen")
        if first_types & SETUP_TYPES and not _visible_ko(obs, v24) and (v24_damage is None or v24_damage < 80):
            categories.append("attack_too_early")
            categories.append("setup_before_attack")

    if "ATTACK" in available and "ATTACK" not in v24_types:
        if (best_damage is not None and best_damage >= 100) or _visible_ko(obs):
            categories.append("attack_too_late")

    if "ATTACH" in first_types and "ATTACH" not in v24_types:
        categories.append("attach_target_choice" if "ATTACH" in v24_types else "attach_timing")
    elif "ATTACH" in first_types and "ATTACH" in v24_types:
        if _attachment_target_key(obs, v24) != _attachment_target_key(obs, first):
            categories.append("attach_target_choice")

    if "EVOLVE" in first_types and "EVOLVE" not in v24_types:
        categories.append("evolve_timing")
    elif "EVOLVE" in v24_types and "EVOLVE" not in first_types:
        categories.append("evolve_timing")

    if "PLAY" in first_types and "PLAY" not in v24_types:
        categories.append("play_supporter_timing")
    elif "PLAY" in v24_types and "PLAY" not in first_types:
        categories.append("play_supporter_timing")

    if "ATTACH" in v24_types and not _attachment_improves(obs, v24) and analyze_board(obs).state.name == "ATTACK_READY":
        categories.append("attach_target_choice")

    if not categories:
        categories.append("unknown")
    return tuple(dict.fromkeys(categories))


def _format_example(obs: dict[str, Any], v24: list[int], first: list[int]) -> str:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    plan = analyze_board(obs)
    return (
        f"turn={current.get('turn')} player={current.get('yourIndex')} "
        f"context={_enum_name(SelectContext, select.get('context'))} state={plan.state.value} "
        f"hp={_opponent_active_hp(obs)} best_damage={_best_attack_damage(obs)} visible_ko={_visible_ko(obs)} "
        f"v24={v24}/{_option_types(obs, v24)} reasons={_candidate_reasons(obs, v24)} "
        f"first={first}/{_option_types(obs, first)}"
    )


def record_decision(summary: Summary, obs: dict[str, Any]) -> None:
    if obs.get("select") is None:
        return
    v24 = _select("v2", obs)
    first = _select("first", obs)
    summary.decisions += 1
    if _visible_ko(obs):
        summary.visible_ko += 1
        if "ATTACK" not in _option_types(obs, v24):
            summary.missed_visible_ko += 1
    if v24 == first:
        return

    summary.divergences += 1
    context = _enum_name(SelectContext, (obs.get("select") or {}).get("context"))
    v24_types = "+".join(_option_types(obs, v24)) or "NONE"
    first_types = "+".join(_option_types(obs, first)) or "NONE"
    summary.patterns[f"{context}: v24={v24_types} first={first_types}"] += 1

    for category in _classify(obs, v24, first):
        summary.categories[category] += 1
        if len(summary.examples[category]) < 5:
            summary.examples[category].append(_format_example(obs, v24, first))


def run(games: int, max_steps: int) -> Summary:
    summary = Summary(games=games)
    for game in range(games):
        v24_player = game % 2
        policy0 = "v2" if v24_player == 0 else "first"
        policy1 = "first" if v24_player == 0 else "v2"
        obs, start_data = battle_start(main.read_deck_csv(), main.read_deck_csv())
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
                    elif result == v24_player:
                        summary.v24_wins += 1
                    else:
                        summary.first_wins += 1
                    break
                if int(current.get("yourIndex") or 0) == v24_player:
                    record_decision(summary, obs)
                policy = policy0 if int(current.get("yourIndex") or 0) == 0 else policy1
                obs = battle_select(_select(policy, obs))
            else:
                summary.timeouts += 1
        except Exception:
            summary.crashes += 1
        finally:
            battle_finish()
    return summary


def print_summary(summary: Summary) -> None:
    print(f"completed={summary.completed}/{summary.games} crashes={summary.crashes} timeouts={summary.timeouts}")
    print(f"v24_wins={summary.v24_wins} first_wins={summary.first_wins} draws={summary.draws}")
    print(f"decisions={summary.decisions} divergences={summary.divergences}")
    print(f"visible_ko={summary.visible_ko} missed_visible_ko={summary.missed_visible_ko}")
    print("categories:")
    for category, count in summary.categories.most_common():
        print(f"  {category}: {count}")
    print("patterns:")
    for pattern, count in summary.patterns.most_common(12):
        print(f"  {pattern}: {count}")
    print("examples:")
    for category, examples in summary.examples.items():
        print(f"  {category}:")
        for example in examples:
            print(f"    - {example}")


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()
    print_summary(run(args.games, args.max_steps))


if __name__ == "__main__":
    main_cli()
