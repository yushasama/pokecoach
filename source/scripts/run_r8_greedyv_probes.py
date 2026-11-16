from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cg.game import battle_finish, battle_select, battle_start

import main
import run_unified_r4_closeout_delta as closeout
from deepmew.rl1_teacher_bc.semantics import legal_action_table


RIPTIDE_ATTACK_ID = 1042
HAMMER_LANCHE_ATTACK_ID = 1046
MEGA_ABOMASNOW_ID = 723
SURFING_BEACH_ID = 1262
BASIC_WATER_ID = 3


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("id", value.get("cardId", value.get("cardID")))
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _own_player(obs: dict[str, Any]) -> dict[str, Any]:
    current = _as_dict(obs.get("current"))
    players = _as_list(current.get("players"))
    your_index = _as_int(current.get("yourIndex"))
    if your_index < 0 or your_index >= len(players):
        return {}

    return _as_dict(players[your_index])


def _own_cards_in_play(obs: dict[str, Any]) -> list[int]:
    player = _own_player(obs)
    cards = []
    for pokemon in _as_list(player.get("active")) + _as_list(player.get("bench")):
        card_id = _card_id(_as_dict(pokemon))
        if card_id is not None:
            cards.append(card_id)

    return cards


def _own_discard_water_count(obs: dict[str, Any]) -> int:
    player = _own_player(obs)

    return sum(1 for card in _as_list(player.get("discard")) if _card_id(_as_dict(card)) == BASIC_WATER_ID)


def _own_deck_count(obs: dict[str, Any]) -> int:
    player = _own_player(obs)
    if player.get("deckCount") is not None:
        return _as_int(player.get("deckCount"))

    return len(_as_list(player.get("deck")))


def _stadium_card_id(obs: dict[str, Any]) -> int | None:
    current = _as_dict(obs.get("current"))
    stadium = current.get("stadium")
    if isinstance(stadium, dict):
        return _card_id(stadium)
    for card in _as_list(stadium):
        card_id = _card_id(card)
        if card_id is not None:
            return card_id

    return None


def _is_surfing_beach_activation(obs: dict[str, Any], option: dict[str, Any]) -> bool:
    return (
        _stadium_card_id(obs) == SURFING_BEACH_ID
        and str(option.get("option_type") or "") == "ABILITY"
        and str(option.get("source_area") or "") == "STADIUM"
    )


@dataclass
class ProbeSummary:
    riptide_attacks: int = 0
    riptide_whiffs: int = 0
    hammer_attacks: int = 0
    hammer_low_deck: int = 0
    surfing_activations: int = 0
    mega_first_turns: list[int] = field(default_factory=list)
    _mega_seen_this_game: bool = False

    def start_game(self) -> None:
        self._mega_seen_this_game = False

    def observe_policy_action(self, obs: dict[str, Any], selected: list[int]) -> None:
        turn = _as_int(_as_dict(obs.get("current")).get("turn"))
        if not self._mega_seen_this_game and MEGA_ABOMASNOW_ID in _own_cards_in_play(obs):
            self.mega_first_turns.append(turn)
            self._mega_seen_this_game = True
        table = legal_action_table(obs, _as_dict(obs.get("select")))
        for index in selected:
            if index < 0 or index >= len(table):
                continue
            option = table[index]
            attack_id = option.get("attack_id")
            source_card_id = option.get("source_card_id")
            if attack_id == RIPTIDE_ATTACK_ID:
                self.riptide_attacks += 1
                self.riptide_whiffs += int(_own_discard_water_count(obs) <= 0)
            if attack_id == HAMMER_LANCHE_ATTACK_ID:
                self.hammer_attacks += 1
                self.hammer_low_deck += int(_own_deck_count(obs) <= 6)
            if source_card_id == SURFING_BEACH_ID or _is_surfing_beach_activation(obs, option):
                self.surfing_activations += 1

    def summary(self) -> dict[str, Any]:
        return {
            "P1_riptide_whiff_rate": round(self.riptide_whiffs / max(1, self.riptide_attacks), 6),
            "P2_mega_latency_mean": round(sum(self.mega_first_turns) / max(1, len(self.mega_first_turns)), 6),
            "P2_mega_latency_max": max(self.mega_first_turns) if self.mega_first_turns else None,
            "P3_hammer_low_deck_rate": round(self.hammer_low_deck / max(1, self.hammer_attacks), 6),
            "P4_surfing_activation_rate": round(self.surfing_activations / max(1, self.riptide_attacks + self.hammer_attacks + self.surfing_activations), 6),
            "raw_counts": {
                "riptide_attacks": self.riptide_attacks,
                "riptide_whiffs": self.riptide_whiffs,
                "hammer_attacks": self.hammer_attacks,
                "hammer_low_deck": self.hammer_low_deck,
                "surfing_activations": self.surfing_activations,
                "mega_seen_games": len(self.mega_first_turns),
            },
        }


def _play_probe_episode(policy: Any, opponent: Any, *, our_deck: list[int], our_seat: int, max_steps: int, probe: ProbeSummary) -> dict[str, int]:
    obs = None
    safety = {"illegal": 0, "crashes": 0, "timeouts": 0}
    try:
        probe.start_game()
        opp_deck = list(getattr(opponent, "deck", None) or main.read_deck_csv())
        deck0 = our_deck if our_seat == 0 else opp_deck
        deck1 = opp_deck if our_seat == 0 else our_deck
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(f"BattleStart failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}")
        for _ in range(max_steps):
            current = _as_dict(obs.get("current"))
            if current.get("result", -1) != -1:
                return safety
            actor_is_policy = _as_int(current.get("yourIndex")) == our_seat
            actor = policy if actor_is_policy else opponent
            selected = closeout._select_from_policy(actor, obs)
            if actor_is_policy:
                probe.observe_policy_action(obs, selected)
            select = _as_dict(obs.get("select"))
            legal_count = len(_as_list(select.get("option")))
            if any(index < 0 or index >= legal_count for index in selected):
                safety["illegal"] += 1
                return safety
            obs = battle_select(selected)
        safety["timeouts"] += 1

        return safety
    except Exception:
        safety["crashes"] += 1

        return safety
    finally:
        if obs is not None:
            battle_finish()


def play_games(policy: Any, opponents: list[Any], deck: list[int], games: int, seeds: list[int], max_steps: int) -> tuple[ProbeSummary, dict[str, int]]:
    probe = ProbeSummary()
    safety = {"illegal": 0, "crashes": 0, "timeouts": 0}
    for opponent in opponents:
        for seed in seeds:
            policy.reset(seed)
            opponent.reset(seed)
            for game_index in range(games):
                row = _play_probe_episode(policy, opponent, our_deck=deck, our_seat=game_index % 2, max_steps=max_steps, probe=probe)
                for key in safety:
                    safety[key] += row[key]

    return probe, safety
