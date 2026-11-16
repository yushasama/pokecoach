"""Safe observational signals for submission replay postmortems.

This module deliberately emits summaries rather than replay observations.  It
never serializes an opponent observation or a full legal-option table, and it
does not describe policy preferences as blunders.  Policy rescoring remains a
separate, explicitly labelled diagnostic surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from deepmew.submission_postmortem.models import ReplayDecision, ReplayEpisode, finite_float


DECISION_SIGNAL_SCHEMA_VERSION = "deepmew.submission_postmortem.decision_signal.v0"
EPISODE_SIGNAL_SCHEMA_VERSION = "deepmew.submission_postmortem.episode_signal.v0"
AGGREGATE_SCHEMA_VERSION = "deepmew.submission_postmortem.aggregate.v0"
DEFAULT_STARTING_PRIZES = 6
ROLE_ACTOR = "actor"
ROLE_OPPONENT = "opponent"

PROVENANCE_ACTOR_VISIBLE = "actor_visible"
PROVENANCE_PUBLIC_EVENT = "public_event"
PROVENANCE_POSTHOC_DECK = "posthoc_deck_metadata"
PROVENANCE_POLICY_RESCORE = "policy_rescore"
PROVENANCE_HEURISTIC = "heuristic"
PROVENANCE_UNAVAILABLE = "unavailable"

_PUBLIC_OPPONENT_CARD_CATEGORIES = frozenset({"ABILITY", "ATTACH", "ATTACK", "EVOLVE", "PLAY"})
_RATING_FIELDS = (
    "status",
    "binding_status",
    "value_status",
    "value",
    "critic_value",
    "entropy",
    "chosen_rank",
    "chosen_probability",
    "chosen_log_probability",
    "margin_to_best",
    "sequence_log_probability",
    "predicted_matches_action",
    "action_agreement",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value

    return ()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()

    return text or None


def _card_id(value: Any) -> int | None:
    card = _mapping(value)
    for key in ("id", "card_id", "cardId", "cardID"):
        result = _integer(card.get(key))
        if result is not None:
            return result

    return None


def _cards(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _energy_count(card: Mapping[str, Any]) -> int:
    energies = _sequence(card.get("energyCards"))
    if energies:
        return len(energies)
    energies = _sequence(card.get("energies"))
    if energies:
        return len(energies)
    count = _integer(card.get("energyCount"))

    return max(0, count or 0)


def _pokemon_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    hp = _integer(card.get("hp"))
    max_hp = _integer(card.get("maxHp"))
    damage = None
    if hp is not None and max_hp is not None:
        damage = max(0, max_hp - hp)

    return {
        "card_id": _card_id(card),
        "hp": hp,
        "max_hp": max_hp,
        "damage": damage,
        "energy_count": _energy_count(card),
    }


def _prizes_remaining(player: Mapping[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, Sequence) and not isinstance(prize, (str, bytes, bytearray)):
        return len(prize)
    for key in ("prizeCount", "prizesRemaining"):
        value = _integer(player.get(key))
        if value is not None:
            return max(0, value)

    return None


def _player_snapshot(player: Mapping[str, Any]) -> dict[str, Any]:
    active = _cards(player.get("active"))
    bench = _cards(player.get("bench"))
    active_rows = [_pokemon_snapshot(card) for card in active]
    bench_rows = [_pokemon_snapshot(card) for card in bench]
    energy_count = sum(int(row["energy_count"] or 0) for row in active_rows + bench_rows)
    viable_board_count = sum(
        1
        for row in active_rows + bench_rows
        if row.get("hp") is None or int(row["hp"]) > 0
    )
    discard_count = len(_sequence(player.get("discard")))
    status = {
        key: bool(player.get(key))
        for key in ("asleep", "burned", "confused", "paralyzed", "poisoned")
        if player.get(key) is not None
    }

    return {
        "deck_count": _integer(player.get("deckCount")),
        "hand_count": _integer(player.get("handCount")),
        "prizes_remaining": _prizes_remaining(player),
        "active_count": len(active_rows),
        "bench_count": len(bench_rows),
        "board_count": len(active_rows) + len(bench_rows),
        "viable_board_count": viable_board_count,
        "energy_count": energy_count,
        "discard_count": discard_count,
        "active": active_rows,
        "bench": bench_rows,
        "status": status,
    }


def _stadium_ids(current: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    for value in _sequence(current.get("stadium")):
        card_id = _card_id(value)
        if card_id is not None:
            result.append(card_id)

    return result


def extract_state_snapshot(decision: ReplayDecision) -> dict[str, Any]:
    """Extract an observation-free actor-visible state summary.

    Only whitelisted public/count fields are returned.  Own hand identities,
    search inputs, logs, and raw observations are intentionally absent.
    """

    current = _mapping(decision.observation.get("current"))
    players = [_mapping(value) for value in _sequence(current.get("players"))]
    your_index = _integer(current.get("yourIndex"))
    if your_index not in (0, 1):
        your_index = decision.seat if decision.seat in (0, 1) else None
    if your_index not in (0, 1) or len(players) <= your_index or len(players) < 2:
        return {
            "available": False,
            "decision_id": decision.decision_id,
            "provenance": PROVENANCE_UNAVAILABLE,
        }

    opponent_index = 1 - your_index
    own = _player_snapshot(players[your_index])
    opponent = _player_snapshot(players[opponent_index])
    own_prizes = _integer(own.get("prizes_remaining"))
    opponent_prizes = _integer(opponent.get("prizes_remaining"))
    prize_lead = None
    if own_prizes is not None and opponent_prizes is not None:
        prize_lead = opponent_prizes - own_prizes
    first_player = _integer(current.get("firstPlayer"))
    first_second = None
    if first_player in (0, 1):
        first_second = "first" if first_player == your_index else "second"

    return {
        "available": True,
        "decision_id": decision.decision_id,
        "episode_id": decision.episode_id,
        "request_step": decision.request_step,
        "seat": decision.seat,
        "turn": _integer(current.get("turn")),
        "turn_action_count": _integer(current.get("turnActionCount")),
        "your_index": your_index,
        "first_player": first_player,
        "first_second_proxy": first_second,
        "prize_lead": prize_lead,
        "board_delta": int(own["board_count"]) - int(opponent["board_count"]),
        "energy_delta": int(own["energy_count"]) - int(opponent["energy_count"]),
        "supporter_played": bool(current.get("supporterPlayed")),
        "stadium_played": bool(current.get("stadiumPlayed")),
        "energy_attached": bool(current.get("energyAttached")),
        "retreated": bool(current.get("retreated")),
        "stadium_card_ids": _stadium_ids(current),
        "own": own,
        "opponent": opponent,
        "provenance": PROVENANCE_ACTOR_VISIBLE,
    }


def _option_category(option: Mapping[str, Any]) -> str:
    value = option.get("option_type", option.get("type", option.get("category")))
    if isinstance(value, str):
        return value.strip().upper() or "UNKNOWN"

    return f"TYPE_{value}" if value is not None else "UNKNOWN"


def _safe_option(option: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    category = _option_category(option)
    result: dict[str, Any] = {"category": category}
    source_card_id = _integer(option.get("source_card_id", option.get("sourceCardId")))
    target_card_id = _integer(option.get("target_card_id", option.get("targetCardId")))
    attack_id = _integer(option.get("attack_id", option.get("attackId")))
    source_name = _text(option.get("source_card_name", option.get("card_name")))
    target_name = _text(option.get("target_card_name"))

    if role == ROLE_ACTOR or category in _PUBLIC_OPPONENT_CARD_CATEGORIES:
        if source_card_id is not None:
            result["source_card_id"] = source_card_id
        if source_name is not None:
            result["source_card_name"] = source_name
    if role == ROLE_ACTOR or category in {"ATTACH", "EVOLVE"}:
        if target_card_id is not None:
            result["target_card_id"] = target_card_id
        if target_name is not None:
            result["target_card_name"] = target_name
    if category == "ATTACK" and attack_id is not None:
        result["attack_id"] = attack_id

    return result


def _safe_rating(rating_summary: Mapping[str, Any] | None, *, role: str) -> dict[str, Any]:
    if role != ROLE_ACTOR or rating_summary is None:
        return {"status": "not_instrumented", "provenance": PROVENANCE_UNAVAILABLE}
    safe: dict[str, Any] = {}
    for key in _RATING_FIELDS:
        value = rating_summary.get(key)
        if isinstance(value, (str, bool)) or value is None:
            safe[key] = value
            continue
        number = finite_float(value)
        if number is not None:
            safe[key] = number
    safe.setdefault("status", "available")
    safe["provenance"] = PROVENANCE_POLICY_RESCORE

    return safe


def build_decision_signal(
    decision: ReplayDecision,
    *,
    legal_table: Sequence[Mapping[str, Any]],
    selected_options: Sequence[Mapping[str, Any]],
    rating_summary: Mapping[str, Any] | None = None,
    role: str = ROLE_ACTOR,
) -> dict[str, Any]:
    """Build one safe decision row without persisting alternatives.

    ``legal_table`` is used only for count validation.  No unselected option is
    copied into the result.  Opponent rows contain realized public semantics
    only and never contain their state snapshot or policy ratings.
    """

    if role not in {ROLE_ACTOR, ROLE_OPPONENT}:
        raise ValueError(f"unsupported decision role: {role!r}")
    safe_options = [_safe_option(_mapping(option), role=role) for option in selected_options]
    categories = [str(option["category"]) for option in safe_options]
    observed_legal_count = len(legal_table)
    count_status = "match" if observed_legal_count == decision.legal_count else "mismatch"
    snapshot = extract_state_snapshot(decision) if role == ROLE_ACTOR else None
    current = _mapping(decision.observation.get("current"))
    turn = _integer(current.get("turn"))
    turn_action_count = _integer(current.get("turnActionCount"))
    first_player = _integer(current.get("firstPlayer"))
    first_second = None
    if first_player in (0, 1) and decision.seat in (0, 1):
        first_second = "first" if first_player == decision.seat else "second"

    return {
        "schema_version": DECISION_SIGNAL_SCHEMA_VERSION,
        "decision_id": decision.decision_id,
        "episode_id": decision.episode_id,
        "request_step": decision.request_step,
        "action_step": decision.action_step,
        "seat": decision.seat,
        "role": role,
        "turn": turn,
        "turn_action_count": turn_action_count,
        "first_player": first_player,
        "first_second_proxy": first_second,
        "select_type": decision.select_type,
        "select_context": decision.select_context,
        "min_count": decision.min_count,
        "max_count": decision.max_count,
        "legal_count": decision.legal_count,
        "observed_legal_count": observed_legal_count,
        "legal_count_status": count_status,
        "selected_count": len(safe_options),
        "selected_categories": categories,
        "action_category_counts": dict(sorted(Counter(categories).items())),
        "selected_options": safe_options,
        "state_snapshot": snapshot,
        "rating": _safe_rating(rating_summary, role=role),
        "provenance": {
            "identity": PROVENANCE_ACTOR_VISIBLE if role == ROLE_ACTOR else PROVENANCE_PUBLIC_EVENT,
            "turn_context": PROVENANCE_ACTOR_VISIBLE if role == ROLE_ACTOR else PROVENANCE_PUBLIC_EVENT,
            "selected_actions": PROVENANCE_ACTOR_VISIBLE if role == ROLE_ACTOR else PROVENANCE_PUBLIC_EVENT,
            "state_snapshot": PROVENANCE_ACTOR_VISIBLE if role == ROLE_ACTOR else PROVENANCE_UNAVAILABLE,
            "rating": PROVENANCE_POLICY_RESCORE if role == ROLE_ACTOR and rating_summary is not None else PROVENANCE_UNAVAILABLE,
        },
    }


def _deck_for_seat(episode: ReplayEpisode, seat: int) -> dict[str, Any]:
    for deck in episode.decks:
        if deck.seat == seat:
            return {
                "stable_hash": deck.stable_hash,
                "team_name": deck.team_name,
                "provenance": PROVENANCE_POSTHOC_DECK,
            }

    return {
        "stable_hash": None,
        "team_name": episode.team_names[seat] if seat < len(episode.team_names) else None,
        "provenance": PROVENANCE_UNAVAILABLE,
    }


def _episode_result(episode: ReplayEpisode, seat: int) -> tuple[str, float | None]:
    reward = finite_float(episode.rewards[seat]) if seat < len(episode.rewards) else None
    if reward is None:
        return "unavailable", None
    if reward > 0:
        return "win", reward
    if reward < 0:
        return "loss", reward

    return "draw", reward


def _selected_options(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for row in rows:
        for option in _sequence(row.get("selected_options")):
            if isinstance(option, Mapping):
                yield option


def _action_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for category in _sequence(row.get("selected_categories")):
            counts[str(category)] += 1

    return counts


def _snapshot_values(rows: Iterable[Mapping[str, Any]], *path: str) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        value: Any = row.get("state_snapshot")
        for key in path:
            value = _mapping(value).get(key)
        if value is not None:
            values.append(value)

    return values


def _first_action_turn(rows: Iterable[Mapping[str, Any]], category: str) -> int | None:
    turns = [
        _integer(row.get("turn"))
        for row in rows
        if category in {str(value) for value in _sequence(row.get("selected_categories"))}
    ]
    available = [turn for turn in turns if turn is not None]

    return min(available) if available else None


def _matches_focal_evolution(
    options: Iterable[Mapping[str, Any]],
    *,
    focal_card_ids: frozenset[int],
    focal_terms: tuple[str, ...],
) -> bool:
    normalized_terms = tuple(term.casefold() for term in focal_terms if term)
    for option in options:
        if _option_category(option) != "EVOLVE":
            continue
        ids = {
            value
            for value in (
                _integer(option.get("source_card_id")),
                _integer(option.get("target_card_id")),
            )
            if value is not None
        }
        if ids & focal_card_ids:
            return True
        names = " ".join(
            str(value)
            for value in (option.get("source_card_name"), option.get("target_card_name"))
            if value
        ).casefold()
        if any(term in names for term in normalized_terms):
            return True

    return False


def _tag(value: bool | None, provenance: str, detail: str) -> dict[str, Any]:
    return {"value": value, "provenance": provenance, "detail": detail}


def _terminal_tag(
    terminal_snapshot: Mapping[str, Any] | None,
    terminal_reason: str | None,
    *,
    kind: str,
) -> dict[str, Any]:
    reason = (terminal_reason or "").casefold()
    own = _mapping(_mapping(terminal_snapshot).get("own"))
    if kind == "boardout":
        explicit = any(term in reason for term in ("boardout", "board out", "no pokemon", "no pokémon"))
        count = _integer(own.get("viable_board_count"))
        if count is None:
            count = _integer(own.get("board_count"))
        if explicit or count == 0:
            provenance = PROVENANCE_PUBLIC_EVENT if explicit else PROVENANCE_ACTOR_VISIBLE
            return _tag(True, provenance, "terminal reason or terminal board count proves no remaining board")
        if terminal_snapshot is not None and count is not None:
            return _tag(False, PROVENANCE_ACTOR_VISIBLE, "terminal board count remained positive")
    if kind == "deckout":
        explicit = any(term in reason for term in ("deckout", "deck out", "deck exhaustion", "no cards in deck"))
        count = _integer(own.get("deck_count"))
        if explicit or count == 0:
            provenance = PROVENANCE_PUBLIC_EVENT if explicit else PROVENANCE_ACTOR_VISIBLE
            return _tag(True, provenance, "terminal reason or terminal deck count proves deck exhaustion")
        if terminal_snapshot is not None and count is not None:
            return _tag(False, PROVENANCE_ACTOR_VISIBLE, "terminal deck count remained positive")

    return _tag(None, PROVENANCE_UNAVAILABLE, "terminal evidence was not available")


def aggregate_public_actions(decision_signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only realized, sanitized opponent public actions."""

    rows = [row for row in decision_signals if row.get("role") == ROLE_OPPONENT]
    category_counts = _action_counts(rows)
    card_ids: Counter[str] = Counter()
    card_names: Counter[str] = Counter()
    attack_ids: Counter[str] = Counter()
    for option in _selected_options(rows):
        category = _option_category(option)
        if category not in _PUBLIC_OPPONENT_CARD_CATEGORIES:
            continue
        card_id = _integer(option.get("source_card_id"))
        card_name = _text(option.get("source_card_name"))
        attack_id = _integer(option.get("attack_id"))
        if card_id is not None:
            card_ids[str(card_id)] += 1
        if card_name is not None:
            card_names[card_name] += 1
        if category == "ATTACK" and attack_id is not None:
            attack_ids[str(attack_id)] += 1

    return {
        "decision_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "source_card_id_counts": dict(sorted(card_ids.items())),
        "source_card_name_counts": dict(sorted(card_names.items())),
        "attack_id_counts": dict(sorted(attack_ids.items())),
        "provenance": PROVENANCE_PUBLIC_EVENT,
    }


def summarize_episode(
    episode: ReplayEpisode,
    *,
    actor_seat: int,
    decision_signals: Sequence[Mapping[str, Any]],
    ordinal: int | None = None,
    opponent_label: str | None = None,
    terminal_snapshot: Mapping[str, Any] | None = None,
    terminal_reason: str | None = None,
    focal_evolution_card_ids: Iterable[int] = (),
    focal_evolution_terms: Iterable[str] = (),
    starting_prizes: int = DEFAULT_STARTING_PRIZES,
) -> dict[str, Any]:
    """Summarize one episode relative to the submitted actor seat."""

    if actor_seat not in (0, 1):
        raise ValueError("actor_seat must be 0 or 1")
    if starting_prizes <= 0:
        raise ValueError("starting_prizes must be positive")
    opponent_seat = 1 - actor_seat
    actor_rows = [
        row
        for row in decision_signals
        if row.get("episode_id") == episode.episode_id and _integer(row.get("seat")) == actor_seat
    ]
    opponent_rows = [
        row
        for row in decision_signals
        if row.get("episode_id") == episode.episode_id and _integer(row.get("seat")) == opponent_seat
    ]
    actor_rows.sort(key=lambda row: (_integer(row.get("request_step")) or 0, str(row.get("decision_id") or "")))
    opponent_rows.sort(key=lambda row: (_integer(row.get("request_step")) or 0, str(row.get("decision_id") or "")))
    actor_battle_rows = [row for row in actor_rows if (_integer(row.get("turn")) or 0) > 0]
    opponent_battle_rows = [row for row in opponent_rows if (_integer(row.get("turn")) or 0) > 0]
    result, reward = _episode_result(episode, actor_seat)
    actor_team = episode.team_names[actor_seat] if actor_seat < len(episode.team_names) else None
    opponent_team = episode.team_names[opponent_seat] if opponent_seat < len(episode.team_names) else None
    actor_deck = _deck_for_seat(episode, actor_seat)
    opponent_deck = _deck_for_seat(episode, opponent_seat)
    first_second_values = _snapshot_values(actor_battle_rows or actor_rows, "first_second_proxy")
    first_second = str(first_second_values[0]) if first_second_values else None

    own_prizes = [_integer(value) for value in _snapshot_values(actor_battle_rows, "own", "prizes_remaining")]
    opp_prizes = [_integer(value) for value in _snapshot_values(actor_battle_rows, "opponent", "prizes_remaining")]
    terminal_own = _mapping(_mapping(terminal_snapshot).get("own"))
    terminal_opponent = _mapping(_mapping(terminal_snapshot).get("opponent"))
    terminal_own_prizes = _integer(terminal_own.get("prizes_remaining"))
    terminal_opponent_prizes = _integer(terminal_opponent.get("prizes_remaining"))
    if terminal_own_prizes is not None:
        own_prizes.append(terminal_own_prizes)
    if terminal_opponent_prizes is not None:
        opp_prizes.append(terminal_opponent_prizes)
    own_prizes = [value for value in own_prizes if value is not None]
    opp_prizes = [value for value in opp_prizes if value is not None]
    prizes_taken = starting_prizes - min(own_prizes) if own_prizes else None
    prizes_conceded = starting_prizes - min(opp_prizes) if opp_prizes else None
    prize_leads = [_integer(value) for value in _snapshot_values(actor_rows, "prize_lead")]
    prize_leads = [value for value in prize_leads if value is not None]
    never_led = max(prize_leads) <= 0 if prize_leads else None

    actor_counts = _action_counts(actor_battle_rows)
    opponent_counts = _action_counts(opponent_battle_rows)
    actor_turns = {_integer(row.get("turn")) for row in actor_battle_rows}
    actor_turns.discard(None)
    focal_ids = frozenset(int(value) for value in focal_evolution_card_ids)
    focal_terms = tuple(str(value).strip() for value in focal_evolution_terms if str(value).strip())
    focal_configured = bool(focal_ids or focal_terms)
    focal_evolved = _matches_focal_evolution(
        _selected_options(actor_battle_rows),
        focal_card_ids=focal_ids,
        focal_terms=focal_terms,
    ) if focal_configured else None

    boardout = _terminal_tag(terminal_snapshot, terminal_reason, kind="boardout")
    deckout = _terminal_tag(terminal_snapshot, terminal_reason, kind="deckout")
    prizeout_value = terminal_opponent_prizes == 0 if terminal_opponent_prizes is not None else None
    prizeout = _tag(
        prizeout_value,
        PROVENANCE_ACTOR_VISIBLE if prizeout_value is not None else PROVENANCE_UNAVAILABLE,
        "terminal actor-visible opponent prize count reached zero",
    )
    zero_prize = _tag(
        prizes_taken == 0 if prizes_taken is not None else None,
        PROVENANCE_ACTOR_VISIBLE if prizes_taken is not None else PROVENANCE_UNAVAILABLE,
        "maximum actor-visible prize progress",
    )
    never_led_tag = _tag(
        never_led,
        PROVENANCE_ACTOR_VISIBLE if never_led is not None else PROVENANCE_UNAVAILABLE,
        "actor-visible prize lead never exceeded zero",
    )
    no_focal_evolution = _tag(
        not focal_evolved if focal_evolved is not None else None,
        PROVENANCE_PUBLIC_EVENT if focal_evolved is not None else PROVENANCE_UNAVAILABLE,
        "configured focal evolution was not observed among realized actor actions",
    )
    tags = {
        "boardout": boardout,
        "deckout": deckout,
        "prizeout": prizeout,
        "zero_prize": zero_prize,
        "never_led": never_led_tag,
        "no_focal_evolution": no_focal_evolution,
    }
    active_tags = [name for name, evidence in tags.items() if evidence.get("value") is True]

    terminal_failure_class = None
    failure_provenance = PROVENANCE_UNAVAILABLE
    if result == "win":
        terminal_failure_class = "not_a_loss"
        failure_provenance = PROVENANCE_PUBLIC_EVENT
    elif result == "draw":
        terminal_failure_class = "draw"
        failure_provenance = PROVENANCE_PUBLIC_EVENT
    elif result == "loss":
        exact_failures = [
            name
            for name, evidence in (("deckout", deckout), ("boardout", boardout), ("prizeout", prizeout))
            if evidence["value"] is True
        ]
        if exact_failures:
            terminal_failure_class = "+".join(exact_failures)
            failure_provenance = PROVENANCE_ACTOR_VISIBLE
        else:
            terminal_failure_class = "loss_unclassified"
            failure_provenance = PROVENANCE_HEURISTIC

    return {
        "schema_version": EPISODE_SIGNAL_SCHEMA_VERSION,
        "episode_id": episode.episode_id,
        "ordinal": ordinal,
        "actor_seat": actor_seat,
        "opponent_seat": opponent_seat,
        "result": result,
        "reward": reward,
        "actor_team": actor_team,
        "opponent_team": opponent_team,
        "actor_deck": actor_deck,
        "opponent_deck": opponent_deck,
        "opponent_label": opponent_label,
        "first_second_proxy": first_second,
        "self_mirror": episode.self_mirror,
        "seat_binding_status": episode.seat_binding_status,
        "alignment_status": episode.alignment_status,
        "decision_counts": {
            "actor": len(actor_rows),
            "opponent": len(opponent_rows),
            "actor_battle": len(actor_battle_rows),
            "opponent_battle": len(opponent_battle_rows),
        },
        "turn_count_proxy": len(actor_turns),
        "action_counts": {
            "actor": dict(sorted(actor_counts.items())),
            "opponent": dict(sorted(opponent_counts.items())),
        },
        "tempo": {
            "first_attack_turn": _first_action_turn(actor_battle_rows, "ATTACK"),
            "first_evolution_turn": _first_action_turn(actor_battle_rows, "EVOLVE"),
            "first_attachment_turn": _first_action_turn(actor_battle_rows, "ATTACH"),
            "maximum_prize_lead": max(prize_leads) if prize_leads else None,
            "provenance": PROVENANCE_ACTOR_VISIBLE,
        },
        "prizes": {
            "taken": prizes_taken,
            "conceded": prizes_conceded,
            "starting_prizes": starting_prizes,
            "provenance": PROVENANCE_ACTOR_VISIBLE if own_prizes or opp_prizes else PROVENANCE_UNAVAILABLE,
        },
        "terminal_failure_class": terminal_failure_class,
        "terminal_failure_provenance": failure_provenance,
        "terminal_state_snapshot": dict(terminal_snapshot) if terminal_snapshot is not None else None,
        "active_tags": active_tags,
        "tag_evidence": tags,
        "opponent_public_actions": aggregate_public_actions(opponent_battle_rows),
        "provenance": {
            "result": PROVENANCE_PUBLIC_EVENT if reward is not None else PROVENANCE_UNAVAILABLE,
            "teams": PROVENANCE_PUBLIC_EVENT,
            "decks": PROVENANCE_POSTHOC_DECK,
            "first_second": PROVENANCE_ACTOR_VISIBLE if first_second is not None else PROVENANCE_UNAVAILABLE,
            "signals": PROVENANCE_HEURISTIC,
        },
    }


def _average(values: Iterable[Any]) -> float | None:
    finite = [value for value in (finite_float(item) for item in values) if value is not None]

    return sum(finite) / len(finite) if finite else None


def aggregate_matchups(episode_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank exposure-adjusted posthoc matchup groups by observed losses."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for summary in episode_summaries:
        opponent_deck = _mapping(summary.get("opponent_deck"))
        key = (
            _text(summary.get("opponent_label"))
            or _text(opponent_deck.get("stable_hash"))
            or _text(summary.get("opponent_team"))
            or "unknown"
        )
        groups.setdefault(key, []).append(summary)
    total_losses = sum(summary.get("result") == "loss" for summary in episode_summaries)
    rows: list[dict[str, Any]] = []
    for key, values in groups.items():
        wins = sum(value.get("result") == "win" for value in values)
        losses = sum(value.get("result") == "loss" for value in values)
        draws = sum(value.get("result") == "draw" for value in values)
        decisive = wins + losses
        rows.append({
            "matchup": key,
            "games": len(values),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "decisive_win_rate": wins / decisive if decisive else None,
            "net_wins": wins - losses,
            "loss_share": losses / total_losses if total_losses else None,
            "average_prizes_taken": _average(_mapping(value.get("prizes")).get("taken") for value in values),
            "average_prizes_conceded": _average(_mapping(value.get("prizes")).get("conceded") for value in values),
            "provenance": PROVENANCE_POSTHOC_DECK,
        })
    rows.sort(key=lambda row: (-int(row["losses"]), float(row["decisive_win_rate"] or 0.0), -int(row["games"]), str(row["matchup"])))

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "episodes": len(episode_summaries),
        "matchups": rows,
        "interpretation": "observed exposure and outcomes; not causal matchup attribution",
        "provenance": PROVENANCE_POSTHOC_DECK,
    }


def aggregate_loss_signals(episode_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count observational signal co-occurrence among losses."""

    losses = [summary for summary in episode_summaries if summary.get("result") == "loss"]
    wins = [summary for summary in episode_summaries if summary.get("result") == "win"]
    tag_names = sorted({
        str(name)
        for summary in episode_summaries
        for name in _mapping(summary.get("tag_evidence"))
    })
    rows: list[dict[str, Any]] = []
    for name in tag_names:
        all_evidence = [_mapping(_mapping(summary.get("tag_evidence")).get(name)).get("value") for summary in episode_summaries]
        loss_evidence = [_mapping(_mapping(summary.get("tag_evidence")).get(name)).get("value") for summary in losses]
        win_evidence = [_mapping(_mapping(summary.get("tag_evidence")).get(name)).get("value") for summary in wins]
        available_losses = [value for value in loss_evidence if isinstance(value, bool)]
        positive_losses = sum(value is True for value in available_losses)
        available_wins = [value for value in win_evidence if isinstance(value, bool)]
        positive_wins = sum(value is True for value in available_wins)
        available_all = [value for value in all_evidence if isinstance(value, bool)]
        loss_rate = positive_losses / len(available_losses) if available_losses else None
        win_rate = positive_wins / len(available_wins) if available_wins else None
        rows.append({
            "signal": name,
            "losses_with_signal": positive_losses,
            "losses_with_evidence": len(available_losses),
            "loss_cooccurrence_rate": loss_rate,
            "wins_with_signal": positive_wins,
            "wins_with_evidence": len(available_wins),
            "win_cooccurrence_rate": win_rate,
            "loss_minus_win_cooccurrence": loss_rate - win_rate
            if loss_rate is not None and win_rate is not None
            else None,
            "all_games_with_signal": sum(value is True for value in available_all),
            "all_games_with_evidence": len(available_all),
            "provenance": PROVENANCE_HEURISTIC,
        })
    rows.sort(
        key=lambda row: (
            -int(row["losses_with_signal"]),
            -float(row["loss_minus_win_cooccurrence"] or 0.0),
            str(row["signal"]),
        )
    )
    failure_counts = Counter(
        str(summary.get("terminal_failure_class") or "unavailable")
        for summary in losses
    )

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "episodes": len(episode_summaries),
        "losses": len(losses),
        "signals": rows,
        "terminal_failure_counts": dict(sorted(failure_counts.items())),
        "interpretation": "observational co-occurrence only; no counterfactual blunder or causal claim",
        "provenance": PROVENANCE_HEURISTIC,
    }


def _episode_order(summary: Mapping[str, Any]) -> tuple[int, int, str]:
    ordinal = _integer(summary.get("ordinal"))
    episode_id = _integer(summary.get("episode_id"))

    return (
        0 if ordinal is not None else 1,
        ordinal if ordinal is not None else episode_id if episode_id is not None else 0,
        str(summary.get("episode_id") or ""),
    )


def _outcome_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = sum(value.get("result") == "win" for value in values)
    losses = sum(value.get("result") == "loss" for value in values)
    draws = sum(value.get("result") == "draw" for value in values)
    decisive = wins + losses

    return {
        "games": len(values),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_win_rate": wins / decisive if decisive else None,
    }


def aggregate_timeline(
    episode_summaries: Sequence[Mapping[str, Any]],
    *,
    window_size: int = 10,
) -> dict[str, Any]:
    """Build ordered rolling and early/middle/late outcome summaries."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    ordered = sorted(episode_summaries, key=_episode_order)
    rolling: list[dict[str, Any]] = []
    for index, summary in enumerate(ordered):
        window = ordered[max(0, index - window_size + 1):index + 1]
        rolling.append({
            "episode_id": summary.get("episode_id"),
            "ordinal": summary.get("ordinal"),
            "result": summary.get("result"),
            "window": _outcome_counts(window),
        })
    phase_values: dict[str, list[Mapping[str, Any]]] = {"early": [], "middle": [], "late": []}
    phase_names = ("early", "middle", "late")
    for index, summary in enumerate(ordered):
        phase_index = min(2, (index * 3) // max(1, len(ordered)))
        phase_values[phase_names[phase_index]].append(summary)
    phases = {
        phase: _outcome_counts(values)
        for phase, values in phase_values.items()
    }

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "episodes": len(ordered),
        "window_size": window_size,
        "phases": phases,
        "rolling": rolling,
        "interpretation": "chronological descriptive outcomes; no Elo values are inferred",
        "provenance": PROVENANCE_HEURISTIC,
    }


__all__ = [
    "aggregate_loss_signals",
    "aggregate_matchups",
    "aggregate_public_actions",
    "aggregate_timeline",
    "build_decision_signal",
    "extract_state_snapshot",
    "summarize_episode",
]
