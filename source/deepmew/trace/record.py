"""Recording helpers for deterministic decision frames."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from cg.api import AreaType, OptionType, SelectContext, SelectType, all_attack, all_card_data
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown
from deepmew.card_semantics import effective_attack_semantics, visible_basic_water_in_discard
from deepmew.deck_profiles import (
    BASIC_WATER_ENERGY_ID,
    HAMMER_LANCHE_ATTACK_ID,
    MEGA_SIGNAL_CARD_ID,
    RIPTIDE_ATTACK_ID,
    WAITRESS_CARD_ID,
)
from deepmew.options import resolve_option_identity
from deepmew.trace.schema import DECISION_FRAME_SCHEMA_VERSION, DecisionFrame, OptionSummary, to_jsonable
from deepmew.replay.information_boundary import InformationMode, redact_public_observation


_ATTACK_BY_ID: dict[int, Any] | None = None
_CARD_BY_ID: dict[int, Any] | None = None


def _enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "None"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _attack_by_id() -> dict[int, Any]:
    global _ATTACK_BY_ID
    if _ATTACK_BY_ID is None:
        _ATTACK_BY_ID = {int(attack.attackId): attack for attack in all_attack()}
    return _ATTACK_BY_ID


def _card_by_id() -> dict[int, Any]:
    global _CARD_BY_ID
    if _CARD_BY_ID is None:
        cards: dict[int, Any] = {}
        for card in all_card_data():
            card_id = _int_or_none(_get(card, "id", _get(card, "cardId")))
            if card_id is not None:
                cards[card_id] = card
        _CARD_BY_ID = cards
    return _CARD_BY_ID


def _card_id(card: Any) -> int | None:
    return _int_or_none(_get(card, "id", _get(card, "cardId")))


def _card_name(card_id: int | None) -> str | None:
    if card_id is None:
        return None
    card = _card_by_id().get(card_id)
    if card is None:
        return None
    return _get(card, "name", _get(card, "cardName"))


def _attack_name(attack_id: int | None) -> str | None:
    if attack_id is None:
        return None
    attack = _attack_by_id().get(attack_id)
    if attack is None:
        return None
    return _get(attack, "name", _get(attack, "attackName"))


def _printed_damage(attack_id: int | None) -> int | None:
    if attack_id is None:
        return None
    attack = _attack_by_id().get(attack_id)
    if attack is None:
        return None
    return _int_or_none(_get(attack, "damage"))


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    return _enum_name(AreaType, value)


def option_summary(index: int, option: dict[str, Any], obs: dict[str, Any] | None = None) -> OptionSummary:
    identity = resolve_option_identity(obs or {}, option)
    option_type = identity.option_type
    attack_id = _int_or_none(option.get("attackId"))
    if attack_id is None:
        attack_id = identity.attack_id
    card_id = identity.card_id
    if card_id is None and option_type != "PLAY":
        card_id = _int_or_none(option.get("cardId", option.get("cardID")))
    card_name = identity.card_name or _card_name(card_id)
    area = identity.target_area or _area_name(option.get("inPlayArea", option.get("area")))
    target = None
    if option.get("target") is not None:
        target = str(option.get("target"))
    elif option.get("inPlayIndex") is not None:
        target = f"inPlayIndex={option.get('inPlayIndex')}"

    effective_damage = None
    if obs is not None:
        try:
            semantics = effective_attack_semantics(option, obs)
            if semantics is not None:
                effective_damage = semantics.effective_damage
        except Exception:
            effective_damage = None

    parts = [option_type]
    if card_id is not None:
        parts.append(f"cardId={card_id}" + (f" {card_name}" if card_name else ""))
    if attack_id is not None:
        attack_label = _attack_name(attack_id)
        parts.append(f"attackId={attack_id}" + (f" {attack_label}" if attack_label else ""))
    if identity.source_area is not None and identity.source_index is not None:
        parts.append(f"source={identity.source_area}[{identity.source_index}]")
    if area:
        parts.append(f"area={area}")
    if target:
        parts.append(target)
    if effective_damage is not None:
        parts.append(f"effective_damage={effective_damage}")

    return OptionSummary(
        index=index,
        option_type=option_type,
        summary=" ".join(parts),
        attack_id=attack_id,
        card_id=card_id,
        card_name=card_name,
        area=area,
        target=target,
        attack_name=_attack_name(attack_id),
        effective_damage=effective_damage,
        printed_damage=_printed_damage(attack_id),
        resolved=identity.resolved,
        resolve_reason=identity.reason,
        source_area=identity.source_area,
        source_index=identity.source_index,
        target_area=identity.target_area,
        target_index=identity.target_index,
        target_card_id=identity.target_card_id,
        target_card_name=identity.target_card_name,
    )


def summarize_options(select: dict[str, Any], obs: dict[str, Any] | None = None) -> list[OptionSummary]:
    return [option_summary(index, option, obs) for index, option in enumerate(select.get("option") or [])]


def summarize_chosen_options(select: dict[str, Any], chosen_indices: list[int], obs: dict[str, Any] | None = None) -> list[OptionSummary]:
    options = select.get("option") or []
    summaries = []
    for index in chosen_indices:
        if isinstance(index, int) and 0 <= index < len(options):
            summaries.append(option_summary(index, options[index], obs))
    return summaries


def _pokemon_summary(pokemon: Any) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    card_id = _card_id(pokemon)
    return {
        "card_id": card_id,
        "name": _card_name(card_id),
        "hp": _int_or_none(_get(pokemon, "hp")),
        "energy_count": len(_get(pokemon, "energyCards", []) or []),
        "tool_count": len(_get(pokemon, "tools", []) or []),
    }


def _player(obs: dict[str, Any], *, own: bool) -> Any | None:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex") or 0)
    player_index = your_index if own else 1 - your_index
    if player_index < 0 or player_index >= len(players):
        return None
    return players[player_index]


def _legal_flags(legal_options: list[OptionSummary]) -> dict[str, bool]:
    types = {summary.option_type for summary in legal_options}
    card_ids = {summary.card_id for summary in legal_options}
    attack_ids = {summary.attack_id for summary in legal_options}
    return {
        "hammer_lanche_legal": HAMMER_LANCHE_ATTACK_ID in attack_ids,
        "riptide_legal": RIPTIDE_ATTACK_ID in attack_ids,
        "attach_legal": "ATTACH" in types,
        "evolve_legal": "EVOLVE" in types,
        "play_legal": "PLAY" in types,
        "mega_signal_legal": MEGA_SIGNAL_CARD_ID in card_ids,
        "waitress_legal": WAITRESS_CARD_ID in card_ids,
    }


def board_summary(obs: dict[str, Any], legal_options: list[OptionSummary]) -> dict[str, Any]:
    current = obs.get("current") or {}
    own = _player(obs, own=True) or {}
    opponent = _player(obs, own=False) or {}
    own_active_list = _get(own, "active", []) or []
    opponent_active_list = _get(opponent, "active", []) or []
    own_bench = [_pokemon_summary(pokemon) for pokemon in (_get(own, "bench", []) or []) if pokemon is not None]
    flags = _legal_flags(legal_options)
    discard = _get(own, "discard", []) or []
    discard_counts = Counter(_card_id(card) for card in discard)
    summary = {
        "turn": _int_or_none(current.get("turn")),
        "your_index": _int_or_none(current.get("yourIndex")),
        "own_active": _pokemon_summary(own_active_list[0] if own_active_list else None),
        "own_bench": own_bench,
        "opponent_active": _pokemon_summary(opponent_active_list[0] if opponent_active_list else None),
        "own_deck_count": _int_or_none(_get(own, "deckCount")),
        "own_discard_basic_water_count": visible_basic_water_in_discard(obs),
        "own_discard_basic_water_count_fallback": discard_counts.get(BASIC_WATER_ENERGY_ID, 0),
    }
    summary.update(flags)
    return summary


def safe_jsonable_observation(obs: dict[str, Any]) -> tuple[Any, list[str]]:
    notes: list[str] = []
    try:
        payload = to_jsonable(redact_public_observation(obs))
        return payload, notes
    except Exception as exc:
        notes.append(f"raw observation omitted: {type(exc).__name__}: {exc}")
        return None, notes


def build_decision_frame(
    *,
    run_id: str,
    game_id: str,
    frame_id: str,
    seed: int | None,
    obs: dict[str, Any],
    chosen_indices: list[int],
    decision_index: int,
    agent_spec: str,
    agent_label: str,
    policy_version: str | None,
    deck_id: str | None = None,
    deck_path: str | None = None,
    final_result: int | None = None,
    final_result_reason: str | None = None,
    notes: list[str] | None = None,
) -> DecisionFrame:
    select = obs.get("select") or {}
    current = obs.get("current") or {}
    legal_options = summarize_options(select, obs)
    raw_observation, raw_notes = safe_jsonable_observation(obs)
    frame_notes = list(notes or []) + raw_notes
    return DecisionFrame(
        schema_version=DECISION_FRAME_SCHEMA_VERSION,
        run_id=run_id,
        game_id=game_id,
        frame_id=frame_id,
        seed=seed,
        seat=_int_or_none(current.get("yourIndex")),
        turn=_int_or_none(current.get("turn")),
        decision_index=decision_index,
        agent_spec=agent_spec,
        agent_label=agent_label,
        policy_version=policy_version,
        deck_id=deck_id,
        deck_path=deck_path,
        select_type=_enum_name(SelectType, select.get("type")),
        select_context=_enum_name(SelectContext, select.get("context")),
        min_count=int(select.get("minCount") or 0),
        max_count=int(select.get("maxCount") or 0),
        legal_options=legal_options,
        chosen_indices=list(chosen_indices),
        chosen_option_summaries=summarize_chosen_options(select, chosen_indices, obs),
        board_summary=board_summary(obs, legal_options),
        final_result=final_result,
        final_result_reason=final_result_reason,
        raw_observation=raw_observation,
        notes=frame_notes,
        information_mode=InformationMode.PUBLIC_BELIEF.value,
    )


class TraceRecorder:
    """Small in-memory trace recorder for arena smoke and diagnosis runs."""

    def __init__(self, *, run_id: str, sample_limit: int | None = None) -> None:
        self.run_id = run_id
        self.sample_limit = sample_limit
        self.frames: list[DecisionFrame] = []
        self.skipped_frames = 0

    def record_decision(
        self,
        *,
        game_id: str,
        seed: int | None,
        obs: dict[str, Any],
        chosen_indices: list[int],
        decision_index: int,
        agent_spec: str,
        agent_label: str,
        policy_version: str | None,
        deck_id: str | None = None,
        deck_path: str | None = "deck.csv",
    ) -> DecisionFrame | None:
        if self.sample_limit is not None and len(self.frames) >= self.sample_limit:
            self.skipped_frames += 1
            return None
        frame_id = f"{game_id}:d{decision_index:04d}"
        frame = build_decision_frame(
            run_id=self.run_id,
            game_id=game_id,
            frame_id=frame_id,
            seed=seed,
            obs=obs,
            chosen_indices=chosen_indices,
            decision_index=decision_index,
            agent_spec=agent_spec,
            agent_label=agent_label,
            policy_version=policy_version,
            deck_id=None,
            deck_path=deck_path,
        )
        self.frames.append(frame)
        return frame

    def finalize_game(self, *, game_id: str, final_result: int | None, final_result_reason: str | None) -> None:
        for frame in self.frames:
            if frame.game_id == game_id:
                frame.final_result = final_result
                frame.final_result_reason = final_result_reason

    def summary(self) -> dict[str, Any]:
        return {
            "frames": len(self.frames),
            "skipped_frames": self.skipped_frames,
            "games": len({frame.game_id for frame in self.frames}),
        }

    def write(self, *, frames_path: Path, summary_path: Path | None = None, report_path: Path | None = None) -> dict[str, str]:
        write_jsonl(frames_path, [frame.to_jsonable() for frame in self.frames])
        outputs = {"frames_jsonl": str(frames_path)}
        if summary_path is not None:
            write_json(summary_path, self.summary())
            outputs["summary_json"] = str(summary_path)
        if report_path is not None:
            lines = [
                "# Trace Run",
                "",
                f"run_id={self.run_id}",
                f"frames={len(self.frames)}",
                f"skipped_frames={self.skipped_frames}",
                f"information_mode={InformationMode.PUBLIC_BELIEF.value}",
                "",
                "This trace is ordered decision-frame evidence, not replay learning.",
            ]
            write_markdown(report_path, "\n".join(lines))
            outputs["report_markdown"] = str(report_path)
        return outputs
