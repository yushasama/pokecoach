"""Prospective, chronology-aware decision traces for offline research."""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cg.api import LogType, SelectContext, SelectType

from deepmew.artifacts.writers import write_json
from deepmew.replay.information_boundary import (
    InformationBoundary,
    InformationMode,
    classify_observation_visibility,
    merge_visualizer_private_state,
    redact_public_observation,
)
from deepmew.replay.canonical_actor_dataset import validate_action_contract
from deepmew.rl1_teacher_bc.semantics import legal_action_table


PLAN_SCHEMA_VERSION = "deepmew.prospective_trace.plan.v1"
TRANCHE_SCHEMA_VERSION = "deepmew.prospective_trace.tranche.v1"
PUBLIC_ROW_SCHEMA_VERSION = "deepmew.prospective_trace.public_decision.v2"
ACTOR_VISIBLE_ROW_SCHEMA_VERSION = "deepmew.prospective_trace.actor_visible_decision.v1"
ORACLE_ROW_SCHEMA_VERSION = "deepmew.prospective_trace.oracle_decision.v1"
PROGRESS_SCHEMA_VERSION = "deepmew.prospective_trace.progress.v1"
PUBLIC_PROJECTION_VERSION = "deepmew.public_trace_projection.v2"

# CABT log rows are actor-perspective observations. Public projection is
# fail-closed per vendor LogType: unknown event types retain only their type and
# player, while known types may add only the fields documented for that event.
# Card identities and per-card serials are deliberately excluded for every
# event so an earlier private draw/search cannot be linked to a later reveal.
PUBLIC_LOG_BASE_FIELDS: tuple[str, ...] = ("type", "playerIndex")
PUBLIC_LOG_FIELDS_BY_TYPE: dict[int, tuple[str, ...]] = {
    int(LogType.HAS_BASIC_POKEMON): ("hasBasicPokemon",),
    int(LogType.MOVE_CARD): ("fromArea", "toArea"),
    int(LogType.MOVE_CARD_REVERSE): ("fromArea", "toArea"),
    int(LogType.ATTACK): ("attackId",),
    int(LogType.HP_CHANGE): ("value", "putDamageCounter"),
    int(LogType.POISONED): ("isRecover",),
    int(LogType.BURNED): ("isRecover",),
    int(LogType.ASLEEP): ("isRecover",),
    int(LogType.PARALYZED): ("isRecover",),
    int(LogType.CONFUSED): ("isRecover",),
    int(LogType.COIN): ("head",),
    int(LogType.RESULT): ("result", "reason"),
}
PRIVATE_ROW_FIELDS: frozenset[str] = frozenset({
    "actor_private_observation",
    "true_private_state",
    "exact_legal_actions",
    "chosen_indices_private",
    "raw_observation",
    "post_observation",
    "legal_set",
    "chosen_action",
})

VISIBILITY = {
    "public_observation_before": "deployment-visible",
    "public_observation_after": "deployment-visible",
    "public_event_delta": "deployment-visible",
    "public_projection_version": PUBLIC_PROJECTION_VERSION,
    "policy_identity": "research-visible",
    "deck_identity": "research-visible",
    "rating_snapshot": "research-visible when time-valid; unavailable otherwise",
    "mature_rating": "audit-only",
    "actor_private_observation": "actor-visible/oracle-only",
    "exact_legal_actions": "actor-visible/oracle-only",
    "true_private_state": "private/oracle-only",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _enum_name(enum_type: Any, value: Any) -> str:
    try:
        return enum_type(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def unavailable_rating_snapshot() -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "rating_snapshot_id": None,
        "snapshot_at_utc": None,
        "algorithm_version": "not_found",
        "rating_before_game": None,
        "games_played_before": None,
        "rating_uncertainty_before": None,
        "rating_after_game_audit_only": None,
        "mature_rating_audit_only": None,
    }


def validate_rating_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    availability = str(snapshot.get("availability") or "")
    if availability not in {"available", "unavailable"}:
        raise ValueError("rating snapshot availability must be available or unavailable")
    if availability == "available":
        required = ("rating_snapshot_id", "snapshot_at_utc", "algorithm_version", "rating_before_game")
        missing = [field for field in required if snapshot.get(field) in (None, "")]
        if missing:
            raise ValueError(f"available rating snapshot missing: {', '.join(missing)}")
    before = snapshot.get("rating_before_game")
    after = snapshot.get("rating_after_game_audit_only")
    if before is None and after is not None and availability == "available":
        raise ValueError("postgame rating cannot replace missing pregame rating")

    return copy.deepcopy(snapshot)


def public_trace_logs(logs: Any) -> list[dict[str, Any]]:
    """Project actor-perspective CABT logs into a seat-neutral public stream."""
    projected: list[dict[str, Any]] = []
    for value in _as_list(logs):
        event = _as_dict(value)
        allowed_fields = _public_log_fields(event)
        public_event = {
            field: copy.deepcopy(event[field])
            for field in allowed_fields
            if field in event and event[field] is not None
        }
        if public_event:
            projected.append(public_event)

    return projected


def _public_log_fields(event: dict[str, Any]) -> tuple[str, ...]:
    event_type = _as_int(event.get("type"))
    specific = PUBLIC_LOG_FIELDS_BY_TYPE.get(event_type, ())

    return PUBLIC_LOG_BASE_FIELDS + specific


def public_trace_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Return seat-neutral public state, omitting legal/private search surfaces."""
    public = redact_public_observation(observation)
    public.pop("search_begin_input", None)
    public.pop("select", None)
    public["logs"] = public_trace_logs(public.get("logs"))
    current = _as_dict(public.get("current"))
    current["looking"] = None
    for player_value in _as_list(current.get("players")):
        player = _as_dict(player_value)
        player["hand"] = None
        player.pop("deck", None)
        player.pop("deckOrder", None)
        player.pop("trueDeckOrder", None)
        prize = player.get("prize")
        if isinstance(prize, list):
            player["prize"] = [None] * len(prize)

    return public


def _assert_public_observation_safe(observation: dict[str, Any], *, field: str) -> None:
    if "select" in observation or "search_begin_input" in observation:
        raise ValueError(f"public trace {field} retained a private selection/search surface")
    current = _as_dict(observation.get("current"))
    if current.get("looking") is not None:
        raise ValueError(f"public trace {field} retained looked-at card identities")
    for player_value in _as_list(current.get("players")):
        player = _as_dict(player_value)
        if player.get("hand") is not None:
            raise ValueError(f"public trace {field} retained a hand")
        if any(key in player for key in ("deck", "deckOrder", "trueDeckOrder")):
            raise ValueError(f"public trace {field} retained a deck-order surface")
        prize = player.get("prize")
        if isinstance(prize, list) and any(card is not None for card in prize):
            raise ValueError(f"public trace {field} retained a prize identity")
    for event in _as_list(observation.get("logs")):
        event_dict = _as_dict(event)
        unexpected = set(event_dict).difference(_public_log_fields(event_dict))
        if unexpected:
            raise ValueError(f"public trace {field} retained unsafe log fields: {sorted(unexpected)}")


def assert_public_trace_row_safe(row: dict[str, Any]) -> None:
    """Fail closed if a public row contains actor-private or hidden-card state."""
    if row.get("information_mode") != InformationMode.PUBLIC_BELIEF.value:
        return
    if row.get("public_projection_version") != PUBLIC_PROJECTION_VERSION:
        raise ValueError("public trace row has an unknown or missing projection version")
    leaked_fields = PRIVATE_ROW_FIELDS.intersection(row)
    if leaked_fields:
        raise ValueError(f"public trace row retained private fields: {sorted(leaked_fields)}")
    observation_fields = (
        "public_observation_before",
        "public_observation_after",
        "end_of_actor_turn_public_observation",
        "next_same_actor_public_observation",
    )
    for field in observation_fields:
        observation = row.get(field)
        if isinstance(observation, dict):
            _assert_public_observation_safe(observation, field=field)
    event_surfaces = (
        _as_list(_as_dict(row.get("public_chosen_action")).get("events")),
        _as_list(row.get("public_event_delta")),
    )
    for events in event_surfaces:
        for event in events:
            event_dict = _as_dict(event)
            unsafe = set(event_dict).difference(_public_log_fields(event_dict))
            if unsafe:
                raise ValueError(f"public trace row retained unsafe event fields: {sorted(unsafe)}")


def actor_private_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep only the acting seat's legitimate private observation."""
    private = redact_public_observation(observation)
    private.pop("search_begin_input", None)
    current = _as_dict(private.get("current"))
    for player_value in _as_list(current.get("players")):
        player = _as_dict(player_value)
        prize = player.get("prize")
        if isinstance(prize, list):
            player["prize"] = [None] * len(prize)

    return private


def assert_actor_visible_trace_row_safe(row: dict[str, Any]) -> None:
    """Fail closed on actor-visible source rows before they reach conversion."""
    if row.get("information_mode") != InformationMode.ACTOR_VISIBLE.value:
        return
    if row.get("schema_version") != ACTOR_VISIBLE_ROW_SCHEMA_VERSION:
        raise ValueError("actor-visible trace row has an unknown schema version")
    if row.get("public_projection_version") != PUBLIC_PROJECTION_VERSION:
        raise ValueError("actor-visible trace row has an unknown public projection version")
    forbidden = {"true_private_state", "actor_private_observation", "exact_legal_actions", "chosen_indices_private"}
    leaked = forbidden.intersection(row)
    if leaked:
        raise ValueError(f"actor-visible trace row retained oracle fields: {sorted(leaked)}")

    seat = _as_int(row.get("seat"), -1)
    actor_seat = _as_int(row.get("actor_seat"), -1)
    if seat not in (0, 1) or actor_seat != seat:
        raise ValueError("actor-visible trace row has inconsistent actor seat metadata")
    raw = row.get("raw_observation")
    if classify_observation_visibility(raw) != InformationMode.ACTOR_VISIBLE.value:
        raise ValueError("actor-visible raw observation failed the visibility classifier")
    if "search_begin_input" in _as_dict(raw):
        raise ValueError("actor-visible raw observation retained opaque search state")
    raw_current = _as_dict(_as_dict(raw).get("current"))
    if _as_int(raw_current.get("yourIndex"), -1) != seat:
        raise ValueError("actor-visible raw observation does not belong to the recorded actor")

    post = row.get("post_observation")
    post_actor = _as_int(_as_dict(_as_dict(post).get("current")).get("yourIndex"), -1)
    expected_post_mode = (
        InformationMode.ACTOR_VISIBLE.value
        if post_actor == seat
        else InformationMode.PUBLIC_BELIEF.value
    )
    if classify_observation_visibility(post) != expected_post_mode:
        raise ValueError("actor-visible post observation violated the control-owner boundary")
    if "search_begin_input" in _as_dict(post):
        raise ValueError("actor-visible post observation retained opaque search state")

    legal_set = row.get("legal_set")
    chosen_action = row.get("chosen_action")
    if not validate_action_contract(chosen_action, legal_set):
        raise ValueError("actor-visible trace row has an invalid action contract")
    select = _as_dict(_as_dict(raw).get("select"))
    if legal_set != legal_action_table(_as_dict(raw), select):
        raise ValueError("actor-visible legal action table does not match the raw observation")
    indices = _as_list(_as_dict(chosen_action).get("indices"))
    expected_semantics = [legal_set[index] for index in indices]
    if _as_dict(chosen_action).get("semantics") != expected_semantics:
        raise ValueError("actor-visible chosen action semantics do not match chosen indices")

    required_ids = ("policy_id", "opponent_policy_id", "deck_hash", "opponent_deck_hash")
    if not all(isinstance(row.get(field), str) and row[field] for field in required_ids):
        raise ValueError("actor-visible trace row is missing policy or deck identity")
    result = _as_int(row.get("result"), -1)
    if result not in (0, 1, 2):
        raise ValueError("actor-visible trace row has no valid terminal result")
    outcome = _as_dict(row.get("outcome"))
    flags = [bool(outcome.get(field)) for field in ("won", "lost", "draw")]
    expected_flags = [result == seat, result in (0, 1) and result != seat, result not in (0, 1)]
    if flags != expected_flags or _as_int(outcome.get("result"), -1) != result:
        raise ValueError("actor-visible trace row has an invalid actor-relative outcome")
    safety_pairs = (
        ("crashed", "crashed"),
        ("timed_out", "timed_out"),
        ("illegal", "illegal_action"),
    )
    if any(bool(row.get(row_field)) != bool(outcome.get(outcome_field)) for row_field, outcome_field in safety_pairs):
        raise ValueError("actor-visible trace row has inconsistent safety fields")


def create_plan(
    path: str | Path,
    *,
    plan_id: str,
    run_id: str,
    tranche_ids: list[str],
    frozen_snapshot: dict[str, Any],
    league_identity: str,
    curriculum_round: int | None,
) -> dict[str, Any]:
    if len(tranche_ids) < 3:
        raise ValueError("prospective walk-forward plan requires at least three ordered tranches")
    if len(set(tranche_ids)) != len(tranche_ids):
        raise ValueError("tranche IDs must be unique")
    declared_at = utc_now()
    rating = validate_rating_snapshot(_as_dict(frozen_snapshot.get("rating_snapshot")))
    snapshot = copy.deepcopy(frozen_snapshot)
    snapshot["rating_snapshot"] = rating
    tranches = []
    for sequence, tranche_id in enumerate(tranche_ids):
        tranches.append({
            "schema_version": TRANCHE_SCHEMA_VERSION,
            "tranche_id": str(tranche_id),
            "tranche_sequence": sequence,
            "predecessor_tranche_id": tranche_ids[sequence - 1] if sequence else None,
            "declared_at_utc": declared_at,
            "status": "declared",
            "started_at_utc": None,
            "completed_at_utc": None,
            "frozen_snapshot": copy.deepcopy(snapshot),
        })
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan_id,
        "run_id": run_id,
        "created_at_utc": declared_at,
        "league_identity": league_identity,
        "curriculum_round": curriculum_round,
        "visibility": copy.deepcopy(VISIBILITY),
        "tranches": tranches,
    }
    write_json(Path(path), plan)

    return plan


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported prospective plan schema: {plan.get('schema_version')!r}")
    tranches = _as_list(plan.get("tranches"))
    sequences = [_as_int(row.get("tranche_sequence")) for row in tranches]
    if sequences != list(range(len(tranches))):
        raise ValueError("prospective tranche_sequence must be contiguous and ordered")

    return plan


def tranche_by_id(plan: dict[str, Any], tranche_id: str) -> dict[str, Any]:
    matches = [row for row in _as_list(plan.get("tranches")) if row.get("tranche_id") == tranche_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate tranche_id: {tranche_id}")

    return matches[0]


def begin_tranche(path: str | Path, tranche_id: str, *, frozen_snapshot: dict[str, Any], resume: bool) -> dict[str, Any]:
    path = Path(path)
    plan = load_plan(path)
    tranche = tranche_by_id(plan, tranche_id)
    expected = json.dumps(tranche.get("frozen_snapshot"), sort_keys=True, separators=(",", ":"))
    actual = json.dumps(frozen_snapshot, sort_keys=True, separators=(",", ":"))
    if expected != actual:
        raise ValueError("tranche frozen policy/deck/rating snapshot does not match declared plan")
    predecessor_id = tranche.get("predecessor_tranche_id")
    if predecessor_id and tranche_by_id(plan, str(predecessor_id)).get("status") != "completed":
        raise ValueError(f"predecessor tranche is not completed: {predecessor_id}")
    status = str(tranche.get("status") or "")
    if status in {"completed", "safety_failed"} and not resume:
        raise ValueError(
            f"tranche is {status}; pass resume to verify or explicitly retry: {tranche_id}"
        )
    if status == "declared" or (status == "safety_failed" and resume):
        tranche["status"] = "running"
        if status == "declared":
            tranche["started_at_utc"] = utc_now()
        else:
            tranche["resumed_after_safety_failure_at_utc"] = utc_now()
    write_json(path, plan)

    return plan


def complete_tranche(path: str | Path, tranche_id: str, *, receipt: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    plan = load_plan(path)
    tranche = tranche_by_id(plan, tranche_id)
    tranche["status"] = "completed"
    tranche["completed_at_utc"] = utc_now()
    tranche["receipt"] = copy.deepcopy(receipt)
    write_json(path, plan)

    return plan


def fail_tranche_safety(
    path: str | Path,
    tranche_id: str,
    *,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Persist a tranche safety hard stop before the triggering exception escapes."""
    plan = load_plan(path)
    tranche = tranche_by_id(plan, tranche_id)
    failure = copy.deepcopy(receipt)
    failure["failed_at_utc"] = utc_now()
    failures = tranche.setdefault("safety_failures", [])
    if not isinstance(failures, list):
        raise ValueError("prospective tranche safety_failures must be a list")
    failures.append(failure)
    tranche["status"] = "safety_failed"
    tranche["safety_failed_at_utc"] = failure["failed_at_utc"]
    tranche["safety_failure_receipt"] = failure
    write_json(path, plan)

    return plan


@dataclass(frozen=True)
class ActorMetadata:
    seat: int
    policy: dict[str, Any]
    deck: dict[str, Any]


def _policy_identity(policy: dict[str, Any]) -> str:
    for field in ("feature_policy_id", "league_member_id", "policy_id", "policy_spec"):
        value = policy.get(field)
        if isinstance(value, str) and value:
            return value

    raise ValueError("prospective actor policy metadata has no stable identity")


class ProspectiveGameRecorder:
    def __init__(
        self,
        *,
        game_id: str,
        run_id: str,
        tranche: dict[str, Any],
        seed: int,
        actors: list[ActorMetadata],
        league_identity: str,
        curriculum_round: int | None,
        boundary: InformationBoundary,
        game_started_at_utc: str,
        game_index: int = 0,
    ) -> None:
        self.game_id = game_id
        self.run_id = run_id
        self.tranche = copy.deepcopy(tranche)
        self.seed = seed
        self.actors = {actor.seat: actor for actor in actors}
        self.league_identity = league_identity
        self.curriculum_round = curriculum_round
        self.boundary = boundary
        self.game_started_at_utc = game_started_at_utc
        self.game_index = game_index
        self.rows: list[dict[str, Any]] = []
        self.actor_indices = {seat: 0 for seat in self.actors}

    def record_transition(
        self,
        before: dict[str, Any],
        selected: list[int],
        after: dict[str, Any],
        *,
        visualizer_before: str | None = None,
    ) -> None:
        before_current = _as_dict(before.get("current"))
        after_current = _as_dict(after.get("current"))
        actor = int(before_current.get("yourIndex"))
        opponent = 1 - actor
        decision_index = len(self.rows)
        actor_decision_index = self.actor_indices[actor]
        self.actor_indices[actor] += 1
        select = _as_dict(before.get("select"))
        post_result = _as_int(after_current.get("result"), -1)
        row: dict[str, Any] = {
            "schema_version": {
                InformationMode.PUBLIC_BELIEF: PUBLIC_ROW_SCHEMA_VERSION,
                InformationMode.ACTOR_VISIBLE: ACTOR_VISIBLE_ROW_SCHEMA_VERSION,
                InformationMode.PRIVILEGED_ORACLE: ORACLE_ROW_SCHEMA_VERSION,
            }[self.boundary.mode],
            "public_projection_version": PUBLIC_PROJECTION_VERSION,
            "information_mode": self.boundary.mode.value,
            "run_id": self.run_id,
            "tranche_id": self.tranche["tranche_id"],
            "tranche_sequence": self.tranche["tranche_sequence"],
            "tranche_declared_at_utc": self.tranche["declared_at_utc"],
            "game_id": self.game_id,
            "game_index": self.game_index,
            "decision_id": f"{self.game_id}:d{decision_index:05d}",
            "decision_index": decision_index,
            "actor_decision_index": actor_decision_index,
            "seed": self.seed,
            "game_started_at_utc": self.game_started_at_utc,
            "game_completed_at_utc": None,
            "actor_seat": actor,
            "opponent_seat": opponent,
            "actor_policy": copy.deepcopy(self.actors[actor].policy),
            "opponent_policy": copy.deepcopy(self.actors[opponent].policy),
            "actor_deck": copy.deepcopy(self.actors[actor].deck),
            "opponent_deck": copy.deepcopy(self.actors[opponent].deck),
            "league_identity": self.league_identity,
            "curriculum_round": self.curriculum_round,
            "rating_snapshot": copy.deepcopy(self.tranche["frozen_snapshot"]["rating_snapshot"]),
            "turn": _as_int(before_current.get("turn"), 0),
            "phase": _enum_name(SelectContext, select.get("context")),
            "select_type": _enum_name(SelectType, select.get("type")),
            "control_owner_before": actor,
            "control_owner_after": _as_int(after_current.get("yourIndex")),
            "control_passed": actor != _as_int(after_current.get("yourIndex")),
            "ended_turn": _as_int(before_current.get("turn")) != _as_int(after_current.get("turn")),
            "terminated": post_result != -1,
            "public_observation_before": public_trace_observation(before),
            "public_observation_after": public_trace_observation(after),
            "public_chosen_action": {
                "source": "post_action_public_logs",
                "events": public_trace_logs(after.get("logs")),
            },
            "public_event_delta": public_trace_logs(after.get("logs")),
            "action_type": None,
            "transition_class": "unknown",
            "transition_class_reason": "no validated effect/RNG classification field",
            "result": None,
            "result_reason": None,
            "crashed": False,
            "timed_out": False,
            "illegal": False,
            "end_of_actor_turn_public_observation": None,
            "next_same_actor_public_observation": None,
            "opponent_decisions_before_return": None,
            "response_span_id": None,
        }
        if self.boundary.mode is InformationMode.PRIVILEGED_ORACLE:
            if visualizer_before is None:
                raise ValueError("oracle prospective row requires visualizer payload")
            row["actor_private_observation"] = actor_private_observation(before)
            row["true_private_state"] = merge_visualizer_private_state(
                redact_public_observation(before), visualizer_before,
            )
            row["exact_legal_actions"] = legal_action_table(before, select)
            row["chosen_indices_private"] = list(selected)
        elif self.boundary.mode is InformationMode.ACTOR_VISIBLE:
            legal_set = legal_action_table(before, select)
            chosen_action = {
                "indices": list(selected),
                "index_space": "option_position",
                "semantics": [copy.deepcopy(legal_set[index]) for index in selected],
            }
            if not selected and _as_int(select.get("minCount"), -1) == 0:
                chosen_action["allows_empty_choice"] = True
            actor_metadata = self.actors[actor]
            opponent_metadata = self.actors[opponent]
            row.update({
                "seat": actor,
                "policy_id": _policy_identity(actor_metadata.policy),
                "opponent_policy_id": _policy_identity(opponent_metadata.policy),
                "deck_id": str(actor_metadata.deck.get("deck_id") or actor_metadata.deck["deck_hash"]),
                "opponent_deck_id": str(
                    opponent_metadata.deck.get("deck_id") or opponent_metadata.deck["deck_hash"]
                ),
                "deck_hash": str(actor_metadata.deck["deck_hash"]),
                "opponent_deck_hash": str(opponent_metadata.deck["deck_hash"]),
                "training_phase": "prospective_actor_visible",
                "raw_observation": actor_private_observation(before),
                "post_observation": (
                    actor_private_observation(after)
                    if _as_int(after_current.get("yourIndex"), -1) == actor
                    else public_trace_observation(after)
                ),
                "legal_set": legal_set,
                "chosen_action": chosen_action,
                "outcome": {},
            })
        self.rows.append(row)

    def finalize(
        self,
        *,
        result: int,
        result_reason: str,
        game_completed_at_utc: str,
        crashed: bool = False,
        timed_out: bool = False,
        illegal: bool = False,
    ) -> list[dict[str, Any]]:
        for row in self.rows:
            row["game_completed_at_utc"] = game_completed_at_utc
            row["result"] = result
            row["result_reason"] = result_reason
            row["crashed"] = crashed
            row["timed_out"] = timed_out
            row["illegal"] = illegal
            if self.boundary.mode is InformationMode.ACTOR_VISIBLE:
                seat = int(row["actor_seat"])
                row["outcome"] = {
                    "won": result == seat,
                    "lost": result in (0, 1) and result != seat,
                    "draw": result not in (0, 1),
                    "result": result,
                    "crashed": crashed,
                    "timed_out": timed_out,
                    "illegal_action": illegal,
                }
        self._backfill_control_spans()
        for row in self.rows:
            if self.boundary.mode is InformationMode.PUBLIC_BELIEF:
                assert_public_trace_row_safe(row)
            elif self.boundary.mode is InformationMode.ACTOR_VISIBLE:
                assert_actor_visible_trace_row_safe(row)

        return copy.deepcopy(self.rows)

    def _backfill_control_spans(self) -> None:
        for root_index, root in enumerate(self.rows):
            root_actor = root["actor_seat"]
            pass_index = None
            for index in range(root_index, len(self.rows)):
                candidate = self.rows[index]
                if candidate["actor_seat"] != root_actor:
                    break
                if candidate["control_passed"] or candidate["terminated"]:
                    pass_index = index
                    root["end_of_actor_turn_public_observation"] = copy.deepcopy(
                        candidate["public_observation_after"]
                    )
                    break
            if pass_index is None or self.rows[pass_index]["terminated"]:
                continue
            response_span_id = f"{self.game_id}:response:d{pass_index:05d}"
            opponent_decisions = 0
            for index in range(pass_index + 1, len(self.rows)):
                candidate = self.rows[index]
                if candidate["actor_seat"] == root_actor:
                    observation = copy.deepcopy(candidate["public_observation_before"])
                    current_actor = _as_int(_as_dict(observation.get("current")).get("yourIndex"))
                    if current_actor != root_actor:
                        raise ValueError("next_same_actor observation does not preserve root actor perspective")
                    root["next_same_actor_public_observation"] = observation
                    root["opponent_decisions_before_return"] = opponent_decisions
                    root["response_span_id"] = response_span_id
                    break
                opponent_decisions += 1


def atomic_write_gzip_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
                for row in rows:
                    compressed.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return path


def scan_durable_shards(shard_dir: str | Path) -> dict[str, Any]:
    games: set[str] = set()
    decisions: set[str] = set()
    rows = 0
    for path in sorted(Path(shard_dir).glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                game_id = str(row.get("game_id") or "")
                decision_id = str(row.get("decision_id") or "")
                if not game_id or not decision_id:
                    raise ValueError(f"{path}:{line_number}: missing stable game/decision identity")
                if decision_id in decisions:
                    raise ValueError(f"duplicate durable decision_id: {decision_id}")
                games.add(game_id)
                decisions.add(decision_id)
                rows += 1

    return {
        "completed_game_ids": sorted(games),
        "completed_decision_ids": sorted(decisions),
        "games": len(games),
        "decisions": rows,
    }


def iter_trace_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl.gz"))
    for source in paths:
        opener = gzip.open if source.suffix == ".gz" else source.open
        with opener(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def is_finite_rating(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
