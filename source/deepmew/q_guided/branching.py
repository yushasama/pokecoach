"""Pure safety utilities for privileged-teacher sibling branching.

This module does not import CABT and never accepts a live simulator state.  The
isolated simulator layer owns exact hidden state and exposes only a zero-argument
``begin_search`` closure to :func:`execute_privileged_teacher`.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PRIVATE_CONTENT_SENTINEL = "__DEEPMEW_PRIVILEGED_HIDDEN_SENTINEL__"

_PLAYER_DECK_FIELDS = frozenset({"deck", "deckOrder", "trueDeckOrder"})
_DEFAULT_FORBIDDEN_KEYS = frozenset({
    "search_begin_input",
    "search_begin_args",
    "search_begin_payload",
    "raw_cabt_hidden_zone_payload",
    "raw_hidden_state",
    "raw_hidden_zones",
    "hidden_state",
    "hidden_zones",
    "hidden_state_hash",
    "exact_state_hash",
    "private_state_hash",
    "private_state_tensor",
    "private_tensor",
    "opponent_hidden_hand",
    "opponent_hidden_deck",
    "opponent_hidden_prize",
    "opponent_deck",
    "opponent_prize",
    "oracle_legal_options",
    "teacher_payload",
    "privileged_teacher_payload",
})
_DEFAULT_FORBIDDEN_VALUES = (PRIVATE_CONTENT_SENTINEL, "search_begin_input")


class LegalSelectionCapExceeded(ValueError):
    """Raised before enumeration when a root exceeds its complete-action cap."""

    def __init__(self, selection_count: int, root_cap: int) -> None:
        self.selection_count = selection_count
        self.root_cap = root_cap
        super().__init__(
            f"complete legal-selection count {selection_count} exceeds root cap {root_cap}"
        )


class PrivilegedTeacherExecutionError(RuntimeError):
    """Fail-closed error whose message cannot contain private engine payloads."""


@dataclass(frozen=True)
class PrivilegedTeacherReceipt:
    """Persistence-safe execution facts; every value is a boolean or a count."""

    actor_observation_valid: bool
    actor_visible_root_unchanged: bool
    aggregate_private_content_absent: bool
    executor_completed: bool
    player_count: int
    legal_option_count: int
    student_input_byte_count: int
    begin_search_call_count: int


@dataclass(frozen=True)
class RootConsistencyReceipt:
    """Actor-visible root comparison without a reconstructive state hash."""

    actor_visible_root_matches: bool
    legal_options_match: bool
    player_count_matches: bool
    expected_byte_count: int
    reconstructed_byte_count: int
    expected_legal_option_count: int
    reconstructed_legal_option_count: int
    expected_player_count: int
    reconstructed_player_count: int


def _selection_bounds(select: Mapping[str, Any]) -> tuple[int, int, int]:
    options = select.get("option")
    if options is None:
        options = []
    if not isinstance(options, list):
        raise TypeError("select.option must be a list")

    option_count = len(options)
    raw_minimum = select.get("minCount")
    raw_maximum = select.get("maxCount")
    minimum = 0 if raw_minimum is None else _require_count(raw_minimum, "minCount")
    maximum = option_count if raw_maximum is None else _require_count(raw_maximum, "maxCount")
    minimum = min(minimum, option_count)
    maximum = max(minimum, min(maximum, option_count))

    return option_count, minimum, maximum


def _require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

    return value


def count_legal_selections(
    select: Mapping[str, Any],
    *,
    order_sensitive: bool,
) -> int:
    """Count every legal selection from ``minCount`` through ``maxCount``."""
    if not isinstance(order_sensitive, bool):
        raise TypeError("order_sensitive must be a boolean")
    option_count, minimum, maximum = _selection_bounds(select)
    count_fn = math.perm if order_sensitive else math.comb

    return sum(count_fn(option_count, count) for count in range(minimum, maximum + 1))


def enumerate_legal_selections(
    select: Mapping[str, Any],
    *,
    order_sensitive: bool,
    root_cap: int | None = None,
) -> list[list[int]]:
    """Enumerate a complete root or reject it before constructing any subset."""
    selection_count = count_legal_selections(select, order_sensitive=order_sensitive)
    if root_cap is not None:
        cap = _require_count(root_cap, "root_cap")
        if selection_count > cap:
            raise LegalSelectionCapExceeded(selection_count, cap)

    option_count, minimum, maximum = _selection_bounds(select)
    iterator = itertools.permutations if order_sensitive else itertools.combinations
    selections: list[list[int]] = []
    for count in range(minimum, maximum + 1):
        selections.extend(
            list(indices)
            for indices in iterator(range(option_count), count)
        )

    return selections


def sample_uniform_legal_selection(
    select: Mapping[str, Any],
    *,
    order_sensitive: bool,
    rng: Any,
    root_cap: int | None = None,
) -> list[int]:
    """Sample uniformly over complete legal selections using only ``rng``."""
    if not hasattr(rng, "randrange"):
        raise TypeError("rng must provide randrange")
    selections = enumerate_legal_selections(
        select,
        order_sensitive=order_sensitive,
        root_cap=root_cap,
    )
    if not selections:
        raise ValueError("root has no legal selections")

    return list(selections[rng.randrange(len(selections))])


def _validate_base_selection(
    selection: Sequence[int],
    select: Mapping[str, Any],
    *,
    order_sensitive: bool,
) -> list[int]:
    if isinstance(selection, (str, bytes)) or not isinstance(selection, Sequence):
        raise TypeError("base_selector must return a sequence of indices")
    selected = list(selection)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise TypeError("base_selector indices must be integers")
    option_count, minimum, maximum = _selection_bounds(select)
    if not minimum <= len(selected) <= maximum:
        raise ValueError("base_selector returned an invalid selection count")
    if len(set(selected)) != len(selected):
        raise ValueError("base_selector returned duplicate indices")
    if any(index < 0 or index >= option_count for index in selected):
        raise ValueError("base_selector returned an out-of-range index")
    if not order_sensitive:
        selected.sort()

    return selected


def select_epsilon_mixture(
    select: Mapping[str, Any],
    *,
    order_sensitive: bool,
    rng: Any,
    epsilon: float,
    base_selector: Callable[[Mapping[str, Any], Any], Sequence[int]],
    root_cap: int | None = None,
) -> list[int]:
    """Mix a passed base selector with uniform legal-root exploration."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be numeric")
    probability = float(epsilon)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not callable(base_selector):
        raise TypeError("base_selector must be callable")

    if probability == 1.0:
        explore = True
    elif probability == 0.0:
        explore = False
    else:
        if not hasattr(rng, "random"):
            raise TypeError("rng must provide random")
        explore = rng.random() < probability
    if explore:
        return sample_uniform_legal_selection(
            select,
            order_sensitive=order_sensitive,
            rng=rng,
            root_cap=root_cap,
        )

    return _validate_base_selection(
        base_selector(select, rng),
        select,
        order_sensitive=order_sensitive,
    )


def _normalize_key(key: str) -> str:
    return key.strip().casefold().replace("-", "_").replace(" ", "_")


def _matches_forbidden_value(value: Any, forbidden_values: Sequence[Any]) -> bool:
    if isinstance(value, (dict, list)):
        return False
    for forbidden in forbidden_values:
        if type(value) is type(forbidden) and value == forbidden:
            return True

    return False


def _observation_private_paths(value: Mapping[str, Any], path: str) -> list[str]:
    current = value.get("current")
    if not isinstance(current, dict) or "players" not in current:
        return []
    players = current.get("players")
    if not isinstance(players, list):
        return [f"{path}.current.players:invalid_observation_players"]
    actor = current.get("yourIndex")
    if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor < len(players):
        return [f"{path}.current.yourIndex:unproven_actor_visibility"]

    violations: list[str] = []
    for index, player in enumerate(players):
        player_path = f"{path}.current.players[{index}]"
        if not isinstance(player, dict):
            violations.append(f"{player_path}:invalid_observation_player")
            continue
        for field in _PLAYER_DECK_FIELDS:
            if field in player and player[field] is not None:
                violations.append(f"{player_path}.{field}:private_deck_order")
        prize = player.get("prize")
        if prize is not None:
            if not isinstance(prize, list):
                violations.append(f"{player_path}.prize:unproven_prize_visibility")
            elif any(card is not None for card in prize):
                violations.append(f"{player_path}.prize:private_prize_identity")
        if index != actor and player.get("hand") is not None:
            violations.append(f"{player_path}.hand:private_opponent_hand")

    return violations


def find_private_content(
    artifact: Any,
    *,
    forbidden_keys: Collection[str] | None = None,
    forbidden_values: Sequence[Any] = (),
) -> list[str]:
    """Return value-free violation paths from a recursively scanned artifact.

    The scan is observation-aware: the actor's visible hand is allowed, while
    opponent hand contents, player deck order, and prize identities fail closed.
    Returned diagnostics never interpolate artifact values.
    """
    keys = set(_DEFAULT_FORBIDDEN_KEYS)
    if forbidden_keys is not None:
        keys.update(forbidden_keys)
    normalized_keys = {_normalize_key(key) for key in keys}
    values = (*_DEFAULT_FORBIDDEN_VALUES, *tuple(forbidden_values))
    violations: list[str] = []

    def visit(value: Any, path: str) -> None:
        if _matches_forbidden_value(value, values):
            violations.append(f"{path}:forbidden_value")
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                violations.append(f"{path}:non_json_number")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            violations.extend(_observation_private_paths(value, path))
            for key, item in value.items():
                if not isinstance(key, str):
                    violations.append(f"{path}.<key>:non_json_key")
                    visit(item, f"{path}.<non_string_key>")
                    continue
                child_path = f"{path}.{key}"
                if _normalize_key(key) in normalized_keys:
                    violations.append(f"{child_path}:forbidden_key")
                visit(item, child_path)
            return
        violations.append(f"{path}:non_json_type")

    visit(artifact, "$")

    return list(dict.fromkeys(violations))


def assert_private_content_absent(
    artifact: Any,
    *,
    forbidden_keys: Collection[str] | None = None,
    forbidden_values: Sequence[Any] = (),
) -> None:
    violations = find_private_content(
        artifact,
        forbidden_keys=forbidden_keys,
        forbidden_values=forbidden_values,
    )
    if violations:
        raise ValueError("private-content scan failed at " + ", ".join(violations))


def _actor_index(current: Mapping[str, Any], player_count: int) -> int:
    actor = current.get("yourIndex")
    if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor < player_count:
        raise ValueError("actor-visible observation has an invalid yourIndex")

    return actor


def project_student_input(
    actor_observation: Mapping[str, Any],
    *,
    teacher_payload: Any = None,
) -> dict[str, Any]:
    """Project only actor-visible input; ``teacher_payload`` is never inspected."""
    del teacher_payload
    if not isinstance(actor_observation, Mapping):
        raise TypeError("actor_observation must be a mapping")
    projected = copy.deepcopy(dict(actor_observation))
    projected.pop("search_begin_input", None)
    current = projected.get("current")
    if not isinstance(current, dict):
        raise ValueError("actor-visible observation must contain current")
    players = current.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("actor-visible observation must contain players")
    actor = _actor_index(current, len(players))
    for index, player in enumerate(players):
        if not isinstance(player, dict):
            raise ValueError("actor-visible player must be an object")
        for field in _PLAYER_DECK_FIELDS:
            if player.get(field) is not None:
                raise ValueError("student projection rejected a private deck ordering")
            player.pop(field, None)
        prize = player.get("prize")
        if prize is not None:
            if not isinstance(prize, list) or any(card is not None for card in prize):
                raise ValueError("student projection rejected a private prize identity")
        if index != actor:
            if player.get("hand") is not None:
                raise ValueError("student projection rejected an opponent hand identity")
            player["hand"] = None

    assert_private_content_absent(projected)

    return projected


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not a finite JSON-safe artifact") from exc

    return encoded.encode("utf-8")


def student_input_bytes(
    actor_observation: Mapping[str, Any],
    *,
    teacher_payload: Any = None,
) -> bytes:
    projected = project_student_input(
        actor_observation,
        teacher_payload=teacher_payload,
    )

    return _canonical_json_bytes(projected)


def _without_transient_fields(projected: dict[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(projected)
    comparable.pop("logs", None)

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if item is not None and key != "playerIndex"
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]

        return value

    return normalize(comparable)


def _root_counts(projected: Mapping[str, Any]) -> tuple[int, int]:
    current = projected.get("current")
    players = current.get("players") if isinstance(current, dict) else None
    player_count = len(players) if isinstance(players, list) else 0
    select = projected.get("select")
    options = select.get("option") if isinstance(select, dict) else None
    legal_option_count = len(options) if isinstance(options, list) else 0

    return player_count, legal_option_count


def build_root_consistency_receipt(
    expected_actor_root: Mapping[str, Any],
    reconstructed_actor_root: Mapping[str, Any],
) -> RootConsistencyReceipt:
    expected = _without_transient_fields(project_student_input(expected_actor_root))
    reconstructed = _without_transient_fields(project_student_input(reconstructed_actor_root))
    expected_bytes = _canonical_json_bytes(expected)
    reconstructed_bytes = _canonical_json_bytes(reconstructed)
    expected_players, expected_options = _root_counts(expected)
    reconstructed_players, reconstructed_options = _root_counts(reconstructed)

    return RootConsistencyReceipt(
        actor_visible_root_matches=expected_bytes == reconstructed_bytes,
        legal_options_match=expected.get("select") == reconstructed.get("select"),
        player_count_matches=expected_players == reconstructed_players,
        expected_byte_count=len(expected_bytes),
        reconstructed_byte_count=len(reconstructed_bytes),
        expected_legal_option_count=expected_options,
        reconstructed_legal_option_count=reconstructed_options,
        expected_player_count=expected_players,
        reconstructed_player_count=reconstructed_players,
    )


def execute_privileged_teacher(
    actor_observation: Mapping[str, Any],
    *,
    begin_search: Callable[[], Any],
    executor: Callable[[Callable[[], Any]], Any],
) -> tuple[Any, PrivilegedTeacherReceipt]:
    """Run aggregate-only teacher work around an opaque SearchBegin closure.

    Neither this function nor ``executor`` receives hidden-zone inputs.  The
    isolated simulator-owned ``begin_search`` closure is the only component
    permitted to capture them.  Its returned search state remains callback-local.
    """
    if not callable(begin_search):
        raise TypeError("begin_search must be callable")
    if not callable(executor):
        raise TypeError("executor must be callable")
    before = student_input_bytes(actor_observation)
    projected = project_student_input(actor_observation)
    player_count, legal_option_count = _root_counts(projected)
    call_count = 0
    is_active = True

    def guarded_begin_search() -> Any:
        nonlocal call_count
        if not is_active:
            raise PrivilegedTeacherExecutionError("privileged SearchBegin closure expired")
        call_count += 1
        try:
            state = begin_search()
        except Exception:
            raise PrivilegedTeacherExecutionError("privileged SearchBegin failed") from None
        if state is None:
            raise PrivilegedTeacherExecutionError("privileged SearchBegin returned no state")

        return state

    try:
        aggregate = executor(guarded_begin_search)
    except PrivilegedTeacherExecutionError:
        raise
    except Exception:
        raise PrivilegedTeacherExecutionError("privileged teacher executor failed") from None
    finally:
        is_active = False
    if call_count == 0:
        raise PrivilegedTeacherExecutionError("privileged teacher did not begin a search")
    after = student_input_bytes(actor_observation)
    if after != before:
        raise PrivilegedTeacherExecutionError("actor-visible root changed during teacher execution")

    assert_private_content_absent(aggregate)
    aggregate_bytes = _canonical_json_bytes(aggregate)
    safe_aggregate = json.loads(aggregate_bytes.decode("utf-8"))
    assert_private_content_absent(safe_aggregate)
    receipt = PrivilegedTeacherReceipt(
        actor_observation_valid=True,
        actor_visible_root_unchanged=True,
        aggregate_private_content_absent=True,
        executor_completed=True,
        player_count=player_count,
        legal_option_count=legal_option_count,
        student_input_byte_count=len(before),
        begin_search_call_count=call_count,
    )

    return safe_aggregate, receipt
