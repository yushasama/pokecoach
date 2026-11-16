"""Sample rule-based teachers for RL1A exact-label collection."""

from __future__ import annotations

import json
import random
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.card_function_discovery.board import read_board_metrics
from deepmew.deck_profiles.current_water_mega_abomasnow import (
    CYRANO_CARD_ID,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.policy import guard_selection, score_candidates
from deepmew.policy.selection import selection_bounds
from deepmew.rl1_teacher_bc.semantics import legal_action_table
from deepmew.rl.deck_identity import normalized_deck_hash

TEACHER_IDS = ("teacher_abomasnow", "teacher_lucario", "teacher_iono", "teacher_dragapult")

NOTEBOOK_POLICY_PATHS = {
    "teacher_abomasnow": Path("league/teachers/teacher_abomasnow.ipynb"),
    "teacher_iono": Path("league/teachers/teacher_iono.ipynb"),
    "teacher_dragapult": Path("league/teachers/teacher_dragapult.ipynb"),
    "teacher_lucario": Path("league/teachers/teacher_lucario.ipynb"),
}

NOTEBOOK_DECKLISTS = {
    "teacher_abomasnow": (
        (721, 2),
        (722, 4),
        (723, 4),
        (1121, 4),
        (1126, 1),
        (1192, 4),
        (1227, 4),
        (1262, 3),
        (3, 34),
    ),
    "teacher_iono": (
        (265, 3),
        (268, 3),
        (269, 3),
        (270, 3),
        (271, 3),
        (1086, 3),
        (1097, 2),
        (1110, 1),
        (1118, 1),
        (1121, 3),
        (1152, 2),
        (1227, 4),
        (1233, 4),
        (1254, 3),
        (4, 22),
    ),
    "teacher_dragapult": (
        (119, 4),
        (120, 4),
        (121, 3),
        (140, 1),
        (184, 1),
        (235, 2),
        (1071, 1),
        (1079, 2),
        (1080, 1),
        (1086, 4),
        (1097, 2),
        (1120, 4),
        (1121, 4),
        (1152, 3),
        (1156, 1),
        (1182, 3),
        (1198, 4),
        (1210, 2),
        (1227, 4),
        (1256, 2),
        (2, 4),
        (5, 4),
    ),
    "teacher_lucario": (
        (673, 2),
        (674, 2),
        (675, 2),
        (676, 3),
        (677, 3),
        (678, 4),
        (1102, 4),
        (1123, 2),
        (1141, 4),
        (1142, 4),
        (1152, 4),
        (1159, 1),
        (1182, 2),
        (1192, 4),
        (1227, 4),
        (1252, 2),
        (6, 13),
    ),
}

TEACHER_ARCHETYPES = {
    "teacher_abomasnow": "Mega Abomasnow ex",
    "teacher_lucario": "Mega Lucario ex",
    "teacher_iono": "Iono",
    "teacher_dragapult": "Dragapult ex",
}

LUCARIO_IDS = frozenset({333, 678, 1141, 1142})
IONO_IDS = frozenset({265, 266, 268, 269, 270, 271})
DRAGAPULT_IDS = frozenset({119, 120, 121, 112, 1079, 1198})


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _expanded_deck(pairs: tuple[tuple[int, int], ...]) -> list[int]:
    deck = [int(card_id) for card_id, count in pairs for _ in range(int(count))]
    if len(deck) != 60:
        raise ValueError(f"teacher decklists must contain 60 cards, got {len(deck)}")
    return deck


def _deck_for(teacher_id: str) -> list[int]:
    return _expanded_deck(NOTEBOOK_DECKLISTS[teacher_id])


def _load_notebook_agent_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        if "def agent" not in source:
            continue
        lines = source.splitlines(True)
        if lines and lines[0].lstrip().startswith("%%"):
            lines = lines[1:]
        return "".join(lines)
    raise ValueError(f"no notebook code cell with def agent found in {path}")


def _score_lookup(obs: dict[str, Any]) -> dict[int, int]:
    return {candidate.index: candidate.score for candidate in score_candidates(obs)}


def _base_scores(obs: dict[str, Any], table: list[dict[str, Any]]) -> dict[int, float]:
    policy_scores = _score_lookup(obs)
    return {int(row["index"]): float(policy_scores.get(int(row["index"]), 0)) for row in table}


def _selected_from_scores(select: dict[str, Any], scores: dict[int, float]) -> list[int]:
    options = _as_list(select.get("option"))
    min_count, max_count = selection_bounds(select, len(options))
    if max_count <= 0:
        return guard_selection([], select)
    ordered = sorted(range(len(options)), key=lambda index: (-scores.get(index, 0.0), index))
    if min_count == max_count:
        selected = ordered[:max_count]
    else:
        positive = [index for index in ordered if scores.get(index, 0.0) > 0.0]
        selected = positive[: max(min_count, min(max_count, len(positive)))]
    return guard_selection(selected, select)


@dataclass(frozen=True)
class SampleTeacherAgent:
    teacher_id: str
    archetype: str
    deck_id: str = ""
    label: str = ""
    teacher_source: str = "sample_rule_based_heuristic_proxy"

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.teacher_id)
        if not self.deck_id:
            object.__setattr__(self, "deck_id", f"sample_notebook_{self.teacher_id}_deck")

    @property
    def deck(self) -> list[int]:
        return _deck_for(self.teacher_id)

    @property
    def deck_hash(self) -> str:
        return normalized_deck_hash(self.deck)

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        select = _as_dict(obs.get("select"))
        if not select:
            return self.deck
        table = legal_action_table(obs, select)
        scores = _base_scores(obs, table)
        board = read_board_metrics(obs, _as_dict(obs.get("current")).get("yourIndex"))
        for row in table:
            index = int(row["index"])
            option_type = str(row.get("option_type") or "")
            source_id = row.get("source_card_id")
            target_id = row.get("target_card_id")
            semantic_key = str(row.get("semantic_key") or "")
            scores[index] += self._profile_bonus(
                option_type=option_type,
                source_id=int(source_id) if source_id is not None else None,
                target_id=int(target_id) if target_id is not None else None,
                semantic_key=semantic_key,
                board=board,
            )
        return _selected_from_scores(select, scores)

    def _profile_bonus(
        self,
        *,
        option_type: str,
        source_id: int | None,
        target_id: int | None,
        semantic_key: str,
        board: Any,
    ) -> float:
        if self.teacher_id == "teacher_abomasnow":
            return _abomasnow_bonus(option_type, source_id, target_id, semantic_key, board)
        if self.teacher_id == "teacher_lucario":
            return _family_bonus(option_type, source_id, target_id, semantic_key, LUCARIO_IDS, aggressive=True)
        if self.teacher_id == "teacher_iono":
            return _family_bonus(option_type, source_id, target_id, semantic_key, IONO_IDS, engine=True)
        if self.teacher_id == "teacher_dragapult":
            return _family_bonus(option_type, source_id, target_id, semantic_key, DRAGAPULT_IDS, setup=True)
        return 0.0


def _abomasnow_bonus(option_type: str, source_id: int | None, target_id: int | None, semantic_key: str, board: Any) -> float:
    bonus = 0.0
    if source_id == SNOVER_CARD_ID:
        bonus += 90.0 if not getattr(board, "in_play_ids", frozenset()) & {SNOVER_CARD_ID, MEGA_ABOMASNOW_CARD_ID} else 30.0
    if source_id == MEGA_ABOMASNOW_CARD_ID:
        bonus += 75.0
    if source_id == MEGA_SIGNAL_CARD_ID:
        bonus += 50.0 if SNOVER_CARD_ID not in getattr(board, "in_play_ids", frozenset()) else 20.0
    if source_id == CYRANO_CARD_ID:
        bonus += 35.0 if SNOVER_CARD_ID not in getattr(board, "in_play_ids", frozenset()) else 10.0
    if source_id == WAITRESS_CARD_ID:
        bonus += 12.0
        if getattr(board, "active_is_only_threat", False) and getattr(board, "backup_attacker_count", 0) == 0:
            bonus -= 30.0
    if source_id == LILLIES_DETERMINATION_CARD_ID:
        bonus += 20.0
    if option_type == "ATTACH" and target_id in {SNOVER_CARD_ID, MEGA_ABOMASNOW_CARD_ID, KYOGRE_CARD_ID}:
        bonus += 35.0
    if option_type == "ATTACK" and getattr(board, "bench_count", 0) == 0:
        bonus -= 20.0
    if option_type in {"PLAY", "EVOLVE"} and getattr(board, "bench_count", 0) == 0:
        bonus += 15.0
    if option_type == "END" and getattr(board, "bench_count", 0) == 0:
        bonus -= 40.0
    return bonus


def _family_bonus(
    option_type: str,
    source_id: int | None,
    target_id: int | None,
    semantic_key: str,
    family_ids: frozenset[int],
    *,
    aggressive: bool = False,
    engine: bool = False,
    setup: bool = False,
) -> float:
    bonus = 0.0
    if source_id in family_ids:
        bonus += 55.0
    if target_id in family_ids:
        bonus += 30.0
    if option_type == "ATTACK" and aggressive:
        bonus += 25.0
    if option_type in {"ABILITY", "SKILL"} and engine:
        bonus += 35.0
    if option_type in {"PLAY", "EVOLVE", "CARD"} and setup:
        bonus += 22.0
    if "ATTACH" in semantic_key and aggressive:
        bonus += 14.0
    return bonus


@dataclass
class NotebookTeacherAgent:
    teacher_id: str
    archetype: str
    notebook_path: Path
    deck: list[int]
    deck_id: str
    fallback: SampleTeacherAgent
    label: str = ""
    teacher_source: str = "sample_rule_based_notebook"

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.teacher_id
        self._module: types.ModuleType | None = None
        self._load_error: str = ""
        self._fallback_errors = 0
        self._reload()

    @property
    def notebook_available(self) -> bool:
        return self._module is not None

    @property
    def fallback_errors(self) -> int:
        return self._fallback_errors

    @property
    def deck_hash(self) -> str:
        return normalized_deck_hash(self.deck)

    def reset(self) -> None:
        self._reload()

    def _reload(self) -> None:
        try:
            source = _load_notebook_agent_source(self.notebook_path)
            module = types.ModuleType(f"deepmew_notebook_policy_{self.teacher_id}")
            module.__dict__["__file__"] = str(self.notebook_path)
            exec(compile(source, str(self.notebook_path), "exec"), module.__dict__)
            if not callable(module.__dict__.get("agent")):
                raise ValueError("notebook policy did not define callable agent")
            module.__dict__["my_deck"] = list(self.deck)
            self._module = module
            self._load_error = ""
        except Exception as exc:
            self._module = None
            self._load_error = f"{type(exc).__name__}: {exc}"

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        select = _as_dict(obs.get("select"))
        if not select:
            return list(self.deck)
        if self._module is not None:
            try:
                raw = self._module.agent(obs)  # type: ignore[attr-defined]
                selected = [int(index) for index in raw] if isinstance(raw, (list, tuple)) else []
                return guard_selection(selected, select)
            except Exception:
                self._fallback_errors += 1
        return guard_selection(self.fallback.select(obs, rng), select)


def make_teacher(teacher_id: str) -> SampleTeacherAgent | NotebookTeacherAgent:
    if teacher_id not in TEACHER_IDS:
        raise ValueError(f"unknown teacher_id {teacher_id!r}; known: {list(TEACHER_IDS)}")
    archetype = TEACHER_ARCHETYPES[teacher_id]
    fallback = SampleTeacherAgent(teacher_id, archetype)
    notebook_path = NOTEBOOK_POLICY_PATHS[teacher_id]
    if not notebook_path.exists():
        return fallback
    return NotebookTeacherAgent(
        teacher_id=teacher_id,
        archetype=archetype,
        notebook_path=notebook_path,
        deck=_deck_for(teacher_id),
        deck_id=f"sample_notebook_{teacher_id}_deck",
        fallback=fallback,
    )
