"""R15: regenerate reference-teacher games against HARD opposition (offline data only).

The original refgames factory (`run_unified_r2_refgames.py`) is lost; this rebuilds it
against the surviving consumer contract (`run_unified_r4_ablation._iter_rows` +
`build_decision_features` + `run_r9_r4f_retrain.encode_refgame_row_r9`), with one
deliberate change: instead of the weak default `v2.4` opponent, each reference teacher
plays the OTHER reference teachers (cross-play). Teachers stay frozen scripted policies;
this is harder-distribution imitation data, not a self-play-to-equilibrium loop
(INVARIANT #2 respected).

Output layout matches regen3: `<out>/shards/decisions_shard_XXX.jsonl.gz` + manifest;
train/val split stays `game_index % 5 == 0 -> val`.

Example
-------
    uv run python scripts/run_r15_hard_refgames.py \
        --out-dir D:/DeepMew_artifacts/datasets/r15/hard_refgames_crossplay \
        --games-per-cell 170
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
import main  # noqa: E402
import run_unified_r4_closeout_delta as closeout  # noqa: E402
from deepmew.rl.observation import ObservationEncoder  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl.opponents import OpponentAgent  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402


SCHEMA_VERSION = "deepmew.r15.hard_refgames.v0"
ROW_SCHEMA_VERSION = "deepmew.refgame_decision_row.rebuilt.v1"
TEACHERS = ("reference_abomasnow", "reference_iono", "reference_dragapult", "reference_lucario")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _own_hand_ids(obs: dict[str, Any], seat: int) -> list[int]:
    players = _as_list(_as_dict(obs.get("current")).get("players"))
    if not (0 <= seat < len(players)):
        return []
    hand = _as_dict(players[seat]).get("hand") if isinstance(players[seat], dict) else None
    ids = []
    for card in hand or []:
        card_id = card.get("id") if isinstance(card, dict) else None
        if isinstance(card_id, int):
            ids.append(card_id)

    return ids


def play_teacher_game(
    teacher: OpponentAgent,
    opponent: OpponentAgent,
    *,
    teacher_seat: int,
    game_id: str,
    game_index: int,
    seed: int,
    max_steps: int,
    encoder: ObservationEncoder,
    raise_on_error: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    safety = {"illegal": 0, "crashes": 0, "timeouts": 0, "completed": 0, "teacher_wins": 0}
    rows: list[dict[str, Any]] = []
    obs = None
    try:
        teacher_deck = list(teacher.deck or main.read_deck_csv())
        opp_deck = list(opponent.deck or main.read_deck_csv())
        deck0 = teacher_deck if teacher_seat == 0 else opp_deck
        deck1 = opp_deck if teacher_seat == 0 else teacher_deck
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(f"BattleStart failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}")
        decision_index = 0
        result = -1
        for _ in range(max_steps):
            current = _as_dict(obs.get("current"))
            result = _as_int(current.get("result"), -1)
            if result != -1:
                break
            actor_is_teacher = _as_int(current.get("yourIndex")) == teacher_seat
            actor = teacher if actor_is_teacher else opponent
            selected = closeout._select_from_policy(actor, obs)
            select = _as_dict(obs.get("select"))
            legal_count = len(_as_list(select.get("option")))
            if any(index < 0 or index >= legal_count for index in selected):
                safety["illegal"] += 1

                return [], safety
            if actor_is_teacher and legal_count:
                legal_set = legal_action_table(obs, select)
                rows.append(
                    {
                        "schema_version": ROW_SCHEMA_VERSION,
                        "game_id": game_id,
                        "game_index": game_index,
                        "decision_index": decision_index,
                        "seed": seed,
                        "seat": teacher_seat,
                        "turn": _as_int(current.get("turn")),
                        "policy_id": teacher.label,
                        "deck_id": teacher.label,
                        "deck_hash": getattr(teacher.deck_identity, "deck_hash", None) or normalized_deck_hash(teacher_deck),
                        "opponent_policy_id": opponent.label,
                        "opponent_deck_id": opponent.label,
                        "opponent_deck_hash": getattr(opponent.deck_identity, "deck_hash", None) or normalized_deck_hash(opp_deck),
                        "raw_observation": obs,
                        "base84": encoder.encode(obs),
                        "own_hand": _own_hand_ids(obs, teacher_seat),
                        "legal_set": legal_set,
                        "action_mask": {
                            "n_options": legal_count,
                            "min_count": _as_int(select.get("minCount")),
                            "max_count": _as_int(select.get("maxCount"), legal_count),
                        },
                        "chosen_action": {
                            "indices": list(selected),
                            "semantics": [legal_set[index] for index in selected if 0 <= index < len(legal_set)],
                        },
                        "outcome": {},
                    }
                )
                decision_index += 1
            obs = battle_select(selected)
        else:
            safety["timeouts"] += 1

            return [], safety
        teacher_won = result == teacher_seat
        draw = result not in (0, 1)
        safety["completed"] += 1
        safety["teacher_wins"] += int(teacher_won)
        outcome = {
            "won": teacher_won,
            "lost": (not teacher_won) and not draw,
            "draw": draw,
            "result": result,
            "crashed": False,
            "timed_out": False,
        }
        for row in rows:
            row["outcome"] = dict(outcome)

        return rows, safety
    except Exception:
        if raise_on_error:
            raise
        safety["crashes"] += 1

        return [], safety
    finally:
        if obs is not None:
            battle_finish()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    encoder = ObservationEncoder()
    teachers = [OpponentAgent.from_str(name) for name in TEACHERS]
    started = time.perf_counter()
    safety_total = Counter()
    rows_by_teacher: Counter[str] = Counter()
    game_index = 0
    row_count = 0
    shard_index = 0
    shard_rows = 0
    shard_handle = gzip.open(shards_dir / f"decisions_shard_{shard_index:03d}.jsonl.gz", "wt", encoding="utf-8")

    def _rotate() -> None:
        nonlocal shard_index, shard_rows, shard_handle
        shard_handle.close()
        shard_index += 1
        shard_rows = 0
        shard_handle = gzip.open(shards_dir / f"decisions_shard_{shard_index:03d}.jsonl.gz", "wt", encoding="utf-8")

    try:
        for teacher in teachers:
            for opponent in teachers:
                if opponent.label == teacher.label:
                    continue
                for game_in_cell in range(int(args.games_per_cell)):
                    seed = int(args.base_seed) + game_index
                    teacher.reset(seed)
                    opponent.reset(seed)
                    game_id = f"{args.game_id_prefix}-{game_index:06d}"
                    rows, safety = play_teacher_game(
                        teacher,
                        opponent,
                        teacher_seat=game_in_cell % 2,
                        game_id=game_id,
                        game_index=game_index,
                        seed=seed,
                        max_steps=int(args.max_steps),
                        encoder=encoder,
                        raise_on_error=bool(args.raise_on_error),
                    )
                    safety_total.update(safety)
                    for row in rows:
                        shard_handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
                        shard_rows += 1
                        row_count += 1
                        if shard_rows >= int(args.shard_rows):
                            _rotate()
                    rows_by_teacher[teacher.label] += len(rows)
                    game_index += 1
                    if game_index % 200 == 0:
                        print(f"games={game_index} rows={row_count} safety={dict(safety_total)}", flush=True)
    finally:
        shard_handle.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "base_seed": int(args.base_seed),
        "games_per_cell": int(args.games_per_cell),
        "teachers": list(TEACHERS),
        "opponents": "reference cross-play (each teacher vs the other three)",
        "hard_arena_note": "opponent pool is the reference teachers themselves, not default v2.4",
        "games_played": game_index,
        "rows": row_count,
        "rows_by_teacher": dict(rows_by_teacher),
        "safety": dict(safety_total),
        "split_rule": "game_index % 5 == 0 -> val",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("games_played", "rows", "rows_by_teacher", "safety", "elapsed_seconds")}, indent=2))

    return manifest


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reference-teacher decision rows against hard cross-play opposition.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games-per-cell", type=int, default=170)
    parser.add_argument("--base-seed", type=int, default=915000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--shard-rows", type=int, default=50000)
    parser.add_argument("--game-id-prefix", default="r15-hard-ref")
    parser.add_argument("--raise-on-error", action="store_true")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
