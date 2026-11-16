"""Reporting-only diagnostics over generated refgame rows."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

UNAVAILABLE = "unavailable"

GENERAL_DIAGNOSTICS = [
    "first_meaningful_attack_turn",
    "no_attack_turn_count",
    "attack_available_but_not_used",
    "missed_lethal",
    "missed_gust_lethal",
    "one_attacker_board_after_turn_2",
    "energy_starved_turn",
    "hand_starved_turn",
    "setup_online_turn",
    "next_attacker_ready",
    "disruption_card_played",
    "disruption_timing_quality",
    "boss_on_setup_target",
    "special_energy_removed",
    "tool_removed",
]

STARMIE_DIAGNOSTICS = [
    "cinderace_started_active",
    "cinderace_used_turbo_flare",
    "turbo_flare_attached_to_future_attacker",
    "staryu_benched_by_turn_1",
    "staryu_count_turn_2",
    "mega_starmie_online_turn",
    "starmie_followup_attacker_ready",
    "energy_stranded_on_cinderace",
    "wally_caused_tempo_loss",
    "boss_used_on_evolving_basic",
]

GRIMMSNARL_DIAGNOSTICS = [
    "impidimp_benched_turn_1",
    "morgrem_online_turn",
    "grimmsnarl_online_turn",
    "rare_candy_success",
    "punk_up_used",
    "punk_up_energy_count",
    "dark_energy_available_for_punk_up",
    "munkidori_online",
    "froslass_online",
    "xerosic_before_enemy_swing_turn",
    "unfair_stamp_before_enemy_swing_turn",
]

ALL_DIAGNOSTICS = GENERAL_DIAGNOSTICS + STARMIE_DIAGNOSTICS + GRIMMSNARL_DIAGNOSTICS


def iter_refgame_rows(refgames_dir: Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for shard in sorted((Path(refgames_dir) / "shards").glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _turn(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("turn"))
    except (TypeError, ValueError):
        return None


def _is_attack_semantic(option: dict[str, Any]) -> bool:
    option_type = str(option.get("option_type") or "").upper()
    select_context = str(option.get("select_context") or "").upper()
    semantic_key = str(option.get("semantic_key") or "").upper()

    return option_type == "ATTACK" or select_context == "ATTACK" or semantic_key.startswith("ATTACK")


def _chosen_attack(row: dict[str, Any]) -> bool:
    chosen = _as_dict(row.get("chosen_action"))

    return any(_is_attack_semantic(s) for s in _as_list(chosen.get("semantics")) if isinstance(s, dict))


def _legal_attack_available(row: dict[str, Any]) -> bool:
    return any(_is_attack_semantic(s) for s in _as_list(row.get("legal_set")) if isinstance(s, dict))


def _own_player(row: dict[str, Any]) -> dict[str, Any] | None:
    obs = _as_dict(row.get("raw_observation"))
    current = _as_dict(obs.get("current"))
    players = _as_list(current.get("players"))
    try:
        seat = int(row.get("seat", current.get("yourIndex", 0)))
    except (TypeError, ValueError):
        seat = 0
    if 0 <= seat < len(players) and isinstance(players[seat], dict):
        return players[seat]

    return None


def _one_attacker_board(row: dict[str, Any]) -> bool | None:
    player = _own_player(row)
    if player is None:
        return None
    active = _as_dict(player.get("active"))
    bench = [p for p in _as_list(player.get("bench")) if isinstance(p, dict)]
    if not active:
        return None

    return len(bench) == 0


def analyze_refgame_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    metadata_counts: dict[str, Counter[str]] = {
        "training_phase": Counter(),
        "opponent_bucket": Counter(),
        "opponent_shell": Counter(),
        "opponent_pilot": Counter(),
    }
    for row in rows:
        row_count += 1
        by_game[str(row.get("game_id") or f"row-{row_count}")].append(row)
        for field, counter in metadata_counts.items():
            counter[str(row.get(field) or "unknown")] += 1

    game_rows: list[dict[str, Any]] = []
    for game_id, game in sorted(by_game.items()):
        summary = {key: UNAVAILABLE for key in ALL_DIAGNOSTICS}
        turns = [t for t in (_turn(row) for row in game) if t is not None]
        attack_turns = [_turn(row) for row in game if _chosen_attack(row) and _turn(row) is not None]
        attack_misses = sum(1 for row in game if _legal_attack_available(row) and not _chosen_attack(row))
        one_attacker_values = [
            _one_attacker_board(row)
            for row in game
            if (_turn(row) is not None and int(_turn(row) or 0) >= 2)
        ]
        one_attacker_known = [v for v in one_attacker_values if v is not None]
        summary["first_meaningful_attack_turn"] = min(attack_turns) if attack_turns else UNAVAILABLE
        summary["no_attack_turn_count"] = attack_misses
        summary["attack_available_but_not_used"] = attack_misses
        if one_attacker_known:
            summary["one_attacker_board_after_turn_2"] = any(one_attacker_known)
        game_rows.append({
            "game_id": game_id,
            "rows": len(game),
            "first_turn": min(turns) if turns else UNAVAILABLE,
            "last_turn": max(turns) if turns else UNAVAILABLE,
            "outcome": _outcome_label(game[-1].get("outcome") or {}),
            "training_phase": str(game[0].get("training_phase") or "unknown"),
            "opponent_bucket": str(game[0].get("opponent_bucket") or "unknown"),
            "opponent_shell": str(game[0].get("opponent_shell") or "unknown"),
            "opponent_pilot": str(game[0].get("opponent_pilot") or "unknown"),
            **summary,
        })

    available_counts = Counter()
    for row in game_rows:
        for key in ALL_DIAGNOSTICS:
            if row.get(key) != UNAVAILABLE:
                available_counts[key] += 1

    return {
        "schema_version": "deepmew.refgame_diagnostics.v0",
        "rows": row_count,
        "games": len(by_game),
        "metadata_counts": {field: dict(counter) for field, counter in metadata_counts.items()},
        "available_counts": dict(available_counts),
        "game_rows": game_rows,
        "unavailable_fields": [key for key in ALL_DIAGNOSTICS if available_counts.get(key, 0) == 0],
    }


def _outcome_label(outcome: dict[str, Any]) -> str:
    if outcome.get("won"):
        return "win"
    if outcome.get("draw"):
        return "draw"
    if outcome:
        return "loss"

    return "unknown"


def write_diagnostics_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnostics.json"
    csv_path = out_dir / "diagnostics.csv"
    md_path = out_dir / "diagnostics.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = ["game_id", "rows", "first_turn", "last_turn", "outcome",
                  "training_phase", "opponent_bucket", "opponent_shell", "opponent_pilot", *ALL_DIAGNOSTICS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.get("game_rows") or []:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    lines = [
        "# Refgame diagnostics",
        "",
        f"- rows: `{result.get('rows', 0)}`",
        f"- games: `{result.get('games', 0)}`",
        "",
        "## Available Fields",
        "",
    ]
    available = result.get("available_counts") or {}
    for key in ALL_DIAGNOSTICS:
        if available.get(key, 0):
            lines.append(f"- `{key}`: {available[key]} games")
    if not available:
        lines.append("- None")
    lines.extend(["", "## Unavailable Fields", ""])
    for key in result.get("unavailable_fields") or []:
        lines.append(f"- `{key}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
