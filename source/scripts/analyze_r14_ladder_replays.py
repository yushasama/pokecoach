"""Analyze downloaded ladder replays for DeepMew's own submissions (R14 Lane 1).

Reads replay JSONs plus the scout manifest that maps episodes to submission ids,
identifies DeepMew's seat by team name, and emits a per-episode CSV plus an
aggregate what-went-right/what-went-wrong report per submission slot.

Example
-------
    uv run python scripts/analyze_r14_ladder_replays.py \
        --replay-dir .research/kaggle_replays/r14-our-slots \
        --scout-manifest .research/kaggle_scout/20260706/20260706T201845Z-kaggle_scout-r14-our-slots/manifest.json \
        --out-dir .research/r14/lane1_replay_taxonomy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_r8_greedyv_probes import ProbeSummary  # noqa: E402
from deepmew.kaggle_replays.decks import extract_decklists  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402


SCHEMA_VERSION = "deepmew.r14.ladder_replay_taxonomy.v0"
DEFAULT_CARD_CSV = ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
SLOT_LABELS = {
    "54404662": {"label": "R4-F reference Abomasnow", "model": "R4-F", "deck": "reference_abomasnow"},
    "54404666": {"label": "v1b reference Abomasnow", "model": "v1b", "deck": "reference_abomasnow"},
    "54404672": {"label": "R4-F Starmie 18ffc1e6", "model": "R4-F", "deck": "starmie_18ffc1e6"},
    "54404676": {"label": "R4-F bd27 Mega Lucario", "model": "R4-F", "deck": "bd27_mega_lucario"},
    "54318639": {"label": "baseline distilled R4-A Abomasnow", "model": "R4-A_distilled", "deck": "reference_abomasnow"},
}


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


def load_card_names(card_csv: Path) -> dict[int, dict[str, Any]]:
    cards: dict[int, dict[str, Any]] = {}
    with card_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            card_id = _as_int(row.get("Card ID"), -1)
            if card_id < 0 or card_id in cards:
                continue
            hp = _as_int(row.get("HP"), 0)
            cards[card_id] = {"name": str(row.get("Card Name") or ""), "is_pokemon": hp > 0, "hp": hp}

    return cards


def episode_submission_map(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for record in _as_list(manifest.get("submission_episode_records")):
        submission_id = str(_as_dict(record).get("submission_id") or "")
        for episode_id in _as_list(_as_dict(record).get("episode_ids")):
            mapping[str(episode_id)] = submission_id

    return mapping


def shell_label(card_ids: list[int], cards: dict[int, dict[str, Any]]) -> str:
    counts = Counter(card_id for card_id in card_ids if cards.get(card_id, {}).get("is_pokemon"))
    if not counts:
        return "Unknown"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -cards[item[0]]["hp"]))
    names: list[str] = []
    for card_id, _count in ranked:
        name = cards[card_id]["name"]
        base = name.split(" ex")[0].strip()
        if base not in names:
            names.append(base)
        if len(names) == 2:
            break

    return "/".join(names)


def _player_state(obs: dict[str, Any], player_index: int) -> dict[str, Any]:
    players = _as_list(_as_dict(obs.get("current")).get("players"))
    if 0 <= player_index < len(players):
        return _as_dict(players[player_index])

    return {}


def _prizes_remaining(obs: dict[str, Any], player_index: int) -> int | None:
    player = _player_state(obs, player_index)
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)

    return None


def analyze_episode(
    path: Path,
    *,
    team: str,
    submission_id: str,
    cards: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    info = _as_dict(data.get("info"))
    team_names = [str(name) for name in _as_list(info.get("TeamNames"))]
    if team not in team_names:
        return None
    our_seat = team_names.index(team)
    opp_seat = 1 - our_seat
    rewards = _as_list(data.get("rewards"))
    our_reward = rewards[our_seat] if len(rewards) == 2 else None
    result = "draw"
    if isinstance(our_reward, (int, float)):
        result = "win" if our_reward > 0 else ("loss" if our_reward < 0 else "draw")

    decks = extract_decklists(data)
    our_cards = list(decks[our_seat].card_ids) if len(decks) == 2 else []
    opp_cards = list(decks[opp_seat].card_ids) if len(decks) == 2 else []

    probe = ProbeSummary()
    probe.start_game()
    decisions = 0
    attacks = 0
    end_choices = 0
    last_obs: dict[str, Any] | None = None
    final_overage: Any = None
    max_turn = 0
    for step in _as_list(data.get("steps")):
        agents = _as_list(step)
        if len(agents) != 2:
            continue
        agent = _as_dict(agents[our_seat])
        obs = _as_dict(agent.get("observation"))
        select = _as_dict(obs.get("select"))
        action = agent.get("action")
        max_turn = max(max_turn, _as_int(_as_dict(obs.get("current")).get("turn")))
        if obs:
            last_obs = obs
            if obs.get("remainingOverageTime") is not None:
                final_overage = obs.get("remainingOverageTime")
        if not (isinstance(action, list) and action and _as_list(select.get("option"))):
            continue
        selected = [index for index in action if isinstance(index, int)]
        decisions += 1
        probe.observe_policy_action(obs, selected)
        table = legal_action_table(obs, select)
        for index in selected:
            if 0 <= index < len(table):
                option_type = str(table[index].get("option_type") or "")
                attacks += int(option_type == "ATTACK")
                end_choices += int(option_type == "END")

    own_prizes = _prizes_remaining(last_obs or {}, our_seat)
    opp_prizes = _prizes_remaining(last_obs or {}, opp_seat)
    own_final = _player_state(last_obs or {}, our_seat)
    active_list = [entry for entry in _as_list(own_final.get("active")) if entry]
    bench_list = [entry for entry in _as_list(own_final.get("bench")) if entry]
    probe_summary = probe.summary()
    counts = _as_dict(probe_summary.get("raw_counts"))
    statuses = [str(status) for status in _as_list(data.get("statuses"))]
    slot = SLOT_LABELS.get(submission_id, {})

    return {
        "episode_id": str(data.get("id") or path.stem),
        "submission_id": submission_id,
        "slot_label": slot.get("label", "unknown"),
        "model": slot.get("model", "unknown"),
        "deck": slot.get("deck", "unknown"),
        "our_seat": our_seat,
        "opponent": team_names[opp_seat] if len(team_names) == 2 else "",
        "opponent_shell": shell_label(opp_cards, cards),
        "result": result,
        "turns": max_turn,
        "decisions": decisions,
        "attacks": attacks,
        "end_choices": end_choices,
        "prizes_taken": (6 - own_prizes) if own_prizes is not None else None,
        "prizes_conceded": (6 - opp_prizes) if opp_prizes is not None else None,
        "own_prizes_remaining": own_prizes,
        "opp_prizes_remaining": opp_prizes,
        "our_status": statuses[our_seat] if len(statuses) == 2 else "",
        "opp_status": statuses[opp_seat] if len(statuses) == 2 else "",
        "final_active_count": len(active_list),
        "final_bench_count": len(bench_list),
        "final_overage_time": final_overage,
        "riptide_attacks": _as_int(counts.get("riptide_attacks")),
        "riptide_whiffs": _as_int(counts.get("riptide_whiffs")),
        "hammer_attacks": _as_int(counts.get("hammer_attacks")),
        "hammer_low_deck": _as_int(counts.get("hammer_low_deck")),
        "surfing_activations": _as_int(counts.get("surfing_activations")),
        "our_deck_size": len(our_cards),
        "replay_path": str(path),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["slot_label"], row["result"])].append(row)

    def _mean(values: list[Any]) -> float | None:
        numeric = [float(value) for value in values if isinstance(value, (int, float))]

        return round(sum(numeric) / len(numeric), 3) if numeric else None

    out: list[dict[str, Any]] = []
    for (slot_label, result), bucket in sorted(grouped.items()):
        riptide_attacks = sum(row["riptide_attacks"] for row in bucket)
        riptide_whiffs = sum(row["riptide_whiffs"] for row in bucket)
        hammer_attacks = sum(row["hammer_attacks"] for row in bucket)
        hammer_low = sum(row["hammer_low_deck"] for row in bucket)
        shells = Counter(row["opponent_shell"] for row in bucket)
        out.append(
            {
                "slot_label": slot_label,
                "result": result,
                "games": len(bucket),
                "mean_turns": _mean([row["turns"] for row in bucket]),
                "mean_prizes_taken": _mean([row["prizes_taken"] for row in bucket]),
                "mean_prizes_conceded": _mean([row["prizes_conceded"] for row in bucket]),
                "no_prize_games": sum(1 for row in bucket if row["prizes_taken"] == 0),
                "empty_active_end": sum(1 for row in bucket if row["final_active_count"] == 0),
                "opp_not_done": sum(1 for row in bucket if row["opp_status"] not in ("DONE", "")),
                "our_not_done": sum(1 for row in bucket if row["our_status"] not in ("DONE", "")),
                "riptide_whiff_rate": round(riptide_whiffs / riptide_attacks, 4) if riptide_attacks else None,
                "riptide_attacks": riptide_attacks,
                "hammer_low_deck_rate": round(hammer_low / hammer_attacks, 4) if hammer_attacks else None,
                "hammer_attacks": hammer_attacks,
                "surfing_activations": sum(row["surfing_activations"] for row in bucket),
                "top_opponent_shells": "; ".join(f"{shell} x{count}" for shell, count in shells.most_common(5)),
            }
        )

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], *, manifest_path: Path) -> str:
    lines = [
        "# R14 Lane 1 - Ladder Replay Taxonomy",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- created_at_utc: `{_now()}`",
        f"- episodes analyzed: {len(rows)}",
        f"- scout manifest: `{manifest_path}`",
        "",
        "## Aggregate by slot and result",
        "",
        "| slot | result | games | turns | prizes taken | prizes conceded | 0-prize games | empty-active end | opp not DONE | riptide whiff | hammer low-deck | top opponent shells |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for agg in aggregates:
        lines.append(
            f"| {agg['slot_label']} | {agg['result']} | {agg['games']} | {agg['mean_turns']} | "
            f"{agg['mean_prizes_taken']} | {agg['mean_prizes_conceded']} | {agg['no_prize_games']} | "
            f"{agg['empty_active_end']} | {agg['opp_not_done']} | {agg['riptide_whiff_rate']} | {agg['hammer_low_deck_rate']} | "
            f"{agg['top_opponent_shells']} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze downloaded ladder replays for DeepMew submissions.")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--scout-manifest", required=True)
    parser.add_argument("--team", default="DeepMew")
    parser.add_argument("--card-csv", default=str(DEFAULT_CARD_CSV))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    cards = load_card_names(Path(args.card_csv))
    mapping = episode_submission_map(Path(args.scout_manifest))
    replay_dir = Path(args.replay_dir)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted(replay_dir.glob("episode-*-replay.json")):
        episode_id = path.stem.replace("episode-", "").replace("-replay", "")
        submission_id = mapping.get(episode_id, "")
        row = analyze_episode(path, team=args.team, submission_id=submission_id, cards=cards)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    aggregates = aggregate(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "episode_taxonomy.csv", rows)
    write_csv(out_dir / "aggregate_by_slot_result.csv", aggregates)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {"replay_dir": str(replay_dir), "scout_manifest": str(args.scout_manifest), "team": args.team},
        "episodes_analyzed": len(rows),
        "episodes_skipped_not_ours": skipped,
        "aggregates": aggregates,
    }
    (out_dir / "taxonomy_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "taxonomy_report.md").write_text(render_report(rows, aggregates, manifest_path=Path(args.scout_manifest)), encoding="utf-8")
    print(json.dumps({"episodes_analyzed": len(rows), "skipped": skipped, "aggregates": aggregates}, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
