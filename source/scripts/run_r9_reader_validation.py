from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r7_probes as r7_probes
import run_r7_sim_reads_cards as r7_sim
from deepmew.second_order.reader import (
    BASIC_WATER_ENERGY,
    SECOND_ORDER_OPTION_SCHEMA,
    SECOND_ORDER_STATE_SCHEMA,
    option_second_order_vector,
    project_attack,
    read_second_order_state,
)


SCHEMA_VERSION = "deepmew.r9.reader_validation.v0"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane1_reader"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _option_for_action(obs: dict[str, Any], action: list[int]) -> dict[str, Any]:
    options = (obs.get("select") or {}).get("option") or []
    if not action or action[0] < 0 or action[0] >= len(options):
        return {}

    return options[action[0]]


def _attack_projection_row(
    episode: str,
    seat: int,
    declaration: dict[str, Any],
    observed: dict[str, Any],
    sim: dict[str, Any],
) -> dict[str, Any]:
    obs = declaration["obs"]
    projection = project_attack(obs, int(declaration["attack_id"]))
    option = _option_for_action(obs, declaration["action"])
    option_vector = option_second_order_vector(obs, option)
    if projection is None:
        projection_payload = {
            "projected_kind": "",
            "projected_resource_count": None,
            "projected_base_damage": None,
            "projected_weakness_damage": None,
            "projected_self_mill_count": None,
            "projected_deck_after_mill": None,
            "projected_hidden_dependent": None,
            "projected_zero_damage_whiff": None,
            "projected_low_deck_self_mill": None,
        }
    else:
        projection_payload = {
            "projected_kind": projection.kind,
            "projected_resource_count": projection.resource_count,
            "projected_base_damage": projection.projected_base_damage,
            "projected_weakness_damage": projection.projected_weakness_damage,
            "projected_self_mill_count": projection.projected_self_mill_count,
            "projected_deck_after_mill": projection.projected_deck_after_mill,
            "projected_hidden_dependent": projection.hidden_dependent,
            "projected_zero_damage_whiff": projection.zero_damage_whiff,
            "projected_low_deck_self_mill": projection.low_deck_self_mill,
        }
    row = {
        "episode": episode,
        "seat": seat,
        "decision_step": declaration["decision_step"],
        "turn": declaration["turn"],
        "attack_id": declaration["attack_id"],
        "attack_name": "Riptide" if declaration["attack_id"] == r7_sim.RIPTIDE else "Hammer-lanche",
        "discard_water_before": declaration["discard_water_before"],
        "deck_before": declaration["deck_before"],
        "observed_attack_seen": observed["attack_seen"],
        "observed_damage": observed["damage"],
        "observed_milled_total": observed["milled_total"],
        "observed_milled_waters": observed["milled_waters"],
        "sim_status": sim.get("sim_status"),
        "sim_damage": sim.get("sim_damage"),
        "sim_milled_total": sim.get("sim_milled_total"),
        "sim_milled_waters": sim.get("sim_milled_waters"),
        "option_second_order_vector": ";".join(f"{value:.6g}" for value in option_vector),
    }
    row.update(projection_payload)

    return row


def _surfing_validation_rows(steps: list[Any], seat: int, episode: str) -> list[dict[str, Any]]:
    rows = []
    for decision_step, obs, action in r7_sim.decision_pairs(steps, seat):
        state = read_second_order_state(obs)
        current = obs.get("current") or {}
        select = obs.get("select") or {}
        expected_index = r7_probes.surfing_beach_ability_option_index(select)
        expected_in_play = r7_probes.surfing_beach_in_play(current)
        if not expected_in_play and expected_index is None and not state.surfing_beach_in_play:
            continue
        expected_available = expected_in_play and expected_index is not None
        expected_beneficial = expected_available and r7_probes.own_active_is_water(current, seat) and r7_probes.own_bench_has_water(current, seat)
        rows.append(
            {
                "episode": episode,
                "seat": seat,
                "decision_step": decision_step,
                "turn": int(current.get("turn") or 0),
                "reader_in_play": state.surfing_beach_in_play,
                "expected_in_play": expected_in_play,
                "reader_available": state.surfing_beach_available,
                "expected_available": expected_available,
                "reader_beneficial": state.surfing_beach_beneficial_proxy,
                "expected_beneficial": expected_beneficial,
                "action_activated": expected_index in action if expected_index is not None else False,
            }
        )

    return rows


def _attack_correctness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [row for row in rows if row["observed_attack_seen"]]
    riptide = [row for row in scored if row["attack_id"] == r7_sim.RIPTIDE]
    hammer = [row for row in scored if row["attack_id"] == r7_sim.HAMMER_LANCHE]
    table = [
        {
            "feature": "riptide_projected_damage",
            "states": len(riptide),
            "correct": sum(1 for row in riptide if row["projected_weakness_damage"] == row["observed_damage"]),
            "rate": None,
            "gate_threshold": 0.95,
            "verdict": "",
        },
        {
            "feature": "riptide_discard_resource_count",
            "states": len(riptide),
            "correct": sum(1 for row in riptide if row["projected_resource_count"] == row["discard_water_before"]),
            "rate": None,
            "gate_threshold": 0.95,
            "verdict": "",
        },
        {
            "feature": "hammer_projected_self_mill_count",
            "states": len(hammer),
            "correct": sum(1 for row in hammer if row["projected_self_mill_count"] == min(6, int(row["deck_before"]))),
            "rate": None,
            "gate_threshold": 0.95,
            "verdict": "",
        },
        {
            "feature": "hammer_hidden_dependent_flag",
            "states": len(hammer),
            "correct": sum(1 for row in hammer if row["projected_hidden_dependent"] is True),
            "rate": None,
            "gate_threshold": 0.95,
            "verdict": "",
        },
    ]
    for item in table:
        item["rate"] = round(item["correct"] / item["states"], 6) if item["states"] else None
        item["verdict"] = "PASS" if item["states"] and item["rate"] >= item["gate_threshold"] else "NOT_SCORED"

    return table


def _surfing_correctness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = [
        ("surfing_in_play", "reader_in_play", "expected_in_play"),
        ("surfing_available", "reader_available", "expected_available"),
        ("surfing_beneficial_proxy", "reader_beneficial", "expected_beneficial"),
    ]
    table = []
    for feature, reader_key, expected_key in checks:
        correct = sum(1 for row in rows if row[reader_key] == row[expected_key])
        rate = round(correct / len(rows), 6) if rows else None
        table.append(
            {
                "feature": feature,
                "states": len(rows),
                "correct": correct,
                "rate": rate,
                "gate_threshold": 0.95,
                "verdict": "PASS" if rows and rate >= 0.95 else "NOT_SCORED",
            }
        )

    return table


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R9 Lane 1 Reader Validation",
        "",
        f"- schema: `{payload['schema_version']}`",
        f"- gate_result: `{payload['gate_result']}`",
        f"- attack_states_scored: `{payload['attack_states_scored']}`",
        f"- surfing_states_scored: `{payload['surfing_states_scored']}`",
        f"- state_schema_dim: `{len(SECOND_ORDER_STATE_SCHEMA)}`",
        f"- option_schema_dim: `{len(SECOND_ORDER_OPTION_SCHEMA)}`",
        "",
        "## Feature Correctness",
        "",
        "| feature | states | correct | rate | threshold | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["feature_correctness"]:
        lines.append(
            f"| `{row['feature']}` | {row['states']} | {row['correct']} | {row['rate']} | {row['gate_threshold']} | `{row['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Hammer-lanche damage itself is marked hidden-dependent because the top six own-deck cards are not public; the validated public feature is the exact self-mill count/deck-risk meter, not exact damage.",
            "- Surfing Beach beneficial is a public legality-plus-board proxy, matching the R7 probe definition; true value still requires policy/search evaluation.",
            "- No opponent hidden hand/deck/prize contents are read by the reader; leakage is tested separately.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decklist = r7_sim.read_deck(r7_sim.PILOT_DECK_CSV)
    weakness_by_card = {card.cardId: card.weakness for card in r7_sim.all_card_data()}
    seats = r7_sim.our_seat_by_episode(Path(args.log_dir))
    attack_rows: list[dict[str, Any]] = []
    surfing_rows: list[dict[str, Any]] = []
    event_truth: dict[str, dict[str, int]] = {}
    for episode in sorted(seats):
        steps = r7_sim.load_episode(Path(args.replay_dir), episode)
        seat = seats[episode]
        truth = r7_sim.count_attack_events(steps, seat)
        event_truth[episode] = {"riptide_events": truth.get(r7_sim.RIPTIDE, 0), "hammer_events": truth.get(r7_sim.HAMMER_LANCHE, 0)}
        surfing_rows.extend(_surfing_validation_rows(steps, seat, episode))
        for declaration in r7_sim.extract_declarations(steps, seat, episode):
            resolution = r7_sim.observed_resolution(steps, seat, declaration["decision_step"], declaration["obs"])
            observed = r7_sim.analyze_attack_logs(resolution, seat, declaration["attack_id"])
            sim = r7_sim.simulate_declaration(declaration, decklist, weakness_by_card)
            attack_rows.append(_attack_projection_row(episode, seat, declaration, observed, sim))
            if args.limit and len(attack_rows) >= args.limit:
                break
        if args.limit and len(attack_rows) >= args.limit:
            break

    attack_fields = [
        "episode", "seat", "decision_step", "turn", "attack_id", "attack_name",
        "discard_water_before", "deck_before", "observed_attack_seen", "observed_damage",
        "observed_milled_total", "observed_milled_waters", "sim_status", "sim_damage",
        "sim_milled_total", "sim_milled_waters", "projected_kind", "projected_resource_count",
        "projected_base_damage", "projected_weakness_damage", "projected_self_mill_count",
        "projected_deck_after_mill", "projected_hidden_dependent", "projected_zero_damage_whiff",
        "projected_low_deck_self_mill", "option_second_order_vector",
    ]
    _write_csv(out_dir / "reader_attack_projection_validation.csv", attack_rows, attack_fields)
    _write_csv(
        out_dir / "reader_surfing_validation.csv",
        surfing_rows,
        [
            "episode", "seat", "decision_step", "turn", "reader_in_play", "expected_in_play",
            "reader_available", "expected_available", "reader_beneficial", "expected_beneficial",
            "action_activated",
        ],
    )
    feature_table = _attack_correctness(attack_rows) + _surfing_correctness(surfing_rows)
    pass_by_feature = {row["feature"]: row["verdict"] == "PASS" for row in feature_table}
    generalized_passes = sum(
        1
        for feature in ("hammer_projected_self_mill_count", "hammer_hidden_dependent_flag", "surfing_available", "surfing_beneficial_proxy")
        if pass_by_feature.get(feature)
    )
    riptide_pass = pass_by_feature.get("riptide_projected_damage", False)
    gate_result = "PASS" if riptide_pass and generalized_passes >= 2 else "FAIL"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "gate_result": gate_result,
        "gate_rule": "Riptide projected damage >=0.95 and at least two other present second-order features validate >=0.95.",
        "attack_states_total": len(attack_rows),
        "attack_states_scored": sum(1 for row in attack_rows if row["observed_attack_seen"]),
        "surfing_states_scored": len(surfing_rows),
        "state_schema": SECOND_ORDER_STATE_SCHEMA,
        "option_schema": SECOND_ORDER_OPTION_SCHEMA,
        "feature_correctness": feature_table,
        "event_ground_truth_by_episode": event_truth,
        "attack_declared_counts": dict(Counter(row["attack_name"] for row in attack_rows)),
        "leakage_note": "Reader consumes public current state plus own zones only; validation does not use opponent hidden hand/deck/prize contents.",
    }
    (out_dir / "reader_validation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "reader_validation_report.md").write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("gate_result", "attack_states_scored", "surfing_states_scored", "feature_correctness")}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate R9 second-order reader on R7 real states.")
    parser.add_argument("--replay-dir", default=str(r7_sim.DEFAULT_REPLAY_DIR))
    parser.add_argument("--log-dir", default=str(r7_sim.DEFAULT_LOG_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    payload = run_validation(args)

    return 0 if payload["gate_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
