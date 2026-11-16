from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r9_r4f_retrain as r4f
from deepmew.second_order.reader import option_second_order_vector, read_second_order_state


SCHEMA_VERSION = "deepmew.r9.feature_leakage.v0"
DEFAULT_EXAMPLES = ROOT / ".research" / "rl1_top_player_bc" / "replay_label_examples.jsonl"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane2_r4f"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _inject_markers(obs: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obs)
    current = _as_dict(mutated.get("current"))
    players = current.get("players")
    your_index = int(current.get("yourIndex") or 0)
    if not isinstance(players, list) or len(players) < 2 or your_index not in (0, 1):
        return mutated
    opp = _as_dict(players[1 - your_index])
    hand_count = int(opp.get("handCount") or len(opp.get("hand") or []))
    deck_count = int(opp.get("deckCount") or len(opp.get("deck") or []))
    prize_count = len(opp.get("prize") or [])
    opp["hand"] = [{"id": 999001 + index} for index in range(hand_count)]
    opp["deck"] = [{"id": 999101 + index} for index in range(deck_count)]
    opp["prize"] = [{"id": 999201 + index} for index in range(prize_count)]
    players[1 - your_index] = opp

    return mutated


def run(args: argparse.Namespace) -> dict[str, Any]:
    checked = 0
    diffs = []
    with Path(args.examples).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            obs = row.get("observation")
            if not isinstance(obs, dict):
                continue
            mutated = _inject_markers(obs)
            state_a = read_second_order_state(obs).vector()
            state_b = read_second_order_state(mutated).vector()
            if state_a != state_b:
                diffs.append({"row": checked, "kind": "state"})
            for option in row.get("legal_action_table") or []:
                if option_second_order_vector(obs, option) != option_second_order_vector(mutated, option):
                    diffs.append({"row": checked, "kind": "option"})
                    break
            encoded_a = r4f.encode_top_player_row_r9(row)
            row_b = dict(row)
            row_b["observation"] = mutated
            encoded_b = r4f.encode_top_player_row_r9(row_b)
            if encoded_a is not None and encoded_b is not None and encoded_a.features != encoded_b.features:
                diffs.append({"row": checked, "kind": "augmented"})
            checked += 1
            if args.limit and checked >= args.limit:
                break
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "examples": str(args.examples),
        "rows_checked": checked,
        "marker_diffs": len(diffs),
        "status": "PASS" if checked and not diffs else "FAIL",
        "diff_examples": diffs[:20],
        "note": "Injected opponent hidden hand/deck/prize marker IDs must not change second-order state, option, or augmented R4-F features.",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_leakage_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "feature_leakage_audit.md").write_text(
        "\n".join(
            [
                "# R9 Feature Leakage Audit",
                "",
                f"- rows_checked: `{payload['rows_checked']}`",
                f"- marker_diffs: `{payload['marker_diffs']}`",
                f"- status: `{payload['status']}`",
                "",
                payload["note"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R9 marker-leakage audit for second-order features.")
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)
    payload = run(args)

    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
