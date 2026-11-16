"""R16: A/B the futile-attack guard on the shipped R4-F distilled runtime.

Motivated by ladder loss 84541591 (2026-07-06): R4-F attacked an ex-immune
Crustle for 0 damage ~13 turns and self-milled with Hammer-lanche to the brink
of deck-out. Arms:
  A (incumbent): bundle DistilledPolicy exactly as shipped.
  B (guarded):   same policy + deepmew.futility_guard post-processing.

Opponents: K3 deck-zoo decklists piloted by first-legal — Crustle-wall decks
(targeted probe) plus non-Crustle control decks (regression watch) — paired
seeds per arm via identical Random seeds and alternating seats.

Usage (repo root):
  python scripts/run_r16_futility_guard_ab.py --bundle-dir <unpacked bundle> \
      --games 64 [--no-write]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_markdown
from deepmew.futility_guard import FutilityGuard
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup

SCHEMA_VERSION = "deepmew.r16_futility_guard_ab.v1"

# Crustle(345)-wall probes + non-Crustle regression controls (K3 deck zoo hashes).
CRUSTLE_DECKS = ("0920dd994f954de6", "d63c6ad09fc480e6", "6415396d35c0f4b3")
CONTROL_DECKS = ("bd27c5c46ec86161", "008816603a8ac836", "86c3817de0b1db80")


def _latest_deck_zoo_decklists() -> Path:
    candidates = sorted((ROOT / ".research" / "runs" / "deck_zoo").glob("*/*/deck_zoo_decklists.jsonl"))
    if not candidates:
        raise FileNotFoundError("no deck_zoo_decklists.jsonl under .research/runs/deck_zoo")
    return candidates[-1]


def _load_decklists(path: Path) -> dict[str, dict[str, Any]]:
    decks: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            decks[str(row.get("deck_hash"))] = row
    return decks


def _deck_cards(row: dict[str, Any]) -> list[int]:
    counts = row.get("card_counts") or {}
    cards: list[int] = []
    for card_id, copies in sorted((int(k), int(v)) for k, v in counts.items()):
        cards.extend([card_id] * copies)
    if len(cards) != 60:
        raise ValueError(f"decklist {row.get('deck_hash')} has {len(cards)} cards")
    return cards


class BundleAgent:
    """SeatAgent adapter around the shipped bundle's DistilledPolicy."""

    def __init__(self, bundle_dir: Path, label: str, *, guarded: bool) -> None:
        self.label = label
        self.guarded = guarded
        self.guard = FutilityGuard()
        self.guard_swaps = 0
        if str(bundle_dir) not in sys.path:
            sys.path.insert(0, str(bundle_dir))
        import r5_distilled_policy as runtime  # noqa: PLC0415 - bundle import by design

        self._runtime = runtime
        self._policy = runtime.DistilledPolicy(
            str(bundle_dir / "r5_policy_weights.npz"),
            str(bundle_dir / "r5_policy_weights.manifest.json"),
        )

    def reset(self) -> None:
        self.guard = FutilityGuard()

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        select = obs.get("select") if isinstance(obs, dict) else None
        if not isinstance(select, dict):
            return []
        try:
            choice = self._policy.select(obs)
        except Exception:
            choice = self._runtime.safe_fallback(select)
        choice = self._runtime.guard_selection([int(i) for i in choice], select)
        if self.guarded:
            adjusted = self.guard.process(obs, select, list(choice))
            if adjusted != choice:
                self.guard_swaps += 1
            choice = self._runtime.guard_selection([int(i) for i in adjusted], select)
        return choice


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = Path(args.bundle_dir).resolve()
    deck_csv = bundle_dir / "deck.csv"
    our_deck = [int(v) for v in deck_csv.read_text(encoding="utf-8").split()][:60]
    decklists_path = _latest_deck_zoo_decklists()
    decks = _load_decklists(decklists_path)

    rows: list[dict[str, Any]] = []
    totals = {"incumbent": [0, 0], "guarded": [0, 0]}  # [wins, losses]
    safety = {"crashes": 0, "timeouts": 0}
    for bucket, hashes in (("crustle_wall", CRUSTLE_DECKS), ("control", CONTROL_DECKS)):
        for deck_hash in hashes:
            row = decks.get(deck_hash)
            if row is None:
                print(f"[skip] deck {deck_hash} not in {decklists_path}")
                continue
            opp_deck = _deck_cards(row)
            opp_agent = SpecAgent.from_str("first-legal")
            for arm, guarded in (("incumbent", False), ("guarded", True)):
                agent = BundleAgent(bundle_dir, f"r4f_{arm}", guarded=guarded)
                summary = run_real_meta_matchup(
                    agent, opp_agent, our_deck, opp_deck,
                    games=args.games, bucket=bucket,
                    opp_deck_hash=deck_hash,
                    opp_archetype=str(row.get("archetype") or "Unknown"),
                    seed=args.seed,
                )
                totals[arm][0] += summary.our_wins
                totals[arm][1] += summary.opp_wins
                safety["crashes"] += summary.crashes
                safety["timeouts"] += summary.timeouts
                rows.append({
                    "bucket": bucket, "deck_hash": deck_hash,
                    "archetype": summary.opp_archetype, "arm": arm,
                    "games": summary.games, "wins": summary.our_wins,
                    "losses": summary.opp_wins, "draws": summary.draws,
                    "crashes": summary.crashes, "timeouts": summary.timeouts,
                    "avg_turns": round(summary.avg_turns, 2),
                    "guard_swaps": agent.guard_swaps,
                    "winrate": round(summary.our_wins / max(1, summary.our_wins + summary.opp_wins), 4),
                    "errors": summary.errors,
                })
                print(f"[{bucket}] {deck_hash} {arm}: {summary.our_wins}-{summary.opp_wins}-{summary.draws} "
                      f"(crash {summary.crashes} / timeout {summary.timeouts}, swaps {agent.guard_swaps})", flush=True)

    def _wr(arm: str) -> float:
        wins, losses = totals[arm]
        return wins / max(1, wins + losses)

    result = {
        "schema_version": SCHEMA_VERSION,
        "bundle_dir": str(bundle_dir),
        "decklists": str(decklists_path),
        "games_per_cell": args.games,
        "seed": args.seed,
        "rows": rows,
        "overall": {arm: {"wins": totals[arm][0], "losses": totals[arm][1], "winrate": round(_wr(arm), 4)} for arm in totals},
        "safety": safety,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=70207)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result["overall"], indent=2))
    print("safety:", result["safety"])
    if args.no_write:
        return
    run_dir = create_run_dir(kind="futility_guard_ab", out_root=ROOT / ".research" / "runs")
    write_json(run_dir.output_path("results.json"), result)
    lines = [
        "# R16 futility-guard A/B",
        "",
        f"- bundle: `{result['bundle_dir']}`",
        f"- games/cell: {result['games_per_cell']}, seed {result['seed']}",
        f"- overall incumbent {result['overall']['incumbent']['winrate']} vs guarded {result['overall']['guarded']['winrate']}",
        f"- safety: {result['safety']}",
        "",
        "| bucket | deck | archetype | arm | W-L-D | winrate | swaps |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['bucket']} | {row['deck_hash']} | {row['archetype']} | {row['arm']} "
            f"| {row['wins']}-{row['losses']}-{row['draws']} | {row['winrate']} | {row['guard_swaps']} |"
        )
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines) + "\n")
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="futility_guard_ab",
        command=[sys.executable, *sys.argv],
        inputs={"bundle_dir": result["bundle_dir"], "decklists": result["decklists"]},
        outputs={"results": "results.json", "report": "report.md"},
        summary={"overall": result["overall"], "safety": result["safety"]},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)


if __name__ == "__main__":
    main()
