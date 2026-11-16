"""R19: best-response corpus generator -> AWR-consumable refgame shards.

Generates training rows of a LEARNER pilot (the exploiter-in-training) piloting the
candidate-band decks against our FROZEN candidate on the root deck, stamped with the
outcome (did the learner beat the candidate?). Feed the output to the AWR trainer so
it gradient-trains a real best-response of the candidate -- the curriculum's
`adversarial` opponent. This is the data half of the adversarial harness; the AWR
trainer itself is unchanged.

Output layout matches the trainers' refgames-dir contract exactly
(`<out>/shards/decisions_shard_XXX.jsonl.gz` + manifest), reusing
run_r18_curriculum_datagen.play_recorded_game for identical row encoding.

INVARIANT #2: reward is the observed outcome vs a FIXED candidate snapshot, not a
self-play-to-Nash signal. Offline/training-only (INVARIANT #8).

Example:
  python scripts/run_r19_best_response_datagen.py --pools <pools.json> \
      --candidate checkpoint:D:/.../bd27.pt --learner checkpoint:D:/.../champion.pt \
      --stages 900-1000 1000-1100 --games 8 \
      --out-dir D:/DeepMew_artifacts/datasets/r19/bd27_best_response
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl.observation import ObservationEncoder  # noqa: E402
import run_r18_curriculum_datagen as dg  # noqa: E402

SCHEMA_VERSION = "deepmew.r19.best_response_datagen.v0"


def _deck_from_candidate(candidate: str) -> list[int]:
    """Resolve the frozen candidate's deck. bundle:<dir> uses that bundle's deck.csv;
    everything else pilots the root deck.csv."""
    kind, _, target = candidate.partition(":")
    if kind == "bundle" and target:
        return [int(v) for v in (Path(target) / "deck.csv").read_text(encoding="utf-8").split()][:60]
    return [int(v) for v in main.read_deck_csv()][:60]


def _candidate_pilot_spec(candidate: str) -> str:
    """Map a curriculum candidate spec to a datagen `_build_pilot` spec."""
    kind, _, target = candidate.partition(":")
    if kind == "spec":
        return target
    if kind == "checkpoint" and target:
        return f"checkpoint:{target}"
    if kind == "bundle":
        # A bundle's distilled runtime isn't a datagen pilot; use the champion checkpoint
        # as the frozen brain over the bundle deck (documented approximation).
        return "exploiter"
    return candidate


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--candidate", required=True, help="FROZEN candidate to best-respond to: spec:/checkpoint:/bundle:")
    ap.add_argument("--learner", default="greedy",
                    help="exploiter-in-training pilot: greedy/v2.4/checkpoint:<path>/exploiter (warm start)")
    ap.add_argument("--stages", nargs="+", default=["900-1000", "1000-1100"])
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--carry-n", type=int, default=0)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--shard-rows", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--deck-policy", default="reference_abomasnow")
    ap.add_argument("--seed", type=int, default=190720)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    encoder = ObservationEncoder()

    cand_deck = _deck_from_candidate(args.candidate)
    cand_hash = normalized_deck_hash(cand_deck)
    cand_spec = _candidate_pilot_spec(args.candidate)
    candidate_pilot = dg._build_pilot(cand_spec, cand_deck, cand_hash)
    cand_label = f"candidate:{args.candidate}"

    pools_payload = json.loads(Path(args.pools).read_text(encoding="utf-8"))
    pools = {b: [entry_from_json(r) for r in rows] for b, rows in pools_payload["pools"].items()}
    shells = {r["deck_hash"]: str(r.get("shell") or "Unknown")
              for rows in pools_payload["pools"].values() for r in rows}

    shard_rows: list[dict[str, Any]] = []
    shard_idx = 0
    totals = {"games": 0, "rows": 0, "learner_wins": 0, "completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    game_index = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal shard_rows, shard_idx
        if not shard_rows:
            return
        path = out_dir / "shards" / f"decisions_shard_{shard_idx:03d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as h:
            for r in shard_rows:
                h.write(json.dumps(r) + "\n")
        shard_idx += 1
        shard_rows = []

    for stage in args.stages:
        for opp in build_stage(pools, stage, top_n=int(args.top_n), carry_n=int(args.carry_n)):
            shell = shells.get(opp.deck_hash, "Unknown")
            learner_pilot = dg._build_pilot(args.learner, list(opp.card_ids), opp.deck_hash)
            learner_label = f"learner:{args.learner}:{shell}"
            for g in range(int(args.games)):
                # our=LEARNER on the band deck; opp=frozen CANDIDATE on root deck.
                rows, safety = dg.play_recorded_game(
                    learner_pilot, candidate_pilot, list(opp.card_ids), cand_deck,
                    our_seat=g % 2, our_label=learner_label, opp_label=cand_label,
                    opp_deck_hash=cand_hash, bucket=stage,
                    game_id=f"br-{stage}-{opp.deck_hash[:8]}-{g}",
                    game_index=game_index, seed=int(args.seed) + game_index,
                    encoder=encoder, max_steps=int(args.max_steps),
                )
                game_index += 1
                totals["games"] += 1
                for k in ("completed", "crashes", "timeouts", "illegal"):
                    totals[k] += safety.get(k, 0)
                totals["learner_wins"] += safety.get("our_wins", 0)
                totals["rows"] += len(rows)
                shard_rows.extend(rows)
                if len(shard_rows) >= int(args.shard_rows):
                    flush()
            print(f"[{stage}] {shell[:26]:26s} done | games={totals['games']} rows={totals['rows']} "
                  f"learner_wins={totals['learner_wins']}/{totals['completed']} crashes={totals['crashes']}", flush=True)
    flush()

    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": dg._now(),
        "candidate": args.candidate, "learner": args.learner, "stages": list(args.stages),
        "candidate_deck_hash": cand_hash, "totals": totals, "shards": shard_idx,
        "learner_win_rate": round(totals["learner_wins"] / max(1, totals["completed"]), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "note": "learner rows are exploiter-perspective vs a FROZEN candidate; feed to AWR "
                "(--refgames-dir) to gradient-train the adversarial best-response (INVARIANT #2).",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
