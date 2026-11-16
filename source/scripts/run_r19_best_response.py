"""R19: adversarial best-response selector for the curriculum `adversarial` tier.

Measures how badly each exploiter BRAIN (a checkpoint or a heuristic spec) piloting
the candidate-band decks beats our FROZEN candidate, ranks them, selects the
strongest as the curriculum's `adversarial` opponent, and registers it into the
league as a frozen member. This is the operator that turns AWR's output (or any
existing checkpoint) into a curriculum opponent.

INVARIANT #2: the target is a FIXED candidate snapshot -> best-response, not
equilibrium. INVARIANT #5: arena is a filter; the selected exploiter is a training
target, not a ladder verdict. Offline/eval-only (INVARIANT #8).

Example (torch-free smoke with heuristic brains):
  python scripts/run_r19_best_response.py --pools <pools.json> \
      --candidate spec:greedy --stage 800-900 --brains v2.4 first-legal --games 8 --no-write

Example (real, checkpoint brains incl. the AWR output):
  python scripts/run_r19_best_response.py --pools <pools.json> \
      --candidate bundle:.research/submissions/20260706-r4f-bd27 --stage 900-1000 \
      --brains D:/DeepMew_artifacts/checkpoints/r18-starmie-awr-iter2-BEST.pt greedy \
      --games 32 --register-league league.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.curriculum import League, LeagueMember, guard_invariant_2  # noqa: E402
from deepmew.curriculum.best_response import ExploiterResult, best_response_summary  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup  # noqa: E402
import run_r17_bucket_arena as r17  # noqa: E402

SCHEMA_VERSION = "deepmew.r19_best_response.v1"
HEURISTIC_SPECS = {"first-legal", "random", "greedy", "legacy-v2", "v2.4", "v4"}


def _brain_label(brain: str) -> str:
    return brain if brain in HEURISTIC_SPECS else Path(brain).stem


def build_exploiter_brain(brain: str, deck: list[int], deck_hash: str, deck_policy: str) -> Any:
    """A SeatAgent for `brain` piloting `deck`. Heuristic specs need no deck binding;
    a checkpoint path binds the deck into an R4-F live policy (build_exploiter)."""
    if brain in HEURISTIC_SPECS:
        return SpecAgent.from_str(brain)
    return r17.build_exploiter(Path(brain), list(deck), deck_hash, deck_policy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--candidate", required=True, help="the FROZEN candidate to be exploited: spec:/bundle:/checkpoint:")
    parser.add_argument("--brains", nargs="+", required=True, help="exploiter brains: heuristic specs and/or checkpoint paths")
    parser.add_argument("--stage", default="900-1000", help="candidate-band stage whose decks the exploiters pilot")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--carry-n", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--min-decided", type=int, default=8, help="min decided games before a brain can be crowned adversarial")
    parser.add_argument("--register-league", default="", help="league JSON to load+append the selected adversarial to (also written back)")
    parser.add_argument("--seed", type=int, default=190719)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    pools_payload = json.loads(Path(args.pools).read_text(encoding="utf-8"))
    pools = {b: [entry_from_json(r) for r in rows] for b, rows in pools_payload["pools"].items()}
    shells = {r["deck_hash"]: str(r.get("shell") or "Unknown")
              for rows in pools_payload["pools"].values() for r in rows}
    stage = build_stage(pools, args.stage, top_n=args.top_n, carry_n=args.carry_n)
    if not stage:
        raise SystemExit(f"stage {args.stage} is empty in this pool")

    # The frozen candidate is the OPPONENT the exploiters try to beat.
    candidate = r17.build_candidate(args.candidate, args.deck_policy)
    cand_deck = list(getattr(candidate, "deck", None)
                     or getattr(getattr(candidate, "inner", None), "deck", None) or [])
    if len(cand_deck) != 60:
        raise SystemExit("candidate did not expose a 60-card deck")

    results: list[ExploiterResult] = []
    safety = {"crashes": 0, "timeouts": 0}
    for brain in args.brains:
        label = _brain_label(brain)
        for opp_index, opp in enumerate(stage):
            shell = shells.get(opp.deck_hash, "Unknown")
            exploiter = build_exploiter_brain(brain, list(opp.card_ids), opp.deck_hash, args.deck_policy)
            # our=exploiter (on the band deck), opp=frozen candidate (on its root deck).
            summary = run_real_meta_matchup(
                exploiter, candidate, list(opp.card_ids), cand_deck,
                games=args.games, bucket=opp.bucket, opp_deck_hash=opp.deck_hash,
                opp_archetype=shell, seed=args.seed + opp_index,
            )
            safety["crashes"] += summary.crashes
            safety["timeouts"] += summary.timeouts
            results.append(ExploiterResult(
                brain=label, ref=brain, deck_hash=opp.deck_hash, shell=shell,
                bucket=opp.bucket, games=summary.games,
                exploiter_wins=summary.our_wins, candidate_wins=summary.opp_wins,
            ))
            print(f"  {label:28s} x {shell[:24]:24s} exploiter {summary.our_wins}-{summary.opp_wins}"
                  f" (wr_vs_cand {summary.our_wins / max(1, summary.our_wins + summary.opp_wins):.3f})", flush=True)

    summary = best_response_summary(results, min_decided=args.min_decided)
    summary.update({"schema_version": SCHEMA_VERSION, "candidate": args.candidate,
                    "stage": args.stage, "games_per_cell": args.games, "safety": safety})
    adversarial = summary["adversarial"]
    print(json.dumps({"adversarial": adversarial, "safety": safety,
                      "top_counters": summary["top_counters"]}, indent=2))
    if adversarial:
        print(f"\n-> wire into curriculum: --adversarial {adversarial['ref']}"
              if adversarial["ref"] not in HEURISTIC_SPECS else
              f"\n-> strongest exploiter is heuristic '{adversarial['ref']}' "
              f"(wr_vs_candidate {adversarial['mean_win_vs_candidate']}); "
              f"a trained checkpoint is needed for a real adversarial tier")

    # Register the selected adversarial into the league as a FROZEN member.
    if args.register_league and adversarial and adversarial["ref"] not in HEURISTIC_SPECS:
        league_path = Path(args.register_league)
        league = (League.from_json(json.loads(league_path.read_text(encoding="utf-8")))
                  if league_path.exists() else League())
        member = LeagueMember(
            member_id=f"adversarial-r{league.round}-{adversarial['brain']}",
            kind="exploiter", round_added=league.round, ref=adversarial["ref"],
            note=f"best-response to {args.candidate} on {args.stage} "
                 f"(wr_vs_candidate {adversarial['mean_win_vs_candidate']})",
        )
        league.add(member)
        guard_invariant_2(league.members, league.round)
        league_path.write_text(json.dumps(league.to_json(), indent=2), encoding="utf-8")
        print(f"registered adversarial into league: {league_path} (member {member.member_id})")

    if args.no_write:
        return
    run_dir = create_run_dir(kind="best_response", out_root=ROOT / ".research" / "runs")
    write_json(run_dir.output_path("results.json"),
               {**summary, "results": [r.to_json() for r in results]})
    lines = ["# R19 adversarial best-response", "",
             f"- candidate (exploited): `{args.candidate}`  stage: {args.stage}  safety: {safety}",
             f"- adversarial: {adversarial}", "",
             "| brain | decks | decided | wr_vs_candidate |", "|---|---|---|---|"]
    for b in summary["brain_rankings"]:
        lines.append(f"| {b['brain']} | {b['decks']} | {b['decided']} | {b['mean_win_vs_candidate']} |")
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines) + "\n")
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="best_response",
        command=[sys.executable, *sys.argv],
        inputs={"pools": args.pools, "candidate": args.candidate, "brains": args.brains},
        outputs={"results": "results.json", "report": "report.md"},
        summary={"adversarial": adversarial, "safety": safety},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)


if __name__ == "__main__":
    main()
