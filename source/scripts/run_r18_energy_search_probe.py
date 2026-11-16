"""R18: energy-placement search probe (BUG-010 diagnostic).

Drives the recovered GreedyV search stack (determinize -> search_begin branch
enumeration -> ValueEvaluator calibration) on a single MAIN decision pulled from a
ladder replay, enumerating ALL legal branches over N determinized worlds and
printing the per-branch expected win-prob. Built to test whether search can resolve
attach-to-ACTIVE vs attach-to-BENCH energy placement (the BUG-010 misallocation).

Self-contained: extracts bd27's deck from its submission tarball and a Mega Starmie
opponent hypothesis from the latest deck-zoo run. Research/offline only; imports the
sourceless greedyv bytecode + cg native lib; does NOT touch main.py/deck.csv/runtime.

Example
-------
    python scripts/run_r18_energy_search_probe.py \
        --replay "<downloads>/7-6-2026-bd27/84708548.json" \
        --seat 1 --step 8 --worlds 8
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CARD_CSV = ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
AREA = {4: "ACTIVE", 5: "BENCH"}
OPTLABEL = {7: "PLAY", 10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END"}


def _card_names() -> dict[str, str]:
    out: dict[str, str] = {}
    for r in csv.reader(open(CARD_CSV, encoding="utf-8")):
        if r:
            out.setdefault(r[0], r[1])
    return out


def _bd27_deck() -> list[int]:
    tp = glob.glob(str(ROOT / ".research/submissions/*bd27*.tar.gz"))
    if not tp:
        raise SystemExit("no bd27 submission tarball found")
    data = tarfile.open(sorted(tp)[0]).extractfile("deck.csv").read().decode()
    return [int(x) for x in data.split()]


def _starmie_hypothesis() -> list[int]:
    files = sorted(
        glob.glob(str(ROOT / ".research/**/deck_zoo_decklists.jsonl"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    for f in files:
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            cc = d.get("card_counts") or {}
            if "1031" in cc or "1030" in cc:
                ids: list[int] = []
                for cid, n in cc.items():
                    ids += [int(cid)] * int(n)
                return ids
    raise SystemExit("no Mega Starmie decklist found in deck-zoo runs")


def _label(op: dict, names: dict) -> str:
    t = op.get("type")
    if t == 8:
        return f"ATTACH hand[{op.get('index')}]->{AREA.get(op.get('inPlayArea'), op.get('inPlayArea'))}[{op.get('inPlayIndex')}]"
    return OPTLABEL.get(t, f"type{t}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--seat", type=int, default=1)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--shortlist-k", type=int, default=24)
    ap.add_argument("--budget-ms", type=float, default=60000.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--value-checkpoint", default=None, help="override the value checkpoint (default: R15 selected)")
    args = ap.parse_args(argv)

    names = _card_names()
    bd27 = _bd27_deck()
    starmie = _starmie_hypothesis()

    d = json.load(open(args.replay))
    obs = d["steps"][args.step][args.seat]["observation"]
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    chose = d["steps"][args.step][args.seat].get("action")
    print(f"=== step {args.step}, {len(opts)} branches; agent chose {chose} ===")
    for j, op in enumerate(opts):
        print(f"  [{j}] {_label(op, names)}")

    import run_r15_greedyv_on_r4f as r15
    from deepmew.rl.metadata import normalized_deck_hash

    class NS:
        pass

    a = NS()
    a.r4f_checkpoint = str(r15.DEFAULT_R4F)
    a.construction_r4a_checkpoint = str(ROOT / ".research/unified_r4/R4-A/selected_model.pt")
    a.value_checkpoint = str(args.value_checkpoint) if args.value_checkpoint else str(r15.DEFAULT_VALUE)
    a.deck_policy = "reference_abomasnow"
    a.worlds = int(args.worlds)
    a.shortlist_k = int(args.shortlist_k)
    a.budget_ms = float(args.budget_ms)
    a.override_margin = 0.0
    a.seed = int(args.seed)

    agent = r15.build_greedyv_on_r4f(a)
    agent.deck = list(bd27)
    agent.shortlist_policy.deck = list(bd27)
    agent.deck_hash = normalized_deck_hash(bd27)
    agent.shortlist_policy.deck_hash = agent.deck_hash
    if hasattr(agent, "candidate_decks"):
        agent.candidate_decks = [list(starmie)]
    agent.reset(int(args.seed))

    dec = agent.decide(obs)
    scored = getattr(dec, "scored", {}) or {}
    print("\n=== SEARCH RESULT ===")
    print("source:", getattr(dec, "source", None), "| selection:", getattr(dec, "selection", None))
    for j in sorted(scored, key=lambda k: -scored[k]):
        star = "  <-- agent's actual choice" if chose and j in chose else ""
        print(f"  branch[{j}] {_label(opts[j], names):40s} win_prob={scored[j]:.4f}{star}")
    st = getattr(agent, "stats", None)
    if st and hasattr(st, "as_dict"):
        s = st.as_dict()
        print(f"stats: search_begin_calls={s.get('search_begin_calls')} "
              f"decision_ms_mean={s.get('decision_ms_mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
