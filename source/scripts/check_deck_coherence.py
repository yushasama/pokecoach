"""Deck coherence checker: flag dead cards (unplayable evolutions) + energy mismatch.

A deck is INCOHERENT when it can't actually deploy/fuel its Pokemon:
  - a Stage 1 with no Basic it evolves from (and no way to get it) is a DEAD card;
  - a Stage 2 needs its Stage 1 (+Basic) OR Rare Candy (+Basic) or it's DEAD;
  - attacker energy types with no matching energy source in the deck are UNFUELABLE.
Dead cards silently shrink a 60-card deck; no piloting/RL fixes them. Offline/eval-only.

    python scripts/check_deck_coherence.py --deck <deck.csv | bundle-dir>
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD_CSV = ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
RARE_CANDY_NAMES = {"rare candy"}
ENERGY_SYMBOLS = re.compile(r"\{([WLFPGRDMCN])\}")  # water/lightning/fighting/psychic/grass/fire/dark/metal/colorless


def _load_cards() -> dict[str, dict]:
    cards: dict[str, dict] = {}
    attacks: dict[str, list[str]] = {}
    for r in csv.reader(open(CARD_CSV, encoding="utf-8")):
        if not r or r[0] == "Card ID":
            continue
        cid = r[0]
        # attack cost symbols live in the move rows (col ~ energy-cost text); collect all {X}
        joined = " ".join(r)
        attacks.setdefault(cid, []).extend(ENERGY_SYMBOLS.findall(joined))
        if cid not in cards:
            cards[cid] = {"name": r[1], "stage": r[4], "rule": r[5] if len(r) > 5 else "",
                          "cat": r[6] if len(r) > 6 else "", "prev": r[7] if len(r) > 7 else "n/a"}
    for cid, c in cards.items():
        c["attack_energy"] = set(attacks.get(cid, []))
    return cards


def _resolve_deck(path: Path) -> list[str]:
    p = path if path.suffix == ".csv" else path / "deck.csv"
    return [x for x in p.read_text(encoding="utf-8").split() if x.strip()]


def check(deck_ids: list[str], cards: dict[str, dict]) -> dict:
    counts = Counter(deck_ids)
    name_to_ids = {}
    for cid in counts:
        nm = cards.get(cid, {}).get("name", "").lower()
        name_to_ids.setdefault(nm, set()).add(cid)
    present_names = {cards.get(cid, {}).get("name", "").lower() for cid in counts}
    has_rare_candy = bool(present_names & RARE_CANDY_NAMES)

    dead, unfuelable, energy_syms = [], [], set()
    # collect energy the deck can PROVIDE (basic energy cards + special energy)
    provided = set()
    for cid in counts:
        c = cards.get(cid, {})
        cat = c.get("cat", "")
        if "Energy" in cat:
            provided |= ENERGY_SYMBOLS.findall(c.get("name", "")) and set(ENERGY_SYMBOLS.findall(c.get("name", ""))) or set()
            # basic {X} energy name carries its symbol; special energy is flexible-ish
            if "Special" in c.get("stage", "") or "Special" in cat:
                provided |= {"*"}  # special energy: treat as wildcard-ish
    for cid, n in counts.items():
        c = cards.get(cid, {})
        cat, stage, prev = c.get("cat", ""), c.get("stage", ""), (c.get("prev") or "n/a").lower()
        is_pokemon = "Pok" in cat or "Pok" in stage
        if not is_pokemon:
            continue
        # evolution-line completeness
        if "Stage 1" in stage and prev not in ("n/a", ""):
            if prev not in present_names:
                dead.append((cid, c["name"], n, f"Stage1 needs Basic '{prev}' (absent)"))
        elif "Stage 2" in stage and prev not in ("n/a", ""):
            has_stage1 = prev in present_names
            if not (has_stage1 or has_rare_candy):
                dead.append((cid, c["name"], n, f"Stage2 needs Stage1 '{prev}' or Rare Candy (both absent)"))
        # attacker fuel: does the deck provide the attack energy types?
        need = c.get("attack_energy", set()) - {"C", "N"}  # colorless is flexible
        if need and "*" not in provided and not (need & provided):
            unfuelable.append((cid, c["name"], n, f"needs {sorted(need)}; deck provides {sorted(provided - {'*'})}"))
        energy_syms |= need

    dead_slots = sum(n for _, _, n, _ in dead)
    return {"total": len(deck_ids), "dead": dead, "dead_slots": dead_slots,
            "unfuelable": unfuelable, "coherent": not dead}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True, nargs="+", help="deck.csv path(s) or bundle dir(s)")
    args = ap.parse_args(argv)
    cards = _load_cards()
    for d in args.deck:
        deck = _resolve_deck(Path(d))
        res = check(deck, cards)
        print(f"\n=== {d} ({res['total']} cards) — {'COHERENT' if res['coherent'] else 'INCOHERENT'} ===")
        if res["dead"]:
            print(f"  DEAD cards ({res['dead_slots']} slots):")
            for cid, nm, n, why in res["dead"]:
                print(f"    {n}x [{cid}] {nm}: {why}")
        if res["unfuelable"]:
            print("  possibly UNFUELABLE attackers:")
            for cid, nm, n, why in res["unfuelable"][:6]:
                print(f"    {n}x [{cid}] {nm}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
