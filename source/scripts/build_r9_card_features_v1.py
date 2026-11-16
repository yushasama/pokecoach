from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cg.api import all_card_data


SCHEMA_VERSION = "deepmew.r9.card_features_v1.v0"
DEFAULT_EN = ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
DEFAULT_V0 = ROOT / ".research" / "r6" / "lane2_mining" / "card_features_v0.csv"
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane1_reader"
PILOT_DECK = ROOT / ".research" / "decks" / "pilot_sample_abomasnow_v0" / "decklist.csv"

TAG_COLUMNS = (
    "second_order_any",
    "self_mill_flag",
    "discard_pile_scaling_flag",
    "energy_recycle_shuffle_back_flag",
    "stadium_enable_flag",
    "free_switch_flag",
    "draw_flag",
    "search_flag",
    "prize_swing_flag",
    "self_discard_cost_flag",
    "heal_loop_flag",
    "own_board_count_scaling_flag",
    "coin_flip_dependence_flag",
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean(value: Any) -> str:
    text = str(value or "")

    return "" if text in {"n/a", "nan", "None"} else text


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _parse_damage(value: str) -> int:
    numbers = [int(match) for match in re.findall(r"\d+", value or "")]

    return max(numbers) if numbers else 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_deck(path: Path) -> list[int]:
    return [int(token) for token in path.read_text(encoding="utf-8").split()]


def _tag_attack(row: dict[str, str]) -> dict[str, bool]:
    text = _lower(row.get("Effect Explanation"))
    damage = _clean(row.get("Damage"))
    stage = _lower(row.get("Stage (Pokemon)/Type (Energy and Trainer)") or row.get("Stage (PokÃ©mon)/Type (Energy and Trainer)"))
    rule = _lower(row.get("Rule"))
    card_name = _lower(row.get("Card Name"))
    is_stadium = "stadium" in stage or "stadium" in card_name
    self_mill = bool(re.search(r"discard\s+the\s+top\s+\d+\s+cards?\s+of\s+your\s+deck", text))
    discard_scaling = "discard pile" in text and "for each" in text and "basic" in text and "energy" in text
    energy_recycle = "shuffle" in text and "into your deck" in text
    free_switch = "switch" in text and ("active" in text or "benched" in text)
    draw = bool(re.search(r"\bdraw\b|\bdraws\b", text))
    search = "search your deck" in text or "look at the top" in text
    prize_swing = "mega pok" in rule or "mega pok" in text or "3 prize" in text or "prize card" in text and "more" in text
    self_discard = "discard" in text and any(phrase in text for phrase in ("from this pokemon", "from this pokémon", "from your hand", "attached to this"))
    heal = "heal" in text or "remove" in text and "damage" in text
    own_board_scaling = "for each of your" in text and any(phrase in text for phrase in ("pokemon in play", "pokémon in play", "benched", "on your bench"))
    coin_flip = "flip a coin" in text or "for each heads" in text
    x_scaling = "×" in damage or "x" in damage.lower() or "for each" in text
    stadium_enable = is_stadium or ("once during each player's turn" in text or "once during each player’s turn" in text)
    tags = {
        "self_mill_flag": self_mill,
        "discard_pile_scaling_flag": discard_scaling,
        "energy_recycle_shuffle_back_flag": energy_recycle,
        "stadium_enable_flag": stadium_enable,
        "free_switch_flag": free_switch,
        "draw_flag": draw,
        "search_flag": search,
        "prize_swing_flag": prize_swing,
        "self_discard_cost_flag": self_discard and not self_mill,
        "heal_loop_flag": heal,
        "own_board_count_scaling_flag": own_board_scaling,
        "coin_flip_dependence_flag": coin_flip,
        "x_scaling_flag": x_scaling,
    }
    tags["second_order_any"] = any(tags[column] for column in TAG_COLUMNS if column != "second_order_any")

    return tags


def _group_en_rows(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[int(row["Card ID"])].append(row)
        except (KeyError, TypeError, ValueError):
            continue

    return dict(grouped)


def _cg_card_map() -> dict[int, Any]:
    return {int(card.cardId): card for card in all_card_data()}


def _card_row(card_id: int, rows: list[dict[str, str]], cg_cards: dict[int, Any], v0_by_id: dict[int, dict[str, str]]) -> dict[str, str]:
    first = rows[0]
    attack_tags = [_tag_attack(row) for row in rows]
    tags = {column: any(tag.get(column, False) for tag in attack_tags) for column in TAG_COLUMNS}
    x_scaling = any(tag.get("x_scaling_flag", False) for tag in attack_tags)
    move_names = [_clean(row.get("Move Name")) for row in rows if _clean(row.get("Move Name"))]
    effect_texts = [_clean(row.get("Effect Explanation")) for row in rows if _clean(row.get("Effect Explanation"))]
    damages = [_parse_damage(_clean(row.get("Damage"))) for row in rows]
    cg_card = cg_cards.get(card_id)
    v0 = v0_by_id.get(card_id, {})
    rule = _clean(first.get("Rule"))
    stage = _clean(first.get("Stage (Pokemon)/Type (Energy and Trainer)") or first.get("Stage (PokÃ©mon)/Type (Energy and Trainer)"))
    card_type = _clean(first.get("Category")) or _clean(first.get("Type")) or _clean(v0.get("card_type"))
    row = {
        "card_id": str(card_id),
        "card_name": _clean(first.get("Card Name")),
        "card_type": card_type,
        "subtype": stage,
        "stage": stage,
        "hp": _clean(first.get("HP")),
        "rule": rule,
        "prize_value": "3" if cg_card is not None and bool(getattr(cg_card, "megaEx", False)) else ("2" if cg_card is not None and bool(getattr(cg_card, "ex", False)) else "1"),
        "cost_vector": ";".join(_clean(row.get("Cost")) for row in rows if _clean(row.get("Cost"))),
        "base_damage_max": str(max(damages) if damages else 0),
        "x_scaling_flag": _bool_text(x_scaling),
        "move_names": "; ".join(move_names),
        "effect_text_short": " | ".join(effect_texts)[:400],
        "cg_present": _bool_text(cg_card is not None),
        "cg_attacks": ";".join(str(value) for value in getattr(cg_card, "attacks", []) or []),
    }
    for column in TAG_COLUMNS:
        row[column] = _bool_text(tags[column])

    return row


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _coverage(rows: list[dict[str, str]], attack_rows: list[dict[str, str]], deck_ids: list[int], cg_cards: dict[int, Any]) -> dict[str, Any]:
    card_count = len(rows)
    attack_count = len([row for row in attack_rows if _clean(row.get("Move Name"))])
    tag_stats = []
    for column in TAG_COLUMNS:
        card_hits = sum(1 for row in rows if row[column] == "True")
        attack_hits = sum(1 for row in attack_rows if _tag_attack(row).get(column, False))
        tag_stats.append(
            {
                "tag": column,
                "card_hits": card_hits,
                "card_fraction": round(card_hits / card_count, 6) if card_count else None,
                "attack_hits": attack_hits,
                "attack_fraction": round(attack_hits / attack_count, 6) if attack_count else None,
            }
        )
    deck_missing_cg = sorted(card_id for card_id in set(deck_ids) if card_id not in cg_cards)
    deck_missing_v1 = sorted(card_id for card_id in set(deck_ids) if card_id not in {int(row["card_id"]) for row in rows})

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "cards_total": card_count,
        "attack_rows_total": attack_count,
        "second_order_cards": sum(1 for row in rows if row["second_order_any"] == "True"),
        "second_order_card_fraction": round(sum(1 for row in rows if row["second_order_any"] == "True") / card_count, 6) if card_count else None,
        "tag_stats": tag_stats,
        "pilot_deck_unique_cards": len(set(deck_ids)),
        "pilot_deck_missing_from_cg": deck_missing_cg,
        "pilot_deck_missing_from_v1": deck_missing_v1,
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R9 Card Features v1 Coverage",
        "",
        f"- schema: `{payload['schema_version']}`",
        f"- cards_total: `{payload['cards_total']}`",
        f"- attack_rows_total: `{payload['attack_rows_total']}`",
        f"- second_order_cards: `{payload['second_order_cards']}`",
        f"- second_order_card_fraction: `{payload['second_order_card_fraction']}`",
        f"- pilot_deck_missing_from_cg: `{payload['pilot_deck_missing_from_cg']}`",
        f"- pilot_deck_missing_from_v1: `{payload['pilot_deck_missing_from_v1']}`",
        "",
        "| tag | card hits | card fraction | attack hits | attack fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["tag_stats"]:
        lines.append(
            f"| `{row['tag']}` | {row['card_hits']} | {row['card_fraction']} | {row['attack_hits']} | {row['attack_fraction']} |"
        )

    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    en_rows = _read_csv(Path(args.en_card_data))
    v0_rows = _read_csv(Path(args.card_features_v0)) if Path(args.card_features_v0).exists() else []
    v0_by_id = {int(row["card_id"]): row for row in v0_rows if row.get("card_id")}
    grouped = _group_en_rows(en_rows)
    cg_cards = _cg_card_map()
    rows = [_card_row(card_id, card_rows, cg_cards, v0_by_id) for card_id, card_rows in sorted(grouped.items())]
    fields = [
        "card_id",
        "card_name",
        "card_type",
        "subtype",
        "stage",
        "hp",
        "rule",
        "prize_value",
        "cost_vector",
        "base_damage_max",
        "x_scaling_flag",
        *TAG_COLUMNS,
        "move_names",
        "cg_attacks",
        "cg_present",
        "effect_text_short",
    ]
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "card_features_v1.csv", rows, fields)
    deck_ids = _read_deck(Path(args.pilot_deck))
    payload = _coverage(rows, en_rows, deck_ids, cg_cards)
    (out_dir / "card_features_v1_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "card_features_v1_coverage.md").write_text(_render_report(payload), encoding="utf-8")

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build R9 card_features_v1 with second-order tags.")
    parser.add_argument("--en-card-data", default=str(DEFAULT_EN))
    parser.add_argument("--card-features-v0", default=str(DEFAULT_V0))
    parser.add_argument("--pilot-deck", default=str(PILOT_DECK))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    payload = build(args)
    print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
