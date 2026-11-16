from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r9_r4f_retrain as r4f
import run_unified_r4_closeout_delta as closeout
from deepmew.real_meta_arena.decklists import expand_card_counts, validate_decklist
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup


SCHEMA_VERSION = "deepmew.r12.deck_pilot_matrix.v0"
DEFAULT_R4F = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"
DEFAULT_POLICY_ID = "top_player_replay"


class PilotPolicyAdapter:
    def __init__(self, policy: Any):
        self.policy = policy
        self.label = str(policy.label)
        self.deck = list(policy.deck)

    def bind_deck(self, deck: list[int], deck_hash: str, policy_id: str) -> None:
        self.deck = list(deck)
        self.policy.deck = list(deck)
        self.policy.deck_hash = str(deck_hash)
        self.policy.policy_id = str(policy_id)

    def select(self, obs: dict[str, Any], rng: Any) -> list[int]:
        _ = rng

        return self.policy.select(obs)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _load_deck_meta(deck_zoo_dir: Path) -> dict[str, dict[str, str]]:
    deck_zoo_csv = deck_zoo_dir / "deck_zoo.csv"
    if not deck_zoo_csv.exists():
        return {}

    return {row["deck_hash"]: row for row in _read_csv(deck_zoo_csv)}


def _find_card_catalog(deck_zoo_dir: Path) -> Path | None:
    runs_dir = deck_zoo_dir.parents[2] if len(deck_zoo_dir.parents) >= 3 else deck_zoo_dir.parent
    candidates = sorted((runs_dir / "meta_kg").glob("*/*/card_catalog.csv"))
    if not candidates:
        return None

    return candidates[-1]


def _load_card_names(deck_zoo_dir: Path, explicit_path: str | None) -> dict[str, str]:
    path = Path(explicit_path) if explicit_path else _find_card_catalog(deck_zoo_dir)
    if path is None or not path.exists():
        return {}
    names: dict[str, str] = {}
    for row in _read_csv(path):
        names[str(row.get("card_id") or "")] = str(row.get("card_name") or "")

    return names


def _name_counts(card_counts: dict[str, Any], card_names: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for card_id, count in card_counts.items():
        name = card_names.get(str(card_id), str(card_id))
        out[name] = out.get(name, 0) + int(count)

    return out


def _style_label(archetype: str, variant_label: str, names: dict[str, int]) -> str:
    has = names.__contains__
    if has("Staryu") or has("Mega Starmie ex"):
        return "Mega Starmie/Staryu spread"
    if has("Crustle") and has("Mega Kangaskhan ex"):
        return "Crustle/Kangaskhan control"
    if has("Dragapult ex"):
        return "Latias/Dragapult stage2 spread"
    if "Alakazam" in archetype or has("Alakazam"):
        return "Alakazam hand/control"
    if "Marnie's Grimmsnarl ex" in archetype:
        return "Marnie's Grimmsnarl tempo"
    if "Dudunsparce" in archetype:
        return "Dudunsparce engine"
    if has("Archaludon ex"):
        return "Archaludon metal"
    if has("Team Rocket's Mewtwo ex"):
        return "Rocket Mewtwo"

    return variant_label or archetype or "Unknown"


def _load_decks(deck_zoo_dir: Path, *, limit: int | None, card_names: dict[str, str]) -> list[dict[str, Any]]:
    rows = _read_jsonl(deck_zoo_dir / "deck_zoo_decklists.jsonl")
    meta = _load_deck_meta(deck_zoo_dir)
    decks: list[dict[str, Any]] = []
    for row in rows:
        deck_hash = str(row["deck_hash"])
        info = meta.get(deck_hash, {})
        card_counts = dict(row.get("card_counts") or {})
        names = _name_counts(card_counts, card_names)
        archetype = str(row.get("archetype") or "Unknown")
        variant_label = info.get("variant_label") or archetype
        decks.append(
            {
                "deck_hash": deck_hash,
                "archetype": archetype,
                "variant_label": variant_label,
                "playstyle": _style_label(archetype, variant_label, names),
                "split": str(row.get("split") or ""),
                "is_canonical": bool(row.get("is_canonical")),
                "card_ids": list(expand_card_counts(card_counts)),
                "name_counts": names,
                "artifact_count": info.get("artifact_count", ""),
                "episode_count": info.get("episode_count", ""),
                "representative_player_ids": info.get("representative_player_ids", ""),
                "representative_episode_ids": info.get("representative_episode_ids", ""),
            }
        )
    decks.sort(key=lambda deck: (deck["playstyle"], deck["archetype"], deck["deck_hash"]))
    if limit is not None:
        decks = decks[:limit]

    return decks


def _validate_decks(decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_decks: list[dict[str, Any]] = []
    for deck in decks:
        validation = validate_decklist(
            type(
                "PilotDeck",
                (),
                {
                    "deck_hash": deck["deck_hash"],
                    "archetype": deck["archetype"],
                    "variant_label": deck["variant_label"],
                    "split": deck["split"],
                    "card_ids": tuple(deck["card_ids"]),
                    "card_count": len(deck["card_ids"]),
                    "is_standard_size": len(deck["card_ids"]) == 60,
                },
            )()
        )
        payload = asdict(validation)
        payload["playstyle"] = deck["playstyle"]
        rows.append(payload)
        if validation.valid:
            valid_decks.append(deck)
    decks[:] = valid_decks

    return rows


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        games = sum(int(row["games"]) for row in group)
        completed = sum(int(row["completed"]) for row in group)
        wins = sum(int(row["wins"]) for row in group)
        losses = sum(int(row["losses"]) for row in group)
        draws = sum(int(row["draws"]) for row in group)
        crashes = sum(int(row["crashes"]) for row in group)
        timeouts = sum(int(row["timeouts"]) for row in group)
        avg_turns_num = sum(float(row["avg_turns"]) * int(row["completed"]) for row in group)
        decided = wins + losses
        payload = {name: value for name, value in zip(group_keys, key, strict=True)}
        payload.update(
            {
                "decks": len({row["pilot_deck_hash"] for row in group}),
                "games": games,
                "completed": completed,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winrate": round(wins / decided, 6) if decided else None,
                "crashes": crashes,
                "timeouts": timeouts,
                "avg_turns": round(avg_turns_num / completed, 3) if completed else None,
            }
        )
        out.append(payload)
    out.sort(key=lambda row: tuple(str(row[key]) for key in group_keys))

    return out


def _policy_delta_rows(by_deck: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in by_deck:
        grouped[str(row["pilot_deck_hash"])][str(row["policy"])] = row
    out: list[dict[str, Any]] = []
    for deck_hash, policies in sorted(grouped.items()):
        r4f_row = policies.get("R4-F_reweighted")
        v1b_row = policies.get("v1b_incumbent")
        if not r4f_row or not v1b_row:
            continue
        out.append(
            {
                "pilot_deck_hash": deck_hash,
                "playstyle": r4f_row["playstyle"],
                "archetype": r4f_row["archetype"],
                "variant_label": r4f_row["variant_label"],
                "r4f_winrate": r4f_row["winrate"],
                "v1b_winrate": v1b_row["winrate"],
                "r4f_minus_v1b": round(float(r4f_row["winrate"]) - float(v1b_row["winrate"]), 6),
                "r4f_wins_losses": f"{r4f_row['wins']}-{r4f_row['losses']}",
                "v1b_wins_losses": f"{v1b_row['wins']}-{v1b_row['losses']}",
            }
        )
    out.sort(key=lambda row: (float(row["r4f_winrate"]), float(row["r4f_minus_v1b"]), row["pilot_deck_hash"]))

    return out


def _render_report(payload: dict[str, Any], by_deck: list[dict[str, Any]], by_playstyle: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    r4f_rows = [row for row in by_deck if row["policy"] == "R4-F_reweighted"]
    strongest = sorted(r4f_rows, key=lambda row: (-float(row["winrate"]), row["pilot_deck_hash"]))[:8]
    weakest = sorted(r4f_rows, key=lambda row: (float(row["winrate"]), row["pilot_deck_hash"]))[:8]

    lines = [
        "# R12 Deck Pilot Matrix",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- deck_zoo_dir: `{payload['inputs']['deck_zoo_dir']}`",
        f"- games_per_deck: `{payload['config']['games_per_deck']}`",
        f"- opponent_policy: `{payload['config']['opponent_policy']}`",
        f"- opponent_mode: `{payload['config']['opponent_mode']}`",
        f"- pilot_policy_id: `{payload['config']['pilot_policy_id']}`",
        f"- valid_decks: `{payload['counts']['valid_decks']}` / `{payload['counts']['input_decks']}`",
        "",
        "## Read This Correctly",
        "",
        "- This is a pilot-skill diagnostic: DeepMew is the pilot of each copied decklist.",
        "- `opponent_mode=same_deck` means the local opponent is `first-legal` piloting the same decklist, so results are pilot quality within that playstyle, not metagame matchup strength.",
        "- Root `deck.csv`, root `main.py`, `cg/`, `deepmew/submissions/`, and default `v2.4` are not modified.",
        "",
        "## Overall",
        "",
        "| policy | decks | games | wins-losses | winrate | crashes | timeouts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_policy"]:
        lines.append(
            f"| {row['policy']} | {row['decks']} | {row['games']} | {row['wins']}-{row['losses']} | "
            f"{row['winrate']} | {row['crashes']} | {row['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## R4-F Strongest Pilot Decks",
            "",
            "| playstyle | archetype | deck_hash | winrate | wins-losses |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in strongest:
        lines.append(
            f"| {row['playstyle']} | {row['archetype']} | `{row['pilot_deck_hash']}` | "
            f"{row['winrate']} | {row['wins']}-{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## R4-F Weakest Pilot Decks",
            "",
            "| playstyle | archetype | deck_hash | winrate | wins-losses |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in weakest:
        lines.append(
            f"| {row['playstyle']} | {row['archetype']} | `{row['pilot_deck_hash']}` | "
            f"{row['winrate']} | {row['wins']}-{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## By Playstyle",
            "",
            "| policy | playstyle | decks | winrate | wins-losses | crashes | timeouts |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_playstyle:
        lines.append(
            f"| {row['policy']} | {row['playstyle']} | {row['decks']} | {row['winrate']} | "
            f"{row['wins']}-{row['losses']} | {row['crashes']} | {row['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## R4-F Minus v1b",
            "",
            "| playstyle | deck_hash | R4-F | v1b | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['playstyle']} | `{row['pilot_deck_hash']}` | {row['r4f_winrate']} | "
            f"{row['v1b_winrate']} | {row['r4f_minus_v1b']} |"
        )

    return "\n".join(lines) + "\n"


def _opponent_deck_for_mode(mode: str, pilot_deck: list[int], reference_deck: list[int]) -> list[int]:
    if mode == "same_deck":
        return list(pilot_deck)
    if mode == "reference_deck":
        return list(reference_deck)

    raise ValueError(f"unknown opponent mode {mode!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deck_zoo_dir = Path(args.deck_zoo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_names = _load_card_names(deck_zoo_dir, args.card_catalog)
    decks = _load_decks(deck_zoo_dir, limit=args.limit_decks, card_names=card_names)
    validation_rows = _validate_decks(decks)
    policies = [
        PilotPolicyAdapter(r4f.R4FLivePolicy("R4-F_reweighted", Path(args.r4f), args.deck_policy)),
        PilotPolicyAdapter(closeout.R4ObsPolicy("v1b_incumbent", Path(args.v1b), args.deck_policy)),
    ]
    reference_deck = list(policies[0].deck)
    opponent = SpecAgent.from_str(args.opponent_policy)
    by_deck: list[dict[str, Any]] = []
    seed = int(args.seed)
    for policy in policies:
        for index, deck in enumerate(decks):
            policy.bind_deck(deck["card_ids"], deck["deck_hash"], args.pilot_policy_id)
            opp_deck = _opponent_deck_for_mode(args.opponent_mode, deck["card_ids"], reference_deck)
            summary = run_real_meta_matchup(
                policy,
                opponent,
                list(deck["card_ids"]),
                opp_deck,
                games=int(args.games),
                bucket=f"pilot_{args.opponent_mode}",
                opp_deck_hash=deck["deck_hash"] if args.opponent_mode == "same_deck" else "reference_deck",
                opp_archetype=deck["archetype"] if args.opponent_mode == "same_deck" else "reference_abomasnow",
                seed=seed + index + (100_000 if policy.label.startswith("v1b") else 0),
                max_steps=int(args.max_steps),
            )
            decided = summary.our_wins + summary.opp_wins
            row = {
                "policy": policy.label,
                "opponent_policy": args.opponent_policy,
                "opponent_mode": args.opponent_mode,
                "pilot_policy_id": args.pilot_policy_id,
                "pilot_deck_hash": deck["deck_hash"],
                "opponent_deck_hash": deck["deck_hash"] if args.opponent_mode == "same_deck" else "reference_deck",
                "playstyle": deck["playstyle"],
                "archetype": deck["archetype"],
                "variant_label": deck["variant_label"],
                "split": deck["split"],
                "games": summary.games,
                "completed": summary.completed,
                "wins": summary.our_wins,
                "losses": summary.opp_wins,
                "draws": summary.draws,
                "winrate": round(summary.our_wins / decided, 6) if decided else None,
                "crashes": summary.crashes,
                "timeouts": summary.timeouts,
                "avg_turns": round(summary.avg_turns, 3),
                "elapsed_seconds": round(summary.elapsed_seconds, 3),
                "artifact_count": deck["artifact_count"],
                "episode_count": deck["episode_count"],
                "representative_player_ids": deck["representative_player_ids"],
                "representative_episode_ids": deck["representative_episode_ids"],
                "name_counts_json": _json_cell(deck["name_counts"]),
                "errors": "; ".join(summary.errors),
            }
            by_deck.append(row)
            print(
                f"[{policy.label}] pilot={deck['playstyle']} {deck['deck_hash']} "
                f"wr={row['winrate']} safety={row['crashes']}/{row['timeouts']}",
                flush=True,
            )
            if args.stop_on_safety and (summary.crashes or summary.timeouts):
                raise RuntimeError(
                    f"safety failure for {policy.label} {deck['deck_hash']}: "
                    f"crashes={summary.crashes}, timeouts={summary.timeouts}, errors={summary.errors[:2]}"
                )
    by_policy = _aggregate(by_deck, ["policy"])
    by_playstyle = _aggregate(by_deck, ["policy", "playstyle"])
    by_archetype = _aggregate(by_deck, ["policy", "archetype"])
    deltas = _policy_delta_rows(by_deck)
    safety = {
        "crashes": sum(int(row["crashes"]) for row in by_deck),
        "timeouts": sum(int(row["timeouts"]) for row in by_deck),
        "draws": sum(int(row["draws"]) for row in by_deck),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "deck_zoo_dir": str(deck_zoo_dir),
            "card_catalog": str(args.card_catalog or _find_card_catalog(deck_zoo_dir) or ""),
            "r4f": str(args.r4f),
            "v1b": str(args.v1b),
        },
        "config": {
            "games_per_deck": int(args.games),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "deck_policy": args.deck_policy,
            "pilot_policy_id": args.pilot_policy_id,
            "opponent_policy": args.opponent_policy,
            "opponent_mode": args.opponent_mode,
        },
        "counts": {
            "input_decks": len(validation_rows),
            "valid_decks": len(decks),
            "arena_rows": len(by_deck),
        },
        "safety": safety,
        "by_policy": by_policy,
        "by_playstyle": by_playstyle,
        "by_archetype": by_archetype,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(out_dir / "deck_validation.csv", validation_rows)
    _write_csv(out_dir / "by_deck.csv", by_deck)
    _write_csv(out_dir / "by_playstyle.csv", by_playstyle)
    _write_csv(out_dir / "by_archetype.csv", by_archetype)
    _write_csv(out_dir / "policy_delta_by_deck.csv", deltas)
    (out_dir / "report.md").write_text(_render_report(payload, by_deck, by_playstyle, deltas), encoding="utf-8")
    print(json.dumps({"by_policy": by_policy, "safety": safety}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run R4-F/v1b as pilots of copied top-player decklists.")
    parser.add_argument("--deck-zoo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--v1b", default=str(DEFAULT_V1B))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--pilot-policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--opponent-policy", default="first-legal")
    parser.add_argument("--opponent-mode", choices=("same_deck", "reference_deck"), default="same_deck")
    parser.add_argument("--card-catalog")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-decks", type=int)
    parser.add_argument("--stop-on-safety", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
