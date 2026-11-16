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


SCHEMA_VERSION = "deepmew.r12.top_player_arena.v0"
DEFAULT_R4F = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"


class PolicyAdapter:
    def __init__(self, policy: Any):
        self.policy = policy
        self.label = str(policy.label)
        self.deck = list(policy.deck)

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


def _load_deck_meta(deck_zoo_dir: Path) -> dict[str, dict[str, str]]:
    deck_zoo_csv = deck_zoo_dir / "deck_zoo.csv"
    if not deck_zoo_csv.exists():
        return {}

    return {row["deck_hash"]: row for row in _read_csv(deck_zoo_csv)}


def _load_decks(deck_zoo_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    path = deck_zoo_dir / "deck_zoo_decklists.jsonl"
    rows = _read_jsonl(path)
    meta = _load_deck_meta(deck_zoo_dir)
    decks: list[dict[str, Any]] = []
    for row in rows:
        deck_hash = str(row["deck_hash"])
        info = meta.get(deck_hash, {})
        decks.append(
            {
                "deck_hash": deck_hash,
                "archetype": str(row.get("archetype") or "Unknown"),
                "variant_label": info.get("variant_label") or str(row.get("archetype") or "Unknown"),
                "split": str(row.get("split") or ""),
                "is_canonical": bool(row.get("is_canonical")),
                "card_ids": list(expand_card_counts(row.get("card_counts") or {})),
                "representative_player_ids": info.get("representative_player_ids", ""),
                "representative_episode_ids": info.get("representative_episode_ids", ""),
            }
        )
    decks.sort(key=lambda deck: (deck["archetype"], deck["deck_hash"]))
    if limit is not None:
        decks = decks[:limit]

    return decks


def _validate_decks(decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_decks: list[dict[str, Any]] = []
    for deck in decks:
        validation = validate_decklist(
            type(
                "ArenaDeck",
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
        rows.append(payload)
        if validation.valid:
            valid_decks.append(deck)
    decks[:] = valid_decks

    return rows


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


def _aggregate(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), str(row[group_key]))].append(row)
    out: list[dict[str, Any]] = []
    for (policy, key), group in groups.items():
        games = sum(int(row["games"]) for row in group)
        completed = sum(int(row["completed"]) for row in group)
        wins = sum(int(row["wins"]) for row in group)
        losses = sum(int(row["losses"]) for row in group)
        draws = sum(int(row["draws"]) for row in group)
        crashes = sum(int(row["crashes"]) for row in group)
        timeouts = sum(int(row["timeouts"]) for row in group)
        avg_turns_num = sum(float(row["avg_turns"]) * int(row["completed"]) for row in group)
        decided = wins + losses
        out.append(
            {
                "policy": policy,
                group_key: key,
                "decks": len(group),
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
    out.sort(key=lambda row: (row["policy"], row.get("winrate") is None, row.get("winrate") or 0.0, row[group_key]))

    return out


def _render_report(payload: dict[str, Any], by_deck: list[dict[str, Any]], by_archetype: list[dict[str, Any]]) -> str:
    r4f_rows = [row for row in by_deck if row["policy"] == "R4-F_reweighted"]
    strongest = sorted(r4f_rows, key=lambda row: (-(row["winrate"] or 0), row["deck_hash"]))[:8]
    weakest = sorted(r4f_rows, key=lambda row: ((row["winrate"] or 0), row["deck_hash"]))[:8]

    lines = [
        "# R12 Fresh Top-Player Arena",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- deck_zoo_dir: `{payload['inputs']['deck_zoo_dir']}`",
        f"- games_per_deck: `{payload['config']['games_per_deck']}`",
        f"- opponent_policy: `{payload['config']['opponent_policy']}`",
        f"- canonical_decks_valid: `{payload['counts']['valid_decks']}` / `{payload['counts']['input_decks']}`",
        "",
        "## Read This Correctly",
        "",
        "- This is a deck-stress arena: top-player decklists are real fresh Kaggle lists, but the opponent pilot is the configured local policy, not the original ladder bot.",
        "- Both seats are alternated. Safety failure is crashes + timeouts; illegal action checks are handled by the engine/selection guards.",
        "- Root `deck.csv`, root `main.py`, `cg/`, `deepmew/submissions/`, and default `v2.4` are not modified.",
        "",
        "## R4-F Strongest Deck Matchups",
        "",
        "| archetype | deck_hash | winrate | wins-losses | crashes | timeouts |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in strongest:
        lines.append(
            f"| {row['archetype']} | `{row['deck_hash']}` | {row['winrate']} | "
            f"{row['wins']}-{row['losses']} | {row['crashes']} | {row['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## R4-F Weakest Deck Matchups",
            "",
            "| archetype | deck_hash | winrate | wins-losses | crashes | timeouts |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in weakest:
        lines.append(
            f"| {row['archetype']} | `{row['deck_hash']}` | {row['winrate']} | "
            f"{row['wins']}-{row['losses']} | {row['crashes']} | {row['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## By Archetype",
            "",
            "| policy | archetype | decks | winrate | wins-losses | crashes | timeouts |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_archetype:
        lines.append(
            f"| {row['policy']} | {row['archetype']} | {row['decks']} | {row['winrate']} | "
            f"{row['wins']}-{row['losses']} | {row['crashes']} | {row['timeouts']} |"
        )

    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    deck_zoo_dir = Path(args.deck_zoo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decks = _load_decks(deck_zoo_dir, args.limit_decks)
    validation_rows = _validate_decks(decks)
    policies = [
        PolicyAdapter(r4f.R4FLivePolicy("R4-F_reweighted", Path(args.r4f), args.deck_policy)),
        PolicyAdapter(closeout.R4ObsPolicy("v1b_incumbent", Path(args.v1b), args.deck_policy)),
    ]
    opponent = SpecAgent.from_str(args.opponent_policy)
    by_deck: list[dict[str, Any]] = []
    seed = int(args.seed)
    for policy in policies:
        for index, deck in enumerate(decks):
            summary = run_real_meta_matchup(
                policy,
                opponent,
                list(policy.deck),
                list(deck["card_ids"]),
                games=int(args.games),
                bucket="fresh_top_player",
                opp_deck_hash=deck["deck_hash"],
                opp_archetype=deck["archetype"],
                seed=seed + index + (100_000 if policy.label.startswith("v1b") else 0),
                max_steps=int(args.max_steps),
            )
            decided = summary.our_wins + summary.opp_wins
            row = {
                "policy": policy.label,
                "opponent_policy": args.opponent_policy,
                "deck_hash": deck["deck_hash"],
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
                "errors": "; ".join(summary.errors),
                "representative_player_ids": deck["representative_player_ids"],
                "representative_episode_ids": deck["representative_episode_ids"],
            }
            by_deck.append(row)
            print(
                f"[{policy.label}] {deck['archetype']} {deck['deck_hash']} "
                f"wr={row['winrate']} safety={row['crashes']}/{row['timeouts']}",
                flush=True,
            )
    by_archetype = _aggregate(by_deck, "archetype")
    by_policy = _aggregate(by_deck, "opponent_policy")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "deck_zoo_dir": str(deck_zoo_dir),
            "r4f": str(args.r4f),
            "v1b": str(args.v1b),
        },
        "config": {
            "games_per_deck": int(args.games),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "deck_policy": args.deck_policy,
            "opponent_policy": args.opponent_policy,
        },
        "counts": {
            "input_decks": len(validation_rows),
            "valid_decks": len(decks),
            "arena_rows": len(by_deck),
        },
        "by_policy": by_policy,
        "by_archetype": by_archetype,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(out_dir / "deck_validation.csv", validation_rows)
    _write_csv(out_dir / "by_deck.csv", by_deck)
    _write_csv(out_dir / "by_archetype.csv", by_archetype)
    (out_dir / "report.md").write_text(_render_report(payload, by_deck, by_archetype), encoding="utf-8")
    print(json.dumps(payload["by_policy"], indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run R4-F/v1b against fresh top-player deck-zoo decklists.")
    parser.add_argument("--deck-zoo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--v1b", default=str(DEFAULT_V1B))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--opponent-policy", default="first-legal")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-decks", type=int)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
