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
from deepmew.real_meta_arena.decklists import expand_card_counts, validate_decklist
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup


SCHEMA_VERSION = "deepmew.r12.deck_choice_panel.v1"
DEFAULT_R4F = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"


class BoundR4FPolicy:
    def __init__(self, checkpoint: Path, deck_policy: str):
        self.policy = r4f.R4FLivePolicy("R4-F_reweighted", checkpoint, deck_policy)
        self.label = self.policy.label
        self.deck = list(self.policy.deck)

    def bind_deck(self, deck: list[int], deck_hash: str, policy_id: str, label: str) -> None:
        self.label = label
        self.deck = list(deck)
        self.policy.label = label
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


def _load_deck_meta(deck_zoo_dir: Path) -> dict[str, dict[str, str]]:
    path = deck_zoo_dir / "deck_zoo.csv"
    if not path.exists():
        return {}

    return {row["deck_hash"]: row for row in _read_csv(path)}


def _load_decks(deck_zoo_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = _read_jsonl(deck_zoo_dir / "deck_zoo_decklists.jsonl")
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
                "card_ids": list(expand_card_counts(row.get("card_counts") or {})),
                "representative_player_ids": info.get("representative_player_ids", ""),
                "representative_episode_ids": info.get("representative_episode_ids", ""),
            }
        )
    decks.sort(key=lambda deck: (deck["archetype"], deck["deck_hash"]))
    if limit is not None:
        decks = decks[:limit]

    return decks


def _validate_decks(decks: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for deck in decks:
        validation = validate_decklist(
            type(
                "ChoiceDeck",
                (),
                {
                    "deck_hash": deck["deck_hash"],
                    "archetype": deck.get("archetype", role),
                    "variant_label": deck.get("variant_label", role),
                    "split": deck.get("split", ""),
                    "card_ids": tuple(deck["card_ids"]),
                    "card_count": len(deck["card_ids"]),
                    "is_standard_size": len(deck["card_ids"]) == 60,
                },
            )()
        )
        payload = asdict(validation)
        payload["role"] = role
        rows.append(payload)
        if validation.valid:
            valid.append(deck)
    decks[:] = valid

    return rows


def _find_deck(decks_by_hash: dict[str, dict[str, Any]], deck_hash: str, *, role: str) -> dict[str, Any]:
    deck = decks_by_hash.get(deck_hash)
    if deck is None:
        raise ValueError(f"{role} deck hash not found in deck zoo: {deck_hash}")

    return dict(deck)


def _reference_abomasnow_deck(base_policy: BoundR4FPolicy) -> dict[str, Any]:
    return {
        "pilot_label": "reference_abomasnow",
        "deck_hash": base_policy.policy.deck_hash,
        "pilot_deck_hash": base_policy.policy.deck_hash,
        "pilot_policy_id": "reference_abomasnow",
        "archetype": "Reference Abomasnow",
        "variant_label": "Reference Abomasnow",
        "split": "reference",
        "card_ids": list(base_policy.deck),
        "representative_player_ids": "",
        "representative_episode_ids": "",
    }


def _pilot_specs(args: argparse.Namespace, base_policy: BoundR4FPolicy, decks_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def append_reference() -> None:
        key = ("reference_abomasnow", str(base_policy.policy.deck_hash))
        if key in seen:
            return
        seen.add(key)
        specs.append(_reference_abomasnow_deck(base_policy))

    def append_copied(deck: dict[str, Any]) -> None:
        key = ("copied", str(deck["deck_hash"]))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "pilot_label": f"{deck['archetype']}:{deck['deck_hash']}",
                "deck_hash": deck["deck_hash"],
                "pilot_deck_hash": deck["deck_hash"],
                "pilot_policy_id": args.copied_policy_id,
                "archetype": deck["archetype"],
                "variant_label": deck["variant_label"],
                "card_ids": list(deck["card_ids"]),
            }
        )

    for raw in args.pilot_deck:
        if raw == "reference_abomasnow":
            append_reference()
            continue
        if raw == "all":
            for deck in sorted(decks_by_hash.values(), key=lambda deck: (deck["archetype"], deck["deck_hash"])):
                append_copied(deck)
            continue
        deck = _find_deck(decks_by_hash, raw, role="pilot")
        append_copied(deck)
    if not specs:
        raise ValueError("at least one --pilot-deck is required")

    return specs


def _opponent_specs(args: argparse.Namespace, base_policy: BoundR4FPolicy, decks_by_hash: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    requested = list(args.opponent_deck or ["all"])

    def append_deck(deck: dict[str, Any]) -> None:
        deck_hash = str(deck["deck_hash"])
        if deck_hash in seen:
            return
        seen.add(deck_hash)
        specs.append(deck)

    for raw in requested:
        if raw == "reference_abomasnow":
            deck = _reference_abomasnow_deck(base_policy)
            deck["opponent_label"] = "reference_abomasnow"
            append_deck(deck)
            continue
        if raw == "all":
            for deck in sorted(decks_by_hash.values(), key=lambda item: (item["archetype"], item["deck_hash"])):
                append_deck(dict(deck))
            continue
        append_deck(_find_deck(decks_by_hash, raw, role="opponent"))
    if not specs:
        raise ValueError("at least one opponent deck is required")

    return specs


def _apply_pilot_shard(pilots: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--pilot-shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--pilot-shard-index must be in [0, --pilot-shard-count)")
    if shard_count == 1:
        return pilots

    return [pilot for index, pilot in enumerate(pilots) if index % shard_count == shard_index]


def _matchup_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["pilot_deck_hash"]),
        str(row["pilot_policy_id"]),
        str(row["opponent_deck_hash"]),
        str(row["opponent_policy"]),
        str(row["games"]),
    )


def _matchup_key_for(pilot: dict[str, Any], opponent: dict[str, Any], args: argparse.Namespace) -> tuple[str, str, str, str, str]:
    return (
        str(pilot["pilot_deck_hash"]),
        str(pilot["pilot_policy_id"]),
        str(opponent["deck_hash"]),
        str(args.opponent_policy),
        str(int(args.games)),
    )


def _aggregate_by_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pilot_label"])].append(row)
    out: list[dict[str, Any]] = []
    for label, group in grouped.items():
        games = sum(int(row["games"]) for row in group)
        completed = sum(int(row["completed"]) for row in group)
        wins = sum(int(row["wins"]) for row in group)
        losses = sum(int(row["losses"]) for row in group)
        draws = sum(int(row["draws"]) for row in group)
        crashes = sum(int(row["crashes"]) for row in group)
        timeouts = sum(int(row["timeouts"]) for row in group)
        avg_turns_num = sum(float(row["avg_turns"]) * int(row["completed"]) for row in group)
        decided = wins + losses
        first = group[0]
        out.append(
            {
                "pilot_label": label,
                "pilot_deck_hash": first["pilot_deck_hash"],
                "pilot_policy_id": first["pilot_policy_id"],
                "archetype": first["pilot_archetype"],
                "variant_label": first["pilot_variant_label"],
                "opponent_decks": len({row["opponent_deck_hash"] for row in group}),
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
    out.sort(key=lambda row: (-(row["winrate"] or 0.0), row["pilot_label"]))

    return out


def _write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    deck_zoo_dir: Path,
    opponent_deck_zoo_dir: Path,
    pilots_requested: int,
    pilots_available: int,
    pilots: list[dict[str, Any]],
    opponent_decks: list[dict[str, Any]],
    opponent_validation: list[dict[str, Any]],
    pilot_validation: list[dict[str, Any]],
    by_matchup: list[dict[str, Any]],
) -> dict[str, Any]:
    by_pilot = _aggregate_by_pilot(by_matchup)
    safety = {
        "crashes": sum(int(row["crashes"]) for row in by_matchup),
        "timeouts": sum(int(row["timeouts"]) for row in by_matchup),
        "draws": sum(int(row["draws"]) for row in by_matchup),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "deck_zoo_dir": str(deck_zoo_dir),
            "pilot_deck_zoo_dir": str(deck_zoo_dir),
            "opponent_deck_zoo_dir": str(opponent_deck_zoo_dir),
            "r4f": str(args.r4f),
        },
        "config": {
            "deck_policy": args.deck_policy,
            "copied_policy_id": args.copied_policy_id,
            "opponent_policy": args.opponent_policy,
            "games_per_opponent_deck": int(args.games),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "pilot_decks": list(args.pilot_deck),
            "opponent_decks": list(args.opponent_deck or ["all"]),
            "pilot_shard_index": int(args.pilot_shard_index),
            "pilot_shard_count": int(args.pilot_shard_count),
            "resume": bool(args.resume),
        },
        "counts": {
            "pilot_decks_requested": pilots_requested,
            "valid_pilot_decks_available": pilots_available,
            "valid_pilot_decks_in_shard": len(pilots),
            "input_opponent_decks": len(opponent_validation),
            "valid_opponent_decks": len(opponent_decks),
            "matchup_rows": len(by_matchup),
            "expected_matchup_rows": len(pilots) * len(opponent_decks),
            "complete": len(by_matchup) == len(pilots) * len(opponent_decks),
        },
        "safety": safety,
        "by_pilot": by_pilot,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(out_dir / "pilot_validation.csv", pilot_validation)
    _write_csv(out_dir / "opponent_validation.csv", opponent_validation)
    _write_csv(out_dir / "by_matchup.csv", by_matchup)
    _write_csv(out_dir / "by_pilot.csv", by_pilot)
    (out_dir / "report.md").write_text(_render_report(payload, by_pilot), encoding="utf-8")

    return payload


def _render_report(payload: dict[str, Any], by_pilot: list[dict[str, Any]]) -> str:
    lines = [
        "# R12 Direct Deck Choice Panel",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- pilot_deck_zoo_dir: `{payload['inputs']['pilot_deck_zoo_dir']}`",
        f"- opponent_deck_zoo_dir: `{payload['inputs']['opponent_deck_zoo_dir']}`",
        f"- opponent_policy: `{payload['config']['opponent_policy']}`",
        f"- games_per_opponent_deck: `{payload['config']['games_per_opponent_deck']}`",
        f"- opponent_decks: `{payload['counts']['valid_opponent_decks']}`",
        "",
        "## Read This Correctly",
        "",
        "- This is the direct deck-choice comparison: R4-F is always the pilot, and only its own deck changes.",
        "- Every candidate pilot deck plays the same requested opponent deck panel.",
        "- Opponent pilot is local `first-legal`, not the original Kaggle ladder bot.",
        "- Reference opponent names in this script are deck choices; the rebuilt reference labels are not distinct live bots.",
        "- Root `deck.csv`, root `main.py`, `cg/`, `deepmew/submissions/`, and default `v2.4` are not modified.",
        "",
        "## By Pilot Deck",
        "",
        "| pilot | policy_id | opponent decks | wins-losses | winrate | crashes | timeouts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_pilot:
        lines.append(
            f"| {row['pilot_label']} | `{row['pilot_policy_id']}` | {row['opponent_decks']} | "
            f"{row['wins']}-{row['losses']} | {row['winrate']} | {row['crashes']} | {row['timeouts']} |"
        )

    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    deck_zoo_dir = Path(args.deck_zoo)
    opponent_deck_zoo_dir = Path(args.opponent_deck_zoo or args.deck_zoo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decks_by_hash = {deck["deck_hash"]: deck for deck in _load_decks(deck_zoo_dir, None)}
    opponent_decks_by_hash = {deck["deck_hash"]: deck for deck in _load_decks(opponent_deck_zoo_dir, None)}
    policy = BoundR4FPolicy(Path(args.r4f), args.deck_policy)
    pilots = _pilot_specs(args, policy, decks_by_hash)
    pilots_requested = len(pilots)
    pilot_validation = _validate_decks(pilots, "pilot")
    pilots_available = len(pilots)
    pilots = _apply_pilot_shard(pilots, int(args.pilot_shard_index), int(args.pilot_shard_count))
    opponent_decks = _opponent_specs(args, policy, opponent_decks_by_hash)
    if args.limit_opponents is not None:
        opponent_decks = opponent_decks[: int(args.limit_opponents)]
    opponent_validation = _validate_decks(opponent_decks, "opponent")
    opponent = SpecAgent.from_str(args.opponent_policy)
    by_matchup: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, str, str, str]] = set()
    matchup_path = out_dir / "by_matchup.csv"
    if args.resume and matchup_path.exists():
        by_matchup = _read_csv(matchup_path)
        completed_keys = {_matchup_key(row) for row in by_matchup}
        print(f"resume loaded {len(by_matchup)} completed matchup rows", flush=True)

    _write_outputs(
        out_dir,
        args,
        deck_zoo_dir,
        opponent_deck_zoo_dir,
        pilots_requested,
        pilots_available,
        pilots,
        opponent_decks,
        opponent_validation,
        pilot_validation,
        by_matchup,
    )
    for pilot_index, pilot in enumerate(pilots):
        policy.bind_deck(
            list(pilot["card_ids"]),
            str(pilot["pilot_deck_hash"]),
            str(pilot["pilot_policy_id"]),
            str(pilot["pilot_label"]),
        )
        for opponent_index, opp in enumerate(opponent_decks):
            key = _matchup_key_for(pilot, opp, args)
            if key in completed_keys:
                print(
                    f"[skip {pilot['pilot_label']}] vs {opp['archetype']} {opp['deck_hash']} "
                    f"games={int(args.games)}",
                    flush=True,
                )
                continue
            row_seed = int(args.seed) + int(args.pilot_shard_index) * 100_000_000 + pilot_index * 1_000_000 + opponent_index
            summary = run_real_meta_matchup(
                policy,
                opponent,
                list(pilot["card_ids"]),
                list(opp["card_ids"]),
                games=int(args.games),
                bucket="deck_choice_panel",
                opp_deck_hash=opp["deck_hash"],
                opp_archetype=opp["archetype"],
                seed=row_seed,
                max_steps=int(args.max_steps),
            )
            decided = summary.our_wins + summary.opp_wins
            row = {
                "policy": "R4-F_reweighted",
                "pilot_label": pilot["pilot_label"],
                "pilot_deck_hash": pilot["pilot_deck_hash"],
                "pilot_policy_id": pilot["pilot_policy_id"],
                "pilot_archetype": pilot["archetype"],
                "pilot_variant_label": pilot["variant_label"],
                "opponent_policy": args.opponent_policy,
                "opponent_deck_hash": opp["deck_hash"],
                "opponent_archetype": opp["archetype"],
                "opponent_variant_label": opp["variant_label"],
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
                "seed": row_seed,
                "max_steps": int(args.max_steps),
                "errors": "; ".join(summary.errors),
            }
            by_matchup.append(row)
            completed_keys.add(key)
            print(
                f"[{pilot['pilot_label']}] vs {opp['archetype']} {opp['deck_hash']} "
                f"wr={row['winrate']} safety={row['crashes']}/{row['timeouts']}",
                flush=True,
            )
            _write_outputs(
                out_dir,
                args,
                deck_zoo_dir,
                opponent_deck_zoo_dir,
                pilots_requested,
                pilots_available,
                pilots,
                opponent_decks,
                opponent_validation,
                pilot_validation,
                by_matchup,
            )
            if args.stop_on_safety and (summary.crashes or summary.timeouts):
                raise RuntimeError(
                    f"safety failure for {pilot['pilot_label']} vs {opp['deck_hash']}: "
                    f"crashes={summary.crashes}, timeouts={summary.timeouts}, errors={summary.errors[:2]}"
                )
    payload = _write_outputs(
        out_dir,
        args,
        deck_zoo_dir,
        opponent_deck_zoo_dir,
        pilots_requested,
        pilots_available,
        pilots,
        opponent_decks,
        opponent_validation,
        pilot_validation,
        by_matchup,
    )
    print(json.dumps({"by_pilot": payload["by_pilot"], "safety": payload["safety"]}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare R4-F pilot decks against the same copied top-player panel.")
    parser.add_argument("--deck-zoo", required=True)
    parser.add_argument("--opponent-deck-zoo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pilot-deck", action="append", required=True)
    parser.add_argument("--opponent-deck", action="append")
    parser.add_argument("--r4f", default=str(DEFAULT_R4F))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--copied-policy-id", default="top_player_replay")
    parser.add_argument("--opponent-policy", default="first-legal")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-opponents", type=int)
    parser.add_argument("--pilot-shard-index", type=int, default=0)
    parser.add_argument("--pilot-shard-count", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-safety", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
