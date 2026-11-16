"""K2B real-meta arena human-readable report."""

from __future__ import annotations

from typing import Any


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def render_report(ctx: dict[str, Any]) -> str:
    counts = ctx["counts"]
    cfg = ctx["config"]
    summary = ctx["summary_rows"]
    invalid = [r for r in ctx["validation_rows"] if not r["valid"]]

    lines: list[str] = [
        "# K2B Real-Meta Arena Report",
        "",
        f"- built_at_utc: {ctx['built_at_utc']}",
        f"- artifact_version: {ctx['artifact_version']}",
        f"- deck_zoo_dir: {ctx['deck_zoo_dir']}",
        f"- imitation_dir: {ctx['imitation_dir']}",
        f"- output_dir: {ctx['output_dir']}",
        f"- games_per_matchup: {cfg.games_per_matchup}; max_decks_per_bucket: {cfg.max_decks_per_bucket}; seed: {cfg.seed}",
        "",
        "## Counts",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in counts.items()]

    lines += ["", "## Output files", ""]
    lines += [f"- {name}: {path}" for name, path in ctx["output_paths"].items()]

    lines += ["", "## Deck Zoo engine validation", "",
              f"- {counts['decklists_valid']}/{counts['decklists_total']} canonical decklists instantiate via battle_start "
              f"({counts['decklists_invalid']} invalid).",
              "- The 2,022-card pool is not real Standard (DEC-006); invalid decks are skipped, never forced."]
    if invalid:
        lines += ["", "### Invalid decks (first 15)", ""]
        lines += _table(
            ["deck_hash", "archetype", "card_count", "errors"],
            [[r["deck_hash"], r["archetype"], r["card_count"], (r["errors"][:80] if r["errors"] else "")] for r in invalid[:15]],
        )

    lines += ["", "## Results: DeepMew vs real-meta buckets (our win rate)", "",
              "our_label = our deck.csv pilot; opp_kind = how the opponent bucket deck is piloted "
              "(spec:first-legal = neutral baseline pilot; replay = K4 archetype action-category distribution)."]
    lines += [""]
    lines += _table(
        ["bucket", "our_label", "opp_kind", "decks", "games", "completed", "our_win_rate", "our_wins", "opp_wins", "draws", "crashes", "timeouts"],
        [[r["bucket"], r["our_label"], r["opp_kind"], r["decks"], r["games"], r["completed"],
          r["our_win_rate"], r["our_wins"], r["opp_wins"], r["draws"], r["crashes"], r["timeouts"]] for r in summary],
    )

    total_games = counts["games_completed"]
    weak = total_games < 100
    lines += [
        "",
        "## Reading these results",
        "",
        f"- Sample size: {total_games} completed games total" + (" - **statistically weak; treat as directional only.**" if weak else "."),
        "- Crashes/timeouts: " + (f"{counts['crashes']} crashes, {counts['timeouts']} timeouts." if (counts['crashes'] or counts['timeouts']) else "0 crashes, 0 timeouts (safe)."),
        "- The opponent bucket decks are real ladder lists piloted by a baseline/replay policy, NOT by their original ladder pilots, so a low win rate reflects deck+pilot, not the ladder bot. This is a filter, not the ladder oracle (INVARIANT #5).",
        "- ReplayPolicy shapes the opponent's action *category* per (select_context, archetype); it does not reconstruct exact card/target choices (K4 limitation).",
        "",
        "## Is K2B usable as a real-meta gate?",
        "",
        "- Yes for safety/instantiation: DeepMew can now play real-meta decks in-engine and the run reports crashes/timeouts/win-rate per bucket under equal settings.",
        "- Treat win rates as directional at this sample size; widen games/decks for a promotion-grade gate. No policy/deck change is gated on this run.",
        "",
        "## Boundary",
        "",
        "- Eval-only. Root `main.py` / `deck.csv` / `cg/` and default `v2.4` are unchanged; ReplayPolicy is eval-only and never imported by the submission runtime; `.research` artifacts are not loaded by the eval/runtime path.",
    ]
    return "\n".join(lines)
