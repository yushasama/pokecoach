"""Markdown reports for the K4.1 fine labels and K5-lite BC comparison."""

from __future__ import annotations

from typing import Any


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def _top(summary: list[dict[str, Any]], key: str = "label", n: int = 12) -> list[list[Any]]:
    return [[r.get(key), r.get("n_examples"), r.get("share")] for r in summary[:n]]


def render_k4_1_report(ctx: dict[str, Any]) -> str:
    c = ctx["counts"]
    lines: list[str] = []
    lines.append("# K4.1 Fine-Grained Imitation Labels")
    lines.append("")
    lines.append(f"- Built: {ctx['built_at_utc']}  ·  artifact: `{ctx['artifact_version']}`")
    lines.append(f"- K2A: `{ctx['meta_kg_dir']}`")
    lines.append(f"- K3 deck zoo: `{ctx['deck_zoo_dir']}`")
    lines.append(f"- Output: `{ctx['output_dir']}`")
    lines.append("")
    lines.append("Deterministic, observational pseudo-labels (not hand annotations, not proven "
                 "full-game counterfactuals). Offline/training-only; never imported by the runtime.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.extend(_table(
        ["metric", "value"],
        [
            ["examples", c["example_count"]],
            ["MAIN-context examples", c["main_context_count"]],
            ["episodes", c["episode_count"]],
            ["decks", c["deck_count"]],
            ["archetypes", c["archetype_count"]],
            ["rows with card_id", c["rows_with_card_id"]],
            ["rows with target", c["rows_with_target"]],
            ["rows with punishment label", c["rows_with_punishment"]],
            ["rows with risk label", c["rows_with_risk"]],
            ["last_stand_triggered rows", c["last_stand_triggered"]],
            ["pilot_rank-known rows (K1.7)", f'{c.get("pilot_rank_known_examples")} ({c.get("pilot_rank_known_share")})'],
            ["pilot top10 rows", c.get("pilot_rank_top10_examples")],
            ["opponent_rank-known rows", c.get("opponent_rank_known_examples")],
            ["missing archetype rows", c["missing_label_rows"]],
            ["missing opponent rows", c["missing_opponent_rows"]],
            ["unresolved identity rows", c["unresolved_identity_rows"]],
        ],
    ))
    lines.append("")
    lines.append("## goal_of_deck distribution")
    lines.append("")
    lines.extend(_table(["goal_of_deck", "n", "share"], _top(ctx["goal_deck_summary"])))
    lines.append("")
    lines.append("## goal_of_play distribution")
    lines.append("")
    lines.extend(_table(["goal_of_play", "n", "share"], _top(ctx["goal_play_summary"])))
    lines.append("")
    lines.append("## punishment labels (decomposed)")
    lines.append("")
    lines.extend(_table(["punishment", "n", "share"], _top(ctx["punishment_summary"], n=10)))
    lines.append("")
    lines.append("## risk justification")
    lines.append("")
    lines.extend(_table(["risk", "n", "share"], _top(ctx["risk_summary"])))
    lines.append("")
    lines.append("## last stand / comeback equity")
    lines.append("")
    lines.extend(_table(["flag", "n", "share/value"], _top(ctx["last_stand_summary"], n=8)))
    lines.append("")
    lines.append("Doctrine: punishment = loss of future agency / hard-to-recover branch, not "
                 "merely taking damage. Risk is contextual; last_stand_mode prefers comeback "
                 "equity over polite survival when passive play loses anyway.")
    lines.append("")
    return "\n".join(lines)


def render_k5_lite_report(ctx: dict[str, Any]) -> str:
    c = ctx["counts"]
    lines: list[str] = []
    lines.append("# K5-lite: Filtered BC vs Reward-Weighted BC")
    lines.append("")
    lines.append(f"- Built: {ctx['built_at_utc']}  ·  artifact: `{ctx['artifact_version']}`")
    lines.append(f"- Output: `{ctx['output_dir']}`")
    lines.append("")
    lines.append("K5-lite is offline corpus-only policy improvement, not full RL. Two transparent "
                 "count-table behavior-cloning methods are compared by held-out top-1 accuracy of "
                 "reproducing the real *fine-grained* choice (PLAY/EVOLVE card, ATTACH target, else "
                 "option category) -- finer than K4's bare action category.")
    lines.append("")
    lines.append("- **Filtered BC**: unit-weight, trusted-teacher rows only (won + confident "
                 "archetype + decent label confidence).")
    lines.append("- **Reward-weighted BC**: all rows, scaled by outcome × teacher trust × "
                 "goal-progress × label-confidence, downweighting punished lines unless the risk "
                 "was justified or a forced last stand.")
    lines.append("")
    lines.append("## Method comparison (held-out top-1 fine-target accuracy)")
    lines.append("")
    lines.extend(_table(
        ["scope", "method", "accuracy", "coverage", "n_test", "n_train"],
        [[r["context_scope"], r["method"], r["accuracy"], r["coverage"], r["n_test"], r["n_train"]]
         for r in ctx["comparison"]],
    ))
    lines.append("")
    lines.append("## Train sizes")
    lines.append("")
    lines.extend(_table(
        ["table", "train_examples"],
        [
            ["filtered_bc", c["filtered_bc_train_examples"]],
            ["reward_weighted_bc", c["reward_weighted_bc_train_examples"]],
            ["conservative_action_value", c.get("value_table_train_examples")],
        ],
    ))
    lines.append("")
    lines.append("## K5 conservative action-value scorer (directional offline sanity)")
    lines.append("")
    lines.append("Not BC argmax: ranks behavior-supported actions by estimated value "
                 "(outcome advantage + goal progress - unjustified punishment + last-stand "
                 "correctness), with an OOD support gate and LCB pessimism. `lift` = realized "
                 "win rate when the player agreed with the scorer's top-value action minus "
                 "when they disagreed (positive = value picks correlate with winning).")
    lines.append("")
    lines.extend(_table(
        ["scope", "n_test", "coverage", "agree_rate", "agreed_win_rate", "disagreed_win_rate", "lift"],
        [[r["scope"], r["n_test"], r["coverage"], r["agree_rate"], r["agreed_win_rate"],
          r["disagreed_win_rate"], r["lift"]] for r in ctx.get("value_eval", [])],
    ))
    lines.append("")
    lines.append("Tables are eval-only artifacts. Runtime promotion requires K2B improvement "
                 "without farming weak ReplayPolicy or regressing anti_overfit_holdout, then a "
                 "gated pure-Python distillation. No default policy change here.")
    lines.append("")
    return "\n".join(lines)
