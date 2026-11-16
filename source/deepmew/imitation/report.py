"""K4 imitation dataset human-readable report."""

from __future__ import annotations

from typing import Any


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def render_report(ctx: dict[str, Any]) -> str:
    counts = ctx["counts"]
    summary = ctx["dataset_summary"]
    buckets = ctx["bucket_summary"]
    baseline = ctx["baseline_rows"]

    def dim(name: str) -> list[list[Any]]:
        return [[r["value"], r["n_examples"]] for r in summary if r["dimension"] == name]

    lines: list[str] = [
        "# K4 Imitation Dataset Report",
        "",
        f"- built_at_utc: {ctx['built_at_utc']}",
        f"- artifact_version: {ctx['artifact_version']}",
        f"- meta_kg_dir: {ctx['meta_kg_dir']}",
        f"- deck_zoo_dir: {ctx['deck_zoo_dir']}",
        f"- output_dir: {ctx['output_dir']}",
        "",
        "## Counts",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in counts.items()]

    lines += ["", "## Output files", ""]
    lines += [f"- {name}: {path}" for name, path in ctx["output_paths"].items()]

    lines += ["", "## Split distribution", ""]
    lines += _table(["split", "n_examples"], dim("split"))
    lines += ["", "## Select-context distribution", ""]
    lines += _table(["select_context", "n_examples"], dim("select_context")[:12])
    lines += ["", "## Chosen action category distribution", ""]
    lines += _table(["action_type", "n_examples"], dim("chosen_option_type")[:12])

    lines += ["", "## Behavior-cloning baseline (held-out top-1 accuracy, MAIN context)", "",
              "Predict the real player's next action category on eval+holdout decisions. "
              "Higher = imitation labels add signal; archetype-conditioning beating context-only "
              "means K2A archetype labels capture real-meta behavior."]
    lines += [""]
    if baseline:
        lines += _table(
            ["model", "accuracy", "coverage", "n_train", "n_test"],
            [[r["model"], r["accuracy"], r["coverage"], r["n_train"], r["n_test"]] for r in baseline],
        )
        best = max(baseline, key=lambda r: r["accuracy"])
        ctx_only = next((r for r in baseline if r["model"] == "by_context"), None)
        arch = next((r for r in baseline if r["model"] == "by_context_archetype"), None)
        if ctx_only and arch:
            delta = round(arch["accuracy"] - ctx_only["accuracy"], 4)
            verdict = "helped" if delta > 0 else ("no change" if delta == 0 else "hurt")
            lines += ["", f"- Archetype conditioning vs context-only: {delta:+} ({verdict}).",
                      f"- Best model: {best['model']} (accuracy={best['accuracy']})."]
    else:
        lines.append("- Baseline not run (insufficient train/test split).")

    lines += ["", "## Top buckets by example count", ""]
    top_buckets = sorted(buckets, key=lambda r: -r["n_examples"])[:15]
    lines += _table(
        ["bucket_type", "bucket_value", "n_examples", "n_episodes", "decision_win_rate"],
        [[r["bucket_type"], r["bucket_value"], r["n_examples"], r["n_episodes"], r["decision_win_rate"]] for r in top_buckets],
    )

    lines += [
        "",
        "## Limitations",
        "",
        "- Label is the chosen action *category* (OptionType), not the exact target/card; a first BC signal, not full play reconstruction.",
        "- `won` is a whole-game outcome attached to every decision (crude credit assignment), not a per-decision value.",
        "- Opponent archetype is missing when the opponent deck was not extracted for that episode (counted in missing_opponent_rows).",
        "- Accuracy is a proxy for imitation quality; it does not by itself prove in-game win-rate improvement.",
        "- Full-game ReplayPolicy-vs-DeepMew evaluation is deferred (needs engine deck-loading); see handoff notes.",
        "",
        "## Handoff (K5 offline RL / K6 submission loop)",
        "",
        "- K5: use split=train examples as behavior-policy data; keep Unknown/Mixed archetypes as explicit context, never coerced.",
        "- A ReplayPolicy can sample action-category distributions per (select_context, archetype) from the train tables.",
        "- K6: after a submission, bucket replay performance by inferred opponent archetype to target patches.",
    ]
    return "\n".join(lines)
