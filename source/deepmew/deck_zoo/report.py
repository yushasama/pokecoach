"""K3 Deck Zoo human-readable report."""

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
    summary = ctx["zoo_summary"]
    sets = ctx["opponent_sets"]

    lines: list[str] = [
        "# K3 Deck Zoo Report",
        "",
        f"- built_at_utc: {ctx['built_at_utc']}",
        f"- artifact_version: {ctx['artifact_version']}",
        f"- meta_kg_dir: {ctx['meta_kg_dir']}",
        f"- output_dir: {ctx['output_dir']}",
        "",
        "## Counts",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in counts.items()]

    lines += ["", "## Output files", ""]
    lines += [f"- {name}: {path}" for name, path in ctx["output_paths"].items()]

    lines += ["", "## Archetypes (deck zoo summary)", ""]
    lines += _table(
        ["archetype", "decks", "canonical", "near_dup", "share", "avg_conf", "train/eval/holdout"],
        [[r["archetype"], r["deck_count"], r["canonical_count"], r["near_duplicate_count"],
          r["meta_share_by_artifact"], r["avg_confidence"],
          f"{r['train_count']}/{r['eval_count']}/{r['holdout_count']}"] for r in summary],
    )

    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in sets:
        by_set.setdefault(row["set_name"], []).append(row)
    for set_name in ("top_meta", "variant_diversity", "anti_overfit_holdout", "unknown_mixed", "engine_first"):
        members = by_set.get(set_name, [])
        lines += ["", f"## Opponent set: {set_name} ({len(members)})", ""]
        if members:
            lines += _table(
                ["rank", "archetype", "variant_label", "confidence", "weight", "split"],
                [[m["rank_in_set"], m["archetype"], m["variant_label"], m["confidence"], m["weight"], m["split"]]
                 for m in members[:25]],
            )
        else:
            lines.append("- (empty in this build)")
            if set_name == "engine_first":
                lines.append("- No engine packages configured yet; engine-first decks currently surface as Unknown. "
                             "See unknown_mixed for candidate engine/toolbox decks.")

    lines += [
        "",
        "## How downstream systems consume this",
        "",
        "- K2B Real-meta Arena: instantiate opponents from arena_opponent_sets.csv (weights = meta share); decklists in deck_zoo_decklists.jsonl.",
        "- arena_matchups.csv: candidate DeepMew-vs-opponent matchups with suggested_games for evaluation scheduling.",
        "- K4 imitation: train on split=train canonical decks; never imitate split=holdout (anti_overfit_holdout).",
        "- unknown_mixed is a robustness/chaos bucket; keep it explicit, do not coerce into known archetypes.",
        "",
        "## Caveats",
        "",
        f"- Near-duplicate threshold = {cfg.dup_similarity_threshold} weighted similarity within the same archetype.",
        "- Splits are deterministic by deck_hash; near-duplicates inherit their canonical's split to avoid imitation leakage.",
        "- Meta shares are corpus-relative (rank 1-300 stratified sample), not absolute ladder truth.",
    ]
    return "\n".join(lines)
