"""Human-readable meta report (`deck_meta_report.md`).

The report explains whether the classifier is sane: top cards, staples,
archetypes, variants, engine-first candidates, unknowns, edge-case checks, name
issues, warnings, and handoff notes for K2B/K3/K4/K5/K6.
"""

from __future__ import annotations

from typing import Any

from deepmew.meta_kg.schemas import (
    BuildWarning,
    STATUS_ENGINE_FIRST,
    STATUS_LOW_CONFIDENCE,
    STATUS_MIXED,
    STATUS_UNKNOWN,
)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_report(ctx: dict[str, Any]) -> str:
    counts = ctx["counts"]
    cfg = ctx["config"]
    catalog = ctx["catalog_rows"]
    staples = ctx["staple_rows"]
    summary = ctx["archetype_summary"]
    assignments = ctx["assignment_rows"]
    packages = ctx["support_packages"]
    warnings: dict[str, BuildWarning] = ctx["warnings"]
    name_lookup = ctx["card_names"]

    def name(cid: str) -> str:
        return name_lookup.get(cid, f"Unknown Card {cid}")

    lines: list[str] = [
        "# K2A Meta Knowledge Graph Report",
        "",
        f"- built_at_utc: {ctx['built_at_utc']}",
        f"- artifact_version: {ctx['artifact_version']}",
        f"- input_dir: {ctx['input_dir']}",
        f"- output_dir: {ctx['output_dir']}",
        "",
        "## Corpus counts",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in counts.items()]

    lines += ["", "## Output files", ""]
    lines += [f"- {name_}: {path}" for name_, path in ctx["output_paths"].items()]

    lines += ["", "## Top global cards", ""]
    top_cards = sorted(catalog, key=lambda r: (-r["deck_count"], r["card_id"]))[:20]
    lines += _table(
        ["card_id", "card_name", "deck_count", "global_deck_freq", "type_guess"],
        [[r["card_id"], r["card_name"], r["deck_count"], r["global_deck_freq"], r["type_guess"]] for r in top_cards],
    )

    lines += ["", "## Staples", "",
              f"thresholds: min_global_deck_freq={cfg.staple_min_global_deck_freq}, "
              f"max_lift_for_staple={cfg.staple_max_lift}, min_entropy_norm={cfg.staple_min_entropy_norm}", ""]
    lines += _table(
        ["card_id", "card_name", "global_deck_freq", "max_archetype_lift", "staple_score", "downweight"],
        [[r["card_id"], r["card_name"], r["global_deck_freq"], r["max_archetype_lift"], r["staple_score"], r["downweight"]]
         for r in staples[:20]],
    )

    lines += ["", "## Archetype summary", ""]
    lines += _table(
        ["archetype", "deck_count", "artifact_count", "episode_count", "meta_share_by_deck", "avg_confidence"],
        [[r["archetype"], r["deck_count"], r["artifact_count"], r["episode_count"], r["meta_share_by_deck"], r["avg_confidence"]]
         for r in summary],
    )

    variants = [r for r in assignments if " / " in r["variant_label"]]
    lines += ["", f"## Variants ({len(variants)} decks with variant labels)", ""]
    lines += _table(
        ["deck_hash", "variant_label", "confidence", "artifact_count"],
        [[r["deck_hash"], r["variant_label"], r["confidence"], r["artifact_count"]] for r in variants[:20]],
    ) if variants else ["- None."]

    engine_first = [r for r in assignments if r["classification_status"] == STATUS_ENGINE_FIRST]
    lines += ["", f"## Engine-first candidates ({len(engine_first)})", ""]
    lines += _table(
        ["deck_hash", "primary_archetype", "confidence", "artifact_count"],
        [[r["deck_hash"], r["primary_archetype"], r["confidence"], r["artifact_count"]] for r in engine_first[:20]],
    ) if engine_first else ["- None."]

    unknowns = [r for r in assignments if r["classification_status"] in (STATUS_UNKNOWN, STATUS_LOW_CONFIDENCE)]
    lines += ["", f"## Unknown / low-confidence decks ({len(unknowns)})", ""]
    lines += _table(
        ["deck_hash", "classification_status", "confidence", "artifact_count", "notes"],
        [[r["deck_hash"], r["classification_status"], r["confidence"], r["artifact_count"], r["notes"]] for r in unknowns[:20]],
    ) if unknowns else ["- None."]

    multi = [r for r in assignments if r["classification_status"] == STATUS_MIXED]
    lines += ["", f"## Multi-core / mixed decks ({len(multi)})", ""]
    lines += _table(
        ["deck_hash", "variant_label", "confidence", "artifact_count"],
        [[r["deck_hash"], r["variant_label"], r["confidence"], r["artifact_count"]] for r in multi[:20]],
    ) if multi else ["- None."]

    lines += ["", "## Split-evolution check", "",
              "Terminal cards dominate primary labels; shared basics are supporting evidence only.",
              "Archetypes sharing line cards but differing in terminal cards stay separate (see archetype summary above)."]

    lines += ["", "## Staple-bridge check", "",
               ctx.get("staple_bridge_note", "Weighted similarity downweights staples; see deck_similarity.csv shared_staples vs shared_core_cards columns.")]

    missing = warnings.get("missing_card_name")
    conflicts = warnings.get("card_name_conflict")
    lines += ["", "## Missing names / name conflicts", "",
              f"- missing_card_name: {missing.count if missing else 0}",
              f"- card_name_conflict: {conflicts.count if conflicts else 0}"]
    if conflicts and conflicts.examples:
        lines.append(f"- conflict examples (card_id): {', '.join(conflicts.examples[:10])}")

    lines += ["", "## Support packages (top)", ""]
    lines += _table(
        ["owner_archetype", "package_type", "package_score", "n_cards"],
        [[p["owner_archetype"], p["package_type"], p["package_score"], len(p["card_ids"])] for p in packages[:15]],
    ) if packages else ["- None."]

    lines += ["", "## Warnings", ""]
    if warnings:
        lines += [f"- {w.code}: count={w.count}" + (f" examples={w.examples[:5]}" if w.examples else "") for w in warnings.values()]
    else:
        lines.append("- None.")

    lines += [
        "",
        "## Handoff notes (K2B / K3 / K4 / K5 / K6)",
        "",
        "- K2B Real-meta Arena: weight opponent mixtures by archetype_summary meta_share_by_deck/artifact; pick representative decks from deck_archetypes (high-confidence first).",
        "- ReplayPolicy: sample opponent traces by primary_archetype/variant_label; start with classification_status=confident only.",
        "- Live belief: load live_belief.json into LiveArchetypeBelief; staples are pre-downweighted.",
        "- K3 Deck Zoo: seed from representative decks per archetype + deck_similarity clusters + engine-first + unknown buckets.",
        "- K4 imitation: join action rows to deck_archetypes by episode/player; keep unknown/low_confidence as explicit buckets.",
        "- K5 offline RL: use archetype/variant/confidence as context features; never collapse Unknown/Mixed.",
        "- K6 submission loop: break down replay performance by inferred opponent archetype/variant.",
        "",
        "## Caveats",
        "",
        "- Deck payloads alone cannot prove matchup-specific punishment rules; action rows are required for timing/exploitation analysis.",
        "- Corpus is a rank 1-300 stratified sample; meta shares are corpus-relative, not absolute ladder truth.",
        "- Card names are display enrichment; card_id is canonical. Future EN metadata enriches, never replaces, the catalog.",
    ]
    return "\n".join(lines)
