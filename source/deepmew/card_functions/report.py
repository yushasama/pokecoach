"""Generate the human-readable v2 classification report (Markdown)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from deepmew.card_functions.schemas import ARTIFACT_VERSION, TAG_ORDER, ClassifiedCard
from deepmew.card_functions.traps import evaluate_all, find_trap_card, TRAP_CASES

_V1_CSV = "data/all_cards_function_tags_from_EN_Card_Data.csv"


def _load_v1_tags(handoff_dir: str | Path | None) -> dict[str, str]:
    if handoff_dir is None:
        return {}
    path = Path(handoff_dir) / _V1_CSV
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[str(row.get("card_id", "")).strip()] = row.get("function_tags", "")
    return out


def _confidence_breakdown(cards: list[ClassifiedCard]) -> Counter:
    return Counter(card.confidence for card in cards)


def _tag_frequency(cards: list[ClassifiedCard]) -> Counter:
    counter: Counter = Counter()
    for card in cards:
        counter.update(card.tags)
    return counter


def _top_review_cards(cards: list[ClassifiedCard], limit: int = 25) -> list[ClassifiedCard]:
    flagged = [c for c in cards if c.needs_manual_review]
    flagged.sort(key=lambda c: (-c.manual_audit_priority, int(c.card_id) if c.card_id.isdigit() else 0))
    return flagged[:limit]


def build_report(
    cards: list[ClassifiedCard],
    *,
    handoff_dir: str | Path | None = None,
    csv_path: str | Path = "",
    csv_output: str | Path = "",
) -> str:
    total = len(cards)
    confidence = _confidence_breakdown(cards)
    review = sum(1 for c in cards if c.needs_manual_review)
    tag_freq = _tag_frequency(cards)
    v1_tags = _load_v1_tags(handoff_dir)
    trap_results = evaluate_all(cards)
    trap_pass = sum(1 for r in trap_results if r["passed"])

    lines: list[str] = []
    lines.append("# CardFunctionClassifier v2 Report")
    lines.append("")
    lines.append(f"Artifact: `{ARTIFACT_VERSION}` (offline/training-only).")
    lines.append("")
    lines.append(
        "This is the first stage of `card text -> function tags -> deck profile -> "
        "matchup threats -> action scores -> reason tags`. It only assigns **function "
        "tags** from `EN_Card_Data.csv` card text. **No** strategic action scoring, **no** "
        "deck-name scripting, and **no** matchup win rates are produced here."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Source card text: `{csv_path}`")
    lines.append(f"- Output CSV: `{csv_output}`")
    lines.append(
        f"- Handoff priority hints: `{handoff_dir}` "
        "(used ONLY to rank manual-audit order, never as truth labels)."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Cards classified (unique `Card ID`): **{total}**")
    lines.append(
        f"- Confidence: high **{confidence.get('high', 0)}**, "
        f"medium **{confidence.get('medium', 0)}**, low **{confidence.get('low', 0)}**"
    )
    lines.append(f"- Needs manual review: **{review}** ({review / total:.1%} of cards)" if total else "")
    lines.append(f"- Known trap cases passing: **{trap_pass}/{len(trap_results)}**")
    lines.append("")

    lines.append("## Tag frequency")
    lines.append("")
    lines.append("| Tag | Cards |")
    lines.append("|---|---|")
    for tag in TAG_ORDER:
        count = tag_freq.get(tag, 0)
        if count:
            lines.append(f"| `{tag}` | {count} |")
    lines.append("")

    lines.append("## Top manual-review cards (by audit priority)")
    lines.append("")
    lines.append("| Priority | Card ID | Name | Conf | Tags |")
    lines.append("|---|---|---|---|---|")
    for card in _top_review_cards(cards):
        tags = ", ".join(t for t in card.tags if t != "needs_manual_review")
        lines.append(
            f"| {card.manual_audit_priority} | {card.card_id} | {card.name} | {card.confidence} | {tags} |"
        )
    lines.append("")

    lines.append("## Known trap test results")
    lines.append("")
    lines.append("| Case | Card ID | Name | Passed | Failures |")
    lines.append("|---|---|---|---|---|")
    for result in trap_results:
        failures = "; ".join(result["failures"]) if result["failures"] else "-"
        passed = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {result['key']} | {result['card_id']} | {result['name']} | {passed} | {failures} |"
        )
    lines.append("")

    lines.append("## v2 vs v1 on known traps")
    lines.append("")
    if v1_tags:
        lines.append("v1's full-pool classifier is provisional and known to contain semantic errors. ")
        lines.append("v2 was built from card text, not from v1 labels. Differences on traps:")
        lines.append("")
        lines.append("| Card | v1 tags | v2 tags |")
        lines.append("|---|---|---|")
        for case in TRAP_CASES:
            card = find_trap_card(cards, case)
            if card is None:
                continue
            v1 = v1_tags.get(card.card_id, "(absent)")
            v2 = ";".join(t for t in card.tags if t != "needs_manual_review")
            lines.append(f"| {card.name} ({card.card_id}) | {v1} | {v2} |")
    else:
        lines.append("_v1 comparison CSV not found; skipped._")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Tags are rule-derived from card text; cards flagged `needs_manual_review` "
        "(prevention, damage-counter, bench, gust/switch, lock, Ability-prevention, energy "
        "movement, status, self-KO, dual Ability+attack, top-meta core, or broad-fallback "
        "tagging) should be human-audited before being used as training labels."
    )
    lines.append("- `EN_Card_Data.csv`'s 2,022 rows are one-row-per-move; they collapse to 1,267 unique cards.")
    lines.append("- This package is never imported by the submission runtime or `cg` (INVARIANT #7/#8).")
    lines.append("")
    return "\n".join(line for line in lines if line is not None)
