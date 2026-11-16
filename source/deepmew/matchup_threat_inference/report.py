"""Generate the MatchupThreatInference v0 report (facts / hypotheses / unresolved).

Three layers, exactly as DeckProfileInference v0:

* **Confirmed card-function evidence (facts)** — the threat/coverage cards cited come
  straight from v2 card-text tags.
* **Inferred threat surfaces & matchup pairings (hypotheses)** — count-weighted
  aggregations. Archetype labels are display handles only.
* **Unresolved / manual-review** — what a human should audit.

The report states plainly that **no matchup win rates** are claimed.
"""

from __future__ import annotations

from collections import Counter

from deepmew.matchup_threat_inference.schemas import (
    ARTIFACT_VERSION,
    COVERAGE_CATEGORIES,
    COVERAGE_GAP_CATEGORIES,
    DeckThreatProfile,
    MatchupPairing,
    THREAT_CATEGORIES,
    threat_sort_key,
)


def build_report(
    profiles: list[DeckThreatProfile],
    pairings: list[MatchupPairing],
    *,
    decklists_path: str = "",
    v2_csv: str = "",
    csv_output: str = "",
    jsonl_output: str = "",
    pairings_output: str = "",
) -> str:
    total = len(profiles)
    threat_freq = Counter(t.category for p in profiles for t in p.all_threats() if t.is_primary)
    coverage_freq = Counter(c.category for p in profiles for c in p.coverage if c.reliable)
    gap_freq = Counter(g for pr in pairings for g in pr.coverage_gaps)
    conf_freq = Counter(p.threat_confidence for p in profiles)
    review_profiles = [p for p in profiles if p.manual_review_reasons]

    lines: list[str] = []
    lines.append("# MatchupThreatInference v0 Report")
    lines.append("")
    lines.append(f"Artifact: `{ARTIFACT_VERSION}` (offline/training-only).")
    lines.append("")
    lines.append(
        "Stage 3 of `card text -> function tags -> deck profile -> matchup threats -> "
        "action scores -> reason tags`. This stage only maps **deck profile -> matchup "
        "threats**. A matchup threat is a structured hypothesis (evidence + confidence + "
        "reason tags), **not** an action score and **not** a win-rate claim."
    )
    lines.append("")
    lines.append(
        "> **No matchup win rates are claimed anywhere in this artifact.** There are no "
        "`win_rate`, `matchup_win_rate`, `expected_win_rate`, or `EV` fields. Threats and "
        "coverage gaps are structural hypotheses about what a deck pressures, exposes, and "
        "can answer — not predictions of who wins. Archetype labels are display handles "
        "only and are never used as inference logic."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- DeckProfileInference v0 decklists (count source): `{decklists_path}`")
    lines.append(f"- CardFunctionClassifier v2 tags: `{v2_csv}`")
    lines.append(f"- Output CSV (per-deck surface): `{csv_output}`")
    lines.append(f"- Output JSONL (per-deck surface): `{jsonl_output}`")
    lines.append(f"- Output JSONL (matchup pairings): `{pairings_output}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Deck threat surfaces generated: **{total}**")
    lines.append(f"- Matchup pairings generated: **{len(pairings)}**")
    lines.append(
        f"- Threat confidence: high **{conf_freq.get('high', 0)}**, "
        f"medium **{conf_freq.get('medium', 0)}**, low **{conf_freq.get('low', 0)}**"
    )
    lines.append(f"- Deck surfaces with manual-review reasons: **{len(review_profiles)}**")
    lines.append("")

    # --- FACTS ---------------------------------------------------------------- #
    lines.append("## Confirmed card-function evidence (facts)")
    lines.append("")
    lines.append(
        "Most-cited threat/coverage evidence cards. Each card's role here is derived from "
        "its v2 card-text tags; the only aggregation is how many decks cite it."
    )
    lines.append("")
    appearances: Counter = Counter()
    names: dict[str, str] = {}
    for profile in profiles:
        for item in profile.all_threats():
            for ref in item.evidence:
                appearances[ref.card_id] += 1
                names[ref.card_id] = ref.name
        for item in profile.coverage:
            for ref in item.evidence:
                appearances[ref.card_id] += 1
                names[ref.card_id] = ref.name
    lines.append("| Card | Decks citing |")
    lines.append("|---|---|")
    for card_id, count in appearances.most_common(20):
        lines.append(f"| {names.get(card_id, card_id)} ({card_id}) | {count} |")
    lines.append("")

    # --- HYPOTHESES ----------------------------------------------------------- #
    lines.append("## Inferred threat surfaces (hypotheses)")
    lines.append("")
    lines.append("### Primary threat frequency (presents + exposes)")
    lines.append("")
    lines.append("| Threat category | Decks (primary) |")
    lines.append("|---|---|")
    for category in sorted(THREAT_CATEGORIES, key=threat_sort_key):
        if threat_freq.get(category):
            lines.append(f"| `{category}` | {threat_freq[category]} |")
    lines.append("")
    lines.append("### Reliable coverage frequency")
    lines.append("")
    lines.append("| Coverage category | Decks (reliable) |")
    lines.append("|---|---|")
    for category in COVERAGE_CATEGORIES:
        if coverage_freq.get(category):
            lines.append(f"| `{category}` | {coverage_freq[category]} |")
    lines.append("")
    lines.append("### Coverage-gap frequency (across matchup pairings)")
    lines.append("")
    lines.append("| Coverage-gap category | Pairings |")
    lines.append("|---|---|")
    for category in COVERAGE_GAP_CATEGORIES:
        if gap_freq.get(category):
            lines.append(f"| `{category}` | {gap_freq[category]} |")
    lines.append("")

    # --- UNRESOLVED ----------------------------------------------------------- #
    lines.append("## Unresolved / manual-review cases")
    lines.append("")
    lines.append(f"- Deck surfaces flagged for review: **{len(review_profiles)}** / {total}")
    low_conf = [p for p in profiles if p.threat_confidence == "low"]
    lines.append(f"- Deck surfaces at low threat confidence: **{len(low_conf)}**")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Threats and coverage gaps are structural hypotheses from card-text tags, **not** "
        "validated game outcomes and **not** matchup win rates."
    )
    lines.append(
        "- `tool_removal` and `stadium_bump` coverage are heuristic (no clean v2 tag): any "
        "Stadium can bump an opponent's Stadium, and tool removal is detected from Item/"
        "Supporter cards that reference a Tool. Both are down-weighted and flagged for audit."
    )
    lines.append(
        "- The three bench-pressure threats (ordinary attack damage, attack-based counters, "
        "Ability-based counters) and their answers are kept strictly separate: Battle Cage "
        "does not answer ordinary bench attack damage, and Rabsca does not answer Ability "
        "counters."
    )
    lines.append(
        "- A one-of tech answer is recorded but never counts as full reliable coverage."
    )
    lines.append("- This package is never imported by the submission runtime or `cg` (INVARIANT #7/#8).")
    lines.append("")
    return "\n".join(lines)
