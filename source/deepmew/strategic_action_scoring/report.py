"""Generate the StrategicActionScorer v0 report (facts / hypotheses / unresolved).

States plainly that no win rates are claimed, no labels are trained, and no runtime
policy was changed. Reports reason-tag frequencies and a few worked examples so a
human can audit the scoring rules.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from deepmew.strategic_action_scoring.schemas import (
    ANSWER_REASON_TAGS,
    ARTIFACT_VERSION,
    BAD_REASON_TAGS,
    REASON_TAGS,
    reason_sort_key,
)


def build_report(
    rows: list[dict[str, Any]],
    *,
    decklists_path: str = "",
    v2_csv: str = "",
    matchup_artifact: str = "",
    csv_output: str = "",
    jsonl_output: str = "",
    n_subjects: int = 0,
    n_opponents: int = 0,
) -> str:
    tag_freq: Counter = Counter(t for r in rows for t in r["reason_tags"])
    kind_freq: Counter = Counter(r["action_kind"] for r in rows)
    conf_freq: Counter = Counter(r["confidence"] for r in rows)

    lines: list[str] = []
    lines.append("# StrategicActionScorer v0 Report")
    lines.append("")
    lines.append(f"Artifact: `{ARTIFACT_VERSION}` (offline/training-only).")
    lines.append("")
    lines.append(
        "Stage 4 of `card text -> function tags -> deck profile -> matchup threats -> "
        "action scores -> reason tags`. This stage maps **matchup threats -> action scores "
        "+ reason tags** with a transparent, fully explainable scorer."
    )
    lines.append("")
    lines.append(
        "> **No win rates / EV are claimed or computed.** Scores are transparent sums of "
        "explicit reason-tagged components (every score ships its component breakdown). "
        "**No labels are trained**, no behaviour cloning or RL runs here, and the runtime "
        "policy (`main.py` / `cg`) is unchanged and never imports this package. Archetype "
        "labels are display-only and never used as scoring logic."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- DeckProfileInference v0 decklists: `{decklists_path}`")
    lines.append(f"- CardFunctionClassifier v2 tags: `{v2_csv}`")
    lines.append(f"- MatchupThreatInference v0 (rebuilt in-process): `{matchup_artifact}`")
    lines.append(f"- Output CSV: `{csv_output}`")
    lines.append(f"- Output JSONL: `{jsonl_output}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Scored candidate-vs-opponent rows: **{len(rows)}**")
    lines.append(f"- Subjects: **{n_subjects}**, representative opponents: **{n_opponents}**")
    lines.append(
        f"- Confidence: high **{conf_freq.get('high', 0)}**, medium **{conf_freq.get('medium', 0)}**, "
        f"low **{conf_freq.get('low', 0)}**"
    )
    lines.append("")
    lines.append("## How the scorer reasons (facts: rules are explicit)")
    lines.append("")
    lines.append(
        "Each candidate's coverage is checked against the opponent's *present* primary "
        "threats using the same ANSWER_MATRIX as MatchupThreatInference v0. The semantic "
        "separations are preserved: counter-placement prevention does not answer ordinary "
        "bench attack damage, attack-effect protection does not answer Ability counters, "
        "and Ability prevention does. Structural answers (tool removal / stadium bump / "
        "energy denial) only earn credit when the opponent dependency is load-bearing; a "
        "one-of tech that answers no live load-bearing threat is penalised."
    )
    lines.append("")
    lines.append("### Positive reason-tag frequency")
    lines.append("")
    lines.append("| Reason tag | Rows |")
    lines.append("|---|---|")
    for tag in ANSWER_REASON_TAGS:
        if tag_freq.get(tag):
            lines.append(f"| `{tag}` | {tag_freq[tag]} |")
    lines.append("")
    lines.append("### Mistake reason-tag frequency")
    lines.append("")
    lines.append("| Reason tag | Rows |")
    lines.append("|---|---|")
    for tag in BAD_REASON_TAGS:
        if tag_freq.get(tag):
            lines.append(f"| `{tag}` | {tag_freq[tag]} |")
    lines.append("")
    lines.append("### Action-kind frequency")
    lines.append("")
    lines.append("| Action kind | Rows |")
    lines.append("|---|---|")
    for kind, count in kind_freq.most_common():
        lines.append(f"| `{kind}` | {count} |")
    lines.append("")

    # --- worked examples (highest and lowest scored, with reasons) ------------ #
    lines.append("## Worked examples (hypotheses)")
    lines.append("")
    tagged = [r for r in rows if r["reason_tags"]]
    top = sorted(tagged, key=lambda r: (-r["score"], r["candidate_id"]))[:6]
    bottom = sorted(tagged, key=lambda r: (r["score"], r["candidate_id"]))[:6]
    lines.append("Best-scored answer plays:")
    lines.append("")
    lines.append("| Card | Action | Score | Reason tags |")
    lines.append("|---|---|---|---|")
    for r in top:
        lines.append(f"| {r['card_name']} | {r['action_kind']} | {r['score']:.2f} | {', '.join(r['reason_tags'])} |")
    lines.append("")
    lines.append("Worst-scored plays (flagged mistakes):")
    lines.append("")
    lines.append("| Card | Action | Score | Reason tags |")
    lines.append("|---|---|---|---|")
    for r in bottom:
        lines.append(f"| {r['card_name']} | {r['action_kind']} | {r['score']:.2f} | {', '.join(r['reason_tags'])} |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Candidates are derived from each deck's own answer/attacker/engine cards, NOT "
        "from a live engine action/state stream — no game state is fabricated. The scorer "
        "is an offline interface; integrating it into the runtime policy is out of scope."
    )
    lines.append(
        "- Scores are relative, transparent component sums for ranking/inspection — they "
        "are **not** win probabilities."
    )
    lines.append(
        "- One-of tech answers are deliberately bounded (reduced credit, tech-timing "
        "penalty when unjustified) so they do not dominate scoring."
    )
    lines.append("- This package is never imported by the submission runtime or `cg` (INVARIANT #7/#8).")
    lines.append(f"- Reason-tag vocabulary is fixed at {len(REASON_TAGS)} tags; ordering is stable.")
    lines.append("")
    _ = reason_sort_key  # stable ordering helper used elsewhere
    return "\n".join(lines)
