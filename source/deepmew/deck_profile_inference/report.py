"""Generate the DeckProfileInference v0 report.

The report deliberately separates three layers:

* **Confirmed card-function evidence** — facts. Each cited card's function tags come
  directly from CardFunctionClassifier v2 card-text parsing.
* **Inferred deck profiles** — hypotheses. Count-weighted aggregations over the
  decklists. Archetype labels are display handles only, never inference inputs.
* **Unresolved / manual-review** — what a human should audit before these profiles
  feed any downstream stage.

No matchup win rates, no action scores, and no deck-name scripting appear here.
"""

from __future__ import annotations

from collections import Counter

from deepmew.deck_profile_inference.schemas import (
    ARTIFACT_VERSION,
    CardRef,
    DeckProfile,
    DEPENDENCY_KEYS,
    V2Card,
)


def _all_cited_card_ids(profile: DeckProfile) -> set[str]:
    ids: set[str] = set()

    def add(refs: list[CardRef]) -> None:
        ids.update(r.card_id for r in refs)

    for refs in (
        profile.main_attackers,
        profile.setup_attackers,
        profile.finishers,
        profile.draw_engine,
        profile.search_engine,
        profile.damage_counter_plan,
        profile.ability_damage_counter_plan,
        profile.switch_gust_plan,
        profile.lock_control_plan,
        profile.healing_recovery_plan,
        profile.disruption_plan,
        profile.status_plan,
    ):
        add(refs)
    for bucket in profile.bench_pressure.values():
        add(bucket)
    for bucket in profile.protection_plan.values():
        add(bucket)
    return ids


def build_report(
    profiles: list[DeckProfile],
    *,
    v2map: dict[str, V2Card],
    decklists_path: str = "",
    csv_output: str = "",
    jsonl_output: str = "",
) -> str:
    total = len(profiles)
    confidence = Counter(p.profile_confidence for p in profiles)
    reason_freq = Counter(t for p in profiles for t in p.reason_tags)
    choke_freq = Counter(c for p in profiles for c in p.choke_points)
    dep_freq = Counter(k for p in profiles for k in p.dependencies)
    review_profiles = [p for p in profiles if p.manual_review_reasons]

    card_appearances: Counter = Counter()
    for profile in profiles:
        for card_id in _all_cited_card_ids(profile):
            card_appearances[card_id] += 1

    lines: list[str] = []
    lines.append("# DeckProfileInference v0 Report")
    lines.append("")
    lines.append(f"Artifact: `{ARTIFACT_VERSION}` (offline/training-only).")
    lines.append("")
    lines.append(
        "Stage 2 of `card text -> function tags -> deck profile -> matchup threats -> "
        "action scores -> reason tags`. This stage only aggregates **function tags -> deck "
        "profile**. Archetype labels are display handles, never inference logic. **No** "
        "action scores, **no** trained labels, **no** matchup win rates."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Decklists (60-card, count source): `{decklists_path}`")
    lines.append(f"- Output CSV: `{csv_output}`")
    lines.append(f"- Output JSONL: `{jsonl_output}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Deck profiles generated: **{total}**")
    lines.append(
        f"- Profile confidence: high **{confidence.get('high', 0)}**, "
        f"medium **{confidence.get('medium', 0)}**, low **{confidence.get('low', 0)}**"
    )
    lines.append(f"- Profiles with manual-review reasons: **{len(review_profiles)}**")
    if total:
        avg_resolved = sum(p.resolved_card_share for p in profiles) / total
        lines.append(f"- Mean resolved-card share (cards found in v2): **{avg_resolved:.3f}**")
    lines.append("")

    # --- FACTS ---------------------------------------------------------------- #
    lines.append("## Confirmed card-function evidence (facts)")
    lines.append("")
    lines.append(
        "Each card's tags below are from v2 card-text parsing; the only aggregation here "
        "is how many of the inferred profiles cite the card in a plan."
    )
    lines.append("")
    lines.append("| Card | Decks citing | v2 tags |")
    lines.append("|---|---|---|")
    for card_id, count in card_appearances.most_common(20):
        card = v2map.get(card_id)
        if card is None:
            continue
        tags = ", ".join(t for t in sorted(card.tags) if t != "needs_manual_review")
        lines.append(f"| {card.name} ({card_id}) | {count} | {tags} |")
    lines.append("")

    # --- HYPOTHESES ----------------------------------------------------------- #
    lines.append("## Inferred deck profiles (hypotheses)")
    lines.append("")
    lines.append("### Reason-tag frequency (generalized deck identities)")
    lines.append("")
    lines.append("| Reason tag | Profiles |")
    lines.append("|---|---|")
    for tag, count in reason_freq.most_common():
        lines.append(f"| `{tag}` | {count} |")
    lines.append("")

    lines.append("### Dependency / choke-point frequency")
    lines.append("")
    lines.append("| Signal | Profiles |")
    lines.append("|---|---|")
    for key in DEPENDENCY_KEYS:
        if dep_freq.get(key):
            lines.append(f"| dependency `{key}` | {dep_freq[key]} |")
    for choke, count in choke_freq.most_common():
        lines.append(f"| choke `{choke}` | {count} |")
    lines.append("")

    lines.append("### Example inferred profiles (archetype label = display handle only)")
    lines.append("")
    lines.append("| Archetype (label) | Deck | Conf | Reason tags | Bench pressure kinds | Choke points |")
    lines.append("|---|---|---|---|---|---|")
    seen_labels: set[str] = set()
    for profile in sorted(profiles, key=lambda p: (p.archetype_label, p.deck_id)):
        if profile.archetype_label in seen_labels:
            continue
        seen_labels.add(profile.archetype_label)
        bench_kinds = ", ".join(profile.bench_pressure.keys()) or "-"
        reasons = ", ".join(profile.reason_tags) or "-"
        chokes = ", ".join(profile.choke_points) or "-"
        lines.append(
            f"| {profile.archetype_label} | {profile.deck_id} | {profile.profile_confidence} "
            f"| {reasons} | {bench_kinds} | {chokes} |"
        )
    lines.append("")

    # --- UNRESOLVED ----------------------------------------------------------- #
    lines.append("## Unresolved / manual-review cases")
    lines.append("")
    lines.append(f"- Profiles flagged for review: **{len(review_profiles)}** / {total}")
    reason_counter: Counter = Counter()
    for profile in review_profiles:
        for reason in profile.manual_review_reasons:
            # Bucket the dynamic counts into a stable family for aggregation.
            family = reason.split(" x")[0].split(" card copies")[0]
            reason_counter[family] += 1
    lines.append("")
    lines.append("| Manual-review reason family | Profiles |")
    lines.append("|---|---|")
    for family, count in reason_counter.most_common(15):
        lines.append(f"| {family} | {count} |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Profiles are count-weighted aggregations of card-text tags, not validated game "
        "outcomes. Choke points are structural dependencies (what *could* disrupt a deck), "
        "**not** matchup win rates."
    )
    lines.append(
        "- `bench_fragility` is a conservative heuristic (non-attacking low-HP support "
        "Pokémon copies) and should be treated as a hypothesis."
    )
    lines.append(
        "- Bench pressure is split three ways on purpose: ordinary bench attack damage, "
        "attack-based damage counters, and Ability-based damage counters are different "
        "threats answered differently."
    )
    lines.append("- This package is never imported by the submission runtime or `cg` (INVARIANT #7/#8).")
    lines.append("")
    return "\n".join(lines)
