"""Curriculum ladder: the default difficulty rungs, auto-advance, and the
INVARIANT #2 frozen-population guard.

The default ladder ascends both axes at once (harder decks AND stronger pilots)
while the gate *descends* -- because the higher ELO bands are structurally harder
for our archetype (e.g. the 900-1000 Archaludon wall OHKOs Mega Abomasnow, so a
0.70 gate up there is not reachable and would stall the curriculum forever). The
gate is "meaningfully positive at this difficulty", not a fixed 0.70.
"""

from __future__ import annotations

from deepmew.curriculum.schemas import (
    FROZEN_KINDS,
    CurriculumConfig,
    CurriculumRung,
    LeagueMember,
    RungResult,
)


def default_curriculum() -> CurriculumConfig:
    """The 5-rung default. Difficulty is monotonic non-decreasing (enforced by
    CurriculumConfig); gates descend as the bands get structurally harder."""
    return CurriculumConfig(
        rungs=(
            CurriculumRung(
                rung_id=0,
                name="warmup-600-800",
                target_bucket="600-800",
                pilot_tiers=("first-legal", "v2.4"),
                gate=0.70,
            ),
            CurriculumRung(
                rung_id=1,
                name="field-600-900-replay",
                target_bucket="800-900",
                pilot_tiers=("v2.4", "replay"),
                gate=0.65,
            ),
            CurriculumRung(
                rung_id=2,
                name="wall-800-1000-exploiter",
                target_bucket="900-1000",
                pilot_tiers=("v2.4", "replay", "exploiter"),
                gate=0.60,
                counter_shells=("Archaludon",),
            ),
            CurriculumRung(
                rung_id=3,
                name="top-1000-1100-exploiter",
                target_bucket="1000-1100",
                pilot_tiers=("replay", "exploiter"),
                gate=0.55,
                counter_shells=("Archaludon", "Crustle"),
            ),
            CurriculumRung(
                rung_id=4,
                name="apex-adversarial",
                target_bucket="1000-1100",
                pilot_tiers=("replay", "exploiter", "adversarial"),
                gate=0.52,
                counter_shells=("Archaludon", "Crustle", "Grimmsnarl"),
            ),
        )
    )


def advance(config: CurriculumConfig, result: RungResult) -> CurriculumRung | None:
    """Given a rung result, return the next rung to run, or None.

    None means either the candidate failed the gate (stay on this rung -- train
    more against it) or it cleared the final rung (curriculum complete). Use
    `result.passed` to disambiguate.
    """
    if not result.passed:
        return None
    return config.next_rung(result.rung_id)


def is_complete(config: CurriculumConfig, result: RungResult) -> bool:
    """True iff the candidate passed the final rung."""
    return result.passed and config.next_rung(result.rung_id) is None


def guard_invariant_2(members: list[LeagueMember], current_round: int) -> None:
    """Enforce INVARIANT #2: the opponent population is frozen, not co-adapting.

    Raises ValueError (which the coder must treat as a HALT + escalate) if:
      - any member's kind is not a frozen kind, or
      - any member was 'added' in a future round (i.e. co-trained with the
        candidate it is meant to evaluate -- the equilibrium-loop smell).

    This does NOT (and cannot) train anything to Nash; it only certifies that the
    league we hand to the arena is a fixed best-response population.
    """
    for member in members:
        if member.kind not in FROZEN_KINDS:
            raise ValueError(
                f"INVARIANT #2 violation: league member {member.member_id!r} kind "
                f"{member.kind!r} is not frozen; opponents must be a fixed population"
            )
        if member.round_added > current_round:
            raise ValueError(
                f"INVARIANT #2 violation: league member {member.member_id!r} was added in "
                f"round {member.round_added} > current round {current_round}; an opponent that "
                f"co-adapts with the candidate it evaluates is the self-play-to-Nash smell"
            )
