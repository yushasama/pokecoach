"""Compact exact diagnostics for full canonical-decision audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from deepmew.replay.contract.schemas import (
    AlignedDecision,
    CanonicalChosenAction,
    CanonicalDecisionRecord,
    CanonicalLegalActionSet,
    SourceEvent,
    ValidationOutcome,
)


AUDIT_DIAGNOSTICS_SCHEMA_VERSION = "deepmew.canonical_decision_audit_diagnostics.v1"
MAX_DIAGNOSTIC_SAMPLES = 5


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _nested_counter_payload(
    values: Mapping[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        key: _counter_payload(values[key])
        for key in sorted(values)
    }


@dataclass
class GameAuditStats:
    source_id: str
    source_version: str
    source_artifact_path: str | None = None
    decisions: int = 0
    complete_alignment: int = 0
    valid_actor_and_critic: int = 0
    forced_critic_only: int = 0
    unresolved_critic_only: int = 0
    rejected: int = 0
    ambiguous: int = 0
    deterministic_repair: int = 0
    repair_attempted: int = 0
    repair_rejected: int = 0
    stale_legacy_legal_set: int = 0
    terminal_rows: int = 0
    actor_indices: dict[int, list[int]] = field(default_factory=dict)
    global_decision_indices: list[int] = field(default_factory=list)
    source_event_positions: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_artifact_path": self.source_artifact_path,
            "decisions": self.decisions,
            "complete_alignment": self.complete_alignment,
            "valid_actor_and_critic": self.valid_actor_and_critic,
            "forced_critic_only": self.forced_critic_only,
            "unresolved_critic_only": self.unresolved_critic_only,
            "rejected": self.rejected,
            "ambiguous": self.ambiguous,
            "deterministic_repair": self.deterministic_repair,
            "repair_attempted": self.repair_attempted,
            "repair_rejected": self.repair_rejected,
            "stale_legacy_legal_set": self.stale_legacy_legal_set,
            "terminal_rows": self.terminal_rows,
            "actor_indices": {
                str(seat): values
                for seat, values in sorted(self.actor_indices.items())
            },
            "global_decision_indices": list(self.global_decision_indices),
            "source_event_positions": list(self.source_event_positions),
        }


@dataclass
class AuditDiagnostics:
    total_candidates: int = 0
    complete_causal_alignment: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    failures: Counter[str] = field(default_factory=Counter)
    first_failures: Counter[str] = field(default_factory=Counter)
    action_families: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    source_formats: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    source_players: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    choice_shapes: Counter[str] = field(default_factory=Counter)
    missing_choice_patterns: Counter[str] = field(default_factory=Counter)
    repair_rules: Counter[str] = field(default_factory=Counter)
    observation_assets: Counter[str] = field(default_factory=Counter)
    games: dict[str, GameAuditStats] = field(default_factory=dict)
    diagnostic_samples: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    exceptions: Counter[str] = field(default_factory=Counter)

    def _sample(self, key: str, value: Mapping[str, Any]) -> None:
        samples = self.diagnostic_samples[key]
        if len(samples) < MAX_DIAGNOSTIC_SAMPLES:
            samples.append(dict(value))

    def observe(
        self,
        *,
        source_event: SourceEvent,
        aligned: AlignedDecision,
        record: CanonicalDecisionRecord,
        legal_actions: CanonicalLegalActionSet,
        chosen: CanonicalChosenAction,
        corrected_pre_observation_reused: bool,
    ) -> None:
        self.total_candidates += 1
        validation = record.validation
        outcome = validation.outcome.value
        self.outcomes[outcome] += 1
        for failure in validation.failures:
            self.failures[failure.value] += 1
        if validation.first_failure is not None:
            self.first_failures[validation.first_failure.value] += 1
        proof = aligned.proof
        alignment_complete = bool(
            proof.pre_action_observation_id
            and proof.legal_observation_id == proof.pre_action_observation_id
            and proof.action_equality_proven
            and proof.actor_seat_proven
            and proof.event_order_proven
            and proof.resulting_transition_proven
            and proof.source_hash_bound
        )
        if alignment_complete:
            self.complete_causal_alignment += 1
        repair_attempted = proof.deterministic_repair_attempted
        repair_succeeded = repair_attempted and proof.deterministic_repair_applied
        repair_rejected = repair_attempted and not repair_succeeded
        selected_counts = [
            len(values) if values is not None else -1
            for values in aligned.stage_selected_choices
        ]
        total_selected = sum(max(0, value) for value in selected_counts)
        if any(value < 0 for value in selected_counts):
            shape = "unresolved"
        elif len(selected_counts) > 1:
            shape = "multi_stage"
        elif total_selected == 0:
            shape = "zero_choice"
        elif total_selected == 1:
            shape = "single_choice"
        else:
            shape = "multi_choice"
        self.choice_shapes[shape] += 1
        action_key = f"{legal_actions.action_family}|{legal_actions.action_template}"
        source_key = f"{source_event.source_id}|{source_event.source_version}"
        player_key = f"{source_key}|seat={source_event.actor_seat}"
        for counter in (
            self.action_families[action_key],
            self.source_formats[source_key],
            self.source_players[player_key],
        ):
            counter["total"] += 1
            counter[f"outcome:{outcome}"] += 1
            counter[f"choice:{shape}"] += 1
            if validation.failures:
                counter["semantic_mismatch"] += 1
            if proof.deterministic_repair_applied:
                counter["deterministic_repair"] += 1
            if repair_attempted:
                counter["repair_attempted"] += 1
                counter["repair_succeeded" if repair_succeeded else "repair_rejected"] += 1
        if repair_attempted:
            self.repair_rules[f"{proof.rule_id}:attempted"] += 1
            self.repair_rules[
                f"{proof.rule_id}:succeeded" if repair_succeeded
                else f"{proof.rule_id}:rejected"
            ] += 1
        stale_legacy = "admission_v0_legal_set_was_post_action" in proof.notes
        if stale_legacy:
            self.repair_rules["stale_legacy_legal_set_detected"] += 1
        asset = aligned.pre_action_event.observation_asset
        pre_asset_eligible = bool(
            alignment_complete
            and asset is not None
            and asset.observation_id == proof.pre_action_observation_id
            and asset.information_mode == "actor_visible"
            and len(asset.source_content_sha256) == 64
        )
        if not pre_asset_eligible:
            self.observation_assets["pre_observation_ineligible"] += 1
        elif source_event.source_format == "kaggle_replay":
            self.observation_assets[
                "normalized_corrected_kaggle_pre_reused"
                if corrected_pre_observation_reused
                else "normalized_corrected_kaggle_pre_new"
            ] += 1
            result_asset = (
                aligned.resulting_event.observation_asset
                if aligned.resulting_event is not None
                else None
            )
            self.observation_assets[
                "normalized_kaggle_result_reused"
                if (
                    result_asset is not None
                    and proof.resulting_transition_proven
                    and result_asset.observation_id == proof.resulting_observation_id
                    and len(result_asset.source_content_sha256) == 64
                )
                else "normalized_kaggle_result_ineligible"
            ] += 1
        elif asset.reusable_admission_asset:
            self.observation_assets["normalized_admission_pre_reused"] += 1
        else:
            self.observation_assets["normalized_admission_pre_new"] += 1
        if validation.failures:
            bounds = [
                f"{stage.minimum_count}:{stage.maximum_count}"
                for stage in legal_actions.stages
            ]
            pattern = (
                f"family={action_key}|selected={selected_counts}|bounds={bounds}|"
                f"first={validation.first_failure.value if validation.first_failure else 'none'}"
            )
            self.missing_choice_patterns[pattern] += 1
            self._sample(
                f"failure:{validation.first_failure.value if validation.first_failure else 'none'}",
                {
                    "binding_id": source_event.binding_id,
                    "local_row": source_event.local_row,
                    "global_row": source_event.global_row,
                    "source_game_id": source_event.source_game_id,
                    "actor_seat": source_event.actor_seat,
                    "causal_decision_index": aligned.causal_decision_index,
                    "action_family": action_key,
                    "selected_counts": selected_counts,
                    "bounds": bounds,
                    "outcome": outcome,
                    "failures": [failure.value for failure in validation.failures],
                },
            )
        ambiguous = outcome == ValidationOutcome.UNKNOWN_FAIL_CLOSED.value
        game = self.games.get(source_event.source_game_id)
        if game is None:
            game = GameAuditStats(
                source_id=source_event.source_id,
                source_version=source_event.source_version,
                source_artifact_path=source_event.source_artifact_path,
            )
            self.games[source_event.source_game_id] = game
        elif (
            game.source_id != source_event.source_id
            or game.source_version != source_event.source_version
            or game.source_artifact_path != source_event.source_artifact_path
        ):
            raise ValueError("canonical game identity has conflicting source lineage")
        game.decisions += 1
        game.complete_alignment += int(alignment_complete)
        game.valid_actor_and_critic += int(
            outcome == ValidationOutcome.VALID_ACTOR_AND_CRITIC.value
        )
        game.forced_critic_only += int(
            outcome == ValidationOutcome.VALID_CRITIC_ONLY_FORCED.value
        )
        game.unresolved_critic_only += int(
            outcome == ValidationOutcome.VALID_CRITIC_ONLY_UNRESOLVED_ACTION.value
        )
        game.rejected += int(outcome.startswith("REJECTED_") or ambiguous)
        game.ambiguous += int(ambiguous)
        game.deterministic_repair += int(proof.deterministic_repair_applied)
        game.repair_attempted += int(repair_attempted)
        game.repair_rejected += int(repair_rejected)
        game.stale_legacy_legal_set += int(stale_legacy)
        game.terminal_rows += int(bool(source_event.transition.get("terminal")))
        game.source_event_positions.append(source_event.source_event_position)
        actor_index = source_event.transition.get("actor_decision_index")
        if isinstance(actor_index, int) and not isinstance(actor_index, bool):
            game.actor_indices.setdefault(int(source_event.actor_seat), []).append(actor_index)
        global_index = source_event.transition.get("global_decision_index")
        if isinstance(global_index, int) and not isinstance(global_index, bool):
            game.global_decision_indices.append(global_index)

    def observe_exception(
        self,
        *,
        binding_id: str,
        local_row: int,
        exception: BaseException,
    ) -> None:
        self.total_candidates += 1
        self.outcomes[ValidationOutcome.UNKNOWN_FAIL_CLOSED.value] += 1
        self.exceptions[type(exception).__name__] += 1
        self._sample(
            f"exception:{type(exception).__name__}",
            {
                "binding_id": binding_id,
                "local_row": local_row,
                "exception_type": type(exception).__name__,
                "message": str(exception)[:400],
            },
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_DIAGNOSTICS_SCHEMA_VERSION,
            "total_candidates": self.total_candidates,
            "complete_causal_alignment": self.complete_causal_alignment,
            "outcomes": _counter_payload(self.outcomes),
            "failures": _counter_payload(self.failures),
            "first_failures": _counter_payload(self.first_failures),
            "action_families": _nested_counter_payload(self.action_families),
            "source_formats": _nested_counter_payload(self.source_formats),
            "source_players": _nested_counter_payload(self.source_players),
            "choice_shapes": _counter_payload(self.choice_shapes),
            "missing_choice_patterns": _counter_payload(self.missing_choice_patterns),
            "repair_rules": _counter_payload(self.repair_rules),
            "observation_assets": _counter_payload(self.observation_assets),
            "games": {
                game_id: self.games[game_id].to_payload()
                for game_id in sorted(self.games)
            },
            "diagnostic_samples": {
                key: list(self.diagnostic_samples[key])
                for key in sorted(self.diagnostic_samples)
            },
            "exceptions": _counter_payload(self.exceptions),
        }


__all__ = [
    "AUDIT_DIAGNOSTICS_SCHEMA_VERSION",
    "MAX_DIAGNOSTIC_SAMPLES",
    "AuditDiagnostics",
    "GameAuditStats",
]
