"""Frozen research-only diagnostic probe shared by all E1 stage boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


PROBE_SCHEMA_VERSION = "deepmew.e1_stage_probe.v1"
PROBE_SEEDS = (71003, 71011)


class StageProbeHardStop(RuntimeError):
    """The fixed stage probe configuration or result is invalid."""


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(data.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class StageProbeConfig:
    participant: str
    opponent_policy_identity: str
    opponent_checkpoint_sha256: str
    deck_identity: str
    opponent_deck_identity: str
    seeds: tuple[int, ...] = PROBE_SEEDS
    games_per_seed: int = 8
    alternating_seats: bool = True
    maximum_game_steps: int = 5000

    def validate(self) -> None:
        identities = (
            self.participant,
            self.opponent_policy_identity,
            self.opponent_checkpoint_sha256,
            self.deck_identity,
            self.opponent_deck_identity,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise StageProbeHardStop("fixed probe identities must be explicit")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise StageProbeHardStop("fixed probe seeds must be unique")
        if self.games_per_seed < 2 or self.games_per_seed % 2 != 0:
            raise StageProbeHardStop("fixed probe games must be positive and seat-balanced")
        if not self.alternating_seats or self.maximum_game_steps <= 0:
            raise StageProbeHardStop("fixed probe requires alternating seats and a step bound")

    @property
    def identity_sha256(self) -> str:
        self.validate()

        return _sha256_json(asdict(self))


@dataclass(frozen=True)
class ProbeGameResult:
    seed: int
    seat: int
    result: int
    steps: int
    legal_action_counts: tuple[int, ...]
    action_entropies: tuple[float, ...]
    critic_probabilities: tuple[float, ...]
    critic_targets: tuple[float, ...]
    q_ranking_correct: int = 0
    q_ranking_total: int = 0
    crash: bool = False
    illegal_action: bool = False
    max_step_timeout: bool = False
    incomplete: bool = False


ProbeRunner = Callable[[str, StageProbeConfig], Sequence[ProbeGameResult]]


def run_fixed_stage_probe(
    checkpoint_sha256: str,
    config: StageProbeConfig,
    runner: ProbeRunner,
) -> dict[str, Any]:
    config.validate()
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise StageProbeHardStop("probe checkpoint hash must be explicit")
    games = tuple(runner(checkpoint_sha256, config))
    expected = len(config.seeds) * config.games_per_seed
    if len(games) != expected:
        raise StageProbeHardStop(f"fixed probe completed {len(games)}/{expected} games")
    expected_schedule = [
        (seed, game_index % 2)
        for seed in config.seeds
        for game_index in range(config.games_per_seed)
    ]
    if [(game.seed, game.seat) for game in games] != expected_schedule:
        raise StageProbeHardStop("fixed probe seed or seat schedule drifted")
    if any(game.result not in {-1, 0, 1} for game in games):
        raise StageProbeHardStop("fixed probe result is invalid")
    wins = sum(game.result == 1 for game in games)
    losses = sum(game.result == -1 for game in games)
    draws = sum(game.result == 0 for game in games)
    decided = wins + losses
    entropies: dict[str, list[float]] = {}
    forced = 0
    decisions = 0
    value_squared_error = []
    value_log_loss = []
    for game in games:
        if len(game.legal_action_counts) != len(game.action_entropies):
            raise StageProbeHardStop("probe legal-count and entropy rows are misaligned")
        if len(game.critic_probabilities) != len(game.critic_targets):
            raise StageProbeHardStop("probe critic rows are misaligned")
        for cardinality, entropy in zip(game.legal_action_counts, game.action_entropies, strict=True):
            if cardinality < 1 or not math.isfinite(entropy):
                raise StageProbeHardStop("probe action diagnostic is invalid")
            entropies.setdefault(str(cardinality), []).append(float(entropy))
            forced += int(cardinality == 1)
            decisions += 1
        for probability, target in zip(game.critic_probabilities, game.critic_targets, strict=True):
            if not 0.0 <= probability <= 1.0 or target not in {0.0, 1.0}:
                raise StageProbeHardStop("probe critic diagnostic is invalid")
            clipped = min(max(probability, 1.0e-7), 1.0 - 1.0e-7)
            value_squared_error.append((probability - target) ** 2)
            value_log_loss.append(-(target * math.log(clipped) + (1.0 - target) * math.log(1.0 - clipped)))
    q_correct = sum(game.q_ranking_correct for game in games)
    q_total = sum(game.q_ranking_total for game in games)
    receipt = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe_config": asdict(config),
        "probe_config_sha256": config.identity_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "completed_games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "point_rate": (wins + 0.5 * draws) / len(games),
        "decisive_win_rate": wins / decided if decided else None,
        "crashes": sum(game.crash for game in games),
        "illegal_actions": sum(game.illegal_action for game in games),
        "max_step_timeouts": sum(game.max_step_timeout for game in games),
        "incomplete_games": sum(game.incomplete for game in games),
        "action_entropy_by_legal_cardinality": {
            key: sum(values) / len(values) for key, values in sorted(entropies.items(), key=lambda item: int(item[0]))
        },
        "forced_decision_rate": forced / decisions if decisions else None,
        "critic_brier": sum(value_squared_error) / len(value_squared_error) if value_squared_error else None,
        "critic_log_loss": sum(value_log_loss) / len(value_log_loss) if value_log_loss else None,
        "q_ranking_accuracy": q_correct / q_total if q_total else None,
        "q_ranking_correct": q_correct,
        "q_ranking_total": q_total,
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)

    return receipt


__all__ = [
    "ProbeGameResult",
    "StageProbeConfig",
    "StageProbeHardStop",
    "run_fixed_stage_probe",
]
