"""Decision-aligned public-belief opponent-response count-table ablation.

This is an offline diagnostic, not a deployable opponent model. Each target
action is represented from the responding actor's legitimate actor-visible
state and exact legal set. Deployable use would require a separately validated
public hidden-state sampler; true private opponent state is never accepted as a
runtime input here.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics as statistics_lib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from deepmew.artifacts.writers import write_json, write_text
from deepmew.imitation.io import (
    iter_action_rows,
    load_episode_seat_decks,
    load_label_map,
    load_split_map,
)
from deepmew.imitation_fine.board import read_board
from deepmew.meta_kg.belief_correction import build_no_staple_model, fingerprint_unique_decks
from deepmew.meta_kg.belief_evaluation import (
    BeliefStatistics,
    extract_public_reveals,
    fit_belief_statistics,
    mapping_fingerprint,
)
from deepmew.meta_kg.io import aggregate_unique_decks, load_deck_artifacts
from deepmew.rl1_teacher_bc.semantics import legal_action_table


SCHEMA_VERSION = "deepmew.opponent_response_belief_ablation.v0"
MODEL_VERSION = "deepmew.response_count_table.v0"
CANONICAL_SPLIT_SCHEMA = "deepmew.value_q_split_manifest.v2"
RESPONSE_INPUT_MODE = "sampled_actor_visible_required"
EVIDENCE_SCALE = 1.0
MIN_CELL_SUPPORT = 8
SMOOTHING_ALPHA = 0.5
ALLOWED_RESPONSE_SPLITS = ("train", "validation", "calibration")

Candidate = tuple[str, str]
StateKey = tuple[str, ...]


@dataclass(frozen=True)
class ResponseExample:
    episode_id: str
    response_split: str
    deck_split: str
    archetype: str
    state_key: StateKey
    legal_candidates: tuple[Candidate, ...]
    chosen_candidate: Candidate
    revealed_card_ids: tuple[str, ...]

    @property
    def action_family(self) -> str:
        return self.chosen_candidate[1]


@dataclass
class CountCell:
    chosen: Counter[Candidate] = field(default_factory=Counter)
    total: int = 0

    def observe(self, candidate: Candidate) -> None:
        self.chosen[candidate] += 1
        self.total += 1


@dataclass
class ResponseCountModel:
    min_support: int = MIN_CELL_SUPPORT
    alpha: float = SMOOTHING_ALPHA
    global_cell: CountCell = field(default_factory=CountCell)
    context_cells: dict[tuple[str, ...], CountCell] = field(default_factory=dict)
    state_cells: dict[StateKey, CountCell] = field(default_factory=dict)
    archetype_context_cells: dict[tuple[str, ...], CountCell] = field(default_factory=dict)
    archetype_state_cells: dict[tuple[str, ...], CountCell] = field(default_factory=dict)

    def observe(self, example: ResponseExample) -> None:
        context_key = example.state_key[:2]
        archetype_context_key = (example.archetype, *context_key)
        archetype_state_key = (example.archetype, *example.state_key)
        self.global_cell.observe(example.chosen_candidate)
        _cell(self.context_cells, context_key).observe(example.chosen_candidate)
        _cell(self.state_cells, example.state_key).observe(example.chosen_candidate)
        _cell(self.archetype_context_cells, archetype_context_key).observe(example.chosen_candidate)
        _cell(self.archetype_state_cells, archetype_state_key).observe(example.chosen_candidate)

    def probabilities_global(self, example: ResponseExample) -> dict[Candidate, float]:
        return self._probabilities(self.global_cell, example.legal_candidates)

    def probabilities_state(self, example: ResponseExample) -> dict[Candidate, float]:
        context_key = example.state_key[:2]
        cells = (
            self.state_cells.get(example.state_key),
            self.context_cells.get(context_key),
            self.global_cell,
        )

        return self._probabilities(_supported_cell(cells, self.min_support), example.legal_candidates)

    def probabilities_archetype(
        self,
        example: ResponseExample,
        archetype: str,
    ) -> dict[Candidate, float]:
        context_key = example.state_key[:2]
        cells = (
            self.archetype_state_cells.get((archetype, *example.state_key)),
            self.archetype_context_cells.get((archetype, *context_key)),
            self.state_cells.get(example.state_key),
            self.context_cells.get(context_key),
            self.global_cell,
        )

        return self._probabilities(_supported_cell(cells, self.min_support), example.legal_candidates)

    def probabilities_belief(
        self,
        example: ResponseExample,
        posterior: Mapping[str, float],
    ) -> dict[Candidate, float]:
        mixed = {candidate: 0.0 for candidate in example.legal_candidates}
        posterior_total = sum(max(0.0, float(weight)) for weight in posterior.values())
        if posterior_total <= 0.0:
            return self.probabilities_state(example)
        for archetype, raw_weight in posterior.items():
            weight = max(0.0, float(raw_weight)) / posterior_total
            if weight == 0.0:
                continue
            component = self.probabilities_archetype(example, archetype)
            for candidate in mixed:
                mixed[candidate] += weight * component[candidate]

        return _normalize(mixed)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        header = {
            "schema_version": MODEL_VERSION,
            "min_support": self.min_support,
            "alpha": self.alpha,
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(json.dumps(("global", _cell_payload(self.global_cell)), separators=(",", ":")).encode("utf-8"))
        for name, table in (
            ("context", self.context_cells),
            ("state", self.state_cells),
            ("archetype_context", self.archetype_context_cells),
            ("archetype_state", self.archetype_state_cells),
        ):
            for key, cell in sorted(table.items()):
                digest.update(
                    json.dumps(
                        (name, list(key), _cell_payload(cell)),
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                )

        return digest.hexdigest()

    def _probabilities(
        self,
        cell: CountCell,
        candidates: Sequence[Candidate],
    ) -> dict[Candidate, float]:
        scores = {
            candidate: float(cell.chosen.get(candidate, 0)) + self.alpha
            for candidate in candidates
        }

        return _normalize(scores)


@dataclass
class MetricAccumulator:
    nll: float = 0.0
    brier: float = 0.0
    top1: int = 0
    top3: int = 0
    confidence_rows: list[tuple[float, int]] = field(default_factory=list)
    samples: int = 0

    def add(
        self,
        probabilities: Mapping[Candidate, float],
        chosen: Candidate,
    ) -> tuple[float, int]:
        probability = max(float(probabilities.get(chosen, 0.0)), 1e-12)
        loss = -math.log(probability)
        ordered = sorted(probabilities, key=lambda candidate: (-probabilities[candidate], candidate))
        correct = int(bool(ordered) and ordered[0] == chosen)
        self.nll += loss
        self.brier += sum(
            (float(value) - float(candidate == chosen)) ** 2
            for candidate, value in probabilities.items()
        )
        self.top1 += correct
        self.top3 += int(chosen in ordered[:3])
        self.confidence_rows.append((float(probabilities[ordered[0]]) if ordered else 0.0, correct))
        self.samples += 1

        return loss, correct

    def result(self, elapsed_seconds: float) -> dict[str, Any]:
        if not self.samples:
            raise ValueError("response metrics require at least one sample")

        return {
            "samples": self.samples,
            "nll": round(self.nll / self.samples, 6),
            "brier": round(self.brier / self.samples, 6),
            "top1_accuracy": round(self.top1 / self.samples, 6),
            "top3_accuracy": round(self.top3 / self.samples, 6),
            "top_label_ece": round(_expected_calibration_error(self.confidence_rows), 6),
            "inference_ms_per_sample": round(1000.0 * elapsed_seconds / self.samples, 6),
        }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _cell(table: dict[Any, CountCell], key: Any) -> CountCell:
    cell = table.get(key)
    if cell is None:
        cell = CountCell()
        table[key] = cell

    return cell


def _supported_cell(cells: Iterable[CountCell | None], min_support: int) -> CountCell:
    fallback = None
    for cell in cells:
        if cell is None:
            continue
        fallback = cell
        if cell.total >= min_support:
            return cell
    if fallback is None:
        raise ValueError("response model has no fallback count cell")

    return fallback


def _normalize(scores: Mapping[Candidate, float]) -> dict[Candidate, float]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)

        return {candidate: uniform for candidate in scores}

    return {candidate: max(0.0, float(value)) / total for candidate, value in scores.items()}


def _cell_payload(cell: CountCell) -> dict[str, Any]:
    return {
        "total": cell.total,
        "chosen": sorted((list(candidate), count) for candidate, count in cell.chosen.items()),
    }


def load_canonical_kaggle_splits(path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CANONICAL_SPLIT_SCHEMA or payload.get("status") != "accepted":
        raise ValueError("response ablation requires the accepted canonical split manifest")
    assignment = payload.get("assignment")
    if not isinstance(assignment, dict):
        raise ValueError("canonical split manifest lacks assignment")
    result = {}
    for canonical_game_id, split in assignment.items():
        if not str(canonical_game_id).startswith("kaggle_ingest:"):
            continue
        episode_id = str(canonical_game_id).split(":", 2)[-1]
        previous = result.get(episode_id)
        if previous is not None and previous != split:
            raise ValueError(f"conflicting canonical split for Kaggle episode {episode_id}")
        result[episode_id] = str(split)
    if not result:
        raise ValueError("canonical split manifest contains no Kaggle episodes")

    return result, {
        "path": str(path),
        "sha256": _sha256(path),
        "assignment_sha256": payload.get("assignment_sha256"),
        "kaggle_episode_counts": dict(sorted(Counter(result.values()).items())),
    }


def _bucket(value: int, boundaries: Sequence[int]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"le{boundary}"

    return f"gt{boundaries[-1]}"


def _hp_bucket(current: int, maximum: int) -> str:
    if maximum <= 0:
        return "empty"
    fraction = max(0.0, min(1.0, current / maximum))
    if fraction <= 0.33:
        return "low"
    if fraction <= 0.67:
        return "mid"

    return "high"


def response_state_key(
    state: dict[str, Any],
    candidates: Sequence[Candidate],
    *,
    select_type: str,
    select_context: str,
) -> StateKey:
    board = read_board(state)
    if not board.valid:
        raise ValueError("response state requires a valid two-seat actor-visible board")
    family_signature = "+".join(sorted({candidate[1] for candidate in candidates}))
    prize_bucket = "ahead" if board.prize_deficit < 0 else "behind" if board.prize_deficit > 0 else "tied"

    return (
        select_type,
        select_context,
        _bucket(board.turn, (1, 3, 5, 8)),
        prize_bucket,
        str(board.my_active_id or 0),
        _hp_bucket(board.my_active_hp, board.my_active_max_hp),
        _bucket(board.my_active_energy, (0, 1, 2, 3)),
        _bucket(board.my_bench_count, (0, 2, 4)),
        str(board.opp_active_id or 0),
        _hp_bucket(board.opp_active_hp, board.opp_active_max_hp),
        _bucket(board.opp_bench_count, (0, 2, 4)),
        _bucket(board.my_hand_count, (2, 5, 8)),
        _bucket(board.my_deck_count, (6, 15, 30)),
        family_signature,
    )


def build_response_example(
    row: dict[str, Any],
    *,
    response_split: str,
    deck_split: str,
    archetype: str,
    revealed_card_ids: Sequence[str],
) -> ResponseExample | None:
    state = row.get("state_json")
    if not isinstance(state, dict):
        return None
    select = state.get("select")
    if not isinstance(select, dict):
        return None
    semantic_options = legal_action_table(state, select)
    if not semantic_options:
        return None
    chosen_indices = row.get("chosen_indices")
    if not isinstance(chosen_indices, list) or len(chosen_indices) != 1:
        return None
    chosen_index = chosen_indices[0]
    if isinstance(chosen_index, bool) or not isinstance(chosen_index, int):
        return None
    if chosen_index < 0 or chosen_index >= len(semantic_options):
        return None

    candidates_by_semantic: dict[str, Candidate] = {}
    for option in semantic_options:
        semantic_key = str(option.get("semantic_key") or "")
        family = str(option.get("option_type") or "UNKNOWN")
        if not semantic_key:
            continue
        candidates_by_semantic.setdefault(semantic_key, (semantic_key, family))
    chosen_option = semantic_options[chosen_index]
    chosen_key = str(chosen_option.get("semantic_key") or "")
    chosen_candidate = candidates_by_semantic.get(chosen_key)
    if chosen_candidate is None or not candidates_by_semantic:
        return None
    candidates = tuple(sorted(candidates_by_semantic.values()))
    if len(candidates) < 2:
        return None
    select_type = str(semantic_options[0].get("select_type") or "UNKNOWN")
    select_context = str(semantic_options[0].get("select_context") or "UNKNOWN")

    return ResponseExample(
        episode_id=str(row.get("game_id") or ""),
        response_split=response_split,
        deck_split=deck_split,
        archetype=archetype,
        state_key=response_state_key(
            state,
            candidates,
            select_type=select_type,
            select_context=select_context,
        ),
        legal_candidates=candidates,
        chosen_candidate=chosen_candidate,
        revealed_card_ids=tuple(revealed_card_ids),
    )


def collect_response_data(
    rows: Iterable[dict[str, Any]],
    *,
    response_splits: Mapping[str, str],
    episode_seat_decks: Mapping[str, Mapping[int, str]],
    labels: Mapping[str, Mapping[str, Any]],
    deck_splits: Mapping[str, str],
    supported_archetypes: Sequence[str],
    max_games: int | None = None,
) -> tuple[ResponseCountModel, dict[str, list[ResponseExample]], dict[str, Any]]:
    supported = set(supported_archetypes)
    model = ResponseCountModel()
    examples = {"validation": [], "calibration": []}
    revealed_order: dict[tuple[str, int], list[str]] = defaultdict(list)
    revealed_seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    accepted_games: set[str] = set()
    stats = Counter()
    train_episodes: set[str] = set()
    evaluation_episodes: dict[str, set[str]] = {
        "validation": set(),
        "calibration": set(),
    }

    for row in rows:
        stats["rows_scanned"] += 1
        episode_id = str(row.get("game_id") or "")
        response_split = response_splits.get(episode_id)
        if response_split not in ALLOWED_RESPONSE_SPLITS:
            stats[f"skipped_response_split_{response_split or 'missing'}"] += 1
            continue
        if max_games is not None and episode_id not in accepted_games and len(accepted_games) >= max_games:
            stats["skipped_max_games"] += 1
            continue
        accepted_games.add(episode_id)
        seat = row.get("player_id")
        if seat not in (0, 1):
            stats["skipped_invalid_seat"] += 1
            continue
        state = row.get("state_json")
        if not isinstance(state, dict):
            stats["skipped_missing_state"] += 1
            continue

        reveal_key = (episode_id, int(seat))
        current_reveals = extract_public_reveals(state, int(seat))
        new_reveals = [card_id for card_id in current_reveals if card_id not in revealed_seen[reveal_key]]
        revealed_seen[reveal_key].update(new_reveals)
        revealed_order[reveal_key].extend(new_reveals)

        deck_hash = str(row.get("deck_hash") or episode_seat_decks.get(episode_id, {}).get(int(seat), ""))
        label = labels.get(deck_hash, {})
        archetype = str(label.get("archetype") or "")
        deck_split = str(deck_splits.get(deck_hash) or "missing")
        if label.get("classification_status") != "confident" or archetype not in supported:
            stats["skipped_unsupported_archetype"] += 1
            continue
        example = build_response_example(
            row,
            response_split=response_split,
            deck_split=deck_split,
            archetype=archetype,
            revealed_card_ids=revealed_order[reveal_key],
        )
        if example is None:
            stats["skipped_action_contract"] += 1
            continue
        stats["examples_built"] += 1
        stats[f"examples_{response_split}"] += 1
        stats[f"deck_split_{response_split}_{deck_split}"] += 1
        if response_split == "train":
            if deck_split != "train":
                stats["skipped_train_nontrain_deck"] += 1
                continue
            model.observe(example)
            stats["train_examples_used"] += 1
            train_episodes.add(episode_id)
        else:
            examples[response_split].append(example)
            evaluation_episodes[response_split].add(episode_id)

    split_episode_sets = {"train": train_episodes, **evaluation_episodes}
    for left, right in (("train", "validation"), ("train", "calibration"), ("validation", "calibration")):
        if split_episode_sets[left].intersection(split_episode_sets[right]):
            raise ValueError(f"response {left}/{right} episode overlap")
    stats["train_episodes"] = len(train_episodes)
    stats["validation_episodes"] = len(evaluation_episodes["validation"])
    stats["calibration_episodes"] = len(evaluation_episodes["calibration"])
    stats["episode_overlap"] = 0
    stats["calibration_examples_scored"] = len(examples["calibration"])
    stats["sealed_test_examples_scored"] = 0
    if model.global_cell.total == 0 or not examples["validation"] or not examples["calibration"]:
        raise ValueError("response ablation requires non-empty train, validation, and calibration examples")

    return model, examples, dict(sorted(stats.items()))


def _posterior(
    belief_model: Any,
    revealed_card_ids: Sequence[str],
) -> dict[str, float]:
    result = belief_model.update(revealed_card_ids, top_k=len(belief_model.archetype_priors))

    return {
        hypothesis.archetype: hypothesis.probability
        for hypothesis in result.hypotheses
    }


def _expected_calibration_error(rows: Sequence[tuple[float, int]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row for row in rows
            if lower <= row[0] < upper or (index == bins - 1 and row[0] == 1.0)
        ]
        if members:
            confidence = sum(row[0] for row in members) / len(members)
            accuracy = sum(row[1] for row in members) / len(members)
            error += len(members) / total * abs(confidence - accuracy)

    return error


def _reveal_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"

    return "8+"


def _slice_result(rows: Sequence[tuple[float, float, float, int, int, int]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "nll_delta_dsu_minus_state": round(sum(row[2] - row[0] for row in rows) / len(rows), 6),
        "nll_delta_dsu_minus_prior_mixture": round(
            sum(row[2] - row[1] for row in rows) / len(rows), 6
        ),
        "top1_delta_dsu_minus_state": round(sum(row[5] - row[3] for row in rows) / len(rows), 6),
        "top1_delta_dsu_minus_prior_mixture": round(
            sum(row[5] - row[4] for row in rows) / len(rows), 6
        ),
    }


def _clustered_delta(
    by_episode: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    means = [sum(values) / len(values) for values in by_episode.values() if values]
    if not means:
        raise ValueError("clustered response delta requires episode values")
    mean = sum(means) / len(means)
    se = statistics_lib.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0

    return {
        "episodes": len(means),
        "mean_nll_delta": round(mean, 6),
        "se": round(se, 6),
        "normal_95_ci": [round(mean - 1.96 * se, 6), round(mean + 1.96 * se, 6)],
    }


def evaluate_response_models(
    model: ResponseCountModel,
    examples: Sequence[ResponseExample],
    statistics: BeliefStatistics,
) -> dict[str, Any]:
    accumulators = {
        "OR0_legal_frequency": MetricAccumulator(),
        "OR1_state_only": MetricAccumulator(),
        "ORX_true_deck_upper_bound": MetricAccumulator(),
        "OR3a_archetype_prior_mixture": MetricAccumulator(),
        "OR3_dsu_deck_posterior": MetricAccumulator(),
    }
    elapsed = Counter()
    slices: dict[str, dict[str, list[tuple[float, float, float, int, int, int]]]] = {
        "reveal_count": defaultdict(list),
        "archetype": defaultdict(list),
        "action_family": defaultdict(list),
        "deck_split": defaultdict(list),
    }
    episode_deltas_vs_state: dict[str, list[float]] = defaultdict(list)
    episode_deltas_vs_prior: dict[str, list[float]] = defaultdict(list)
    belief_model = build_no_staple_model(statistics, evidence_scale=EVIDENCE_SCALE)

    for example in examples:
        started = time.perf_counter()
        global_probabilities = model.probabilities_global(example)
        elapsed["OR0_legal_frequency"] += time.perf_counter() - started

        started = time.perf_counter()
        state_probabilities = model.probabilities_state(example)
        elapsed["OR1_state_only"] += time.perf_counter() - started

        started = time.perf_counter()
        true_probabilities = model.probabilities_archetype(example, example.archetype)
        elapsed["ORX_true_deck_upper_bound"] += time.perf_counter() - started

        started = time.perf_counter()
        prior_probabilities = model.probabilities_belief(example, statistics.priors)
        elapsed["OR3a_archetype_prior_mixture"] += time.perf_counter() - started

        started = time.perf_counter()
        posterior = _posterior(belief_model, example.revealed_card_ids)
        dsu_probabilities = model.probabilities_belief(example, posterior)
        elapsed["OR3_dsu_deck_posterior"] += time.perf_counter() - started

        accumulators["OR0_legal_frequency"].add(global_probabilities, example.chosen_candidate)
        state_loss, state_correct = accumulators["OR1_state_only"].add(
            state_probabilities, example.chosen_candidate
        )
        accumulators["ORX_true_deck_upper_bound"].add(
            true_probabilities, example.chosen_candidate
        )
        prior_loss, prior_correct = accumulators["OR3a_archetype_prior_mixture"].add(
            prior_probabilities, example.chosen_candidate
        )
        dsu_loss, dsu_correct = accumulators["OR3_dsu_deck_posterior"].add(
            dsu_probabilities, example.chosen_candidate
        )
        row = (
            state_loss,
            prior_loss,
            dsu_loss,
            state_correct,
            prior_correct,
            dsu_correct,
        )
        slices["reveal_count"][_reveal_bucket(len(example.revealed_card_ids))].append(row)
        slices["archetype"][example.archetype].append(row)
        slices["action_family"][example.action_family].append(row)
        slices["deck_split"][example.deck_split].append(row)
        episode_deltas_vs_state[example.episode_id].append(dsu_loss - state_loss)
        episode_deltas_vs_prior[example.episode_id].append(dsu_loss - prior_loss)

    metrics = {
        name: accumulator.result(float(elapsed[name]))
        for name, accumulator in accumulators.items()
    }
    state = metrics["OR1_state_only"]
    prior = metrics["OR3a_archetype_prior_mixture"]
    dsu = metrics["OR3_dsu_deck_posterior"]
    true_deck = metrics["ORX_true_deck_upper_bound"]
    contribution = {
        "dsu_nll_delta_vs_state": round(dsu["nll"] - state["nll"], 6),
        "dsu_brier_delta_vs_state": round(dsu["brier"] - state["brier"], 6),
        "dsu_top1_delta_vs_state": round(dsu["top1_accuracy"] - state["top1_accuracy"], 6),
        "dsu_nll_delta_vs_prior_mixture": round(dsu["nll"] - prior["nll"], 6),
        "dsu_brier_delta_vs_prior_mixture": round(dsu["brier"] - prior["brier"], 6),
        "dsu_top1_delta_vs_prior_mixture": round(
            dsu["top1_accuracy"] - prior["top1_accuracy"], 6
        ),
        "true_deck_nll_delta_vs_state": round(true_deck["nll"] - state["nll"], 6),
        "true_deck_top1_delta_vs_state": round(
            true_deck["top1_accuracy"] - state["top1_accuracy"], 6
        ),
        "true_deck_nll_delta_vs_prior_mixture": round(true_deck["nll"] - prior["nll"], 6),
    }
    clustered = {
        "dsu_minus_state": _clustered_delta(episode_deltas_vs_state),
        "dsu_minus_prior_mixture": _clustered_delta(episode_deltas_vs_prior),
    }
    ci_upper = clustered["dsu_minus_prior_mixture"]["normal_95_ci"][1]
    dsu_passes_proxy = (
        contribution["dsu_nll_delta_vs_prior_mixture"] < 0.0
        and ci_upper < 0.0
        and contribution["dsu_top1_delta_vs_prior_mixture"] >= 0.0
    )
    feature_decision = "diagnostic only" if dsu_passes_proxy else "quarantine"

    failure_slices = {
        dimension: {
            key: _slice_result(rows)
            for key, rows in sorted(groups.items())
            if rows
        }
        for dimension, groups in slices.items()
    }
    damaged_slices = [
        f"{dimension}:{key}"
        for dimension, groups in failure_slices.items()
        for key, metrics in groups.items()
        if metrics["nll_delta_dsu_minus_prior_mixture"] > 0.0
    ]
    feature_audit = {
        "feature": "DSU strict-public deck posterior in opponent-response prediction",
        "purpose": "condition a legal-choice response table on inferred opponent deck structure",
        "baseline": "OR3a_archetype_prior_mixture on the identical scored decisions and legal candidate sets",
        "benefit": contribution,
        "cost": {
            "prior_mixture_ms_per_sample": prior["inference_ms_per_sample"],
            "dsu_ms_per_sample": dsu["inference_ms_per_sample"],
            "incremental_ms_per_sample": round(
                dsu["inference_ms_per_sample"] - prior["inference_ms_per_sample"], 6
            ),
        },
        "damage": {"positive_nll_delta_slices": damaged_slices},
        "uniqueness": "bounded by the true-deck upper-bound comparison; no gameplay wins measured",
        "precision": "not_applicable_variable_legal_set_multiclass",
        "recall": "not_applicable_variable_legal_set_multiclass",
        "mean_true_objective_delta": "not_measured",
        "throughput_impact": "reported as incremental inference ms/sample",
        "marginal_value_per_second": "not_estimable_from_offline_action-likelihood proxy",
        "failure_clustering": damaged_slices,
        "decision": feature_decision,
        "next_action": (
            "collect a new response-specific holdout before any Q proposal"
            if feature_decision == "diagnostic only"
            else "keep DSU out of response/Q and inspect clustered damage only if a new signal appears"
        ),
    }

    return {
        "metrics": metrics,
        "marginal_contribution": contribution,
        "game_clustered_delta": clustered,
        "failure_slices": failure_slices,
        "feature_audit": feature_audit,
        "dsu_response_feature_decision": feature_decision,
        "classification_reason": (
            "chronology-split offline proxy improvement under paired episode groups; no gameplay value measured"
            if feature_decision == "diagnostic only"
            else "DSU response conditioning did not clear the paired validation proxy gate"
        ),
        "unavailable_ablations": {
            "OR2_known_policy_family": "not_found_in_kaggle_replay_rows",
            "OR4_playstyle": "playstyle_inference_not_implemented",
            "OR5_playstyle_strength": "playstyle_missing_and_strength_only_partial_metadata",
            "OR6_value_trend": "decision_aligned_oof_value_join_not_implemented",
        },
        "marginal_value_per_second": "not_estimable_from_offline_action-likelihood proxy",
    }


def render_response_report(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    primary = evaluation["calibration"]
    contribution = primary["marginal_contribution"]
    clustered = primary["game_clustered_delta"]
    clustered_unique = clustered["dsu_minus_prior_mixture"]
    lines = [
        "# Opponent-response DSU contribution ablation",
        "",
        f"Verdict: **{evaluation['dsu_response_feature_decision']}**.",
        "",
        "This is an offline train/validation/calibration diagnostic. It is not wired into V, Q, policy, runtime, or submissions.",
        "Exact actor legal actions and private hand are legitimate only inside the responding actor view; deployable use remains blocked without a validated public hidden-state sampler.",
        "",
        "## Split and boundary",
        "",
        f"- Canonical response train episodes: {result['dataset']['train_episodes']}.",
        f"- Canonical validation episodes: {result['dataset']['validation_episodes']}.",
        f"- Canonical calibration episodes: {result['dataset']['calibration_episodes']}.",
        f"- Cross-split episode overlap: {result['dataset']['episode_overlap']}.",
        f"- Canonical calibration examples scored: {result['dataset']['calibration_examples_scored']}.",
        f"- Sealed-test examples scored: {result['dataset']['sealed_test_examples_scored']}.",
        "- Response training excludes actors whose deck hash is not in the K3 train split.",
        "",
        "## Chronological calibration ablation",
        "",
        "| Variant | NLL | Brier | Top-1 | Top-3 | ECE | ms/row |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in primary["metrics"].items():
        lines.append(
            f"| {name} | {metrics['nll']:.4f} | {metrics['brier']:.4f} | "
            f"{metrics['top1_accuracy']:.4f} | {metrics['top3_accuracy']:.4f} | "
            f"{metrics['top_label_ece']:.4f} | {metrics['inference_ms_per_sample']:.4f} |"
        )
    lines.extend([
        "",
        "## Marginal contribution",
        "",
        f"- DSU minus state NLL: `{contribution['dsu_nll_delta_vs_state']:+.6f}`.",
        f"- DSU minus state Brier: `{contribution['dsu_brier_delta_vs_state']:+.6f}`.",
        f"- DSU minus state top-1: `{contribution['dsu_top1_delta_vs_state']:+.6f}`.",
        f"- DSU minus archetype-prior mixture NLL: `{contribution['dsu_nll_delta_vs_prior_mixture']:+.6f}`.",
        f"- DSU minus archetype-prior mixture Brier: `{contribution['dsu_brier_delta_vs_prior_mixture']:+.6f}`.",
        f"- DSU minus archetype-prior mixture top-1: `{contribution['dsu_top1_delta_vs_prior_mixture']:+.6f}`.",
        f"- True-deck upper-bound minus state NLL: `{contribution['true_deck_nll_delta_vs_state']:+.6f}`.",
        f"- Game-clustered posterior-minus-prior NLL delta: `{clustered_unique['mean_nll_delta']:+.6f} +/- {clustered_unique['se']:.6f} SE`; 95% CI `{clustered_unique['normal_95_ci']}` over {clustered_unique['episodes']} episodes.",
        "",
        f"Classification reason: {evaluation['classification_reason']}.",
        f"Validation posterior-minus-prior NLL delta before this gate: `{evaluation['validation']['marginal_contribution']['dsu_nll_delta_vs_prior_mixture']:+.6f}`.",
        "",
        "## Feature contribution audit",
        "",
        f"- Feature: {primary['feature_audit']['feature']}.",
        f"- Baseline: {primary['feature_audit']['baseline']}.",
        f"- Cost: {primary['feature_audit']['cost']['incremental_ms_per_sample']:+.6f} ms/sample.",
        f"- Damage clusters: {primary['feature_audit']['damage']['positive_nll_delta_slices']}.",
        f"- Decision: **{primary['feature_audit']['decision']}**.",
        f"- Next action: {primary['feature_audit']['next_action']}.",
        "",
        "Policy-family, playstyle, strength-combination, and V-trend ablations remain explicitly unavailable; they were not silently approximated.",
        "",
    ])

    return "\n".join(lines)


def run_response_ablation(
    *,
    meta_kg_dir: str | Path,
    deck_zoo_dir: str | Path,
    ingest_root: str | Path,
    opponent_decks_root: str | Path,
    canonical_split_manifest: str | Path,
    dsu_correction_result: str | Path,
    output_dir: str | Path,
    max_games: int | None = None,
) -> dict[str, Any]:
    correction = json.loads(Path(dsu_correction_result).read_text(encoding="utf-8"))
    selected_variant = (correction.get("evaluation") or {}).get("selected_variant")
    if selected_variant != "B2_uncalibrated_no_staple":
        raise ValueError("response ablation requires the damage-gated uncalibrated no-staple posterior")
    if (correction.get("evaluation") or {}).get("feature_classification") != "diagnostic only":
        raise ValueError("response ablation requires a diagnostic-only DSU correction")

    labels = load_label_map(meta_kg_dir)
    deck_splits = load_split_map(deck_zoo_dir)
    deck_artifacts = load_deck_artifacts(opponent_decks_root)
    unique_decks, conflicts = aggregate_unique_decks(deck_artifacts)
    statistics = fit_belief_statistics(unique_decks, labels, deck_splits)
    response_splits, split_identity = load_canonical_kaggle_splits(canonical_split_manifest)
    episode_seat_decks = load_episode_seat_decks(opponent_decks_root)
    started = time.perf_counter()
    model, examples, dataset = collect_response_data(
        iter_action_rows(ingest_root),
        response_splits=response_splits,
        episode_seat_decks=episode_seat_decks,
        labels=labels,
        deck_splits=deck_splits,
        supported_archetypes=statistics.archetypes,
        max_games=max_games,
    )
    training_seconds = time.perf_counter() - started
    validation_evaluation = evaluate_response_models(model, examples["validation"], statistics)
    if validation_evaluation["dsu_response_feature_decision"] != "diagnostic only":
        raise ValueError("response DSU feature did not clear validation; calibration remains closed")
    calibration_evaluation = evaluate_response_models(model, examples["calibration"], statistics)
    final_decision = calibration_evaluation["dsu_response_feature_decision"]
    if final_decision == "diagnostic only":
        calibration_evaluation["feature_audit"]["next_action"] = (
            "keep DSU response diagnostic; predeclare an action-family gate and validate a public hidden-state sampler before Q"
        )
    evaluation = {
        "validation": validation_evaluation,
        "calibration": calibration_evaluation,
        "dsu_response_feature_decision": final_decision,
        "classification_reason": (
            "canonical chronological calibration confirms the paired offline proxy; no gameplay value measured"
            if final_decision == "diagnostic only"
            else "the validation DSU response signal did not replicate on canonical chronological calibration"
        ),
    }
    action_tables = (correction.get("input_identity") or {}).get("action_tables")
    result = {
        "schema_version": SCHEMA_VERSION,
        "information_mode": RESPONSE_INPUT_MODE,
        "deployment_status": "blocked_without_validated_public_hidden_state_sampler",
        "consumed_response_splits": ["train", "validation", "calibration"],
        "protected_splits": {"calibration_consumed": True, "sealed_test_consumed": False},
        "training": {
            "seconds": round(training_seconds, 6),
            "examples": model.global_cell.total,
            "min_cell_support": model.min_support,
            "smoothing_alpha": model.alpha,
            "model_sha256": model.fingerprint(),
            "archetypes": list(statistics.archetypes),
        },
        "dataset": dataset,
        "input_identity": {
            "meta_kg_dir": str(Path(meta_kg_dir)),
            "deck_zoo_dir": str(Path(deck_zoo_dir)),
            "ingest_root": str(Path(ingest_root)),
            "opponent_decks_root": str(Path(opponent_decks_root)),
            "canonical_split": split_identity,
            "dsu_correction_result": str(Path(dsu_correction_result)),
            "dsu_correction_sha256": _sha256(dsu_correction_result),
            "action_tables": action_tables,
            "label_split_sha256": mapping_fingerprint(labels, deck_splits),
            "unique_decks_sha256": fingerprint_unique_decks(unique_decks),
            "deck_hash_conflicts": len(conflicts),
            "implementation_sha256": _sha256(Path(__file__)),
        },
        "artifact_identity": {
            "checkpoint": "not_persisted_diagnostic_count_table_only",
            "generation": "offline_pre_generation_diagnostic",
            "deck_scope": "K3-train actors for fitting; canonical validation actors for scoring",
            "policy_family": "not_available",
            "league_snapshot": "not_applicable_offline_diagnostic",
            "safety_status": "not_instrumented_in_source_replays",
            "selection_status": "diagnostic_only_not_selected_for_runtime",
        },
        "evaluation": evaluation,
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    write_text(output_dir / "REPORT.md", render_response_report(result))

    return result
