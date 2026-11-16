from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from checkpoint_preservation import preserve_checkpoint
import run_unified_r4_ablation as r4
import main
from deepmew.imitation_fine.board import read_board
from deepmew.imitation_fine.labelers import label_decision
from deepmew.imitation_fine.schemas import GOAL_OF_PLAY_VOCAB, UNKNOWN, FineConfig
from deepmew.option_identity_bc import EncodedDecision
from deepmew.policy.selection import guard_selection
from deepmew.rl.observation import ObservationEncoder
from deepmew.rl.metadata import normalized_deck_hash
from deepmew.rl.opponents import OpponentAgent
from deepmew.rl1_teacher_bc.semantics import legal_action_table
from deepmew.second_order.reader import SECOND_ORDER_OPTION_SCHEMA, SECOND_ORDER_STATE_SCHEMA, augment_option_features


SCHEMA_VERSION = "deepmew.r9.r4f_retrain.v0"
DEFAULT_REGEN3 = Path(r"D:\DeepMew_artifacts\datasets\unified_r2\refgames_dec061_regen3_clean")
DEFAULT_OUT = ROOT / ".research" / "r9" / "lane2_r4f"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"
DEFAULT_TOP_PLAYER = ROOT / ".research" / "rl1_top_player_bc" / "replay_label_examples.jsonl"

CONFIDENT_THRESHOLD = 0.70
INTENT_VOCAB = tuple(GOAL_OF_PLAY_VOCAB)
INTENT_TO_INDEX = {label: index for index, label in enumerate(INTENT_VOCAB)}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _option_for_target(legal_set: list[dict[str, Any]], target: int) -> dict[str, Any]:
    if target < 0 or target >= len(legal_set):
        return {}

    return _as_dict(legal_set[target])


def _goal_of_play(row: dict[str, Any], legal_set: list[dict[str, Any]], target: int) -> str:
    existing = row.get("goal_of_play")
    if isinstance(existing, str) and existing in INTENT_TO_INDEX:
        return existing
    option = _option_for_target(legal_set, target)
    option_type = str(option.get("option_type") or option.get("type") or "")
    source_card = option.get("source_card_id", option.get("source_card"))
    target_card = option.get("target_card_id")
    card_id = source_card if source_card not in (None, "") else target_card
    target_area = str(option.get("target_area") or "")
    result = label_decision(option_type, _as_int(card_id, None), target_area, read_board(row.get("raw_observation")), FineConfig())

    return result.goal_of_play if result.goal_of_play in INTENT_TO_INDEX else UNKNOWN


def _augment_encoded(row: dict[str, Any], encoded: EncodedDecision, *, source: str) -> EncodedDecision | None:
    obs = _as_dict(row.get("raw_observation"))
    legal_set = [_as_dict(option) for option in _as_list(row.get("legal_set"))[: encoded.legal_count]]
    if not obs or len(legal_set) != encoded.legal_count:
        return None
    features = augment_option_features(obs, legal_set, encoded.features)
    metadata = dict(encoded.metadata)
    metadata.update(
        {
            "game_id": row.get("game_id") or row.get("episode_id"),
            "decision_index": row.get("decision_index") or row.get("step"),
            "chosen_indices": list(_as_list(_as_dict(row.get("chosen_action")).get("indices"))),
            "min_count": _as_int(_as_dict(row.get("action_mask")).get("min_count")),
            "max_count": _as_int(_as_dict(row.get("action_mask")).get("max_count")),
            "outcome": _as_dict(row.get("outcome")),
            "source": source,
            "rank_bucket": row.get("rank_bucket", ""),
        }
    )
    metadata["goal_of_play"] = _goal_of_play(row, legal_set, encoded.target_index)

    return EncodedDecision(features=features, target_index=encoded.target_index, legal_count=encoded.legal_count, metadata=metadata)


def encode_refgame_row_r9(row: dict[str, Any]) -> EncodedDecision | None:
    encoded = r4.build_decision_features(row)
    if encoded is None:
        return None

    return _augment_encoded(row, encoded, source="regen3")


def encode_top_player_row_r9(row: dict[str, Any]) -> EncodedDecision | None:
    observation = _as_dict(row.get("observation"))
    legal_set = [_as_dict(option) for option in _as_list(row.get("legal_action_table"))]
    selected = [_as_int(index, -1) for index in _as_list(row.get("selected_action_indices"))]
    selected = [index for index in selected if index >= 0]
    if not observation or not legal_set or not selected:
        return None
    wrapped = {
        "raw_observation": observation,
        "base84": ObservationEncoder().encode(observation),
        "policy_id": "top_player_replay",
        "deck_hash": row.get("deck_hash") or "",
        "legal_set": legal_set,
        "action_mask": {
            "n_options": len(legal_set),
            "min_count": _as_int(_as_dict(row.get("action_mask")).get("min_count")),
            "max_count": _as_int(_as_dict(row.get("action_mask")).get("max_count"), len(legal_set)),
        },
        "chosen_action": {"indices": selected},
        "outcome": {"won": row.get("reward") == 1, "lost": row.get("reward") == 0},
        "episode_id": row.get("episode_id"),
        "step": row.get("step"),
        "rank_bucket": row.get("rank_bucket"),
    }
    encoded = r4.build_decision_features(wrapped)
    if encoded is None:
        return None

    return _augment_encoded(wrapped, encoded, source="top_player")


def iter_refgame_encoded_r9(refgames_dir: Path, *, split: str, limit: int | None) -> Iterable[EncodedDecision]:
    emitted = 0
    for row in r4._iter_rows(refgames_dir, split=split, max_game_index=None):
        encoded = encode_refgame_row_r9(row)
        if encoded is None:
            continue
        yield encoded
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def iter_top_player_encoded(path: Path, *, limit: int | None) -> Iterable[EncodedDecision]:
    emitted = 0
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            encoded = encode_top_player_row_r9(json.loads(line))
            if encoded is None:
                continue
            yield encoded
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def collect_r9_validation(refgames_dir: Path, *, limit: int) -> list[EncodedDecision]:
    return list(iter_refgame_encoded_r9(refgames_dir, split="val", limit=limit))


@dataclass(frozen=True)
class R4FStepBatch:
    features: torch.Tensor
    option_meta: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor
    intent_target: torch.Tensor


def _intent_index(sample: r4.StepSample) -> int:
    return INTENT_TO_INDEX.get(str(sample.metadata.get("goal_of_play") or UNKNOWN), INTENT_TO_INDEX[UNKNOWN])


def _r4f_row_weight(sample: r4.StepSample) -> float:
    weight = float(sample.weight)
    source = str(sample.metadata.get("source") or "")
    context = str(sample.metadata.get("select_context") or "")
    if source == "top_player":
        weight *= 1.75
    if context in {"MAIN", "ATTACH_FROM"}:
        weight *= 1.20

    return max(0.05, min(4.0, weight))


def batch_steps_r4f(samples: list[r4.StepSample]) -> R4FStepBatch:
    features, option_meta, target, mask, weights = r4.batch_steps(samples)
    intent = torch.tensor([_intent_index(sample) for sample in samples], dtype=torch.long)
    adjusted = torch.tensor([_r4f_row_weight(sample) for sample in samples], dtype=torch.float32)

    return R4FStepBatch(features=features, option_meta=option_meta, target=target, mask=mask, weights=weights * adjusted, intent_target=intent)


class R4FIntentAutoregressiveScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, depth: int = 3, embed_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.encoder = r4.OptionEncoderScorer(input_dim, hidden_dim=hidden_dim, depth=depth, embed_dim=embed_dim, dropout=dropout).encoder
        self.option_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + 5, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.stop_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + 5, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.intent_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(INTENT_VOCAB)),
        )

    def encode_options(self, option_features: torch.Tensor) -> torch.Tensor:
        batch, options, dim = option_features.shape
        flat = option_features.reshape(batch * options, dim)

        return self.encoder(flat).reshape(batch, options, -1)

    def forward(self, option_features: torch.Tensor, option_meta: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_options(option_features)
        option_scores = self.option_head(torch.cat([embeddings, option_meta], dim=-1)).squeeze(-1)
        pooled = embeddings.mean(dim=1)
        step_meta = option_meta[:, 0, :] if option_meta.shape[1] else torch.zeros((option_features.shape[0], 5), dtype=option_features.dtype)
        stop_score = self.stop_head(torch.cat([pooled, step_meta], dim=-1)).squeeze(-1).unsqueeze(1)

        return torch.cat([option_scores, stop_score], dim=1)

    def intent_logits(self, option_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_options(option_features)

        return self.intent_head(embeddings.mean(dim=1))


def _copy_checkpoint_weights(model: R4FIntentAutoregressiveScorer, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_state = payload["model_state"]
    target_state = model.state_dict()
    copied = []
    partial = []
    for key, value in source_state.items():
        if key not in target_state:
            continue
        if target_state[key].shape == value.shape:
            target_state[key] = value
            copied.append(key)
            continue
        if key == "encoder.0.weight" and target_state[key].shape[0] == value.shape[0] and target_state[key].shape[1] > value.shape[1]:
            new_value = target_state[key].clone()
            new_value[:, : value.shape[1]] = value
            new_value[:, value.shape[1]:] = 0.0
            target_state[key] = new_value
            partial.append(key)
    model.load_state_dict(target_state)

    return {
        "source_checkpoint": str(checkpoint),
        "source_input_dim": int(payload["input_dim"]),
        "copied_keys": len(copied),
        "partial_keys": partial,
        "source_schema_version": payload.get("schema_version"),
    }


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, *, gamma: float) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)

    return ((1.0 - pt) ** gamma) * ce


def _iter_train_epoch(args: argparse.Namespace) -> Iterable[EncodedDecision]:
    top_player = list(iter_top_player_encoded(Path(args.top_player_examples), limit=args.top_player_rows)) if args.top_player_rows > 0 else []
    top_index = 0
    interval = max(1, int(args.top_player_interval))
    for ref_index, sample in enumerate(iter_refgame_encoded_r9(Path(args.refgames_dir), split="train", limit=args.train_rows if args.train_rows > 0 else None), start=1):
        yield sample
        if top_player and ref_index % interval == 0:
            yield top_player[top_index % len(top_player)]
            top_index += 1
    while top_player and top_index < len(top_player) and args.train_rows > 0:
        yield top_player[top_index]
        top_index += 1


def train_r4f(model: R4FIntentAutoregressiveScorer, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows_seen = 0
    step_rows_seen = 0
    batches = 0
    action_loss_total = 0.0
    intent_loss_total = 0.0
    source_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    while True:
        batch: list[r4.StepSample] = []
        for encoded in _iter_train_epoch(args):
            rows_seen += 1
            source_counts[str(encoded.metadata.get("source") or "")] += 1
            intent_counts[str(encoded.metadata.get("goal_of_play") or UNKNOWN)] += 1
            batch.extend(r4.make_step_samples(encoded, weighted=False))
            if len(batch) < args.batch_size:
                continue
            packed = batch_steps_r4f(batch)
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
            action_loss_by_row = focal_cross_entropy(scores, packed.target, gamma=args.focal_gamma)
            action_loss = (action_loss_by_row * packed.weights).sum() / packed.weights.sum()
            intent_loss = torch.nn.functional.cross_entropy(model.intent_logits(packed.features), packed.intent_target)
            loss = action_loss + args.intent_loss_weight * intent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_rows_seen += len(batch)
            batches += 1
            action_loss_total += float(action_loss.item())
            intent_loss_total += float(intent_loss.item())
            if args.progress_every and batches % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"r4f batches={batches} rows={rows_seen} steps={step_rows_seen} action_loss={action_loss.item():.6f} intent_loss={intent_loss.item():.6f} elapsed={elapsed:.1f}", file=sys.stderr, flush=True)
            batch = []
            if time.perf_counter() - started >= args.min_train_seconds:
                break
        if batch and time.perf_counter() - started < args.min_train_seconds:
            packed = batch_steps_r4f(batch)
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
            action_loss_by_row = focal_cross_entropy(scores, packed.target, gamma=args.focal_gamma)
            action_loss = (action_loss_by_row * packed.weights).sum() / packed.weights.sum()
            intent_loss = torch.nn.functional.cross_entropy(model.intent_logits(packed.features), packed.intent_target)
            loss = action_loss + args.intent_loss_weight * intent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_rows_seen += len(batch)
            batches += 1
            action_loss_total += float(action_loss.item())
            intent_loss_total += float(intent_loss.item())
        if time.perf_counter() - started >= args.min_train_seconds:
            break

    return {
        "rows_seen": rows_seen,
        "step_rows_seen": step_rows_seen,
        "batches": batches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_action_loss": round(action_loss_total / max(1, batches), 6),
        "mean_intent_loss": round(intent_loss_total / max(1, batches), 6),
        "source_counts": dict(source_counts),
        "top_intents": [{"intent": key, "rows": value} for key, value in intent_counts.most_common(12)],
    }


def select_r4f_from_features(
    model: R4FIntentAutoregressiveScorer,
    features: list[list[float]],
    legal_count: int,
    min_count: int,
    max_count: int,
) -> list[int]:
    min_count = max(0, min(min_count, legal_count))
    max_count = max(min_count, min(max_count or legal_count, legal_count))
    selected: list[int] = []
    for step_index in range(max_count + 1):
        step = r4.StepSample(
            features=features,
            legal_count=legal_count,
            selected=list(selected),
            target=legal_count,
            step_index=step_index,
            min_count=min_count,
            max_count=max_count,
            weight=1.0,
            metadata={},
        )
        packed = batch_steps_r4f([step])
        with torch.no_grad():
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)[0].numpy()
        choice = int(np.argmax(scores))
        if choice >= legal_count:
            break
        if choice not in selected:
            selected.append(choice)
        if len(selected) >= max_count:
            break

    return selected


def evaluate_r4f(model: R4FIntentAutoregressiveScorer, samples: list[EncodedDecision], *, batch_size: int) -> dict[str, Any]:
    model.eval()
    steps: list[r4.StepSample] = []
    for sample in samples:
        steps.extend(r4.make_step_samples(sample, weighted=False))
    first_steps = [step for step in steps if step.step_index == 0]
    exact = 0
    total = 0
    all_scores = []
    all_targets = []
    all_masks = []
    metadata = []
    intent_correct = 0
    intent_total = 0
    with torch.no_grad():
        for start in range(0, len(first_steps), batch_size):
            chunk = first_steps[start:start + batch_size]
            packed = batch_steps_r4f(chunk)
            all_scores.append(model(packed.features, packed.option_meta))
            all_targets.append(packed.target)
            all_masks.append(packed.mask)
            metadata.extend(step.metadata for step in chunk)
            intent_pred = model.intent_logits(packed.features).argmax(dim=1)
            intent_correct += int((intent_pred == packed.intent_target).sum().item())
            intent_total += int(packed.intent_target.numel())
        for sample in samples[: min(5000, len(samples))]:
            expected = [int(index) for index in _as_list(sample.metadata.get("chosen_indices"))]
            predicted = select_r4f_from_features(
                model,
                sample.features,
                sample.legal_count,
                _as_int(sample.metadata.get("min_count")),
                _as_int(sample.metadata.get("max_count")),
            )
            exact += int(predicted == expected[:len(predicted)] and len(predicted) == len(expected))
            total += 1
    max_options = max(scores.shape[1] for scores in all_scores)
    padded_scores = []
    padded_masks = []
    for scores, mask in zip(all_scores, all_masks):
        if scores.shape[1] == max_options:
            padded_scores.append(scores)
            padded_masks.append(mask)
            continue
        pad = max_options - scores.shape[1]
        padded_scores.append(torch.nn.functional.pad(scores, (0, pad), value=-1e9))
        padded_masks.append(torch.nn.functional.pad(mask, (0, pad), value=False))
    scores = torch.cat(padded_scores, dim=0)
    target = torch.cat(all_targets, dim=0)
    mask = torch.cat(padded_masks, dim=0)
    metrics = r4.metrics_for_scores(scores, target, mask, metadata)
    metrics["sequence_exact_sampled"] = round(exact / max(1, total), 6)
    metrics["step_rows"] = len(steps)
    metrics["intent_top1"] = round(intent_correct / max(1, intent_total), 6)

    return metrics


def load_r4a_model(checkpoint: Path) -> r4.AutoregressiveOptionScorer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = r4.AutoregressiveOptionScorer(
        int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
        embed_dim=int(payload["embed_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    return model


def load_r4f_model(checkpoint: Path) -> tuple[R4FIntentAutoregressiveScorer, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = R4FIntentAutoregressiveScorer(
        int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
        embed_dim=int(payload["embed_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    return model, payload


class R4FLivePolicy:
    def __init__(self, label: str, checkpoint: Path, deck_policy: str):
        self.label = label
        self.checkpoint = checkpoint
        self.model, self.payload = load_r4f_model(checkpoint)
        self.agent = OpponentAgent.from_str(deck_policy)
        self.policy_id = deck_policy
        self.deck = list(self.agent.deck or main.read_deck_csv())
        self.deck_hash = self.agent.deck_identity.deck_hash or normalized_deck_hash(self.deck)

    def reset(self, seed: int) -> None:
        _ = seed

    def bind_timer(self, timer: Any | None) -> None:
        _ = timer

    def select(self, obs: dict[str, Any]) -> list[int]:
        select = _as_dict(obs.get("select"))
        if not select:
            return list(self.deck)
        table = legal_action_table(obs, select)
        encoded = r4.build_live_features(obs, table, policy_id=self.policy_id, deck_hash=self.deck_hash)
        if encoded is None:
            return guard_selection([0], select)
        features = augment_option_features(obs, table[: encoded.legal_count], encoded.features)
        min_count = _as_int(select.get("minCount"))
        max_count = _as_int(select.get("maxCount"), default=encoded.legal_count)
        chosen = select_r4f_from_features(self.model, features, encoded.legal_count, min_count, max_count)

        return guard_selection(chosen, select)


def collect_base_validation(refgames_dir: Path, *, limit: int) -> list[EncodedDecision]:
    return r4.collect_encoded(refgames_dir, split="val", limit=limit, max_game_index=None, feature_mode="base")


def action_ece_on_top_player(model: R4FIntentAutoregressiveScorer, examples: Path, *, limit: int, batch_size: int) -> dict[str, Any]:
    samples = [
        sample
        for sample in iter_top_player_encoded(examples, limit=limit)
        if _as_int(sample.metadata.get("max_count")) == 1 and _as_int(sample.metadata.get("min_count")) <= 1
    ]
    if not samples:
        return {"rows_scored": 0, "ece": None}
    confidences = []
    correct = []
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        first_steps = [r4.make_step_samples(sample, weighted=False)[0] for sample in chunk]
        packed = batch_steps_r4f(first_steps)
        with torch.no_grad():
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
            probs = torch.softmax(scores[:, :-1], dim=1)
        for sample_index, sample in enumerate(chunk):
            legal = sample.legal_count
            row_probs = probs[sample_index, :legal]
            conf = float(row_probs.max().item())
            pred = int(row_probs.argmax().item())
            target = int(sample.target_index)
            is_correct = int(pred == target)
            context = str(sample.metadata.get("select_context") or "")
            confidences.append(conf)
            correct.append(is_correct)
            by_context[context]["n"] += 1
            by_context[context]["correct"] += is_correct
            if conf >= CONFIDENT_THRESHOLD and not is_correct:
                by_context[context]["confident_wrong"] += 1
    ece = 0.0
    bins = []
    for bucket in range(10):
        lo = bucket / 10
        hi = (bucket + 1) / 10
        indices = [index for index, conf in enumerate(confidences) if (conf > lo or bucket == 0 and conf >= lo) and conf <= hi]
        if not indices:
            bins.append({"bucket": f"{lo:.1f}-{hi:.1f}", "count": 0, "mean_conf": None, "accuracy": None, "gap": None})
            continue
        mean_conf = sum(confidences[index] for index in indices) / len(indices)
        accuracy = sum(correct[index] for index in indices) / len(indices)
        gap = abs(mean_conf - accuracy)
        ece += len(indices) / len(confidences) * gap
        bins.append({"bucket": f"{lo:.1f}-{hi:.1f}", "count": len(indices), "mean_conf": round(mean_conf, 4), "accuracy": round(accuracy, 4), "gap": round(gap, 4)})

    return {
        "rows_scored": len(samples),
        "ece": round(ece, 6),
        "overall_agreement_with_expert": round(sum(correct) / len(correct), 6),
        "confident_wrong_count": sum(1 for conf, ok in zip(confidences, correct) if conf >= CONFIDENT_THRESHOLD and not ok),
        "by_context": {
            key: {
                "n": counts["n"],
                "agreement": round(counts["correct"] / counts["n"], 6),
                "confident_wrong": counts["confident_wrong"],
            }
            for key, counts in sorted(by_context.items())
        },
        "bins": bins,
    }


def save_r4f_checkpoint(path: Path, model: R4FIntentAutoregressiveScorer, args: argparse.Namespace, train: dict[str, Any], init: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_kind": "r4f_intent_autoregressive",
            "model_state": model.state_dict(),
            "input_dim": int(args.input_dim),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "embed_dim": int(args.embed_dim),
            "dropout": float(args.dropout),
            "feature_mode": "second_order",
            "intent_vocab": INTENT_VOCAB,
            "state_schema": SECOND_ORDER_STATE_SCHEMA,
            "option_schema": SECOND_ORDER_OPTION_SCHEMA,
            "train": train,
            "initialization": init,
        },
        path,
    )


def render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# R9 R4-F Retrain Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model_path']}`",
        f"- initialized_from: `{payload['initialization'].get('source_checkpoint')}`",
        f"- train_rows_seen: `{payload['train']['rows_seen']}`",
        f"- elapsed_seconds: `{payload['train']['elapsed_seconds']}`",
        f"- input_dim: `{payload['input_dim']}`",
        f"- added_second_order_dims: `{payload['added_second_order_dims']}`",
        "",
        "## Offline Common Split",
        "",
        "| model | top1 | top3 | seq exact | intent top1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["offline_common_split"].items():
        lines.append(
            f"| `{name}` | {metrics.get('top1')} | {metrics.get('top3')} | {metrics.get('sequence_exact_sampled')} | {metrics.get('intent_top1', '')} |"
        )
    calibration = payload.get("top_player_calibration") or {}
    lines.extend(
        [
            "",
            "## Top-Player Calibration",
            "",
            f"- rows_scored: `{calibration.get('rows_scored')}`",
            f"- action_ece: `{calibration.get('ece')}`",
            f"- confident_wrong_count: `{calibration.get('confident_wrong_count')}`",
            "",
            "## Caveats",
            "",
            "- This runner derives auxiliary intent labels with the existing K4.1 deterministic labeler when a persisted fine-label row is absent.",
            "- Full panel/probe adjudication is a separate R9 step after this offline gate; this script does not select or bundle a challenger.",
        ]
    )

    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_r9 = collect_r9_validation(Path(args.refgames_dir), limit=args.val_rows)
    if not val_r9:
        raise RuntimeError("no R9 validation rows found")
    args.input_dim = len(val_r9[0].features[0])
    torch.manual_seed(args.seed)
    model = R4FIntentAutoregressiveScorer(
        args.input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )
    init = _copy_checkpoint_weights(model, Path(args.init_checkpoint)) if args.init_checkpoint else {}
    train = train_r4f(model, args)
    metrics_r4f = evaluate_r4f(model, val_r9, batch_size=args.batch_size)
    val_base = collect_base_validation(Path(args.refgames_dir), limit=args.val_rows)
    r4a_model = load_r4a_model(Path(args.r4a_checkpoint))
    v1b_model = load_r4a_model(Path(args.v1b_checkpoint))
    metrics_r4a = r4.evaluate_autoregressive(r4a_model, val_base, batch_size=args.batch_size, weighted=False)
    metrics_v1b = r4.evaluate_autoregressive(v1b_model, val_base, batch_size=args.batch_size, weighted=False)
    calibration = action_ece_on_top_player(model, Path(args.top_player_examples), limit=args.calibration_rows, batch_size=args.batch_size)
    model_path = out_dir / "selected_ar_bc_r4f_candidate.pt"
    save_r4f_checkpoint(model_path, model, args, train, init)
    preservation = preserve_checkpoint(
        model_path,
        mission=str(args.backup_mission),
        name="r4f-candidate",
        reason="R4-F candidate checkpoint produced by R9/R12 source recipe",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_offline_not_panel_adjudicated",
        "model_path": str(model_path),
        "checkpoint_preservation": preservation,
        "input_dim": args.input_dim,
        "added_second_order_dims": len(SECOND_ORDER_STATE_SCHEMA) + len(SECOND_ORDER_OPTION_SCHEMA),
        "feature_schema": {
            "state": SECOND_ORDER_STATE_SCHEMA,
            "option": SECOND_ORDER_OPTION_SCHEMA,
        },
        "initialization": init,
        "train": train,
        "offline_common_split": {
            "R4-A_current": metrics_r4a,
            "v1b_incumbent": metrics_v1b,
            "R4-F_candidate": metrics_r4f,
        },
        "top_player_calibration": calibration,
        "decision_rule": "offline gate only here: seq-exact and MAIN/action ECE must improve before panel/probes; panel/probes still decide selection.",
    }
    _write_json(out_dir / "r4f_retrain_summary.json", payload)
    _write_text(out_dir / "r4f_retrain_summary.md", render_summary(payload))
    print(json.dumps({k: payload[k] for k in ("status", "input_dim", "train", "offline_common_split", "top_player_calibration")}, indent=2, sort_keys=True, default=str)[:6000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R9 Lane 2: R4-F second-order AR retrain with intent head + focal loss.")
    parser.add_argument("--refgames-dir", default=str(DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--r4a-checkpoint", default=str(DEFAULT_R4A))
    parser.add_argument("--v1b-checkpoint", default=str(DEFAULT_V1B))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_V1B))
    parser.add_argument("--top-player-examples", default=str(DEFAULT_TOP_PLAYER))
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--top-player-rows", type=int, default=5000)
    parser.add_argument("--top-player-interval", type=int, default=8)
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--calibration-rows", type=int, default=5000)
    parser.add_argument("--min-train-seconds", type=float, default=3600.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--intent-loss-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=904400)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--backup-mission", default="r12")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
