from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
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
from cg.api import OptionType, SelectContext

from checkpoint_preservation import preserve_checkpoint
from deepmew.option_identity_bc import EncodedDecision
from deepmew.rl1_teacher_bc.semantics import build_option_features, legal_action_table

DEFAULT_REGEN3 = Path(r"D:\DeepMew_artifacts\datasets\unified_r2\refgames_dec061_regen3_clean")
DEFAULT_OUT = ROOT / ".research" / "unified_r4" / "R4-A"
SCHEMA_VERSION = "deepmew.unified_r4_ablation.rebuilt.v0"
REFERENCE_POLICIES = ("reference_abomasnow", "reference_iono", "reference_dragapult", "reference_lucario")
HASH_BINS = 16


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_features(value: Any, bins: int = HASH_BINS) -> list[float]:
    if value is None or value == "":
        return [0.0] * bins
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()

    return [digest[index] / 255.0 for index in range(bins)]


def _policy_id_features(policy_id: Any) -> list[float]:
    text = str(policy_id or "")

    return [1.0 if text == policy else 0.0 for policy in REFERENCE_POLICIES]


def _option_type_name(option: dict[str, Any]) -> str:
    raw = option.get("type")
    try:
        return OptionType(raw).name
    except (TypeError, ValueError):
        return str(option.get("option_type") or "UNKNOWN")


def _active0(player: dict[str, Any]) -> dict[str, Any]:
    active = _as_list(player.get("active"))

    return _as_dict(active[0]) if active else {}


def _energy_count(pokemon: dict[str, Any]) -> int:
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    cards = pokemon.get("energyCards")
    if isinstance(cards, list):
        return len(cards)

    return _as_int(energies, 0)


def _hp_ratio(pokemon: dict[str, Any]) -> float:
    max_hp = _as_int(pokemon.get("maxHp"), 0)
    if max_hp <= 0:
        return 0.0
    hp = _as_int(pokemon.get("hp"), max_hp)

    return max(0.0, min(1.0, hp / max_hp))


def _bench_danger(obs: dict[str, Any]) -> list[float]:
    current = _as_dict(obs.get("current"))
    players = _as_list(current.get("players"))
    your_index = _as_int(current.get("yourIndex"))
    if len(players) < 2 or your_index not in (0, 1):
        return [0.0] * 6
    own = _as_dict(players[your_index])
    opp = _as_dict(players[1 - your_index])
    own_bench = len(_as_list(own.get("bench")))
    opp_bench = len(_as_list(opp.get("bench")))
    own_prizes = len(_as_list(own.get("prize")))
    opp_prizes = len(_as_list(opp.get("prize")))

    return [
        1.0 if own_bench == 0 else 0.0,
        min(own_bench, 5) / 5.0,
        min(opp_bench, 5) / 5.0,
        1.0 - _hp_ratio(_active0(own)),
        1.0 - _hp_ratio(_active0(opp)),
        (opp_prizes - own_prizes) / 6.0,
    ]


def encode_base(obs: dict[str, Any]) -> list[float]:
    current = _as_dict(obs.get("current"))
    select = _as_dict(obs.get("select"))
    players = _as_list(current.get("players"))
    your_index = _as_int(current.get("yourIndex"), 0)
    options = _as_list(select.get("option"))
    ctx_value = select.get("context")
    ctx_vec = [1.0 if ctx.value == ctx_value else 0.0 for ctx in SelectContext]
    opt_counts = {opt.name: 0 for opt in OptionType}
    for option in options:
        name = _option_type_name(_as_dict(option))
        if name in opt_counts:
            opt_counts[name] += 1
    denom = max(1, len(options))
    opt_vec = [opt_counts[opt.name] / denom for opt in OptionType]
    min_count = _as_int(select.get("minCount"), 0)
    max_count = _as_int(select.get("maxCount"), 0)
    if len(players) >= 2 and your_index in (0, 1):
        me = _as_dict(players[your_index])
        opp = _as_dict(players[1 - your_index])
        my_active = _active0(me)
        opp_active = _active0(opp)
        my_prizes = len(_as_list(me.get("prize")))
        opp_prizes = len(_as_list(opp.get("prize")))
        my_energy = _energy_count(my_active)
        my_bench = len(_as_list(me.get("bench")))
        opp_bench = len(_as_list(opp.get("bench")))
        my_hand = _as_int(me.get("handCount"), len(_as_list(me.get("hand"))))
        my_deck = _as_int(me.get("deckCount"), 60)
    else:
        my_prizes = opp_prizes = 6
        my_active = opp_active = {}
        my_energy = my_bench = opp_bench = my_hand = 0
        my_deck = 60
    first_player = _as_int(current.get("firstPlayer"), -1)

    scalars = [
        min(len(options), 30) / 30.0,
        min(min_count, 10) / 10.0,
        min(max_count, 10) / 10.0,
        min(_as_int(current.get("turn"), 0), 40) / 40.0,
        my_prizes / 6.0,
        opp_prizes / 6.0,
        _hp_ratio(my_active),
        _hp_ratio(opp_active),
        min(my_energy, 6) / 6.0,
        min(my_bench, 5) / 5.0,
        min(opp_bench, 5) / 5.0,
        min(my_hand, 12) / 12.0,
        min(my_deck, 60) / 60.0,
        1.0 if current.get("supporterPlayed") else 0.0,
        1.0 if current.get("stadiumPlayed") else 0.0,
        1.0 if current.get("energyAttached") else 0.0,
        1.0 if current.get("retreated") else 0.0,
        1.0 if first_player == your_index else 0.0,
    ]

    return ctx_vec + opt_vec + scalars


def build_feature_matrix(obs: dict[str, Any], legal_set: list[dict[str, Any]], *, policy_id: str, deck_hash: str) -> list[list[float]]:
    state = [float(value) for value in encode_base(obs)]
    state.extend(_bench_danger(obs))
    state.extend(_policy_id_features(policy_id))
    state.extend(_hash_features(deck_hash))

    return [state + build_option_features(_as_dict(option)) for option in legal_set]


def _chosen_indices(row: dict[str, Any]) -> list[int]:
    chosen = _as_dict(row.get("chosen_action"))

    return [_as_int(index, -1) for index in _as_list(chosen.get("indices")) if _as_int(index, -1) >= 0]


def build_decision_features(row: dict[str, Any]) -> EncodedDecision | None:
    obs = _as_dict(row.get("raw_observation"))
    select = _as_dict(obs.get("select"))
    if not obs or not select:
        return None
    legal_set = [_as_dict(option) for option in _as_list(row.get("legal_set"))]
    if not legal_set:
        legal_set = legal_action_table(obs, select)
    if not legal_set:
        return None
    legal_count = len(legal_set)
    chosen = _chosen_indices(row)
    target = chosen[0] if chosen else legal_count
    if target < 0 or target > legal_count:
        return None
    metadata = {
        "select_context": str(legal_set[0].get("select_context") or ""),
        "policy_id": row.get("policy_id") or "",
        "deck_hash": row.get("deck_hash") or "",
        "chosen_indices": chosen,
        "min_count": _as_int(_as_dict(row.get("action_mask")).get("min_count")),
        "max_count": _as_int(_as_dict(row.get("action_mask")).get("max_count"), legal_count),
        "game_index": row.get("game_index"),
        "decision_index": row.get("decision_index"),
        "outcome": _as_dict(row.get("outcome")),
    }
    features = build_feature_matrix(
        obs,
        legal_set,
        policy_id=str(row.get("policy_id") or ""),
        deck_hash=str(row.get("deck_hash") or ""),
    )

    return EncodedDecision(features=features, target_index=target, legal_count=legal_count, metadata=metadata)


def build_live_features(obs: dict[str, Any], legal_set: list[dict[str, Any]], *, policy_id: str, deck_hash: str) -> EncodedDecision | None:
    if not obs or not legal_set:
        return None
    features = build_feature_matrix(obs, legal_set, policy_id=policy_id, deck_hash=deck_hash)
    metadata = {
        "select_context": str(legal_set[0].get("select_context") or ""),
        "policy_id": policy_id,
        "deck_hash": deck_hash,
    }

    return EncodedDecision(features=features, target_index=len(legal_set), legal_count=len(legal_set), metadata=metadata)


def _iter_shard_paths(refgames_dir: Path) -> list[Path]:
    shard_dir = refgames_dir / "shards"

    return sorted(shard_dir.glob("decisions_shard_*.jsonl.gz"))


def _row_in_split(row: dict[str, Any], split: str) -> bool:
    if split not in {"train", "val", "all"}:
        raise ValueError(f"unknown split {split!r}")
    if split == "all":
        return True
    game_index = _as_int(row.get("game_index"), _as_int(row.get("seed"), 0))
    is_val = game_index % 5 == 0

    return is_val if split == "val" else not is_val


def _iter_rows(refgames_dir: Path, *, split: str, max_game_index: int | None = None) -> Iterable[dict[str, Any]]:
    for shard_path in _iter_shard_paths(refgames_dir):
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if max_game_index is not None and _as_int(row.get("game_index"), 0) > max_game_index:
                    continue
                if _row_in_split(row, split):
                    yield row


def iter_encoded(
    refgames_dir: Path,
    *,
    split: str,
    limit: int | None,
    max_game_index: int | None = None,
    feature_mode: str = "base",
) -> Iterable[EncodedDecision]:
    if feature_mode != "base":
        raise ValueError("rebuilt R4 ablation surface only supports feature_mode='base'")
    emitted = 0
    for row in _iter_rows(refgames_dir, split=split, max_game_index=max_game_index):
        encoded = build_decision_features(row)
        if encoded is None:
            continue
        yield encoded
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def collect_encoded(
    refgames_dir: Path,
    *,
    split: str,
    limit: int | None,
    max_game_index: int | None = None,
    feature_mode: str = "base",
) -> list[EncodedDecision]:
    return list(
        iter_encoded(
            refgames_dir,
            split=split,
            limit=limit,
            max_game_index=max_game_index,
            feature_mode=feature_mode,
        )
    )


@dataclass(frozen=True)
class StepSample:
    features: list[list[float]]
    legal_count: int
    selected: list[int]
    target: int
    step_index: int
    min_count: int
    max_count: int
    weight: float
    metadata: dict[str, Any]


def make_step_samples(sample: EncodedDecision, *, weighted: bool) -> list[StepSample]:
    selected = [int(index) for index in _as_list(sample.metadata.get("chosen_indices")) if int(index) < sample.legal_count]
    if not selected and sample.target_index < sample.legal_count:
        selected = [int(sample.target_index)]
    min_count = _as_int(sample.metadata.get("min_count"))
    max_count = _as_int(sample.metadata.get("max_count"), sample.legal_count)
    max_count = max(min_count, min(max_count, sample.legal_count))
    weight = 1.0
    if weighted and _as_dict(sample.metadata.get("outcome")).get("won"):
        weight = 1.25
    steps: list[StepSample] = []
    prefix: list[int] = []
    for step_index, choice in enumerate(selected[:max_count]):
        steps.append(
            StepSample(
                features=sample.features,
                legal_count=sample.legal_count,
                selected=list(prefix),
                target=choice,
                step_index=step_index,
                min_count=min_count,
                max_count=max_count,
                weight=weight,
                metadata=sample.metadata,
            )
        )
        prefix.append(choice)
    if len(prefix) >= min_count:
        steps.append(
            StepSample(
                features=sample.features,
                legal_count=sample.legal_count,
                selected=list(prefix),
                target=sample.legal_count,
                step_index=len(prefix),
                min_count=min_count,
                max_count=max_count,
                weight=weight,
                metadata=sample.metadata,
            )
        )

    return steps


def batch_steps(samples: list[StepSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("empty batch")
    max_options = max(sample.legal_count for sample in samples)
    input_dim = len(samples[0].features[0])
    features = torch.zeros((len(samples), max_options, input_dim), dtype=torch.float32)
    option_meta = torch.zeros((len(samples), max_options, 5), dtype=torch.float32)
    target = torch.zeros(len(samples), dtype=torch.long)
    mask = torch.zeros((len(samples), max_options + 1), dtype=torch.bool)
    weights = torch.zeros(len(samples), dtype=torch.float32)
    for row_index, sample in enumerate(samples):
        selected_set = set(sample.selected)
        for option_index in range(sample.legal_count):
            features[row_index, option_index] = torch.tensor(sample.features[option_index], dtype=torch.float32)
            option_meta[row_index, option_index] = torch.tensor(
                [
                    1.0 if option_index in selected_set else 0.0,
                    min(len(sample.selected), 8) / 8.0,
                    min(len(selected_set), 8) / 8.0,
                    max(0, sample.min_count - len(selected_set)) / max(1, sample.legal_count),
                    max(0, sample.max_count - len(selected_set)) / max(1, sample.legal_count),
                ],
                dtype=torch.float32,
            )
            if option_index not in selected_set and len(selected_set) < sample.max_count:
                mask[row_index, option_index] = True
        if len(selected_set) >= sample.min_count:
            mask[row_index, max_options] = True
        if len(selected_set) >= sample.max_count:
            mask[row_index, :max_options] = False
            mask[row_index, max_options] = True
        target[row_index] = sample.target if sample.target < sample.legal_count else max_options
        weights[row_index] = float(sample.weight)

    return features, option_meta, target, mask, weights


class OptionEncoderScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, depth: int = 3, embed_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, embed_dim))
        layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*layers)


class AutoregressiveOptionScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, depth: int = 3, embed_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.encoder = OptionEncoderScorer(input_dim, hidden_dim=hidden_dim, depth=depth, embed_dim=embed_dim, dropout=dropout).encoder
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


def metrics_for_scores(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    masked = scores.masked_fill(~mask, -1e9)
    pred = masked.argmax(dim=1)
    top1 = (pred == target).float().mean().item() if target.numel() else 0.0
    top3_hits = []
    for row_index in range(masked.shape[0]):
        k = min(3, int(mask[row_index].sum().item()))
        topk = set(int(index) for index in torch.topk(masked[row_index], k=max(1, k)).indices.tolist())
        top3_hits.append(int(int(target[row_index]) in topk))
    by_context: dict[str, Counter[str]] = {}
    for ok, meta in zip((pred == target).tolist(), metadata):
        context = str(meta.get("select_context") or "")
        by_context.setdefault(context, Counter())
        by_context[context]["n"] += 1
        by_context[context]["correct"] += int(ok)

    return {
        "top1": round(float(top1), 6),
        "top3": round(sum(top3_hits) / max(1, len(top3_hits)), 6),
        "by_select_context": {
            key: {"n": value["n"], "top1": round(value["correct"] / max(1, value["n"]), 6)}
            for key, value in sorted(by_context.items())
        },
    }


def evaluate_autoregressive(model: AutoregressiveOptionScorer, samples: list[EncodedDecision], *, batch_size: int, weighted: bool) -> dict[str, Any]:
    model.eval()
    steps = [step for sample in samples for step in make_step_samples(sample, weighted=weighted)]
    first_steps = [step for step in steps if step.step_index == 0]
    scores_all = []
    targets = []
    masks = []
    metadata = []
    with torch.no_grad():
        for start in range(0, len(first_steps), batch_size):
            chunk = first_steps[start:start + batch_size]
            features, option_meta, target, mask, _weights = batch_steps(chunk)
            scores_all.append(model(features, option_meta))
            targets.append(target)
            masks.append(mask)
            metadata.extend(step.metadata for step in chunk)
    max_options = max(score.shape[1] for score in scores_all)
    padded_scores = []
    padded_masks = []
    for score, mask in zip(scores_all, masks):
        pad = max_options - score.shape[1]
        padded_scores.append(torch.nn.functional.pad(score, (0, pad), value=-1e9) if pad else score)
        padded_masks.append(torch.nn.functional.pad(mask, (0, pad), value=False) if pad else mask)
    metrics = metrics_for_scores(torch.cat(padded_scores), torch.cat(targets), torch.cat(padded_masks), metadata)
    metrics["step_rows"] = len(steps)
    metrics["sequence_exact_sampled"] = metrics["top1"]

    return metrics


def train_autoregressive(model: AutoregressiveOptionScorer, samples: Iterable[EncodedDecision], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rows_seen = 0
    step_rows_seen = 0
    batches = 0
    loss_total = 0.0
    batch: list[StepSample] = []
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    for encoded in samples:
        rows_seen += 1
        batch.extend(make_step_samples(encoded, weighted=True))
        if len(batch) < int(args.batch_size):
            continue
        features, option_meta, target, mask, weights = batch_steps(batch)
        scores = model(features, option_meta).masked_fill(~mask, -1e9)
        loss_by_row = torch.nn.functional.cross_entropy(scores, target, reduction="none")
        loss = (loss_by_row * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_rows_seen += len(batch)
        batches += 1
        loss_total += float(loss.item())
        if progress_every and batches % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"r4 batches={batches} rows={rows_seen} steps={step_rows_seen} loss={loss.item():.6f} elapsed={elapsed:.1f}",
                file=sys.stderr,
                flush=True,
            )
        batch = []
        if rows_seen >= int(args.train_rows):
            break
    if batch:
        features, option_meta, target, mask, weights = batch_steps(batch)
        scores = model(features, option_meta).masked_fill(~mask, -1e9)
        loss_by_row = torch.nn.functional.cross_entropy(scores, target, reduction="none")
        loss = (loss_by_row * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_rows_seen += len(batch)
        batches += 1
        loss_total += float(loss.item())

    return {
        "rows_seen": rows_seen,
        "step_rows_seen": step_rows_seen,
        "batches": batches,
        "mean_loss": round(loss_total / max(1, batches), 6),
    }


def save_checkpoint(path: Path, model: AutoregressiveOptionScorer, args: argparse.Namespace, train: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_kind": "autoregressive_option_scorer",
            "model_state": model.state_dict(),
            "input_dim": int(args.input_dim),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "embed_dim": int(args.embed_dim),
            "dropout": float(args.dropout),
            "feature_mode": "base",
            "total_rows_seen": int(train["rows_seen"]),
            "train": train,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = collect_encoded(Path(args.refgames_dir), split="val", limit=int(args.val_rows), feature_mode="base")
    if not val:
        raise RuntimeError("no validation rows found")
    args.input_dim = len(val[0].features[0])
    torch.manual_seed(int(args.seed))
    model = AutoregressiveOptionScorer(args.input_dim, hidden_dim=args.hidden_dim, depth=args.depth, embed_dim=args.embed_dim, dropout=args.dropout)
    train_samples = iter_encoded(Path(args.refgames_dir), split="train", limit=int(args.train_rows), feature_mode="base")
    train = train_autoregressive(model, train_samples, args)
    metrics = evaluate_autoregressive(model, val, batch_size=int(args.batch_size), weighted=False)
    model_path = out_dir / "selected_model.pt"
    save_checkpoint(model_path, model, args, train)
    preservation = preserve_checkpoint(
        model_path,
        mission=str(args.backup_mission),
        name="r4a-source",
        reason="R4-A source checkpoint rebuilt from source recipe",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_rebuilt_source_not_original_r4a",
        "model_path": str(model_path),
        "checkpoint_preservation": preservation,
        "input_dim": int(args.input_dim),
        "train": train,
        "offline_common_split": {"rebuilt_R4_A": metrics},
        "note": "Source was rebuilt after cleanup; this is reproducible from live modules but is not claimed byte-identical to the deleted checkpoint.",
    }
    (out_dir / "r4a_rebuild_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "r4a_rebuild_summary.md").write_text(render_summary(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:6000])

    return payload


def render_summary(payload: dict[str, Any]) -> str:
    metrics = payload["offline_common_split"]["rebuilt_R4_A"]
    lines = [
        "# R4-A Rebuild Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model_path']}`",
        f"- rows_seen: `{payload['train']['rows_seen']}`",
        f"- input_dim: `{payload['input_dim']}`",
        f"- top1: `{metrics.get('top1')}`",
        f"- top3: `{metrics.get('top3')}`",
        f"- seq_exact_proxy: `{metrics.get('sequence_exact_sampled')}`",
        "",
        payload["note"],
        "",
    ]

    return "\n".join(lines)


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuilt R4-A autoregressive option-identity BC training entrypoint.")
    parser.add_argument("--refgames-dir", default=str(DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--val-rows", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=440004)
    parser.add_argument("--backup-mission", default="r12")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
