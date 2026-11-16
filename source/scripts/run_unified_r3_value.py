"""Rebuilt value-source module for GreedyV (R15).

The original ``run_unified_r3_value.py`` (trainer + ``ValueMLP`` + ``_features``) and
its ``selected_value_source_v0.pt`` checkpoint were lost; this rebuilds both against
the surviving consumer contract in ``deepmew/greedyv/value`` (recovered from bytecode):

    payload = torch.load(ckpt)                      # feature_config, input_dim, arch, model_state
    model = value_source.ValueMLP(input_dim, payload["arch"])
    features = value_source._features({"raw_observation": obs}, feature_config)
    logit = model(torch.tensor([features]))[0].item()   # P(win) for obs's yourIndex player

Features are PUBLIC/OWN-state only, oriented to the observation's ``yourIndex`` player
(the second-order reader plus generic board counts). Research-only: torch.

Train
-----
    uv run python scripts/run_unified_r3_value.py \
        --refgames-dir D:/DeepMew_artifacts/datasets/r15/hard_refgames_crossplay \
        --out .research/unified_r3/value/selected_value_source_v0.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from checkpoint_preservation import preserve_checkpoint  # noqa: E402
from deepmew.second_order.reader import SECOND_ORDER_STATE_SCHEMA, read_second_order_state  # noqa: E402


SCHEMA_VERSION = "deepmew.r15.value_source.rebuilt.v0"
FEATURE_CONFIG_V1 = "r15_state_v1"
# v2 adds board-development / attacker-readiness features (bench energy is invisible
# to v1, and v1 rewards keeping cards in hand -> value net prefers END over building
# a board; see BUG-010 / R18 END-bias). v2 = v1 features + 4 dev features per side.
FEATURE_CONFIG_V2 = "r15_state_v2"
DEFAULT_ARCH = {"hidden_dims": [256, 128], "dropout": 0.0}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _player(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = _as_list(current.get("players"))

    return _as_dict(players[index]) if 0 <= index < len(players) else {}


def _active_entry(player: dict[str, Any]) -> dict[str, Any]:
    active = _as_list(player.get("active"))

    return _as_dict(active[0]) if active else {}


def _hp_fraction(pokemon: dict[str, Any]) -> float:
    max_hp = _as_int(pokemon.get("maxHp"))
    if max_hp <= 0:
        return 0.0

    return max(0.0, min(1.0, _as_int(pokemon.get("hp")) / max_hp))


def _bench_hp_sum_norm(player: dict[str, Any]) -> float:
    total = sum(_hp_fraction(_as_dict(entry)) for entry in _as_list(player.get("bench")) if entry)

    return total / 5.0


def _side_features(player: dict[str, Any], *, own_side: bool) -> list[float]:
    active = _active_entry(player)
    prize = player.get("prize")
    prizes_remaining = len(prize) if isinstance(prize, list) else 6
    hand = player.get("hand")
    hand_count = len(hand) if isinstance(hand, list) else _as_int(player.get("handCount"))
    deck_count = _as_int(player.get("deckCount"))
    discard = _as_list(player.get("discard"))
    bench = [entry for entry in _as_list(player.get("bench")) if entry]
    energies = len(_as_list(active.get("energies"))) + len(_as_list(active.get("energyCards")))
    status = sum(
        1.0
        for flag in ("asleep", "burned", "confused", "paralyzed", "poisoned")
        if player.get(flag)
    )
    _ = own_side

    return [
        prizes_remaining / 6.0,
        hand_count / 12.0,
        deck_count / 60.0,
        len(discard) / 60.0,
        len(bench) / 5.0,
        1.0 if active else 0.0,
        _hp_fraction(active),
        min(1.0, energies / 4.0),
        len(_as_list(active.get("tools"))) and 1.0 or 0.0,
        _bench_hp_sum_norm(player),
        status / 5.0,
    ]


def _energy_units(pokemon: dict[str, Any]) -> int:
    """Count of energy UNITS attached to a Pokemon (each element of `energies`
    is one attached energy providing that type)."""
    return len(_as_list(pokemon.get("energies")))


def _dev_features(player: dict[str, Any]) -> list[float]:
    """Board-development / attacker-readiness signals (v2).

    v1 makes bench energy invisible and rewards a full hand, so the value net
    prefers END over committing energy to the board (BUG-010 / R18 END-bias).
    These give the net explicit visibility into where energy actually sits and
    whether any single Pokemon is fueled enough to threaten an attack.
    """
    active = _active_entry(player)
    bench = [_as_dict(entry) for entry in _as_list(player.get("bench")) if entry]
    active_e = _energy_units(active)
    bench_e = sum(_energy_units(entry) for entry in bench)
    in_play = ([active] if active else []) + bench
    max_e = max((_energy_units(p) for p in in_play), default=0)
    total_e = active_e + bench_e
    return [
        min(1.0, bench_e / 6.0),          # bench energy (invisible in v1)
        min(1.0, total_e / 8.0),          # total energy committed to the board
        min(1.0, max_e / 4.0),            # attacker-readiness: best-fueled Pokemon
        len(in_play) / 6.0,               # board width (active + bench)
    ]


def _features(row: dict[str, Any], feature_config: str) -> list[float]:
    cfg = str(feature_config)
    if cfg not in (FEATURE_CONFIG_V1, FEATURE_CONFIG_V2):
        raise ValueError(f"unknown feature_config {feature_config!r}")
    obs = _as_dict(row.get("raw_observation"))
    current = _as_dict(obs.get("current"))
    seat = _as_int(current.get("yourIndex"))
    opp_seat = 1 - seat if seat in (0, 1) else 1
    own = _player(current, seat)
    opp = _player(current, opp_seat)
    second_order = read_second_order_state(obs).vector()
    vector = list(_side_features(own, own_side=True))
    vector.extend(_side_features(opp, own_side=False))
    vector.extend(
        [
            min(1.0, _as_int(current.get("turn")) / 30.0),
            1.0 if _as_int(current.get("firstPlayer"), -1) == seat else 0.0,
            1.0 if current.get("energyAttached") else 0.0,
            1.0 if current.get("supporterPlayed") else 0.0,
            1.0 if _as_list(current.get("stadium")) else 0.0,
        ]
    )
    vector.extend(float(value) for value in second_order)
    if cfg == FEATURE_CONFIG_V2:
        # appended AFTER v1 so v1 remains a strict prefix (train/inference agree
        # because both call this same function with the checkpoint's config).
        vector.extend(_dev_features(own))
        vector.extend(_dev_features(opp))

    return vector


FEATURE_DIM_V1 = 11 * 2 + 5 + len(SECOND_ORDER_STATE_SCHEMA)
FEATURE_DIM_V2 = FEATURE_DIM_V1 + 4 * 2


class ValueMLP(torch.nn.Module):
    def __init__(self, input_dim: int, arch: dict[str, Any]):
        super().__init__()
        hidden_dims = [int(width) for width in _as_list(_as_dict(arch).get("hidden_dims"))] or [256, 128]
        dropout = float(_as_dict(arch).get("dropout") or 0.0)
        layers: list[torch.nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            layers.append(torch.nn.Linear(previous, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _iter_shard_rows(refgames_dir: Path) -> Any:
    for shard_path in sorted((refgames_dir / "shards").glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _load_dataset(refgames_dir: Path, *, limit: int | None, feature_config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    is_val: list[bool] = []
    for row in _iter_shard_rows(refgames_dir):
        outcome = _as_dict(row.get("outcome"))
        if outcome.get("draw"):
            continue
        features.append(_features(row, feature_config))
        labels.append(1.0 if outcome.get("won") else 0.0)
        is_val.append(_as_int(row.get("game_index")) % 5 == 0)
        if limit is not None and len(features) >= limit:
            break

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(is_val, dtype=bool),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    feature_config = str(args.feature_config)
    features, labels, is_val = _load_dataset(Path(args.refgames_dir), limit=args.limit if args.limit > 0 else None, feature_config=feature_config)
    train_x = torch.from_numpy(features[~is_val])
    train_y = torch.from_numpy(labels[~is_val])
    val_x = torch.from_numpy(features[is_val])
    val_y = torch.from_numpy(labels[is_val])
    model = ValueMLP(features.shape[1], DEFAULT_ARCH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    batch_size = int(args.batch_size)
    for epoch in range(int(args.epochs)):
        model.train()
        order = torch.randperm(train_x.shape[0])
        epoch_loss = 0.0
        batches = 0
        for start in range(0, train_x.shape[0], batch_size):
            index = order[start:start + batch_size]
            logits = model(train_x[index])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(val_logits, val_y).item())
            val_pred = (val_logits > 0).float()
            val_acc = float((val_pred == val_y).float().mean().item())
            order_idx = torch.argsort(val_logits)
            ranks = torch.empty_like(order_idx, dtype=torch.float32)
            ranks[order_idx] = torch.arange(val_logits.shape[0], dtype=torch.float32)
            positives = val_y > 0.5
            n_pos = int(positives.sum().item())
            n_neg = val_y.shape[0] - n_pos
            auc = float((ranks[positives].sum().item() - n_pos * (n_pos - 1) / 2) / max(1, n_pos * n_neg))
        print(f"epoch={epoch} train_loss={epoch_loss / max(1, batches):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={auc:.4f}", flush=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "feature_config": feature_config,
        "input_dim": int(features.shape[1]),
        "arch": dict(DEFAULT_ARCH),
        "model_state": model.state_dict(),
        "train": {
            "refgames_dir": str(args.refgames_dir),
            "rows_train": int(train_x.shape[0]),
            "rows_val": int(val_x.shape[0]),
            "epochs": int(args.epochs),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": auc,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    preservation = preserve_checkpoint(
        out_path,
        mission=str(args.backup_mission),
        name="value-source-v0-rebuilt",
        reason="Rebuilt GreedyV value model trained on hard cross-play refgames",
    )
    print(json.dumps({"out": str(out_path), "input_dim": payload["input_dim"], "train": payload["train"], "preservation": preservation}, indent=2, default=str))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the rebuilt GreedyV value source (P(win) for yourIndex player).")
    parser.add_argument("--refgames-dir", required=True)
    parser.add_argument("--feature-config", default=FEATURE_CONFIG_V2, choices=[FEATURE_CONFIG_V1, FEATURE_CONFIG_V2],
                        help="v2 adds board-development/attacker-readiness features (BUG-010 END-bias fix)")
    parser.add_argument("--out", default=str(ROOT / ".research" / "unified_r3" / "value" / "selected_value_source_v0.pt"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=915001)
    parser.add_argument("--backup-mission", default="r15")
    args = parser.parse_args(argv)
    train(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
