from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from cg.game import battle_finish, battle_select, battle_start

import main
import run_unified_r4_ablation as r4
from deepmew.policy.selection import guard_selection
from deepmew.rl.metadata import normalized_deck_hash
from deepmew.rl1_teacher_bc.semantics import legal_action_table

PANEL_PATH = ROOT / ".research" / "r12" / "rebuilt_eval_suite.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class EpisodeResult:
    result: int | None
    our_seat: int
    illegal: bool = False
    crashed: bool = False
    timed_out: bool = False
    turns: int = 0
    error: str | None = None

    def is_win(self, seat: int) -> bool:
        return self.result == seat

    def is_loss(self, seat: int) -> bool:
        return self.result in (0, 1) and self.result != seat

    def is_draw(self) -> bool:
        return self.result == 2


class R4ObsPolicy:
    def __init__(self, label: str, checkpoint: Path, deck_policy: str):
        self.label = label
        self.checkpoint = Path(checkpoint)
        self.payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model = r4.AutoregressiveOptionScorer(
            int(self.payload["input_dim"]),
            hidden_dim=int(self.payload["hidden_dim"]),
            depth=int(self.payload["depth"]),
            embed_dim=int(self.payload["embed_dim"]),
            dropout=float(self.payload.get("dropout", 0.0)),
        )
        self.model.load_state_dict(self.payload["model_state"])
        self.model.eval()
        self.policy_id = deck_policy
        self.deck = main.read_deck_csv()
        self.deck_hash = normalized_deck_hash(self.deck)

    def reset(self, seed: int) -> None:
        _ = seed

    def bind_timer(self, timer: Any | None) -> None:
        _ = timer

    def _select_from_features(self, features: list[list[float]], legal_count: int, min_count: int, max_count: int) -> list[int]:
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
            packed = r4.batch_steps([step])
            with torch.no_grad():
                scores = self.model(packed[0], packed[1]).masked_fill(~packed[3], -1e9)[0].numpy()
            choice = int(np.argmax(scores))
            if choice >= legal_count:
                break
            if choice not in selected:
                selected.append(choice)
            if len(selected) >= max_count:
                break

        return selected

    def select(self, obs: dict[str, Any]) -> list[int]:
        select = _as_dict(obs.get("select"))
        if not select:
            return list(self.deck)
        table = legal_action_table(obs, select)
        encoded = r4.build_live_features(obs, table, policy_id=self.policy_id, deck_hash=self.deck_hash)
        if encoded is None:
            return guard_selection([0], select)
        min_count = max(0, min(_as_int(select.get("minCount")), encoded.legal_count))
        max_count = max(min_count, min(_as_int(select.get("maxCount"), encoded.legal_count), encoded.legal_count))
        chosen = self._select_from_features(encoded.features, encoded.legal_count, min_count, max_count)

        return guard_selection(chosen, select)


def _select_from_policy(policy: Any, obs: dict[str, Any]) -> list[int]:
    select = obs.get("select") if isinstance(obs, dict) else None
    if not isinstance(select, dict):
        return list(getattr(policy, "deck", None) or main.read_deck_csv())

    return policy.select(obs)


def _play_obs_episode(
    policy: Any,
    *,
    opponent: Any,
    our_seat: int,
    our_deck: list[int],
    max_steps: int,
    opponent_seed: int,
    policy_seed: int,
    phase_timer: Any | None = None,
) -> EpisodeResult:
    _ = phase_timer
    obs = None
    try:
        opponent.reset(opponent_seed)
        policy.reset(policy_seed)
        deck0 = our_deck if our_seat == 0 else list(getattr(opponent, "deck", None) or main.read_deck_csv())
        deck1 = list(getattr(opponent, "deck", None) or main.read_deck_csv()) if our_seat == 0 else our_deck
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(f"BattleStart failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}")
        for turn_index in range(max_steps):
            current = _as_dict(obs.get("current"))
            if current.get("result", -1) != -1:
                return EpisodeResult(result=int(current.get("result")), our_seat=our_seat, turns=_as_int(current.get("turn"), turn_index))
            player_index = _as_int(current.get("yourIndex"))
            actor = policy if player_index == our_seat else opponent
            selected = _select_from_policy(actor, obs)
            select = _as_dict(obs.get("select"))
            legal_count = len(select.get("option") or [])
            illegal = any(index < 0 or index >= legal_count for index in selected)
            obs = battle_select(selected)
            if illegal:
                return EpisodeResult(result=None, our_seat=our_seat, illegal=True, turns=turn_index)

        return EpisodeResult(result=None, our_seat=our_seat, timed_out=True, turns=max_steps)
    except Exception as exc:
        return EpisodeResult(result=None, our_seat=our_seat, crashed=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        if obs is not None:
            battle_finish()


def _slice_keys(per: dict[str, Any], name: str) -> list[str]:
    keys = sorted(per)
    if name == "overall":
        return keys
    if name == "trivial":
        return [key for key in keys if key in {"random", "first-legal"}]
    if name == "v2_4_legacy":
        return [key for key in keys if key.startswith("v2.4") or key == "legacy-v2"]
    if name == "references":
        return [key for key in keys if key.startswith("reference_")]
    if name == "iono":
        return [key for key in keys if "iono" in key]

    return keys


def _slice_winrate(per: dict[str, Any], keys: list[str]) -> tuple[int, int, float | None]:
    wins = 0
    completed = 0
    for key in keys:
        row = per.get(key) or {}
        wins += int(row.get("wins", 0))
        completed += int(row.get("wins", 0)) + int(row.get("losses", 0))
    if completed <= 0:
        return wins, completed, None

    return wins, completed, wins / completed


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R4 closeout compatibility helpers; import-only by default.")
    parser.parse_args(argv)
    print(json.dumps({"panel_path": str(PANEL_PATH), "status": "helpers_importable"}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
