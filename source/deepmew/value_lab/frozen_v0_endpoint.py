"""Hash-bound inference endpoint for the frozen actor-visible V0 model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from deepmew.value_lab.calibration import apply_calibrator
from deepmew.value_lab.calibration_selection import load_calibrator_bundle
from deepmew.value_lab.crossfit import FOLD_UNIT_SCHEMA_VERSION, SNAPSHOT_SPEC
from deepmew.value_lab.mc_screen import build_snapshot_model
from deepmew.value_lab.models import parameter_count


ACTOR_PERSPECTIVE = "actor"
FROZEN_INPUT_START = 0
FROZEN_INPUT_STOP = 49
FROZEN_ARCHITECTURE = (49, 256, 128, 1)
FROZEN_PARAMETER_COUNT = 45_825
FROZEN_CHECKPOINT_ROLE = "final_full_outer_train"
FROZEN_CALIBRATOR_METHOD = "isotonic"
TERMINAL_OUTCOMES = frozenset({0.0, 0.5, 1.0})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256")


@dataclass(frozen=True)
class FrozenV0Spec:
    """Immutable identities and shape contract for one frozen V0 endpoint."""

    checkpoint_sha256: str
    calibrator_bundle_sha256: str
    perspective: str = ACTOR_PERSPECTIVE
    input_start: int = FROZEN_INPUT_START
    input_stop: int = FROZEN_INPUT_STOP
    architecture: tuple[int, ...] = FROZEN_ARCHITECTURE

    def __post_init__(self) -> None:
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _validate_sha256(self.calibrator_bundle_sha256, "calibrator_bundle_sha256")
        if self.perspective != ACTOR_PERSPECTIVE:
            raise ValueError("frozen V0 is defined only from the acting player's perspective")
        if (self.input_start, self.input_stop) != (FROZEN_INPUT_START, FROZEN_INPUT_STOP):
            raise ValueError("frozen V0 requires the raw S1 feature slice [0:49]")
        if self.architecture != FROZEN_ARCHITECTURE:
            raise ValueError(f"frozen V0 architecture must be {FROZEN_ARCHITECTURE}")


class FrozenV0Endpoint:
    """CPU inference endpoint that cannot silently drift or flip perspective."""

    def __init__(
        self,
        model: torch.nn.Module,
        calibrator: Mapping[str, Any],
        spec: FrozenV0Spec,
    ) -> None:
        self._model = model.eval()
        self._calibrator = dict(calibrator)
        self.spec = spec

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        calibrator_root: Path,
        calibrator_receipt: Mapping[str, Any],
        spec: FrozenV0Spec,
    ) -> "FrozenV0Endpoint":
        """Load only the exact frozen architecture, weights, and isotonic bundle."""
        checkpoint_path = Path(checkpoint_path).resolve()
        calibrator_root = Path(calibrator_root).resolve()
        if _sha256_file(checkpoint_path) != spec.checkpoint_sha256.lower():
            raise ValueError("frozen V0 checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_model_config = {
            "input_dim": FROZEN_ARCHITECTURE[0],
            "hidden_dims": list(FROZEN_ARCHITECTURE[1:-1]),
            "dropout": 0.0,
        }
        if (
            checkpoint.get("schema_version") != FOLD_UNIT_SCHEMA_VERSION
            or checkpoint.get("role") != FROZEN_CHECKPOINT_ROLE
            or checkpoint.get("spec") != asdict(SNAPSHOT_SPEC)
            or checkpoint.get("model_config") != expected_model_config
            or checkpoint.get("parameter_count") != FROZEN_PARAMETER_COUNT
        ):
            raise ValueError("frozen V0 checkpoint architecture or role mismatch")
        model = build_snapshot_model(SNAPSHOT_SPEC)
        model.load_state_dict(checkpoint.get("model_state", {}), strict=True)
        if parameter_count(model) != FROZEN_PARAMETER_COUNT:
            raise ValueError("frozen V0 parameter count drifted")
        if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
            raise ValueError("frozen V0 checkpoint contains non-finite parameters")

        receipt = dict(calibrator_receipt)
        if receipt.get("method") != FROZEN_CALIBRATOR_METHOD:
            raise ValueError("frozen V0 requires the selected isotonic calibrator")
        if receipt.get("bundle_sha256") != spec.calibrator_bundle_sha256.lower():
            raise ValueError("frozen V0 calibrator receipt hash mismatch")
        calibrator = load_calibrator_bundle(calibrator_root, receipt)
        if calibrator.get("method") != FROZEN_CALIBRATOR_METHOD:
            raise ValueError("frozen V0 calibrator method drifted")
        calibration_probe = apply_calibrator(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            calibrator,
        )
        if not np.isfinite(calibration_probe).all():
            raise ValueError("frozen V0 calibrator produced non-finite values")

        return cls(model, calibrator, spec)

    def score(
        self,
        s1: np.ndarray,
        *,
        perspective: str = ACTOR_PERSPECTIVE,
        terminal_outcome: float | None = None,
    ) -> float | np.ndarray:
        """Score float32 S1 rows, applying sigmoid then the frozen calibrator.

        Terminal outcomes are returned exactly and are never calibrated. There
        is intentionally no opponent-perspective/complement path.
        """
        if perspective != ACTOR_PERSPECTIVE or perspective != self.spec.perspective:
            raise ValueError("frozen V0 accepts only acting-player perspective; do not complement V")
        if not isinstance(s1, np.ndarray):
            raise TypeError("s1 must be a numpy.ndarray")
        if s1.dtype != np.float32:
            raise TypeError("s1 must have exact dtype float32")
        if s1.ndim not in {1, 2} or s1.shape[-1] != self.spec.input_stop - self.spec.input_start:
            raise ValueError("s1 must have shape [49] or [N, 49]")
        if s1.ndim == 2 and s1.shape[0] == 0:
            raise ValueError("s1 batch cannot be empty")
        if not np.isfinite(s1).all():
            raise ValueError("s1 contains non-finite values")

        is_single = s1.ndim == 1
        rows = s1[None, :] if is_single else s1
        if terminal_outcome is not None:
            if isinstance(terminal_outcome, bool) or not isinstance(terminal_outcome, (int, float)):
                raise TypeError("terminal_outcome must be a numeric 0.0, 0.5, or 1.0")
            terminal_value = float(terminal_outcome)
            if not math.isfinite(terminal_value) or terminal_value not in TERMINAL_OUTCOMES:
                raise ValueError("terminal_outcome must be exactly 0.0, 0.5, or 1.0")
            if is_single:
                return terminal_value

            return np.full(rows.shape[0], terminal_value, dtype=np.float64)

        tensor = torch.from_numpy(np.ascontiguousarray(rows)).unsqueeze(1)
        mask = torch.ones((rows.shape[0], 1), dtype=torch.bool)
        with torch.inference_mode():
            logits = self._model(tensor, mask)[:, 0]
            raw_probability = torch.sigmoid(logits).cpu().numpy().astype(np.float64, copy=False)
        calibrated = apply_calibrator(raw_probability, self._calibrator)
        if not np.isfinite(calibrated).all() or np.any((calibrated < 0.0) | (calibrated > 1.0)):
            raise FloatingPointError("frozen V0 produced a non-finite or out-of-range probability")
        if is_single:
            return float(calibrated[0])

        return np.asarray(calibrated, dtype=np.float64)
