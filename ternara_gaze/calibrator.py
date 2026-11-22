"""Pure-integer Q12 9-point gaze calibrator (on-chip compatible).

Why this exists:
- Every user has different eye geometry, glasses, camera angle, and monitor size.
- During a 5-second 9-point calibration, we record the frozen 0-DSP ternary backbone's
  penultimate 32-dim int8 vector `h2`, raw `(gaze_x, gaze_y)`, and optional normalized
  face/eye-box anchor `(head_x, head_y)` in [-128, +128].
- We fit a regularized linear head and quantize it to fixed-point Q12 int32 weights.
- At inference time, prediction is 100% integer (add/multiply-free optional, or single
  DSP-free Q12 readout), producing screen coordinates `(sx_u16, sy_u16)` in [0, 65535].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

Q_SHIFT = 12
Q_SCALE = 1 << Q_SHIFT  # 4096
CAL_FEAT_DIM = 37  # 32 (h2) + 2 (raw gaze) + 2 (head anchor) + 1 (bias=64)

NINE_POINT_GRID_NORM: Tuple[Tuple[float, float], ...] = (
    (0.12, 0.14), (0.50, 0.14), (0.88, 0.14),
    (0.12, 0.50), (0.50, 0.50), (0.88, 0.50),
    (0.12, 0.86), (0.50, 0.86), (0.88, 0.86),
)


def build_cal_feature(
    h2: np.ndarray,
    gaze_x: int,
    gaze_y: int,
    head_x: int = 0,
    head_y: int = 0,
) -> np.ndarray:
    """Assemble the 37-element int32 calibration feature vector."""
    h = np.asarray(h2, dtype=np.int32).reshape(-1)
    if h.size != 32:
        raise ValueError(f"Expected 32-element h2 vector, got {h.size}")
    return np.concatenate([
        h,
        np.array([int(gaze_x), int(gaze_y), int(head_x), int(head_y), 64], dtype=np.int32),
    ])


@dataclass(frozen=True)
class IntegerGazeCalibrator:
    """Fixed-point Q12 screen coordinate mapper (0 floats at inference)."""

    wx_q12: np.ndarray  # (37,) int32
    wy_q12: np.ndarray  # (37,) int32

    def __post_init__(self) -> None:
        if self.wx_q12.shape != (CAL_FEAT_DIM,) or self.wy_q12.shape != (CAL_FEAT_DIM,):
            raise ValueError(f"Calibrator weights must have shape ({CAL_FEAT_DIM},)")

    @classmethod
    def identity_default(cls) -> "IntegerGazeCalibrator":
        """Uncalibrated fallback: maps raw gaze_x/gaze_y in [-128,+128] to [0..65535]."""
        wx = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        wy = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        # sx = (gaze_x * 256 + 32768) -> Q12 coeff on gaze_x = 256 * 4096 / 1 = 1048576
        # Wait: let's map output in Q12 where 0..65535 is represented directly after >> 12:
        # (gaze_x * (256 << 12) + 64 * (512 << 12)) >> 12 = 256*gaze_x + 32768
        wx[32] = 256 << Q_SHIFT
        wx[34] = 128 << Q_SHIFT  # gentle head_x assist
        wx[36] = 512 << Q_SHIFT  # 64 * 512 = 32768 (center)
        wy[33] = 256 << Q_SHIFT
        wy[35] = 128 << Q_SHIFT  # gentle head_y assist
        wy[36] = 512 << Q_SHIFT
        return cls(wx_q12=wx, wy_q12=wy)

    @classmethod
    def fit(
        cls,
        features: Sequence[np.ndarray] | np.ndarray,
        targets_norm: Sequence[Tuple[float, float]] | np.ndarray,
        ridge_lambda: float = 2500.0,
    ) -> "IntegerGazeCalibrator":
        """Fit regularized ridge regression on calibration samples and quantize to Q12 int32."""
        X = np.asarray(features, dtype=np.float64)
        Y = np.asarray(targets_norm, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != CAL_FEAT_DIM:
            raise ValueError(f"Expected X shape (N, {CAL_FEAT_DIM}), got {X.shape}")
        if Y.ndim != 2 or Y.shape != (X.shape[0], 2):
            raise ValueError(f"Expected Y shape ({X.shape[0]}, 2), got {Y.shape}")
        if X.shape[0] < 4:
            raise ValueError("Need at least 4 calibration samples")

        # Scale targets to [0, 65535]
        Y_u16 = np.clip(Y, 0.0, 1.0) * 65535.0
        reg = np.eye(CAL_FEAT_DIM, dtype=np.float64) * ridge_lambda
        reg[-1, -1] = 1e-3  # do not shrink bias term
        A = X.T @ X + reg
        W = np.linalg.solve(A, X.T @ Y_u16)  # (37, 2)
        wx_q12 = np.rint(W[:, 0] * Q_SCALE).astype(np.int64)
        wy_q12 = np.rint(W[:, 1] * Q_SCALE).astype(np.int64)
        wx_q12 = np.clip(wx_q12, -(1 << 30), (1 << 30) - 1).astype(np.int32)
        wy_q12 = np.clip(wy_q12, -(1 << 30), (1 << 30) - 1).astype(np.int32)
        return cls(wx_q12=wx_q12, wy_q12=wy_q12)

    def predict_u16(
        self,
        h2: np.ndarray,
        gaze_x: int,
        gaze_y: int,
        head_x: int = 0,
        head_y: int = 0,
    ) -> Tuple[int, int]:
        """Pure-integer screen coordinate prediction in [0, 65535]."""
        f = build_cal_feature(h2, gaze_x, gaze_y, head_x, head_y).astype(np.int64)
        sx = int(np.dot(f, self.wx_q12.astype(np.int64))) >> Q_SHIFT
        sy = int(np.dot(f, self.wy_q12.astype(np.int64))) >> Q_SHIFT
        sx = max(0, min(65535, sx))
        sy = max(0, min(65535, sy))
        return sx, sy

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ternara_gaze_cal_q12_v1",
            "q_shift": Q_SHIFT,
            "wx_q12": [int(v) for v in self.wx_q12],
            "wy_q12": [int(v) for v in self.wy_q12],
        }
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntegerGazeCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "ternara_gaze_cal_q12_v1":
            raise ValueError(f"Unsupported calibration format: {data.get('format')}")
        return cls(
            wx_q12=np.asarray(data["wx_q12"], dtype=np.int32),
            wy_q12=np.asarray(data["wy_q12"], dtype=np.int32),
        )
