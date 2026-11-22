"""Pure-integer Q12 corner-first polynomial gaze calibrator (on-chip compatible, zero overfitting).

Uses 14 physically orthogonal integer features instead of high-dimensional raw hidden activations:
  [0..3]   pupil relative offsets (pup_lx, pup_ly, pup_rx, pup_ry) in [-128..+127]
  [4..5]   Q27 0-DSP ternary backbone raw (gaze_x, gaze_y)
  [6..7]   head position anchor (head_x, head_y) in [-128..+127]
  [8..9]   head orientation (yaw, pitch) in [-128..+127]
  [10..12] mild quadratic interaction terms: (px_avg^2>>7, py_avg^2>>7, (px_avg*py_avg)>>7)
  [13]     constant bias = 64

Fitted over 13 corner-first calibration targets (4 true extreme corners FIRST, then center,
4 edge midpoints, and 4 inner quadrants) with per-target robust median/MAD outlier rejection.
At inference time: 100% pure integer dot product + `>> 12` shift (0 floats).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np

Q_SHIFT = 12
Q_SCALE = 1 << Q_SHIFT  # 4096
CAL_FEAT_DIM = 14

# 13-point corner-first calibration grid:
# 4 TRUE extreme corners first, then center, 4 edge midpoints, 4 inner quadrants
CORNER_FIRST_GRID: Tuple[Tuple[float, float, str], ...] = (
    (0.04, 0.05, "1/13  TOP-LEFT CORNER"),
    (0.96, 0.05, "2/13  TOP-RIGHT CORNER"),
    (0.96, 0.94, "3/13  BOTTOM-RIGHT CORNER"),
    (0.04, 0.94, "4/13  BOTTOM-LEFT CORNER"),
    (0.50, 0.50, "5/13  SCREEN CENTER"),
    (0.50, 0.05, "6/13  TOP EDGE"),
    (0.96, 0.50, "7/13  RIGHT EDGE"),
    (0.50, 0.94, "8/13  BOTTOM EDGE"),
    (0.04, 0.50, "9/13  LEFT EDGE"),
    (0.26, 0.28, "10/13 UPPER-LEFT"),
    (0.74, 0.28, "11/13 UPPER-RIGHT"),
    (0.74, 0.72, "12/13 LOWER-RIGHT"),
    (0.26, 0.72, "13/13 LOWER-LEFT"),
)

# Backwards-compatible alias
NINE_POINT_GRID_NORM: Tuple[Tuple[float, float], ...] = tuple((x, y) for x, y, _ in CORNER_FIRST_GRID[:9])


def dark_pupil_centroid_int(crop_u4: np.ndarray) -> Tuple[int, int]:
    """Pure-integer dark-pupil centroid inside a (16, 32) 4-bit eye crop -> (px, py) in [-128, +127]."""
    c = np.asarray(crop_u4, dtype=np.int32)
    # Focus on central eye aperture (avoid outer brow/eyelash frame)
    roi = c[3:13, 5:27]  # 10 x 22
    max_v = int(roi.max())
    min_v = int(roi.min())
    thresh = min_v + max(1, (max_v - min_v) >> 2)
    # Weight darker pixels linearly: w = max(0, thresh - pixel + 1)
    weights = np.maximum(0, thresh - roi + 1)
    total_w = int(weights.sum())
    if total_w <= 0:
        return 0, 0
    yy, xx = np.mgrid[0:10, 0:22]
    cx_q8 = (int((weights * xx).sum()) << 8) // total_w  # [0..21*256]
    cy_q8 = (int((weights * yy).sum()) << 8) // total_w  # [0..9*256]
    # Center around midpoint (10.5, 4.5) and scale to [-128, +127]
    px = max(-128, min(127, ((cx_q8 - 2688) * 12) >> 8))
    py = max(-128, min(127, ((cy_q8 - 1152) * 26) >> 8))
    return int(px), int(py)


def build_cal_feature(
    h2: Optional[np.ndarray],
    gaze_x: int,
    gaze_y: int,
    head_x: int = 0,
    head_y: int = 0,
    pup_lx: int = 0,
    pup_ly: int = 0,
    pup_rx: int = 0,
    pup_ry: int = 0,
    yaw: int = 0,
    pitch: int = 0,
) -> np.ndarray:
    """Assemble the 14-element physically orthogonal int32 calibration feature vector."""
    del h2  # intentionally excluded to avoid 37-dim overfitting on small calibration grids
    px_avg = (int(pup_lx) + int(pup_rx) + int(yaw)) // 2
    py_avg = (int(pup_ly) + int(pup_ry) + int(pitch)) // 2
    q_xx = max(-128, min(127, (px_avg * px_avg) >> 7))
    q_yy = max(-128, min(127, (py_avg * py_avg) >> 7))
    q_xy = max(-128, min(127, (px_avg * py_avg) >> 7))
    return np.array(
        [
            int(pup_lx),
            int(pup_ly),
            int(pup_rx),
            int(pup_ry),
            int(gaze_x),
            int(gaze_y),
            int(head_x),
            int(head_y),
            int(yaw),
            int(pitch),
            q_xx,
            q_yy,
            q_xy,
            64,
        ],
        dtype=np.int32,
    )


def filter_target_outliers(samples: Sequence[np.ndarray], keep_ratio: float = 0.65) -> List[np.ndarray]:
    """Drop saccade/blink outlier frames within a single calibration target cluster."""
    if len(samples) <= 4:
        return list(samples)
    arr = np.asarray(samples, dtype=np.float64)
    med = np.median(arr, axis=0)
    dists = np.linalg.norm(arr - med, axis=1)
    k = max(4, int(round(len(samples) * keep_ratio)))
    order = np.argsort(dists)[:k]
    return [samples[int(i)] for i in order]


@dataclass(frozen=True)
class IntegerGazeCalibrator:
    """Fixed-point Q12 screen coordinate mapper (0 floats at inference)."""

    wx_q12: np.ndarray  # (14,) int32
    wy_q12: np.ndarray  # (14,) int32
    blink_thresh_l: int = 24
    blink_thresh_r: int = 24
    fit_mae_pct_x: float = 0.0
    fit_mae_pct_y: float = 0.0

    def __post_init__(self) -> None:
        if self.wx_q12.shape != (CAL_FEAT_DIM,) or self.wy_q12.shape != (CAL_FEAT_DIM,):
            raise ValueError(f"Calibrator weights must have shape ({CAL_FEAT_DIM},)")

    @classmethod
    def identity_default(cls) -> "IntegerGazeCalibrator":
        wx = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        wy = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        wx[0] = 180 << Q_SHIFT   # pup_lx
        wx[2] = 180 << Q_SHIFT   # pup_rx
        wx[4] = 96 << Q_SHIFT    # q27 gaze_x
        wx[8] = 220 << Q_SHIFT   # yaw
        wx[13] = 512 << Q_SHIFT  # center
        wy[1] = 180 << Q_SHIFT   # pup_ly
        wy[3] = 180 << Q_SHIFT   # pup_ry
        wy[5] = 96 << Q_SHIFT    # q27 gaze_y
        wy[9] = 220 << Q_SHIFT   # pitch
        wy[13] = 512 << Q_SHIFT
        return cls(wx_q12=wx, wy_q12=wy)

    @classmethod
    def fit(
        cls,
        features: Sequence[np.ndarray] | np.ndarray,
        targets_norm: Sequence[Tuple[float, float]] | np.ndarray,
        ridge_lambda: float = 1200.0,
        blink_thresh_l: int = 24,
        blink_thresh_r: int = 24,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> "IntegerGazeCalibrator":
        """Fit weighted ridge regression (giving extra weight to extreme corners) and quantize to Q12."""
        X = np.asarray(features, dtype=np.float64)
        Y = np.asarray(targets_norm, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != CAL_FEAT_DIM:
            raise ValueError(f"Expected X shape (N, {CAL_FEAT_DIM}), got {X.shape}")
        if Y.ndim != 2 or Y.shape != (X.shape[0], 2):
            raise ValueError(f"Expected Y shape ({X.shape[0]}, 2), got {Y.shape}")
        if X.shape[0] < 4:
            raise ValueError("Need at least 4 calibration samples")

        if sample_weights is None:
            w = np.ones(X.shape[0], dtype=np.float64)
        else:
            w = np.asarray(sample_weights, dtype=np.float64)

        Y_u16 = np.clip(Y, 0.0, 1.0) * 65535.0
        Xw = X * w[:, None]
        reg = np.eye(CAL_FEAT_DIM, dtype=np.float64) * ridge_lambda
        # Stronger regularization on quadratic terms so edges stay smooth
        reg[10, 10] = ridge_lambda * 6.0
        reg[11, 11] = ridge_lambda * 6.0
        reg[12, 12] = ridge_lambda * 6.0
        reg[-1, -1] = 1e-3

        A = Xw.T @ X + reg
        W = np.linalg.solve(A, Xw.T @ Y_u16)
        wx_q12 = np.clip(np.rint(W[:, 0] * Q_SCALE), -(1 << 30), (1 << 30) - 1).astype(np.int32)
        wy_q12 = np.clip(np.rint(W[:, 1] * Q_SCALE), -(1 << 30), (1 << 30) - 1).astype(np.int32)

        # Measure integer fit error in % of screen
        pred_x = np.clip((X.astype(np.int64) @ wx_q12.astype(np.int64)) >> Q_SHIFT, 0, 65535)
        pred_y = np.clip((X.astype(np.int64) @ wy_q12.astype(np.int64)) >> Q_SHIFT, 0, 65535)
        mae_x = float(np.mean(np.abs(pred_x - Y_u16[:, 0])) / 655.35)
        mae_y = float(np.mean(np.abs(pred_y - Y_u16[:, 1])) / 655.35)

        return cls(
            wx_q12=wx_q12,
            wy_q12=wy_q12,
            blink_thresh_l=int(blink_thresh_l),
            blink_thresh_r=int(blink_thresh_r),
            fit_mae_pct_x=round(mae_x, 2),
            fit_mae_pct_y=round(mae_y, 2),
        )

    def predict_u16(
        self,
        h2: Optional[np.ndarray],
        gaze_x: int,
        gaze_y: int,
        head_x: int = 0,
        head_y: int = 0,
        pup_lx: int = 0,
        pup_ly: int = 0,
        pup_rx: int = 0,
        pup_ry: int = 0,
        yaw: int = 0,
        pitch: int = 0,
    ) -> Tuple[int, int]:
        """Pure-integer screen coordinate prediction in [0, 65535]."""
        f = build_cal_feature(
            h2, gaze_x, gaze_y, head_x, head_y, pup_lx, pup_ly, pup_rx, pup_ry, yaw, pitch
        ).astype(np.int64)
        sx = int(np.dot(f, self.wx_q12.astype(np.int64))) >> Q_SHIFT
        sy = int(np.dot(f, self.wy_q12.astype(np.int64))) >> Q_SHIFT
        return max(0, min(65535, sx)), max(0, min(65535, sy))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ternara_gaze_cal_q12_v2",
            "q_shift": Q_SHIFT,
            "feat_dim": CAL_FEAT_DIM,
            "wx_q12": [int(v) for v in self.wx_q12],
            "wy_q12": [int(v) for v in self.wy_q12],
            "blink_thresh_l": int(self.blink_thresh_l),
            "blink_thresh_r": int(self.blink_thresh_r),
            "fit_mae_pct_x": self.fit_mae_pct_x,
            "fit_mae_pct_y": self.fit_mae_pct_y,
        }
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntegerGazeCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "ternara_gaze_cal_q12_v2":
            raise ValueError(f"Unsupported or outdated calibration format: {data.get('format')}; please re-run calibrate.")
        return cls(
            wx_q12=np.asarray(data["wx_q12"], dtype=np.int32),
            wy_q12=np.asarray(data["wy_q12"], dtype=np.int32),
            blink_thresh_l=int(data.get("blink_thresh_l", 24)),
            blink_thresh_r=int(data.get("blink_thresh_r", 24)),
            fit_mae_pct_x=float(data.get("fit_mae_pct_x", 0.0)),
            fit_mae_pct_y=float(data.get("fit_mae_pct_y", 0.0)),
        )
