"""Pure-integer Q12 Axis-Pure Nose-Anchored + Eye-Refined gaze calibrator (0 floats, 0 cross-axis leakage).

Feature vector layout (CAL_FEAT_DIM = 14):
  X-axis active slots [0, 2, 4, 6, 13]: nose_x, yaw,   pup_avg_x, q27_gx, bias(64)
  Y-axis active slots [1, 3, 5, 7, 13]: nose_y, pitch, pup_avg_y, q27_gy, bias(64)

Decoupling X and Y fits guarantees:
  - Moving nose/eyes horizontally NEVER leaks into vertical cursor motion.
  - Moving nose/eyes vertically NEVER leaks into horizontal cursor motion.
  - Only 5 parameters per axis fitted over 9 corner-first targets (~250 frames) -> zero overfitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

Q_SHIFT = 12
Q_SCALE = 1 << Q_SHIFT  # 4096
CAL_FEAT_DIM = 14

X_ACTIVE_IDX = (0, 2, 4, 6, 13)
Y_ACTIVE_IDX = (1, 3, 5, 7, 13)

# 9-point corner-first calibration grid (4 TRUE extreme corners FIRST, then center + 4 edges):
CORNER_FIRST_GRID: Tuple[Tuple[float, float, str], ...] = (
    (0.05, 0.06, "1/9  TOP-LEFT CORNER"),
    (0.95, 0.06, "2/9  TOP-RIGHT CORNER"),
    (0.95, 0.93, "3/9  BOTTOM-RIGHT CORNER"),
    (0.05, 0.93, "4/9  BOTTOM-LEFT CORNER"),
    (0.50, 0.50, "5/9  SCREEN CENTER"),
    (0.50, 0.06, "6/9  TOP EDGE"),
    (0.95, 0.50, "7/9  RIGHT EDGE"),
    (0.50, 0.93, "8/9  BOTTOM EDGE"),
    (0.05, 0.50, "9/9  LEFT EDGE"),
)

NINE_POINT_GRID_NORM: Tuple[Tuple[float, float], ...] = tuple((x, y) for x, y, _ in CORNER_FIRST_GRID)


def dark_pupil_centroid_int(crop_u4: np.ndarray) -> Tuple[int, int]:
    """Pure-integer dark-pupil centroid inside a (16, 32) 4-bit eye crop -> (px, py) in [-128, +127]."""
    c = np.asarray(crop_u4, dtype=np.int32)
    roi = c[3:13, 5:27]  # 10 x 22
    max_v = int(roi.max())
    min_v = int(roi.min())
    thresh = min_v + max(1, (max_v - min_v) >> 2)
    weights = np.maximum(0, thresh - roi + 1)
    total_w = int(weights.sum())
    if total_w <= 0:
        return 0, 0
    yy, xx = np.mgrid[0:10, 0:22]
    cx_q8 = (int((weights * xx).sum()) << 8) // total_w
    cy_q8 = (int((weights * yy).sum()) << 8) // total_w
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
    """Assemble the 14-element SNR-ordered int32 calibration feature vector (nose-first + eye-refined)."""
    del h2
    nose_x = int(head_x)
    nose_y = int(head_y)
    pup_avg_x = (int(pup_lx) + int(pup_rx)) >> 1
    pup_avg_y = (int(pup_ly) + int(pup_ry)) >> 1
    pup_diff_x = (int(pup_lx) - int(pup_rx)) >> 1
    pup_diff_y = (int(pup_ly) - int(pup_ry)) >> 1
    return np.array(
        [
            nose_x,
            nose_y,
            int(yaw),
            int(pitch),
            pup_avg_x,
            pup_avg_y,
            int(gaze_x),
            int(gaze_y),
            pup_diff_x,
            pup_diff_y,
            0,
            0,
            0,
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


def _fit_axis(
    X_full: np.ndarray,
    y_u16: np.ndarray,
    w: np.ndarray,
    active_idx: Tuple[int, ...],
    ridge_lambda: float,
) -> np.ndarray:
    """Fit a clean 5-parameter axis-pure weighted ridge regression -> (14,) int32 Q12 vector."""
    X_sub = X_full[:, active_idx]
    Xw = X_sub * w[:, None]
    # [nose, pose, pup_avg, q27_gaze, bias]
    tier = np.array([1.0, 1.0, 4.5, 6.0, 1e-6], dtype=np.float64)
    reg = np.diag(tier * float(ridge_lambda))
    w_sub = np.linalg.solve(Xw.T @ X_sub + reg, Xw.T @ y_u16)

    out = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
    lim = (1 << 30) - 1
    for j, col in enumerate(active_idx):
        out[col] = int(np.clip(np.rint(w_sub[j] * Q_SCALE), -lim, lim))
    return out


@dataclass(frozen=True)
class IntegerGazeCalibrator:
    """Fixed-point Q12 axis-pure screen coordinate mapper (0 floats at inference)."""

    wx_q12: np.ndarray  # (14,) int32 (non-zero only on X_ACTIVE_IDX)
    wy_q12: np.ndarray  # (14,) int32 (non-zero only on Y_ACTIVE_IDX)
    blink_thresh_l: int = 24
    blink_thresh_r: int = 24
    ear_thresh_l: int = 340
    ear_thresh_r: int = 340
    fit_mae_pct_x: float = 0.0
    fit_mae_pct_y: float = 0.0
    online_updates: int = 0

    def __post_init__(self) -> None:
        if self.wx_q12.shape != (CAL_FEAT_DIM,) or self.wy_q12.shape != (CAL_FEAT_DIM,):
            raise ValueError(f"Calibrator weights must have shape ({CAL_FEAT_DIM},)")

    @classmethod
    def identity_default(cls) -> "IntegerGazeCalibrator":
        wx = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        wy = np.zeros(CAL_FEAT_DIM, dtype=np.int32)
        wx[0] = 320 << Q_SHIFT   # nose_x
        wx[2] = 280 << Q_SHIFT   # yaw
        wx[4] = 120 << Q_SHIFT   # pup_avg_x
        wx[13] = 512 << Q_SHIFT  # center
        wy[1] = 360 << Q_SHIFT   # nose_y
        wy[3] = 300 << Q_SHIFT   # pitch
        wy[5] = 120 << Q_SHIFT   # pup_avg_y
        wy[13] = 512 << Q_SHIFT
        return cls(wx_q12=wx, wy_q12=wy)

    @classmethod
    def fit(
        cls,
        features: Sequence[np.ndarray] | np.ndarray,
        targets_norm: Sequence[Tuple[float, float]] | np.ndarray,
        ridge_lambda: float = 600.0,
        blink_thresh_l: int = 24,
        blink_thresh_r: int = 24,
        ear_thresh_l: int = 340,
        ear_thresh_r: int = 340,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> "IntegerGazeCalibrator":
        """Fit strictly axis-pure weighted ridge regression (zero cross-axis leakage)."""
        X = np.asarray(features, dtype=np.float64)
        Y = np.asarray(targets_norm, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != CAL_FEAT_DIM:
            raise ValueError(f"Expected X shape (N, {CAL_FEAT_DIM}), got {X.shape}")
        if Y.ndim != 2 or Y.shape != (X.shape[0], 2):
            raise ValueError(f"Expected Y shape ({X.shape[0]}, 2), got {Y.shape}")
        if X.shape[0] < 4:
            raise ValueError("Need at least 4 calibration samples")

        w = np.ones(X.shape[0], dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
        Y_u16 = np.clip(Y, 0.0, 1.0) * 65535.0

        wx_q12 = _fit_axis(X, Y_u16[:, 0], w, X_ACTIVE_IDX, ridge_lambda)
        wy_q12 = _fit_axis(X, Y_u16[:, 1], w, Y_ACTIVE_IDX, ridge_lambda)

        pred_x = np.clip((X.astype(np.int64) @ wx_q12.astype(np.int64)) >> Q_SHIFT, 0, 65535)
        pred_y = np.clip((X.astype(np.int64) @ wy_q12.astype(np.int64)) >> Q_SHIFT, 0, 65535)
        mae_x = float(np.mean(np.abs(pred_x - Y_u16[:, 0])) / 655.35)
        mae_y = float(np.mean(np.abs(pred_y - Y_u16[:, 1])) / 655.35)

        return cls(
            wx_q12=wx_q12,
            wy_q12=wy_q12,
            blink_thresh_l=int(blink_thresh_l),
            blink_thresh_r=int(blink_thresh_r),
            ear_thresh_l=int(ear_thresh_l),
            ear_thresh_r=int(ear_thresh_r),
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

    def online_learn_step(
        self,
        feat: np.ndarray,
        target_x_u16: int,
        target_y_u16: int,
        lr: float = 0.14,
        max_error_u16: int = 19000,
    ) -> Tuple["IntegerGazeCalibrator", bool]:
        """Safe axis-pure NLMS online adaptation step (never leaks across X/Y axes)."""
        f = np.asarray(feat, dtype=np.float64)
        if f.shape != (CAL_FEAT_DIM,):
            return self, False

        pred_x = max(0, min(65535, int(np.dot(f.astype(np.int64), self.wx_q12.astype(np.int64))) >> Q_SHIFT))
        pred_y = max(0, min(65535, int(np.dot(f.astype(np.int64), self.wy_q12.astype(np.int64))) >> Q_SHIFT))
        tx = max(0, min(65535, int(target_x_u16)))
        ty = max(0, min(65535, int(target_y_u16)))

        ex = float(tx - pred_x)
        ey = float(ty - pred_y)
        if abs(ex) > max_error_u16 or abs(ey) > max_error_u16:
            return self, False

        lim = (1 << 30) - 1
        new_wx = self.wx_q12.astype(np.int64).copy()
        new_wy = self.wy_q12.astype(np.int64).copy()

        # Axis-pure X update
        tier_sub = np.array([1.0, 1.0, 0.28, 0.20, 1.6], dtype=np.float64)
        fx = f[list(X_ACTIVE_IDX)]
        dx_step = (lr / (float(np.sum(tier_sub * fx * fx)) + 8000.0)) * float(Q_SCALE)
        for j, col in enumerate(X_ACTIVE_IDX):
            new_wx[col] += int(np.rint(ex * dx_step * tier_sub[j] * fx[j]))

        # Axis-pure Y update
        fy = f[list(Y_ACTIVE_IDX)]
        dy_step = (lr / (float(np.sum(tier_sub * fy * fy)) + 8000.0)) * float(Q_SCALE)
        for j, col in enumerate(Y_ACTIVE_IDX):
            new_wy[col] += int(np.rint(ey * dy_step * tier_sub[j] * fy[j]))

        return (
            IntegerGazeCalibrator(
                wx_q12=np.clip(new_wx, -lim, lim).astype(np.int32),
                wy_q12=np.clip(new_wy, -lim, lim).astype(np.int32),
                blink_thresh_l=self.blink_thresh_l,
                blink_thresh_r=self.blink_thresh_r,
                ear_thresh_l=self.ear_thresh_l,
                ear_thresh_r=self.ear_thresh_r,
                fit_mae_pct_x=self.fit_mae_pct_x,
                fit_mae_pct_y=self.fit_mae_pct_y,
                online_updates=self.online_updates + 1,
            ),
            True,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ternara_gaze_cal_q12_v4",
            "q_shift": Q_SHIFT,
            "feat_dim": CAL_FEAT_DIM,
            "wx_q12": [int(v) for v in self.wx_q12],
            "wy_q12": [int(v) for v in self.wy_q12],
            "blink_thresh_l": int(self.blink_thresh_l),
            "blink_thresh_r": int(self.blink_thresh_r),
            "ear_thresh_l": int(self.ear_thresh_l),
            "ear_thresh_r": int(self.ear_thresh_r),
            "fit_mae_pct_x": self.fit_mae_pct_x,
            "fit_mae_pct_y": self.fit_mae_pct_y,
            "online_updates": int(self.online_updates),
        }
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntegerGazeCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "ternara_gaze_cal_q12_v4":
            raise ValueError(f"Unsupported or outdated calibration format: {data.get('format')}; please re-run calibrate.")
        return cls(
            wx_q12=np.asarray(data["wx_q12"], dtype=np.int32),
            wy_q12=np.asarray(data["wy_q12"], dtype=np.int32),
            blink_thresh_l=int(data.get("blink_thresh_l", 24)),
            blink_thresh_r=int(data.get("blink_thresh_r", 24)),
            ear_thresh_l=int(data.get("ear_thresh_l", 340)),
            ear_thresh_r=int(data.get("ear_thresh_r", 340)),
            fit_mae_pct_x=float(data.get("fit_mae_pct_x", 0.0)),
            fit_mae_pct_y=float(data.get("fit_mae_pct_y", 0.0)),
            online_updates=int(data.get("online_updates", 0)),
        )
