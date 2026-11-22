"""Pure-integer Q12 Nose-Anchored + Eye-Refined corner-first gaze calibrator (0 floats at inference).

Uses 14 physically orthogonal integer features ordered by signal-to-noise ratio (SNR):
  [0..1]   HD nose-tip pointer (nose_x, nose_y) in [-128..+127]       -- PRIMARY ultra-low-noise backbone
  [2..3]   3D nose-to-eye orientation (yaw, pitch) in [-128..+127]    -- PRIMARY head rotation backbone
  [4..5]   averaged rigid pupils ((pup_lx+pup_rx)//2, (pup_ly+pup_ry)//2) -- bounded fine eye refinement
  [6..7]   Q27 0-DSP ternary backbone raw (gaze_x, gaze_y)            -- bounded fine eye refinement
  [8..9]   pupil vergence/asymmetry ((pup_lx-pup_rx)//2, (pup_ly-pup_ry)//2)
  [10..12] mild quadratic terms of the clean nose+yaw/pitch signal
  [13]     constant bias = 64

Tiered ridge regularization ensures the rock-solid nose pointer carries the coarse screen zone
without ever amplifying webcam pupil noise into cursor wander.
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

NINE_POINT_GRID_NORM: Tuple[Tuple[float, float], ...] = tuple((x, y) for x, y, _ in CORNER_FIRST_GRID[:9])


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

    # Clean quadratic interaction on the low-noise nose+orientation signal
    ax = (nose_x + int(yaw)) >> 1
    ay = (nose_y + int(pitch)) >> 1
    q_xx = max(-128, min(127, (ax * ax) >> 7))
    q_yy = max(-128, min(127, (ay * ay) >> 7))
    q_xy = max(-128, min(127, (ax * ay) >> 7))
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
    ear_thresh_l: int = 165
    ear_thresh_r: int = 165
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
        wx[0] = 260 << Q_SHIFT   # nose_x
        wx[2] = 240 << Q_SHIFT   # yaw
        wx[4] = 120 << Q_SHIFT   # pup_avg_x
        wx[13] = 512 << Q_SHIFT  # center
        wy[1] = 280 << Q_SHIFT   # nose_y
        wy[3] = 260 << Q_SHIFT   # pitch
        wy[5] = 120 << Q_SHIFT   # pup_avg_y
        wy[13] = 512 << Q_SHIFT
        return cls(wx_q12=wx, wy_q12=wy)

    @classmethod
    def fit(
        cls,
        features: Sequence[np.ndarray] | np.ndarray,
        targets_norm: Sequence[Tuple[float, float]] | np.ndarray,
        ridge_lambda: float = 500.0,
        blink_thresh_l: int = 24,
        blink_thresh_r: int = 24,
        ear_thresh_l: int = 165,
        ear_thresh_r: int = 165,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> "IntegerGazeCalibrator":
        """Fit SNR-tiered weighted ridge regression (nose-primary, eye-refined) and quantize to Q12."""
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

        # SNR-tiered regularization:
        #   [0..3] nose_x, nose_y, yaw, pitch -> primary low-noise backbone (1.0x lambda)
        #   [4..7] averaged pupils + Q27 eye  -> bounded fine correction (4.5x lambda)
        #   [8..12] vergence + quadratic      -> heavily damped against noise (14.0x lambda)
        #   [13]   constant bias              -> unregularized
        tier = np.array(
            [1.0, 1.0, 1.0, 1.0, 4.5, 4.5, 5.5, 5.5, 14.0, 14.0, 14.0, 14.0, 14.0, 1e-6],
            dtype=np.float64,
        )
        reg = np.diag(tier * float(ridge_lambda))

        A = Xw.T @ X + reg
        W = np.linalg.solve(A, Xw.T @ Y_u16)
        wx_q12 = np.clip(np.rint(W[:, 0] * Q_SCALE), -(1 << 30), (1 << 30) - 1).astype(np.int32)
        wy_q12 = np.clip(np.rint(W[:, 1] * Q_SCALE), -(1 << 30), (1 << 30) - 1).astype(np.int32)

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
        """Safe bounded SNR-tiered NLMS online adaptation step (returns (updated_cal, applied)).

        Triggered when the user clicks with their real mouse/trackpad or confirms a magnetically
        locked UI button, continuously adapting the Q12 model to posture, lighting, and gaze habits.
        """
        f = np.asarray(feat, dtype=np.float64)
        if f.shape != (CAL_FEAT_DIM,):
            return self, False

        pred_x = max(0, min(65535, int(np.dot(f.astype(np.int64), self.wx_q12.astype(np.int64))) >> Q_SHIFT))
        pred_y = max(0, min(65535, int(np.dot(f.astype(np.int64), self.wy_q12.astype(np.int64))) >> Q_SHIFT))
        tx = max(0, min(65535, int(target_x_u16)))
        ty = max(0, min(65535, int(target_y_u16)))

        ex = float(tx - pred_x)
        ey = float(ty - pred_y)
        # Safety guard: skip if user clicked without looking near the target (>29% screen away)
        if abs(ex) > max_error_u16 or abs(ey) > max_error_u16:
            return self, False

        # SNR-tiered learning weights: strong adaptation on bias & nose/pose, gentle on eyes
        tier_inv = np.array(
            [1.0, 1.0, 1.0, 1.0, 0.28, 0.28, 0.22, 0.22, 0.06, 0.06, 0.04, 0.04, 0.04, 1.6],
            dtype=np.float64,
        )
        denom = float(np.sum(tier_inv * (f * f))) + 8000.0
        step_scale = (lr / denom) * float(Q_SCALE)
        dwx = np.rint(ex * step_scale * tier_inv * f).astype(np.int64)
        dwy = np.rint(ey * step_scale * tier_inv * f).astype(np.int64)

        lim = (1 << 30) - 1
        new_wx = np.clip(self.wx_q12.astype(np.int64) + dwx, -lim, lim).astype(np.int32)
        new_wy = np.clip(self.wy_q12.astype(np.int64) + dwy, -lim, lim).astype(np.int32)

        return (
            IntegerGazeCalibrator(
                wx_q12=new_wx,
                wy_q12=new_wy,
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
            "format": "ternara_gaze_cal_q12_v3",
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
        if data.get("format") != "ternara_gaze_cal_q12_v3":
            raise ValueError(f"Unsupported or outdated calibration format: {data.get('format')}; please re-run calibrate.")
        return cls(
            wx_q12=np.asarray(data["wx_q12"], dtype=np.int32),
            wy_q12=np.asarray(data["wy_q12"], dtype=np.int32),
            blink_thresh_l=int(data.get("blink_thresh_l", 24)),
            blink_thresh_r=int(data.get("blink_thresh_r", 24)),
            ear_thresh_l=int(data.get("ear_thresh_l", 165)),
            ear_thresh_r=int(data.get("ear_thresh_r", 165)),
            fit_mae_pct_x=float(data.get("fit_mae_pct_x", 0.0)),
            fit_mae_pct_y=float(data.get("fit_mae_pct_y", 0.0)),
            online_updates=int(data.get("online_updates", 0)),
        )
