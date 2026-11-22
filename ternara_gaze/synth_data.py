"""Physically-grounded synthetic dual-eye 4-bit (16x32) generator for Ternara-Gaze QAT.

Generates paired left/right eye crops modeling:
- Sclera almond aperture with independent upper/lower eyelids (open, squinting, closed, winking)
- Iris + darker pupil shifting with normalized gaze (gx, gy) in [-1, +1]
- Optional corneal specular glint (visible/IR light source reflection)
- Realistic camera/crop jitter, lighting gradients, glasses rim occlusion, and 4-bit quantization noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .oracle import EYE_H, EYE_W, extract_features_int


@dataclass(frozen=True)
class SynthBatch:
    left_u4: np.ndarray       # (N, 16, 32) uint8 in [0..15]
    right_u4: np.ndarray      # (N, 16, 32) uint8 in [0..15]
    features: np.ndarray      # (N, 256) int16
    targets: np.ndarray       # (N, 4) float32: (gx, gy, blink_l, blink_r)


def _render_single_eye(
    rng: np.random.Generator,
    gx: float,
    gy: float,
    closed: bool,
    is_right: bool,
) -> np.ndarray:
    yy, xx = np.mgrid[0:EYE_H, 0:EYE_W].astype(np.float32)
    # Normalize coordinates to [-1, 1] across eye box
    nx = (xx - (EYE_W - 1) * 0.5) / ((EYE_W - 1) * 0.5)
    ny = (yy - (EYE_H - 1) * 0.5) / ((EYE_H - 1) * 0.5)

    # Small random crop alignment jitter
    jx = float(rng.uniform(-0.08, 0.08))
    jy = float(rng.uniform(-0.08, 0.08))
    nx = nx - jx
    ny = ny - jy

    # Skin background with illumination gradient
    skin_base = float(rng.uniform(7.5, 11.0))
    grad_x = float(rng.uniform(-1.8, 1.8))
    grad_y = float(rng.uniform(-1.5, 1.5))
    img = skin_base + grad_x * nx + grad_y * ny

    # Eyelid aperture (almond shape)
    if closed:
        aperture = float(rng.uniform(0.05, 0.16))
    else:
        aperture = float(rng.uniform(0.55, 0.85))
    # Slight vertical lid follow with gaze_y
    lid_center_y = 0.12 * gy
    arch = np.maximum(0.0, 1.0 - (nx / 0.88) ** 2)
    upper_y = lid_center_y - aperture * arch
    lower_y = lid_center_y + aperture * 0.88 * arch
    inside_eye = (ny >= upper_y) & (ny <= lower_y)

    # Sclera brightness
    sclera_val = float(rng.uniform(11.5, 14.5))
    img = np.where(inside_eye, sclera_val + 0.8 * grad_x * nx, img)

    # Eyelash / eyelid crease shadow
    crease_dist = np.abs(ny - upper_y)
    lid_shadow = np.exp(-((crease_dist / 0.16) ** 2)) * (np.abs(nx) < 0.90)
    img = img - float(rng.uniform(2.8, 4.5)) * lid_shadow

    if closed:
        # Closed eyelid seam
        seam = np.exp(-(((ny - lid_center_y) / 0.14) ** 2)) * (np.abs(nx) < 0.86)
        img = img - float(rng.uniform(3.0, 4.5)) * seam
    else:
        # Iris + dark pupil centered at gaze offset
        # Slight horizontal convergence asymmetry between left/right eyes
        conv = -0.04 if is_right else 0.04
        px = 0.44 * gx + conv
        py = 0.36 * gy + lid_center_y * 0.5
        # Aspect-corrected radial distance (EYE_W = 2 * EYE_H)
        dx = (nx - px) * 2.0
        dy = ny - py
        r = np.sqrt(dx * dx + dy * dy)

        iris_r = float(rng.uniform(0.50, 0.62))
        pupil_r = iris_r * float(rng.uniform(0.45, 0.58))
        iris_val = float(rng.uniform(3.2, 6.2))
        pupil_val = float(rng.uniform(0.4, 2.0))

        img = np.where(inside_eye & (r < iris_r), iris_val, img)
        img = np.where(inside_eye & (r < pupil_r), pupil_val, img)

        # Optional corneal glint
        if rng.random() < 0.75:
            glint_x = px + float(rng.uniform(-0.10, 0.10))
            glint_y = py + float(rng.uniform(-0.10, 0.06))
            gr = np.sqrt(((nx - glint_x) * 2.0) ** 2 + (ny - glint_y) ** 2)
            img = np.where(inside_eye & (gr < 0.14), 15.0, img)

    # Occasional glasses rim near top/bottom edge
    if rng.random() < 0.20:
        rim_y = float(rng.choice([-0.82, 0.82]))
        rim = np.abs(ny - rim_y) < 0.12
        img = np.where(rim, float(rng.uniform(1.0, 3.0)), img)

    # Sensor noise + 4-bit quantization [0..15]
    img = img + rng.normal(0.0, float(rng.uniform(0.25, 0.75)), size=img.shape)
    return np.clip(np.rint(img), 0, 15).astype(np.uint8)


def generate_dataset(n_samples: int, seed: int = 20261122) -> SynthBatch:
    """Generate `n_samples` dual-eye 4-bit samples with balanced gaze, blinks, and winks."""
    rng = np.random.default_rng(seed)
    left_arr = np.empty((n_samples, EYE_H, EYE_W), dtype=np.uint8)
    right_arr = np.empty((n_samples, EYE_H, EYE_W), dtype=np.uint8)
    feats = np.empty((n_samples, 256), dtype=np.int16)
    targets = np.empty((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        gx = float(rng.uniform(-1.0, 1.0))
        gy = float(rng.uniform(-1.0, 1.0))
        u = float(rng.random())
        if u < 0.70:
            bl, br = False, False
        elif u < 0.82:
            bl, br = True, True    # both eyes closed (deliberate blink)
        elif u < 0.91:
            bl, br = True, False   # left wink
        else:
            bl, br = False, True   # right wink

        l = _render_single_eye(rng, gx, gy, closed=bl, is_right=False)
        r = _render_single_eye(rng, gx, gy, closed=br, is_right=True)
        left_arr[i] = l
        right_arr[i] = r
        feats[i] = extract_features_int(l, r)
        targets[i] = (
            gx,
            gy,
            1.0 if bl else -1.0,
            1.0 if br else -1.0,
        )

    return SynthBatch(left_u4=left_arr, right_u4=right_arr, features=feats, targets=targets)
