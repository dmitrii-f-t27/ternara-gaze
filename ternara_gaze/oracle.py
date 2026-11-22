"""Bit-exact pure-integer, multiplication-free Q27 oracle for Ternara-Gaze.

Every inference operation uses strictly:
  - 4-bit unsigned eye pixels (0..15) in two 16x32 ROIs (512 bytes packed)
  - exact integer mean-centering (sum >> 9, because 16*32 = 512 = 2^9; zero dividers)
  - 2x2 sum pooling (16x32 -> 8x16 = 128 ints per eye -> 256 ints total)
  - ternary dot products using ONLY add, subtract, or skip (ZERO multiplications, ZERO DSPs)
  - arithmetic right-shift and clamp [0, 127] (int7 ReLU activations)

Outputs:
  - raw_out: (gaze_x_q8, gaze_y_q8, blink_left_q8, blink_right_q8) signed int32
  - h2: 32-element int8 penultimate feature vector (used for on-chip 9-point calibration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .pack5 import TernaryCartridge

EYE_H = 16
EYE_W = 32
EYE_PIXELS = EYE_H * EYE_W  # 512 = 2^9
FRAME_PACKED_BYTES = EYE_PIXELS  # 2 eyes * 512 nibbles / 2 = 512 bytes
FEATURE_DIM = 256  # 2 eyes * (8 * 16)


@dataclass(frozen=True)
class OracleOutput:
    gaze_x: int        # signed integer, nominal [-128, +128]
    gaze_y: int        # signed integer, nominal [-128, +128]
    blink_left: int    # > 0 means left eye closed
    blink_right: int   # > 0 means right eye closed
    h2: np.ndarray     # (32,) int8 penultimate features for on-chip calibration


def pack_dual_eye_nibbles(left_u4: np.ndarray, right_u4: np.ndarray) -> bytes:
    """Pack two (16, 32) 4-bit eye crops (values 0..15) into 512 bytes (2 pixels/byte)."""
    l = np.asarray(left_u4, dtype=np.uint8)
    r = np.asarray(right_u4, dtype=np.uint8)
    if l.shape != (EYE_H, EYE_W) or r.shape != (EYE_H, EYE_W):
        raise ValueError(f"Expected ({EYE_H}, {EYE_W}) eye crops, got {l.shape} and {r.shape}")
    if np.any(l > 15) or np.any(r > 15):
        raise ValueError("4-bit eye pixels must be in [0, 15]")
    flat = np.concatenate([l.reshape(-1), r.reshape(-1)])  # 1024 nibbles
    packed = ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)
    return packed.astype(np.uint8).tobytes()


def unpack_dual_eye_nibbles(payload: bytes | bytearray | memoryview) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack 512 bytes into two (16, 32) uint8 arrays with values in [0, 15]."""
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size != FRAME_PACKED_BYTES:
        raise ValueError(f"Expected {FRAME_PACKED_BYTES} bytes, got {raw.size}")
    hi = (raw >> 4) & 0x0F
    lo = raw & 0x0F
    flat = np.empty((EYE_PIXELS * 2,), dtype=np.uint8)
    flat[0::2] = hi
    flat[1::2] = lo
    left = flat[:EYE_PIXELS].reshape(EYE_H, EYE_W)
    right = flat[EYE_PIXELS:].reshape(EYE_H, EYE_W)
    return left, right


def extract_features_int(left_u4: np.ndarray, right_u4: np.ndarray) -> np.ndarray:
    """Pure-integer, division-free, multiplication-free eye feature extractor.

    For each 16x32 4-bit eye crop:
      1. Compute exact mean via sum >> 9 (since 16*32 = 512 = 2^9).
      2. Compute 2x2 block sum (4 pixels in [0..15], sum in [0..60]).
      3. Subtract 4 * mean (= mean << 2), giving signed contrast in [-60, +60].
      4. Clamp to [-64, +63] (int7 signed range).
    Returns a (256,) int16 array.
    """
    feats = []
    for crop in (left_u4, right_u4):
        c = np.asarray(crop, dtype=np.int32)
        if c.shape != (EYE_H, EYE_W):
            raise ValueError(f"Expected ({EYE_H}, {EYE_W}) eye crop, got {c.shape}")
        mean = int(c.sum()) >> 9
        mean4 = mean << 2
        # 2x2 sum pooling -> (8, 16)
        pooled = (
            c[0::2, 0::2]
            + c[0::2, 1::2]
            + c[1::2, 0::2]
            + c[1::2, 1::2]
            - mean4
        )
        pooled = np.clip(pooled, -64, 63).astype(np.int16).reshape(-1)
        feats.append(pooled)
    return np.concatenate(feats)


def _ternary_matvec_pure(w: np.ndarray, x: np.ndarray, b: np.ndarray) -> list[int]:
    """Reference multiplication-free ternary matrix-vector product (strictly + / - / skip)."""
    rows, cols = w.shape
    x_list = [int(v) for v in x]
    out: list[int] = []
    for r in range(rows):
        acc = int(b[r])
        row = w[r]
        for c in range(cols):
            wt = int(row[c])
            if wt == 1:
                acc += x_list[c]
            elif wt == -1:
                acc -= x_list[c]
        out.append(acc)
    return out


def _relu_shift_clamp_pure(vals: list[int], shift: int) -> list[int]:
    """Arithmetic right shift + ReLU clamp to [0, 127]."""
    res: list[int] = []
    for v in vals:
        s = v >> shift
        if s < 0:
            s = 0
        elif s > 127:
            s = 127
        res.append(s)
    return res


def oracle_forward_pure(
    cartridge: TernaryCartridge,
    left_u4: np.ndarray,
    right_u4: np.ndarray,
) -> OracleOutput:
    """Bit-exact scalar reference oracle (zero floats, zero multiplications)."""
    x = extract_features_int(left_u4, right_u4)
    z1 = _ternary_matvec_pure(cartridge.w1, x, cartridge.b1)
    h1 = _relu_shift_clamp_pure(z1, cartridge.s1)
    z2 = _ternary_matvec_pure(cartridge.w2, np.asarray(h1, dtype=np.int32), cartridge.b2)
    h2 = _relu_shift_clamp_pure(z2, cartridge.s2)
    out = _ternary_matvec_pure(cartridge.w3, np.asarray(h2, dtype=np.int32), cartridge.b3)
    return OracleOutput(
        gaze_x=int(out[0]),
        gaze_y=int(out[1]),
        blink_left=int(out[2]),
        blink_right=int(out[3]),
        h2=np.asarray(h2, dtype=np.int8),
    )


def oracle_forward_fast(
    cartridge: TernaryCartridge,
    left_u4: np.ndarray,
    right_u4: np.ndarray,
) -> OracleOutput:
    """Vectorized integer forward pass, guaranteed bit-identical to `oracle_forward_pure`."""
    x = extract_features_int(left_u4, right_u4).astype(np.int32)
    z1 = cartridge.w1.astype(np.int32) @ x + cartridge.b1.astype(np.int32)
    h1 = np.clip(np.right_shift(z1, cartridge.s1), 0, 127).astype(np.int32)
    z2 = cartridge.w2.astype(np.int32) @ h1 + cartridge.b2.astype(np.int32)
    h2 = np.clip(np.right_shift(z2, cartridge.s2), 0, 127).astype(np.int32)
    out = cartridge.w3.astype(np.int32) @ h2 + cartridge.b3.astype(np.int32)
    return OracleOutput(
        gaze_x=int(out[0]),
        gaze_y=int(out[1]),
        blink_left=int(out[2]),
        blink_right=int(out[3]),
        h2=h2.astype(np.int8),
    )


def oracle_forward_features_batch(
    cartridge: TernaryCartridge,
    features_int: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch integer forward pass on pre-extracted (N, 256) int features.
    Returns (out_int32 (N, 4), h2_int8 (N, 32)).
    """
    x = np.asarray(features_int, dtype=np.int32)
    z1 = x @ cartridge.w1.astype(np.int32).T + cartridge.b1.astype(np.int32)
    h1 = np.clip(np.right_shift(z1, cartridge.s1), 0, 127).astype(np.int32)
    z2 = h1 @ cartridge.w2.astype(np.int32).T + cartridge.b2.astype(np.int32)
    h2 = np.clip(np.right_shift(z2, cartridge.s2), 0, 127).astype(np.int32)
    out = h2 @ cartridge.w3.astype(np.int32).T + cartridge.b3.astype(np.int32)
    return out, h2.astype(np.int8)
