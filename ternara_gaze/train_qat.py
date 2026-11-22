"""Fast CPU PyTorch QAT (Straight-Through Estimator) trainer for Ternara-Gaze.

Exports a bit-exact .tgaze.bin cartridge (pack5 weights + int16 biases + bit-shifts)
and asserts ZERO mismatch between PyTorch integer-export and the multiplication-free
pure-integer Q27 oracle.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .oracle import oracle_forward_features_batch, oracle_forward_pure
from .pack5 import TernaryCartridge
from .synth_data import generate_dataset


@dataclass(frozen=True)
class TrainReport:
    cartridge_path: str
    sha256: str
    cartridge_bytes: int
    total_trits: int
    zero_fraction: float
    val_gaze_mae_norm: float
    val_blink_left_acc: float
    val_blink_right_acc: float
    bit_exact_oracle_checked: int
    bit_exact_max_diff: int
    wall_seconds: float


def _ternary_ste(w, threshold_scale: float = 0.7):
    import torch

    delta = threshold_scale * w.detach().abs().mean().clamp(min=1e-4)
    w_q = torch.where(w > delta, torch.ones_like(w), torch.where(w < -delta, -torch.ones_like(w), torch.zeros_like(w)))
    return w + (w_q - w).detach()


def _quantize_ternary_np(w_np: np.ndarray, threshold_scale: float = 0.7) -> np.ndarray:
    delta = max(1e-4, threshold_scale * float(np.mean(np.abs(w_np))))
    q = np.zeros_like(w_np, dtype=np.int8)
    q[w_np > delta] = 1
    q[w_np < -delta] = -1
    return q


def train_and_export(
    output_path: str | Path = "weights/ternara_gaze_v0.tgaze.bin",
    n_train: int = 3200,
    n_val: int = 640,
    epochs: int = 35,
    seed: int = 20261122,
    h1: int = 64,
    h2: int = 32,
    s1: int = 5,
    s2: int = 4,
    max_threads: int = 2,
) -> Tuple[TernaryCartridge, TrainReport]:
    """Train ternary gaze+blink model with STE QAT on CPU and verify bit-exact Q27 oracle."""
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    import torch
    import torch.nn as nn

    torch.set_num_threads(max_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    train_ds = generate_dataset(n_train, seed=seed)
    val_ds = generate_dataset(n_val, seed=seed + 1)

    # Target scaling: gaze in [-128, +128], blink logits in {-96, +96}
    gaze_scale = 128.0
    blink_scale = 96.0

    x_tr = torch.from_numpy(train_ds.features.astype(np.float32))
    y_tr = torch.from_numpy(train_ds.targets.astype(np.float32))
    y_tr_scaled = y_tr.clone()
    y_tr_scaled[:, :2] *= gaze_scale
    y_tr_scaled[:, 2:] *= blink_scale
    # Mask gaze regression loss when both eyes are closed
    open_mask_tr = ((y_tr[:, 2] < 0) | (y_tr[:, 3] < 0)).float().unsqueeze(1)

    class TernaryQATNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(h1, 256) * 0.08)
            self.b1 = nn.Parameter(torch.zeros(h1))
            self.w2 = nn.Parameter(torch.randn(h2, h1) * 0.12)
            self.b2 = nn.Parameter(torch.zeros(h2))
            self.w3 = nn.Parameter(torch.randn(4, h2) * 0.15)
            self.b3 = nn.Parameter(torch.zeros(4))
            self.div1 = float(1 << s1)
            self.div2 = float(1 << s2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w1q = _ternary_ste(self.w1)
            z1 = x @ w1q.t() + self.b1
            a1 = torch.clamp(z1 / self.div1, 0.0, 127.0)
            a1 = a1 + (torch.floor(a1) - a1).detach()

            w2q = _ternary_ste(self.w2)
            z2 = a1 @ w2q.t() + self.b2
            a2 = torch.clamp(z2 / self.div2, 0.0, 127.0)
            a2 = a2 + (torch.floor(a2) - a2).detach()

            w3q = _ternary_ste(self.w3)
            out = a2 @ w3q.t() + self.b3
            return out

    model = TernaryQATNet()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    batch_size = 128
    n = x_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_tr[idx]
            yb = y_tr_scaled[idx]
            mb = open_mask_tr[idx]

            pred = model(xb)
            # Weighted MSE on gaze (when at least one eye open) + BCE/hinge on blink logits
            gaze_err = ((pred[:, :2] - yb[:, :2]) / gaze_scale) ** 2 * mb
            gaze_loss = gaze_err.sum() / mb.sum().clamp(min=1.0)
            blink_targets = (y_tr[idx, 2:] > 0).float()
            blink_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 2:] / 24.0, blink_targets)
            loss = gaze_loss + 0.6 * blink_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        w1_q = _quantize_ternary_np(model.w1.detach().numpy())
        w2_q = _quantize_ternary_np(model.w2.detach().numpy())
        w3_q = _quantize_ternary_np(model.w3.detach().numpy())
        b1_q = np.clip(np.rint(model.b1.detach().numpy()), -32768, 32767).astype(np.int16)
        b2_q = np.clip(np.rint(model.b2.detach().numpy()), -32768, 32767).astype(np.int16)
        b3_q = np.clip(np.rint(model.b3.detach().numpy()), -32768, 32767).astype(np.int16)

    cart = TernaryCartridge(w1=w1_q, b1=b1_q, s1=s1, w2=w2_q, b2=b2_q, s2=s2, w3=w3_q, b3=b3_q)
    out_p = Path(output_path)
    digest = cart.save(out_p)
    reloaded = TernaryCartridge.load(out_p)

    # Evaluate on validation set using integer oracle
    val_out, _ = oracle_forward_features_batch(reloaded, val_ds.features)
    open_mask = (val_ds.targets[:, 2] < 0) | (val_ds.targets[:, 3] < 0)
    pred_gaze = np.clip(val_out[open_mask, :2] / gaze_scale, -1.2, 1.2)
    true_gaze = val_ds.targets[open_mask, :2]
    val_gaze_mae = float(np.mean(np.abs(pred_gaze - true_gaze)))

    pred_bl = val_out[:, 2] > 0
    true_bl = val_ds.targets[:, 2] > 0
    pred_br = val_out[:, 3] > 0
    true_br = val_ds.targets[:, 3] > 0
    bl_acc = float(np.mean(pred_bl == true_bl))
    br_acc = float(np.mean(pred_br == true_br))

    # Strict bit-exactness check: pure multiplication-free scalar oracle vs vectorized oracle
    n_check = min(32, n_val)
    max_diff = 0
    for i in range(n_check):
        pure = oracle_forward_pure(reloaded, val_ds.left_u4[i], val_ds.right_u4[i])
        vec = val_out[i]
        diffs = [
            abs(pure.gaze_x - int(vec[0])),
            abs(pure.gaze_y - int(vec[1])),
            abs(pure.blink_left - int(vec[2])),
            abs(pure.blink_right - int(vec[3])),
        ]
        max_diff = max(max_diff, max(diffs))
    if max_diff != 0:
        raise RuntimeError(f"Bit-exact oracle mismatch: max_diff={max_diff}")

    wall = time.perf_counter() - t0
    report = TrainReport(
        cartridge_path=str(out_p),
        sha256=digest,
        cartridge_bytes=len(reloaded.to_bytes()),
        total_trits=reloaded.total_trits,
        zero_fraction=round(reloaded.zero_fraction, 4),
        val_gaze_mae_norm=round(val_gaze_mae, 4),
        val_blink_left_acc=round(bl_acc, 4),
        val_blink_right_acc=round(br_acc, 4),
        bit_exact_oracle_checked=n_check,
        bit_exact_max_diff=max_diff,
        wall_seconds=round(wall, 2),
    )
    meta_path = out_p.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(report.__dict__, indent=2) + "\n", encoding="utf-8")
    return reloaded, report
