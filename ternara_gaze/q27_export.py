"""Clean-room Q27 hardware source emitter, 8-bit pack5 ROM generator, and FPGA budget estimator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .pack5 import TernaryCartridge, pack5_encode


@dataclass(frozen=True)
class FpgaBudget:
    board: str
    clock_mhz: float
    lanes_parallel: int
    total_trits: int
    pack5_bytes: int
    bias_bytes: int
    total_rom_bytes: int
    bram_blocks_used: int
    bram_blocks_total: int
    bram_block_type: str
    dsp_used: int
    dram_bytes_used: int
    cycles_per_frame: int
    latency_us: float
    max_fps: float
    est_lut_count: int
    est_lut_pct: float


def estimate_fpga_budgets(cartridge: TernaryCartridge, lanes_parallel: int = 16) -> Dict[str, FpgaBudget]:
    """Compute honest, conservative FPGA resource & cycle budgets for AX7203 and ECP5-85F."""
    trits = cartridge.total_trits
    pack5_bytes = (trits + 4) // 5
    bias_bytes = 2 * (cartridge.h1 + cartridge.h2 + cartridge.out_dim)
    total_rom_bytes = pack5_bytes + bias_bytes

    # Cycles: 256-input feature prep (~64 cycles streaming) + ternary MACs / lanes_parallel
    mac_cycles = (
        (cartridge.in_dim * cartridge.h1 + lanes_parallel - 1) // lanes_parallel
        + (cartridge.h1 * cartridge.h2 + lanes_parallel - 1) // lanes_parallel
        + (cartridge.h2 * cartridge.out_dim + lanes_parallel - 1) // lanes_parallel
    )
    cycles = 64 + mac_cycles + 12  # pipeline flush

    # LUT estimate: 16-lane ternary adder tree + pack5 decoder ROM (243x10b) + FSM + UART
    est_lut6 = 1450 + 48 * lanes_parallel
    est_lut4 = int(est_lut6 * 1.45)

    ax_clk = 25.0
    ax_lat_us = round(cycles / ax_clk, 2)
    ax_bram = max(1, (total_rom_bytes + 4095) // 4096)  # 4 KB per RAMB36E1 in strict 8-bit mode

    ecp_clk = 25.0
    ecp_lat_us = round(cycles / ecp_clk, 2)
    ecp_ebr = max(1, (total_rom_bytes + 2047) // 2048)  # 2 KB per ECP5 18Kb EBR in strict 8-bit mode

    return {
        "ax7203_xc7a200t": FpgaBudget(
            board="ALINX AX7203 (XC7A200T)",
            clock_mhz=ax_clk,
            lanes_parallel=lanes_parallel,
            total_trits=trits,
            pack5_bytes=pack5_bytes,
            bias_bytes=bias_bytes,
            total_rom_bytes=total_rom_bytes,
            bram_blocks_used=ax_bram,
            bram_blocks_total=365,
            bram_block_type="RAMB36E1 (strict 8-bit lane, 0 parity reliance)",
            dsp_used=0,
            dram_bytes_used=0,
            cycles_per_frame=cycles,
            latency_us=ax_lat_us,
            max_fps=round(1_000_000.0 / ax_lat_us, 1),
            est_lut_count=est_lut6,
            est_lut_pct=round(100.0 * est_lut6 / 134600.0, 2),
        ),
        "ecp5_85f": FpgaBudget(
            board="Lattice ECP5-85F (Kickstarter Target)",
            clock_mhz=ecp_clk,
            lanes_parallel=lanes_parallel,
            total_trits=trits,
            pack5_bytes=pack5_bytes,
            bias_bytes=bias_bytes,
            total_rom_bytes=total_rom_bytes,
            bram_blocks_used=ecp_ebr,
            bram_blocks_total=208,
            bram_block_type="DP16KD EBR (strict 8-bit lane)",
            dsp_used=0,
            dram_bytes_used=0,
            cycles_per_frame=cycles,
            latency_us=ecp_lat_us,
            max_fps=round(1_000_000.0 / ecp_lat_us, 1),
            est_lut_count=est_lut4,
            est_lut_pct=round(100.0 * est_lut4 / 83640.0, 2),
        ),
    }


def export_q27_bundle(cartridge: TernaryCartridge, out_dir: str | Path) -> Dict[str, str]:
    """Emit clean-room Q27 v0.3-draft core source, strict 8-bit pack5 ROM hex, and budget JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_trits = np.concatenate([
        cartridge.w1.reshape(-1),
        cartridge.w2.reshape(-1),
        cartridge.w3.reshape(-1),
    ])
    packed = pack5_encode(all_trits)
    hex_path = out / "ternara_gaze_pack5.hex"
    hex_path.write_text("".join(f"{b:02x}\n" for b in packed), encoding="utf-8")

    biases_all = np.concatenate([cartridge.b1, cartridge.b2, cartridge.b3]).astype(np.int16)
    bias_hex_path = out / "ternara_gaze_biases_i16.hex"
    bias_hex_path.write_text(
        "".join(f"{int(v) & 0xFFFF:04x}\n" for v in biases_all),
        encoding="utf-8",
    )

    budgets = estimate_fpga_budgets(cartridge)
    budget_path = out / "fpga_budget.json"
    budget_path.write_text(
        json.dumps({k: v.__dict__ for k, v in budgets.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    q27_src = f"""// Ternara-Gaze v0 — Clean-Room Q27 Hardware Kernel Specification (v0.3 pack5 profile)
// Generated from cartridge sha256: {cartridge.sha256()}
// Hardware invariants:
//   - 0 DSP slices, 0 floating-point units, 0 external DRAM
//   - Strict 8-bit BRAM ROM lanes (radix-243 pack5: 5 trits/byte; bytes > 242 raise FAULT)
//   - Input: 2x (16x32) 4-bit grayscale eye ROIs = 512 bytes packed nibbles
//   - Mean centering via exact right-shift >> 9 (16*32 = 512 = 2^9; zero dividers)
//   - Topology: {cartridge.in_dim} -> {cartridge.h1} (>>{cartridge.s1}) -> {cartridge.h2} (>>{cartridge.s2}) -> {cartridge.out_dim}
//   - Total weights: {cartridge.total_trits} trits ({len(packed)} pack5 bytes; zero sparsity {cartridge.zero_fraction*100:.1f}%)

module ternara_gaze_core
  param IN_DIM  = {cartridge.in_dim}
  param H1_DIM  = {cartridge.h1}
  param H2_DIM  = {cartridge.h2}
  param OUT_DIM = {cartridge.out_dim}
  param SHIFT_1 = {cartridge.s1}
  param SHIFT_2 = {cartridge.s2}
  param PACK5_BYTES = {len(packed)}

  rom8  weights_pack5[PACK5_BYTES] = "ternara_gaze_pack5.hex"
  rom16 biases_i16[{len(biases_all)}] = "ternara_gaze_biases_i16.hex"

  // Step 1: Stream dual 16x32 u4 eye crops -> 2x2 sum pool - (sum >> 9) << 2 -> int7[256]
  stage feature_extract(left_u4[512], right_u4[512]) -> x[256]:
    for eye in 0..1:
      mean = (sum_all(eye) >> 9)
      for r in 0..7, c in 0..15:
        x[eye*128 + r*16 + c] = clamp_i7(pool2x2_sum(eye, r, c) - (mean << 2))

  // Step 2: Multiplication-free ternary layers (add / sub / skip only)
  stage layer1(x[256]) -> h1[{cartridge.h1}]:
    for o in 0..{cartridge.h1 - 1}:
      acc = biases_i16[o] + ternary_dot_pack5(weights_pack5, o * 256, x, 256)
      h1[o] = clamp_u7(acc >> SHIFT_1)

  stage layer2(h1[{cartridge.h1}]) -> h2[{cartridge.h2}]:
    for o in 0..{cartridge.h2 - 1}:
      acc = biases_i16[{cartridge.h1} + o] + ternary_dot_pack5(weights_pack5, {cartridge.in_dim * cartridge.h1} + o * {cartridge.h1}, h1, {cartridge.h1})
      h2[o] = clamp_u7(acc >> SHIFT_2)

  stage head(h2[{cartridge.h2}]) -> out[4]:
    for o in 0..3:
      out[o] = biases_i16[{cartridge.h1 + cartridge.h2} + o] + ternary_dot_pack5(weights_pack5, {cartridge.in_dim * cartridge.h1 + cartridge.h1 * cartridge.h2} + o * {cartridge.h2}, h2, {cartridge.h2})
end
"""
    q27_path = out / "ternara_gaze_v0.q27"
    q27_path.write_text(q27_src, encoding="utf-8")

    return {
        "q27": str(q27_path),
        "pack5_hex": str(hex_path),
        "biases_hex": str(bias_hex_path),
        "budget_json": str(budget_path),
    }
