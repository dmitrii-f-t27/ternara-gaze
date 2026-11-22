# Ternara-Gaze FPGA Hardware Contract (V0 → V0.5 → V1)

## 1. Strict Numerical & Memory Invariants
- **No floating point, no DSP48E1 / MULT18X18D, no external DRAM.**
- **Input ROI:** Two `16×32` 4-bit unsigned luminance eye crops (`0..15`), packed 2 pixels/byte = **`512 bytes/frame`**.
- **Exact division-free mean centering:** Because `16 × 32 = 512 = 2^9`, per-eye mean is `sum_512 >> 9`. Each `2×2` block computes `p00 + p01 + p10 + p11 - (mean << 2)` clamped to signed `int7 [-64, +63]`.
- **Weight storage (`pack5`):** Five ternary weights `t0..t4 ∈ {-1, 0, +1}` encode into one unsigned 8-bit byte `b = Σ (t_k + 1) * 3^k ∈ [0, 242]`.
  - **Strict 8-bit memory lanes:** Never relies on 9-bit BRAM parity bits (`INITP`) — avoids 7-Series parity routing traps.
  - **Fault contract:** Any ROM byte `b >= 243` asserts `FAULT` (`STATUS bit 7`) and halts inference cleanly.

## 2. Framed UART Protocol (V0.5 — Host Eye Cropper ↔ FPGA Inference Core)
Default baud: `921,600` (8N1). `518 B` TX + `46 B` RX = `564 B` (~6.1 ms/frame → `> 160 FPS` wire speed).

### Host → FPGA (`518 bytes`)
| Offset | Size | Field |
|---|---:|---|
| `0..1` | 2 | Sync `0x54 0x47` (`"TG"`) |
| `2` | 1 | `FLAGS` (`bit 0`: return 32-byte `h2` feature vector) |
| `3` | 1 | `SEQ` (`uint8` sequence counter) |
| `4..515` | 512 | Packed dual-eye 4-bit nibbles (left `16×32` row-major, then right `16×32`) |
| `516..517` | 2 | `CRC16-CCITT` (big-endian, poly `0x1021`, init `0xFFFF`) over bytes `0..515` |

### FPGA → Host (`14 bytes` compact or `46 bytes` with `h2`)
| Offset | Size | Field |
|---|---:|---|
| `0..1` | 2 | Sync `0x47 0x54` (`"GT"`) |
| `2` | 1 | `STATUS` (`bit 0`: `h2` attached; `bit 7`: `pack5` / Q27 FAULT) |
| `3` | 1 | `SEQ` (echoed) |
| `4..11` | 8 | `4 × int16` little-endian: `(gaze_x, gaze_y, blink_left, blink_right)` |
| `12..43` | 0 or 32 | Optional `32 × int8` penultimate `h2` vector for on-chip / Q12 calibration |
| last 2 | 2 | `CRC16-CCITT` (big-endian) |

## 3. Target Board Budgets (`16`-lane parallel ternary adder tree @ `25 MHz`)
| Board | LUTs (est.) | DSP | DRAM | BRAM (8-bit lanes) | Cycles/Frame | Latency |
|---|---:|---:|---:|---|---:|---:|
| **ALINX AX7203 (`XC7A200T`)** | ~`2,218` (`1.65%`) | `0` | `0 B` | `1 / 365 RAMB36E1` | `1,236` | **`49.44 µs`** |
| **Lattice `ECP5-85F`** | ~`3,216` (`3.85%`) | `0` | `0 B` | `2 / 208 DP16KD` | `1,236` | **`49.44 µs`** |

## 4. V1 Autonomous USB-HID Hardware Path (Kickstarter Target)
Replace the host UART cropper with an onboard DVP/MIPI IR sensor (`850 nm` dual-LED illumination) + streaming FPGA eye-ROI extractor + `USB-IF Eye Tracking HID` / standard USB Mouse HID endpoint — zero host drivers, zero host CPU usage, 100% physical video isolation.
