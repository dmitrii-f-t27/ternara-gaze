# Ternara-Gaze

**Privacy-first, FPGA-native ternary AI eye-tracking mouse (`0 DSP`, `0 DRAM`, `< 4 KB` `pack5` weight cartridge, bit-exact Q27 integer oracle).**

Control your cursor with your eyes and click with a deliberate blink or wink — engineered from day one so the entire neural inference runs inside a single 8-bit BRAM block on an Artix-7 or Lattice ECP5 FPGA with **zero floating-point units, zero DSP multipliers, and zero raw video ever leaving the hardware boundary**.

> **Hardware Transparency Note:** On current FPGAs (Xilinx 7-Series / Lattice ECP5), Ternara-Gaze runs **ternary arithmetic (`W ∈ {-1, 0, +1}`) encoded on binary silicon** via `pack5` (5 trits per 8-bit byte, radix-243) — it is not a native physical 3-voltage-level ternary chip. The same `Q27` hardware IR targets FPGAs today and open-source ASIC flows (`Tiny Tapeout` / Sky130 / IHP) next.

---

## Why FPGA + Ternary AI for Eye Tracking?

- **Hardware-enforced privacy:** Only two `32×16` 4-bit grayscale eye ROIs (`512 bytes`) enter the inference core; on the autonomous FPGA build, the host OS only ever sees standard **USB-HID mouse/eye-tracker coordinates** — raw camera imagery never touches the operating system.
- **0 DSP, 0 DRAM, `< 4 KB` weights:** `18,560` ternary weights (`44.7%` natural zero sparsity) pack into **`3,712 bytes`** using strict 8-bit `pack5` lanes (`3^5 = 243 < 256`, zero 9-bit parity-lane hazards). Total cartridge `.tgaze.bin` with `int16` biases + header + CRC32 is **`3,944 bytes`** — fitting inside **1 `RAMB36E1`** on Artix-7 or **2 `EBR` blocks** on Lattice ECP5-85F.
- **Sub-50 µs core latency @ 25 MHz:** Multiplication-free ternary adder tree (`+`, `-`, skip only) completes a full frame in `1,236 cycles = 49.44 µs` (`> 20,000 FPS` compute ceiling).
- **13-point corner-first `Q12` calibration:** Pins the **4 true screen corners first** (plus center, edges, and inner quadrants) using 14 physically orthogonal integer features (sub-pixel pupil offsets + Q27 backbone + head position/yaw/pitch + mild quadratic terms) with per-target MAD outlier rejection and an on-screen visual target banner.
- **Non-invasive floating second cursor (Computer-Use style):** Renders an independent 60 FPS transparent click-through cyan reticle that follows your eyes while leaving your physical mouse/trackpad 100% free; deliberate blinks (`260–750 ms`) or left/right winks perform an atomic click at the reticle and immediately restore your physical pointer.

---

## Architecture

```text
[Camera / macOS AVFoundation+Vision]
        │  ONLY 2 × (16×32) 4-bit eye crops = 512 bytes (no raw face video stored)
        ▼
[Exact Integer Feature Prep]  mean = sum >> 9 (zero dividers), 2×2 pool -> 256 × int7
        ▼
[Ternary Backbone (Q27 / pack5, 0 DSP, 0 float, 0 multiplications)]
  256 ──► 64 (>>5, int7 ReLU) ──► 32 (>>4, int7 ReLU) ──► 4 (gaze_x, gaze_y, blink_L, blink_R)
                                    │
                                    ▼  32-dim int8 penultimate features
[Pure-Integer Q12 13-Point Corner-First Calibrator] ──► floating 2nd cursor (x_u16, y_u16)
        ▼
[Fixed-Point 1-Euro Filter + Safe Blink/Wink State Machine]
        ▼
[macOS CoreGraphics Mouse / Future Autonomous FPGA USB-HID]
```

---

## Verified v0 Numbers (Measured)

| Metric | Measured Result |
|---|---:|
| Ternary weights (`256 → 64 → 32 → 4`) | `18,560` trits (`44.67%` zeros) |
| Weight cartridge size (`.tgaze.bin`, `pack5` + biases + CRC32) | **`3,944 bytes`** |
| Validation gaze MAE (640 held-out frames, pure integer oracle) | **`0.112`** (~5.6% screen span) |
| Validation blink accuracy (Left / Right) | **`99.22%` / `99.53%`** |
| Bit-exactness (multiplication-free scalar oracle vs UART loopback) | **`0` max diff (`40/40`)** |
| `pack5` reserved-byte fault rejection (`243..255`) | **`13/13` rejected** |
| Target AX7203 (`XC7A200T`) budget @ 25 MHz | `0 DSP`, `0 DRAM`, `1/365 RAMB36E1`, **`49.44 µs/frame`** |
| Target Lattice `ECP5-85F` budget @ 25 MHz | `0 DSP`, `0 DRAM`, `2/208 EBR`, **`49.44 µs/frame`** |

---

## Quickstart (macOS, Zero OpenCV / Zero MediaPipe)

Requires Python 3.10+ (`numpy`, optional `torch` for retraining) and Apple Command Line Tools (`swiftc`).

### 1. Verify the bit-exact Q27 oracle, UART framing & FPGA export
```bash
python3 -m ternara_gaze.cli verify --weights weights/ternara_gaze_v0.tgaze.bin
python3 -m ternara_gaze.cli demo-sim --weights weights/ternara_gaze_v0.tgaze.bin
python3 -m ternara_gaze.cli export-q27 --weights weights/ternara_gaze_v0.tgaze.bin --out-dir build/q27_export
pytest -v
```

### 2. Build the native macOS camera + cursor bridge (`AVFoundation` + `Vision` + `CoreGraphics`)
```bash
bash macos/build_macos.sh
```

### 3. Calibrate (13 corner-first points, ~28 seconds) & run live second cursor
> Grant **Camera** and **Accessibility** permissions to your terminal app in *System Settings → Privacy & Security*.

```bash
# 13-point corner-first visual calibration (saves pure-integer Q12 v2 weights to weights/user_calibration.json)
python3 -m ternara_gaze.cli calibrate

# Preview mode (floating cyan second cursor only, clicks disabled)
python3 -m ternara_gaze.cli run

# Live floating second cursor + atomic clicks (deliberate blink/left wink = Left Click; right wink = Right Click)
# Touch your trackpad/mouse anytime to pause gaze control automatically, or press Ctrl+C to stop.
python3 -m ternara_gaze.cli run --live
```

### 4. Switch seamlessly to live FPGA UART backend
```bash
python3 -m ternara_gaze.cli run --backend fpga-uart --port /dev/tty.usbserial-XXXX --baudrate 921600 --live
```

---

## Retrain Cartridge from Scratch (~2 seconds on 2 CPU threads)

```bash
python3 -m ternara_gaze.cli train --output weights/ternara_gaze_v0.tgaze.bin --export-q27 build/q27_export
```

---

## Roadmap

- [x] **V0 (This Release):** Bit-exact 0-DSP ternary gaze+blink model, `pack5` radix-243 cartridge (`< 4 KB`), 13-point corner-first pure-integer `Q12` calibration, smooth 60 FPS floating second cursor with atomic clicks, framed UART protocol, clean-room Q27 hardware export, and native zero-OpenCV macOS bridge.
- [ ] **V0.5 (Live AX7203 FPGA):** Synthesize `ternara_gaze_core` with Vivado-free `q27rtl2` (`openXC7`), verify byte-exact board lockstep over UART, and demonstrate `< 125 ms` hot weight-cartridge swaps via `q27_bitpack`.
- [ ] **V1 (Autonomous Hardware Prototype — ECP5-85F / Artix-7):** Direct IR camera sensor (`OV2640`/`OV7670` + `850 nm` LEDs) → FPGA `Q27` inference → native **USB-IF Eye Tracking HID** output with zero host software required.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE). Clean-room Ternara codebase (`0%` third-party RTL/compiler code).
