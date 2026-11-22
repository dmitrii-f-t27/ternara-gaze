"""Command-line interface for Ternara-Gaze."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .calibrator import (
    NINE_POINT_GRID_NORM,
    IntegerGazeCalibrator,
    build_cal_feature,
)
from .cursor_filter import FilterConfig, GazeCursorController
from .oracle import oracle_forward_pure, unpack_dual_eye_nibbles
from .pack5 import Pack5FaultError, TernaryCartridge, pack5_decode
from .q27_export import estimate_fpga_budgets, export_q27_bundle
from .synth_data import generate_dataset
from .train_qat import train_and_export
from .uart_bridge import FpgaUartBackend, GazeInferenceBackend, OracleLoopbackBackend

DEFAULT_WEIGHTS = Path("weights/ternara_gaze_v0.tgaze.bin")
DEFAULT_CAL = Path("weights/user_calibration.json")
DEFAULT_BRIDGE = Path("build/gaze_bridge")


def _make_backend(args: argparse.Namespace, cartridge: Optional[TernaryCartridge]) -> GazeInferenceBackend:
    if args.backend == "fpga-uart":
        if not args.port:
            raise SystemExit("--port is required for --backend fpga-uart")
        return FpgaUartBackend(port=args.port, baudrate=args.baudrate)
    if cartridge is None:
        raise SystemExit("Cartridge required for oracle backend")
    return OracleLoopbackBackend(cartridge)


def cmd_train(args: argparse.Namespace) -> int:
    cart, report = train_and_export(
        output_path=args.output,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        seed=args.seed,
        max_threads=args.threads,
    )
    print(json.dumps(report.__dict__, indent=2))
    if args.export_q27:
        paths = export_q27_bundle(cart, args.export_q27)
        print("Exported Q27 bundle:", json.dumps(paths, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cart = TernaryCartridge.load(args.weights)
    backend = OracleLoopbackBackend(cart)
    ds = generate_dataset(args.samples, seed=99991)

    max_diff = 0
    t0 = time.perf_counter()
    for i in range(args.samples):
        pure = oracle_forward_pure(cart, ds.left_u4[i], ds.right_u4[i])
        loop = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        diffs = [
            abs(pure.gaze_x - loop.gaze_x),
            abs(pure.gaze_y - loop.gaze_y),
            abs(pure.blink_left - loop.blink_left),
            abs(pure.blink_right - loop.blink_right),
            int(np.max(np.abs(pure.h2.astype(np.int32) - loop.h2.astype(np.int32)))),
        ]
        max_diff = max(max_diff, max(diffs))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Fault check: reserved pack5 bytes 243..255 must raise Pack5FaultError
    faults_rejected = 0
    for bad_byte in range(243, 256):
        try:
            pack5_decode(bytes([bad_byte]), 5)
        except Pack5FaultError:
            faults_rejected += 1

    budgets = estimate_fpga_budgets(cart)
    summary = {
        "weights": str(args.weights),
        "sha256": cart.sha256(),
        "cartridge_bytes": len(cart.to_bytes()),
        "total_trits": cart.total_trits,
        "zero_fraction": round(cart.zero_fraction, 4),
        "samples_checked": args.samples,
        "bit_exact_pure_vs_uart_loopback_max_diff": max_diff,
        "pack5_reserved_bytes_rejected": f"{faults_rejected}/13",
        "avg_pure_plus_loopback_ms": round(elapsed_ms / max(1, args.samples), 3),
        "fpga_ax7203_latency_us": budgets["ax7203_xc7a200t"].latency_us,
        "fpga_ecp5_latency_us": budgets["ecp5_85f"].latency_us,
        "dsp_used": 0,
        "status": "PASS" if (max_diff == 0 and faults_rejected == 13) else "FAIL",
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def cmd_export_q27(args: argparse.Namespace) -> int:
    cart = TernaryCartridge.load(args.weights)
    paths = export_q27_bundle(cart, args.out_dir)
    budgets = estimate_fpga_budgets(cart)
    print(json.dumps({"exported": paths, "budgets": {k: v.__dict__ for k, v in budgets.items()}}, indent=2))
    return 0


def cmd_demo_sim(args: argparse.Namespace) -> int:
    cart = TernaryCartridge.load(args.weights)
    backend = OracleLoopbackBackend(cart)
    ds = generate_dataset(60, seed=4242)

    # Fit 9-point style integer calibrator on synthetic samples
    open_idx = [i for i in range(len(ds.targets)) if ds.targets[i, 2] < 0 and ds.targets[i, 3] < 0]
    cal_feats = []
    cal_tgts = []
    for i in open_idx[:36]:
        out = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        cal_feats.append(build_cal_feature(out.h2, out.gaze_x, out.gaze_y, 0, 0))
        gx, gy = float(ds.targets[i, 0]), float(ds.targets[i, 1])
        cal_tgts.append(((gx + 1.0) * 0.5, (gy + 1.0) * 0.5))
    cal = IntegerGazeCalibrator.fit(cal_feats, cal_tgts)
    ctrl = GazeCursorController()

    # Simulate fixation + deliberate 350ms blink click
    clicks = 0
    now_ms = 0
    for step_idx in range(25):
        i = open_idx[step_idx % len(open_idx)]
        out = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        bl, br = out.blink_left, out.blink_right
        if 10 <= step_idx <= 16:
            bl, br = 40, 40  # simulated 350 ms deliberate blink
        sx, sy = cal.predict_u16(out.h2, out.gaze_x, out.gaze_y, 0, 0)
        dec = ctrl.step(sx, sy, bl, br, now_ms)
        if dec.click != "none":
            clicks += 1
        now_ms += 50
    print(json.dumps({
        "demo": "synthetic_end_to_end",
        "calibrated_samples": len(cal_feats),
        "frames_simulated": 25,
        "final_cursor_u16": [dec.x_u16, dec.y_u16],
        "deliberate_blink_clicks_detected": clicks,
        "status": "PASS" if clicks == 1 else "CHECK",
    }, indent=2))
    return 0


def _ensure_bridge(bridge_path: Path) -> Path:
    if bridge_path.exists():
        return bridge_path
    build_script = Path("macos/build_macos.sh")
    if build_script.exists():
        subprocess.run(["bash", str(build_script)], check=True)
        if bridge_path.exists():
            return bridge_path
    raise SystemExit(f"macOS gaze bridge not found at {bridge_path}. Run: bash macos/build_macos.sh")


def cmd_calibrate(args: argparse.Namespace) -> int:
    bridge = _ensure_bridge(Path(args.bridge))
    cart = TernaryCartridge.load(args.weights) if args.backend == "oracle" else None
    backend = _make_backend(args, cart)

    labels = [
        "TOP-LEFT", "TOP-CENTER", "TOP-RIGHT",
        "MIDDLE-LEFT", "CENTER", "MIDDLE-RIGHT",
        "BOTTOM-LEFT", "BOTTOM-CENTER", "BOTTOM-RIGHT",
    ]
    print("Ternara-Gaze 9-Point Calibration (zero raw video saved)")
    print("Keep your head comfortable and look at each screen region as prompted.\n")

    proc = subprocess.Popen(
        [str(bridge)],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    feats: List[np.ndarray] = []
    targets: List[Tuple[float, float]] = []

    try:
        for (tx, ty), name in zip(NINE_POINT_GRID_NORM, labels):
            print(f"  -> Look at {name:14s} ({int(tx*100):2d}%, {int(ty*100):2d}%) — hold steady...", flush=True)
            # Warmup 0.6s then collect frames for 1.2s
            t_start = time.monotonic()
            collected = 0
            while time.monotonic() - t_start < 1.8:
                line = proc.stdout.readline()
                if not line:
                    raise SystemExit("Camera bridge stopped unexpectedly. Check macOS Camera permission.")
                if time.monotonic() - t_start < 0.6:
                    continue
                msg = json.loads(line)
                left_u4, right_u4 = unpack_dual_eye_nibbles(bytes.fromhex(msg["eyes_hex"]))
                out = backend.infer(left_u4, right_u4, want_h2=True)
                if out.blink_left <= 12 and out.blink_right <= 12:
                    feats.append(build_cal_feature(out.h2, out.gaze_x, out.gaze_y, msg["head_x"], msg["head_y"]))
                    targets.append((tx, ty))
                    collected += 1
            print(f"     captured {collected} open-eye frames", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=3)

    if len(feats) < 12:
        raise SystemExit(f"Too few open-eye frames captured ({len(feats)}). Ensure good lighting and retry.")

    cal = IntegerGazeCalibrator.fit(feats, targets)
    cal.save(args.cal_out)
    print(f"\nSaved pure-integer Q12 calibration ({len(feats)} samples) -> {args.cal_out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bridge = _ensure_bridge(Path(args.bridge))
    cart = TernaryCartridge.load(args.weights) if args.backend == "oracle" else None
    backend = _make_backend(args, cart)

    cal_path = Path(args.calibration)
    if cal_path.exists():
        cal = IntegerGazeCalibrator.load(cal_path)
        print(f"Loaded calibration: {cal_path}")
    else:
        cal = IntegerGazeCalibrator.identity_default()
        print("No user calibration found; using default identity map (run `ternara-gaze calibrate` for best accuracy).")

    ctrl = GazeCursorController(FilterConfig())
    cmd = [str(bridge)]
    if args.live:
        cmd.append("--live")
        print("LIVE MOUSE MODE ACTIVE: deliberate 300-600ms blink or left wink = Left Click; right wink = Right Click.")
        print("SAFETY: Touch your trackpad/mouse anytime to pause gaze control for 1.5s, or press Ctrl+C to stop.")
    else:
        print("DRY-RUN MODE (pass --live to move the actual macOS cursor). Press Ctrl+C to stop.")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if args.live else None,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    last_sent_pos: Optional[Tuple[int, int]] = None
    frames = 0
    try:
        for line in proc.stdout:
            msg = json.loads(line)
            now_ms = int(msg["ts_ms"])
            mx, my = int(msg["mx_u16"]), int(msg["my_u16"])

            # Detect physical mouse override if cursor moved >2% away from our last commanded position
            if args.live and last_sent_pos is not None:
                if abs(mx - last_sent_pos[0]) + abs(my - last_sent_pos[1]) > 1400:
                    ctrl.notify_physical_mouse_moved(now_ms)
                    last_sent_pos = (mx, my)

            left_u4, right_u4 = unpack_dual_eye_nibbles(bytes.fromhex(msg["eyes_hex"]))
            out = backend.infer(left_u4, right_u4, want_h2=True)
            sx, sy = cal.predict_u16(out.h2, out.gaze_x, out.gaze_y, msg["head_x"], msg["head_y"])
            dec = ctrl.step(sx, sy, out.blink_left, out.blink_right, now_ms)

            if args.live and proc.stdin is not None and not dec.paused_by_user:
                if not dec.frozen:
                    proc.stdin.write(f"M {dec.x_u16} {dec.y_u16}\n")
                    last_sent_pos = (dec.x_u16, dec.y_u16)
                if dec.click == "left_click":
                    proc.stdin.write("L\n")
                elif dec.click == "right_click":
                    proc.stdin.write("R\n")
                proc.stdin.flush()

            frames += 1
            if frames % 10 == 0 or dec.click != "none":
                print(
                    f"frame={frames:5d} raw=({out.gaze_x:+4d},{out.gaze_y:+4d}) "
                    f"blink=({out.blink_left:+4d},{out.blink_right:+4d}) "
                    f"cursor=({dec.x_u16:5d},{dec.y_u16:5d}) click={dec.click} "
                    f"paused={dec.paused_by_user}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped cleanly.")
    finally:
        proc.terminate()
        proc.wait(timeout=3)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ternara-gaze", description="Privacy-first FPGA-native ternary eye-tracking mouse")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="Train ternary QAT model and export pack5 .tgaze.bin cartridge")
    p_tr.add_argument("--output", default=str(DEFAULT_WEIGHTS))
    p_tr.add_argument("--n-train", type=int, default=3200)
    p_tr.add_argument("--n-val", type=int, default=640)
    p_tr.add_argument("--epochs", type=int, default=35)
    p_tr.add_argument("--seed", type=int, default=20261122)
    p_tr.add_argument("--threads", type=int, default=2)
    p_tr.add_argument("--export-q27", default="build/q27_export")
    p_tr.set_defaults(func=cmd_train)

    p_ver = sub.add_parser("verify", help="Verify bit-exact Q27 oracle, UART framing, and pack5 fault rejection")
    p_ver.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_ver.add_argument("--samples", type=int, default=40)
    p_ver.set_defaults(func=cmd_verify)

    p_exp = sub.add_parser("export-q27", help="Export clean-room Q27 source, pack5 hex ROM, and FPGA budget")
    p_exp.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_exp.add_argument("--out-dir", default="build/q27_export")
    p_exp.set_defaults(func=cmd_export_q27)

    p_sim = sub.add_parser("demo-sim", help="Run camera-free synthetic end-to-end calibration & blink-click demo")
    p_sim.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_sim.set_defaults(func=cmd_demo_sim)

    p_cal = sub.add_parser("calibrate", help="Run 9-point webcam calibration (macOS)")
    p_cal.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_cal.add_argument("--cal-out", default=str(DEFAULT_CAL))
    p_cal.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    p_cal.add_argument("--backend", choices=["oracle", "fpga-uart"], default="oracle")
    p_cal.add_argument("--port", default="")
    p_cal.add_argument("--baudrate", type=int, default=921600)
    p_cal.set_defaults(func=cmd_calibrate)

    p_run = sub.add_parser("run", help="Run live gaze-to-mouse tracking (macOS)")
    p_run.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_run.add_argument("--calibration", default=str(DEFAULT_CAL))
    p_run.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    p_run.add_argument("--backend", choices=["oracle", "fpga-uart"], default="oracle")
    p_run.add_argument("--port", default="")
    p_run.add_argument("--baudrate", type=int, default=921600)
    p_run.add_argument("--live", action="store_true", help="Actually move the macOS cursor and emit clicks")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
