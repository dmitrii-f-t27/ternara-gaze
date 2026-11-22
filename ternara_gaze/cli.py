"""Command-line interface for Ternara-Gaze (HD Eye-Lock + Blink-to-Advance + Floating 2nd Cursor)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibrator import (
    CORNER_FIRST_GRID,
    IntegerGazeCalibrator,
    build_cal_feature,
    dark_pupil_centroid_int,
    filter_target_outliers,
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

RU_TARGET_LABELS: Tuple[str, ...] = (
    "1/13  ЛЕВЫЙ ВЕРХНИЙ УГОЛ",
    "2/13  ПРАВЫЙ ВЕРХНИЙ УГОЛ",
    "3/13  ПРАВЫЙ НИЖНИЙ УГОЛ",
    "4/13  ЛЕВЫЙ НИЖНИЙ УГОЛ",
    "5/13  ЦЕНТР ЭКРАНА",
    "6/13  ВЕРХНИЙ КРАЙ",
    "7/13  ПРАВЫЙ КРАЙ",
    "8/13  НИЖНИЙ КРАЙ",
    "9/13  ЛЕВЫЙ КРАЙ",
    "10/13 ВЕРХ-ЛЕВО",
    "11/13 ВЕРХ-ПРАВО",
    "12/13 НИЗ-ПРАВО",
    "13/13 НИЗ-ЛЕВО",
)


def _speak_async(text: str) -> None:
    """Speak a Russian voice prompt on macOS without blocking the UI."""
    say_bin = shutil.which("say")
    if not say_bin:
        return
    try:
        subprocess.Popen(
            [say_bin, "-v", "Milena", "-r", "185", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _play_chime_async() -> None:
    """Play a crisp confirmation pop sound on macOS when a blink advances to the next dot."""
    afplay = shutil.which("afplay")
    sound = Path("/System/Library/Sounds/Pop.aiff")
    if not afplay or not sound.exists():
        return
    try:
        subprocess.Popen(
            [afplay, str(sound)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _extract_eye_kinematics(
    msg: Dict[str, object], left_u4: np.ndarray, right_u4: np.ndarray
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Extract (head_x, head_y, pup_lx, pup_ly, pup_rx, pup_ry, yaw, pitch) with pure-integer pupil fallback."""
    head_x = int(msg.get("nose_x", msg.get("head_x", 0)))
    head_y = int(msg.get("nose_y", msg.get("head_y", 0)))
    yaw = int(msg.get("yaw", 0))
    pitch = int(msg.get("pitch", 0))
    pup_lx = int(msg.get("pup_lx", 0))
    pup_ly = int(msg.get("pup_ly", 0))
    pup_rx = int(msg.get("pup_rx", 0))
    pup_ry = int(msg.get("pup_ry", 0))
    if pup_lx == 0 and pup_ly == 0:
        pup_lx, pup_ly = dark_pupil_centroid_int(left_u4)
    if pup_rx == 0 and pup_ry == 0:
        pup_rx, pup_ry = dark_pupil_centroid_int(right_u4)
    return head_x, head_y, pup_lx, pup_ly, pup_rx, pup_ry, yaw, pitch


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

    open_idx = [i for i in range(len(ds.targets)) if ds.targets[i, 2] < 0 and ds.targets[i, 3] < 0]
    cal_feats = []
    cal_tgts = []
    for i in open_idx[:36]:
        out = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        plx, ply = dark_pupil_centroid_int(ds.left_u4[i])
        prx, pry = dark_pupil_centroid_int(ds.right_u4[i])
        cal_feats.append(build_cal_feature(out.h2, out.gaze_x, out.gaze_y, 0, 0, plx, ply, prx, pry, 0, 0))
        gx, gy = float(ds.targets[i, 0]), float(ds.targets[i, 1])
        cal_tgts.append(((gx + 1.0) * 0.5, (gy + 1.0) * 0.5))
    cal = IntegerGazeCalibrator.fit(cal_feats, cal_tgts)
    ctrl = GazeCursorController()

    clicks = 0
    now_ms = 0
    for step_idx in range(25):
        i = open_idx[step_idx % len(open_idx)]
        out = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        plx, ply = dark_pupil_centroid_int(ds.left_u4[i])
        prx, pry = dark_pupil_centroid_int(ds.right_u4[i])
        bl, br = out.blink_left, out.blink_right
        if 10 <= step_idx <= 16:
            bl, br = 40, 40
        sx, sy = cal.predict_u16(out.h2, out.gaze_x, out.gaze_y, 0, 0, plx, ply, prx, pry, 0, 0)
        dec = ctrl.step(sx, sy, bl, br, now_ms)
        if dec.click != "none":
            clicks += 1
        now_ms += 50
    print(json.dumps({
        "demo": "synthetic_end_to_end",
        "calibrated_samples": len(cal_feats),
        "fit_mae_pct_x": cal.fit_mae_pct_x,
        "fit_mae_pct_y": cal.fit_mae_pct_y,
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

    print("=========================================================================", flush=True)
    print("  TERNARA-GAZE v3: ЗАПУСК ТЕСТА КАЛИБРОВКИ ГЛАЗ (HD 720p EYE-LOCK)", flush=True)
    print("=========================================================================", flush=True)
    print("  1. Убедись в окошке на экране, что камера четко зафиксировала оба глаза.", flush=True)
    print("  2. Смотри на зелёную точку по углам экрана, пока кольцо заполняется.", flush=True)
    print("  3. Когда кольцо станет ЗОЛОТЫМ — МОРГНИ, чтобы перейти к следующей точке!\n", flush=True)

    proc = subprocess.Popen(
        [str(bridge), "--overlay"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stdin is not None

    all_feats: List[np.ndarray] = []
    all_targets: List[Tuple[float, float]] = []
    all_weights: List[float] = []
    open_bl: List[int] = []
    open_br: List[int] = []
    open_ear_l: List[int] = []
    open_ear_r: List[int] = []
    blink_bl_peaks: List[int] = []
    blink_br_peaks: List[int] = []
    blink_ear_l_mins: List[int] = []
    blink_ear_r_mins: List[int] = []
    raw_total = 0

    try:
        # Phase 0: Voice + On-Screen Intro Card + Live HD Eye-Lock verification (4.0 s)
        _speak_async(
            "Запускаем тест калибровки. Смотрите на зелёную точку и слегка поворачивайте нос в её сторону. "
            "Когда кольцо станет золотым — моргните, чтобы перейти к следующей точке."
        )
        t_intro = time.monotonic()
        intro_dur = 4.2
        while True:
            elapsed = time.monotonic() - t_intro
            if elapsed >= intro_dur:
                break
            rem = max(1, int(np.ceil(intro_dur - elapsed)))
            proc.stdin.write(f"I {rem} Камера HD 720p фиксирует оба глаза — проверьте зелёные рамки\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise SystemExit("Camera bridge stopped unexpectedly.")
            msg = json.loads(line)
            el = int(msg.get("ear_l", 0))
            er = int(msg.get("ear_r", 0))
            if el > 80 and er > 80:
                open_ear_l.append(el)
                open_ear_r.append(er)

        # Phases 1..13: 4 Extreme Corners first, then center/edges/inner + Blink-to-Advance!
        settle_sec = 0.70
        capture_sec = 1.15
        blink_timeout_sec = 6.5

        for idx, (tx, ty, eng_label) in enumerate(CORNER_FIRST_GRID):
            label = RU_TARGET_LABELS[idx] if idx < len(RU_TARGET_LABELS) else eng_label
            tx_u16 = int(round(tx * 65535))
            ty_u16 = int(round(ty * 65535))
            print(f"  -> {label:28s} ({int(tx * 100):2d}%, {int(ty * 100):2d}%) ...", flush=True)
            proc.stdin.write(f"T {tx_u16} {ty_u16} 0 AIM {label}\n")
            proc.stdin.flush()

            # Stage A+B: Settle + steady open-eye fixation recording
            t_start = time.monotonic()
            point_feats: List[np.ndarray] = []
            point_ear_l: List[int] = []
            point_ear_r: List[int] = []
            point_bl: List[int] = []
            point_br: List[int] = []

            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= settle_sec + capture_sec:
                    break
                line = proc.stdout.readline()
                if not line:
                    raise SystemExit("Camera bridge stopped unexpectedly.")
                if elapsed < settle_sec:
                    proc.stdin.write(f"T {tx_u16} {ty_u16} 0 AIM {label}\n")
                    proc.stdin.flush()
                    continue
                pct = min(100, int((elapsed - settle_sec) / capture_sec * 100))
                proc.stdin.write(f"T {tx_u16} {ty_u16} {pct} REC {label}\n")
                proc.stdin.flush()

                msg = json.loads(line)
                left_u4, right_u4 = unpack_dual_eye_nibbles(bytes.fromhex(msg["eyes_hex"]))
                out = backend.infer(left_u4, right_u4, want_h2=True)
                hx, hy, plx, ply, prx, pry, yaw, pitch = _extract_eye_kinematics(msg, left_u4, right_u4)
                el = int(msg.get("ear_l", 0))
                er = int(msg.get("ear_r", 0))

                point_feats.append(
                    build_cal_feature(out.h2, out.gaze_x, out.gaze_y, hx, hy, plx, ply, prx, pry, yaw, pitch)
                )
                point_bl.append(out.blink_left)
                point_br.append(out.blink_right)
                if el > 0 and er > 0:
                    point_ear_l.append(el)
                    point_ear_r.append(er)
                raw_total += 1

            # Establish local open-eye baseline at THIS exact gaze angle before asking for blink
            local_ear_l = int(np.median(point_ear_l)) if point_ear_l else 240
            local_ear_r = int(np.median(point_ear_r)) if point_ear_r else 240
            local_bl = int(np.percentile(point_bl, 85)) if point_bl else 10
            local_br = int(np.percentile(point_br, 85)) if point_br else 10

            open_bl.extend(point_bl)
            open_br.extend(point_br)
            open_ear_l.extend(point_ear_l)
            open_ear_r.extend(point_ear_r)

            kept = filter_target_outliers(point_feats, keep_ratio=0.65)
            point_weight = 2.6 if idx < 4 else (1.4 if idx < 9 else 1.0)
            for f in kept:
                all_feats.append(f)
                all_targets.append((tx, ty))
                all_weights.append(point_weight)

            # Stage C: Turn dot GOLD and wait for user's real blink to advance!
            proc.stdin.write(f"T {tx_u16} {ty_u16} 100 BLINK {label}\n")
            proc.stdin.flush()
            t_wait = time.monotonic()
            closed_streak = 0
            blink_seen = False
            min_el_seen = local_ear_l
            min_er_seen = local_ear_r
            max_bl_seen = local_bl
            max_br_seen = local_br
            ear_cut_l = max(85, int(local_ear_l * 0.74))
            ear_cut_r = max(85, int(local_ear_r * 0.74))

            while True:
                if time.monotonic() - t_wait >= blink_timeout_sec:
                    print(
                        f"     captured {len(kept)} stable frames (auto-advanced after {blink_timeout_sec:.0f}s)",
                        flush=True,
                    )
                    break
                line = proc.stdout.readline()
                if not line:
                    raise SystemExit("Camera bridge stopped unexpectedly.")
                msg = json.loads(line)
                left_u4, right_u4 = unpack_dual_eye_nibbles(bytes.fromhex(msg["eyes_hex"]))
                out = backend.infer(left_u4, right_u4, want_h2=False)
                el = int(msg.get("ear_l", 0))
                er = int(msg.get("ear_r", 0))

                is_ear_blink = (el > 0 and er > 0 and el < ear_cut_l and er < ear_cut_r)
                is_q27_blink = (out.blink_left > local_bl + 16 and out.blink_right > local_br + 16)
                is_closed = is_ear_blink or is_q27_blink

                if is_closed:
                    closed_streak += 1
                    if el > 0:
                        min_el_seen = min(min_el_seen, el)
                    if er > 0:
                        min_er_seen = min(min_er_seen, er)
                    max_bl_seen = max(max_bl_seen, out.blink_left)
                    max_br_seen = max(max_br_seen, out.blink_right)
                    if closed_streak >= 2:
                        blink_seen = True
                else:
                    if blink_seen:
                        # Eyes reopened cleanly -> confirmed deliberate blink!
                        blink_ear_l_mins.append(min_el_seen)
                        blink_ear_r_mins.append(min_er_seen)
                        blink_bl_peaks.append(max_bl_seen)
                        blink_br_peaks.append(max_br_seen)
                        _play_chime_async()
                        print(
                            f"     captured {len(kept)} stable frames + BLINK confirmed "
                            f"(EAR {local_ear_l}->{min_el_seen} | {local_ear_r}->{min_er_seen})",
                            flush=True,
                        )
                        time.sleep(0.22)
                        break
                    closed_streak = 0
    finally:
        try:
            proc.stdin.write("Q\n")
            proc.stdin.flush()
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=3)

    if len(all_feats) < 16:
        raise SystemExit(f"Too few stable frames captured ({len(all_feats)}). Ensure good lighting and retry.")

    # Safe open-vs-blink thresholds: strictly separated from normal open eyes so cursor never false-freezes
    open_bl_p95 = int(np.percentile(open_bl, 95)) if open_bl else 10
    open_br_p95 = int(np.percentile(open_br, 95)) if open_br else 10
    if blink_bl_peaks:
        tl = max(open_bl_p95 + 22, int((open_bl_p95 + int(np.median(blink_bl_peaks))) // 2))
        tr = max(open_br_p95 + 22, int((open_br_p95 + int(np.median(blink_br_peaks))) // 2))
    else:
        tl = open_bl_p95 + 26
        tr = open_br_p95 + 26

    open_el_med = int(np.median(open_ear_l)) if open_ear_l else 260
    open_er_med = int(np.median(open_ear_r)) if open_ear_r else 260
    open_el_p10 = int(np.percentile(open_ear_l, 10)) if open_ear_l else 210
    open_er_p10 = int(np.percentile(open_ear_r, 10)) if open_ear_r else 210
    max_safe_el = min(open_el_p10 - 35, int(open_el_med * 0.70))
    max_safe_er = min(open_er_p10 - 35, int(open_er_med * 0.70))
    if blink_ear_l_mins:
        mid_el = int((open_el_p10 + int(np.median(blink_ear_l_mins))) // 2)
        mid_er = int((open_er_p10 + int(np.median(blink_ear_r_mins))) // 2)
        ear_tl = max(65, min(max_safe_el, mid_el))
        ear_tr = max(65, min(max_safe_er, mid_er))
    else:
        ear_tl = max(65, max_safe_el)
        ear_tr = max(65, max_safe_er)

    cal = IntegerGazeCalibrator.fit(
        all_feats,
        all_targets,
        ridge_lambda=500.0,
        blink_thresh_l=tl,
        blink_thresh_r=tr,
        ear_thresh_l=ear_tl,
        ear_thresh_r=ear_tr,
        sample_weights=all_weights,
    )
    cal.save(args.cal_out)
    _speak_async("Калибровка завершена. Включаю плавный второй курсор.")
    print(
        f"\nSaved Q12 v3 Nose+Eye HD Lock calibration -> {args.cal_out}\n"
        f"  Stable samples : {len(all_feats)} / {raw_total} raw frames across 13 targets\n"
        f"  Fit accuracy   : MAE X = {cal.fit_mae_pct_x:.2f}% | MAE Y = {cal.fit_mae_pct_y:.2f}% of screen\n"
        f"  Blink thresh   : Q27=({tl},{tr}) | HD EAR=({ear_tl},{ear_tr}) (from {len(blink_ear_l_mins)} real blinks)",
        flush=True,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bridge = _ensure_bridge(Path(args.bridge))
    cart = TernaryCartridge.load(args.weights) if args.backend == "oracle" else None
    backend = _make_backend(args, cart)

    cal_path = Path(args.calibration)
    if cal_path.exists():
        try:
            cal = IntegerGazeCalibrator.load(cal_path)
            print(
                f"Loaded Q12 v3 Nose+Eye calibration: {cal_path} "
                f"(fit MAE X={cal.fit_mae_pct_x:.2f}%, Y={cal.fit_mae_pct_y:.2f}%, "
                f"EAR=({cal.ear_thresh_l},{cal.ear_thresh_r}))",
                flush=True,
            )
        except ValueError as exc:
            print(f"Warning: {exc} Falling back to default map.", flush=True)
            cal = IntegerGazeCalibrator.identity_default()
    else:
        cal = IntegerGazeCalibrator.identity_default()
        print("No user calibration found; using default map (run `calibrate` first for best accuracy).", flush=True)

    ctrl = GazeCursorController(
        FilterConfig(
            blink_threshold_l=cal.blink_thresh_l,
            blink_threshold_r=cal.blink_thresh_r,
            ear_threshold_l=cal.ear_thresh_l,
            ear_threshold_r=cal.ear_thresh_r,
        )
    )
    cmd = [str(bridge), "--overlay"]
    if args.live:
        cmd.append("--live")
        print("SECOND VIRTUAL CURSOR + FIXATION-LOCK + ATOMIC CLICK ACTIVE:", flush=True)
        print("  - Your main physical mouse/trackpad stays 100% free and untouched!", flush=True)
        print("  - Fixation-lock eliminates tremor when looking at a button; glides fast on eye jumps.", flush=True)
        print("  - Deliberate blink (0.24-0.75s) or left wink = Left Click; right wink = Right Click.", flush=True)
        print("  - Press Ctrl+C in this window anytime to stop.", flush=True)
    else:
        print("OVERLAY PREVIEW MODE (floating cyan reticle only, clicks disabled; pass --live to enable clicks).", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stdin is not None

    frames = 0
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            now_ms = int(msg["ts_ms"])

            left_u4, right_u4 = unpack_dual_eye_nibbles(bytes.fromhex(msg["eyes_hex"]))
            out = backend.infer(left_u4, right_u4, want_h2=True)
            hx, hy, plx, ply, prx, pry, yaw, pitch = _extract_eye_kinematics(msg, left_u4, right_u4)
            el = int(msg["ear_l"]) if "ear_l" in msg and int(msg["ear_l"]) > 0 else None
            er = int(msg["ear_r"]) if "ear_r" in msg and int(msg["ear_r"]) > 0 else None

            sx, sy = cal.predict_u16(out.h2, out.gaze_x, out.gaze_y, hx, hy, plx, ply, prx, pry, yaw, pitch)
            dec = ctrl.step(sx, sy, out.blink_left, out.blink_right, now_ms, ear_l=el, ear_r=er)

            proc.stdin.write(f"G {dec.x_u16} {dec.y_u16} {dec.hold_progress_pct} {1 if dec.locked else 0}\n")
            if dec.click == "left_click":
                proc.stdin.write(f"L {dec.x_u16} {dec.y_u16}\n")
            elif dec.click == "right_click":
                proc.stdin.write(f"R {dec.x_u16} {dec.y_u16}\n")
            proc.stdin.flush()

            frames += 1
            if frames % 15 == 0 or dec.click != "none":
                print(
                    f"frame={frames:5d} nose=({hx:+4d},{hy:+4d}) pup=({plx:+4d},{ply:+4d}) "
                    f"pose=({yaw:+4d},{pitch:+4d}) ear=({el or 0:3d},{er or 0:3d}) "
                    f"reticle=({dec.x_u16:5d},{dec.y_u16:5d}) lock={int(dec.locked)} click={dec.click}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped cleanly.", flush=True)
    finally:
        try:
            proc.stdin.write("Q\n")
            proc.stdin.flush()
        except Exception:
            pass
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

    p_cal = sub.add_parser("calibrate", help="Run 13-point HD Eye-Lock corner-first calibration with blink-to-advance")
    p_cal.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_cal.add_argument("--cal-out", default=str(DEFAULT_CAL))
    p_cal.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    p_cal.add_argument("--backend", choices=["oracle", "fpga-uart"], default="oracle")
    p_cal.add_argument("--port", default="")
    p_cal.add_argument("--baudrate", type=int, default=921600)
    p_cal.set_defaults(func=cmd_calibrate)

    p_run = sub.add_parser("run", help="Run smooth fixation-lock floating second-cursor gaze tracking (macOS)")
    p_run.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_run.add_argument("--calibration", default=str(DEFAULT_CAL))
    p_run.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    p_run.add_argument("--backend", choices=["oracle", "fpga-uart"], default="oracle")
    p_run.add_argument("--port", default="")
    p_run.add_argument("--baudrate", type=int, default=921600)
    p_run.add_argument("--live", action="store_true", help="Enable atomic Computer-Use clicks at the second cursor")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
