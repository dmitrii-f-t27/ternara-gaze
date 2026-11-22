import numpy as np

from ternara_gaze.calibrator import IntegerGazeCalibrator, build_cal_feature
from ternara_gaze.cursor_filter import FilterConfig, GazeCursorController
from ternara_gaze.oracle import (
    extract_features_int,
    oracle_forward_fast,
    oracle_forward_features_batch,
    oracle_forward_pure,
    pack_dual_eye_nibbles,
    unpack_dual_eye_nibbles,
)
from ternara_gaze.pack5 import TernaryCartridge
from ternara_gaze.synth_data import generate_dataset


def test_nibble_pack_and_oracle_bit_exactness():
    rng = np.random.default_rng(42)
    cart = TernaryCartridge(
        w1=rng.integers(-1, 2, size=(64, 256), dtype=np.int8),
        b1=rng.integers(-50, 50, size=64, dtype=np.int16),
        s1=5,
        w2=rng.integers(-1, 2, size=(32, 64), dtype=np.int8),
        b2=rng.integers(-50, 50, size=32, dtype=np.int16),
        s2=4,
        w3=rng.integers(-1, 2, size=(4, 32), dtype=np.int8),
        b3=rng.integers(-50, 50, size=4, dtype=np.int16),
    )
    ds = generate_dataset(12, seed=101)
    batch_out, batch_h2 = oracle_forward_features_batch(cart, ds.features)

    for i in range(12):
        packed = pack_dual_eye_nibbles(ds.left_u4[i], ds.right_u4[i])
        assert len(packed) == 512
        l_dec, r_dec = unpack_dual_eye_nibbles(packed)
        assert np.array_equal(l_dec, ds.left_u4[i])
        assert np.array_equal(r_dec, ds.right_u4[i])

        pure = oracle_forward_pure(cart, l_dec, r_dec)
        fast = oracle_forward_fast(cart, l_dec, r_dec)
        assert pure.gaze_x == fast.gaze_x == int(batch_out[i, 0])
        assert pure.gaze_y == fast.gaze_y == int(batch_out[i, 1])
        assert pure.blink_left == fast.blink_left == int(batch_out[i, 2])
        assert pure.blink_right == fast.blink_right == int(batch_out[i, 3])
        assert np.array_equal(pure.h2, fast.h2)
        assert np.array_equal(pure.h2, batch_h2[i])


def test_mean_centering_invariance():
    # Shifting brightness uniformly by +3 (without clipping) keeps contrast identical
    base_l = np.full((16, 32), 4, dtype=np.uint8)
    base_l[4:12, 10:22] = 8
    base_r = base_l.copy()
    f1 = extract_features_int(base_l, base_r)
    f2 = extract_features_int(base_l + 3, base_r + 3)
    assert np.array_equal(f1, f2)


def test_integer_calibrator_fit_and_roundtrip(tmp_path):
    rng = np.random.default_rng(77)
    feats = []
    targets = []
    for _ in range(45):
        gx = int(rng.integers(-100, 101))
        gy = int(rng.integers(-100, 101))
        h2 = rng.integers(0, 64, size=32, dtype=np.int8)
        feats.append(build_cal_feature(h2, gx, gy, 0, 0))
        targets.append(((gx + 100) / 200.0, (gy + 100) / 200.0))

    cal = IntegerGazeCalibrator.fit(feats, targets, ridge_lambda=100.0)
    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)
    reloaded = IntegerGazeCalibrator.load(cal_path)

    sx, sy = reloaded.predict_u16(np.zeros(32, dtype=np.int8), 0, 0, 0, 0)
    assert 24000 <= sx <= 41000
    assert 24000 <= sy <= 41000


def test_cursor_controller_blink_and_override_safety():
    ctrl = GazeCursorController(FilterConfig())

    # Initial placement
    d0 = ctrl.step(30000, 30000, -50, -50, now_ms=0)
    assert d0.click == "none"

    # Fast reflex blink (120 ms) -> MUST NOT click
    ctrl.step(30000, 30000, 40, 40, now_ms=100)
    d_reflex = ctrl.step(30000, 30000, -50, -50, now_ms=220)
    assert d_reflex.click == "none"

    # Deliberate blink (360 ms) -> MUST trigger left_click and freeze during closure
    d_close = ctrl.step(55000, 55000, 40, 40, now_ms=400)
    assert d_close.frozen is True
    assert d_close.x_u16 == 30000  # frozen while blinking!
    d_open = ctrl.step(30000, 30000, -50, -50, now_ms=760)
    assert d_open.click == "left_click"

    # Eyes held closed > 900 ms (resting) -> MUST NOT click
    ctrl.step(30000, 30000, 40, 40, now_ms=2000)
    d_rest = ctrl.step(30000, 30000, -50, -50, now_ms=3000)
    assert d_rest.click == "none"

    # Right wink (300 ms) -> right_click
    ctrl.step(30000, 30000, -50, 40, now_ms=4000)
    d_rwink = ctrl.step(30000, 30000, -50, -50, now_ms=4300)
    assert d_rwink.click == "right_click"

    # Physical mouse override pauses gaze control
    ctrl.notify_physical_mouse_moved(now_ms=5000)
    d_paused = ctrl.step(60000, 60000, -50, -50, now_ms=5500)
    assert d_paused.paused_by_user is True
    assert d_paused.x_u16 == 30000
