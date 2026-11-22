import numpy as np

from ternara_gaze.calibrator import (
    CORNER_FIRST_GRID,
    IntegerGazeCalibrator,
    build_cal_feature,
    dark_pupil_centroid_int,
    filter_target_outliers,
)
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


def test_dark_pupil_centroid_and_outlier_filter():
    crop_left = np.full((16, 32), 12, dtype=np.uint8)
    crop_left[6:10, 7:11] = 1  # dark pupil on left side
    crop_right = np.full((16, 32), 12, dtype=np.uint8)
    crop_right[6:10, 21:25] = 1  # dark pupil on right side

    px_l, _ = dark_pupil_centroid_int(crop_left)
    px_r, _ = dark_pupil_centroid_int(crop_right)
    assert px_l < -30
    assert px_r > 30

    cluster = [np.array([10, 10], dtype=np.int32) for _ in range(8)]
    cluster.extend([np.array([120, -110], dtype=np.int32), np.array([-115, 115], dtype=np.int32)])
    kept = filter_target_outliers(cluster, keep_ratio=0.7)
    assert len(kept) == 7
    assert all(int(v[0]) == 10 and int(v[1]) == 10 for v in kept)


def test_integer_calibrator_corner_first_fit_and_roundtrip(tmp_path):
    assert len(CORNER_FIRST_GRID) == 13
    # First 4 targets must be the 4 extreme corners
    for tx, ty, _ in CORNER_FIRST_GRID[:4]:
        assert tx <= 0.05 or tx >= 0.95
        assert ty <= 0.06 or ty >= 0.93

    rng = np.random.default_rng(77)
    feats = []
    targets = []
    weights = []
    for idx, (tx, ty, _) in enumerate(CORNER_FIRST_GRID):
        for _ in range(6):
            px = int((tx - 0.5) * 180 + rng.integers(-3, 4))
            py = int((ty - 0.5) * 180 + rng.integers(-3, 4))
            gx = int((tx - 0.5) * 160 + rng.integers(-4, 5))
            gy = int((ty - 0.5) * 160 + rng.integers(-4, 5))
            feats.append(build_cal_feature(None, gx, gy, 0, 0, px, py, px, py, 0, 0))
            targets.append((tx, ty))
            weights.append(2.5 if idx < 4 else 1.0)

    cal = IntegerGazeCalibrator.fit(feats, targets, ridge_lambda=100.0, sample_weights=weights)
    assert cal.fit_mae_pct_x < 3.0
    assert cal.fit_mae_pct_y < 3.0

    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)
    reloaded = IntegerGazeCalibrator.load(cal_path)

    # Center check
    sx, sy = reloaded.predict_u16(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert 29000 <= sx <= 36500
    assert 29000 <= sy <= 36500

    # Extreme corners check (top-left & bottom-right)
    tl_x, tl_y = reloaded.predict_u16(None, -74, -72, 0, 0, -83, -81, -83, -81, 0, 0)
    br_x, br_y = reloaded.predict_u16(None, 74, 70, 0, 0, 83, 79, 83, 79, 0, 0)
    assert tl_x < 6000 and tl_y < 6500
    assert br_x > 59000 and br_y > 58500


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
