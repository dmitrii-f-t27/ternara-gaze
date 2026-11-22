import json

import numpy as np
import pytest

from ternara_gaze.oracle import oracle_forward_pure
from ternara_gaze.pack5 import TernaryCartridge
from ternara_gaze.q27_export import estimate_fpga_budgets, export_q27_bundle
from ternara_gaze.synth_data import generate_dataset
from ternara_gaze.uart_bridge import (
    OracleLoopbackBackend,
    decode_rx_frame,
    decode_tx_frame,
    encode_tx_frame,
)


def _dummy_cart() -> TernaryCartridge:
    rng = np.random.default_rng(11)
    return TernaryCartridge(
        w1=rng.integers(-1, 2, size=(64, 256), dtype=np.int8),
        b1=rng.integers(-20, 20, size=64, dtype=np.int16),
        s1=5,
        w2=rng.integers(-1, 2, size=(32, 64), dtype=np.int8),
        b2=rng.integers(-20, 20, size=32, dtype=np.int16),
        s2=4,
        w3=rng.integers(-1, 2, size=(4, 32), dtype=np.int8),
        b3=rng.integers(-20, 20, size=4, dtype=np.int16),
    )


def test_uart_framing_and_crc_faults():
    cart = _dummy_cart()
    backend = OracleLoopbackBackend(cart)
    ds = generate_dataset(5, seed=22)

    for i in range(5):
        pure = oracle_forward_pure(cart, ds.left_u4[i], ds.right_u4[i])
        out = backend.infer(ds.left_u4[i], ds.right_u4[i], want_h2=True)
        assert (pure.gaze_x, pure.gaze_y, pure.blink_left, pure.blink_right) == (
            out.gaze_x, out.gaze_y, out.blink_left, out.blink_right
        )
        assert np.array_equal(pure.h2, out.h2)

    tx = bytearray(encode_tx_frame(ds.left_u4[0], ds.right_u4[0], seq=3))
    assert len(tx) == 518
    tx[100] ^= 0xFF
    with pytest.raises(ValueError, match="CRC16"):
        decode_tx_frame(bytes(tx))

    with pytest.raises(ValueError):
        decode_rx_frame(b"short")


def test_q27_export_and_fpga_budget(tmp_path):
    cart = _dummy_cart()
    paths = export_q27_bundle(cart, tmp_path)
    q27_text = (tmp_path / "ternara_gaze_v0.q27").read_text(encoding="utf-8")
    assert "module ternara_gaze_core" in q27_text
    assert "0 DSP slices" in q27_text

    hex_lines = (tmp_path / "ternara_gaze_pack5.hex").read_text(encoding="utf-8").strip().splitlines()
    assert len(hex_lines) == 3712
    assert all(0 <= int(line, 16) <= 242 for line in hex_lines)

    budgets = estimate_fpga_budgets(cart)
    assert budgets["ax7203_xc7a200t"].dsp_used == 0
    assert budgets["ax7203_xc7a200t"].dram_bytes_used == 0
    assert budgets["ax7203_xc7a200t"].bram_blocks_used == 1
    assert budgets["ax7203_xc7a200t"].latency_us < 100.0
    assert budgets["ecp5_85f"].dsp_used == 0
    assert budgets["ecp5_85f"].bram_blocks_used == 2

    budget_json = json.loads((tmp_path / "fpga_budget.json").read_text(encoding="utf-8"))
    assert "ax7203_xc7a200t" in budget_json and "ecp5_85f" in budget_json
