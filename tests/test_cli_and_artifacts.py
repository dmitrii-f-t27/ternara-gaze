from pathlib import Path

from ternara_gaze.cli import main as cli_main

ROOT = Path(__file__).resolve().parent.parent


def test_committed_cartridge_verify_and_demo_sim():
    weights = ROOT / "weights" / "ternara_gaze_v0.tgaze.bin"
    assert weights.exists()
    assert cli_main(["verify", "--weights", str(weights), "--samples", "20"]) == 0
    assert cli_main(["demo-sim", "--weights", str(weights)]) == 0


def test_cli_export_q27(tmp_path):
    weights = ROOT / "weights" / "ternara_gaze_v0.tgaze.bin"
    assert cli_main(["export-q27", "--weights", str(weights), "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "ternara_gaze_v0.q27").exists()
    assert (tmp_path / "ternara_gaze_pack5.hex").exists()
    assert (tmp_path / "fpga_budget.json").exists()
