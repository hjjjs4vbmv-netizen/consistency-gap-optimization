"""Regression test for the post-calibration bounded summary."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summary_preserves_pass_without_reopening_old_factorial(tmp_path: Path):
    receipt = {
        "verdict": "PASS_CALIBRATED",
        "oracle": {"finite": True},
        "source_preserved": True,
        "assets_preserved": True,
        "plateaus": [{"epsilons": [0.004, 0.002, 0.001]}],
        "protocol": {"claim_ceiling": "one-cell calibration only"},
        "rows": [],
    }
    for index, epsilon in enumerate((0.06, 0.03, 0.015, 0.008, 0.004)):
        error = 0.2 / (index + 1)
        receipt["rows"].append({
            "epsilon": epsilon,
            "finite": True,
            "source_preserved": True,
            "tangent_vs_oracle": {
                "relative_error": error, "cosine": 0.99,
                "norm_ratio": 1.0},
            "action_vs_oracle": {
                "relative_error": error / 2, "cosine": 0.995,
                "norm_ratio": 1.0},
            "parameter_resolution": {
                "branch_distinct_fraction": 1.0,
                "realized_direction_relative_error": 1e-5,
                "realized_direction_cosine": 1.0},
        })
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "analysis.jvp_harness_calibration.summarize",
        "--receipt", str(receipt_path), "--out", str(tmp_path),
    ], check=True)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "PASS_CALIBRATED"
    assert summary["old_factorial_reopened"] is False
    assert "one-cell" in summary["claim_ceiling"]
