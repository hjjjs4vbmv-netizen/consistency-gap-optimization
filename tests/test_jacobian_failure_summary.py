"""NO-GO reporting for a failed preregistered correctness stage."""

from __future__ import annotations

import argparse
import json

from analysis.jacobian_failure_factorial import summarize


def test_failed_correctness_gate_stops_formal_without_missing_cell_claim(
        tmp_path, monkeypatch):
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    cell = {
        "arm": "A",
        "batch_id": 2026082601,
        "direction_id": 2026082611,
        "regime": "A_squared_gn_fp32",
        "epsilon_grid": [0.03, 0.02, 0.015, 0.01],
        "status": "FAIL_CLOSED",
        "detail": {
            "finite": True,
            "source_preserved": True,
            "branches": [
                {"output_fd_epsilon": epsilon, "finite": True}
                for epsilon in (0.03, 0.02, 0.015, 0.01)
            ],
            "convergence": {
                "passed": False,
                "tolerance": 0.05,
                "finest_adjacent_pair": {
                    "coarse_epsilon": 0.015,
                    "fine_epsilon": 0.01,
                    "relative_change": 0.10665712476398169,
                },
                "epsilon_metrics": [
                    {"epsilon": 0.03, "finite": True, "jvp_norm": 10.5},
                    {"epsilon": 0.02, "finite": True, "jvp_norm": 12.9,
                     "coarse_epsilon": 0.03, "relative_error": 0.31,
                     "cosine": 0.98, "norm_ratio": 1.22},
                    {"epsilon": 0.015, "finite": True, "jvp_norm": 13.6,
                     "coarse_epsilon": 0.02, "relative_error": 0.13,
                     "cosine": 0.99, "norm_ratio": 1.05},
                    {"epsilon": 0.01, "finite": True, "jvp_norm": 13.9,
                     "coarse_epsilon": 0.015,
                     "relative_error": 0.10665712476398169,
                     "cosine": 0.9945608319866657,
                     "norm_ratio": 1.0175},
                ],
            },
        },
    }
    (receipt_root / "armA_batch2026082601_dir2026082611_"
                    "A_squared_gn_fp32.json").write_text(
        json.dumps(cell), encoding="utf-8")
    (receipt_root / "correctness_gate.json").write_text(json.dumps({
        "status": "NO_GO",
        "formal_admissible": False,
    }), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(summarize, "parse_args", lambda: argparse.Namespace(
        receipt_root=receipt_root, out=out))

    summarize.main()

    result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["expected_cell_count"] == 1
    assert result["missing_cells"] == []
    assert result["formal_not_run_by_design"] is True
    assert result["formal_observed_cell_count"] == 0
    assert result["verdict"] == "NO-GO"
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "not run, as required by the frozen stop rule" in report
    assert "cannot localize the PR #87 failure" in report
