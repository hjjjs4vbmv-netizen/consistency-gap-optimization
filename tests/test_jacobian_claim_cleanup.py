"""Claim-ceiling checks for the calibrated Jacobian audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_names_the_parameter_partial():
    summary = json.loads((
        ROOT / "analysis" / "jacobian_failure_factorial_v2" / "summary.json"
    ).read_text(encoding="utf-8"))
    claim = summary["claim"]
    assert "parameter-to-augmented-state" in claim
    assert "complete augmented-state derivative" not in claim


def test_old_coarse_grid_report_is_marked_superseded():
    report = (
        ROOT / "analysis" / "jacobian_failure_factorial" / "REPORT.md"
    ).read_text(encoding="utf-8")
    assert "superseded for\nfield-level interpretation" in report
    assert "passes 32/32 formal cells" in report
