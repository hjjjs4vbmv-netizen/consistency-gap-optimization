"""Checks for the committed calibrated-factorial outputs."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "jacobian_failure_factorial_v2" / "summary.json"


def test_committed_summary_supports_only_transition_level_go():
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["integrity_passed"] is True
    assert result["expected_cell_count"] == 160
    assert result["unique_cell_count"] == 160
    assert result["transition_level_verdict"] == (
        "GO_PRODUCTION_TRANSITION_SEPARATION")
    assert result["internal_component_verdict"] == (
        "HOLD_INTERNAL_COMPONENT_ATTRIBUTION")


def test_committed_summary_retains_cell_level_status_counts():
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    by_regime = result["by_regime"]
    assert by_regime["A_squared_gn_fp32"]["status_counts"] == {
        "PASS": 30, "FAIL_CLOSED": 2}
    for regime in (
            "B_real_loss_gn_fp32", "C_full_field_fp32",
            "E_full_field_pseudohuber_fp32"):
        assert by_regime[regime]["status_counts"] == {"PASS": 32}
    assert by_regime["D_production_algorithmic"]["status_counts"] == {
        "FAIL_CLOSED": 32}
