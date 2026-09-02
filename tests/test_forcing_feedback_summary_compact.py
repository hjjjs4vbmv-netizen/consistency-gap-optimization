"""Scientific-label and repository-size contracts for forcing-feedback output."""
from __future__ import annotations

from analysis.nonlinear_dynamics_gate.compact_existing_summary import compact, relabel


def fixture():
    step_receipts = [{"k": 0, "large": [1, 2, 3]}]
    return {
        "schema_version": 1,
        "mechanism_by_arm_and_block": {
            "B:state:theta": {
                "classification": "trajectory_feedback_amplification"}},
        "mechanism_decision_rules": {
            "trajectory_feedback_amplification": "old rule"},
        "interpretation_guard": "old guard",
        "instrumentation": {"replay_receipt": {
            "step_replay_receipts": step_receipts}},
        "run_receipt": {
            "state_hashes_k_0_through_horizon": ["a", "b"],
            "step_replay_receipts": step_receipts,
        },
    }


def test_relabel_removes_amplification_claim():
    result = relabel(fixture())
    item = result["mechanism_by_arm_and_block"]["B:state:theta"]
    assert item["classification"] == "persistent_state_feedback_dominance"
    assert "trajectory_feedback_amplification" not in result[
        "mechanism_decision_rules"]


def test_compact_replaces_full_step_receipts_with_hash():
    result = compact(relabel(fixture()))
    assert "step_replay_receipts" not in result["run_receipt"]
    assert result["run_receipt"]["step_replay_receipt_count"] == 1
    assert len(result["run_receipt"]["step_replay_receipts_sha256"]) == 64
