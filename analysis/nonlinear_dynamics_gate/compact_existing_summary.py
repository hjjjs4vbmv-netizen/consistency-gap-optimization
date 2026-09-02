#!/usr/bin/env python3
"""Relabel and compact a previously generated forcing-feedback summary."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.nonlinear_dynamics_gate.decompose_forcing_feedback import (
    _canonical_sha256,
    _compact_run_receipt,
    write_report,
)


OLD_LABEL = "trajectory_feedback_amplification"
NEW_LABEL = "persistent_state_feedback_dominance"


def relabel(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(summary))
    result["schema_version"] = max(2, int(result.get("schema_version", 1)))
    migrated = False
    for item in result["mechanism_by_arm_and_block"].values():
        if item["classification"] == OLD_LABEL:
            item["classification"] = NEW_LABEL
            migrated = True
    rules = result["mechanism_decision_rules"]
    if OLD_LABEL in rules:
        rules[NEW_LABEL] = rules.pop(OLD_LABEL)
        migrated = True
    if migrated:
        result["interpretation_guard"] = (
            "R_over_b is a scale diagnostic, never a contribution percentage; "
            "large state-block values identify history-dominated state propagation, "
            "not a dynamical gain or quality mechanism."
        )
    old_gate = result.pop("amplification_claim_gate", None)
    if old_gate is not None:
        result["strong_expansion_claim_gate"] = old_gate
    for item in result["mechanism_by_arm_and_block"].values():
        if "amplification_term_allowed" in item:
            item["strong_expansion_claim_allowed"] = item.pop(
                "amplification_term_allowed")
        if "amplification_gate_reason" in item:
            item["strong_expansion_gate_reason"] = item.pop(
                "amplification_gate_reason")
    dominance = result["mechanism_decision_rules"].get(NEW_LABEL)
    if isinstance(dominance, str):
        result["mechanism_decision_rules"][NEW_LABEL] = dominance.replace(
            "not an amplification claim", "not a stronger expansion claim")
    propagation = result.get("pre_transition_propagation_interpretation", {})
    if "G_gt_1_and_alignment_approximately_1" in propagation:
        propagation["G_gt_1_and_alignment_approximately_1"] = (
            "possible same-direction expansion, subject to the stronger causal claim gate"
        )
    if "low_alignment" in propagation:
        propagation["low_alignment"] = (
            "rotation or complex deformation rather than simple same-direction expansion"
        )
    return result


def compact(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(summary))
    receipt = result["run_receipt"]
    step_receipts = receipt["step_replay_receipts"]
    result["instrumentation"]["replay_receipt"] = {
        "state_hashes_k_0_through_horizon": receipt[
            "state_hashes_k_0_through_horizon"],
        "step_replay_receipt_count": len(step_receipts),
        "step_replay_receipts_sha256": _canonical_sha256(step_receipts),
    }
    result["run_receipt"] = _compact_run_receipt(receipt)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    full = relabel(source)
    compact_summary = compact(full)
    if args.full_output is not None:
        args.full_output.write_text(
            json.dumps(full, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    args.compact_output.write_text(
        json.dumps(compact_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.report_output is not None:
        write_report(args.report_output, compact_summary)


if __name__ == "__main__":
    main()
