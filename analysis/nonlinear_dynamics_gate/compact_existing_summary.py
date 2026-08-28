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
    result["schema_version"] = 2
    for item in result["mechanism_by_arm_and_block"].values():
        if item["classification"] == OLD_LABEL:
            item["classification"] = NEW_LABEL
    rules = result["mechanism_decision_rules"]
    rules[NEW_LABEL] = rules.pop(OLD_LABEL)
    result["interpretation_guard"] = (
        "R_over_b is a scale diagnostic, never a contribution percentage; "
        "large state-block values identify history-dominated state propagation, "
        "not a dynamical gain or quality mechanism."
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
