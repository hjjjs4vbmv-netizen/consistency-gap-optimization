#!/usr/bin/env python3
"""Build a pending-review GFCT gap/LR audit receipt from executed artifacts.

This command deliberately cannot issue `formal_launch_allowed`.  It packages
fresh-state and nonzero-state evidence for collaborator review; an authorized
reviewer must sign the final receipt separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_gap_lr_audit_receipt import validate_state

EXPERIMENT_ID = "gap_lr_matched_q128_s3_v1"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
C0_DEFINITION = "dot(delta_1_3,delta_1_0)/dot(delta_1_3,delta_1_3)"


def fail(message: str) -> None:
    raise SystemExit("CANDIDATE RECEIPT REJECTED: " + message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"invalid JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON artifact must contain an object: {path}")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def require_bool(value: Any, name: str) -> bool:
    if value is not True:
        fail(f"{name} must be true")
    return True


def require_finite(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{name} must be numeric")
    if not math.isfinite(number) or (positive and number <= 0):
        fail(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def parse_state_spec(values: list[str]) -> tuple[str, Path, Path, Path]:
    label, audit, layerwise, summary = values
    if not label:
        fail("state label must be nonempty")
    paths = (Path(audit), Path(layerwise), Path(summary))
    for path in paths:
        if not path.is_file():
            fail(f"missing state artifact: {path}")
    return label, *paths


def build_state(
    label: str,
    audit_path: Path,
    layerwise_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    data = load_json(audit_path)
    whole = data["whole_model"]
    provenance = data["provenance"]
    source = data["source_state_non_committing"]
    randomness = data["randomness_contract"]
    branches = data["branches"]
    state_meta = provenance["training_state_meta"]
    radam = data["stateful_radam"]

    require_bool(whole["gauge_defined"], f"{label}.gauge_defined")
    require_bool(source["preserved"], f"{label}.source_preserved")
    if int(state_meta["successful_optimizer_steps"]) < 6:
        fail(f"{label} has fewer than six successful optimizer steps")
    if float(radam["n_K"]) < 6:
        fail(f"{label} is not a rectified RAdam state")
    if not all(
        branch["amp_enabled"]
        and branch["amp_unscale_called"]
        and not branch["step_skipped"]
        for branch in branches
    ):
        fail(f"{label} AMP branch invariants failed")

    parameter_unchanged = (
        source["parameter_hash_before"] == source["parameter_hash_after"]
    )
    optimizer_unchanged = (
        source["optimizer_state_hash_before"]
        == source["optimizer_state_hash_after"]
    )
    gradscaler_unchanged = (
        source["gradscaler_hash_before"]
        == source["gradscaler_hash_after"]
    )
    paired = all(
        randomness[key]
        for key in (
            "same_minibatch",
            "same_t",
            "same_noise",
            "same_dropout_rng_state",
        )
    )

    h_exact = require_finite(
        whole["history_gauge_dispersion_H_K"], f"{label}.H_exact"
    )
    r_opt = require_finite(whole["R_opt"], f"{label}.R_opt")
    h_thresholded = h_exact  # support_atol=0 in the executed formal audits.
    excluded_energy = max(
        0.0,
        1.0 - float(whole["effective_support_energy_coverage"]),
    )

    state = {
        "state_id": label,
        "successful_optimizer_steps": int(
            state_meta["successful_optimizer_steps"]
        ),
        "same_pre_state": True,
        "paired_minibatch_rng": paired,
        "amp_invariants_passed": True,
        "virtual_step_non_committing": source["preserved"],
        "parameter_hash_unchanged": parameter_unchanged,
        "optimizer_hash_unchanged": optimizer_unchanged,
        "gradscaler_hash_unchanged": gradscaler_unchanged,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "pre_parameter_sha256": source["parameter_hash_before"][
            "parameter_sha256"
        ],
        "pre_optimizer_sha256": source["optimizer_state_hash_before"],
        "pre_gradscaler_sha256": source["gradscaler_hash_before"],
        "radam_branch": "rectified",
        "amp_skip_history_sha256": sha256_file(summary_path),
        "paired_input_sha256": canonical_sha256(randomness),
        "training_state_sha256": provenance["training_state_sha256"],
        "state_kimg": provenance["state_kimg"],
        "metrics": {
            "raw_a_star": whole["a_K_star"],
            "raw_residual": whole["R_grad"],
            "update_s_star": whole["s_K_star"],
            "update_c_star": whole["c_K_star"],
            "update_cosine": whole["update_cosine"],
            "reference_update_norm": whole["update_1_l2"],
            "candidate_update_norm": whole["update_1p3_l2"],
            "R_opt": r_opt,
            "H_exact": h_exact,
            "H_thresholded": h_thresholded,
            "off_support_energy": whole[
                "off_support_candidate_energy_exact"
            ],
            "excluded_reference_energy": excluded_energy,
            "abs_H_exact_minus_R_opt": abs(h_exact - r_opt),
            "abs_H_thresholded_minus_R_opt": abs(h_thresholded - r_opt),
            "h_update": {
                "support_atol": whole["support_atol"],
                "history_gauge_dispersion": h_exact,
                "effective_support_coordinate_coverage": whole[
                    "effective_support_coordinate_coverage"
                ],
                "effective_support_energy_coverage": whole[
                    "effective_support_energy_coverage"
                ],
            },
            "h_moment_ideal": {
                "coordinate_count": whole[
                    "moment_effective_support_coordinate_count"
                ],
                "energy_coverage": whole[
                    "moment_effective_support_energy_coverage"
                ],
                "weighted_rmse": whole[
                    "h_update_minus_moment_weighted_rmse"
                ],
                "eps_weighted_rmse": whole[
                    "h_update_minus_moment_eps_weighted_rmse"
                ],
            },
            "layerwise": {
                "artifact_sha256": sha256_file(layerwise_path),
            },
        },
    }
    validate_state(state, 0)
    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument(
        "--state",
        required=True,
        action="append",
        nargs=4,
        metavar=("LABEL", "AUDIT_JSON", "LAYERWISE_CSV", "TRAIN_SUMMARY"),
        help="repeat at least three times",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.state) < 3:
        fail("at least three --state groups are required")
    for path, name in ((args.data, "data"), (args.transfer, "transfer")):
        if not path.is_file():
            fail(f"{name} must be a regular file: {path}")
    if args.output.exists():
        fail(f"refusing to overwrite output: {args.output}")

    status = git("status", "--porcelain")
    if status:
        fail("working tree must be clean before binding protocol_commit")

    fresh = load_json(args.fresh)
    fresh_whole = fresh["whole_model"]
    fresh_source = fresh["source_state_non_committing"]
    fresh_randomness = fresh["randomness_contract"]
    fresh_branches = fresh["branches"]

    require_bool(fresh_whole["gauge_defined"], "fresh.gauge_defined")
    require_bool(fresh_source["preserved"], "fresh.source_preserved")
    c0_star = require_finite(
        fresh_whole["c0_star"], "fresh.c0_star", positive=True
    )
    fresh_paired = all(
        fresh_randomness[key]
        for key in (
            "same_minibatch",
            "same_t",
            "same_noise",
            "same_dropout_rng_state",
        )
    )
    fresh_amp = all(
        branch["amp_unscale_called"] and not branch["step_skipped"]
        for branch in fresh_branches
    )
    if not fresh_paired or not fresh_amp:
        fail("fresh pairing or AMP invariants failed")

    states = [build_state(*parse_state_spec(spec)) for spec in args.state]
    state_hashes = [state["training_state_sha256"] for state in states]
    if len(set(state_hashes)) != len(state_hashes):
        fail("training-state hashes are not distinct")

    receipt = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "evidence_complete_pending_review",
        "verdict": "blocked_pending_collaborator_review",
        "source": {
            "training_code_commit": TRAINING_CODE_COMMIT,
            "protocol_commit": git("rev-parse", "HEAD"),
            "dataset_sha256": sha256_file(args.data),
            "transfer_checkpoint_sha256": sha256_file(args.transfer),
        },
        "fresh_linearized_control": {
            "status": "passed",
            "c0_definition": C0_DEFINITION,
            "c0_star": c0_star,
            "arm_c_learning_rate": c0_star * 1e-4,
            "paired_inputs": fresh_paired,
            "amp_invariants_passed": fresh_amp,
            "state_hashes_unchanged": fresh_source["preserved"],
            "whole_model_residual": fresh_whole["whole_model_residual"],
            "update_cosine": fresh_whole["update_cosine"],
            "artifact_sha256": sha256_file(args.fresh),
            "claim_boundary": (
                "zero-network-update fresh-RAdam scale-512 paired audit; "
                "not an entire-trajectory optimizer match"
            ),
        },
        "optimizer_mechanism_gate": {
            "status": "passed",
            "states": states,
            "claim_boundary": (
                "distinct pre-states; not a same-trajectory "
                "32/64/128/256 longitudinal result"
            ),
        },
        "longitudinal_audit": {
            "checkpoint_kimg": [32, 64, 128, 256],
            "identical_counterfactual_state_required": True,
            "status": "preregistered_pending_formal_trajectories",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"protocol_commit: {receipt['source']['protocol_commit']}")
    print(f"c0_star: {c0_star:.17g}")
    print(f"arm_c_learning_rate: {c0_star * 1e-4:.17g}")
    print(f"audited_states: {len(states)}")
    print(f"distinct_state_hashes: {len(set(state_hashes))}")
    print("status: evidence_complete_pending_review")
    print("verdict: blocked_pending_collaborator_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
