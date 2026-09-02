#!/usr/bin/env python3
"""Fail-closed validator for the seed-4/5 gap-LR replication launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
SOURCE_EXPERIMENT_ID = "gap_lr_matched_q128_s3_v1"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
SOURCE_RECEIPT_SHA256 = "6487fbcc5f63817c8e3a91968f45fb13437d1c580afa73966bdf0ad8061bb9fa"
DATA_SHA256 = "a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
C0_STAR = 1.2963523762588691
C_LR = 0.00012963523762588692
NEW_SEEDS = [4, 5]


def fail(message: str) -> None:
    raise SystemExit("SEED REPLICATION RECEIPT REJECTED: " + message)


def need(obj: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        fail(f"missing {where}.{key}")
    return obj[key]


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo), text=True
    ).strip()


def same_number(value: Any, expected: float, label: str) -> None:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        fail(f"{label} must be numeric: {exc}")
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=0.0, abs_tol=1e-18
    ):
        fail(f"{label}={observed!r}, expected {expected!r}")


def validate_source_receipt(receipt: dict[str, Any]) -> None:
    if need(receipt, "schema_version", "source_receipt") != 1:
        fail("source receipt schema mismatch")
    if need(receipt, "experiment_id", "source_receipt") != SOURCE_EXPERIMENT_ID:
        fail("source experiment mismatch")
    if need(receipt, "status", "source_receipt") != "passed":
        fail("source receipt did not pass")
    if need(receipt, "verdict", "source_receipt") != "formal_launch_allowed":
        fail("source receipt does not authorize its formal launch")
    source = need(receipt, "source", "source_receipt")
    if need(source, "training_code_commit", "source_receipt.source") != TRAINING_CODE_COMMIT:
        fail("source receipt training commit mismatch")
    if need(source, "dataset_sha256", "source_receipt.source") != DATA_SHA256:
        fail("source receipt dataset hash mismatch")
    if need(source, "transfer_checkpoint_sha256", "source_receipt.source") != TRANSFER_SHA256:
        fail("source receipt transfer hash mismatch")
    fresh = need(receipt, "fresh_linearized_control", "source_receipt")
    if need(fresh, "status", "source_receipt.fresh_linearized_control") != "passed":
        fail("source fresh-linearized control did not pass")
    same_number(need(fresh, "c0_star", "source_receipt.fresh_linearized_control"), C0_STAR, "source c0_star")
    gate = need(receipt, "optimizer_mechanism_gate", "source_receipt")
    if need(gate, "status", "source_receipt.optimizer_mechanism_gate") != "passed":
        fail("source optimizer mechanism gate did not pass")
    states = need(gate, "states", "source_receipt.optimizer_mechanism_gate")
    if not isinstance(states, list) or len(states) < 3:
        fail("source optimizer mechanism gate has fewer than three states")


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 1 or matrix.get("experiment_id") != EXPERIMENT_ID:
        fail("matrix identity mismatch")
    if matrix.get("source_experiment_id") != SOURCE_EXPERIMENT_ID:
        fail("matrix source experiment mismatch")
    if matrix.get("status") != "frozen_seed_replication":
        fail("matrix is not frozen")
    if matrix.get("replication_axis") != "training_seed":
        fail("matrix changes an axis other than training seed")
    if matrix.get("existing_formal_seeds") != [3]:
        fail("matrix existing-seed declaration mismatch")
    if matrix.get("new_formal_seeds") != NEW_SEEDS:
        fail("matrix new seeds must be exactly [4, 5]")
    if matrix.get("final_training_seed_count") != 3:
        fail("matrix final seed count must be three")

    shared = need(matrix, "shared_training", "matrix")
    expected_shared = {
        "dataset_sha256": DATA_SHA256,
        "transfer_checkpoint_sha256": TRANSFER_SHA256,
        "q": 128,
        "k": 8,
        "b": 1,
        "c": 0,
        "duration_kimg": 256,
        "batch_size": 128,
        "batch_gpu": 16,
        "optimizer": "RAdam",
        "dropout": 0.2,
        "augment": 0,
        "ema_beta": 0.9993,
        "fp16": True,
        "amp": True,
        "double_ticks": 10000,
        "mapping": "global_sigmoid",
    }
    for key, expected in expected_shared.items():
        if shared.get(key) != expected:
            fail(f"matrix shared_training.{key} changed")
    same_number(shared.get("c0_star"), C0_STAR, "matrix c0_star")

    arms = matrix.get("arms")
    if not isinstance(arms, list) or len(arms) != 3:
        fail("matrix must contain exactly A/B/C")
    expected_arms = [
        ("A", 1.0, 0.0001, "arm_a_g1_0_lr_fixed_s{seed}"),
        ("B", 1.3, 0.0001, "arm_b_g1_3_lr_fixed_s{seed}"),
        ("C", 1.3, C_LR, "arm_c_g1_3_lr_matched_s{seed}"),
    ]
    for item, (arm, gap, lr, template) in zip(arms, expected_arms):
        if item.get("arm") != arm or item.get("run_id_template") != template:
            fail(f"matrix arm {arm} identity changed")
        same_number(item.get("gap_scale"), gap, f"matrix arm {arm} gap")
        same_number(item.get("learning_rate"), lr, f"matrix arm {arm} learning rate")

    policy = need(matrix, "execution_policy", "matrix")
    if policy.get("seed_group_order") != NEW_SEEDS:
        fail("seed group order changed")
    if policy.get("within_seed_arm_order") != ["A", "B", "C"]:
        fail("within-seed A/B/C order changed")
    for key in ("one_seed_group_at_a_time", "second_seed_group_queued", "stop_on_first_failure"):
        if policy.get(key) is not True:
            fail(f"execution policy {key} must be true")
    if policy.get("automatic_retry") is not False:
        fail("automatic retry must be disabled")


def validate_replication_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != 1 or receipt.get("experiment_id") != EXPERIMENT_ID:
        fail("replication receipt identity mismatch")
    if receipt.get("status") != "passed":
        fail("replication receipt status is not passed")
    if receipt.get("verdict") != "formal_seed_replication_launch_allowed":
        fail("replication receipt verdict does not allow launch")
    authorization = need(receipt, "authorization", "replication_receipt")
    if authorization.get("authorized_scope") != "cross-training-seed replication only":
        fail("authorization scope mismatch")
    if authorization.get("authorized_new_seeds") != NEW_SEEDS:
        fail("authorization must name exactly seeds 4 and 5")
    if authorization.get("only_seed_changes") is not True:
        fail("authorization does not enforce seed-only changes")
    contract = need(receipt, "replication_contract", "replication_receipt")
    if contract.get("source_experiment_id") != SOURCE_EXPERIMENT_ID:
        fail("replication contract source mismatch")
    if contract.get("existing_formal_seed") != 3:
        fail("replication contract existing seed mismatch")
    if contract.get("new_formal_seeds") != NEW_SEEDS:
        fail("replication contract seed mismatch")
    if contract.get("required_arms_per_seed") != ["A", "B", "C"]:
        fail("replication contract does not require A/B/C")
    same_number(contract.get("c0_star"), C0_STAR, "replication receipt c0_star")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-audit-receipt", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    args = parser.parse_args()

    for path, label in (
        (args.receipt, "replication receipt"),
        (args.source_audit_receipt, "source audit receipt"),
        (args.matrix, "matrix"),
        (args.data, "dataset"),
        (args.transfer, "transfer checkpoint"),
    ):
        if not path.is_file():
            fail(f"{label} does not exist: {path}")

    receipt = load_json(args.receipt, "replication receipt")
    source_receipt = load_json(args.source_audit_receipt, "source audit receipt")
    matrix = load_json(args.matrix, "matrix")
    validate_replication_receipt(receipt)
    validate_source_receipt(source_receipt)
    validate_matrix(matrix)

    source = need(receipt, "source", "replication_receipt")
    if source.get("training_code_commit") != TRAINING_CODE_COMMIT:
        fail("replication receipt training commit mismatch")
    if source.get("source_audit_receipt_sha256") != SOURCE_RECEIPT_SHA256:
        fail("replication receipt source-audit hash mismatch")
    if source.get("dataset_sha256") != DATA_SHA256 or source.get("transfer_checkpoint_sha256") != TRANSFER_SHA256:
        fail("replication receipt input hashes changed")
    if digest(args.source_audit_receipt) != SOURCE_RECEIPT_SHA256:
        fail("source audit receipt file hash mismatch")
    if digest(args.data) != DATA_SHA256:
        fail("dataset file hash mismatch")
    if digest(args.transfer) != TRANSFER_SHA256:
        fail("transfer checkpoint file hash mismatch")
    if digest(args.matrix) != source.get("matrix_sha256"):
        fail("matrix file hash mismatch")

    protocol_commit = source.get("protocol_commit")
    if not isinstance(protocol_commit, str) or git(args.repo, "rev-parse", "HEAD") != protocol_commit:
        fail("HEAD does not equal replication protocol commit")
    if git(args.repo, "status", "--porcelain", "--untracked-files=no"):
        fail("tracked worktree is not clean")
    result = subprocess.run(
        ["git", "diff", "--quiet", TRAINING_CODE_COMMIT, "--", "ct_train.py", "training"],
        cwd=str(args.repo),
    )
    if result.returncode != 0:
        fail("working training implementation differs from frozen training commit")

    print(format(C0_STAR, ".17g"))


if __name__ == "__main__":
    try:
        main()
    except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        fail(str(exc))
