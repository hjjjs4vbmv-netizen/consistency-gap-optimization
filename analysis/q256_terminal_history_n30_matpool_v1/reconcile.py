#!/usr/bin/env python3
"""Reconcile the known n30 postcheck receipt bug without altering old evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


FALSE_ERROR = "suffix trajectory receipt failed:"
ZERO_FIELDS = (
    "loss_nonfinite_count",
    "sanitized_grad_nonfinite_count",
    "update_nonfinite_count",
    "model_nonfinite_count",
    "ema_nonfinite_count",
    "factor_nonfinite_count",
    "nonpositive_denominator_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        current = load(path)
        if current != value:
            raise RuntimeError(f"existing reconciliation differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    os.unlink(temporary)


def audit_suffix(cell_dir: Path, cell: str, protocol_sha: str) -> dict:
    compute_path = cell_dir / "compute_completion_receipt.json"
    trajectory_path = cell_dir / "trajectory_completion_receipt.json"
    manifest_path = cell_dir / "formal_run_manifest.json"
    false_path = cell_dir / "postcheck_failure_receipt.json"
    for path in (compute_path, trajectory_path, manifest_path, false_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing required artifact: {path}")
    compute = load(compute_path)
    trajectory = load(trajectory_path)
    manifest = load(manifest_path)
    false = load(false_path)
    if compute.get("status") != "PASS" or compute.get("exit_code") != 0:
        raise RuntimeError(f"compute did not pass: {cell_dir}")
    if trajectory.get("status") != "PASS":
        raise RuntimeError(f"trajectory receipt did not pass: {cell_dir}")
    if manifest.get("branch") != cell or manifest.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(f"manifest identity mismatch: {cell_dir}")
    if false.get("status") != "FAIL" or FALSE_ERROR not in false.get("error", ""):
        raise RuntimeError(f"not the known false postcheck: {false_path}")
    telemetry = cell_dir / "schedule_switch_training_telemetry_v1.csv"
    with telemetry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    attempts = [int(float(row["attempted_iteration"])) for row in rows]
    if attempts != list(range(4001, 8001)):
        raise RuntimeError(f"telemetry coverage mismatch: {telemetry}")
    for row in rows:
        if row.get("continuation_arm") != "A":
            raise RuntimeError(f"continuation arm mismatch: {telemetry}")
        for field in ZERO_FIELDS:
            if int(float(row[field])) != 0:
                raise RuntimeError(
                    f"nonzero semantic failure {field} at {row['attempted_iteration']}"
                )
    return {
        "schema": "ect.q256.terminal-history-postcheck-reconciliation/v1",
        "status": "PASS",
        "cell": cell,
        "reason": "original validator incorrectly required a branch field in the PASS trajectory receipt; branch is validated from the frozen formal manifest",
        "compute_completion_receipt_sha256": sha256(compute_path),
        "trajectory_completion_receipt_sha256": sha256(trajectory_path),
        "formal_run_manifest_sha256": sha256(manifest_path),
        "superseded_false_postcheck_receipt_sha256": sha256(false_path),
        "telemetry_sha256": sha256(telemetry),
        "telemetry_rows": len(rows),
        "attempt_range": [4001, 8000],
        "semantic_nonfinite_count": 0,
        "protocol_sha256": protocol_sha,
        "original_evidence_modified": False,
        "automatic_retry_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    protocol_sha = load(root / "control" / "protocol.json").get("implementation_commit")
    companion = (root / "control" / "protocol.sha256").read_text().split()[0]
    if len(companion) != 64:
        raise RuntimeError("invalid protocol SHA companion")
    reconciled = []
    scientific = []
    for seed_dir in sorted((root / "training").glob("seed*")):
        seed = int(seed_dir.name[4:])
        for child in seed_dir.glob("*/compute_completion_receipt.json"):
            receipt = load(child)
            if receipt.get("status") != "PASS":
                scientific.append({
                    "seed": seed,
                    "cell": child.parent.name,
                    "receipt": str(child),
                    "receipt_sha256": sha256(child),
                    "status": receipt.get("status"),
                    "exit_code": receipt.get("exit_code"),
                })
        for cell in ("AA", "BA"):
            cell_dir = seed_dir / cell
            false_path = cell_dir / "postcheck_failure_receipt.json"
            if not false_path.is_file():
                continue
            false = load(false_path)
            if FALSE_ERROR not in false.get("error", ""):
                continue
            record = audit_suffix(cell_dir, cell, companion)
            destination = cell_dir / "postcheck_reconciliation_v1.json"
            write_once(destination, record)
            reconciled.append({"seed": seed, "cell": cell, "path": str(destination)})
    summary = {
        "schema": "ect.q256.terminal-history-reconciliation-summary/v1",
        "status": "PASS",
        "known_validator_bug": "trajectory PASS receipt has no branch field; original code falsely classified it as failure",
        "reconciled_cells": reconciled,
        "scientific_compute_failures": scientific,
        "protocol_sha256": companion,
        "original_evidence_modified": False,
        "automatic_retry_performed": False,
    }
    destination = root / "control" / "postcheck_reconciliation_summary_v1.json"
    # This summary is a live snapshot; only publish once the node is complete.
    if (root / "node_completion_receipt.json").is_file():
        write_once(destination, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
