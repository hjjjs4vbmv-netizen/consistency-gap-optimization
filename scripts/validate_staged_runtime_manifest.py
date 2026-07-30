#!/usr/bin/env python3
"""Validate a machine-local staged-evaluation manifest against its frozen matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_staged_evaluation


def fail(message: str) -> None:
    raise SystemExit(f"[validate_staged_runtime_manifest] ERROR: {message}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cells_by_id(manifest: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        fail(f"{label} manifest must contain a non-empty cells list")
    indexed: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("checkpoint_id"), str):
            fail(f"{label} manifest has a cell without a string checkpoint_id")
        checkpoint_id = cell["checkpoint_id"]
        if not checkpoint_id or checkpoint_id in indexed:
            fail(f"{label} manifest checkpoint_id is empty or duplicated: {checkpoint_id!r}")
        indexed[checkpoint_id] = cell
    return indexed


def validate(
    frozen_path: Path,
    runtime_path: Path,
    *,
    allow_missing_inputs: bool = False,
) -> list[dict[str, str]]:
    """Return validated binding rows or raise SystemExit on any mismatch."""
    frozen = load_json(frozen_path, "frozen matrix")
    if frozen.get("manifest_kind") != "frozen-logical-checkpoint-matrix":
        fail("frozen matrix has an unexpected manifest_kind")
    if frozen.get("protocol") != run_staged_evaluation.PROTOCOL_ID:
        fail(f"frozen matrix protocol must be {run_staged_evaluation.PROTOCOL_ID!r}")
    if frozen.get("runtime_binding", {}).get("versioned_paths") is not False:
        fail("frozen matrix must declare versioned_paths=false")

    runtime = load_json(runtime_path, "runtime manifest")
    frozen_cells = cells_by_id(frozen, "frozen")
    runtime_cells = cells_by_id(runtime, "runtime")
    if frozen_cells.keys() != runtime_cells.keys():
        missing = sorted(frozen_cells.keys() - runtime_cells.keys())
        extra = sorted(runtime_cells.keys() - frozen_cells.keys())
        fail(f"runtime checkpoint IDs differ from frozen matrix; missing={missing}, extra={extra}")
    frozen_policy = frozen.get("formal_promotion_policy")
    if frozen_policy is not None and runtime.get("formal_promotion_policy") != frozen_policy:
        fail("runtime formal_promotion_policy must exactly match the frozen matrix")

    # Reuse the evaluator's strict checkpoint/receipt schema checks so this
    # validator cannot accept a runtime manifest which the formal runner rejects.
    evaluator_cells, _ = run_staged_evaluation.load_cells(runtime_path, allow_missing_inputs)
    evaluator_by_id = {cell["checkpoint_id"]: cell for cell in evaluator_cells}
    rows: list[dict[str, str]] = []
    required_identity_fields = (
        "method", "training_seed", "budget_kimg", "schedule_q",
        "schedule_identity", "global_gap_scale", "checkpoint_sha256",
        "executed_training_source_commit",
    )
    for checkpoint_id, frozen_cell in frozen_cells.items():
        runtime_cell = runtime_cells[checkpoint_id]
        missing = [field for field in required_identity_fields if field not in runtime_cell]
        if missing:
            fail(f"runtime cell {checkpoint_id} is missing frozen identity fields: {missing}")
        for field in required_identity_fields:
            if runtime_cell[field] != frozen_cell.get(field):
                fail(
                    f"runtime cell {checkpoint_id} differs from frozen {field}: "
                    f"{runtime_cell[field]!r} != {frozen_cell.get(field)!r}"
                )

        receipt_path = Path(str(runtime_cell.get("integrity_receipt", ""))).expanduser()
        expected_receipt = frozen_cell.get("training_integrity_receipt")
        if not isinstance(expected_receipt, dict):
            fail(f"frozen cell {checkpoint_id} lacks training_integrity_receipt")
        if receipt_path.name != expected_receipt.get("receipt_filename"):
            fail(
                f"runtime receipt filename mismatch for {checkpoint_id}: "
                f"{receipt_path.name!r} != {expected_receipt.get('receipt_filename')!r}"
            )
        if receipt_path.is_file():
            expected_receipt_sha = expected_receipt.get("receipt_sha256")
            if expected_receipt_sha and sha256_file(receipt_path) != expected_receipt_sha:
                fail(f"runtime receipt SHA256 mismatch for {checkpoint_id}: {receipt_path}")
        elif not allow_missing_inputs:
            fail(f"runtime receipt not found: {receipt_path}")

        receipt = run_staged_evaluation.verify_integrity_receipt(
            evaluator_by_id[checkpoint_id], allow_missing_inputs
        )
        rows.append({
            "checkpoint_id": checkpoint_id,
            "checkpoint": str(evaluator_by_id[checkpoint_id]["checkpoint"]),
            "integrity_receipt": str(receipt["path"]),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True, help="Git-tracked logical matrix")
    parser.add_argument("--runtime", type=Path, required=True, help="machine-local manifest with paths")
    parser.add_argument(
        "--allow-missing-inputs", action="store_true",
        help="validate only structure and identity; never use this before formal evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = validate(
        args.frozen, args.runtime, allow_missing_inputs=args.allow_missing_inputs
    )
    print(f"Validated {len(rows)} runtime bindings against {args.frozen}")


if __name__ == "__main__":
    main()
