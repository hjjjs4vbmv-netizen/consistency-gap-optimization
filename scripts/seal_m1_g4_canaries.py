#!/usr/bin/env python3
"""Seal the five no-quality G4 canary classes for formal admission."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots
from scripts import run_m1_g4_canary as canary
from scripts import validate_m1_evaluation_job as validation


SCHEMA = "ect.m1.g4-canary-seal/v1"
EXPECTED_CLASSES = {
    ("ONLINE", "B0"), ("E_KEEP", "B0"),
    ("E_512", "B0"), ("E_512", "B1"), ("E_512", "B2"),
}


class G4SealError(RuntimeError):
    pass


def validate_canary(
    payload, training, gates, gates_sha256, evaluation_manifest_sha256
):
    expected = {
        "schema": canary.SCHEMA, "status": "PASS",
        "protocol_id": slots.PROTOCOL_ID, "quality_eligible": False,
        "quality_generation": False, "quality_metrics_executed": False,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "training_runtime_receipt_sha256": training[
            "training_runtime_receipt_sha256"
        ],
        "implementation_commit": training["implementation_commit"],
        "training_gates_receipt_sha256": gates_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "evaluation_dataset_sha256": validation.DATASET_SHA256,
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise G4SealError(
            f"G4 canary top-level binding mismatch: {','.join(mismatches)}"
        )
    if payload.get("roster_slot") not in {"S01", "S02"}:
        raise G4SealError("G4 canary must use a frozen gate roster slot")
    if payload.get("implementation_checkout") != {
        "head": training["implementation_commit"], "clean": True,
    }:
        raise G4SealError("G4 canary lacks verified clean implementation checkout")
    if validation.verify_implementation_checkout(
        training["implementation_commit"]
    ) != payload["implementation_checkout"]:
        raise G4SealError("G4 implementation checkout changed after canary")
    readout, block = payload.get("readout"), payload.get("block")
    if (readout, block) not in EXPECTED_CLASSES:
        raise G4SealError("unexpected G4 readout/block class")
    snapshot = payload.get("snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("source_attempted_iteration") != 4_032
        or snapshot.get("source_cur_nimg") != 516_096
        or snapshot.get("quality_eligible") is not False
        or snapshot.get("gate_state") is not True
    ):
        raise G4SealError("G4 canary is not bound to an attempt-4032 gate export")
    for key in (
        "snapshot_sha256", "terminal_state_sha256", "branch_manifest_sha256",
        "frozen_source_state_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get(key, ""))) is None:
            raise G4SealError(f"G4 snapshot provenance missing: {key}")
    evaluation_manifest = Path(str(payload.get("evaluation_manifest_path", "")))
    export_receipt = Path(str(payload.get("gate_export_receipt_path", "")))
    if (
        evaluation_manifest.is_symlink() or not evaluation_manifest.is_file()
        or validation.sha256_file(evaluation_manifest) != evaluation_manifest_sha256
        or export_receipt.is_symlink() or not export_receipt.is_file()
        or validation.sha256_file(export_receipt)
        != payload.get("gate_export_receipt_sha256")
    ):
        raise G4SealError("G4 manifest/export artifact binding mismatch")
    try:
        canary.validate_evaluation_manifest(evaluation_manifest, training)
    except (slots.SlotError, validation.ValidationError) as exc:
        raise G4SealError(f"G4 evaluation manifest is not training-bound: {exc}") from exc
    pseudo_slot = {
        "seed": str(payload.get("seed")), "branch": payload.get("branch"),
        "readout": payload.get("readout"),
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "frozen_source_state_sha256": snapshot.get("frozen_source_state_sha256"),
    }
    try:
        observed_snapshot = validation.load_snapshot_receipt(
            export_receipt, pseudo_slot, training, gate=True
        )
    except validation.ValidationError as exc:
        raise G4SealError(f"G4 gate export provenance mismatch: {exc}") from exc
    if any(observed_snapshot.get(key) != value for key, value in snapshot.items()):
        raise G4SealError("G4 snapshot fields differ from the bound gate export")
    try:
        gate_training_state = canary.validate_gate_export(
            gates, int(payload["seed"]), payload["branch"], observed_snapshot
        )
    except validation.ValidationError as exc:
        raise G4SealError(f"G4 gate state does not match G1-G3: {exc}") from exc
    if payload.get("gate_training_state") != gate_training_state:
        raise G4SealError("G4 canary gate-state binding mismatch")
    checks, probe = payload.get("checks"), payload.get("runtime_probe")
    evaluator, runtime = payload.get("evaluator"), payload.get("runtime")
    if not isinstance(checks, dict) or checks.get("status") != "G4_NO_QUALITY_CANARY_PASS":
        raise G4SealError("G4 dry-run checks did not pass")
    if (
        not isinstance(probe, dict) or probe.get("cuda_available") is not True
        or any(probe.get(key) != value for key, value in validation.EXPECTED_RUNTIME_PROBE.items())
        or not isinstance(runtime, dict)
        or probe.get("pip_freeze_sha256")
        != runtime.get("runtime_pip_freeze_sha256")
    ):
        raise G4SealError("G4 live runtime probe mismatch")
    if not isinstance(evaluator, dict) or evaluator.get("evaluator_commit") != slots.EVALUATOR_COMMIT:
        raise G4SealError("G4 evaluator identity mismatch")
    gpu_resource = payload.get("gpu_resource_probe")
    disk_resource = payload.get("disk_resource_probe")
    if (
        not isinstance(gpu_resource, dict)
        or "A100" not in str(gpu_resource.get("name", ""))
        or not isinstance(gpu_resource.get("free_mib"), int)
        or gpu_resource["free_mib"] < 35_000
        or not isinstance(gpu_resource.get("utilization_percent"), int)
        or gpu_resource["utilization_percent"] > 5
        or not isinstance(disk_resource, dict)
        or any(
            not isinstance(disk_resource.get(name), dict)
            or disk_resource[name].get("minimum_free_bytes") != 5 << 30
            or not isinstance(disk_resource[name].get("free_bytes"), int)
            or disk_resource[name]["free_bytes"] < 5 << 30
            for name in ("output", "cache")
        )
    ):
        raise G4SealError("G4 basic GPU/disk resource probe mismatch")
    runtime_sha = runtime.get("runtime_integrity_receipt_sha256") if isinstance(runtime, dict) else None
    runtime_origin = runtime.get("runtime_origin") if isinstance(runtime, dict) else None
    if re.fullmatch(r"[0-9a-f]{64}", str(runtime_sha or "")) is None:
        raise G4SealError("G4 runtime receipt identity missing")
    if runtime_origin not in {"ORIGINAL_FROZEN_ARCHIVE", "REBUILT_NOT_BYTE_IDENTICAL"}:
        raise G4SealError("G4 runtime origin is missing or invalid")
    runtime_receipt_path = Path(str(runtime.get("runtime_integrity_receipt", "")))
    if (
        runtime_receipt_path.is_symlink() or not runtime_receipt_path.is_file()
        or validation.sha256_file(runtime_receipt_path) != runtime_sha
    ):
        raise G4SealError("G4 runtime receipt file/hash mismatch")
    try:
        observed_runtime = validation.verify_runtime(
            Path(runtime["runtime_base"]), Path(runtime["runtime_environment"]),
            runtime_receipt_path,
        )
    except (KeyError, validation.ValidationError) as exc:
        raise G4SealError(f"G4 runtime artifact mismatch: {exc}") from exc
    if observed_runtime != runtime:
        raise G4SealError("G4 runtime fields differ from its current receipt")
    evaluator_source = Path(str(payload.get("evaluator_source", "")))
    archive_value = evaluator.get("evaluator_archive") if isinstance(evaluator, dict) else None
    try:
        observed_evaluator = validation.verify_evaluator(
            evaluator_source, slots.EVALUATOR_COMMIT,
            None if archive_value is None else Path(str(archive_value)),
        )
        observed_dataset = validation.verify_evaluation_dataset(
            Path(str(payload.get("evaluation_dataset", "")))
        )
    except validation.ValidationError as exc:
        raise G4SealError(f"G4 evaluator/data artifact mismatch: {exc}") from exc
    if observed_evaluator != evaluator or observed_dataset != validation.DATASET_SHA256:
        raise G4SealError("G4 evaluator/data fields differ from current artifacts")
    command = payload.get("command")
    start, end = slots.BLOCKS[block]
    required = {"--dry_run", "--nfe=1", "--fp16=False", f"--sample-seeds={start}-{end}"}
    if not isinstance(command, list) or not required.issubset(set(command)):
        raise G4SealError("G4 dry-run command mismatch")
    return str(runtime_sha), str(runtime_origin)


def seal(
    training, gates_path: Path, evaluation_manifest_sha256: str,
    receipts_dir: Path,
) -> dict[str, Any]:
    gates_path = gates_path.resolve(strict=True)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    gates_sha = validation.sha256_file(gates_path)
    if (
        gates.get("schema") != "ect.m1.training-gates/v1"
        or gates.get("status") != "PASS"
        or gates.get("training_manifest_sha256") != training["training_manifest_sha256"]
    ):
        raise G4SealError("G1-G3 receipt is not a PASS for this training manifest")
    paths = sorted(receipts_dir.glob("*-g4-canary.json"))
    if len(paths) != len(EXPECTED_CLASSES):
        raise G4SealError("G4 requires exactly five canary receipts")
    classes, trajectories, runtimes, receipts = set(), set(), set(), []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise G4SealError(f"invalid G4 canary receipt: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtimes.add(
            validate_canary(
                payload, training, gates, gates_sha, evaluation_manifest_sha256
            )
        )
        classes.add((payload["readout"], payload["block"]))
        trajectories.add((
            payload["roster_slot"], payload["seed"], payload["branch"],
            payload["snapshot"]["source_state_path"],
            payload["snapshot"]["terminal_state_sha256"],
            payload["snapshot"]["branch_manifest_path"],
            payload["snapshot"]["branch_manifest_sha256"],
        ))
        receipts.append({
            "gate_id": payload["gate_id"], "path": str(path.resolve()),
            "sha256": validation.sha256_file(path),
        })
    if classes != EXPECTED_CLASSES:
        raise G4SealError("G4 five-class coverage mismatch")
    if len(trajectories) != 1:
        raise G4SealError("G4 five classes must use one gate trajectory")
    if len(runtimes) != 1:
        raise G4SealError("G4 canaries must share one frozen evaluation runtime")
    runtime_receipt, runtime_origin = next(iter(runtimes))
    (
        roster_slot, seed, branch, terminal_path, terminal_sha256,
        branch_manifest_path, branch_manifest_sha256,
    ) = next(iter(trajectories))
    return {
        "schema": SCHEMA, "status": "PASS", "protocol_id": slots.PROTOCOL_ID,
        "quality_eligible": False, "quality_generation": False,
        "quality_metrics_executed": False,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "training_runtime_receipt_sha256": training[
            "training_runtime_receipt_sha256"
        ],
        "implementation_commit": training["implementation_commit"],
        "training_gates_receipt_path": str(gates_path),
        "training_gates_receipt_sha256": gates_sha,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "runtime_integrity_receipt_sha256": runtime_receipt,
        "runtime_origin": runtime_origin,
        "evaluator_commit": slots.EVALUATOR_COMMIT,
        "evaluation_dataset_sha256": validation.DATASET_SHA256,
        "gate_trajectory": {
            "roster_slot": roster_slot, "seed": seed, "branch": branch,
            "terminal_state_path": terminal_path,
            "terminal_state_sha256": terminal_sha256,
            "branch_manifest_path": branch_manifest_path,
            "branch_manifest_sha256": branch_manifest_sha256,
        },
        "covered_classes": [list(item) for item in sorted(classes)],
        "canary_count": len(receipts), "canary_receipts": receipts,
        "claim_boundary": "dry-run only; no quality generation, 50k throughput, or peak-memory measurement",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-gates-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = slots.load_training_identity(args.training_manifest)
    evaluation_manifest = args.evaluation_manifest.resolve(strict=True)
    payload = seal(
        training, args.training_gates_receipt,
        validation.sha256_file(evaluation_manifest),
        args.receipts_dir.resolve(strict=True),
    )
    validation.atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
