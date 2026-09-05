#!/usr/bin/env python3
"""Run one no-quality G4 canary from an attempt-4032 M1 gate export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots
from scripts import run_m1_evaluation_job as worker
from scripts import validate_m1_evaluation_job as validation


SCHEMA = "ect.m1.g4-no-quality-canary/v1"


def validate_evaluation_manifest(path: Path, training: dict) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    slots.validate_slots(rows)
    expected_rows = slots.build_slots(training["roster"], training)
    if any(
        any(observed[field] != str(expected[field]) for field in slots.FIELDS)
        for observed, expected in zip(rows, expected_rows)
    ):
        raise validation.ValidationError(
            "evaluation manifest differs from the canonical training-bound 320 slots"
        )
    return rows


def validate_gate_export(gates: dict, seed: int, branch: str, snapshot: dict) -> dict:
    gate_seed = [row for row in gates.get("seeds", []) if row.get("seed") == seed]
    artifact = (
        gate_seed[0].get("artifacts", {}).get(f"{branch}_continuous_state")
        if len(gate_seed) == 1 else None
    )
    if (
        len(gate_seed) != 1
        or gate_seed[0].get("status") != "PASS"
        or gate_seed[0].get("manifest_sha256_by_branch", {}).get(branch)
        != snapshot["branch_manifest_sha256"]
        or not isinstance(artifact, dict)
        or artifact.get("path") != snapshot["source_state_path"]
        or artifact.get("sha256") != snapshot["terminal_state_sha256"]
    ):
        raise validation.ValidationError(
            "gate export is not the bound G1-G3 continuous state/branch manifest"
        )
    return {"path": artifact["path"], "sha256": artifact["sha256"]}


def run(args: argparse.Namespace) -> int:
    training = slots.load_training_identity(args.training_manifest)
    implementation_checkout = validation.verify_implementation_checkout(
        training["implementation_commit"]
    )
    evaluation_manifest_path = args.evaluation_manifest.resolve(strict=True)
    validate_evaluation_manifest(evaluation_manifest_path, training)
    gates_path = args.training_gates_receipt.resolve(strict=True)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if (
        gates.get("schema") != "ect.m1.training-gates/v1"
        or gates.get("status") != "PASS"
        or gates.get("training_manifest_sha256")
        != training["training_manifest_sha256"]
    ):
        raise validation.ValidationError("G1-G3 receipt is not a PASS for this training manifest")
    roster = [
        row for row in training["roster"]
        if row["roster_slot"] == args.roster_slot
    ]
    if len(roster) != 1 or args.roster_slot not in {"S01", "S02"}:
        raise validation.ValidationError("G4 is restricted to frozen roster slots S01/S02")
    seed = roster[0]["seed"]
    pseudo_slot = {
        "seed": str(seed),
        "branch": args.branch,
        "readout": args.readout,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "implementation_checkout": implementation_checkout,
        "evaluation_manifest_path": str(evaluation_manifest_path),
        "evaluation_manifest_sha256": validation.sha256_file(
            evaluation_manifest_path
        ),
        "frozen_source_state_sha256": training["sources"][
            (seed, "B" if args.branch.endswith("_B") else "A")
        ],
    }
    export_path = args.gate_export_receipt.resolve(strict=True)
    snapshot = validation.load_snapshot_receipt(
        export_path, pseudo_slot, training, gate=True
    )
    gate_training_state = validate_gate_export(
        gates, seed, args.branch, snapshot
    )
    evaluator_repo = args.evaluator_repo.resolve(strict=True)
    evaluator = validation.verify_evaluator(
        evaluator_repo, slots.EVALUATOR_COMMIT, args.evaluator_archive
    )
    runtime = validation.verify_runtime(
        args.runtime_base, args.runtime_env, args.runtime_receipt
    )
    dataset = args.evaluation_dataset.resolve(strict=True)
    validation.verify_evaluation_dataset(dataset)
    worker.validate_cache_root(args.cache_root)
    gpu_resource = worker.gpu_resource_probe(args.gpu_index)
    gpu_uuid = str(gpu_resource["uuid"])
    gpu_row = ", ".join(str(gpu_resource[key]) for key in (
        "index", "uuid", "name", "free_mib", "utilization_percent"
    ))
    disk_resource = {
        "output": worker.disk_resource_probe(args.output_root),
        "cache": worker.disk_resource_probe(args.cache_root),
    }
    if args.block not in slots.READOUT_BLOCKS[args.readout]:
        raise validation.ValidationError("G4 block is not valid for this readout")
    start, end = slots.BLOCKS[args.block]
    gate_id = f"{args.roster_slot}-{args.branch}-{args.readout}-{args.block}"
    command_slot = {
        "slot_id": f"G4-{gate_id}",
        "block": args.block,
        "readout": args.readout,
        "sample_seed_start": str(start),
        "sample_seed_end": str(end),
        "nfe": "1",
        "precision": "fp32",
        "metrics": "kid50k_full,fid50k_full",
        "metric_seed": str(slots.METRIC_SEED),
    }
    runtime_python = Path(runtime["runtime_python"])
    master_port = 54_000 + args.gpu_index * 100 + (
        int(args.roster_slot[1:]) * 20
        + slots.BRANCHES.index(args.branch) * 3
        + tuple(slots.READOUT_BLOCKS).index(args.readout)
    )
    command = worker.build_command(
        command_slot,
        snapshot["snapshot_path"],
        dataset,
        args.output_root.resolve() / "g4-dry-run" / gate_id,
        evaluator_repo,
        runtime_python,
        master_port,
    )
    environment = worker.runtime_env(
        Path(runtime["runtime_base"]),
        Path(runtime["runtime_environment"]),
        args.cache_root.resolve(strict=True),
        args.gpu_index,
        master_port,
        runtime.get("runtime_library_paths"),
    )
    checks = worker.run_canary(
        command,
        snapshot["snapshot_path"],
        runtime_python,
        evaluator_repo,
        environment,
        runtime["runtime_pip_freeze_sha256"],
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "protocol_id": slots.PROTOCOL_ID,
        "quality_eligible": False,
        "quality_generation": False,
        "quality_metrics_executed": False,
        "gate_id": gate_id,
        "roster_slot": args.roster_slot,
        "seed": seed,
        "branch": args.branch,
        "readout": args.readout,
        "block": args.block,
        "training_manifest_path": training["training_manifest_path"],
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "implementation_checkout": implementation_checkout,
        "training_runtime_receipt_sha256": training[
            "training_runtime_receipt_sha256"
        ],
        "evaluation_manifest_path": str(evaluation_manifest_path),
        "evaluation_manifest_sha256": validation.sha256_file(
            evaluation_manifest_path
        ),
        "training_gates_receipt_path": str(gates_path),
        "training_gates_receipt_sha256": validation.sha256_file(gates_path),
        "gate_export_receipt_path": str(export_path),
        "gate_export_receipt_sha256": validation.sha256_file(export_path),
        "gate_training_state": gate_training_state,
        "snapshot": {
            key: snapshot[key]
            for key in (
                "snapshot_path", "snapshot_sha256", "source_state_path",
                "terminal_state_sha256", "branch_manifest_path",
                "branch_manifest_sha256", "frozen_source_state_sha256",
                "source_attempted_iteration", "source_cur_nimg",
                "quality_eligible", "gate_state",
            )
        },
        "runtime": runtime,
        "runtime_probe": checks["runtime_probe"],
        "evaluator": evaluator,
        "evaluator_source": str(evaluator_repo),
        "evaluation_dataset": str(dataset),
        "evaluation_dataset_sha256": validation.DATASET_SHA256,
        "gpu_uuid": gpu_uuid,
        "gpu_identity_row": gpu_row,
        "gpu_resource_probe": gpu_resource,
        "disk_resource_probe": disk_resource,
        "command": command + ["--dry_run"],
        "checks": checks,
    }
    validation.atomic_json(args.receipt.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--training-gates-receipt", type=Path, required=True)
    parser.add_argument("--gate-export-receipt", type=Path, required=True)
    parser.add_argument("--roster-slot", required=True)
    parser.add_argument("--branch", choices=slots.BRANCHES, required=True)
    parser.add_argument("--readout", choices=slots.READOUT_BLOCKS, required=True)
    parser.add_argument("--block", choices=slots.BLOCKS, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--evaluator-archive", type=Path)
    parser.add_argument("--runtime-base", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "protocol_id": slots.PROTOCOL_ID,
            "quality_eligible": False,
            "quality_generation": False,
            "quality_metrics_executed": False,
            "roster_slot": args.roster_slot,
            "branch": args.branch,
            "readout": args.readout,
            "block": args.block,
            "training_manifest_path": str(args.training_manifest),
            "evaluation_manifest_path": str(args.evaluation_manifest),
            "training_gates_receipt_path": str(args.training_gates_receipt),
            "gate_export_receipt_path": str(args.gate_export_receipt),
            "runtime_receipt_path": str(args.runtime_receipt),
            "evaluator_repo": str(args.evaluator_repo),
            "evaluation_dataset": str(args.evaluation_dataset),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        validation.atomic_json(args.receipt.resolve(), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
