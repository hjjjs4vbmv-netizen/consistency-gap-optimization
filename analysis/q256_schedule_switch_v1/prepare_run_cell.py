#!/usr/bin/env python3
"""Prepare one immutable parity/formal schedule-switch output cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"missing regular source artifact: {source}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def copy_csv_prefix(source: Path, destination: Path) -> dict:
    with source.open("rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [
            row for row in reader
            if int(float(row["attempted_iteration"]))
            <= schedule_switch.SWITCH_ATTEMPT
        ]
    attempts = [int(float(row["attempted_iteration"])) for row in rows]
    if attempts != list(range(1, schedule_switch.SWITCH_ATTEMPT + 1)):
        raise RuntimeError(f"source CSV does not have an exact 1..4000 prefix: {source}")
    if int(float(rows[-1]["processed_nimg"])) != schedule_switch.SWITCH_NIMG:
        raise RuntimeError(f"source CSV prefix does not end at 512000: {source}")
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "derived_path": str(destination.resolve()),
        "derived_sha256": sha256_file(destination),
        "rows": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-kind", choices=("parity", "formal"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--seed", type=int, choices=range(14, 19), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    branches = (
        schedule_switch.PARITY_BRANCHES
        if args.run_kind == "parity"
        else schedule_switch.FORMAL_BRANCHES
    )
    if args.branch not in branches:
        raise RuntimeError("branch is incompatible with run kind")
    origin, continuation = branches[args.branch]
    inventory_path = args.inventory.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    if inventory.get("status") != "PASS" or inventory.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("source inventory is not PASS or protocol-bound")
    cell = next(
        item for item in inventory["cells"]
        if item["seed"] == args.seed and item["origin_arm"] == origin
    )
    source_state = cell["source_artifacts"]["training_state"]
    source_state_path = Path(source_state["path"]).resolve(strict=True)
    if source_state_path.stat().st_size != source_state["bytes"]:
        raise RuntimeError("source-state size changed since inventory")
    if sha256_file(source_state_path) != source_state["sha256"]:
        raise RuntimeError("source-state SHA changed since inventory")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cell_dir = Path(cell["canonical_cell_dir"])
    train_prefix = copy_csv_prefix(
        cell_dir / "train_summary.csv", output / "train_summary.csv"
    )
    telemetry_prefix = copy_csv_prefix(
        cell_dir / "factorial_training_telemetry_v1.csv",
        output / "source_factorial_training_telemetry_v1.csv",
    )
    copy_exclusive(
        cell_dir / "initial_state_receipt_v1.json",
        output / "initial_state_receipt_v1.json",
    )
    copy_exclusive(
        cell_dir / "training_options.json",
        output / "source_training_options.json",
    )
    final_kimg = 640 if args.run_kind == "parity" else 1024
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": schedule_switch.PROTOCOL,
        "run_kind": args.run_kind,
        "branch": args.branch,
        "seed": args.seed,
        "origin_arm": origin,
        "continuation_arm": continuation,
        "switch_kimg": schedule_switch.SWITCH_KIMG,
        "final_kimg": final_kimg,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "implementation_commit": args.implementation_commit,
        "source_inventory_path": str(inventory_path),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_checkpoint_manifest_sha256": cell[
            "source_checkpoint_manifest"
        ]["sha256"],
        "source_state": {
            "path": str(source_state_path),
            "bytes": source_state["bytes"],
            "sha256": source_state["sha256"],
            "internal_state_sha256": cell["internal_state_sha256"],
        },
        "source_history_prefix": {
            "train_summary": train_prefix,
            "factorial_telemetry": telemetry_prefix,
        },
        "immutable_output_root": str(output),
    }
    manifest_path = output / "formal_run_manifest.json"
    reproducibility.atomic_json_dump(manifest, manifest_path, overwrite=False)
    loaded = schedule_switch.load_run_manifest(manifest_path)
    if loaded != manifest:
        raise RuntimeError("written run manifest failed canonical reload")
    receipt = {
        "schema": "ect.q256.schedule-switch-cell-preparation/v1",
        "status": "PASS",
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "source_files_modified": False,
    }
    reproducibility.atomic_json_dump(
        receipt, output / "cell_preparation_receipt.json", overwrite=False
    )
    print(json.dumps({"status": "PASS", "output": str(output),
                      "manifest_sha256": receipt["run_manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
