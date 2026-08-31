#!/usr/bin/env python3
"""Create one immutable P2 branch directory and source-bound manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(18, *pulse_chase.SEEDS), required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--branch", choices=tuple(pulse_chase.BRANCHES), required=True)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matched-randomness-audit", action="store_true")
    args = parser.parse_args()
    inventory_path = args.source_inventory.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "PASS"
        or inventory.get("seed") != args.seed
        or inventory.get("run_kind") != args.run_kind
    ):
        raise RuntimeError("source inventory is not PASS for this seed")
    protocol_path = args.protocol.resolve(strict=True)
    protocol_sha = pulse_chase.sha256_file(protocol_path)
    if inventory.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("source inventory protocol mismatch")
    if inventory.get("implementation_commit") != args.implementation_commit:
        raise RuntimeError("source inventory implementation mismatch")
    if inventory.get("asset_sha256") != pulse_chase.ASSET_SHA256:
        raise RuntimeError("source inventory asset mismatch")
    expected_gpu = 0 if args.seed <= 23 else 1
    if args.run_kind == "formal" and args.gpu_index != expected_gpu:
        raise RuntimeError("seed is assigned to the other physical GPU")
    source = inventory["source_state"]
    source_path = Path(source["path"]).resolve(strict=True)
    if source_path.stat().st_size != source["bytes"]:
        raise RuntimeError("source byte count changed")
    if pulse_chase.sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("source file SHA256 changed")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    branch_spec = pulse_chase.BRANCHES[args.branch]
    manifest = {
        "schema": pulse_chase.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": pulse_chase.PROTOCOL,
        "run_kind": args.run_kind,
        "seed": args.seed,
        "branch": args.branch,
        "pulse_arm": branch_spec["pulse_arm"],
        "chase_arm": "A",
        "source_kimg": 384,
        "pulse_end_kimg": 512,
        "chase_end_kimg": 640,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "implementation_commit": args.implementation_commit,
        "asset_sha256": dict(pulse_chase.ASSET_SHA256),
        "source_inventory_path": str(inventory_path),
        "source_inventory_sha256": pulse_chase.sha256_file(inventory_path),
        "source_state": source,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "immutable_output_root": str(output),
        "matched_randomness_audit": bool(args.matched_randomness_audit),
    }
    path = output / "formal_run_manifest.json"
    reproducibility.atomic_json_dump(manifest, path, overwrite=False)
    if pulse_chase.load_run_manifest(path) != manifest:
        raise RuntimeError("manifest canonical reload mismatch")
    reproducibility.atomic_json_dump(
        {
            "schema": "ect.q256.p2-cell-preparation/v1",
            "status": "PASS",
            "manifest_sha256": pulse_chase.sha256_file(path),
            "source_files_modified": False,
        },
        output / "cell_preparation_receipt.json",
        overwrite=False,
    )
    print(json.dumps({"status": "PASS", "manifest": str(path),
                      "sha256": pulse_chase.sha256_file(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
