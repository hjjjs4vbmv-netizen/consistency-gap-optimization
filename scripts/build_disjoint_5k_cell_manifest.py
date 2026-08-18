#!/usr/bin/env python3
"""Build a deterministic PR #53 metric-cell binding manifest.

The manifest binds every metric receipt to an ordered sample-seed range and to
the exact numbered checkpoint record.  It fails open neither on missing metric
files nor on missing checkpoint hashes: the two known seed-3 B/C snapshot-hash
gaps remain explicit and are counted in the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "evidence" / "gap_artifact_manifest_v1.json"
OUTPUT = ROOT / "evidence" / "disjoint_5k_cell_manifest_v1.json"
CHECKSUM_OUTPUT = ROOT / "evidence" / "disjoint_5k_cell_manifest_v1.sha256"
PATH_RE = re.compile(
    r"blocks/block_(\d+)_(\d+)/seed([345])/arm_([abc])/"
    r"metric-(fid5k_full|kid5k_full)\.jsonl"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ValueError(
            f"cannot read git:{commit}:{path}: {result.stderr.decode().strip()}"
        )
    return result.stdout


def parse_checksums(data: bytes) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in data.decode().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        parsed[relative] = digest
    return parsed


def build(root: Path = ROOT) -> dict[str, Any]:
    canonical = json.loads(
        (root / "evidence" / "gap_artifact_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = canonical["evidence_bundles"]["disjoint_fid_kid_5k"]
    metric_root = bundle["raw_metric_root"]
    commit = bundle["artifact_commit"]
    checksums = parse_checksums(
        git_bytes(root, commit, f"{metric_root}/RESULT_SHA256SUMS.txt")
    )
    checkpoint_by_key = {
        (record["training_seed"], record["arm"].lower()): record
        for record in canonical["checkpoint_records"]
        if record["id"] in bundle["checkpoint_record_ids"]
    }

    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    for relative, expected_sha in sorted(checksums.items()):
        match = PATH_RE.fullmatch(relative)
        if match is None:
            continue
        start, end, seed, arm, metric = match.groups()
        source_path = f"{metric_root}/{relative}"
        raw_bytes = git_bytes(root, commit, source_path)
        observed_sha = sha256_bytes(raw_bytes)
        if observed_sha != expected_sha:
            raise ValueError(f"metric checksum mismatch: {relative}")
        raw = json.loads(raw_bytes)
        if raw.get("metric") != metric:
            raise ValueError(f"metric name mismatch: {relative}")
        key = (int(start), int(end), int(seed), arm)
        grouped.setdefault(key, []).append(
            {
                "metric": metric,
                "receipt_path": source_path,
                "receipt_sha256": expected_sha,
                "reported_value": raw["results"][metric],
            }
        )

    cells: list[dict[str, Any]] = []
    for (start, end, seed, arm), metrics in sorted(grouped.items()):
        checkpoint = checkpoint_by_key[(seed, arm)]
        snapshot = checkpoint["network_snapshot"]
        snapshot_hash = snapshot["sha256"]
        cells.append(
            {
                "cell_id": f"block_{start}_{end}/seed{seed}/arm_{arm}",
                "training_seed": seed,
                "arm": arm.upper(),
                "sample_binding": {
                    "ordered_sample_seed_range": f"{start}-{end}",
                    "first_seed": start,
                    "last_seed": end,
                    "count": end - start + 1,
                    "ordering": "ascending_inclusive",
                },
                "checkpoint_binding": {
                    "checkpoint_record_id": checkpoint["id"],
                    "run_id": checkpoint["run_id"],
                    "state_id": checkpoint["state_id"],
                    "network_snapshot_filename": snapshot["path"],
                    "network_snapshot_sha256": snapshot_hash,
                    "training_state_sha256": checkpoint["training_state"]["sha256"],
                    "status": (
                        "HASH_BOUND" if snapshot_hash is not None else "BLOCKED_BY_B005"
                    ),
                },
                "metric_receipts": sorted(metrics, key=lambda item: item["metric"]),
            }
        )

    receipt_count = sum(len(cell["metric_receipts"]) for cell in cells)
    hash_bound_receipts = sum(
        len(cell["metric_receipts"])
        for cell in cells
        if cell["checkpoint_binding"]["status"] == "HASH_BOUND"
    )
    blocked_records = sorted(
        {
            cell["checkpoint_binding"]["checkpoint_record_id"]
            for cell in cells
            if cell["checkpoint_binding"]["status"] != "HASH_BOUND"
        }
    )
    return {
        "schema_version": 1,
        "manifest_id": "pr53-disjoint-5k-cell-bindings-v1",
        "generated_at": "2026-08-17T00:00:00+08:00",
        "source_artifact_commit": bundle["artifact_commit"],
        "metric_root": bundle["raw_metric_root"],
        "sample_policy": {
            "range_from_frozen_block_directory": True,
            "ordering": "ascending_inclusive",
            "count_per_cell": 5000,
        },
        "summary": {
            "cells": len(cells),
            "metric_receipts": receipt_count,
            "sample_range_bound_receipts": receipt_count,
            "checkpoint_hash_bound_receipts": hash_bound_receipts,
            "checkpoint_hash_unbound_receipts": receipt_count - hash_bound_receipts,
            "checkpoint_records_blocked_by_b005": blocked_records,
            "b006_status": (
                "HASH_BOUND_54_OF_54"
                if receipt_count == hash_bound_receipts
                else "PARTIAL_BLOCKED_ONLY_BY_B005"
            ),
        },
        "cells": cells,
    }


def write_outputs(output: Path, checksum_output: Path) -> None:
    payload = (json.dumps(build(), indent=2, sort_keys=False) + "\n").encode()
    output.write_bytes(payload)
    checksum_output.write_text(
        f"{sha256_bytes(payload)}  {output.name}\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checksum-output", type=Path, default=CHECKSUM_OUTPUT)
    args = parser.parse_args()
    write_outputs(args.output, args.checksum_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
