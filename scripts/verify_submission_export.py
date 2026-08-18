#!/usr/bin/env python3
"""Self-contained verifier for the history-free anonymous submission export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_file(root: Path, filename: str) -> int:
    checksum_path = root / filename
    count = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        if SHA256_RE.fullmatch(expected) is None:
            raise RuntimeError(f"invalid digest in {checksum_path}: {expected}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"checksum mismatch: {path}")
        count += 1
    return count


def verify_lightweight(root: Path) -> dict[str, int]:
    release_files = verify_checksum_file(root, "RELEASE_SHA256SUMS")
    bundle = root / "results" / "publication_v2_regenerated"
    bundle_files = verify_checksum_file(bundle, "SHA256SUMS")
    manifest = json.loads((bundle / "publication_v2_cell_manifest.json").read_text())
    matrix = manifest["matrix"]
    expected = {
        "cells": 27,
        "metric_receipts": 54,
        "checkpoint_hash_bound_receipts": 54,
        "sample_range_bound_receipts": 54,
        "retained_sample_arrays": 27,
        "retained_feature_arrays": 54,
        "retry_count": 0,
    }
    if manifest.get("status") != "PASS" or any(matrix.get(k) != v for k, v in expected.items()):
        raise RuntimeError("publication-v2 manifest accounting mismatch")
    if len(manifest.get("cells", [])) != 27:
        raise RuntimeError("publication-v2 manifest does not contain 27 cells")
    cell_manifest = root / "evidence" / "disjoint_5k_cell_manifest_v1.json"
    checksum_line = (root / "evidence" / "disjoint_5k_cell_manifest_v1.sha256").read_text().strip()
    expected_cell_sha, expected_name = checksum_line.split("  ", maxsplit=1)
    if expected_name != cell_manifest.name or sha256_file(cell_manifest) != expected_cell_sha:
        raise RuntimeError("historical cell-manifest detached checksum mismatch")
    historical = json.loads(cell_manifest.read_text())
    if len(historical.get("cells", [])) != 27:
        raise RuntimeError("historical cell manifest does not contain 27 cells")
    if sum(len(cell.get("metric_receipts", [])) for cell in historical["cells"]) != 54:
        raise RuntimeError("historical cell manifest does not contain 54 receipts")
    recovery = json.loads((root / "evidence" / "b005_recovery_receipt.json").read_text())
    if recovery.get("status") != "PASS" or len(recovery.get("recovered_checkpoints", [])) != 2:
        raise RuntimeError("B005 recovery receipt failed")
    balanced_root = root / "analysis" / "balanced_beta"
    balanced = json.loads((balanced_root / "summary.json").read_text(encoding="utf-8"))
    balanced_arrays = 0
    for label in ("k32", "k64", "k128", "k256"):
        configs = balanced.get(label, {}).get("configs", {})
        for config in ("standard", "balanced_0.9", "balanced_0.99", "balanced_0.999"):
            if config not in configs:
                raise RuntimeError(f"missing balanced-beta cell: {label}/{config}")
            path = balanced_root / label / "raw_h20" / f"{config}_h_actual.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            row = next(
                item for item in configs[config]["horizons"]
                if item["horizon_steps"] == 20
            )
            if array.ndim != 1 or array.shape[0] != row["effective_coords"]:
                raise RuntimeError(f"balanced-beta h20 shape mismatch: {label}/{config}")
            balanced_arrays += 1
    provenance = json.loads((balanced_root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("git", {}).get("repository") != "anonymous-submission-repository":
        raise RuntimeError("balanced-beta anonymous provenance was not sanitized")
    return {
        "release_files": release_files,
        "bundle_files": bundle_files,
        "balanced_beta_arrays": balanced_arrays,
    }


def verify_data(root: Path, data_root: Path) -> dict[str, int]:
    manifest = json.loads(
        (root / "results" / "publication_v2_regenerated" / "publication_v2_cell_manifest.json").read_text()
    )
    arrays = 0
    receipts = 0
    for cell in manifest["cells"]:
        sample = cell["artifacts"]["samples"]
        feature_records = cell["artifacts"]["features"].values()
        for record in (sample, *feature_records):
            path = data_root / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"data hash mismatch: {record['path']}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
                raise RuntimeError(f"data shape/dtype mismatch: {record['path']}")
            arrays += 1
        for metric, record in cell["artifacts"]["metric_receipts"].items():
            path = data_root / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"receipt hash mismatch: {record['path']}")
            rows = path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(rows[0]) if len(rows) == 1 else {}
            if payload.get("metric") != metric or metric not in payload.get("results", {}):
                raise RuntimeError(f"invalid metric receipt: {record['path']}")
            receipts += 1
    if arrays != 81 or receipts != 54:
        raise RuntimeError("data-plane accounting mismatch")
    return {"arrays": arrays, "receipts": receipts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = verify_lightweight(root)
    if args.data_root is not None:
        result.update(verify_data(root, args.data_root.resolve()))
    elif args.require_data:
        raise RuntimeError("--require-data requires --data-root")
    print(json.dumps({"status": "PASS", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
