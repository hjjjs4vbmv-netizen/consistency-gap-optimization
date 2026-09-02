#!/usr/bin/env python3
"""Verify downloaded q256 audit artifacts against the server matrix receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.results_root.resolve()
    receipt_path = root / "v2_primary_matrix_validation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_PRIMARY_12_STATE_MATRIX":
        raise ValueError("server matrix receipt is not PASS_PRIMARY_12_STATE_MATRIX")

    verified = []
    for cell in receipt["cells"]:
        artifact_dir = root / "v2_primary" / Path(cell["artifact_dir"]).name
        manifest_path = artifact_dir / "target_component_manifest.json"
        observed_manifest = sha256(manifest_path)
        if observed_manifest != cell["manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch: {artifact_dir.name}")
        observed_artifacts = {}
        for filename, expected in sorted(cell["artifact_sha256"].items()):
            observed = sha256(artifact_dir / filename)
            if observed != expected:
                raise ValueError(f"artifact hash mismatch: {artifact_dir.name}/{filename}")
            observed_artifacts[filename] = observed
        verified.append(
            {
                "cell": artifact_dir.name,
                "manifest_sha256": observed_manifest,
                "artifact_sha256": observed_artifacts,
            }
        )

    implementation_snapshot = {
        "fixed_randomness_helper": root
        / "implementation_snapshot"
        / "analysis"
        / "gap_gradient_hook.py",
        "dataset": root / "implementation_snapshot" / "training" / "dataset.py",
        "networks": root / "implementation_snapshot" / "training" / "networks.py",
    }
    snapshot_hashes = {label: sha256(path) for label, path in implementation_snapshot.items()}
    expected_implementation = receipt["common_implementation_sha256"]
    for label, observed in snapshot_hashes.items():
        if observed != expected_implementation[label]:
            raise ValueError(f"implementation snapshot mismatch: {label}")

    payload = {
        "schema": "ect.q256.target-component-audit-download-integrity/v1",
        "status": "PASS_LOCAL_DOWNLOAD_INTEGRITY",
        "server_receipt_sha256": sha256(receipt_path),
        "verified_cell_count": len(verified),
        "verified_cells": verified,
        "differing_server_implementation_files_preserved": snapshot_hashes,
        "note": (
            "The current local checkout differs from the server execution snapshot in these three files. "
            "Their exact server versions are preserved under implementation_snapshot and match the hashes "
            "bound into every formal manifest."
        ),
    }
    args.out.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
