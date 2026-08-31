#!/usr/bin/env python3
"""Gate unsealing on all 60 metric-blind SEALED_PASS receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = pulse_chase.sha256_file(manifest_path)
    paths = sorted(args.receipts.resolve(strict=True).glob("*.json"))
    if any("FAILED" in path.name for path in paths):
        raise RuntimeError("failed evaluation receipt present")
    receipts = [json.loads(path.read_text()) for path in paths]
    failures = []
    if len(receipts) != 60:
        failures.append(f"expected 60 receipts, found {len(receipts)}")
    indices = []
    features = []
    for receipt in receipts:
        indices.append(receipt.get("job_index"))
        features.append({"job_index": receipt.get("job_index"),
                         "generated_feature_sha256": receipt.get("generated_feature_sha256")})
        if (
            receipt.get("status") != "SEALED_PASS"
            or receipt.get("frozen_manifest_sha256") != manifest_sha
            or receipt.get("kid_fid_shared_feature_identity") is not True
            or receipt.get("numeric_results_exposed_in_receipt") is not False
        ):
            failures.append(f"invalid seal receipt job {receipt.get('job_index')}")
    if sorted(indices) != list(range(60)):
        failures.append("receipt indices are not exactly 0..59")
    payload = {
        "schema": "ect.q256.p2-evaluation-seal-audit/v1",
        "status": "ALL_60_SEALED_PASS" if not failures else "FAIL_CLOSED",
        "unseal_authorized": not failures,
        "manifest_sha256": manifest_sha,
        "receipt_count": len(receipts),
        "generated_feature_hashes": sorted(features, key=lambda x: x["job_index"]),
        "failures": failures,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "receipt_count": len(receipts)}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
