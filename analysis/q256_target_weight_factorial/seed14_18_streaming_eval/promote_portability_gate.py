#!/usr/bin/env python3
"""Promote an exact known-result replay into the A100-40GB portability gate."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import time
from pathlib import Path


ROOT = Path("/root/q256_eval")
OUTPUT = ROOT / "portability_gate.json"
EXPECTED = {
    "A": {
        "checkpoint_sha256": "e39c197cad28c9b7cb7028aba303fde842f566935ede6da0dc9899684899ba89",
        "generated_feature_sha256": "18e864eb591183b35523a0985c956806c1747fffa3c2c5b766d1df2fa71118e3",
        "fid50k_full": 9.669405820297698,
        "kid50k_full": 0.005999107735235234,
    },
    "B": {
        "checkpoint_sha256": "862bb0262706421e8d06c26215b7904b4f95d8a0c90d7eebd2b849cb5bea0d76",
        "generated_feature_sha256": "38faa1894312225304340585e215001616b371ace1a2388092b44a1c855d6416",
        "fid50k_full": 8.686351455868838,
        "kid50k_full": 0.005436252860360346,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    args = parser.parse_args()
    expected = EXPECTED[args.arm]
    receipt_path = ROOT / "receipts" / f"seed3-arm{args.arm}-k1024-nfe1.json"
    if OUTPUT.exists():
        raise SystemExit(f"refuse existing gate: {OUTPUT}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("seed") != 3
        or receipt.get("arm") != args.arm
        or receipt.get("nfe") != 1
    ):
        raise SystemExit("calibration receipt identity mismatch")
    for key in ("checkpoint_sha256", "generated_feature_sha256"):
        if receipt.get(key) != expected[key]:
            raise SystemExit(f"portability hash mismatch for {key}")
    for metric in ("fid50k_full", "kid50k_full"):
        if not math.isclose(
            float(receipt["metrics"][metric]), expected[metric], rel_tol=0, abs_tol=1e-12
        ):
            raise SystemExit(f"portability metric mismatch for {metric}")
    payload = {
        "schema": "ect.q256.a100-40gb-portability-gate/v1",
        "status": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_arm": args.arm,
        "calibration_receipt": str(receipt_path),
        "calibration_receipt_sha256": sha256_file(receipt_path),
        **expected,
        "verdict": "bit_exact_generated_features_and_metrics",
        "metric_numerical_semantics_changed": False,
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, OUTPUT)
    temporary.unlink()
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
