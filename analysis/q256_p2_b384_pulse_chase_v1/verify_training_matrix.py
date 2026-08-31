#!/usr/bin/env python3
"""Seal the exact 10-source/20-branch formal P2 training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-cost", type=Path, required=True)
    args = parser.parse_args()
    root = args.formal_root.resolve(strict=True)
    protocol_sha = pulse_chase.sha256_file(args.protocol.resolve(strict=True))
    sources, branches, pairs, costs = [], [], [], []
    failures = []
    for seed in pulse_chase.SEEDS:
        seed_dir = root / "seeds" / f"seed{seed}"
        source = json.loads((seed_dir / "source" / "source_inventory.json").read_text())
        pair = json.loads((seed_dir / "pair_integrity_receipt.json").read_text())
        sources.append(source)
        pairs.append(pair)
        if source.get("status") != "PASS" or source.get("protocol_sha256") != protocol_sha:
            failures.append(f"seed{seed} source is not protocol-bound PASS")
        if pair.get("status") != "PASS":
            failures.append(f"seed{seed} pair integrity is not PASS")
        source_cost = json.loads(
            (seed_dir / "source" / "source_completion_receipt.json").read_text()
        )
        costs.append({"seed": seed, "cell": "source", "gpu_index": source_cost["gpu_index"],
                      "elapsed_seconds": source_cost["elapsed_seconds"],
                      "gpu_hours": source_cost["elapsed_seconds"] / 3600})
        for branch_name in pulse_chase.BRANCHES:
            receipt = json.loads(
                (seed_dir / branch_name / "trajectory_completion_receipt.json").read_text()
            )
            branches.append(receipt)
            if receipt.get("status") != "PASS" or receipt.get("protocol_sha256") != protocol_sha:
                failures.append(f"seed{seed} {branch_name} is not PASS")
            cost = json.loads(
                (seed_dir / branch_name / "compute_cost_receipt.json").read_text()
            )
            costs.append({"seed": seed, "cell": branch_name,
                          "gpu_index": receipt["gpu_index"],
                          "elapsed_seconds": cost["elapsed_seconds"],
                          "gpu_hours": cost["elapsed_seconds"] / 3600})
    if len(sources) != 10 or len(branches) != 20 or len(pairs) != 10:
        failures.append("formal training matrix cardinality mismatch")
    payload = {
        "schema": "ect.q256.p2-training-integrity/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "protocol_sha256": protocol_sha,
        "source_count": len(sources),
        "branch_count": len(branches),
        "pair_count": len(pairs),
        "seeds": list(pulse_chase.SEEDS),
        "sources": sources,
        "branches": branches,
        "pairs": pairs,
        "failures": failures,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    with args.compute_cost.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(costs[0]))
        writer.writeheader(); writer.writerows(costs)
    print(json.dumps({"status": payload["status"], "sources": len(sources),
                      "branches": len(branches)}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
