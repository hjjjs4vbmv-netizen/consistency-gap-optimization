#!/usr/bin/env python3
"""Build compact final compute/provenance manifests after analysis PASS."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    analysis = json.loads((root / "results" / "analysis.json").read_text())
    seal = json.loads((root / "evaluation" / "evaluation_seal_audit.json").read_text())
    training = json.loads((root / "training" / "training_integrity_report.json").read_text())
    if analysis.get("status") != "PASS" or seal.get("status") != "ALL_60_SEALED_PASS" or training.get("status") != "PASS":
        raise RuntimeError("experiment is not complete and auditable")
    compute = []
    with (root / "training" / "compute_cost.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            compute.append({"phase": "training", **row})
    for path in sorted((root / "evaluation" / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        compute.append({
            "phase": "evaluation", "seed": receipt["seed"],
            "cell": f"{receipt['branch']}-kimg{receipt['budget_kimg']}-nfe{receipt['nfe']}",
            "gpu_index": receipt["gpu_index"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "gpu_hours": receipt["elapsed_seconds"] / 3600,
        })
    compute_path = root / "compute_cost.csv"
    with compute_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compute[0]))
        writer.writeheader(); writer.writerows(compute)
    command_path = root / "REPRODUCE_COMMAND.txt"
    if not command_path.is_file():
        raise RuntimeError("missing exact reproduction command")
    include_suffixes = {".json", ".csv", ".md", ".txt"}
    excluded_names = {"SHA256SUMS.txt"}
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in include_suffixes
        and path.name not in excluded_names and "logs" not in path.parts
        and not path.name.endswith(".process.log")
    ]
    sums = "".join(
        f"{pulse_chase.sha256_file(path)}  {path.relative_to(root)}\n"
        for path in sorted(files)
    )
    (root / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest_files": len(files),
                      "compute_rows": len(compute)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
