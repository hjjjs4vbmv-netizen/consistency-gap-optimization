#!/usr/bin/env python3
"""Freeze the exact 80-job q256 crossed-switch evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


SEEDS = tuple(range(14, 19))
BRANCHES = ("A_to_B", "B_to_A")
BUDGETS = (640, 768, 896, 1024)
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    protocol_sha = sha256_file(protocol_path)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("canonical evaluation dataset SHA256 mismatch")
    jobs = []
    for seed in SEEDS:
        for branch in BRANCHES:
            run_dir = formal_root / f"seed{seed}" / branch
            completion_path = run_dir / "trajectory_completion_receipt.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("status") != "PASS":
                raise RuntimeError(f"training cell is not PASS: {run_dir}")
            manifest = schedule_switch.load_run_manifest(
                run_dir / "formal_run_manifest.json"
            )
            if manifest["protocol_sha256"] != protocol_sha:
                raise RuntimeError(f"training protocol mismatch: {run_dir}")
            for budget in BUDGETS:
                checkpoint = run_dir / f"kimg{budget:04d}" / "network-snapshot.pkl"
                if not checkpoint.is_file() or checkpoint.is_symlink():
                    raise RuntimeError(f"missing evaluation snapshot: {checkpoint}")
                checkpoint_sha = sha256_file(checkpoint)
                for nfe in (1, 2):
                    jobs.append({
                        "job_index": len(jobs),
                        "seed": seed,
                        "branch": branch,
                        "origin_arm": manifest["origin_arm"],
                        "continuation_arm": manifest["continuation_arm"],
                        "budget_kimg": budget,
                        "nfe": nfe,
                        "mid_t": "" if nfe == 1 else "0.821",
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_sha,
                        "dataset_path": str(dataset),
                        "dataset_sha256": DATASET_SHA256,
                        "sample_count": 50000,
                        "sample_seed_start": 0,
                        "sample_seed_end": 49999,
                        "metric_seed": 20260730,
                        "metrics_in_order": "kid50k_full,fid50k_full",
                        "precision": "fp32",
                        "evaluator_commit": EVALUATOR_COMMIT,
                        "protocol_sha256": protocol_sha,
                        "status": "FROZEN_NOT_RUN",
                    })
    expected = {
        (seed, branch, budget, nfe)
        for seed in SEEDS for branch in BRANCHES
        for budget in BUDGETS for nfe in (1, 2)
    }
    actual = {
        (job["seed"], job["branch"], job["budget_kimg"], job["nfe"])
        for job in jobs
    }
    if len(jobs) != 80 or actual != expected:
        raise RuntimeError("evaluation matrix is not exactly 80 unique jobs")
    payload = {
        "schema": "ect.q256.schedule-switch-frozen-evaluation/v1",
        "status": "FROZEN_NOT_RUN",
        "job_count": 80,
        "protocol_sha256": protocol_sha,
        "evaluator_commit": EVALUATOR_COMMIT,
        "dataset_sha256": DATASET_SHA256,
        "metrics_executed": False,
        "jobs": jobs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    reproducibility.atomic_json_dump(payload, args.output_json, overwrite=False)
    with args.output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    print(json.dumps({"status": "FROZEN_NOT_RUN", "job_count": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
