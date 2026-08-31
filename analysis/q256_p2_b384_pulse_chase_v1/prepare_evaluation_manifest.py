#!/usr/bin/env python3
"""Freeze the exact 60-job P2 evaluation matrix after 20/20 training PASS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--training-integrity", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-sif", type=Path, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--kid-real-features", type=Path, required=True)
    parser.add_argument("--fid-real-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.formal_root.resolve(strict=True)
    integrity_path = args.training_integrity.resolve(strict=True)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    protocol_path = args.protocol.resolve(strict=True)
    protocol_sha = pulse_chase.sha256_file(protocol_path)
    if (
        integrity.get("status") != "PASS"
        or integrity.get("source_count") != 10
        or integrity.get("branch_count") != 20
        or integrity.get("protocol_sha256") != protocol_sha
    ):
        raise RuntimeError("training integrity is not exact 10-source/20-branch PASS")
    dataset = args.dataset.resolve(strict=True)
    runtime = args.runtime_sif.resolve(strict=True)
    if pulse_chase.sha256_file(dataset) != pulse_chase.ASSET_SHA256["dataset"]:
        raise RuntimeError("evaluation dataset hash mismatch")
    if pulse_chase.sha256_file(runtime) != pulse_chase.ASSET_SHA256["runtime_sif"]:
        raise RuntimeError("evaluation runtime hash mismatch")
    evaluator = args.evaluator_repo.resolve(strict=True)
    if subprocess.check_output(
        ["git", "-C", str(evaluator), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("evaluator repository is dirty")
    evaluator_commit = subprocess.check_output(
        ["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True
    ).strip()
    code_files = [
        evaluator / "ct_eval.py",
        evaluator / "metrics" / "metric_utils.py",
        evaluator / "metrics" / "frechet_inception_distance.py",
        evaluator / "metrics" / "kernel_inception_distance.py",
    ]
    evaluator_hashes = {
        str(path.relative_to(evaluator)): pulse_chase.sha256_file(path)
        for path in code_files
    }
    real_features = {
        "kid50k_full": {
            "path": str(args.kid_real_features.resolve(strict=True)),
            "sha256": pulse_chase.sha256_file(
                args.kid_real_features.resolve(strict=True)
            ),
        },
        "fid50k_full": {
            "path": str(args.fid_real_features.resolve(strict=True)),
            "sha256": pulse_chase.sha256_file(
                args.fid_real_features.resolve(strict=True)
            ),
        },
    }
    jobs = []
    for seed in pulse_chase.SEEDS:
        for branch in pulse_chase.BRANCHES:
            branch_root = root / "seeds" / f"seed{seed}" / branch
            completion = json.loads(
                (branch_root / "trajectory_completion_receipt.json").read_text()
            )
            if completion.get("status") != "PASS":
                raise RuntimeError(f"training branch not PASS: {branch_root}")
            endpoints = {item["kimg"]: item for item in completion["endpoints"]}
            for budget, nfe in ((512, 1), (640, 1), (640, 2)):
                checkpoint = endpoints[budget]["ema_snapshot"]
                jobs.append({
                    "job_index": len(jobs),
                    "seed": seed,
                    "branch": branch,
                    "budget_kimg": budget,
                    "nfe": nfe,
                    "mid_t": None if nfe == 1 else 0.821,
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "sample_count": 50000,
                    "generation_seed_start": 0,
                    "generation_seed_end": 49999,
                    "metric_seed": 20260730,
                    "precision": "fp32",
                    "metrics_in_order": ["kid50k_full", "fid50k_full"],
                    "shared_generated_features_required": True,
                    "status": "FROZEN_NOT_RUN",
                })
    expected = {
        (seed, branch, budget, nfe)
        for seed in pulse_chase.SEEDS for branch in pulse_chase.BRANCHES
        for budget, nfe in ((512, 1), (640, 1), (640, 2))
    }
    actual = {(j["seed"], j["branch"], j["budget_kimg"], j["nfe"])
              for j in jobs}
    if len(jobs) != 60 or actual != expected:
        raise RuntimeError("evaluation matrix is not exactly 60 unique jobs")
    payload = {
        "schema": "ect.q256.p2-frozen-evaluation-manifest/v1",
        "status": "FROZEN_NOT_RUN",
        "results_unviewed": True,
        "job_count": 60,
        "protocol_sha256": protocol_sha,
        "training_integrity_path": str(integrity_path),
        "training_integrity_sha256": pulse_chase.sha256_file(integrity_path),
        "dataset": {"path": str(dataset), "sha256": pulse_chase.sha256_file(dataset)},
        "runtime_sif": {"path": str(runtime), "sha256": pulse_chase.sha256_file(runtime)},
        "evaluator": {"repo": str(evaluator), "commit": evaluator_commit,
                      "code_sha256": evaluator_hashes},
        "real_features": real_features,
        "jobs": jobs,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": "FROZEN_NOT_RUN", "job_count": 60,
                      "manifest_sha256": pulse_chase.sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
