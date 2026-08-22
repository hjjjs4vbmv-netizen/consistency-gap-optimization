#!/usr/bin/env python3
"""Freeze, but do not execute, the replay learning-curve metric matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
EXPECTED_ARMS = ("A", "B", "C", "D")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, action="append", required=True
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--fid-reference-sha256", required=True)
    parser.add_argument("--kid-reference-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    inventory = []
    for path in args.inventory:
        with path.open(newline="", encoding="utf-8") as handle:
            inventory.extend(csv.DictReader(handle))
    assert len(inventory) == 84
    keys = {
        (int(row["seed"]), row["arm"], int(row["budget_kimg"]))
        for row in inventory
    }
    expected = {
        (seed, arm, budget)
        for seed in (3, 4, 5)
        for arm in EXPECTED_ARMS
        for budget in EXPECTED_BUDGETS
    }
    assert keys == expected
    assert all(row["status"] == "PASS" for row in inventory)

    jobs = []
    for row in sorted(
        inventory,
        key=lambda item: (
            int(item["seed"]),
            item["arm"],
            int(item["budget_kimg"]),
        ),
    ):
        seed = int(row["seed"])
        arm = row["arm"]
        budget = int(row["budget_kimg"])
        snapshot = (
            args.archive_root
            / "runs/q256-target-weight-replay-curve-v1"
            / f"seed{seed}"
            / f"arm{arm}"
            / f"network-snapshot-kimg{budget:06d}.pkl"
        )
        for phase, nfe, mid_t in (
            ("primary_nfe1", 1, []),
            ("secondary_nfe2", 2, [0.821]),
        ):
            jobs.append(
                {
                    "job_index": len(jobs),
                    "phase": phase,
                    "seed": seed,
                    "arm": arm,
                    "budget_kimg": budget,
                    "nfe": nfe,
                    "mid_t": json.dumps(mid_t),
                    "checkpoint_path": str(snapshot),
                    "checkpoint_sha256": row["ema_snapshot_sha256"],
                    "metrics": "kid50k_full,fid50k_full",
                    "sample_count": 50_000,
                    "sample_seed_start": 0,
                    "sample_seed_end": 49_999,
                    "metric_seed": 20_260_730,
                    "evaluation_precision": "fp32",
                    "evaluation_batch": 512,
                    "metric_generator_batch": 128,
                    "kid_subset_size": 1000,
                    "kid_num_subsets": 100,
                    "kid_display_scale": 1.0,
                    "dataset_sha256": args.dataset_sha256,
                    "detector_sha256": args.detector_sha256,
                    "fid_reference_sha256": args.fid_reference_sha256,
                    "kid_reference_sha256": args.kid_reference_sha256,
                    "evaluator_commit": args.evaluator_commit,
                    "status": "FROZEN_NOT_RUN",
                }
            )

    assert len(jobs) == 168
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    payload = {
        "schema": "ect.q256.replay-curve-frozen-evaluation/v1",
        "training_complete_required": True,
        "metrics_executed": False,
        "primary": {
            "phase": "primary_nfe1",
            "job_count": 84,
            "nfe": 1,
        },
        "secondary": {
            "phase": "secondary_nfe2",
            "job_count": 84,
            "nfe": 2,
            "mid_t": [0.821],
        },
        "jobs": jobs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("EVALUATION_PLAN_PASS jobs=168 metrics_executed=false")


if __name__ == "__main__":
    main()
