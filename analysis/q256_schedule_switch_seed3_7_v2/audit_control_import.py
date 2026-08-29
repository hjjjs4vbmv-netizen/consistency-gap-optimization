#!/usr/bin/env python3
"""Audit PR79/seed6-7 A/B controls before mixed-cohort analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility


SEEDS = (3, 4, 5, 6, 7)
ARMS = ("A", "B")
BUDGETS = (512, 640, 768, 896, 1024)
NFES = (1, 2)
DATASET_SHA = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
FROZEN_EVALUATOR = "d6aba02fb88e9db0993623895eb2228ed717d810"
SEED3_5_EVALUATOR = "c8721a05227f3ff171f8dc1f559a64d58281c0ae"
SEED6_7_EVALUATOR = "9d06ccc72545d4189af1b86de7f629f9c09d3f73"
EXPECTED_REUSE_DIFF_SHA = "4171ce39ed0a5a3b9d6e11928febccb84dadac976297d5f7af84f278e2ab4adb"
EVALUATOR_PATHS = (
    "ct_eval.py", "metrics", "dnnlib", "torch_utils",
    "training/networks.py", "training/dataset.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_receipt(root: Path, seed: int, arm: str, budget: int, nfe: int) -> tuple[Path, dict]:
    matches = []
    for path in root.rglob("*.json"):
        name = path.name
        if (
            f"seed{seed}-arm{arm}-" not in name
            or f"nfe{nfe}" not in name
            or not (f"kimg{budget}" in name or f"k{budget}" in name)
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") == "PASS":
            matches.append((path, payload))
    if not matches:
        raise RuntimeError(
            f"missing PASS control receipt seed={seed} arm={arm} budget={budget} nfe={nfe}"
        )
    identities = {
        (
            item[1].get("checkpoint_sha256"),
            item[1].get("generated_feature_sha256"),
        )
        for item in matches
    }
    if len(identities) != 1:
        raise RuntimeError("duplicate control receipts disagree")
    return matches[0]


def evaluator_diff(repo: Path, left: str, right: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "diff", f"{left}..{right}", "--", *EVALUATOR_PATHS]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed3-5-results", type=Path, required=True)
    parser.add_argument("--seed3-5-receipts", type=Path, required=True)
    parser.add_argument("--seed3-5-frozen-manifest", type=Path, required=True)
    parser.add_argument("--seed3-5-summary", type=Path, required=True)
    parser.add_argument("--seed6-7-results", type=Path, required=True)
    parser.add_argument("--seed6-7-receipts", type=Path, required=True)
    parser.add_argument("--seed6-7-completion", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol"] != "q256_ab_crossed_switch_seed3_7_v2":
        raise RuntimeError("wrong protocol")
    protocol_sha = sha256_file(protocol_path)
    frozen35 = json.loads(args.seed3_5_frozen_manifest.read_text(encoding="utf-8"))
    summary35 = json.loads(args.seed3_5_summary.read_text(encoding="utf-8"))
    completion67 = json.loads(args.seed6_7_completion.read_text(encoding="utf-8"))
    if summary35.get("status") != "PASS" or summary35.get("generated_feature_identity") != "PASS: 168/168 KID/FID pairs byte-identical":
        raise RuntimeError("seed3-5 shared-feature summary is not PASS")
    if completion67.get("status") != "PASS":
        raise RuntimeError("seed6-7 archive completion is not PASS")
    jobs35 = frozen35.get("jobs", [])
    if len(jobs35) != 168 or any(job.get("dataset_sha256") != DATASET_SHA for job in jobs35):
        raise RuntimeError("seed3-5 frozen evaluator dataset identity mismatch")
    if {job.get("evaluator_commit") for job in jobs35} != {SEED3_5_EVALUATOR}:
        raise RuntimeError("seed3-5 evaluator commit mismatch")

    rows35 = read_csv(args.seed3_5_results)
    rows67 = read_csv(args.seed6_7_results)
    controls = []
    for seed in SEEDS:
        rows = rows35 if seed <= 5 else rows67
        receipt_root = args.seed3_5_receipts if seed <= 5 else args.seed6_7_receipts
        for arm in ARMS:
            for budget in BUDGETS:
                for nfe in NFES:
                    matches = [
                        row for row in rows
                        if int(row["seed"]) == seed and row["arm"] == arm
                        and int(row["budget_kimg"]) == budget
                        and int(row["nfe"]) == nfe and row.get("status") == "PASS"
                    ]
                    if len(matches) != 1:
                        raise RuntimeError("control result matrix has missing/duplicate cell")
                    row = matches[0]
                    receipt_path, receipt = find_receipt(
                        receipt_root, seed, arm, budget, nfe
                    )
                    expected_checkpoint = (
                        row["checkpoint_sha256"] if seed <= 5
                        else row["snapshot_sha256"]
                    )
                    if receipt.get("checkpoint_sha256") != expected_checkpoint:
                        raise RuntimeError("control checkpoint SHA mismatch")
                    if receipt.get("generated_feature_sha256") != row["generated_feature_sha256"]:
                        raise RuntimeError("control generated-feature SHA mismatch")
                    if seed <= 5:
                        fid = float(receipt["fid50k_full"])
                        kid = float(receipt["kid50k_full"])
                    else:
                        metrics = receipt["metrics"]
                        fid = float(metrics["fid50k_full"])
                        kid = float(metrics["kid50k_full"])
                        if receipt.get("dataset_sha256") != DATASET_SHA:
                            raise RuntimeError("seed6-7 control dataset mismatch")
                        if receipt.get("evaluator_source_git_head") != SEED6_7_EVALUATOR:
                            raise RuntimeError("seed6-7 evaluator commit mismatch")
                    if fid != float(row["fid50k_full"]) or kid != float(row["kid50k_full"]):
                        raise RuntimeError("control metric/receipt mismatch")
                    controls.append({
                        "seed": seed, "arm": arm, "budget_kimg": budget,
                        "nfe": nfe, "fid50k_full": fid, "kid50k_full": kid,
                        "checkpoint_sha256": expected_checkpoint,
                        "generated_feature_sha256": row["generated_feature_sha256"],
                        "receipt_path": str(receipt_path.resolve()),
                        "receipt_sha256": sha256_file(receipt_path),
                    })
    if len(controls) != 100:
        raise RuntimeError("control import is not exactly 100 cells")

    diff67 = evaluator_diff(repo, SEED6_7_EVALUATOR, FROZEN_EVALUATOR)
    if diff67:
        raise RuntimeError("seed6-7 evaluator numerical source differs from frozen evaluator")
    diff35 = evaluator_diff(repo, SEED3_5_EVALUATOR, FROZEN_EVALUATOR)
    diff35_sha = hashlib.sha256(diff35).hexdigest()
    if diff35_sha != EXPECTED_REUSE_DIFF_SHA:
        raise RuntimeError("seed3-5 evaluator diff is outside the audited reuse-only patch")
    payload = {
        "schema": "ect.q256.schedule-switch-control-import-audit/v1",
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "control_cells": len(controls),
        "dataset_sha256": DATASET_SHA,
        "frozen_evaluator_commit": FROZEN_EVALUATOR,
        "seed6_7_evaluator": {
            "commit": SEED6_7_EVALUATOR,
            "relevant_tree_diff": "empty",
        },
        "seed3_5_evaluator": {
            "commit": SEED3_5_EVALUATOR,
            "relevant_diff_sha256": diff35_sha,
            "difference": "shared generated-feature reuse and artifact copying only",
            "original_generated_feature_identity": summary35["generated_feature_identity"],
            "metric_numerical_semantics_changed": False,
        },
        "controls": controls,
    }
    reproducibility.atomic_json_dump(payload, args.output_json, overwrite=False)
    lines = [
        "# q256 seed3-7 archived-control compatibility audit", "",
        "Status: **PASS**", "",
        f"Protocol SHA256: `{protocol_sha}`", "",
        "- 100/100 A/B control cells joined to PASS receipts.",
        "- Checkpoint, generated-feature, dataset, sampling, and metric values match.",
        "- Seed6-7 evaluator code is identical to the frozen evaluator in all relevant files.",
        "- Seed3-5 differs only by later shared-feature reuse; its original 168/168 jobs already produced byte-identical KID/FID features.",
        "- No control was selected or discarded by quality.", "",
    ]
    with args.output_report.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "controls": len(controls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
