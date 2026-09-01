#!/usr/bin/env python3
"""Validate one job and emit a metric-blind SEALED_PASS receipt."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def validate_metric(path: Path, metric: str) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError("metric file must contain one JSON line")
    payload = json.loads(lines[0])
    value = float(payload["results"][metric])
    if payload.get("metric") != metric or not math.isfinite(value) or value < 0:
        raise RuntimeError("invalid sealed metric payload")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--job-cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_NOT_RUN" or frozen.get("job_count") != 60:
        raise RuntimeError("manifest is not frozen 60-job P2 evaluation")
    job = frozen["jobs"][args.job_index]
    if job["job_index"] != args.job_index:
        raise RuntimeError("job index mismatch")
    checkpoint = Path(job["checkpoint_path"]).resolve(strict=True)
    if pulse_chase.sha256_file(checkpoint) != job["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")
    evaluator = Path(frozen["evaluator"]["repo"]).resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != frozen["evaluator"]["commit"]:
        raise RuntimeError("evaluator commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(evaluator), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("evaluator repository is dirty")
    for name, digest in frozen["evaluator"]["code_sha256"].items():
        if pulse_chase.sha256_file(evaluator / name) != digest:
            raise RuntimeError(f"evaluator code hash mismatch: {name}")
    job_cache = args.job_cache.resolve(strict=True)
    for record in frozen["real_features"].values():
        source = Path(record["path"]).resolve(strict=True)
        matches = list(job_cache.rglob(source.name))
        if len(matches) != 1 or pulse_chase.sha256_file(matches[0]) != record["sha256"]:
            raise RuntimeError("job real-feature cache hash mismatch")
    detector = frozen["feature_detector"]
    detector_source = Path(detector["path"]).resolve(strict=True)
    detector_matches = list(job_cache.rglob(detector_source.name))
    if (
        len(detector_matches) != 1
        or pulse_chase.sha256_file(detector_matches[0]) != detector["sha256"]
    ):
        raise RuntimeError("job feature-detector hash mismatch")
    job_dir = args.job_dir.resolve(strict=True)
    required = [
        "training_options.json", "generated-samples.npy",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
        "metric-kid50k_full.jsonl", "metric-fid50k_full.jsonl",
    ]
    artifacts = {}
    for name in required:
        path = job_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing evaluation artifact: {name}")
        artifacts[name] = {"bytes": path.stat().st_size,
                           "sha256": pulse_chase.sha256_file(path)}
    kid_feature = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
    fid_feature = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
    if kid_feature != fid_feature:
        raise RuntimeError("KID/FID did not use byte-identical generated features")
    validate_metric(job_dir / "metric-kid50k_full.jsonl", "kid50k_full")
    validate_metric(job_dir / "metric-fid50k_full.jsonl", "fid50k_full")
    options = json.loads((job_dir / "training_options.json").read_text())
    if options.get("sample_seeds") != list(range(50000)):
        raise RuntimeError("generation seeds are not exactly 0..49999")
    if (
        options.get("seed") != 20260730
        or options.get("network_kwargs", {}).get("use_fp16") is not False
    ):
        raise RuntimeError("metric seed or precision mismatch")
    expected_mid = [] if job["nfe"] == 1 else [0.821]
    if options.get("mid_t") != expected_mid:
        raise RuntimeError("NFE/mid_t mismatch")
    receipt = {
        "schema": "ect.q256.p2-evaluation-sealed-job/v1",
        "status": "SEALED_PASS",
        "job_index": args.job_index,
        "seed": job["seed"], "branch": job["branch"],
        "budget_kimg": job["budget_kimg"], "nfe": job["nfe"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "generated_feature_sha256": kid_feature,
        "kid_fid_shared_feature_identity": True,
        "sealed_metric_artifact_sha256": {
            metric: artifacts[f"metric-{metric}.jsonl"]["sha256"]
            for metric in ("kid50k_full", "fid50k_full")
        },
        "frozen_manifest_sha256": pulse_chase.sha256_file(manifest_path),
        "evaluator_commit": head,
        "runtime_sif_sha256": frozen["runtime_sif"]["sha256"],
        "real_features": frozen["real_features"],
        "feature_detector": frozen["feature_detector"],
        "gpu_index": args.gpu_index, "gpu_uuid": args.gpu_uuid,
        "elapsed_seconds": args.elapsed_seconds,
        "job_dir": str(job_dir),
        "artifacts": artifacts,
        "numeric_results_exposed_in_receipt": False,
    }
    reproducibility.atomic_json_dump(receipt, args.receipt, overwrite=False)
    print(json.dumps({"status": "SEALED_PASS", "job_index": args.job_index,
                      "generated_feature_sha256": kid_feature}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
