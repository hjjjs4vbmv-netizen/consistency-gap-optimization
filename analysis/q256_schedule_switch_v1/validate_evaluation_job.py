#!/usr/bin/env python3
"""Validate one frozen q256 crossed-switch KID/FID-50k job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility


EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
RUNTIME_SHA256 = "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
METRICS = ("kid50k_full", "fid50k_full")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric(path: Path, metric: str) -> float:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"metric file must contain one JSON line: {path}")
    payload = json.loads(lines[0])
    if payload.get("metric") != metric or payload.get("num_gpus") != 1:
        raise RuntimeError(f"metric identity mismatch: {path}")
    value = float(payload["results"][metric])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid metric value: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--runtime-sif", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    args = parser.parse_args()
    frozen_path = args.evaluation_manifest.resolve(strict=True)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("metrics_executed") is not False or frozen.get("job_count") != 80:
        raise RuntimeError("evaluation manifest is not the frozen 80-job matrix")
    job = frozen["jobs"][args.job_index]
    if job["job_index"] != args.job_index:
        raise RuntimeError("evaluation job index mismatch")
    job_dir = args.job_dir.resolve(strict=True)
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise RuntimeError("refuse existing evaluation receipt")
    checkpoint = Path(job["checkpoint_path"]).resolve(strict=True)
    dataset = Path(job["dataset_path"]).resolve(strict=True)
    evaluator_repo = args.evaluator_repo.resolve(strict=True)
    runtime_sif = args.runtime_sif.resolve(strict=True)
    runtime_receipt_path = args.runtime_receipt.resolve(strict=True)
    runtime_receipt = json.loads(runtime_receipt_path.read_text(encoding="utf-8"))
    if sha256_file(checkpoint) != job["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("dataset SHA256 mismatch")
    if (runtime_receipt.get("status") != "PASS"
            or runtime_receipt.get("runtime_sif_sha256") != RUNTIME_SHA256
            or Path(runtime_receipt.get("runtime_sif", "")).resolve()
            != runtime_sif):
        raise RuntimeError("runtime SIF SHA256 mismatch")
    evaluator_head = subprocess.check_output(
        ["git", "-C", str(evaluator_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if evaluator_head != EVALUATOR_COMMIT:
        raise RuntimeError("evaluator source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(evaluator_repo), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("evaluator source is dirty")
    required = (
        "log.txt", "training_options.json", "generated-samples.npy",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
        "metric-kid50k_full.jsonl", "metric-fid50k_full.jsonl",
    )
    artifacts = {}
    for name in required:
        path = job_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing evaluation artifact: {path}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    kid_feature_sha = artifacts[
        "generated-features-kid50k_full-repeat00.npy"
    ]["sha256"]
    fid_feature_sha = artifacts[
        "generated-features-fid50k_full-repeat00.npy"
    ]["sha256"]
    if kid_feature_sha != fid_feature_sha:
        raise RuntimeError("KID/FID generated features are not byte-identical")
    options = json.loads((job_dir / "training_options.json").read_text())
    expected_options = {
        "batch_size": 512,
        "metrics": list(METRICS),
        "metric_repeats": 1,
        "metric_generator_batch": 128,
        "retain_generated_artifacts": True,
        "seed": 20260730,
    }
    for key, expected in expected_options.items():
        if options.get(key) != expected:
            raise RuntimeError(f"evaluation option mismatch: {key}")
    if options.get("sample_seeds") != list(range(50000)):
        raise RuntimeError("sample seeds are not exactly 0..49999")
    if options.get("mid_t") != ([] if job["nfe"] == 1 else [0.821]):
        raise RuntimeError("NFE/mid_t mismatch")
    if Path(options["resume_pkl"]).resolve() != checkpoint:
        raise RuntimeError("evaluation checkpoint binding mismatch")
    if Path(options["dataset_kwargs"]["path"]).resolve() != dataset:
        raise RuntimeError("evaluation dataset binding mismatch")
    metrics = {
        metric: read_metric(job_dir / f"metric-{metric}.jsonl", metric)
        for metric in METRICS
    }
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    matching = [row.strip() for row in gpu_rows if args.gpu_uuid in row]
    if len(matching) != 1 or "A100" not in matching[0]:
        raise RuntimeError("GPU identity mismatch")
    payload = {
        "schema": "ect.q256.schedule-switch-evaluation-job/v1",
        "status": "PASS",
        **{key: job[key] for key in (
            "job_index", "seed", "branch", "origin_arm", "continuation_arm",
            "budget_kimg", "nfe", "protocol_sha256",
        )},
        "mid_t": None if job["nfe"] == 1 else 0.821,
        "sample_count": 50000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20260730,
        "precision": "fp32",
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": job["checkpoint_sha256"],
        "generated_feature_sha256": kid_feature_sha,
        "kid_fid_shared_feature_identity": True,
        "dataset": str(dataset),
        "dataset_sha256": DATASET_SHA256,
        "evaluator_source": str(evaluator_repo),
        "evaluator_commit": EVALUATOR_COMMIT,
        "runtime_sif": str(runtime_sif),
        "runtime_sif_sha256": RUNTIME_SHA256,
        "runtime_integrity_receipt": str(runtime_receipt_path),
        "runtime_integrity_receipt_sha256": sha256_file(runtime_receipt_path),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "gpu_identity_row": matching[0],
        "elapsed_seconds": args.elapsed_seconds,
        "job_dir": str(job_dir),
        "artifacts": artifacts,
        "frozen_evaluation_manifest_sha256": sha256_file(frozen_path),
    }
    reproducibility.atomic_json_dump(payload, receipt, overwrite=False)
    print(json.dumps({"status": "PASS", "job_index": args.job_index,
                      "metrics": metrics, "feature_sha256": kid_feature_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
