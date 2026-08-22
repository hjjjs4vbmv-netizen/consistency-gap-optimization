#!/usr/bin/env python3
"""Validate one frozen q256 FID/KID job and publish an immutable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


EXPECTED_DATASET_SHA256 = (
    "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
)
EXPECTED_SIF_SHA256 = (
    "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6"
)
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
METRICS = ("kid50k_full", "fid50k_full")


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def read_metric(path: Path, metric: str) -> tuple[float, dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        fail(f"metric output must contain one JSON line: {path}")
    value = json.loads(lines[0])
    if value.get("metric") != metric or value.get("num_gpus") != 1:
        fail(f"metric identity mismatch: {path}")
    number = float(value["results"][metric])
    if not math.isfinite(number) or number < 0:
        fail(f"non-finite or negative metric: {path}")
    return number, value


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--budget-kimg", type=int, required=True)
    parser.add_argument("--nfe", type=int, choices=(1, 2), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    args = parser.parse_args()

    job_dir = args.job_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    repo = args.repo.resolve(strict=True)
    source_archive = args.source_archive.resolve(strict=True)
    sif = args.sif.resolve(strict=True)
    runtime_receipt = load_json(args.runtime_receipt.resolve(strict=True))
    if args.receipt.exists():
        fail(f"refuse existing receipt: {args.receipt}")
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        fail("checkpoint SHA256 mismatch")
    if sha256_file(dataset) != EXPECTED_DATASET_SHA256:
        fail("dataset SHA256 mismatch")
    if (
        runtime_receipt.get("status") != "PASS"
        or runtime_receipt.get("runtime_sif_sha256") != EXPECTED_SIF_SHA256
        or Path(runtime_receipt.get("runtime_sif", "")).resolve() != sif
    ):
        fail("runtime integrity receipt mismatch")
    portability_gate = None
    if not args.calibration:
        portability_gate = load_json(args.portability_gate.resolve(strict=True))
        if (
            portability_gate.get("status") != "PASS"
            or portability_gate.get("verdict")
            != "bit_exact_generated_features_and_metrics"
            or portability_gate.get("metric_numerical_semantics_changed") is not False
        ):
            fail("A100 40GB portability gate is not PASS")
    if sha256_file(source_archive) != EXPECTED_SOURCE_ARCHIVE_SHA256:
        fail("evaluator source archive hash mismatch")
    subprocess.run(
        ["tar", "-df", str(source_archive), "-C", str(repo)], check=True
    )

    required = [
        "log.txt",
        "training_options.json",
        "generated-samples.npy",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
        "metric-kid50k_full.jsonl",
        "metric-fid50k_full.jsonl",
    ]
    artifacts = {}
    for name in required:
        artifact = job_dir / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size <= 0:
            fail(f"missing evaluation artifact: {artifact}")
        artifacts[name] = {
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    if artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"] != artifacts[
        "generated-features-fid50k_full-repeat00.npy"
    ]["sha256"]:
        fail("KID/FID generated features are not byte-identical")
    if not (job_dir / "log.txt").read_text(
        encoding="utf-8", errors="replace"
    ).rstrip().endswith("Exiting..."):
        fail("evaluation log lacks terminal Exiting marker")

    options = load_json(job_dir / "training_options.json")
    exact = {
        "batch_size": 512,
        "metrics": list(METRICS),
        "metric_repeats": 1,
        "metric_generator_batch": 128,
        "retain_generated_artifacts": True,
        "seed": 20_260_730,
    }
    for key, expected in exact.items():
        if options.get(key) != expected:
            fail(f"evaluation option mismatch for {key}")
    if options.get("sample_seeds") != list(range(50_000)):
        fail("sample seeds are not exactly 0..49999")
    if options.get("mid_t") != ([] if args.nfe == 1 else [0.821]):
        fail("NFE/mid_t mismatch")
    if Path(options["resume_pkl"]).resolve() != checkpoint:
        fail("evaluation checkpoint binding mismatch")
    if Path(options["dataset_kwargs"]["path"]).resolve() != dataset:
        fail("evaluation dataset binding mismatch")

    values = {}
    for metric in METRICS:
        values[metric], _ = read_metric(job_dir / f"metric-{metric}.jsonl", metric)
    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpu_rows = [row.strip() for row in gpu_output.splitlines() if args.gpu_uuid in row]
    if len(gpu_rows) != 1 or "A100" not in gpu_rows[0]:
        fail("selected GPU identity is unavailable")

    payload = {
        "schema": "ect.q256.seed14-18.streaming-evaluation-job/v1",
        "status": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": "secondary_precision_extension_portability_runtime",
        "metric_numerical_semantics_changed": False,
        "seed": args.seed,
        "arm": args.arm,
        "budget_kimg": args.budget_kimg,
        "nfe": args.nfe,
        "mid_t": None if args.nfe == 1 else 0.821,
        "sample_count": 50_000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20_260_730,
        "precision": "fp32",
        "metrics": values,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "dataset": str(dataset),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "evaluator_source": str(repo),
        "evaluator_source_commit": EVALUATOR_COMMIT,
        "evaluator_source_archive": str(source_archive),
        "evaluator_source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "runtime_sif": str(sif),
        "runtime_sif_sha256": EXPECTED_SIF_SHA256,
        "runtime_integrity_receipt": str(args.runtime_receipt.resolve()),
        "runtime_integrity_receipt_sha256": sha256_file(args.runtime_receipt),
        "portability_gate": (
            None if portability_gate is None else str(args.portability_gate.resolve())
        ),
        "portability_gate_sha256": (
            None if portability_gate is None else sha256_file(args.portability_gate)
        ),
        "gpu_uuid": args.gpu_uuid,
        "gpu_identity_row": gpu_rows[0],
        "elapsed_seconds": args.elapsed_seconds,
        "generated_feature_sha256": artifacts[
            "generated-features-fid50k_full-repeat00.npy"
        ]["sha256"],
        "job_dir": str(job_dir),
        "artifacts": artifacts,
    }
    write_exclusive(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"[q256-stream-validate] ERROR: {exc}") from exc
