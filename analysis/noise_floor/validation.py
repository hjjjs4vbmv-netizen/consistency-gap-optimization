"""Validate retained evaluation artifacts and resumable PASS receipts."""

import hashlib
import json
import math
from pathlib import Path


METRICS = ("kid50k_full", "fid50k_full")
REQUIRED = (
    "log.txt",
    "training_options.json",
    "generated-samples.npy",
    "generated-features-kid50k_full-repeat00.npy",
    "generated-features-fid50k_full-repeat00.npy",
    "metric-kid50k_full.jsonl",
    "metric-fid50k_full.jsonl",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric(path, metric):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("metric file does not contain one row: {}".format(path))
    payload = json.loads(lines[0])
    value = float(payload["results"][metric])
    if (payload.get("metric") != metric or payload.get("num_gpus") != 1
            or not math.isfinite(value)):
        raise RuntimeError("invalid metric receipt: {}".format(path))
    return value


def validate_attempt(manifest, job, target):
    for name in REQUIRED:
        path = target / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("missing artifact: {}".format(path))
    kid_sha = sha256(target / REQUIRED[3])
    fid_sha = sha256(target / REQUIRED[4])
    if kid_sha != fid_sha:
        raise RuntimeError("KID/FID generated features are not byte-identical")
    options = json.loads((target / "training_options.json").read_text())
    expected = {
        "batch_size": 512,
        "metrics": list(METRICS),
        "metric_repeats": 1,
        "metric_generator_batch": 128,
        "retain_generated_artifacts": True,
        "seed": manifest["metric_seed"],
    }
    for key, value in expected.items():
        if options.get(key) != value:
            raise RuntimeError("evaluation option mismatch: {}".format(key))
    block = job["block"]
    if options.get("sample_seeds") != list(range(block["start"], block["end"] + 1)):
        raise RuntimeError("sample seed range mismatch")
    expected_mid = [] if job["nfe"] == 1 else [0.821]
    if options.get("mid_t") != expected_mid:
        raise RuntimeError("NFE/mid_t mismatch")
    if Path(options["resume_pkl"]).resolve() != Path(job["checkpoint"]["path"]).resolve():
        raise RuntimeError("checkpoint binding mismatch")
    dataset = Path(options["dataset_kwargs"]["path"]).resolve()
    if dataset != Path(manifest["dataset"]).resolve():
        raise RuntimeError("dataset binding mismatch")
    metrics = {
        metric: read_metric(target / "metric-{}.jsonl".format(metric), metric)
        for metric in METRICS
    }
    return metrics, kid_sha


def validate_receipt(manifest, job, receipt):
    expected = {
        "status": "PASS",
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "checkpoint_id": job["checkpoint"]["id"],
        "checkpoint": job["checkpoint"]["path"],
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "block": job["block"],
        "nfe": job["nfe"],
        "metric_seed": manifest["metric_seed"],
        "evaluator_commit": manifest["evaluator_commit"],
        "dataset_sha256": manifest["dataset_sha256"],
        "runtime_sha256": manifest["runtime_sha256"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError("receipt binding mismatch: {}".format(key))
    attempt = int(receipt["attempt"])
    expected_dir = (
        Path(manifest["output_root"]) / "jobs" / job["job_id"]
        / "attempt-{:02d}".format(attempt)
    ).resolve()
    if Path(receipt["job_dir"]).resolve() != expected_dir:
        raise RuntimeError("receipt job directory mismatch")
    metrics, feature_sha = validate_attempt(manifest, job, expected_dir)
    if receipt.get("metrics") != metrics:
        raise RuntimeError("receipt metric values mismatch")
    if receipt.get("generated_feature_sha256") != feature_sha:
        raise RuntimeError("receipt feature hash mismatch")
    return metrics
