#!/usr/bin/env python3
"""Bind and run the 34-cell opaque evaluation extension for replacement seed209."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


RUN_ROOT = Path("/root/q128_fresh_regime_history_n8_v1")
SOURCE_REPO = Path("/root/consistency-gap-optimization")
EVAL_ROOT = RUN_ROOT / "evaluation"
REPL_ROOT = EVAL_ROOT / "replacement209"
BASE_RUNTIME = RUN_ROOT / "runtime_receipts/eval_runtime/q128_fresh_opaque_eval.py"
THIS_SCRIPT = Path(__file__).resolve()
EXTENSION = REPL_ROOT / "control/replacement209_evaluation_extension.json"
LAUNCH_RECEIPT = REPL_ROOT / "control/launch_receipt.json"
RUNTIME_FREEZE = REPL_ROOT / "control/runtime_code_freeze.json"
GPU = 4
GPU_UUID = "GPU-6a4b6766-c00f-c1a5-059c-e098fc8b12c4"
SEED = 209
FAILED_SEED = 205


def load_base():
    spec = importlib.util.spec_from_file_location("q128_eval_base", BASE_RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen base evaluation runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_GPU_UUIDS[GPU] = GPU_UUID
    return module


base = load_base()


def oid(parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def read_rows(path: Path) -> tuple[dict, ...]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return tuple(csv.DictReader(handle))


def verify_training_complete() -> list[dict]:
    receipt = base.load_json(RUN_ROOT / "formal/seed209/seed_completion_receipt.json")
    expected = {"status": "PASS", "seed": 209, "replacement_for": 205, "gpu_index": 4}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"seed209 completion receipt mismatch: {key}")
    for branch in ("BA", "AB"):
        branch_receipt = base.load_json(RUN_ROOT / f"formal/seed209/{branch}/trajectory_completion_receipt.json")
        if branch_receipt.get("status") != "PASS" or branch_receipt.get("seed") != 209:
            raise RuntimeError(f"invalid {branch} completion receipt")
    deviations = []
    for trajectory in ("armA", "armBsame", "armBmatch", "armCmatch", "armDmatch", "BA", "AB"):
        root = RUN_ROOT / f"formal/seed209/{trajectory}"
        telemetry_name = "schedule_switch_training_telemetry_v1.csv" if trajectory in {"BA", "AB"} else "factorial_training_telemetry_v1.csv"
        rows = read_rows(root / telemetry_name)
        if not rows or float(rows[-1]["processed_kimg"]) < 1024.0:
            raise RuntimeError(f"incomplete telemetry: {trajectory}")
        if any(int(row.get(field, "0")) for row in rows for field in ("loss_nonfinite_count", "model_nonfinite_count", "ema_nonfinite_count")):
            raise RuntimeError(f"terminal scientific-state nonfinite: {trajectory}")
        for row in rows:
            if int(row.get("step_skipped", "0")) and float(row.get("processed_nimg", "0")) > 10_000:
                deviations.append({
                    "trajectory": trajectory,
                    "attempted_iteration": int(row["attempted_iteration"]),
                    "processed_nimg": int(float(row["processed_nimg"])),
                    "loss_nonfinite_count": int(row["loss_nonfinite_count"]),
                    "raw_grad_nonfinite_count": int(row["raw_grad_nonfinite_count"]),
                    "model_nonfinite_count": int(row["model_nonfinite_count"]),
                    "ema_nonfinite_count": int(row["ema_nonfinite_count"]),
                })
        for budget in ((512, 768, 1024) if trajectory.startswith("arm") else (640, 768, 896, 1024)):
            snapshot = root / f"network-snapshot-kimg{budget:06d}.pkl"
            if snapshot.is_symlink() or not snapshot.is_file() or snapshot.stat().st_size <= 0:
                raise RuntimeError(f"missing checkpoint: {snapshot}")
    return deviations


def gpu4_idle() -> None:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    if any(row.strip().startswith(GPU_UUID + ",") for row in rows):
        raise RuntimeError("GPU4 still has a compute process")


def prepare() -> None:
    os.umask(0o077)
    if REPL_ROOT.exists():
        raise RuntimeError(f"refuse pre-existing replacement evaluation root: {REPL_ROOT}")
    for name in ("control", "jobs", "receipts", "private", "opaque_logs"):
        (REPL_ROOT / name).mkdir(parents=True, mode=0o700, exist_ok=False)
    base.verify_runtime_freeze()
    base.require_frozen_inputs()
    base.require_evaluator()
    gpu4_idle()
    deviations = verify_training_complete()

    frozen = base.load_json(base.FROZEN_MANIFEST)
    templates = tuple(job for job in frozen["jobs"] if job["seed"] == FAILED_SEED)
    if len(templates) != 34:
        raise RuntimeError("seed205 template must contain 34 jobs")
    jobs = []
    checkpoint_hashes = {}
    protocol_id = frozen["protocol_id"]
    for template in templates:
        job = dict(template)
        job["seed"] = SEED
        job["opaque_id"] = oid((protocol_id, SEED, job["trajectory"], job["budget_kimg"], job["nfe"]))
        job["checkpoint_path"] = job["checkpoint_path"].replace("/seed205/", "/seed209/")
        checkpoint = Path(job["checkpoint_path"])
        if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"missing replacement checkpoint: {checkpoint}")
        checkpoint_hashes.setdefault(str(checkpoint), base.sha256_file(checkpoint))
        job["checkpoint_sha256"] = checkpoint_hashes[str(checkpoint)]
        job["replacement_of_seed"] = FAILED_SEED
        job["assigned_eval_gpu"] = GPU
        job["status"] = "BOUND_NOT_RUN"
        jobs.append(job)
    seven = base.load_json(base.BOUND_MANIFEST)
    existing_oids = {job["opaque_id"] for job in seven["jobs"]}
    if len(jobs) != 34 or len({job["opaque_id"] for job in jobs}) != 34 or existing_oids.intersection(job["opaque_id"] for job in jobs):
        raise RuntimeError("replacement extension opaque-id uniqueness failure")

    base.atomic_json(EXTENSION, {
        "schema": "ect.q128-fresh-replacement-evaluation-extension/v1",
        "status": "BOUND_NOT_RUN",
        "created_utc": base.utc_now(),
        "protocol_id": protocol_id,
        "protocol_sha256": base.PROTOCOL_SHA256,
        "frozen_manifest_sha256": base.FROZEN_MANIFEST_SHA256,
        "replacement_seed": SEED,
        "replacement_of_seed": FAILED_SEED,
        "effective_cohort": [201, 202, 203, 204, 206, 207, 208, 209],
        "old_seed205_jobs_excluded": True,
        "job_count": 34,
        "quality_values_decoded": False,
        "opaque_id_rule": "sha256(protocol_id|seed|trajectory|budget_kimg|nfe)[:20]",
        "jobs": jobs,
    })
    base.atomic_json(REPL_ROOT / "control/effective_cohort_roster.json", {
        "schema": "ect.q128-fresh-effective-cohort/v1",
        "status": "FROZEN_BEFORE_SEED209_QUALITY",
        "created_utc": base.utc_now(),
        "effective_cohort": [201, 202, 203, 204, 206, 207, 208, 209],
        "failed_seed": 205,
        "replacement_seed": 209,
        "pooling_with_old_n3_forbidden": True,
    })
    base.atomic_json(REPL_ROOT / "control/pre_quality_integrity_deviation_record.json", {
        "schema": "ect.q128-fresh-seed209-pre-quality-integrity-deviation/v1",
        "status": "PROTOCOL_DEVIATION_PENDING_ADJUDICATION" if deviations else "NO_NEW_LATE_AMP_SKIP",
        "created_utc": base.utc_now(),
        "recorded_before_first_seed209_quality_job": True,
        "quality_values_observed": False,
        "late_amp_skip_event_count": len(deviations),
        "events": deviations,
        "interpretation": "evaluation launch is not a waiver and does not determine confirmatory validity",
    })
    cache = REPL_ROOT / "cache"
    shutil.copytree(base.CACHE_SOURCE, cache)
    runtime_sha = base.sha256_file(THIS_SCRIPT)
    base.atomic_json(RUNTIME_FREEZE, {
        "schema": "ect.q128-fresh-seed209-evaluation-runtime-freeze/v1",
        "status": "FROZEN_BEFORE_FIRST_SEED209_QUALITY_JOB",
        "created_utc": base.utc_now(),
        "runtime_script": str(THIS_SCRIPT),
        "runtime_script_sha256": runtime_sha,
        "base_runtime_script": str(BASE_RUNTIME),
        "base_runtime_script_sha256": base.sha256_file(BASE_RUNTIME),
        "replacement_extension_sha256": base.sha256_file(EXTENSION),
        "quality_values_observed": False,
    })
    base.atomic_json(LAUNCH_RECEIPT, {
        "schema": "ect.q128-fresh-seed209-evaluation-launch/v1",
        "status": "PREPARED_NOT_STARTED",
        "created_utc": base.utc_now(),
        "seed": SEED,
        "gpu_index": GPU,
        "gpu_uuid": GPU_UUID,
        "job_count": 34,
        "extension_sha256": base.sha256_file(EXTENSION),
        "runtime_script_sha256": runtime_sha,
        "first_quality_job_started": False,
        "quality_values_decoded": False,
    })
    print(json.dumps({"status": "PREPARED", "seed": SEED, "jobs": 34}))


def verify_own_freeze() -> str:
    receipt = base.load_json(RUNTIME_FREEZE)
    actual = base.sha256_file(THIS_SCRIPT)
    if receipt.get("status") != "FROZEN_BEFORE_FIRST_SEED209_QUALITY_JOB" or receipt.get("runtime_script_sha256") != actual:
        raise RuntimeError("seed209 evaluation runtime changed after freeze")
    if receipt.get("replacement_extension_sha256") != base.sha256_file(EXTENSION):
        raise RuntimeError("replacement extension changed after freeze")
    return actual


def run() -> None:
    os.umask(0o077)
    runtime_sha = verify_own_freeze()
    base.verify_runtime_freeze()
    base.require_frozen_inputs()
    base.require_evaluator()
    uuid, _ = base.gpu_identity(GPU)
    if uuid != GPU_UUID:
        raise RuntimeError("GPU4 UUID mismatch")
    manifest = base.load_json(EXTENSION)
    if manifest.get("job_count") != 34 or manifest.get("quality_values_decoded") is not False:
        raise RuntimeError("invalid replacement extension")
    launch = base.load_json(LAUNCH_RECEIPT)
    if launch.get("status") != "PREPARED_NOT_STARTED" or launch.get("first_quality_job_started") is not False:
        raise RuntimeError("seed209 launch receipt is not launchable")
    launch.update({"status": "RUNNING_SEALED", "started_utc": base.utc_now(), "pid": os.getpid(), "first_quality_job_started": True})
    base.atomic_json(LAUNCH_RECEIPT, launch, overwrite=True)
    queue_path = REPL_ROOT / "opaque_logs/gpu4.queue.log"
    with queue_path.open("a", encoding="utf-8", buffering=1) as queue:
        for job in manifest["jobs"]:
            oid_value = job["opaque_id"]
            receipt_path = REPL_ROOT / f"receipts/{oid_value}.json"
            if receipt_path.exists():
                old = base.load_json(receipt_path)
                if old.get("status") == "SEALED_PASS" and old.get("opaque_id") == oid_value:
                    continue
                raise RuntimeError(f"non-pass existing seed209 receipt: {oid_value}")
            checkpoint = Path(job["checkpoint_path"])
            if base.sha256_file(checkpoint) != job["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {oid_value}")
            job_dir = REPL_ROOT / f"jobs/{oid_value}"
            process_log = REPL_ROOT / f"private/{oid_value}.process.log"
            if job_dir.exists() or process_log.exists():
                raise RuntimeError(f"refuse nonterminal output: {oid_value}")
            port = 45000 + (int(job["job_index"]) % 900)
            command = [
                str(base.RUNTIME_PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                f"--master_port={port}", str(base.EVALUATOR_REPO / "ct_eval.py"),
                "--resume", str(checkpoint), "--outdir", str(job_dir), "--nosubdir",
                "--data", str(base.DATASET), "--cond=False", "--arch=ddpmpp", "--precond=ct",
                "--dropout=0.2", "--augment=0", "--xflip=False", "--fp16=False", "--cache=True",
                "--workers=1", "--eval-batch=512", "--metric-generator-batch=128", f"--nfe={job['nfe']}",
                "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1", "--sample-seeds=0-49999",
                "--seed=20260730", "--retain-generated-artifacts", f"--desc={oid_value}",
            ]
            if job["nfe"] == 2:
                if job.get("mid_t") != 0.821:
                    raise RuntimeError(f"NFE2 mid_t mismatch: {oid_value}")
                command.append("--mid_t=0.821")
            elif job.get("mid_t") is not None:
                raise RuntimeError(f"NFE1 carries mid_t: {oid_value}")
            env = os.environ.copy()
            env.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(GPU),
                "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "DNNLIB_CACHE_DIR": str(REPL_ROOT / "cache"), "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port), "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
            })
            queue.write(f"{base.utc_now()} START {oid_value} gpu=4\n")
            started = time.time()
            sealed_on_failure = []
            try:
                with process_log.open("xb") as handle:
                    result = subprocess.run(command, cwd=base.EVALUATOR_REPO, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT, timeout=6 * 60 * 60)
                if result.returncode != 0:
                    raise RuntimeError(f"evaluator exit code {result.returncode}")
                payload = base.validate_and_seal(job, job_dir, process_log, GPU, time.time() - started)
                payload["schema"] = "ect.q128-fresh-seed209-opaque-evaluation-job/v1"
                payload["runtime_script_sha256"] = runtime_sha
                payload["replacement_extension_sha256"] = base.sha256_file(EXTENSION)
                payload["bound_manifest_sha256"] = base.sha256_file(EXTENSION)
                base.atomic_json(receipt_path, payload)
                queue.write(f"{base.utc_now()} SEALED_PASS {oid_value}\n")
            except Exception as exc:
                try:
                    sealed_on_failure = base.seal_sensitive_outputs(job_dir, process_log)
                except Exception as seal_exc:
                    exc = RuntimeError(f"{exc}; sealing_failure={seal_exc}")
                base.atomic_json(receipt_path, {
                    "schema": "ect.q128-fresh-seed209-opaque-evaluation-job/v1",
                    "status": "FAILED_CLOSED", "opaque_id": oid_value, "failed_utc": base.utc_now(),
                    "error": f"{type(exc).__name__}: {exc}", "quality_values_in_receipt": False,
                    "quality_values_decoded": False, "sealed_scalar_artifacts": sealed_on_failure,
                })
                queue.write(f"{base.utc_now()} FAILED_CLOSED {oid_value}\n")
                launch = base.load_json(LAUNCH_RECEIPT)
                launch.update({"status": "FAILED_CLOSED", "failed_utc": base.utc_now(), "failed_opaque_id": oid_value})
                base.atomic_json(LAUNCH_RECEIPT, launch, overwrite=True)
                raise
    count = sum(base.load_json(path).get("status") == "SEALED_PASS" for path in (REPL_ROOT / "receipts").glob("*.json"))
    launch = base.load_json(LAUNCH_RECEIPT)
    launch.update({"status": "SEALED_PASS_SEED209" if count == 34 else "FAILED_CLOSED",
                   "completed_utc": base.utc_now(), "sealed_pass_count": count, "expected_count": 34,
                   "quality_values_decoded": False})
    base.atomic_json(LAUNCH_RECEIPT, launch, overwrite=True)
    if count != 34:
        raise SystemExit(4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
