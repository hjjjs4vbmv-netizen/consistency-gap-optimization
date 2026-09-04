#!/usr/bin/env python3
"""Fail-closed rolling evaluator for the completed fresh-q128 seeds.

This runtime adapter binds the frozen opaque evaluation matrix to immutable
checkpoint hashes, runs one completed seed per idle GPU, validates the frozen
FP32 FID/KID semantics, and seals all scalar-quality-bearing text before
publishing a SEALED_PASS receipt.  It never prints or stores metric values in
receipts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_ROOT = Path("/root/q128_fresh_regime_history_n8_v1")
SOURCE_REPO = Path("/root/consistency-gap-optimization")
EVAL_ROOT = RUN_ROOT / "evaluation"
FROZEN_MANIFEST = SOURCE_REPO / "analysis/q128_fresh_regime_history_n8_v1/evaluation_manifest.json"
BOUND_MANIFEST = EVAL_ROOT / "control/bound_manifest_completed7.json"
EVALUATOR_REPO = EVAL_ROOT / "control/evaluator_5abd4bd"
RUNTIME_PYTHON = Path("/root/miniconda3/envs/myconda/bin/python")
DATASET = Path("/mnt/ect_project/q256_seed14_18_eval_assets_20260822/cifar10-32x32-canonical-08c9ed1b2b1c.zip")
CACHE_SOURCE = Path("/mnt/ect_project/q256_seed14_18_eval_assets_20260822/cache")
SOURCE_COMMIT = "5abd4bd074f6987110f29a0adb93e24e842450bd"
FROZEN_MANIFEST_SHA256 = "34f296c1eac0042bb6d37cd692c3a1ab008ef87caa7417b4ce9f0f33dde6e17e"
CT_EVAL_SHA256 = "938941b612bd766fbf552e84d4e127daedb19594b39a5603b7c735d89d47d325"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
PROTOCOL_SHA256 = "e908da54c47d2f3faa9dd699f7d1a345446fdb11a71ad32642ec74197f59b3bf"
COMPLETED_GPU_MAP = {201: 0, 202: 1, 203: 2, 204: 3, 206: 5, 207: 6, 208: 7}
EXPECTED_GPU_UUIDS = {
    0: "GPU-4c7f9706-4aa9-3e0d-77dc-7d60770de75b",
    1: "GPU-34962947-daa7-4a85-dbcb-ec375884c9bf",
    2: "GPU-97fbe27b-7894-7bde-fe12-deeeffb3b355",
    3: "GPU-d769c7d9-1684-b1d6-f9a5-922fec514974",
    5: "GPU-61e3a676-4e43-4daf-eddf-7e7754557267",
    6: "GPU-9015e7ca-971b-7dfa-c636-928a213fc1e1",
    7: "GPU-b0e88c86-df21-f3b5-3c8a-fdcf025c9eb4",
}
METRICS = ("kid50k_full", "fid50k_full")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_frozen_inputs() -> None:
    if sha256_file(FROZEN_MANIFEST) != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("frozen evaluation manifest hash mismatch")
    if sha256_file(DATASET) != DATASET_SHA256:
        raise RuntimeError("canonical dataset hash mismatch")
    protocol_path = SOURCE_REPO / "analysis/q128_fresh_regime_history_n8_v1/protocol.json"
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("protocol hash mismatch")


def require_evaluator() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(EVALUATOR_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError("evaluator commit mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(EVALUATOR_REPO), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("evaluator worktree is dirty")
    if sha256_file(EVALUATOR_REPO / "ct_eval.py") != CT_EVAL_SHA256:
        raise RuntimeError("ct_eval.py hash mismatch")


def ensure_evaluator_worktree() -> None:
    if not EVALUATOR_REPO.exists():
        EVALUATOR_REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "worktree", "add", "--detach", str(EVALUATOR_REPO), SOURCE_COMMIT],
            check=True,
        )
    require_evaluator()


def scan_late_amp_skips() -> list[dict]:
    rows: list[dict] = []
    for seed in COMPLETED_GPU_MAP:
        seed_root = RUN_ROOT / f"formal/seed{seed}"
        paths = sorted(seed_root.glob("arm*/factorial_training_telemetry_v1.csv"))
        paths += sorted(seed_root.glob("*/schedule_switch_training_telemetry_v1.csv"))
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for record in csv.DictReader(handle):
                    if int(record.get("step_skipped", "0")) != 1:
                        continue
                    processed = int(float(record["processed_nimg"]))
                    if processed <= 10_000:
                        continue
                    rows.append({
                        "seed": seed,
                        "trajectory_dir": path.parent.name,
                        "telemetry_path": str(path),
                        "attempted_iteration": int(record["attempted_iteration"]),
                        "processed_nimg": processed,
                        "loss_nonfinite_count": int(record["loss_nonfinite_count"]),
                        "raw_grad_nonfinite_count": int(record["raw_grad_nonfinite_count"]),
                        "sanitized_grad_nonfinite_count": int(record["sanitized_grad_nonfinite_count"]),
                        "update_nonfinite_count": int(record["update_nonfinite_count"]),
                        "model_nonfinite_count": int(record["model_nonfinite_count"]),
                        "ema_nonfinite_count": int(record["ema_nonfinite_count"]),
                        "update_norm": float(record["update_norm"]),
                    })
    return rows


def prepare() -> None:
    if EVAL_ROOT.exists():
        raise RuntimeError(f"refuse pre-existing evaluation root: {EVAL_ROOT}")
    os.umask(0o077)
    for name in ("control", "jobs", "receipts", "private", "opaque_logs", "worker_caches"):
        (EVAL_ROOT / name).mkdir(parents=True, mode=0o700, exist_ok=False)
    require_frozen_inputs()
    ensure_evaluator_worktree()

    script_path = Path(__file__).resolve()
    runtime_sha = sha256_file(script_path)
    atomic_json(EVAL_ROOT / "control/runtime_code_freeze.json", {
        "schema": "ect.q128-fresh-opaque-evaluation-runtime-freeze/v1",
        "status": "FROZEN_BEFORE_FIRST_QUALITY_JOB",
        "created_utc": utc_now(),
        "runtime_script": str(script_path),
        "runtime_script_sha256": runtime_sha,
        "evaluator_repo": str(EVALUATOR_REPO),
        "evaluator_commit": SOURCE_COMMIT,
        "ct_eval_sha256": CT_EVAL_SHA256,
        "frozen_manifest": str(FROZEN_MANIFEST),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "quality_values_observed": False,
    })

    frozen = load_json(FROZEN_MANIFEST)
    if frozen.get("job_count") != 272 or frozen.get("quality_values_decoded") is not False:
        raise RuntimeError("invalid frozen 272-job manifest")
    jobs = []
    checkpoint_cache: dict[str, str] = {}
    for seed, gpu in COMPLETED_GPU_MAP.items():
        completion = RUN_ROOT / f"formal/seed{seed}/seed_completion_receipt.json"
        receipt = load_json(completion)
        if receipt.get("status") != "PASS" or receipt.get("seed") != seed or receipt.get("gpu_index") != gpu:
            raise RuntimeError(f"invalid seed completion receipt: {seed}")
        seed_jobs = [j for j in frozen["jobs"] if j["seed"] == seed]
        if len(seed_jobs) != 34:
            raise RuntimeError(f"seed {seed} does not have 34 frozen jobs")
        for source_job in seed_jobs:
            job = dict(source_job)
            checkpoint = Path(job["checkpoint_path"])
            if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise RuntimeError(f"missing immutable checkpoint: {checkpoint}")
            key = str(checkpoint)
            if key not in checkpoint_cache:
                checkpoint_cache[key] = sha256_file(checkpoint)
            job["checkpoint_sha256"] = checkpoint_cache[key]
            job["assigned_eval_gpu"] = gpu
            job["status"] = "BOUND_NOT_RUN"
            jobs.append(job)
    if len(jobs) != 238 or len({j["opaque_id"] for j in jobs}) != 238:
        raise RuntimeError("completed-seven matrix must contain 238 unique jobs")
    atomic_json(BOUND_MANIFEST, {
        "schema": "ect.q128-fresh-bound-evaluation-manifest/v1",
        "status": "BOUND_NOT_RUN",
        "created_utc": utc_now(),
        "selected_completed_seeds": sorted(COMPLETED_GPU_MAP),
        "pending_replacement_seed": 209,
        "failed_seed_excluded": 205,
        "job_count": len(jobs),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "quality_values_decoded": False,
        "decode_gate": "all PRIMARY and KEY_SECONDARY jobs for the final effective n=8 cohort must be SEALED_PASS",
        "jobs": jobs,
    })

    deviations = scan_late_amp_skips()
    atomic_json(EVAL_ROOT / "control/pre_quality_integrity_deviation_record.json", {
        "schema": "ect.q128-fresh-pre-quality-integrity-deviation/v1",
        "status": "PROTOCOL_DEVIATION_PENDING_ADJUDICATION",
        "created_utc": utc_now(),
        "recorded_before_first_quality_job": True,
        "quality_values_observed": False,
        "issue": "late AMP GradScaler skips occurred after the frozen 10000-nimg warmup gate",
        "completed_seed_skip_event_count": len(deviations),
        "events": deviations,
        "interpretation": "rolling blind evaluation is not a waiver and does not restore confirmatory status",
    })

    key = EVAL_ROOT / "control/decode.key"
    with key.open("xb") as handle:
        handle.write(os.urandom(32).hex().encode("ascii") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(key, stat.S_IRUSR)
    for gpu in COMPLETED_GPU_MAP.values():
        cache = EVAL_ROOT / f"worker_caches/gpu{gpu}"
        shutil.copytree(CACHE_SOURCE, cache)
    atomic_json(EVAL_ROOT / "control/rolling_launch_receipt.json", {
        "schema": "ect.q128-fresh-rolling-evaluation-launch/v1",
        "status": "PREPARED_NOT_STARTED",
        "created_utc": utc_now(),
        "completed_seeds": sorted(COMPLETED_GPU_MAP),
        "seed_to_eval_gpu": {str(k): v for k, v in COMPLETED_GPU_MAP.items()},
        "excluded_training_gpu": 4,
        "pending_seed": 209,
        "job_count": 238,
        "runtime_script_sha256": runtime_sha,
        "bound_manifest_sha256": sha256_file(BOUND_MANIFEST),
        "first_quality_job_started": False,
        "quality_values_decoded": False,
    })
    print(json.dumps({"status": "PREPARED", "jobs": 238, "seeds": 7}))


def verify_runtime_freeze() -> str:
    receipt = load_json(EVAL_ROOT / "control/runtime_code_freeze.json")
    actual = sha256_file(Path(__file__).resolve())
    if receipt.get("status") != "FROZEN_BEFORE_FIRST_QUALITY_JOB" or receipt.get("runtime_script_sha256") != actual:
        raise RuntimeError("runtime adapter changed after freeze")
    return actual


def gpu_identity(gpu: int) -> tuple[str, str]:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,name", "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid, model = [part.strip() for part in output.split(",", 1)]
    if uuid != EXPECTED_GPU_UUIDS[gpu] or "A100" not in model:
        raise RuntimeError(f"GPU identity mismatch for index {gpu}")
    return uuid, model


def read_metric_without_disclosure(path: Path, metric: str) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"metric file line count mismatch: {path.name}")
    payload = json.loads(lines[0])
    if payload.get("metric") != metric or payload.get("num_gpus") != 1:
        raise RuntimeError(f"metric identity mismatch: {path.name}")
    value = float(payload["results"][metric])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"nonfinite or negative metric: {path.name}")


def seal_file(path: Path, key: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"cannot seal missing/symlink file: {path}")
    sealed = path.with_name(path.name + ".sealed")
    if sealed.exists():
        raise RuntimeError(f"refuse existing seal: {sealed}")
    subprocess.run([
        "openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt",
        "-pass", f"file:{key}", "-in", str(path), "-out", str(sealed),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sealed.stat().st_size <= 0:
        raise RuntimeError(f"empty seal: {sealed}")
    payload = {"sealed_path": str(sealed), "sha256": sha256_file(sealed), "bytes": sealed.stat().st_size}
    path.unlink()
    return payload


def seal_sensitive_outputs(job_dir: Path, process_log: Path) -> list[dict]:
    key = EVAL_ROOT / "control/decode.key"
    candidates = []
    if process_log.exists():
        candidates.append(process_log)
    if job_dir.exists():
        candidates.extend(sorted(job_dir.glob("*.jsonl")))
        candidates.extend(sorted(job_dir.glob("*.log")))
        candidates.extend(sorted(job_dir.glob("*.txt")))
    sealed = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen or path.name.endswith(".sealed"):
            continue
        seen.add(resolved)
        sealed.append(seal_file(path, key))
    return sealed


def validate_and_seal(job: dict, job_dir: Path, process_log: Path, gpu: int, elapsed: float) -> dict:
    required_binary = (
        "training_options.json", "generated-samples.npy",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
    )
    artifacts = {}
    for name in required_binary:
        path = job_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing evaluation artifact: {name}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    kid_feature = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
    fid_feature = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
    if kid_feature != fid_feature:
        raise RuntimeError("KID/FID generated features are not byte-identical")
    for metric in METRICS:
        read_metric_without_disclosure(job_dir / f"metric-{metric}.jsonl", metric)
    options = load_json(job_dir / "training_options.json")
    expected = {
        "batch_size": 512,
        "metrics": list(METRICS),
        "metric_repeats": 1,
        "metric_generator_batch": 128,
        "retain_generated_artifacts": True,
        "seed": 20260730,
    }
    for key, value in expected.items():
        if options.get(key) != value:
            raise RuntimeError(f"evaluation option mismatch: {key}")
    if options.get("sample_seeds") != list(range(50_000)):
        raise RuntimeError("sample seeds are not exactly 0..49999")
    if options.get("mid_t") != ([] if job["nfe"] == 1 else [0.821]):
        raise RuntimeError("NFE/mid_t mismatch")
    if Path(options["resume_pkl"]).resolve() != Path(job["checkpoint_path"]).resolve():
        raise RuntimeError("checkpoint binding mismatch")
    if Path(options["dataset_kwargs"]["path"]).resolve() != DATASET.resolve():
        raise RuntimeError("dataset binding mismatch")
    sealed = seal_sensitive_outputs(job_dir, process_log)
    uuid, model = gpu_identity(gpu)
    return {
        "schema": "ect.q128-fresh-opaque-evaluation-job/v1",
        "status": "SEALED_PASS",
        "opaque_id": job["opaque_id"],
        "job_index": job["job_index"],
        "category": job["category"],
        "nfe": job["nfe"],
        "mid_t": None if job["nfe"] == 1 else 0.821,
        "sample_count": 50_000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20260730,
        "precision": "fp32",
        "metrics_validated_but_not_disclosed": list(METRICS),
        "quality_values_in_receipt": False,
        "quality_values_decoded": False,
        "checkpoint_sha256": job["checkpoint_sha256"],
        "generated_feature_sha256": kid_feature,
        "kid_fid_shared_feature_identity": True,
        "dataset_sha256": DATASET_SHA256,
        "evaluator_commit": SOURCE_COMMIT,
        "ct_eval_sha256": CT_EVAL_SHA256,
        "runtime_script_sha256": verify_runtime_freeze(),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "bound_manifest_sha256": sha256_file(BOUND_MANIFEST),
        "gpu_index": gpu,
        "gpu_uuid": uuid,
        "gpu_model": model,
        "elapsed_seconds": elapsed,
        "completed_utc": utc_now(),
        "artifacts": artifacts,
        "sealed_scalar_artifacts": sealed,
    }


def worker(seed: int, gpu: int) -> None:
    os.umask(0o077)
    if COMPLETED_GPU_MAP.get(seed) != gpu or gpu == 4:
        raise RuntimeError("invalid rolling evaluation seed/GPU assignment")
    verify_runtime_freeze()
    require_frozen_inputs()
    require_evaluator()
    gpu_identity(gpu)
    bound = load_json(BOUND_MANIFEST)
    if bound.get("quality_values_decoded") is not False or bound.get("job_count") != 238:
        raise RuntimeError("invalid bound manifest")
    jobs = [j for j in bound["jobs"] if j["seed"] == seed]
    if len(jobs) != 34:
        raise RuntimeError(f"seed {seed} does not have 34 bound jobs")
    worker_log = EVAL_ROOT / f"opaque_logs/gpu{gpu}.queue.log"
    with worker_log.open("a", encoding="utf-8", buffering=1) as queue:
        for job in jobs:
            oid = job["opaque_id"]
            receipt_path = EVAL_ROOT / f"receipts/{oid}.json"
            if receipt_path.exists():
                receipt = load_json(receipt_path)
                if receipt.get("status") == "SEALED_PASS" and receipt.get("opaque_id") == oid:
                    continue
                raise RuntimeError(f"non-pass existing receipt: {oid}")
            checkpoint = Path(job["checkpoint_path"])
            if sha256_file(checkpoint) != job["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {oid}")
            job_dir = EVAL_ROOT / f"jobs/{oid}"
            process_log = EVAL_ROOT / f"private/{oid}.process.log"
            if job_dir.exists() or process_log.exists():
                raise RuntimeError(f"refuse nonterminal pre-existing output: {oid}")
            cache = EVAL_ROOT / f"worker_caches/gpu{gpu}"
            port = 41000 + gpu * 1000 + (int(job["job_index"]) % 900)
            command = [
                str(RUNTIME_PYTHON), "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=1", f"--master_port={port}", str(EVALUATOR_REPO / "ct_eval.py"),
                "--resume", str(checkpoint), "--outdir", str(job_dir), "--nosubdir",
                "--data", str(DATASET), "--cond=False", "--arch=ddpmpp", "--precond=ct",
                "--dropout=0.2", "--augment=0", "--xflip=False", "--fp16=False",
                "--cache=True", "--workers=1", "--eval-batch=512", "--metric-generator-batch=128",
                f"--nfe={job['nfe']}", "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
                "--sample-seeds=0-49999", "--seed=20260730", "--retain-generated-artifacts",
                f"--desc={oid}",
            ]
            if job["nfe"] == 2:
                if job.get("mid_t") != 0.821:
                    raise RuntimeError(f"NFE2 mid_t mismatch: {oid}")
                command.append("--mid_t=0.821")
            elif job.get("mid_t") is not None:
                raise RuntimeError(f"NFE1 carries mid_t: {oid}")
            env = os.environ.copy()
            env.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "DNNLIB_CACHE_DIR": str(cache),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
            })
            queue.write(f"{utc_now()} START {oid} gpu={gpu}\n")
            start = time.time()
            failure = None
            sealed_on_failure = []
            try:
                with process_log.open("xb") as log_handle:
                    result = subprocess.run(
                        command, cwd=EVALUATOR_REPO, env=env,
                        stdout=log_handle, stderr=subprocess.STDOUT,
                        timeout=6 * 60 * 60,
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"evaluator exit code {result.returncode}")
                payload = validate_and_seal(job, job_dir, process_log, gpu, time.time() - start)
                atomic_json(receipt_path, payload)
                queue.write(f"{utc_now()} SEALED_PASS {oid}\n")
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                try:
                    sealed_on_failure = seal_sensitive_outputs(job_dir, process_log)
                except Exception as seal_exc:
                    failure += f"; sealing_failure={type(seal_exc).__name__}: {seal_exc}"
                    for path in (process_log,):
                        if path.exists():
                            os.chmod(path, 0)
                atomic_json(receipt_path, {
                    "schema": "ect.q128-fresh-opaque-evaluation-job/v1",
                    "status": "FAILED_CLOSED",
                    "opaque_id": oid,
                    "failed_utc": utc_now(),
                    "error": failure,
                    "quality_values_in_receipt": False,
                    "quality_values_decoded": False,
                    "sealed_scalar_artifacts": sealed_on_failure,
                })
                queue.write(f"{utc_now()} FAILED_CLOSED {oid}\n")
                raise


def coordinator() -> None:
    verify_runtime_freeze()
    receipt_path = EVAL_ROOT / "control/rolling_launch_receipt.json"
    launch = load_json(receipt_path)
    if launch.get("status") != "PREPARED_NOT_STARTED" or launch.get("first_quality_job_started") is not False:
        raise RuntimeError("rolling launch receipt is not launchable")
    launch.update({
        "status": "RUNNING_SEALED",
        "started_utc": utc_now(),
        "first_quality_job_started": True,
        "quality_values_decoded": False,
        "coordinator_pid": os.getpid(),
    })
    atomic_json(receipt_path, launch, overwrite=True)
    processes = {}
    for seed, gpu in COMPLETED_GPU_MAP.items():
        log = (EVAL_ROOT / f"opaque_logs/worker_seed{seed}.supervisor.log").open("ab", buffering=0)
        proc = subprocess.Popen(
            [str(RUNTIME_PYTHON), str(Path(__file__).resolve()), "worker", "--seed", str(seed), "--gpu", str(gpu)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=False,
        )
        processes[seed] = (proc, log)
    atomic_json(EVAL_ROOT / "control/worker_pids.json", {
        "schema": "ect.q128-fresh-opaque-evaluation-workers/v1",
        "status": "RUNNING",
        "created_utc": utc_now(),
        "workers": [
            {"seed": seed, "gpu_index": COMPLETED_GPU_MAP[seed], "pid": proc.pid}
            for seed, (proc, _) in processes.items()
        ],
    })
    failures = []
    for seed, (proc, log) in processes.items():
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append({"seed": seed, "gpu_index": COMPLETED_GPU_MAP[seed], "exit_code": code})
    receipt_count = 0
    for path in (EVAL_ROOT / "receipts").glob("*.json"):
        if load_json(path).get("status") == "SEALED_PASS":
            receipt_count += 1
    launch = load_json(receipt_path)
    launch.update({
        "status": "SEALED_PASS_COMPLETED7" if not failures and receipt_count == 238 else "FAILED_CLOSED",
        "completed_utc": utc_now(),
        "sealed_pass_count": receipt_count,
        "expected_count": 238,
        "worker_failures": failures,
        "quality_values_decoded": False,
    })
    atomic_json(receipt_path, launch, overwrite=True)
    if launch["status"] != "SEALED_PASS_COMPLETED7":
        raise SystemExit(4)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--gpu", type=int, required=True)
    sub.add_parser("coordinator")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        worker(args.seed, args.gpu)
    else:
        coordinator()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
