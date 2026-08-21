#!/usr/bin/env python3
"""Run one seed's frozen q256 FID/KID evaluation matrix.

This is a five-seed extension of the preregistered seed3-5 evaluator.  It keeps
the numerical contract unchanged and only distributes one complete seed to one
GPU so the five independent replicates can run concurrently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
TRAINING_COMMIT = "458205192722883df393a8d017c26e6fa46f48f7"
PARENT_TRAINING_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
ARMS = ("A", "B", "C", "D")
NFE_SETTINGS = {1: [], 2: [0.821]}
METRICS = ("kid50k_full", "fid50k_full")
SAMPLE_COUNT = 50_000
METRIC_SEED = 20_260_730
PROTOCOL = "q256-target-weight-1024k-formal-evaluation-v1"


class EvaluationFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise EvaluationFailure(f"cache template contains a symlink: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not files:
        raise EvaluationFailure(f"cache template is empty: {root}")
    return {"root": str(root), "files": files, "tree_sha256": canonical_sha256(files)}


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def latest_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EvaluationFailure(f"empty training summary: {path}")
    row = rows[-1]
    attempted = int(row["attempted_iteration"])
    successful = int(row["successful_optimizer_steps"])
    processed = float(row["processed_kimg"])
    loss = float(row["loss"])
    skipped = int(row["step_skipped"])
    if attempted != 8000 or not (0 < successful <= attempted) or not math.isclose(processed, 1024.0):
        raise EvaluationFailure(f"training endpoint mismatch: {path}: {row}")
    if not math.isfinite(loss) or skipped != 0:
        raise EvaluationFailure(f"invalid final training row: {path}: {row}")
    return {
        "attempted_iteration": attempted,
        "successful_optimizer_steps": successful,
        "processed_kimg": processed,
        "loss": loss,
        "step_skipped": skipped,
        "summary_sha256": sha256_file(path),
    }


def validate_training_cell(training_root: Path, seed: int, arm: str) -> dict[str, Any]:
    run_dir = (training_root / f"seed{seed}" / f"arm{arm}").resolve(strict=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise EvaluationFailure(f"invalid training run directory: {run_dir}")
    checkpoint = run_dir / "network-snapshot-latest.pkl"
    summary = run_dir / "train_summary.csv"
    initial_receipt = run_dir / "initial_state_receipt_v1.json"
    log_path = run_dir / "log.txt"
    for path in (checkpoint, summary, initial_receipt, log_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise EvaluationFailure(f"missing required training artifact: {path}")
    receipt = json.loads(initial_receipt.read_text(encoding="utf-8"))
    factorial = receipt.get("factorial", {})
    if int(receipt.get("seed", -1)) != seed or factorial.get("arm") != arm:
        raise EvaluationFailure(f"initial-state seed/arm binding mismatch: {initial_receipt}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Exiting..." not in log_text or "Traceback (most recent call last)" in log_text:
        raise EvaluationFailure(f"training log lacks a clean completion marker: {log_path}")
    return {
        "seed": seed,
        "arm": arm,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "training_endpoint": latest_summary(summary),
        "initial_state_receipt": str(initial_receipt),
        "initial_state_receipt_sha256": sha256_file(initial_receipt),
        "training_log": str(log_path),
        "training_log_sha256": sha256_file(log_path),
    }


def run_process(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path,
    timeout_seconds: int,
) -> int:
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command), cwd=cwd, env=dict(env), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            raise EvaluationFailure(
                f"evaluation job exceeded {timeout_seconds} seconds: {' '.join(command)}"
            )


def build_jobs(
    *, source_root: Path, output_root: Path, dataset: Path,
    cells: Sequence[Mapping[str, Any]], base_port: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in cells:
        for nfe, mid_t in NFE_SETTINGS.items():
            job_id = f"seed{cell['seed']}-arm{cell['arm']}-nfe{nfe}"
            target = output_root / "jobs" / job_id
            command = [
                "bash", str(source_root / "scripts" / "evaluate_checkpoint.sh"),
                "1", str(base_port + len(jobs)), str(cell["checkpoint"]),
                "--outdir", str(target), "--nosubdir",
                "--data", str(dataset), "--cond=False", "--arch=ddpmpp",
                "--precond=ct", "--dropout=0.2", "--augment=0", "--xflip=False",
                "--fp16=False", "--cache=True", "--workers=3", "--eval-batch=512",
                "--metric-generator-batch=128", f"--nfe={nfe}",
                *(["--mid_t=0.821"] if nfe == 2 else []),
                f"--metrics={','.join(METRICS)}", "--metric-repeats=1",
                "--sample-seeds=0-49999", f"--seed={METRIC_SEED}",
                "--retain-generated-artifacts",
                f"--desc={PROTOCOL}-{job_id}",
            ]
            jobs.append({
                "job_id": job_id,
                "seed": cell["seed"],
                "arm": cell["arm"],
                "nfe": nfe,
                "mid_t": mid_t,
                "checkpoint": cell["checkpoint"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "training_run": cell["run_dir"],
                "sample_count": SAMPLE_COUNT,
                "sample_seeds": "0-49999",
                "metric_seed": METRIC_SEED,
                "metrics": list(METRICS),
                "precision": "fp32",
                "output_directory": str(target),
                "command_argv": command,
            })
    if len(jobs) != 8:
        raise EvaluationFailure(f"one-seed matrix must contain 8 jobs, got {len(jobs)}")
    return jobs


def runtime_manifest() -> dict[str, Any]:
    import numpy
    import scipy
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    manifest = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": names,
        "hardware_deviation_from_seed3_5": "A100-PCIE-40GB instead of A100-SXM4-80GB; numerical contract unchanged",
    }
    expected = {
        "python_version": "3.10.12",
        "torch_version": "2.2.0a0+81ea7a4",
        "torch_cuda_version": "12.3",
        "cuda_available": True,
        "cuda_device_count": 1,
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise EvaluationFailure(f"runtime mismatch for {key}: {manifest[key]} != {value}")
    if len(names) != 1 or "A100" not in names[0]:
        raise EvaluationFailure(f"expected exactly one visible A100, got {names}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=range(14, 19))
    parser.add_argument("--gpu", type=int, required=True, choices=range(5))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--cache-template", type=Path)
    parser.add_argument("--job-timeout-seconds", type=int, default=21600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = utc_now()
    output_root = args.output_root.resolve()
    failure_path = output_root.parent / f"seed{args.seed}-EVALUATION_FAILURE.json"
    try:
        if args.gpu != args.seed - 14:
            raise EvaluationFailure("seed/GPU binding must be seed14->0 through seed18->4")
        source_root = args.source_root.resolve(strict=True)
        training_root = args.training_root.resolve(strict=True)
        dataset = args.dataset.resolve(strict=True)
        source_archive = args.source_archive.resolve(strict=True)
        if sha256_file(dataset) != DATASET_SHA256:
            raise EvaluationFailure("canonical CIFAR-10 archive SHA256 mismatch")
        if sha256_file(source_archive) != args.source_archive_sha256:
            raise EvaluationFailure("evaluator source archive SHA256 mismatch")
        cache_template = None
        cache_template_manifest = None
        if args.cache_template is not None:
            cache_template = args.cache_template.resolve(strict=True)
            if cache_template.is_symlink() or not cache_template.is_dir():
                raise EvaluationFailure(f"invalid cache template: {cache_template}")
            cache_template_manifest = cache_manifest(cache_template)
        if output_root.exists() or output_root.is_symlink():
            raise EvaluationFailure(f"refusing to overwrite evaluation output: {output_root}")
        output_root.mkdir(parents=False)
        if cache_template is not None:
            copied_cache = output_root / "evaluator_cache"
            shutil.copytree(cache_template, copied_cache, copy_function=shutil.copy2)
            copied_manifest = cache_manifest(copied_cache)
            if copied_manifest["files"] != cache_template_manifest["files"]:
                raise EvaluationFailure("copied evaluator cache differs from the verified template")
        sys.path.insert(0, str(source_root))
        from scripts import run_q256_target_weight_evaluation as frozen

        if frozen.DATASET_SHA256 != DATASET_SHA256:
            raise EvaluationFailure("frozen evaluator dataset contract drift")
        if tuple(frozen.METRICS) != METRICS or frozen.SAMPLE_COUNT != SAMPLE_COUNT:
            raise EvaluationFailure("frozen evaluator metric contract drift")
        cells = [validate_training_cell(training_root, args.seed, arm) for arm in ARMS]
        jobs = build_jobs(
            source_root=source_root, output_root=output_root, dataset=dataset,
            cells=cells, base_port=args.base_port,
        )
        runtime = runtime_manifest()
        plan = {
            "schema": "q256-target-weight-seed14-18-1024k-frozen-evaluation-plan-v1",
            "status": "RUNNING",
            "created_at_utc": utc_now(),
            "training_source_commit": TRAINING_COMMIT,
            "parent_256k_training_source_commit": PARENT_TRAINING_COMMIT,
            "training_budget_kimg": 1024,
            "evaluator_source_commit": EVALUATOR_COMMIT,
            "evaluator_source_archive": str(source_archive),
            "evaluator_source_archive_sha256": args.source_archive_sha256,
            "seed": args.seed,
            "physical_gpu_index": args.gpu,
            "dataset": {"path": str(dataset), "sha256": DATASET_SHA256, "bytes": dataset.stat().st_size},
            "cache_bootstrap": None if cache_template_manifest is None else {
                "source": str(cache_template),
                "source_tree_sha256": cache_template_manifest["tree_sha256"],
                "copied_tree_sha256": copied_manifest["tree_sha256"],
                "purpose": "recover from incomplete external detector download; numerical evaluator unchanged",
            },
            "protocol": {
                "precision": "fp32", "sample_count": SAMPLE_COUNT,
                "sample_seeds": "0-49999", "metric_seed": METRIC_SEED,
                "metrics": list(METRICS), "metric_repeats": 1,
                "eval_batch": 512, "metric_generator_batch": 128, "workers": 3,
                "nfe_modes": {"1": [], "2": [0.821]},
                "retain_generated_artifacts": True,
                "selection_policy": "all_exact_final_1024kimg_cells_no_intermediate_selection",
            },
            "runtime": runtime,
            "training_cells": cells,
            "jobs": jobs,
        }
        write_json_exclusive(output_root / "evaluation_plan.json", plan)
        receipts: list[dict[str, Any]] = []
        env = dict(os.environ)
        env["DNNLIB_CACHE_DIR"] = str(output_root / "evaluator_cache")
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["PYTHONUNBUFFERED"] = "1"
        for job in jobs:
            target = Path(job["output_directory"])
            if target.exists() or target.is_symlink():
                raise EvaluationFailure(f"job output unexpectedly exists: {target}")
            before_checkpoint = sha256_file(Path(job["checkpoint"]))
            job_started = utc_now()
            process_log = output_root / "process_logs" / f"{job['job_id']}.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            returncode = run_process(
                job["command_argv"], cwd=source_root, env=env,
                log_path=process_log, timeout_seconds=args.job_timeout_seconds,
            )
            if returncode != 0:
                raise EvaluationFailure(f"{job['job_id']} exited with status {returncode}")
            validated = frozen.validate_job_outputs(job, dataset=dataset, output_root=output_root)
            after_checkpoint = sha256_file(Path(job["checkpoint"]))
            if before_checkpoint != after_checkpoint or after_checkpoint != job["checkpoint_sha256"]:
                raise EvaluationFailure(f"checkpoint changed during {job['job_id']}")
            receipt = {
                "schema": "q256-target-weight-seed14-18-1024k-evaluation-job-receipt-v1",
                "status": "PASS", "job": job, "started_at_utc": job_started,
                "completed_at_utc": utc_now(), "process_log": str(process_log),
                "process_log_sha256": sha256_file(process_log), "validation": validated,
            }
            receipt_path = output_root / "receipts" / f"{job['job_id']}.json"
            write_json_exclusive(receipt_path, receipt)
            receipts.append({
                "job_id": job["job_id"], "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path), "metrics": validated["metrics"],
            })
            print(f"JOB_PASS {job['job_id']}", flush=True)
        completion = {
            "schema": "q256-target-weight-seed14-18-1024k-evaluation-worker-completion-v1",
            "status": "WORKER_PASS", "seed": args.seed, "physical_gpu_index": args.gpu,
            "started_at_utc": started, "completed_at_utc": utc_now(),
            "evaluation_plan": str(output_root / "evaluation_plan.json"),
            "evaluation_plan_sha256": sha256_file(output_root / "evaluation_plan.json"),
            "jobs_completed": len(receipts), "job_receipts": receipts,
        }
        write_json_exclusive(output_root / "WORKER_PASS.json", completion)
        print(f"WORKER_PASS seed={args.seed} jobs={len(receipts)}", flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "schema": "q256-target-weight-seed14-18-1024k-evaluation-failure-v1",
            "status": "FAIL", "seed": args.seed, "physical_gpu_index": args.gpu,
            "started_at_utc": started, "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        }
        if not failure_path.exists():
            write_json_exclusive(failure_path, failure)
        print(f"EVALUATION_FAILURE seed={args.seed}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
