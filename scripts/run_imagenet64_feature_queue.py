#!/usr/bin/env python3
"""Run a deterministic shard of the 108-job ImageNet-64 feature matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 102, 103)
METHODS = ("IA", "IB")
MILESTONES_KIMG = tuple(range(2560, 12801, 1280))
NFES = (1, 2)
FEATURE_SHAPE = (50_000, 2_048)
JOB_COUNT = len(SEEDS) * len(METHODS) * len(MILESTONES_KIMG) * len(NFES)


@dataclass(frozen=True)
class Job:
    seed: int
    method: str
    kimg: int
    nfe: int

    @property
    def name(self) -> str:
        return f"seed{self.seed}-{self.method}-kimg{self.kimg:06d}-nfe{self.nfe}"


def jobs() -> tuple[Job, ...]:
    return tuple(
        Job(seed=seed, method=method, kimg=kimg, nfe=nfe)
        for kimg in MILESTONES_KIMG
        for seed in SEEDS
        for method in METHODS
        for nfe in NFES
    )


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(token.strip() for token in value.split(",") if token.strip())
    if not 1 <= len(gpus) <= 3 or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus requires one to three distinct GPU IDs")
    return gpus


def shard_jobs(
    matrix: tuple[Job, ...], shard_index: int, shard_count: int
) -> tuple[Job, ...]:
    return tuple(
        job for index, job in enumerate(matrix)
        if index % shard_count == shard_index
    )


def worker_port(base_port: int, shard_index: int, worker_index: int) -> int:
    return base_port + shard_index * 3 + worker_index


def snapshot_path(run_root: Path, job: Job) -> Path:
    return (
        run_root
        / f"seed{job.seed}"
        / job.method
        / f"network-snapshot-kimg{job.kimg:06d}-0.050.pkl"
    )


def job_dir(feature_root: Path, job: Job) -> Path:
    return (
        feature_root
        / f"seed{job.seed}"
        / job.method
        / f"kimg{job.kimg:06d}"
        / f"nfe{job.nfe}"
    )


def feature_path(feature_root: Path, job: Job) -> Path:
    return job_dir(feature_root, job) / "features.final.npy"


def validate_feature_file(path: Path, job: Job) -> None:
    try:
        features = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{job.name} has an invalid existing feature file") from exc
    if features.shape != FEATURE_SHAPE or features.dtype != np.float32:
        raise RuntimeError(
            f"{job.name} existing feature header is {features.shape} "
            f"{features.dtype}, expected {FEATURE_SHAPE} float32"
        )


def evaluation_command(
    args: argparse.Namespace, job: Job, worker_index: int
) -> list[str]:
    target_dir = job_dir(args.feature_root, job)
    command = [
        str(args.python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        f"--master_port={worker_port(args.base_port, args.shard_index, worker_index)}",
        str(args.repo / "ct_eval.py"),
        f"--outdir={target_dir}",
        "--nosubdir",
        f"--data={args.data}",
        "--cond=True",
        "--arch=edm2",
        "--preset=edm2-img64-s",
        "--fp16=False",
        "--bench=False",
        "--cache=False",
        "--workers=1",
        "--eval-batch=128",
        "--metric-generator-batch=32",
        f"--resume={snapshot_path(args.run_root, job)}",
        f"--nfe={job.nfe}",
        "--metrics=none",
        "--sample-seeds=0-49999",
        "--seed=20260730",
        "--feature-only",
        f"--feature-output={feature_path(args.feature_root, job)}",
        f"--desc={job.name}",
    ]
    if job.nfe == 2:
        command.append("--mid_t=1.526")
    return command


def run_job(
    args: argparse.Namespace,
    job: Job,
    gpu: str,
    worker_index: int,
) -> Job:
    target_dir = job_dir(args.feature_root, job)
    target_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    log_path = target_dir / "feature.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            evaluation_command(args, job, worker_index),
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.job_timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"{job.name} exceeded its job timeout")
    if returncode != 0:
        raise RuntimeError(f"{job.name} failed with exit code {returncode}")
    output = feature_path(args.feature_root, job)
    validate_feature_file(output, job)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_gpus, default=("0", "1", "2"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=29610)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 24 * 60 * 60)
    parser.add_argument("--job-timeout-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in ("run_root", "feature_root", "data", "repo", "python"):
        setattr(args, name, getattr(args, name).resolve())

    worker_count = len(args.gpus)
    if (
        args.timeout_seconds <= 0
        or args.job_timeout_seconds <= 0
        or args.poll_seconds <= 0
    ):
        parser.error("timeouts must be positive")
    full_matrix = jobs()
    if len(full_matrix) != JOB_COUNT:
        raise RuntimeError(
            f"internal feature matrix must contain exactly {JOB_COUNT} jobs"
        )
    if not 1 <= args.shard_count <= len(full_matrix):
        parser.error(f"--shard-count must be between 1 and {JOB_COUNT}")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if (
        args.base_port < 1
        or worker_port(args.base_port, args.shard_index, worker_count - 1) > 65535
    ):
        parser.error("--base-port does not leave enough shard worker ports")
    matrix = shard_jobs(full_matrix, args.shard_index, args.shard_count)

    if args.dry_run:
        for index, job in enumerate(matrix):
            worker_index = index % worker_count
            gpu = args.gpus[worker_index]
            print(
                f"CUDA_VISIBLE_DEVICES={gpu} "
                + shlex.join(evaluation_command(args, job, worker_index))
            )
        print(
            f"total={JOB_COUNT} shard={len(matrix)} "
            f"shard_index={args.shard_index} shard_count={args.shard_count} "
            f"workers={worker_count} retries=0 outputs=features-only"
        )
        return

    if not (args.repo / "ct_eval.py").is_file():
        parser.error(f"missing evaluator: {args.repo / 'ct_eval.py'}")
    if not args.data.exists():
        parser.error(f"missing dataset: {args.data}")

    pending = []
    for job in matrix:
        output = feature_path(args.feature_root, job)
        if output.exists():
            validate_feature_file(output, job)
        else:
            pending.append(job)
    deadline = time.monotonic() + args.timeout_seconds
    free_workers = list(range(worker_count))
    running: dict[Future[Job], tuple[int, Job]] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while pending or running:
            for future, (worker_index, job) in list(running.items()):
                if not future.done():
                    continue
                del running[future]
                free_workers.append(worker_index)
                try:
                    future.result()
                    print(f"DONE {job.name}", flush=True)
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"FAILED {exc}", file=sys.stderr, flush=True)

            ready = next(
                (
                    job for job in pending
                    if snapshot_path(args.run_root, job).is_file()
                    and snapshot_path(args.run_root, job).stat().st_size > 0
                ),
                None,
            )
            while ready is not None and free_workers:
                pending.remove(ready)
                worker_index = free_workers.pop(0)
                gpu = args.gpus[worker_index]
                future = executor.submit(
                    run_job, args, ready, gpu, worker_index
                )
                running[future] = (worker_index, ready)
                print(f"START {ready.name} gpu={gpu}", flush=True)
                ready = next(
                    (
                        job for job in pending
                        if snapshot_path(args.run_root, job).is_file()
                        and snapshot_path(args.run_root, job).stat().st_size > 0
                    ),
                    None,
                )

            if pending and not running and time.monotonic() >= deadline:
                missing = ", ".join(job.name for job in pending[:5])
                parser.error(
                    f"timed out waiting for {len(pending)} snapshots; first: {missing}"
                )
            if pending or running:
                time.sleep(args.poll_seconds)

    if failures:
        raise SystemExit(
            f"{len(failures)} feature jobs failed; no jobs were retried automatically"
        )
    print(
        f"COMPLETE total={JOB_COUNT} shard={len(matrix)} "
        f"shard_index={args.shard_index} shard_count={args.shard_count} "
        f"workers={worker_count} retries=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
