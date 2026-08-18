#!/usr/bin/env python3
"""Prepare or execute the frozen 27-cell publication-v2 evaluation.

The runner is intentionally fail-closed: it verifies every frozen input before
launch, refuses to overwrite an existing result root, uses one process per GPU,
never retries a failed cell, and validates the retained samples/features before
marking a cell complete.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EVALUATOR_SEED = 20260730
SOURCE_COMMIT = "1c8971e78637a31f044a5e05e262c08adf15c5d2"
BLOCKS = ((5000, 9999), (10000, 14999), (15000, 19999))
CHECKPOINT_SHA256 = {
    (3, "A"): "fa48bf5a3c7488678e3efd79d43229196bba6e24bacad2c7fc4cda5c2ff1c32b",
    (3, "B"): "a698182f3bbc8307fe1c36c229e5b50772f7fe7e532868353ddf5e395c0ee4db",
    (3, "C"): "0caf658fdffc30a5d9fd3d143da1a86a7cf40152403e7235cb2b8ae392bc1639",
    (4, "A"): "ec724a4705cab6a789f05404a2fc82b362d5e3ef3aa5ed24735b82583059b684",
    (4, "B"): "e6adb0548babb1de2aaa4a55e22ae4adfbe4d7daae2f2547e11e46628b726595",
    (4, "C"): "b5d19259a9089ba2bc8b8cb90e7dcd669b065a364efbb4f99736aae5bdded31e",
    (5, "A"): "97837ecba0f11d5b7d25c1eada17adf8ce5d5671ceae6553291f1405c5c16455",
    (5, "B"): "fce3c1f2c14357b617f51e7220dd3dfe0e02c3e9894318678d7e167bff6af36a",
    (5, "C"): "48e7fa22cef49b158b9b99da71f20c472149ebced9028b6f5c165653a2762852",
}
DATASET_SHA256 = "a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1"
METRICS = ("fid5k_full", "kid5k_full")
CONFIRMATION_TOKEN = "PUBLICATION_V2_27_CELL_V2_APPROVED"
MINIMUM_FREE_GPU_MIB = 70000
STABLE_GPU_PROBES = 3
RESOURCE_POLL_SECONDS = 30
RESOURCE_WAIT_TIMEOUT_SECONDS = 1800


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty required file: {path}")


def require_sha256(path: Path, expected: str) -> None:
    require_file(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, observed {observed}"
        )


def parse_cache_manifest(cache_root: Path) -> dict[str, str]:
    manifest = cache_root / "SHA256SUMS"
    require_file(manifest)
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        entries[relative] = digest
    if len(entries) != 3:
        raise RuntimeError(f"expected 3 dnnlib cache entries, found {len(entries)}")
    for relative, expected in entries.items():
        require_sha256(cache_root / relative, expected)
    return entries


def validate_inputs(root: Path) -> dict[str, Any]:
    source = root / "source" / "repo"
    python = root / "runtime" / "ect-publication-py310-v2" / "bin" / "python"
    dataset = root / "frozen_inputs" / "dataset" / "cifar10-32x32.zip"
    cache_root = root / "runtime" / "dnnlib-cache-v2"
    require_file(source / "ct_eval.py")
    require_file(source / "metrics" / "metric_utils.py")
    require_file(python)
    require_sha256(dataset, DATASET_SHA256)
    checkpoints: dict[str, dict[str, str]] = {}
    for (seed, arm), expected in sorted(CHECKPOINT_SHA256.items()):
        path = root / "frozen_inputs" / "checkpoints" / f"s{seed}-{arm}-k256000.pkl"
        require_sha256(path, expected)
        checkpoints[f"s{seed}-{arm}"] = {"path": str(path), "sha256": expected}
    cache_entries = parse_cache_manifest(cache_root)
    return {
        "source": str(source),
        "python": str(python),
        "dataset": {"path": str(dataset), "sha256": DATASET_SHA256},
        "checkpoints": checkpoints,
        "dnnlib_cache": {
            "path": str(cache_root),
            "entries": cache_entries,
            "manifest_sha256": sha256_file(cache_root / "SHA256SUMS"),
        },
    }


def build_cells(root: Path) -> list[dict[str, Any]]:
    source = root / "source" / "repo"
    python = root / "runtime" / "ect-publication-py310-v2" / "bin" / "python"
    dataset = root / "frozen_inputs" / "dataset" / "cifar10-32x32.zip"
    output = root / "output" / "regenerated-disjoint-5k-v2"
    cells: list[dict[str, Any]] = []
    for start, end in BLOCKS:
        for seed in (3, 4, 5):
            for arm in ("A", "B", "C"):
                checkpoint = (
                    root
                    / "frozen_inputs"
                    / "checkpoints"
                    / f"s{seed}-{arm}-k256000.pkl"
                )
                outdir = (
                    output
                    / "blocks"
                    / f"block_{start}_{end}"
                    / f"seed{seed}"
                    / f"arm_{arm.lower()}"
                )
                command = [
                    str(python),
                    str(source / "ct_eval.py"),
                    f"--data={dataset}",
                    f"--outdir={outdir}",
                    "--nosubdir",
                    "--cond=False",
                    "--arch=ddpmpp",
                    "--precond=ct",
                    "--dropout=0.2",
                    "--augment=0",
                    "--fp16=False",
                    "--eval-batch=64",
                    "--metric-generator-batch=32",
                    f"--seed={EVALUATOR_SEED}",
                    f"--resume={checkpoint}",
                    "--nfe=1",
                    f"--metrics={','.join(METRICS)}",
                    "--metric-repeats=1",
                    f"--sample-seeds={start}-{end}",
                    "--retain-generated-artifacts",
                ]
                cells.append(
                    {
                        "cell_id": f"block_{start}_{end}/seed{seed}/arm_{arm.lower()}",
                        "block": [start, end],
                        "training_seed": seed,
                        "arm": arm,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": CHECKPOINT_SHA256[(seed, arm)],
                        "outdir": str(outdir),
                        "command": command,
                    }
                )
    if len(cells) != 27:
        raise AssertionError(f"internal contract error: {len(cells)} cells")
    return cells


def build_contract(root: Path, gpus: list[str], timeout_seconds: int) -> dict[str, Any]:
    inputs = validate_inputs(root)
    cells = build_cells(root)
    for index, cell in enumerate(cells):
        cell["assigned_gpu"] = gpus[index % len(gpus)]
        cell["master_port"] = 29600 + (index % len(gpus))
    source = root / "source" / "repo"
    runner = source / "scripts" / Path(__file__).name
    return {
        "schema_version": 1,
        "contract_id": "publication-v2-regenerated-disjoint-5k-v2",
        "supersedes_failed_contract_sha256": "01a59322556e148599f4662805e72f0903897a432250018e5729a8a4238636e9",
        "prepared_at_utc": utc_now(),
        "source_commit": SOURCE_COMMIT,
        "runner_sha256": sha256_file(runner),
        "evaluation_source_sha256": {
            "ct_eval.py": sha256_file(source / "ct_eval.py"),
            "metrics/metric_utils.py": sha256_file(source / "metrics" / "metric_utils.py"),
            "metrics/metric_main.py": sha256_file(source / "metrics" / "metric_main.py"),
        },
        "inputs": inputs,
        "protocol": {
            "evidence_class": "regenerated evaluation provenance",
            "nfe": 1,
            "mid_t": [],
            "metrics": list(METRICS),
            "metric_repeats": 1,
            "evaluator_seed": EVALUATOR_SEED,
            "sample_blocks": [f"{start}-{end}" for start, end in BLOCKS],
            "sample_count_per_cell": 5000,
            "precision": "FP32",
            "conditioned": False,
            "architecture": "ddpmpp",
            "preconditioning": "ct",
            "dropout": 0.2,
            "augment": 0,
            "preview_batch_size": 64,
            "metric_generator_batch_size": 32,
            "retry_policy": "none",
            "overwrite_policy": "refuse",
            "retained_artifacts": [
                "generated-samples.npy",
                "generated-features-fid5k_full-repeat00.npy",
                "generated-features-kid5k_full-repeat00.npy",
            ],
        },
        "execution": {
            "gpus": gpus,
            "one_process_per_gpu": True,
            "total_timeout_seconds": timeout_seconds,
            "monitor_interval_seconds": 30,
            "output_root": str(root / "output" / "regenerated-disjoint-5k-v2"),
            "resource_gate": {
                "minimum_free_gpu_mib": MINIMUM_FREE_GPU_MIB,
                "stable_probes": STABLE_GPU_PROBES,
                "poll_seconds": RESOURCE_POLL_SECONDS,
                "wait_timeout_seconds": RESOURCE_WAIT_TIMEOUT_SECONDS,
            },
        },
        "success_criteria": {
            "completed_cells": 27,
            "metric_receipts": 54,
            "one_json_line_per_metric": True,
            "generated_samples": {"shape": [5000, 3, 32, 32], "dtype": "uint8"},
            "generated_features_each": {"shape": [5000, 2048], "dtype": "float32"},
            "all_outputs_sha256_bound": True,
            "no_retries": True,
        },
        "cells": cells,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_metric_receipt(path: Path, metric: str) -> dict[str, Any]:
    require_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one JSONL row: {path}")
    receipt = json.loads(lines[0])
    value = receipt.get("results", {}).get(metric)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"missing or non-finite {metric} value: {path}")
    return receipt


def validate_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
    require_file(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != dtype:
        raise RuntimeError(
            f"invalid array {path}: expected {shape}/{dtype}, got {array.shape}/{array.dtype}"
        )


def validate_cell(cell: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(cell["outdir"])
    receipts = {
        metric: validate_metric_receipt(outdir / f"metric-{metric}.jsonl", metric)
        for metric in METRICS
    }
    samples = outdir / "generated-samples.npy"
    fid_features = outdir / "generated-features-fid5k_full-repeat00.npy"
    kid_features = outdir / "generated-features-kid5k_full-repeat00.npy"
    validate_npy(samples, (5000, 3, 32, 32), np.dtype("uint8"))
    validate_npy(fid_features, (5000, 2048), np.dtype("float32"))
    validate_npy(kid_features, (5000, 2048), np.dtype("float32"))
    artifacts = sorted(
        path
        for path in outdir.iterdir()
        if path.is_file() and path.name not in {"artifact_sha256.json", "completion.json"}
    )
    hashes = {path.name: sha256_file(path) for path in artifacts}
    write_json_atomic(outdir / "artifact_sha256.json", hashes)
    return {
        "cell_id": cell["cell_id"],
        "status": "completed",
        "finished_at_utc": utc_now(),
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "sample_seed_range": f"{cell['block'][0]}-{cell['block'][1]}",
        "metrics": {metric: receipts[metric]["results"][metric] for metric in METRICS},
        "artifact_sha256_manifest": "artifact_sha256.json",
        "artifact_sha256_manifest_sha256": sha256_file(outdir / "artifact_sha256.json"),
    }


def run_process(
    cell: dict[str, Any], root: Path, deadline: float, stop_event: threading.Event
) -> dict[str, Any]:
    if stop_event.is_set():
        return {"cell_id": cell["cell_id"], "status": "not_started_after_failure"}
    outdir = Path(cell["outdir"])
    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing cell: {outdir}")
    wait_for_gpu_capacity(str(cell["assigned_gpu"]), deadline)
    outdir.mkdir(parents=True)
    command = list(cell["command"])
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(cell["assigned_gpu"]),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(cell["master_port"]),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "DNNLIB_CACHE_DIR": str(root / "runtime" / "dnnlib-cache-v2"),
            "PYTHONPYCACHEPREFIX": str(root / "output" / "pycache"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    metadata = {
        "cell": cell,
        "started_at_utc": utc_now(),
        "exact_command_shell": shlex.join(command),
        "environment_overrides": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "DNNLIB_CACHE_DIR",
                "PYTHONPYCACHEPREFIX",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "retry_index": 0,
    }
    write_json_atomic(outdir / "experiment_meta.json", metadata)
    log_path = outdir / "runner.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=root / "source" / "repo",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(f"global timeout reached in {cell['cell_id']}")
            print(
                f"[{utc_now()}] running {cell['cell_id']} on GPU "
                f"{cell['assigned_gpu']} ({int(remaining)}s remaining)",
                flush=True,
            )
            time.sleep(min(30, remaining))
        if process.returncode != 0:
            raise RuntimeError(
                f"cell failed with exit code {process.returncode}: {cell['cell_id']}"
            )
    completion = validate_cell(cell)
    write_json_atomic(outdir / "completion.json", completion)
    print(f"[{utc_now()}] completed {cell['cell_id']}", flush=True)
    return completion


def query_free_gpu_mib(gpu: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    free_by_gpu: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, free_mib = [item.strip() for item in line.split(",", maxsplit=1)]
        free_by_gpu[index] = int(free_mib)
    if gpu not in free_by_gpu:
        raise RuntimeError(f"nvidia-smi did not report GPU {gpu}")
    return free_by_gpu[gpu]


def wait_for_gpu_capacity(gpu: str, global_deadline: float) -> None:
    resource_deadline = min(
        global_deadline, time.monotonic() + RESOURCE_WAIT_TIMEOUT_SECONDS
    )
    stable = 0
    while time.monotonic() < resource_deadline:
        free_mib = query_free_gpu_mib(gpu)
        if free_mib >= MINIMUM_FREE_GPU_MIB:
            stable += 1
        else:
            stable = 0
        print(
            f"[{utc_now()}] GPU {gpu} resource gate: {free_mib} MiB free, "
            f"stable {stable}/{STABLE_GPU_PROBES}",
            flush=True,
        )
        if stable >= STABLE_GPU_PROBES:
            return
        time.sleep(RESOURCE_POLL_SECONDS)
    raise TimeoutError(
        f"GPU {gpu} did not sustain {MINIMUM_FREE_GPU_MIB} MiB free for "
        f"{STABLE_GPU_PROBES} probes"
    )


def execute(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    output_root = Path(contract["execution"]["output_root"])
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    write_json_atomic(output_root / "run_contract.json", contract)
    gpus = list(contract["execution"]["gpus"])
    grouped = {gpu: [] for gpu in gpus}
    for cell in contract["cells"]:
        grouped[cell["assigned_gpu"]].append(cell)
    deadline = time.monotonic() + int(contract["execution"]["total_timeout_seconds"])
    stop_event = threading.Event()

    def worker(gpu: str) -> list[dict[str, Any]]:
        results = []
        for cell in grouped[gpu]:
            if stop_event.is_set():
                results.append(
                    {"cell_id": cell["cell_id"], "status": "not_started_after_failure"}
                )
                continue
            try:
                results.append(run_process(cell, root, deadline, stop_event))
            except BaseException as error:
                stop_event.set()
                results.append(
                    {
                        "cell_id": cell["cell_id"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        return results

    all_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            all_results.extend(future.result())
    completed = [result for result in all_results if result["status"] == "completed"]
    summary = {
        "contract_id": contract["contract_id"],
        "finished_at_utc": utc_now(),
        "status": "completed" if len(completed) == 27 else "failed",
        "completed_cells": len(completed),
        "total_cells": 27,
        "results": sorted(all_results, key=lambda item: item["cell_id"]),
    }
    write_json_atomic(output_root / "run_summary.json", summary)
    if summary["status"] != "completed":
        raise RuntimeError(f"evaluation incomplete: {len(completed)}/27 cells")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.publication_root.resolve()
    if not root.is_absolute() or not (root / "source" / "repo").is_dir():
        raise RuntimeError(f"invalid publication root: {root}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise RuntimeError("the frozen contract requires exactly two distinct GPUs")
    if args.timeout_seconds <= 0:
        raise RuntimeError("timeout must be positive")
    contract_path = root / "logs" / "publication_v2_eval_contract_v2.json"
    if args.prepare_only:
        contract = build_contract(root, gpus, args.timeout_seconds)
        write_json_atomic(contract_path, contract)
        print(json.dumps({
            "status": "prepared",
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "cells": len(contract["cells"]),
            "gpus": gpus,
            "timeout_seconds": args.timeout_seconds,
        }, indent=2))
        return 0
    require_file(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "publication-v2-regenerated-disjoint-5k-v2":
        raise RuntimeError(f"unexpected contract identity: {contract_path}")
    if contract.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("runner SHA256 differs from the frozen contract")
    if contract.get("execution", {}).get("gpus") != gpus:
        raise RuntimeError("GPU selection differs from the frozen contract")
    if contract.get("execution", {}).get("total_timeout_seconds") != args.timeout_seconds:
        raise RuntimeError("timeout differs from the frozen contract")
    current_inputs = validate_inputs(root)
    if current_inputs != contract.get("inputs"):
        raise RuntimeError("current frozen inputs differ from the frozen contract")
    if args.confirmation_token != CONFIRMATION_TOKEN:
        raise RuntimeError(
            "execution requires the post-review confirmation token " + CONFIRMATION_TOKEN
        )
    summary = execute(contract, root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
