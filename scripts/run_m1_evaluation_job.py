#!/usr/bin/env python3
"""Run one frozen M1 NFE1 evaluation slot with its exact readout and seed block."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots
from scripts import validate_m1_evaluation_job as validation


def build_command(
    slot: Mapping[str, str],
    snapshot_path: str,
    dataset: Path,
    output: Path,
    evaluator_repo: Path,
    runtime_python: Path,
    master_port: int,
) -> list[str]:
    start, end = int(slot["sample_seed_start"]), int(slot["sample_seed_end"])
    if (
        slot["block"] not in slots.BLOCKS
        or slots.BLOCKS[slot["block"]] != (start, end)
    ):
        raise validation.ValidationError("slot block and sample range disagree")
    if (
        slot["readout"] not in slots.READOUT_BLOCKS
        or slot["block"] not in slots.READOUT_BLOCKS[slot["readout"]]
    ):
        raise validation.ValidationError("slot readout does not permit this generation block")
    if slot["nfe"] != "1" or slot["precision"] != "fp32":
        raise validation.ValidationError("M1 worker only supports NFE1 FP32")
    return [
        str(runtime_python), "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", f"--master_port={master_port}",
        str(evaluator_repo / "ct_eval.py"),
        "--resume", snapshot_path, "--outdir", str(output), "--nosubdir",
        "--data", str(dataset), "--cond=False", "--arch=ddpmpp", "--precond=ct",
        "--dropout=0.2", "--augment=0", "--xflip=False", "--fp16=False",
        "--cache=True", "--workers=1", "--eval-batch=512",
        "--metric-generator-batch=128", "--nfe=1",
        f"--metrics={slot['metrics']}", "--metric-repeats=1",
        f"--sample-seeds={start}-{end}", f"--seed={slot['metric_seed']}",
        "--retain-generated-artifacts", f"--desc=m1-{slot['slot_id']}",
    ]


def runtime_env(
    runtime_base: Path,
    runtime_environment: Path,
    cache: Path,
    gpu_index: int,
    master_port: int,
    runtime_library_paths: list[str] | None = None,
) -> dict[str, str]:
    base = runtime_base.resolve()
    environment = runtime_environment.resolve()
    library_paths = [
        Path(path) for path in (
            runtime_library_paths or [
                str(base / "lib/python3.11/site-packages/torch/lib"),
                str(base / "lib"),
            ]
        )
    ] + [Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu")]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(gpu_index),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        DNNLIB_CACHE_DIR=str(cache),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        LD_LIBRARY_PATH=":".join(str(path) for path in library_paths),
        PATH=f"{environment / 'bin'}:{base / 'bin'}:/usr/bin:/bin",
    )
    return env


def gpu_identity(gpu_index: int) -> tuple[str, str]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    matches = [
        row.strip()
        for row in rows
        if row.split(",", 1)[0].strip() == str(gpu_index)
    ]
    if len(matches) != 1 or "A100" not in matches[0]:
        raise validation.ValidationError("assigned GPU is not one identifiable A100")
    parts = [part.strip() for part in matches[0].split(",")]
    return parts[1], matches[0]


def gpu_resource_probe(gpu_index: int) -> dict[str, object]:
    rows = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    matches = [
        [part.strip() for part in row.split(",")]
        for row in rows if row.split(",", 1)[0].strip() == str(gpu_index)
    ]
    if len(matches) != 1 or len(matches[0]) != 5:
        raise validation.ValidationError("assigned GPU resource probe is ambiguous")
    index, uuid, name, free_mib, utilization = matches[0]
    result = {
        "index": int(index), "uuid": uuid, "name": name,
        "free_mib": int(free_mib), "utilization_percent": int(utilization),
    }
    if (
        "A100" not in name or result["free_mib"] < 35_000
        or result["utilization_percent"] > 5
    ):
        raise validation.ValidationError("A100 free-memory/idle resource gate failed")
    return result


def disk_resource_probe(path: Path, minimum_free_bytes: int = 5 << 30) -> dict[str, object]:
    target = path.resolve()
    while not target.exists():
        if target.parent == target:
            raise validation.ValidationError("cannot locate filesystem for output path")
        target = target.parent
    free_bytes = shutil.disk_usage(target).free
    if free_bytes < minimum_free_bytes:
        raise validation.ValidationError("evaluation filesystem has insufficient free space")
    return {
        "filesystem_probe_path": str(target), "free_bytes": free_bytes,
        "minimum_free_bytes": minimum_free_bytes,
    }


def copy_cache(cache_root: Path, destination: Path) -> None:
    source = validate_cache_root(cache_root)
    try:
        shutil.copytree(source, destination, copy_function=os.link)
    except OSError:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def validate_cache_root(cache_root: Path) -> Path:
    source = cache_root.resolve(strict=True)
    if not (source / "downloads").is_dir():
        raise validation.ValidationError("cache template lacks downloads directory")
    return source


def run_canary(
    command: list[str],
    snapshot_path: str,
    runtime_python: Path,
    evaluator_repo: Path,
    environment: dict[str, str],
    expected_pip_freeze_sha256: str | None = None,
) -> dict[str, object]:
    probe = validation.probe_live_runtime(
        runtime_python, environment, expected_pip_freeze_sha256
    )
    snapshot_code = (
        "import pickle,sys;"
        "payload=pickle.load(open(sys.argv[1],'rb'));"
        "assert isinstance(payload,dict) and payload.get('ema') is not None;"
        "payload['ema'].eval();"
        "print('M1_SNAPSHOT_LOAD_PASS')"
    )
    subprocess.run(
        [str(runtime_python), "-c", snapshot_code, snapshot_path],
        cwd=evaluator_repo,
        env=environment,
        check=True,
        timeout=300,
    )
    subprocess.run(
        command + ["--dry_run"],
        cwd=evaluator_repo,
        env=environment,
        check=True,
        timeout=600,
    )
    return {"status": "G4_NO_QUALITY_CANARY_PASS", "runtime_probe": probe}


def run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, bool, float]:
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=6 * 3600)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    return return_code, timed_out, time.monotonic() - started


def execute(args: argparse.Namespace) -> int:
    manifest_path = args.manifest_csv.resolve(strict=True)
    slot = validation.load_slot(manifest_path, args.slot_id)
    training = slots.load_training_identity(args.training_manifest)
    implementation_checkout = validation.verify_implementation_checkout(
        training["implementation_commit"]
    )
    snapshot_receipt_path = args.snapshot_receipt.resolve(strict=True)
    snapshot = validation.load_snapshot_receipt(
        snapshot_receipt_path, slot, training
    )
    evaluator_repo = args.evaluator_repo.resolve(strict=True)
    evaluator = validation.verify_evaluator(
        evaluator_repo, slot["evaluator_commit"], args.evaluator_archive
    )
    runtime = validation.verify_runtime(
        args.runtime_base, args.runtime_env, args.runtime_receipt
    )
    dataset = args.evaluation_dataset.resolve(strict=True)
    validation.verify_evaluation_dataset(dataset)

    runtime_python = Path(runtime["runtime_python"])
    master_port = 52_000 + args.gpu_index * 100 + int(slot["slot_index"]) % 100
    output_root = args.output_root.resolve()
    job_dir = output_root / "jobs" / slot["slot_id"] / f"attempt{args.attempt:02d}"
    receipt_path = output_root / "receipts" / f"{slot['slot_id']}-attempt{args.attempt:02d}.json"
    cache_dir = output_root / "job_caches" / slot["slot_id"] / f"attempt{args.attempt:02d}"
    log_path = output_root / "logs" / f"{slot['slot_id']}-attempt{args.attempt:02d}.process.log"
    command = build_command(
        slot, snapshot["snapshot_path"], dataset, job_dir, evaluator_repo,
        runtime_python, master_port,
    )
    if args.dry_run:
        print(json.dumps({"slot_id": slot["slot_id"], "command": command}, indent=2))
        return 0
    validate_cache_root(args.cache_root)
    if any(path.exists() for path in (job_dir, receipt_path, cache_dir, log_path)):
        raise validation.ValidationError("refuse pre-existing output for this slot attempt")

    job_dir.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    copy_cache(args.cache_root, cache_dir)
    gpu_resource = gpu_resource_probe(args.gpu_index)
    gpu_uuid = str(gpu_resource["uuid"])
    gpu_row = ", ".join(str(gpu_resource[key]) for key in (
        "index", "uuid", "name", "free_mib", "utilization_percent"
    ))
    disk_resource = disk_resource_probe(output_root)
    execution_environment = runtime_env(
        Path(runtime["runtime_base"]),
        Path(runtime["runtime_environment"]),
        cache_dir,
        args.gpu_index,
        master_port,
        runtime.get("runtime_library_paths"),
    )
    live_runtime_probe = validation.probe_live_runtime(
        runtime_python, execution_environment,
        runtime["runtime_pip_freeze_sha256"],
    )
    code, timed_out, elapsed = run_process(
        command, evaluator_repo,
        execution_environment,
        log_path,
    )
    try:
        payload = validation.validate_output(
            slot,
            snapshot,
            job_dir,
            dataset,
            process_exit_code=code,
            process_hard_timeout=timed_out,
        )
    except validation.ValidationError as exc:
        failure = {
            "schema": validation.RECEIPT_SCHEMA,
            "status": "INCOMPLETE_TECHNICAL",
            "slot_id": slot["slot_id"],
            "attempt": args.attempt,
            "exit_code": code,
            "hard_timeout": timed_out,
            "sample_seed_start": int(slot["sample_seed_start"]),
            "sample_seed_end": int(slot["sample_seed_end"]),
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "terminal_state_sha256": snapshot["terminal_state_sha256"],
            "branch_manifest_sha256": snapshot["branch_manifest_sha256"],
            "training_manifest_sha256": snapshot["training_manifest_sha256"],
            "implementation_commit": snapshot["implementation_commit"],
            "frozen_source_state_sha256": snapshot["frozen_source_state_sha256"],
            "manifest_sha256": validation.sha256_file(manifest_path),
            "validation_error": str(exc),
        }
        failure.update(runtime)
        failure["training_runtime_receipt_sha256"] = training[
            "training_runtime_receipt_sha256"
        ]
        failure["live_runtime_probe"] = live_runtime_probe
        failure["implementation_checkout"] = implementation_checkout
        failure["gpu_resource_probe"] = gpu_resource
        failure["disk_resource_probe"] = disk_resource
        validation.atomic_json(receipt_path, failure)
        return 2
    payload.update(runtime)
    payload["training_runtime_receipt_sha256"] = training[
        "training_runtime_receipt_sha256"
    ]
    payload["live_runtime_probe"] = live_runtime_probe
    payload["implementation_checkout"] = implementation_checkout
    payload.update(evaluator)
    payload.update(
        attempt=args.attempt,
        elapsed_seconds=elapsed,
        gpu_index=args.gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_identity_row=gpu_row,
        gpu_resource_probe=gpu_resource,
        disk_resource_probe=disk_resource,
        evaluator_source=str(evaluator_repo),
        evaluation_dataset_sha256=validation.DATASET_SHA256,
        manifest_sha256=validation.sha256_file(manifest_path),
        snapshot_receipt_sha256=validation.sha256_file(snapshot_receipt_path),
        process_exit_code=code,
        process_hard_timeout=timed_out,
    )
    validation.atomic_json(receipt_path, payload)
    print(json.dumps(payload["result_row"], sort_keys=True))
    return 0 if payload["metrics"]["fid50k_full"]["status"] == "SEALED_PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, choices=range(3), default=0)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--evaluator-archive", type=Path)
    parser.add_argument("--runtime-base", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
