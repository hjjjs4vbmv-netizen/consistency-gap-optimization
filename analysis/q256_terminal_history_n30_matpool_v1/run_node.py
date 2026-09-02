#!/usr/bin/env python3
"""Fail-closed two-node launcher for the q256 terminal history replication."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training import reproducibility, schedule_switch  # noqa: E402


EXPERIMENT_ID = schedule_switch.TERMINAL_HISTORY_N30_PROTOCOL
SCHEMA = "ect.q256.terminal-history-n30-protocol/v1"
SEEDS = tuple(range(50, 80))
CELLS = {"AA": ("A", "A"), "BA": ("B", "A")}
ARM_FACTORS = {"A": (1.0, 1.0), "B": (1.1, 1.1)}
NODE_SEEDS = {
    "node8": tuple(range(50, 66)),
    "node7": tuple(range(66, 80)),
}
NODE_GPU_COUNTS = {"node8": 8, "node7": 7}
PREFIX_MILESTONES = (512,)
SUFFIX_MILESTONES = (640, 768, 896, 1024)
TAPE_FIELDS = ("attempted_iteration", "batch_sha256", "t_sha256", "base_r_sha256")
ZERO_FIELDS = (
    "loss_nonfinite_count",
    "sanitized_grad_nonfinite_count",
    "update_nonfinite_count",
    "model_nonfinite_count",
    "ema_nonfinite_count",
    "factor_nonfinite_count",
    "nonpositive_denominator_count",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    os.unlink(temporary)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_protocol(path: Path) -> dict:
    path = path.resolve(strict=True)
    companion = path.with_name("protocol.sha256")
    expected = companion.read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("protocol SHA256 companion mismatch")
    protocol = load_json(path)
    if protocol.get("schema") != SCHEMA or protocol.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("protocol identity mismatch")
    if protocol.get("seeds") != list(SEEDS) or protocol.get("cells") != list(CELLS):
        raise RuntimeError("protocol seed/cell matrix mismatch")
    if protocol.get("node_seeds") != {key: list(value) for key, value in NODE_SEEDS.items()}:
        raise RuntimeError("protocol node assignment mismatch")
    if protocol.get("node_gpu_counts") != NODE_GPU_COUNTS:
        raise RuntimeError("protocol GPU counts mismatch")
    if protocol.get("training", {}).get("switch_kimg") != 512:
        raise RuntimeError("protocol switch point mismatch")
    if protocol.get("training", {}).get("final_kimg") != 1024:
        raise RuntimeError("protocol endpoint mismatch")
    if protocol.get("analysis", {}).get("primary_contrast") != "logFID_BA-logFID_AA":
        raise RuntimeError("protocol primary contrast mismatch")
    return protocol


def query_gpus() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected GPU row: {line}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "driver_version": fields[4],
                "pci_bus_id": fields[5],
            }
        )
    return rows


def compute_apps() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit():
            rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1],
                         "process_name": fields[2], "used_gpu_memory": fields[3]})
    return rows


def runtime_fingerprint(python: Path) -> dict:
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    probe = subprocess.check_output(
        [
            str(python),
            "-c",
            (
                "import json,platform,torch,numpy,scipy,PIL,click,psutil;"
                "print(json.dumps({'python':platform.python_version(),"
                "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
                "'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,"
                "'scipy':scipy.__version__,'pillow':PIL.__version__,"
                "'click':click.__version__,'psutil':psutil.__version__},sort_keys=True))"
            ),
        ],
        text=True,
    )
    return {
        "python_executable": str(python.resolve(strict=True)),
        "pip_freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "probe": json.loads(probe),
    }


def node_preflight(protocol_path: Path, node_id: str, receipt: Path) -> None:
    protocol = validate_protocol(protocol_path)
    expected_count = NODE_GPU_COUNTS[node_id]
    expected_seeds = NODE_SEEDS[node_id]
    gpus = query_gpus()
    if len(gpus) != expected_count or any("A100" not in row["name"] for row in gpus):
        raise RuntimeError(f"{node_id} requires exactly {expected_count} visible A100 GPUs")
    if compute_apps():
        raise RuntimeError("preflight requires all visible GPUs to be compute-idle")
    dataset = Path(protocol["assets"]["dataset"]["path"])
    transfer = Path(protocol["assets"]["transfer"]["path"])
    if sha256_file(dataset) != protocol["assets"]["dataset"]["sha256"]:
        raise RuntimeError("dataset SHA256 mismatch")
    if sha256_file(transfer) != protocol["assets"]["transfer"]["sha256"]:
        raise RuntimeError("transfer SHA256 mismatch")
    runtime = runtime_fingerprint(Path(protocol["runtime"]["python"]))
    if runtime["pip_freeze_sha256"] != protocol["runtime"]["pip_freeze_sha256"]:
        raise RuntimeError("runtime pip-freeze SHA256 mismatch")
    if runtime["probe"] != protocol["runtime"]["probe"]:
        raise RuntimeError("runtime package probe mismatch")
    output_root = Path(protocol["paths"]["output_root"])
    if output_root.exists():
        raise RuntimeError(f"fresh output root already exists: {output_root}")
    free_bytes = os.statvfs(output_root.parent).f_bavail * os.statvfs(output_root.parent).f_frsize
    minimum = int(protocol["storage"]["minimum_free_bytes_by_node"][node_id])
    if free_bytes < minimum:
        raise RuntimeError(f"insufficient storage: {free_bytes} < {minimum}")
    atomic_json(receipt, {
        "schema": "ect.q256.terminal-history-node-preflight/v1",
        "status": "PASS",
        "observed_at": utc_now(),
        "hostname": socket.gethostname(),
        "node_id": node_id,
        "seeds": list(expected_seeds),
        "gpus": gpus,
        "gpu_idle": True,
        "runtime": runtime,
        "dataset_sha256": protocol["assets"]["dataset"]["sha256"],
        "transfer_sha256": protocol["assets"]["transfer"]["sha256"],
        "free_bytes": free_bytes,
        "protocol_sha256": sha256_file(protocol_path),
    })


def cell_environment(gpu: int, python: Path) -> dict[str, str]:
    env = os.environ.copy()
    prefix = python.parent.parent
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_CACHE_DISABLE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(49000 + gpu),
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PATH=f"{prefix / 'bin'}:{env.get('PATH', '')}",
        LD_LIBRARY_PATH=f"{torch_lib}:{prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
    )
    return env


def training_command(
    protocol: dict,
    run_dir: Path,
    seed: int,
    arm: str,
    gpu: int,
    *,
    prefix: bool,
    manifest: Path | None = None,
    source: Path | None = None,
) -> list[str]:
    python = Path(protocol["runtime"]["python"])
    target, denominator = ARM_FACTORS[arm]
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", f"--master_port={49000 + gpu}", str(REPO_ROOT / "ct_train.py"),
        f"--data={protocol['assets']['dataset']['path']}", f"--outdir={run_dir}", "--nosubdir",
        "--cond=False", "--arch=ddpmpp", "--precond=ect", "--batch=128", "--batch-gpu=16",
        "--optim=RAdam", "--lr=0.0001", "--dropout=0.2", "--augment=0", "--xflip=False",
        "--mean=-1.1", "--std=2.0", "--mapping=sigmoid", "--global-gap-scale=1.0",
        "--factorial-protocol=q256_target_weight_v1", f"--target-gap-scale={target}",
        f"--denominator-gap-scale={denominator}", "-q", "256", "-k", "8", "-b", "1", "-c", "0",
        "--double=10000", "--ema_beta=0.9993", f"--seed={seed}", "--fp16=True", "--tf32=False",
        "--ls=1.0", "--enable_amp=True", "--bench=False", "--cache=True", "--workers=1",
        "--metrics=none", "--duration=1.024", "--tick=10", "--snap=0", "--dump=0", "--ckpt=10",
        "--sample_every=26", "--eval_every=50", "--mid_t=0.821", "--adaptive-update-kimg=0.5",
    ]
    if prefix:
        command.extend([
            "--immutable-checkpoint-kimg=512",
            "--stop-after-attempts=4000",
            f"--planned-pause-protocol={EXPERIMENT_ID}",
            f"--transfer={protocol['assets']['transfer']['path']}",
        ])
    else:
        if manifest is None or source is None:
            raise RuntimeError("suffix command requires manifest and source")
        command.extend([
            "--immutable-checkpoint-kimg=640,768,896,1024",
            f"--schedule-switch-manifest={manifest}",
            f"--resume={source}",
        ])
    return command


def assert_gpu_idle(gpu_uuid: str, *, allow_pid: int | None = None) -> None:
    rows = [row for row in compute_apps() if row["gpu_uuid"] == gpu_uuid]
    if allow_pid is not None:
        rows = [row for row in rows if row["pid"] != allow_pid]
    if rows:
        raise RuntimeError(f"GPU is not exclusive: {rows}")


def run_cell(protocol: dict, run_dir: Path, gpu: int, command: list[str], label: str) -> dict:
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
    gpu_record = query_gpus()[gpu]
    assert_gpu_idle(gpu_record["uuid"])
    protocol_sha = protocol["protocol_sha256_runtime"]
    atomic_json(run_dir / "launch_receipt.json", {
        "schema": "ect.q256.terminal-history-cell-launch/v1",
        "status": "START",
        "label": label,
        "started_at": utc_now(),
        "hostname": socket.gethostname(),
        "gpu": gpu_record,
        "command": command,
        "protocol_sha256": protocol_sha,
        "automatic_retry_count": 0,
    })
    start = time.monotonic()
    launcher_log = (run_dir / "launcher.log").open("xb")
    env = cell_environment(gpu, Path(protocol["runtime"]["python"]))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    total_attempts = 8000 if any(
        item.startswith("--schedule-switch-manifest=") for item in command
    ) else 4000
    monitor = subprocess.Popen(
        [
            protocol["runtime"]["python"],
            str(Path(__file__).with_name("monitor.py")),
            "--pid", str(process.pid),
            "--run-dir", str(run_dir),
            "--gpu-index", str(gpu),
            "--gpu-uuid", gpu_record["uuid"],
            "--total-attempts", str(total_attempts),
            "--interval-seconds", "30",
            "--stall-seconds", "300",
            "--min-free-bytes", str(100 * 1024**3),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    timed_out = False
    try:
        returncode = process.wait(timeout=6 * 3600)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    finally:
        launcher_log.close()
    try:
        monitor.wait(timeout=90)
    except subprocess.TimeoutExpired:
        monitor.terminate()
        monitor.wait(timeout=15)
    status = "PASS" if returncode == 0 and not timed_out else "FAIL"
    receipt = {
        "schema": "ect.q256.terminal-history-cell-completion/v1",
        "status": status,
        "label": label,
        "ended_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start,
        "exit_code": returncode,
        "hard_timeout": timed_out,
        "automatic_retry_count": 0,
        "protocol_sha256": protocol_sha,
    }
    atomic_json(run_dir / "compute_completion_receipt.json", receipt)
    if status == "PASS":
        assert_gpu_idle(gpu_record["uuid"])
    return receipt


def audit_telemetry(path: Path, *, arm: str, first: int, last: int) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    attempts = [int(float(row["attempted_iteration"])) for row in rows]
    if attempts != list(range(first, last + 1)):
        raise RuntimeError(f"telemetry attempt coverage mismatch: {path}")
    nonfinite = []
    for row in rows:
        if row.get("arm") != arm:
            raise RuntimeError(f"telemetry arm mismatch: {path}")
        for field in ZERO_FIELDS:
            if int(float(row[field])) != 0:
                nonfinite.append({"attempt": int(row["attempted_iteration"]), "field": field,
                                  "value": row[field]})
    if nonfinite:
        raise RuntimeError(f"semantic nonfinite telemetry: {nonfinite[:10]}")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path),
            "first_attempt": first, "last_attempt": last, "semantic_nonfinite": 0}


def export_prefix(run_dir: Path, seed: int, arm: str, protocol_sha: str) -> dict:
    state_path = run_dir / "training-state-kimg000512.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["cur_nimg"]) != 512000 or int(state["attempted_iteration"]) != 4000:
        raise RuntimeError("prefix source counter mismatch")
    factorial = state.get("factorial", {})
    target, denominator = ARM_FACTORS[arm]
    if (factorial.get("arm") != arm
            or float(factorial.get("target_gap_scale")) != target
            or float(factorial.get("denominator_gap_scale")) != denominator):
        raise RuntimeError("prefix factorial identity mismatch")
    hashes = schedule_switch.internal_state_hashes(state)
    audit = audit_telemetry(
        run_dir / "factorial_training_telemetry_v1.csv", arm=arm, first=1, last=4000
    )
    source = {
        "path": str(state_path.resolve()),
        "bytes": state_path.stat().st_size,
        "sha256": sha256_file(state_path),
        "internal_state_sha256": hashes,
    }
    receipt = {
        "schema": "ect.q256.terminal-history-prefix-source/v1",
        "status": "PASS",
        "seed": seed,
        "arm": arm,
        "source_kimg": 512,
        "training_state": source,
        "telemetry_audit": audit,
        "protocol_sha256": protocol_sha,
    }
    atomic_json(run_dir / "source_state_receipt.json", receipt)
    return receipt


def copy_csv_prefix(source: Path, destination: Path) -> dict:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [row for row in reader if int(float(row["attempted_iteration"])) <= 4000]
    if [int(float(row["attempted_iteration"])) for row in rows] != list(range(1, 4001)):
        raise RuntimeError(f"source CSV does not contain exact attempts 1..4000: {source}")
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return {"source_path": str(source.resolve()), "source_sha256": sha256_file(source),
            "derived_path": str(destination.resolve()), "derived_sha256": sha256_file(destination),
            "rows": len(rows)}


def copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        while block := reader.read(8 * 1024 * 1024):
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())


def prepare_resume_history(prefix_dir: Path, output: Path) -> dict:
    record = {
        "train_summary": copy_csv_prefix(prefix_dir / "train_summary.csv", output / "train_summary.csv"),
        "factorial_telemetry": copy_csv_prefix(
            prefix_dir / "factorial_training_telemetry_v1.csv",
            output / "source_factorial_training_telemetry_v1.csv",
        ),
    }
    copy_exclusive(prefix_dir / "initial_state_receipt_v1.json", output / "initial_state_receipt_v1.json")
    copy_exclusive(prefix_dir / "training_options.json", output / "source_training_options.json")
    return record


def suffix_manifest(
    protocol: dict,
    protocol_path: Path,
    seed: int,
    cell: str,
    source_receipt: dict,
    output: Path,
) -> Path:
    origin, continuation = CELLS[cell]
    output.mkdir(parents=True, exist_ok=False)
    source = source_receipt["training_state"]
    prefix_dir = Path(source["path"]).parent
    source_receipt_path = prefix_dir / "source_state_receipt.json"
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": EXPERIMENT_ID,
        "run_kind": "formal",
        "branch": cell,
        "seed": seed,
        "origin_arm": origin,
        "continuation_arm": continuation,
        "switch_kimg": 512,
        "final_kimg": 1024,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": protocol["implementation_commit"],
        "source_checkpoint_manifest_sha256": sha256_file(source_receipt_path),
        "source_state": source,
        "source_history_prefix": prepare_resume_history(prefix_dir, output),
        "immutable_output_root": str(output),
    }
    path = output / "formal_run_manifest.json"
    atomic_json(path, manifest)
    schedule_switch.load_run_manifest(path)
    return path


def compare_tapes(paths: list[Path], labels: list[str], destination: Path,
                  protocol_sha: str, *, first: int, last: int) -> None:
    tapes = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [int(float(row["attempted_iteration"])) for row in rows] != list(range(first, last + 1)):
            raise RuntimeError(f"randomness tape coverage mismatch: {path}")
        tapes.append([tuple(row[field] for field in TAPE_FIELDS) for row in rows])
    matched = tapes[0] == tapes[1]
    atomic_json(destination, {
        "schema": "ect.q256.terminal-history-matched-randomness/v1",
        "status": "PASS" if matched else "FAIL",
        "labels": labels,
        "fields": list(TAPE_FIELDS),
        "series_sha256": [canonical_sha256(value) for value in tapes],
        "protocol_sha256": protocol_sha,
    })
    if not matched:
        raise RuntimeError(f"matched randomness failed: {labels}")


def export_suffix(protocol: dict, run_dir: Path, manifest: Path, cell: str) -> dict:
    env = cell_environment(0, Path(protocol["runtime"]["python"]))
    subprocess.run(
        [
            protocol["runtime"]["python"],
            str(REPO_ROOT / "analysis/q256_schedule_switch_v1/export_milestones.py"),
            "--run-dir", str(run_dir),
            "--manifest", str(manifest),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    audit = audit_telemetry(
        run_dir / "schedule_switch_training_telemetry_v1.csv",
        arm="A",
        first=4001,
        last=8000,
    )
    receipt = load_json(run_dir / "trajectory_completion_receipt.json")
    if receipt.get("status") != "PASS" or receipt.get("branch") != cell:
        raise RuntimeError(f"suffix trajectory receipt failed: {run_dir}")
    atomic_json(run_dir / "terminal_history_telemetry_audit.json", {
        "schema": "ect.q256.terminal-history-telemetry-audit/v1",
        "status": "PASS",
        "cell": cell,
        "audit": audit,
        "protocol_sha256": protocol["protocol_sha256_runtime"],
    })
    return receipt


def attempt_prefix(protocol: dict, seed_root: Path, seed: int, arm: str, gpu: int) -> tuple[dict, dict | None]:
    run_dir = seed_root / f"prefix_{arm}"
    command = training_command(protocol, run_dir, seed, arm, gpu, prefix=True)
    completion = run_cell(protocol, run_dir, gpu, command, f"seed{seed}:prefix_{arm}")
    if completion["status"] != "PASS":
        return completion, None
    try:
        source = export_prefix(run_dir, seed, arm, protocol["protocol_sha256_runtime"])
    except Exception as exc:
        atomic_json(run_dir / "postcheck_failure_receipt.json", {
            "schema": "ect.q256.terminal-history-postcheck-failure/v1",
            "status": "FAIL",
            "label": f"seed{seed}:prefix_{arm}",
            "error": repr(exc),
            "automatic_retry_count": 0,
            "protocol_sha256": protocol["protocol_sha256_runtime"],
        })
        return {**completion, "status": "FAIL_POSTCHECK", "error": repr(exc)}, None
    return completion, source


def attempt_suffix(protocol: dict, protocol_path: Path, seed_root: Path, seed: int,
                   cell: str, source: dict | None, gpu: int) -> dict:
    run_dir = seed_root / cell
    if source is None:
        receipt = {
            "schema": "ect.q256.terminal-history-cell-not-run/v1",
            "status": "NOT_RUN_SOURCE_FAILURE",
            "seed": seed,
            "cell": cell,
            "reason": f"prefix_{CELLS[cell][0]} did not yield a validated 512-kimg source",
            "automatic_retry_count": 0,
            "protocol_sha256": protocol["protocol_sha256_runtime"],
        }
        atomic_json(seed_root / f"{cell}_not_run.json", receipt)
        return receipt
    manifest = suffix_manifest(protocol, protocol_path, seed, cell, source, run_dir)
    source_path = Path(source["training_state"]["path"])
    command = training_command(protocol, run_dir, seed, "A", gpu, prefix=False,
                               manifest=manifest, source=source_path)
    completion = run_cell(protocol, run_dir, gpu, command, f"seed{seed}:{cell}")
    if completion["status"] != "PASS":
        return completion
    try:
        export_suffix(protocol, run_dir, manifest, cell)
    except Exception as exc:
        atomic_json(run_dir / "postcheck_failure_receipt.json", {
            "schema": "ect.q256.terminal-history-postcheck-failure/v1",
            "status": "FAIL",
            "label": f"seed{seed}:{cell}",
            "error": repr(exc),
            "automatic_retry_count": 0,
            "protocol_sha256": protocol["protocol_sha256_runtime"],
        })
        return {**completion, "status": "FAIL_POSTCHECK", "error": repr(exc)}
    return completion


def run_seed(protocol: dict, protocol_path: Path, seed: int, gpu: int) -> dict:
    output_root = Path(protocol["paths"]["output_root"])
    seed_root = output_root / "training" / f"seed{seed}"
    seed_root.mkdir(parents=True, exist_ok=False)
    prefix_order = ("A", "B") if seed % 2 == 0 else ("B", "A")
    suffix_order = ("AA", "BA") if seed % 2 == 0 else ("BA", "AA")
    sources: dict[str, dict | None] = {"A": None, "B": None}
    statuses: dict[str, dict] = {}
    for arm in prefix_order:
        completion, source = attempt_prefix(protocol, seed_root, seed, arm, gpu)
        statuses[f"prefix_{arm}"] = completion
        sources[arm] = source
    if all(sources.values()):
        try:
            compare_tapes(
                [seed_root / "prefix_A" / "factorial_training_telemetry_v1.csv",
                 seed_root / "prefix_B" / "factorial_training_telemetry_v1.csv"],
                ["prefix_A", "prefix_B"],
                seed_root / "prefix_matched_randomness_receipt.json",
                protocol["protocol_sha256_runtime"], first=1, last=4000,
            )
        except Exception as exc:
            atomic_json(seed_root / "prefix_randomness_failure.json", {
                "schema": "ect.q256.terminal-history-randomness-failure/v1",
                "status": "FAIL", "error": repr(exc),
                "protocol_sha256": protocol["protocol_sha256_runtime"],
            })
            sources = {"A": None, "B": None}
    for cell in suffix_order:
        statuses[cell] = attempt_suffix(
            protocol, protocol_path, seed_root, seed, cell,
            sources[CELLS[cell][0]], gpu,
        )
    if all(statuses[cell].get("status") == "PASS" for cell in CELLS):
        try:
            compare_tapes(
                [seed_root / "AA" / "schedule_switch_training_telemetry_v1.csv",
                 seed_root / "BA" / "schedule_switch_training_telemetry_v1.csv"],
                ["AA", "BA"], seed_root / "suffix_matched_randomness_receipt.json",
                protocol["protocol_sha256_runtime"], first=4001, last=8000,
            )
        except Exception as exc:
            atomic_json(seed_root / "suffix_randomness_failure.json", {
                "schema": "ect.q256.terminal-history-randomness-failure/v1",
                "status": "FAIL", "error": repr(exc),
                "protocol_sha256": protocol["protocol_sha256_runtime"],
            })
            statuses["randomness"] = {"status": "FAIL", "error": repr(exc)}
    success = all(statuses[cell].get("status") == "PASS" for cell in CELLS) \
        and statuses.get("randomness", {"status": "PASS"})["status"] == "PASS"
    receipt = {
        "schema": "ect.q256.terminal-history-seed-completion/v1",
        "status": "PASS" if success else "COMPLETE_WITH_FAILURE",
        "seed": seed,
        "gpu_index": gpu,
        "prefix_order": list(prefix_order),
        "suffix_order": list(suffix_order),
        "trajectory_status": {cell: statuses[cell].get("status") for cell in CELLS},
        "prefix_status": {arm: statuses[f"prefix_{arm}"].get("status") for arm in ARM_FACTORS},
        "automatic_retry_count": 0,
        "protocol_sha256": protocol["protocol_sha256_runtime"],
        "completed_at": utc_now(),
    }
    atomic_json(seed_root / "seed_completion_receipt.json", receipt)
    return receipt


def worker(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = validate_protocol(protocol_path)
    protocol["protocol_sha256_runtime"] = sha256_file(protocol_path)
    node_seeds = NODE_SEEDS[args.node_id]
    gpu_count = NODE_GPU_COUNTS[args.node_id]
    expected = tuple(node_seeds[args.gpu_index::gpu_count])
    if tuple(args.seeds) != expected:
        raise RuntimeError(f"GPU {args.gpu_index} expected seeds {expected}, got {args.seeds}")
    receipts = []
    for seed in expected:
        try:
            receipts.append(run_seed(protocol, protocol_path, seed, args.gpu_index))
        except Exception as exc:
            seed_root = Path(protocol["paths"]["output_root"]) / "training" / f"seed{seed}"
            atomic_json(seed_root / "seed_worker_failure.json", {
                "schema": "ect.q256.terminal-history-seed-worker-failure/v1",
                "status": "FAIL",
                "seed": seed,
                "gpu_index": args.gpu_index,
                "error": repr(exc),
                "automatic_retry_count": 0,
                "protocol_sha256": protocol["protocol_sha256_runtime"],
                "failed_at": utc_now(),
            })
            receipts.append({"seed": seed, "status": "FAIL", "error": repr(exc)})
    atomic_json(Path(protocol["paths"]["output_root"]) / "logs" / f"gpu{args.gpu_index}_worker_receipt.json", {
        "schema": "ect.q256.terminal-history-gpu-worker/v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in receipts) else "COMPLETE_WITH_FAILURES",
        "node_id": args.node_id,
        "gpu_index": args.gpu_index,
        "seeds": list(expected),
        "seed_receipts": receipts,
        "automatic_retry_count": 0,
        "protocol_sha256": protocol["protocol_sha256_runtime"],
        "completed_at": utc_now(),
    })


def launch(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = validate_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    preflight = load_json(args.preflight.resolve(strict=True))
    if (preflight.get("status") != "PASS" or preflight.get("node_id") != args.node_id
            or preflight.get("protocol_sha256") != protocol_sha):
        raise RuntimeError("preflight receipt mismatch")
    output_root = Path(protocol["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "training").mkdir()
    logs = output_root / "logs"
    logs.mkdir()
    control = output_root / "control"
    control.mkdir()
    copy_exclusive(protocol_path, control / "protocol.json")
    copy_exclusive(protocol_path.with_name("protocol.sha256"), control / "protocol.sha256")
    copy_exclusive(args.preflight, control / "preflight.json")
    python = protocol["runtime"]["python"]
    workers = []
    registry = []
    node_seeds = NODE_SEEDS[args.node_id]
    gpu_count = NODE_GPU_COUNTS[args.node_id]
    for gpu in range(gpu_count):
        seeds = tuple(node_seeds[gpu::gpu_count])
        command = [python, str(Path(__file__).resolve()), "worker",
                   "--protocol", str(protocol_path), "--node-id", args.node_id,
                   "--gpu-index", str(gpu), "--seeds", *map(str, seeds)]
        log_path = logs / f"gpu{gpu}.log"
        log = log_path.open("xb")
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        workers.append((gpu, seeds, process, log, log_path))
        registry.append({"gpu_index": gpu, "seeds": list(seeds), "pid": process.pid,
                         "log_path": str(log_path)})
    atomic_json(logs / "worker_registry.json", {
        "schema": "ect.q256.terminal-history-worker-registry/v1",
        "status": "STARTED", "node_id": args.node_id,
        "workers": registry, "started_at": utc_now(),
        "protocol_sha256": protocol_sha,
    })
    infrastructure_failures = []
    for gpu, seeds, process, log, log_path in workers:
        returncode = process.wait()
        log.close()
        if returncode:
            infrastructure_failures.append({"gpu_index": gpu, "seeds": list(seeds),
                                            "exit_code": returncode, "log_path": str(log_path)})
    seed_receipts = sorted(output_root.glob("training/seed*/seed_completion_receipt.json"))
    scientific_failures = []
    for receipt_path in seed_receipts:
        receipt = load_json(receipt_path)
        if receipt.get("status") != "PASS":
            scientific_failures.append({"seed": receipt.get("seed"),
                                        "status": receipt.get("status"),
                                        "path": str(receipt_path)})
    status = "PASS"
    if infrastructure_failures:
        status = "INFRASTRUCTURE_FAILURE"
    elif scientific_failures:
        status = "COMPLETE_WITH_SCIENTIFIC_FAILURES"
    atomic_json(output_root / "node_completion_receipt.json", {
        "schema": "ect.q256.terminal-history-node-completion/v1",
        "status": status,
        "node_id": args.node_id,
        "expected_seeds": list(node_seeds),
        "seed_receipt_count": len(seed_receipts),
        "infrastructure_failures": infrastructure_failures,
        "scientific_failures": scientific_failures,
        "automatic_retry_count": 0,
        "protocol_sha256": protocol_sha,
        "completed_at": utc_now(),
    })
    if infrastructure_failures:
        raise RuntimeError(f"worker infrastructure failures: {infrastructure_failures}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--protocol", type=Path, required=True)
    preflight.add_argument("--node-id", choices=tuple(NODE_SEEDS), required=True)
    preflight.add_argument("--receipt", type=Path, required=True)
    preflight.set_defaults(func=lambda args: node_preflight(args.protocol, args.node_id, args.receipt))
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--protocol", type=Path, required=True)
    launch_parser.add_argument("--node-id", choices=tuple(NODE_SEEDS), required=True)
    launch_parser.add_argument("--preflight", type=Path, required=True)
    launch_parser.set_defaults(func=launch)
    work = sub.add_parser("worker")
    work.add_argument("--protocol", type=Path, required=True)
    work.add_argument("--node-id", choices=tuple(NODE_SEEDS), required=True)
    work.add_argument("--gpu-index", type=int, required=True)
    work.add_argument("--seeds", type=int, nargs="+", required=True)
    work.set_defaults(func=worker)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
