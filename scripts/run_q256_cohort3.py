#!/usr/bin/env python3
"""Fail-closed execution layer for prospective q256 Cohort III training.

This program is training-only.  It never imports or invokes the FID/KID
evaluation path.  Run it inside the exact NGC PyTorch 24.01 container with all
five GPUs visible and a private /dev/shm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import verify_q256_target_weight_arm as production  # noqa: E402
from training import reproducibility  # noqa: E402


SCHEMA_PREFIX = "ect.q256.target-weight-factorial-cohort3"
BASE_COMMIT = "64e56392883248668a92aa6c18c0cec3d1ef796f"
FORMAL_SEMANTICS_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EXPECTED_BRANCH = "experiment/q256-cohort3-seed8-12"
EXPECTED_IMAGE = "nvcr.io/nvidia/pytorch:24.01-py3"
EXPECTED_PYTHON = "3.10.12"
EXPECTED_TORCH = "2.2.0a0+81ea7a4"
EXPECTED_TORCH_CUDA = "12.3"
EXPECTED_DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
EXPECTED_TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
DATASET = Path(
    "/data/raw/ECT/datasets/"
    "cifar10-32x32-canonical-08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372.zip"
)
TRANSFER = Path("/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl")
DEFAULT_RUN_ROOT = Path(
    "/data/raw/ECT/ect_runs/q256-target-weight-factorial-cohort3-seed8-12"
)
PREREGISTRATION = (
    REPO_ROOT
    / "analysis"
    / "q256_target_weight_factorial"
    / "cohort3_seed8_12"
    / "preregistration.json"
)
PREREGISTRATION_DIGESTS = PREREGISTRATION.with_suffix(".sha256")
SMOKE_SEED = 314159
SEED_GPU = OrderedDict(((8, 0), (9, 1), (10, 2), (11, 3), (12, 4)))
ARMS = OrderedDict(
    (
        ("A", ("1.0", "1.0")),
        ("B", ("1.1", "1.1")),
        ("C", ("1.1", "1.0")),
        ("D", ("1.0", "1.1")),
    )
)
CORE_TRAINING_FILES = (
    "ct_train.py",
    "training/loss.py",
    "training/schedules.py",
    "training/ct_training_loop.py",
    "training/reproducibility.py",
    "torch_utils/misc.py",
)
REQUIRED_RUN_ARTIFACTS = (
    "training_options.json",
    "initial_state_receipt_v1.json",
    "factorial_training_telemetry_v1.csv",
    "train_summary.csv",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "log.txt",
    "final.png",
)
FATAL_PATTERNS = (
    "traceback (most recent call last)",
    "cuda error",
    "cuda out of memory",
    "outofmemoryerror",
    "bus error",
    "dataloader worker",
    "nonpositive denominator",
    "non-finite loss",
    "non-finite update",
    "non-finite model",
    "non-finite ema",
)
WALL_CLOCK_FIELDS = {"elapsed_sec", "gpu_hours_cumulative", "peak_vram_gb"}
TARGET_FIELDS = (
    "target_r_sha256",
    "target_delta_sha256",
    "target_r_zero_count",
    "target_r_equal_t_count",
    "target_scaled_to_zero_count",
    "target_delta_min",
    "target_delta_max",
    "target_delta_mean",
)
DENOMINATOR_FIELDS = (
    "denominator_r_sha256",
    "denominator_delta_sha256",
    "denominator_r_zero_count",
    "denominator_r_equal_t_count",
    "denominator_scaled_to_zero_count",
    "denominator_delta_min",
    "denominator_delta_max",
    "denominator_delta_mean",
)


class CohortError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CohortError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checked_output(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", "") or str(exc)
        fail(f"command failed ({shlex.join(command)}): {detail.strip()}")


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected one JSON object: {path}")
    return value


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"{label} is missing, empty, or a symlink: {path}")
    return path.resolve(strict=True)


def validate_run_root(raw: Path) -> Path:
    if not raw.is_absolute():
        fail(f"run root must be absolute: {raw}")
    resolved = raw.resolve(strict=False)
    allowed = Path("/data/raw/ECT/ect_runs").resolve(strict=False)
    try:
        resolved.relative_to(allowed)
    except ValueError:
        fail(f"run root must stay under {allowed}: {resolved}")
    if resolved == allowed:
        fail("run root must not be the broad ect_runs directory")
    return resolved


def git_record() -> dict[str, Any]:
    branch = checked_output(["git", "symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch != EXPECTED_BRANCH:
        fail(f"wrong branch: {branch!r} != {EXPECTED_BRANCH!r}")
    status = checked_output(["git", "status", "--porcelain", "--untracked-files=all"])
    if status:
        fail(f"source worktree is dirty: {'; '.join(status.splitlines()[:12])}")
    head = checked_output(["git", "rev-parse", "HEAD"])
    tree = checked_output(["git", "rev-parse", "HEAD^{tree}"])
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            FORMAL_SEMANTICS_COMMIT,
            BASE_COMMIT,
            "--",
            *CORE_TRAINING_FILES,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        fail(f"formal-source semantic diff is nonempty: {diff.stdout.strip()}")
    entries = []
    for relative in (
        *CORE_TRAINING_FILES,
        "scripts/run_q256_cohort3.py",
        "scripts/launch_q256_cohort3_tmux.sh",
    ):
        path = require_regular(REPO_ROOT / relative, f"source file {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "branch": branch,
        "head": head,
        "tree": tree,
        "clean": True,
        "base_commit": BASE_COMMIT,
        "formal_semantics_commit": FORMAL_SEMANTICS_COMMIT,
        "formal_semantics_diff_empty": True,
        "files": entries,
        "content_sha256": canonical_sha256(entries),
    }


def preregistration_record() -> dict[str, Any]:
    require_regular(PREREGISTRATION, "Cohort III preregistration")
    require_regular(PREREGISTRATION_DIGESTS, "Cohort III preregistration digest file")
    expected = None
    for line in PREREGISTRATION_DIGESTS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "preregistration.json":
            expected = parts[0]
    actual = sha256_file(PREREGISTRATION)
    if expected != actual:
        fail(f"preregistration digest mismatch: sidecar={expected}, actual={actual}")
    payload = load_json(PREREGISTRATION)
    if payload.get("status") != "frozen_before_any_formal_cohort3_training_or_quality_evaluation":
        fail("preregistration does not carry the frozen prospective status")
    if payload.get("cohort", {}).get("seeds") != list(SEED_GPU):
        fail("preregistration seed list differs from the execution layer")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO_ROOT)),
        "sha256": actual,
        "created_utc": payload.get("created_utc"),
    }


def asset_record(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = require_regular(path, label)
    observed = sha256_file(resolved)
    if observed != expected:
        fail(f"{label} SHA256 mismatch: {observed} != {expected}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def gpu_inventory() -> list[dict[str, Any]]:
    raw = checked_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    records = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            fail(f"unexpected nvidia-smi GPU row: {line!r}")
        records.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
            }
        )
    if len(records) != 5 or [record["index"] for record in records] != list(range(5)):
        fail(f"exactly five physical GPUs indexed 0..4 are required: {records}")
    for record in records:
        if "A100" not in record["name"] or record["memory_total_mib"] < 40000:
            fail(f"GPU is not an expected A100: {record}")
    return records


def compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        fail(f"cannot query GPU processes: {result.stderr.strip()}")
    records = []
    for line in result.stdout.splitlines():
        if not line.strip() or "No running" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            records.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "used_memory_mib": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
                }
            )
    return records


def assert_gpu_idle(index: int, inventory: Sequence[Mapping[str, Any]]) -> None:
    uuid = str(inventory[index]["uuid"])
    foreign = [record for record in compute_processes() if record["gpu_uuid"] == uuid]
    if foreign:
        fail(f"physical GPU {index} ({uuid}) is not exclusive/idle: {foreign}")


def memory_record() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        match = re.search(r"([0-9]+)", raw)
        if match:
            values[key] = int(match.group(1)) * 1024
    return {key: values.get(key, 0) for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}


def runtime_record() -> dict[str, Any]:
    image = os.environ.get("ECT_COHORT3_RUNTIME_IMAGE")
    image_id = os.environ.get("ECT_COHORT3_RUNTIME_IMAGE_ID")
    if image != EXPECTED_IMAGE:
        fail(f"runtime image attestation mismatch: {image!r} != {EXPECTED_IMAGE!r}")
    if not image_id or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        fail("ECT_COHORT3_RUNTIME_IMAGE_ID must be the inspected sha256 image ID")
    if platform.python_version() != EXPECTED_PYTHON:
        fail(f"Python mismatch: {platform.python_version()} != {EXPECTED_PYTHON}")
    if torch.__version__ != EXPECTED_TORCH:
        fail(f"PyTorch mismatch: {torch.__version__} != {EXPECTED_TORCH}")
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        fail(f"CUDA runtime mismatch: {torch.version.cuda} != {EXPECTED_TORCH_CUDA}")
    cudnn = torch.backends.cudnn.version()
    if cudnn is None or not str(cudnn).startswith("8"):
        fail(f"cuDNN 8.x required, got {cudnn!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 5:
        fail(f"exactly five CUDA devices must be visible, got {torch.cuda.device_count()}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        fail("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    shm = shutil.disk_usage("/dev/shm")
    if shm.total < 60 * 1024**3:
        fail(f"private /dev/shm is below the 60-GiB safety floor: {shm.total}")
    identity = {
        "image": image,
        "image_id": image_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn,
        "cuda_device_count": torch.cuda.device_count(),
        "deterministic_algorithms_required": True,
        "cudnn_deterministic_required": True,
        "cudnn_benchmark_required": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dev_shm_total_bytes": shm.total,
        "gpus": gpu_inventory(),
    }
    record = {
        **identity,
        "dev_shm_free_bytes": shm.free,
        "host_memory": memory_record(),
        "runtime_digest_scope": identity,
        "runtime_digest": canonical_sha256(identity),
    }
    return record


def require_preflight(run_root: Path, *, require_smoke: bool = False) -> dict[str, Any]:
    receipt = load_json(run_root / "provenance" / "preflight.json")
    if receipt.get("status") != "PASS":
        fail("preflight receipt is not PASS")
    source = git_record()
    prereg = preregistration_record()
    assets = {
        "dataset": asset_record(DATASET, EXPECTED_DATASET_SHA256, "canonical dataset"),
        "transfer": asset_record(TRANSFER, EXPECTED_TRANSFER_SHA256, "transfer checkpoint"),
    }
    runtime = runtime_record()
    exact = {
        "source_head": source["head"],
        "source_content_sha256": source["content_sha256"],
        "preregistration_sha256": prereg["sha256"],
        "dataset_sha256": assets["dataset"]["sha256"],
        "transfer_sha256": assets["transfer"]["sha256"],
        "runtime_digest": runtime["runtime_digest"],
    }
    for field, expected in exact.items():
        if receipt.get("bindings", {}).get(field) != expected:
            fail(f"preflight binding changed for {field}")
    if require_smoke:
        smoke = load_json(run_root / "engineering" / "smoke_gate.json")
        if smoke.get("status") != "PASS" or smoke.get("bindings") != exact:
            fail("cross-GPU/exact-resume smoke gate is absent, failed, or stale")
    return {"receipt": receipt, "source": source, "preregistration": prereg, "assets": assets, "runtime": runtime, "bindings": exact}


def init_preflight(run_root: Path) -> dict[str, Any]:
    run_root = validate_run_root(run_root)
    if run_root.exists():
        fail(f"refusing existing experiment root: {run_root}")
    source = git_record()
    prereg = preregistration_record()
    assets = {
        "dataset": asset_record(DATASET, EXPECTED_DATASET_SHA256, "canonical dataset"),
        "transfer": asset_record(TRANSFER, EXPECTED_TRANSFER_SHA256, "transfer checkpoint"),
    }
    runtime = runtime_record()
    active = compute_processes()
    if active:
        fail(f"foreign GPU processes prevent five-GPU exclusivity: {active}")
    storage = shutil.disk_usage(run_root.parent)
    if storage.free < 50 * 1024**3:
        fail(f"durable storage is below 50 GiB: {storage.free}")
    bindings = {
        "source_head": source["head"],
        "source_content_sha256": source["content_sha256"],
        "preregistration_sha256": prereg["sha256"],
        "dataset_sha256": assets["dataset"]["sha256"],
        "transfer_sha256": assets["transfer"]["sha256"],
        "runtime_digest": runtime["runtime_digest"],
    }
    run_root.mkdir(mode=0o700, parents=True)
    for relative in ("provenance", "engineering", "formal", "monitoring", "handoff"):
        (run_root / relative).mkdir(mode=0o700)
    receipt = {
        "schema": f"{SCHEMA_PREFIX}-preflight/v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "source": source,
        "preregistration": prereg,
        "assets": assets,
        "runtime": runtime,
        "gpu_exclusivity": {"active_compute_processes": active, "pass": True},
        "storage": {"total_bytes": storage.total, "used_bytes": storage.used, "free_bytes": storage.free, "minimum_free_bytes": 50 * 1024**3},
        "bindings": bindings,
        "quality_evaluation_executed": False,
    }
    write_json_exclusive(run_root / "provenance" / "preflight.json", receipt)
    return receipt


def training_command(
    *, run_dir: Path, arm: str, seed: int, mode: str, resume: Path | None = None,
    stop_after_attempts: int | None = None,
) -> list[str]:
    if arm not in ARMS or mode not in {"smoke", "formal"}:
        fail(f"invalid arm/mode: {arm}/{mode}")
    target, denominator = ARMS[arm]
    command = [
        sys.executable,
        str(REPO_ROOT / "ct_train.py"),
        f"--data={DATASET}",
        f"--outdir={run_dir}",
        "--nosubdir",
        "--cond=False",
        "--arch=ddpmpp",
        "--precond=ect",
        "--batch=128",
        "--batch-gpu=16",
        "--optim=RAdam",
        "--lr=0.0001",
        "--dropout=0.2",
        "--augment=0",
        "--xflip=False",
        "--mean=-1.1",
        "--std=2.0",
        "--mapping=sigmoid",
        "--global-gap-scale=1.0",
        "--factorial-protocol=q256_target_weight_v1",
        f"--target-gap-scale={target}",
        f"--denominator-gap-scale={denominator}",
        "-q", "256", "-k", "8", "-b", "1", "-c", "0",
        "--double=10000",
        "--ema_beta=0.9993",
        f"--seed={seed}",
        "--fp16=True",
        "--tf32=False",
        "--ls=1.0",
        "--enable_amp=True",
        "--bench=False",
        "--cache=True",
        "--workers=1",
        "--metrics=none",
        f"--duration={'0.004096' if mode == 'smoke' else '0.256'}",
        "--tick=10",
        "--snap=0",
        "--dump=0",
        "--ckpt=10",
        "--sample_every=26",
        "--eval_every=50",
        "--mid_t=0.821",
        "--adaptive-update-kimg=0.5",
    ]
    command.append(f"--transfer={TRANSFER}" if resume is None else f"--resume={resume}")
    if stop_after_attempts is not None:
        if mode != "smoke" or resume is not None or stop_after_attempts != 16:
            fail("planned pause is fresh-smoke-only and frozen at 16 attempts")
        command.append("--stop-after-attempts=16")
    return command


def process_environment(gpu_index: int, port: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def canonical_csv_digest(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [
        {key: value for key, value in row.items() if key not in WALL_CLOCK_FIELDS}
        for row in rows
    ]
    return canonical_sha256(normalized)


def fatal_matches(paths: Iterable[Path]) -> list[str]:
    matches = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for pattern in FATAL_PATTERNS:
            if pattern in text:
                matches.append(f"{path.name}:{pattern}")
    return sorted(set(matches))


def verify_computational_run(
    run_dir: Path, *, arm: str, seed: int, mode: str,
    fresh_options_path: Path | None = None,
) -> dict[str, Any]:
    paths = {name: require_regular(run_dir / name, name) for name in REQUIRED_RUN_ARTIFACTS}
    if list(run_dir.glob("network-snapshot-[0-9]*.pkl")) or list(run_dir.glob("training-state-[0-9]*.pt")):
        fail(f"numbered artifacts violate the frozen q256 cadence: {run_dir}")
    matches = fatal_matches((paths["log.txt"], run_dir / "runner.log", run_dir / "runner-resume.log"))
    if matches:
        fail(f"fatal log pattern(s) in {run_dir}: {matches}")
    options_source = fresh_options_path or paths["training_options.json"]
    options = load_json(options_source)
    options_info = production.validate_training_options(options, arm, seed, mode)
    initial = load_json(paths["initial_state_receipt_v1.json"])
    initial_info = production.validate_initial_receipt(initial, arm, seed, options_info)
    rows = production.read_telemetry(paths["factorial_training_telemetry_v1.csv"])
    telemetry = production.validate_telemetry(rows, arm, mode, None)
    state_info = production.validate_training_state(
        paths["training-state-latest.pt"], arm, seed, mode, telemetry
    )
    if state_info["trajectory_config_sha256"] != initial_info["trajectory_config_sha256"]:
        fail("initial/final trajectory config mismatch")
    snapshot = production.validate_snapshot(
        paths["network-snapshot-latest.pkl"], arm, options_info, state_info
    )
    with paths["train_summary.csv"].open("r", newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    expected_attempts = 32 if mode == "smoke" else 2000
    expected_nimg = 4096 if mode == "smoke" else 256000
    if len(summary) != expected_attempts:
        fail(f"train_summary rows mismatch: {len(summary)} != {expected_attempts}")
    if int(summary[-1]["attempted_iteration"]) != expected_attempts:
        fail("train_summary final attempted iteration mismatch")
    if int(summary[-1]["processed_nimg"]) != expected_nimg:
        fail("train_summary final processed images mismatch")
    state = state_info.pop("state")
    computational = {
        "model_sha256": reproducibility.module_state_sha256(state["net"]),
        "ema_sha256": reproducibility.module_state_sha256(state["ema"]),
        "optimizer_state_sha256": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler_state_sha256": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng_sampler_state_sha256": reproducibility.state_sha256(state["rank_states"]),
        "loss_state_sha256": reproducibility.state_sha256(state["loss_fn_state"]),
        "telemetry_computational_sha256": canonical_csv_digest(paths["factorial_training_telemetry_v1.csv"]),
        "train_summary_computational_sha256": canonical_csv_digest(paths["train_summary.csv"]),
    }
    artifacts = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    report = {
        "schema": f"{SCHEMA_PREFIX}-cell-completion/v1",
        "status": "PASS",
        "finished_utc": utc_now(),
        "run_dir": str(run_dir),
        "mode": mode,
        "seed": seed,
        "arm": arm,
        "target_gap_scale": ARMS[arm][0],
        "denominator_gap_scale": ARMS[arm][1],
        "attempted_iterations": telemetry["attempts"],
        "accepted_optimizer_updates": telemetry["successful_optimizer_steps"],
        "amp_skip_attempts": telemetry["amp_skip_attempts"],
        "amp_skip_count": len(telemetry["amp_skip_attempts"]),
        "processed_images": telemetry["processed_nimg"],
        "processed_kimg": telemetry["processed_nimg"] / 1000,
        "initial_common_state_sha256": initial_info["common_initial_state_sha256"],
        "trajectory_config_sha256": initial_info["trajectory_config_sha256"],
        "snapshot_ema_sha256": snapshot["ema_sha256"],
        "computational_state": computational,
        "artifacts": artifacts,
        "fatal_log_patterns": [],
        "quality_metrics_computed": False,
    }
    return report


def launch_fresh(
    *, run_root: Path, run_dir: Path, arm: str, seed: int, mode: str,
    gpu_index: int, port: int, stop_after_attempts: int | None = None,
) -> tuple[int, dict[str, Any]]:
    context = require_preflight(run_root)
    inventory = context["runtime"]["gpus"]
    assert_gpu_idle(gpu_index, inventory)
    if run_dir.exists():
        fail(f"refusing existing run directory: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o700)
    command = training_command(
        run_dir=run_dir, arm=arm, seed=seed, mode=mode,
        stop_after_attempts=stop_after_attempts,
    )
    manifest = {
        "schema": f"{SCHEMA_PREFIX}-launch/v1",
        "status": "authorized_to_start",
        "launch_kind": "fresh_transfer",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "source": context["source"],
        "preregistration": context["preregistration"],
        "assets": context["assets"],
        "runtime": context["runtime"],
        "runtime_digest": context["runtime"]["runtime_digest"],
        "gpu": inventory[gpu_index],
        "seed": seed,
        "arm": arm,
        "target_gap_scale": ARMS[arm][0],
        "denominator_gap_scale": ARMS[arm][1],
        "mode": mode,
        "attempt_budget": 32 if mode == "smoke" else 2000,
        "processed_image_budget": 4096 if mode == "smoke" else 256000,
        "stop_after_attempts": stop_after_attempts,
        "exact_command_argv": command,
        "exact_command_shell": shlex.join(command),
        "process_environment": {
            key: process_environment(gpu_index, port)[key]
            for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
        },
        "quality_metrics_computed": False,
    }
    write_json_exclusive(run_dir / "cohort3_launch_manifest.json", manifest)
    log_path = run_dir / "runner.log"
    start = time.monotonic()
    with log_path.open("xb") as output:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=process_environment(gpu_index, port),
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        pid = process.pid
        returncode = process.wait()
    exit_record = {
        "schema": f"{SCHEMA_PREFIX}-runner-exit/v1",
        "started_utc": manifest["created_utc"],
        "finished_utc": utc_now(),
        "pid": pid,
        "exit_code": returncode,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json_exclusive(run_dir / "runner_exit.json", exit_record)
    return returncode, manifest


def run_one_fresh(
    *, run_root: Path, run_dir: Path, arm: str, seed: int, mode: str,
    gpu_index: int, port: int,
) -> dict[str, Any]:
    returncode, _manifest = launch_fresh(
        run_root=run_root, run_dir=run_dir, arm=arm, seed=seed, mode=mode,
        gpu_index=gpu_index, port=port,
    )
    if returncode != 0:
        fail(f"training exited nonzero ({returncode}): {run_dir}")
    report = verify_computational_run(run_dir, arm=arm, seed=seed, mode=mode)
    write_json_exclusive(run_dir / "cell_completion.json", report)
    return report


def resume_planned_smoke(
    *, run_root: Path, run_dir: Path, gpu_index: int, port: int,
) -> dict[str, Any]:
    returncode, original_manifest = launch_fresh(
        run_root=run_root, run_dir=run_dir, arm="A", seed=SMOKE_SEED,
        mode="smoke", gpu_index=gpu_index, port=port, stop_after_attempts=16,
    )
    if returncode != 0:
        fail(f"planned 16-attempt pause exited nonzero: {returncode}")
    state_path = require_regular(run_dir / "training-state-latest.pt", "planned-pause state")
    pause_state = production.torch_load_trusted(state_path)
    if pause_state.get("attempted_iteration") != 16:
        fail("planned-pause state is not at attempt 16")
    fresh_options = run_dir / "training_options.fresh.json"
    shutil.copyfile(run_dir / "training_options.json", fresh_options)
    pause_receipt = {
        "schema": f"{SCHEMA_PREFIX}-planned-pause/v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "attempted_iteration": 16,
        "state_sha256": sha256_file(state_path),
        "fresh_options_sha256": sha256_file(fresh_options),
        "original_launch_manifest_sha256": sha256_file(run_dir / "cohort3_launch_manifest.json"),
    }
    write_json_exclusive(run_dir / "planned_pause_receipt.json", pause_receipt)
    command = training_command(
        run_dir=run_dir, arm="A", seed=SMOKE_SEED, mode="smoke", resume=state_path
    )
    context = require_preflight(run_root)
    inventory = context["runtime"]["gpus"]
    assert_gpu_idle(gpu_index, inventory)
    resume_manifest = {
        "schema": f"{SCHEMA_PREFIX}-resume-launch/v1",
        "status": "authorized_to_resume_planned_gate",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "gpu": inventory[gpu_index],
        "runtime_digest": context["runtime"]["runtime_digest"],
        "source_head": context["source"]["head"],
        "preregistration_sha256": context["preregistration"]["sha256"],
        "resume_state": str(state_path),
        "resume_state_sha256": sha256_file(state_path),
        "original_launch": original_manifest,
        "exact_command_argv": command,
        "exact_command_shell": shlex.join(command),
    }
    write_json_exclusive(run_dir / "cohort3_resume_manifest.json", resume_manifest)
    with (run_dir / "runner-resume.log").open("xb") as output:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=process_environment(gpu_index, port),
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        returncode = process.wait()
    write_json_exclusive(
        run_dir / "runner_resume_exit.json",
        {"schema": f"{SCHEMA_PREFIX}-runner-resume-exit/v1", "finished_utc": utc_now(), "pid": process.pid, "exit_code": returncode},
    )
    if returncode != 0:
        fail(f"planned exact resume exited nonzero: {returncode}")
    report = verify_computational_run(
        run_dir, arm="A", seed=SMOKE_SEED, mode="smoke",
        fresh_options_path=fresh_options,
    )
    report["planned_exact_resume"] = True
    write_json_exclusive(run_dir / "cell_completion.json", report)
    return report


def smoke_monitor(stop: threading.Event, output: Path) -> None:
    rows = []
    while not stop.wait(1.0):
        shm = shutil.disk_usage("/dev/shm")
        rows.append(
            {
                "timestamp_utc": utc_now(),
                "dev_shm_used_bytes": shm.used,
                "dev_shm_free_bytes": shm.free,
                "host_memory": memory_record(),
                "gpu_compute_processes": compute_processes(),
            }
        )
    write_json_exclusive(output, {"schema": f"{SCHEMA_PREFIX}-smoke-resource-monitor/v1", "samples": rows})


def run_smoke_gate(run_root: Path) -> dict[str, Any]:
    run_root = validate_run_root(run_root)
    context = require_preflight(run_root)
    gate_path = run_root / "engineering" / "smoke_gate.json"
    if gate_path.exists():
        fail(f"smoke gate already exists: {gate_path}")
    for index in range(5):
        assert_gpu_idle(index, context["runtime"]["gpus"])
    smoke_root = run_root / "engineering" / "cross_gpu"
    if smoke_root.exists():
        fail(f"refusing existing cross-GPU smoke root: {smoke_root}")
    smoke_root.mkdir(mode=0o700)
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=smoke_monitor,
        args=(monitor_stop, run_root / "engineering" / "smoke_resource_monitor.json"),
        daemon=True,
    )
    monitor.start()
    reports: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(
                    run_one_fresh,
                    run_root=run_root,
                    run_dir=smoke_root / f"gpu{index}",
                    arm="A",
                    seed=SMOKE_SEED,
                    mode="smoke",
                    gpu_index=index,
                    port=29910 + index,
                ): index
                for index in range(5)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    reports[index] = future.result()
                except BaseException as exc:
                    errors[index] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor_stop.set()
        monitor.join(timeout=10)
    if errors or len(reports) != 5:
        fail(f"five-GPU concurrent smoke failed: {errors}")
    comparison_keys = (
        "model_sha256",
        "ema_sha256",
        "optimizer_state_sha256",
        "gradscaler_state_sha256",
        "rank_rng_sampler_state_sha256",
        "loss_state_sha256",
        "telemetry_computational_sha256",
        "train_summary_computational_sha256",
    )
    reference = reports[0]
    for index, report in reports.items():
        if report["initial_common_state_sha256"] != reference["initial_common_state_sha256"]:
            fail(f"cross-GPU initial state diverged on GPU {index}")
        for key in comparison_keys:
            if report["computational_state"][key] != reference["computational_state"][key]:
                fail(f"cross-GPU deterministic parity failed on GPU {index}: {key}")
    resumed_dir = run_root / "engineering" / "exact_resume" / "gpu0"
    resumed = resume_planned_smoke(
        run_root=run_root, run_dir=resumed_dir, gpu_index=0, port=29920
    )
    for key in comparison_keys:
        if resumed["computational_state"][key] != reference["computational_state"][key]:
            fail(f"uninterrupted versus 16+resume mismatch: {key}")
    gate = {
        "schema": f"{SCHEMA_PREFIX}-smoke-gate/v1",
        "status": "PASS",
        "finished_utc": utc_now(),
        "smoke_seed": SMOKE_SEED,
        "arm": "A",
        "attempts": 32,
        "concurrent_gpu_count": 5,
        "cross_gpu_byte_identical_computational_state": True,
        "uninterrupted_vs_16_plus_resume_exact": True,
        "comparison_keys": list(comparison_keys),
        "gpu_reports": {str(index): report for index, report in sorted(reports.items())},
        "resume_report": resumed,
        "bindings": context["bindings"],
        "quality_metrics_computed": False,
    }
    write_json_exclusive(gate_path, gate)
    return gate


def compare_rows(
    left: list[dict[str, str]], right: list[dict[str, str]], fields: Iterable[str], label: str
) -> None:
    if len(left) != len(right):
        fail(f"{label} row count differs")
    for attempt, (a, b) in enumerate(zip(left, right, strict=True), start=1):
        for field in fields:
            if a[field] != b[field]:
                fail(f"{label} differs at attempt {attempt}, field={field}")


def verify_seed(run_root: Path, seed: int) -> dict[str, Any]:
    seed_dir = run_root / "formal" / f"seed{seed}"
    reports = {arm: load_json(seed_dir / f"arm{arm}" / "cell_completion.json") for arm in ARMS}
    initials = {report["initial_common_state_sha256"] for report in reports.values()}
    accepted = {report["accepted_optimizer_updates"] for report in reports.values()}
    skip_counts = {report["amp_skip_count"] for report in reports.values()}
    if len(initials) != 1:
        fail(f"seed{seed} arm initial states differ")
    if len(accepted) != 1:
        fail(f"seed{seed} accepted optimizer-update counts differ")
    if len(skip_counts) != 1:
        fail(f"seed{seed} AMP skip counts differ")
    rows: dict[str, list[dict[str, str]]] = {}
    for arm in ARMS:
        with (seed_dir / f"arm{arm}" / "factorial_training_telemetry_v1.csv").open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            rows[arm] = list(csv.DictReader(handle))
    compare_rows(rows["A"], rows["D"], TARGET_FIELDS, "A/D shared target")
    compare_rows(rows["B"], rows["C"], TARGET_FIELDS, "B/C shared target")
    compare_rows(rows["A"], rows["C"], DENOMINATOR_FIELDS, "A/C shared denominator")
    compare_rows(rows["B"], rows["D"], DENOMINATOR_FIELDS, "B/D shared denominator")
    receipt = {
        "schema": f"{SCHEMA_PREFIX}-seed-completion/v1",
        "status": "PASS",
        "finished_utc": utc_now(),
        "seed": seed,
        "gpu_index": SEED_GPU[seed],
        "arm_order": list(ARMS),
        "four_arm_complete": True,
        "common_initial_state": True,
        "common_accepted_update_count": next(iter(accepted)),
        "common_amp_skip_count": next(iter(skip_counts)),
        "telemetry_identities": {
            "A_D_shared_target": True,
            "B_C_shared_target": True,
            "A_C_shared_denominator": True,
            "B_D_shared_denominator": True,
        },
        "cells": reports,
        "quality_metrics_computed": False,
    }
    write_json_exclusive(seed_dir / "seed_completion.json", receipt)
    return receipt


def run_formal_queue(run_root: Path, seed: int, gpu_index: int) -> dict[str, Any]:
    run_root = validate_run_root(run_root)
    context = require_preflight(run_root, require_smoke=True)
    if SEED_GPU.get(seed) != gpu_index:
        fail(f"frozen mapping requires seed {seed} on GPU {SEED_GPU.get(seed)}, not {gpu_index}")
    seed_dir = run_root / "formal" / f"seed{seed}"
    if seed_dir.exists():
        fail(f"refusing existing formal seed directory: {seed_dir}")
    seed_dir.mkdir(mode=0o700)
    queue_manifest = {
        "schema": f"{SCHEMA_PREFIX}-seed-queue/v1",
        "status": "authorized_to_start",
        "created_utc": utc_now(),
        "seed": seed,
        "gpu": context["runtime"]["gpus"][gpu_index],
        "arm_order": list(ARMS),
        "bindings": context["bindings"],
        "quality_metrics_computed": False,
    }
    write_json_exclusive(seed_dir / "queue_manifest.json", queue_manifest)
    try:
        for ordinal, arm in enumerate(ARMS):
            run_one_fresh(
                run_root=run_root,
                run_dir=seed_dir / f"arm{arm}",
                arm=arm,
                seed=seed,
                mode="formal",
                gpu_index=gpu_index,
                port=30000 + gpu_index * 10 + ordinal,
            )
        return verify_seed(run_root, seed)
    except BaseException as exc:
        failure = {
            "schema": f"{SCHEMA_PREFIX}-seed-failure/v1",
            "status": "FAIL",
            "finished_utc": utc_now(),
            "seed": seed,
            "gpu_index": gpu_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "queue_stopped_fail_closed": True,
            "quality_metrics_computed": False,
        }
        write_json_exclusive(seed_dir / "seed_failure.json", failure)
        raise


def last_progress(run_dir: Path) -> dict[str, Any] | None:
    telemetry = run_dir / "factorial_training_telemetry_v1.csv"
    if not telemetry.is_file():
        return None
    with telemetry.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[-1]
    return {
        "attempted_iteration": int(row["attempted_iteration"]),
        "accepted_optimizer_steps": int(row["successful_optimizer_steps"]),
        "processed_kimg": float(row["processed_kimg"]),
        "amp_skip_count": sum(int(item["step_skipped"]) for item in rows),
        "finite_health": all(
            int(row[field]) == 0
            for field in (
                "loss_nonfinite_count", "sanitized_grad_nonfinite_count",
                "update_nonfinite_count", "model_nonfinite_count",
                "ema_nonfinite_count", "factor_nonfinite_count",
                "nonpositive_denominator_count",
            )
        ),
    }


def status_record(run_root: Path) -> dict[str, Any]:
    run_root = validate_run_root(run_root)
    gpus = gpu_inventory()
    processes = compute_processes()
    cells = []
    for seed, gpu_index in SEED_GPU.items():
        for arm in ARMS:
            run_dir = run_root / "formal" / f"seed{seed}" / f"arm{arm}"
            state = "not_started"
            if (run_dir / "cell_completion.json").is_file():
                state = "complete"
            elif run_dir.is_dir():
                state = "active_or_incomplete"
            cells.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpus[gpu_index]["uuid"],
                    "state": state,
                    "progress": last_progress(run_dir),
                    "fatal_log_patterns": fatal_matches((run_dir / "runner.log", run_dir / "log.txt")),
                }
            )
    shm = shutil.disk_usage("/dev/shm")
    disk = shutil.disk_usage(run_root)
    failures = [
        str(path) for path in sorted((run_root / "formal").glob("seed*/seed_failure.json"))
    ]
    return {
        "schema": f"{SCHEMA_PREFIX}-status/v1",
        "timestamp_utc": utc_now(),
        "cells": cells,
        "complete_cells": sum(cell["state"] == "complete" for cell in cells),
        "seed_completion_count": len(list((run_root / "formal").glob("seed*/seed_completion.json"))),
        "failures": failures,
        "gpu_compute_processes": processes,
        "dev_shm": {"total_bytes": shm.total, "used_bytes": shm.used, "free_bytes": shm.free},
        "host_memory": memory_record(),
        "durable_disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "quality_metrics_computed": False,
    }


def monitor(run_root: Path, interval: int) -> int:
    if interval < 120 or interval > 300:
        fail("monitor interval must stay within 120..300 seconds")
    output = run_root / "monitoring" / "formal_monitor.jsonl"
    if output.exists():
        fail(f"refusing an existing monitor log: {output}")
    with output.open("x", encoding="utf-8") as handle:
        while True:
            record = status_record(run_root)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if record["failures"]:
                return 1
            if record["complete_cells"] == 20 and record["seed_completion_count"] == 5:
                return 0
            time.sleep(interval)


def freeze_handoff(run_root: Path) -> dict[str, Any]:
    run_root = validate_run_root(run_root)
    require_preflight(run_root, require_smoke=True)
    rows = []
    for seed, gpu_index in SEED_GPU.items():
        seed_receipt = load_json(run_root / "formal" / f"seed{seed}" / "seed_completion.json")
        if seed_receipt.get("status") != "PASS":
            fail(f"seed{seed} completion is not PASS")
        for arm in ARMS:
            run_dir = run_root / "formal" / f"seed{seed}" / f"arm{arm}"
            cell = load_json(run_dir / "cell_completion.json")
            launch = load_json(run_dir / "cohort3_launch_manifest.json")
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "run_directory": str(run_dir),
                    "checkpoint_filename": "network-snapshot-latest.pkl",
                    "checkpoint_sha256": sha256_file(run_dir / "network-snapshot-latest.pkl"),
                    "training_state_filename": "training-state-latest.pt",
                    "training_state_sha256": sha256_file(run_dir / "training-state-latest.pt"),
                    "source_commit": launch["source"]["head"],
                    "preregistration_sha256": launch["preregistration"]["sha256"],
                    "gpu_index": gpu_index,
                    "gpu_uuid": launch["gpu"]["uuid"],
                    "runtime_digest": launch["runtime_digest"],
                    "attempts": cell["attempted_iterations"],
                    "accepted_steps": cell["accepted_optimizer_updates"],
                    "amp_skips": cell["amp_skip_count"],
                    "deviations_or_incidents": "none",
                }
            )
    if len(rows) != 20:
        fail(f"handoff requires exactly 20 cells, got {len(rows)}")
    handoff_dir = run_root / "handoff"
    json_path = handoff_dir / "checkpoint_handoff.json"
    csv_path = handoff_dir / "checkpoint_handoff.csv"
    md_path = handoff_dir / "checkpoint_handoff.md"
    for path in (json_path, csv_path, md_path):
        if path.exists():
            fail(f"refusing existing handoff artifact: {path}")
    payload = {
        "schema": f"{SCHEMA_PREFIX}-checkpoint-handoff/v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "cell_count": 20,
        "formal_quality_evaluation_executed": False,
        "cells": rows,
    }
    write_json_exclusive(json_path, payload)
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# q256 Cohort III checkpoint handoff",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {payload['created_utc'][:10]}",
        "- Verification Status: VERIFIED",
        "- Version Label: q256_cohort3_checkpoint_handoff_v1",
        "",
        "Training completion: **20/20**. Formal FID/KID evaluation was not run.",
        "",
        "| Seed | Arm | Attempts | Accepted | AMP skips | GPU UUID | Checkpoint SHA256 | State SHA256 |",
        "|---:|:---:|---:|---:|---:|:---|:---|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['attempts']} | {row['accepted_steps']} | "
            f"{row['amp_skips']} | `{row['gpu_uuid']}` | `{row['checkpoint_sha256']}` | "
            f"`{row['training_state_sha256']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke", "status", "handoff"):
        child = sub.add_parser(name)
        child.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    queue = sub.add_parser("queue")
    queue.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    queue.add_argument("--seed", type=int, choices=tuple(SEED_GPU), required=True)
    queue.add_argument("--gpu-index", type=int, choices=tuple(range(5)), required=True)
    watch = sub.add_parser("monitor")
    watch.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    watch.add_argument("--interval", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = make_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = init_preflight(args.run_root)
        elif args.command == "smoke":
            result = run_smoke_gate(args.run_root)
        elif args.command == "queue":
            result = run_formal_queue(args.run_root, args.seed, args.gpu_index)
        elif args.command == "status":
            result = status_record(args.run_root)
        elif args.command == "monitor":
            return monitor(args.run_root, args.interval)
        elif args.command == "handoff":
            result = freeze_handoff(args.run_root)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except CohortError as exc:
        print(f"[run_q256_cohort3] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
