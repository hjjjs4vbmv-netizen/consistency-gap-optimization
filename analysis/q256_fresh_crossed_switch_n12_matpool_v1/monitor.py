#!/usr/bin/env python3
"""Advisory-only monitor for one formal single-GPU trajectory.

The monitor never kills training.  It emits append-only JSONL observations and
an immutable final receipt.  The outer launcher owns the hard timeout policy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

SCHEMA = "ect.q256.fresh-crossed-switch-monitor/v1"
ERROR_MARKERS = (
    "cuda error",
    "cuda out of memory",
    "traceback (most recent call last)",
    "no space left on device",
    "hash mismatch",
    "nonfinite update",
    "non-finite update",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    os.unlink(temporary)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("ab", buffering=0) as handle:
        handle.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(handle.fileno())


def descendants(root_pid: int) -> set[int]:
    try:
        root = psutil.Process(root_pid)
        return {root_pid, *(child.pid for child in root.children(recursive=True))}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set()


def owns_cuda_device(pids: set[int]) -> bool:
    for pid in pids:
        fd_root = Path(f"/proc/{pid}/fd")
        try:
            for descriptor in fd_root.iterdir():
                try:
                    if os.readlink(descriptor).startswith("/dev/nvidia"):
                        return True
                except (FileNotFoundError, PermissionError, OSError):
                    continue
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return False


def classify_gpu_apps(apps: list[dict], gpu_uuid: str, owned: set[int],
                      *, alive: bool, owned_cuda_context: bool) -> tuple[list[dict], list[dict]]:
    selected = [row for row in apps if row["gpu_uuid"] == gpu_uuid]
    direct_owned = [row for row in selected if row["pid"] in owned]
    unmatched = [row for row in selected if row["pid"] not in owned]
    namespace_owned = []
    if (alive and owned_cuda_context and len(unmatched) == 1
            and unmatched[0].get("process_name") == "[Not Found]"):
        namespace_owned = unmatched
        unmatched = []
    return direct_owned + namespace_owned, unmatched


def gpu_rows() -> list[dict]:
    query = (
        "index,uuid,name,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,power.draw"
    )
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 8:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": float(fields[3]),
                "memory_used_mib": float(fields[4]),
                "utilization_percent": float(fields[5]),
                "temperature_c": float(fields[6]),
                "power_w": None if fields[7] in {"N/A", "[N/A]"} else float(fields[7]),
            }
        )
    return rows


def compute_apps() -> list[dict]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        if "No running processes" in exc.output:
            return []
        raise
    rows = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "gpu_uuid": fields[1],
                "process_name": fields[2],
                "used_gpu_memory_mib": float(fields[3]),
            }
        )
    return rows


def progress(run_dir: Path, total_attempts: int, start_monotonic: float) -> dict:
    path = run_dir / "train_summary.csv"
    result = {
        "attempted_iteration": None,
        "successful_optimizer_steps": None,
        "processed_nimg": None,
        "eta_seconds": None,
    }
    if total_attempts <= 0 or not path.is_file() or path.stat().st_size == 0:
        return result
    try:
        with path.open("rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return result
        last = rows[-1]
        attempted = int(float(last["attempted_iteration"]))
        result.update(
            attempted_iteration=attempted,
            successful_optimizer_steps=int(float(last["successful_optimizer_steps"])),
            processed_nimg=int(float(last["processed_nimg"])),
        )
        elapsed = max(time.monotonic() - start_monotonic, 1.0)
        completed = max(attempted - int(float(rows[0]["attempted_iteration"])) + 1, 1)
        remaining = max(total_attempts - attempted, 0)
        result["eta_seconds"] = remaining * elapsed / completed
    except (OSError, KeyError, ValueError):
        result["parse_warning"] = True
    return result


def scan_log(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        block = handle.read()
    lowered = block.decode("utf-8", errors="replace").lower()
    markers = [marker for marker in ERROR_MARKERS if marker in lowered]
    return size, markers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--total-attempts", type=int, required=True)
    parser.add_argument("--log-name", default="log.txt")
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--stall-seconds", type=int, default=300)
    parser.add_argument("--min-free-bytes", type=int, default=100 * 1024**3)
    args = parser.parse_args()
    if not 10 <= args.interval_seconds <= 300:
        raise SystemExit("monitor interval must be in [10, 300]")
    run_dir = args.run_dir.resolve(strict=True)
    observations = run_dir / "monitor.jsonl"
    final_path = run_dir / "monitor_completion_receipt.json"
    start_wall = utc_now()
    start_monotonic = time.monotonic()
    last_log_size = -1
    last_log_growth = start_monotonic
    log_offset = 0
    alert_counts: dict[str, int] = {}
    samples = 0
    while True:
        owned = descendants(args.pid)
        alive = bool(owned)
        gpus = gpu_rows()
        selected = next((row for row in gpus if row["index"] == args.gpu_index), None)
        apps = compute_apps()
        owned_apps, foreign = classify_gpu_apps(
            apps, args.gpu_uuid, owned, alive=alive,
            owned_cuda_context=owns_cuda_device(owned))
        log_path = run_dir / args.log_name
        log_size = log_path.stat().st_size if log_path.is_file() else 0
        if log_size != last_log_size:
            last_log_growth = time.monotonic()
            last_log_size = log_size
        stall_seconds = time.monotonic() - last_log_growth
        log_offset, markers = scan_log(log_path, log_offset)
        usage = shutil.disk_usage(run_dir)
        alerts = []
        if selected is None or selected["uuid"] != args.gpu_uuid:
            alerts.append("GPU_IDENTITY_MISMATCH")
        if foreign:
            alerts.append("FOREIGN_GPU_PROCESS")
        if alive and stall_seconds >= args.stall_seconds:
            alerts.append("OUTPUT_STALL_SUSPECTED")
        if usage.free < args.min_free_bytes:
            alerts.append("DISK_LOW")
        if markers:
            alerts.append("LOG_ERROR_MARKER")
        for alert in alerts:
            alert_counts[alert] = alert_counts.get(alert, 0) + 1
        row = {
            "schema": SCHEMA,
            "observed_at": utc_now(),
            "root_pid": args.pid,
            "owned_pids": sorted(owned),
            "process_alive": alive,
            "gpu": selected,
            "compute_apps": apps,
            "owned_compute_apps": owned_apps,
            "foreign_compute_apps": foreign,
            "log_size_bytes": log_size,
            "log_stall_seconds": round(stall_seconds, 3),
            "disk_free_bytes": usage.free,
            "disk_total_bytes": usage.total,
            "progress": progress(run_dir, args.total_attempts, start_monotonic),
            "log_error_markers": markers,
            "alerts": alerts,
        }
        append_jsonl(observations, row)
        samples += 1
        if not alive:
            break
        time.sleep(args.interval_seconds)
    final = {
        "schema": "ect.q256.fresh-crossed-switch-monitor-completion/v1",
        "status": "PASS" if not alert_counts else "ADVISORY_ALERTS",
        "started_at": start_wall,
        "ended_at": utc_now(),
        "root_pid": args.pid,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "sample_count": samples,
        "alert_counts": alert_counts,
        "automatic_kill_performed": False,
        "observations_path": str(observations),
    }
    atomic_json(final_path, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
