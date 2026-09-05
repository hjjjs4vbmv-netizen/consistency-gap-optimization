#!/usr/bin/env python3
"""Run one M1 seed as four serial, single-GPU branches."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training import m1, schedule_switch


REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERS = (
    ("K_A", "K_B", "R_A", "R_B"),
    ("K_B", "R_A", "R_B", "K_A"),
    ("R_A", "R_B", "K_A", "K_B"),
    ("R_B", "K_A", "K_B", "R_A"),
)
BRANCHES = {"K_A": "A", "K_B": "B", "R_A": "A", "R_B": "B"}
SCIENTIFIC_FAILURE_MARKERS = (
    "FloatingPointError: non-finite",
    "FloatingPointError: target realized",
    "FloatingPointError: denominator realized",
    "FloatingPointError: strict factorial training invariant failure",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def equal_state(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equal_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def parse_slot(value: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"S(0[1-9]|1[0-6])", value)
    if match is None:
        raise ValueError("slot must be S01 through S16")
    index = int(match.group(1)) - 1
    return value, ORDERS[index % len(ORDERS)]


def source_path(source_root: Path, seed: int, arm: str) -> Path:
    return (
        source_root / f"seed{seed}" / f"prefix_{arm}"
        / "training-state-kimg000512.pt"
    ).resolve(strict=True)


def manifest_value(
    *, seed: int, branch: str, source: Path, run_dir: Path
) -> dict:
    origin = BRANCHES[branch]
    return {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": m1.PROTOCOL_ID,
        "run_kind": "formal",
        "branch": branch,
        "seed": seed,
        "origin_arm": origin,
        "continuation_arm": "A",
        "switch_kimg": 512,
        "final_kimg": 1024,
        "source_state": {"path": str(source)},
        "immutable_output_root": str(run_dir.resolve()),
        "m1_shadow_update": True,
    }


def load_source(source: Path, manifest: dict) -> dict:
    state = torch.load(source, map_location="cpu", weights_only=False)
    schedule_switch.verify_source_state(state, manifest)
    return state


def check_paired_random_streams(source_root: Path, seed: int) -> None:
    streams = {}
    for arm, branch in (("A", "K_A"), ("B", "K_B")):
        source = source_path(source_root, seed, arm)
        manifest = manifest_value(
            seed=seed, branch=branch, source=source, run_dir=Path("/")
        )
        state = load_source(source, manifest)
        rank = state["rank_states"][0]
        streams[arm] = (
            copy.deepcopy(rank["rng_state"]),
            copy.deepcopy(rank["sampler_state"]),
        )
        del state
    if not equal_state(streams["A"][0], streams["B"][0]):
        raise RuntimeError(f"seed {seed} A/B source RNG states differ")
    if not equal_state(streams["A"][1], streams["B"][1]):
        raise RuntimeError(f"seed {seed} A/B source sampler states differ")


def runtime_environment(gpu: int, runtime_python: Path) -> dict:
    prefix = runtime_python.resolve(strict=True).parent.parent
    site = prefix / "lib/python3.11/site-packages"
    libraries = [prefix / "lib", site / "torch/lib"]
    nvidia = site / "nvidia"
    if nvidia.is_dir():
        libraries.extend(sorted(nvidia.glob("*/lib")))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_DISABLE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "LD_LIBRARY_PATH": ":".join(str(path) for path in libraries),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
    })
    return env


def training_command(
    runtime_python: Path, dataset: Path, seed: int, manifest: Path,
    resume: Path,
) -> list[str]:
    return [
        str(runtime_python), "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", str(REPO_ROOT / "ct_train.py"),
        f"--data={dataset}", f"--outdir={manifest.parent}", "--nosubdir",
        "--cond=False", "--arch=ddpmpp", "--precond=ect",
        "--batch=128", "--batch-gpu=16", "--optim=RAdam", "--lr=0.0001",
        "--dropout=0.2", "--augment=0", "--xflip=False",
        "--mean=-1.1", "--std=2.0", "--mapping=sigmoid",
        "--global-gap-scale=1.0", "--factorial-protocol=q256_target_weight_v1",
        "--target-gap-scale=1.0", "--denominator-gap-scale=1.0",
        "-q", "256", "-k", "8", "-b", "1", "-c", "0",
        "--double=10000", "--ema_beta=0.9993", f"--seed={seed}",
        "--fp16=True", "--tf32=False", "--ls=1.0", "--enable_amp=True",
        "--bench=False", "--cache=True", "--workers=1", "--metrics=none",
        "--duration=1.024", "--tick=10", "--snap=0", "--dump=0",
        "--ckpt=10", "--sample_every=26", "--eval_every=50",
        "--mid_t=0.821", "--adaptive-update-kimg=0.5",
        f"--schedule-switch-manifest={manifest}", f"--resume={resume}",
        "--immutable-checkpoint-kimg=640,768,896,1024",
    ]


def validate_branch_state(path: Path, manifest: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    schedule_switch.verify_switched_state(state, manifest)
    m1.validate_resumed_state(state, manifest)
    attempt = int(state.get("attempted_iteration", -1))
    nimg = int(state.get("cur_nimg", -1))
    if not 4000 <= attempt <= 8000 or nimg != attempt * 128:
        raise RuntimeError("M1 resume progress is inconsistent")
    return {"state": state, "attempt": attempt, "nimg": nimg}


def select_resume(run_dir: Path, source: Path, manifest: dict) -> tuple[Path, int]:
    candidates = [run_dir / "training-state-latest.pt"] + [
        run_dir / f"training-state-kimg{kimg:06d}.pt"
        for kimg in (1024, 896, 768, 640, 512)
    ]
    valid = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            result = validate_branch_state(path, manifest)
            valid.append((result["attempt"], path))
            del result["state"]
        except Exception as exc:
            print(f"Ignoring unusable resume candidate {path}: {exc}", flush=True)
    if valid:
        attempt, path = max(valid, key=lambda item: item[0])
        return path.resolve(), attempt
    if any(run_dir.iterdir()):
        allowed = {"formal_run_manifest.json", "branch_status.json"}
        if any(path.name not in allowed for path in run_dir.iterdir()):
            raise RuntimeError(f"existing branch has no usable checkpoint: {run_dir}")
    state = load_source(source, manifest)
    del state
    return source, schedule_switch.SWITCH_ATTEMPT


def scientific_failure(log: Path) -> bool:
    text = log.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in SCIENTIFIC_FAILURE_MARKERS)


def run_branch(
    *, slot: str, seed: int, branch: str, gpu: int, source_root: Path,
    dataset: Path, runtime_python: Path, output_root: Path,
) -> dict:
    run_dir = output_root / slot / branch
    run_dir.mkdir(parents=True, exist_ok=True)
    source = source_path(source_root, seed, BRANCHES[branch])
    manifest = manifest_value(
        seed=seed, branch=branch, source=source, run_dir=run_dir
    )
    manifest_path = run_dir / "formal_run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"existing manifest differs: {manifest_path}")
    else:
        write_json(manifest_path, manifest)
    schedule_switch.load_run_manifest(manifest_path)

    status_path = run_dir / "branch_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") in {"PASS", "SCIENTIFIC_FAILURE"}:
            return old

    resume, resume_attempt = select_resume(run_dir, source, manifest)
    if resume_attempt == 8000:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        m1.validate_terminal_state(state, manifest)
        result = {
            "slot": slot, "seed": seed, "branch": branch, "status": "PASS",
            "gpu": gpu, "host": socket.gethostname(), "source_arm": BRANCHES[branch],
            "source_path": str(source), "resume_path": str(resume),
            "resume_attempt": resume_attempt, "terminal_path": str(resume),
            "ended_utc": utc_now(),
        }
        write_json(status_path, result)
        return result

    attempts = sorted(run_dir.glob("train-attempt-*.log"))
    log = run_dir / f"train-attempt-{len(attempts) + 1:02d}.log"
    command = training_command(runtime_python, dataset, seed, manifest_path, resume)
    started = utc_now()
    print(
        f"M1_START slot={slot} seed={seed} branch={branch} gpu={gpu} "
        f"resume_attempt={resume_attempt}", flush=True,
    )
    with log.open("xb") as handle:
        result = subprocess.run(
            command, cwd=REPO_ROOT,
            env=runtime_environment(gpu, runtime_python),
            stdout=handle, stderr=subprocess.STDOUT,
        )
    record = {
        "slot": slot, "seed": seed, "branch": branch,
        "order_position": None, "host": socket.gethostname(), "gpu": gpu,
        "source_arm": BRANCHES[branch], "source_path": str(source),
        "resume_path": str(resume), "resume_attempt": resume_attempt,
        "command": command, "log_path": str(log.resolve()),
        "started_utc": started, "ended_utc": utc_now(),
        "exit_code": result.returncode,
    }
    if result.returncode != 0:
        record["status"] = (
            "SCIENTIFIC_FAILURE" if scientific_failure(log) else "TECHNICAL_FAILURE"
        )
        write_json(status_path, record)
        if record["status"] == "SCIENTIFIC_FAILURE":
            return record
        raise RuntimeError(f"M1 training failed: {run_dir}; see {log}")

    terminal = run_dir / "training-state-kimg001024.pt"
    state = torch.load(terminal.resolve(strict=True), map_location="cpu", weights_only=False)
    m1.validate_terminal_state(state, manifest)
    record.update({"status": "PASS", "terminal_path": str(terminal.resolve())})
    write_json(status_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--next-slot")
    parser.add_argument("--next-seed", type=int)
    args = parser.parse_args()

    if (args.next_slot is None) != (args.next_seed is None):
        raise ValueError("--next-slot and --next-seed must be supplied together")
    jobs = [(args.slot, args.seed)]
    if args.next_slot is not None:
        jobs.append((args.next_slot, args.next_seed))
    source_root = args.source_root.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    runtime_python = args.runtime_python.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lock_path = Path(f"/tmp/m1-gpu-{args.gpu}.lock")
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"GPU {args.gpu} already has an M1 lane") from exc

    for slot_value, seed in jobs:
        slot, order = parse_slot(slot_value)
        check_paired_random_streams(source_root, seed)
        results = {}
        for position, branch in enumerate(order, start=1):
            result = run_branch(
                slot=slot, seed=seed, branch=branch, gpu=args.gpu,
                source_root=source_root, dataset=dataset,
                runtime_python=runtime_python, output_root=output_root,
            )
            result["order_position"] = position
            write_json(output_root / slot / branch / "branch_status.json", result)
            results[branch] = result
        status = (
            "PASS" if all(row["status"] == "PASS" for row in results.values())
            else "COMPLETE_WITH_SCIENTIFIC_FAILURES"
        )
        write_json(output_root / slot / "slot_status.json", {
            "slot": slot, "seed": seed, "order": list(order),
            "host": socket.gethostname(), "gpu": args.gpu,
            "status": status, "branches": results, "ended_utc": utc_now(),
        })
        print(f"M1_SLOT_{status} slot={slot} seed={seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
