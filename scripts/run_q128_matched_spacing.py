#!/usr/bin/env python3
"""Fail-closed launcher for the frozen q128 five-arm matched-spacing block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/q128_matched_spacing_v1.frozen.json"


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("schema") == "ect.q128-matched-spacing-training/v1", "unexpected schema")
    require(config.get("status") == "FROZEN_READY_FOR_TECHNICAL_PREFLIGHT", "config is not frozen")
    require(config.get("quality_unblinded_before_freeze") is False, "protocol was quality-unblinded")
    calibration = config.get("calibration", {})
    require(calibration.get("quality_blind") is True, "calibration must be quality-blind")
    require(calibration.get("selected_g128_star") == 0.55, "frozen g128* changed")
    require(calibration.get("optimum_on_boundary") is False, "calibration optimum is on boundary")
    training = config.get("training", {})
    require(training.get("schedule_q") == 128, "q must be 128")
    require(training.get("training_seeds") == [3, 4, 5], "seeds changed")
    require(training.get("final_budget_kimg") == 1024, "budget changed")
    require(
        training.get("immutable_checkpoint_kimg") == [256, 384, 512, 640, 768, 896, 1024],
        "checkpoint plan changed",
    )
    expected_arms = {
        "A": (1.0, 1.0),
        "Bsame": (1.1, 1.1),
        "Bmatch": (0.55, 0.55),
        "Cmatch": (0.55, 1.0),
        "Dmatch": (1.0, 0.55),
    }
    arms = training.get("arms", {})
    require(set(arms) == set(expected_arms), "five-arm matrix changed")
    for arm, factors in expected_arms.items():
        actual = arms[arm]
        require(
            (actual.get("target_gap_scale"), actual.get("denominator_gap_scale")) == factors,
            f"arm {arm} factors changed",
        )
    shared = training.get("shared_hyperparameters", {})
    require(shared.get("factorial_protocol") == "q128_matched_spacing_v1", "protocol id changed")
    require(shared.get("double_ticks") == 10000, "stage schedule changed")
    evaluation = config.get("evaluation", {})
    require(evaluation.get("job_count") == 210, "evaluation matrix size changed")
    require(evaluation.get("quality_block_blind") is True, "evaluation must remain block-blind")


def verify_source(repo: Path, config: dict[str, Any]) -> None:
    for relative, expected in config["source_contract"]["file_sha256"].items():
        path = repo / relative
        require(path.is_file(), f"missing frozen source file: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"source SHA256 mismatch for {relative}: {actual}")


def verify_calibration(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected = config["calibration"]
    require(path.is_file(), f"missing calibration manifest: {path}")
    require(sha256_file(path) == expected["manifest_sha256"], "calibration manifest SHA256 mismatch")
    manifest = read_json(path)
    require(manifest.get("status") == "FROZEN_PASS", "calibration is not PASS")
    require(manifest.get("quality_blind") is True, "calibration was not quality-blind")
    require(manifest.get("quality_metrics_read") == [], "calibration read quality metrics")
    require(manifest.get("selected_g128_star") == expected["selected_g128_star"], "g128* mismatch")
    require(manifest.get("sample_count") == expected["sample_count"], "calibration sample count mismatch")
    require(manifest.get("t_sample_raw_sha256") == expected["t_sample_raw_sha256"], "t sample hash mismatch")
    require(manifest.get("search", {}).get("optimum_on_boundary") is False, "calibration boundary failure")
    return manifest


def runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    sandbox = args.sandbox_root.resolve()
    library_parts = [
        sandbox / "usr/local/lib/python3.10/dist-packages/torch/lib",
        sandbox / "usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib",
        sandbox / "usr/local/cuda/compat/lib",
        sandbox / "usr/local/nvidia/lib",
        sandbox / "usr/local/nvidia/lib64",
        sandbox / "lib",
        sandbox / "lib/x86_64-linux-gnu",
        sandbox / "opt/hpcx/clusterkit/lib",
        sandbox / "opt/hpcx/hcoll/lib",
        sandbox / "opt/hpcx/nccl_rdma_sharp_plugin/lib",
        sandbox / "opt/hpcx/ompi/lib",
        sandbox / "opt/hpcx/sharp/lib",
        sandbox / "opt/hpcx/ucc/lib",
        sandbox / "opt/hpcx/ucx/lib",
        sandbox / "usr/local/cuda/targets/x86_64-linux/lib",
        sandbox / "usr/local/lib",
    ]
    path_parts = [
        sandbox / "usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin",
        sandbox / "usr/local/mpi/bin",
        sandbox / "usr/local/nvidia/bin",
        sandbox / "usr/local/cuda/bin",
        sandbox / "usr/local/sbin",
        sandbox / "usr/local/bin",
        sandbox / "usr/sbin",
        sandbox / "usr/bin",
        sandbox / "sbin",
        sandbox / "bin",
        sandbox / "usr/local/ucx/bin",
        sandbox / "opt/tensorrt/bin",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "USE_EXPERIMENTAL_CUDNN_V8_API": "1",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(str(path) for path in library_parts),
        "PATH": ":".join(str(path) for path in path_parts),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(args.master_port),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def training_command(args: argparse.Namespace, config: dict[str, Any], seed: int,
                     arm: str, run_dir: Path, *, smoke_attempts: int | None = None) -> list[str]:
    training = config["training"]
    shared = training["shared_hyperparameters"]
    factors = training["arms"][arm]
    checkpoints = ",".join(str(value) for value in training["immutable_checkpoint_kimg"])
    command = [
        str(args.runtime_python.resolve()),
        str(args.repo.resolve() / "ct_train.py"),
        f"--data={args.dataset.resolve()}",
        f"--outdir={run_dir}",
        "--nosubdir",
        "--cond=False",
        f"--arch={shared['architecture']}",
        f"--precond={shared['preconditioning']}",
        f"--batch={shared['global_batch']}",
        f"--batch-gpu={shared['batch_gpu']}",
        f"--optim={shared['optimizer']}",
        f"--lr={shared['learning_rate']}",
        f"--dropout={shared['dropout']}",
        f"--augment={shared['augment']}",
        f"--xflip={shared['xflip']}",
        "--mean=-1.1",
        "--std=2.0",
        f"--mapping={shared['mapping']}",
        "--global-gap-scale=1.0",
        f"--factorial-protocol={shared['factorial_protocol']}",
        f"--target-gap-scale={factors['target_gap_scale']}",
        f"--denominator-gap-scale={factors['denominator_gap_scale']}",
        "-q", str(training["schedule_q"]),
        "-k", str(shared["k"]),
        "-b", str(shared["b"]),
        "-c", str(shared["c"]),
        f"--double={shared['double_ticks']}",
        f"--ema_beta={shared['ema_beta']}",
        f"--seed={seed}",
        f"--fp16={shared['fp16']}",
        f"--tf32={shared['tf32']}",
        "--ls=1.0",
        f"--enable_amp={shared['amp']}",
        "--bench=False",
        "--cache=True",
        f"--workers={shared['workers']}",
        "--metrics=none",
        f"--duration={training['final_budget_kimg'] / 1000}",
        "--tick=10",
        "--snap=0",
        "--dump=0",
        "--ckpt=10",
        "--sample_every=10000",
        "--eval_every=10000",
        "--mid_t=0.821",
        "--adaptive-update-kimg=0.5",
        f"--immutable-checkpoint-kimg={checkpoints}",
        f"--transfer={args.transfer.resolve()}",
    ]
    if smoke_attempts is not None:
        command.append(f"--stop-after-attempts={smoke_attempts}")
    return command


def preflight(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    verify_source(args.repo.resolve(), config)
    calibration = verify_calibration(args.calibration_manifest.resolve(), config)
    assets = config["assets"]
    for label, path, expected in (
        ("dataset", args.dataset.resolve(), assets["dataset_sha256"]),
        ("transfer", args.transfer.resolve(), assets["transfer_sha256"]),
        ("runtime", args.runtime_sif.resolve(), assets["runtime_sif_sha256"]),
    ):
        require(path.is_file(), f"missing {label}: {path}")
        require(sha256_file(path) == expected, f"{label} SHA256 mismatch")
    require(args.runtime_python.resolve().is_file(), "runtime sandbox Python is missing")
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    require(gpu_query.returncode == 0 and "A100" in gpu_query.stdout, "A100 GPU preflight failed")
    receipt = {
        "schema": "ect.q128-matched-spacing-node-preflight/v1",
        "status": "GO",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": os.uname().nodename,
        "gpu": gpu_query.stdout.strip(),
        "config_sha256": sha256_file(args.config.resolve()),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest.resolve()),
        "selected_g128_star": calibration["selected_g128_star"],
        "dataset_sha256": assets["dataset_sha256"],
        "transfer_sha256": assets["transfer_sha256"],
        "runtime_sif_sha256": assets["runtime_sif_sha256"],
        "source_file_sha256": config["source_contract"]["file_sha256"],
        "run_root": str(args.run_root.resolve()),
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.receipt_out.exists(), f"refusing to overwrite receipt: {args.receipt_out}")
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.receipt_out.write_text(payload, encoding="utf-8")
    Path(str(args.receipt_out) + ".sha256").write_text(
        f"{hashlib.sha256(payload.encode()).hexdigest()}  {args.receipt_out.name}\n",
        encoding="utf-8",
    )
    return receipt


def load_receipt(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    require(args.preflight_receipt is not None, "--preflight-receipt is required")
    receipt = read_json(args.preflight_receipt.resolve())
    require(receipt.get("status") == "GO", "preflight is not GO")
    require(receipt.get("config_sha256") == sha256_file(args.config.resolve()), "config changed after preflight")
    require(receipt.get("run_root") == str(args.run_root.resolve()), "run root changed after preflight")
    verify_source(args.repo.resolve(), config)
    verify_calibration(args.calibration_manifest.resolve(), config)
    return receipt


def execute_cell(args: argparse.Namespace, config: dict[str, Any], seed: int,
                 arm: str, *, smoke: bool = False) -> None:
    require(seed in config["training"]["training_seeds"], f"unsupported seed: {seed}")
    require(arm in config["training"]["arms"], f"unsupported arm: {arm}")
    root = args.smoke_root.resolve() if smoke else args.run_root.resolve()
    run_dir = root / f"seed{seed}" / f"arm{arm}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    require(not run_dir.exists(), f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir()
    command = training_command(args, config, seed, arm, run_dir, smoke_attempts=16 if smoke else None)
    launch_record = {
        "schema": "ect.q128-matched-spacing-launch/v1",
        "seed": seed,
        "arm": arm,
        "smoke": smoke,
        "command": command,
        "scientific_identity": config["training"]["arms"][arm],
        "config_sha256": sha256_file(args.config.resolve()),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest.resolve()),
        "preflight_receipt_sha256": sha256_file(args.preflight_receipt.resolve()),
    }
    (run_dir / "launch_record.json").write_text(
        json.dumps(launch_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[q128-match] START seed={seed} arm={arm} smoke={smoke}", flush=True)
    result = subprocess.run(command, cwd=args.repo.resolve(), env=runtime_environment(args), check=False)
    require(result.returncode == 0, f"training failed seed={seed} arm={arm}")
    if smoke:
        print(f"[q128-match] SMOKE_PASS seed={seed} arm={arm}", flush=True)
        return
    export = [
        str(args.runtime_python.resolve()),
        str(args.repo.resolve() / "scripts/export_second_q_ab_snapshots.py"),
        "--run-root", str(args.run_root.resolve()),
        "--cells", f"{seed}:{arm}",
        "--summary-out", str(run_dir / "ema_export_summary.json"),
    ]
    require(
        subprocess.run(export, cwd=args.repo.resolve(), env=runtime_environment(args), check=False).returncode == 0,
        f"snapshot export failed seed={seed} arm={arm}",
    )
    for budget in config["training"]["immutable_checkpoint_kimg"]:
        stem = f"kimg{budget:06d}"
        require((run_dir / f"training-state-{stem}.pt").is_file(), f"missing state {stem}")
        require((run_dir / f"network-snapshot-{stem}.pkl").is_file(), f"missing snapshot {stem}")
        require((run_dir / f"network-snapshot-{stem}.receipt.json").is_file(), f"missing receipt {stem}")
    print(f"[q128-match] PASS seed={seed} arm={arm}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "preflight", "smoke", "run", "run-queue"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--job-root", type=Path, default=Path("/root/q128_matched_spacing_v1"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/q128_matched_spacing_v1/runs"))
    parser.add_argument("--smoke-root", type=Path, default=Path("/root/q128_matched_spacing_v1/smoke"))
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/ect_project/datasets/cifar10-32x32.zip"))
    parser.add_argument("--transfer", type=Path, default=Path("/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"))
    parser.add_argument("--runtime-sif", type=Path, default=Path("/mnt/ect_project/q256_target_weight_1024k/runtime/ect-pytorch2401-deterministic.sif"))
    parser.add_argument("--sandbox-root", type=Path, default=Path("/root/q128_runtime/sandbox"))
    parser.add_argument("--runtime-python", type=Path, default=Path("/root/q128_runtime/sandbox/usr/bin/python"))
    parser.add_argument("--calibration-manifest", type=Path, default=Path("/root/q128_matched_spacing_v1/calibration/calibration_manifest.json"))
    parser.add_argument("--receipt-out", type=Path, default=Path("/root/q128_matched_spacing_v1/preflight/node_preflight.json"))
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("A", "Bsame", "Bmatch", "Cmatch", "Dmatch"))
    parser.add_argument("--master-port", type=int, default=29641)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = read_json(args.config.resolve())
        validate_config(config)
        if args.action == "validate":
            verify_source(args.repo.resolve(), config)
            print(json.dumps({"status": "PASS", "protocol_id": config["protocol_id"]}, sort_keys=True))
            return 0
        if args.action == "preflight":
            print(json.dumps(preflight(args, config), indent=2, sort_keys=True))
            return 0
        load_receipt(args, config)
        require(args.arm is not None, "--arm is required")
        if args.action in {"smoke", "run"}:
            require(args.seed is not None, "--seed is required")
            execute_cell(args, config, args.seed, args.arm, smoke=args.action == "smoke")
        else:
            for seed in config["training"]["training_seeds"]:
                execute_cell(args, config, seed, args.arm)
        return 0
    except ContractError as exc:
        print(f"[q128-match] NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
