#!/usr/bin/env python3
"""Fail-closed launcher for the frozen second-q q128 A/B training matrix.

The default ``validate`` action is local and read-only. ``preflight`` checks the
machine-local runtime without starting training. ``run`` starts exactly one
paired training seed (A followed by B); launch one process per seed/GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/second_q_ab_q128_learning_curve.frozen.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ACCEPTED_VERDICTS = {"SEMANTIC_EQUIVALENT", "NOT_EQUIVALENT"}


class ContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("schema") == "ect.second-q-ab-learning-curve/v1", "unexpected config schema")
    require(config.get("status") == "FROZEN_WAITING_ROLE_E", "config is not in the frozen waiting state")

    scope = config.get("scope", {})
    require(scope.get("included_arms") == ["A", "B"], "only A/B arms are allowed")
    exclusions = set(scope.get("excluded_work", []))
    require({"new_q256_training_seeds", "arm_C", "arm_D"}.issubset(exclusions), "scope exclusions are incomplete")

    gate = config.get("provenance_gate", {})
    require(gate.get("launch_requires_role_e_verdict") is True, "Role E launch gate must be enabled")
    require(set(gate.get("accepted_verdicts", [])) == ACCEPTED_VERDICTS, "accepted Role E verdicts changed")
    require(gate.get("selected_execution_path") == "canonical_fresh_rerun", "launcher only supports canonical fresh rerun")
    for name in ("legacy_q128_dataset_sha256", "q256_canonical_training_dataset_sha256"):
        require(bool(SHA256_RE.fullmatch(str(gate.get(name, "")))), f"invalid {name}")

    source = config.get("source_contract", {})
    require(bool(GIT_COMMIT_RE.fullmatch(str(source.get("reference_training_commit", "")))), "invalid training reference commit")
    require(source.get("allowed_change_from_reference") == "schedule q: 256 -> 128 only", "only-q source contract changed")
    require(len(source.get("training_paths_required_byte_equivalent", [])) >= 7, "training path contract is incomplete")

    training = config.get("training", {})
    require(training.get("schedule_q") == 128, "second-q schedule must be q=128")
    require(training.get("training_seeds") == [3, 4, 5], "training seeds must remain 3/4/5")
    require(training.get("final_budget_kimg") == 1024, "final budget must remain 1024 kimg")
    require(training.get("immutable_checkpoint_kimg") == [256, 384, 512, 640, 768, 896, 1024], "checkpoint plan changed")
    require(bool(SHA256_RE.fullmatch(str(training.get("transfer_checkpoint_sha256", "")))), "invalid transfer SHA256")
    arms = training.get("arms", {})
    require(set(arms) == {"A", "B"}, "training matrix must contain only A/B")
    require(arms["A"].get("target_gap_scale") == 1.0 and arms["A"].get("denominator_gap_scale") == 1.0, "arm A identity changed")
    require(arms["B"].get("target_gap_scale") == 1.1 and arms["B"].get("denominator_gap_scale") == 1.1, "arm B identity changed")

    evaluation = config.get("evaluation", {})
    require(evaluation.get("precision") == "fp32", "evaluation precision changed")
    require(evaluation.get("sample_count") == 50000, "sample count changed")
    require((evaluation.get("sample_seed_start"), evaluation.get("sample_seed_end")) == (0, 49999), "sample seeds changed")
    require(evaluation.get("metric_seed") == 20260730, "metric seed changed")
    primary = evaluation.get("primary", {})
    require(primary.get("metric") == "fid50k_full" and primary.get("nfe") == 1 and primary.get("mid_t") == [], "primary endpoint changed")
    require(primary.get("budgets_kimg") == [512, 640, 768, 896, 1024], "primary budget curve changed")
    secondary_nfe2 = evaluation.get("secondary", {}).get("nfe2", {})
    require(secondary_nfe2.get("nfe") == 2 and secondary_nfe2.get("mid_t") == [0.821], "NFE2 contract changed")


def validate_verdict(verdict: dict[str, Any], config: dict[str, Any]) -> str:
    require(verdict.get("schema") == "ect.role-e.dataset-semantic-equivalence-verdict/v1", "unexpected Role E verdict schema")
    require(verdict.get("role") == "Role E", "dataset verdict is not signed by Role E")
    decision = str(verdict.get("verdict", ""))
    require(decision in ACCEPTED_VERDICTS, f"Role E verdict is not conclusive: {decision or 'MISSING'}")
    gate = config["provenance_gate"]
    require(verdict.get("legacy_q128_dataset_sha256") == gate["legacy_q128_dataset_sha256"], "legacy q128 dataset identity mismatch")
    require(verdict.get("q256_canonical_training_dataset_sha256") == gate["q256_canonical_training_dataset_sha256"], "q256 canonical dataset identity mismatch")
    for name in ("legacy_semantic_manifest_sha256", "canonical_semantic_manifest_sha256", "evidence_manifest_sha256"):
        require(bool(SHA256_RE.fullmatch(str(verdict.get(name, "")))), f"invalid or missing Role E {name}")
    require(bool(verdict.get("audit_id")), "missing Role E audit_id")
    require(str(verdict.get("signed_off_utc", "")).endswith("Z"), "Role E signed_off_utc must be UTC")
    return decision


def run_checked(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ContractError(f"command failed ({result.returncode}): {shlex.join(command)}{': ' + detail if detail else ''}")
    return (result.stdout or "").strip()


def runtime_hash_from_manifest(manifest: Path) -> str:
    require(manifest.is_file(), f"missing runtime release manifest: {manifest}")
    matches: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[-1] == "./runtime/ect-pytorch2401-deterministic.sif":
            matches.append(fields[0])
    require(len(matches) == 1 and bool(SHA256_RE.fullmatch(matches[0])), "runtime SIF identity missing or ambiguous in release manifest")
    return matches[0]


def ensure_within(child: Path, parent: Path, label: str) -> None:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} is outside declared root: {child}") from exc


def machine_preflight(args: argparse.Namespace, config: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    decision = validate_verdict(verdict, config)
    repo = args.repo.resolve()
    dataset = args.dataset.resolve()
    transfer = args.transfer.resolve()
    runtime_python = args.runtime_python.resolve()
    runtime_sif = args.runtime_sif.resolve()
    run_root = args.run_root.resolve()

    require(repo.is_dir(), f"missing repository: {repo}")
    require(dataset.is_file(), f"missing canonical dataset: {dataset}")
    require(transfer.is_file(), f"missing transfer checkpoint: {transfer}")
    require(runtime_python.is_file() and os.access(runtime_python, os.X_OK), f"missing runtime Python: {runtime_python}")
    require(runtime_sif.is_file(), f"missing runtime SIF: {runtime_sif}")
    ensure_within(run_root, args.job_root.resolve(), "run root")

    expected_dataset = config["provenance_gate"]["q256_canonical_training_dataset_sha256"]
    expected_transfer = config["training"]["transfer_checkpoint_sha256"]
    actual_dataset = sha256_file(dataset)
    actual_transfer = sha256_file(transfer)
    require(actual_dataset == expected_dataset, f"canonical dataset SHA256 mismatch: {actual_dataset}")
    require(actual_transfer == expected_transfer, f"transfer checkpoint SHA256 mismatch: {actual_transfer}")

    source = config["source_contract"]
    reference = source["reference_training_commit"]
    paths = source["training_paths_required_byte_equivalent"]
    run_checked(["git", "-C", str(repo), "cat-file", "-e", f"{reference}^{{commit}}"])
    run_checked(["git", "-C", str(repo), "diff", "--quiet", f"{reference}..HEAD", "--", *paths])
    dirty = run_checked(["git", "-C", str(repo), "status", "--porcelain"], capture=True)
    require(not dirty, "source worktree is dirty")
    git_commit = run_checked(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True)

    expected_runtime = runtime_hash_from_manifest(args.runtime_manifest)
    actual_runtime = sha256_file(runtime_sif)
    require(actual_runtime == expected_runtime, f"runtime SIF SHA256 mismatch: {actual_runtime}")

    return {
        "status": "GO",
        "role_e_verdict": decision,
        "execution_path": "canonical_fresh_rerun",
        "config_sha256": sha256_file(args.config.resolve()),
        "role_e_verdict_sha256": sha256_file(args.verdict.resolve()),
        "git_commit": git_commit,
        "dataset_sha256": actual_dataset,
        "transfer_sha256": actual_transfer,
        "runtime_sif_sha256": actual_runtime,
        "run_root": str(run_root),
    }


def training_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    seed: int,
    arm: str,
    run_dir: Path,
) -> list[str]:
    training = config["training"]
    shared = training["shared_hyperparameters"]
    arm_config = training["arms"][arm]
    checkpoints = ",".join(str(value) for value in training["immutable_checkpoint_kimg"])
    return [
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
        "--factorial-protocol=q256_target_weight_v1",
        f"--target-gap-scale={arm_config['target_gap_scale']}",
        f"--denominator-gap-scale={arm_config['denominator_gap_scale']}",
        "-q",
        str(training["schedule_q"]),
        "-k",
        str(shared["k"]),
        "-b",
        str(shared["b"]),
        "-c",
        str(shared["c"]),
        f"--double={shared['double']}",
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
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
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
        }
    )
    return env


def execute_seed(args: argparse.Namespace, config: dict[str, Any], receipt: dict[str, Any]) -> None:
    require(args.seed in config["training"]["training_seeds"], f"unsupported seed: {args.seed}")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    seed_root = run_root / f"seed{args.seed}"
    seed_root.mkdir(exist_ok=False)
    (seed_root / "preflight_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = runtime_environment(args)
    for arm in ("A", "B"):
        run_dir = seed_root / f"arm{arm}"
        run_dir.mkdir(exist_ok=False)
        command = training_command(args, config, args.seed, arm, run_dir)
        launch_record = {
            "schema": "ect.second-q-training-launch/v1",
            "seed": args.seed,
            "arm": arm,
            "command": command,
            "scientific_identity": config["training"]["arms"][arm],
            "preflight": receipt,
        }
        (run_dir / "launch_record.json").write_text(json.dumps(launch_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[second-q] START seed={args.seed} arm={arm}", flush=True)
        result = subprocess.run(command, cwd=args.repo.resolve(), env=env, check=False)
        if result.returncode != 0:
            raise ContractError(f"training crashed for seed={args.seed} arm={arm}; no automatic retry was attempted")
        for budget in config["training"]["immutable_checkpoint_kimg"]:
            state = run_dir / f"training-state-kimg{budget:06d}.pt"
            snapshot = run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
            require(state.is_file() and state.stat().st_size > 0, f"missing immutable state: {state}")
            require(snapshot.is_file() and snapshot.stat().st_size > 0, f"missing immutable snapshot: {snapshot}")
        print(f"[second-q] PASS seed={args.seed} arm={arm}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verdict", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--job-root", type=Path, default=Path("/root/second_q_q128_ab_v1"))
    parser.add_argument("--repo", type=Path, default=Path("/root/second_q_q128_ab_v1/source/recurrence_of_ect"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/second_q_q128_ab_v1/runs/second-q-q128-ab-v1"))
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/ect_project/datasets/cifar10-32x32.zip"))
    parser.add_argument("--transfer", type=Path, default=Path("/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"))
    parser.add_argument("--runtime-sif", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/ect-pytorch2401-deterministic.sif"))
    parser.add_argument("--runtime-manifest", type=Path, default=Path("/mnt/ect_project/q256_target_weight_1024k/SHA256SUMS.release.txt"))
    parser.add_argument("--sandbox-root", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/sandbox"))
    parser.add_argument("--runtime-python", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/sandbox/usr/bin/python"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = read_json(args.config)
        validate_config(config)
        if args.action == "validate":
            print(json.dumps({"status": "WAIT_ROLE_E", "config": str(args.config), "protocol_id": config["protocol_id"]}, sort_keys=True))
            return 0
        require(args.verdict is not None, "--verdict is required for preflight and run")
        verdict = read_json(args.verdict)
        receipt = machine_preflight(args, config, verdict)
        if args.action == "preflight":
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        require(args.seed is not None, "--seed is required for run")
        execute_seed(args, config, receipt)
        return 0
    except ContractError as exc:
        print(f"[second-q] NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
