#!/usr/bin/env python3
"""Fail-closed launcher for the frozen second-q q128 A/B training matrix.

The default ``validate`` action is local and read-only. ``preflight`` checks the
machine-local runtime and exhaustively smokes the canonical dataset without
starting training. ``run`` starts exactly one paired training seed (A followed
by B); launch one process per seed/GPU.
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
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/second_q_ab_q128_learning_curve_v2.frozen.json"
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
    schema = config.get("schema")
    require(schema in {"ect.second-q-ab-learning-curve/v1", "ect.second-q-ab-learning-curve/v2"}, "unexpected config schema")
    if schema.endswith("/v1"):
        require(config.get("status") == "FROZEN_WAITING_ROLE_E", "v1 config is not in the frozen waiting state")
    else:
        require(config.get("status") == "FROZEN_CANONICAL_DATASET_READY_FOR_PREFLIGHT", "v2 config is not frozen for canonical preflight")
        amendment = config.get("amendment", {})
        require(amendment.get("supersedes_protocol_commit") == "05157e7a0532b02184e2c38d051fe8c4c8aabac4", "v2 does not supersede the immutable v1 freeze")
        require(amendment.get("reason") == "unresolved dataset identity", "unexpected v2 amendment reason")
        require(amendment.get("scientific_results_observed_before_amendment") is False, "v2 must be pre-result")
        require(amendment.get("training_started_before_amendment") is False, "v2 must be pre-training")

    scope = config.get("scope", {})
    require(scope.get("included_arms") == ["A", "B"], "only A/B arms are allowed")
    exclusions = set(scope.get("excluded_work", []))
    require({"new_q256_training_seeds", "arm_C", "arm_D"}.issubset(exclusions), "scope exclusions are incomplete")

    gate = config.get("provenance_gate", {})
    if schema.endswith("/v1"):
        require(gate.get("launch_requires_role_e_verdict") is True, "Role E launch gate must be enabled")
        require(set(gate.get("accepted_verdicts", [])) == ACCEPTED_VERDICTS, "accepted Role E verdicts changed")
    else:
        require(gate.get("required_preflight_status") == "GO_CANONICAL_DATASET", "v2 canonical GO status changed")
        require(gate.get("role_e_semantic_equivalence_verdict_required") is False, "v2 must not wait on legacy ZIP equivalence")
        require("legacy_q128_dataset_equivalence_recovery" in exclusions, "v2 must exclude legacy ZIP recovery")
        smoke = gate.get("dataset_loader_smoke", {})
        require(smoke.get("sample_count") == 50000 and smoke.get("class_count") == 10, "dataset-loader smoke population changed")
        require(smoke.get("image_shape") == [3, 32, 32] and smoke.get("image_dtype") == "uint8", "dataset-loader image contract changed")
        require(smoke.get("preprocessing") == "float32(uint8_image) / 127.5 - 1.0", "dataset preprocessing changed")
        require(bool(SHA256_RE.fullmatch(str(smoke.get("loader_sha256", "")))), "invalid dataset loader SHA256")
    require(gate.get("selected_execution_path") == "canonical_fresh_rerun", "launcher only supports canonical fresh rerun")
    for name in ("legacy_q128_dataset_sha256", "q256_canonical_training_dataset_sha256"):
        require(bool(SHA256_RE.fullmatch(str(gate.get(name, "")))), f"invalid {name}")

    source = config.get("source_contract", {})
    require(bool(GIT_COMMIT_RE.fullmatch(str(source.get("reference_training_commit", "")))), "invalid training reference commit")
    expected_source_change = (
        "schedule q: 256 -> 128 only"
        if schema.endswith("/v1")
        else "scientific CLI q: 256 -> 128; validation-only source amendment admits q=128 to the otherwise unchanged strict path"
    )
    require(source.get("allowed_change_from_reference") == expected_source_change, "only-q source contract changed")
    minimum_frozen_paths = 7 if schema.endswith("/v1") else 6
    require(len(source.get("training_paths_required_byte_equivalent", [])) >= minimum_frozen_paths, "training path contract is incomplete")
    if schema.endswith("/v2"):
        q_scope = source.get("strict_protocol_q_scope_amendment", {})
        require(q_scope.get("optimizer_steps_before_amendment") == 0, "q-scope amendment must precede optimizer steps")
        require(q_scope.get("scientific_results_observed_before_amendment") is False, "q-scope amendment must be pre-result")
        require(q_scope.get("scientific_math_changed") is False, "q-scope amendment must be validation-only")
        require(q_scope.get("path") == "training/loss.py", "unexpected q-scope amendment path")
        require(bool(SHA256_RE.fullmatch(str(q_scope.get("amended_file_sha256", "")))), "invalid amended loss SHA256")
        runtime_execution = config.get("runtime_execution", {})
        require(runtime_execution.get("training_started_before_runtime_amendment") is False, "runtime amendment must be pre-training")
        require(runtime_execution.get("scientific_results_observed_before_runtime_amendment") is False, "runtime amendment must be pre-result")
        require(runtime_execution.get("cell_mode") == "one_seed_by_arm_cell_per_single_gpu_process", "v2 cell execution mode changed")
        require(runtime_execution.get("max_concurrent_cells") == 6, "v2 must freeze six concurrent cells")
        require(runtime_execution.get("shared_preflight_receipt_required") is True, "v2 requires one shared preflight receipt")
        require(set(runtime_execution.get("gpu_assignment", {}).values()) == {
            "seed3-armA", "seed3-armB", "seed4-armA",
            "seed4-armB", "seed5-armA", "seed5-armB",
        }, "v2 GPU assignment does not cover exactly six cells")
        artifact_export = config.get("artifact_export", {})
        require(
            artifact_export.get("script")
            == "scripts/export_second_q_ab_snapshots.py",
            "v2 artifact exporter changed",
        )
        require(
            artifact_export.get("required_snapshot_count") == 42,
            "v2 snapshot matrix must contain 42 outputs",
        )
        require(
            artifact_export.get("rng_unchanged_required") is True,
            "v2 snapshot export must preserve RNG",
        )

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
    if schema.endswith("/v2"):
        require(primary.get("execution_priority_is_not_selection") is True, "execution priority must not become selection")
        require(primary.get("all_frozen_budgets_mandatory") is True, "all primary budgets must remain mandatory")
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


def canonical_loader_smoke(
    args: argparse.Namespace, config: dict[str, Any], env: dict[str, str]
) -> dict[str, Any]:
    gate = config["provenance_gate"]
    smoke = gate["dataset_loader_smoke"]
    evaluation = config["evaluation"]
    command = [
        str(args.runtime_python.resolve()),
        str(args.repo.resolve() / smoke["script"]),
        "--repo",
        str(args.repo.resolve()),
        "--dataset",
        str(args.dataset.resolve()),
        "--expected-dataset-sha256",
        gate["q256_canonical_training_dataset_sha256"],
        "--expected-loader-sha256",
        smoke["loader_sha256"],
        "--detector-sha256",
        evaluation["detector_sha256"],
        "--fid-reference-sha256",
        evaluation["fid_reference_sha256"],
        "--kid-reference-sha256",
        evaluation["kid_reference_sha256"],
    ]
    result = subprocess.run(
        command,
        cwd=args.repo.resolve(),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ContractError(
            "canonical dataset-loader smoke failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("canonical dataset-loader smoke did not return JSON") from exc
    require(receipt.get("status") == "PASS", "canonical dataset-loader smoke did not pass")
    return receipt


def machine_preflight(
    args: argparse.Namespace,
    config: dict[str, Any],
    verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    is_v2 = config["schema"].endswith("/v2")
    if is_v2:
        require(verdict is None, "v2 canonical binding does not accept a legacy equivalence verdict")
        decision = "NOT_REQUIRED_CANONICAL_BINDING"
    else:
        require(verdict is not None, "v1 requires a Role E verdict")
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
    if is_v2:
        q_scope = source["strict_protocol_q_scope_amendment"]
        amended_path = repo / q_scope["path"]
        require(
            sha256_file(amended_path) == q_scope["amended_file_sha256"],
            "validation-only q-scope amendment SHA256 mismatch",
        )
    dirty = run_checked(["git", "-C", str(repo), "status", "--porcelain"], capture=True)
    require(not dirty, "source worktree is dirty")
    git_commit = run_checked(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True)

    expected_runtime = runtime_hash_from_manifest(args.runtime_manifest)
    actual_runtime = sha256_file(runtime_sif)
    require(actual_runtime == expected_runtime, f"runtime SIF SHA256 mismatch: {actual_runtime}")

    loader_receipt = None
    if is_v2:
        loader_receipt = canonical_loader_smoke(args, config, runtime_environment(args))

    receipt = {
        "status": "GO_CANONICAL_DATASET" if is_v2 else "GO",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "role_e_verdict": decision,
        "execution_path": "canonical_fresh_rerun",
        "config_sha256": sha256_file(args.config.resolve()),
        "git_commit": git_commit,
        "dataset_sha256": actual_dataset,
        "dataset_path": str(dataset),
        "dataset_stat": {"size": dataset.stat().st_size, "mtime_ns": dataset.stat().st_mtime_ns},
        "transfer_sha256": actual_transfer,
        "transfer_path": str(transfer),
        "transfer_stat": {"size": transfer.stat().st_size, "mtime_ns": transfer.stat().st_mtime_ns},
        "runtime_sif_sha256": actual_runtime,
        "runtime_sif_path": str(runtime_sif),
        "runtime_sif_stat": {"size": runtime_sif.stat().st_size, "mtime_ns": runtime_sif.stat().st_mtime_ns},
        "run_root": str(run_root),
    }
    if verdict is not None:
        receipt["role_e_verdict_sha256"] = sha256_file(args.verdict.resolve())
    if loader_receipt is not None:
        receipt["dataset_loader_smoke"] = loader_receipt
    return receipt


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


def write_preflight_receipt(path: Path, receipt: dict[str, Any], job_root: Path) -> None:
    ensure_within(path, job_root, "preflight receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite preflight receipt: {path}")
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def load_bound_preflight_receipt(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    require(args.preflight_receipt is not None, "--preflight-receipt is required for run")
    path = args.preflight_receipt.resolve()
    ensure_within(path, args.job_root.resolve(), "preflight receipt")
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), "missing preflight receipt or SHA256 sidecar")
    fields = sidecar.read_text(encoding="utf-8").split()
    require(len(fields) == 2 and fields[1] == path.name, "invalid preflight receipt sidecar")
    require(fields[0] == sha256_file(path), "preflight receipt SHA256 mismatch")
    receipt = read_json(path)
    require(receipt.get("status") == "GO_CANONICAL_DATASET", "preflight receipt is not GO_CANONICAL_DATASET")
    require(receipt.get("execution_path") == "canonical_fresh_rerun", "preflight execution path mismatch")
    require(receipt.get("config_sha256") == sha256_file(args.config.resolve()), "preflight config binding mismatch")
    require(receipt.get("run_root") == str(args.run_root.resolve()), "preflight run-root binding mismatch")
    require(receipt.get("dataset_sha256") == config["provenance_gate"]["q256_canonical_training_dataset_sha256"], "preflight dataset identity mismatch")
    require(receipt.get("transfer_sha256") == config["training"]["transfer_checkpoint_sha256"], "preflight transfer identity mismatch")
    require(receipt.get("dataset_loader_smoke", {}).get("status") == "PASS", "preflight loader smoke did not pass")

    repo = args.repo.resolve()
    git_commit = run_checked(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True)
    require(receipt.get("git_commit") == git_commit, "source commit changed after preflight")
    dirty = run_checked(["git", "-C", str(repo), "status", "--porcelain"], capture=True)
    require(not dirty, "source worktree became dirty after preflight")
    for label, file_path in (
        ("dataset", args.dataset.resolve()),
        ("transfer", args.transfer.resolve()),
        ("runtime_sif", args.runtime_sif.resolve()),
    ):
        require(file_path.is_file(), f"{label} disappeared after preflight")
        expected_path = receipt.get(f"{label}_path")
        expected_stat = receipt.get(f"{label}_stat", {})
        actual_stat = file_path.stat()
        require(expected_path == str(file_path), f"{label} path changed after preflight")
        require(
            expected_stat == {"size": actual_stat.st_size, "mtime_ns": actual_stat.st_mtime_ns},
            f"{label} file changed after preflight",
        )
    return receipt


def execute_cell(args: argparse.Namespace, config: dict[str, Any], receipt: dict[str, Any]) -> None:
    require(args.seed in config["training"]["training_seeds"], f"unsupported seed: {args.seed}")
    require(args.arm in config["training"]["arms"], f"unsupported arm: {args.arm}")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    seed_root = run_root / f"seed{args.seed}"
    seed_root.mkdir(exist_ok=True)

    env = runtime_environment(args)
    run_dir = seed_root / f"arm{args.arm}"
    run_dir.mkdir(exist_ok=False)
    command = training_command(args, config, args.seed, args.arm, run_dir)
    launch_record = {
        "schema": "ect.second-q-training-launch/v3",
        "seed": args.seed,
        "arm": args.arm,
        "command": command,
        "scientific_identity": config["training"]["arms"][args.arm],
        "preflight_receipt_path": str(args.preflight_receipt.resolve()),
        "preflight_receipt_sha256": sha256_file(args.preflight_receipt.resolve()),
    }
    (run_dir / "launch_record.json").write_text(json.dumps(launch_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[second-q] START seed={args.seed} arm={args.arm}", flush=True)
    result = subprocess.run(command, cwd=args.repo.resolve(), env=env, check=False)
    if result.returncode != 0:
        raise ContractError(f"training crashed for seed={args.seed} arm={args.arm}; no automatic retry was attempted")
    export_command = [
        str(args.runtime_python.resolve()),
        str(args.repo.resolve() / "scripts/export_second_q_ab_snapshots.py"),
        "--run-root",
        str(run_root),
        "--cells",
        f"{args.seed}:{args.arm}",
        "--summary-out",
        str(run_dir / "ema_export_summary.json"),
    ]
    export_result = subprocess.run(
        export_command,
        cwd=args.repo.resolve(),
        env=env,
        check=False,
    )
    if export_result.returncode != 0:
        raise ContractError(
            f"EMA snapshot export failed for seed={args.seed} arm={args.arm}"
        )
    for budget in config["training"]["immutable_checkpoint_kimg"]:
        state = run_dir / f"training-state-kimg{budget:06d}.pt"
        snapshot = run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
        receipt_path = snapshot.with_suffix(".receipt.json")
        require(state.is_file() and state.stat().st_size > 0, f"missing immutable state: {state}")
        require(snapshot.is_file() and snapshot.stat().st_size > 0, f"missing immutable snapshot: {snapshot}")
        require(
            receipt_path.is_file() and receipt_path.stat().st_size > 0,
            f"missing immutable snapshot receipt: {receipt_path}",
        )
    print(f"[second-q] PASS seed={args.seed} arm={args.arm}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verdict", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--job-root", type=Path, default=Path("/root/second_q_q128_ab_v2"))
    parser.add_argument("--repo", type=Path, default=Path("/root/second_q_q128_ab_v2/source/recurrence_of_ect"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/second_q_q128_ab_v2/runs/second-q-q128-ab-v2"))
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/ect_project/datasets/cifar10-32x32.zip"))
    parser.add_argument("--transfer", type=Path, default=Path("/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"))
    parser.add_argument("--runtime-sif", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/ect-pytorch2401-deterministic.sif"))
    parser.add_argument("--runtime-manifest", type=Path, default=Path("/mnt/ect_project/q256_target_weight_1024k/SHA256SUMS.release.txt"))
    parser.add_argument("--sandbox-root", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/sandbox"))
    parser.add_argument("--runtime-python", type=Path, default=Path("/root/q256_target_weight_1024k/runtime/sandbox/usr/bin/python"))
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = read_json(args.config)
        validate_config(config)
        if args.action == "validate":
            print(json.dumps({"status": config["status"], "config": str(args.config), "protocol_id": config["protocol_id"]}, sort_keys=True))
            return 0
        if config["schema"].endswith("/v2"):
            require(args.verdict is None, "--verdict is not used by v2 canonical binding")
            verdict = None
        else:
            require(args.verdict is not None, "--verdict is required for v1 preflight and run")
            verdict = read_json(args.verdict)
        if args.action == "preflight":
            receipt = machine_preflight(args, config, verdict)
            if args.receipt_out is not None:
                write_preflight_receipt(
                    args.receipt_out.resolve(), receipt, args.job_root.resolve()
                )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        require(args.seed is not None, "--seed is required for run")
        require(args.arm is not None, "--arm is required for one-cell-per-GPU run")
        receipt = load_bound_preflight_receipt(args, config)
        execute_cell(args, config, receipt)
        return 0
    except ContractError as exc:
        print(f"[second-q] NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
