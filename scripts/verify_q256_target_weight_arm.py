#!/usr/bin/env python3
"""Fail-closed verifier for one q256 target-geometry x weighting arm.

The artifacts loaded here are trusted training outputs from this repository.
Both the network snapshot and the full training state use pickle-backed PyTorch
serialization and must not be supplied by an untrusted party.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import reproducibility


VERIFIER_VERSION = "2"
PROTOCOL = "q256_target_weight_v1"
EXPERIMENT_ID = "q256-target-weight-factorial"
LAUNCH_SCHEMA = "ect.q256.target-weight-factorial-launch/v2"
AUTHORIZATION_SCHEMA = "ect.q256.target-weight-factorial-launch-authorization/v2"
TELEMETRY_SCHEMA = "ect.q256.target-weight-training-telemetry/v1"
VALIDATION_SCHEMA = "ect.q256.target-weight-arm-validation/v2"
HASH_RECEIPT_SCHEMA = "ect.q256.target-weight-arm-artifact-hashes/v2"
AMP_SKIP_WARMUP_PROCESSED_NIMG = 10_000
AMP_SKIP_POLICY = {
    "schema": "ect.q256.target-weight-amp-skip-policy/v2",
    "kind": "observe_then_require_cross_arm_count_equivalence_within_seed",
    "allowed_region": "tick_0_amp_warmup_only",
    "warmup_processed_nimg_exclusive_upper_bound": AMP_SKIP_WARMUP_PROCESSED_NIMG,
    "require_finite_loss": True,
    "require_raw_nonfinite_exactly_on_skipped_attempts": True,
    "require_cross_arm_equal_skip_count_within_seed": True,
    "require_cross_arm_equal_successful_update_count_within_seed": True,
    "allow_objective_dependent_skip_locations": True,
}

VALIDATION_FILENAME = "q256_target_weight_arm_validation_v2.json"
HASH_RECEIPT_FILENAME = "q256_target_weight_arm_artifact_hashes_v2.json"

ARMS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}

MODES = {
    "smoke": {"attempts": 32, "processed_nimg": 4096, "total_kimg": 4},
    "formal": {"attempts": 2000, "processed_nimg": 256000, "total_kimg": 256},
}

DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
TRANSFER_SOURCE_POLICY = {
    "schema": "ect.q256.authoritative-transfer-source-policy/v1",
    "required_target_coverage": "all_parameters_and_buffers",
    "allowed_source_extras": {
        "model.map_augment.weight": {
            "shape": [128, 9],
            "dtype": "torch.float32",
            "tensor_bytes_sha256": (
                "4500f8ac1eb5cc8dd4096595a798c8ea4793d42f8433014ab67e41d5ceb70de0"
            ),
            "reason": (
                "authoritative checkpoint augmentation map unused by augment=0 target"
            ),
        }
    },
}

# This tuple deliberately duplicates the writer's public v1 contract. A
# missing, reordered, or additional column is a version mismatch, not a
# best-effort compatibility case.
TELEMETRY_FIELDS = (
    "schema",
    "protocol",
    "arm",
    "target_gap_scale",
    "denominator_gap_scale",
    "attempted_iteration",
    "successful_optimizer_steps",
    "processed_nimg",
    "processed_kimg",
    "stage",
    "loss",
    "loss_nonfinite_count",
    "raw_grad_norm",
    "raw_grad_finite_norm",
    "raw_grad_nonfinite_count",
    "sanitized_grad_norm",
    "sanitized_grad_nonfinite_count",
    "update_norm",
    "update_nonfinite_count",
    "model_norm",
    "model_nonfinite_count",
    "ema_norm",
    "ema_nonfinite_count",
    "sample_count",
    "batch_sha256",
    "t_sha256",
    "base_r_sha256",
    "target_r_sha256",
    "denominator_r_sha256",
    "target_delta_sha256",
    "denominator_delta_sha256",
    "base_r_zero_count",
    "target_r_zero_count",
    "target_r_equal_t_count",
    "target_scaled_to_zero_count",
    "denominator_r_zero_count",
    "denominator_r_equal_t_count",
    "denominator_scaled_to_zero_count",
    "target_delta_min",
    "target_delta_max",
    "target_delta_mean",
    "denominator_delta_min",
    "denominator_delta_max",
    "denominator_delta_mean",
    "factor_nonfinite_count",
    "nonpositive_denominator_count",
    "learning_rate",
    "grad_scale_before",
    "grad_scale_after",
    "step_skipped",
    "elapsed_sec",
    "gpu_hours_cumulative",
)

CORE_ARTIFACTS = (
    "launch_manifest.json",
    "training_options.json",
    "initial_state_receipt_v1.json",
    "factorial_training_telemetry_v1.csv",
    "train_summary.csv",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "log.txt",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_UINT_RE = re.compile(r"^(0|[1-9][0-9]*)$")


class VerificationError(RuntimeError):
    """The run is incomplete, inconsistent, or outside the frozen protocol."""


def fail(message: str) -> None:
    raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object: {path}")
    return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object/dict")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a list")
    return value


def require_exact_keys(value: dict[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        fail(f"{label} keys mismatch: missing={missing}, extra={extra}")


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        fail(f"{label} must be an integer, got bool")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and _UINT_RE.fullmatch(value):
        result = int(value)
    else:
        fail(f"{label} must be a canonical non-negative integer, got {value!r}")
    if minimum is not None and result < minimum:
        fail(f"{label} must be >= {minimum}, got {result}")
    return result


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        fail(f"{label} must be a finite number, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        fail(f"{label} must be a finite number: {exc}")
    if not math.isfinite(result):
        fail(f"{label} must be finite, got {result!r}")
    return result


def exact_number(value: Any, expected: float, label: str) -> None:
    actual = finite_float(value, label)
    if actual != float(expected):
        fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def exact_value(value: Any, expected: Any, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        fail(f"{label} mismatch: expected {expected!r}, got {value!r}")


def expected_factorial(arm: str) -> dict[str, Any]:
    target, denominator = ARMS[arm]
    return {
        "enabled": True,
        "protocol": PROTOCOL,
        "arm": arm,
        "target_gap_scale": target,
        "denominator_gap_scale": denominator,
    }


def validate_factorial(value: Any, arm: str, label: str) -> dict[str, Any]:
    factorial = require_dict(value, label)
    expected = expected_factorial(arm)
    require_exact_keys(factorial, expected, label)
    exact_value(factorial["enabled"], True, f"{label}.enabled")
    exact_value(factorial["protocol"], PROTOCOL, f"{label}.protocol")
    exact_value(factorial["arm"], arm, f"{label}.arm")
    exact_number(
        factorial["target_gap_scale"], expected["target_gap_scale"],
        f"{label}.target_gap_scale",
    )
    exact_number(
        factorial["denominator_gap_scale"],
        expected["denominator_gap_scale"],
        f"{label}.denominator_gap_scale",
    )
    return factorial


def validate_training_options(
    options: dict[str, Any], arm: str, seed: int, mode: str
) -> dict[str, Any]:
    expected = MODES[mode]
    loss = require_dict(options.get("loss_kwargs"), "training_options.loss_kwargs")
    target_scale, denominator_scale = ARMS[arm]

    exact_value(options.get("seed"), seed, "training_options.seed")
    exact_value(
        options.get("total_kimg"), expected["total_kimg"],
        "training_options.total_kimg",
    )
    exact_value(options.get("batch_size"), 128, "training_options.batch_size")
    exact_value(options.get("batch_gpu"), 16, "training_options.batch_gpu")
    exact_value(options.get("enable_amp"), True, "training_options.enable_amp")
    exact_value(options.get("enable_tf32"), False, "training_options.enable_tf32")
    exact_number(options.get("loss_scaling"), 1.0, "training_options.loss_scaling")
    exact_number(options.get("ema_beta"), 0.9993, "training_options.ema_beta")
    exact_number(options.get("kimg_per_tick"), 10, "training_options.kimg_per_tick")
    exact_value(options.get("snapshot_ticks"), None, "training_options.snapshot_ticks")
    exact_value(options.get("state_dump_ticks"), None, "training_options.state_dump_ticks")
    exact_value(options.get("ckpt_ticks"), 10, "training_options.ckpt_ticks")
    exact_value(options.get("sample_ticks"), 26, "training_options.sample_ticks")
    exact_value(options.get("double_ticks"), 10000, "training_options.double_ticks")
    exact_value(options.get("metrics"), [], "training_options.metrics")
    if options.get("resume_state_dump") is not None:
        fail("formal/smoke arm must be fresh, not a resumed training_options run")
    if not isinstance(options.get("resume_pkl"), str) or not options["resume_pkl"]:
        fail("training_options.resume_pkl must bind the fresh transfer source")

    exact_value(loss.get("class_name"), "training.loss.ECMLoss", "loss.class_name")
    exact_value(loss.get("factorial_protocol"), PROTOCOL, "loss.factorial_protocol")
    exact_number(loss.get("target_gap_scale"), target_scale, "loss.target_gap_scale")
    exact_number(
        loss.get("denominator_gap_scale"), denominator_scale,
        "loss.denominator_gap_scale",
    )
    exact_value(loss.get("adj"), "sigmoid", "loss.adj")
    exact_number(loss.get("global_gap_scale"), 1.0, "loss.global_gap_scale")
    exact_number(loss.get("q"), 256.0, "loss.q")
    exact_number(loss.get("c"), 0.0, "loss.c")
    exact_number(loss.get("k"), 8.0, "loss.k")
    exact_number(loss.get("b"), 1.0, "loss.b")

    optimizer = require_dict(
        options.get("optimizer_kwargs"), "training_options.optimizer_kwargs"
    )
    exact_value(
        optimizer.get("class_name"), "torch.optim.RAdam", "optimizer.class_name"
    )
    exact_number(optimizer.get("lr"), 1e-4, "optimizer.lr")
    if optimizer.get("betas") != [0.9, 0.999]:
        fail(f"optimizer.betas mismatch: {optimizer.get('betas')!r}")
    exact_number(optimizer.get("eps"), 1e-8, "optimizer.eps")
    exact_number(optimizer.get("weight_decay", 0.0), 0.0, "optimizer.weight_decay")

    network = require_dict(options.get("network_kwargs"), "training_options.network_kwargs")
    exact_value(
        network.get("class_name"), "training.networks.ECMPrecond",
        "network.class_name",
    )
    exact_value(network.get("use_fp16"), True, "network.use_fp16")
    exact_number(network.get("dropout"), 0.2, "network.dropout")
    if options.get("augment_kwargs") is not None:
        fail("training_options.augment_kwargs must be null for augmentation=0")

    dataset = require_dict(options.get("dataset_kwargs"), "training_options.dataset_kwargs")
    exact_value(dataset.get("use_labels"), False, "dataset.use_labels")
    exact_value(dataset.get("xflip"), False, "dataset.xflip")
    exact_value(dataset.get("resolution"), 32, "dataset.resolution")
    exact_value(dataset.get("max_size"), 50000, "dataset.max_size")
    if not isinstance(dataset.get("path"), str) or not dataset["path"]:
        fail("dataset.path must be a non-empty string")

    return {
        "dataset_path": dataset["path"],
        "transfer_path": options["resume_pkl"],
        "batch_size": options["batch_size"],
        "batch_gpu": options["batch_gpu"],
        "factorial": expected_factorial(arm),
    }


def expected_launch_training_contract(mode: str, arm: str, seed: int) -> dict[str, Any]:
    target, denominator = ARMS[arm]
    phase = MODES[mode]
    return {
        "phase": mode,
        "arm": arm,
        "seed": seed,
        "target_gap_scale": str(target),
        "denominator_gap_scale": str(denominator),
        "factorial_protocol": PROTOCOL,
        "mapping": "sigmoid",
        "global_gap_scale": "1.0",
        "duration_mimg": "0.004096" if mode == "smoke" else "0.256",
        "requested_kimg": "4.096" if mode == "smoke" else "256",
        "ct_train_total_kimg": phase["total_kimg"],
        "expected_processed_nimg": phase["processed_nimg"],
        "expected_optimizer_attempts": phase["attempts"],
        "q": 256,
        "k": 8,
        "b": 1,
        "c": 0,
        "batch": 128,
        "batch_gpu": 16,
        "optimizer": {
            "name": "RAdam",
            "lr": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "dropout": 0.2,
        "augmentation": 0,
        "xflip": False,
        "fp16": True,
        "amp": True,
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "ema_beta": 0.9993,
        "tick_kimg": 10,
        "numbered_snapshots": False,
        "numbered_state_dumps": False,
        "latest_checkpoint_ticks": 10,
        "preview_ticks": 26,
        "training_metrics": [],
    }


def resolve_bound_run_artifact(run_dir: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        fail(f"{label} path must be a non-empty relative string")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        fail(f"{label} path must be relative to the run directory")
    joined = run_dir / candidate
    if joined.is_symlink():
        fail(f"{label} must not be a symlink: {joined}")
    resolved = joined.resolve(strict=False)
    try:
        resolved.relative_to(run_dir)
    except ValueError:
        fail(f"{label} escapes the run directory: {raw_path!r}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        fail(f"{label} is missing, empty, or a symlink: {resolved}")
    return resolved


def validate_launch_manifest(
    manifest: dict[str, Any], run_dir: Path, arm: str, seed: int, mode: str,
    options_info: dict[str, Any],
) -> dict[str, Any]:
    exact = {
        "schema": LAUNCH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "launch_kind": "fresh_transfer",
        "status": "authorized_to_start",
        "original_launch_manifest_sha256": None,
    }
    for field, expected in exact.items():
        exact_value(manifest.get(field), expected, f"launch manifest.{field}")
    try:
        recorded_run_dir = Path(manifest.get("run_directory", "")).resolve()
    except (OSError, TypeError) as exc:
        fail(f"launch manifest run_directory is invalid: {exc}")
    if recorded_run_dir != run_dir:
        fail(
            "launch manifest run_directory mismatch: "
            f"expected {run_dir}, got {recorded_run_dir}"
        )
    expected_training = expected_launch_training_contract(mode, arm, seed)
    if manifest.get("training") != expected_training:
        fail(
            "launch manifest training contract mismatch: "
            f"expected={expected_training!r}, actual={manifest.get('training')!r}"
        )

    source = require_dict(manifest.get("source"), "launch manifest.source")
    exact_value(source.get("git_clean"), True, "launch manifest.source.git_clean")
    exact_value(
        source.get("git_branch"), "experiment/q256-target-weight-factorial",
        "launch manifest.source.git_branch",
    )
    require_git_oid(source.get("git_head"), "launch manifest.source.git_head")
    require_sha256(
        source.get("content_sha256"), "launch manifest.source.content_sha256"
    )
    preregistration = require_dict(
        manifest.get("preregistration"), "launch manifest.preregistration"
    )
    require_sha256(
        preregistration.get("sha256"), "launch manifest.preregistration.sha256"
    )

    assets = require_dict(manifest.get("assets"), "launch manifest.assets")
    bound_assets = (
        ("dataset", DATASET_SHA256, options_info["dataset_path"]),
        ("transfer", TRANSFER_SHA256, options_info["transfer_path"]),
    )
    for name, expected_sha, expected_path in bound_assets:
        asset = require_dict(assets.get(name), f"launch manifest.assets.{name}")
        exact_value(asset.get("sha256"), expected_sha, f"launch asset {name} SHA256")
        exact_value(
            asset.get("resolved_path"), expected_path,
            f"launch asset {name} resolved_path",
        )
        if strict_int(asset.get("size_bytes"), f"launch asset {name} size", minimum=1) < 1:
            fail(f"launch asset {name} is empty")

    runtime = require_dict(manifest.get("runtime"), "launch manifest.runtime")
    exact_value(runtime.get("cuda_available"), True, "launch runtime.cuda_available")
    exact_value(runtime.get("cuda_device_count"), 1, "launch runtime.cuda_device_count")
    gpu = require_dict(manifest.get("gpu"), "launch manifest.gpu")
    if "A100" not in str(gpu.get("name", "")):
        fail(f"launch GPU is not an A100: {gpu.get('name')!r}")
    if strict_int(gpu.get("memory_total_mib"), "launch GPU memory", minimum=80000) < 80000:
        fail("launch GPU does not provide the frozen 80GB capacity")
    environment = require_dict(
        manifest.get("process_environment"), "launch manifest.process_environment"
    )
    exact_value(environment.get("WORLD_SIZE"), "1", "launch environment.WORLD_SIZE")
    exact_value(environment.get("RANK"), "0", "launch environment.RANK")
    exact_value(environment.get("LOCAL_RANK"), "0", "launch environment.LOCAL_RANK")

    authorization = require_dict(
        manifest.get("authorization"), "launch manifest.authorization"
    )
    receipt_path = resolve_bound_run_artifact(
        run_dir, authorization.get("receipt_path"), "authorization receipt"
    )
    recorded_receipt_sha = require_sha256(
        authorization.get("receipt_sha256"), "authorization receipt SHA256"
    )
    if sha256_file(receipt_path) != recorded_receipt_sha:
        fail("run-contained authorization receipt hash mismatch")
    authorization_payload = load_json(receipt_path, "authorization receipt")
    exact_value(
        authorization_payload.get("schema"),
        AUTHORIZATION_SCHEMA,
        "authorization receipt schema",
    )
    exact_value(
        authorization_payload.get("amp_skip_policy"),
        AMP_SKIP_POLICY,
        "authorization AMP skip policy",
    )
    verifier_contract = require_dict(
        manifest.get("post_training_verifier"),
        "launch manifest.post_training_verifier",
    )
    if "expected_amp_skip_attempts" not in authorization_payload:
        fail(
            "authorization must explicitly record expected_amp_skip_attempts, "
            "including null in observe mode"
        )
    if "expected_skip_attempts" not in verifier_contract:
        fail(
            "launch manifest must explicitly record expected_skip_attempts, "
            "including null in observe mode"
        )
    authorized_skip_attempts = authorization_payload.get(
        "expected_amp_skip_attempts"
    )
    if authorized_skip_attempts is not None and (
        not isinstance(authorized_skip_attempts, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in authorized_skip_attempts
        )
        or authorized_skip_attempts != sorted(set(authorized_skip_attempts))
    ):
        fail("authorization has a malformed expected AMP skip signature")
    exact_value(
        verifier_contract.get("expected_skip_attempts"),
        authorized_skip_attempts,
        "launch/authorization expected AMP skip signature",
    )
    gate_records = require_list(
        authorization.get("gate_receipts"), "launch manifest authorization gates"
    )
    if not gate_records:
        fail("launch manifest contains no bound PASS gate receipts")
    bound_paths = {str(receipt_path.relative_to(run_dir)): receipt_path}
    seen_gate_names: set[str] = set()
    for index, value in enumerate(gate_records):
        gate = require_dict(value, f"launch authorization gate {index}")
        name = gate.get("name")
        if not isinstance(name, str) or not name or name in seen_gate_names:
            fail(f"launch authorization gate {index} has invalid/duplicate name")
        seen_gate_names.add(name)
        path = resolve_bound_run_artifact(
            run_dir, gate.get("path"), f"authorization gate {name!r}"
        )
        recorded_sha = require_sha256(
            gate.get("sha256"), f"authorization gate {name!r} SHA256"
        )
        if sha256_file(path) != recorded_sha:
            fail(f"authorization gate {name!r} hash mismatch")
        relative = str(path.relative_to(run_dir))
        if relative in bound_paths:
            fail(f"authorization artifact is bound more than once: {relative}")
        bound_paths[relative] = path
    return {
        "source_git_head": source["git_head"],
        "source_content_sha256": source["content_sha256"],
        "bound_artifacts": bound_paths,
        "expected_skip_attempts": authorized_skip_attempts,
    }


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        fail(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _GIT_OID_RE.fullmatch(value):
        fail(f"{label} must be a lowercase 40-character Git object id, got {value!r}")
    return value


def validate_sampler_state(
    value: Any, *, seed: int, consumed_samples: int, dataset_size: int = 50000
) -> dict[str, Any]:
    label = "sampler_state"
    sampler = require_dict(value, label)
    expected = {
        "schema": "ect.infinite-sampler/v1",
        "dataset_size": dataset_size,
        "rank": 0,
        "num_replicas": 1,
        "shuffle": True,
        "seed": seed,
        "window_size": 0.5,
        "consumed_samples": consumed_samples,
    }
    require_exact_keys(sampler, expected, label)
    for field, expected_value in expected.items():
        exact_value(sampler[field], expected_value, f"{label}.{field}")
    return sampler


def validate_initial_receipt(
    receipt: dict[str, Any], arm: str, seed: int, options_info: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema", "seed", "attempted_iteration", "processed_nimg",
        "factorial", "dataset_path", "transfer_path", "world_size",
        "batch_size", "batch_gpu", "trajectory_config",
        "trajectory_config_sha256", "hashes",
        "common_initial_state_sha256", "rank_states",
    }
    require_exact_keys(receipt, required, "initial receipt")
    exact_value(
        receipt["schema"], reproducibility.INITIAL_RECEIPT_SCHEMA,
        "initial receipt.schema",
    )
    exact_value(receipt["seed"], seed, "initial receipt.seed")
    exact_value(receipt["attempted_iteration"], 0, "initial receipt.attempted_iteration")
    exact_value(receipt["processed_nimg"], 0, "initial receipt.processed_nimg")
    validate_factorial(receipt["factorial"], arm, "initial receipt.factorial")
    exact_value(
        receipt["dataset_path"], options_info["dataset_path"],
        "initial receipt.dataset_path",
    )
    exact_value(
        receipt["transfer_path"], options_info["transfer_path"],
        "initial receipt.transfer_path",
    )
    exact_value(receipt["world_size"], 1, "initial receipt.world_size")
    exact_value(receipt["batch_size"], 128, "initial receipt.batch_size")
    exact_value(receipt["batch_gpu"], 16, "initial receipt.batch_gpu")
    trajectory = require_dict(
        receipt["trajectory_config"], "initial receipt.trajectory_config"
    )
    exact_value(
        trajectory.get("schema"), reproducibility.TRAJECTORY_CONFIG_SCHEMA,
        "initial receipt.trajectory_config.schema",
    )
    trajectory_sha256 = require_sha256(
        receipt["trajectory_config_sha256"],
        "initial receipt.trajectory_config_sha256",
    )
    if reproducibility.state_sha256(trajectory) != trajectory_sha256:
        fail("initial receipt trajectory config hash mismatch")
    exact_value(trajectory.get("seed"), seed, "trajectory_config.seed")
    exact_value(
        trajectory.get("batch_size"), 128, "trajectory_config.batch_size"
    )
    exact_value(
        trajectory.get("batch_gpu"), 16, "trajectory_config.batch_gpu"
    )
    exact_value(
        trajectory.get("authoritative_transfer_source_policy"),
        TRANSFER_SOURCE_POLICY,
        "trajectory_config.authoritative_transfer_source_policy",
    )
    loss_kwargs = require_dict(
        trajectory.get("loss_kwargs"), "trajectory_config.loss_kwargs"
    )
    validate_factorial(
        {
            "enabled": True,
            "protocol": loss_kwargs.get("factorial_protocol"),
            "arm": arm,
            "target_gap_scale": loss_kwargs.get("target_gap_scale"),
            "denominator_gap_scale": loss_kwargs.get(
                "denominator_gap_scale"
            ),
        },
        arm,
        "trajectory_config.loss_kwargs factorial",
    )

    ranks = require_list(receipt["rank_states"], "initial receipt.rank_states")
    if len(ranks) != 1:
        fail(f"initial receipt must contain exactly one rank state, got {len(ranks)}")
    rank = require_dict(ranks[0], "initial receipt.rank_states[0]")
    require_exact_keys(
        rank,
        ("rank", "world_size", "rng_sha256", "sampler_sha256", "sampler_state"),
        "initial receipt.rank_states[0]",
    )
    exact_value(rank["rank"], 0, "initial receipt rank")
    exact_value(rank["world_size"], 1, "initial receipt rank world_size")
    rng_sha = require_sha256(rank["rng_sha256"], "initial receipt rng_sha256")
    sampler = validate_sampler_state(rank["sampler_state"], seed=seed, consumed_samples=0)
    sampler_sha = require_sha256(
        rank["sampler_sha256"], "initial receipt sampler_sha256"
    )
    actual_sampler_sha = reproducibility.state_sha256(sampler)
    if sampler_sha != actual_sampler_sha:
        fail(
            "initial receipt sampler hash mismatch: "
            f"recorded={sampler_sha}, actual={actual_sampler_sha}"
        )

    hashes = require_dict(receipt["hashes"], "initial receipt.hashes")
    require_exact_keys(
        hashes,
        ("model", "ema", "optimizer", "gradscaler", "rank_rng", "rank_sampler"),
        "initial receipt.hashes",
    )
    for name in ("model", "ema", "optimizer", "gradscaler"):
        require_sha256(hashes[name], f"initial receipt.hashes.{name}")
    if hashes["model"] != hashes["ema"]:
        fail("initial model and EMA hashes differ before optimizer attempt 1")
    if hashes["rank_rng"] != [rng_sha]:
        fail("initial receipt rank_rng hash list does not bind rank_states")
    if hashes["rank_sampler"] != [sampler_sha]:
        fail("initial receipt rank_sampler hash list does not bind rank_states")
    common = require_sha256(
        receipt["common_initial_state_sha256"],
        "initial receipt.common_initial_state_sha256",
    )
    actual_common = reproducibility.state_sha256(hashes)
    if common != actual_common:
        fail(
            "initial common-state hash mismatch: "
            f"recorded={common}, actual={actual_common}"
        )
    return {
        "common_initial_state_sha256": common,
        "trajectory_config_sha256": trajectory_sha256,
    }


def read_telemetry(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        fail(f"cannot read telemetry {path}: {exc}")
    if fieldnames != TELEMETRY_FIELDS:
        fail(
            "factorial telemetry header is not exact v1: "
            f"expected={TELEMETRY_FIELDS!r}, actual={fieldnames!r}"
        )
    if not rows:
        fail("factorial telemetry contains no optimizer attempts")
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            fail(f"telemetry row {row_number} has fields beyond the exact v1 header")
    return rows


def validate_telemetry(
    rows: list[dict[str, str]], arm: str, mode: str,
    expected_skip_attempts: list[int] | None,
) -> dict[str, Any]:
    expected = MODES[mode]
    if len(rows) != expected["attempts"]:
        fail(
            f"telemetry attempt count mismatch for {mode}: "
            f"expected {expected['attempts']}, got {len(rows)}"
        )
    target_scale, denominator_scale = ARMS[arm]
    skips: list[int] = []
    previous_elapsed = -1.0
    cumulative_skips = 0
    count_fields = (
        "loss_nonfinite_count", "raw_grad_nonfinite_count",
        "sanitized_grad_nonfinite_count", "update_nonfinite_count",
        "model_nonfinite_count", "ema_nonfinite_count", "sample_count",
        "base_r_zero_count", "target_r_zero_count",
        "target_r_equal_t_count", "target_scaled_to_zero_count",
        "denominator_r_zero_count", "denominator_r_equal_t_count",
        "denominator_scaled_to_zero_count", "factor_nonfinite_count",
        "nonpositive_denominator_count",
    )
    always_finite_fields = (
        "loss", "raw_grad_finite_norm", "sanitized_grad_norm",
        "update_norm", "model_norm", "ema_norm", "target_delta_min",
        "target_delta_max", "target_delta_mean", "denominator_delta_min",
        "denominator_delta_max", "denominator_delta_mean",
        "learning_rate", "grad_scale_before", "grad_scale_after",
        "elapsed_sec", "gpu_hours_cumulative",
    )

    for attempt, row in enumerate(rows, start=1):
        label = f"telemetry row {attempt + 1}"
        exact_value(row["schema"], TELEMETRY_SCHEMA, f"{label}.schema")
        exact_value(row["protocol"], PROTOCOL, f"{label}.protocol")
        exact_value(row["arm"], arm, f"{label}.arm")
        exact_number(row["target_gap_scale"], target_scale, f"{label}.target_gap_scale")
        exact_number(
            row["denominator_gap_scale"], denominator_scale,
            f"{label}.denominator_gap_scale",
        )
        exact_value(
            strict_int(row["attempted_iteration"], f"{label}.attempted_iteration"),
            attempt,
            f"{label}.attempted_iteration",
        )
        processed_nimg = strict_int(row["processed_nimg"], f"{label}.processed_nimg")
        exact_value(processed_nimg, attempt * 128, f"{label}.processed_nimg")
        processed_kimg = finite_float(row["processed_kimg"], f"{label}.processed_kimg")
        if processed_kimg != processed_nimg / 1000:
            fail(f"{label}.processed_kimg does not match processed_nimg")
        exact_value(strict_int(row["stage"], f"{label}.stage"), 0, f"{label}.stage")

        counts = {
            name: strict_int(row[name], f"{label}.{name}") for name in count_fields
        }
        digest_fields = (
            "batch_sha256", "t_sha256", "base_r_sha256",
            "target_r_sha256", "denominator_r_sha256",
            "target_delta_sha256", "denominator_delta_sha256",
        )
        for name in digest_fields:
            require_sha256(row[name], f"{label}.{name}")
        if target_scale == denominator_scale:
            if row["target_r_sha256"] != row["denominator_r_sha256"]:
                fail(f"{label} native-arm target/denominator time digests differ")
            if row["target_delta_sha256"] != row["denominator_delta_sha256"]:
                fail(f"{label} native-arm target/denominator gap digests differ")
        values = {
            name: finite_float(row[name], f"{label}.{name}")
            for name in always_finite_fields
        }
        if values["loss"] < 0:
            fail(f"{label}.loss must be non-negative")
        step_skipped = strict_int(row["step_skipped"], f"{label}.step_skipped")
        if step_skipped not in (0, 1):
            fail(f"{label}.step_skipped must be 0 or 1")
        if step_skipped:
            skips.append(attempt)
            cumulative_skips += 1
            if processed_nimg >= AMP_SKIP_WARMUP_PROCESSED_NIMG:
                fail(
                    f"{label} AMP skip occurred after the frozen tick-0 "
                    "warm-up region"
                )
        expected_successes = attempt - cumulative_skips
        successes = strict_int(
            row["successful_optimizer_steps"],
            f"{label}.successful_optimizer_steps",
        )
        if successes != expected_successes:
            fail(
                f"{label}.successful_optimizer_steps mismatch: "
                f"expected {expected_successes}, got {successes}"
            )

        if counts["sample_count"] != 128:
            fail(f"{label}.sample_count must equal the exact batch size 128")
        for name in (
            "base_r_zero_count", "target_r_zero_count",
            "target_scaled_to_zero_count", "denominator_r_zero_count",
            "denominator_scaled_to_zero_count",
        ):
            if counts[name] > counts["sample_count"]:
                fail(f"{label}.{name} exceeds sample_count")
        if counts["target_r_equal_t_count"] != 0:
            fail(f"{label} records a non-positive target gap")
        if counts["denominator_r_equal_t_count"] != 0:
            fail(f"{label} records a non-positive denominator gap")
        for name in (
            "loss_nonfinite_count", "sanitized_grad_nonfinite_count",
            "update_nonfinite_count", "model_nonfinite_count",
            "ema_nonfinite_count", "factor_nonfinite_count",
            "nonpositive_denominator_count",
        ):
            if counts[name] != 0:
                fail(f"{label}.{name} must be zero, got {counts[name]}")

        raw_nonfinite = counts["raw_grad_nonfinite_count"]
        if bool(raw_nonfinite) != bool(step_skipped):
            fail(f"{label} raw-gradient non-finite status does not match AMP skip")
        try:
            raw_norm = float(row["raw_grad_norm"])
        except (TypeError, ValueError, OverflowError) as exc:
            fail(f"{label}.raw_grad_norm is invalid: {exc}")
        if step_skipped:
            if raw_norm != float("inf"):
                fail(f"{label}.raw_grad_norm must be +inf on an AMP skip")
            if values["grad_scale_after"] >= values["grad_scale_before"]:
                fail(f"{label} AMP skip did not reduce GradScaler scale")
            if values["update_norm"] != 0.0:
                fail(f"{label} skipped optimizer attempt changed parameters")
        else:
            if not math.isfinite(raw_norm) or raw_norm < 0:
                fail(f"{label}.raw_grad_norm must be finite and non-negative")
            if values["grad_scale_after"] < values["grad_scale_before"]:
                fail(f"{label} GradScaler scale fell without a recorded skip")
            if values["update_norm"] <= 0:
                fail(f"{label} successful optimizer update norm is not positive")

        for name in (
            "raw_grad_finite_norm", "sanitized_grad_norm", "model_norm",
            "ema_norm",
        ):
            if values[name] < 0:
                fail(f"{label}.{name} must be non-negative")
        for prefix in ("target_delta", "denominator_delta"):
            minimum = values[f"{prefix}_min"]
            maximum = values[f"{prefix}_max"]
            mean = values[f"{prefix}_mean"]
            if minimum <= 0 or not minimum <= mean <= maximum:
                fail(
                    f"{label}.{prefix} must have 0 < min <= mean <= max, "
                    f"got {(minimum, mean, maximum)!r}"
                )
        exact_number(values["learning_rate"], 1e-4, f"{label}.learning_rate")
        if values["grad_scale_before"] <= 0 or values["grad_scale_after"] <= 0:
            fail(f"{label} GradScaler scales must be positive")
        elapsed = values["elapsed_sec"]
        if elapsed < previous_elapsed:
            fail(f"{label}.elapsed_sec moved backwards")
        previous_elapsed = elapsed
        if not math.isclose(
            values["gpu_hours_cumulative"], elapsed / 3600,
            rel_tol=0.0, abs_tol=1e-8,
        ):
            fail(f"{label}.gpu_hours_cumulative does not match elapsed_sec")

    if expected_skip_attempts is not None and skips != expected_skip_attempts:
        fail(
            "AMP skip-attempt signature mismatch: "
            f"expected {expected_skip_attempts}, got {skips}"
        )
    return {
        "attempts": len(rows),
        "successful_optimizer_steps": len(rows) - len(skips),
        "processed_nimg": expected["processed_nimg"],
        "amp_skip_attempts": skips,
        "elapsed_sec": previous_elapsed,
    }


def validate_rng_state(value: Any) -> None:
    state = require_dict(value, "training-state RNG state")
    keys = (
        "schema", "python", "numpy", "torch_cpu", "torch_cuda_all",
        "torch_cuda_device_count",
    )
    require_exact_keys(state, keys, "training-state RNG state")
    exact_value(
        state["schema"], reproducibility.RNG_STATE_SCHEMA,
        "training-state RNG schema",
    )
    if not isinstance(state["python"], tuple) or len(state["python"]) != 3:
        fail("training-state Python RNG payload is malformed")
    if not isinstance(state["numpy"], tuple) or len(state["numpy"]) != 5:
        fail("training-state NumPy RNG payload is malformed")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        fail("training-state CPU Torch RNG payload is not a tensor")
    cuda_count = strict_int(
        state["torch_cuda_device_count"], "training-state CUDA RNG device count"
    )
    cuda_states = require_list(state["torch_cuda_all"], "training-state CUDA RNG states")
    if len(cuda_states) != cuda_count:
        fail("training-state CUDA RNG count does not match its state list")
    if any(not isinstance(item, torch.Tensor) for item in cuda_states):
        fail("training-state CUDA RNG list contains a non-tensor")


def validate_finite_tree(value: Any, label: str) -> int:
    stack = [value]
    tensors = 0
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            tensors += 1
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all().item()
            ):
                fail(f"{label} contains a non-finite tensor")
        elif isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.inexact) and not bool(np.isfinite(item).all()):
                fail(f"{label} contains a non-finite NumPy array")
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            fail(f"{label} contains a non-finite scalar")
    return tensors


def validate_module(module: Any, label: str) -> int:
    if not isinstance(module, torch.nn.Module):
        fail(f"{label} must be a torch.nn.Module")
    named = list(module.named_parameters()) + list(module.named_buffers())
    if not named:
        fail(f"{label} has no parameters or buffers")
    checked = 0
    for name, tensor in named:
        checked += 1
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            fail(f"{label} contains non-finite tensor {name!r}")
    return checked


def torch_load_trusted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases predating the weights_only argument.
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - version-specific branch.
            fail(f"cannot load trusted training state {path}: {exc}")
    except Exception as exc:
        fail(f"cannot load trusted training state {path}: {exc}")


def validate_training_state(
    path: Path, arm: str, seed: int, mode: str, telemetry: dict[str, Any]
) -> dict[str, Any]:
    state = torch_load_trusted(path)
    state = require_dict(state, "training state")
    required = (
        "reproducibility_schema", "net", "ema", "optimizer_state",
        "gradscaler_state", "rank_states", "factorial",
        "attempted_iteration", "successful_optimizer_steps", "cur_nimg",
        "cur_tick", "tick_start_nimg", "elapsed_sec", "loss_fn_state",
        "trajectory_config", "trajectory_config_sha256",
        "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size",
    )
    missing = [name for name in required if name not in state]
    if missing:
        fail(f"self-contained training state is missing: {missing}")
    exact_value(
        state["reproducibility_schema"], reproducibility.TRAINING_STATE_SCHEMA,
        "training-state reproducibility_schema",
    )
    validate_factorial(state["factorial"], arm, "training-state.factorial")
    trajectory = require_dict(
        state["trajectory_config"], "training-state trajectory_config"
    )
    exact_value(
        trajectory.get("schema"), reproducibility.TRAJECTORY_CONFIG_SCHEMA,
        "training-state trajectory_config.schema",
    )
    trajectory_sha256 = require_sha256(
        state["trajectory_config_sha256"],
        "training-state trajectory_config_sha256",
    )
    if reproducibility.state_sha256(trajectory) != trajectory_sha256:
        fail("training-state trajectory config hash mismatch")
    expected = MODES[mode]
    exact_value(
        state["attempted_iteration"], expected["attempts"],
        "training-state.attempted_iteration",
    )
    exact_value(
        state["successful_optimizer_steps"],
        telemetry["successful_optimizer_steps"],
        "training-state.successful_optimizer_steps",
    )
    exact_value(
        state["cur_nimg"], expected["processed_nimg"],
        "training-state.cur_nimg",
    )
    exact_value(
        state["tick_start_nimg"], expected["processed_nimg"],
        "training-state.tick_start_nimg",
    )
    if strict_int(state["cur_tick"], "training-state.cur_tick", minimum=1) < 1:
        fail("training-state.cur_tick must be positive")
    elapsed = finite_float(state["elapsed_sec"], "training-state.elapsed_sec")
    if not math.isclose(elapsed, telemetry["elapsed_sec"], rel_tol=0.0, abs_tol=5e-6):
        fail("training-state elapsed_sec does not match final telemetry row")

    net_tensors = validate_module(state["net"], "training-state net")
    ema_tensors = validate_module(state["ema"], "training-state EMA")
    optimizer = require_dict(state["optimizer_state"], "training-state optimizer")
    if not {"state", "param_groups"}.issubset(optimizer):
        fail("training-state optimizer lacks state/param_groups")
    optimizer_tensors = validate_finite_tree(optimizer, "training-state optimizer")
    scaler = require_dict(state["gradscaler_state"], "training-state GradScaler")
    if not scaler:
        fail("training-state GradScaler state is empty")
    scaler_tensors = validate_finite_tree(scaler, "training-state GradScaler")

    ranks = require_list(state["rank_states"], "training-state rank_states")
    if len(ranks) != 1:
        fail(f"training state must contain exactly one rank, got {len(ranks)}")
    rank = require_dict(ranks[0], "training-state rank 0")
    require_exact_keys(
        rank, ("rank", "world_size", "rng_state", "sampler_state"),
        "training-state rank 0",
    )
    exact_value(rank["rank"], 0, "training-state rank")
    exact_value(rank["world_size"], 1, "training-state world_size")
    validate_rng_state(rank["rng_state"])
    validate_sampler_state(
        rank["sampler_state"], seed=seed,
        consumed_samples=expected["processed_nimg"],
    )
    validate_finite_tree(rank["rng_state"], "training-state RNG")

    loss_state = require_dict(state["loss_fn_state"], "training-state loss_fn_state")
    exact_value(loss_state.get("schedule_name"), "sigmoid", "loss_fn_state.schedule_name")
    exact_value(loss_state.get("stage"), 0, "loss_fn_state.stage")
    exact_number(loss_state.get("ratio"), 255 / 256, "loss_fn_state.ratio")
    exact_value(loss_state.get("schedule"), {}, "loss_fn_state.schedule")

    grid_z = require_list(state["snapshot_grid_z"], "training-state snapshot_grid_z")
    grid_c = require_list(state["snapshot_grid_c"], "training-state snapshot_grid_c")
    if not grid_z or len(grid_z) != len(grid_c):
        fail("training-state preview tensors are empty or unpaired")
    if any(not isinstance(value, torch.Tensor) for value in grid_z + grid_c):
        fail("training-state preview state contains a non-tensor")
    validate_finite_tree(grid_z + grid_c, "training-state preview state")
    grid_size = state["snapshot_grid_size"]
    if not (
        isinstance(grid_size, tuple) and len(grid_size) == 2
        and all(isinstance(value, (int, np.integer)) and int(value) > 0 for value in grid_size)
    ):
        fail("training-state snapshot_grid_size is invalid")

    return {
        "state": state,
        "net_tensors_checked": net_tensors,
        "ema_tensors_checked": ema_tensors,
        "optimizer_tensors_checked": optimizer_tensors,
        "gradscaler_tensors_checked": scaler_tensors,
        "state_ema_sha256": reproducibility.module_state_sha256(state["ema"]),
        "trajectory_config_sha256": trajectory_sha256,
    }


def validate_snapshot(
    path: Path, arm: str, options_info: dict[str, Any], state_info: dict[str, Any]
) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            snapshot = pickle.load(handle)
    except Exception as exc:
        fail(f"cannot load trusted network snapshot {path}: {exc}")
    snapshot = require_dict(snapshot, "network snapshot")
    required = ("ema", "loss_fn", "augment_pipe", "dataset_kwargs")
    missing = [name for name in required if name not in snapshot]
    if missing:
        fail(f"network snapshot missing fields: {missing}")
    ema_tensors = validate_module(snapshot["ema"], "network snapshot EMA")
    snapshot_ema_sha = reproducibility.module_state_sha256(snapshot["ema"])
    if snapshot_ema_sha != state_info["state_ema_sha256"]:
        fail("network snapshot EMA does not exactly match self-contained state EMA")

    loss_fn = snapshot["loss_fn"]
    validate_factorial(getattr(loss_fn, "factorial", None), arm, "snapshot loss factorial")
    exact_value(
        getattr(getattr(loss_fn, "schedule", None), "name", None),
        "sigmoid", "snapshot loss schedule",
    )
    exact_number(getattr(loss_fn, "q", None), 256.0, "snapshot loss q")
    exact_number(getattr(loss_fn, "c", None), 0.0, "snapshot loss c")
    if snapshot["augment_pipe"] is not None:
        fail("network snapshot augment_pipe must be null")
    dataset = require_dict(snapshot["dataset_kwargs"], "network snapshot dataset_kwargs")
    exact_value(dataset.get("path"), options_info["dataset_path"], "snapshot dataset path")
    exact_value(dataset.get("use_labels"), False, "snapshot dataset use_labels")
    exact_value(dataset.get("xflip"), False, "snapshot dataset xflip")
    return {
        "ema_tensors_checked": ema_tensors,
        "ema_sha256": snapshot_ema_sha,
    }


def parse_expected_skip_attempts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON skip-attempt list: {exc}")
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("skip attempts must be a JSON array or comma list")
    result: list[int] = []
    for item in parsed:
        text = str(item).strip()
        if not _UINT_RE.fullmatch(text) or int(text) < 1:
            raise argparse.ArgumentTypeError(
                f"skip attempts must be positive canonical integers, got {item!r}"
            )
        result.append(int(text))
    if result != sorted(set(result)):
        raise argparse.ArgumentTypeError("skip attempts must be strictly increasing and unique")
    return result


def verify_run(
    run_dir: Path,
    *,
    arm: str,
    seed: int,
    mode: str,
    expected_skip_attempts: list[int] | None = None,
    write_receipts: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if arm not in ARMS:
        fail(f"unknown arm {arm!r}")
    if mode not in MODES:
        fail(f"unknown mode {mode!r}")
    if mode == "smoke" and seed != 3:
        fail("the frozen 32-attempt smoke gate uses seed 3")
    if mode == "formal" and seed not in (3, 4, 5):
        fail("formal seed must be one of 3, 4, 5")
    if expected_skip_attempts is not None:
        if expected_skip_attempts != sorted(set(expected_skip_attempts)):
            fail("expected skip attempts must be strictly increasing and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < 1 or value > MODES[mode]["attempts"]
            for value in expected_skip_attempts
        ):
            fail("expected skip attempts are outside the selected mode")
        if any(
            value * 128 >= AMP_SKIP_WARMUP_PROCESSED_NIMG
            for value in expected_skip_attempts
        ):
            fail("expected skip attempts extend beyond the tick-0 warm-up region")

    if not run_dir.is_dir():
        fail(f"run directory does not exist: {run_dir}")
    paths = {name: run_dir / name for name in CORE_ARTIFACTS}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            fail(f"missing or empty required artifact {name}: {path}")
    numbered = sorted(run_dir.glob("network-snapshot-[0-9]*.pkl"))
    numbered += sorted(run_dir.glob("training-state-[0-9]*.pt"))
    if numbered:
        fail(f"frozen no-numbered-artifact policy violated: {numbered}")

    validation_path = run_dir / VALIDATION_FILENAME
    hash_receipt_path = run_dir / HASH_RECEIPT_FILENAME
    if write_receipts:
        for path in (validation_path, hash_receipt_path):
            if path.exists():
                fail(f"immutable verifier output already exists: {path}")

    log_text = paths["log.txt"].read_text(encoding="utf-8", errors="replace")
    if "Exiting..." not in log_text:
        fail("log.txt lacks the clean completion marker")
    if "Traceback (most recent call last)" in log_text:
        fail("log.txt contains a Python traceback")

    options = load_json(paths["training_options.json"], "training options")
    options_info = validate_training_options(options, arm, seed, mode)
    launch_manifest = load_json(paths["launch_manifest.json"], "launch manifest")
    launch_info = validate_launch_manifest(
        launch_manifest, run_dir, arm, seed, mode, options_info
    )
    exact_value(
        expected_skip_attempts,
        launch_info["expected_skip_attempts"],
        "verifier/authorization expected AMP skip signature",
    )
    initial = load_json(paths["initial_state_receipt_v1.json"], "initial receipt")
    initial_info = validate_initial_receipt(initial, arm, seed, options_info)
    telemetry_rows = read_telemetry(paths["factorial_training_telemetry_v1.csv"])
    telemetry_info = validate_telemetry(
        telemetry_rows, arm, mode, expected_skip_attempts
    )
    state_info = validate_training_state(
        paths["training-state-latest.pt"], arm, seed, mode, telemetry_info
    )
    exact_value(
        state_info["trajectory_config_sha256"],
        initial_info["trajectory_config_sha256"],
        "initial/training-state trajectory config hash",
    )
    snapshot_info = validate_snapshot(
        paths["network-snapshot-latest.pkl"], arm, options_info, state_info
    )

    report = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "verifier_version": VERIFIER_VERSION,
        "run_dir": str(run_dir),
        "mode": mode,
        "arm": arm,
        "seed": seed,
        "factorial": expected_factorial(arm),
        "attempted_iterations": telemetry_info["attempts"],
        "successful_optimizer_steps": telemetry_info["successful_optimizer_steps"],
        "processed_nimg": telemetry_info["processed_nimg"],
        "amp_skip_attempts": telemetry_info["amp_skip_attempts"],
        "amp_skip_signature_expected_value_enforced": expected_skip_attempts is not None,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "initial_common_state_sha256": initial_info["common_initial_state_sha256"],
        "trajectory_config_sha256": initial_info["trajectory_config_sha256"],
        "snapshot_ema_sha256": snapshot_info["ema_sha256"],
        "source_git_head": launch_info["source_git_head"],
        "source_content_sha256": launch_info["source_content_sha256"],
        "finite_tensor_checks": {
            "state_net": state_info["net_tensors_checked"],
            "state_ema": state_info["ema_tensors_checked"],
            "state_optimizer": state_info["optimizer_tensors_checked"],
            "state_gradscaler": state_info["gradscaler_tensors_checked"],
            "snapshot_ema": snapshot_info["ema_tensors_checked"],
        },
    }

    if write_receipts:
        reproducibility.atomic_json_dump(report, validation_path, overwrite=False)
        artifact_paths = {
            **paths,
            **launch_info["bound_artifacts"],
        }
        artifact_hashes = {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        }
        artifact_hashes[VALIDATION_FILENAME] = {
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        }
        hash_receipt = {
            "schema": HASH_RECEIPT_SCHEMA,
            "status": "passed",
            "verifier_version": VERIFIER_VERSION,
            "run_dir": str(run_dir),
            "mode": mode,
            "arm": arm,
            "seed": seed,
            "artifacts": artifact_hashes,
        }
        reproducibility.atomic_json_dump(
            hash_receipt, hash_receipt_path, overwrite=False
        )
        report["validation_receipt"] = str(validation_path)
        report["artifact_hash_receipt"] = str(hash_receipt_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument(
        "--expected-skip-attempts",
        type=parse_expected_skip_attempts,
        default=None,
        metavar="LIST|JSON",
        help=(
            "optional frozen AMP skip signature, e.g. '1,2,3' or '[1,2,3]'; "
            "when omitted the observed signature is extracted but not guessed"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "re-run every production validation against existing immutable "
            "artifacts without writing or replacing receipts"
        ),
    )
    args = parser.parse_args()
    try:
        report = verify_run(
            args.run_dir,
            arm=args.arm,
            seed=args.seed,
            mode=args.mode,
            expected_skip_attempts=args.expected_skip_attempts,
            write_receipts=not args.check_only,
        )
    except VerificationError as exc:
        raise SystemExit(f"[verify_q256_target_weight_arm] ERROR: {exc}") from exc
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
