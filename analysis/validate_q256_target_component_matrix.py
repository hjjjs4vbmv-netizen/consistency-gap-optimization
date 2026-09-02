"""Fail-closed validator for the frozen 12-state q=256 primary audit matrix.

The validator consumes exactly one published audit artifact directory for each
cell in seeds {3,4,5} x budgets {256,512,768,1024} kimg, all from arm A.  It
does not load checkpoints, execute a model, or use a GPU.  It verifies each
artifact's manifest and hash-bound CSVs, then checks that the stochastic probe
realization and implementation/runtime contracts are identical across cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.gap_gradient_hook import sha256_file
from analysis.q256_target_component_audit import (
    ARM_FACTORS,
    CANONICAL_CIFAR10_SHA256,
    MAX_IDENTITY_RELATIVE_TOLERANCE,
    PRIMARY_AUDIT_SEED,
    PRIMARY_BATCHES,
    PRIMARY_BATCH_SIZE,
    PRIMARY_LOSS_CONTRACT,
    PRIMARY_STATE_ARM,
    PRIMARY_STATE_BUDGETS_KIMG,
    PRIMARY_TRAINING_SEEDS,
    REPLAY_RECEIPT_SCHEMA,
    SHA256_PATTERN,
    implementation_hashes,
)


PRIMARY_SCHEMA = "ect.q256.target-component-audit-primary/v2"
PRIMARY_STATUS = "PASS_PRIMARY_COMMON_STATE_GRADIENT_AUDIT"
MATRIX_SCHEMA = "ect.q256.target-component-audit-primary-matrix/v2"
MATRIX_STATUS = "PASS_PRIMARY_12_STATE_MATRIX"
MANIFEST_FILENAME = "target_component_manifest.json"
LAYER_CSV_FILENAME = "target_component_layers.csv"
BATCH_CSV_FILENAME = "target_component_batches.csv"
EXPECTED_ARTIFACT_FILENAMES = frozenset(
    {LAYER_CSV_FILENAME, BATCH_CSV_FILENAME}
)
EXPECTED_MATRIX = frozenset(
    (seed, budget)
    for seed in PRIMARY_TRAINING_SEEDS
    for budget in PRIMARY_STATE_BUDGETS_KIMG
)
REQUIRED_IMPLEMENTATION_LABELS = frozenset(
    {
        "runner",
        "protocol",
        "protocol_amendment_001",
        "fixed_randomness_helper",
        "loss",
        "schedules",
        "reproducibility",
        "dataset",
        "networks",
        "dnnlib_init",
        "dnnlib_util",
        "torch_utils_persistence",
        "torch_utils_distributed",
        "matrix_validator",
    }
)
BATCH_HASH_COLUMNS = (
    "images_sha256",
    "labels_sha256",
    "t_sha256",
    "eps_sha256",
    "dropout_rng_sha256",
    "baseline_target_r_sha256",
    "enlarged_target_r_sha256",
    "baseline_denominator_r_sha256",
    "enlarged_denominator_r_sha256",
)
EXPECTED_IDENTITY_ERROR_KEYS = frozenset(
    {
        "max_identity_d_equals_s_a_relative_l2",
        "max_identity_b_equals_s_c_relative_l2",
        "max_loss_identity_d_equals_s_a_relative_l2",
        "max_loss_identity_b_equals_s_c_relative_l2",
    }
)


class MatrixValidationError(ValueError):
    """Raised when any primary-matrix admission gate is not proven."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MatrixValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixValidationError(f"cannot read valid JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MatrixValidationError(f"JSON root is not an object: {path}")
    return payload


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise MatrixValidationError(f"{label} is not a lowercase SHA256 digest")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MatrixValidationError(f"{label} is not an object")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixValidationError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MatrixValidationError(f"{label} is not finite")
    return number


def _expect_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise MatrixValidationError(
            f"{label}={observed!r}, expected {expected!r}"
        )


def _expect_typed_equal(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise MatrixValidationError(
            f"{label}={observed!r} ({type(observed).__name__}), expected "
            f"{expected!r} ({type(expected).__name__})"
        )


def _read_batch_hash_contract(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise MatrixValidationError(f"batch CSV has no header: {path}")
            if len(fieldnames) != len(set(fieldnames)):
                raise MatrixValidationError(f"batch CSV has duplicate columns: {path}")
            required = {"batch_index", "sample_count", *BATCH_HASH_COLUMNS}
            missing = sorted(required - set(fieldnames))
            if missing:
                raise MatrixValidationError(
                    f"batch CSV is missing required columns {missing}: {path}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise MatrixValidationError(f"cannot read batch CSV {path}: {error}") from error

    if len(rows) != PRIMARY_BATCHES:
        raise MatrixValidationError(
            f"batch CSV has {len(rows)} rows, expected {PRIMARY_BATCHES}: {path}"
        )
    contract: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            batch_index = int(row["batch_index"])
            sample_count = int(row["sample_count"])
        except (TypeError, ValueError) as error:
            raise MatrixValidationError(
                f"non-integer batch metadata at {path}:{row_number}"
            ) from error
        if str(batch_index) != row["batch_index"]:
            raise MatrixValidationError(
                f"non-canonical batch_index at {path}:{row_number}"
            )
        if str(sample_count) != row["sample_count"]:
            raise MatrixValidationError(
                f"non-canonical sample_count at {path}:{row_number}"
            )
        if batch_index in seen_indices:
            raise MatrixValidationError(f"duplicate batch_index {batch_index}: {path}")
        seen_indices.add(batch_index)
        if sample_count != PRIMARY_BATCH_SIZE:
            raise MatrixValidationError(
                f"batch {batch_index} sample_count={sample_count}, expected "
                f"{PRIMARY_BATCH_SIZE}: {path}"
            )
        item: dict[str, Any] = {
            "batch_index": batch_index,
            "sample_count": sample_count,
        }
        for column in BATCH_HASH_COLUMNS:
            item[column] = _require_digest(
                row.get(column), f"{path}:{row_number}:{column}"
            )
        contract.append(item)
    expected_indices = set(range(PRIMARY_BATCHES))
    if seen_indices != expected_indices:
        raise MatrixValidationError(
            f"batch indices are {sorted(seen_indices)}, expected "
            f"{sorted(expected_indices)}: {path}"
        )
    return sorted(contract, key=lambda row: row["batch_index"])


def _state_loss_contract(state: dict[str, Any], label: str) -> dict[str, Any]:
    expected = {
        "loss_stage": 0,
        "loss_q": 256.0,
        "loss_P_mean": PRIMARY_LOSS_CONTRACT["P_mean"],
        "loss_P_std": PRIMARY_LOSS_CONTRACT["P_std"],
        "loss_sigma_data": PRIMARY_LOSS_CONTRACT["sigma_data"],
        "loss_k": PRIMARY_LOSS_CONTRACT["k"],
        "loss_b": PRIMARY_LOSS_CONTRACT["b"],
        "loss_c": 0.0,
        "state_arm": PRIMARY_STATE_ARM,
    }
    observed: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key == "state_arm":
            value = state.get(key)
        else:
            value = _require_finite_number(state.get(key), f"{label}.state.{key}")
        _expect_equal(value, expected_value, f"{label}.state.{key}")
        observed[key] = value
    return observed


def _runtime_core(manifest: dict[str, Any], label: str) -> dict[str, Any]:
    runtime = _require_mapping(manifest.get("runtime_contract"), f"{label}.runtime")
    gpu_name = runtime.get("gpu_name")
    torch_version = manifest.get("torch_version")
    cuda_version = manifest.get("cuda_version")
    cudnn_version = runtime.get("cudnn_version")
    python_version = runtime.get("python_version")
    platform_value = runtime.get("platform")
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise MatrixValidationError(f"{label} has no GPU name")
    if not isinstance(torch_version, str) or not torch_version.strip():
        raise MatrixValidationError(f"{label} has no PyTorch version")
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        raise MatrixValidationError(f"{label} has no CUDA version")
    if isinstance(cudnn_version, bool) or not isinstance(cudnn_version, int):
        raise MatrixValidationError(f"{label} has no integer cuDNN version")
    if cudnn_version <= 0:
        raise MatrixValidationError(f"{label} has invalid cuDNN version")
    if not isinstance(python_version, str) or not python_version.strip():
        raise MatrixValidationError(f"{label} has no Python version")
    if not isinstance(platform_value, str) or not platform_value.strip():
        raise MatrixValidationError(f"{label} has no platform provenance")
    for key, expected in {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }.items():
        _expect_typed_equal(runtime.get(key), expected, f"{label}.runtime.{key}")
    _expect_equal(
        runtime.get("torch_version"), torch_version, f"{label}.runtime.torch_version"
    )
    _expect_equal(
        runtime.get("cuda_version"), cuda_version, f"{label}.runtime.cuda_version"
    )
    _expect_equal(
        runtime.get("cublas_workspace_config"),
        ":4096:8",
        f"{label}.runtime.cublas_workspace_config",
    )
    return {
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "cublas_workspace_config": runtime["cublas_workspace_config"],
        "python_version": python_version,
        "platform": platform_value,
    }


def _validate_one_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise MatrixValidationError(f"artifact path is not a directory: {resolved}")
    manifest_path = resolved / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MatrixValidationError(f"missing {MANIFEST_FILENAME}: {resolved}")
    manifest = _load_json(manifest_path)
    label = str(resolved)

    fixed_fields = {
        "schema": PRIMARY_SCHEMA,
        "status": PRIMARY_STATUS,
        "run_kind": "primary",
        "estimand": "fp32_reference_one_sided_stop_gradient_objective_gradient",
        "audit_seed": PRIMARY_AUDIT_SEED,
        "batches": PRIMARY_BATCHES,
        "batch_size": PRIMARY_BATCH_SIZE,
        "force_fp32": True,
        "amp_used": False,
        "optimizer_constructed_or_stepped": False,
        "network_state_preserved": True,
        "identity_gate_passed": True,
        "audit_gate_passed": True,
        "arm_factors": {
            arm: list(factors) for arm, factors in ARM_FACTORS.items()
        },
    }
    for key, expected in fixed_fields.items():
        _expect_typed_equal(manifest.get(key), expected, f"{label}.{key}")
    device = manifest.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise MatrixValidationError(f"{label}.device is not CUDA: {device!r}")
    _expect_equal(
        manifest.get("dataset_sha256"),
        CANONICAL_CIFAR10_SHA256,
        f"{label}.dataset_sha256",
    )
    _expect_equal(
        _require_finite_number(
            manifest.get("identity_relative_tolerance"),
            f"{label}.identity_relative_tolerance",
        ),
        MAX_IDENTITY_RELATIVE_TOLERANCE,
        f"{label}.identity_relative_tolerance",
    )
    identity_errors = _require_mapping(
        manifest.get("identity_errors"), f"{label}.identity_errors"
    )
    if set(identity_errors) != EXPECTED_IDENTITY_ERROR_KEYS:
        raise MatrixValidationError(
            f"{label}.identity_errors keys={sorted(identity_errors)}, expected "
            f"{sorted(EXPECTED_IDENTITY_ERROR_KEYS)}"
        )
    for name, value in identity_errors.items():
        error = _require_finite_number(value, f"{label}.identity_errors.{name}")
        if error > MAX_IDENTITY_RELATIVE_TOLERANCE:
            raise MatrixValidationError(
                f"{label}.identity_errors.{name}={error} exceeds gate"
            )
    layerwise_summary = _require_mapping(
        manifest.get("layerwise_summary"), f"{label}.layerwise_summary"
    )
    _expect_equal(
        layerwise_summary.get("energy_reconstruction_gate_passed"),
        True,
        f"{label}.layerwise_summary.energy_reconstruction_gate_passed",
    )

    training_seed = manifest.get("training_seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise MatrixValidationError(f"{label}.training_seed is not an integer")
    state = _require_mapping(manifest.get("state"), f"{label}.state")
    budget = state.get("state_kimg")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise MatrixValidationError(f"{label}.state.state_kimg is not an integer")
    if (training_seed, budget) not in EXPECTED_MATRIX:
        raise MatrixValidationError(
            f"unexpected primary cell seed={training_seed}, budget={budget}"
        )
    _expect_equal(
        state.get("cur_nimg"), budget * 1000, f"{label}.state.cur_nimg"
    )
    loss_contract = _state_loss_contract(state, label)

    receipt = _require_mapping(
        manifest.get("checkpoint_receipt_payload"),
        f"{label}.checkpoint_receipt_payload",
    )
    for key, expected in {
        "schema": REPLAY_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": training_seed,
        "arm": PRIMARY_STATE_ARM,
        "budget_kimg": budget,
    }.items():
        _expect_equal(receipt.get(key), expected, f"{label}.receipt.{key}")
    for key in (
        "training_state_sha256",
        "checkpoint_sha256",
        "checkpoint_receipt_sha256",
    ):
        _require_digest(manifest.get(key), f"{label}.{key}")

    implementation = _require_mapping(
        manifest.get("implementation_sha256"), f"{label}.implementation_sha256"
    )
    missing_implementation = sorted(
        REQUIRED_IMPLEMENTATION_LABELS - set(implementation)
    )
    if missing_implementation:
        raise MatrixValidationError(
            f"{label} implementation hashes miss {missing_implementation}"
        )
    for name, digest in implementation.items():
        _require_digest(digest, f"{label}.implementation_sha256.{name}")

    runtime = _require_mapping(manifest.get("runtime_contract"), f"{label}.runtime")
    runtime_core = _runtime_core(manifest, label)
    artifact_hashes = _require_mapping(
        manifest.get("artifact_sha256"), f"{label}.artifact_sha256"
    )
    if set(artifact_hashes) != EXPECTED_ARTIFACT_FILENAMES:
        raise MatrixValidationError(
            f"{label}.artifact_sha256 keys={sorted(artifact_hashes)}, expected "
            f"{sorted(EXPECTED_ARTIFACT_FILENAMES)}"
        )
    for filename in sorted(EXPECTED_ARTIFACT_FILENAMES):
        expected_digest = _require_digest(
            artifact_hashes.get(filename), f"{label}.artifact_sha256.{filename}"
        )
        artifact_path = resolved / filename
        if not artifact_path.is_file():
            raise MatrixValidationError(f"missing hash-bound artifact: {artifact_path}")
        observed_digest = sha256_file(artifact_path)
        if observed_digest != expected_digest:
            raise MatrixValidationError(
                f"artifact SHA256 mismatch for {artifact_path}: "
                f"{observed_digest} != {expected_digest}"
            )
    batch_contract = _read_batch_hash_contract(resolved / BATCH_CSV_FILENAME)

    trajectory_hash = _require_digest(
        state.get("trajectory_config_sha256"),
        f"{label}.state.trajectory_config_sha256",
    )
    trajectory_dynamics_hash = _require_digest(
        state.get("trajectory_dynamics_sha256"),
        f"{label}.state.trajectory_dynamics_sha256",
    )
    trajectory_total_kimg = state.get("trajectory_total_kimg")
    if (
        isinstance(trajectory_total_kimg, bool)
        or not isinstance(trajectory_total_kimg, int)
        or trajectory_total_kimg < budget
    ):
        raise MatrixValidationError(
            f"{label}.state.trajectory_total_kimg={trajectory_total_kimg!r}, "
            f"expected an integer >= {budget}"
        )
    return {
        "cell": (training_seed, budget),
        "artifact_dir": str(resolved),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "implementation_sha256": implementation,
        "runtime_contract": runtime,
        "runtime_core": runtime_core,
        "loss_contract": loss_contract,
        "trajectory_config_sha256": trajectory_hash,
        "trajectory_dynamics_sha256": trajectory_dynamics_hash,
        "trajectory_total_kimg": trajectory_total_kimg,
        "batch_hash_contract": batch_contract,
    }


def validate_primary_matrix(artifact_dirs: Iterable[Path]) -> dict[str, Any]:
    """Validate and summarize exactly the frozen twelve primary cells."""
    paths = [Path(path) for path in artifact_dirs]
    if len(paths) != len(EXPECTED_MATRIX):
        raise MatrixValidationError(
            f"received {len(paths)} artifact directories, expected "
            f"{len(EXPECTED_MATRIX)}"
        )
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise MatrixValidationError("artifact directory paths are not unique")
    records = [_validate_one_artifact(path) for path in resolved_paths]

    cells = [record["cell"] for record in records]
    if len(set(cells)) != len(cells):
        raise MatrixValidationError(f"duplicate matrix cells: {cells}")
    if set(cells) != EXPECTED_MATRIX:
        missing = sorted(EXPECTED_MATRIX - set(cells))
        unexpected = sorted(set(cells) - EXPECTED_MATRIX)
        raise MatrixValidationError(
            f"matrix cell mismatch; missing={missing}, unexpected={unexpected}"
        )

    reference = records[0]
    for record in records[1:]:
        cell_label = f"seed={record['cell'][0]},budget={record['cell'][1]}"
        for key in (
            "implementation_sha256",
            "runtime_contract",
            "runtime_core",
            "loss_contract",
            "batch_hash_contract",
        ):
            if record[key] != reference[key]:
                raise MatrixValidationError(
                    f"{key} differs at {cell_label} from the matrix reference"
                )

    current_implementation = implementation_hashes()
    if reference["implementation_sha256"] != current_implementation:
        raise MatrixValidationError(
            "recorded implementation_sha256 does not match the validator's "
            "current implementation closure"
        )

    for seed in sorted(PRIMARY_TRAINING_SEEDS):
        seed_hashes = {
            record["trajectory_dynamics_sha256"]
            for record in records
            if record["cell"][0] == seed
        }
        if len(seed_hashes) != 1:
            raise MatrixValidationError(
                "trajectory_dynamics_sha256 differs across budgets for seed "
                f"{seed}"
            )

    sorted_records = sorted(records, key=lambda record: record["cell"])
    return {
        "schema": MATRIX_SCHEMA,
        "status": MATRIX_STATUS,
        "source_artifact_count": len(sorted_records),
        "expected_training_seeds": sorted(PRIMARY_TRAINING_SEEDS),
        "expected_budgets_kimg": sorted(PRIMARY_STATE_BUDGETS_KIMG),
        "state_arm": PRIMARY_STATE_ARM,
        "audit_seed": PRIMARY_AUDIT_SEED,
        "batches": PRIMARY_BATCHES,
        "batch_size": PRIMARY_BATCH_SIZE,
        "dataset_sha256": CANONICAL_CIFAR10_SHA256,
        "common_implementation_sha256": reference["implementation_sha256"],
        "common_runtime_contract": reference["runtime_contract"],
        "common_runtime_core": reference["runtime_core"],
        "common_loss_contract": reference["loss_contract"],
        "common_batch_hash_contract": reference["batch_hash_contract"],
        "cells": [
            {
                "training_seed": record["cell"][0],
                "budget_kimg": record["cell"][1],
                "artifact_dir": record["artifact_dir"],
                "manifest_sha256": record["manifest_sha256"],
                "artifact_sha256": record["artifact_sha256"],
                "trajectory_config_sha256": record[
                    "trajectory_config_sha256"
                ],
                "trajectory_dynamics_sha256": record[
                    "trajectory_dynamics_sha256"
                ],
                "trajectory_total_kimg": record["trajectory_total_kimg"],
            }
            for record in sorted_records
        ],
    }


def publish_validation_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one validation record without overwriting any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as error:
        raise MatrixValidationError(f"output path already exists: {path}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_dirs",
        nargs="+",
        type=Path,
        help="the twelve published primary audit artifact directories",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = validate_primary_matrix(args.artifact_dirs)
    publish_validation_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
