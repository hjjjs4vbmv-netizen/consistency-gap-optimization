import csv
import json
from pathlib import Path

import pytest
import torch

from analysis.gap_gradient_hook import sha256_file
from analysis.q256_target_component_audit import (
    CANONICAL_CIFAR10_SHA256,
    configure_deterministic_runtime,
    implementation_hashes,
)
from analysis.validate_q256_target_component_matrix import (
    BATCH_CSV_FILENAME,
    BATCH_HASH_COLUMNS,
    LAYER_CSV_FILENAME,
    MANIFEST_FILENAME,
    MATRIX_STATUS,
    MatrixValidationError,
    publish_validation_json,
    validate_primary_matrix,
)


SEEDS = (3, 4, 5)
BUDGETS = (256, 512, 768, 1024)


def digest(character: str) -> str:
    return character * 64


def batch_rows(*, altered=False):
    rows = []
    for batch_index in range(8):
        row = {"batch_index": batch_index, "sample_count": 16}
        for offset, column in enumerate(BATCH_HASH_COLUMNS):
            character = format((batch_index + offset) % 16, "x")
            row[column] = digest(character)
        rows.append(row)
    if altered:
        rows[4]["eps_sha256"] = digest("f")
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_artifact(root: Path, seed: int, budget: int) -> Path:
    out = root / f"seed{seed}-kimg{budget}"
    out.mkdir(parents=True)
    layer_path = out / LAYER_CSV_FILENAME
    layer_path.write_text("layer,r_tar\nall,0.5\n", encoding="utf-8")
    batch_path = out / BATCH_CSV_FILENAME
    write_csv(batch_path, batch_rows())
    implementation = implementation_hashes()
    runtime = {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
        "cudnn_version": 90100,
        "python_version": "3.10.14",
        "platform": "Linux-6.8-x86_64",
        "torch_version": "2.3.0a0+nv24.04",
        "cuda_version": "12.4",
        "gpu_name": "NVIDIA A100 80GB PCIe",
    }
    manifest = {
        "schema": "ect.q256.target-component-audit-primary/v2",
        "status": "PASS_PRIMARY_COMMON_STATE_GRADIENT_AUDIT",
        "run_kind": "primary",
        "estimand": "fp32_reference_one_sided_stop_gradient_objective_gradient",
        "training_seed": seed,
        "audit_seed": 20260823,
        "batches": 8,
        "batch_size": 16,
        "device": "cuda",
        "force_fp32": True,
        "amp_used": False,
        "optimizer_constructed_or_stepped": False,
        "network_state_preserved": True,
        "identity_relative_tolerance": 1e-4,
        "identity_errors": {
            "max_identity_d_equals_s_a_relative_l2": 1e-6,
            "max_identity_b_equals_s_c_relative_l2": 1e-6,
            "max_loss_identity_d_equals_s_a_relative_l2": 1e-6,
            "max_loss_identity_b_equals_s_c_relative_l2": 1e-6,
        },
        "identity_gate_passed": True,
        "audit_gate_passed": True,
        "layerwise_summary": {"energy_reconstruction_gate_passed": True},
        "arm_factors": {
            "A": [1.0, 1.0],
            "B": [1.1, 1.1],
            "C": [1.1, 1.0],
            "D": [1.0, 1.1],
        },
        "dataset_sha256": CANONICAL_CIFAR10_SHA256,
        "training_state_sha256": digest("7"),
        "checkpoint_sha256": digest("8"),
        "checkpoint_receipt_sha256": digest("9"),
        "checkpoint_receipt_payload": {
            "schema": "ect.q256.replay-ema-export/v1",
            "status": "PASS",
            "seed": seed,
            "arm": "A",
            "budget_kimg": budget,
        },
        "implementation_sha256": implementation,
        "runtime_contract": runtime,
        "torch_version": "2.3.0a0+nv24.04",
        "cuda_version": "12.4",
        "state": {
            "cur_nimg": budget * 1000,
            "loss_stage": 0,
            "loss_q": 256.0,
            "loss_P_mean": -1.1,
            "loss_P_std": 2.0,
            "loss_sigma_data": 0.5,
            "loss_k": 8.0,
            "loss_b": 1.0,
            "loss_c": 0.0,
            "state_arm": "A",
            "state_kimg": budget,
            "trajectory_config_sha256": digest(
                format((seed + budget) % 16, "x")
            ),
            "trajectory_dynamics_sha256": digest(str(seed)),
            "trajectory_total_kimg": 256 if budget == 256 else 1024,
        },
        "artifact_sha256": {
            LAYER_CSV_FILENAME: sha256_file(layer_path),
            BATCH_CSV_FILENAME: sha256_file(batch_path),
        },
    }
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def make_matrix(tmp_path: Path):
    return [
        make_artifact(tmp_path, seed, budget)
        for seed in SEEDS
        for budget in BUDGETS
    ]


def load_manifest(artifact: Path):
    return json.loads((artifact / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def save_manifest(artifact: Path, manifest):
    (artifact / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_valid_primary_matrix_and_non_overwriting_publication(tmp_path):
    artifacts = make_matrix(tmp_path / "matrix")
    payload = validate_primary_matrix(reversed(artifacts))

    assert payload["status"] == MATRIX_STATUS
    assert payload["source_artifact_count"] == 12
    assert [(cell["training_seed"], cell["budget_kimg"]) for cell in payload["cells"]] == [
        (seed, budget) for seed in SEEDS for budget in BUDGETS
    ]
    assert len(payload["common_batch_hash_contract"]) == 8
    out = tmp_path / "validation.json"
    publish_validation_json(out, payload)
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == MATRIX_STATUS
    with pytest.raises(MatrixValidationError, match="already exists"):
        publish_validation_json(out, payload)


def test_matrix_rejects_missing_or_duplicate_cells(tmp_path):
    artifacts = make_matrix(tmp_path / "matrix")
    with pytest.raises(MatrixValidationError, match="received 11"):
        validate_primary_matrix(artifacts[:-1])
    with pytest.raises(MatrixValidationError, match="paths are not unique"):
        validate_primary_matrix(artifacts[:-1] + [artifacts[0]])


def test_matrix_rejects_self_consistent_cross_state_batch_hash_drift(tmp_path):
    artifacts = make_matrix(tmp_path / "matrix")
    changed = artifacts[-1]
    batch_path = changed / BATCH_CSV_FILENAME
    write_csv(batch_path, batch_rows(altered=True))
    manifest = load_manifest(changed)
    manifest["artifact_sha256"][BATCH_CSV_FILENAME] = sha256_file(batch_path)
    save_manifest(changed, manifest)

    with pytest.raises(MatrixValidationError, match="batch_hash_contract differs"):
        validate_primary_matrix(artifacts)


def test_matrix_rejects_csv_tampering_and_primary_cublas_smoke_config(tmp_path):
    artifacts = make_matrix(tmp_path / "matrix")
    batch_path = artifacts[0] / BATCH_CSV_FILENAME
    batch_path.write_text(batch_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(MatrixValidationError, match="artifact SHA256 mismatch"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-cublas")
    manifest = load_manifest(artifacts[0])
    manifest["runtime_contract"]["cublas_workspace_config"] = ":16:8"
    save_manifest(artifacts[0], manifest)
    with pytest.raises(MatrixValidationError, match="cublas_workspace_config"):
        validate_primary_matrix(artifacts)


def test_matrix_rejects_implementation_runtime_and_loss_contract_drift(tmp_path):
    artifacts = make_matrix(tmp_path / "matrix")
    manifest = load_manifest(artifacts[3])
    manifest["implementation_sha256"]["runner"] = digest("a")
    save_manifest(artifacts[3], manifest)
    with pytest.raises(MatrixValidationError, match="implementation_sha256 differs"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-runtime")
    manifest = load_manifest(artifacts[3])
    manifest["runtime_contract"]["cudnn_version"] = 90200
    save_manifest(artifacts[3], manifest)
    with pytest.raises(MatrixValidationError, match="runtime_contract differs"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-nondeterministic")
    for artifact in artifacts:
        manifest = load_manifest(artifact)
        manifest["runtime_contract"]["deterministic_algorithms"] = False
        save_manifest(artifact, manifest)
    with pytest.raises(MatrixValidationError, match="deterministic_algorithms"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-loss")
    manifest = load_manifest(artifacts[3])
    manifest["state"]["loss_k"] = 7.0
    save_manifest(artifacts[3], manifest)
    with pytest.raises(MatrixValidationError, match="state.loss_k"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-dynamics")
    manifest = load_manifest(artifacts[3])
    manifest["state"]["trajectory_dynamics_sha256"] = digest("f")
    save_manifest(artifacts[3], manifest)
    with pytest.raises(MatrixValidationError, match="trajectory_dynamics_sha256"):
        validate_primary_matrix(artifacts)

    artifacts = make_matrix(tmp_path / "matrix-horizon")
    manifest = load_manifest(artifacts[3])
    manifest["state"]["trajectory_total_kimg"] = 512
    save_manifest(artifacts[3], manifest)
    with pytest.raises(MatrixValidationError, match="trajectory_total_kimg"):
        validate_primary_matrix(artifacts)


def test_runtime_cublas_policy_and_cudnn_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake-gpu")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    preflight = configure_deterministic_runtime(
        torch.device("cuda"), run_kind="primary", preflight_only=True
    )
    assert "cudnn_version" in preflight
    assert preflight["cublas_workspace_config"] is None

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(SystemExit, match="only for smoke"):
        configure_deterministic_runtime(
            torch.device("cuda"), run_kind="primary", preflight_only=False
        )
    smoke = configure_deterministic_runtime(
        torch.device("cuda"), run_kind="smoke", preflight_only=False
    )
    assert smoke["cublas_workspace_config"] == ":16:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    primary = configure_deterministic_runtime(
        torch.device("cuda"), run_kind="primary", preflight_only=False
    )
    assert primary["cublas_workspace_config"] == ":4096:8"
