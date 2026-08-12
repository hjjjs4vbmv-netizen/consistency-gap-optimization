#!/usr/bin/env python3
"""Fail-closed contracts shared by the blind seed-replication audit tools."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
SOURCE_AUDIT_RECEIPT_SHA256 = (
    "6487fbcc5f63817c8e3a91968f45fb13437d1c580afa73966bdf0ad8061bb9fa"
)
MATRIX_SHA256 = "113a4676916e045f95a1928dd6fa163552515ce589a3721b8873bb72f389ad77"
DATA_SHA256 = "a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"

RUNS = (
    (4, "A", "arm_a_g1_0_lr_fixed_s4"),
    (4, "B", "arm_b_g1_3_lr_fixed_s4"),
    (4, "C", "arm_c_g1_3_lr_matched_s4"),
    (5, "A", "arm_a_g1_0_lr_fixed_s5"),
    (5, "B", "arm_b_g1_3_lr_fixed_s5"),
    (5, "C", "arm_c_g1_3_lr_matched_s5"),
)
EXPECTED_RUNTIME = {
    "arm_a_g1_0_lr_fixed_s4": {
        "seed": 4,
        "arm": "A",
        "gap_scale": 1.0,
        "learning_rate": 0.0001,
        "segment_id": "original",
        "logged_gpu_index": 1,
        "port": 29841,
    },
    "arm_b_g1_3_lr_fixed_s4": {
        "seed": 4,
        "arm": "B",
        "gap_scale": 1.3,
        "learning_rate": 0.0001,
        "segment_id": "seed4_recovery",
        "logged_gpu_index": 0,
        "port": 29842,
    },
    "arm_c_g1_3_lr_matched_s4": {
        "seed": 4,
        "arm": "C",
        "gap_scale": 1.3,
        "learning_rate": 0.00012963523762588692,
        "segment_id": "seed4_recovery",
        "logged_gpu_index": 0,
        "port": 29843,
    },
    "arm_a_g1_0_lr_fixed_s5": {
        "seed": 5,
        "arm": "A",
        "gap_scale": 1.0,
        "learning_rate": 0.0001,
        "segment_id": "seed5_recovery",
        "logged_gpu_index": 1,
        "port": 29851,
    },
    "arm_b_g1_3_lr_fixed_s5": {
        "seed": 5,
        "arm": "B",
        "gap_scale": 1.3,
        "learning_rate": 0.0001,
        "segment_id": "seed5_recovery",
        "logged_gpu_index": 1,
        "port": 29852,
    },
    "arm_c_g1_3_lr_matched_s5": {
        "seed": 5,
        "arm": "C",
        "gap_scale": 1.3,
        "learning_rate": 0.00012963523762588692,
        "segment_id": "seed5_recovery",
        "logged_gpu_index": 1,
        "port": 29853,
    },
}
EXPECTED_OVERLAP_PAIRS = {
    frozenset(("arm_b_g1_3_lr_fixed_s4", "arm_a_g1_0_lr_fixed_s5")),
    frozenset(("arm_c_g1_3_lr_matched_s4", "arm_b_g1_3_lr_fixed_s5")),
}


def expected_launcher_events(segment_id: str) -> list[dict[str, Any]]:
    by_segment = {
        "original": ["arm_a_g1_0_lr_fixed_s4"],
        "seed4_recovery": [
            "arm_b_g1_3_lr_fixed_s4",
            "arm_c_g1_3_lr_matched_s4",
        ],
        "seed5_recovery": [
            "arm_a_g1_0_lr_fixed_s5",
            "arm_b_g1_3_lr_fixed_s5",
            "arm_c_g1_3_lr_matched_s5",
        ],
    }
    if segment_id not in by_segment:
        raise ValueError(f"unknown launcher segment: {segment_id}")
    events: list[dict[str, Any]] = []
    for run_id in by_segment[segment_id]:
        item = EXPECTED_RUNTIME[run_id]
        events.append(
            {
                "event": "START",
                "seed": item["seed"],
                "arm": item["arm"],
                "gap_scale": item["gap_scale"],
                "learning_rate": item["learning_rate"],
                "logged_gpu_index": item["logged_gpu_index"],
                "port": item["port"],
            }
        )
        if segment_id != "original":
            events.append(
                {
                    "event": "DONE",
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "integrity": "passed",
                }
            )
    if segment_id == "seed4_recovery":
        events.append(
            {"event": "COMPLETE", "marker": "SEED4_REMAINING_ARMS_COMPLETE"}
        )
    elif segment_id == "seed5_recovery":
        events.append({"event": "COMPLETE", "marker": "SEED5_ALL_ARMS_COMPLETE"})
    return events


def expected_deviations() -> list[dict[str, Any]]:
    return [
        {
            "id": "D1",
            "field": "launcher_continuity",
            "planned": "single fail-stop launcher",
            "observed": "original launcher plus two manual recovery launchers",
            "materiality": "runtime provenance",
            "evidence_refs": [
                "runtime.launcher_segments",
                "runtime.runs",
                "evidence_manifest.original_launcher_log",
                "evidence_manifest.seed4_recovery_launcher_log",
                "evidence_manifest.seed5_recovery_launcher_log",
            ],
            "confidence": "direct launcher-log evidence",
            "acceptance_rationale": (
                "all recovery runs were fresh starts and passed artifact integrity"
            ),
        },
        {
            "id": "D2",
            "field": "execution_mode",
            "planned": "fully serial",
            "observed": (
                "two training-phase overlap intervals with different logged GPU indices"
            ),
            "materiality": (
                "runtime/performance; no training-definition change observed"
            ),
            "evidence_refs": [
                "runtime.directly_observed_overlaps",
                "runtime.hardware",
            ],
            "confidence": (
                "direct application timestamp overlap; GPU index is launcher assertion"
            ),
            "acceptance_rationale": (
                "the different indices map to equivalent A100 entries in one pre-launch "
                "sidecar; per-run UUID is not attested and performance claims are excluded"
            ),
        },
        {
            "id": "D3",
            "field": "gpu_assignment",
            "planned": "all runs on logged GPU index 1",
            "observed": "seed 4 B/C logged on GPU index 0",
            "materiality": "possible low-order numerical environment effect",
            "evidence_refs": ["runtime.runs", "runtime.hardware"],
            "confidence": (
                "direct index assertion; UUID mapping inferred from one sidecar"
            ),
            "acceptance_rationale": (
                "same GPU model, memory, driver, code, inputs, and training definition"
            ),
        },
        {
            "id": "D4",
            "field": "initialization_evidence",
            "planned": "identical model-init preview files",
            "observed": "seed 5 A versus B/C differs by at most one 8-bit level",
            "materiality": (
                "diagnostic preview only; historical parameter hash absent"
            ),
            "evidence_refs": [
                "initialization.model_init_previews",
                "initialization.reconstructed_expected_initialization",
            ],
            "confidence": (
                "direct PNG comparison plus deterministic reconstruction of expected "
                "tensor state"
            ),
            "acceptance_rationale": (
                "all six expected tensor-state hashes match and preview drift is only one LSB"
            ),
        },
        {
            "id": "D5",
            "field": "inline_verification_continuity",
            "planned": "seed4 A verified inline before launcher continuation",
            "observed": (
                "original verifier output/exit status lost; same artifact set passed "
                "post-hoc re-verification"
            ),
            "materiality": "verification provenance",
            "evidence_refs": [
                "runtime.launcher_segments",
                "per_run_integrity.public_receipts.seed4_A",
                "missing_evidence",
            ],
            "confidence": (
                "training completion and post-hoc receipt direct; original failure "
                "mechanism unknown"
            ),
            "acceptance_rationale": (
                "all seed4 A artifacts were recomputed and passed strengthened "
                "verification; exact original failure not claimed"
            ),
        },
    ]
EXPECTED_INTERFACE = {"img_resolution": 32, "img_channels": 3, "label_dim": 0}
EXPECTED_SOURCE_ONLY = [
    {
        "name": "model.map_augment.weight",
        "shape": [128, 9],
        "dtype": "float32",
        "raw_bytes": 4608,
    }
]
EXPECTED_MODULE_SUMMARY = {
    "schema": "ECT_CANONICAL_TORCH_MODULE_V1",
    "tensor_count": 424,
    "kind_counts": {"buffer": 8, "parameter": 416},
    "dtype_counts": {"float32": 424},
    "total_raw_bytes": 222_931_084,
}
PROTECTED_TOOL_PATHS = (
    "scripts/gap_lr_seed_replication_contract.py",
    "scripts/reconstruct_gap_lr_seed_initialization.py",
    "scripts/verify_gap_lr_seed_replication_run.py",
    "scripts/build_gap_lr_seed_replication_blind_evidence.py",
    "scripts/adjudicate_gap_lr_seed_replication.py",
)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def is_exact_int(value: Any, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def is_finite_number(value: Any, *, positive: bool = False) -> bool:
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    return not positive or value > 0


def exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values while preserving int/float/bool distinctions."""

    def encode(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    try:
        return encode(left) == encode(right)
    except (TypeError, ValueError):
        return False


def loads_strict(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def load_json_object(path: Path) -> dict[str, Any]:
    value = loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_tooling_checkout(
    repo: Path,
    tooling_commit: str,
    *,
    executed_file: Path,
    expected_relative: str,
    imported_files: dict[str, Path] | None = None,
) -> dict[str, str]:
    """Bind the running tool and its audit dependencies to exact Git blobs."""

    if re.fullmatch(r"[0-9a-f]{40}", tooling_commit) is None:
        raise ValueError("adjudication tooling commit must be a full Git SHA")
    repo = repo.resolve()
    if Path(__file__).resolve() != (
        repo / "scripts/gap_lr_seed_replication_contract.py"
    ).resolve():
        raise ValueError("imported contract module is not the --repo committed path")
    if executed_file.resolve() != (repo / expected_relative).resolve():
        raise ValueError(f"executed tool is not the --repo committed path: {expected_relative}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    if head != tooling_commit:
        raise ValueError(
            f"repository HEAD {head} does not equal tooling commit {tooling_commit}"
        )

    blob_hashes: dict[str, str] = {}
    for relative in PROTECTED_TOOL_PATHS:
        working = repo / relative
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{tooling_commit}:{relative}"], cwd=str(repo)
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"tool is absent from tooling commit: {relative}") from exc
        if not working.is_file() or working.read_bytes() != committed:
            raise ValueError(f"working audit tool differs from committed blob: {relative}")
        blob_hashes[relative] = hashlib.sha256(committed).hexdigest()

    for relative, origin in (imported_files or {}).items():
        if origin.resolve() != (repo / relative).resolve():
            raise ValueError(f"imported module is not the --repo committed path: {relative}")
    return blob_hashes


def validate_initialization_report(
    report: dict[str, Any],
    *,
    tooling_commit: str,
    reconstruction_source_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate and recompute every cross-run initialization claim."""

    expected_top = {
        "schema_version",
        "receipt_type",
        "status",
        "experiment_id",
        "quality_blind",
        "bindings",
        "interpretation",
        "canonicalization",
        "runs",
        "cross_run",
    }
    if set(report) != expected_top:
        raise ValueError("initialization report top-level schema mismatch")
    if (
        not is_exact_int(report.get("schema_version"))
        or report.get("schema_version") != 1
        or report.get("receipt_type")
        != "gap_lr_seed_initialization_reconstruction"
        or report.get("status") != "passed"
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("quality_blind", {}).get(
            "generation_quality_metrics_accessed"
        )
        is not False
    ):
        raise ValueError("initialization report identity/quality-blind boundary mismatch")

    expected_bindings = {
        "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
        "adjudication_tooling_commit": tooling_commit,
        "training_code_commit": TRAINING_CODE_COMMIT,
        "source_audit_receipt_sha256": SOURCE_AUDIT_RECEIPT_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "dataset_sha256": DATA_SHA256,
        "transfer_checkpoint_sha256": TRANSFER_SHA256,
        "tool_source_sha256": reconstruction_source_sha256,
    }
    if not exact_json_equal(report.get("bindings"), expected_bindings):
        raise ValueError("initialization report frozen-input/tool bindings mismatch")
    if not exact_json_equal(report.get("interpretation"), {
        "hash_kind": "reconstructed_expected_initialization_hash",
        "historical_observed_preupdate_hash_captured": False,
        "does_not_attest_historical_process_memory": True,
        "rng_state_reconstructed": False,
        "scope": "transferred tensor state only",
    }):
        raise ValueError("initialization reconstruction interpretation mismatch")
    if not exact_json_equal(report.get("quality_blind"), {
        "generation_quality_metrics_accessed": False,
        "inputs_read": [
            "receipt-bound training_options.json",
            "frozen transfer checkpoint",
            "frozen implementation modules",
            "frozen dataset metadata",
        ],
        "inputs_explicitly_not_read": [
            "FID",
            "KID",
            "quality-evaluation outputs",
            "trained network snapshots",
            "training states",
        ],
    }):
        raise ValueError("initialization quality-blind scope mismatch")
    if not exact_json_equal(report.get("canonicalization"), {
        "schema": "ECT_CANONICAL_TORCH_MODULE_V1",
        "ordering": "UTF-8 fully-qualified tensor name, then kind",
        "fields": ["kind", "name", "dtype", "rank", "shape", "nbytes", "raw_bytes"],
        "raw_bytes": "detach, CPU, contiguous, row-major, little-endian",
        "metadata_integer_encoding": "unsigned 64-bit big-endian",
        "excluded": ["module mode", "requires_grad", "non-tensor attributes"],
    }):
        raise ValueError("initialization canonicalization contract mismatch")

    runs = report.get("runs")
    expected_by_id = {run_id: (seed, arm) for seed, arm, run_id in RUNS}
    if not isinstance(runs, dict) or set(runs) != set(expected_by_id):
        raise ValueError("initialization reconstruction run identity set mismatch")

    run_hashes: dict[str, str] = {}
    contract_hashes: dict[str, str] = {}
    for run_id, (seed, arm) in expected_by_id.items():
        row = runs[run_id]
        expected_row_keys = {
            "seed",
            "arm",
            "training_options_sha256",
            "internal_integrity_receipt_sha256",
            "interface_kwargs",
            "initialization_contract_sha256",
            "copy_contract",
            "ema_copy_contract_equal",
            "net",
            "ema",
        }
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError(f"initialization row schema mismatch: {run_id}")
        if (
            not is_exact_int(row.get("seed"))
            or row.get("seed") != seed
            or row.get("arm") != arm
        ):
            raise ValueError(f"initialization row identity mismatch: {run_id}")
        if not is_sha256(row.get("training_options_sha256")) or not is_sha256(
            row.get("internal_integrity_receipt_sha256")
        ):
            raise ValueError(f"initialization row receipt/options binding invalid: {run_id}")
        if not exact_json_equal(row.get("interface_kwargs"), EXPECTED_INTERFACE):
            raise ValueError(f"initialization interface mismatch: {run_id}")
        if not is_sha256(row.get("initialization_contract_sha256")):
            raise ValueError(f"initialization contract hash invalid: {run_id}")

        copy_contract = row.get("copy_contract")
        if not exact_json_equal(copy_contract, {
            "source_tensor_count": 425,
            "destination_tensor_count": 424,
            "missing_destination_names": [],
            "source_only_ignored_by_destination_iterating_copy": EXPECTED_SOURCE_ONLY,
            "shape_dtype_mismatches": [],
            "all_destination_tensors_covered": True,
        }) or row.get("ema_copy_contract_equal") is not True:
            raise ValueError(f"transfer-copy coverage mismatch: {run_id}")

        for label in ("net", "ema"):
            summary = row.get(label)
            expected_summary_keys = set(EXPECTED_MODULE_SUMMARY) | {"sha256"}
            if not isinstance(summary, dict) or set(summary) != expected_summary_keys:
                raise ValueError(f"canonical {label} summary schema mismatch: {run_id}")
            if (
                not is_sha256(summary.get("sha256"))
                or not exact_json_equal(
                    {key: summary.get(key) for key in EXPECTED_MODULE_SUMMARY},
                    EXPECTED_MODULE_SUMMARY,
                )
            ):
                raise ValueError(f"canonical {label} summary mismatch: {run_id}")
        if row["net"]["sha256"] != row["ema"]["sha256"]:
            raise ValueError(f"reconstructed net/EMA hashes differ: {run_id}")
        run_hashes[run_id] = row["net"]["sha256"]
        contract_hashes[run_id] = row["initialization_contract_sha256"]

    if len(set(run_hashes.values())) != 1 or len(set(contract_hashes.values())) != 1:
        raise ValueError("recomputed cross-run initialization equality failed")
    cross = report.get("cross_run")
    expected_cross = {
        "all_six_reconstructed_net_hashes_equal": True,
        "all_six_initialization_contract_hashes_equal": True,
        "all_six_dataset_interfaces_equal": True,
        "distinct_reconstructed_net_hashes": sorted(set(run_hashes.values())),
        "distinct_initialization_contract_hashes": sorted(
            set(contract_hashes.values())
        ),
        "run_hashes": run_hashes,
    }
    if not exact_json_equal(cross, expected_cross):
        raise ValueError("initialization cross-run derived fields are inconsistent")
    return runs
