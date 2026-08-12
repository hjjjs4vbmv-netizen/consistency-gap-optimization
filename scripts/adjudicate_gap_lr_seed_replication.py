#!/usr/bin/env python3
"""Quality-blind formal adjudication for the seed-4/5 replication package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import gap_lr_seed_replication_contract as audit_contract


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
EXPECTED_RUN_KEYS = {
    "seed4_A",
    "seed4_B",
    "seed4_C",
    "seed5_A",
    "seed5_B",
    "seed5_C",
}
EXPECTED_RUNS = {
    "seed4_A": ("arm_a_g1_0_lr_fixed_s4", 4, "A", 1.0, 0.0001),
    "seed4_B": ("arm_b_g1_3_lr_fixed_s4", 4, "B", 1.3, 0.0001),
    "seed4_C": (
        "arm_c_g1_3_lr_matched_s4",
        4,
        "C",
        1.3,
        0.00012963523762588692,
    ),
    "seed5_A": ("arm_a_g1_0_lr_fixed_s5", 5, "A", 1.0, 0.0001),
    "seed5_B": ("arm_b_g1_3_lr_fixed_s5", 5, "B", 1.3, 0.0001),
    "seed5_C": (
        "arm_c_g1_3_lr_matched_s5",
        5,
        "C",
        1.3,
        0.00012963523762588692,
    ),
}
EXPECTED_RUN_IDS = {item[0] for item in EXPECTED_RUNS.values()}
EXPECTED_ARTIFACT_KEYS = {
    "training_options",
    "stats",
    "train_summary",
    "log",
    "model_init_image",
    "data_image",
    "final_ema_snapshot",
    "final_training_state",
    "protocol_commit",
    "training_code_commit",
    "source_audit_receipt_sha256",
    *{f"network_snapshot_{index:06d}" for index in range(1, 9)},
    *{f"training_state_{index:06d}" for index in range(1, 9)},
}
EXPECTED_DEVIATIONS = {"D1", "D2", "D3", "D4", "D5"}
REQUIRED_EXCLUSIONS = {
    "protocol-exact execution",
    "historically observed bitwise pre-update parameter identity",
    "bitwise training equivalence across devices",
    "throughput, latency, or GPU performance comparisons",
}
FORBIDDEN_PUBLIC_TEXT = (
    "/data/",
    "/Users/",
    "172.16.",
    "ECT001@",
    "GPU-d791",
    "GPU-ef9e",
)
SHA256_KEYS = {
    "sha256",
    "internal_receipt_sha256",
    "verifier_source_sha256",
    "execution_protocol_commit",
    "adjudication_tooling_commit",
    "training_code_commit",
    "source_audit_receipt_sha256",
    "matrix_sha256",
    "dataset_sha256",
    "transfer_checkpoint_sha256",
    "objective_evidence_sha256",
    "initialization_reconstruction_sha256",
    "adjudicator_source_sha256",
    "tool_source_sha256",
    "evidence_builder_source_sha256",
}


def fail(message: str) -> None:
    raise SystemExit("BLIND ADJUDICATION REJECTED: " + message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return audit_contract.load_json_object(path)
    except ValueError as exc:
        fail(str(exc))


def public_text_is_sanitized(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if any(marker in text for marker in FORBIDDEN_PUBLIC_TEXT):
        return False
    try:
        value = audit_contract.loads_strict(text)
    except ValueError:
        return False

    def safe_string(item: str, key: str | None = None) -> bool:
        if key in SHA256_KEYS or (key is not None and key.endswith("_sha256")):
            return bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item))
        forbidden_patterns = (
            r"(?:^|[=:'\"(\s])/(?!/)",
            r"[A-Za-z]:[\/]",
            r"\\\\[^\\\s]+\\",
            r"(?:^|[/\\])\.\.(?:[/\\]|$)",
            r"(?:^|\s)[^\s/@]+@[^\s/]+",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"\bGPU-[0-9a-fA-F-]{16,}\b",
            r"\b[A-Za-z][A-Za-z0-9+.-]*://",
        )
        if any(re.search(pattern, item) for pattern in forbidden_patterns):
            return False
        if "::" in item or item.count(":") >= 4:
            if re.search(r"\b[0-9A-Fa-f:]{3,}\b", item):
                return False
        if re.fullmatch(r"[A-Za-z0-9.-]+", item) and re.search(
            r"(?:^|[-.])(internal|private|localhost|host|server)(?:[-.]|$)",
            item,
            flags=re.IGNORECASE,
        ):
            return False
        return True

    def safe(item: Any, key: str | None = None) -> bool:
        if isinstance(item, dict):
            return all(
                safe_string(str(child_key)) and safe(child, str(child_key))
                for child_key, child in item.items()
            )
        if isinstance(item, list):
            return all(safe(child, key) for child in item)
        if not isinstance(item, str):
            return True
        return safe_string(item, key)

    return safe(value)


def validate_public_receipt(
    receipt: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    arm: str,
    gap: float,
    learning_rate: float,
    run_verifier_source_sha256: str,
) -> None:
    expected_top = {
        "schema_version",
        "receipt_type",
        "status",
        "experiment_id",
        "run_id",
        "seed",
        "arm",
        "gap_scale",
        "learning_rate",
        "bindings",
        "completion",
        "final_training_state",
        "final_ema_snapshot",
        "artifact_manifest",
        "verified_at_utc",
        "publication",
    }
    if set(receipt) != expected_top:
        raise ValueError("public receipt top-level schema mismatch")
    if (
        not audit_contract.is_exact_int(receipt.get("schema_version"))
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_type")
        != "gap_lr_seed_replication_run_integrity_public"
        or receipt.get("status") != "passed"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("run_id") != run_id
        or not audit_contract.is_exact_int(receipt.get("seed"))
        or receipt.get("seed") != seed
        or receipt.get("arm") != arm
        or type(receipt.get("gap_scale")) is not float
        or not audit_contract.is_finite_number(receipt.get("gap_scale"))
        or not math.isclose(receipt["gap_scale"], gap, rel_tol=0.0, abs_tol=1e-18)
        or type(receipt.get("learning_rate")) is not float
        or not audit_contract.is_finite_number(receipt.get("learning_rate"))
        or not math.isclose(
            receipt["learning_rate"], learning_rate, rel_tol=0.0, abs_tol=1e-18
        )
    ):
        raise ValueError("public receipt identity/configuration mismatch")

    bindings = receipt.get("bindings", {})
    expected_bindings = {
        "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
        "training_code_commit": audit_contract.TRAINING_CODE_COMMIT,
        "source_audit_receipt_sha256": audit_contract.SOURCE_AUDIT_RECEIPT_SHA256,
        "matrix_sha256": audit_contract.MATRIX_SHA256,
        "dataset_sha256": audit_contract.DATA_SHA256,
        "transfer_checkpoint_sha256": audit_contract.TRANSFER_SHA256,
        "verifier_source_sha256": run_verifier_source_sha256,
    }
    if (
        set(bindings) != set(expected_bindings) | {"internal_receipt_sha256"}
        or any(bindings.get(key) != value for key, value in expected_bindings.items())
        or not audit_contract.is_sha256(bindings.get("internal_receipt_sha256"))
    ):
        raise ValueError("public receipt frozen-input/verifier binding mismatch")

    completion = receipt.get("completion", {})
    summary = completion.get("summary", {})
    stats = completion.get("stats", {})
    if (
        set(completion) != {"budget_kimg", "summary", "stats"}
        or not audit_contract.is_exact_int(completion.get("budget_kimg"))
        or completion.get("budget_kimg") != 256
        or set(summary)
        != {
            "rows",
            "final_processed_kimg",
            "attempted_iterations",
            "successful_optimizer_steps",
            "amp_skipped_steps",
            "max_allowed_amp_skips",
            "final_gradscaler_scale",
            "amp_contract_passed",
        }
        or not audit_contract.is_exact_int(summary.get("rows"))
        or summary.get("rows") != 2000
        or type(summary.get("final_processed_kimg")) is not float
        or summary.get("final_processed_kimg") != 256.0
        or not audit_contract.is_exact_int(summary.get("attempted_iterations"))
        or summary.get("attempted_iterations") != 2000
        or not audit_contract.is_exact_int(summary.get("amp_skipped_steps"))
        or not 0 <= summary["amp_skipped_steps"] <= 16
        or summary.get("max_allowed_amp_skips") != 16
        or not audit_contract.is_exact_int(summary.get("max_allowed_amp_skips"))
        or not audit_contract.is_exact_int(
            summary.get("successful_optimizer_steps")
        )
        or summary.get("successful_optimizer_steps")
        != 2000 - summary["amp_skipped_steps"]
        or not audit_contract.is_finite_number(
            summary.get("final_gradscaler_scale"), positive=True
        )
        or type(summary.get("final_gradscaler_scale")) is not float
        or summary.get("amp_contract_passed") is not True
        or set(stats) != {"records", "final_kimg"}
        or stats.get("records") != 9
        or not audit_contract.is_exact_int(stats.get("records"), minimum=1)
        or type(stats.get("final_kimg")) is not float
        or not audit_contract.is_finite_number(stats.get("final_kimg"))
        or not math.isclose(stats["final_kimg"], 256.0, rel_tol=0.0, abs_tol=1e-3)
    ):
        raise ValueError("public receipt completion/AMP summary mismatch")

    state = receipt.get("final_training_state", {})
    if (
        set(state)
        != {
            "cur_nimg",
            "attempted_iteration",
            "successful_optimizer_steps",
            "gradscaler_scale",
            "optimizer_parameter_states",
            "tensors_checked",
        }
        or not audit_contract.is_exact_int(state.get("cur_nimg"))
        or state.get("cur_nimg") != 256000
        or not audit_contract.is_exact_int(state.get("attempted_iteration"))
        or state.get("attempted_iteration") != 2000
        or not audit_contract.is_exact_int(state.get("successful_optimizer_steps"))
        or state.get("successful_optimizer_steps")
        != summary.get("successful_optimizer_steps")
        or state.get("gradscaler_scale") != summary.get("final_gradscaler_scale")
        or type(state.get("gradscaler_scale")) is not float
        or not audit_contract.is_exact_int(state.get("optimizer_parameter_states"))
        or state.get("optimizer_parameter_states") != 416
        or not audit_contract.is_exact_int(state.get("tensors_checked"))
        or state.get("tensors_checked") != 1248
    ):
        raise ValueError("public receipt final training-state mismatch")
    snapshot = receipt.get("final_ema_snapshot", {})
    if not audit_contract.exact_json_equal(snapshot, {
        "ema_present": True,
        "ema_finite": True,
        "ema_tensors_checked": 424,
    }):
        raise ValueError("public receipt final EMA mismatch")

    artifacts = receipt.get("artifact_manifest", {})
    if set(artifacts) != EXPECTED_ARTIFACT_KEYS:
        raise ValueError("public receipt artifact manifest key mismatch")
    for item in artifacts.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"sha256", "size_bytes"}
            or not audit_contract.is_sha256(item.get("sha256"))
        or not audit_contract.is_exact_int(item.get("size_bytes"), minimum=1)
            or item["size_bytes"] <= 0
        ):
            raise ValueError("public receipt artifact manifest entry mismatch")
    verified = datetime.fromisoformat(str(receipt.get("verified_at_utc")))
    if verified.tzinfo is None:
        raise ValueError("public receipt verification timestamp lacks timezone")
    if not audit_contract.exact_json_equal(receipt.get("publication"), {
        "sanitized_for_github": True,
        "absolute_paths_removed": True,
        "raw_artifacts_retained_external_to_git": True,
    }):
        raise ValueError("public receipt publication boundary mismatch")


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not an ISO timestamp string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} is not explicitly UTC")
    return parsed


def validate_runtime(
    runtime: dict[str, Any],
    manifest: dict[str, Any],
    public_receipts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, datetime]]:
    if set(runtime) != {
        "planned",
        "hardware",
        "software",
        "launcher_segments",
        "runs",
        "directly_observed_overlaps",
        "clock_model",
    }:
        raise ValueError("runtime schema mismatch")
    if not audit_contract.exact_json_equal(runtime.get("planned"), {
        "seed_order": [4, 5],
        "arm_order": ["A", "B", "C"],
        "gpu_index": 1,
        "execution_mode": "fully_serial",
        "automatic_retry": False,
    }):
        raise ValueError("planned runtime contract mismatch")

    hardware = runtime.get("hardware", {})
    if set(hardware) != {
        "devices",
        "prelaunch_sidecar_entries_equivalent",
        "snapshot_scope",
        "per_run_cuda_uuid_attested",
        "full_gpu_uuids_retained_only_in_internal_sidecar",
    }:
        raise ValueError("runtime hardware schema mismatch")
    devices = hardware.get("devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise ValueError("runtime hardware device count mismatch")
    expected_devices: dict[int, dict[str, Any]] = {}
    signatures = set()
    for row in devices:
        if not isinstance(row, dict) or set(row) != {
            "device_alias",
            "logged_gpu_index",
            "name",
            "driver_version",
            "memory_total",
        }:
            raise ValueError("runtime hardware device schema mismatch")
        index = row.get("logged_gpu_index")
        if (
            not audit_contract.is_exact_int(index)
            or index not in (0, 1)
            or row.get("device_alias") != f"device_{index}"
        ):
            raise ValueError("runtime hardware index/alias mismatch")
        if (
            row.get("name") != "NVIDIA A100 80GB PCIe"
            or row.get("memory_total") != "81920 MiB"
            or not isinstance(row.get("driver_version"), str)
            or re.fullmatch(r"\d+\.\d+\.\d+", row["driver_version"]) is None
        ):
            raise ValueError("runtime hardware entry mismatch")
        if index in expected_devices:
            raise ValueError("duplicate runtime hardware index")
        expected_devices[index] = row
        signatures.add((row["name"], row["driver_version"], row["memory_total"]))
    if (
        set(expected_devices) != {0, 1}
        or len(signatures) != 1
        or hardware.get("prelaunch_sidecar_entries_equivalent") is not True
        or hardware.get("snapshot_scope")
        != "single pre-original-launch sidecar; not per-run attestation"
        or hardware.get("per_run_cuda_uuid_attested") is not False
        or hardware.get("full_gpu_uuids_retained_only_in_internal_sidecar")
        is not True
    ):
        raise ValueError("runtime hardware derived claim mismatch")

    software = runtime.get("software")
    if not isinstance(software, dict) or set(software) != {
        "python",
        "torch",
        "cuda",
        "cudnn",
    }:
        raise ValueError("runtime software schema mismatch")
    software_patterns = {
        "python": r"\d+\.\d+\.\d+",
        "torch": r"\d+\.\d+\.\d+(?:\+[A-Za-z0-9.]+)?",
        "cuda": r"\d+\.\d+",
        "cudnn": r"\d+",
    }
    if any(
        not isinstance(software[key], str)
        or re.fullmatch(pattern, software[key]) is None
        for key, pattern in software_patterns.items()
    ):
        raise ValueError("runtime software value mismatch")

    segments = runtime.get("launcher_segments")
    if not isinstance(segments, list) or len(segments) != 3:
        raise ValueError("launcher segment count mismatch")
    segments_by_id: dict[str, dict[str, Any]] = {}
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("launcher segment entry is not an object")
        segment_id = row.get("segment_id")
        common = {
            "segment_id",
            "kind",
            "launcher_log_sha256",
            "launcher_log_size_bytes",
            "exact_command_preserved",
            "committed_launcher_reconstructible",
            "events",
        }
        expected_keys = common if segment_id == "original" else common | {
            "pid",
            "pid_file_sha256",
            "pid_file_mtime_filesystem_clock_utc",
        }
        if set(row) != expected_keys or segment_id in segments_by_id:
            raise ValueError("launcher segment schema/identity mismatch")
        if (
            segment_id not in {"original", "seed4_recovery", "seed5_recovery"}
            or row.get("kind")
            != ("original" if segment_id == "original" else "manual_recovery")
            or not audit_contract.is_sha256(row.get("launcher_log_sha256"))
            or not audit_contract.is_exact_int(
                row.get("launcher_log_size_bytes"), minimum=1
            )
            or row.get("exact_command_preserved") is not False
            or row.get("committed_launcher_reconstructible")
            is not (segment_id == "original")
            or not audit_contract.exact_json_equal(
                row.get("events"),
                audit_contract.expected_launcher_events(str(segment_id)),
            )
        ):
            raise ValueError("launcher segment content mismatch")
        if segment_id != "original":
            if (
                not audit_contract.is_exact_int(row.get("pid"), minimum=1)
                or not audit_contract.is_sha256(row.get("pid_file_sha256"))
            ):
                raise ValueError("recovery launcher PID evidence mismatch")
            _utc_datetime(
                row.get("pid_file_mtime_filesystem_clock_utc"),
                f"{segment_id} PID mtime",
            )
        segments_by_id[segment_id] = row
    if set(segments_by_id) != {"original", "seed4_recovery", "seed5_recovery"}:
        raise ValueError("launcher segment identity set mismatch")

    runs = runtime.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(audit_contract.EXPECTED_RUNTIME):
        raise ValueError("runtime run identity set mismatch")
    parsed: dict[str, dict[str, datetime]] = {}
    ports = set()
    for run_id, expected in audit_contract.EXPECTED_RUNTIME.items():
        row = runs[run_id]
        expected_keys = {
            "seed",
            "arm",
            "segment_id",
            "port",
            "logged_gpu_index",
            "device_alias",
            "gpu_evidence",
            "first_progress_at_utc",
            "last_progress_at_utc",
            "exit_marker_at_utc",
            "interval_definition",
            "interval_scope",
            "exit_timestamp_source",
            "exit_timestamp_timezone_assumption",
            "strengthened_integrity_verified_at_utc",
            "historical_integrity_receipt",
            "public_receipt",
            "public_receipt_sha256",
            "external_training_log_sha256",
        }
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError(f"runtime row schema mismatch: {run_id}")
        observed_mapping = {
            field: row.get(field)
            for field in ("seed", "arm", "segment_id", "port", "logged_gpu_index")
        }
        expected_mapping = {
            field: expected[field]
            for field in ("seed", "arm", "segment_id", "port", "logged_gpu_index")
        }
        if not audit_contract.exact_json_equal(observed_mapping, expected_mapping):
            raise ValueError(f"runtime matrix mismatch: {run_id}")
        ports.add(row["port"])
        index = row["logged_gpu_index"]
        if (
            row.get("device_alias") != f"device_{index}"
            or not audit_contract.exact_json_equal(row.get("gpu_evidence"), {
                "logged_index": "direct launcher assertion",
                "uuid_mapping": (
                    "inference from one pre-launch hardware sidecar; index stability "
                    "and per-run UUID were not attested"
                ),
            })
            or row.get("interval_definition")
            != "first stats record through timestamped exit marker"
            or row.get("interval_scope")
            != "observed training-phase lower bound, not exact process lifetime"
            or row.get("exit_timestamp_source")
            != "destroy_process_group warning immediately after clean Exiting marker"
            or row.get("exit_timestamp_timezone_assumption")
            != "Asia/Shanghai (UTC+08:00)"
            or not audit_contract.is_sha256(row.get("public_receipt_sha256"))
            or not audit_contract.is_sha256(row.get("external_training_log_sha256"))
        ):
            raise ValueError(f"runtime evidence boundary mismatch: {run_id}")
        public_key = f"seed{expected['seed']}_{expected['arm']}"
        expected_manifest = manifest.get(public_key, {})
        historical = row.get("historical_integrity_receipt")
        expected_historical_role = (
            "posthoc_reverification_after_original_launcher_exit"
            if run_id == "arm_a_g1_0_lr_fixed_s4"
            else "inline_before_launcher_done"
        )
        if (
            not isinstance(historical, dict)
            or set(historical)
            != {
                "role",
                "schema_version",
                "receipt_sha256",
                "receipt_size_bytes",
                "verified_at_utc",
                "artifact_manifest_equal_to_strengthened_receipt",
                "retained_external_to_git",
            }
            or historical.get("role") != expected_historical_role
            or not audit_contract.is_exact_int(historical.get("schema_version"))
            or historical.get("schema_version") != 1
            or not audit_contract.is_sha256(historical.get("receipt_sha256"))
            or not audit_contract.is_exact_int(
                historical.get("receipt_size_bytes"), minimum=1
            )
            or historical.get("artifact_manifest_equal_to_strengthened_receipt")
            is not True
            or historical.get("retained_external_to_git") is not True
        ):
            raise ValueError(f"historical receipt evidence mismatch: {run_id}")
        if (
            row.get("public_receipt") != expected_manifest.get("file")
            or row.get("public_receipt_sha256") != expected_manifest.get("sha256")
            or row.get("strengthened_integrity_verified_at_utc")
            != public_receipts.get(public_key, {}).get("verified_at_utc")
        ):
            raise ValueError(f"runtime/public receipt binding mismatch: {run_id}")
        first = _utc_datetime(row.get("first_progress_at_utc"), f"{run_id} first")
        last = _utc_datetime(row.get("last_progress_at_utc"), f"{run_id} last")
        exit_at = _utc_datetime(row.get("exit_marker_at_utc"), f"{run_id} exit")
        strengthened_verified = _utc_datetime(
            row.get("strengthened_integrity_verified_at_utc"),
            f"{run_id} strengthened verified",
        )
        historical_verified = _utc_datetime(
            historical.get("verified_at_utc"), f"{run_id} historical verified"
        )
        if not first <= last <= exit_at <= strengthened_verified:
            raise ValueError(f"runtime timestamps are non-monotonic: {run_id}")
        parsed[run_id] = {
            "first": first,
            "last": last,
            "exit": exit_at,
            "strengthened_verified": strengthened_verified,
            "historical_verified": historical_verified,
        }
    if len(ports) != 6:
        raise ValueError("runtime ports are not unique")

    a4 = parsed["arm_a_g1_0_lr_fixed_s4"]
    b4 = parsed["arm_b_g1_3_lr_fixed_s4"]
    c4 = parsed["arm_c_g1_3_lr_matched_s4"]
    a5 = parsed["arm_a_g1_0_lr_fixed_s5"]
    b5 = parsed["arm_b_g1_3_lr_fixed_s5"]
    c5 = parsed["arm_c_g1_3_lr_matched_s5"]
    if not (
        a4["exit"] <= min(b4["first"], a5["first"])
        and b4["exit"] <= c4["first"]
        and a5["exit"] <= b5["first"]
        and b5["exit"] <= c5["first"]
        and a4["historical_verified"] <= min(b4["first"], a5["first"])
        and b4["historical_verified"] <= c4["first"]
        and a5["historical_verified"] <= b5["first"]
        and b5["historical_verified"] <= c5["first"]
    ):
        raise ValueError("cross-run launcher/verification chronology mismatch")

    recomputed: dict[frozenset[str], dict[str, Any]] = {}
    run_ids = list(audit_contract.EXPECTED_RUNTIME)
    for index, left_id in enumerate(run_ids):
        for right_id in run_ids[index + 1 :]:
            start = max(parsed[left_id]["first"], parsed[right_id]["first"])
            end = min(parsed[left_id]["exit"], parsed[right_id]["exit"])
            if start < end:
                pair = frozenset((left_id, right_id))
                recomputed[pair] = {"start": start, "end": end}
    if set(recomputed) != audit_contract.EXPECTED_OVERLAP_PAIRS:
        raise ValueError("recomputed runtime overlap pair set mismatch")
    reported = runtime.get("directly_observed_overlaps")
    if not isinstance(reported, list) or len(reported) != 2:
        raise ValueError("reported runtime overlap count mismatch")
    seen_pairs = set()
    for row in reported:
        if not isinstance(row, dict) or set(row) != {
            "runs",
            "directly_observed_overlap_start_utc",
            "directly_observed_overlap_end_utc",
            "duration_seconds",
            "logged_gpu_indices",
            "different_logged_gpu_indices",
        }:
            raise ValueError("reported overlap schema mismatch")
        listed = row.get("runs")
        if (
            not isinstance(listed, list)
            or len(listed) != 2
            or listed[0] == listed[1]
        ):
            raise ValueError("reported overlap run pair mismatch")
        pair = frozenset(listed)
        if pair not in recomputed or pair in seen_pairs:
            raise ValueError("reported overlap identity mismatch")
        seen_pairs.add(pair)
        expected = recomputed[pair]
        start = _utc_datetime(
            row.get("directly_observed_overlap_start_utc"), "overlap start"
        )
        end = _utc_datetime(
            row.get("directly_observed_overlap_end_utc"), "overlap end"
        )
        expected_indices = [
            audit_contract.EXPECTED_RUNTIME[run_id]["logged_gpu_index"]
            for run_id in listed
        ]
        if (
            start != expected["start"]
            or end != expected["end"]
            or not audit_contract.is_finite_number(
                row.get("duration_seconds"), positive=True
            )
            or type(row.get("duration_seconds")) is not float
            or not math.isclose(
                row["duration_seconds"],
                (end - start).total_seconds(),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not audit_contract.exact_json_equal(
                row.get("logged_gpu_indices"), expected_indices
            )
            or row.get("different_logged_gpu_indices")
            is not (expected_indices[0] != expected_indices[1])
            or expected_indices[0] == expected_indices[1]
        ):
            raise ValueError("reported overlap derived fields mismatch")
    if seen_pairs != set(recomputed):
        raise ValueError("reported overlap set is incomplete")

    clock = runtime.get("clock_model", {})
    offsets = clock.get("observed_filesystem_minus_application_offset_seconds", {})
    if (
        set(clock)
        != {
            "application_clock_sources",
            "filesystem_clock_not_used_as_application_time",
            "acceptance_gate",
            "observed_filesystem_minus_application_offset_seconds",
        }
        or clock.get("application_clock_sources")
        != [
            "stats.jsonl epoch timestamps",
            "timestamped process-exit warning",
            "historical and strengthened receipt verified_at_utc",
        ]
        or clock.get("filesystem_clock_not_used_as_application_time") is not True
        or clock.get("acceptance_gate")
        != "diagnostic only; no exact process-start claim"
        or not isinstance(offsets, dict)
        or set(offsets) != {"minimum", "maximum"}
        or not audit_contract.is_finite_number(offsets.get("minimum"))
        or not audit_contract.is_finite_number(offsets.get("maximum"))
        or type(offsets.get("minimum")) is not float
        or type(offsets.get("maximum")) is not float
        or not 30 <= offsets["minimum"] <= offsets["maximum"] <= 120
        or offsets["maximum"] - offsets["minimum"] > 2
    ):
        raise ValueError("runtime clock diagnostic mismatch")
    return parsed


def validate_deviations(evidence: dict[str, Any]) -> None:
    deviations = evidence.get("deviations")
    if not audit_contract.exact_json_equal(
        deviations, audit_contract.expected_deviations()
    ):
        raise ValueError("canonical deviation records mismatch")

    missing = evidence.get("missing_evidence")
    if missing != [
        "historical post-transfer/pre-forward parameter hash",
        "per-run CUDA UUID attestation",
        "per-run recovery software snapshot",
        "complete recovery command lines and process environments",
        "original seed4-A verifier failure output and exit status",
    ]:
        raise ValueError("missing-evidence inventory mismatch")


def validate_initialization_evidence(
    evidence_init: dict[str, Any],
    initialization: dict[str, Any],
    initialization_report_sha256: str,
) -> None:
    if not isinstance(evidence_init, dict) or set(evidence_init) != {
        "historical_observed_preupdate_parameter_hash",
        "reconstructed_expected_initialization",
        "model_init_previews",
        "data_images",
    }:
        raise ValueError("initialization evidence schema mismatch")
    reconstructed = evidence_init.get("reconstructed_expected_initialization")
    distinct = initialization["cross_run"]["distinct_reconstructed_net_hashes"]
    if not audit_contract.exact_json_equal(reconstructed, {
        "status": "passed",
        "report_sha256": initialization_report_sha256,
        "hash_kind": "reconstructed_expected_initialization_hash",
        "all_six_equal": True,
        "distinct_hashes": distinct,
        "historical_process_attestation": False,
    }) or evidence_init.get("historical_observed_preupdate_parameter_hash") != (
        "not_captured"
    ):
        raise ValueError("initialization reconstruction evidence mismatch")

    previews = evidence_init.get("model_init_previews")
    if not isinstance(previews, dict) or set(previews) != {"4", "5"}:
        raise ValueError("model-init preview seed set mismatch")
    run_ids = {
        "4": {
            "A": "arm_a_g1_0_lr_fixed_s4",
            "B": "arm_b_g1_3_lr_fixed_s4",
            "C": "arm_c_g1_3_lr_matched_s4",
        },
        "5": {
            "A": "arm_a_g1_0_lr_fixed_s5",
            "B": "arm_b_g1_3_lr_fixed_s5",
            "C": "arm_c_g1_3_lr_matched_s5",
        },
    }
    for seed, expected_ids in run_ids.items():
        row = previews[seed]
        if not isinstance(row, dict) or set(row) != {
            "sha256",
            "pairwise_against_A",
            "exact_file_hashes_equal",
            "max_abs_channel_delta_lsb",
            "role",
        }:
            raise ValueError(f"model-init preview schema mismatch: seed {seed}")
        hashes = row.get("sha256")
        pairs = row.get("pairwise_against_A")
        if (
            not isinstance(hashes, dict)
            or set(hashes) != {"A", "B", "C"}
            or not all(audit_contract.is_sha256(value) for value in hashes.values())
            or not isinstance(pairs, list)
            or len(pairs) != 2
            or row.get("role")
            != "generated FP16 diagnostic preview; not a parameter hash"
            or row.get("exact_file_hashes_equal") is not (len(set(hashes.values())) == 1)
        ):
            raise ValueError(f"model-init preview content mismatch: seed {seed}")
        observed_max = 0
        for arm, pair in zip(("B", "C"), pairs):
            if not isinstance(pair, dict) or set(pair) != {
                "first",
                "second",
                "shape_hwc",
                "exact_pixel_values_equal",
                "max_abs_channel_delta_lsb",
                "differing_channel_values",
                "differing_pixels",
                "positive_one_lsb",
                "negative_one_lsb",
                "greater_than_one_lsb",
            }:
                raise ValueError(f"model-init pair schema mismatch: seed {seed} arm {arm}")
            numeric_keys = (
                "max_abs_channel_delta_lsb",
                "differing_channel_values",
                "differing_pixels",
                "positive_one_lsb",
                "negative_one_lsb",
                "greater_than_one_lsb",
            )
            shape = pair.get("shape_hwc")
            if (
                pair.get("first") != expected_ids["A"]
                or pair.get("second") != expected_ids[arm]
                or not isinstance(shape, list)
                or len(shape) != 3
                or shape[-1] != 3
                or not all(audit_contract.is_exact_int(value, minimum=1) for value in shape)
                or any(
                    not audit_contract.is_exact_int(pair.get(key), minimum=0)
                    for key in numeric_keys
                )
                or pair["max_abs_channel_delta_lsb"] > 1
                or pair["greater_than_one_lsb"] != 0
                or pair["differing_channel_values"]
                != pair["positive_one_lsb"] + pair["negative_one_lsb"]
                or pair["differing_pixels"] > pair["differing_channel_values"]
                or pair.get("exact_pixel_values_equal")
                is not (pair["differing_channel_values"] == 0)
                or (
                    pair["max_abs_channel_delta_lsb"] == 0
                    and pair["differing_channel_values"] != 0
                )
                or (
                    pair["max_abs_channel_delta_lsb"] == 1
                    and pair["differing_channel_values"] == 0
                )
            ):
                raise ValueError(f"model-init pair values mismatch: seed {seed} arm {arm}")
            observed_max = max(observed_max, pair["max_abs_channel_delta_lsb"])
        if (
            not audit_contract.is_exact_int(
                row.get("max_abs_channel_delta_lsb"), minimum=0
            )
            or row.get("max_abs_channel_delta_lsb") != observed_max
        ):
            raise ValueError(f"model-init preview derived max mismatch: seed {seed}")

    data_images = evidence_init.get("data_images")
    if (
        not isinstance(data_images, dict)
        or set(data_images) != {"all_six_sha256_equal", "sha256"}
        or data_images.get("all_six_sha256_equal") is not True
        or not audit_contract.is_sha256(data_images.get("sha256"))
    ):
        raise ValueError("data-image identity evidence mismatch")


def validate_evidence_manifest(
    evidence_manifest: dict[str, Any], initialization_report_sha256: str
) -> None:
    expected_files = {
        "launch_provenance": "launch_provenance.txt",
        "hardware": "hardware.txt",
        "software": "software.txt",
        "original_launcher_log": (
            "gap_lr_matched_q128_s45_replication_v1.launcher.log"
        ),
        "seed4_recovery_launcher_log": "seed4_resume.launcher.log",
        "seed5_recovery_launcher_log": "seed5_resume.launcher.log",
        "initialization_reconstruction": "initialization_reconstruction.json",
    }
    if not isinstance(evidence_manifest, dict) or set(evidence_manifest) != set(
        expected_files
    ):
        raise ValueError("evidence manifest identity set mismatch")
    for label, filename in expected_files.items():
        row = evidence_manifest[label]
        if (
            not isinstance(row, dict)
            or set(row) != {"file", "sha256", "size_bytes", "retained_external_to_git"}
            or row.get("file") != filename
            or not audit_contract.is_sha256(row.get("sha256"))
            or not audit_contract.is_exact_int(row.get("size_bytes"), minimum=1)
            or row.get("retained_external_to_git")
            is not (label != "initialization_reconstruction")
        ):
            raise ValueError(f"evidence manifest entry mismatch: {label}")
    if (
        evidence_manifest["initialization_reconstruction"]["sha256"]
        != initialization_report_sha256
    ):
        raise ValueError("evidence manifest initialization hash mismatch")


def evaluate(
    evidence: dict[str, Any],
    initialization: dict[str, Any],
    public_receipts: dict[str, dict[str, Any]],
    *,
    tooling_commit: str,
    reconstruction_source_sha256: str,
    evidence_builder_source_sha256: str,
    run_verifier_source_sha256: str,
) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    affected_runs: set[str] = set()

    expected_evidence_keys = {
        "schema_version",
        "receipt_type",
        "status",
        "experiment_id",
        "quality_blind",
        "bindings",
        "per_run_integrity",
        "configuration_contract",
        "initialization",
        "runtime",
        "deviations",
        "missing_evidence",
        "claim_exclusions",
        "evidence_manifest",
        "publication",
    }
    evidence_bindings = evidence.get("bindings", {})
    expected_binding_values = {
        "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
        "adjudication_tooling_commit": tooling_commit,
        "training_code_commit": audit_contract.TRAINING_CODE_COMMIT,
        "source_audit_receipt_sha256": audit_contract.SOURCE_AUDIT_RECEIPT_SHA256,
        "matrix_sha256": audit_contract.MATRIX_SHA256,
        "dataset_sha256": audit_contract.DATA_SHA256,
        "transfer_checkpoint_sha256": audit_contract.TRANSFER_SHA256,
        "evidence_builder_source_sha256": evidence_builder_source_sha256,
    }
    if (
        set(evidence) != expected_evidence_keys
        or not isinstance(evidence_bindings, dict)
        or not audit_contract.is_exact_int(evidence.get("schema_version"))
        or evidence.get("receipt_type")
        != "gap_lr_seed_replication_quality_blind_evidence"
        or evidence.get("schema_version") != 1
        or evidence.get("status") != "adjudication_ready"
        or evidence.get("experiment_id") != EXPERIMENT_ID
        or any(
            evidence_bindings.get(key) != value
            for key, value in expected_binding_values.items()
        )
        or set(evidence_bindings)
        != set(expected_binding_values) | {"initialization_reconstruction_sha256"}
        or not audit_contract.is_sha256(
            evidence_bindings.get("initialization_reconstruction_sha256")
        )
    ):
        failures.append("objective evidence identity/binding failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
    if not audit_contract.exact_json_equal(evidence.get("quality_blind"), {
        "generation_quality_metrics_accessed": False,
        "decision_frozen_before_quality_evaluation": True,
        "attestation_kind": "workflow-scope declaration; not cryptographic proof",
        "excluded_inputs": ["FID", "KID", "quality-evaluation outputs"],
    }):
        failures.append("quality-blind boundary was not preserved")
        affected_runs.update(EXPECTED_RUN_KEYS)
    if not audit_contract.exact_json_equal(evidence.get("publication"), {
        "sanitized_for_github": True,
        "absolute_paths_hostnames_accounts_ips_and_full_gpu_uuids_removed": True,
    }):
        failures.append("public evidence publication boundary failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
    try:
        validate_evidence_manifest(
            evidence.get("evidence_manifest", {}),
            evidence_bindings.get("initialization_reconstruction_sha256", ""),
        )
    except (KeyError, TypeError, ValueError):
        failures.append("objective evidence manifest failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    integrity = evidence.get("per_run_integrity", {})
    manifest = integrity.get("public_receipts", {}) if isinstance(integrity, dict) else {}
    if (
        not isinstance(integrity, dict)
        or not isinstance(manifest, dict)
        or set(integrity) != {
            "passed_runs",
            "required_runs",
            "all_artifact_hashes_recomputed",
            "public_receipts",
        }
        or not audit_contract.is_exact_int(integrity.get("passed_runs"))
        or not audit_contract.is_exact_int(integrity.get("required_runs"))
        or integrity.get("passed_runs") != 6
        or integrity.get("required_runs") != 6
        or integrity.get("all_artifact_hashes_recomputed") is not True
        or set(manifest) != EXPECTED_RUN_KEYS
        or set(public_receipts) != EXPECTED_RUN_KEYS
    ):
        failures.append("six-run artifact integrity gate failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
    for key in EXPECTED_RUN_KEYS & set(public_receipts):
        receipt = public_receipts[key]
        run_id, seed, arm, gap, lr = EXPECTED_RUNS[key]
        entry = manifest.get(key, {})
        try:
            validate_public_receipt(
                receipt,
                run_id=run_id,
                seed=seed,
                arm=arm,
                gap=gap,
                learning_rate=lr,
                run_verifier_source_sha256=run_verifier_source_sha256,
            )
        except (TypeError, ValueError):
            receipt_valid = False
        else:
            receipt_valid = True
        if (
            not receipt_valid
            or set(entry)
            != {"run_id", "file", "sha256", "internal_receipt_sha256"}
            or entry.get("run_id") != run_id
            or not isinstance(entry.get("file"), str)
            or not audit_contract.is_sha256(entry.get("sha256"))
            or entry.get("internal_receipt_sha256")
            != receipt.get("bindings", {}).get("internal_receipt_sha256")
        ):
            failures.append(f"public per-run receipt failed: {key}")
            affected_runs.add(key)

    config = evidence.get("configuration_contract", {})
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "within_seed_allowed_differences",
            "within_seed_passed",
            "between_seed_allowed_differences",
            "between_seed_passed",
        }
        or config.get("within_seed_allowed_differences")
        != [
            "loss_kwargs.global_gap_scale",
            "optimizer_kwargs.lr",
            "run_dir",
        ]
        or not audit_contract.exact_json_equal(
            config.get("within_seed_passed"), {"4": True, "5": True}
        )
        or config.get("between_seed_allowed_differences") != ["seed", "run_dir"]
        or not audit_contract.exact_json_equal(
            config.get("between_seed_passed"),
            {"A": True, "B": True, "C": True},
        )
    ):
        failures.append("allowed-difference configuration contract failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    try:
        init_runs = audit_contract.validate_initialization_report(
            initialization,
            tooling_commit=tooling_commit,
            reconstruction_source_sha256=reconstruction_source_sha256,
        )
    except ValueError:
        failures.append("expected initialization reconstruction failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
        init_runs = {}

    for key in EXPECTED_RUN_KEYS & set(public_receipts):
        run_id = EXPECTED_RUNS[key][0]
        init_row = init_runs.get(run_id, {})
        public = public_receipts[key]
        if (
            init_row.get("internal_integrity_receipt_sha256")
            != public.get("bindings", {}).get("internal_receipt_sha256")
            or init_row.get("training_options_sha256")
            != public.get("artifact_manifest", {})
            .get("training_options", {})
            .get("sha256")
        ):
            failures.append(f"initialization/public receipt cross-binding failed: {key}")
            affected_runs.add(key)

    evidence_init = evidence.get("initialization", {})
    try:
        validate_initialization_evidence(
            evidence_init,
            initialization,
            evidence_bindings.get("initialization_reconstruction_sha256", ""),
        )
    except (KeyError, TypeError, ValueError):
        failures.append("initialization evidence boundary/content failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    runtime = evidence.get("runtime", {})
    try:
        validate_runtime(runtime, manifest, public_receipts)
    except (KeyError, TypeError, ValueError):
        failures.append("runtime-deviation acceptance conditions failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    try:
        validate_deviations(evidence)
    except (KeyError, TypeError, ValueError):
        failures.append("deviation set is incomplete")
        affected_runs.update(EXPECTED_RUN_KEYS)
    if evidence.get("claim_exclusions") != sorted(REQUIRED_EXCLUSIONS):
        failures.append("required claim exclusions are missing")
        affected_runs.update(EXPECTED_RUN_KEYS)

    verdict = "rerun_required" if failures else "machine_recommends_acceptance"
    return verdict, failures, sorted(affected_runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--initialization-reconstruction", required=True, type=Path)
    parser.add_argument("--public-receipt-dir", required=True, type=Path)
    parser.add_argument("--adjudication-tooling-commit", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.adjudication_tooling_commit):
        fail("adjudication tooling commit must be a full Git SHA")
    try:
        tooling_blobs = audit_contract.validate_tooling_checkout(
            args.repo,
            args.adjudication_tooling_commit,
            executed_file=Path(__file__),
            expected_relative="scripts/adjudicate_gap_lr_seed_replication.py",
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        fail(str(exc))

    evidence = load_json(args.evidence)
    initialization = load_json(args.initialization_reconstruction)
    manifest = evidence.get("per_run_integrity", {}).get("public_receipts", {})
    receipts: dict[str, dict[str, Any]] = {}
    receipt_bindings: dict[str, Any] = {}
    for key in sorted(EXPECTED_RUN_KEYS):
        entry = manifest.get(key, {})
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            fail(f"unsafe or missing public receipt filename for {key}")
        path = args.public_receipt_dir / filename
        if not path.is_file() or not public_text_is_sanitized(path):
            fail(f"public receipt is missing or unsanitized: {key}")
        digest = file_sha256(path)
        if digest != entry.get("sha256"):
            fail(f"public receipt hash mismatch: {key}")
        receipts[key] = load_json(path)
        receipt_bindings[key] = {"file": filename, "sha256": digest}

    if not public_text_is_sanitized(args.evidence) or not public_text_is_sanitized(
        args.initialization_reconstruction
    ):
        fail("public evidence package contains a forbidden internal identifier")
    if (
        evidence.get("bindings", {}).get("initialization_reconstruction_sha256")
        != file_sha256(args.initialization_reconstruction)
    ):
        fail("initialization reconstruction hash mismatch")
    if (
        evidence.get("bindings", {}).get("adjudication_tooling_commit")
        != args.adjudication_tooling_commit
        or initialization.get("bindings", {}).get("adjudication_tooling_commit")
        != args.adjudication_tooling_commit
    ):
        fail("adjudication tooling commit binding mismatch")

    verdict, failures, affected_runs = evaluate(
        evidence,
        initialization,
        receipts,
        tooling_commit=args.adjudication_tooling_commit,
        reconstruction_source_sha256=tooling_blobs[
            "scripts/reconstruct_gap_lr_seed_initialization.py"
        ],
        evidence_builder_source_sha256=tooling_blobs[
            "scripts/build_gap_lr_seed_replication_blind_evidence.py"
        ],
        run_verifier_source_sha256=tooling_blobs[
            "scripts/verify_gap_lr_seed_replication_run.py"
        ],
    )
    receipt = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_blind_adjudication",
        "status": "machine_recommendation_ready",
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_blind": {
            "generation_quality_metrics_accessed": False,
            "decision_frozen_before_quality_evaluation": True,
            "attestation_kind": "workflow-scope declaration; not cryptographic proof",
        },
        "bindings": {
            "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
            "adjudication_tooling_commit": args.adjudication_tooling_commit,
            "objective_evidence_sha256": file_sha256(args.evidence),
            "initialization_reconstruction_sha256": file_sha256(
                args.initialization_reconstruction
            ),
            "public_per_run_receipts": receipt_bindings,
            "adjudicator_source_sha256": tooling_blobs[
                "scripts/adjudicate_gap_lr_seed_replication.py"
            ],
        },
        "decision_policy": {
            "accept_if": [
                "all six runs pass strengthened integrity and artifact rehash",
                "only preregistered option differences are present",
                "all destination tensors are covered by the frozen transfer",
                "all six reconstructed expected initialization hashes match",
                "preview drift is no greater than one 8-bit level",
                "overlapping runs use different logged indices whose one pre-launch sidecar entries are equivalent A100 devices",
                "all deviations and claim exclusions are explicit",
            ],
            "rerun_if": [
                "any integrity/configuration/hash binding fails",
                "transfer coverage or reconstructed initialization equality fails",
                "preview drift exceeds one 8-bit level",
                "overlap occurs on the same logged GPU or non-equivalent devices",
            ],
        },
        "decision": {
            "protocol_exact": False,
            "scientific_replication_use": False,
            "quality_evaluation_seed4_seed5_authorized": False,
            "performance_benchmark_use": False,
            "historical_bitwise_initialization_claim": False,
            "rerun_affected_runs": affected_runs,
            "failed_conditions": failures,
        },
        "documented_deviations": ["D1", "D2", "D3", "D4", "D5"],
        "claim_exclusions": sorted(REQUIRED_EXCLUSIONS),
        "adjudicator": {
            "role": "Collaborator",
            "independent_external_signature_present": False,
            "machine_policy_evaluated": True,
        },
        "next_step": (
            "obtain independent quality-blind review bound to this candidate "
            "receipt before issuing accepted_with_documented_deviation"
            if verdict == "machine_recommends_acceptance"
            else "rerun the identified affected runs before quality evaluation"
        ),
        "publication": {
            "sanitized_for_github": True,
            "raw_training_and_quality_artifacts_committed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if verdict == "rerun_required":
        raise SystemExit(4)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        fail(str(exc))
