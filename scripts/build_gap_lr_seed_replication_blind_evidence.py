#!/usr/bin/env python3
"""Build sanitized, quality-blind evidence for seed-replication adjudication."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts import verify_gap_lr_seed_replication_run as run_verifier


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
SOURCE_AUDIT_RECEIPT_SHA256 = "6487fbcc5f63817c8e3a91968f45fb13437d1c580afa73966bdf0ad8061bb9fa"
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
LOCAL_TZ = timezone(timedelta(hours=8))


def fail(message: str) -> None:
    raise SystemExit("BLIND EVIDENCE REJECTED: " + message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def same_number(value: Any, expected: float, label: str) -> None:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        fail(f"{label} must be numeric: {exc}")
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-18):
        fail(f"{label}={observed!r}, expected {expected!r}")


def validate_internal_receipt(
    receipt_path: Path,
    run_dir: Path,
    seed: int,
    arm: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema_version") != 2
        or receipt.get("receipt_type") != "gap_lr_seed_replication_run_integrity"
        or receipt.get("status") != "passed"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("seed") != seed
        or receipt.get("arm") != arm
        or receipt.get("execution_protocol_commit") != EXECUTION_PROTOCOL_COMMIT
        or receipt.get("training_code_commit") != TRAINING_CODE_COMMIT
        or Path(receipt.get("run_dir", "")).resolve() != run_dir.resolve()
    ):
        fail(f"internal receipt identity mismatch for seed {seed} arm {arm}")
    completion = receipt.get("completion", {})
    summary = completion.get("summary", {})
    state = receipt.get("final_training_state", {})
    snapshot = receipt.get("final_ema_snapshot", {})
    if (
        completion.get("budget_kimg") != 256
        or summary.get("rows") != 2000
        or summary.get("attempted_iterations") != 2000
        or summary.get("amp_contract_passed") is not True
        or not 0 <= int(summary.get("amp_skipped_steps", -1)) <= 16
        or state.get("cur_nimg") != 256000
        or state.get("attempted_iteration") != 2000
        or state.get("successful_optimizer_steps")
        != summary.get("successful_optimizer_steps")
        or state.get("gradscaler_scale") != summary.get("final_gradscaler_scale")
        or state.get("optimizer_parameter_states") != 416
        or state.get("tensors_checked") != 1248
        or snapshot.get("ema_present") is not True
        or snapshot.get("ema_finite") is not True
        or snapshot.get("ema_tensors_checked") != 424
    ):
        fail(f"completion/state/AMP contract mismatch for seed {seed} arm {arm}")

    expected_source_hash = file_sha256(
        Path(__file__).with_name("verify_gap_lr_seed_replication_run.py")
    )
    if receipt.get("verifier", {}).get("source_sha256") != expected_source_hash:
        fail(f"verifier source hash mismatch for seed {seed} arm {arm}")

    paths = run_verifier.require_files(run_dir.resolve())
    expected_keys = set(paths)
    hashes = receipt.get("artifact_sha256", {})
    sizes = receipt.get("artifact_size_bytes", {})
    if set(hashes) != expected_keys or set(sizes) != expected_keys:
        fail(f"artifact manifest key mismatch for seed {seed} arm {arm}")
    for name, path in sorted(paths.items()):
        if sizes[name] != path.stat().st_size:
            fail(f"artifact size mismatch for {seed}{arm}:{name}")
        if hashes[name] != file_sha256(path):
            fail(f"artifact SHA256 mismatch for {seed}{arm}:{name}")
    return receipt, paths


def public_receipt(
    internal: dict[str, Any], internal_path: Path, run_id: str
) -> dict[str, Any]:
    artifacts = {
        name: {
            "sha256": digest,
            "size_bytes": internal["artifact_size_bytes"][name],
        }
        for name, digest in sorted(internal["artifact_sha256"].items())
    }
    return {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_run_integrity_public",
        "status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "seed": internal["seed"],
        "arm": internal["arm"],
        "gap_scale": internal["gap_scale"],
        "learning_rate": internal["learning_rate"],
        "bindings": {
            "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
            "training_code_commit": TRAINING_CODE_COMMIT,
            "source_audit_receipt_sha256": SOURCE_AUDIT_RECEIPT_SHA256,
            "matrix_sha256": MATRIX_SHA256,
            "dataset_sha256": DATA_SHA256,
            "transfer_checkpoint_sha256": TRANSFER_SHA256,
            "internal_receipt_sha256": file_sha256(internal_path),
            "verifier_source_sha256": internal["verifier"]["source_sha256"],
        },
        "completion": {
            "budget_kimg": internal["completion"]["budget_kimg"],
            "summary": {
                key: internal["completion"]["summary"][key]
                for key in (
                    "rows",
                    "final_processed_kimg",
                    "attempted_iterations",
                    "successful_optimizer_steps",
                    "amp_skipped_steps",
                    "max_allowed_amp_skips",
                    "final_gradscaler_scale",
                    "amp_contract_passed",
                )
            },
            "stats": {
                key: internal["completion"]["stats"][key]
                for key in ("records", "final_kimg")
            },
        },
        "final_training_state": {
            key: internal["final_training_state"][key]
            for key in (
                "cur_nimg",
                "attempted_iteration",
                "successful_optimizer_steps",
                "gradscaler_scale",
                "optimizer_parameter_states",
                "tensors_checked",
            )
        },
        "final_ema_snapshot": {
            key: internal["final_ema_snapshot"][key]
            for key in (
                "ema_present",
                "ema_finite",
                "ema_tensors_checked",
            )
        },
        "artifact_manifest": artifacts,
        "verified_at_utc": internal["verified_at_utc"],
        "publication": {
            "sanitized_for_github": True,
            "absolute_paths_removed": True,
            "raw_artifacts_retained_external_to_git": True,
        },
    }


def normalized_within_seed(options: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(options)
    value.pop("run_dir", None)
    value["loss_kwargs"].pop("global_gap_scale", None)
    value["optimizer_kwargs"].pop("lr", None)
    return value


def normalized_between_seeds(options: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(options)
    value.pop("run_dir", None)
    value.pop("seed", None)
    return value


def preview_pair(first: Path, second: Path) -> dict[str, Any]:
    left = np.asarray(Image.open(first).convert("RGB"), dtype=np.int16)
    right = np.asarray(Image.open(second).convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        fail(f"preview shape mismatch: {first.name} vs {second.name}")
    delta = left - right
    absolute = np.abs(delta)
    return {
        "first": first.parent.name,
        "second": second.parent.name,
        "shape_hwc": list(left.shape),
        "exact_pixel_values_equal": bool(np.array_equal(left, right)),
        "max_abs_channel_delta_lsb": int(absolute.max(initial=0)),
        "differing_channel_values": int(np.count_nonzero(delta)),
        "differing_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
        "positive_one_lsb": int(np.count_nonzero(delta == 1)),
        "negative_one_lsb": int(np.count_nonzero(delta == -1)),
        "greater_than_one_lsb": int(np.count_nonzero(absolute > 1)),
    }


def parse_exit_marker(path: Path, reference_utc: datetime) -> datetime:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Exiting\.\.\.\s+\[rank0\]:\[W(\d{3}) "
        r"(\d{2}):(\d{2}):(\d{2})\.(\d+)[^\n]*destroy_process_group",
        text,
    )
    if len(matches) != 1:
        fail(f"expected one timestamped clean-exit marker in {path.name}")
    month_day, hour, minute, second, fraction = matches[-1]
    month = int(month_day[:-2])
    day = int(month_day[-2:])
    microsecond = int((fraction + "000000")[:6])
    local = datetime(
        reference_utc.year,
        month,
        day,
        int(hour),
        int(minute),
        int(second),
        microsecond,
        tzinfo=LOCAL_TZ,
    )
    return local.astimezone(timezone.utc)


def stats_interval(path: Path) -> tuple[datetime, datetime]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 9:
        fail(f"{path} does not have nine stats records")
    timestamps = [float(row["timestamp"]) for row in rows]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        fail(f"{path} stats timestamps are not strictly increasing")
    return (
        datetime.fromtimestamp(timestamps[0], timezone.utc),
        datetime.fromtimestamp(timestamps[-1], timezone.utc),
    )


def parse_start_events(segment_id: str, path: Path) -> list[dict[str, Any]]:
    events = []
    pattern = re.compile(
        r"START seed=(\d+) arm=([ABC]) gap=([^ ]+) lr=([^ ]+) gpu=(\d+) port=(\d+)"
    )
    for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
        seed, arm, gap, lr, gpu, port = match.groups()
        events.append(
            {
                "segment_id": segment_id,
                "seed": int(seed),
                "arm": arm,
                "gap_scale": float(gap),
                "learning_rate": float(lr),
                "logged_gpu_index": int(gpu),
                "port": int(port),
            }
        )
    return events


def parse_done_events(segment_id: str, path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        (int(seed), arm)
        for seed, arm in re.findall(
            r"DONE seed=(\d+) arm=([ABC]) integrity=passed", text
        )
    ]


def hardware_summary(path: Path) -> dict[str, Any]:
    devices = []
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
        if len(row) != 5:
            fail("hardware sidecar row must contain five columns")
        index, name, uuid, driver, memory = [item.strip() for item in row]
        devices.append(
            {
                "device_alias": f"device_{index}",
                "logged_gpu_index": int(index),
                "name": name,
                "driver_version": driver,
                "memory_total": memory,
            }
        )
    indices = [item["logged_gpu_index"] for item in devices]
    raw_uuids = [row[2].strip() for row in csv.reader(path.read_text(encoding="utf-8").splitlines())]
    if sorted(indices) != [0, 1] or len(set(raw_uuids)) != 2:
        fail("hardware sidecar must bind two unique GPU indices and UUIDs")
    signatures = {
        (item["name"], item["driver_version"], item["memory_total"])
        for item in devices
    }
    return {
        "devices": devices,
        "same_model_driver_and_memory": (
            len(devices) == 2
            and len(signatures) == 1
            and all(item["name"] == "NVIDIA A100 80GB PCIe" for item in devices)
        ),
            "full_gpu_uuids_and_stable_hashes_retained_only_in_internal_sidecar": True,
    }


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--internal-receipt-dir", required=True, type=Path)
    parser.add_argument("--initialization-reconstruction", required=True, type=Path)
    parser.add_argument("--original-launcher-log", required=True, type=Path)
    parser.add_argument("--adjudication-tooling-commit", required=True)
    parser.add_argument("--public-receipt-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.adjudication_tooling_commit):
        fail("adjudication tooling commit must be a full Git SHA")

    root = args.experiment_root.resolve()
    provenance_path = root / "launch_provenance.txt"
    hardware_path = root / "hardware.txt"
    software_path = root / "software.txt"
    seed4_recovery = root / "logs" / "seed4_resume.launcher.log"
    seed5_recovery = root / "logs" / "seed5_resume.launcher.log"
    for path in (
        provenance_path,
        hardware_path,
        software_path,
        seed4_recovery,
        seed5_recovery,
        args.original_launcher_log,
        args.initialization_reconstruction,
    ):
        if not path.is_file():
            fail(f"missing required evidence: {path}")

    provenance = parse_env(provenance_path)
    expected_provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_commit": EXECUTION_PROTOCOL_COMMIT,
        "training_code_commit": TRAINING_CODE_COMMIT,
        "source_audit_receipt_sha256": SOURCE_AUDIT_RECEIPT_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "dataset_sha256": DATA_SHA256,
        "transfer_sha256": TRANSFER_SHA256,
        "new_seeds": "4,5",
        "arm_order": "A,B,C",
        "gpu": "1",
        "automatic_retry": "false",
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            fail(f"launch provenance {key} mismatch")

    init_report = load_json(args.initialization_reconstruction)
    if (
        init_report.get("status") != "passed"
        or init_report.get("experiment_id") != EXPERIMENT_ID
        or init_report.get("quality_blind", {}).get(
            "generation_quality_metrics_accessed"
        )
        is not False
        or init_report.get("cross_run", {}).get(
            "all_six_reconstructed_net_hashes_equal"
        )
        is not True
        or init_report.get("bindings", {}).get("adjudication_tooling_commit")
        != args.adjudication_tooling_commit
        or init_report.get("cross_run", {}).get(
            "all_six_initialization_contract_hashes_equal"
        )
        is not True
    ):
        fail("initialization reconstruction did not pass")

    run_records: dict[str, dict[str, Any]] = {}
    options: dict[int, dict[str, dict[str, Any]]] = {4: {}, 5: {}}
    public_receipt_manifest: dict[str, Any] = {}
    preview_paths: dict[int, dict[str, Path]] = {4: {}, 5: {}}
    data_hashes: dict[int, dict[str, str]] = {4: {}, 5: {}}
    clock_offsets = []

    segment_paths = {
        "original": args.original_launcher_log,
        "seed4_recovery": seed4_recovery,
        "seed5_recovery": seed5_recovery,
    }
    start_events = []
    for segment_id, path in segment_paths.items():
        start_events.extend(parse_start_events(segment_id, path))
    event_keys = [(item["seed"], item["arm"]) for item in start_events]
    if len(event_keys) != len(set(event_keys)):
        fail("duplicate launcher START event; possible retry or relaunch")
    event_map = {(item["seed"], item["arm"]): item for item in start_events}
    if set(event_map) != {(seed, arm) for seed, arm, _run_id in RUNS}:
        fail("launcher START events do not cover exactly the six runs")
    expected_by_segment = {
        "original": [(4, "A")],
        "seed4_recovery": [(4, "B"), (4, "C")],
        "seed5_recovery": [(5, "A"), (5, "B"), (5, "C")],
    }
    for segment_id, expected in expected_by_segment.items():
        observed_starts = [
            (item["seed"], item["arm"])
            for item in start_events
            if item["segment_id"] == segment_id
        ]
        if observed_starts != expected:
            fail(f"launcher START order mismatch for {segment_id}")
        observed_done = parse_done_events(segment_id, segment_paths[segment_id])
        if segment_id == "original":
            if observed_done:
                fail("original launcher unexpectedly records DONE")
        elif observed_done != expected:
            fail(f"launcher DONE order mismatch for {segment_id}")

    hardware = hardware_summary(hardware_path)
    device_by_index = {
        item["logged_gpu_index"]: item for item in hardware["devices"]
    }

    for seed, arm, run_id in RUNS:
        run_dir = root / run_id
        internal_path = args.internal_receipt_dir / f"seed{seed}_{arm}.integrity.json"
        internal, paths = validate_internal_receipt(
            internal_path, run_dir, seed, arm
        )
        if (
            init_report.get("runs", {})
            .get(run_id, {})
            .get("internal_integrity_receipt_sha256")
            != file_sha256(internal_path)
        ):
            fail(f"initialization reconstruction receipt binding mismatch: {run_id}")
        public = public_receipt(internal, internal_path, run_id)
        public_name = f"seed{seed}_{arm}.integrity.public.json"
        public_path = args.public_receipt_dir / public_name
        write_json(public_path, public)
        public_receipt_manifest[f"seed{seed}_{arm}"] = {
            "run_id": run_id,
            "file": public_name,
            "sha256": file_sha256(public_path),
            "internal_receipt_sha256": file_sha256(internal_path),
        }

        options[seed][arm] = load_json(paths["training_options"])
        preview_paths[seed][arm] = paths["model_init_image"]
        data_hashes[seed][arm] = file_sha256(paths["data_image"])
        first_progress, last_progress = stats_interval(paths["stats"])
        external_log = root / "logs" / f"seed{seed}_{arm}.log"
        exit_marker = parse_exit_marker(external_log, last_progress)
        event = event_map[(seed, arm)]
        same_number(
            event["gap_scale"],
            internal["gap_scale"],
            f"logged gap seed{seed}_{arm}",
        )
        same_number(
            event["learning_rate"],
            internal["learning_rate"],
            f"logged learning rate seed{seed}_{arm}",
        )
        device = device_by_index[event["logged_gpu_index"]]
        run_records[run_id] = {
            "seed": seed,
            "arm": arm,
            "segment_id": event["segment_id"],
            "port": event["port"],
            "logged_gpu_index": event["logged_gpu_index"],
            "device_alias": device["device_alias"],
            "gpu_evidence": {
                "logged_index": "direct launcher assertion",
                "uuid_mapping": (
                    "inference from one pre-launch hardware sidecar; index stability "
                    "and per-run UUID were not attested"
                ),
            },
            "first_progress_at_utc": iso(first_progress),
            "last_progress_at_utc": iso(last_progress),
            "exit_marker_at_utc": iso(exit_marker),
            "interval_definition": "first stats record through timestamped exit marker",
            "interval_scope": "observed training-phase lower bound, not exact process lifetime",
            "exit_timestamp_source": "destroy_process_group warning immediately after clean Exiting marker",
            "exit_timestamp_timezone_assumption": "Asia/Shanghai (UTC+08:00)",
            "integrity_verified_at_utc": internal["verified_at_utc"],
            "public_receipt": public_name,
            "public_receipt_sha256": file_sha256(public_path),
            "external_training_log_sha256": file_sha256(external_log),
        }
        verified = datetime.fromisoformat(internal["verified_at_utc"])
        if not first_progress <= last_progress <= exit_marker <= verified:
            fail(f"non-monotonic application chronology for {run_id}")
        mtime = datetime.fromtimestamp(internal_path.stat().st_mtime, timezone.utc)
        clock_offsets.append((mtime - verified).total_seconds())

    normalized_within = {
        str(seed): len(
            {
                json.dumps(normalized_within_seed(options[seed][arm]), sort_keys=True)
                for arm in ("A", "B", "C")
            }
        )
        == 1
        for seed in (4, 5)
    }
    normalized_between = {
        arm: normalized_between_seeds(options[4][arm])
        == normalized_between_seeds(options[5][arm])
        for arm in ("A", "B", "C")
    }
    if not all(normalized_within.values()) or not all(normalized_between.values()):
        fail("configuration allowed-difference contract failed")

    previews: dict[str, Any] = {}
    for seed in (4, 5):
        pairs = [
            preview_pair(preview_paths[seed]["A"], preview_paths[seed][arm])
            for arm in ("B", "C")
        ]
        previews[str(seed)] = {
            "sha256": {
                arm: file_sha256(path)
                for arm, path in preview_paths[seed].items()
            },
            "pairwise_against_A": pairs,
            "exact_file_hashes_equal": len(
                {file_sha256(path) for path in preview_paths[seed].values()}
            )
            == 1,
            "max_abs_channel_delta_lsb": max(
                item["max_abs_channel_delta_lsb"] for item in pairs
            ),
            "role": "generated FP16 diagnostic preview; not a parameter hash",
        }
    if max(item["max_abs_channel_delta_lsb"] for item in previews.values()) > 1:
        fail("model-init preview drift exceeds one LSB")
    if any(len(set(items.values())) != 1 for items in data_hashes.values()):
        fail("within-seed data images differ")
    all_data_hashes = {
        value for per_seed in data_hashes.values() for value in per_seed.values()
    }
    if len(all_data_hashes) != 1:
        fail("data images differ across seeds")

    intervals = []
    for run_id, item in run_records.items():
        intervals.append(
            (
                run_id,
                datetime.fromisoformat(item["first_progress_at_utc"]),
                datetime.fromisoformat(item["exit_marker_at_utc"]),
                item["logged_gpu_index"],
            )
        )
    overlaps = []
    for index, (left_id, left_start, left_end, left_gpu) in enumerate(intervals):
        for right_id, right_start, right_end, right_gpu in intervals[index + 1 :]:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start < end:
                overlaps.append(
                    {
                        "runs": [left_id, right_id],
                        "directly_observed_overlap_start_utc": iso(start),
                        "directly_observed_overlap_end_utc": iso(end),
                        "duration_seconds": (end - start).total_seconds(),
                        "logged_gpu_indices": [left_gpu, right_gpu],
                        "different_logged_gpu_indices": left_gpu != right_gpu,
                    }
                )
    if len(overlaps) != 2 or not all(
        item["different_logged_gpu_indices"] for item in overlaps
    ):
        fail("runtime overlap pattern differs from the documented deviation")

    segments = []
    for segment_id, path in segment_paths.items():
        segment: dict[str, Any] = {
            "segment_id": segment_id,
            "kind": "original" if segment_id == "original" else "manual_recovery",
            "launcher_log_sha256": file_sha256(path),
            "launcher_log_size_bytes": path.stat().st_size,
            "exact_command_preserved": False,
            "committed_launcher_reconstructible": segment_id == "original",
        }
        pid_path = root / "logs" / f"{segment_id.replace('_recovery', '_resume')}.pid"
        if segment_id != "original" and pid_path.is_file():
            segment["pid"] = int(pid_path.read_text(encoding="utf-8").strip())
            segment["pid_file_sha256"] = file_sha256(pid_path)
            segment["pid_file_mtime_filesystem_clock_utc"] = iso(
                datetime.fromtimestamp(pid_path.stat().st_mtime, timezone.utc)
            )
        segments.append(segment)

    launch_time = datetime.fromisoformat(provenance["launch_utc"].replace("Z", "+00:00"))
    provenance_mtime = datetime.fromtimestamp(
        provenance_path.stat().st_mtime, timezone.utc
    )
    clock_offsets.append((provenance_mtime - launch_time).total_seconds())
    if not (
        30.0 <= min(clock_offsets) <= max(clock_offsets) <= 120.0
        and max(clock_offsets) - min(clock_offsets) <= 2.0
    ):
        fail("filesystem/application clock anchors are inconsistent")

    evidence_manifest = {}
    for label, path in {
        "launch_provenance": provenance_path,
        "hardware": hardware_path,
        "software": software_path,
        "original_launcher_log": args.original_launcher_log,
        "seed4_recovery_launcher_log": seed4_recovery,
        "seed5_recovery_launcher_log": seed5_recovery,
        "initialization_reconstruction": args.initialization_reconstruction,
    }.items():
        evidence_manifest[label] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "retained_external_to_git": label
            not in {"initialization_reconstruction"},
        }

    software = parse_env(software_path)
    evidence = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_quality_blind_evidence",
        "status": "adjudication_ready",
        "experiment_id": EXPERIMENT_ID,
        "quality_blind": {
            "generation_quality_metrics_accessed": False,
            "decision_frozen_before_quality_evaluation": True,
            "attestation_kind": "workflow-scope declaration; not cryptographic proof",
            "excluded_inputs": ["FID", "KID", "quality-evaluation outputs"],
        },
        "bindings": {
            "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
            "adjudication_tooling_commit": args.adjudication_tooling_commit,
            "training_code_commit": TRAINING_CODE_COMMIT,
            "source_audit_receipt_sha256": SOURCE_AUDIT_RECEIPT_SHA256,
            "matrix_sha256": MATRIX_SHA256,
            "dataset_sha256": DATA_SHA256,
            "transfer_checkpoint_sha256": TRANSFER_SHA256,
            "initialization_reconstruction_sha256": file_sha256(
                args.initialization_reconstruction
            ),
            "evidence_builder_source_sha256": file_sha256(Path(__file__)),
        },
        "per_run_integrity": {
            "passed_runs": 6,
            "required_runs": 6,
            "all_artifact_hashes_recomputed": True,
            "public_receipts": public_receipt_manifest,
        },
        "configuration_contract": {
            "within_seed_allowed_differences": [
                "loss_kwargs.global_gap_scale",
                "optimizer_kwargs.lr",
                "run_dir",
            ],
            "within_seed_passed": normalized_within,
            "between_seed_allowed_differences": ["seed", "run_dir"],
            "between_seed_passed": normalized_between,
        },
        "initialization": {
            "historical_observed_preupdate_parameter_hash": "not_captured",
            "reconstructed_expected_initialization": {
                "status": "passed",
                "report_sha256": file_sha256(args.initialization_reconstruction),
                "hash_kind": "reconstructed_expected_initialization_hash",
                "all_six_equal": True,
                "distinct_hashes": init_report["cross_run"][
                    "distinct_reconstructed_net_hashes"
                ],
                "historical_process_attestation": False,
            },
            "model_init_previews": previews,
            "data_images": {
                "all_six_sha256_equal": True,
                "sha256": next(iter(all_data_hashes)),
            },
        },
        "runtime": {
            "planned": {
                "seed_order": [4, 5],
                "arm_order": ["A", "B", "C"],
                "gpu_index": 1,
                "execution_mode": "fully_serial",
                "automatic_retry": False,
            },
            "hardware": hardware,
            "software": software,
            "launcher_segments": segments,
            "runs": run_records,
            "directly_observed_overlaps": overlaps,
            "clock_model": {
                "application_clock_sources": [
                    "stats.jsonl epoch timestamps",
                    "timestamped process-exit warning",
                    "receipt verified_at_utc",
                ],
                "filesystem_clock_not_used_as_application_time": True,
                "acceptance_gate": "diagnostic only; no exact process-start claim",
                "observed_filesystem_minus_application_offset_seconds": {
                    "minimum": min(clock_offsets),
                    "maximum": max(clock_offsets),
                },
            },
        },
        "deviations": [
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
                "acceptance_rationale": "all recovery runs were fresh starts and passed artifact integrity",
            },
            {
                "id": "D2",
                "field": "execution_mode",
                "planned": "fully serial",
                "observed": "two directly observed cross-GPU overlap intervals",
                "materiality": "runtime/performance; no training-definition change observed",
                "evidence_refs": ["runtime.directly_observed_overlaps", "runtime.hardware"],
                "confidence": "direct application timestamp overlap; GPU index is launcher assertion",
                "acceptance_rationale": "overlaps used different logged indices on equivalent A100 devices; performance claims excluded",
            },
            {
                "id": "D3",
                "field": "gpu_assignment",
                "planned": "all runs on logged GPU index 1",
                "observed": "seed 4 B/C logged on GPU index 0",
                "materiality": "possible low-order numerical environment effect",
                "evidence_refs": ["runtime.runs", "runtime.hardware"],
                "confidence": "direct index assertion; UUID mapping inferred from one sidecar",
                "acceptance_rationale": "same GPU model, memory, driver, code, inputs, and training definition",
            },
            {
                "id": "D4",
                "field": "initialization_evidence",
                "planned": "identical model-init preview files",
                "observed": "seed 5 A versus B/C differs by at most one 8-bit level",
                "materiality": "diagnostic preview only; historical parameter hash absent",
                "evidence_refs": ["initialization.model_init_previews", "initialization.reconstructed_expected_initialization"],
                "confidence": "direct PNG comparison plus deterministic reconstruction of expected tensor state",
                "acceptance_rationale": "all six expected tensor-state hashes match and preview drift is only one LSB",
            },
            {
                "id": "D5",
                "field": "inline_verification_continuity",
                "planned": "seed4 A verified inline before launcher continuation",
                "observed": "original verifier output/exit status lost; same artifact set passed post-hoc re-verification",
                "materiality": "verification provenance",
                "evidence_refs": [
                    "runtime.launcher_segments",
                    "per_run_integrity.public_receipts.seed4_A",
                    "missing_evidence",
                ],
                "confidence": "training completion and post-hoc receipt direct; original failure mechanism unknown",
                "acceptance_rationale": "all seed4 A artifacts were recomputed and passed strengthened verification; exact original failure not claimed",
            },
        ],
        "missing_evidence": [
            "historical post-transfer/pre-forward parameter hash",
            "per-run CUDA UUID attestation",
            "per-run recovery software snapshot",
            "complete recovery command lines and process environments",
            "original seed4-A verifier failure output and exit status",
        ],
        "claim_exclusions": [
            "protocol-exact execution",
            "historically observed bitwise pre-update parameter identity",
            "bitwise training equivalence across devices",
            "throughput, latency, or GPU performance comparisons",
        ],
        "evidence_manifest": evidence_manifest,
        "publication": {
            "sanitized_for_github": True,
            "absolute_paths_hostnames_accounts_ips_and_full_gpu_uuids_removed": True,
        },
    }
    write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fail(str(exc))
