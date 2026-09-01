#!/usr/bin/env python3
"""Fail-closed orchestration for the fresh q256 n=12 crossed-switch run."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import shutil
import signal
import socket
import subprocess
import sys
import time
import math
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training import reproducibility, schedule_switch  # noqa: E402

EXPERIMENT_ID = "q256_fresh_crossed_switch_n12_matpool_v1"
PROTOCOL_SCHEMA = "ect.q256.fresh-crossed-switch-protocol/v1"
RUNTIME_SCHEMA = "ect.q256.rebuilt-runtime/v1"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
SEEDS = tuple(range(31, 43))
ELEVEN_SEED_EXCLUSION = 38
ELEVEN_SEEDS = tuple(seed for seed in SEEDS if seed != ELEVEN_SEED_EXCLUSION)
ELEVEN_JOB_COUNT = len(ELEVEN_SEEDS) * 22
ARMS = {"A": (1.0, 1.0), "B": (1.1, 1.1)}
CELLS = {"AA": ("A", "A"), "AB": ("A", "B"), "BA": ("B", "A"), "BB": ("B", "B")}
SUFFIX_ORDERS = (
    ("AA", "AB", "BA", "BB"),
    ("AB", "BB", "AA", "BA"),
    ("BA", "AA", "BB", "AB"),
    ("BB", "BA", "AB", "AA"),
)
TAPE_FIELDS = ("attempted_iteration", "batch_sha256", "t_sha256", "base_r_sha256")
HEX40 = __import__("re").compile(r"^[0-9a-f]{40}$")
HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: object, *, overwrite: bool = False) -> None:
    reproducibility.atomic_json_dump(payload, path, overwrite=overwrite)


def load_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def validate_eleven_seed_authorization(path: Path, protocol_path: Path,
                                       *, require_commit: bool = False) -> dict:
    path = path.resolve(strict=True)
    value = load_json(path)
    expected = {
        "status": "AUTHOR_APPROVED",
        "protocol_sha256": sha256_file(protocol_path),
        "excluded_seed": ELEVEN_SEED_EXCLUSION,
        "included_seeds": list(ELEVEN_SEEDS),
        "expected_evaluation_jobs": ELEVEN_JOB_COUNT,
        "original_n12_claim_abandoned": True,
        "decode_forbidden_before_full_amended_seal": True,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("eleven-seed authorization binding mismatch")
    if require_commit:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if value.get("amendment_commit") != head:
            raise RuntimeError("eleven-seed authorization commit mismatch")
    return value


def validate_evaluation_recovery1_authorization(path: Path, protocol_path: Path,
                                                *, require_commit: bool = False) -> dict:
    path = path.resolve(strict=True)
    value = load_json(path)
    if (value.get("status") != "AUTHOR_APPROVED"
            or value.get("protocol_sha256") != sha256_file(protocol_path)
            or value.get("manual_evaluation_recovery_index") != 1
            or value.get("expected_evaluation_jobs") != ELEVEN_JOB_COUNT
            or value.get("automatic_retry_count") != 0
            or value.get("original_failure_must_be_preserved") is not True
            or value.get("decode_forbidden_before_full_recovery_seal") is not True):
        raise RuntimeError("evaluation recovery1 authorization binding mismatch")
    eleven_path = Path(value.get("eleven_seed_authorization_path", ""))
    validate_eleven_seed_authorization(eleven_path, protocol_path)
    if value.get("eleven_seed_authorization_sha256") != sha256_file(eleven_path):
        raise RuntimeError("evaluation recovery1 eleven-seed authorization mismatch")
    if require_commit:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if value.get("amendment_commit") != head:
            raise RuntimeError("evaluation recovery1 commit mismatch")
    return value


def copy_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"missing regular source artifact: {source}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def copy_csv_prefix(source: Path, destination: Path) -> dict:
    with source.open("rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [row for row in reader
                if int(float(row["attempted_iteration"])) <= schedule_switch.SWITCH_ATTEMPT]
    attempts = [int(float(row["attempted_iteration"])) for row in rows]
    if attempts != list(range(1, schedule_switch.SWITCH_ATTEMPT + 1)):
        raise RuntimeError(f"source CSV does not contain exact attempts 1..4000: {source}")
    if int(float(rows[-1]["processed_nimg"])) != schedule_switch.SWITCH_NIMG:
        raise RuntimeError(f"source CSV does not end at 512000 images: {source}")
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    return {"source_path": str(source.resolve()), "source_sha256": sha256_file(source),
            "derived_path": str(destination.resolve()), "derived_sha256": sha256_file(destination),
            "rows": len(rows)}


def prepare_resume_history(prefix_dir: Path, output: Path) -> dict:
    record = {
        "train_summary": copy_csv_prefix(prefix_dir / "train_summary.csv", output / "train_summary.csv"),
        "factorial_telemetry": copy_csv_prefix(
            prefix_dir / "factorial_training_telemetry_v1.csv",
            output / "source_factorial_training_telemetry_v1.csv"),
    }
    copy_exclusive(prefix_dir / "initial_state_receipt_v1.json",
                   output / "initial_state_receipt_v1.json")
    copy_exclusive(prefix_dir / "training_options.json", output / "source_training_options.json")
    return record


def assignment(seed: int) -> dict:
    if seed not in SEEDS:
        raise RuntimeError(f"formal seed outside 31-42: {seed}")
    position = seed - 31
    gpu = position % 6
    wave = position // 6 + 1
    if wave == 1:
        prefix_order = ("A", "B") if seed % 2 == 1 else ("B", "A")
    else:
        prefix_order = ("B", "A") if seed % 2 == 1 else ("A", "B")
    return {
        "seed": seed,
        "gpu_index": gpu,
        "wave": wave,
        "prefix_order": list(prefix_order),
        "suffix_order": list(SUFFIX_ORDERS[position % 4]),
    }


def planned_evaluation_jobs() -> list[dict]:
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            jobs.append({"seed": seed, "kind": "prefix", "cell": arm,
                         "budget_kimg": 512, "nfe": 1})
        for cell in CELLS:
            for budget in (640, 768, 896, 1024):
                jobs.append({"seed": seed, "kind": "suffix", "cell": cell,
                             "budget_kimg": budget, "nfe": 1})
            jobs.append({"seed": seed, "kind": "suffix", "cell": cell,
                         "budget_kimg": 1024, "nfe": 2})
    if len(jobs) != 264:
        raise AssertionError("internal evaluation plan length mismatch")
    return jobs


def query_gpus() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected GPU row: {line}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "driver_version": fields[4],
                "pci_bus_id": fields[5],
            }
        )
    return rows


def compute_apps() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            rows.append(
                {"pid": int(fields[0]), "gpu_uuid": fields[1],
                 "process_name": fields[2], "memory_mib": float(fields[3])}
            )
    return rows


def assert_gpu_exclusive(gpu_uuid: str, receipt_path: Path, phase: str,
                         *, release_grace_seconds: int = 0) -> None:
    deadline = time.monotonic() + release_grace_seconds
    while True:
        apps = [row for row in compute_apps() if row["gpu_uuid"] == gpu_uuid]
        if not apps or time.monotonic() >= deadline:
            break
        time.sleep(1)
    payload = {
        "schema": "ect.q256.gpu-exclusivity/v1",
        "status": "PASS" if not apps else "FAIL",
        "observed_at": utc_now(),
        "phase": phase,
        "gpu_uuid": gpu_uuid,
        "compute_apps": apps,
    }
    atomic_json(receipt_path, payload)
    if apps:
        raise RuntimeError(f"foreign GPU process at {phase}: {apps}")


def validate_runtime(runtime: dict) -> None:
    if runtime.get("schema") != RUNTIME_SCHEMA or runtime.get("status") != "PASS":
        raise RuntimeError("rebuilt runtime manifest is not PASS")
    for key in ("environment_archive", "explicit_lock", "pip_freeze", "requirements", "probe_receipt"):
        artifact = runtime.get(key)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"runtime manifest missing {key}")
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"runtime artifact mismatch: {key}")
    prefix = Path(runtime["environment_prefix"])
    if not (prefix / "bin" / "python").is_file():
        raise RuntimeError("runtime environment Python is missing")
    live_freeze = subprocess.run(
        [str(prefix / "bin" / "python"), "-m", "pip", "freeze", "--all"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(live_freeze).hexdigest() != runtime["pip_freeze"]["sha256"]:
        raise RuntimeError("live runtime package set differs from frozen pip manifest")


def validate_protocol(protocol: dict, protocol_path: Path | None = None) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("protocol schema mismatch")
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("protocol identity mismatch")
    if protocol.get("seeds") != list(SEEDS):
        raise RuntimeError("formal seeds must be exactly 31-42")
    if protocol.get("evaluation", {}).get("expected_jobs") != 264:
        raise RuntimeError("evaluation matrix must contain exactly 264 jobs")
    if protocol.get("evaluation", {}).get("task_definitions") != planned_evaluation_jobs():
        raise RuntimeError("evaluation task definitions mismatch")
    if protocol.get("training", {}).get("world_size") != 1:
        raise RuntimeError("world_size must be one")
    if len(protocol.get("gpu_assignment", [])) != 12:
        raise RuntimeError("GPU assignment must contain twelve rows")
    if protocol["gpu_assignment"] != [assignment(seed) for seed in SEEDS]:
        raise RuntimeError("GPU/seed/order assignment mismatch")
    storage = protocol.get("storage_plan", {})
    if (storage.get("full_state_count") != 228 or storage.get("ema_snapshot_count") != 228
            or storage.get("headroom_fraction", 0) < 0.30
            or storage.get("minimum_free_bytes", 0) < math.ceil(storage.get("estimated_bytes", 0) * 1.30)
            or storage.get("kid_fid_generated_features_are_hardlinked_after_byte_identity_validation") is not True):
        raise RuntimeError("storage plan does not preserve the 30% headroom contract")
    if protocol_path is not None:
        companion = protocol_path.with_name("protocol.sha256")
        expected = companion.read_text(encoding="ascii").split()[0]
        if sha256_file(protocol_path) != expected:
            raise RuntimeError("protocol companion SHA256 mismatch")


def validate_runtime_parity(report: dict, implementation_commit: str,
                            runtime_manifest_sha256: str) -> None:
    if report.get("schema") != "ect.q256.fresh-crossed-switch-runtime-parity/v1":
        raise RuntimeError("runtime parity schema mismatch")
    if report.get("status") != "PASS" or report.get("automatic_retry_count") != 0:
        raise RuntimeError("runtime parity did not pass on its first attempt")
    if report.get("implementation_commit") != implementation_commit:
        raise RuntimeError("runtime parity implementation commit mismatch")
    if report.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise RuntimeError("runtime parity manifest mismatch")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 2:
        raise RuntimeError("runtime parity must contain exactly A and B comparisons")
    if {item.get("arm") for item in comparisons} != {"A", "B"}:
        raise RuntimeError("runtime parity arm coverage mismatch")
    required = ("parameters", "ema", "optimizer", "gradscaler", "rng",
                "sampler", "counters", "complete_state")
    if any(item.get("status") != "PASS" or
           any(item.get(key) is not True for key in required)
           for item in comparisons):
        raise RuntimeError("runtime parity subsystem comparison mismatch")


def freeze_protocol(args: argparse.Namespace) -> None:
    if not HEX40.fullmatch(args.implementation_commit):
        raise RuntimeError("invalid implementation commit")
    runtime = load_json(args.runtime_manifest.resolve(strict=True))
    validate_runtime(runtime)
    gpus = query_gpus()
    if len(gpus) != 6 or any("A100" not in row["name"] for row in gpus):
        raise RuntimeError("protocol freeze requires exactly six A100 GPUs")
    assets = {
        "dataset": {"path": str(args.dataset.resolve(strict=True)), "sha256": DATASET_SHA256},
        "transfer": {"path": str(args.transfer.resolve(strict=True)), "sha256": TRANSFER_SHA256},
        "runtime_manifest": {
            "path": str(args.runtime_manifest.resolve(strict=True)),
            "sha256": sha256_file(args.runtime_manifest),
        },
        "runtime_parity": {
            "path": str(args.runtime_parity.resolve(strict=True)),
            "sha256": sha256_file(args.runtime_parity),
        },
        "evaluator_source": {
            "path": str(args.evaluator_source.resolve(strict=True)),
            "sha256": sha256_file(args.evaluator_source),
            "commit": EVALUATOR_COMMIT,
        },
        "detector": {"path": str(args.detector.resolve(strict=True)), "sha256": sha256_file(args.detector)},
        "real_features": [
            {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
            for path in args.real_features
        ],
        "storage_samples": {
            "full_state": {"path": str(args.storage_sample_state.resolve(strict=True)),
                           "bytes": args.storage_sample_state.stat().st_size,
                           "sha256": sha256_file(args.storage_sample_state)},
            "ema_snapshot": {"path": str(args.storage_sample_snapshot.resolve(strict=True)),
                             "bytes": args.storage_sample_snapshot.stat().st_size,
                             "sha256": sha256_file(args.storage_sample_snapshot)},
        },
    }
    validate_runtime_parity(
        load_json(args.runtime_parity), args.implementation_commit,
        assets["runtime_manifest"]["sha256"],
    )
    if assets["dataset"]["sha256"] != sha256_file(args.dataset):
        raise RuntimeError("canonical dataset SHA256 mismatch")
    if assets["transfer"]["sha256"] != sha256_file(args.transfer):
        raise RuntimeError("transfer checkpoint SHA256 mismatch")
    cache_root = args.evaluator_cache.resolve(strict=True)
    for record in [assets["detector"], *assets["real_features"]]:
        if not Path(record["path"]).resolve().is_relative_to(cache_root):
            raise RuntimeError("detector/real features must reside in the frozen evaluator cache")
    full_state_count = 228
    snapshot_count = 228
    generated_samples_bytes = 264 * 50_000 * 3 * 32 * 32
    shared_generated_features_bytes = 264 * 50_000 * 2048 * 4
    runtime_archive_bytes = runtime["environment_archive"]["bytes"]
    estimated_bytes = (full_state_count * assets["storage_samples"]["full_state"]["bytes"]
                       + snapshot_count * assets["storage_samples"]["ema_snapshot"]["bytes"]
                       + generated_samples_bytes + shared_generated_features_bytes
                       + runtime_archive_bytes)
    reservation_bytes = math.ceil(estimated_bytes * 1.30)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "frozen_at": utc_now(),
        "implementation_commit": args.implementation_commit,
        "protocol_hash_contract": "SHA256 of canonical frozen protocol bytes is stored in protocol.sha256",
        "runtime_amendment": {
            "status": "AUTHOR_APPROVED_BEFORE_FORMAL_EXECUTION",
            "reason": "original SIF unavailable on current node and historical archive endpoints unreachable",
            "old_runtime_sha256": "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea",
            "new_runtime_manifest_sha256": assets["runtime_manifest"]["sha256"],
            "required_new_runtime_parity": True,
        },
        "storage_amendment": {
            "status": "AUTHOR_ACCEPTED_EPHEMERAL_NODE_STORAGE_WITH_LATER_RETURN",
            "formal_output_root": str(args.output_root.absolute()),
            "control_root": str(args.control_root.absolute()),
            "instance_must_not_be_released_before_verified_return": True,
        },
        "storage_plan": {
            "full_state_count": full_state_count, "ema_snapshot_count": snapshot_count,
            "evaluation_generated_samples_bytes": generated_samples_bytes,
            "evaluation_shared_features_bytes": shared_generated_features_bytes,
            "runtime_archive_bytes": runtime_archive_bytes,
            "estimated_bytes": estimated_bytes, "headroom_fraction": 0.30,
            "minimum_free_bytes": reservation_bytes,
            "kid_fid_generated_features_are_hardlinked_after_byte_identity_validation": True,
        },
        "hostname": socket.gethostname(),
        "gpus": gpus,
        "gpu_assignment": [assignment(seed) for seed in SEEDS],
        "paths": {
            "repository_root": str(args.repo.resolve(strict=True)),
            "asset_root": str(args.asset_root.resolve(strict=True)),
            "control_root": str(args.control_root.absolute()),
            "evaluator_cache_root": str(args.evaluator_cache.resolve(strict=True)),
            "formal_output_root": str(args.output_root.absolute()),
            "logs_root": str((args.output_root / "logs").absolute()),
            "immutable_archive_root": str((args.output_root / "archive").absolute()),
        },
        "assets": assets,
        "seeds": list(SEEDS),
        "training": {
            "dataset": "canonical CIFAR-10 32x32",
            "q": 256, "mapping": "official sigmoid", "architecture": "ddpmpp",
            "precondition": "ect", "optimizer": "RAdam", "learning_rate": 1e-4,
            "global_batch": 128, "batch_gpu": 16, "dropout": 0.2,
            "augment": 0, "xflip": False, "ema_beta": 0.9993,
            "fp16": True, "amp": True, "tf32": False,
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8", "world_size": 1,
            "stop_condition": "attempted_iterations",
            "prefix_stop_attempt": 4000, "suffix_stop_attempt": 8000,
            "arms": {"A": [1.0, 1.0], "B": [1.1, 1.1]},
            "prefix_milestones_kimg": {"A": [512], "B": [384, 512]},
            "suffix_milestones_kimg": [640, 768, 896, 1024],
        },
        "evaluation": {
            "precision": "FP32", "samples_per_job": 50000,
            "generation_seed_range": [0, 49999], "metric_seed": 20260730,
            "nfe2_mid_t": 0.821, "shared_generated_features": True,
            "source_512_nfe1_jobs": 24, "suffix_milestone_nfe1_jobs": 192,
            "suffix_1024_nfe2_jobs": 48, "expected_jobs": 264,
            "opaque_ids": True, "shuffle_seed": 20260831,
            "round_robin_gpu_indices": [0, 1, 2, 3, 4, 5],
            "seal_before_decode": True,
            "task_definitions": planned_evaluation_jobs(),
        },
        "statistics": {
            "primary_outcome": "log(FID50k at 1024 kimg, NFE1)",
            "delta": "log(1.03)", "sample_size": 12,
            "history_contrast": "0.5*((Y_BA-Y_AA)+(Y_BB-Y_AB))",
            "current_contrast": "0.5*((Y_AB-Y_AA)+(Y_BB-Y_BA))",
            "interaction": "Y_BB-Y_BA-Y_AB+Y_AA",
            "checkpoint_quality": {
                "Q": "logFID_B_512-logFID_A_512",
                "H_A": "logFID_BA_1024-logFID_AA_1024", "G": "H_A-Q",
            },
            "interval_method": "Student t intervals on twelve seed-level contrasts",
            "sign_flip": "exact two-sided enumeration of all 2^12 sign assignments using absolute mean",
            "primary_decisions": {
                "strong_success": "95% CI upper < -log(1.03), negative count >=10, and every LOSO mean <0",
                "informative_practical_null": "90% CI wholly inside [-log(1.03),+log(1.03)]",
                "weak_directional_replication": "not strong/equivalent and 95% CI upper <0",
                "inconclusive": "95% CI covers 0 and 90% CI is not wholly inside equivalence band",
                "opposite_direction_falsification": "95% CI lower >0",
            },
            "primary_decision_precedence": ["strong_success", "informative_practical_null", "weak_directional_replication", "opposite_direction_falsification", "inconclusive"],
            "interaction_claim_gate": "90% CI wholly inside [-log(1.03),+log(1.03)]",
            "formal_secondary_family": ["two-sided exact sign-flip C", "two-sided exact sign-flip I"],
            "multiple_testing": "Holm correction at familywise alpha 0.05 for C and I",
            "descriptive_only": ["NFE2", "KID", "640/768/896", "AULC", "BA single cell", "checkpoint-quality subgroup"],
        },
        "governance": {
            "no_interim_fid_kid": True, "no_p2_access": True,
            "no_seed_replacement": True, "no_result_driven_stopping": True,
            "no_automatic_retry": True, "only_hard_timeout_may_auto_kill": True,
        },
    }
    validate_protocol(protocol)
    args.destination.mkdir(parents=True, exist_ok=True)
    protocol_path = args.destination / "protocol.json"
    if protocol_path.exists() or (args.destination / "protocol.sha256").exists():
        raise RuntimeError("protocol destination is not fresh")
    atomic_json(protocol_path, protocol)
    digest = sha256_file(protocol_path)
    companion = args.destination / "protocol.sha256"
    with companion.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  protocol.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "protocol": str(protocol_path), "sha256": digest}))


def preflight(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    runtime_manifest_path = Path(protocol["assets"]["runtime_manifest"]["path"])
    if sha256_file(runtime_manifest_path) != protocol["assets"]["runtime_manifest"]["sha256"]:
        raise RuntimeError("runtime manifest SHA mismatch")
    validate_runtime(load_json(runtime_manifest_path))
    parity_record = protocol["assets"]["runtime_parity"]
    parity_path = Path(parity_record["path"])
    if sha256_file(parity_path) != parity_record["sha256"]:
        raise RuntimeError("runtime parity receipt mismatch")
    validate_runtime_parity(
        load_json(parity_path), protocol["implementation_commit"],
        protocol["assets"]["runtime_manifest"]["sha256"],
    )
    for label in ("dataset", "transfer", "evaluator_source", "detector"):
        record = protocol["assets"][label]
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"asset mismatch: {label}")
    for record in protocol["assets"]["real_features"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("real-feature asset mismatch")
    for record in protocol["assets"]["storage_samples"].values():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError("storage-size sample mismatch")
    cache_root = Path(protocol["paths"]["evaluator_cache_root"]).resolve(strict=True)
    for record in [protocol["assets"]["detector"], *protocol["assets"]["real_features"]]:
        if not Path(record["path"]).resolve(strict=True).is_relative_to(cache_root):
            raise RuntimeError("evaluator cache binding mismatch")
    gpus = query_gpus()
    if gpus != protocol["gpus"]:
        raise RuntimeError("current GPU inventory differs from frozen protocol")
    if compute_apps():
        raise RuntimeError("GPU exclusivity preflight failed")
    if socket.gethostname() != protocol["hostname"]:
        raise RuntimeError("hostname differs from frozen protocol")
    repo = Path(protocol["paths"]["repository_root"])
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("formal worktree is dirty")
    ancestor = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", protocol["implementation_commit"], head]
    )
    if ancestor.returncode != 0:
        raise RuntimeError("implementation commit is not an ancestor of formal HEAD")
    output_root = Path(protocol["paths"]["formal_output_root"])
    if output_root.exists():
        raise RuntimeError("formal output root must not already exist")
    usage = shutil.disk_usage(output_root.parent)
    minimum = max(args.minimum_free_gib * 1024**3,
                  int(protocol["storage_plan"]["minimum_free_bytes"]))
    if usage.free < minimum:
        raise RuntimeError(f"insufficient free storage: {usage.free} < {minimum}")
    receipt = {
        "schema": "ect.q256.fresh-crossed-switch-preflight/v1", "status": "PASS",
        "observed_at": utc_now(), "hostname": socket.gethostname(), "gpus": gpus,
        "git_head": head, "git_clean": True, "free_bytes": usage.free,
        "minimum_free_bytes": minimum, "protocol_sha256": sha256_file(protocol_path),
        "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
        "gpu_exclusivity": "PASS", "output_root_absent": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


def training_command(
    protocol: dict, run_dir: Path, seed: int, arm: str, gpu: int,
    *, prefix: bool, manifest: Path | None = None, source: Path | None = None,
    final_kimg: int = 1024,
) -> list[str]:
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    target, denominator = ARMS[arm]
    command = [
        python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        f"--master_port={48100 + gpu}", str(REPO_ROOT / "ct_train.py"),
        f"--data={protocol['assets']['dataset']['path']}", f"--outdir={run_dir}", "--nosubdir",
        "--cond=False", "--arch=ddpmpp", "--precond=ect", "--batch=128", "--batch-gpu=16",
        "--optim=RAdam", "--lr=0.0001", "--dropout=0.2", "--augment=0", "--xflip=False",
        "--mean=-1.1", "--std=2.0", "--mapping=sigmoid", "--global-gap-scale=1.0",
        "--factorial-protocol=q256_target_weight_v1", f"--target-gap-scale={target}",
        f"--denominator-gap-scale={denominator}", "-q", "256", "-k", "8", "-b", "1", "-c", "0",
        "--double=10000", "--ema_beta=0.9993", f"--seed={seed}", "--fp16=True", "--tf32=False",
        "--ls=1.0", "--enable_amp=True", "--bench=False", "--cache=True", "--workers=1",
        "--metrics=none", f"--duration={final_kimg / 1000:.3f}", "--tick=10", "--snap=0", "--dump=0", "--ckpt=10",
        "--sample_every=26", "--eval_every=50", "--mid_t=0.821", "--adaptive-update-kimg=0.5",
    ]
    if prefix:
        milestones = "512" if arm == "A" else "384,512"
        command.extend([
            f"--immutable-checkpoint-kimg={milestones}", "--stop-after-attempts=4000",
            f"--planned-pause-protocol={'q256_fresh_crossed_switch_n12_matpool_engineering_v1' if seed == 20260831 else EXPERIMENT_ID}",
            f"--transfer={protocol['assets']['transfer']['path']}",
        ])
    else:
        if manifest is None or source is None:
            raise RuntimeError("suffix command requires manifest and source")
        milestones = "640" if final_kimg == 640 else "640,768,896,1024"
        command.extend([
            f"--immutable-checkpoint-kimg={milestones}",
            f"--schedule-switch-manifest={manifest}", f"--resume={source}",
        ])
    return command


def cell_environment(gpu: int, runtime: dict) -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(runtime["environment_prefix"])
    torch_lib = prefix / "lib" / f"python{runtime['probe']['python_major_minor']}" / "site-packages" / "torch" / "lib"
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_CACHE_DISABLE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(48100 + gpu),
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PATH=f"{prefix / 'bin'}:{env.get('PATH', '')}",
        LD_LIBRARY_PATH=f"{torch_lib}:{prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
    )
    return env


def run_cell(protocol: dict, run_dir: Path, gpu: int, command: list[str], label: str) -> None:
    if run_dir.exists():
        existing = {path.name for path in run_dir.iterdir()}
        prepared_resume = {
            "formal_run_manifest.json", "train_summary.csv",
            "source_factorial_training_telemetry_v1.csv", "initial_state_receipt_v1.json",
            "source_training_options.json",
        }
        if existing not in ({"formal_run_manifest.json"}, prepared_resume,
                            prepared_resume | {"source_state_receipt.json"}):
            raise RuntimeError(f"pre-existing cell is not a fresh prepared suffix: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    gpu_uuid = protocol["gpus"][gpu]["uuid"]
    protocol_sha = protocol.get("protocol_sha256_runtime")
    if protocol_sha is None:
        protocol_sha = command and next(
            (item.split("=", 1)[1] for item in command if item.startswith("--schedule-switch-manifest=")),
            None,
        )
        protocol_sha = (load_json(Path(protocol_sha))["protocol_sha256"]
                        if protocol_sha else None)
    atomic_json(run_dir / "preparation_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-cell-preparation/v1", "status": "PASS",
        "label": label, "prepared_at": utc_now(), "hostname": socket.gethostname(),
        "gpu_index": gpu, "gpu_uuid": gpu_uuid, "world_size": 1,
        "command": command, "runtime_manifest_sha256": protocol["assets"]["runtime_manifest"]["sha256"],
        "protocol_sha256": protocol_sha,
    })
    atomic_json(run_dir / "trajectory_manifest.json", {
        "schema": "ect.q256.fresh-crossed-switch-cell-trajectory/v1", "status": "PASS",
        "label": label, "command": command, "gpu_index": gpu, "gpu_uuid": gpu_uuid,
        "formal_schedule_switch_manifest": (str(run_dir / "formal_run_manifest.json")
                                             if (run_dir / "formal_run_manifest.json").is_file() else None),
    })
    atomic_json(run_dir / "matpool_gpu_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-matpool-gpu/v1", "status": "PASS",
        "hostname": socket.gethostname(), "gpu": protocol["gpus"][gpu], "world_size": 1,
    })
    assert_gpu_exclusive(gpu_uuid, run_dir / "gpu_exclusivity_before.json", "before")
    start = time.monotonic()
    start_receipt = {
        "schema": "ect.q256.fresh-crossed-switch-compute-start/v1", "status": "START",
        "label": label, "started_at": utc_now(), "gpu_index": gpu, "gpu_uuid": gpu_uuid,
        "command": command, "runtime_manifest_sha256": protocol["assets"]["runtime_manifest"]["sha256"],
    }
    atomic_json(run_dir / "compute_start_receipt.json", start_receipt)
    outer_log = (run_dir / "launcher.log").open("xb")
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=cell_environment(gpu, runtime), stdout=outer_log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    duration = next(float(item.split("=", 1)[1]) for item in command if item.startswith("--duration="))
    total_attempts = (4000 if any(item == "--stop-after-attempts=4000" for item in command)
                      else int(duration * 1_000_000 // 128))
    monitor = subprocess.Popen(
        [python, str(Path(__file__).with_name("monitor.py")), "--pid", str(process.pid),
         "--run-dir", str(run_dir), "--gpu-index", str(gpu), "--gpu-uuid", gpu_uuid,
         "--total-attempts", str(total_attempts), "--interval-seconds", "30", "--stall-seconds", "300",
         "--min-free-bytes", str(100 * 1024**3)],
        cwd=REPO_ROOT, env=cell_environment(gpu, runtime),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    timed_out = False
    try:
        returncode = process.wait(timeout=6 * 3600)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    finally:
        outer_log.close()
    monitor.wait(timeout=90)
    completion = {
        "schema": "ect.q256.fresh-crossed-switch-compute-completion/v1",
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "label": label, "ended_at": utc_now(), "elapsed_seconds": time.monotonic() - start,
        "exit_code": returncode, "hard_timeout": timed_out,
    }
    atomic_json(run_dir / "compute_completion_receipt.json", completion)
    atomic_json(run_dir / "compute_time_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-compute-time/v1",
        "status": completion["status"], "label": label,
        "elapsed_seconds": completion["elapsed_seconds"], "gpu_hours": completion["elapsed_seconds"] / 3600,
        "gpu_index": gpu, "gpu_uuid": gpu_uuid,
    })
    if returncode != 0 or timed_out:
        raise RuntimeError(f"cell failed without retry: {label}, exit={returncode}, timeout={timed_out}")
    assert_gpu_exclusive(gpu_uuid, run_dir / "gpu_exclusivity_after.json", "after",
                         release_grace_seconds=30)


def export_prefix(run_dir: Path, seed: int, arm: str, protocol_sha: str) -> dict:
    milestones = (512,) if arm == "A" else (384, 512)
    records = []
    for kimg in milestones:
        state_path = run_dir / f"training-state-kimg{kimg:06d}.pt"
        if not state_path.is_file() or state_path.is_symlink():
            raise RuntimeError(f"missing prefix state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state["attempted_iteration"]) != kimg * 1000 // 128 or int(state["cur_nimg"]) != kimg * 1000:
            raise RuntimeError("prefix state counter mismatch")
        hashes = schedule_switch.internal_state_hashes(state)
        milestone_dir = run_dir / f"kimg{kimg:04d}"
        milestone_dir.mkdir(exist_ok=False)
        linked = milestone_dir / "training-state.pt"
        os.link(state_path, linked)
        snapshot = {
            "ema": copy.deepcopy(state["ema"]).eval().requires_grad_(False),
            "loss_fn": None, "augment_pipe": None,
            "dataset_kwargs": dict(state["trajectory_config"]["dataset_kwargs"]),
        }
        snapshot_path = milestone_dir / "network-snapshot.pkl"
        reproducibility.atomic_pickle_dump(snapshot, snapshot_path)
        receipt = {
            "schema": "ect.q256.fresh-crossed-switch-prefix-milestone/v1", "status": "PASS",
            "seed": seed, "arm": arm, "milestone_kimg": kimg,
            "attempted_iteration": int(state["attempted_iteration"]),
            "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
            "cur_nimg": int(state["cur_nimg"]),
            "training_state": {"path": str(linked), "bytes": linked.stat().st_size,
                               "sha256": sha256_file(linked), "internal_state_sha256": hashes},
            "network_snapshot": {"path": str(snapshot_path), "bytes": snapshot_path.stat().st_size,
                                 "sha256": sha256_file(snapshot_path), "ema_internal_sha256": hashes["ema"]},
            "protocol_sha256": protocol_sha,
        }
        receipt_path = milestone_dir / "milestone_receipt.json"
        atomic_json(receipt_path, receipt)
        records.append(receipt)
    source = records[-1]
    source_receipt = {
        "schema": "ect.q256.fresh-crossed-switch-source-state/v1", "status": "PASS",
        "seed": seed, "arm": arm, "source_kimg": 512,
        "training_state": source["training_state"], "protocol_sha256": protocol_sha,
        "milestone_receipt_sha256": sha256_file(run_dir / "kimg0512" / "milestone_receipt.json"),
    }
    atomic_json(run_dir / "source_state_receipt.json", source_receipt)
    return source_receipt


def compare_tapes(paths: list[Path], labels: list[str], destination: Path,
                  protocol_sha: str, *, first_attempt: int, last_attempt: int) -> None:
    series = []
    for path in paths:
        with path.open("rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        attempts = [int(float(row["attempted_iteration"])) for row in rows]
        if attempts != list(range(first_attempt, last_attempt + 1)):
            raise RuntimeError(f"matched-randomness attempt sequence mismatch: {path}")
        series.append([tuple(row[field] for field in TAPE_FIELDS) for row in rows])
    matches = all(item == series[0] for item in series[1:])
    receipt = {
        "schema": "ect.q256.fresh-crossed-switch-matched-randomness/v1",
        "status": "PASS" if matches else "FAIL", "labels": labels,
        "fields": list(TAPE_FIELDS), "row_counts": [len(item) for item in series],
        "series_sha256": [canonical_sha256(item) for item in series],
        "protocol_sha256": protocol_sha,
    }
    atomic_json(destination, receipt)
    if not matches:
        raise RuntimeError(f"matched-randomness tape mismatch: {labels}")


def suffix_manifest(protocol: dict, protocol_path: Path, seed: int, cell: str,
                    source_receipt: dict, output: Path, *,
                    experiment_protocol: str = schedule_switch.FRESH_N12_PROTOCOL,
                    implementation_commit: str | None = None,
                    numeric_recovery_v2: dict | None = None) -> Path:
    origin, continuation = CELLS[cell]
    source = source_receipt["training_state"]
    source_receipt_path = Path(protocol["paths"]["formal_output_root"]) / "training" / f"seed{seed}" / f"prefix_{origin}" / "source_state_receipt.json"
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": experiment_protocol,
        "run_kind": "formal", "branch": cell, "seed": seed,
        "origin_arm": origin, "continuation_arm": continuation,
        "switch_kimg": 512, "final_kimg": 1024,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit or protocol["implementation_commit"],
        "source_checkpoint_manifest_sha256": sha256_file(source_receipt_path),
        "source_state": source,
        "source_history_prefix": prepare_resume_history(source_receipt_path.parent, output),
        "immutable_output_root": str(output),
    }
    if numeric_recovery_v2 is not None:
        manifest["numeric_recovery_v2"] = numeric_recovery_v2
    path = output / "formal_run_manifest.json"
    atomic_json(path, manifest)
    atomic_json(output / "source_state_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-suffix-source-state/v1", "status": "PASS",
        "seed": seed, "cell": cell, "origin_arm": origin,
        "source_state": source, "source_state_receipt": str(source_receipt_path),
        "source_state_receipt_sha256": sha256_file(source_receipt_path),
        "protocol_sha256": sha256_file(protocol_path),
    })
    schedule_switch.load_run_manifest(path)
    return path


def validate_completed_compute(run_dir: Path, label: str) -> None:
    receipt = load_json((run_dir / "compute_completion_receipt.json").resolve(strict=True))
    if (receipt.get("status") != "PASS" or receipt.get("label") != label
            or receipt.get("exit_code") != 0 or receipt.get("hard_timeout") is not False):
        raise RuntimeError(f"recovery refused non-PASS compute receipt: {run_dir}")


def validate_prefix_source(run_dir: Path, seed: int, arm: str,
                           protocol_sha: str) -> dict:
    validate_completed_compute(run_dir, f"seed{seed}:prefix_{arm}")
    receipt_path = (run_dir / "source_state_receipt.json").resolve(strict=True)
    receipt = load_json(receipt_path)
    if (receipt.get("status") != "PASS" or receipt.get("seed") != seed
            or receipt.get("arm") != arm
            or receipt.get("source_kimg") != 512
            or receipt.get("protocol_sha256") != protocol_sha):
        raise RuntimeError(f"recovery refused prefix source receipt: {receipt_path}")
    state = receipt.get("training_state", {})
    state_path = Path(state.get("path", ""))
    if (not state_path.is_file() or state_path.is_symlink()
            or sha256_file(state_path) != state.get("sha256")):
        raise RuntimeError(f"recovery refused prefix source state: {state_path}")
    milestone = run_dir / "kimg0512" / "milestone_receipt.json"
    if (not milestone.is_file()
            or sha256_file(milestone) != receipt.get("milestone_receipt_sha256")):
        raise RuntimeError(f"recovery refused prefix milestone binding: {milestone}")
    return receipt


def validate_pass_receipt(path: Path, protocol_sha: str) -> None:
    receipt = load_json(path.resolve(strict=True))
    if receipt.get("status") != "PASS" or receipt.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(f"recovery refused receipt: {path}")


def worker(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    validate_runtime(runtime)
    gpu = args.gpu_index
    expected_seeds = [31 + gpu, 37 + gpu]
    if args.seeds and list(args.seeds) != expected_seeds:
        raise RuntimeError(f"worker GPU {gpu} must execute seeds {expected_seeds}")
    output_root = Path(protocol["paths"]["formal_output_root"])
    protocol_sha = sha256_file(protocol_path)
    protocol["protocol_sha256_runtime"] = protocol_sha
    recovery = bool(getattr(args, "recovery", False))
    for seed in expected_seeds:
        plan = assignment(seed)
        seed_root = output_root / "training" / f"seed{seed}"
        existing_seed = seed_root.exists()
        if existing_seed and not recovery:
            raise RuntimeError(f"seed root already exists: {seed_root}")
        seed_root.mkdir(parents=True, exist_ok=recovery)
        sources = {}
        if existing_seed:
            if (seed_root / "seed_completion_receipt.json").exists():
                validate_pass_receipt(seed_root / "seed_completion_receipt.json", protocol_sha)
                continue
            for arm in ("A", "B"):
                sources[arm] = validate_prefix_source(
                    seed_root / f"prefix_{arm}", seed, arm, protocol_sha)
            validate_pass_receipt(
                seed_root / "prefix_matched_randomness_receipt.json", protocol_sha)
        else:
            for arm in plan["prefix_order"]:
                run_dir = seed_root / f"prefix_{arm}"
                command = training_command(protocol, run_dir, seed, arm, gpu, prefix=True)
                run_cell(protocol, run_dir, gpu, command, f"seed{seed}:prefix_{arm}")
                sources[arm] = export_prefix(run_dir, seed, arm, protocol_sha)
            compare_tapes(
                [seed_root / "prefix_A" / "factorial_training_telemetry_v1.csv",
                 seed_root / "prefix_B" / "factorial_training_telemetry_v1.csv"],
                ["prefix_A", "prefix_B"], seed_root / "prefix_matched_randomness_receipt.json", protocol_sha,
                first_attempt=1, last_attempt=4000,
            )
        for cell in plan["suffix_order"]:
            origin, continuation = CELLS[cell]
            run_dir = seed_root / cell
            partial = run_dir.exists()
            if partial:
                if not recovery:
                    raise RuntimeError(f"suffix root already exists: {run_dir}")
                validate_completed_compute(run_dir, f"seed{seed}:{cell}")
                manifest = (run_dir / "formal_run_manifest.json").resolve(strict=True)
                loaded = schedule_switch.load_run_manifest(manifest)
                if (loaded.get("seed") != seed or loaded.get("branch") != cell
                        or loaded.get("protocol_sha256") != protocol_sha):
                    raise RuntimeError(f"recovery refused suffix manifest: {manifest}")
            else:
                run_dir.mkdir(exist_ok=False)
                manifest = suffix_manifest(protocol, protocol_path, seed, cell, sources[origin], run_dir)
                source = Path(sources[origin]["training_state"]["path"])
                command = training_command(protocol, run_dir, seed, continuation, gpu, prefix=False,
                                           manifest=manifest, source=source)
                run_cell(protocol, run_dir, gpu, command, f"seed{seed}:{cell}")
            export = [str(Path(runtime["environment_prefix"]) / "bin" / "python"),
                      str(REPO_ROOT / "analysis/q256_schedule_switch_v1/export_milestones.py"),
                      "--run-dir", str(run_dir), "--manifest", str(manifest)]
            trajectory = run_dir / "trajectory_completion_receipt.json"
            if trajectory.exists():
                validate_pass_receipt(trajectory, protocol_sha)
            else:
                if partial:
                    export.append("--recover-partial-export")
                subprocess.run(export, cwd=REPO_ROOT, env=cell_environment(gpu, runtime), check=True)
                validate_pass_receipt(trajectory, protocol_sha)
        compare_tapes(
            [seed_root / cell / "schedule_switch_training_telemetry_v1.csv" for cell in ("AA", "AB", "BA", "BB")],
            ["AA", "AB", "BA", "BB"], seed_root / "suffix_matched_randomness_receipt.json", protocol_sha,
            first_attempt=4001, last_attempt=8000,
        )
        atomic_json(seed_root / "seed_completion_receipt.json", {
            "schema": "ect.q256.fresh-crossed-switch-seed-completion/v1", "status": "PASS",
            "seed": seed, "gpu_index": gpu, "gpu_uuid": protocol["gpus"][gpu]["uuid"],
            "prefix_order": plan["prefix_order"], "suffix_order": plan["suffix_order"],
            "protocol_sha256": protocol_sha,
        })


def authorize_recovery(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    validate_protocol(load_json(protocol_path), protocol_path)
    failure_path = args.original_failure.resolve(strict=True)
    failure = load_json(failure_path)
    if (failure.get("status") != "FAIL" or failure.get("automatic_retry_count") != 0
            or failure.get("protocol_sha256") != sha256_file(protocol_path)):
        raise RuntimeError("manual recovery authorization requires the canonical first-attempt FAIL receipt")
    if not HEX40.fullmatch(args.repair_commit):
        raise RuntimeError("invalid repair commit")
    archive_path = args.archive.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    copy_exclusive(failure_path, archive_path)
    atomic_json(args.destination.resolve(), {
        "schema": "ect.q256.fresh-crossed-switch-manual-recovery-authorization/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": utc_now(),
        "scope": "repair receipt serialization, finish partial exports, and continue missing cells",
        "completed_compute_must_not_be_rerun": True,
        "automatic_retry_count": 0, "manual_recovery_index": 1,
        "protocol_sha256": sha256_file(protocol_path),
        "repair_commit": args.repair_commit,
        "original_failure_receipt": {"path": str(archive_path),
                                     "sha256": sha256_file(archive_path)},
    })


def recovery_launch(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    protocol_sha = sha256_file(protocol_path)
    preflight_receipt = load_json(args.preflight_receipt.resolve(strict=True))
    if (preflight_receipt.get("status") != "PASS"
            or preflight_receipt.get("protocol_sha256") != protocol_sha):
        raise RuntimeError("manual recovery requires the original PASS preflight receipt")
    authorization = load_json(args.authorization.resolve(strict=True))
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    archived = authorization.get("original_failure_receipt", {})
    if (authorization.get("status") != "AUTHOR_APPROVED"
            or authorization.get("protocol_sha256") != protocol_sha
            or authorization.get("repair_commit") != current_commit
            or authorization.get("manual_recovery_index") != 1
            or authorization.get("completed_compute_must_not_be_rerun") is not True
            or sha256_file(Path(archived.get("path", ""))) != archived.get("sha256")):
        raise RuntimeError("manual recovery authorization binding mismatch")
    output_root = Path(protocol["paths"]["formal_output_root"])
    if not output_root.is_dir():
        raise RuntimeError("formal output root is missing")
    logs = output_root / "logs"
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    processes = []
    registry = []
    for gpu in range(6):
        command = [python, str(Path(__file__).resolve()), "recovery-worker",
                   "--protocol", str(protocol_path), "--gpu-index", str(gpu),
                   "--seeds", str(31 + gpu), str(37 + gpu)]
        log_path = logs / f"gpu{gpu}.recovery1.log"
        log_handle = log_path.open("xb")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=os.environ.copy(),
                                   stdout=log_handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append((gpu, process, log_handle, log_path))
        registry.append({"gpu_index": gpu, "gpu_uuid": protocol["gpus"][gpu]["uuid"],
                         "seeds": [31 + gpu, 37 + gpu], "pid": process.pid,
                         "log_path": str(log_path), "command": command})
    atomic_json(logs / "recovery1_worker_registry.json", {
        "schema": "ect.q256.fresh-crossed-switch-recovery-worker-registry/v1",
        "status": "STARTED", "started_at": utc_now(), "workers": registry,
        "protocol_sha256": protocol_sha, "repair_commit": current_commit,
    })
    failures = []
    for gpu, process, log_handle, log_path in processes:
        returncode = process.wait(); log_handle.close()
        if returncode != 0:
            failures.append({"gpu_index": gpu, "exit_code": returncode,
                             "log_path": str(log_path)})
    completed = len(list((output_root / "training").glob("seed*/seed_completion_receipt.json")))
    receipt = {
        "schema": "ect.q256.fresh-crossed-switch-training-matrix-completion/v1",
        "status": "PASS" if not failures and completed == 12 else "FAIL",
        "ended_at": utc_now(), "worker_failures": failures,
        "completed_seed_receipts": completed, "expected_seed_receipts": 12,
        "protocol_sha256": protocol_sha, "automatic_retry_count": 0,
        "manual_recovery_count": 1, "repair_commit": current_commit,
        "original_failure_receipt_sha256": archived["sha256"],
    }
    atomic_json(output_root / "training_matrix_recovery1_completion_receipt.json", receipt)
    atomic_json(output_root / "training_matrix_completion_receipt.json", receipt, overwrite=True)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"manual recovery failed closed: {receipt}")
    print(json.dumps(receipt, sort_keys=True))


def directory_manifest(root: Path) -> list[dict]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"regular directory required: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"archive refuses symlink: {path}")
        if path.is_file():
            records.append({"path": str(path.relative_to(root)),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return records


def directory_stat_manifest(root: Path) -> list[dict]:
    """Identity manifest suitable for an atomic same-filesystem directory rename."""
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"regular directory required: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"archive refuses symlink: {path}")
        if path.is_file():
            stat = path.stat()
            records.append({"path": str(path.relative_to(root)), "bytes": stat.st_size,
                            "device": stat.st_dev, "inode": stat.st_ino,
                            "mtime_ns": stat.st_mtime_ns})
    return records


def authorize_numeric_recovery2(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    validate_protocol(load_json(protocol_path), protocol_path)
    protocol_sha = sha256_file(protocol_path)
    matrix_path = args.recovery1_failure.resolve(strict=True)
    matrix = load_json(matrix_path)
    if (matrix.get("status") != "FAIL" or matrix.get("protocol_sha256") != protocol_sha
            or matrix.get("manual_recovery_count") != 1
            or matrix.get("automatic_retry_count") != 0
            or matrix.get("completed_seed_receipts") != 11):
        raise RuntimeError("numeric recovery v2 requires the completed recovery1 FAIL receipt")
    failed_path = args.failed_compute.resolve(strict=True)
    failed = load_json(failed_path)
    if (failed.get("status") != "FAIL" or failed.get("label") != "seed38:AB"
            or failed.get("exit_code") != 1 or failed.get("hard_timeout") is not False):
        raise RuntimeError("numeric recovery v2 requires the exact seed38/AB failure")
    if not HEX40.fullmatch(args.repair_commit):
        raise RuntimeError("invalid numeric recovery v2 repair commit")
    output_root = Path(load_json(protocol_path)["paths"]["formal_output_root"])
    archive_root = output_root / "archive" / "manual-recovery-2"
    archive_run = archive_root / "seed38_AB_failed_attempt1"
    if archive_run.exists():
        raise RuntimeError("numeric recovery v2 archive target already exists")
    archived_matrix = archive_root / "training_matrix_recovery1_completion_receipt.json"
    archive_root.mkdir(parents=True, exist_ok=True)
    copy_exclusive(matrix_path, archived_matrix)
    atomic_json(args.destination.resolve(), {
        "schema": "ect.q256.fresh-crossed-switch-numeric-recovery-v2-authorization/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": utc_now(),
        "manual_recovery_index": 2, "automatic_retry_count": 0,
        "scope": "archive failed seed38/AB; rerun seed38/AB and missing seed38/AA only",
        "protocol_sha256": protocol_sha, "repair_commit": args.repair_commit,
        "max_recoverable_nonfinite_loss_attempts_per_cell": 1,
        "managed_overflow_requirements": {
            "amp_step_skipped": True, "sanitized_gradients_finite": True,
            "update_norm_zero": True, "model_and_ema_finite": True,
            "target_and_denominator_factors_finite": True,
            "denominator_positive": True,
        },
        "recovery1_failure": {"path": str(archived_matrix),
                              "sha256": sha256_file(archived_matrix)},
        "failed_compute_receipt_sha256": sha256_file(failed_path),
        "failed_run_dir": str(failed_path.parent),
        "archive_run_dir": str(archive_run),
        "original_failure_must_be_preserved": True,
        "quality_metrics_observed_before_amendment": False,
    })


def numeric_recovery2(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    protocol_sha = sha256_file(protocol_path)
    authorization_path = args.authorization.resolve(strict=True)
    authorization = load_json(authorization_path)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    recovery1 = authorization.get("recovery1_failure", {})
    if (authorization.get("status") != "AUTHOR_APPROVED"
            or authorization.get("manual_recovery_index") != 2
            or authorization.get("protocol_sha256") != protocol_sha
            or authorization.get("repair_commit") != current_commit
            or authorization.get("max_recoverable_nonfinite_loss_attempts_per_cell") != 1
            or authorization.get("original_failure_must_be_preserved") is not True
            or sha256_file(Path(recovery1.get("path", ""))) != recovery1.get("sha256")):
        raise RuntimeError("numeric recovery v2 authorization binding mismatch")
    output_root = Path(protocol["paths"]["formal_output_root"])
    seed_root = output_root / "training" / "seed38"
    failed_run = Path(authorization["failed_run_dir"])
    archive_run = Path(authorization["archive_run_dir"])
    failed_receipt = failed_run / "compute_completion_receipt.json"
    if sha256_file(failed_receipt) != authorization["failed_compute_receipt_sha256"]:
        raise RuntimeError("seed38/AB failure changed after authorization")
    before = directory_manifest(failed_run)
    archive_run.parent.mkdir(parents=True, exist_ok=True)
    os.rename(failed_run, archive_run)
    after = directory_manifest(archive_run)
    if before != after:
        raise RuntimeError("failed seed38/AB archive identity mismatch")
    archive_receipt = archive_run.parent / "seed38_AB_failed_attempt1_archive_receipt.json"
    atomic_json(archive_receipt, {
        "schema": "ect.q256.failed-run-archive/v1", "status": "PASS",
        "archived_at": utc_now(), "source_path": str(failed_run),
        "archive_path": str(archive_run), "file_count": len(after),
        "directory_manifest_sha256": canonical_sha256(after),
        "failed_compute_receipt_sha256": authorization["failed_compute_receipt_sha256"],
        "protocol_sha256": protocol_sha,
    })
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    validate_runtime(runtime)
    sources = {
        arm: validate_prefix_source(seed_root / f"prefix_{arm}", 38, arm, protocol_sha)
        for arm in ("A", "B")
    }
    numeric_binding = {
        "authorization_sha256": sha256_file(authorization_path),
        "failed_compute_receipt_sha256": authorization["failed_compute_receipt_sha256"],
        "max_recoverable_nonfinite_loss_attempts": 1,
    }
    for cell in ("AB", "AA"):
        origin, continuation = CELLS[cell]
        run_dir = seed_root / cell
        run_dir.mkdir(exist_ok=False)
        manifest = suffix_manifest(
            protocol, protocol_path, 38, cell, sources[origin], run_dir,
            experiment_protocol=(
                schedule_switch.FRESH_N12_NUMERIC_RECOVERY_V2_PROTOCOL
            ),
            implementation_commit=current_commit,
            numeric_recovery_v2=numeric_binding,
        )
        command = training_command(
            protocol, run_dir, 38, continuation, 1, prefix=False,
            manifest=manifest, source=Path(sources[origin]["training_state"]["path"]),
        )
        run_cell(protocol, run_dir, 1, command, f"seed38:{cell}")
        subprocess.run(
            [str(Path(runtime["environment_prefix"]) / "bin" / "python"),
             str(REPO_ROOT / "analysis/q256_schedule_switch_v1/export_milestones.py"),
             "--run-dir", str(run_dir), "--manifest", str(manifest)],
            cwd=REPO_ROOT, env=cell_environment(1, runtime), check=True,
        )
        validate_pass_receipt(run_dir / "trajectory_completion_receipt.json", protocol_sha)
    compare_tapes(
        [seed_root / cell / "schedule_switch_training_telemetry_v1.csv"
         for cell in ("AA", "AB", "BA", "BB")],
        ["AA", "AB", "BA", "BB"],
        seed_root / "suffix_matched_randomness_receipt.json", protocol_sha,
        first_attempt=4001, last_attempt=8000,
    )
    plan = assignment(38)
    atomic_json(seed_root / "seed_completion_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-seed-completion/v1",
        "status": "PASS", "seed": 38, "gpu_index": 1,
        "gpu_uuid": protocol["gpus"][1]["uuid"],
        "prefix_order": plan["prefix_order"], "suffix_order": plan["suffix_order"],
        "protocol_sha256": protocol_sha, "numeric_recovery_v2": numeric_binding,
    })
    completed = len(list((output_root / "training").glob("seed*/seed_completion_receipt.json")))
    if completed != 12:
        raise RuntimeError(f"numeric recovery v2 completed seed count mismatch: {completed}")
    receipt = {
        "schema": "ect.q256.fresh-crossed-switch-training-matrix-completion/v1",
        "status": "PASS", "ended_at": utc_now(), "worker_failures": [],
        "completed_seed_receipts": 12, "expected_seed_receipts": 12,
        "protocol_sha256": protocol_sha, "automatic_retry_count": 0,
        "manual_recovery_count": 2, "manual_numeric_retry_count": 1,
        "repair_commit": current_commit,
        "numeric_recovery_v2_authorization_sha256": sha256_file(authorization_path),
        "recovery1_failure_receipt_sha256": recovery1["sha256"],
        "failed_run_archive_receipt_sha256": sha256_file(archive_receipt),
    }
    atomic_json(output_root / "training_matrix_recovery2_completion_receipt.json", receipt)
    atomic_json(output_root / "training_matrix_completion_receipt.json", receipt, overwrite=True)
    print(json.dumps(receipt, sort_keys=True))


def authorize_eleven_seed(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    protocol_sha = sha256_file(protocol_path)
    if not HEX40.fullmatch(args.amendment_commit):
        raise RuntimeError("invalid eleven-seed amendment commit")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if head != args.amendment_commit:
        raise RuntimeError("eleven-seed amendment commit is not current HEAD")
    numeric_auth_path = args.numeric_recovery2_authorization.resolve(strict=True)
    numeric_auth = load_json(numeric_auth_path)
    if (numeric_auth.get("status") != "AUTHOR_APPROVED"
            or numeric_auth.get("protocol_sha256") != protocol_sha
            or numeric_auth.get("manual_recovery_index") != 2):
        raise RuntimeError("eleven-seed amendment requires the numeric recovery v2 authorization")
    failed_path = args.failed_compute.resolve(strict=True)
    failed = load_json(failed_path)
    if (failed.get("status") != "FAIL" or failed.get("label") != "seed38:AB"
            or failed.get("exit_code") != 1 or failed.get("hard_timeout") is not False):
        raise RuntimeError("eleven-seed amendment requires the terminal seed38/AB v2 failure")
    output_root = Path(protocol["paths"]["formal_output_root"])
    if (output_root / "training" / "seed38" / "AA").exists():
        raise RuntimeError("seed38/AA must remain absent for the eleven-seed amendment")
    if (output_root / "evaluation").exists():
        raise RuntimeError("eleven-seed amendment requires evaluation to be unstarted")
    completed = []
    seed_receipt_hashes = {}
    for seed in ELEVEN_SEEDS:
        receipt_path = output_root / "training" / f"seed{seed}" / "seed_completion_receipt.json"
        receipt = load_json(receipt_path.resolve(strict=True))
        if (receipt.get("status") != "PASS" or receipt.get("seed") != seed
                or receipt.get("protocol_sha256") != protocol_sha):
            raise RuntimeError(f"invalid completed seed receipt: seed{seed}")
        completed.append(seed)
        seed_receipt_hashes[str(seed)] = sha256_file(receipt_path)
    payload = {
        "schema": "ect.q256.fresh-crossed-switch-eleven-seed-amendment/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": utc_now(),
        "protocol_sha256": protocol_sha, "amendment_commit": head,
        "scope": "exclude terminally failed seed38 and continue with the eleven complete seeds",
        "excluded_seed": ELEVEN_SEED_EXCLUSION, "included_seeds": completed,
        "expected_evaluation_jobs": ELEVEN_JOB_COUNT,
        "original_n12_claim_abandoned": True,
        "analysis_population": "AUTHOR_AMENDED_N11_COMPLETE_CASE",
        "minimum_negative_directions_for_strong_success": 10,
        "decode_forbidden_before_full_amended_seal": True,
        "automatic_retry_count": 0,
        "quality_metrics_observed_before_amendment": False,
        "numeric_recovery2_authorization_sha256": sha256_file(numeric_auth_path),
        "terminal_failed_compute_receipt_sha256": sha256_file(failed_path),
        "seed_completion_receipt_sha256": seed_receipt_hashes,
    }
    destination = args.destination.resolve()
    atomic_json(destination, payload)
    atomic_json(output_root / "training_matrix_11seed_completion_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-training-matrix-amended/v1",
        "status": "PASS", "ended_at": utc_now(),
        "protocol_sha256": protocol_sha,
        "eleven_seed_authorization_sha256": sha256_file(destination),
        "completed_seed_receipts": len(ELEVEN_SEEDS),
        "expected_seed_receipts": len(ELEVEN_SEEDS),
        "included_seeds": list(ELEVEN_SEEDS), "excluded_seed": ELEVEN_SEED_EXCLUSION,
        "original_n12_claim_abandoned": True, "automatic_retry_count": 0,
    })
    print(json.dumps({"status": "PASS", "included_seeds": list(ELEVEN_SEEDS),
                      "expected_evaluation_jobs": ELEVEN_JOB_COUNT}, sort_keys=True))


def authorize_evaluation_recovery1(args: argparse.Namespace) -> None:
    """Bind one author-approved, non-automatic recovery of the failed blind matrix."""
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    protocol_sha = sha256_file(protocol_path)
    eleven_path = args.eleven_seed_authorization.resolve(strict=True)
    eleven = validate_eleven_seed_authorization(eleven_path, protocol_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if not HEX40.fullmatch(args.amendment_commit) or head != args.amendment_commit:
        raise RuntimeError("evaluation recovery amendment is not current HEAD")
    matrix_path = args.matrix_failure.resolve(strict=True)
    matrix = load_json(matrix_path)
    expected_failures = [{"gpu_index": gpu, "exit_code": 1} for gpu in range(6)]
    if (matrix.get("status") != "FAIL"
            or matrix.get("expected_receipts") != ELEVEN_JOB_COUNT
            or matrix.get("validated_receipts") != 0
            or matrix.get("automatic_retry_count") != 0
            or matrix.get("worker_failures") != expected_failures):
        raise RuntimeError("evaluation recovery requires the exact six-worker matrix failure")
    output_root = Path(protocol["paths"]["formal_output_root"])
    eval_source = output_root / "evaluation_11seed"
    failure_receipts = sorted(eval_source.glob("jobs/*/failure_receipt.json"))
    if len(failure_receipts) != 6:
        raise RuntimeError("evaluation recovery requires six preserved job failures")
    for receipt_path in failure_receipts:
        receipt = load_json(receipt_path)
        if (receipt.get("status") != "FAIL" or receipt.get("exit_code") != 1
                or receipt.get("hard_timeout") is not False
                or receipt.get("automatic_retry_count") != 0):
            raise RuntimeError(f"unexpected evaluation failure receipt: {receipt_path}")
    legacy_metric_artifacts = sorted(eval_source.glob("jobs/*/metric-*.jsonl"))
    if len(legacy_metric_artifacts) != 6:
        raise RuntimeError("evaluation recovery requires exactly six preserved unsealed metric artifacts")
    if any(path.name != "metric-kid50k_full.jsonl" or path.stat().st_size <= 0
           for path in legacy_metric_artifacts):
        raise RuntimeError("unexpected legacy metric artifact identity; contents remain unread")
    if list((eval_source / "receipts").glob("*.json")) \
            or list((eval_source / "seals").glob("*.json")):
        raise RuntimeError("evaluation recovery requires zero validated receipts and zero seals")
    control_root = Path(protocol["paths"]["control_root"])
    old_public = control_root / "frozen_242_job_evaluation_manifest_11seed.json"
    old_private = control_root / "sealed_private_evaluation_map_11seed.json"
    for path in (old_public, old_private):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing failed evaluation control artifact: {path}")
    archive_root = output_root / "archive" / "manual-evaluation-recovery-1"
    eval_archive = archive_root / "evaluation_11seed_failed_attempt1"
    public_archive = archive_root / old_public.name
    private_archive = archive_root / old_private.name
    for path in (archive_root, eval_archive, public_archive, private_archive,
                 output_root / "evaluation_11seed_recovery1",
                 control_root / "frozen_242_job_evaluation_manifest_11seed_recovery1.json",
                 control_root / "sealed_private_evaluation_map_11seed_recovery1.json"):
        if path.exists():
            raise RuntimeError(f"evaluation recovery destination already exists: {path}")
    cache_source = args.cache_source.resolve(strict=True)
    if not cache_source.is_dir() or cache_source.is_symlink():
        raise RuntimeError("regular source evaluator cache required")
    cache_destination = args.cache_destination.resolve()
    if cache_destination.exists():
        raise RuntimeError("fresh evaluator cache destination required")
    free = shutil.disk_usage(cache_destination.parent.resolve(strict=True)).free
    if free < 400 * 1024**3:
        raise RuntimeError("evaluation recovery requires at least 400 GiB free")
    payload = {
        "schema": "ect.q256.eleven-seed-evaluation-recovery/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": utc_now(),
        "manual_evaluation_recovery_index": 1,
        "protocol_sha256": protocol_sha, "amendment_commit": head,
        "expected_evaluation_jobs": ELEVEN_JOB_COUNT, "automatic_retry_count": 0,
        "original_failure_must_be_preserved": True,
        "decode_forbidden_before_full_recovery_seal": True,
        "quality_metrics_observed_before_amendment": False,
        "legacy_unsealed_metric_artifact_count": len(legacy_metric_artifacts),
        "legacy_unsealed_metric_artifact_names": [path.name for path in legacy_metric_artifacts],
        "legacy_metric_artifact_contents_read": False,
        "eleven_seed_authorization_path": str(eleven_path),
        "eleven_seed_authorization_sha256": sha256_file(eleven_path),
        "eleven_seed_amendment_commit": eleven["amendment_commit"],
        "matrix_failure_path": str(matrix_path),
        "matrix_failure_sha256": sha256_file(matrix_path),
        "failed_evaluation_source": str(eval_source),
        "failed_evaluation_archive": str(eval_archive),
        "old_public_manifest_source": str(old_public),
        "old_public_manifest_archive": str(public_archive),
        "old_public_manifest_sha256": sha256_file(old_public),
        "old_private_map_source": str(old_private),
        "old_private_map_archive": str(private_archive),
        "old_private_map_sha256": sha256_file(old_private),
        "cache_source": str(cache_source), "cache_destination": str(cache_destination),
        "evaluation_dir": "evaluation_11seed_recovery1",
        "analysis_dir": "analysis_11seed_recovery1",
        "minimum_free_gib": 400,
    }
    atomic_json(args.destination.resolve(), payload)
    print(json.dumps({"status": "AUTHOR_APPROVED", "manual_evaluation_recovery_index": 1,
                      "expected_evaluation_jobs": ELEVEN_JOB_COUNT}, sort_keys=True))


def prepare_evaluation_recovery1(args: argparse.Namespace) -> None:
    """Preserve the failed attempt, clone cache, and pass a non-metric storage gate."""
    protocol_path = args.protocol.resolve(strict=True)
    authorization_path = args.authorization.resolve(strict=True)
    authorization = validate_evaluation_recovery1_authorization(
        authorization_path, protocol_path, require_commit=True
    )
    if sha256_file(Path(authorization["matrix_failure_path"])) != authorization["matrix_failure_sha256"]:
        raise RuntimeError("failed matrix changed after recovery authorization")
    source = Path(authorization["failed_evaluation_source"])
    archive = Path(authorization["failed_evaluation_archive"])
    before = directory_stat_manifest(source)
    archive.parent.mkdir(parents=True, exist_ok=False)
    os.rename(source, archive)
    after = directory_stat_manifest(archive)
    if before != after:
        raise RuntimeError("failed blind evaluation archive identity mismatch")
    control_specs = (
        (Path(authorization["old_public_manifest_source"]),
         Path(authorization["old_public_manifest_archive"]),
         authorization["old_public_manifest_sha256"]),
        (Path(authorization["old_private_map_source"]),
         Path(authorization["old_private_map_archive"]),
         authorization["old_private_map_sha256"]),
    )
    for src, dst, expected_sha in control_specs:
        if sha256_file(src) != expected_sha:
            raise RuntimeError(f"old control artifact changed: {src}")
        os.rename(src, dst)
        if sha256_file(dst) != expected_sha:
            raise RuntimeError(f"old control archive identity mismatch: {dst}")
    cache_source = Path(authorization["cache_source"])
    cache_destination = Path(authorization["cache_destination"])
    source_manifest = directory_manifest(cache_source)
    shutil.copytree(cache_source, cache_destination, copy_function=shutil.copy2)
    destination_manifest = directory_manifest(cache_destination)
    if source_manifest != destination_manifest:
        raise RuntimeError("evaluator cache copy identity mismatch")
    free_before = shutil.disk_usage(cache_destination).free
    if free_before < authorization["minimum_free_gib"] * 1024**3:
        raise RuntimeError("free-space gate failed after evaluator cache copy")
    smoke = cache_destination / ".q256_recovery1_storage_smoke.tmp"
    block = b"\0" * (8 * 1024 * 1024)
    with smoke.open("xb") as handle:
        for _ in range(128):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if smoke.stat().st_size != 1024**3:
        raise RuntimeError("storage smoke file size mismatch")
    smoke.unlink()
    free_after = shutil.disk_usage(cache_destination).free
    receipt = {
        "schema": "ect.q256.eleven-seed-evaluation-recovery-preparation/v1",
        "status": "PASS", "prepared_at": utc_now(),
        "protocol_sha256": authorization["protocol_sha256"],
        "evaluation_recovery_authorization_sha256": sha256_file(authorization_path),
        "failed_evaluation_archive": str(archive), "archived_file_count": len(after),
        "archived_directory_manifest_sha256": canonical_sha256(after),
        "cache_source": str(cache_source), "cache_destination": str(cache_destination),
        "cache_file_count": len(destination_manifest),
        "cache_directory_manifest_sha256": canonical_sha256(destination_manifest),
        "storage_smoke_bytes_written_and_fsynced": 1024**3,
        "storage_smoke_temp_removed": not smoke.exists(),
        "free_bytes_after_smoke": free_after, "metrics_executed": False,
        "metric_values_observed": False,
    }
    atomic_json(archive.parent / "evaluation_recovery1_preparation_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "cache_file_count": len(destination_manifest),
                      "free_bytes_after_smoke": free_after}, sort_keys=True))


def launch(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    preflight_receipt = load_json(args.preflight_receipt.resolve(strict=True))
    if preflight_receipt.get("status") != "PASS":
        raise RuntimeError("formal launch requires PASS preflight receipt")
    if preflight_receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("preflight receipt is not bound to frozen protocol")
    output_root = Path(protocol["paths"]["formal_output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    archive = output_root / "archive"
    logs.mkdir()
    archive.mkdir()
    runtime = load_json(Path(protocol["assets"]["runtime_manifest"]["path"]))
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    processes = []
    registry = []
    for gpu in range(6):
        command = [
            python, str(Path(__file__).resolve()), "worker", "--protocol", str(protocol_path),
            "--gpu-index", str(gpu), "--seeds", str(31 + gpu), str(37 + gpu),
        ]
        log_path = logs / f"gpu{gpu}.log"
        log_handle = log_path.open("xb")
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=os.environ.copy(), stdout=log_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes.append((gpu, process, log_handle, log_path))
        registry.append(
            {"gpu_index": gpu, "gpu_uuid": protocol["gpus"][gpu]["uuid"],
             "seeds": [31 + gpu, 37 + gpu], "pid": process.pid,
             "log_path": str(log_path), "command": command}
        )
    atomic_json(logs / "worker_registry.json", {
        "schema": "ect.q256.fresh-crossed-switch-worker-registry/v1",
        "status": "STARTED", "started_at": utc_now(), "workers": registry,
        "protocol_sha256": sha256_file(protocol_path),
    })
    failures = []
    for gpu, process, log_handle, log_path in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append({"gpu_index": gpu, "exit_code": returncode, "log_path": str(log_path)})
    completed = len(list((output_root / "training").glob("seed*/seed_completion_receipt.json")))
    receipt = {
        "schema": "ect.q256.fresh-crossed-switch-training-matrix-completion/v1",
        "status": "PASS" if not failures and completed == 12 else "FAIL",
        "ended_at": utc_now(), "worker_failures": failures,
        "completed_seed_receipts": completed, "expected_seed_receipts": 12,
        "protocol_sha256": sha256_file(protocol_path), "automatic_retry_count": 0,
    }
    atomic_json(output_root / "training_matrix_completion_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"training matrix failed closed: {receipt}")
    print(json.dumps(receipt, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-protocol")
    freeze.add_argument("--implementation-commit", required=True)
    freeze.add_argument("--runtime-manifest", type=Path, required=True)
    freeze.add_argument("--runtime-parity", type=Path, required=True)
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--transfer", type=Path, required=True)
    freeze.add_argument("--evaluator-source", type=Path, required=True)
    freeze.add_argument("--detector", type=Path, required=True)
    freeze.add_argument("--real-features", type=Path, nargs=2, required=True)
    freeze.add_argument("--storage-sample-state", type=Path, required=True)
    freeze.add_argument("--storage-sample-snapshot", type=Path, required=True)
    freeze.add_argument("--repo", type=Path, required=True)
    freeze.add_argument("--asset-root", type=Path, required=True)
    freeze.add_argument("--evaluator-cache", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--control-root", type=Path, required=True)
    freeze.add_argument("--destination", type=Path, required=True)
    freeze.set_defaults(func=freeze_protocol)
    pre = sub.add_parser("preflight")
    pre.add_argument("--protocol", type=Path, required=True)
    pre.add_argument("--receipt", type=Path, required=True)
    pre.add_argument("--minimum-free-gib", type=int, default=500)
    pre.set_defaults(func=preflight)
    work = sub.add_parser("worker")
    work.add_argument("--protocol", type=Path, required=True)
    work.add_argument("--gpu-index", type=int, choices=range(6), required=True)
    work.add_argument("--seeds", type=int, nargs="*")
    work.set_defaults(func=worker)
    recovery_work = sub.add_parser("recovery-worker")
    recovery_work.add_argument("--protocol", type=Path, required=True)
    recovery_work.add_argument("--gpu-index", type=int, choices=range(6), required=True)
    recovery_work.add_argument("--seeds", type=int, nargs="*")
    recovery_work.set_defaults(func=worker, recovery=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--protocol", type=Path, required=True)
    launch_parser.add_argument("--preflight-receipt", type=Path, required=True)
    launch_parser.set_defaults(func=launch)
    authorize = sub.add_parser("authorize-recovery")
    authorize.add_argument("--protocol", type=Path, required=True)
    authorize.add_argument("--original-failure", type=Path, required=True)
    authorize.add_argument("--archive", type=Path, required=True)
    authorize.add_argument("--repair-commit", required=True)
    authorize.add_argument("--destination", type=Path, required=True)
    authorize.set_defaults(func=authorize_recovery)
    recovery = sub.add_parser("recovery-launch")
    recovery.add_argument("--protocol", type=Path, required=True)
    recovery.add_argument("--preflight-receipt", type=Path, required=True)
    recovery.add_argument("--authorization", type=Path, required=True)
    recovery.set_defaults(func=recovery_launch)
    authorize_v2 = sub.add_parser("authorize-numeric-recovery2")
    authorize_v2.add_argument("--protocol", type=Path, required=True)
    authorize_v2.add_argument("--recovery1-failure", type=Path, required=True)
    authorize_v2.add_argument("--failed-compute", type=Path, required=True)
    authorize_v2.add_argument("--repair-commit", required=True)
    authorize_v2.add_argument("--destination", type=Path, required=True)
    authorize_v2.set_defaults(func=authorize_numeric_recovery2)
    recovery_v2 = sub.add_parser("numeric-recovery2")
    recovery_v2.add_argument("--protocol", type=Path, required=True)
    recovery_v2.add_argument("--authorization", type=Path, required=True)
    recovery_v2.set_defaults(func=numeric_recovery2)
    eleven = sub.add_parser("authorize-eleven-seed")
    eleven.add_argument("--protocol", type=Path, required=True)
    eleven.add_argument("--numeric-recovery2-authorization", type=Path, required=True)
    eleven.add_argument("--failed-compute", type=Path, required=True)
    eleven.add_argument("--amendment-commit", required=True)
    eleven.add_argument("--destination", type=Path, required=True)
    eleven.set_defaults(func=authorize_eleven_seed)
    evaluation_authorize = sub.add_parser("authorize-evaluation-recovery1")
    evaluation_authorize.add_argument("--protocol", type=Path, required=True)
    evaluation_authorize.add_argument("--eleven-seed-authorization", type=Path, required=True)
    evaluation_authorize.add_argument("--matrix-failure", type=Path, required=True)
    evaluation_authorize.add_argument("--amendment-commit", required=True)
    evaluation_authorize.add_argument("--cache-source", type=Path, required=True)
    evaluation_authorize.add_argument("--cache-destination", type=Path, required=True)
    evaluation_authorize.add_argument("--destination", type=Path, required=True)
    evaluation_authorize.set_defaults(func=authorize_evaluation_recovery1)
    evaluation_prepare = sub.add_parser("prepare-evaluation-recovery1")
    evaluation_prepare.add_argument("--protocol", type=Path, required=True)
    evaluation_prepare.add_argument("--authorization", type=Path, required=True)
    evaluation_prepare.set_defaults(func=prepare_evaluation_recovery1)
    verify = sub.add_parser("verify-protocol")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.set_defaults(func=lambda args: validate_protocol(load_json(args.protocol), args.protocol))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
