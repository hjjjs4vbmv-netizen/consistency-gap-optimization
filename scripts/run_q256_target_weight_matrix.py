#!/usr/bin/env python3
"""Fail-closed launcher for the frozen q256 target/weight 2x2 factorial.

The default ``matrix`` command only prints a plan; matrix execution needs an
explicit ``--execute``.  Every fresh arm launch also needs a source-bound PASS
authorization receipt.  The ``arm`` subcommand is normally reached through
run_q256_target_weight_arm.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_SCRIPT = REPO_ROOT / "scripts" / "run_q256_target_weight_arm.sh"
PREREGISTRATION = (
    REPO_ROOT / "analysis" / "q256_target_weight_factorial" / "preregistration.json"
)
EXPERIMENT_ID = "q256-target-weight-factorial"
FACTORIAL_PROTOCOL = "q256_target_weight_v1"
AUTHORIZATION_SCHEMA = "ect.q256.target-weight-factorial-launch-authorization/v2"
LAUNCH_SCHEMA = "ect.q256.target-weight-factorial-launch/v2"
MATRIX_SCHEMA = "ect.q256.target-weight-factorial-matrix/v2"
EXPECTED_BRANCH = "experiment/q256-target-weight-factorial"
VALIDATION_FILENAME = "q256_target_weight_arm_validation_v2.json"
HASH_RECEIPT_FILENAME = "q256_target_weight_arm_artifact_hashes_v2.json"
CORE_ARM_ARTIFACTS = (
    "launch_manifest.json",
    "training_options.json",
    "initial_state_receipt_v1.json",
    "factorial_training_telemetry_v1.csv",
    "train_summary.csv",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "log.txt",
)
VALIDATION_SCHEMA = "ect.q256.target-weight-arm-validation/v2"
HASH_RECEIPT_SCHEMA = "ect.q256.target-weight-arm-artifact-hashes/v2"
RUNNER_COMPLETION_SCHEMA = "ect.q256.target-weight-factorial-runner-completion/v2"
PLANNED_PAUSE_STATUS = "PLANNED_PAUSE_PASS"
PLANNED_PAUSE_ATTEMPTS = 16
PLANNED_PAUSE_EVIDENCE_DIR = "exact_resume_pause_evidence_v1"
PLANNED_PAUSE_EVIDENCE_MANIFEST = "pause_evidence_manifest.json"
PLANNED_PAUSE_EVIDENCE_SCHEMA = "ect.q256.exact-resume-pause-evidence/v1"
ROLE_E_AB_PARITY_SCHEMA = "ect.q256.target-weight-role-e-ab-parity/v1"
SMOKE_MATRIX_VALIDATION_SCHEMA = (
    "ect.q256.target-weight-smoke-matrix-validation/v2"
)
EXPECTED_DATASET_SHA256 = (
    "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
)
EXPECTED_TRANSFER_SHA256 = (
    "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
)
DEFAULT_DATA = Path(
    "/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip"
)
DEFAULT_TRANSFER = Path(
    "/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
)
DEFAULT_RUNS_ROOT = Path(
    "/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819"
)
DEFAULT_RUNTIME_SANDBOX = Path("/data/temp/ect001-pytorch2401-sandbox")
RUNTIME_BIND_SPECS = (
    "/data/raw:/data/raw",
    "/data/temp:/data/temp",
)
DEFAULT_TRAINING_PYTHON = "python"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_TORCH_VERSION = "2.2.0a0+81ea7a4"
EXPECTED_TORCH_CUDA_VERSION = "12.3"
IN_SANDBOX_ENV = "ECT_Q256_LAUNCHER_IN_SANDBOX"
GPU_MONITOR_SCHEMA = "ect.q256.in-run-gpu-exclusivity-monitor/v2"
GPU_MONITOR_CADENCE_GRACE_SECONDS = 0.25
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

# Ordered deliberately: every seed sees the same arm order.
ARMS = OrderedDict(
    (
        ("A", {"target_gap_scale": "1.0", "denominator_gap_scale": "1.0"}),
        ("B", {"target_gap_scale": "1.1", "denominator_gap_scale": "1.1"}),
        ("C", {"target_gap_scale": "1.1", "denominator_gap_scale": "1.0"}),
        ("D", {"target_gap_scale": "1.0", "denominator_gap_scale": "1.1"}),
    )
)

PHASES = {
    # ct_train converts Mimg to int(kimg).  0.004096 therefore commits after
    # 32 x 128 = 4,096 images while retaining the requested 4.096-kimg label.
    "smoke": {
        "duration_mimg": "0.004096",
        "requested_kimg": "4.096",
        "ct_train_total_kimg": 4,
        "expected_processed_nimg": 4096,
        "expected_attempts": 32,
        "seeds": (3,),
    },
    "formal": {
        "duration_mimg": "0.256",
        "requested_kimg": "256",
        "ct_train_total_kimg": 256,
        "expected_processed_nimg": 256000,
        "expected_attempts": 2000,
        "seeds": (3, 4, 5),
    },
}

_SOURCE_SUFFIXES = {".py", ".cu", ".cpp", ".c", ".cc", ".h", ".hpp"}
_SOURCE_PREFIXES = ("dnnlib/", "metrics/", "torch_utils/", "training/")
ROLE_E_AB_PARITY_SOURCE_FILES = (
    "ct_train.py",
    "training/loss.py",
    "training/schedules.py",
    "training/ct_training_loop.py",
    "training/reproducibility.py",
    "torch_utils/misc.py",
    "scripts/run_q256_target_weight_correctness_gate.py",
    "scripts/run_q256_target_weight_evaluation.py",
    "scripts/run_q256_target_weight_matrix.py",
    "scripts/verify_q256_target_weight_arm.py",
    "scripts/verify_q256_target_weight_smoke_matrix.py",
    "tests/test_q256_target_weight_factorial.py",
    "tests/test_schedules.py",
    "tests/test_exact_resume_state.py",
    "tests/test_training_cli_compat.py",
    "tests/test_q256_target_weight_launcher.py",
    "tests/test_q256_target_weight_verifier.py",
    "tests/test_q256_target_weight_smoke_matrix_verifier.py",
    "tests/test_q256_target_weight_evaluation.py",
    "tests/test_q256_target_weight_correctness_gate.py",
    "analysis/q256_target_weight_factorial/preregistration_amendment_002.json",
)
ROLE_E_REQUIRED_TEST_CASES = (
    (
        "tests.test_q256_target_weight_factorial.CanonicalParityTest",
        "test_A_is_bitwise_equal_to_native_sigmoid",
    ),
    (
        "tests.test_q256_target_weight_factorial.CanonicalParityTest",
        "test_B_is_bitwise_equal_to_native_global_sigmoid_g110",
    ),
    (
        "tests.test_q256_target_weight_factorial.CanonicalParityTest",
        "test_cuda_amp_A_and_B_match_native_full_forward_gradients_and_rng",
    ),
)
_SOURCE_EXACT = {
    "ct_train.py",
    "scripts/run_q256_target_weight_arm.sh",
    "scripts/run_q256_target_weight_correctness_gate.py",
    "scripts/run_q256_target_weight_matrix.py",
    "scripts/verify_q256_target_weight_arm.py",
    "scripts/verify_q256_target_weight_smoke_matrix.py",
    "tests/test_q256_target_weight_correctness_gate.py",
    *ROLE_E_AB_PARITY_SOURCE_FILES,
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_STATE_RE = re.compile(r"^training-state-(?:latest|\d+)\.pt$")
_GPU_RE = re.compile(r"^(?:\d+|GPU-[A-Za-z0-9-]+)$")

# Exact public v1 writer/verifier contract.  Keep synchronized with
# training.ct_training_loop and verify_q256_target_weight_arm.py.
FACTORIAL_TELEMETRY_FIELDS = (
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
FACTORIAL_DIGEST_FIELDS = (
    "batch_sha256",
    "t_sha256",
    "base_r_sha256",
    "target_r_sha256",
    "denominator_r_sha256",
    "target_delta_sha256",
    "denominator_delta_sha256",
)

TRAIN_SUMMARY_FIELDS = (
    "attempted_iteration",
    "successful_optimizer_steps",
    "processed_nimg",
    "processed_kimg",
    "loss",
    "grad_scale",
    "step_skipped",
    "schedule",
    "stage",
    "next_loop_cur_tick",
    "loss_ema",
    "loss_reference",
    "correction",
    "signal_updates",
    "adaptive_active",
    "r_over_t_mean",
    "gap_mean",
    "gap_over_sigmoid_gap_mean",
    "lower_gap_clip_rate",
    "upper_gap_clip_rate",
    "elapsed_sec",
    "peak_vram_gb",
)

PLANNED_PAUSE_ARTIFACTS = (
    "launch_manifest.json",
    "training_options.json",
    "initial_state_receipt_v1.json",
    "train_summary.csv",
    "factorial_training_telemetry_v1.csv",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "log.txt",
)

AUTHORIZATION_GATE_CONTRACTS = {
    "smoke": OrderedDict(
        (
            (
                "role_e_ab_parity",
                {
                    "schema": ROLE_E_AB_PARITY_SCHEMA,
                    "status": "PASS",
                    "require_exact_resume": False,
                },
            ),
        )
    ),
    "formal": OrderedDict(
        (
            (
                "role_e_ab_parity",
                {
                    "schema": ROLE_E_AB_PARITY_SCHEMA,
                    "status": "PASS",
                    "require_exact_resume": False,
                },
            ),
            (
                "four_arm_smoke_matrix",
                {
                    "schema": SMOKE_MATRIX_VALIDATION_SCHEMA,
                    "status": "passed",
                    "require_exact_resume": False,
                },
            ),
            (
                "exact_resume",
                {
                    "schema": SMOKE_MATRIX_VALIDATION_SCHEMA,
                    "status": "passed",
                    "require_exact_resume": True,
                },
            ),
        )
    ),
}


class LaunchError(RuntimeError):
    """A fail-closed preflight or execution error."""


class ProcessCleanupError(LaunchError):
    """The launcher cannot prove that its child process tree is fully gone."""


PROCESS_CLEANUP_UNCONFIRMED_EXIT_CODE = 86
MATRIX_CHILD_REGISTRY_ENV = "ECT_Q256_MATRIX_CHILD_REGISTRY"
MATRIX_OUTER_PID_ENV = "ECT_Q256_MATRIX_OUTER_PID"
MATRIX_CHILD_TOKEN_SCHEMA = "ect.q256.matrix-child-ownership/v1"
MATRIX_OUTER_WRAPPER_CODE = r'''
import os
import sys

command = sys.argv[1:]
if not command:
    raise SystemExit("matrix outer wrapper received no command")
os.environ["ECT_Q256_MATRIX_OUTER_PID"] = str(os.getpid())
os.execvpe(command[0], command, os.environ)
'''
MATRIX_CHILD_WRAPPER_CODE = r'''
import ctypes
import hashlib
import json
import os
import signal
import sys

registry_dir, label, expected_parent_raw, *command = sys.argv[1:]
expected_parent = int(expected_parent_raw)
matrix_outer_pid = int(os.environ["ECT_Q256_MATRIX_OUTER_PID"])
if not command:
    raise SystemExit("matrix child wrapper received no command")
if sys.platform != "linux":
    raise SystemExit("matrix child ownership wrapper requires Linux")
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
if os.getppid() != expected_parent:
    os.kill(os.getpid(), signal.SIGKILL)
raw = open("/proc/self/stat", "rb").read()
close = raw.rfind(b")")
fields = raw[close + 2:].split()
payload = {
    "schema": "ect.q256.matrix-child-ownership/v1",
    "label": label,
    "matrix_outer_pid": matrix_outer_pid,
    "launcher_parent_pid": expected_parent,
    "pid": os.getpid(),
    "pgid": os.getpgrp(),
    "sid": os.getsid(0),
    "starttime": int(fields[19]),
    "command_sha256": hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
}
if payload["pid"] != payload["pgid"] or payload["pid"] != payload["sid"]:
    raise SystemExit("matrix child wrapper is not its own session leader")
encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
name = f"{label}-{os.getpid()}.json"
temporary = os.path.join(registry_dir, f".{name}.tmp")
final = os.path.join(registry_dir, name)
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(fd, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, final)
os.unlink(temporary)
directory_fd = os.open(registry_dir, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
os.execvpe(command[0], command, os.environ)
'''


def fail(message: str) -> None:
    raise LaunchError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checked_output(
    args: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> str:
    try:
        return subprocess.check_output(
            list(args), cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        detail = str(output).strip() or str(exc)
        fail(f"command failed ({shlex.join(list(args))}): {detail}")


def _selected_source_path(path: str) -> bool:
    if path in _SOURCE_EXACT:
        return True
    return path.endswith(tuple(_SOURCE_SUFFIXES)) and path.startswith(_SOURCE_PREFIXES)


def source_snapshot(repo_root: Path = REPO_ROOT, *, require_clean: bool = True) -> dict:
    """Return the clean committed HEAD and a deterministic training-source manifest."""

    repo_root = repo_root.resolve()
    if checked_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root) != "true":
        fail(f"not a Git worktree: {repo_root}")
    # Git 1.8.3.1 (the production host) supports the original --porcelain
    # spelling, but not the later explicit ``--porcelain=v1`` alias.
    status = checked_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root
    )
    if require_clean and status:
        preview = "; ".join(status.splitlines()[:12])
        fail(f"formal source must be a clean committed HEAD; status={preview}")
    # ``git branch --show-current`` was added long after the server's Git
    # 1.8.3.1.  symbolic-ref is available there and also fails closed for a
    # detached HEAD instead of silently returning an empty branch name.
    branch = checked_output(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo_root
    )
    if require_clean and branch != EXPECTED_BRANCH:
        fail(f"wrong execution branch: {branch!r} != {EXPECTED_BRANCH!r}")
    head = checked_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        fail(f"invalid Git HEAD: {head!r}")
    tree = checked_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root)
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        fail(f"invalid Git tree: {tree!r}")
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=repo_root, stderr=subprocess.STDOUT
    )
    tracked = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    selected = sorted(path for path in tracked if _selected_source_path(path))
    missing = sorted(path for path in _SOURCE_EXACT if path not in selected)
    if missing:
        fail(f"source manifest is missing required tracked files: {missing}")
    entries = []
    for relative in selected:
        path = repo_root / relative
        if not path.is_file():
            fail(f"tracked training source is not a regular file: {path}")
        entries.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return {
        "git_head": head,
        "git_tree": tree,
        "git_branch": branch,
        "git_clean": not bool(status),
        "manifest_algorithm": "canonical-json-sha256-v1",
        "content_sha256": canonical_sha256(entries),
        "file_count": len(entries),
        "files": entries,
    }


def verify_asset(path: Path, expected_sha256: str, label: str) -> dict:
    expanded = path.expanduser()
    if not expanded.is_file():
        fail(f"{label} not found: {expanded}")
    resolved = expanded.resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        fail(f"{label} SHA256 mismatch: {actual} != {expected_sha256} ({resolved})")
    return {
        "path": str(expanded.absolute()),
        "resolved_path": str(resolved),
        "sha256": actual,
        "size_bytes": resolved.stat().st_size,
    }


def preregistration_record() -> dict:
    if not PREREGISTRATION.is_file():
        fail(f"preregistration is missing: {PREREGISTRATION}")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO_ROOT)),
        "sha256": sha256_file(PREREGISTRATION),
    }


def _load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{label} must be a JSON object: {path}")
    return payload


def expected_authorization_seeds(phase: str) -> list[int]:
    return list(PHASES[phase]["seeds"])


def parse_expected_skip_attempts(raw: str | None, phase: str) -> list[int] | None:
    if raw is None:
        return None
    try:
        if raw.lstrip().startswith("["):
            values = json.loads(raw)
        else:
            values = [] if not raw.strip() else [int(item.strip()) for item in raw.split(",")]
    except (ValueError, json.JSONDecodeError) as exc:
        fail(f"invalid --expected-skip-attempts: {exc}")
    return validate_amp_skip_signature(values, phase=phase, label="expected")


def validate_amp_skip_signature(
    values: object, *, phase: str, label: str
) -> list[int]:
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        fail(f"{label} AMP skip attempts must be a list of integers")
    if values != sorted(set(values)):
        fail(f"{label} AMP skip attempts must be strictly increasing and unique")
    maximum = min(
        int(PHASES[phase]["expected_attempts"]),
        (AMP_SKIP_WARMUP_PROCESSED_NIMG - 1) // 128,
    )
    if any(value < 1 or value > maximum for value in values):
        fail(
            f"{label} AMP skip attempts must lie within the tick-0 warm-up region "
            f"1..{maximum}"
        )
    return list(values)


def authorization_gate_contract(phase: str) -> Mapping[str, Mapping[str, object]]:
    contract = AUTHORIZATION_GATE_CONTRACTS.get(phase)
    if contract is None:
        fail(f"no authorization gate contract for phase={phase!r}")
    return contract


def _source_file_sha256_map(source: Mapping[str, object], label: str) -> dict[str, str]:
    entries = source.get("files")
    if not isinstance(entries, list):
        fail(f"{label} source manifest has no file list")
    result: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"{label} source manifest entry {index} is not an object")
        relative = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in result
            or not isinstance(digest, str)
            or not _SHA_RE.fullmatch(digest)
        ):
            fail(f"{label} source manifest entry {index} is invalid or duplicated")
        result[relative] = digest
    return result


def role_e_gate_runtime_scope(runtime: Mapping[str, object]) -> dict:
    """Translate the launcher probe into the independently emitted Role-E fields."""

    field_map = OrderedDict(
        (
            ("python", "python_version"),
            ("platform", "platform"),
            ("torch", "torch_version"),
            ("torch_cuda", "torch_cuda_version"),
            ("cudnn", "torch_cudnn_version"),
            ("cuda_device_count", "cuda_device_count"),
            ("gpu_uuid", "visible_gpu_uuid"),
            ("gpu_name", "visible_gpu_name_nvidia_smi"),
            ("gpu_memory_mib", "visible_gpu_memory_mib_nvidia_smi"),
            ("compute_capability", "visible_device_compute_capability"),
            ("launcher_software_sha256", "software_sha256"),
            ("critical_runtime_files", "critical_runtime_files"),
        )
    )
    missing = [runtime_key for runtime_key in field_map.values() if runtime_key not in runtime]
    if missing:
        fail(f"runtime probe lacks Role-E authorization fields: {missing}")
    scope = {
        gate_key: runtime[runtime_key]
        for gate_key, runtime_key in field_map.items()
    }
    if (
        not isinstance(scope["gpu_uuid"], str)
        or not str(scope["gpu_uuid"]).startswith("GPU-")
        or not isinstance(scope["gpu_name"], str)
        or not scope["gpu_name"]
        or isinstance(scope["gpu_memory_mib"], bool)
        or not isinstance(scope["gpu_memory_mib"], int)
        or scope["gpu_memory_mib"] <= 0
        or scope["cuda_device_count"] != 1
        or not isinstance(scope["launcher_software_sha256"], str)
        or not _SHA_RE.fullmatch(scope["launcher_software_sha256"])
    ):
        fail("runtime probe has an invalid Role-E GPU/software identity")
    capability = scope["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
    ):
        fail("runtime probe has an invalid CUDA compute capability")
    return scope


def _parse_role_e_junit(path: Path, label: str) -> dict:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        fail(f"{label} is not valid JUnit XML: {exc}")
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [
        (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        for case in cases
    ]
    invalid_counts = {
        f"{classname}::{name}": identities.count((classname, name))
        for classname, name in ROLE_E_REQUIRED_TEST_CASES
        if identities.count((classname, name)) != 1
    }
    if invalid_counts:
        fail(f"{label} required test identities are not unique: {invalid_counts}")
    required_failures = []
    for case in cases:
        identity = (
            case.attrib.get("classname", ""), case.attrib.get("name", "")
        )
        if identity in ROLE_E_REQUIRED_TEST_CASES and (
            case.find("failure") is not None
            or case.find("error") is not None
            or case.find("skipped") is not None
        ):
            required_failures.append(f"{identity[0]}::{identity[1]}")
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "required_test_cases": [
            {"classname": classname, "name": name}
            for classname, name in ROLE_E_REQUIRED_TEST_CASES
        ],
        "required_test_failures": required_failures,
    }


def _resolve_role_e_evidence(
    payload: Mapping[str, object],
    label: str,
    override: Mapping[str, Path] | None = None,
) -> dict[str, dict]:
    evidence = payload.get("evidence")
    expected_keys = {
        "pytest_log",
        "pytest_log_sha256",
        "pytest_junit",
        "pytest_junit_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        fail(f"{label} has a malformed evidence binding")
    records: dict[str, dict] = {}
    for artifact in ("pytest_log", "pytest_junit"):
        declared_path = evidence.get(artifact)
        declared_sha = evidence.get(f"{artifact}_sha256")
        if not isinstance(declared_sha, str) or not _SHA_RE.fullmatch(declared_sha):
            fail(f"{label} has an invalid {artifact} SHA256")
        if override is not None:
            raw_path = override.get(artifact)
            if raw_path is None:
                fail(f"{label} lacks copied {artifact} evidence")
            path = Path(raw_path)
        else:
            if not isinstance(declared_path, str) or not declared_path:
                fail(f"{label} has an invalid {artifact} path")
            path = Path(declared_path).expanduser()
        if path.is_symlink():
            fail(f"{label} {artifact} must not be a symlink")
        try:
            path = path.resolve(strict=True)
        except OSError as exc:
            fail(f"{label} {artifact} cannot be resolved: {exc}")
        if not path.is_file() or path.stat().st_size <= 0:
            fail(f"{label} {artifact} must be a non-empty regular file")
        actual_sha = sha256_file(path)
        if actual_sha != declared_sha:
            fail(f"{label} {artifact} hash mismatch")
        records[artifact] = {
            "source_path": str(path),
            "sha256": actual_sha,
        }
    parsed = _parse_role_e_junit(
        Path(records["pytest_junit"]["source_path"]),
        f"{label} pytest_junit",
    )
    if payload.get("pytest_exit_code") != 0 or payload.get("junit") != parsed:
        fail(f"{label} pytest exit/JUnit summary does not match immutable evidence")
    if (
        parsed["failures"] != 0
        or parsed["errors"] != 0
        or parsed["skipped"] != 0
        or parsed["required_test_failures"]
    ):
        fail(f"{label} JUnit evidence is not an all-pass, zero-skip gate")
    expected_contract = {
        "required_test_cases": parsed["required_test_cases"],
        "all_collected_tests_passed_without_skip": True,
    }
    if payload.get("assertion_contract") != expected_contract:
        fail(f"{label} assertion contract differs from the parsed JUnit evidence")
    archive_sha = payload.get("executed_git_archive_sha256")
    if not isinstance(archive_sha, str) or not _SHA_RE.fullmatch(archive_sha):
        fail(f"{label} lacks a valid executed Git-archive hash")
    return records


def validate_gate_receipt_payload(
    *,
    phase: str,
    name: str,
    payload: Mapping[str, object],
    label: str,
    source: Mapping[str, object],
    role_e_runtime: Mapping[str, object],
    expected_skip_attempts: list[int] | None,
    revalidation_command: Sequence[str] | None = None,
    revalidation_env: Mapping[str, str] | None = None,
    scope_cache: dict[str, dict] | None = None,
    role_e_evidence_override: Mapping[str, Path] | None = None,
) -> dict:
    contract = authorization_gate_contract(phase)
    spec = contract.get(name)
    if spec is None:
        fail(f"{label} has an unauthorized logical gate name: {name!r}")
    expected_schema = spec["schema"]
    expected_status = spec["status"]
    if payload.get("schema") != expected_schema:
        fail(
            f"{label} schema mismatch: "
            f"{payload.get('schema')!r} != {expected_schema!r}"
        )
    if payload.get("status") != expected_status:
        fail(
            f"{label} status mismatch: "
            f"{payload.get('status')!r} != {expected_status!r}"
        )

    checks = {"schema": expected_schema, "status": expected_status}
    if expected_schema == ROLE_E_AB_PARITY_SCHEMA:
        gate_source = payload.get("source")
        if not isinstance(gate_source, dict):
            fail(f"{label} has no structured source identity")
        if gate_source.get("commit") != source.get("git_head"):
            fail(f"{label} was produced from a stale Git commit")
        if gate_source.get("tree") != source.get("git_tree"):
            fail(f"{label} was produced from a stale Git tree")
        if gate_source.get("launcher_content_sha256") != source.get("content_sha256"):
            fail(f"{label} launcher source-content binding is stale")
        if gate_source.get("branch") != source.get("git_branch"):
            fail(f"{label} source branch differs from the authorized source")
        if gate_source.get("clean") is not True:
            fail(f"{label} was not produced from a clean source tree")
        gate_files = gate_source.get("files")
        if not isinstance(gate_files, dict) or set(gate_files) != set(
            ROLE_E_AB_PARITY_SOURCE_FILES
        ):
            fail(f"{label} does not bind the exact Role-E critical-file set")
        current_files = _source_file_sha256_map(source, label)
        for relative in ROLE_E_AB_PARITY_SOURCE_FILES:
            digest = gate_files.get(relative)
            if (
                not isinstance(digest, str)
                or not _SHA_RE.fullmatch(digest)
                or current_files.get(relative) != digest
            ):
                fail(f"{label} critical source hash is stale for {relative!r}")
        manifest = "".join(
            f"{relative}\t{gate_files[relative]}\n"
            for relative in sorted(gate_files)
        )
        expected_manifest_sha = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
        if gate_source.get("manifest_sha256") != expected_manifest_sha:
            fail(f"{label} Role-E critical-file manifest hash is invalid")
        gate_runtime = payload.get("runtime")
        if not isinstance(gate_runtime, dict):
            fail(f"{label} has no structured CUDA runtime identity")
        if gate_runtime.get("cuda_visible_devices") != gate_runtime.get("gpu_uuid"):
            fail(f"{label} did not bind CUDA_VISIBLE_DEVICES to its full GPU UUID")
        if set(role_e_runtime) != {
            "python",
            "platform",
            "torch",
            "torch_cuda",
            "cudnn",
            "cuda_device_count",
            "gpu_uuid",
            "gpu_name",
            "gpu_memory_mib",
            "compute_capability",
            "launcher_software_sha256",
            "critical_runtime_files",
        }:
            fail("authorization has a malformed Role-E runtime scope")
        for field, expected in role_e_runtime.items():
            if gate_runtime.get(field) != expected:
                fail(f"{label} runtime field {field!r} differs from the launch scope")
        evidence_records = _resolve_role_e_evidence(
            payload,
            label,
            override=role_e_evidence_override,
        )
        checks.update(
            source_git_head=source["git_head"],
            source_content_sha256=source["content_sha256"],
            runtime_software_sha256=role_e_runtime["launcher_software_sha256"],
            gpu_uuid=role_e_runtime["gpu_uuid"],
            critical_file_count=len(gate_files),
            evidence=evidence_records,
        )
    if expected_schema == SMOKE_MATRIX_VALIDATION_SCHEMA:
        if payload.get("mode") != "smoke" or payload.get("seed") != 3:
            fail(f"{label} is not the frozen seed-3 smoke matrix")
        if payload.get("source_git_head") != source.get("git_head"):
            fail(f"{label} was produced from a stale Git commit")
        if payload.get("source_content_sha256") != source.get("content_sha256"):
            fail(f"{label} was produced from stale source content")
        observed_skip_attempts_by_arm = payload.get("amp_skip_attempts_by_arm")
        if (
            not isinstance(observed_skip_attempts_by_arm, dict)
            or set(observed_skip_attempts_by_arm) != set(ARMS)
        ):
            fail(f"{label} has malformed per-arm AMP skip signatures")
        for arm, signature in observed_skip_attempts_by_arm.items():
            if (
                not isinstance(signature, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in signature
                )
                or signature != sorted(set(signature))
            ):
                fail(f"{label} arm {arm} has a malformed AMP skip signature")
            if expected_skip_attempts is not None and signature != expected_skip_attempts:
                fail(
                    f"{label} arm {arm} AMP skip signature differs from the "
                    "authorized value"
                )
        observed_skip_count = payload.get("amp_skip_count")
        if (
            isinstance(observed_skip_count, bool)
            or not isinstance(observed_skip_count, int)
            or observed_skip_count < 0
            or any(
                len(signature) != observed_skip_count
                for signature in observed_skip_attempts_by_arm.values()
            )
        ):
            fail(f"{label} has a malformed cross-arm AMP skip count")
        observed_successful_steps = payload.get("successful_optimizer_steps")
        if (
            isinstance(observed_successful_steps, bool)
            or not isinstance(observed_successful_steps, int)
            or observed_successful_steps < 0
        ):
            fail(f"{label} has a malformed successful optimizer-step count")
        if payload.get("amp_skip_policy") != AMP_SKIP_POLICY:
            fail(f"{label} AMP skip policy differs from the frozen policy")
        if payload.get("amp_skip_signature_expected_value_enforced") is not (
            expected_skip_attempts is not None
        ):
            fail(f"{label} AMP skip enforcement mode differs from authorization")
        arms = payload.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            fail(f"{label} does not bind exactly the A/B/C/D smoke arms")
        if any(not isinstance(arms[arm], dict) for arm in ARMS):
            fail(f"{label} has a malformed smoke-arm binding")
        cache_key = canonical_sha256(
            {
                "payload": payload,
                "source_git_head": source.get("git_head"),
                "source_content_sha256": source.get("content_sha256"),
                "expected_skip_attempts": expected_skip_attempts,
            }
        )
        cached = scope_cache.get(cache_key) if scope_cache is not None else None
        if cached is None:
            arm_bindings = {}
            runner_completions = {}
            resolved_run_dirs: set[Path] = set()
            for arm in ARMS:
                binding = arms[arm]
                if set(binding) != {
                    "run_dir",
                    "validation_receipt_sha256",
                    "artifact_hash_receipt_sha256",
                    "runner_completion_path",
                    "runner_completion_sha256",
                }:
                    fail(f"{label} arm {arm} has a malformed receipt binding")
                raw_run_dir = binding.get("run_dir")
                if not isinstance(raw_run_dir, str) or not raw_run_dir:
                    fail(f"{label} arm {arm} has no run directory")
                unresolved = Path(raw_run_dir).expanduser()
                if unresolved.is_symlink():
                    fail(f"{label} arm {arm} run directory must not be a symlink")
                try:
                    run_dir = unresolved.resolve(strict=True)
                except OSError as exc:
                    fail(f"{label} arm {arm} run directory cannot be resolved: {exc}")
                if not run_dir.is_dir() or run_dir in resolved_run_dirs:
                    fail(f"{label} arm {arm} run directory is invalid or reused")
                if raw_run_dir != str(run_dir):
                    fail(f"{label} arm {arm} run directory is not canonical")
                resolved_run_dirs.add(run_dir)
                verified = validate_existing_verifier_receipts(
                    run_dir,
                    phase="smoke",
                    arm=arm,
                    seed=3,
                    expected_skip_attempts=expected_skip_attempts,
                )
                if verified is None:
                    fail(f"{label} arm {arm} lacks immutable PASS receipts")
                deep_validation = deep_revalidate_existing_arm(
                    run_dir,
                    phase="smoke",
                    arm=arm,
                    seed=3,
                    expected_skip_attempts=expected_skip_attempts,
                    runtime_command=revalidation_command,
                    process_env=revalidation_env,
                )
                if (
                    verified.get("amp_skip_attempts")
                    != observed_skip_attempts_by_arm[arm]
                ):
                    fail(
                        f"{label} arm {arm} AMP skip signature differs from "
                        "its matrix binding"
                    )
                if (
                    verified.get("successful_optimizer_steps")
                    != observed_successful_steps
                ):
                    fail(
                        f"{label} arm {arm} successful optimizer-step count "
                        "differs from the matrix value"
                    )
                runner_completions[arm] = validate_existing_runner_completion(
                    run_dir
                )
                for field in ("path", "sha256"):
                    if binding.get(f"runner_completion_{field}") != (
                        runner_completions[arm].get(field)
                    ):
                        fail(
                            f"{label} arm {arm} has a stale runner completion "
                            f"{field} binding"
                        )
                for field in (
                    "validation_receipt_sha256",
                    "artifact_hash_receipt_sha256",
                ):
                    if binding.get(field) != verified.get(field):
                        fail(f"{label} arm {arm} has a stale {field} binding")
                arm_validation = _load_json(
                    run_dir / VALIDATION_FILENAME,
                    f"{label} arm {arm} validation receipt",
                )
                if arm_validation.get("run_dir") != str(run_dir):
                    fail(f"{label} arm {arm} validation receipt targets another run")
                if (
                    arm_validation.get("source_git_head") != source.get("git_head")
                    or arm_validation.get("source_content_sha256")
                    != source.get("content_sha256")
                ):
                    fail(f"{label} arm {arm} immutable PASS is source-stale")
                arm_bindings[arm] = dict(binding)
                arm_bindings[arm]["fresh_revalidation_report_sha256"] = (
                    deep_validation["report_sha256"]
                )
            matrix_revalidation = deep_revalidate_smoke_matrix(
                payload,
                runtime_command=revalidation_command,
                process_env=revalidation_env,
            )
            cached = {
                "arm_bindings": arm_bindings,
                "runner_completions": runner_completions,
                "matrix_revalidation": matrix_revalidation,
            }
            if scope_cache is not None:
                scope_cache[cache_key] = cached
        checks.update(
            mode="smoke",
            seed=3,
            arms=list(ARMS),
            source_git_head=source["git_head"],
            source_content_sha256=source["content_sha256"],
            amp_skip_attempts_by_arm={
                arm: list(observed_skip_attempts_by_arm[arm]) for arm in ARMS
            },
            amp_skip_count=observed_skip_count,
            successful_optimizer_steps=observed_successful_steps,
            amp_skip_policy=dict(AMP_SKIP_POLICY),
            arm_bindings=cached["arm_bindings"],
            runner_completions=cached["runner_completions"],
            matrix_revalidation=cached["matrix_revalidation"],
        )
    if spec["require_exact_resume"]:
        exact_resume = payload.get("exact_resume")
        if not isinstance(exact_resume, dict) or exact_resume.get("status") != "passed":
            fail(f"{label} lacks a passing top-level exact_resume object")
        resume_arm = exact_resume.get("arm")
        if resume_arm not in ARMS:
            fail(f"{label} exact_resume has an invalid arm")
        exact_resume_bindings = {}
        resolved_resume_dirs = []
        for prefix in ("uninterrupted", "resumed"):
            raw_run_dir = exact_resume.get(f"{prefix}_run_dir")
            if not isinstance(raw_run_dir, str) or not raw_run_dir:
                fail(f"{label} exact_resume lacks {prefix}_run_dir")
            unresolved = Path(raw_run_dir).expanduser()
            if unresolved.is_symlink():
                fail(f"{label} exact_resume {prefix} directory is a symlink")
            try:
                run_dir = unresolved.resolve(strict=True)
            except OSError as exc:
                fail(
                    f"{label} exact_resume {prefix} directory cannot be "
                    f"resolved: {exc}"
                )
            if not run_dir.is_dir() or raw_run_dir != str(run_dir):
                fail(f"{label} exact_resume {prefix} directory is not canonical")
            resolved_resume_dirs.append(run_dir)
            verified = validate_existing_verifier_receipts(
                run_dir,
                phase="smoke",
                arm=str(resume_arm),
                seed=3,
                expected_skip_attempts=expected_skip_attempts,
            )
            if verified is None:
                fail(f"{label} exact_resume {prefix} lacks arm PASS receipts")
            deep_validation = deep_revalidate_existing_arm(
                run_dir,
                phase="smoke",
                arm=str(resume_arm),
                seed=3,
                expected_skip_attempts=expected_skip_attempts,
                runtime_command=revalidation_command,
                process_env=revalidation_env,
            )
            for field in (
                "validation_receipt_sha256",
                "artifact_hash_receipt_sha256",
            ):
                if exact_resume.get(f"{prefix}_{field}") != verified.get(field):
                    fail(
                        f"{label} exact_resume {prefix} has a stale {field}"
                    )
            arm_validation = _load_json(
                run_dir / VALIDATION_FILENAME,
                f"{label} exact_resume {prefix} arm validation",
            )
            if (
                arm_validation.get("source_git_head") != source.get("git_head")
                or arm_validation.get("source_content_sha256")
                != source.get("content_sha256")
            ):
                fail(f"{label} exact_resume {prefix} arm PASS is source-stale")
            runner = validate_existing_runner_completion(run_dir)
            if exact_resume.get(f"{prefix}_runner_completion_sha256") != runner.get(
                "sha256"
            ):
                fail(
                    f"{label} exact_resume {prefix} has a stale runner completion"
                )
            exact_resume_bindings[prefix] = {
                "run_dir": str(run_dir),
                "verifier": verified,
                "fresh_revalidation": deep_validation,
                "runner_completion": runner,
            }
        if resolved_resume_dirs[0] == resolved_resume_dirs[1]:
            fail(f"{label} exact_resume reuses one directory for both branches")
        provenance = validate_exact_resume_provenance(
            resolved_resume_dirs[0],
            resolved_resume_dirs[1],
            arm=str(resume_arm),
            seed=3,
            runtime_command=revalidation_command,
            process_env=revalidation_env,
        )
        if (
            exact_resume.get("provenance") != provenance
            or exact_resume.get("provenance_sha256")
            != provenance.get("provenance_sha256")
        ):
            fail(f"{label} exact_resume provenance binding is stale")
        checks["exact_resume_status"] = "passed"
        checks["exact_resume_bindings"] = exact_resume_bindings
        checks["exact_resume_provenance"] = provenance
    return checks


def validate_authorization(
    receipt_path: Path,
    *,
    phase: str,
    arm: str,
    seed: int,
    source: Mapping[str, object],
    dataset: Mapping[str, object],
    transfer: Mapping[str, object],
    runtime_sandbox: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
    expected_skip_attempts: list[int] | None = None,
    revalidation_command: Sequence[str] | None = None,
    revalidation_env: Mapping[str, str] | None = None,
) -> dict:
    """Validate an explicit PASS receipt and every gate artifact bound by it."""

    path = receipt_path.expanduser().resolve(strict=True)
    payload = _load_json(path, "authorization receipt")
    expected_skip_attempts = (
        None if expected_skip_attempts is None else list(expected_skip_attempts)
    )
    if runtime_sandbox is None or runtime is None:
        fail("authorization validation requires the frozen sandbox and CUDA runtime")
    if "expected_amp_skip_attempts" not in payload:
        fail(
            "authorization must explicitly record expected_amp_skip_attempts, "
            "including null in observe mode"
        )
    current_role_e_runtime = role_e_gate_runtime_scope(runtime)
    exact = {
        "schema": AUTHORIZATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "authorized",
        "gates_status": "PASS",
        "allowed_arms": list(ARMS),
        "allowed_seeds": expected_authorization_seeds(phase),
        "source_git_head": source["git_head"],
        "source_content_sha256": source["content_sha256"],
        "preregistration_sha256": preregistration_record()["sha256"],
        "dataset_sha256": dataset["sha256"],
        "transfer_sha256": transfer["sha256"],
        "expected_amp_skip_attempts": expected_skip_attempts,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "role_e_gate_runtime": current_role_e_runtime,
        "runtime_sandbox_tree_metadata_sha256": runtime_sandbox[
            "sandbox_tree_metadata_sha256"
        ],
        "runtime_sandbox_critical_files_sha256": runtime_sandbox[
            "critical_files_sha256"
        ],
        "runtime_software_sha256": runtime["software_sha256"],
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            fail(
                f"authorization field {key!r} does not match frozen scope: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    if arm not in payload["allowed_arms"] or seed not in payload["allowed_seeds"]:
        fail(f"authorization does not cover arm={arm}, seed={seed}")
    if not isinstance(payload.get("authorization_id"), str) or not payload["authorization_id"].strip():
        fail("authorization receipt requires a non-empty authorization_id")
    if not isinstance(payload.get("authorized_by"), str) or not payload["authorized_by"].strip():
        fail("authorization receipt requires a non-empty authorized_by")
    if not isinstance(payload.get("issued_at_utc"), str) or not _UTC_RE.fullmatch(payload["issued_at_utc"]):
        fail("authorization issued_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    gate_receipts = payload.get("gate_receipts")
    contract = authorization_gate_contract(phase)
    if not isinstance(gate_receipts, list):
        fail("authorization gate_receipts must be a list")
    if len(gate_receipts) != len(contract):
        fail(
            f"phase={phase} authorization requires exactly logical gates "
            f"{list(contract)}, got {len(gate_receipts)} entries"
        )
    validated_gates = []
    seen_names = set()
    gate_scope_cache: dict[str, dict] = {}
    for index, item in enumerate(gate_receipts):
        if not isinstance(item, dict):
            fail(f"gate_receipts[{index}] must be an object")
        if set(item) != {"name", "schema", "status", "path", "sha256"}:
            fail(
                f"gate_receipts[{index}] must contain exactly "
                "name/schema/status/path/sha256"
            )
        name = item.get("name")
        declared_sha = item.get("sha256")
        raw_path = item.get("path")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            fail(f"invalid or duplicate gate receipt name at index {index}")
        spec = contract.get(name)
        if spec is None:
            fail(f"phase={phase} authorization contains an extra gate: {name!r}")
        if item.get("schema") != spec["schema"] or item.get("status") != spec["status"]:
            fail(f"authorization declaration for gate {name!r} mismatches its contract")
        if not isinstance(declared_sha, str) or not _SHA_RE.fullmatch(declared_sha):
            fail(f"invalid gate receipt SHA256 for {name!r}")
        if not isinstance(raw_path, str) or not raw_path:
            fail(f"gate receipt {name!r} is missing path")
        gate_path = Path(raw_path).expanduser()
        if not gate_path.is_absolute():
            gate_path = path.parent / gate_path
        if gate_path.is_symlink():
            fail(f"gate receipt {name!r} must not be a symlink: {gate_path}")
        gate_path = gate_path.resolve(strict=True)
        if not gate_path.is_file() or gate_path.stat().st_size <= 0:
            fail(f"gate receipt {name!r} must be a non-empty regular JSON file")
        actual_sha = sha256_file(gate_path)
        if actual_sha != declared_sha:
            fail(f"gate receipt hash mismatch for {name!r}: {actual_sha} != {declared_sha}")
        gate_payload = _load_json(gate_path, f"gate receipt {name!r}")
        logical_checks = validate_gate_receipt_payload(
            phase=phase,
            name=name,
            payload=gate_payload,
            label=f"gate receipt {name!r}",
            source=source,
            role_e_runtime=current_role_e_runtime,
            expected_skip_attempts=expected_skip_attempts,
            revalidation_command=revalidation_command,
            revalidation_env=revalidation_env,
            scope_cache=gate_scope_cache,
        )
        seen_names.add(name)
        validated_gates.append(
            {
                "name": name,
                "schema": spec["schema"],
                "status": spec["status"],
                "source_path": str(gate_path),
                "sha256": actual_sha,
                "logical_checks": logical_checks,
            }
        )
    if seen_names != set(contract):
        fail(
            f"phase={phase} authorization gate set mismatch: "
            f"expected={list(contract)}, actual={sorted(seen_names)}"
        )
    if phase == "formal":
        matrix_gates = {
            item["name"]: item
            for item in validated_gates
            if item["name"] in {"four_arm_smoke_matrix", "exact_resume"}
        }
        if set(matrix_gates) != {"four_arm_smoke_matrix", "exact_resume"}:
            fail("formal authorization lacks the paired smoke/exact matrix gates")
        smoke_gate = matrix_gates["four_arm_smoke_matrix"]
        exact_gate = matrix_gates["exact_resume"]
        if (
            smoke_gate["source_path"] != exact_gate["source_path"]
            or smoke_gate["sha256"] != exact_gate["sha256"]
        ):
            fail(
                "formal smoke-matrix and exact-resume gates must be two logical "
                "checks over the same immutable receipt"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload": payload,
        "validated_gate_receipts": validated_gates,
    }


def authorization_template(
    *,
    phase: str,
    source: Mapping[str, object],
    dataset: Mapping[str, object],
    transfer: Mapping[str, object],
    runtime_sandbox: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
) -> dict:
    """Return a deliberately non-authorizing template for review and signing."""

    template = {
        "schema": AUTHORIZATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "NOT_AUTHORIZED",
        "gates_status": "NOT_VERIFIED",
        "authorization_id": "REPLACE_AFTER_GATE_REVIEW",
        "authorized_by": "REPLACE_WITH_REVIEWER_IDENTITY",
        "issued_at_utc": "REPLACE_WITH_UTC_TIMESTAMP",
        "allowed_arms": list(ARMS),
        "allowed_seeds": expected_authorization_seeds(phase),
        "source_git_head": source["git_head"],
        "source_content_sha256": source["content_sha256"],
        "preregistration_sha256": preregistration_record()["sha256"],
        "dataset_sha256": dataset["sha256"],
        "transfer_sha256": transfer["sha256"],
        "expected_amp_skip_attempts": None,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "gate_receipts": [
            {
                "name": name,
                "schema": spec["schema"],
                "status": spec["status"],
                "path": f"REPLACE_WITH_{name.upper()}_RECEIPT_PATH",
                "sha256": "0" * 64,
            }
            for name, spec in authorization_gate_contract(phase).items()
        ],
    }
    if runtime_sandbox is not None and runtime is not None:
        template.update(
            {
                "role_e_gate_runtime": role_e_gate_runtime_scope(runtime),
                "runtime_sandbox_tree_metadata_sha256": runtime_sandbox[
                    "sandbox_tree_metadata_sha256"
                ],
                "runtime_sandbox_critical_files_sha256": runtime_sandbox[
                    "critical_files_sha256"
                ],
                "runtime_software_sha256": runtime["software_sha256"],
            }
        )
    return template


def validate_arm_seed_phase(phase: str, arm: str, seed: int) -> None:
    if phase not in PHASES:
        fail(f"unsupported phase: {phase!r}")
    if arm not in ARMS:
        fail(f"unsupported arm: {arm!r}; expected one of {list(ARMS)}")
    if seed not in PHASES[phase]["seeds"]:
        fail(f"phase={phase} permits only seeds {PHASES[phase]['seeds']}, got {seed}")


def build_training_command(
    *,
    python_bin: str | Path,
    data: Path,
    transfer: Path,
    outdir: Path,
    phase: str,
    arm: str,
    seed: int,
    resume: Path | None = None,
    runtime_command: Sequence[str] | None = None,
    stop_after_attempts: int | None = None,
) -> list[str]:
    validate_arm_seed_phase(phase, arm, seed)
    if stop_after_attempts is not None:
        if phase != "smoke" or resume is not None:
            fail("--stop-after-attempts is gate-only, smoke-only, and fresh-only")
        if (
            isinstance(stop_after_attempts, bool)
            or stop_after_attempts != PLANNED_PAUSE_ATTEMPTS
        ):
            fail("--stop-after-attempts must equal the frozen gate target 16")
    factors = ARMS[arm]
    command = [
        *(list(runtime_command) if runtime_command is not None else [str(python_bin)]),
        str(REPO_ROOT / "ct_train.py"),
        f"--data={data}",
        f"--outdir={outdir}",
        "--nosubdir",
        "--cond=False",
        "--arch=ddpmpp",
        "--precond=ect",
        "--batch=128",
        "--batch-gpu=16",
        "--optim=RAdam",
        "--lr=0.0001",
        "--dropout=0.2",
        "--augment=0",
        "--xflip=False",
        "--mean=-1.1",
        "--std=2.0",
        "--mapping=sigmoid",
        "--global-gap-scale=1.0",
        f"--factorial-protocol={FACTORIAL_PROTOCOL}",
        f"--target-gap-scale={factors['target_gap_scale']}",
        f"--denominator-gap-scale={factors['denominator_gap_scale']}",
        "-q",
        "256",
        "-k",
        "8",
        "-b",
        "1",
        "-c",
        "0",
        "--double=10000",
        "--ema_beta=0.9993",
        f"--seed={seed}",
        "--fp16=True",
        "--tf32=False",
        "--ls=1.0",
        "--enable_amp=True",
        "--bench=True",
        "--cache=True",
        "--workers=1",
        "--metrics=none",
        f"--duration={PHASES[phase]['duration_mimg']}",
        "--tick=10",
        "--snap=0",
        "--dump=0",
        "--ckpt=10",
        "--sample_every=26",
        "--eval_every=50",
        "--mid_t=0.821",
        "--adaptive-update-kimg=0.5",
    ]
    if resume is None:
        command.append(f"--transfer={transfer}")
    else:
        command.append(f"--resume={resume}")
    if stop_after_attempts is not None:
        command.append(f"--stop-after-attempts={stop_after_attempts}")
    return command


def training_contract(phase: str, arm: str, seed: int) -> dict:
    validate_arm_seed_phase(phase, arm, seed)
    phase_spec = PHASES[phase]
    return {
        "phase": phase,
        "arm": arm,
        "seed": seed,
        **ARMS[arm],
        "factorial_protocol": FACTORIAL_PROTOCOL,
        "mapping": "sigmoid",
        "global_gap_scale": "1.0",
        "duration_mimg": phase_spec["duration_mimg"],
        "requested_kimg": phase_spec["requested_kimg"],
        "ct_train_total_kimg": phase_spec["ct_train_total_kimg"],
        "expected_processed_nimg": phase_spec["expected_processed_nimg"],
        "expected_optimizer_attempts": phase_spec["expected_attempts"],
        "q": 256,
        "k": 8,
        "b": 1,
        "c": 0,
        "batch": 128,
        "batch_gpu": 16,
        "optimizer": {"name": "RAdam", "lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8},
        "dropout": 0.2,
        "augmentation": 0,
        "xflip": False,
        "fp16": True,
        "amp": True,
        "tf32": False,
        "ema_beta": 0.9993,
        "tick_kimg": 10,
        "numbered_snapshots": False,
        "numbered_state_dumps": False,
        "latest_checkpoint_ticks": 10,
        "preview_ticks": 26,
        "training_metrics": [],
    }


def _parse_csv_line(line: str, expected_fields: int, label: str) -> list[str]:
    values = [part.strip() for part in line.split(",")]
    if len(values) != expected_fields:
        fail(f"unexpected nvidia-smi {label} row: {line!r}")
    return values


def query_gpu(gpu: str) -> dict:
    if not _GPU_RE.fullmatch(gpu):
        fail(f"GPU must be an explicit index or GPU UUID, got {gpu!r}")
    fields = "index,uuid,name,memory.total,driver_version,compute_mode"
    output = checked_output(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"GPU selector {gpu!r} resolved to {len(lines)} devices, expected exactly one")
    index, gpu_uuid, name, memory_mib, driver, compute_mode = _parse_csv_line(lines[0], 6, "GPU")
    record = {
        "requested_selector": gpu,
        "physical_index": int(index),
        "uuid": gpu_uuid,
        "name": name,
        "memory_total_mib": int(memory_mib),
        "driver_version": driver,
        "compute_mode": compute_mode,
    }
    if "A100" not in name or record["memory_total_mib"] < 80000:
        fail(
            "frozen server protocol requires one A100 80GB GPU, got "
            f"name={name!r} memory_total_mib={record['memory_total_mib']}"
        )
    return record


def query_gpu_compute_processes(
    gpu_uuid: str, *, timeout_seconds: float = 5.0
) -> list[dict[str, object]]:
    query = "gpu_uuid,pid,process_name,used_gpu_memory"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = (getattr(exc, "output", "") or "").strip()
        fail(f"cannot audit GPU compute processes: {detail or exc}")
    processes = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = _parse_csv_line(line, 4, "compute-process")
        if fields[0] == gpu_uuid:
            try:
                pid = int(fields[1])
            except ValueError:
                fail(f"invalid compute-process PID from nvidia-smi: {fields[1]!r}")
            processes.append(
                {
                    "pid": pid,
                    "process_name": fields[2],
                    "used_gpu_memory_mib": fields[3],
                }
            )
    return processes


def assert_gpu_idle(gpu_record: Mapping[str, object]) -> dict:
    query = "gpu_uuid,pid,process_name,used_gpu_memory"
    busy = query_gpu_compute_processes(str(gpu_record["uuid"]))
    if busy:
        fail(f"selected GPU is not exclusive/idle: uuid={gpu_record['uuid']} processes={busy}")
    return {
        "checked_utc": utc_now(),
        "gpu_uuid": gpu_record["uuid"],
        "compute_process_count": 0,
        "query": query,
    }


def process_tree_pids(
    root_pid: int, *, timeout_seconds: float = 5.0
) -> set[int]:
    """Return the live process tree rooted at an explicitly launched PID."""

    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid="],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = (getattr(exc, "output", "") or "").strip()
        fail(f"cannot audit launched process descendants: {detail or exc}")
    children: dict[int, list[int]] = defaultdict(list)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, parent = (int(value) for value in fields)
        except ValueError:
            continue
        children[parent].append(pid)
    result = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def linux_process_snapshot() -> dict[int, dict[str, int]]:
    """Read PID ancestry and stable start identities from one Linux /proc scan."""

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        fail("bounded process-group escalation requires Linux /proc")
    records: dict[int, dict[str, int]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        fail(f"cannot scan /proc for bounded escalation: {exc}")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_bytes()
            close = raw.rfind(b")")
            if close < 0:
                continue
            fields = raw[close + 2 :].split()
            if len(fields) < 20:
                continue
            pid = int(entry.name)
            records[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "sid": int(fields[3]),
                "starttime": int(fields[19]),
            }
        except (OSError, ValueError):
            # /proc is inherently racing and unrelated processes may expose
            # malformed/non-text comm fields.  Missing entries are ignored;
            # callers separately require the exact launcher-owned root.
            continue
    return records


def snapshot_descendants(
    snapshot: Mapping[int, Mapping[str, int]], root_pid: int
) -> dict[int, Mapping[str, int]]:
    if root_pid not in snapshot:
        return {}
    children: dict[int, list[int]] = defaultdict(list)
    for pid, record in snapshot.items():
        children[int(record["ppid"])].append(pid)
    descendants: dict[int, Mapping[str, int]] = {}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in descendants or pid not in snapshot:
            continue
        descendants[pid] = snapshot[pid]
        pending.extend(children.get(pid, ()))
    return descendants


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        fail(f"cannot audit launcher-owned process group {pgid}: {exc}")


def wait_for_process_groups_gone(
    process_groups: Iterable[int], *, timeout_seconds: float
) -> list[int]:
    groups = sorted(set(int(value) for value in process_groups))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = [pgid for pgid in groups if process_group_exists(pgid)]
        if not remaining:
            return []
        time.sleep(0.05)
    return [pgid for pgid in groups if process_group_exists(pgid)]


def load_matrix_child_tokens(
    registry_dir: Path, *, matrix_outer_pid: int
) -> list[dict[str, object]]:
    if (
        registry_dir.is_symlink()
        or not registry_dir.is_dir()
        or not registry_dir.is_absolute()
    ):
        raise ProcessCleanupError(
            f"matrix child registry is missing, relative, or a symlink: {registry_dir}"
        )
    tokens: list[dict[str, object]] = []
    expected_keys = {
        "schema",
        "label",
        "matrix_outer_pid",
        "launcher_parent_pid",
        "pid",
        "pgid",
        "sid",
        "starttime",
        "command_sha256",
    }
    try:
        entries = sorted(registry_dir.iterdir())
    except OSError as exc:
        raise ProcessCleanupError(
            f"cannot enumerate matrix child registry {registry_dir}: {exc}"
        ) from exc
    for path in entries:
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ProcessCleanupError(
                f"matrix child registry contains an incomplete/unexpected entry: {path}"
            )
        try:
            token = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProcessCleanupError(
                f"cannot validate matrix child ownership token {path}: {exc}"
            ) from exc
        if not isinstance(token, dict) or set(token) != expected_keys:
            raise ProcessCleanupError(
                f"matrix child ownership token has the wrong schema keys: {path}"
            )
        if token.get("schema") != MATRIX_CHILD_TOKEN_SCHEMA:
            raise ProcessCleanupError(
                f"matrix child ownership token has the wrong schema: {path}"
            )
        if token.get("matrix_outer_pid") != matrix_outer_pid:
            raise ProcessCleanupError(
                f"matrix child token is bound to another outer process: {path}"
            )
        integers = (
            "matrix_outer_pid",
            "launcher_parent_pid",
            "pid",
            "pgid",
            "sid",
            "starttime",
        )
        if any(
            isinstance(token.get(key), bool)
            or not isinstance(token.get(key), int)
            or int(token[key]) <= 1
            for key in integers
        ):
            raise ProcessCleanupError(
                f"matrix child token has an invalid process identity: {path}"
            )
        if token["pid"] != token["pgid"] or token["pid"] != token["sid"]:
            raise ProcessCleanupError(
                f"matrix child token is not a private session leader: {path}"
            )
        if not isinstance(token.get("label"), str) or not token["label"]:
            raise ProcessCleanupError(f"matrix child token label is invalid: {path}")
        if not isinstance(token.get("command_sha256"), str) or not _SHA_RE.fullmatch(
            str(token["command_sha256"])
        ):
            raise ProcessCleanupError(
                f"matrix child token command hash is invalid: {path}"
            )
        tokens.append({**token, "token_path": str(path)})
    pids = [int(token["pid"]) for token in tokens]
    if len(pids) != len(set(pids)):
        raise ProcessCleanupError("matrix child registry contains duplicate PIDs")
    return tokens


def drain_matrix_child_registry(
    registry_dir: Path, *, matrix_outer_pid: int
) -> dict[str, object]:
    """Stop and prove absence of inner sessions registered by an arm wrapper."""

    tokens = load_matrix_child_tokens(
        registry_dir, matrix_outer_pid=matrix_outer_pid
    )
    actions: list[dict[str, object]] = []
    for token in tokens:
        sid = int(token["sid"])
        snapshot = linux_process_snapshot()
        members = {
            pid: record
            for pid, record in snapshot.items()
            if int(record["sid"]) == sid
        }
        if not members:
            continue
        root = members.get(int(token["pid"]))
        if root is not None and int(root["starttime"]) != int(token["starttime"]):
            raise ProcessCleanupError(
                "matrix child PID identity changed before inner-session drain"
            )
        groups = sorted({int(record["pgid"]) for record in members.values()})
        if os.getpgrp() in groups or any(pgid < 2 for pgid in groups):
            raise ProcessCleanupError(
                "matrix child registry resolved an unsafe process group"
            )
        signals_sent: list[str] = []
        for signum, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
            sent = False
            for pgid in groups:
                try:
                    os.killpg(pgid, signum)
                    sent = True
                except ProcessLookupError:
                    continue
            if sent:
                signals_sent.append(name)
            deadline = time.monotonic() + (0.5 if signum == signal.SIGTERM else 2.0)
            while time.monotonic() < deadline:
                current = linux_process_snapshot()
                if not any(int(record["sid"]) == sid for record in current.values()):
                    break
                time.sleep(0.05)
            current = linux_process_snapshot()
            if not any(int(record["sid"]) == sid for record in current.values()):
                break
        remaining = {
            pid: record
            for pid, record in linux_process_snapshot().items()
            if int(record["sid"]) == sid
        }
        if remaining:
            raise ProcessCleanupError(
                "matrix-owned inner session remains after SIGTERM/SIGKILL: "
                f"sid={sid} pids={sorted(remaining)}"
            )
        actions.append(
            {
                "token_path": token["token_path"],
                "pid": token["pid"],
                "sid": sid,
                "process_groups": groups,
                "signals": signals_sent,
            }
        )
    return {
        "registry_directory": str(registry_dir),
        "token_count": len(tokens),
        "active_inner_sessions_drained": actions,
        "cleanup_confirmed": True,
    }


def kill_verified_process_tree(
    process: subprocess.Popen[bytes], *, seed: int, arm: str
) -> dict[str, object]:
    """SIGKILL only stable process groups descended from our session leader."""

    if process.poll() is not None:
        if process_group_exists(process.pid):
            raise ProcessCleanupError(
                "launcher leader exited but its process group still exists; "
                "ownership can no longer be safely re-established"
            )
        return {
            "seed": seed,
            "arm": arm,
            "outer_process_group": process.pid,
            "verified_descendant_pids": [],
            "vanished_short_lived_descendant_pids": [],
            "killed_process_groups": [],
            "signal": "ALREADY_EXITED",
        }
    first_snapshot = linux_process_snapshot()
    first_descendants = snapshot_descendants(first_snapshot, process.pid)
    first_root = first_descendants.get(process.pid)
    if first_root is None:
        raise ProcessLookupError(process.pid)
    second_snapshot = linux_process_snapshot()
    second_descendants = snapshot_descendants(second_snapshot, process.pid)
    second_root = second_descendants.get(process.pid)
    if (
        second_root is None
        or second_root["starttime"] != first_root["starttime"]
        or second_root["pgid"] != first_root["pgid"]
        or int(second_root["pgid"]) != process.pid
    ):
        fail("launcher-owned process identity changed during final drain audit")
    stable_descendants = {
        pid: second_descendants[pid]
        for pid, identity in first_descendants.items()
        if pid in second_descendants
        and second_descendants[pid]["starttime"] == identity["starttime"]
    }
    descendant_pgids = {
        int(identity["pgid"]) for identity in stable_descendants.values()
    }
    descendant_pgids.add(process.pid)
    if os.getpgrp() in descendant_pgids:
        fail("final drain resolved the matrix process group; refusing broad kill")
    ordered_pgids = sorted(
        descendant_pgids, key=lambda pgid: pgid == process.pid
    )
    for pgid in ordered_pgids:
        if pgid < 2:
            fail(f"final drain resolved an invalid process group: {pgid}")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        fail("launcher-owned process survived verified SIGKILL drain")
    remaining_groups = wait_for_process_groups_gone(
        ordered_pgids, timeout_seconds=2.0
    )
    if remaining_groups:
        raise ProcessCleanupError(
            "launcher-owned descendant process groups remain after verified "
            f"SIGKILL drain: {remaining_groups}"
        )
    return {
        "seed": seed,
        "arm": arm,
        "outer_process_group": process.pid,
        "verified_descendant_pids": sorted(stable_descendants),
        "vanished_short_lived_descendant_pids": sorted(
            set(first_descendants) - set(stable_descendants)
        ),
        "killed_process_groups": ordered_pgids,
        "signal": "SIGKILL",
    }


def terminate_own_process_group(process: subprocess.Popen[bytes]) -> list[str]:
    """Stop and confirm every snapshotted process group in our child tree.

    The leader is deliberately not polled/reaped until after group escalation;
    this prevents its PID/PGID from being reused while a TERM-ignoring child
    still holds the stdout pipe or GPU context.
    """

    if process.returncode is not None:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return []
        fail(
            "refuse to signal a bare PID/PGID after the launcher-owned leader "
            "was already reaped"
        )
    snapshot = linux_process_snapshot() if Path("/proc").is_dir() else {}
    descendants = snapshot_descendants(snapshot, process.pid)
    root = descendants.get(process.pid)
    process_groups = {process.pid}
    if root is not None:
        if int(root["pgid"]) != process.pid or int(root["sid"]) != process.pid:
            fail("launcher child is not the expected start_new_session leader")
        process_groups.update(int(item["pgid"]) for item in descendants.values())
    elif Path("/proc").is_dir():
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return []
        fail(
            "refuse to signal an existing process group without the exact "
            "launcher-owned /proc root identity"
        )
    if os.getpgrp() in process_groups or any(pgid < 2 for pgid in process_groups):
        fail("launcher child drain resolved an unsafe process group")

    ordered_groups = sorted(process_groups, key=lambda pgid: pgid == process.pid)
    signals_sent: list[str] = []
    term_sent = False
    for pgid in ordered_groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            continue
    if term_sent:
        signals_sent.append("SIGTERM")
    if process.returncode is None:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not any(process_group_exists(pgid) for pgid in ordered_groups):
            break
        time.sleep(0.1)
    kill_sent = False
    for pgid in ordered_groups:
        if not process_group_exists(pgid):
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            continue
    if kill_sent:
        signals_sent.append("SIGKILL")
    if process.returncode is None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            fail("launcher child leader survived SIGTERM/SIGKILL escalation")
    group_deadline = time.monotonic() + 2.0
    while time.monotonic() < group_deadline:
        if not any(process_group_exists(pgid) for pgid in ordered_groups):
            break
        time.sleep(0.05)
    remaining = [
        pgid for pgid in ordered_groups if process_group_exists(pgid)
    ]
    if remaining:
        fail(f"launcher-owned process groups remain after SIGKILL: {remaining}")
    return signals_sent


def stop_and_reap_process_group(
    process: subprocess.Popen[bytes], *, label: str
) -> list[str]:
    """Terminate a launcher-owned session and confirm its leader is reaped."""

    signals_sent: list[str] = []
    try:
        signals_sent = terminate_own_process_group(process)
    except LaunchError as exc:
        raise ProcessCleanupError(
            f"{label} could not prove full process-tree cleanup: {exc}"
        ) from exc
    if process.poll() is None:
        raise ProcessCleanupError(
            f"{label} child remains live after verified stop/reap"
        )
    return signals_sent


@contextlib.contextmanager
def gpu_lock(gpu: str, lock_root: Path) -> Iterator[dict]:
    root = lock_root.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu)
    path = root / f"{EXPERIMENT_ID}-gpu-{safe}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    handle = os.fdopen(fd, "r+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail(f"GPU launcher lock is already held: {path}")
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "acquired_utc": utc_now()}))
        handle.flush()
        os.fsync(handle.fileno())
        yield {"path": str(path), "mechanism": "fcntl.flock(LOCK_EX|LOCK_NB)"}
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def runtime_prefix(runtime_sandbox: Path, python_bin: str) -> tuple[list[str], dict]:
    sandbox = runtime_sandbox.expanduser()
    if sandbox.is_symlink() or not sandbox.is_dir():
        fail(f"runtime sandbox must be a real directory: {sandbox}")
    sandbox = sandbox.resolve(strict=True)
    inside_value = os.environ.get(IN_SANDBOX_ENV)
    if inside_value not in (None, "1"):
        fail(f"{IN_SANDBOX_ENV} must be unset or exactly '1'")
    if not python_bin or "/" in python_bin:
        fail("--python-bin must be the in-sandbox executable name (normally 'python')")
    if inside_value == "1":
        if sys.version_info < (3, 10):
            fail("already-inside launcher Python must be version 3.10 or newer")
        if not sys.executable or not Path(sys.executable).is_file():
            fail("already-inside launcher sys.executable is not a regular file")
        return [sys.executable], {
            "sandbox_path": str(sandbox),
            "bind_specs": list(RUNTIME_BIND_SPECS),
            "invocation_mode": "already_inside_runtime_sandbox",
            "already_inside_runtime_sandbox": True,
            "outer_apptainer_executable": None,
            "outer_apptainer_version": None,
            "python_command": sys.executable,
            "launcher_python_version": platform.python_version(),
        }
    apptainer_raw = shutil.which("apptainer")
    if not apptainer_raw:
        fail("apptainer executable not found on the host")
    apptainer = Path(apptainer_raw).resolve(strict=True)
    version = checked_output([str(apptainer), "--version"])
    command = [str(apptainer), "exec", "--nv"]
    for bind_spec in RUNTIME_BIND_SPECS:
        command.extend(("--bind", bind_spec))
    command.extend((str(sandbox), python_bin))
    return command, {
        "sandbox_path": str(sandbox),
        "bind_specs": list(RUNTIME_BIND_SPECS),
        "invocation_mode": "host_apptainer_exec",
        "already_inside_runtime_sandbox": False,
        "apptainer_executable": str(apptainer),
        "apptainer_executable_sha256": sha256_file(apptainer),
        "apptainer_version": version,
        "python_command": python_bin,
    }


def runtime_sandbox_fingerprint(runtime_record: Mapping[str, object]) -> dict:
    """Bind a writable sandbox without pretending it is a single image file.

    The tree digest covers type, mode, size, mtime, and symlink target for every
    entry.  Critical identity files are additionally content-hashed.  Exact
    Python/package/binary hashes are added by ``runtime_environment``.
    """

    root = Path(str(runtime_record["sandbox_path"]))
    tree_digest = hashlib.sha256()
    entry_count = 0
    regular_file_count = 0
    symlink_count = 0
    total_regular_bytes = 0
    critical = []
    critical_prefixes = (
        ".singularity.d/",
        "etc/os-release",
        "etc/ld.so.cache",
    )
    def walk_error(exc: OSError) -> None:
        fail(f"cannot walk runtime sandbox {root}: {exc}")

    for current, dirnames, filenames in os.walk(
        root, topdown=True, onerror=walk_error, followlinks=False
    ):
        dirnames.sort()
        filenames.sort()
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                stat = path.lstat()
            except OSError as exc:
                fail(f"cannot stat runtime sandbox entry {path}: {exc}")
            if path.is_symlink():
                kind = "symlink"
                target = os.readlink(path)
                symlink_count += 1
            elif path.is_dir():
                kind = "directory"
                target = ""
            elif path.is_file():
                kind = "file"
                target = ""
                regular_file_count += 1
                total_regular_bytes += stat.st_size
            else:
                kind = "other"
                target = ""
            line = (
                f"{relative}\0{kind}\0{stat.st_mode:o}\0{stat.st_size}\0"
                f"{stat.st_mtime_ns}\0{target}\n"
            ).encode("utf-8", errors="surrogateescape")
            tree_digest.update(line)
            entry_count += 1
            if kind == "file" and any(
                relative == prefix or relative.startswith(prefix)
                for prefix in critical_prefixes
            ):
                critical.append(
                    {
                        "path": relative,
                        "size_bytes": stat.st_size,
                        "sha256": sha256_file(path),
                    }
                )
    result = dict(runtime_record)
    result.update(
        {
            "sandbox_fingerprint_algorithm": "full-tree-metadata-plus-critical-content-sha256-v1",
            "sandbox_tree_metadata_sha256": tree_digest.hexdigest(),
            "sandbox_entry_count": entry_count,
            "sandbox_regular_file_count": regular_file_count,
            "sandbox_symlink_count": symlink_count,
            "sandbox_total_regular_bytes": total_regular_bytes,
            "critical_files": critical,
            "critical_files_sha256": canonical_sha256(critical),
        }
    )
    return result


def runtime_environment(runtime_command: Sequence[str], env: Mapping[str, str]) -> dict:
    code = r'''import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import PIL
import numpy
import torch

def digest(path):
    if not path or not os.path.isfile(path):
        return None
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

modules = {
    "python": sys.executable,
    "torch": torch.__file__,
    "torch_C": getattr(torch._C, "__file__", None),
    "numpy": numpy.__file__,
    "PIL": PIL.__file__,
}
packages = sorted(
    (str(dist.metadata.get("Name") or ""), str(dist.version))
    for dist in importlib.metadata.distributions()
)
device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
selector = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_row = subprocess.check_output(
    [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
    stderr=subprocess.STDOUT,
).strip()
gpu_fields = [value.strip() for value in gpu_row.split(",")]
if len(gpu_fields) != 3:
    raise RuntimeError(f"unexpected visible GPU identity row: {gpu_row!r}")
print(json.dumps({
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "visible_device_name": None if device is None else device.name,
    "visible_device_total_memory": None if device is None else device.total_memory,
    "visible_device_compute_capability": (
        None if device is None else [device.major, device.minor]
    ),
    "visible_gpu_uuid": gpu_fields[0],
    "visible_gpu_name_nvidia_smi": gpu_fields[1],
    "visible_gpu_memory_mib_nvidia_smi": int(gpu_fields[2]),
    "already_inside_runtime_sandbox": (
        os.environ.get("ECT_Q256_LAUNCHER_IN_SANDBOX") == "1"
    ),
    "critical_runtime_files": {
        name: {"path": path, "sha256": digest(path)}
        for name, path in modules.items()
    },
    "installed_distributions": packages,
}, sort_keys=True))'''
    output = checked_output([*runtime_command, "-c", code], env=env)
    try:
        record = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"training Python environment probe returned invalid JSON: {exc}: {output!r}")
    if record.get("cuda_available") is not True or record.get("cuda_device_count") != 1:
        fail(f"explicit CUDA binding must expose exactly one usable GPU: {record}")
    if (
        not isinstance(record.get("visible_gpu_uuid"), str)
        or not str(record["visible_gpu_uuid"]).startswith("GPU-")
        or record.get("visible_device_name") != record.get("visible_gpu_name_nvidia_smi")
        or isinstance(record.get("visible_gpu_memory_mib_nvidia_smi"), bool)
        or not isinstance(record.get("visible_gpu_memory_mib_nvidia_smi"), int)
        or record["visible_gpu_memory_mib_nvidia_smi"] <= 0
    ):
        fail(f"PyTorch/nvidia-smi visible GPU identity mismatch: {record}")
    observed_python = str(record.get("python_version", "")).split()[0]
    exact_versions = {
        "python": (observed_python, EXPECTED_PYTHON_VERSION),
        "torch": (record.get("torch_version"), EXPECTED_TORCH_VERSION),
        "torch CUDA": (record.get("torch_cuda_version"), EXPECTED_TORCH_CUDA_VERSION),
    }
    for label, (observed, expected) in exact_versions.items():
        if observed != expected:
            fail(f"frozen runtime {label} mismatch: {observed!r} != {expected!r}")
    expected_inside = os.environ.get(IN_SANDBOX_ENV) == "1"
    if record.get("already_inside_runtime_sandbox") is not expected_inside:
        fail("runtime probe disagrees with the launcher sandbox invocation mode")
    software = {
        key: record[key]
        for key in (
            "python_executable",
            "python_version",
            "platform",
            "torch_version",
            "torch_cuda_version",
            "torch_cudnn_version",
            "already_inside_runtime_sandbox",
            "critical_runtime_files",
            "installed_distributions",
        )
    }
    record["software_sha256"] = canonical_sha256(software)
    return record


def host_environment() -> dict:
    """Collect non-secret host/container identity fields for the launch receipt."""

    record: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
    }
    for label, raw_path in (
        ("os_release", Path("/etc/os-release")),
        ("self_cgroup", Path("/proc/self/cgroup")),
        ("pid1_cgroup", Path("/proc/1/cgroup")),
    ):
        try:
            data = raw_path.read_bytes()
        except OSError:
            continue
        record[label] = {
            "path": str(raw_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "text": data[:16384].decode("utf-8", errors="replace"),
            "truncated": len(data) > 16384,
        }
    visible_container_env = {}
    for key in (
        "CONTAINER_IMAGE",
        "CUDA_DRIVER_VERSION",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        if key in os.environ:
            visible_container_env[key] = os.environ[key]
    record["container_environment"] = visible_container_env
    return record


def build_process_environment(gpu: str, master_port: int) -> dict[str, str]:
    if not (1024 <= master_port <= 65535):
        fail(f"master port outside 1024..65535: {master_port}")
    env = dict(os.environ)
    for key in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def assert_master_port_available(master_port: int) -> None:
    """Catch an already-bound rendezvous port immediately before startup."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", master_port))
        except OSError as exc:
            fail(f"MASTER_PORT {master_port} is unavailable: {exc}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_run_contained_file(run_dir: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        fail(f"{label} path escapes the run")
    root = run_dir.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            fail(f"{label} path contains a symlink: {current}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        fail(f"{label} cannot be resolved: {exc}")
    if not _is_within(resolved, root):
        fail(f"{label} resolves outside the run: {resolved}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        fail(f"{label} is not a non-empty regular file: {resolved}")
    return resolved


def validate_runs_root(path: Path) -> Path:
    root = path.expanduser()
    if not root.is_absolute():
        fail(f"runs root must be absolute: {root}")
    resolved = root.resolve(strict=False)
    if str(resolved) in {"/", str(Path.home().resolve())}:
        fail(f"refusing broad runs root: {resolved}")
    return resolved


def storage_preflight(runs_root: Path, phase: str) -> dict:
    probe = runs_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        fail(f"cannot resolve storage filesystem for runs root: {runs_root}")
    usage = shutil.disk_usage(probe)
    minimum_gib = 50 if phase == "formal" else 5
    free_gib = usage.free / (1024 ** 3)
    if free_gib < minimum_gib:
        fail(
            f"insufficient free storage for phase={phase}: "
            f"{free_gib:.3f} GiB < {minimum_gib} GiB"
        )
    return {
        "probe_path": str(probe.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_gib": minimum_gib,
    }


def create_fresh_run_dir(outdir: Path, runs_root: Path) -> Path:
    root = validate_runs_root(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved = outdir.expanduser().resolve(strict=False)
    if not _is_within(resolved, root):
        fail(f"fresh outdir must be inside runs root {root}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"fresh run directory must not already exist: {resolved}")
    return resolved


def atomic_json_exclusive(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    published = False
    try:
        fd = os.open(temporary, flags, 0o640)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            fail(f"refuse to overwrite immutable artifact: {path}")
        published = True
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            if not published:
                raise


def _copy_file_exclusive_fsync(source: Path | str, target: Path) -> None:
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with Path(source).open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def copy_authorization_into_run(run_dir: Path, authorization: Mapping[str, object]) -> dict:
    auth_dir = run_dir / "authorization"
    auth_dir.mkdir(mode=0o750)
    auth_target = auth_dir / "authorization_receipt.json"
    _copy_file_exclusive_fsync(authorization["path"], auth_target)
    gates = []
    gate_evidence = []
    for index, item in enumerate(authorization["validated_gate_receipts"], start=1):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(item["name"]))
        target = auth_dir / f"gate-{index:02d}-{safe}.json"
        _copy_file_exclusive_fsync(item["source_path"], target)
        gates.append(
            {
                "name": item["name"],
                "schema": item["schema"],
                "status": item["status"],
                "path": str(target.relative_to(run_dir)),
                "sha256": sha256_file(target),
            }
        )
        logical_checks = item.get("logical_checks")
        evidence = (
            logical_checks.get("evidence")
            if isinstance(logical_checks, dict)
            else None
        )
        if evidence is not None:
            if not isinstance(evidence, dict) or set(evidence) != {
                "pytest_log", "pytest_junit"
            }:
                fail(f"validated gate {item['name']!r} has malformed evidence")
            for artifact, record in sorted(evidence.items()):
                if not isinstance(record, dict):
                    fail(f"validated gate evidence {artifact!r} is malformed")
                suffix = ".xml" if artifact == "pytest_junit" else ".log"
                evidence_target = auth_dir / f"gate-{index:02d}-{safe}-{artifact}{suffix}"
                _copy_file_exclusive_fsync(
                    record["source_path"], evidence_target
                )
                evidence_sha = sha256_file(evidence_target)
                if evidence_sha != record.get("sha256"):
                    fail(f"copied gate evidence changed for {item['name']!r}/{artifact}")
                gate_evidence.append(
                    {
                        "name": item["name"],
                        "artifact": artifact,
                        "path": str(evidence_target.relative_to(run_dir)),
                        "sha256": evidence_sha,
                    }
                )
    dir_fd = os.open(auth_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {
        "receipt_path": str(auth_target.relative_to(run_dir)),
        "receipt_sha256": sha256_file(auth_target),
        "gate_receipts": gates,
        "gate_evidence": gate_evidence,
    }


def verify_internal_authorization(
    run_dir: Path,
    launch_manifest: Mapping[str, object],
    *,
    expected_phase: str | None = None,
    revalidation_command: Sequence[str] | None = None,
    revalidation_env: Mapping[str, str] | None = None,
) -> dict:
    record = launch_manifest.get("authorization")
    if not isinstance(record, dict):
        fail("original launch manifest is missing authorization binding")
    raw_receipt_path = record.get("receipt_path")
    if not isinstance(raw_receipt_path, str) or not raw_receipt_path:
        fail("run-contained authorization receipt path is invalid")
    receipt_path = _resolve_run_contained_file(
        run_dir, Path(raw_receipt_path), "run-contained authorization receipt"
    )
    if sha256_file(receipt_path) != record.get("receipt_sha256"):
        fail("run-contained authorization receipt is missing or hash-mismatched")
    authorization_payload = _load_json(
        receipt_path, "run-contained authorization receipt"
    )
    if "expected_amp_skip_attempts" not in authorization_payload:
        fail(
            "run-contained authorization must explicitly record "
            "expected_amp_skip_attempts, including null in observe mode"
        )
    phase = authorization_payload.get("phase")
    if not isinstance(phase, str):
        fail("run-contained authorization receipt has no phase")
    if expected_phase is not None and phase != expected_phase:
        fail(
            "run-contained authorization phase mismatch: "
            f"{phase!r} != {expected_phase!r}"
        )
    source = launch_manifest.get("source")
    runtime = launch_manifest.get("runtime")
    sandbox = launch_manifest.get("runtime_sandbox")
    assets = launch_manifest.get("assets")
    if (
        not isinstance(source, dict)
        or not isinstance(runtime, dict)
        or not isinstance(sandbox, dict)
        or not isinstance(assets, dict)
        or not isinstance(assets.get("dataset"), dict)
        or not isinstance(assets.get("transfer"), dict)
    ):
        fail("original launch manifest lacks authorization source/runtime/assets scope")
    role_e_runtime = role_e_gate_runtime_scope(runtime)
    verifier_contract = launch_manifest.get("post_training_verifier")
    if not isinstance(verifier_contract, dict):
        fail("original launch manifest lacks the post-training verifier contract")
    if "expected_skip_attempts" not in verifier_contract:
        fail(
            "original launch manifest must explicitly record the expected AMP "
            "skip signature, including null in observe mode"
        )
    expected_skip_attempts = verifier_contract.get("expected_skip_attempts")
    if expected_skip_attempts is not None:
        if (
            not isinstance(expected_skip_attempts, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in expected_skip_attempts
            )
            or expected_skip_attempts != sorted(set(expected_skip_attempts))
        ):
            fail("original launch manifest has an invalid AMP skip signature")
    frozen_top_level = {
        "schema": AUTHORIZATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "authorized",
        "gates_status": "PASS",
        "source_git_head": source.get("git_head"),
        "source_content_sha256": source.get("content_sha256"),
        "dataset_sha256": assets["dataset"].get("sha256"),
        "transfer_sha256": assets["transfer"].get("sha256"),
        "expected_amp_skip_attempts": expected_skip_attempts,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "role_e_gate_runtime": role_e_runtime,
        "runtime_sandbox_tree_metadata_sha256": sandbox.get(
            "sandbox_tree_metadata_sha256"
        ),
        "runtime_sandbox_critical_files_sha256": sandbox.get(
            "critical_files_sha256"
        ),
        "runtime_software_sha256": runtime.get("software_sha256"),
    }
    for field, expected in frozen_top_level.items():
        if authorization_payload.get(field) != expected:
            fail(f"run-contained authorization scope mismatch for {field!r}")
    contract = authorization_gate_contract(phase)
    declarations = authorization_payload.get("gate_receipts")
    copied_gates = record.get("gate_receipts")
    copied_evidence = record.get("gate_evidence")
    if (
        not isinstance(declarations, list)
        or not isinstance(copied_gates, list)
        or not isinstance(copied_evidence, list)
    ):
        fail("run-contained authorization gate records must be lists")
    if len(declarations) != len(contract) or len(copied_gates) != len(contract):
        fail("run-contained authorization gate count differs from its phase contract")
    declaration_by_name = {}
    for item in declarations:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "schema",
            "status",
            "path",
            "sha256",
        }:
            fail("run-contained authorization has a malformed gate declaration")
        name = item.get("name")
        if not isinstance(name, str) or name in declaration_by_name:
            fail("run-contained authorization has a duplicate/invalid gate name")
        declaration_by_name[name] = item
    if set(declaration_by_name) != set(contract):
        fail("run-contained authorization logical gate set differs from its phase contract")
    if phase == "formal" and (
        declaration_by_name["four_arm_smoke_matrix"].get("sha256")
        != declaration_by_name["exact_resume"].get("sha256")
    ):
        fail(
            "run-contained smoke-matrix and exact-resume gates must bind the "
            "same immutable receipt"
        )

    evidence_by_gate: dict[str, dict[str, Path]] = defaultdict(dict)
    for evidence_item in copied_evidence:
        if not isinstance(evidence_item, dict) or set(evidence_item) != {
            "name", "artifact", "path", "sha256"
        }:
            fail("run-contained copied gate evidence binding is malformed")
        evidence_name = evidence_item.get("name")
        artifact = evidence_item.get("artifact")
        if (
            evidence_name not in contract
            or artifact not in {"pytest_log", "pytest_junit"}
            or artifact in evidence_by_gate[evidence_name]
        ):
            fail("run-contained copied gate evidence is duplicated or unexpected")
        raw_evidence_path = evidence_item.get("path")
        declared_evidence_sha = evidence_item.get("sha256")
        if (
            not isinstance(raw_evidence_path, str)
            or not isinstance(declared_evidence_sha, str)
            or not _SHA_RE.fullmatch(declared_evidence_sha)
        ):
            fail("run-contained copied gate evidence path/hash is invalid")
        evidence_path = _resolve_run_contained_file(
            run_dir,
            Path(raw_evidence_path),
            f"run-contained gate evidence {evidence_name!r}/{artifact}",
        )
        if sha256_file(evidence_path) != declared_evidence_sha:
            fail("run-contained copied gate evidence is hash-mismatched")
        evidence_by_gate[evidence_name][artifact] = evidence_path
    if set(evidence_by_gate) != {"role_e_ab_parity"} or set(
        evidence_by_gate["role_e_ab_parity"]
    ) != {"pytest_log", "pytest_junit"}:
        fail("run-contained Role-E evidence set is incomplete or has extras")

    seen_names = set()
    gate_scope_cache: dict[str, dict] = {}
    for item in copied_gates:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "schema",
            "status",
            "path",
            "sha256",
        }:
            fail("run-contained copied gate binding is malformed")
        name = item.get("name")
        if not isinstance(name, str) or name in seen_names or name not in contract:
            fail("run-contained copied gate has a duplicate/invalid logical name")
        declaration = declaration_by_name[name]
        spec = contract[name]
        for field in ("schema", "status"):
            if item.get(field) != declaration.get(field) or item.get(field) != spec.get(field):
                fail(f"run-contained gate {name!r} binding mismatch for {field}")
        if item.get("sha256") != declaration.get("sha256"):
            fail(f"run-contained gate {name!r} binding mismatch for sha256")
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            fail(f"run-contained gate {name!r} has an invalid copied path")
        path = _resolve_run_contained_file(
            run_dir, Path(raw_path), f"run-contained gate {name!r}"
        )
        if sha256_file(path) != item.get("sha256"):
            fail(f"run-contained gate receipt is missing or hash-mismatched: {path}")
        gate_payload = _load_json(path, f"run-contained gate {name!r}")
        validate_gate_receipt_payload(
            phase=phase,
            name=name,
            payload=gate_payload,
            label=f"run-contained gate {name!r}",
            source=source,
            role_e_runtime=role_e_runtime,
            expected_skip_attempts=expected_skip_attempts,
            revalidation_command=revalidation_command,
            revalidation_env=revalidation_env,
            scope_cache=gate_scope_cache,
            role_e_evidence_override=evidence_by_gate.get(name),
        )
        seen_names.add(name)
    if seen_names != set(contract):
        fail("run-contained copied gate set is incomplete")
    return dict(record)


def validate_resume(
    state: Path,
    *,
    runs_root: Path,
    phase: str,
    arm: str,
    seed: int,
    source: Mapping[str, object],
    dataset: Mapping[str, object],
    transfer: Mapping[str, object],
    gpu: Mapping[str, object],
    runtime_sandbox: Mapping[str, object],
    runtime: Mapping[str, object],
    expected_skip_attempts: list[int] | None,
    outdir_override: Path | None,
    supplied_authorization: Path | None,
    revalidation_command: Sequence[str] | None,
    revalidation_env: Mapping[str, str] | None,
) -> tuple[Path, dict, dict, dict | None, Path]:
    if state.is_symlink() or not state.is_file() or not _STATE_RE.fullmatch(state.name):
        fail(f"resume must be a regular in-run training-state-*.pt: {state}")
    state = state.expanduser().resolve(strict=True)
    run_dir = state.parent
    root = validate_runs_root(runs_root)
    if not _is_within(run_dir, root):
        fail(f"resume run must remain inside runs root {root}: {run_dir}")
    if outdir_override is not None and outdir_override.expanduser().resolve(strict=True) != run_dir:
        fail("--outdir must be the exact parent directory of --resume")
    manifest_path = run_dir / "launch_manifest.json"
    manifest = _load_json(manifest_path, "original launch manifest")
    if (
        manifest.get("schema") != LAUNCH_SCHEMA
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("launch_kind") != "fresh_transfer"
        or manifest.get("status") != "authorized_to_start"
        or manifest.get("run_directory") != str(run_dir)
        or manifest.get("resume_state") is not None
        or manifest.get("resume_state_sha256") is not None
        or manifest.get("validated_planned_pause_completion") is not None
        or manifest.get("original_gate_control") is not None
        or manifest.get("original_launch_manifest_sha256") is not None
    ):
        fail("resume source is not an untouched original fresh launch manifest")
    exact = {
        "schema": LAUNCH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source.git_head": source["git_head"],
        "source.content_sha256": source["content_sha256"],
        "assets.dataset.sha256": dataset["sha256"],
        "assets.transfer.sha256": transfer["sha256"],
        "training.phase": phase,
        "training.arm": arm,
        "training.seed": seed,
        "gpu.uuid": gpu["uuid"],
        "runtime_sandbox.sandbox_tree_metadata_sha256": runtime_sandbox[
            "sandbox_tree_metadata_sha256"
        ],
        "runtime_sandbox.critical_files_sha256": runtime_sandbox[
            "critical_files_sha256"
        ],
        "runtime_sandbox.already_inside_runtime_sandbox": runtime_sandbox[
            "already_inside_runtime_sandbox"
        ],
        "runtime_sandbox.bind_specs": runtime_sandbox["bind_specs"],
        "runtime.software_sha256": runtime["software_sha256"],
        "runtime.already_inside_runtime_sandbox": runtime[
            "already_inside_runtime_sandbox"
        ],
        "post_training_verifier.expected_skip_attempts": expected_skip_attempts,
    }
    for dotted, expected in exact.items():
        value: object = manifest
        for key in dotted.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value != expected:
            fail(f"resume identity mismatch for {dotted}: {value!r} != {expected!r}")
    if manifest.get("training") != training_contract(phase, arm, seed):
        fail("resume training contract differs from the original immutable launch")
    gate_control = manifest.get("gate_control")
    if not isinstance(gate_control, dict):
        fail("resume launch manifest is missing the versioned gate_control")
    if gate_control.get("kind") not in {"none", "planned_exact_resume_pause"}:
        fail("resume launch manifest has an unknown gate_control kind")
    if gate_control.get("scientific_training_contract_unchanged") is not True:
        fail("resume gate_control does not preserve the scientific training contract")
    original_stop = gate_control.get("stop_after_attempts")
    expected_gate_kind = (
        "planned_exact_resume_pause" if original_stop is not None else "none"
    )
    expected_gate_control = {
        "kind": expected_gate_kind,
        "stop_after_attempts": original_stop,
        "scientific_training_contract_unchanged": True,
    }
    if gate_control != expected_gate_control:
        fail("resume gate_control is not the exact versioned launch contract")
    if original_stop is not None and (
        phase != "smoke"
        or isinstance(original_stop, bool)
        or original_stop != PLANNED_PAUSE_ATTEMPTS
    ):
        fail("resume launch manifest contains an invalid gate-only stop target")
    auth_record = verify_internal_authorization(
        run_dir,
        manifest,
        expected_phase=phase,
        revalidation_command=revalidation_command,
        revalidation_env=revalidation_env,
    )
    if supplied_authorization is not None:
        supplied = supplied_authorization.expanduser().resolve(strict=True)
        if sha256_file(supplied) != auth_record["receipt_sha256"]:
            fail("supplied resume authorization differs from original launch authorization")
    required = (
        "training_options.json",
        "initial_state_receipt_v1.json",
        "factorial_training_telemetry_v1.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        fail(f"strict self-contained resume is missing in-run artifacts: {missing}")
    existing_pass = validate_existing_verifier_receipts(
        run_dir,
        phase=phase,
        arm=arm,
        seed=seed,
        expected_skip_attempts=expected_skip_attempts,
    )
    if existing_pass is not None:
        fail(
            "run already has immutable PASS verifier receipts; direct resume/repeat "
            "is forbidden (matrix continuation may validate and skip it)"
        )
    planned_pause_completion = None
    effective_resume_state = state
    if original_stop is not None:
        if state.name != "training-state-latest.pt":
            fail("planned exact resume requires the run-contained training-state-latest.pt")
        prior_resume_manifests = sorted(run_dir.glob("resume_launch_manifest-*.json"))
        if prior_resume_manifests:
            fail(
                "planned exact-resume gate was already consumed or attempted; "
                f"audit required: {prior_resume_manifests}"
            )
        planned_pause_completion = validate_planned_pause_completion(
            run_dir,
            launch_manifest=manifest,
            stop_after_attempts=original_stop,
        )
        pause_evidence = preserve_planned_pause_evidence(
            run_dir, planned_pause_completion
        )
        planned_pause_completion = {
            **planned_pause_completion,
            "evidence": pause_evidence,
        }
        effective_resume_state = _resolve_run_contained_file(
            run_dir,
            Path(str(pause_evidence["resume_state_path"])),
            "preserved exact-resume state",
        )
    return (
        run_dir,
        manifest,
        auth_record,
        planned_pause_completion,
        effective_resume_state,
    )


def next_unique_name(run_dir: Path, prefix: str, suffix: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_dir / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}{suffix}"


def stream_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    monitored_gpu_uuid: str | None = None,
    gpu_monitor_record: dict[str, object] | None = None,
    gpu_monitor_interval_seconds: float = 1.0,
) -> int:
    if monitored_gpu_uuid is not None and gpu_monitor_record is None:
        fail("GPU monitoring requires a mutable evidence record")
    if gpu_monitor_interval_seconds <= 0:
        fail("GPU monitor interval must be positive")
    launch_command = list(command)
    registry_raw = env.get(MATRIX_CHILD_REGISTRY_ENV)
    if registry_raw is not None:
        registry_dir = Path(registry_raw)
        if (
            registry_dir.is_symlink()
            or not registry_dir.is_dir()
            or not registry_dir.is_absolute()
        ):
            fail("matrix child ownership registry is not a fixed absolute directory")
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", log_path.name)[:80]
        launch_command = [
            sys.executable,
            "-c",
            MATRIX_CHILD_WRAPPER_CODE,
            str(registry_dir.resolve(strict=True)),
            label,
            str(os.getpid()),
            *launch_command,
        ]
    with log_path.open("xb") as raw_log:
        managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        is_main_thread = threading.current_thread() is threading.main_thread()
        previous_signal_handlers: dict[int, object] = {}
        pending_signals: list[int] = []
        process: subprocess.Popen[bytes] | None = None
        if is_main_thread:
            def stop_launched_process(signum: int, _frame: object) -> None:
                if gpu_monitor_record is not None:
                    gpu_monitor_record["received_signal"] = {
                        "received_utc": utc_now(),
                        "signal": signal.Signals(signum).name,
                    }
                if process is None:
                    pending_signals.append(signum)
                    return
                if process.returncode is None:
                    stop_and_reap_process_group(process, label="stream-signal")
                raise LaunchError(
                    f"launcher received signal {signal.Signals(signum).name}; "
                    "stopping its own child process group"
                )

            for signum in managed_signals:
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, stop_launched_process)
        try:
            process = subprocess.Popen(
                launch_command,
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            for signum, handler in previous_signal_handlers.items():
                signal.signal(signum, handler)
            raise
        if pending_signals:
            stop_and_reap_process_group(process, label="stream-start-signal")
            for signum, handler in previous_signal_handlers.items():
                signal.signal(signum, handler)
            raise LaunchError(
                "launcher received signal "
                f"{signal.Signals(pending_signals[0]).name} while starting its "
                "child; the new child process group was stopped"
            )
        monitor_stop = threading.Event()
        monitor_thread = None
        monitor_exceptions: list[BaseException] = []
        if monitored_gpu_uuid is not None:
            assert gpu_monitor_record is not None
            monitor_started_monotonic = time.monotonic()
            cadence_grace_seconds = max(
                0.05, gpu_monitor_interval_seconds * 0.25
            )
            probe_timeout_seconds = max(
                0.05, min(0.4, gpu_monitor_interval_seconds * 0.4)
            )
            gpu_monitor_record.update(
                {
                    "schema": GPU_MONITOR_SCHEMA,
                    "status": "RUNNING",
                    "gpu_uuid": monitored_gpu_uuid,
                    "root_process_pid": process.pid,
                    "poll_interval_seconds": gpu_monitor_interval_seconds,
                    "cadence_grace_seconds": cadence_grace_seconds,
                    "probe_timeout_seconds": probe_timeout_seconds,
                    "started_utc": utc_now(),
                    "checks_completed": 0,
                    "first_check_offset_seconds": None,
                    "last_check_offset_seconds": None,
                    "monitor_duration_seconds": None,
                    "first_check_started_utc": None,
                    "last_check_started_utc": None,
                    "max_observed_poll_gap_seconds": 0.0,
                    "max_observed_check_duration_seconds": 0.0,
                    "max_observed_schedule_lateness_seconds": 0.0,
                    "foreign_process_incident": None,
                    "own_process_group_signals": [],
                    "received_signal": None,
                }
            )

            def monitor_gpu_exclusivity() -> None:
                assert gpu_monitor_record is not None
                next_check_started = time.monotonic()
                previous_check_started: float | None = None

                def stop_for_audit(kind: str, **details: object) -> None:
                    gpu_monitor_record["foreign_process_incident"] = {
                        "detected_utc": utc_now(),
                        "kind": kind,
                        **details,
                    }
                    gpu_monitor_record["status"] = "STOPPED_FOR_AUDIT"
                    gpu_monitor_record["own_process_group_signals"] = (
                        stop_and_reap_process_group(
                            process, label="stream-monitor-audit"
                        )
                    )

                while not monitor_stop.is_set() and process.poll() is None:
                    remaining = next_check_started - time.monotonic()
                    if remaining > 0 and monitor_stop.wait(remaining):
                        return
                    check_started = time.monotonic()
                    check_started_utc = utc_now()
                    lateness = max(0.0, check_started - next_check_started)
                    gpu_monitor_record["max_observed_schedule_lateness_seconds"] = max(
                        float(
                            gpu_monitor_record[
                                "max_observed_schedule_lateness_seconds"
                            ]
                        ),
                        lateness,
                    )
                    if lateness > cadence_grace_seconds:
                        stop_for_audit(
                            "GPU_MONITOR_CADENCE_MISSED",
                            schedule_lateness_seconds=lateness,
                            allowed_grace_seconds=cadence_grace_seconds,
                        )
                        return
                    if previous_check_started is not None:
                        gap = check_started - previous_check_started
                        gpu_monitor_record["max_observed_poll_gap_seconds"] = max(
                            float(
                                gpu_monitor_record[
                                    "max_observed_poll_gap_seconds"
                                ]
                            ),
                            gap,
                        )
                        if gap > gpu_monitor_interval_seconds + cadence_grace_seconds:
                            stop_for_audit(
                                "GPU_MONITOR_CADENCE_MISSED",
                                poll_gap_seconds=gap,
                                allowed_gap_seconds=(
                                    gpu_monitor_interval_seconds
                                    + cadence_grace_seconds
                                ),
                            )
                            return
                    else:
                        gpu_monitor_record["first_check_started_utc"] = check_started_utc
                        gpu_monitor_record["first_check_offset_seconds"] = (
                            check_started - monitor_started_monotonic
                        )
                    gpu_monitor_record["last_check_started_utc"] = check_started_utc
                    gpu_monitor_record["last_check_offset_seconds"] = (
                        check_started - monitor_started_monotonic
                    )
                    previous_check_started = check_started
                    try:
                        compute_processes = query_gpu_compute_processes(
                            monitored_gpu_uuid,
                            timeout_seconds=probe_timeout_seconds,
                        )
                        allowed_pids = process_tree_pids(
                            process.pid,
                            timeout_seconds=probe_timeout_seconds,
                        )
                    except Exception as exc:
                        stop_for_audit(
                            "GPU_AUDIT_FAILED",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        return
                    check_duration = time.monotonic() - check_started
                    gpu_monitor_record["max_observed_check_duration_seconds"] = max(
                        float(
                            gpu_monitor_record[
                                "max_observed_check_duration_seconds"
                            ]
                        ),
                        check_duration,
                    )
                    if check_duration > (
                        gpu_monitor_interval_seconds + cadence_grace_seconds
                    ):
                        stop_for_audit(
                            "GPU_MONITOR_CADENCE_MISSED",
                            check_duration_seconds=check_duration,
                            allowed_check_duration_seconds=(
                                gpu_monitor_interval_seconds
                                + cadence_grace_seconds
                            ),
                        )
                        return
                    gpu_monitor_record["checks_completed"] = (
                        int(gpu_monitor_record["checks_completed"]) + 1
                    )
                    foreign = [
                        record
                        for record in compute_processes
                        if int(record["pid"]) not in allowed_pids
                    ]
                    if foreign:
                        gpu_monitor_record["foreign_process_incident"] = {
                            "detected_utc": utc_now(),
                            "kind": "FOREIGN_GPU_COMPUTE_PROCESS",
                            "foreign_processes": foreign,
                            "allowed_process_tree_pids": sorted(allowed_pids),
                        }
                        gpu_monitor_record["status"] = "EXCLUSIVITY_LOST"
                        gpu_monitor_record["own_process_group_signals"] = (
                            stop_and_reap_process_group(
                                process, label="stream-monitor-foreign-process"
                            )
                        )
                        return
                    next_check_started += gpu_monitor_interval_seconds

            def monitor_gpu_exclusivity_guarded() -> None:
                try:
                    monitor_gpu_exclusivity()
                except BaseException as exc:
                    monitor_exceptions.append(exc)
                    assert gpu_monitor_record is not None
                    gpu_monitor_record["status"] = "STOPPED_FOR_AUDIT"
                    if gpu_monitor_record.get("foreign_process_incident") is None:
                        gpu_monitor_record["foreign_process_incident"] = {
                            "detected_utc": utc_now(),
                            "kind": "GPU_MONITOR_UNCAUGHT_EXCEPTION",
                            "error": f"{type(exc).__name__}: {exc}",
                        }

            try:
                monitor_thread = threading.Thread(
                    target=monitor_gpu_exclusivity_guarded,
                    name=f"gpu-exclusivity-{monitored_gpu_uuid}",
                    daemon=True,
                )
                monitor_thread.start()
            except BaseException:
                monitor_stop.set()
                stop_and_reap_process_group(
                    process, label="stream-monitor-start-failure"
                )
                if process.stdout is not None:
                    process.stdout.close()
                for signum, handler in previous_signal_handlers.items():
                    signal.signal(signum, handler)
                raise
        assert process.stdout is not None
        try:
            for chunk in iter(lambda: process.stdout.read(65536), b""):
                raw_log.write(chunk)
                raw_log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            returncode = process.wait()
            remaining_groups = wait_for_process_groups_gone(
                [process.pid], timeout_seconds=2.0
            )
            if remaining_groups:
                raise ProcessCleanupError(
                    "stream root exited while its launcher-owned process group "
                    f"still exists after teardown grace: {remaining_groups}"
                )
        except ProcessCleanupError:
            raise
        except BaseException:
            stop_and_reap_process_group(process, label="stream-read-failure")
            raise
        finally:
            monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=max(12.0, gpu_monitor_interval_seconds * 2))
                if monitor_thread.is_alive() and gpu_monitor_record is not None:
                    gpu_monitor_record["status"] = "STOPPED_FOR_AUDIT"
                    gpu_monitor_record["foreign_process_incident"] = {
                        "detected_utc": utc_now(),
                        "kind": "GPU_MONITOR_DID_NOT_STOP",
                    }
            process.stdout.close()
            for signum, handler in previous_signal_handlers.items():
                signal.signal(signum, handler)
        if monitor_exceptions:
            raise monitor_exceptions[0]
        if gpu_monitor_record is not None:
            gpu_monitor_record["finished_utc"] = utc_now()
            gpu_monitor_record["monitor_duration_seconds"] = (
                time.monotonic() - monitor_started_monotonic
            )
            if gpu_monitor_record.get("status") == "RUNNING":
                if int(gpu_monitor_record.get("checks_completed", 0)) < 1:
                    gpu_monitor_record["status"] = "STOPPED_FOR_AUDIT"
                    gpu_monitor_record["foreign_process_incident"] = {
                        "detected_utc": utc_now(),
                        "kind": "GPU_MONITOR_PERFORMED_NO_CHECKS",
                    }
                else:
                    gpu_monitor_record["status"] = "PASS"
        return returncode


def read_factorial_telemetry(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FACTORIAL_TELEMETRY_FIELDS:
                fail(
                    "factorial telemetry schema is not the exact 52-field v1 "
                    f"contract: {tuple(reader.fieldnames or ())!r}"
                )
            rows = list(reader)
    except OSError as exc:
        fail(f"cannot read factorial telemetry {path}: {exc}")
    if not rows:
        fail("factorial telemetry is empty")
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            fail(f"factorial telemetry row {row_number} has extra fields")
    return rows


def verify_completed_run(run_dir: Path, phase: str, arm: str) -> dict:
    required = (
        "training_options.json",
        "initial_state_receipt_v1.json",
        "train_summary.csv",
        "factorial_training_telemetry_v1.csv",
        "network-snapshot-latest.pkl",
        "training-state-latest.pt",
    )
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"completed run is missing required non-empty artifact: {path}")
    telemetry_path = run_dir / "factorial_training_telemetry_v1.csv"
    rows = read_factorial_telemetry(telemetry_path)
    expected_attempts = int(PHASES[phase]["expected_attempts"])
    expected_nimg = int(PHASES[phase]["expected_processed_nimg"])
    if len(rows) != expected_attempts:
        fail(f"telemetry attempts mismatch: {len(rows)} != {expected_attempts}")
    last = rows[-1]
    exact = {
        "schema": "ect.q256.target-weight-training-telemetry/v1",
        "protocol": FACTORIAL_PROTOCOL,
        "arm": arm,
        "attempted_iteration": str(expected_attempts),
        "processed_nimg": str(expected_nimg),
    }
    for key, expected in exact.items():
        if last.get(key) != expected:
            fail(f"final telemetry {key} mismatch: {last.get(key)!r} != {expected!r}")
    artifacts = {}
    for name in required:
        path = run_dir / name
        artifacts[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {
        "status": "PASS",
        "attempted_iterations": len(rows),
        "successful_optimizer_steps": int(last["successful_optimizer_steps"]),
        "processed_nimg": int(last["processed_nimg"]),
        "amp_skip_attempts": [
            int(row["attempted_iteration"])
            for row in rows
            if row.get("step_skipped", "").lower() in {"1", "true"}
        ],
        "artifacts": artifacts,
    }


def _strict_csv_int(value: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value or ""):
        fail(f"{label} must be a canonical non-negative integer, got {value!r}")
    return int(value)


def _finite_csv_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        fail(f"{label} must be finite: {exc}")
    if not (float("-inf") < result < float("inf")):
        fail(f"{label} must be finite, got {value!r}")
    return result


def verify_planned_pause_run(
    run_dir: Path,
    *,
    arm: str,
    seed: int,
    stop_after_attempts: int,
    runtime_command: Sequence[str],
    process_env: Mapping[str, str],
) -> dict:
    """Verify a gate-only durable pause without pretending it is a full arm."""

    if stop_after_attempts != PLANNED_PAUSE_ATTEMPTS:
        fail("planned-pause verification is reserved for the frozen 16-attempt gate")
    expected_nimg = stop_after_attempts * 128
    required = PLANNED_PAUSE_ARTIFACTS
    for name in required:
        path = run_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            fail(f"planned pause is missing a required regular artifact: {path}")
    for forbidden in (VALIDATION_FILENAME, HASH_RECEIPT_FILENAME):
        if (run_dir / forbidden).exists():
            fail(f"planned pause must not create a full-arm verifier receipt: {forbidden}")

    manifest = _load_json(run_dir / "launch_manifest.json", "launch manifest")
    gate_control = manifest.get("gate_control")
    expected_gate = {
        "kind": "planned_exact_resume_pause",
        "stop_after_attempts": stop_after_attempts,
        "scientific_training_contract_unchanged": True,
    }
    if gate_control != expected_gate:
        fail(f"planned-pause launch gate_control mismatch: {gate_control!r}")
    if manifest.get("training") != training_contract("smoke", arm, seed):
        fail("planned pause changed the frozen scientific training contract")

    options = _load_json(run_dir / "training_options.json", "training options")
    if options.get("stop_after_attempts") != stop_after_attempts:
        fail("training_options does not bind the planned pause target")
    if options.get("total_kimg") != 4 or options.get("resume_state_dump") is not None:
        fail("planned pause must be a fresh full-smoke contract with total_kimg=4")
    initial = _load_json(run_dir / "initial_state_receipt_v1.json", "initial receipt")
    if initial.get("attempted_iteration") != 0 or initial.get("processed_nimg") != 0:
        fail("planned pause initial receipt was not captured before attempt 1")
    if initial.get("seed") != seed or initial.get("factorial", {}).get("arm") != arm:
        fail("planned pause initial receipt identity mismatch")

    rows = read_factorial_telemetry(
        run_dir / "factorial_training_telemetry_v1.csv"
    )
    if len(rows) != stop_after_attempts:
        fail(
            "planned pause telemetry attempt count mismatch: "
            f"{len(rows)} != {stop_after_attempts}"
        )
    target_scale = float(ARMS[arm]["target_gap_scale"])
    denominator_scale = float(ARMS[arm]["denominator_gap_scale"])
    skip_count = 0
    previous_elapsed = -1.0
    count_fields = (
        "loss_nonfinite_count",
        "raw_grad_nonfinite_count",
        "sanitized_grad_nonfinite_count",
        "update_nonfinite_count",
        "model_nonfinite_count",
        "ema_nonfinite_count",
        "sample_count",
        "base_r_zero_count",
        "target_r_zero_count",
        "target_r_equal_t_count",
        "target_scaled_to_zero_count",
        "denominator_r_zero_count",
        "denominator_r_equal_t_count",
        "denominator_scaled_to_zero_count",
        "factor_nonfinite_count",
        "nonpositive_denominator_count",
    )
    finite_fields = (
        "loss",
        "raw_grad_finite_norm",
        "sanitized_grad_norm",
        "update_norm",
        "model_norm",
        "ema_norm",
        "target_delta_min",
        "target_delta_max",
        "target_delta_mean",
        "denominator_delta_min",
        "denominator_delta_max",
        "denominator_delta_mean",
        "learning_rate",
        "grad_scale_before",
        "grad_scale_after",
        "elapsed_sec",
        "gpu_hours_cumulative",
    )
    for attempt, row in enumerate(rows, start=1):
        label = f"planned telemetry attempt {attempt}"
        exact = {
            "schema": "ect.q256.target-weight-training-telemetry/v1",
            "protocol": FACTORIAL_PROTOCOL,
            "arm": arm,
            "attempted_iteration": str(attempt),
            "processed_nimg": str(attempt * 128),
            "stage": "0",
        }
        for field, expected in exact.items():
            if row[field] != expected:
                fail(f"{label}.{field} mismatch: {row[field]!r} != {expected!r}")
        if float(row["target_gap_scale"]) != target_scale:
            fail(f"{label}.target_gap_scale mismatch")
        if float(row["denominator_gap_scale"]) != denominator_scale:
            fail(f"{label}.denominator_gap_scale mismatch")
        if _finite_csv_float(row["processed_kimg"], f"{label}.processed_kimg") != attempt * 128 / 1000:
            fail(f"{label}.processed_kimg mismatch")
        for field in FACTORIAL_DIGEST_FIELDS:
            if not _SHA_RE.fullmatch(row[field]):
                fail(f"{label}.{field} is not a lowercase SHA256 digest")
        if target_scale == denominator_scale:
            if row["target_r_sha256"] != row["denominator_r_sha256"]:
                fail(f"{label} native target/denominator time digests differ")
            if row["target_delta_sha256"] != row["denominator_delta_sha256"]:
                fail(f"{label} native target/denominator gap digests differ")
        counts = {
            field: _strict_csv_int(row[field], f"{label}.{field}")
            for field in count_fields
        }
        if counts["sample_count"] != 128:
            fail(f"{label}.sample_count must equal 128")
        for field in (
            "base_r_zero_count",
            "target_r_zero_count",
            "target_scaled_to_zero_count",
            "denominator_r_zero_count",
            "denominator_scaled_to_zero_count",
        ):
            if counts[field] > counts["sample_count"]:
                fail(f"{label}.{field} exceeds sample_count")
        for field in (
            "loss_nonfinite_count",
            "sanitized_grad_nonfinite_count",
            "update_nonfinite_count",
            "model_nonfinite_count",
            "ema_nonfinite_count",
            "target_r_equal_t_count",
            "denominator_r_equal_t_count",
            "factor_nonfinite_count",
            "nonpositive_denominator_count",
        ):
            if counts[field] != 0:
                fail(f"{label}.{field} must be zero")
        step_skipped = _strict_csv_int(row["step_skipped"], f"{label}.step_skipped")
        if step_skipped not in (0, 1):
            fail(f"{label}.step_skipped must be 0 or 1")
        if step_skipped and attempt * 128 >= AMP_SKIP_WARMUP_PROCESSED_NIMG:
            fail(f"{label} AMP skip occurred after the tick-0 warm-up region")
        skip_count += step_skipped
        successes = _strict_csv_int(
            row["successful_optimizer_steps"],
            f"{label}.successful_optimizer_steps",
        )
        if successes != attempt - skip_count:
            fail(f"{label}.successful_optimizer_steps mismatch")
        raw_nonfinite = counts["raw_grad_nonfinite_count"]
        if bool(raw_nonfinite) != bool(step_skipped):
            fail(f"{label} raw-gradient non-finite status does not match AMP skip")
        values = {
            field: _finite_csv_float(row[field], f"{label}.{field}")
            for field in finite_fields
        }
        if values["loss"] < 0:
            fail(f"{label}.loss must be non-negative")
        try:
            raw_norm = float(row["raw_grad_norm"])
        except (TypeError, ValueError, OverflowError) as exc:
            fail(f"{label}.raw_grad_norm is invalid: {exc}")
        if step_skipped:
            if raw_norm != float("inf"):
                fail(f"{label}.raw_grad_norm must be +inf on an AMP skip")
            if values["grad_scale_after"] >= values["grad_scale_before"]:
                fail(f"{label} AMP skip did not reduce GradScaler scale")
            if values["update_norm"] != 0:
                fail(f"{label} skipped optimizer attempt changed parameters")
        else:
            if not math.isfinite(raw_norm) or raw_norm < 0:
                fail(f"{label}.raw_grad_norm must be finite and non-negative")
            if values["grad_scale_after"] < values["grad_scale_before"]:
                fail(f"{label} GradScaler scale fell without a recorded skip")
            if values["update_norm"] <= 0:
                fail(f"{label} successful optimizer update norm is not positive")
        for field in (
            "raw_grad_finite_norm",
            "sanitized_grad_norm",
            "model_norm",
            "ema_norm",
        ):
            if values[field] < 0:
                fail(f"{label}.{field} must be non-negative")
        for prefix in ("target_delta", "denominator_delta"):
            minimum = values[f"{prefix}_min"]
            maximum = values[f"{prefix}_max"]
            mean = values[f"{prefix}_mean"]
            if minimum <= 0 or not minimum <= mean <= maximum:
                fail(f"{label}.{prefix} must satisfy 0 < min <= mean <= max")
        if values["learning_rate"] != 1e-4:
            fail(f"{label}.learning_rate mismatch")
        if values["grad_scale_before"] <= 0 or values["grad_scale_after"] <= 0:
            fail(f"{label} GradScaler scales must be positive")
        if values["elapsed_sec"] < previous_elapsed:
            fail(f"{label}.elapsed_sec regressed")
        if not math.isclose(
            values["gpu_hours_cumulative"],
            values["elapsed_sec"] / 3600,
            rel_tol=0,
            abs_tol=1e-8,
        ):
            fail(f"{label}.gpu_hours_cumulative does not match elapsed_sec")
        previous_elapsed = values["elapsed_sec"]

    with (run_dir / "train_summary.csv").open(
        "rt", newline="", encoding="utf-8"
    ) as handle:
        summary_reader = csv.DictReader(handle)
        if tuple(summary_reader.fieldnames or ()) != TRAIN_SUMMARY_FIELDS:
            fail("planned pause train_summary header is not the exact current schema")
        summary_rows = list(summary_reader)
    if len(summary_rows) != stop_after_attempts:
        fail("planned pause train_summary row count does not match the pause target")
    for attempt, row in enumerate(summary_rows, start=1):
        if None in row:
            fail("planned pause train_summary contains extra cells")
        if _strict_csv_int(row["attempted_iteration"], "train_summary attempt") != attempt:
            fail("planned pause train_summary attempt sequence is not contiguous")
        if _strict_csv_int(row["processed_nimg"], "train_summary processed_nimg") != attempt * 128:
            fail("planned pause train_summary processed_nimg sequence mismatch")
        if row["successful_optimizer_steps"] != rows[attempt - 1]["successful_optimizer_steps"]:
            fail("planned pause train_summary successful-step count differs from telemetry")
        if row["step_skipped"] != rows[attempt - 1]["step_skipped"]:
            fail("planned pause train_summary AMP skip marker differs from telemetry")
        if row["schedule"] != "sigmoid" or row["stage"] != "0":
            fail("planned pause train_summary schedule identity mismatch")
        if _strict_csv_int(row["next_loop_cur_tick"], "train_summary next tick") != 1:
            fail("planned pause changed tick control outside a natural maintenance boundary")

    log_text = (run_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    marker = f"Planned pause after {stop_after_attempts} attempts; exiting."
    if marker not in log_text or "Traceback (most recent call last)" in log_text:
        fail("planned pause log lacks its clean marker or contains a traceback")
    if "Exiting..." in log_text:
        fail("planned pause log incorrectly claims full-budget completion")

    state_probe_code = r'''import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scripts import verify_q256_target_weight_arm as v
from training import reproducibility

run_dir = Path(sys.argv[1]).resolve()
arm = sys.argv[2]
seed = int(sys.argv[3])
attempts = int(sys.argv[4])
processed_nimg = int(sys.argv[5])
successful_steps = int(sys.argv[6])
elapsed_sec = float(sys.argv[7])
next_loop_tick = int(sys.argv[8])
expected_tick_start_nimg = int(sys.argv[9])

options = v.load_json(run_dir / "training_options.json", "training options")
options_info = v.validate_training_options(options, arm, seed, "smoke")
v.exact_value(options.get("stop_after_attempts"), attempts, "stop_after_attempts")
initial = v.load_json(run_dir / "initial_state_receipt_v1.json", "initial receipt")
initial_info = v.validate_initial_receipt(initial, arm, seed, options_info)

state = v.torch_load_trusted(run_dir / "training-state-latest.pt")
state = v.require_dict(state, "planned-pause training state")
required = (
    "reproducibility_schema", "net", "ema", "optimizer_state",
    "gradscaler_state", "rank_states", "factorial", "attempted_iteration",
    "successful_optimizer_steps", "cur_nimg", "cur_tick", "tick_start_nimg",
    "elapsed_sec", "loss_fn_state", "snapshot_grid_z", "snapshot_grid_c",
    "snapshot_grid_size", "trajectory_config", "trajectory_config_sha256",
)
missing = [name for name in required if name not in state]
if missing:
    v.fail(f"planned-pause self-contained state is missing: {missing}")
v.exact_value(
    state["reproducibility_schema"], reproducibility.TRAINING_STATE_SCHEMA,
    "reproducibility_schema",
)
v.validate_factorial(state["factorial"], arm, "training-state.factorial")
v.exact_value(state["attempted_iteration"], attempts, "attempted_iteration")
v.exact_value(
    state["successful_optimizer_steps"], successful_steps,
    "successful_optimizer_steps",
)
v.exact_value(state["cur_nimg"], processed_nimg, "cur_nimg")
v.exact_value(state["cur_tick"], next_loop_tick, "cur_tick")
v.exact_value(
    state["tick_start_nimg"], expected_tick_start_nimg, "tick_start_nimg"
)
if not math.isclose(float(state["elapsed_sec"]), elapsed_sec, rel_tol=0, abs_tol=5e-6):
    v.fail("training-state elapsed_sec does not match partial telemetry")
trajectory = v.require_dict(state["trajectory_config"], "trajectory_config")
v.exact_value(
    trajectory.get("schema"), reproducibility.TRAJECTORY_CONFIG_SCHEMA,
    "trajectory_config.schema",
)
trajectory_sha256 = v.require_sha256(
    state["trajectory_config_sha256"], "trajectory_config_sha256"
)
if reproducibility.state_sha256(trajectory) != trajectory_sha256:
    v.fail("training-state trajectory config hash mismatch")
v.exact_value(
    trajectory_sha256, initial_info["trajectory_config_sha256"],
    "initial/training-state trajectory config hash",
)
net_tensors = v.validate_module(state["net"], "training-state net")
ema_tensors = v.validate_module(state["ema"], "training-state EMA")
optimizer = v.require_dict(state["optimizer_state"], "optimizer state")
if not {"state", "param_groups"}.issubset(optimizer):
    v.fail("planned-pause optimizer state lacks state/param_groups")
v.validate_finite_tree(optimizer, "optimizer state")
scaler = v.require_dict(state["gradscaler_state"], "GradScaler state")
if not scaler:
    v.fail("planned-pause GradScaler state is empty")
v.validate_finite_tree(scaler, "GradScaler state")
ranks = v.require_list(state["rank_states"], "rank states")
if len(ranks) != 1:
    v.fail("planned-pause state must contain exactly one rank")
rank = v.require_dict(ranks[0], "rank state")
v.require_exact_keys(
    rank, ("rank", "world_size", "rng_state", "sampler_state"), "rank state"
)
v.exact_value(rank["rank"], 0, "rank")
v.exact_value(rank["world_size"], 1, "world_size")
v.validate_rng_state(rank["rng_state"])
v.validate_sampler_state(
    rank["sampler_state"], seed=seed, consumed_samples=processed_nimg
)
v.validate_finite_tree(rank["rng_state"], "RNG state")
loss_state = v.require_dict(state["loss_fn_state"], "loss_fn_state")
v.exact_value(loss_state.get("schedule_name"), "sigmoid", "schedule_name")
v.exact_value(loss_state.get("stage"), 0, "schedule stage")
v.exact_number(loss_state.get("ratio"), 255 / 256, "schedule ratio")
v.exact_value(loss_state.get("schedule"), {}, "schedule state")
grid_z = v.require_list(state["snapshot_grid_z"], "snapshot_grid_z")
grid_c = v.require_list(state["snapshot_grid_c"], "snapshot_grid_c")
if not grid_z or len(grid_z) != len(grid_c):
    v.fail("planned-pause preview tensors are empty or unpaired")
if any(not isinstance(value, torch.Tensor) for value in grid_z + grid_c):
    v.fail("planned-pause preview state contains a non-tensor")
v.validate_finite_tree(grid_z + grid_c, "preview state")
grid_size = state["snapshot_grid_size"]
if not (
    isinstance(grid_size, tuple) and len(grid_size) == 2
    and all(
        isinstance(value, (int, np.integer)) and int(value) > 0
        for value in grid_size
    )
):
    v.fail("planned-pause snapshot_grid_size is invalid")
state_info = {
    "state_ema_sha256": reproducibility.module_state_sha256(state["ema"]),
}
snapshot_info = v.validate_snapshot(
    run_dir / "network-snapshot-latest.pkl", arm, options_info, state_info
)
print(json.dumps({
    "status": "PASS",
    "net_tensors_checked": net_tensors,
    "ema_tensors_checked": ema_tensors,
    "snapshot_ema_tensors_checked": snapshot_info["ema_tensors_checked"],
    "component_sha256": {
        "net": reproducibility.module_state_sha256(state["net"]),
        "ema": state_info["state_ema_sha256"],
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "rng": reproducibility.state_sha256(rank["rng_state"]),
        "sampler": reproducibility.state_sha256(rank["sampler_state"]),
        "loss": reproducibility.state_sha256(state["loss_fn_state"]),
        "trajectory": trajectory_sha256,
        "control_state": reproducibility.state_sha256({
            "reproducibility_schema": state["reproducibility_schema"],
            "factorial": state["factorial"],
            "attempted_iteration": state["attempted_iteration"],
            "successful_optimizer_steps": state["successful_optimizer_steps"],
            "cur_nimg": state["cur_nimg"],
            "cur_tick": state["cur_tick"],
            "tick_start_nimg": state["tick_start_nimg"],
            "trajectory_config_sha256": trajectory_sha256,
        }),
    },
}, sort_keys=True))'''
    last = rows[-1]
    last_summary = summary_rows[-1]
    probe_output = checked_output(
        [
            *runtime_command,
            "-c",
            state_probe_code,
            str(run_dir),
            arm,
            str(seed),
            str(stop_after_attempts),
            str(expected_nimg),
            last["successful_optimizer_steps"],
            last["elapsed_sec"],
            last_summary["next_loop_cur_tick"],
            "128",
        ],
        cwd=REPO_ROOT,
        env=process_env,
    )
    try:
        state_probe = json.loads(probe_output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"planned-pause state probe returned invalid JSON: {exc}")
    if state_probe.get("status") != "PASS":
        fail(f"planned-pause state probe did not pass: {state_probe!r}")
    component_sha256 = state_probe.get("component_sha256")
    expected_components = {
        "net",
        "ema",
        "optimizer",
        "gradscaler",
        "rng",
        "sampler",
        "loss",
        "trajectory",
        "control_state",
    }
    if not isinstance(component_sha256, dict) or set(component_sha256) != expected_components:
        fail("planned-pause state probe returned an incomplete component hash set")
    for name, digest in component_sha256.items():
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            fail(f"planned-pause state probe returned an invalid {name} digest")

    artifacts = {}
    for name in required:
        path = run_dir / name
        artifacts[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "status": "PASS",
        "stop_after_attempts": stop_after_attempts,
        "attempted_iterations": len(rows),
        "successful_optimizer_steps": int(last["successful_optimizer_steps"]),
        "processed_nimg": int(last["processed_nimg"]),
        "amp_skip_attempts": [
            int(row["attempted_iteration"])
            for row in rows
            if row["step_skipped"] == "1"
        ],
        "state_probe": state_probe,
        "artifacts": artifacts,
    }


def validate_planned_pause_completion(
    run_dir: Path,
    *,
    launch_manifest: Mapping[str, object],
    stop_after_attempts: int,
) -> dict:
    """Bind exact-resume authorization to an immutable, deeply checked pause."""

    if stop_after_attempts != PLANNED_PAUSE_ATTEMPTS:
        fail("planned-pause completion must bind the frozen 16-attempt gate")
    completion_path = run_dir / "runner_completion.json"
    if (
        completion_path.is_symlink()
        or not completion_path.is_file()
        or completion_path.stat().st_size <= 0
    ):
        fail("planned exact resume requires the immutable runner_completion.json")
    completion = _load_json(completion_path, "planned-pause completion")
    exact = {
        "schema": RUNNER_COMPLETION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": PLANNED_PAUSE_STATUS,
        "returncode": 0,
        "launch_manifest": "launch_manifest.json",
        "runner_log": "runner.log",
        "full_arm_verifier_invoked": False,
        "resume_required": True,
    }
    for field, expected in exact.items():
        value = completion.get(field)
        if isinstance(value, bool) and field == "returncode":
            fail("planned-pause completion returncode must be integer zero")
        if value != expected:
            fail(
                f"planned-pause completion {field} mismatch: "
                f"{value!r} != {expected!r}"
            )
    for field in ("started_utc", "finished_utc"):
        value = completion.get(field)
        if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
            fail(f"planned-pause completion has invalid {field}: {value!r}")
    manifest_path = run_dir / "launch_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if completion.get("launch_manifest_sha256") != manifest_sha256:
        fail("planned-pause completion does not bind the immutable launch manifest")
    if launch_manifest.get("gate_control") != {
        "kind": "planned_exact_resume_pause",
        "stop_after_attempts": stop_after_attempts,
        "scientific_training_contract_unchanged": True,
    }:
        fail("planned-pause completion was requested for a different launch gate")
    launch_gpu = launch_manifest.get("gpu")
    if not isinstance(launch_gpu, dict):
        fail("planned-pause launch manifest has no GPU identity")
    launch_gpu_uuid = launch_gpu.get("uuid")
    if not isinstance(launch_gpu_uuid, str) or not launch_gpu_uuid.startswith("GPU-"):
        fail("planned-pause launch manifest has an invalid GPU UUID")
    validate_gpu_idle_record(
        completion.get("final_prelaunch_gpu_idle_check"),
        label="planned-pause final-prelaunch",
        expected_gpu_uuid=launch_gpu_uuid,
    )
    validate_gpu_monitor_record(
        completion.get("training_gpu_exclusivity_monitor"),
        label="planned-pause training",
        expected_gpu_uuid=launch_gpu_uuid,
    )
    validate_gpu_idle_record(
        completion.get("post_training_gpu_idle_check"),
        label="planned-pause post-training",
        expected_gpu_uuid=launch_gpu_uuid,
    )

    runner_log = run_dir / "runner.log"
    if runner_log.is_symlink() or not runner_log.is_file() or runner_log.stat().st_size <= 0:
        fail("planned-pause runner log is missing, empty, or a symlink")
    if completion.get("runner_log_sha256") != sha256_file(runner_log):
        fail("planned-pause runner log changed after the immutable completion receipt")

    postcheck = completion.get("planned_pause_verification")
    if not isinstance(postcheck, dict):
        fail("planned-pause completion lacks its deep verification report")
    expected_postcheck = {
        "status": "PASS",
        "stop_after_attempts": stop_after_attempts,
        "attempted_iterations": stop_after_attempts,
        "processed_nimg": stop_after_attempts * 128,
    }
    for field, expected in expected_postcheck.items():
        if postcheck.get(field) != expected:
            fail(f"planned-pause verification report mismatch for {field}")
    state_probe = postcheck.get("state_probe")
    if not isinstance(state_probe, dict) or state_probe.get("status") != "PASS":
        fail("planned-pause completion lacks a passing self-contained state probe")
    components = state_probe.get("component_sha256")
    expected_components = {
        "net",
        "ema",
        "optimizer",
        "gradscaler",
        "rng",
        "sampler",
        "loss",
        "trajectory",
        "control_state",
    }
    if not isinstance(components, dict) or set(components) != expected_components:
        fail("planned-pause completion has an incomplete state component hash set")
    for name, digest in components.items():
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            fail(f"planned-pause completion has an invalid {name} state digest")
    artifacts = postcheck.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(PLANNED_PAUSE_ARTIFACTS):
        fail("planned-pause verification report has an incomplete artifact binding set")
    for relative_name in PLANNED_PAUSE_ARTIFACTS:
        path = run_dir / relative_name
        binding = artifacts.get(relative_name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            fail(f"planned-pause bound artifact is missing, empty, or a symlink: {path}")
        if not _is_within(path.resolve(strict=True), run_dir.resolve(strict=True)):
            fail(f"planned-pause bound artifact escapes its run directory: {path}")
        if not isinstance(binding, dict) or set(binding) != {"sha256", "size_bytes"}:
            fail(f"planned-pause artifact binding is malformed: {relative_name}")
        digest = binding.get("sha256")
        size = binding.get("size_bytes")
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            fail(f"planned-pause artifact has invalid SHA256: {relative_name}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            fail(f"planned-pause artifact has invalid byte count: {relative_name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            fail(f"planned-pause artifact changed after verification: {path}")
    if any(
        field in completion
        for field in (
            "verification",
            "verifier_command_argv",
            "verifier_log",
            "verifier_returncode",
        )
    ):
        fail("planned-pause completion improperly claims a full-arm verifier run")
    return {
        "path": completion_path.name,
        "sha256": sha256_file(completion_path),
        "status": PLANNED_PAUSE_STATUS,
        "stop_after_attempts": stop_after_attempts,
        "artifact_bindings_sha256": canonical_sha256(artifacts),
    }


def _copy_file_exclusive(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        fail(f"immutable pause-evidence target already exists: {target}")
    try:
        source_handle = source.open("rb")
        target_fd = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
    except OSError as exc:
        fail(f"cannot create immutable pause evidence {target}: {exc}")
    try:
        with source_handle, os.fdopen(target_fd, "wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
    except Exception:
        # A partial exclusive file intentionally remains as fail-closed evidence;
        # exact-resume attempts are never silently retried in the same directory.
        raise


def preserve_planned_pause_evidence(
    run_dir: Path,
    planned_completion: Mapping[str, object],
) -> dict[str, object]:
    """Copy the 16-attempt pause state before latest artifacts are replaced."""

    completion_path = run_dir / "runner_completion.json"
    completion = _load_json(completion_path, "planned-pause completion")
    postcheck = completion.get("planned_pause_verification")
    if not isinstance(postcheck, dict) or not isinstance(
        postcheck.get("artifacts"), dict
    ):
        fail("planned-pause completion lacks artifact bindings for preservation")
    pause_bindings = postcheck["artifacts"]
    evidence_dir = run_dir / PLANNED_PAUSE_EVIDENCE_DIR
    try:
        evidence_dir.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"planned-pause evidence directory already exists: {evidence_dir}")

    sources = list(PLANNED_PAUSE_ARTIFACTS) + [
        "runner.log",
        "runner_completion.json",
    ]
    copied: dict[str, dict[str, object]] = {}
    for name in sources:
        source = run_dir / name
        if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
            fail(f"planned-pause evidence source is invalid: {source}")
        if name in pause_bindings:
            binding = pause_bindings[name]
            expected_sha = binding.get("sha256") if isinstance(binding, dict) else None
            expected_size = (
                binding.get("size_bytes") if isinstance(binding, dict) else None
            )
        elif name == "runner.log":
            expected_sha = completion.get("runner_log_sha256")
            expected_size = source.stat().st_size
        else:
            expected_sha = planned_completion.get("sha256")
            expected_size = source.stat().st_size
        if (
            not isinstance(expected_sha, str)
            or not _SHA_RE.fullmatch(expected_sha)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or source.stat().st_size != expected_size
            or sha256_file(source) != expected_sha
        ):
            fail(f"planned-pause evidence source binding is stale: {source}")
        target = evidence_dir / name
        _copy_file_exclusive(source, target)
        if target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
            fail(f"planned-pause evidence copy verification failed: {target}")
        copied[name] = {
            "source_path": name,
            "evidence_path": str(target.relative_to(run_dir)),
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }

    manifest = {
        "schema": PLANNED_PAUSE_EVIDENCE_SCHEMA,
        "created_utc": utc_now(),
        "run_directory": str(run_dir.resolve(strict=True)),
        "planned_pause_completion_sha256": planned_completion["sha256"],
        "files": copied,
    }
    manifest_path = evidence_dir / PLANNED_PAUSE_EVIDENCE_MANIFEST
    atomic_json_exclusive(manifest_path, manifest)
    dir_fd = os.open(evidence_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {
        "schema": PLANNED_PAUSE_EVIDENCE_SCHEMA,
        "directory": PLANNED_PAUSE_EVIDENCE_DIR,
        "manifest_path": str(manifest_path.relative_to(run_dir)),
        "manifest_sha256": sha256_file(manifest_path),
        "planned_pause_completion_sha256": planned_completion["sha256"],
        "resume_state_path": str(
            (evidence_dir / "training-state-latest.pt").relative_to(run_dir)
        ),
    }


def validate_preserved_planned_pause_evidence(
    run_dir: Path,
    evidence: object,
    *,
    arm: str,
    seed: int,
    runtime_command: Sequence[str] | None,
    process_env: Mapping[str, str] | None,
) -> dict[str, object]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema",
        "directory",
        "manifest_path",
        "manifest_sha256",
        "planned_pause_completion_sha256",
        "resume_state_path",
    }:
        fail("exact-resume pause evidence binding is malformed")
    if evidence.get("schema") != PLANNED_PAUSE_EVIDENCE_SCHEMA:
        fail("exact-resume pause evidence has the wrong schema")
    if evidence.get("directory") != PLANNED_PAUSE_EVIDENCE_DIR:
        fail("exact-resume pause evidence uses an unexpected directory")
    manifest_path = _resolve_run_contained_file(
        run_dir,
        Path(str(evidence.get("manifest_path"))),
        "exact-resume pause evidence manifest",
    )
    declared_manifest_sha = evidence.get("manifest_sha256")
    if (
        not isinstance(declared_manifest_sha, str)
        or not _SHA_RE.fullmatch(declared_manifest_sha)
        or sha256_file(manifest_path) != declared_manifest_sha
    ):
        fail("exact-resume pause evidence manifest hash is stale")
    manifest = _load_json(manifest_path, "exact-resume pause evidence manifest")
    if (
        manifest.get("schema") != PLANNED_PAUSE_EVIDENCE_SCHEMA
        or manifest.get("run_directory") != str(run_dir.resolve(strict=True))
        or manifest.get("planned_pause_completion_sha256")
        != evidence.get("planned_pause_completion_sha256")
    ):
        fail("exact-resume pause evidence manifest identity differs")
    expected_files = set(PLANNED_PAUSE_ARTIFACTS) | {
        "runner.log",
        "runner_completion.json",
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        fail("exact-resume pause evidence file set is incomplete")
    evidence_dir = manifest_path.parent
    if evidence_dir.name != PLANNED_PAUSE_EVIDENCE_DIR:
        fail("exact-resume pause evidence manifest is outside its fixed directory")
    for name in sorted(expected_files):
        binding = files.get(name)
        if not isinstance(binding, dict) or set(binding) != {
            "source_path",
            "evidence_path",
            "sha256",
            "size_bytes",
        }:
            fail(f"exact-resume pause evidence binding is malformed: {name}")
        if binding.get("source_path") != name:
            fail(f"exact-resume pause evidence source name differs: {name}")
        path = _resolve_run_contained_file(
            run_dir,
            Path(str(binding.get("evidence_path"))),
            f"exact-resume pause evidence {name}",
        )
        if path.parent != evidence_dir or path.name != name:
            fail(f"exact-resume pause evidence path differs: {name}")
        digest = binding.get("sha256")
        size = binding.get("size_bytes")
        if (
            not isinstance(digest, str)
            or not _SHA_RE.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            fail(f"exact-resume pause evidence file changed: {name}")
    copied_manifest = _load_json(
        evidence_dir / "launch_manifest.json", "preserved pause launch manifest"
    )
    pause_record = validate_planned_pause_completion(
        evidence_dir,
        launch_manifest=copied_manifest,
        stop_after_attempts=PLANNED_PAUSE_ATTEMPTS,
    )
    if pause_record.get("sha256") != evidence.get(
        "planned_pause_completion_sha256"
    ):
        fail("preserved pause completion differs from its resume binding")
    if not runtime_command or process_env is None:
        fail("preserved pause evidence requires canonical runtime revalidation")
    fresh_postcheck = verify_planned_pause_run(
        evidence_dir,
        arm=arm,
        seed=seed,
        stop_after_attempts=PLANNED_PAUSE_ATTEMPTS,
        runtime_command=runtime_command,
        process_env=process_env,
    )
    copied_completion = _load_json(
        evidence_dir / "runner_completion.json",
        "preserved planned-pause completion",
    )
    if copied_completion.get("planned_pause_verification") != fresh_postcheck:
        fail(
            "fresh planned-pause state revalidation differs from the preserved "
            "completion report"
        )
    expected_resume_state = evidence_dir / "training-state-latest.pt"
    if evidence.get("resume_state_path") != str(expected_resume_state.relative_to(run_dir)):
        fail("exact-resume pause evidence binds another resume state")
    return {
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": declared_manifest_sha,
        "planned_pause_completion_sha256": pause_record["sha256"],
        "resume_state": str(expected_resume_state),
        "resume_state_sha256": sha256_file(expected_resume_state),
        "fresh_postcheck_sha256": canonical_sha256(fresh_postcheck),
        "files_sha256": {
            name: files[name]["sha256"] for name in sorted(expected_files)
        },
    }


def build_verifier_command(
    *,
    python_bin: str | Path,
    run_dir: Path,
    phase: str,
    arm: str,
    seed: int,
    expected_skip_attempts: str | None = None,
    runtime_command: Sequence[str] | None = None,
) -> list[str]:
    command = [
        *(list(runtime_command) if runtime_command is not None else [str(python_bin)]),
        str(REPO_ROOT / "scripts" / "verify_q256_target_weight_arm.py"),
        "--run-dir",
        str(run_dir),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--mode",
        phase,
    ]
    if expected_skip_attempts is not None:
        command += ["--expected-skip-attempts", expected_skip_attempts]
    return command


def validate_existing_verifier_receipts(
    run_dir: Path,
    *,
    phase: str,
    arm: str,
    seed: int,
    expected_skip_attempts: list[int] | None = None,
) -> dict | None:
    """Validate immutable PASS receipts without creating or replacing them."""

    expected_skip_attempts = (
        None if expected_skip_attempts is None else list(expected_skip_attempts)
    )
    validation_path = run_dir / VALIDATION_FILENAME
    hashes_path = run_dir / HASH_RECEIPT_FILENAME
    exists = (validation_path.exists(), hashes_path.exists())
    if exists == (False, False):
        return None
    if exists != (True, True):
        fail(f"partial verifier receipt set requires audit: {validation_path}, {hashes_path}")
    validation = _load_json(validation_path, "arm validation receipt")
    hashes = _load_json(hashes_path, "arm artifact-hash receipt")
    expected_common = {"status": "passed", "mode": phase, "arm": arm, "seed": seed}
    if validation.get("schema") != VALIDATION_SCHEMA:
        fail(f"invalid validation receipt schema: {validation.get('schema')!r}")
    if hashes.get("schema") != HASH_RECEIPT_SCHEMA:
        fail(f"invalid artifact-hash receipt schema: {hashes.get('schema')!r}")
    canonical_run_dir = str(run_dir.resolve(strict=True))
    if (
        validation.get("run_dir") != canonical_run_dir
        or hashes.get("run_dir") != canonical_run_dir
    ):
        fail("verifier PASS receipts are bound to another run directory")
    for key, expected in expected_common.items():
        if validation.get(key) != expected or hashes.get(key) != expected:
            fail(f"verifier PASS receipt identity mismatch for {key!r}")
    observed_skip_attempts = validate_amp_skip_signature(
        validation.get("amp_skip_attempts"),
        phase=phase,
        label="immutable verifier receipt",
    )
    if (
        expected_skip_attempts is not None
        and observed_skip_attempts != expected_skip_attempts
    ):
        fail(
            "immutable verifier receipt AMP skip signature differs from "
            "the authorized signature"
        )
    expected_value_was_enforced = expected_skip_attempts is not None
    if (
        validation.get("amp_skip_signature_expected_value_enforced")
        is not expected_value_was_enforced
    ):
        fail("immutable verifier receipt AMP skip enforcement mode differs")
    if validation.get("amp_skip_policy") != AMP_SKIP_POLICY:
        fail("immutable verifier receipt AMP skip policy differs")
    successful_optimizer_steps = validation.get("successful_optimizer_steps")
    expected_successful_steps = (
        PHASES[phase]["expected_attempts"] - len(observed_skip_attempts)
    )
    if successful_optimizer_steps != expected_successful_steps:
        fail(
            "immutable verifier receipt has an invalid successful "
            "optimizer-step count"
        )
    initial_common_state_sha256 = validation.get("initial_common_state_sha256")
    if (
        not isinstance(initial_common_state_sha256, str)
        or not _SHA_RE.fullmatch(initial_common_state_sha256)
    ):
        fail("immutable verifier receipt lacks the initial common-state SHA256")
    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        fail("artifact-hash receipt has no bound artifacts")
    required_artifacts = set(CORE_ARM_ARTIFACTS) | {VALIDATION_FILENAME}
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    if missing_artifacts:
        fail(
            "artifact-hash receipt is missing required verifier artifacts: "
            f"{missing_artifacts}"
        )
    for raw_relative, binding in artifacts.items():
        if not isinstance(raw_relative, str) or not raw_relative:
            fail("artifact-hash receipt contains an invalid path")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"artifact-hash receipt path escapes the run: {raw_relative!r}")
        path = run_dir / relative
        if path.is_symlink() or not path.is_file():
            fail(f"PASS-bound artifact is missing or a symlink: {path}")
        resolved = path.resolve(strict=True)
        if not _is_within(resolved, run_dir.resolve()):
            fail(f"PASS-bound artifact resolves outside the run: {path}")
        if not isinstance(binding, dict):
            fail(f"invalid artifact binding for {raw_relative!r}")
        expected_sha = binding.get("sha256")
        expected_size = binding.get("bytes")
        if not isinstance(expected_sha, str) or not _SHA_RE.fullmatch(expected_sha):
            fail(f"invalid artifact SHA256 for {raw_relative!r}")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            fail(f"invalid artifact byte count for {raw_relative!r}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            fail(f"PASS-bound artifact changed after verification: {path}")
    return {
        "validation_receipt": validation_path.name,
        "validation_receipt_sha256": sha256_file(validation_path),
        "artifact_hash_receipt": hashes_path.name,
        "artifact_hash_receipt_sha256": sha256_file(hashes_path),
        "amp_skip_attempts": validation.get("amp_skip_attempts"),
        "successful_optimizer_steps": successful_optimizer_steps,
        "initial_common_state_sha256": initial_common_state_sha256,
    }


def deep_revalidate_existing_arm(
    run_dir: Path,
    *,
    phase: str,
    arm: str,
    seed: int,
    expected_skip_attempts: list[int] | None,
    runtime_command: Sequence[str] | None,
    process_env: Mapping[str, str] | None,
    stop_event: threading.Event | None = None,
    register_process: Callable[[subprocess.Popen[bytes]], None] | None = None,
    unregister_process: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> dict[str, object]:
    """Freshly execute the production verifier and compare its full report.

    Hash-self-consistent JSON is not sufficient evidence that tensors, state,
    telemetry, and the launch contract are valid.  This check-only execution is
    therefore mandatory before a completed run may satisfy a later gate or be
    skipped by a resumed matrix.
    """

    if not runtime_command or process_env is None:
        fail("fresh arm revalidation requires the canonical runtime command and environment")
    command = [
        *runtime_command,
        str(REPO_ROOT / "scripts" / "verify_q256_target_weight_arm.py"),
        "--run-dir",
        str(run_dir),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--mode",
        phase,
        "--check-only",
    ]
    if expected_skip_attempts is not None:
        command += [
            "--expected-skip-attempts",
            json.dumps(expected_skip_attempts, separators=(",", ":")),
        ]
    returncode, output = _run_deep_revalidation_command(
        command,
        process_env=process_env,
        timeout_seconds=600.0,
        label="fresh arm revalidation",
        stop_event=stop_event,
        register_process=register_process,
        unregister_process=unregister_process,
    )
    if returncode != 0:
        fail(
            "fresh arm revalidation failed for "
            f"phase={phase} seed={seed} arm={arm}: {output.strip()}"
        )
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"fresh arm revalidation returned invalid JSON: {exc}")
    immutable = _load_json(
        run_dir / VALIDATION_FILENAME, "immutable arm validation receipt"
    )
    if report != immutable:
        fail(
            "fresh arm revalidation report differs from the immutable PASS "
            f"receipt for phase={phase} seed={seed} arm={arm}"
        )
    return {
        "status": "PASS",
        "command_argv": command,
        "report_sha256": canonical_sha256(report),
    }


def deep_revalidate_smoke_matrix(
    payload: Mapping[str, object],
    *,
    runtime_command: Sequence[str] | None,
    process_env: Mapping[str, str] | None,
    stop_event: threading.Event | None = None,
    register_process: Callable[[subprocess.Popen[bytes]], None] | None = None,
    unregister_process: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> dict[str, object]:
    """Freshly rerun the full cross-arm and exact-resume production verifier."""

    if not runtime_command or process_env is None:
        fail("fresh smoke-matrix revalidation requires canonical runtime scope")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        fail("fresh smoke-matrix revalidation lacks exact A/B/C/D bindings")
    command = [
        *runtime_command,
        str(REPO_ROOT / "scripts" / "verify_q256_target_weight_smoke_matrix.py"),
    ]
    for arm in ARMS:
        binding = arms.get(arm)
        run_dir = binding.get("run_dir") if isinstance(binding, dict) else None
        if not isinstance(run_dir, str) or not run_dir:
            fail(f"fresh smoke-matrix revalidation lacks arm {arm} run directory")
        command += [f"--arm-{arm.lower()}-run-dir", run_dir]
    command += ["--seed", "3", "--check-only"]
    exact_resume = payload.get("exact_resume")
    if exact_resume is not None:
        if not isinstance(exact_resume, dict):
            fail("fresh smoke-matrix revalidation has malformed exact_resume")
        for key in ("uninterrupted_run_dir", "resumed_run_dir", "arm"):
            if not isinstance(exact_resume.get(key), str) or not exact_resume.get(key):
                fail(f"fresh smoke-matrix revalidation lacks exact_resume.{key}")
        command += [
            "--uninterrupted-run-dir",
            exact_resume["uninterrupted_run_dir"],
            "--resumed-run-dir",
            exact_resume["resumed_run_dir"],
            "--resume-arm",
            exact_resume["arm"],
        ]
    returncode, output = _run_deep_revalidation_command(
        command,
        process_env=process_env,
        timeout_seconds=1800.0,
        label="fresh smoke-matrix revalidation",
        stop_event=stop_event,
        register_process=register_process,
        unregister_process=unregister_process,
    )
    if returncode != 0:
        fail(f"fresh smoke-matrix revalidation failed: {output.strip()}")
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"fresh smoke-matrix revalidation returned invalid JSON: {exc}")
    if report != payload:
        fail("fresh smoke-matrix revalidation differs from immutable gate receipt")
    return {
        "status": "PASS",
        "command_argv": command,
        "report_sha256": canonical_sha256(report),
    }


def _run_deep_revalidation_command(
    command: Sequence[str],
    *,
    process_env: Mapping[str, str],
    timeout_seconds: float,
    label: str,
    stop_event: threading.Event | None,
    register_process: Callable[[subprocess.Popen[bytes]], None] | None,
    unregister_process: Callable[[subprocess.Popen[bytes]], None] | None,
) -> tuple[int, str]:
    """Run a check-only verifier without allowing an untracked child to survive.

    Matrix workers provide registration callbacks and a stop event.  In that
    mode the verifier owns a separate process group, is visible to the matrix
    signal handler, and is synchronously reaped before this function returns.
    Pre-thread authorization checks keep the caller's process group so normal
    terminal signals continue to reach both parent and verifier.
    """

    managed = stop_event is not None or register_process is not None
    if managed and (stop_event is None or register_process is None or unregister_process is None):
        fail(f"{label} has an incomplete managed-process contract")
    process: subprocess.Popen[bytes] | None = None
    registered = False
    cleanup_confirmed = False
    raw_output = b""
    try:
        if stop_event is not None and stop_event.is_set():
            fail(f"{label} refused to start after matrix audit stop was requested")
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=dict(process_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=managed,
        )
        if register_process is not None:
            registered = True
            register_process(process)
        deadline = time.monotonic() + timeout_seconds
        while True:
            if stop_event is not None and stop_event.is_set():
                stop_and_reap_process_group(process, label=label)
                cleanup_confirmed = True
                raw_output, _ = process.communicate(timeout=1.0)
                fail(f"{label} stopped because matrix audit stop was requested")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if managed:
                    stop_and_reap_process_group(process, label=label)
                    cleanup_confirmed = True
                    raw_output, _ = process.communicate()
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    process.communicate()
                fail(f"{label} timed out after {timeout_seconds:.0f} seconds")
            try:
                raw_output, _ = process.communicate(timeout=min(1.0, remaining))
                if managed and process_group_exists(process.pid):
                    raise ProcessCleanupError(
                        f"{label} leader exited but its process group remains"
                    )
                cleanup_confirmed = True
                break
            except subprocess.TimeoutExpired:
                continue
    except OSError as exc:
        fail(f"{label} could not execute: {exc}")
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    if managed:
                        stop_and_reap_process_group(process, label=label)
                        cleanup_confirmed = True
                    else:
                        process.terminate()
                        try:
                            process.wait(timeout=10.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2.0)
                        cleanup_confirmed = True
                elif managed and not cleanup_confirmed:
                    if process_group_exists(process.pid):
                        raise ProcessCleanupError(
                            f"{label} exited without proving its process group gone"
                        )
                    cleanup_confirmed = True
                elif not managed:
                    cleanup_confirmed = True
                if process.stdout is not None and not process.stdout.closed:
                    trailing_output, _ = process.communicate()
                    if not raw_output:
                        raw_output = trailing_output or b""
            finally:
                if (
                    registered
                    and unregister_process is not None
                    and cleanup_confirmed
                ):
                    unregister_process(process)
    output = (raw_output or b"").decode("utf-8", errors="replace")
    assert process is not None and process.returncode is not None
    return process.returncode, output


def default_outdir(runs_root: Path, phase: str, arm: str, seed: int) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = uuid.uuid4().hex[:10]
    return runs_root / phase / f"seed{seed}" / f"arm{arm}-{stamp}-{nonce}"


def run_arm(args: argparse.Namespace) -> int:
    validate_arm_seed_phase(args.phase, args.arm, args.seed)
    stop_after_attempts = args.stop_after_attempts
    if stop_after_attempts is not None:
        if args.phase != "smoke" or args.resume is not None:
            fail("--stop-after-attempts is gate-only, smoke-only, and fresh-only")
        if stop_after_attempts != PLANNED_PAUSE_ATTEMPTS:
            fail("--stop-after-attempts must equal the frozen gate target 16")
    expected_skip_values = parse_expected_skip_attempts(
        args.expected_skip_attempts, args.phase
    )
    expected_skip_argument = (
        json.dumps(expected_skip_values, separators=(",", ":"))
        if expected_skip_values is not None
        else None
    )
    runs_root = validate_runs_root(args.runs_root)
    storage = storage_preflight(runs_root, args.phase)
    data_path = args.data.expanduser()
    transfer_path = args.transfer.expanduser()
    python_bin = str(args.python_bin)
    runtime_command, runtime_base = runtime_prefix(args.runtime_sandbox, python_bin)
    sandbox = runtime_sandbox_fingerprint(runtime_base)
    source = source_snapshot(require_clean=True)
    dataset = verify_asset(data_path, EXPECTED_DATASET_SHA256, "canonical dataset")
    transfer = verify_asset(transfer_path, EXPECTED_TRANSFER_SHA256, "authoritative transfer")
    process_env = build_process_environment(args.gpu, args.master_port)

    resolved_gpu = query_gpu(args.gpu)
    with gpu_lock(str(resolved_gpu["uuid"]), args.lock_root) as lock_record:
        gpu = query_gpu(args.gpu)
        if gpu["uuid"] != resolved_gpu["uuid"]:
            fail("GPU selector changed identity while acquiring the canonical UUID lock")
        gpu_idle_checks = [assert_gpu_idle(gpu)]
        runtime = runtime_environment(runtime_command, process_env)
        gpu_idle_checks.append(assert_gpu_idle(gpu))

        if args.resume is None:
            if args.authorization_receipt is None:
                fail("fresh launch requires --authorization-receipt")
            authorization = validate_authorization(
                args.authorization_receipt,
                phase=args.phase,
                arm=args.arm,
                seed=args.seed,
                source=source,
                dataset=dataset,
                transfer=transfer,
                runtime_sandbox=sandbox,
                runtime=runtime,
                expected_skip_attempts=expected_skip_values,
                revalidation_command=runtime_command,
                revalidation_env=process_env,
            )
            requested = args.outdir or default_outdir(runs_root, args.phase, args.arm, args.seed)
            run_dir = create_fresh_run_dir(requested, runs_root)
            internal_auth = copy_authorization_into_run(run_dir, authorization)
            original_manifest = None
            planned_pause_completion = None
            effective_resume_state = None
        else:
            (
                run_dir,
                original_manifest,
                internal_auth,
                planned_pause_completion,
                effective_resume_state,
            ) = validate_resume(
                args.resume,
                runs_root=runs_root,
                phase=args.phase,
                arm=args.arm,
                seed=args.seed,
                source=source,
                dataset=dataset,
                transfer=transfer,
                gpu=gpu,
                runtime_sandbox=sandbox,
                runtime=runtime,
                expected_skip_attempts=expected_skip_values,
                outdir_override=args.outdir,
                supplied_authorization=args.authorization_receipt,
                revalidation_command=runtime_command,
                revalidation_env=process_env,
            )

        command = build_training_command(
            python_bin=python_bin,
            data=Path(dataset["resolved_path"]),
            transfer=Path(transfer["resolved_path"]),
            outdir=run_dir,
            phase=args.phase,
            arm=args.arm,
            seed=args.seed,
            resume=effective_resume_state,
            runtime_command=runtime_command,
            stop_after_attempts=stop_after_attempts,
        )
        manifest = {
            "schema": LAUNCH_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "created_utc": utc_now(),
            "launch_kind": "resume" if args.resume else "fresh_transfer",
            "status": "authorized_to_start",
            "run_directory": str(run_dir),
            "source": source,
            "preregistration": preregistration_record(),
            "assets": {"dataset": dataset, "transfer": transfer},
            "storage": storage,
            "authorization": internal_auth,
            "training": training_contract(args.phase, args.arm, args.seed),
            "gate_control": {
                "kind": (
                    "planned_exact_resume_pause"
                    if stop_after_attempts is not None
                    else "none"
                ),
                "stop_after_attempts": stop_after_attempts,
                "scientific_training_contract_unchanged": True,
            },
            "gpu": gpu,
            "gpu_lock": lock_record,
            "gpu_exclusivity": {
                "policy": "canonical-uuid-flock-plus-zero-compute-processes",
                "pre_manifest_checks": gpu_idle_checks,
            },
            "runtime_sandbox": sandbox,
            "runtime": runtime,
            "process_environment": {
                key: process_env[key]
                for key in (
                    "CUDA_DEVICE_ORDER",
                    "CUDA_VISIBLE_DEVICES",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "PYTHONUNBUFFERED",
                    "PYTHONDONTWRITEBYTECODE",
                )
            },
            "host": host_environment(),
            "exact_command_argv": command,
            "exact_command_shell": shlex.join(command),
            "resume_state": (
                str(effective_resume_state)
                if effective_resume_state is not None
                else None
            ),
            "resume_state_sha256": (
                sha256_file(effective_resume_state)
                if effective_resume_state is not None
                else None
            ),
            "validated_planned_pause_completion": planned_pause_completion,
            "original_gate_control": (
                original_manifest.get("gate_control") if original_manifest else None
            ),
            "original_launch_manifest_sha256": (
                sha256_file(run_dir / "launch_manifest.json") if original_manifest else None
            ),
            "post_training_verifier": {
                "path": "scripts/verify_q256_target_weight_arm.py",
                "expected_skip_attempts": expected_skip_values,
                "receipts_must_be_new_after_complete_budget": True,
                "deferred_for_planned_pause": stop_after_attempts is not None,
            },
        }
        if args.resume is None:
            manifest_path = run_dir / "launch_manifest.json"
            log_path = run_dir / "runner.log"
            completion_path = run_dir / "runner_completion.json"
        else:
            manifest_path = next_unique_name(run_dir, "resume_launch_manifest", ".json")
            token = manifest_path.stem.removeprefix("resume_launch_manifest-")
            log_path = run_dir / f"runner-resume-{token}.log"
            completion_path = run_dir / f"runner-completion-resume-{token}.json"
        atomic_json_exclusive(manifest_path, manifest)

        print(f"[q256-target-weight] run_dir={run_dir}", flush=True)
        print(f"[q256-target-weight] gpu={gpu['physical_index']} uuid={gpu['uuid']}", flush=True)
        print(f"[q256-target-weight] command={shlex.join(command)}", flush=True)
        # Close the provenance TOCTOU window as tightly as practical.  Any
        # source, asset, authorization, or writable-sandbox drift stops before
        # the first optimizer attempt.
        try:
            final_source = source_snapshot(require_clean=True)
            if (
                final_source["git_head"] != source["git_head"]
                or final_source["content_sha256"] != source["content_sha256"]
            ):
                fail("source changed between manifest creation and launch")
            final_dataset = verify_asset(data_path, EXPECTED_DATASET_SHA256, "canonical dataset")
            final_transfer = verify_asset(transfer_path, EXPECTED_TRANSFER_SHA256, "authoritative transfer")
            if final_dataset["sha256"] != dataset["sha256"] or final_transfer["sha256"] != transfer["sha256"]:
                fail("asset changed between manifest creation and launch")
            final_sandbox = runtime_sandbox_fingerprint(runtime_base)
            for key in ("sandbox_tree_metadata_sha256", "critical_files_sha256"):
                if final_sandbox[key] != sandbox[key]:
                    fail(f"runtime sandbox changed between probe and launch: {key}")
            verify_internal_authorization(
                run_dir,
                manifest,
                expected_phase=args.phase,
                revalidation_command=runtime_command,
                revalidation_env=process_env,
            )
            if planned_pause_completion is not None:
                preserved = validate_preserved_planned_pause_evidence(
                    run_dir,
                    planned_pause_completion.get("evidence"),
                    arm=args.arm,
                    seed=args.seed,
                    runtime_command=runtime_command,
                    process_env=process_env,
                )
                if (
                    manifest.get("resume_state") != preserved["resume_state"]
                    or manifest.get("resume_state_sha256")
                    != preserved["resume_state_sha256"]
                ):
                    fail("resume manifest state differs from preserved pause evidence")
            storage_preflight(runs_root, args.phase)
            final_gpu_idle_check = assert_gpu_idle(gpu)
            assert_master_port_available(args.master_port)
        except LaunchError as exc:
            atomic_json_exclusive(
                completion_path,
                {
                    "schema": RUNNER_COMPLETION_SCHEMA,
                    "experiment_id": EXPERIMENT_ID,
                    "finished_utc": utc_now(),
                    "status": "FAILED_FINAL_PREFLIGHT_NO_TRAINING_STARTED",
                    "error": str(exc),
                    "launch_manifest": manifest_path.name,
                    "launch_manifest_sha256": sha256_file(manifest_path),
                },
            )
            raise
        started = utc_now()
        training_gpu_monitor: dict[str, object] = {}
        returncode = stream_process(
            command,
            cwd=REPO_ROOT,
            env=process_env,
            log_path=log_path,
            monitored_gpu_uuid=str(gpu["uuid"]),
            gpu_monitor_record=training_gpu_monitor,
        )
        completion: dict[str, object] = {
            "schema": RUNNER_COMPLETION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": returncode,
            "launch_manifest": manifest_path.name,
            "launch_manifest_sha256": sha256_file(manifest_path),
            "runner_log": log_path.name,
            "runner_log_sha256": sha256_file(log_path),
            "final_prelaunch_gpu_idle_check": final_gpu_idle_check,
            "training_gpu_exclusivity_monitor": training_gpu_monitor,
        }
        if training_gpu_monitor.get("status") != "PASS":
            completion["status"] = "STOPPED_GPU_EXCLUSIVITY_LOST"
            atomic_json_exclusive(completion_path, completion)
            fail(
                "selected GPU lost exclusivity during training; our process group "
                f"was stopped and evidence was written to {completion_path}"
            )
        if returncode != 0:
            completion["status"] = "FAILED_PROCESS"
            atomic_json_exclusive(completion_path, completion)
            fail(f"training process exited with code {returncode}; inspect {log_path}")
        try:
            completion["post_training_gpu_idle_check"] = assert_gpu_idle(gpu)
        except LaunchError as exc:
            completion["status"] = "STOPPED_POST_TRAINING_GPU_EXCLUSIVITY_LOST"
            completion["error"] = str(exc)
            atomic_json_exclusive(completion_path, completion)
            raise
        if stop_after_attempts is not None:
            try:
                completion["planned_pause_verification"] = verify_planned_pause_run(
                    run_dir,
                    arm=args.arm,
                    seed=args.seed,
                    stop_after_attempts=stop_after_attempts,
                    runtime_command=runtime_command,
                    process_env=process_env,
                )
            except Exception as exc:
                error = (
                    str(exc)
                    if isinstance(exc, LaunchError)
                    else "planned-pause postcheck crashed fail-closed: "
                    f"{type(exc).__name__}: {exc}"
                )
                completion["status"] = "FAILED_PLANNED_PAUSE_POSTCONDITION"
                completion["error"] = error
                completion["full_arm_verifier_invoked"] = False
                atomic_json_exclusive(completion_path, completion)
                if isinstance(exc, LaunchError):
                    raise
                raise LaunchError(error) from exc
            completion["status"] = PLANNED_PAUSE_STATUS
            completion["resume_required"] = True
            completion["full_arm_verifier_invoked"] = False
            atomic_json_exclusive(completion_path, completion)
            print(
                "[q256-target-weight] PLANNED_PAUSE_PASS "
                f"resume={run_dir / 'training-state-latest.pt'} "
                f"receipt={completion_path}",
                flush=True,
            )
            return 0
        try:
            completion["launcher_postcheck"] = verify_completed_run(
                run_dir, args.phase, args.arm
            )
        except LaunchError as exc:
            completion["status"] = "FAILED_POSTCONDITION"
            completion["error"] = str(exc)
            atomic_json_exclusive(completion_path, completion)
            raise

        verifier_command = build_verifier_command(
            python_bin=python_bin,
            run_dir=run_dir,
            phase=args.phase,
            arm=args.arm,
            seed=args.seed,
            expected_skip_attempts=expected_skip_argument,
            runtime_command=runtime_command,
        )
        verifier_log = (
            run_dir / "arm_verifier.log"
            if args.resume is None
            else next_unique_name(run_dir, "arm-verifier-resume", ".log")
        )
        print(f"[q256-target-weight] verifier={shlex.join(verifier_command)}", flush=True)
        verifier_gpu_monitor: dict[str, object] = {}
        verifier_returncode = stream_process(
            verifier_command,
            cwd=REPO_ROOT,
            env=process_env,
            log_path=verifier_log,
            monitored_gpu_uuid=str(gpu["uuid"]),
            gpu_monitor_record=verifier_gpu_monitor,
        )
        completion.update(
            {
                "verifier_command_argv": verifier_command,
                "verifier_log": verifier_log.name,
                "verifier_log_sha256": sha256_file(verifier_log),
                "verifier_returncode": verifier_returncode,
                "verifier_gpu_exclusivity_monitor": verifier_gpu_monitor,
            }
        )
        if verifier_gpu_monitor.get("status") != "PASS":
            completion["status"] = "STOPPED_VERIFIER_GPU_EXCLUSIVITY_LOST"
            atomic_json_exclusive(completion_path, completion)
            fail(
                "selected GPU lost exclusivity during verification; our verifier "
                f"was stopped and evidence was written to {completion_path}"
            )
        if verifier_returncode != 0:
            completion["status"] = "FAILED_VERIFIER"
            atomic_json_exclusive(completion_path, completion)
            fail(f"arm verifier failed; inspect {verifier_log}")
        try:
            verifier_receipts = validate_existing_verifier_receipts(
                run_dir,
                phase=args.phase,
                arm=args.arm,
                seed=args.seed,
                expected_skip_attempts=expected_skip_values,
            )
            if verifier_receipts is None:
                fail("arm verifier exited zero without immutable PASS receipts")
            completion["verification"] = verifier_receipts
        except LaunchError as exc:
            completion["status"] = "FAILED_VERIFIER_RECEIPT"
            completion["error"] = str(exc)
            atomic_json_exclusive(completion_path, completion)
            raise
        try:
            completion["post_verifier_gpu_idle_check"] = assert_gpu_idle(gpu)
        except LaunchError as exc:
            completion["status"] = "STOPPED_POST_VERIFIER_GPU_EXCLUSIVITY_LOST"
            completion["error"] = str(exc)
            atomic_json_exclusive(completion_path, completion)
            raise
        completion["status"] = "PASS"
        completion["finished_utc"] = utc_now()
        atomic_json_exclusive(completion_path, completion)
        print(f"[q256-target-weight] PASS receipt={completion_path}", flush=True)
        return 0


@dataclass(frozen=True)
class MatrixJob:
    seed: int
    arm: str
    gpu: str
    master_port: int
    outdir: Path | None
    resume: Path | None
    command: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "arm": self.arm,
            "gpu": self.gpu,
            "master_port": self.master_port,
            "outdir": str(self.outdir) if self.outdir else None,
            "resume": str(self.resume) if self.resume else None,
            "command_argv": list(self.command),
            "command_shell": shlex.join(self.command),
        }


def validate_gpu_monitor_record(
    monitor: object,
    *,
    label: str,
    expected_gpu_uuid: str | None = None,
) -> str:
    if not isinstance(monitor, dict) or monitor.get("status") != "PASS":
        fail(f"{label} GPU monitor is not passing")
    if monitor.get("schema") != GPU_MONITOR_SCHEMA:
        fail(f"{label} GPU monitor has the wrong schema")
    gpu_uuid = monitor.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        fail(f"{label} GPU monitor has an invalid UUID")
    if expected_gpu_uuid is not None and gpu_uuid != expected_gpu_uuid:
        fail(f"{label} GPU monitor observed another GPU UUID")
    root_pid = monitor.get("root_process_pid")
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid < 1:
        fail(f"{label} GPU monitor has an invalid root PID")
    interval = monitor.get("poll_interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) != 1.0
    ):
        fail(f"{label} GPU monitor has an invalid interval")
    grace = monitor.get("cadence_grace_seconds")
    timeout_seconds = monitor.get("probe_timeout_seconds")
    if (
        isinstance(grace, bool)
        or not isinstance(grace, (int, float))
        or float(grace) != GPU_MONITOR_CADENCE_GRACE_SECONDS
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not (0 < float(timeout_seconds) <= 0.4)
    ):
        fail(f"{label} GPU monitor has an invalid cadence contract")
    parsed_timestamps: dict[str, dt.datetime] = {}
    for timestamp in ("started_utc", "finished_utc"):
        value = monitor.get(timestamp)
        if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
            fail(f"{label} GPU monitor has an invalid {timestamp}")
        parsed_timestamps[timestamp] = dt.datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=dt.timezone.utc)
    for timestamp in ("first_check_started_utc", "last_check_started_utc"):
        value = monitor.get(timestamp)
        if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
            fail(f"{label} GPU monitor has an invalid {timestamp}")
        parsed_timestamps[timestamp] = dt.datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=dt.timezone.utc)
    if not (
        parsed_timestamps["started_utc"]
        <= parsed_timestamps["first_check_started_utc"]
        <= parsed_timestamps["last_check_started_utc"]
        <= parsed_timestamps["finished_utc"]
    ):
        fail(f"{label} GPU monitor timestamps are not ordered")
    checks = monitor.get("checks_completed")
    if isinstance(checks, bool) or not isinstance(checks, int) or checks < 1:
        fail(f"{label} GPU monitor performed no checks")
    if monitor.get("foreign_process_incident") is not None:
        fail(f"{label} GPU monitor records an incident")
    if monitor.get("own_process_group_signals") != []:
        fail(f"{label} GPU monitor sent a stop signal")
    max_gap = monitor.get("max_observed_poll_gap_seconds")
    max_duration = monitor.get("max_observed_check_duration_seconds")
    max_lateness = monitor.get("max_observed_schedule_lateness_seconds")
    for field, value, upper_bound in (
        (
            "max_observed_poll_gap_seconds",
            max_gap,
            float(interval) + float(grace),
        ),
        (
            "max_observed_check_duration_seconds",
            max_duration,
            float(interval) + float(grace),
        ),
        (
            "max_observed_schedule_lateness_seconds",
            max_lateness,
            float(grace),
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or float(value) > upper_bound
        ):
            fail(f"{label} GPU monitor has an invalid {field}")
    first_offset = monitor.get("first_check_offset_seconds")
    last_offset = monitor.get("last_check_offset_seconds")
    measured_duration = monitor.get("monitor_duration_seconds")
    for field, value in (
        ("first_check_offset_seconds", first_offset),
        ("last_check_offset_seconds", last_offset),
        ("monitor_duration_seconds", measured_duration),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            fail(f"{label} GPU monitor has an invalid {field}")
    if not (
        float(first_offset) <= float(grace)
        and float(first_offset) <= float(last_offset) <= float(measured_duration)
        and float(measured_duration) - float(last_offset)
        <= float(interval) + float(grace)
    ):
        fail(f"{label} GPU monitor has an unaudited start or tail interval")
    started = parsed_timestamps["started_utc"]
    finished = parsed_timestamps["finished_utc"]
    duration_seconds = (finished - started).total_seconds()
    if duration_seconds < 0:
        fail(f"{label} GPU monitor finished before it started")
    first_wall_offset = (
        parsed_timestamps["first_check_started_utc"] - started
    ).total_seconds()
    last_wall_offset = (
        parsed_timestamps["last_check_started_utc"] - started
    ).total_seconds()
    if any(
        abs(wall_value - monotonic_value) > 1.5
        for wall_value, monotonic_value in (
            (duration_seconds, float(measured_duration)),
            (first_wall_offset, float(first_offset)),
            (last_wall_offset, float(last_offset)),
        )
    ):
        fail(f"{label} GPU monitor wall and monotonic clocks disagree")
    minimum_checks = max(
        1,
        math.floor(
            max(
                0.0,
                float(measured_duration)
                - float(first_offset)
                + float(grace),
            )
            / float(interval)
        ),
    )
    if checks < minimum_checks:
        fail(
            f"{label} GPU monitor cadence is not evidenced: "
            f"checks={checks}, required>={minimum_checks}"
        )
    return gpu_uuid


def validate_gpu_idle_record(
    record: object,
    *,
    label: str,
    expected_gpu_uuid: str,
) -> None:
    if not isinstance(record, dict):
        fail(f"runner completion lacks the {label} idle check")
    if (
        record.get("gpu_uuid") != expected_gpu_uuid
        or record.get("compute_process_count") != 0
        or record.get("query") != "gpu_uuid,pid,process_name,used_gpu_memory"
        or not isinstance(record.get("checked_utc"), str)
        or not _UTC_RE.fullmatch(record["checked_utc"])
    ):
        fail(f"runner completion has a malformed {label} idle check")


def validate_runner_completion_receipt(path: Path) -> dict[str, object]:
    payload = _load_json(path, "runner completion receipt")
    exact = {
        "schema": RUNNER_COMPLETION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "returncode": 0,
        "verifier_returncode": 0,
    }
    for field, expected in exact.items():
        if field in {"returncode", "verifier_returncode"} and isinstance(
            payload.get(field), bool
        ):
            fail(f"runner completion field {field!r} must be integer zero")
        if payload.get(field) != expected:
            fail(
                f"runner completion field {field!r} mismatch: "
                f"{payload.get(field)!r} != {expected!r}"
            )
    for field in ("started_utc", "finished_utc"):
        value = payload.get(field)
        if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
            fail(f"runner completion has an invalid {field}")
    file_bindings = {}
    for name_field, sha_field in (
        ("launch_manifest", "launch_manifest_sha256"),
        ("runner_log", "runner_log_sha256"),
        ("verifier_log", "verifier_log_sha256"),
    ):
        name = payload.get(name_field)
        digest = payload.get(sha_field)
        if not isinstance(name, str) or not name:
            fail(f"runner completion lacks {name_field}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"runner completion {name_field} escapes its run directory")
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            fail(f"runner completion has an invalid {sha_field}")
        file_bindings[name_field] = {"path": name, "sha256": digest}
    monitor_uuid = None
    for label in ("training", "verifier"):
        monitor = payload.get(f"{label}_gpu_exclusivity_monitor")
        gpu_uuid = validate_gpu_monitor_record(
            monitor,
            label=f"runner completion {label}",
            expected_gpu_uuid=monitor_uuid,
        )
        if monitor_uuid is None:
            monitor_uuid = gpu_uuid
    assert monitor_uuid is not None
    validate_gpu_idle_record(
        payload.get("final_prelaunch_gpu_idle_check"),
        label="final-prelaunch",
        expected_gpu_uuid=monitor_uuid,
    )
    for field, label in (
        ("post_training_gpu_idle_check", "post-training"),
        ("post_verifier_gpu_idle_check", "post-verifier"),
    ):
        validate_gpu_idle_record(
            payload.get(field),
            label=label,
            expected_gpu_uuid=monitor_uuid,
        )
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        fail("runner completion lacks immutable verifier receipt bindings")
    for field in ("validation_receipt_sha256", "artifact_hash_receipt_sha256"):
        value = verification.get(field)
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            fail(f"runner completion has an invalid verifier binding {field!r}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "training_gpu_monitor_checks": payload[
            "training_gpu_exclusivity_monitor"
        ]["checks_completed"],
        "verifier_gpu_monitor_checks": payload[
            "verifier_gpu_exclusivity_monitor"
        ]["checks_completed"],
        "gpu_uuid": monitor_uuid,
        "file_bindings": file_bindings,
        "verification": verification,
    }


def validate_existing_runner_completion(run_dir: Path) -> dict[str, object]:
    candidates = []
    primary = run_dir / "runner_completion.json"
    if primary.exists():
        candidates.append(primary)
    candidates.extend(sorted(run_dir.glob("runner-completion-resume-*.json")))
    if not candidates:
        fail(f"run has no runner completion receipt: {run_dir}")
    passing = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            fail(f"runner completion candidate is not a regular file: {path}")
        payload = _load_json(path, "runner completion candidate")
        if payload.get("status") == "PASS":
            passing.append(validate_runner_completion_receipt(path))
    if len(passing) != 1:
        fail(
            "run must have exactly one monitor-bound PASS runner completion; "
            f"found {len(passing)} in {run_dir}"
        )
    record = passing[0]
    resolved_files = {}
    for label, binding in record["file_bindings"].items():
        path = _resolve_run_contained_file(
            run_dir,
            Path(binding["path"]),
            f"runner completion {label}",
        )
        if sha256_file(path) != binding["sha256"]:
            fail(f"runner completion has a stale {label} hash")
        resolved_files[label] = path
    launch_manifest = _load_json(
        resolved_files["launch_manifest"], "runner-bound launch manifest"
    )
    if launch_manifest.get("run_directory") != str(run_dir.resolve(strict=True)):
        fail("runner-bound launch manifest targets another run directory")
    launch_gpu = launch_manifest.get("gpu")
    if not isinstance(launch_gpu, dict) or launch_gpu.get("uuid") != record.get(
        "gpu_uuid"
    ):
        fail("runner monitor GPU UUID differs from the launch manifest")
    arm_validation = _load_json(
        run_dir / VALIDATION_FILENAME, "runner-bound arm validation receipt"
    )
    launch_training = launch_manifest.get("training")
    if not isinstance(launch_training, dict):
        fail("runner-bound launch manifest lacks the training identity")
    expected_training_identity = {
        "phase": arm_validation.get("mode"),
        "arm": arm_validation.get("arm"),
        "seed": arm_validation.get("seed"),
    }
    for field, expected in expected_training_identity.items():
        if launch_training.get(field) != expected:
            fail(f"runner-bound launch manifest training {field} differs")
    verification = record["verification"]
    expected_bindings = {
        "validation_receipt_sha256": run_dir / VALIDATION_FILENAME,
        "artifact_hash_receipt_sha256": run_dir / HASH_RECEIPT_FILENAME,
    }
    for field, path in expected_bindings.items():
        if path.is_symlink() or not path.is_file():
            fail(f"runner completion bound verifier artifact is missing: {path}")
        if verification.get(field) != sha256_file(path):
            fail(f"runner completion has a stale verifier binding: {field}")
    return record


def validate_exact_resume_provenance(
    uninterrupted_dir: Path,
    resumed_dir: Path,
    *,
    arm: str,
    seed: int,
    runtime_command: Sequence[str] | None,
    process_env: Mapping[str, str] | None,
) -> dict[str, object]:
    """Prove that the second leg is exactly one 16-attempt pause + resume."""

    uninterrupted_dir = uninterrupted_dir.expanduser().resolve(strict=True)
    resumed_dir = resumed_dir.expanduser().resolve(strict=True)
    if uninterrupted_dir == resumed_dir:
        fail("exact-resume provenance requires two distinct run directories")
    expected_training = training_contract("smoke", arm, seed)
    none_gate = {
        "kind": "none",
        "stop_after_attempts": None,
        "scientific_training_contract_unchanged": True,
    }
    pause_gate = {
        "kind": "planned_exact_resume_pause",
        "stop_after_attempts": PLANNED_PAUSE_ATTEMPTS,
        "scientific_training_contract_unchanged": True,
    }

    uninterrupted_manifest_path = uninterrupted_dir / "launch_manifest.json"
    uninterrupted_manifest = _load_json(
        uninterrupted_manifest_path, "uninterrupted launch manifest"
    )
    if (
        uninterrupted_manifest.get("schema") != LAUNCH_SCHEMA
        or uninterrupted_manifest.get("launch_kind") != "fresh_transfer"
        or uninterrupted_manifest.get("run_directory") != str(uninterrupted_dir)
        or uninterrupted_manifest.get("training") != expected_training
        or uninterrupted_manifest.get("gate_control") != none_gate
        or uninterrupted_manifest.get("resume_state") is not None
        or uninterrupted_manifest.get("resume_state_sha256") is not None
        or uninterrupted_manifest.get("validated_planned_pause_completion") is not None
        or uninterrupted_manifest.get("original_gate_control") is not None
        or uninterrupted_manifest.get("original_launch_manifest_sha256") is not None
    ):
        fail("exact-resume uninterrupted leg is not one uninterrupted fresh launch")
    forbidden_uninterrupted = []
    for pattern in (
        "resume_launch_manifest-*.json",
        "runner-completion-resume-*.json",
        "runner-resume-*.log",
        "arm-verifier-resume-*.log",
    ):
        forbidden_uninterrupted.extend(uninterrupted_dir.glob(pattern))
    if forbidden_uninterrupted or (uninterrupted_dir / PLANNED_PAUSE_EVIDENCE_DIR).exists():
        fail("exact-resume uninterrupted leg contains resume/pause artifacts")
    uninterrupted_runner = validate_existing_runner_completion(uninterrupted_dir)
    if (
        uninterrupted_runner["file_bindings"]["launch_manifest"]["path"]
        != "launch_manifest.json"
        or uninterrupted_runner["file_bindings"]["launch_manifest"]["sha256"]
        != sha256_file(uninterrupted_manifest_path)
    ):
        fail("exact-resume uninterrupted runner does not bind its fresh manifest")

    original_manifest_path = resumed_dir / "launch_manifest.json"
    original_manifest = _load_json(
        original_manifest_path, "planned exact-resume original launch manifest"
    )
    if (
        original_manifest.get("schema") != LAUNCH_SCHEMA
        or original_manifest.get("launch_kind") != "fresh_transfer"
        or original_manifest.get("run_directory") != str(resumed_dir)
        or original_manifest.get("training") != expected_training
        or original_manifest.get("gate_control") != pause_gate
        or original_manifest.get("resume_state") is not None
        or original_manifest.get("resume_state_sha256") is not None
        or original_manifest.get("validated_planned_pause_completion") is not None
        or original_manifest.get("original_gate_control") is not None
        or original_manifest.get("original_launch_manifest_sha256") is not None
    ):
        fail("exact-resume resumed leg did not begin as the fixed fresh planned pause")
    resume_manifests = sorted(resumed_dir.glob("resume_launch_manifest-*.json"))
    resume_completions = sorted(
        resumed_dir.glob("runner-completion-resume-*.json")
    )
    resume_logs = sorted(resumed_dir.glob("runner-resume-*.log"))
    verifier_logs = sorted(resumed_dir.glob("arm-verifier-resume-*.log"))
    if not all(
        len(paths) == 1
        for paths in (resume_manifests, resume_completions, resume_logs, verifier_logs)
    ):
        fail("exact-resume resumed leg must contain exactly one resume attempt")
    for path in (
        resume_manifests[0],
        resume_completions[0],
        resume_logs[0],
        verifier_logs[0],
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            fail(f"exact-resume provenance artifact is invalid: {path}")

    resume_manifest_path = resume_manifests[0]
    resume_manifest = _load_json(resume_manifest_path, "exact-resume launch manifest")
    planned_record = resume_manifest.get("validated_planned_pause_completion")
    if not isinstance(planned_record, dict) or planned_record.get("status") != (
        PLANNED_PAUSE_STATUS
    ):
        fail("exact-resume manifest lacks the validated planned pause binding")
    evidence_record = planned_record.get("evidence")
    evidence_validation = validate_preserved_planned_pause_evidence(
        resumed_dir,
        evidence_record,
        arm=arm,
        seed=seed,
        runtime_command=runtime_command,
        process_env=process_env,
    )
    copied_hashes = evidence_validation.get("files_sha256")
    if (
        not isinstance(copied_hashes, dict)
        or copied_hashes.get("launch_manifest.json")
        != sha256_file(original_manifest_path)
        or copied_hashes.get("training_options.json")
        != sha256_file(resumed_dir / "training_options.json")
        or copied_hashes.get("initial_state_receipt_v1.json")
        != sha256_file(resumed_dir / "initial_state_receipt_v1.json")
    ):
        fail("exact-resume pause evidence was spliced from another run")
    primary_completion = resumed_dir / "runner_completion.json"
    if (
        primary_completion.is_symlink()
        or not primary_completion.is_file()
        or sha256_file(primary_completion) != planned_record.get("sha256")
        or planned_record.get("sha256")
        != evidence_validation["planned_pause_completion_sha256"]
    ):
        fail("exact-resume planned-pause completion binding is stale")
    original_manifest_sha = sha256_file(original_manifest_path)
    expected_resume_state = evidence_validation["resume_state"]
    expected_resume_state_sha = evidence_validation["resume_state_sha256"]
    if (
        resume_manifest.get("schema") != LAUNCH_SCHEMA
        or resume_manifest.get("launch_kind") != "resume"
        or resume_manifest.get("status") != "authorized_to_start"
        or resume_manifest.get("run_directory") != str(resumed_dir)
        or resume_manifest.get("training") != expected_training
        or resume_manifest.get("gate_control") != none_gate
        or resume_manifest.get("original_gate_control") != pause_gate
        or resume_manifest.get("original_launch_manifest_sha256")
        != original_manifest_sha
        or resume_manifest.get("resume_state") != expected_resume_state
        or resume_manifest.get("resume_state_sha256") != expected_resume_state_sha
    ):
        fail("exact-resume launch manifest does not close the pause-to-resume chain")
    for field in (
        "source",
        "assets",
        "authorization",
        "gpu",
        "runtime_sandbox",
        "runtime",
        "process_environment",
    ):
        if resume_manifest.get(field) != original_manifest.get(field):
            fail(f"exact-resume launch changed trajectory identity field {field!r}")
    command = resume_manifest.get("exact_command_argv")
    assets = original_manifest.get("assets")
    if (
        not isinstance(assets, dict)
        or not isinstance(assets.get("dataset"), dict)
        or not isinstance(assets.get("transfer"), dict)
    ):
        fail("exact-resume original launch has malformed asset bindings")
    expected_command = build_training_command(
        python_bin=DEFAULT_TRAINING_PYTHON,
        data=Path(str(assets["dataset"].get("resolved_path"))),
        transfer=Path(str(assets["transfer"].get("resolved_path"))),
        outdir=resumed_dir,
        phase="smoke",
        arm=arm,
        seed=seed,
        resume=Path(expected_resume_state),
        runtime_command=runtime_command,
    )
    if (
        not isinstance(command, list)
        or command != expected_command
        or resume_manifest.get("exact_command_shell") != shlex.join(expected_command)
    ):
        fail("exact-resume command differs from the frozen preserved-state command")
    resumed_runner = validate_existing_runner_completion(resumed_dir)
    if (
        resumed_runner["path"] != str(resume_completions[0].resolve(strict=True))
        or resumed_runner["file_bindings"]["launch_manifest"]["path"]
        != resume_manifest_path.name
        or resumed_runner["file_bindings"]["launch_manifest"]["sha256"]
        != sha256_file(resume_manifest_path)
        or resumed_runner["file_bindings"]["runner_log"]["path"]
        != resume_logs[0].name
        or resumed_runner["file_bindings"]["verifier_log"]["path"]
        != verifier_logs[0].name
    ):
        fail("exact-resume PASS runner does not bind the unique resume attempt")

    record = {
        "status": "PASS",
        "arm": arm,
        "seed": seed,
        "uninterrupted_launch_manifest_sha256": sha256_file(
            uninterrupted_manifest_path
        ),
        "uninterrupted_runner_completion_sha256": uninterrupted_runner["sha256"],
        "planned_launch_manifest_sha256": original_manifest_sha,
        "planned_pause_completion_sha256": planned_record["sha256"],
        "pause_evidence_manifest_sha256": evidence_validation["manifest_sha256"],
        "pause_launch_manifest_sha256": copied_hashes["launch_manifest.json"],
        "pause_initial_receipt_sha256": copied_hashes[
            "initial_state_receipt_v1.json"
        ],
        "pause_state_sha256": expected_resume_state_sha,
        "resume_launch_manifest_sha256": sha256_file(resume_manifest_path),
        "resume_runner_completion_sha256": resumed_runner["sha256"],
        "resume_attempt_count": 1,
    }
    record["provenance_sha256"] = canonical_sha256(record)
    return record


def parse_seed_gpu(values: Sequence[str], phase: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for value in values:
        match = re.fullmatch(r"(\d+)=(.+)", value)
        if not match:
            fail(f"--seed-gpu must be SEED=GPU, got {value!r}")
        seed, gpu = int(match.group(1)), match.group(2)
        if seed in mapping:
            fail(f"duplicate GPU assignment for seed {seed}")
        if not _GPU_RE.fullmatch(gpu):
            fail(f"invalid explicit GPU selector: {gpu!r}")
        mapping[seed] = gpu
    expected = set(PHASES[phase]["seeds"])
    if set(mapping) != expected:
        fail(f"phase={phase} requires exact seed GPU map {sorted(expected)}, got {sorted(mapping)}")
    return mapping


def parse_resume_cells(values: Sequence[str], phase: str) -> dict[tuple[int, str], Path]:
    parsed: dict[tuple[int, str], Path] = {}
    for value in values:
        match = re.fullmatch(r"(\d+):([A-D])=(.+)", value)
        if not match:
            fail(f"--resume-cell must be SEED:ARM=STATE, got {value!r}")
        key = (int(match.group(1)), match.group(2))
        validate_arm_seed_phase(phase, key[1], key[0])
        if key in parsed:
            fail(f"duplicate resume cell: {key}")
        parsed[key] = Path(match.group(3)).expanduser()
    return parsed


def validate_resume_cell_set(
    resume_cells: Mapping[tuple[int, str], Path], phase: str
) -> None:
    if not resume_cells:
        return
    expected = {
        (seed, arm) for seed in PHASES[phase]["seeds"] for arm in ARMS
    }
    if set(resume_cells) != expected:
        fail(
            "resume execution requires the complete phase cell set before any "
            f"cell starts: expected={sorted(expected)}, actual={sorted(resume_cells)}"
        )


def make_matrix_jobs(
    *,
    phase: str,
    seed_gpu: Mapping[int, str],
    runs_root: Path,
    matrix_id: str,
    authorization_receipt: Path | None,
    data: Path,
    transfer: Path,
    python_bin: str | Path,
    lock_root: Path,
    base_port: int,
    resume_cells: Mapping[tuple[int, str], Path] | None = None,
    runtime_sandbox: Path = DEFAULT_RUNTIME_SANDBOX,
    expected_skip_attempts: str | None = None,
) -> list[MatrixJob]:
    resume_cells = dict(resume_cells or {})
    validate_resume_cell_set(resume_cells, phase)
    cells: Iterable[tuple[int, str]]
    if resume_cells:
        cells = sorted(resume_cells, key=lambda item: (item[0], list(ARMS).index(item[1])))
    else:
        cells = ((seed, arm) for seed in PHASES[phase]["seeds"] for arm in ARMS)
    jobs = []
    for ordinal, (seed, arm) in enumerate(cells):
        resume = resume_cells.get((seed, arm))
        outdir = None if resume else runs_root / phase / matrix_id / f"seed{seed}" / f"arm{arm}"
        command = [
            "bash",
            str(ARM_SCRIPT),
            "--phase",
            phase,
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--gpu",
            seed_gpu[seed],
            "--master-port",
            str(base_port + ordinal),
            "--runs-root",
            str(runs_root),
            "--data",
            str(data),
            "--transfer",
            str(transfer),
            "--python-bin",
            str(python_bin),
            "--runtime-sandbox",
            str(runtime_sandbox),
            "--lock-root",
            str(lock_root),
        ]
        if expected_skip_attempts is not None:
            command += ["--expected-skip-attempts", expected_skip_attempts]
        if authorization_receipt is not None:
            command += ["--authorization-receipt", str(authorization_receipt)]
        if resume is None:
            command += ["--outdir", str(outdir)]
        else:
            command += ["--resume", str(resume)]
        jobs.append(
            MatrixJob(
                seed=seed,
                arm=arm,
                gpu=seed_gpu[seed],
                master_port=base_port + ordinal,
                outdir=outdir,
                resume=resume,
                command=tuple(command),
            )
        )
    return jobs


def verify_matrix_skip_equivalence(
    jobs: Sequence[MatrixJob],
    *,
    phase: str,
    expected_skip_attempts: list[int] | None = None,
) -> dict:
    expected_cells = {
        (seed, arm) for seed in PHASES[phase]["seeds"] for arm in ARMS
    }
    actual_cells = {(job.seed, job.arm) for job in jobs}
    if actual_cells != expected_cells or len(jobs) != len(expected_cells):
        fail(
            "AMP skip equivalence requires the complete phase cell set: "
            f"expected={sorted(expected_cells)}, actual={sorted(actual_cells)}"
        )
    by_seed: dict[int, list[tuple[str, list[int], int, str]]] = defaultdict(list)
    for job in jobs:
        run_dir = job.resume.resolve(strict=True).parent if job.resume else job.outdir
        assert run_dir is not None
        with (run_dir / "factorial_training_telemetry_v1.csv").open(
            "rt", newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        skips = [
            int(row["attempted_iteration"])
            for row in rows
            if row.get("step_skipped", "").lower() in {"1", "true"}
        ]
        skips = validate_amp_skip_signature(
            skips,
            phase=phase,
            label=f"matrix telemetry seed={job.seed} arm={job.arm}",
        )
        immutable = validate_existing_verifier_receipts(
            run_dir,
            phase=phase,
            arm=job.arm,
            seed=job.seed,
            expected_skip_attempts=expected_skip_attempts,
        )
        if immutable is None:
            fail(f"matrix cell lacks immutable verifier receipts: {run_dir}")
        if immutable["amp_skip_attempts"] != skips:
            fail(
                "matrix telemetry AMP skip signature differs from its immutable "
                f"verifier receipt for seed={job.seed} arm={job.arm}"
            )
        by_seed[job.seed].append(
            (
                job.arm,
                skips,
                int(immutable["successful_optimizer_steps"]),
                str(immutable["initial_common_state_sha256"]),
            )
        )
    result = {}
    for seed, arm_values in sorted(by_seed.items()):
        reference_skip_count = len(arm_values[0][1])
        reference_successful_steps = arm_values[0][2]
        reference_initial = arm_values[0][3]
        skip_count_mismatched = [
            (arm, len(skips))
            for arm, skips, _successful, _initial in arm_values
            if len(skips) != reference_skip_count
        ]
        if skip_count_mismatched:
            fail(f"arm-specific AMP skip count for seed {seed}: {arm_values}")
        successful_steps_mismatched = [
            (arm, successful)
            for arm, _skips, successful, _initial in arm_values
            if successful != reference_successful_steps
        ]
        if successful_steps_mismatched:
            fail(
                f"arm-specific successful optimizer-step count for seed {seed}: "
                f"{arm_values}"
            )
        initial_mismatched = [
            (arm, initial)
            for arm, _skips, _successful, initial in arm_values
            if initial != reference_initial
        ]
        if initial_mismatched:
            fail(
                f"arm-specific initial common state for seed {seed}: {arm_values}"
            )
        result[str(seed)] = {
            "arms": [arm for arm, _skips, _successful, _initial in arm_values],
            "skip_attempts_by_arm": {
                arm: skips for arm, skips, _successful, _initial in arm_values
            },
            "skip_count": reference_skip_count,
            "successful_optimizer_steps": reference_successful_steps,
            "initial_common_state_sha256": reference_initial,
        }
    return result


def run_matrix(args: argparse.Namespace) -> int:
    runs_root = validate_runs_root(args.runs_root)
    expected_skip_values = parse_expected_skip_attempts(
        args.expected_skip_attempts, args.phase
    )
    expected_skip_argument = (
        json.dumps(expected_skip_values, separators=(",", ":"))
        if expected_skip_values is not None
        else None
    )
    seed_gpu = parse_seed_gpu(args.seed_gpu, args.phase)
    resume_cells = parse_resume_cells(args.resume_cell, args.phase)
    validate_resume_cell_set(resume_cells, args.phase)
    matrix_id = args.matrix_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", matrix_id):
        fail(f"invalid --matrix-id: {matrix_id!r}")
    jobs = make_matrix_jobs(
        phase=args.phase,
        seed_gpu=seed_gpu,
        runs_root=runs_root,
        matrix_id=matrix_id,
        authorization_receipt=args.authorization_receipt,
        data=args.data,
        transfer=args.transfer,
        python_bin=args.python_bin,
        lock_root=args.lock_root,
        base_port=args.base_port,
        resume_cells=resume_cells,
        runtime_sandbox=args.runtime_sandbox,
        expected_skip_attempts=expected_skip_argument,
    )
    plan = {
        "schema": MATRIX_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "matrix_id": matrix_id,
        "phase": args.phase,
        "mode": "resume_selected_cells" if resume_cells else "fresh_exact_matrix",
        "seed_gpu_binding": {str(key): value for key, value in sorted(seed_gpu.items())},
        "arm_order": list(ARMS),
        "expected_amp_skip_attempts": expected_skip_values,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "expected_cell_count": len(resume_cells) if resume_cells else len(PHASES[args.phase]["seeds"]) * 4,
        "jobs": [job.as_dict() for job in jobs],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.authorization_receipt is None and not resume_cells:
        fail("fresh matrix execution requires --authorization-receipt")

    # Matrix preflight repeats the expensive gates once before any run directory
    # is created.  Each arm repeats them under its own GPU flock immediately
    # before launch, closing the time-of-check/time-of-use window.
    storage = storage_preflight(runs_root, args.phase)
    source = source_snapshot(require_clean=True)
    dataset = verify_asset(args.data, EXPECTED_DATASET_SHA256, "canonical dataset")
    transfer = verify_asset(args.transfer, EXPECTED_TRANSFER_SHA256, "authoritative transfer")
    runtime_command, runtime_base = runtime_prefix(
        args.runtime_sandbox, str(args.python_bin)
    )
    sandbox = runtime_sandbox_fingerprint(runtime_base)
    resolved_by_selector = {
        selector: query_gpu(selector) for selector in sorted(set(seed_gpu.values()))
    }
    unique_by_uuid = {
        str(record["uuid"]): record for record in resolved_by_selector.values()
    }
    if len(unique_by_uuid) != len(resolved_by_selector):
        fail(
            "multiple --seed-gpu selector spellings resolve to the same physical GPU; "
            "use one canonical selector consistently"
        )
    if len(unique_by_uuid) != 1:
        fail(
            "the frozen Role-E authorization is bound to one physical GPU; "
            "map every seed to that same GPU UUID and run the matrix serially"
        )
    with contextlib.ExitStack() as stack:
        for gpu_uuid in sorted(unique_by_uuid):
            stack.enter_context(gpu_lock(gpu_uuid, args.lock_root))
        matrix_gpu_idle_checks = [
            assert_gpu_idle(record) for record in unique_by_uuid.values()
        ]
        probe_selector = sorted(resolved_by_selector)[0]
        probe_env = build_process_environment(probe_selector, args.base_port)
        runtime = runtime_environment(runtime_command, probe_env)
        matrix_gpu_idle_checks.extend(
            assert_gpu_idle(record) for record in unique_by_uuid.values()
        )
        for job in jobs:
            assert_master_port_available(job.master_port)
        final_sandbox = runtime_sandbox_fingerprint(runtime_base)
        for key in ("sandbox_tree_metadata_sha256", "critical_files_sha256"):
            if final_sandbox[key] != sandbox[key]:
                fail(f"runtime sandbox changed during matrix preflight: {key}")
        if not resume_cells:
            assert args.authorization_receipt is not None
            for seed in PHASES[args.phase]["seeds"]:
                validate_authorization(
                    args.authorization_receipt,
                    phase=args.phase,
                    arm="A",
                    seed=seed,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                    expected_skip_attempts=expected_skip_values,
                    revalidation_command=runtime_command,
                    revalidation_env=probe_env,
                )
    plan["preflight"] = {
        "source_git_head": source["git_head"],
        "source_content_sha256": source["content_sha256"],
        "dataset_sha256": dataset["sha256"],
        "transfer_sha256": transfer["sha256"],
        "runtime_sandbox_tree_metadata_sha256": sandbox[
            "sandbox_tree_metadata_sha256"
        ],
        "runtime_software_sha256": runtime["software_sha256"],
        "gpus": list(unique_by_uuid.values()),
        "gpu_idle_checks": matrix_gpu_idle_checks,
        "storage": storage,
        "all_selected_gpus_idle_before_any_cell": True,
    }
    matrix_dir = runs_root / args.phase / matrix_id
    matrix_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        matrix_dir.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"matrix directory already exists: {matrix_dir}")
    plan_path = matrix_dir / "matrix_plan.json"
    process_registry_root = matrix_dir / "process_registry"
    process_registry_root.mkdir(mode=0o750)

    queues: dict[str, list[MatrixJob]] = defaultdict(list)
    for job in jobs:
        queues[job.gpu].append(job)
    stop_event = threading.Event()
    failures: list[dict] = []
    completed: list[dict] = []
    skipped_pass: list[dict] = []
    mutex = threading.RLock()
    active_processes: dict[tuple[int, str], subprocess.Popen[bytes]] = {}
    matrix_stop_actions: list[dict[str, object]] = []
    cleanup_unconfirmed: list[dict[str, object]] = []
    received_signal: dict[str, object] | None = None
    live_seed_identity: dict[int, dict[str, object]] = {}
    stop_requested_monotonic: float | None = None

    def register_live_cell_identity(
        job: MatrixJob, immutable: Mapping[str, object]
    ) -> str | None:
        signature = immutable.get("amp_skip_attempts")
        if not isinstance(signature, list):
            return f"seed {job.seed} arm {job.arm} has malformed AMP skip attempts"
        successful_steps = immutable.get("successful_optimizer_steps")
        if isinstance(successful_steps, bool) or not isinstance(successful_steps, int):
            return f"seed {job.seed} arm {job.arm} has malformed successful steps"
        initial_sha = immutable.get("initial_common_state_sha256")
        with mutex:
            existing = live_seed_identity.get(job.seed)
            if existing is None:
                live_seed_identity[job.seed] = {
                    "amp_skip_attempts_by_arm": {job.arm: signature},
                    "amp_skip_count": len(signature),
                    "successful_optimizer_steps": successful_steps,
                    "initial_common_state_sha256": initial_sha,
                    "arms": [job.arm],
                }
                return None
            if existing["amp_skip_count"] != len(signature):
                return (
                    f"seed {job.seed} AMP skip count changed at arm {job.arm}: "
                    f"{existing['amp_skip_count']} != {len(signature)}"
                )
            if existing["successful_optimizer_steps"] != successful_steps:
                return (
                    f"seed {job.seed} successful optimizer-step count changed at "
                    f"arm {job.arm}: {existing['successful_optimizer_steps']} != "
                    f"{successful_steps}"
                )
            if existing["initial_common_state_sha256"] != initial_sha:
                return (
                    f"seed {job.seed} initial common state changed at arm {job.arm}: "
                    f"{existing['initial_common_state_sha256']} != {initial_sha}"
                )
            arms = existing["arms"]
            assert isinstance(arms, list)
            if job.arm in arms:
                return f"seed {job.seed} arm {job.arm} was registered twice"
            arms.append(job.arm)
            signatures = existing["amp_skip_attempts_by_arm"]
            assert isinstance(signatures, dict)
            signatures[job.arm] = signature
        return None

    def request_matrix_stop(
        reason: str, *, exclude: tuple[int, str] | None = None
    ) -> None:
        nonlocal stop_requested_monotonic
        stop_event.set()
        with mutex:
            if stop_requested_monotonic is None:
                stop_requested_monotonic = time.monotonic()
            targets = [
                (key, process)
                for key, process in active_processes.items()
                if key != exclude and process.poll() is None
            ]
        dispatched = []
        for key, process in targets:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                dispatched.append(
                    {
                        "seed": key[0],
                        "arm": key[1],
                        "process_group": process.pid,
                        "signal": "SIGTERM",
                    }
                )
            except ProcessLookupError:
                continue
        with mutex:
            matrix_stop_actions.append(
                {
                    "requested_utc": utc_now(),
                    "reason": reason,
                    "targets": dispatched,
                }
            )

    previous_matrix_signal_handlers: dict[int, object] = {}

    def handle_matrix_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = {
            "received_utc": utc_now(),
            "signal": signal.Signals(signum).name,
        }
        request_matrix_stop(f"matrix received {signal.Signals(signum).name}")

    managed_matrix_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    if not hasattr(signal, "pthread_sigmask"):
        fail("platform cannot protect matrix plan-to-supervisor signal handoff")
    previous_matrix_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, managed_matrix_signals
    )
    plan_sha256 = hashlib.sha256(
        (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    try:
        for signum in managed_matrix_signals:
            previous_matrix_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_matrix_signal)
        atomic_json_exclusive(plan_path, plan)
    except BaseException:
        for signum, handler in previous_matrix_signal_handlers.items():
            signal.signal(signum, handler)
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_matrix_mask)

    def worker_impl(gpu: str, queue: Sequence[MatrixJob]) -> None:
        for job in queue:
            if stop_event.is_set():
                return
            key = (job.seed, job.arm)

            def register_worker_process(process: subprocess.Popen[bytes]) -> None:
                with mutex:
                    active_processes[key] = process

            def unregister_worker_process(process: subprocess.Popen[bytes]) -> None:
                with mutex:
                    if active_processes.get(key) is process:
                        active_processes.pop(key, None)

            if job.resume is not None:
                try:
                    run_dir = job.resume.expanduser().resolve(strict=True).parent
                    existing = validate_existing_verifier_receipts(
                        run_dir,
                        phase=args.phase,
                        arm=job.arm,
                        seed=job.seed,
                        expected_skip_attempts=expected_skip_values,
                    )
                except (OSError, LaunchError) as exc:
                    with mutex:
                        failures.append(
                            {
                                "seed": job.seed,
                                "arm": job.arm,
                                "gpu": gpu,
                                "returncode": None,
                                "preflight_error": str(exc),
                            }
                        )
                    request_matrix_stop(
                        f"resume preflight failed for seed={job.seed} arm={job.arm}",
                        exclude=(job.seed, job.arm),
                    )
                    return
                if existing is not None:
                    try:
                        deep_revalidation = deep_revalidate_existing_arm(
                            run_dir,
                            phase=args.phase,
                            arm=job.arm,
                            seed=job.seed,
                            expected_skip_attempts=expected_skip_values,
                            runtime_command=runtime_command,
                            process_env=build_process_environment(
                                job.gpu, job.master_port
                            ),
                            stop_event=stop_event,
                            register_process=register_worker_process,
                            unregister_process=unregister_worker_process,
                        )
                        runner_completion = validate_existing_runner_completion(
                            run_dir
                        )
                    except LaunchError as exc:
                        with mutex:
                            failures.append(
                                {
                                    "seed": job.seed,
                                    "arm": job.arm,
                                    "gpu": gpu,
                                    "returncode": None,
                                    "preflight_error": str(exc),
                                }
                            )
                        request_matrix_stop(
                            "existing PASS lacks a monitor-bound runner completion "
                            f"for seed={job.seed} arm={job.arm}",
                            exclude=(job.seed, job.arm),
                        )
                        return
                    if stop_event.is_set():
                        return
                    identity_error = register_live_cell_identity(job, existing)
                    if identity_error is not None:
                        with mutex:
                            failures.append(
                                {
                                    "seed": job.seed,
                                    "arm": job.arm,
                                    "gpu": gpu,
                                    "returncode": None,
                                    "cross_arm_identity_error": identity_error,
                                }
                            )
                        request_matrix_stop(
                            identity_error, exclude=(job.seed, job.arm)
                        )
                        return
                    with mutex:
                        skipped_pass.append(
                            {
                                "seed": job.seed,
                                "arm": job.arm,
                                "gpu": gpu,
                                "status": "SKIPPED_EXISTING_IMMUTABLE_PASS",
                                "receipt": existing,
                                "fresh_revalidation": deep_revalidation,
                                "runner_completion": runner_completion,
                            }
                        )
                    print(
                        f"[matrix] validated immutable PASS; skip seed={job.seed} "
                        f"arm={job.arm}",
                        flush=True,
                    )
                    continue
            if stop_event.is_set():
                return
            print(f"[matrix] starting seed={job.seed} arm={job.arm} gpu={gpu}", flush=True)
            run_dir = (
                job.resume.expanduser().resolve(strict=True).parent
                if job.resume is not None
                else job.outdir
            )
            assert run_dir is not None
            prior_resume_completions = (
                set(run_dir.glob("runner-completion-resume-*.json"))
                if job.resume is not None
                else set()
            )
            process: subprocess.Popen[bytes] | None = None
            outer_cleanup_confirmed = False
            inner_registry_audit: dict[str, object] | None = None
            registry_dir = process_registry_root / f"seed{job.seed}-arm{job.arm}"
            try:
                if stop_event.is_set():
                    return
                registry_dir.mkdir(mode=0o750)
                outer_env = dict(os.environ)
                outer_env[MATRIX_CHILD_REGISTRY_ENV] = str(
                    registry_dir.resolve(strict=True)
                )
                process = subprocess.Popen(
                    [sys.executable, "-c", MATRIX_OUTER_WRAPPER_CODE, *job.command],
                    cwd=REPO_ROOT,
                    env=outer_env,
                    start_new_session=True,
                )
                register_worker_process(process)
                with mutex:
                    stop_was_already_requested = stop_event.is_set()
                if stop_was_already_requested and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                returncode = process.wait()
                inner_registry_audit = drain_matrix_child_registry(
                    registry_dir, matrix_outer_pid=process.pid
                )
                if returncode == 0 and inner_registry_audit["token_count"] != 2:
                    raise ProcessCleanupError(
                        "successful arm wrapper did not publish exactly two "
                        "training/verifier ownership tokens"
                    )
                outer_cleanup_confirmed = True
            except ProcessCleanupError:
                raise
            except Exception as exc:
                stop_signals = []
                if process is not None and process.poll() is None:
                    stop_signals = stop_and_reap_process_group(
                        process, label="matrix-arm-wrapper"
                    )
                if process is not None:
                    inner_registry_audit = drain_matrix_child_registry(
                        registry_dir, matrix_outer_pid=process.pid
                    )
                outer_cleanup_confirmed = True
                with mutex:
                    failures.append(
                        {
                            "seed": job.seed,
                            "arm": job.arm,
                            "gpu": gpu,
                            "returncode": None,
                            "worker_exception": f"{type(exc).__name__}: {exc}",
                            "own_process_group_signals": stop_signals,
                            "inner_registry_audit": inner_registry_audit,
                        }
                    )
                request_matrix_stop(
                    f"matrix worker crashed for seed={job.seed} arm={job.arm}",
                    exclude=key,
                )
                return
            finally:
                if process is not None and outer_cleanup_confirmed:
                    unregister_worker_process(process)
            if returncode == PROCESS_CLEANUP_UNCONFIRMED_EXIT_CODE:
                detail = {
                    "seed": job.seed,
                    "arm": job.arm,
                    "gpu": gpu,
                    "returncode": returncode,
                    "reason": (
                        "arm runner could not prove that its child process tree "
                        "was fully removed"
                    ),
                }
                with mutex:
                    cleanup_unconfirmed.append(detail)
                    failures.append(dict(detail))
                request_matrix_stop(
                    f"unconfirmed child cleanup for seed={job.seed} arm={job.arm}",
                    exclude=key,
                )
                return
            runner_completion = None
            immutable = None
            drained_inner_sessions = (
                inner_registry_audit.get("active_inner_sessions_drained", [])
                if isinstance(inner_registry_audit, dict)
                else []
            )
            completion_error = (
                "arm wrapper exited while registered inner sessions were still active"
                if drained_inner_sessions
                else None
            )
            if returncode == 0 and completion_error is None:
                try:
                    if job.resume is None:
                        completion_path = run_dir / "runner_completion.json"
                    else:
                        new_completions = set(
                            run_dir.glob("runner-completion-resume-*.json")
                        ) - prior_resume_completions
                        if len(new_completions) != 1:
                            fail(
                                "successful resume did not create exactly one new "
                                "runner completion receipt"
                            )
                        completion_path = next(iter(new_completions))
                    runner_completion = validate_existing_runner_completion(run_dir)
                    if runner_completion.get("path") != str(
                        completion_path.resolve(strict=True)
                    ):
                        fail("new runner completion is not the unique bound PASS")
                    immutable = validate_existing_verifier_receipts(
                        run_dir,
                        phase=args.phase,
                        arm=job.arm,
                        seed=job.seed,
                        expected_skip_attempts=expected_skip_values,
                    )
                    if immutable is None:
                        fail("successful cell lacks immutable verifier receipts")
                except (OSError, LaunchError) as exc:
                    completion_error = str(exc)
            identity_error = None
            if runner_completion is not None and immutable is not None:
                identity_error = register_live_cell_identity(job, immutable)
                if identity_error is not None:
                    completion_error = identity_error
                    runner_completion = None
            with mutex:
                record = {
                    "seed": job.seed,
                    "arm": job.arm,
                    "gpu": gpu,
                    "returncode": returncode,
                    "inner_registry_audit": inner_registry_audit,
                }
                if runner_completion is not None:
                    record["runner_completion"] = runner_completion
                    completed.append(record)
                else:
                    if completion_error is not None:
                        record[
                            "cross_arm_identity_error"
                            if identity_error is not None
                            else "postcondition_error"
                        ] = completion_error
                    failures.append(record)
            if runner_completion is None:
                request_matrix_stop(
                    f"cell failed for seed={job.seed} arm={job.arm}",
                    exclude=key,
                )
                return

    def worker(gpu: str, queue: Sequence[MatrixJob]) -> None:
        try:
            worker_impl(gpu, queue)
        except ProcessCleanupError as exc:
            with mutex:
                cleanup_unconfirmed.append(
                    {
                        "seed": None,
                        "arm": None,
                        "gpu": gpu,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            request_matrix_stop(
                f"matrix worker could not prove child cleanup on GPU {gpu}"
            )
        except BaseException as exc:
            with mutex:
                failures.append(
                    {
                        "seed": None,
                        "arm": None,
                        "gpu": gpu,
                        "returncode": None,
                        "worker_uncaught_exception": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
            request_matrix_stop(f"uncaught matrix worker error on GPU {gpu}")

    threads = [
        threading.Thread(target=worker, args=(gpu, queue), name=f"gpu-{gpu}", daemon=False)
        for gpu, queue in sorted(queues.items())
    ]
    started_threads: list[threading.Thread] = []

    def force_drain_registered(reason: str) -> int:
        with mutex:
            targets = list(active_processes.items())
        actions = []
        for key, process in targets:
            confirmed = False
            try:
                action = kill_verified_process_tree(
                    process, seed=key[0], arm=key[1]
                )
                actions.append(action)
                confirmed = True
            except (OSError, LaunchError) as exc:
                action = {
                    "seed": key[0],
                    "arm": key[1],
                    "outer_process_group": process.pid,
                    "signal": "SIGKILL_NOT_CONFIRMED",
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
                actions.append(action)
                with mutex:
                    if action not in cleanup_unconfirmed:
                        cleanup_unconfirmed.append(action)
            if confirmed:
                with mutex:
                    if active_processes.get(key) is process:
                        active_processes.pop(key, None)
        if actions:
            with mutex:
                matrix_stop_actions.append(
                    {
                        "requested_utc": utc_now(),
                        "reason": reason,
                        "targets": actions,
                    }
                )
        with mutex:
            return len(active_processes)

    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        for thread in started_threads:
            shutdown_timeout_recorded = False
            while thread.is_alive():
                thread.join(timeout=1.0)
                with mutex:
                    stop_started = stop_requested_monotonic
                if (
                    thread.is_alive()
                    and stop_started is not None
                    and time.monotonic() - stop_started >= 10.0
                ):
                    force_drain_registered(
                        "bounded verified drain after SIGTERM grace"
                    )
                if (
                    thread.is_alive()
                    and stop_started is not None
                    and time.monotonic() - stop_started >= 20.0
                    and not shutdown_timeout_recorded
                ):
                    with mutex:
                        failures.append(
                            {
                                "seed": None,
                                "arm": None,
                                "gpu": thread.name,
                                "returncode": None,
                                "worker_shutdown_timeout": True,
                            }
                        )
                    shutdown_timeout_recorded = True
    except BaseException as exc:
        with mutex:
            failures.append(
                {
                    "seed": None,
                    "arm": None,
                    "gpu": None,
                    "returncode": None,
                    "matrix_supervisor_exception": f"{type(exc).__name__}: {exc}",
                }
            )
        request_matrix_stop("matrix supervisor failed closed")
    finally:
        if any(thread.is_alive() for thread in started_threads):
            request_matrix_stop("matrix worker shutdown before all workers exited")
        while any(thread.is_alive() for thread in started_threads):
            for thread in started_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            with mutex:
                stop_started = stop_requested_monotonic
            if (
                stop_started is not None
                and time.monotonic() - stop_started >= 10.0
            ):
                force_drain_registered(
                    "final supervisor drain before joining all workers"
                )
        while force_drain_registered(
            "final registered-child drain before terminal receipt"
        ):
            with mutex:
                if cleanup_unconfirmed:
                    break
            time.sleep(0.1)

    def restore_matrix_signal_handlers() -> None:
        for signum, handler in previous_matrix_signal_handlers.items():
            signal.signal(signum, handler)

    with mutex:
        unresolved_cleanup = list(cleanup_unconfirmed)
        unresolved_active = sorted(active_processes)
    if unresolved_cleanup or unresolved_active:
        audit_path = matrix_dir / "matrix_cleanup_unconfirmed.json"
        atomic_json_exclusive(
            audit_path,
            {
                "schema": "ect.q256.target-weight-cleanup-unconfirmed/v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "CLEANUP_UNCONFIRMED",
                "created_utc": utc_now(),
                "matrix_plan_sha256": plan_sha256,
                "details": unresolved_cleanup,
                "active_cell_keys": [
                    {"seed": seed, "arm": arm}
                    for seed, arm in unresolved_active
                ],
                "terminal_completion_forbidden": True,
            },
        )
        restore_matrix_signal_handlers()
        raise ProcessCleanupError(
            "matrix terminal receipt is forbidden because child cleanup could "
            f"not be proven; audit={audit_path}"
        )

    completion = {
        "schema": "ect.q256.target-weight-factorial-matrix-completion/v2",
        "experiment_id": EXPERIMENT_ID,
        "finished_utc": utc_now(),
        "matrix_plan_sha256": plan_sha256,
        "completed": sorted(completed, key=lambda item: (item["seed"], list(ARMS).index(item["arm"]))),
        "skipped_existing_pass": sorted(
            skipped_pass,
            key=lambda item: (item["seed"], list(ARMS).index(item["arm"])),
        ),
        "failures": failures,
        "received_signal": received_signal,
        "stop_actions": matrix_stop_actions,
        "live_seed_identity": {
            str(seed): value for seed, value in sorted(live_seed_identity.items())
        },
    }
    if (
        failures
        or received_signal is not None
        or len(completed) + len(skipped_pass) != len(jobs)
    ):
        completion["status"] = "STOPPED_FOR_AUDIT"
        completion["not_started"] = [
            {"seed": job.seed, "arm": job.arm, "gpu": job.gpu}
            for job in jobs
            if (job.seed, job.arm)
            not in {
                (item["seed"], item["arm"])
                for item in completed + failures + skipped_pass
            }
        ]
        atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
        restore_matrix_signal_handlers()
        fail(f"matrix stopped after failure; receipt={matrix_dir / 'matrix_completion.json'}")
    try:
        completion["amp_skip_equivalence"] = verify_matrix_skip_equivalence(
            jobs,
            phase=args.phase,
            expected_skip_attempts=expected_skip_values,
        )
    except BaseException as exc:
        completion["status"] = "STOPPED_FOR_AUDIT"
        completion["post_matrix_error"] = f"{type(exc).__name__}: {exc}"
        atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
        restore_matrix_signal_handlers()
        raise
    managed_signal_set = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    if not hasattr(signal, "pthread_sigmask"):
        completion["status"] = "STOPPED_FOR_AUDIT"
        completion["post_matrix_error"] = (
            "platform cannot linearize the final signal/receipt commit"
        )
        atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
        restore_matrix_signal_handlers()
        fail("matrix platform lacks pthread_sigmask; final PASS is forbidden")
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, managed_signal_set)
    final_stopped = False
    try:
        completion["final_signal_commit_boundary_utc"] = utc_now()
        pending = signal.sigpending().intersection(managed_signal_set)
        if pending and received_signal is None:
            received_signal = {
                "received_utc": utc_now(),
                "signal": "+".join(
                    sorted(signal.Signals(signum).name for signum in pending)
                ),
                "observed_while_signals_blocked_for_final_commit": True,
            }
        completion["received_signal"] = received_signal
        if received_signal is not None or stop_event.is_set():
            completion["status"] = "STOPPED_FOR_AUDIT"
            completion["post_matrix_error"] = (
                "signal/stop request observed before final receipt commit"
            )
            final_stopped = True
        else:
            completion["status"] = "PASS"
        atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
        restore_matrix_signal_handlers()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if final_stopped:
        fail(f"matrix stopped before final commit; receipt={matrix_dir / 'matrix_completion.json'}")
    print(f"[matrix] PASS receipt={matrix_dir / 'matrix_completion.json'}", flush=True)
    return 0


def add_common_asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ECT_DATA_PATH", DEFAULT_DATA)))
    parser.add_argument(
        "--transfer", type=Path, default=Path(os.environ.get("ECT_TRANSFER_PATH", DEFAULT_TRANSFER))
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    arm = subparsers.add_parser("arm", help="launch one authorized cell")
    arm.add_argument("--phase", "--mode", dest="phase", choices=tuple(PHASES), required=True)
    arm.add_argument("--arm", choices=tuple(ARMS), required=True)
    arm.add_argument("--seed", type=int, required=True)
    arm.add_argument("--gpu", required=True, help="physical GPU index or UUID; never inferred")
    arm.add_argument("--master-port", type=int, required=True)
    arm.add_argument(
        "--stop-after-attempts",
        type=int,
        help="gate-only fresh smoke pause at the frozen 16-attempt boundary",
    )
    arm.add_argument(
        "--expected-skip-attempts",
        help=(
            "optional prospectively frozen AMP skip signature; when omitted, "
            "each arm records its observed tick-0 warm-up signature and the "
            "matrix gate requires equal skip and successful-update counts "
            "within each seed"
        ),
    )
    arm.add_argument("--authorization-receipt", type=Path)
    arm.add_argument("--resume", type=Path)
    arm.add_argument("--outdir", type=Path)
    arm.add_argument(
        "--runs-root", type=Path, default=Path(os.environ.get("ECT_FACTORIAL_RUNS_ROOT", DEFAULT_RUNS_ROOT))
    )
    arm.add_argument(
        "--python-bin", default=os.environ.get("ECT_TRAINING_PYTHON", DEFAULT_TRAINING_PYTHON)
    )
    arm.add_argument(
        "--runtime-sandbox",
        type=Path,
        default=Path(os.environ.get("ECT_RUNTIME_SANDBOX", DEFAULT_RUNTIME_SANDBOX)),
    )
    arm.add_argument(
        "--lock-root", type=Path, default=Path(os.environ.get("ECT_FACTORIAL_LOCK_ROOT", "/tmp/ect-q256-target-weight-locks"))
    )
    add_common_asset_args(arm)
    arm.set_defaults(func=run_arm)

    matrix = subparsers.add_parser("matrix", help="print or explicitly execute the frozen matrix")
    matrix.add_argument("--phase", "--mode", dest="phase", choices=tuple(PHASES), required=True)
    matrix.add_argument(
        "--seed-gpu",
        action="append",
        default=[],
        metavar="SEED=GPU",
        help="required once for every phase seed; seeds never migrate between GPUs",
    )
    matrix.add_argument("--authorization-receipt", type=Path)
    matrix.add_argument("--execute", action="store_true", help="required to start any subprocess")
    matrix.add_argument("--matrix-id")
    matrix.add_argument("--base-port", type=int, default=29800)
    matrix.add_argument(
        "--expected-skip-attempts",
        help=(
            "optional prospectively frozen AMP skip signature; when omitted, "
            "the matrix records per-arm tick-0 warm-up signatures and compares "
            "skip and successful-update counts"
        ),
    )
    matrix.add_argument("--resume-cell", action="append", default=[], metavar="SEED:ARM=STATE")
    matrix.add_argument(
        "--runs-root", type=Path, default=Path(os.environ.get("ECT_FACTORIAL_RUNS_ROOT", DEFAULT_RUNS_ROOT))
    )
    matrix.add_argument(
        "--python-bin", default=os.environ.get("ECT_TRAINING_PYTHON", DEFAULT_TRAINING_PYTHON)
    )
    matrix.add_argument(
        "--runtime-sandbox",
        type=Path,
        default=Path(os.environ.get("ECT_RUNTIME_SANDBOX", DEFAULT_RUNTIME_SANDBOX)),
    )
    matrix.add_argument(
        "--lock-root", type=Path, default=Path(os.environ.get("ECT_FACTORIAL_LOCK_ROOT", "/tmp/ect-q256-target-weight-locks"))
    )
    add_common_asset_args(matrix)
    matrix.set_defaults(func=run_matrix)

    template = subparsers.add_parser(
        "authorization-template", help="print a non-authorizing source-bound review template"
    )
    template.add_argument("--phase", choices=tuple(PHASES), required=True)
    template.add_argument("--gpu", required=True, help="idle physical GPU index or UUID for runtime probe")
    template.add_argument("--master-port", type=int, default=29790)
    template.add_argument(
        "--python-bin", default=os.environ.get("ECT_TRAINING_PYTHON", DEFAULT_TRAINING_PYTHON)
    )
    template.add_argument(
        "--runtime-sandbox",
        type=Path,
        default=Path(os.environ.get("ECT_RUNTIME_SANDBOX", DEFAULT_RUNTIME_SANDBOX)),
    )
    template.add_argument(
        "--lock-root", type=Path, default=Path(os.environ.get("ECT_FACTORIAL_LOCK_ROOT", "/tmp/ect-q256-target-weight-locks"))
    )
    add_common_asset_args(template)

    def print_template(args: argparse.Namespace) -> int:
        source = source_snapshot(require_clean=True)
        dataset = verify_asset(args.data, EXPECTED_DATASET_SHA256, "canonical dataset")
        transfer = verify_asset(args.transfer, EXPECTED_TRANSFER_SHA256, "authoritative transfer")
        runtime_command, runtime_base = runtime_prefix(args.runtime_sandbox, str(args.python_bin))
        sandbox = runtime_sandbox_fingerprint(runtime_base)
        process_env = build_process_environment(args.gpu, args.master_port)
        resolved_gpu = query_gpu(args.gpu)
        with gpu_lock(str(resolved_gpu["uuid"]), args.lock_root):
            gpu = query_gpu(args.gpu)
            if gpu["uuid"] != resolved_gpu["uuid"]:
                fail("GPU selector changed identity while acquiring the canonical UUID lock")
            assert_gpu_idle(gpu)
            runtime = runtime_environment(runtime_command, process_env)
            assert_gpu_idle(gpu)
        print(
            json.dumps(
                authorization_template(
                    phase=args.phase,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                ),
                indent=2,
            )
        )
        return 0

    template.set_defaults(func=print_template)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ProcessCleanupError as exc:
        print(
            "[run_q256_target_weight_matrix] CLEANUP_UNCONFIRMED: " f"{exc}",
            file=sys.stderr,
        )
        return PROCESS_CLEANUP_UNCONFIRMED_EXIT_CODE
    except LaunchError as exc:
        print(f"[run_q256_target_weight_matrix] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
