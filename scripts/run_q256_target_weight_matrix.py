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
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_SCRIPT = REPO_ROOT / "scripts" / "run_q256_target_weight_arm.sh"
PREREGISTRATION = (
    REPO_ROOT / "analysis" / "q256_target_weight_factorial" / "preregistration.json"
)
EXPERIMENT_ID = "q256-target-weight-factorial"
FACTORIAL_PROTOCOL = "q256_target_weight_v1"
AUTHORIZATION_SCHEMA = "ect.q256.target-weight-factorial-launch-authorization/v1"
LAUNCH_SCHEMA = "ect.q256.target-weight-factorial-launch/v1"
MATRIX_SCHEMA = "ect.q256.target-weight-factorial-matrix/v1"
EXPECTED_BRANCH = "experiment/q256-target-weight-factorial"
VALIDATION_FILENAME = "q256_target_weight_arm_validation_v1.json"
HASH_RECEIPT_FILENAME = "q256_target_weight_arm_artifact_hashes_v1.json"
VALIDATION_SCHEMA = "ect.q256.target-weight-arm-validation/v1"
HASH_RECEIPT_SCHEMA = "ect.q256.target-weight-arm-artifact-hashes/v1"
RUNNER_COMPLETION_SCHEMA = "ect.q256.target-weight-factorial-runner-completion/v1"
PLANNED_PAUSE_STATUS = "PLANNED_PAUSE_PASS"
PLANNED_PAUSE_ATTEMPTS = 16
ROLE_E_AB_PARITY_SCHEMA = "ect.q256.target-weight-role-e-ab-parity/v1"
SMOKE_MATRIX_VALIDATION_SCHEMA = (
    "ect.q256.target-weight-smoke-matrix-validation/v1"
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


def parse_expected_skip_attempts(raw: str | None, phase: str) -> list[int]:
    if raw is None:
        return []
    try:
        if raw.lstrip().startswith("["):
            values = json.loads(raw)
        else:
            values = [] if not raw.strip() else [int(item.strip()) for item in raw.split(",")]
    except (ValueError, json.JSONDecodeError) as exc:
        fail(f"invalid --expected-skip-attempts: {exc}")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        fail("expected skip attempts must be a JSON/comma list of integers")
    if values != sorted(set(values)):
        fail("expected skip attempts must be strictly increasing and unique")
    maximum = int(PHASES[phase]["expected_attempts"])
    if any(value < 1 or value > maximum for value in values):
        fail(f"expected skip attempts must lie within 1..{maximum}")
    if values:
        fail("the frozen AMP skip-attempt signature is exactly []")
    return values


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
    expected_skip_attempts: list[int],
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
        if payload.get("amp_skip_attempts") != expected_skip_attempts:
            fail(f"{label} AMP skip signature is not the frozen empty list")
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
            resolved_run_dirs: set[Path] = set()
            for arm in ARMS:
                binding = arms[arm]
                if set(binding) != {
                    "run_dir",
                    "validation_receipt_sha256",
                    "artifact_hash_receipt_sha256",
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
            cached = {"arm_bindings": arm_bindings}
            if scope_cache is not None:
                scope_cache[cache_key] = cached
        checks.update(
            mode="smoke",
            seed=3,
            arms=list(ARMS),
            source_git_head=source["git_head"],
            source_content_sha256=source["content_sha256"],
            amp_skip_attempts=list(expected_skip_attempts),
            arm_bindings=cached["arm_bindings"],
        )
    if spec["require_exact_resume"]:
        exact_resume = payload.get("exact_resume")
        if not isinstance(exact_resume, dict) or exact_resume.get("status") != "passed":
            fail(f"{label} lacks a passing top-level exact_resume object")
        checks["exact_resume_status"] = "passed"
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
) -> dict:
    """Validate an explicit PASS receipt and every gate artifact bound by it."""

    path = receipt_path.expanduser().resolve(strict=True)
    payload = _load_json(path, "authorization receipt")
    expected_skip_attempts = list(expected_skip_attempts or [])
    if expected_skip_attempts:
        fail("authorization AMP skip-attempt signature must be exactly []")
    if runtime_sandbox is None or runtime is None:
        fail("authorization validation requires the frozen sandbox and CUDA runtime")
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
        "expected_amp_skip_attempts": [],
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


def assert_gpu_idle(gpu_record: Mapping[str, object]) -> dict:
    query = "gpu_uuid,pid,process_name,used_gpu_memory"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.output or "").strip()
        fail(f"cannot audit GPU compute processes: {detail or exc}")
    busy = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = _parse_csv_line(line, 4, "compute-process")
        if fields[0] == gpu_record["uuid"]:
            busy.append({"pid": fields[1], "process_name": fields[2], "used_gpu_memory_mib": fields[3]})
    if busy:
        fail(f"selected GPU is not exclusive/idle: uuid={gpu_record['uuid']} processes={busy}")
    return {
        "checked_utc": utc_now(),
        "gpu_uuid": gpu_record["uuid"],
        "compute_process_count": 0,
        "query": query,
    }


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
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o640)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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
        "expected_amp_skip_attempts": [],
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
            expected_skip_attempts=[],
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
) -> tuple[Path, dict, dict, dict | None]:
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
        run_dir, manifest, expected_phase=phase
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
    return run_dir, manifest, auth_record, planned_pause_completion


def next_unique_name(run_dir: Path, prefix: str, suffix: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_dir / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}{suffix}"


def stream_process(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> int:
    with log_path.open("xb") as raw_log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            raw_log.write(chunk)
            raw_log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()


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
    and all(isinstance(value, int) and value > 0 for value in grid_size)
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


def build_verifier_command(
    *,
    python_bin: str | Path,
    run_dir: Path,
    phase: str,
    arm: str,
    seed: int,
    expected_skip_attempts: str | None = "[]",
    runtime_command: Sequence[str] | None = None,
) -> list[str]:
    if expected_skip_attempts not in (None, "[]"):
        fail("arm verifier AMP skip-attempt signature must be canonical []")
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
    command += [
        "--expected-skip-attempts",
        "[]" if expected_skip_attempts is None else expected_skip_attempts,
    ]
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

    expected_skip_attempts = list(expected_skip_attempts or [])
    if expected_skip_attempts:
        fail("immutable verifier AMP skip-attempt signature must be exactly []")
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
    if validation.get("amp_skip_attempts") != expected_skip_attempts:
        fail(
            "immutable verifier receipt AMP skip signature differs from "
            "the frozen empty signature"
        )
    if validation.get("amp_skip_signature_expected_value_enforced") is not True:
        fail("immutable verifier receipt did not preregister the AMP skip signature")
    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        fail("artifact-hash receipt has no bound artifacts")
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
    }


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
            )
            requested = args.outdir or default_outdir(runs_root, args.phase, args.arm, args.seed)
            run_dir = create_fresh_run_dir(requested, runs_root)
            internal_auth = copy_authorization_into_run(run_dir, authorization)
            original_manifest = None
            planned_pause_completion = None
        else:
            (
                run_dir,
                original_manifest,
                internal_auth,
                planned_pause_completion,
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
            )

        command = build_training_command(
            python_bin=python_bin,
            data=Path(dataset["resolved_path"]),
            transfer=Path(transfer["resolved_path"]),
            outdir=run_dir,
            phase=args.phase,
            arm=args.arm,
            seed=args.seed,
            resume=args.resume.expanduser().resolve(strict=True) if args.resume else None,
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
            "resume_state": str(args.resume.expanduser().resolve(strict=True)) if args.resume else None,
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
            )
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
        returncode = stream_process(command, cwd=REPO_ROOT, env=process_env, log_path=log_path)
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
        }
        if returncode != 0:
            completion["status"] = "FAILED_PROCESS"
            atomic_json_exclusive(completion_path, completion)
            fail(f"training process exited with code {returncode}; inspect {log_path}")
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
        verifier_returncode = stream_process(
            verifier_command,
            cwd=REPO_ROOT,
            env=process_env,
            log_path=verifier_log,
        )
        completion.update(
            {
                "verifier_command_argv": verifier_command,
                "verifier_log": verifier_log.name,
                "verifier_log_sha256": sha256_file(verifier_log),
                "verifier_returncode": verifier_returncode,
            }
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
            completion["status"] = "PASS"
        except LaunchError as exc:
            completion["status"] = "FAILED_VERIFIER_RECEIPT"
            completion["error"] = str(exc)
            atomic_json_exclusive(completion_path, completion)
            raise
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
    if expected_skip_attempts not in (None, "[]"):
        fail("matrix AMP skip-attempt signature must be canonical []")
    expected_skip_attempts = "[]"
    resume_cells = dict(resume_cells or {})
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


def verify_matrix_skip_identity(jobs: Sequence[MatrixJob]) -> dict:
    by_seed: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
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
        by_seed[job.seed].append((job.arm, skips))
    result = {}
    for seed, arm_values in sorted(by_seed.items()):
        reference = arm_values[0][1]
        mismatched = [(arm, skips) for arm, skips in arm_values if skips != reference]
        if mismatched:
            fail(f"arm-specific AMP skip pattern for seed {seed}: {arm_values}")
        result[str(seed)] = {"arms": [arm for arm, _ in arm_values], "skip_attempts": reference}
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
    atomic_json_exclusive(plan_path, plan)

    queues: dict[str, list[MatrixJob]] = defaultdict(list)
    for job in jobs:
        queues[job.gpu].append(job)
    stop_event = threading.Event()
    failures: list[dict] = []
    completed: list[dict] = []
    skipped_pass: list[dict] = []
    mutex = threading.Lock()

    def worker(gpu: str, queue: Sequence[MatrixJob]) -> None:
        for job in queue:
            if stop_event.is_set():
                return
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
                        stop_event.set()
                    return
                if existing is not None:
                    with mutex:
                        skipped_pass.append(
                            {
                                "seed": job.seed,
                                "arm": job.arm,
                                "gpu": gpu,
                                "status": "SKIPPED_EXISTING_IMMUTABLE_PASS",
                                "receipt": existing,
                            }
                        )
                    print(
                        f"[matrix] validated immutable PASS; skip seed={job.seed} "
                        f"arm={job.arm}",
                        flush=True,
                    )
                    continue
            print(f"[matrix] starting seed={job.seed} arm={job.arm} gpu={gpu}", flush=True)
            result = subprocess.run(job.command, cwd=REPO_ROOT, check=False)
            with mutex:
                record = {"seed": job.seed, "arm": job.arm, "gpu": gpu, "returncode": result.returncode}
                if result.returncode == 0:
                    completed.append(record)
                else:
                    failures.append(record)
                    stop_event.set()
            if result.returncode != 0:
                return

    threads = [
        threading.Thread(target=worker, args=(gpu, queue), name=f"gpu-{gpu}", daemon=False)
        for gpu, queue in sorted(queues.items())
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completion = {
        "schema": "ect.q256.target-weight-factorial-matrix-completion/v1",
        "experiment_id": EXPERIMENT_ID,
        "finished_utc": utc_now(),
        "matrix_plan_sha256": sha256_file(plan_path),
        "completed": sorted(completed, key=lambda item: (item["seed"], list(ARMS).index(item["arm"]))),
        "skipped_existing_pass": sorted(
            skipped_pass,
            key=lambda item: (item["seed"], list(ARMS).index(item["arm"])),
        ),
        "failures": failures,
    }
    if failures or len(completed) + len(skipped_pass) != len(jobs):
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
        fail(f"matrix stopped after failure; receipt={matrix_dir / 'matrix_completion.json'}")
    try:
        completion["amp_skip_identity"] = verify_matrix_skip_identity(jobs)
    except LaunchError as exc:
        completion["status"] = "STOPPED_FOR_AUDIT"
        completion["post_matrix_error"] = str(exc)
        atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
        raise
    completion["status"] = "PASS"
    atomic_json_exclusive(matrix_dir / "matrix_completion.json", completion)
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
        help="optional explicit spelling of the frozen empty AMP skip signature []",
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
        help="optional explicit spelling of the frozen empty AMP skip signature []",
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
    except LaunchError as exc:
        print(f"[run_q256_target_weight_matrix] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
