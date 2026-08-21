#!/usr/bin/env python3
"""Run the immutable Role-E q256 factorial correctness gate on one GPU.

This gate is intentionally narrower than the training smoke gate.  It proves
that the frozen factorized A/B controls reproduce the canonical production
paths, verifies the same-state/same-batch denominator gradient manipulation,
and records the exact source and CUDA runtime used for those proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import torch


SCHEMA = "ect.q256.target-weight-role-e-correctness/v2"
GIT_STATUS_COMMAND = (
    "git", "status", "--porcelain", "--untracked-files=all"
)
REQUIRED_TEST_CASES = (
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
        "test_clip_free_denominator_gradient_scaling_identities",
    ),
    (
        "tests.test_q256_target_weight_factorial.CanonicalParityTest",
        "test_cuda_amp_A_and_B_match_native_full_forward_gradients_and_rng",
    ),
)
TEST_FILES = (
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
SOURCE_FILES = (
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
    "analysis/q256_target_weight_factorial/preregistration_amendment_002.json",
) + TEST_FILES


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_output(command: Sequence[str], *, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=str(cwd), text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise GateError(
            f"command failed ({exc.returncode}): {' '.join(command)}\n{exc.output}"
        ) from exc


def verify_launcher_source_matches_head(
    repo_root: Path, launcher_source: Mapping[str, object]
) -> None:
    entries = launcher_source.get("files")
    if not isinstance(entries, list):
        raise GateError("launcher source identity lacks a file manifest")
    raw_flags = subprocess.check_output(
        ["git", "ls-files", "-v", "-z"], cwd=str(repo_root)
    )
    index_flags: dict[str, str] = {}
    for raw in raw_flags.split(b"\0"):
        if not raw:
            continue
        decoded = raw.decode("utf-8")
        if len(decoded) < 3 or decoded[1] != " ":
            raise GateError(f"invalid git ls-files -v record: {decoded!r}")
        index_flags[decoded[2:]] = decoded[0]
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateError("launcher source identity has a malformed file entry")
        relative = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise GateError("launcher source identity has an invalid file entry")
        if index_flags.get(relative) != "H":
            raise GateError(
                f"source file has an unsafe index flag: {relative}: "
                f"{index_flags.get(relative)!r}"
            )
        head_blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{relative}"],
            cwd=str(repo_root),
        )
        if hashlib.sha256(head_blob).hexdigest() != digest:
            raise GateError(f"working source differs from HEAD blob: {relative}")


def source_identity(repo_root: Path) -> dict[str, object]:
    status = checked_output(
        GIT_STATUS_COMMAND,
        cwd=repo_root,
    )
    if status:
        raise GateError("Role-E gate requires a clean committed source tree")
    files: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise GateError(f"required source file is missing or unsafe: {relative}")
        files[relative] = sha256_file(path)
    manifest = "".join(f"{name}\t{files[name]}\n" for name in sorted(files))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts import run_q256_target_weight_matrix as launcher

    launcher_source = launcher.source_snapshot(repo_root, require_clean=True)
    verify_launcher_source_matches_head(repo_root, launcher_source)
    return {
        "commit": checked_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "tree": checked_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root),
        "branch": checked_output(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo_root
        ),
        "clean": True,
        "files": files,
        "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "launcher_content_sha256": launcher_source["content_sha256"],
    }


def require_visible_gpu_uuid(expected_gpu_uuid: str) -> str:
    if not expected_gpu_uuid.startswith("GPU-"):
        raise GateError("--expected-gpu-uuid must be a full NVIDIA GPU UUID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_gpu_uuid:
        raise GateError(
            "Role-E gate requires CUDA_VISIBLE_DEVICES to equal the expected "
            f"full GPU UUID exactly: {visible!r} != {expected_gpu_uuid!r}"
        )
    return visible


def runtime_identity(expected_gpu_uuid: str) -> dict[str, object]:
    visible_gpu_uuid = require_visible_gpu_uuid(expected_gpu_uuid)
    if torch.cuda.device_count() != 1:
        raise GateError(
            "Role-E CUDA gate requires exactly one CUDA device to be visible; "
            f"found {torch.cuda.device_count()}"
        )
    query = checked_output(
        [
            "nvidia-smi",
            "-i",
            expected_gpu_uuid,
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path.cwd(),
    )
    fields = [value.strip() for value in query.split(",")]
    if len(fields) != 3 or fields[0] != expected_gpu_uuid:
        raise GateError(f"unexpected nvidia-smi identity: {query!r}")
    props = torch.cuda.get_device_properties(0)
    if props.name != fields[1]:
        raise GateError(
            f"PyTorch/NVIDIA GPU name mismatch: {props.name!r} != {fields[1]!r}"
        )
    from scripts import run_q256_target_weight_matrix as launcher

    launcher_runtime = launcher.runtime_environment(
        [sys.executable], os.environ.copy()
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": visible_gpu_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "gpu_memory_mib": int(fields[2]),
        "compute_capability": [props.major, props.minor],
        "launcher_software_sha256": launcher_runtime["software_sha256"],
        "critical_runtime_files": launcher_runtime["critical_runtime_files"],
    }


def parse_junit(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [
        (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        for case in cases
    ]
    invalid_counts = {
        f"{classname}::{name}": identities.count((classname, name))
        for classname, name in REQUIRED_TEST_CASES
        if identities.count((classname, name)) != 1
    }
    if invalid_counts:
        raise GateError(
            "required frozen correctness tests must each be collected exactly once: "
            f"{invalid_counts}"
        )
    required_failures = []
    for case in cases:
        identity = (
            case.attrib.get("classname", ""), case.attrib.get("name", "")
        )
        if identity in REQUIRED_TEST_CASES and (
            case.find("failure") is not None
            or case.find("error") is not None
            or case.find("skipped") is not None
        ):
            required_failures.append(f"{identity[0]}::{identity[1]}")
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "required_test_cases": [
            {"classname": classname, "name": name}
            for classname, name in REQUIRED_TEST_CASES
        ],
        "required_test_failures": required_failures,
    }


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def seal_existing_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def extract_head_archive(repo_root: Path, destination: Path) -> str:
    archive = destination / "source-head.tar"
    subprocess.check_call(
        ["git", "archive", "--format=tar", f"--output={archive}", "HEAD"],
        cwd=str(repo_root),
    )
    archive_sha256 = sha256_file(archive)
    execution_root = destination / "source"
    execution_root.mkdir()
    with tarfile.open(archive, "r:") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise GateError(f"unsafe member in committed source archive: {member.name}")
        handle.extractall(execution_root, members=members)
    archive.unlink()
    return archive_sha256


def run_gate(
    *, repo_root: Path, output: Path, expected_gpu_uuid: str
) -> Mapping[str, object]:
    output = output.resolve()
    try:
        output.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise GateError("gate output and evidence must be outside the source tree")
    source_before = source_identity(repo_root)
    runtime = runtime_identity(expected_gpu_uuid)
    if output.exists():
        raise GateError(f"immutable gate receipt already exists: {output}")
    evidence_dir = output.parent / f"{output.stem}.evidence"
    if evidence_dir.exists():
        raise GateError(f"immutable gate evidence directory already exists: {evidence_dir}")
    evidence_dir.mkdir(parents=True)
    log_path = evidence_dir / "pytest.log"
    junit_path = evidence_dir / "pytest.xml"
    with tempfile.TemporaryDirectory(
        prefix="q256-role-e-head-", dir=str(output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        archive_sha256 = extract_head_archive(repo_root, temporary_root)
        execution_root = temporary_root / "source"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit_path}",
            *TEST_FILES,
        ]
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(temporary_root / "pycache")
        environment["PYTHONPATH"] = str(execution_root)
        completed = subprocess.run(
            command,
            cwd=str(execution_root),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    write_exclusive(log_path, completed.stdout.encode("utf-8", errors="replace"))
    if not junit_path.is_file():
        raise GateError("pytest did not produce the required JUnit evidence")
    seal_existing_file(junit_path)
    junit = parse_junit(junit_path)
    source_after = source_identity(repo_root)
    passed = (
        completed.returncode == 0
        and junit["failures"] == 0
        and junit["errors"] == 0
        and junit["skipped"] == 0
        and not junit["required_test_failures"]
        and source_after == source_before
    )
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "scope": (
            "canonical A/B factorized parity, clip-free denominator gradient "
            "scaling identities, and q256 correctness gates"
        ),
        "source": source_before,
        "executed_git_archive_sha256": archive_sha256,
        "runtime": runtime,
        "command": command,
        "pytest_exit_code": completed.returncode,
        "junit": junit,
        "evidence": {
            "pytest_log": str(log_path),
            "pytest_log_sha256": sha256_file(log_path),
            "pytest_junit": str(junit_path),
            "pytest_junit_sha256": sha256_file(junit_path),
        },
        "assertion_contract": {
            "required_test_cases": junit["required_test_cases"],
            "all_collected_tests_passed_without_skip": passed,
        },
    }
    write_exclusive(
        output,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if not passed:
        raise GateError(f"Role-E q256 correctness gate failed; receipt={output}")
    return receipt


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        receipt = run_gate(
            repo_root=args.repo_root.resolve(strict=True),
            output=args.output,
            expected_gpu_uuid=args.expected_gpu_uuid,
        )
    except (GateError, OSError, ValueError, ET.ParseError) as exc:
        print(f"[q256-role-e-gate] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[q256-role-e-gate] PASS tests={receipt['junit']['tests']} "
        f"receipt={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
