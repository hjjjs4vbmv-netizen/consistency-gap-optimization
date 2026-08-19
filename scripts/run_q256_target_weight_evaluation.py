#!/usr/bin/env python3
"""Run the immutable formal evaluation for the q256 target/weight factorial.

The input is one completed 12-cell formal training matrix.  Every final
checkpoint is evaluated exactly once at NFE=1 and once at NFE=2.  FID and KID
share each 50,000-seed job.  This runner never reads an intermediate quality
measurement and has no checkpoint-selection path.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "q256-target-weight-formal-evaluation-v1"
PLAN_SCHEMA = "ect.q256.target-weight-evaluation-plan/v1"
JOB_LAUNCH_SCHEMA = "ect.q256.target-weight-evaluation-job-launch/v1"
JOB_RECEIPT_SCHEMA = "ect.q256.target-weight-evaluation-job-receipt/v1"
COMPLETION_SCHEMA = "ect.q256.target-weight-evaluation-completion/v1"
BLOCK_SCHEMA = "ect.q256.target-weight-sampling-block-diagnostics/v1"

TRAINING_MATRIX_SCHEMA = "ect.q256.target-weight-factorial-matrix/v1"
TRAINING_COMPLETION_SCHEMA = "ect.q256.target-weight-factorial-matrix-completion/v1"
TRAINING_VALIDATION_SCHEMA = "ect.q256.target-weight-arm-validation/v1"
TRAINING_HASH_SCHEMA = "ect.q256.target-weight-arm-artifact-hashes/v1"
TRAINING_PROTOCOL = "q256_target_weight_v1"
EXPERIMENT_ID = "q256-target-weight-factorial"
EXPECTED_BRANCH = "experiment/q256-target-weight-factorial"

VALIDATION_FILENAME = "q256_target_weight_arm_validation_v1.json"
HASH_RECEIPT_FILENAME = "q256_target_weight_arm_artifact_hashes_v1.json"
CHECKPOINT_FILENAME = "network-snapshot-latest.pkl"

DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
DEFAULT_DATASET = Path(
    "/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip"
)
INCEPTION_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
    "inception-2015-12-05.pt"
)

ARMS = ("A", "B", "C", "D")
SEEDS = (3, 4, 5)
NFE_SETTINGS = {1: [], 2: [0.821]}
METRICS = ("kid50k_full", "fid50k_full")
SAMPLE_COUNT = 50_000
SAMPLE_SEEDS = "0-49999"
METRIC_SEED = 20_260_730
BLOCK_SIZE = 5_000
BLOCK_COUNT = SAMPLE_COUNT // BLOCK_SIZE
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_TORCH_VERSION = "2.2.0a0+81ea7a4"
EXPECTED_TORCH_CUDA_VERSION = "12.3"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GPU_SELECTOR_RE = re.compile(r"^(?:[0-9]+|GPU-[A-Za-z0-9-]+)$")
_SOURCE_SUFFIXES = (".py", ".sh", ".cu", ".cpp", ".c", ".cc", ".h", ".hpp")
_SOURCE_PREFIXES = ("dnnlib/", "metrics/", "torch_utils/", "training/")
_SOURCE_EXACT = {
    "ct_eval.py",
    "scripts/collect_q256_target_weight_results.py",
    "scripts/evaluate_checkpoint.sh",
    "scripts/run_q256_target_weight_evaluation.py",
}


class EvaluationError(RuntimeError):
    """A formal-evaluation precondition or postcondition failed."""


def fail(message: str) -> None:
    raise EvaluationError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object: {path}")
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        fail(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return value


def resolve_within(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{label} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"{label} escapes its root: {raw!r}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing or a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError:
        fail(f"{label} resolves outside its root: {path}")
    return resolved


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail(f"refuse to overwrite immutable artifact: {path}")
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checked_output(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            list(args), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = str(getattr(exc, "output", "")).strip() or str(exc)
        fail(f"command failed ({shlex.join(list(args))}): {detail}")


def _selected_source(path: str) -> bool:
    return path in _SOURCE_EXACT or (
        path.startswith(_SOURCE_PREFIXES) and path.endswith(_SOURCE_SUFFIXES)
    )


def source_snapshot(*, require_clean: bool = True) -> dict[str, Any]:
    if _checked_output(["git", "rev-parse", "--is-inside-work-tree"]) != "true":
        fail(f"not a Git worktree: {REPO_ROOT}")
    status = _checked_output(
        ["git", "status", "--porcelain", "--untracked-files=all"]
    )
    if require_clean and status:
        fail(f"formal evaluator source is dirty: {'; '.join(status.splitlines()[:12])}")
    # ``git branch --show-current`` was added long after Git 1.8.3.1, which is
    # still the frozen server host version.  symbolic-ref is available there.
    branch = _checked_output(["git", "symbolic-ref", "--quiet", "--short", "HEAD"])
    if require_clean and branch != EXPECTED_BRANCH:
        fail(f"wrong evaluator branch: {branch!r} != {EXPECTED_BRANCH!r}")
    head = _checked_output(["git", "rev-parse", "HEAD"])
    if not _GIT_RE.fullmatch(head):
        fail(f"invalid Git HEAD: {head!r}")
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, stderr=subprocess.STDOUT
    )
    tracked = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    selected = sorted(item for item in tracked if _selected_source(item))
    missing = sorted(_SOURCE_EXACT - set(selected))
    if missing:
        fail(f"evaluator source snapshot is missing tracked files: {missing}")
    entries = []
    for relative in selected:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            fail(f"evaluator source is not a regular file: {path}")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "git_head": head,
        "git_branch": branch,
        "git_clean": not bool(status),
        "content_algorithm": "canonical-json-sha256-v1",
        "content_sha256": canonical_sha256(entries),
        "file_count": len(entries),
        "files": entries,
    }


def validate_hash_receipt(run_dir: Path, receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != TRAINING_HASH_SCHEMA or receipt.get("status") != "passed":
        fail(f"training artifact-hash receipt did not pass: {run_dir}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        fail(f"training artifact-hash receipt has no artifacts: {run_dir}")
    required = {
        CHECKPOINT_FILENAME,
        "launch_manifest.json",
        "training_options.json",
        VALIDATION_FILENAME,
    }
    if not required.issubset(artifacts):
        fail(f"training artifact-hash receipt is missing {sorted(required - set(artifacts))}")
    for relative, binding in artifacts.items():
        path = resolve_within(run_dir, relative, f"training artifact {relative!r}")
        if not isinstance(binding, dict):
            fail(f"invalid hash binding for {path}")
        expected_sha = require_sha256(binding.get("sha256"), f"hash for {path}")
        expected_bytes = binding.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
            fail(f"invalid byte count for {path}: {expected_bytes!r}")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            fail(f"PASS-bound training artifact changed: {path}")


def validate_training_run(run_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        fail(f"training run is not a regular directory: {run_dir}")
    validation_path = run_dir / VALIDATION_FILENAME
    hashes_path = run_dir / HASH_RECEIPT_FILENAME
    validation = load_json(validation_path, "training validation receipt")
    hashes = load_json(hashes_path, "training artifact-hash receipt")
    expected = {"status": "passed", "mode": "formal", "arm": arm, "seed": seed}
    if validation.get("schema") != TRAINING_VALIDATION_SCHEMA:
        fail(f"wrong training validation schema: {validation_path}")
    if hashes.get("schema") != TRAINING_HASH_SCHEMA:
        fail(f"wrong training hash schema: {hashes_path}")
    for field, value in expected.items():
        if validation.get(field) != value or hashes.get(field) != value:
            fail(f"training PASS identity mismatch for {field}: {run_dir}")
    validate_hash_receipt(run_dir, hashes)

    launch = load_json(run_dir / "launch_manifest.json", "training launch manifest")
    if launch.get("schema") != "ect.q256.target-weight-factorial-launch/v1":
        fail(f"wrong training launch schema: {run_dir}")
    if Path(str(launch.get("run_directory", ""))).resolve() != run_dir:
        fail(f"training launch run_directory mismatch: {run_dir}")
    training = launch.get("training")
    if not isinstance(training, dict):
        fail(f"training launch has no training contract: {run_dir}")
    exact_training = {
        "phase": "formal",
        "arm": arm,
        "seed": seed,
        "factorial_protocol": TRAINING_PROTOCOL,
        "ct_train_total_kimg": 256,
        "expected_processed_nimg": 256000,
        "expected_optimizer_attempts": 2000,
    }
    for field, value in exact_training.items():
        if training.get(field) != value:
            fail(f"training contract {field} mismatch: {run_dir}")
    assets = launch.get("assets")
    if not isinstance(assets, dict) or not isinstance(assets.get("dataset"), dict):
        fail(f"training launch lacks dataset binding: {run_dir}")
    if assets["dataset"].get("sha256") != DATASET_SHA256:
        fail(f"training used a noncanonical dataset: {run_dir}")
    source = launch.get("source")
    if not isinstance(source, dict):
        fail(f"training launch lacks source binding: {run_dir}")
    if source.get("git_branch") != EXPECTED_BRANCH or source.get("git_clean") is not True:
        fail(f"training source is not the clean frozen branch: {run_dir}")
    source_head = source.get("git_head")
    source_content = source.get("content_sha256")
    if not _GIT_RE.fullmatch(str(source_head)):
        fail(f"training source Git OID is invalid: {run_dir}")
    require_sha256(source_content, "training source content SHA256")
    if validation.get("source_git_head") != source_head:
        fail(f"training validation/source Git OID mismatch: {run_dir}")
    if validation.get("source_content_sha256") != source_content:
        fail(f"training validation/source content mismatch: {run_dir}")
    preregistration = launch.get("preregistration")
    if not isinstance(preregistration, dict):
        fail(f"training launch lacks preregistration binding: {run_dir}")
    preregistration_sha = require_sha256(
        preregistration.get("sha256"), "training preregistration SHA256"
    )
    if preregistration.get("path") != "analysis/q256_target_weight_factorial/preregistration.json":
        fail(f"training launch binds the wrong preregistration path: {run_dir}")

    checkpoint = run_dir / CHECKPOINT_FILENAME
    checkpoint_binding = hashes["artifacts"][CHECKPOINT_FILENAME]
    checkpoint_sha = require_sha256(
        checkpoint_binding.get("sha256"), "final checkpoint SHA256"
    )
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_binding["bytes"],
        "training_validation_receipt": str(validation_path),
        "training_validation_receipt_sha256": sha256_file(validation_path),
        "training_hash_receipt": str(hashes_path),
        "training_hash_receipt_sha256": sha256_file(hashes_path),
        "training_source_git_head": source_head,
        "training_source_content_sha256": source_content,
        "preregistration_path": preregistration["path"],
        "preregistration_sha256": preregistration_sha,
        "initial_common_state_sha256": require_sha256(
            validation.get("initial_common_state_sha256"),
            "training initial common-state SHA256",
        ),
        "amp_skip_attempts": validation.get("amp_skip_attempts"),
    }


def load_training_matrix(matrix_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_dir = matrix_dir.expanduser().resolve(strict=True)
    plan_path = matrix_dir / "matrix_plan.json"
    completion_path = matrix_dir / "matrix_completion.json"
    plan = load_json(plan_path, "training matrix plan")
    completion = load_json(completion_path, "training matrix completion")
    if plan.get("schema") != TRAINING_MATRIX_SCHEMA:
        fail(f"wrong training matrix schema: {plan_path}")
    if plan.get("experiment_id") != EXPERIMENT_ID or plan.get("phase") != "formal":
        fail("evaluation requires the exact formal q256 target/weight matrix")
    if plan.get("expected_cell_count") != 12:
        fail("training matrix must predeclare exactly 12 cells")
    if plan.get("mode") not in ("fresh_exact_matrix", "resume_selected_cells"):
        fail(f"unsupported training matrix mode: {plan.get('mode')!r}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 12:
        fail("training matrix plan must contain exactly 12 jobs")
    if completion.get("schema") != TRAINING_COMPLETION_SCHEMA:
        fail(f"wrong training matrix completion schema: {completion_path}")
    if completion.get("status") != "PASS" or completion.get("failures") != []:
        fail("training matrix has not completed with an exact PASS")
    if completion.get("matrix_plan_sha256") != sha256_file(plan_path):
        fail("training matrix completion does not bind the current plan")

    expected_cells = {(seed, arm) for seed in SEEDS for arm in ARMS}
    seen: set[tuple[int, str]] = set()
    cells = []
    for raw in jobs:
        if not isinstance(raw, dict):
            fail("training matrix job must be an object")
        try:
            key = (int(raw["seed"]), str(raw["arm"]))
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"malformed training matrix job: {exc}")
        if key not in expected_cells or key in seen:
            fail(f"unexpected or duplicate training matrix cell: {key}")
        seen.add(key)
        if raw.get("outdir") and raw.get("resume"):
            fail(f"training matrix cell binds both outdir and resume: {key}")
        if raw.get("outdir"):
            run_dir = Path(str(raw["outdir"]))
        elif raw.get("resume"):
            run_dir = Path(str(raw["resume"])).parent
        else:
            fail(f"training matrix cell has no resolved run: {key}")
        cells.append(validate_training_run(run_dir, key[1], key[0]))
    if seen != expected_cells:
        fail(f"incomplete training matrix: missing={sorted(expected_cells - seen)}")

    completion_cells = []
    for field in ("completed", "skipped_existing_pass"):
        values = completion.get(field, [])
        if not isinstance(values, list):
            fail(f"training completion {field} must be a list")
        for raw in values:
            if not isinstance(raw, dict):
                fail(f"training completion {field} contains a non-object")
            completion_cells.append((int(raw["seed"]), str(raw["arm"])))
    if set(completion_cells) != expected_cells or len(completion_cells) != 12:
        fail("training completion does not account for the exact 12 cells")

    heads = {cell["training_source_git_head"] for cell in cells}
    contents = {cell["training_source_content_sha256"] for cell in cells}
    preregistrations = {cell["preregistration_sha256"] for cell in cells}
    if len(heads) != 1 or len(contents) != 1 or len(preregistrations) != 1:
        fail("the 12 training cells do not share one exact source and preregistration")
    return sorted(cells, key=lambda item: (item["seed"], ARMS.index(item["arm"]))), {
        "matrix_dir": str(matrix_dir),
        "matrix_plan": str(plan_path),
        "matrix_plan_sha256": sha256_file(plan_path),
        "matrix_completion": str(completion_path),
        "matrix_completion_sha256": sha256_file(completion_path),
        "training_source_git_head": next(iter(heads)),
        "training_source_content_sha256": next(iter(contents)),
        "preregistration_path": "analysis/q256_target_weight_factorial/preregistration.json",
        "preregistration_sha256": next(iter(preregistrations)),
        "cell_count": 12,
        "selection_policy": "all_exact_final_256kimg_cells_no_intermediate_selection",
    }


def verify_dataset(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        fail(f"dataset is missing or a symlink: {path}")
    actual = sha256_file(path)
    if actual != DATASET_SHA256:
        fail(f"canonical dataset SHA256 mismatch: {actual} != {DATASET_SHA256}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def build_jobs(
    cells: Sequence[Mapping[str, Any]], output_root: Path, base_port: int
) -> list[dict[str, Any]]:
    output_root = output_root.expanduser().resolve()
    jobs = []
    for cell in cells:
        for nfe, mid_t in NFE_SETTINGS.items():
            job_id = f"seed{cell['seed']}-arm{cell['arm']}-nfe{nfe}"
            target = output_root / "jobs" / job_id
            command = [
                "bash",
                str(REPO_ROOT / "scripts" / "evaluate_checkpoint.sh"),
                "1",
                str(base_port + len(jobs)),
                str(cell["checkpoint"]),
                "--outdir",
                str(target),
                "--nosubdir",
                "--data",
                "__DATASET_PATH__",
                "--cond=False",
                "--arch=ddpmpp",
                "--precond=ct",
                "--dropout=0.2",
                "--augment=0",
                "--xflip=False",
                "--fp16=False",
                "--cache=True",
                "--workers=3",
                "--eval-batch=512",
                "--metric-generator-batch=128",
                f"--nfe={nfe}",
                *(["--mid_t=0.821"] if nfe == 2 else []),
                f"--metrics={','.join(METRICS)}",
                "--metric-repeats=1",
                f"--sample-seeds={SAMPLE_SEEDS}",
                f"--seed={METRIC_SEED}",
                "--retain-generated-artifacts",
                f"--desc={PROTOCOL}-{job_id}",
            ]
            jobs.append(
                {
                    "job_id": job_id,
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "nfe": nfe,
                    "mid_t": mid_t,
                    "checkpoint": cell["checkpoint"],
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                    "training_run": cell["run_dir"],
                    "training_validation_receipt": cell["training_validation_receipt"],
                    "training_validation_receipt_sha256": cell[
                        "training_validation_receipt_sha256"
                    ],
                    "training_hash_receipt": cell["training_hash_receipt"],
                    "training_hash_receipt_sha256": cell[
                        "training_hash_receipt_sha256"
                    ],
                    "sample_count": SAMPLE_COUNT,
                    "sample_seeds": SAMPLE_SEEDS,
                    "metric_seed": METRIC_SEED,
                    "metrics": list(METRICS),
                    "precision": "fp32",
                    "output_directory": str(target),
                    "command_argv_template": command,
                }
            )
    if len(jobs) != 24:
        fail(f"formal evaluation must contain exactly 24 jobs, got {len(jobs)}")
    return jobs


def materialize_command(job: Mapping[str, Any], dataset: Path) -> list[str]:
    return [str(dataset) if value == "__DATASET_PATH__" else value for value in job["command_argv_template"]]


def capture_runtime() -> dict[str, Any]:
    try:
        import numpy
        import scipy
        import torch
    except ImportError as exc:
        fail(f"formal evaluator runtime is incomplete: {exc}")
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": gpu_names,
    }


def validate_runtime(runtime: Mapping[str, Any]) -> None:
    exact = {
        "python_version": EXPECTED_PYTHON_VERSION,
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
        "cuda_available": True,
        "cuda_device_count": 1,
    }
    for field, expected in exact.items():
        if runtime.get(field) != expected:
            fail(
                f"frozen evaluator runtime {field} mismatch: "
                f"{runtime.get(field)!r} != {expected!r}"
            )
    names = runtime.get("cuda_device_names")
    if not isinstance(names, list) or len(names) != 1 or "A100" not in names[0]:
        fail(f"frozen evaluator runtime did not expose one A100: {names!r}")


def query_gpu(selector: str) -> dict[str, Any]:
    if not _GPU_SELECTOR_RE.fullmatch(selector):
        fail(f"invalid explicit GPU selector: {selector!r}")
    output = _checked_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    matches = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            fail(f"malformed nvidia-smi GPU row: {line!r}")
        index, uuid, name, memory = parts
        if selector in (index, uuid):
            matches.append(
                {
                    "physical_index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "memory_total_mib": int(memory),
                }
            )
    if len(matches) != 1:
        fail(f"GPU selector {selector!r} resolved to {len(matches)} devices")
    gpu = matches[0]
    if "A100" not in gpu["name"] or gpu["memory_total_mib"] < 80_000:
        fail(f"formal evaluator requires one A100 80GB GPU, got {gpu}")
    return gpu


def assert_gpu_idle(gpu: Mapping[str, Any]) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        # nvidia-smi returns a nonzero code on some drivers when no process is
        # present.  Only the explicit no-process message is admissible.
        output = str(exc.output).strip()
        if "No running processes found" not in output:
            fail(f"cannot audit GPU compute processes: {output or exc}")
        output = ""
    processes = []
    for line in output.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[0] == gpu["uuid"]:
            processes.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "name": parts[2]})
    if processes:
        fail(f"assigned GPU has compute processes; stop for audit: {processes}")
    return {"checked_utc": utc_now(), "gpu_uuid": gpu["uuid"], "compute_processes": []}


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail(f"another formal evaluator holds lock: {path}")
        record = {"path": str(path), "pid": os.getpid(), "acquired_utc": utc_now()}
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield record
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def read_metric(path: Path, metric: str, checkpoint: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing raw metric result: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"raw metric file must contain exactly one record: {path}")
    try:
        payload = json.loads(lines[0])
        value = float(payload["results"][metric])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        fail(f"malformed raw metric record {path}: {exc}")
    if payload.get("metric") != metric or payload.get("num_gpus") != 1:
        fail(f"raw metric identity mismatch: {path}")
    if not math.isfinite(value) or (metric.startswith("fid") and value < 0):
        fail(f"invalid metric value in {path}: {value}")
    snapshot_raw = payload.get("snapshot_pkl")
    if not isinstance(snapshot_raw, str):
        fail(f"raw metric lacks checkpoint binding: {path}")
    if (path.parent / snapshot_raw).resolve() != checkpoint.resolve(strict=True):
        fail(f"raw metric checkpoint mismatch: {path}")
    return {"metric": metric, "value": value, "raw_path": str(path), "raw_sha256": sha256_file(path)}


def validate_evaluation_options(
    path: Path, *, job: Mapping[str, Any], dataset: Path, checkpoint: Path
) -> None:
    options = load_json(path, "evaluation options")
    dataset_kwargs = options.get("dataset_kwargs")
    network_kwargs = options.get("network_kwargs")
    if not isinstance(dataset_kwargs, dict) or not isinstance(network_kwargs, dict):
        fail(f"evaluation options lack dataset/network contracts: {path}")
    exact = {
        "batch_size": 512,
        "seed": METRIC_SEED,
        "metrics": list(METRICS),
        "metric_repeats": 1,
        "retain_generated_artifacts": True,
        "metric_generator_batch": 128,
    }
    for field, value in exact.items():
        if options.get(field) != value:
            fail(f"evaluation option {field} mismatch: {path}")
    if options.get("mid_t") != job["mid_t"]:
        fail(f"evaluation NFE/mid_t mismatch: {path}")
    seeds = options.get("sample_seeds")
    if seeds != list(range(SAMPLE_COUNT)):
        fail(f"evaluation sample seeds are not exactly 0-49999: {path}")
    if Path(str(options.get("resume_pkl", ""))).resolve() != checkpoint.resolve(strict=True):
        fail(f"evaluation checkpoint mismatch: {path}")
    if Path(str(dataset_kwargs.get("path", ""))).resolve() != dataset.resolve(strict=True):
        fail(f"evaluation dataset path mismatch: {path}")
    expected_dataset = {"use_labels": False, "xflip": False, "resolution": 32, "max_size": 50000}
    for field, value in expected_dataset.items():
        if dataset_kwargs.get(field) != value:
            fail(f"evaluation dataset option {field} mismatch: {path}")
    if network_kwargs.get("class_name") != "training.networks.ECMPrecond":
        fail(f"evaluation network class mismatch: {path}")
    if network_kwargs.get("use_fp16") is not False or network_kwargs.get("dropout") != 0.2:
        fail(f"evaluation precision/dropout mismatch: {path}")


def build_sampling_block_diagnostics(
    features_path: Path, output_path: Path, *, sample_count: int = SAMPLE_COUNT,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    """Describe fixed contiguous seed-block feature variation without inference."""
    try:
        import numpy as np
    except ImportError as exc:
        fail(f"NumPy is required for block diagnostics: {exc}")
    try:
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        fail(f"cannot load retained Inception features {features_path}: {exc}")
    if features.ndim != 2 or features.shape[0] != sample_count or features.shape[1] <= 0:
        fail(f"unexpected retained feature shape: {features.shape}")
    if sample_count % block_size != 0:
        fail("sampling diagnostic requires equal fixed blocks")
    full_mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    blocks = []
    for start in range(0, sample_count, block_size):
        stop = start + block_size
        block = np.asarray(features[start:stop], dtype=np.float64)
        if not bool(np.isfinite(block).all()):
            fail(f"non-finite retained features in block {start}-{stop - 1}")
        block_mean = block.mean(axis=0)
        centered = block - block_mean
        blocks.append(
            {
                "block_index": start // block_size,
                "sample_seed_start": start,
                "sample_seed_end": stop - 1,
                "sample_count": block_size,
                "feature_mean_l2_distance_from_full": float(
                    np.linalg.norm(block_mean - full_mean)
                ),
                "feature_variance_trace": float(
                    np.square(centered).sum(dtype=np.float64) / block_size
                ),
            }
        )
    payload = {
        "schema": BLOCK_SCHEMA,
        "status": "descriptive_variation_only",
        "feature_source": features_path.name,
        "feature_source_sha256": sha256_file(features_path),
        "sample_seed_range": f"0-{sample_count - 1}",
        "sample_count": sample_count,
        "fixed_block_size": block_size,
        "fixed_block_count": len(blocks),
        "independent_training_replicate_contribution": 0,
        "quality_endpoint": False,
        "selection_criterion": False,
        "blocks": blocks,
    }
    write_json_exclusive(output_path, payload)
    return payload


def hash_regular_tree(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail(f"evaluation output contains a symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(root))
            artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not artifacts:
        fail(f"evaluation output is empty: {root}")
    return artifacts


def hash_cache_tree(cache_root: Path) -> dict[str, Any]:
    artifacts = hash_regular_tree(cache_root)
    return {
        "root": str(cache_root),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tree_sha256": canonical_sha256(artifacts),
        "inception_detector_url": INCEPTION_URL,
    }


def validate_job_outputs(
    job: Mapping[str, Any], *, dataset: Path, output_root: Path
) -> dict[str, Any]:
    target = Path(job["output_directory"])
    if target.is_symlink() or not target.is_dir():
        fail(f"evaluation job produced no regular output directory: {target}")
    checkpoint = Path(job["checkpoint"])
    if sha256_file(checkpoint) != job["checkpoint_sha256"]:
        fail(f"checkpoint changed during evaluation: {checkpoint}")
    options_path = target / "training_options.json"
    validate_evaluation_options(options_path, job=job, dataset=dataset, checkpoint=checkpoint)
    log_path = target / "log.txt"
    if not log_path.is_file():
        fail(f"evaluation job lacks log.txt: {target}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Exiting..." not in log_text or "Traceback (most recent call last)" in log_text:
        fail(f"evaluation job lacks a clean completion marker: {target}")
    metrics = [
        read_metric(target / f"metric-{metric}.jsonl", metric, checkpoint)
        for metric in METRICS
    ]
    kid_features = target / "generated-features-kid50k_full-repeat00.npy"
    fid_features = target / "generated-features-fid50k_full-repeat00.npy"
    samples = target / "generated-samples.npy"
    for path in (kid_features, fid_features, samples):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            fail(f"missing retained formal evaluation artifact: {path}")
    kid_sha = sha256_file(kid_features)
    fid_sha = sha256_file(fid_features)
    if kid_sha != fid_sha:
        fail("FID and KID did not use bit-identical retained Inception features")
    block_path = target / "sampling_block_diagnostics_v1.json"
    block = build_sampling_block_diagnostics(fid_features, block_path)
    if block["fixed_block_count"] != BLOCK_COUNT:
        fail("formal sampling diagnostics do not contain the exact ten 5k blocks")
    artifacts = hash_regular_tree(target)
    required = {
        "training_options.json",
        "log.txt",
        "metric-kid50k_full.jsonl",
        "metric-fid50k_full.jsonl",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
        "generated-samples.npy",
        "sampling_block_diagnostics_v1.json",
    }
    if not required.issubset(artifacts):
        fail(f"evaluation output is incomplete: {sorted(required - set(artifacts))}")
    return {
        "metrics": metrics,
        "sampling_block_diagnostics": {
            "path": str(block_path),
            "sha256": sha256_file(block_path),
        },
        "artifacts": artifacts,
        "artifacts_tree_sha256": canonical_sha256(artifacts),
        "cache": hash_cache_tree(output_root / "evaluator_cache"),
    }


def stream_process(
    command: Sequence[str], *, env: Mapping[str, str], log_path: Path
) -> int:
    try:
        log = log_path.open("x", encoding="utf-8")
    except FileExistsError:
        fail(f"refuse to overwrite evaluator process log: {log_path}")
    with log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
        os.fsync(log.fileno())
    return returncode


def build_plan(
    *, matrix: Mapping[str, Any], cells: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any], output_root: Path, source: Mapping[str, Any],
    runtime: Mapping[str, Any], gpu: Mapping[str, Any], base_port: int,
    locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    jobs = build_jobs(cells, output_root, base_port)
    return {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "status": "authorized_exact_matrix",
        "selection_policy": "all_12_final_256kimg_checkpoints_no_intermediate_selection",
        "independent_unit": {"name": "training_seed", "values": list(SEEDS), "n": 3},
        "training_matrix": dict(matrix),
        "training_cells": list(cells),
        "dataset": dict(dataset),
        "evaluator_source": dict(source),
        "runtime": dict(runtime),
        "gpu": dict(gpu),
        "locks": list(locks),
        "precision": "fp32",
        "sample_count_per_job": SAMPLE_COUNT,
        "sample_seed_range": SAMPLE_SEEDS,
        "metric_seed": METRIC_SEED,
        "metrics_per_job": list(METRICS),
        "metric_repeats": 1,
        "nfe_modes": {str(key): value for key, value in NFE_SETTINGS.items()},
        "job_count": 24,
        "jobs": jobs,
    }


def execute(args: argparse.Namespace) -> int:
    matrix_dir = args.matrix_dir.expanduser().resolve(strict=True)
    output_root = args.outdir.expanduser().resolve()
    if output_root.exists():
        fail(f"refuse to reuse or overwrite evaluation root: {output_root}")
    if not output_root.parent.is_dir():
        fail(f"evaluation parent directory does not exist: {output_root.parent}")
    cells, matrix = load_training_matrix(matrix_dir)
    dataset = verify_dataset(args.data)
    source = source_snapshot(require_clean=True)
    gpu = query_gpu(args.gpu)
    lock_root = args.lock_root.expanduser().resolve()
    output_lock_path = lock_root / f"output-{hashlib.sha256(str(output_root).encode()).hexdigest()}.lock"
    gpu_lock_path = lock_root / f"gpu-{gpu['uuid']}.lock"

    with exclusive_lock(output_lock_path) as output_lock, exclusive_lock(gpu_lock_path) as gpu_lock:
        if output_root.exists():
            fail(f"evaluation root appeared while acquiring lock: {output_root}")
        initial_idle = assert_gpu_idle(gpu)
        process_env = dict(os.environ)
        process_env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "DNNLIB_CACHE_DIR": str(output_root / "evaluator_cache"),
            }
        )
        # This process has deliberately not imported Torch yet.  Apply the
        # same one-device visibility before capture_runtime() imports it so
        # both the runner receipt and every child observe the identical GPU.
        for key in (
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "PYTHONUNBUFFERED",
            "PYTHONDONTWRITEBYTECODE",
            "DNNLIB_CACHE_DIR",
        ):
            os.environ[key] = process_env[key]
        runtime = capture_runtime()
        validate_runtime(runtime)
        locks = [output_lock, gpu_lock]
        output_root.mkdir(mode=0o750)
        for name in ("manifests", "receipts", "process_logs", "jobs"):
            (output_root / name).mkdir()
        plan = build_plan(
            matrix=matrix,
            cells=cells,
            dataset=dataset,
            output_root=output_root,
            source=source,
            runtime=runtime,
            gpu=gpu,
            base_port=args.base_port,
            locks=locks,
        )
        plan["initial_gpu_idle_check"] = initial_idle
        plan_path = output_root / "evaluation_plan.json"
        write_json_exclusive(plan_path, plan)
        plan_sha = sha256_file(plan_path)
        completed = []
        cache_sha: str | None = None

        for ordinal, job in enumerate(plan["jobs"], start=1):
            target = Path(job["output_directory"])
            if target.exists():
                fail(f"refuse to append to evaluation job output: {target}")
            if source_snapshot(require_clean=True)["content_sha256"] != source["content_sha256"]:
                fail("evaluator source changed after plan authorization")
            if verify_dataset(args.data)["sha256"] != dataset["sha256"]:
                fail("dataset changed after plan authorization")
            if sha256_file(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
                fail(f"checkpoint changed before job {job['job_id']}")
            idle = assert_gpu_idle(gpu)
            command = materialize_command(job, Path(dataset["path"]))
            launch = {
                "schema": JOB_LAUNCH_SCHEMA,
                "protocol": PROTOCOL,
                "status": "authorized_to_start",
                "created_utc": utc_now(),
                "ordinal": ordinal,
                "job": job,
                "evaluation_plan": plan_path.name,
                "evaluation_plan_sha256": plan_sha,
                "dataset": dataset,
                "evaluator_source_git_head": source["git_head"],
                "evaluator_source_content_sha256": source["content_sha256"],
                "gpu": gpu,
                "gpu_idle_check": idle,
                "exact_command_argv": command,
                "exact_command_shell": shlex.join(command),
            }
            launch_path = output_root / "manifests" / f"{job['job_id']}.json"
            write_json_exclusive(launch_path, launch)
            log_path = output_root / "process_logs" / f"{job['job_id']}.log"
            print(f"[{ordinal}/24] {job['job_id']}", flush=True)
            started = time.time()
            returncode = stream_process(command, env=process_env, log_path=log_path)
            base_receipt: dict[str, Any] = {
                "schema": JOB_RECEIPT_SCHEMA,
                "protocol": PROTOCOL,
                "job_id": job["job_id"],
                "seed": job["seed"],
                "arm": job["arm"],
                "nfe": job["nfe"],
                "mid_t": job["mid_t"],
                "checkpoint": job["checkpoint"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                "dataset_sha256": dataset["sha256"],
                "evaluator_source_git_head": source["git_head"],
                "evaluator_source_content_sha256": source["content_sha256"],
                "sample_count": SAMPLE_COUNT,
                "sample_seed_range": SAMPLE_SEEDS,
                "metric_seed": METRIC_SEED,
                "precision": "fp32",
                "launch_manifest": str(launch_path),
                "launch_manifest_sha256": sha256_file(launch_path),
                "process_log": str(log_path),
                "process_log_sha256": sha256_file(log_path),
                "returncode": returncode,
                "elapsed_seconds": round(time.time() - started, 3),
                "finished_utc": utc_now(),
            }
            receipt_path = output_root / "receipts" / f"{job['job_id']}.json"
            if returncode != 0:
                base_receipt["status"] = "failed_process"
                write_json_exclusive(receipt_path, base_receipt)
                write_json_exclusive(
                    output_root / "evaluation_completion.json",
                    {
                        "schema": COMPLETION_SCHEMA,
                        "protocol": PROTOCOL,
                        "status": "STOPPED_FOR_AUDIT",
                        "evaluation_plan_sha256": plan_sha,
                        "completed_job_ids": completed,
                        "failed_job_id": job["job_id"],
                        "failed_receipt": str(receipt_path),
                    },
                )
                fail(f"evaluation process failed for {job['job_id']}; no later job started")
            try:
                outputs = validate_job_outputs(
                    job, dataset=Path(dataset["path"]), output_root=output_root
                )
            except EvaluationError as exc:
                base_receipt.update({"status": "failed_postcondition", "error": str(exc)})
                write_json_exclusive(receipt_path, base_receipt)
                write_json_exclusive(
                    output_root / "evaluation_completion.json",
                    {
                        "schema": COMPLETION_SCHEMA,
                        "protocol": PROTOCOL,
                        "status": "STOPPED_FOR_AUDIT",
                        "evaluation_plan_sha256": plan_sha,
                        "completed_job_ids": completed,
                        "failed_job_id": job["job_id"],
                        "failed_receipt": str(receipt_path),
                        "error": str(exc),
                    },
                )
                raise
            if cache_sha is None:
                cache_sha = outputs["cache"]["tree_sha256"]
            elif outputs["cache"]["tree_sha256"] != cache_sha:
                fail("canonical evaluator cache changed after the first complete job")
            base_receipt.update({"status": "passed", **outputs})
            write_json_exclusive(receipt_path, base_receipt)
            completed.append(job["job_id"])

        completion = {
            "schema": COMPLETION_SCHEMA,
            "protocol": PROTOCOL,
            "experiment_id": EXPERIMENT_ID,
            "status": "PASS",
            "finished_utc": utc_now(),
            "evaluation_plan": str(plan_path),
            "evaluation_plan_sha256": plan_sha,
            "job_count": 24,
            "completed_job_ids": completed,
            "failed_job_id": None,
            "cache_tree_sha256": cache_sha,
            "selection_policy": "all_12_final_256kimg_checkpoints_no_intermediate_selection",
            "independent_training_seed_n": 3,
        }
        write_json_exclusive(output_root / "evaluation_completion.json", completion)
        print(f"Formal evaluation PASS: {output_root / 'evaluation_completion.json'}")
        return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="explicit physical GPU index or UUID")
    parser.add_argument("--base-port", type=int, default=31_800)
    parser.add_argument(
        "--lock-root", type=Path, default=Path("/data/temp/ECT001-q256-evaluation-locks")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return execute(args)
    except EvaluationError as exc:
        print(f"[q256-target-weight-evaluation] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
