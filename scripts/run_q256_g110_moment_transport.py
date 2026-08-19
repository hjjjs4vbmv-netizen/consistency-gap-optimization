#!/usr/bin/env python3
"""Fail-closed orchestration for the q256 g=1.10 moment-transport study.

The runner is intentionally a thin wrapper around ``ct_train.py``.  It never
changes training semantics: a frozen JSON manifest supplies the source
checkpoints, immutable identities, seed coefficients, runtime command prefix,
and the exact training protocol.  Four phases are exposed:

``prepare``
    Verify every immutable input, copy the matching EMA snapshots, and create
    one a=1 and one real-a transformed source for every seed.
``smoke``
    Run seed 3 (or the manifest's frozen smoke seed) four times for 32 updates:
    ordinary/no-op-transform and two independent real-a repeats.  Frozen
    validator/comparator commands must issue GO receipts.
``formal``
    After all external and internal GO gates, run the compatibility-frozen fresh
    arms (G/T or F/G/T) for seeds 3, 4, and 5 at 512, 768, and 1024 kimg.  F is
    the unmodified fixed schedule, G is unmodified global g=1.10, and T is the
    transported global g=1.10 arm.  Every selected arm for a seed uses the same
    explicitly selected GPU and serial queue.
``status``
    Read plans, receipts, and tmux state without changing anything.

The manifest schema is ``ect.q256.g110-moment-transport-runner/v1``.  Commands
are argument arrays and are never evaluated by a shell.  Template commands for
checkpoint validation/comparison may use the placeholders documented in
``render_template_command`` below.  A successful command has immutable stdout,
stderr, command, and exit receipts; interrupted or failed output is never
silently reused or overwritten.
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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence


SCHEMA = "ect.q256.g110-moment-transport-runner/v1"
PLAN_SCHEMA = "ect.q256.g110-moment-transport-plan/v1"
EXIT_SCHEMA = "ect.q256.g110-moment-transport-exit/v1"
GATE_SCHEMA = "ect.q256.g110-moment-transport-gate/v1"
PREPARE_SCHEMA = "ect.q256.g110-moment-transport-prepare/v1"
EXPECTED_SEEDS = (3, 4, 5)
EXPECTED_ENDPOINTS = (512, 768, 1024)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class RunnerError(RuntimeError):
    """Raised when an invariant cannot be established."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RunnerError(f"{label} must be a JSON array")
    return value


def _require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{label} must be a non-empty string path")
    path = Path(value)
    if not path.is_absolute():
        raise RunnerError(f"{label} must be absolute: {path}")
    return path


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise RunnerError(f"{label} must be a 64-character SHA256 digest")
    return value.lower()


def load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read {label} {path}: {exc}") from exc
    return _require_mapping(value, label)


def _json_value(payload: Mapping[str, Any], dotted_key: str, label: str) -> Any:
    if not isinstance(dotted_key, str) or not dotted_key:
        raise RunnerError(f"{label} verdict_key must be a non-empty string")
    value: Any = payload
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise RunnerError(f"{label} lacks verdict field {dotted_key!r}")
        value = value[component]
    return value


def _safe_child(root: Path, *parts: str) -> Path:
    for part in parts:
        if not isinstance(part, str) or not part or part in (".", ".."):
            raise RunnerError(f"unsafe output path component: {part!r}")
        candidate = Path(part)
        if candidate.is_absolute() or len(candidate.parts) != 1:
            raise RunnerError(f"unsafe output path component: {part!r}")
    resolved_root = root.resolve(strict=False)
    child = resolved_root.joinpath(*parts).resolve(strict=False)
    try:
        child.relative_to(resolved_root)
    except ValueError as exc:
        raise RunnerError(f"output escapes run_root: {child}") from exc
    return child


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RunnerError(f"no existing parent for {path}")
        candidate = candidate.parent
    return candidate


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if path.exists() or path.is_symlink():
        raise RunnerError(f"refusing to overwrite receipt: {path}")
    temp_path: Path | None = None
    try:
        for counter in range(1000):
            candidate = path.parent / (
                f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}-{counter}"
            )
            try:
                handle = candidate.open("xb")
                temp_path = candidate
                break
            except FileExistsError:
                continue
        else:
            raise RunnerError(f"cannot allocate atomic temporary receipt beside {path}")
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RunnerError(f"refusing to overwrite receipt: {path}") from exc
        except OSError as exc:
            raise RunnerError(
                f"atomic no-replace publication failed for receipt {path}: {exc}"
            ) from exc
        try:
            directory_fd = os.open(
                str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an immutable plan, or prove an existing plan is identical."""
    if path.is_symlink():
        raise RunnerError(f"immutable JSON must not be a symlink: {path}")
    if path.exists():
        existing = load_json(path, "existing immutable JSON")
        if canonical_json(existing) != canonical_json(payload):
            raise RunnerError(
                f"existing immutable JSON differs from requested payload: {path}"
            )
        return
    try:
        _write_new_json(path, payload)
    except RunnerError as exc:
        # Multiple GPU workers may publish the same phase plan concurrently.
        # Losing the no-replace race is safe only after reading the winner and
        # proving byte-independent canonical equality.
        if not path.exists() or path.is_symlink():
            raise
        existing = load_json(path, "concurrently published immutable JSON")
        if canonical_json(existing) != canonical_json(payload):
            raise RunnerError(
                f"concurrently published immutable JSON differs from requested payload: {path}"
            ) from exc


def _artifact(spec: Any, label: str, *, verify: bool) -> dict[str, Any]:
    raw = _require_mapping(spec, label)
    path = _require_absolute_path(raw.get("path"), f"{label}.path")
    expected = _require_sha256(raw.get("sha256"), f"{label}.sha256")
    result = {"path": str(path), "sha256": expected}
    if verify:
        if not path.is_file() or path.is_symlink():
            raise RunnerError(f"{label} is not a regular non-symlink file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RunnerError(
                f"{label} SHA256 mismatch for {path}: observed {observed}, expected {expected}"
            )
        result["observed_sha256"] = observed
        result["size_bytes"] = path.stat().st_size
    return result


def _command_array(value: Any, label: str) -> list[str]:
    raw = _require_sequence(value, label)
    command = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item or "\x00" in item:
            raise RunnerError(f"{label}[{index}] must be a non-empty NUL-free string")
        command.append(item)
    if not command:
        raise RunnerError(f"{label} must not be empty")
    return command


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def _number_arg(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError(f"training numeric value is invalid: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RunnerError(f"training numeric value is non-finite: {value!r}")
    if isinstance(value, int):
        return str(value)
    # Python's repr is the shortest decimal that round-trips to the frozen
    # binary float (e.g. 1.1 rather than 1.1000000000000001).
    return repr(numeric)


def _assert_exact_protocol(training: Mapping[str, Any]) -> None:
    expected: Mapping[str, Any] = {
        "source_kimg": 256,
        "endpoints_kimg": list(EXPECTED_ENDPOINTS),
        "smoke_steps": 32,
        "cond": False,
        "arch": "ddpmpp",
        "precond": "ect",
        "batch": 128,
        "batch_gpu": 16,
        "optim": "RAdam",
        "lr": 0.0001,
        "dropout": 0.2,
        "augment": 0,
        "xflip": False,
        "mapping": "global_sigmoid",
        "global_gap_scale": 1.10,
        "q": 256,
        "k": 8,
        "b": 1,
        "c": 0,
        "double": 10000,
        "ema_beta": 0.9993,
        "fp16": True,
        "enable_amp": True,
        "tf32": False,
        "ls": 1.0,
        "metrics": "none",
        "tick": 10,
        "snap": 0,
        "dump": 0,
        "ckpt": 10,
        "sample_every": 26,
        "eval_every": 50,
        "adaptive_update_kimg": 0.5,
    }
    for key, expected_value in expected.items():
        if key not in training:
            raise RunnerError(f"training protocol is missing frozen field {key!r}")
        actual = training[key]
        if isinstance(expected_value, float):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise RunnerError(f"training.{key} must be numeric")
            if float(actual) != expected_value:
                raise RunnerError(
                    f"training.{key} differs from frozen protocol: {actual!r} != {expected_value!r}"
                )
        elif actual != expected_value:
            raise RunnerError(
                f"training.{key} differs from frozen protocol: {actual!r} != {expected_value!r}"
            )
    permitted = set(expected) | {"workers", "bench", "cache"}
    unknown = sorted(set(training) - permitted)
    if unknown:
        raise RunnerError(f"training protocol has unsupported fields: {unknown}")
    for optional in ("workers",):
        if optional in training and (
            isinstance(training[optional], bool)
            or not isinstance(training[optional], int)
            or training[optional] <= 0
        ):
            raise RunnerError(f"training.{optional} must be a positive integer")
    for optional in ("bench", "cache"):
        if optional in training and not isinstance(training[optional], bool):
            raise RunnerError(f"training.{optional} must be boolean")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Return a normalized manifest or raise before any output is created."""
    if manifest.get("schema") != SCHEMA:
        raise RunnerError(f"manifest schema must be {SCHEMA!r}")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not ID_RE.fullmatch(experiment_id):
        raise RunnerError("experiment_id contains unsafe characters")

    paths = _require_mapping(manifest.get("paths"), "paths")
    repo_root = _require_absolute_path(paths.get("repo_root"), "paths.repo_root")
    run_root = _require_absolute_path(paths.get("run_root"), "paths.run_root")
    transformer = _require_absolute_path(paths.get("transformer"), "paths.transformer")
    if run_root == Path("/") or run_root.resolve(strict=False) == repo_root.resolve(
        strict=False
    ):
        raise RunnerError("run_root must be a dedicated directory, not / or repo_root")
    if run_root.exists() and run_root.is_symlink():
        raise RunnerError(f"run_root cannot be a symlink: {run_root}")
    if verify_artifacts:
        if not repo_root.is_dir():
            raise RunnerError(f"repo_root is not a directory: {repo_root}")
        if not transformer.is_file() or transformer.is_symlink():
            raise RunnerError(f"transformer is not a regular file: {transformer}")

    runtime = _require_mapping(manifest.get("runtime"), "runtime")
    python_command = _command_array(
        runtime.get("python_command"), "runtime.python_command"
    )
    worker_command = _command_array(
        runtime.get("worker_command"), "runtime.worker_command"
    )
    git_command = _command_array(
        runtime.get("git_command", ["git"]), "runtime.git_command"
    )
    nvidia_command = _command_array(
        runtime.get("nvidia_smi_command", ["nvidia-smi"]),
        "runtime.nvidia_smi_command",
    )
    tmux_binary = runtime.get("tmux_binary", "tmux")
    if not isinstance(tmux_binary, str) or not tmux_binary or "/" in tmux_binary:
        raise RunnerError(
            "runtime.tmux_binary must be a command name without path separators"
        )
    master_port_base = runtime.get("master_port_base")
    if (
        isinstance(master_port_base, bool)
        or not isinstance(master_port_base, int)
        or not 1024 <= master_port_base <= 65533
    ):
        raise RunnerError("runtime.master_port_base must be an integer in [1024,65533]")

    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    execution_commit = provenance.get("execution_commit")
    if not isinstance(execution_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", execution_commit
    ):
        raise RunnerError(
            "provenance.execution_commit must be a lowercase 40-hex commit"
        )
    dataset = _artifact(
        provenance.get("dataset"), "provenance.dataset", verify=verify_artifacts
    )
    container_identity = _artifact(
        provenance.get("container_identity"),
        "provenance.container_identity",
        verify=verify_artifacts,
    )
    core_raw = _require_sequence(provenance.get("core_code"), "provenance.core_code")
    if not core_raw:
        raise RunnerError("provenance.core_code must not be empty")
    core_code = [
        _artifact(item, f"provenance.core_code[{index}]", verify=verify_artifacts)
        for index, item in enumerate(core_raw)
    ]
    core_paths = {Path(item["path"]).resolve(strict=False) for item in core_code}
    required_core = {
        (repo_root / "ct_train.py").resolve(strict=False),
        (repo_root / "training" / "ct_training_loop.py").resolve(strict=False),
        (repo_root / "training" / "loss.py").resolve(strict=False),
        (repo_root / "training" / "schedules.py").resolve(strict=False),
        (repo_root / "training" / "networks.py").resolve(strict=False),
        (repo_root / "training" / "dataset.py").resolve(strict=False),
        (repo_root / "scripts" / "run_q256_g110_moment_transport.py").resolve(
            strict=False
        ),
        transformer.resolve(strict=False),
    }
    if not required_core.issubset(core_paths):
        missing = sorted(str(item) for item in required_core - core_paths)
        raise RunnerError(f"core_code is missing required training files: {missing}")

    training = _require_mapping(manifest.get("training"), "training")
    _assert_exact_protocol(training)

    resources = _require_mapping(manifest.get("resources"), "resources")
    min_disk = resources.get("min_free_disk_bytes")
    min_gpu = resources.get("min_free_gpu_mib")
    max_util = resources.get("max_gpu_utilization_pct")
    if isinstance(min_disk, bool) or not isinstance(min_disk, int) or min_disk <= 0:
        raise RunnerError("resources.min_free_disk_bytes must be a positive integer")
    if isinstance(min_gpu, bool) or not isinstance(min_gpu, int) or min_gpu <= 0:
        raise RunnerError("resources.min_free_gpu_mib must be a positive integer")
    if (
        isinstance(max_util, bool)
        or not isinstance(max_util, int)
        or not 0 <= max_util <= 100
    ):
        raise RunnerError(
            "resources.max_gpu_utilization_pct must be an integer in [0,100]"
        )

    smoke = _require_mapping(manifest.get("smoke"), "smoke")
    smoke_seed = smoke.get("seed")
    if smoke_seed not in EXPECTED_SEEDS:
        raise RunnerError(f"smoke.seed must be one of {EXPECTED_SEEDS}")
    validator_template = _command_array(
        smoke.get("validator_command"), "smoke.validator_command"
    )
    comparator_template = _command_array(
        smoke.get("comparator_command"), "smoke.comparator_command"
    )
    validator_gate = _normalize_result_gate(
        smoke.get("validator_gate"), "smoke.validator_gate"
    )
    comparator_gate = _normalize_result_gate(
        smoke.get("comparator_gate"), "smoke.comparator_gate"
    )

    formal = _require_mapping(manifest.get("formal"), "formal")
    arms_raw = _require_sequence(formal.get("arms"), "formal.arms")
    arms = list(arms_raw)
    if arms not in (["G", "T"], ["F", "G", "T"]):
        raise RunnerError('formal.arms must be exactly ["G","T"] or ["F","G","T"]')
    compatibility_report = _artifact(
        formal.get("compatibility_report"),
        "formal.compatibility_report",
        verify=verify_artifacts,
    )
    formal_validator_template = _command_array(
        formal.get("validator_command"), "formal.validator_command"
    )
    formal_validator_gate = _normalize_result_gate(
        formal.get("validator_gate"), "formal.validator_gate"
    )
    for label, command in (
        ("smoke.validator_command", validator_template),
        ("smoke.comparator_command", comparator_template),
        ("formal.validator_command", formal_validator_template),
    ):
        script_paths = {
            Path(argument).resolve(strict=False)
            for argument in command
            if Path(argument).is_absolute() and Path(argument).suffix == ".py"
        }
        missing_scripts = sorted(str(path) for path in script_paths - core_paths)
        if missing_scripts:
            raise RunnerError(f"{label} executes unhashed scripts: {missing_scripts}")

    seeds_raw = _require_sequence(manifest.get("seeds"), "seeds")
    if len(seeds_raw) != len(EXPECTED_SEEDS):
        raise RunnerError(f"manifest must contain exactly seeds {EXPECTED_SEEDS}")
    seeds = []
    seen = set()
    for index, item in enumerate(seeds_raw):
        raw = _require_mapping(item, f"seeds[{index}]")
        seed = raw.get("seed")
        if seed not in EXPECTED_SEEDS or seed in seen:
            raise RunnerError(f"seeds[{index}].seed is invalid or duplicated: {seed!r}")
        seen.add(seed)
        gpu = raw.get("gpu")
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise RunnerError(f"seeds[{index}].gpu must be a nonnegative integer")
        coefficient = raw.get("coefficient")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise RunnerError(f"seeds[{index}].coefficient must be numeric")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise RunnerError(f"seeds[{index}].coefficient must be finite and > 0")
        source_state = _artifact(
            raw.get("source_state"),
            f"seeds[{index}].source_state",
            verify=verify_artifacts,
        )
        source_snapshot = _artifact(
            raw.get("source_snapshot"),
            f"seeds[{index}].source_snapshot",
            verify=verify_artifacts,
        )
        state_path = Path(source_state["path"])
        snapshot_path = Path(source_snapshot["path"])
        match = re.fullmatch(r"training-state-(\d+|latest)\.pt", state_path.name)
        if not match:
            raise RunnerError(
                f"source state has an invalid resume filename: {state_path}"
            )
        expected_snapshot_name = f"network-snapshot-{match.group(1)}.pkl"
        if (
            snapshot_path.parent != state_path.parent
            or snapshot_path.name != expected_snapshot_name
        ):
            raise RunnerError(
                "source state and snapshot must be a matching adjacent resume pair: "
                f"{state_path}, {snapshot_path}"
            )
        seeds.append(
            {
                "seed": seed,
                "gpu": gpu,
                "coefficient": coefficient,
                "source_state": source_state,
                "source_snapshot": source_snapshot,
            }
        )
    if seen != set(EXPECTED_SEEDS):
        raise RunnerError(f"manifest seed set differs from {EXPECTED_SEEDS}")
    seeds.sort(key=lambda row: row["seed"])
    if master_port_base + max(row["gpu"] for row in seeds) > 65535:
        raise RunnerError(
            "runtime.master_port_base plus assigned GPU exceeds port 65535"
        )
    smoke_row = next(row for row in seeds if row["seed"] == smoke_seed)
    if smoke.get("gpu") is not None and smoke.get("gpu") != smoke_row["gpu"]:
        raise RunnerError("smoke.gpu must equal the frozen GPU assigned to smoke.seed")

    formal_gates = _normalize_gate_specs(manifest.get("formal_gates"), "formal_gates")
    smoke_gates = _normalize_gate_specs(manifest.get("smoke_gates", []), "smoke_gates")
    if not smoke_gates:
        raise RunnerError("smoke_gates must contain the frozen scientific GO receipt")
    if not formal_gates:
        raise RunnerError("formal_gates must contain the frozen scientific GO receipt")
    smoke_preflight = [
        gate for gate in smoke_gates if gate.get("binding") == "preflight"
    ]
    formal_preflight = [
        gate for gate in formal_gates if gate.get("binding") == "preflight"
    ]
    if len(smoke_preflight) != 1 or len(formal_preflight) != 1:
        raise RunnerError(
            "smoke_gates and formal_gates must each contain exactly one preflight binding"
        )
    if any(
        smoke_preflight[0][key] != formal_preflight[0][key]
        for key in ("path", "sha256", "verdict_key", "expected")
    ):
        raise RunnerError(
            "smoke and formal must bind the identical frozen preflight receipt"
        )

    normalized = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "paths": {
            "repo_root": str(repo_root),
            "run_root": str(run_root),
            "transformer": str(transformer),
        },
        "runtime": {
            "python_command": python_command,
            "worker_command": worker_command,
            "git_command": git_command,
            "nvidia_smi_command": nvidia_command,
            "tmux_binary": tmux_binary,
            "master_port_base": master_port_base,
        },
        "provenance": {
            "execution_commit": execution_commit,
            "dataset": dataset,
            "container_identity": container_identity,
            "core_code": core_code,
        },
        "training": dict(training),
        "resources": dict(resources),
        "smoke": {
            "seed": smoke_seed,
            "gpu": smoke_row["gpu"],
            "validator_command": validator_template,
            "comparator_command": comparator_template,
            "validator_gate": validator_gate,
            "comparator_gate": comparator_gate,
        },
        "formal": {
            "arms": arms,
            "compatibility_report": compatibility_report,
            "validator_command": formal_validator_template,
            "validator_gate": formal_validator_gate,
        },
        "smoke_gates": smoke_gates,
        "formal_gates": formal_gates,
        "seeds": seeds,
    }
    verify_compatibility_fresh_arms(normalized)
    return normalized


def _normalize_result_gate(value: Any, label: str) -> dict[str, Any]:
    raw = _require_mapping(value, label)
    key = raw.get("verdict_key")
    expected = raw.get("expected")
    if not isinstance(key, str) or not key:
        raise RunnerError(f"{label}.verdict_key must be a non-empty string")
    if expected is None or isinstance(expected, (Mapping, list)):
        raise RunnerError(f"{label}.expected must be a scalar JSON value")
    return {"verdict_key": key, "expected": expected}


def _normalize_gate_specs(value: Any, label: str) -> list[dict[str, Any]]:
    raw_list = _require_sequence(value, label)
    normalized = []
    names = set()
    for index, value_item in enumerate(raw_list):
        raw = _require_mapping(value_item, f"{label}[{index}]")
        name = raw.get("name")
        if not isinstance(name, str) or not ID_RE.fullmatch(name) or name in names:
            raise RunnerError(f"{label}[{index}].name is invalid or duplicated")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "path": str(
                    _require_absolute_path(raw.get("path"), f"{label}[{index}].path")
                ),
                "sha256": _require_sha256(
                    raw.get("sha256"), f"{label}[{index}].sha256"
                ),
                "verdict_key": raw.get("verdict_key", "verdict"),
                "expected": raw.get("expected", "GO"),
                "binding": raw.get("binding"),
            }
        )
        if normalized[-1]["binding"] not in (None, "preflight"):
            raise RunnerError(f"{label}[{index}].binding is unsupported")
    return normalized


def load_manifest(
    path: Path, *, verify_artifacts: bool = True
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"manifest is not a regular non-symlink file: {path}")
    observed_sha = sha256_file(path)
    return validate_manifest(
        load_json(path, "execution manifest"), verify_artifacts=verify_artifacts
    ), observed_sha


def verify_external_gates(
    gates: Sequence[Mapping[str, Any]], config: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    receipts = []
    for gate in gates:
        label = f"gate {gate['name']}"
        path = Path(str(gate["path"]))
        if not path.is_file() or path.is_symlink():
            raise RunnerError(f"{label} receipt is not a regular file: {path}")
        observed = sha256_file(path)
        if observed != gate["sha256"]:
            raise RunnerError(
                f"{label} SHA256 mismatch: observed {observed}, expected {gate['sha256']}"
            )
        payload = load_json(path, label)
        actual = _json_value(payload, str(gate["verdict_key"]), label)
        if actual != gate["expected"]:
            raise RunnerError(
                f"{label} is not eligible: {gate['verdict_key']}={actual!r}, "
                f"expected {gate['expected']!r}"
            )
        binding = None
        if gate.get("binding") == "preflight":
            if config is None:
                raise RunnerError(
                    "preflight gate binding requires the normalized execution manifest"
                )
            binding = verify_preflight_binding(payload, config)
        receipts.append(
            {
                "name": gate["name"],
                "path": str(path),
                "sha256": observed,
                "binding": binding,
            }
        )
    return receipts


def verify_preflight_binding(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("status") != "GO"
        or payload.get("formal_training_authorized") is not True
    ):
        raise RunnerError("preflight receipt does not authorize formal training")
    frozen = _require_mapping(payload.get("frozen_a_s"), "preflight frozen_a_s")
    assets = _require_mapping(payload.get("source_assets"), "preflight source_assets")
    if set(frozen) != {str(seed) for seed in EXPECTED_SEEDS} or set(assets) != {
        str(seed) for seed in EXPECTED_SEEDS
    }:
        raise RunnerError("preflight seed set differs from the execution manifest")
    bindings = {}
    for row in config["seeds"]:
        seed_key = str(row["seed"])
        coefficient = frozen.get(seed_key)
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or float(coefficient) != float(row["coefficient"])
        ):
            raise RunnerError(f"preflight coefficient differs for seed {seed_key}")
        asset = _require_mapping(
            assets.get(seed_key), f"preflight source_assets.{seed_key}"
        )
        if asset.get("source_state_sha256") != row["source_state"]["sha256"]:
            raise RunnerError(f"preflight source state differs for seed {seed_key}")
        if asset.get("checkpoint_sha256") != row["source_snapshot"]["sha256"]:
            raise RunnerError(f"preflight source snapshot differs for seed {seed_key}")
        bindings[seed_key] = {
            "coefficient": float(coefficient),
            "source_state_sha256": asset["source_state_sha256"],
            "checkpoint_sha256": asset["checkpoint_sha256"],
        }
    return {"kind": "preflight", "seeds": bindings}


def verify_compatibility_fresh_arms(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the selected fresh formal arms to the fail-closed control audit.

    G is always relaunched beside T in this intervention.  F is also required
    for *all* seeds when the compatibility audit marks F ``fresh_required`` for
    any seed; this keeps the independent training-seed matrix paired.  A report
    that requests fresh F while the manifest selects only G/T is rejected.
    """
    artifact = config["formal"]["compatibility_report"]
    path = Path(str(artifact["path"]))
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"compatibility report is not a regular file: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != artifact["sha256"]:
        raise RunnerError(
            f"compatibility report SHA256 mismatch: {observed_sha} != {artifact['sha256']}"
        )
    report = load_json(path, "compatibility report")
    if report.get("schema_version") != 1 or report.get("status") not in (
        "PASS",
        "FAIL",
    ):
        raise RunnerError("compatibility report has an unsupported schema/status")
    seed_reports = _require_mapping(report.get("seeds"), "compatibility report seeds")
    rows_raw = _require_sequence(report.get("rows"), "compatibility report rows")
    row_index: dict[tuple[int, str, str, str], Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows_raw):
        identity_item = _require_mapping(raw_row, f"compatibility report rows[{index}]")
        key = (
            identity_item.get("seed"),
            identity_item.get("arm"),
            identity_item.get("scope"),
            identity_item.get("field"),
        )
        if not isinstance(key[0], int) or any(
            not isinstance(item, str) for item in key[1:]
        ):
            raise RunnerError("compatibility report contains an invalid identity row")
        if key in row_index:
            raise RunnerError(f"compatibility report duplicates identity row {key}")
        row_index[key] = identity_item
    decisions: dict[str, dict[str, str]] = {}
    fresh_f = False
    for seed in EXPECTED_SEEDS:
        seed_config = next(item for item in config["seeds"] if item["seed"] == seed)
        seed_report = _require_mapping(
            seed_reports.get(str(seed)), f"compatibility report seed {seed}"
        )
        controls = _require_mapping(
            seed_report.get("controls"), f"compatibility report seed {seed} controls"
        )
        decisions[str(seed)] = {}
        for arm in ("F", "G"):
            control = _require_mapping(
                controls.get(arm), f"compatibility report seed {seed} arm {arm}"
            )
            decision = control.get("reuse_decision")
            if decision not in ("reusable", "fresh_required"):
                raise RunnerError(
                    f"compatibility report seed {seed} arm {arm} has invalid reuse_decision"
                )
            decisions[str(seed)][arm] = str(decision)
            if arm == "F" and decision == "fresh_required":
                fresh_f = True
            source_expectations = {
                "full_state_sha256": seed_config["source_state"]["sha256"],
                "checkpoint_sha256": seed_config["source_snapshot"]["sha256"],
            }
            protocol_expectations: Mapping[str, Any] = {
                "q": 256,
                "batch.batch_size": config["training"]["batch"],
                "batch.batch_gpu": config["training"]["batch_gpu"],
                "augmentation": {"augment_kwargs": None, "dataset_xflip": False},
                "precision.use_fp16": True,
                "precision.enable_amp": True,
                "precision.enable_tf32": False,
                "precision.loss_scaling": 1.0,
                "checkpoint_cadence.kimg_per_tick": 10,
                "checkpoint_cadence.snapshot_ticks": None,
                "checkpoint_cadence.state_dump_ticks": None,
                "checkpoint_cadence.ckpt_ticks": 10,
                "checkpoint_cadence.sample_ticks": 26,
                "checkpoint_cadence.eval_ticks": 50,
                "data.byte_sha256": config["provenance"]["dataset"]["sha256"],
                "start_kimg": 256,
                "endpoints_kimg": [512, 768, 1024],
                "schedule": "sigmoid" if arm == "F" else "global_sigmoid",
                "gap_scale": 1.0 if arm == "F" else 1.10,
            }
            for field, expected_value in source_expectations.items():
                identity_row = row_index.get((seed, arm, "source", field))
                if (
                    identity_row is None
                    or identity_row.get("expected") != expected_value
                ):
                    raise RunnerError(
                        f"compatibility report source identity differs for seed {seed} arm {arm} {field}"
                    )
            for field, expected_value in protocol_expectations.items():
                identity_row = row_index.get((seed, arm, "run", field))
                if (
                    identity_row is None
                    or identity_row.get("expected") != expected_value
                ):
                    raise RunnerError(
                        f"compatibility report protocol identity differs for seed {seed} arm {arm} {field}"
                    )
    expected_arms = ["F", "G", "T"] if fresh_f else ["G", "T"]
    selected = list(config["formal"]["arms"])
    if selected != expected_arms:
        raise RunnerError(
            "formal.arms is inconsistent with compatibility fresh requirements: "
            f"selected={selected}, required={expected_arms}"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "status": report["status"],
        "selected_fresh_arms": selected,
        "decisions": decisions,
    }


def verify_git_provenance(config: Mapping[str, Any]) -> None:
    repo = Path(config["paths"]["repo_root"])
    command = list(config["runtime"]["git_command"])
    try:
        head = subprocess.check_output(
            command + ["rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            command + ["status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"cannot verify Git provenance: {exc}") from exc
    if head != config["provenance"]["execution_commit"]:
        raise RunnerError(
            f"execution HEAD mismatch: observed {head}, expected "
            f"{config['provenance']['execution_commit']}"
        )
    if dirty:
        first = dirty.splitlines()[:8]
        raise RunnerError(f"formal/smoke worktree is not clean: {first}")


def _duration_arg(end_kimg: int) -> str:
    if isinstance(end_kimg, bool) or not isinstance(end_kimg, int) or end_kimg <= 0:
        raise RunnerError(f"end_kimg must be a positive integer: {end_kimg!r}")
    whole, fractional = divmod(end_kimg, 1000)
    return f"{whole}.{fractional:03d}".rstrip("0").rstrip(".")


TRAINING_FLAG_ORDER = (
    "cond",
    "arch",
    "precond",
    "batch",
    "batch_gpu",
    "optim",
    "lr",
    "dropout",
    "augment",
    "xflip",
    "mapping",
    "global_gap_scale",
    "q",
    "k",
    "b",
    "c",
    "double",
    "ema_beta",
    "fp16",
    "enable_amp",
    "tf32",
    "ls",
    "metrics",
    "tick",
    "snap",
    "dump",
    "ckpt",
    "sample_every",
    "eval_every",
    "adaptive_update_kimg",
    "workers",
    "bench",
    "cache",
)


def build_training_command(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: Path,
    resume_state: Path,
    end_kimg: int,
    arm: str = "G",
) -> list[str]:
    if arm not in ("F", "G", "T"):
        raise RunnerError(f"unknown formal arm: {arm!r}")
    training = config["training"]
    command = list(config["runtime"]["python_command"])
    command.extend(
        [
            str(Path(config["paths"]["repo_root"]) / "ct_train.py"),
            f"--data={config['provenance']['dataset']['path']}",
            f"--outdir={output_dir}",
            "--nosubdir",
            f"--duration={_duration_arg(end_kimg)}",
            f"--seed={seed}",
            f"--resume={resume_state}",
        ]
    )
    short_flags = {"q": "-q", "k": "-k", "b": "-b", "c": "-c"}
    names = {
        "batch_gpu": "batch-gpu",
        "global_gap_scale": "global-gap-scale",
        "enable_amp": "enable_amp",
        "ema_beta": "ema_beta",
        "sample_every": "sample_every",
        "eval_every": "eval_every",
        "adaptive_update_kimg": "adaptive-update-kimg",
    }
    for key in TRAINING_FLAG_ORDER:
        if key not in training:
            continue
        value = training[key]
        if arm == "F" and key == "mapping":
            value = "sigmoid"
        elif arm == "F" and key == "global_gap_scale":
            value = 1.0
        rendered = (
            _bool_arg(value)
            if isinstance(value, bool)
            else (_number_arg(value) if isinstance(value, (int, float)) else str(value))
        )
        if key in short_flags:
            command.extend([short_flags[key], rendered])
        else:
            command.append(f"--{names.get(key, key.replace('_', '-'))}={rendered}")
    return command


def _prepared_paths(
    config: Mapping[str, Any], seed: int, variant: str
) -> tuple[Path, Path, Path]:
    root = Path(config["paths"]["run_root"])
    directory = _safe_child(root, "staged", f"seed{seed}", variant)
    return (
        directory / "training-state-latest.pt",
        directory / "network-snapshot-latest.pkl",
        directory / "training-state-latest.pt.manifest.json",
    )


def _make_job(
    *,
    job_id: str,
    kind: str,
    phase: str,
    seed: int,
    gpu: int,
    command: Sequence[str],
    record_dir: Path,
    input_artifacts: Sequence[Mapping[str, str]] = (),
    output_artifacts: Sequence[Mapping[str, Any]] = (),
    result_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not ID_RE.fullmatch(job_id):
        raise RunnerError(f"unsafe job_id: {job_id!r}")
    return {
        "job_id": job_id,
        "kind": kind,
        "phase": phase,
        "seed": seed,
        "gpu": gpu,
        "command": list(command),
        "record_dir": str(record_dir),
        "input_artifacts": [dict(item) for item in input_artifacts],
        "output_artifacts": [dict(item) for item in output_artifacts],
        "result_gate": dict(result_gate) if result_gate is not None else None,
    }


def _distributed_environment(config: Mapping[str, Any], gpu: int) -> dict[str, str]:
    port = int(config["runtime"]["master_port_base"]) + int(gpu)
    if not 1024 <= port <= 65535:
        raise RunnerError(f"derived MASTER_PORT is out of range for GPU {gpu}: {port}")
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }


def render_template_command(
    template: Sequence[str], values: Mapping[str, Any]
) -> list[str]:
    """Render a no-shell command template.

    Supported values supplied by this runner include ``run_dir``, ``state``,
    ``snapshot``, ``expected_nimg``, ``left_state``, ``right_state``,
    ``left_snapshot``, ``right_snapshot``, ``result_receipt``, ``seed``,
    ``arm``, ``endpoint_kimg``, ``expected_mapping``, and
    ``expected_gap_scale``.  Validator templates also receive
    ``expected_method`` (``fixed`` or ``global110``), ``checkpoint_id``, and
    ``training_run_id``.
    """
    rendered = []
    string_values = {key: str(value) for key, value in values.items()}
    for argument in template:
        try:
            value = argument.format_map(string_values)
        except KeyError as exc:
            raise RunnerError(
                f"unknown command-template placeholder {exc.args[0]!r}"
            ) from exc
        if "{" in value or "}" in value or not value or "\x00" in value:
            raise RunnerError(f"invalid rendered command argument: {value!r}")
        rendered.append(value)
    return rendered


def build_prepare_jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["paths"]["run_root"])
    jobs = []
    for row in config["seeds"]:
        seed = row["seed"]
        for variant, coefficient in (("noop", 1.0), ("transport", row["coefficient"])):
            state, snapshot, sidecar = _prepared_paths(config, seed, variant)
            record = _safe_child(
                root, "_runner", "commands", "prepare", f"seed{seed}-{variant}"
            )
            command = list(config["runtime"]["python_command"])
            command.extend(
                [
                    config["paths"]["transformer"],
                    "--source",
                    row["source_state"]["path"],
                    "--output",
                    str(state),
                    "--coefficient",
                    format(coefficient, ".17g"),
                    "--manifest",
                    str(sidecar),
                    "--expected-source-sha256",
                    row["source_state"]["sha256"],
                ]
            )
            jobs.append(
                _make_job(
                    job_id=f"prepare-seed{seed}-{variant}",
                    kind="transform",
                    phase="prepare",
                    seed=seed,
                    gpu=row["gpu"],
                    command=command,
                    record_dir=record,
                    input_artifacts=[row["source_state"], row["source_snapshot"]],
                    output_artifacts=[
                        {"path": str(state), "required": True},
                        {"path": str(snapshot), "required": True},
                        {"path": str(sidecar), "required": True},
                    ],
                )
            )
    return jobs


def _smoke_run_dir(config: Mapping[str, Any], seed: int, name: str) -> Path:
    return _safe_child(Path(config["paths"]["run_root"]), "smoke", f"seed{seed}", name)


def _formal_run_dir(
    config: Mapping[str, Any], seed: int, arm: str, endpoint: int
) -> Path:
    return _safe_child(
        Path(config["paths"]["run_root"]), "formal", f"seed{seed}", arm, f"{endpoint}k"
    )


def _train_output_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(run_dir / "training-state-latest.pt"), "required": True},
        {"path": str(run_dir / "network-snapshot-latest.pkl"), "required": True},
        {"path": str(run_dir / "train_summary.csv"), "required": True},
        {"path": str(run_dir / "log.txt"), "required": True},
    ]


def _resume_pair_artifacts(resume_state: Path) -> list[dict[str, str]]:
    match = re.fullmatch(r"training-state-(\d+|latest)\.pt", resume_state.name)
    if not match:
        raise RunnerError(f"invalid resume-state filename: {resume_state}")
    snapshot = resume_state.with_name(f"network-snapshot-{match.group(1)}.pkl")
    return [{"path": str(resume_state)}, {"path": str(snapshot)}]


def _validator_job(
    config: Mapping[str, Any],
    *,
    phase: str,
    job_id: str,
    seed: int,
    gpu: int,
    arm: str,
    endpoint_kimg: int,
    run_dir: Path,
    expected_nimg: int | None = None,
) -> dict[str, Any]:
    section = config["smoke"] if phase == "smoke" else config["formal"]
    record_dir = _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "commands", phase, job_id
    )
    result_receipt = record_dir / "result.json"
    state = run_dir / "training-state-latest.pt"
    snapshot = run_dir / "network-snapshot-latest.pkl"
    if expected_nimg is None:
        expected_nimg = endpoint_kimg * 1000
    if (
        isinstance(expected_nimg, bool)
        or not isinstance(expected_nimg, int)
        or expected_nimg <= 0
    ):
        raise RunnerError(f"invalid validator expected_nimg: {expected_nimg!r}")
    expected_mapping = "sigmoid" if arm == "F" else "global_sigmoid"
    expected_gap_scale = 1.0 if arm == "F" else 1.10
    expected_method = "fixed" if arm == "F" else "global110"
    checkpoint_id = f"q256-{phase}-seed{seed}-{arm}-{endpoint_kimg}k"
    training_run_id = checkpoint_id
    command = render_template_command(
        section["validator_command"],
        {
            "run_dir": run_dir,
            "state": state,
            "snapshot": snapshot,
            "expected_nimg": expected_nimg,
            "result_receipt": result_receipt,
            "seed": seed,
            "arm": arm,
            "endpoint_kimg": endpoint_kimg,
            "expected_mapping": expected_mapping,
            "expected_gap_scale": expected_gap_scale,
            "expected_method": expected_method,
            "checkpoint_id": checkpoint_id,
            "training_run_id": training_run_id,
        },
    )
    return _make_job(
        job_id=job_id,
        kind="validate",
        phase=phase,
        seed=seed,
        gpu=gpu,
        command=command,
        record_dir=record_dir,
        input_artifacts=[{"path": str(state)}, {"path": str(snapshot)}],
        output_artifacts=[{"path": str(result_receipt), "required": True}],
        result_gate={
            "path": str(result_receipt),
            **section["validator_gate"],
            "bindings": {
                "expected.arm": arm,
                "expected.seed": seed,
                "expected.nimg": expected_nimg,
                "expected.schedule": expected_mapping,
                "expected.global_gap_scale": expected_gap_scale,
            },
        },
    )


def build_smoke_jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = config["smoke"]["seed"]
    row = next(item for item in config["seeds"] if item["seed"] == seed)
    gpu = row["gpu"]
    source_kimg = config["training"]["source_kimg"]
    steps = config["training"]["smoke_steps"]
    batch = config["training"]["batch"]
    images = steps * batch
    if images % 1000 != 96:
        # With the frozen 256k source and batch 128, 32 steps stop at 260.096k;
        # ct_train's integer-kimg budget ends after exactly 32 updates at 260k.
        raise RunnerError("frozen smoke step arithmetic unexpectedly changed")
    smoke_end = source_kimg + images // 1000
    noop_state, _, _ = _prepared_paths(config, seed, "noop")
    transport_state, _, _ = _prepared_paths(config, seed, "transport")
    variants = [
        ("noop-direct", Path(row["source_state"]["path"])),
        ("noop-rewrite", noop_state),
        ("transport-repeat1", transport_state),
        ("transport-repeat2", transport_state),
    ]
    jobs: list[dict[str, Any]] = []
    run_dirs: dict[str, Path] = {}
    for name, resume in variants:
        run_dir = _smoke_run_dir(config, seed, name)
        run_dirs[name] = run_dir
        train_id = f"smoke-seed{seed}-{name}-train"
        record_dir = _safe_child(
            Path(config["paths"]["run_root"]), "_runner", "commands", "smoke", train_id
        )
        jobs.append(
            _make_job(
                job_id=train_id,
                kind="train",
                phase="smoke",
                seed=seed,
                gpu=gpu,
                command=build_training_command(
                    config,
                    seed=seed,
                    output_dir=run_dir,
                    resume_state=resume,
                    end_kimg=smoke_end,
                ),
                record_dir=record_dir,
                input_artifacts=_resume_pair_artifacts(resume),
                output_artifacts=_train_output_artifacts(run_dir),
            )
        )
        jobs.append(
            _validator_job(
                config,
                phase="smoke",
                job_id=f"smoke-seed{seed}-{name}-validate",
                seed=seed,
                gpu=gpu,
                arm=name,
                endpoint_kimg=smoke_end,
                run_dir=run_dir,
                expected_nimg=source_kimg * 1000 + images,
            )
        )

    comparisons = (
        ("noop", "noop-direct", "noop-rewrite"),
        ("transport-repeat", "transport-repeat1", "transport-repeat2"),
    )
    for label, left_name, right_name in comparisons:
        job_id = f"smoke-seed{seed}-{label}-compare"
        record_dir = _safe_child(
            Path(config["paths"]["run_root"]), "_runner", "commands", "smoke", job_id
        )
        result_receipt = record_dir / "result.json"
        left = run_dirs[left_name]
        right = run_dirs[right_name]
        command = render_template_command(
            config["smoke"]["comparator_command"],
            {
                "left_state": left / "training-state-latest.pt",
                "right_state": right / "training-state-latest.pt",
                "left_snapshot": left / "network-snapshot-latest.pkl",
                "right_snapshot": right / "network-snapshot-latest.pkl",
                "result_receipt": result_receipt,
                "seed": seed,
                "arm": label,
                "endpoint_kimg": smoke_end,
            },
        )
        jobs.append(
            _make_job(
                job_id=job_id,
                kind="compare",
                phase="smoke",
                seed=seed,
                gpu=gpu,
                command=command,
                record_dir=record_dir,
                input_artifacts=[
                    {"path": str(left / "training-state-latest.pt")},
                    {"path": str(right / "training-state-latest.pt")},
                    {"path": str(left / "network-snapshot-latest.pkl")},
                    {"path": str(right / "network-snapshot-latest.pkl")},
                ],
                output_artifacts=[{"path": str(result_receipt), "required": True}],
                result_gate={
                    "path": str(result_receipt),
                    **config["smoke"]["comparator_gate"],
                },
            )
        )
    return jobs


def build_formal_jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    root = Path(config["paths"]["run_root"])
    for row in config["seeds"]:
        seed = row["seed"]
        gpu = row["gpu"]
        transported, _, _ = _prepared_paths(config, seed, "transport")
        previous = {
            arm: (transported if arm == "T" else Path(row["source_state"]["path"]))
            for arm in config["formal"]["arms"]
        }
        for endpoint in config["training"]["endpoints_kimg"]:
            for arm in config["formal"]["arms"]:
                run_dir = _formal_run_dir(config, seed, arm, endpoint)
                train_id = f"formal-seed{seed}-{arm}-{endpoint}k-train"
                record_dir = _safe_child(
                    root, "_runner", "commands", "formal", train_id
                )
                jobs.append(
                    _make_job(
                        job_id=train_id,
                        kind="train",
                        phase="formal",
                        seed=seed,
                        gpu=gpu,
                        command=build_training_command(
                            config,
                            seed=seed,
                            output_dir=run_dir,
                            resume_state=previous[arm],
                            end_kimg=endpoint,
                            arm=arm,
                        ),
                        record_dir=record_dir,
                        input_artifacts=_resume_pair_artifacts(previous[arm]),
                        output_artifacts=_train_output_artifacts(run_dir),
                    )
                )
                jobs.append(
                    _validator_job(
                        config,
                        phase="formal",
                        job_id=f"formal-seed{seed}-{arm}-{endpoint}k-validate",
                        seed=seed,
                        gpu=gpu,
                        arm=arm,
                        endpoint_kimg=endpoint,
                        run_dir=run_dir,
                    )
                )
                previous[arm] = run_dir / "training-state-latest.pt"
    return jobs


def build_plan(
    config: Mapping[str, Any], manifest_sha256: str, phase: str
) -> dict[str, Any]:
    if phase == "prepare":
        jobs = build_prepare_jobs(config)
    elif phase == "smoke":
        jobs = build_smoke_jobs(config)
    elif phase == "formal":
        jobs = build_formal_jobs(config)
    else:
        raise RunnerError(f"cannot build a plan for phase {phase!r}")
    fingerprints = []
    for job in jobs:
        job["environment"] = _distributed_environment(config, int(job["gpu"]))
        fingerprint = sha256_bytes(canonical_json(job).encode())
        job["fingerprint"] = fingerprint
        fingerprints.append(fingerprint)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": config["experiment_id"],
        "phase": phase,
        "manifest_sha256": manifest_sha256,
        "execution_commit": config["provenance"]["execution_commit"],
        "job_count": len(jobs),
        "jobs_sha256": sha256_bytes(canonical_json(fingerprints).encode()),
        "jobs": jobs,
    }


def _query_gpu(config: Mapping[str, Any], gpu: int) -> dict[str, Any]:
    command = list(config["runtime"]["nvidia_smi_command"])
    command.extend(
        [
            "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"nvidia-smi preflight failed: {exc}") from exc
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            index, free, total, util = (
                int(fields[0]),
                int(fields[2]),
                int(fields[3]),
                int(fields[4]),
            )
        except ValueError:
            continue
        if index == gpu:
            return {
                "index": index,
                "name": fields[1],
                "memory_free_mib": free,
                "memory_total_mib": total,
                "utilization_pct": util,
                "raw": line,
            }
    raise RunnerError(f"assigned GPU {gpu} is absent from nvidia-smi output")


def resource_preflight(
    config: Mapping[str, Any], gpu: int, *, require_idle: bool = True
) -> dict[str, Any]:
    run_root = Path(config["paths"]["run_root"])
    disk_path = _nearest_existing(run_root)
    usage = shutil.disk_usage(disk_path)
    minimum_disk = config["resources"]["min_free_disk_bytes"]
    if usage.free < minimum_disk:
        raise RunnerError(
            f"insufficient disk space at {disk_path}: {usage.free} < {minimum_disk} bytes"
        )
    snapshot = _query_gpu(config, gpu)
    if snapshot["memory_free_mib"] < config["resources"]["min_free_gpu_mib"]:
        raise RunnerError(
            f"GPU {gpu} has insufficient free memory: {snapshot['memory_free_mib']} MiB"
        )
    if (
        require_idle
        and snapshot["utilization_pct"] > config["resources"]["max_gpu_utilization_pct"]
    ):
        raise RunnerError(
            f"GPU {gpu} utilization is too high: {snapshot['utilization_pct']}%"
        )
    return {
        "disk": {
            "path": str(disk_path),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "required_free_bytes": minimum_disk,
        },
        "gpu": snapshot,
    }


def _observe_inputs(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed = []
    for index, raw in enumerate(job.get("input_artifacts", [])):
        path = Path(str(raw.get("path", "")))
        if not path.is_file() or path.is_symlink():
            raise RunnerError(
                f"job {job['job_id']} input {index} is not a regular file: {path}"
            )
        digest = sha256_file(path)
        expected = raw.get("sha256")
        if expected is not None and digest != expected:
            raise RunnerError(
                f"job {job['job_id']} input hash mismatch for {path}: {digest} != {expected}"
            )
        observed.append(
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        )
    return observed


def _observe_outputs(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed = []
    for index, raw in enumerate(job.get("output_artifacts", [])):
        path = Path(str(raw.get("path", "")))
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RunnerError(
                f"job {job['job_id']} output {index} is missing/empty: {path}"
            )
        observed.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return observed


def _verify_result_gate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    gate = job.get("result_gate")
    if gate is None:
        return None
    path = Path(str(gate["path"]))
    payload = load_json(path, f"job {job['job_id']} result receipt")
    observed = _json_value(
        payload, gate["verdict_key"], f"job {job['job_id']} result receipt"
    )
    if observed != gate["expected"]:
        raise RunnerError(
            f"job {job['job_id']} result gate failed: {observed!r} != {gate['expected']!r}"
        )
    bindings = gate.get("bindings", {})
    if not isinstance(bindings, Mapping):
        raise RunnerError(f"job {job['job_id']} result bindings must be a mapping")
    for dotted_key, expected_value in bindings.items():
        actual_value = _json_value(
            payload, str(dotted_key), f"job {job['job_id']} result receipt"
        )
        if actual_value != expected_value:
            raise RunnerError(
                f"job {job['job_id']} result binding {dotted_key!r} differs: "
                f"{actual_value!r} != {expected_value!r}"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "verdict_key": gate["verdict_key"],
        "expected": gate["expected"],
        "bindings": dict(bindings),
    }


def _completed_receipt(
    job: Mapping[str, Any], manifest_sha256: str
) -> Mapping[str, Any] | None:
    exit_path = Path(job["record_dir"]) / "exit.json"
    if not exit_path.exists():
        return None
    receipt = load_json(exit_path, f"job {job['job_id']} exit receipt")
    if receipt.get("schema") != EXIT_SCHEMA or receipt.get("status") != "completed":
        raise RunnerError(
            f"job {job['job_id']} has a non-success receipt and cannot be overwritten"
        )
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise RunnerError(
            f"job {job['job_id']} receipt belongs to a different manifest"
        )
    if receipt.get("job_fingerprint") != job["fingerprint"]:
        raise RunnerError(f"job {job['job_id']} receipt fingerprint mismatch")
    expected_outputs = {
        item["path"]: item["sha256"] for item in receipt.get("outputs", [])
    }
    for raw in job.get("output_artifacts", []):
        path = Path(str(raw["path"]))
        if str(path) not in expected_outputs or not path.is_file():
            raise RunnerError(
                f"completed job {job['job_id']} has a missing recorded output: {path}"
            )
        if sha256_file(path) != expected_outputs[str(path)]:
            raise RunnerError(f"completed job {job['job_id']} output changed: {path}")
    _verify_result_gate(job)
    return receipt


def execute_job(
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    completed = _completed_receipt(job, manifest_sha256)
    if completed is not None:
        return completed
    expected_environment = _distributed_environment(config, int(job["gpu"]))
    declared_environment = job.get("environment")
    if (
        declared_environment is not None
        and declared_environment != expected_environment
    ):
        raise RunnerError(
            f"job {job['job_id']} distributed environment differs from manifest"
        )
    record_dir = Path(job["record_dir"])
    if record_dir.exists():
        raise RunnerError(
            f"job {job['job_id']} has an incomplete record directory; refusing overwrite: {record_dir}"
        )
    if job["kind"] == "train":
        training_state_outputs = [
            Path(str(item["path"]))
            for item in job.get("output_artifacts", [])
            if Path(str(item["path"])).name == "training-state-latest.pt"
        ]
        if len(training_state_outputs) != 1:
            raise RunnerError(
                f"train job {job['job_id']} has no unique output run directory"
            )
        run_dir = training_state_outputs[0].parent
        if run_dir.exists() or run_dir.is_symlink():
            raise RunnerError(
                f"refusing to start train job in an existing output directory: {run_dir}"
            )
    # All checks below are read-only.  Perform them before creating a record
    # directory so a temporarily busy GPU or not-yet-ready dependency remains
    # safely retryable rather than looking like an interrupted command.
    inputs = _observe_inputs(job)
    before = resource_preflight(
        config,
        int(job["gpu"]),
        require_idle=job["kind"] == "train",
    )
    record_dir.mkdir(parents=True, exist_ok=False)
    command_path = record_dir / "command.json"
    stdout_path = record_dir / "stdout.log"
    stderr_path = record_dir / "stderr.log"
    command_receipt = {
        "job_id": job["job_id"],
        "fingerprint": job["fingerprint"],
        "command": job["command"],
        "cwd": config["paths"]["repo_root"],
        "environment": expected_environment,
        "created_utc": utc_now(),
    }
    _write_new_json(command_path, command_receipt)
    environment = os.environ.copy()
    environment.update(expected_environment)
    started = utc_now()
    start_mono = time.monotonic()
    return_code: int | None = None
    error: str | None = None
    outputs: list[dict[str, Any]] = []
    result_gate: dict[str, Any] | None = None
    try:
        with (
            stdout_path.open("xb") as stdout_handle,
            stderr_path.open("xb") as stderr_handle,
        ):
            process = subprocess.run(
                list(job["command"]),
                cwd=config["paths"]["repo_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
            return_code = process.returncode
        if return_code != 0:
            raise RunnerError(f"command exited with code {return_code}")
        outputs = _observe_outputs(job)
        result_gate = _verify_result_gate(job)
        status = "completed"
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    try:
        after_gpu = _query_gpu(config, int(job["gpu"]))
    except RunnerError as exc:
        after_gpu = {"error": str(exc)}
        if status == "completed":
            status = "failed"
            error = f"post-command GPU snapshot failed: {exc}"
    receipt = {
        "schema": EXIT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "manifest_sha256": manifest_sha256,
        "execution_commit": config["provenance"]["execution_commit"],
        "job_id": job["job_id"],
        "job_fingerprint": job["fingerprint"],
        "kind": job["kind"],
        "phase": job["phase"],
        "seed": job["seed"],
        "gpu": job["gpu"],
        "status": status,
        "exit_code": return_code,
        "error": error,
        "started_utc": started,
        "finished_utc": utc_now(),
        "duration_seconds": time.monotonic() - start_mono,
        "command": job["command"],
        "environment": expected_environment,
        "resource_preflight": before,
        "gpu_after": after_gpu,
        "inputs": inputs,
        "outputs": outputs,
        "result_gate": result_gate,
        "logs": {
            "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
            "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
            "command": {"path": str(command_path), "sha256": sha256_file(command_path)},
        },
    }
    _write_new_json(record_dir / "exit.json", receipt)
    if status != "completed":
        raise RunnerError(f"job {job['job_id']} failed: {error}")
    return receipt


def _copy_or_verify_snapshot(source: Mapping[str, Any], target: Path) -> dict[str, Any]:
    expected = source["sha256"]
    source_path = Path(source["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            not target.is_file()
            or target.is_symlink()
            or sha256_file(target) != expected
        ):
            raise RunnerError(
                f"existing staged snapshot is not the frozen source: {target}"
            )
        return {"path": str(target), "sha256": expected, "status": "verified_existing"}
    temp_path = (
        target.parent / f".{target.name}.copy-tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with (
            source_path.open("rb") as source_handle,
            temp_path.open("xb") as target_handle,
        ):
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        observed = sha256_file(temp_path)
        if observed != expected:
            raise RunnerError(f"staged snapshot temporary hash mismatch: {temp_path}")
        try:
            os.link(temp_path, target, follow_symlinks=False)
        except FileExistsError:
            if (
                not target.is_file()
                or target.is_symlink()
                or sha256_file(target) != expected
            ):
                raise RunnerError(
                    f"concurrent staged snapshot differs from source: {target}"
                )
            return {
                "path": str(target),
                "sha256": expected,
                "status": "verified_concurrent",
            }
        return {"path": str(target), "sha256": observed, "status": "copied"}
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _validate_prepare_receipt(
    config: Mapping[str, Any], manifest_sha256: str
) -> Mapping[str, Any]:
    path = _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "prepare_receipt.json"
    )
    receipt = load_json(path, "prepare receipt")
    if (
        receipt.get("schema") != PREPARE_SCHEMA
        or receipt.get("verdict") != "GO"
        or receipt.get("experiment_id") != config["experiment_id"]
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("execution_commit") != config["provenance"]["execution_commit"]
    ):
        raise RunnerError(
            f"prepare receipt is not a GO for this frozen execution: {path}"
        )
    plan = build_plan(config, manifest_sha256, "prepare")
    if receipt.get("jobs_sha256") != plan["jobs_sha256"]:
        raise RunnerError("prepare receipt jobs_sha256 differs from the frozen plan")
    expected_artifacts = set()
    for row in config["seeds"]:
        for variant in ("noop", "transport"):
            expected_artifacts.update(
                str(item) for item in _prepared_paths(config, row["seed"], variant)
            )
    artifact_rows = receipt.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise RunnerError("prepare receipt artifacts must be a complete list")
    observed_artifacts = {
        str(row.get("path", "")): row
        for row in artifact_rows
        if isinstance(row, Mapping)
    }
    if (
        len(observed_artifacts) != len(artifact_rows)
        or set(observed_artifacts) != expected_artifacts
    ):
        raise RunnerError(
            "prepare receipt artifact set is incomplete, duplicated, or unexpected"
        )
    for artifact_path_text, row in observed_artifacts.items():
        artifact_path = Path(artifact_path_text)
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise RunnerError(
                f"prepared artifact changed or disappeared: {artifact_path}"
            )
        if sha256_file(artifact_path) != row.get("sha256"):
            raise RunnerError(f"prepared artifact hash changed: {artifact_path}")

    exit_rows = receipt.get("job_exit_receipts")
    if not isinstance(exit_rows, list):
        raise RunnerError("prepare receipt job_exit_receipts must be a complete list")
    observed_exits = {
        str(row.get("job_id", "")): row for row in exit_rows if isinstance(row, Mapping)
    }
    expected_jobs = {job["job_id"]: job for job in plan["jobs"]}
    if len(observed_exits) != len(exit_rows) or set(observed_exits) != set(
        expected_jobs
    ):
        raise RunnerError(
            "prepare receipt command set is incomplete, duplicated, or unexpected"
        )
    for job_id, job in expected_jobs.items():
        row = observed_exits[job_id]
        exit_path = Path(job["record_dir"]) / "exit.json"
        if row.get("path") != str(exit_path):
            raise RunnerError(f"prepare receipt path mismatch for {job_id}")
        if not exit_path.is_file() or sha256_file(exit_path) != row.get("sha256"):
            raise RunnerError(
                f"prepare command receipt changed or disappeared: {exit_path}"
            )
        _completed_receipt(job, manifest_sha256)
    return receipt


def run_prepare(
    config: Mapping[str, Any], plan: Mapping[str, Any], manifest_sha256: str
) -> Mapping[str, Any]:
    root = Path(config["paths"]["run_root"])
    receipt_path = _safe_child(root, "_runner", "prepare_receipt.json")
    if receipt_path.exists():
        return _validate_prepare_receipt(config, manifest_sha256)
    snapshot_staging = []
    for row in config["seeds"]:
        for variant in ("noop", "transport"):
            _, snapshot, _ = _prepared_paths(config, row["seed"], variant)
            snapshot_staging.append(
                _copy_or_verify_snapshot(row["source_snapshot"], snapshot)
            )
    for job in plan["jobs"]:
        execute_job(config, job, manifest_sha256=manifest_sha256)
    artifacts = []
    for row in config["seeds"]:
        # Re-hash immutable sources after every transformation.
        for label in ("source_state", "source_snapshot"):
            path = Path(row[label]["path"])
            if sha256_file(path) != row[label]["sha256"]:
                raise RunnerError(f"immutable {label} changed during prepare: {path}")
        for variant in ("noop", "transport"):
            state, snapshot, sidecar = _prepared_paths(config, row["seed"], variant)
            for path in (state, snapshot, sidecar):
                artifacts.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    receipt = {
        "schema": PREPARE_SCHEMA,
        "verdict": "GO",
        "experiment_id": config["experiment_id"],
        "manifest_sha256": manifest_sha256,
        "execution_commit": config["provenance"]["execution_commit"],
        "jobs_sha256": plan["jobs_sha256"],
        "created_utc": utc_now(),
        "snapshot_staging": snapshot_staging,
        "artifacts": artifacts,
        "job_exit_receipts": [
            {
                "job_id": job["job_id"],
                "path": str(Path(job["record_dir"]) / "exit.json"),
                "sha256": sha256_file(Path(job["record_dir"]) / "exit.json"),
            }
            for job in plan["jobs"]
        ],
    }
    _write_new_json(receipt_path, receipt)
    return receipt


def _phase_gate_path(config: Mapping[str, Any], phase: str) -> Path:
    return _safe_child(
        Path(config["paths"]["run_root"]), "_runner", f"{phase}_gate.json"
    )


def _write_phase_gate(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest_sha256: str,
    phase: str,
) -> Mapping[str, Any]:
    path = _phase_gate_path(config, phase)
    receipts = []
    for job in plan["jobs"]:
        exit_path = Path(job["record_dir"]) / "exit.json"
        receipt = _completed_receipt(job, manifest_sha256)
        if receipt is None:
            raise RunnerError(
                f"cannot issue {phase} GO: job is incomplete: {job['job_id']}"
            )
        receipts.append(
            {
                "job_id": job["job_id"],
                "path": str(exit_path),
                "sha256": sha256_file(exit_path),
            }
        )
    payload = {
        "schema": GATE_SCHEMA,
        "phase": phase,
        "verdict": "GO",
        "experiment_id": config["experiment_id"],
        "manifest_sha256": manifest_sha256,
        "execution_commit": config["provenance"]["execution_commit"],
        "jobs_sha256": plan["jobs_sha256"],
        "created_utc": utc_now(),
        "exit_receipts": receipts,
    }
    if path.exists():
        existing = load_json(path, f"existing {phase} gate")
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        requested = dict(payload)
        requested.pop("created_utc", None)
        if canonical_json(comparable) != canonical_json(requested):
            raise RunnerError(
                f"existing {phase} gate differs from completed plan: {path}"
            )
        return existing
    try:
        _write_new_json(path, payload)
        return payload
    except RunnerError as exc:
        # Concurrent per-GPU workers can finish together.  Only accept the
        # winner's gate when it is identical apart from its creation time.
        if not path.exists():
            raise
        existing = load_json(path, f"concurrent {phase} gate")
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        requested = dict(payload)
        requested.pop("created_utc", None)
        if canonical_json(comparable) != canonical_json(requested):
            raise exc
        return existing


def _validate_internal_gate(
    config: Mapping[str, Any], phase: str, manifest_sha256: str
) -> Mapping[str, Any]:
    path = _phase_gate_path(config, phase)
    receipt = load_json(path, f"{phase} gate")
    if (
        receipt.get("schema") != GATE_SCHEMA
        or receipt.get("phase") != phase
        or receipt.get("verdict") != "GO"
        or receipt.get("experiment_id") != config["experiment_id"]
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("execution_commit") != config["provenance"]["execution_commit"]
    ):
        raise RunnerError(f"{phase} gate is not a GO for this execution: {path}")
    plan = build_plan(config, manifest_sha256, phase)
    if receipt.get("jobs_sha256") != plan["jobs_sha256"]:
        raise RunnerError(f"{phase} gate jobs_sha256 differs from the frozen plan")
    rows = receipt.get("exit_receipts")
    if not isinstance(rows, list):
        raise RunnerError(f"{phase} gate exit_receipts must be a complete list")
    indexed = {
        str(row.get("job_id", "")): row for row in rows if isinstance(row, Mapping)
    }
    expected = {job["job_id"]: job for job in plan["jobs"]}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise RunnerError(
            f"{phase} gate job set is incomplete, duplicated, or unexpected"
        )
    for job_id, job in expected.items():
        row = indexed[job_id]
        path_item = Path(job["record_dir"]) / "exit.json"
        if row.get("path") != str(path_item):
            raise RunnerError(f"{phase} gate receipt path mismatch for {job_id}")
        if not path_item.is_file() or sha256_file(path_item) != row.get("sha256"):
            raise RunnerError(f"{phase} gate dependency changed: {path_item}")
        _completed_receipt(job, manifest_sha256)
    return receipt


@contextlib.contextmanager
def phase_lock(config: Mapping[str, Any], name: str) -> Iterator[None]:
    path = _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "locks", f"{name}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"another runner holds lock {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            canonical_json(
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "acquired_utc": utc_now(),
                    "name": name,
                }
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _environment_snapshot(
    config: Mapping[str, Any], phase: str, gpu: int, manifest_sha256: str
) -> Mapping[str, Any]:
    snapshot = {
        "schema": "ect.q256.g110-moment-transport-environment/v1",
        "captured_utc": utc_now(),
        "phase": phase,
        "gpu": gpu,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "manifest_sha256": manifest_sha256,
        "execution_commit": config["provenance"]["execution_commit"],
        "container_identity": config["provenance"]["container_identity"],
        "dataset": config["provenance"]["dataset"],
        "formal_arms": config["formal"]["arms"],
        "compatibility": verify_compatibility_fresh_arms(config),
        "gpu_snapshot": _query_gpu(config, gpu),
    }
    root = Path(config["paths"]["run_root"])
    name = f"{phase}-gpu{gpu}-{time.time_ns()}-{os.getpid()}.json"
    path = _safe_child(root, "_runner", "environment", name)
    _write_new_json(path, snapshot)
    return {"path": str(path), "sha256": sha256_file(path)}


def run_worker(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest_sha256: str,
    *,
    phase: str,
    gpu: int,
) -> None:
    jobs = [job for job in plan["jobs"] if job["gpu"] == gpu]
    if not jobs:
        raise RunnerError(f"phase {phase} has no jobs assigned to GPU {gpu}")
    with phase_lock(config, f"{phase}-gpu{gpu}"):
        _environment_snapshot(config, phase, gpu, manifest_sha256)
        for job in jobs:
            execute_job(config, job, manifest_sha256=manifest_sha256)
    # The phase GO is written only when every GPU's queue is complete.  A
    # racing worker may see another queue incomplete, which is expected.
    try:
        _write_phase_gate(config, plan, manifest_sha256, phase)
    except RunnerError as exc:
        if "job is incomplete" not in str(exc):
            raise


def _tmux_session_name(config: Mapping[str, Any], phase: str, gpu: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "-", config["experiment_id"])
    return f"{base}-{phase}-gpu{gpu}"[:120]


def _tmux_has_session(binary: str, name: str) -> bool:
    result = subprocess.run(
        [binary, "has-session", "-t", name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _tmux_pane_state(binary: str, name: str) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                binary,
                "display-message",
                "-p",
                "-t",
                name,
                "#{pane_dead}\t#{pane_start_command}\t#{pane_pid}\t#{pane_dead_status}",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).rstrip("\n")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"cannot inspect tmux session {name}: {exc}") from exc
    fields = output.split("\t", 3)
    if len(fields) != 4 or fields[0] not in ("0", "1"):
        raise RunnerError(f"unexpected tmux pane identity for {name}: {output!r}")
    return {
        "dead": fields[0] == "1",
        "start_command": fields[1],
        "pane_pid": fields[2],
        "dead_status": fields[3],
    }


def _next_worker_record_dir(config: Mapping[str, Any], phase: str, gpu: int) -> Path:
    base = _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "workers", f"{phase}-gpu{gpu}"
    )
    base.mkdir(parents=True, exist_ok=True)
    attempts = []
    for child in base.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})", child.name)
        if match and child.is_dir() and not child.is_symlink():
            attempts.append(int(match.group(1)))
    attempt = max(attempts, default=0) + 1
    return _safe_child(base, f"attempt-{attempt:04d}")


def _register_external_worker(
    config: Mapping[str, Any],
    *,
    phase: str,
    gpu: int,
    manifest_sha256: str,
    process_argv: Sequence[str],
) -> Path:
    """Self-register a host-tmux-managed worker before any gate or job work."""
    for _ in range(1000):
        record_dir = _next_worker_record_dir(config, phase, gpu)
        try:
            record_dir.mkdir()
            break
        except FileExistsError:
            continue
        except OSError as exc:
            raise RunnerError(
                f"cannot create external worker record {record_dir}: {exc}"
            ) from exc
    else:
        raise RunnerError(
            f"cannot allocate external worker record for {phase} GPU {gpu}"
        )
    _write_new_json(
        record_dir / "launch.json",
        {
            "schema": "ect.q256.g110-moment-transport-worker-launch/v1",
            "experiment_id": config["experiment_id"],
            "manifest_sha256": manifest_sha256,
            "execution_commit": config["provenance"]["execution_commit"],
            "phase": phase,
            "gpu": gpu,
            "session": None,
            "dispatch": "external-host-tmux-foreground",
            "command": list(process_argv),
            "pid": os.getpid(),
            "host": platform.node(),
            "created_utc": utc_now(),
        },
    )
    return _start_worker_record(
        config,
        record_dir=record_dir,
        phase=phase,
        gpu=gpu,
        manifest_sha256=manifest_sha256,
    )


def launch_tmux_workers(
    config: Mapping[str, Any],
    manifest_path: Path,
    plan: Mapping[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    binary = config["runtime"]["tmux_binary"]
    if shutil.which(binary) is None:
        raise RunnerError(f"tmux command not found: {binary}")
    gpus = sorted({int(job["gpu"]) for job in plan["jobs"]})
    launches = []
    for gpu in gpus:
        name = _tmux_session_name(config, phase, gpu)
        if _tmux_has_session(binary, name):
            worker_base = _safe_child(
                Path(config["paths"]["run_root"]),
                "_runner",
                "workers",
                f"{phase}-gpu{gpu}",
            )
            launches_for_name = (
                sorted(worker_base.glob("attempt-*/launch.json"))
                if worker_base.is_dir()
                else []
            )
            if not launches_for_name:
                raise RunnerError(
                    f"tmux session {name} has no frozen worker launch identity"
                )
            identity = load_json(
                launches_for_name[-1], "existing worker launch identity"
            )
            if (
                identity.get("manifest_sha256") != plan["manifest_sha256"]
                or identity.get("phase") != phase
                or identity.get("gpu") != gpu
                or identity.get("session") != name
            ):
                raise RunnerError(
                    f"tmux session {name} belongs to another worker identity"
                )
            pane = _tmux_pane_state(binary, name)
            if (
                pane["dead"]
                or not pane["pane_pid"].isdigit()
                or identity.get("shell_command") != pane["start_command"]
            ):
                raise RunnerError(
                    f"tmux session {name} does not match its frozen live worker command"
                )
            launches.append(
                {
                    "gpu": gpu,
                    "session": name,
                    "status": "already_running",
                    "worker_record_dir": str(launches_for_name[-1].parent),
                    "pane_pid": pane["pane_pid"],
                }
            )
            continue
        worker_record_dir = _next_worker_record_dir(config, phase, gpu)
        command = [
            *config["runtime"]["worker_command"],
            str(
                Path(config["paths"]["repo_root"])
                / "scripts"
                / "run_q256_g110_moment_transport.py"
            ),
            "--manifest",
            str(manifest_path),
            "--phase",
            phase,
            "--worker-gpu",
            str(gpu),
            "--expected-manifest-sha256",
            plan["manifest_sha256"],
            "--worker-record-dir",
            str(worker_record_dir),
        ]
        # Target tmux is 1.8: it has no new-session -c and accepts the pane
        # command as one string.  Every dynamic token is shell-quoted, and a
        # dedicated pane log survives even when the tmux session exits early.
        pane_log = worker_record_dir / "pane.log"
        shell_command = (
            f"cd {shlex.quote(config['paths']['repo_root'])} && exec "
            + " ".join(shlex.quote(argument) for argument in command)
            + f" > {shlex.quote(str(pane_log))} 2>&1"
        )
        launch_identity = {
            "schema": "ect.q256.g110-moment-transport-worker-launch/v1",
            "experiment_id": config["experiment_id"],
            "manifest_sha256": plan["manifest_sha256"],
            "execution_commit": config["provenance"]["execution_commit"],
            "phase": phase,
            "gpu": gpu,
            "session": name,
            "command": command,
            "shell_command": shell_command,
            "pane_log": str(pane_log),
            "created_utc": utc_now(),
        }
        _write_new_json(worker_record_dir / "launch.json", launch_identity)
        tmux_command = [binary, "new-session", "-d", "-s", name, shell_command]
        result = subprocess.run(
            tmux_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            _write_new_json(
                worker_record_dir / "launch_failure.json",
                {
                    **launch_identity,
                    "tmux_command": tmux_command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "failed_utc": utc_now(),
                },
            )
            raise RunnerError(f"tmux launch failed for {name}: {result.stderr.strip()}")
        launches.append(
            {
                "gpu": gpu,
                "session": name,
                "status": "launched",
                "command": command,
                "worker_record_dir": str(worker_record_dir),
                "pane_log": str(pane_log),
            }
        )
    return launches


def _plan_path(config: Mapping[str, Any], phase: str) -> Path:
    return _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "plans", f"{phase}.json"
    )


def _start_worker_record(
    config: Mapping[str, Any],
    *,
    record_dir: Path,
    phase: str,
    gpu: int,
    manifest_sha256: str,
) -> Path:
    expected_base = _safe_child(
        Path(config["paths"]["run_root"]), "_runner", "workers", f"{phase}-gpu{gpu}"
    )
    resolved = record_dir.resolve(strict=True)
    try:
        resolved.relative_to(expected_base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RunnerError(
            f"worker record directory is outside its frozen queue: {record_dir}"
        ) from exc
    launch = load_json(resolved / "launch.json", "worker launch identity")
    if (
        launch.get("manifest_sha256") != manifest_sha256
        or launch.get("execution_commit") != config["provenance"]["execution_commit"]
        or launch.get("experiment_id") != config["experiment_id"]
        or launch.get("phase") != phase
        or launch.get("gpu") != gpu
    ):
        raise RunnerError("worker launch identity differs from the frozen queue")
    if (resolved / "exit.json").exists():
        raise RunnerError(f"worker attempt already has an exit receipt: {resolved}")
    _write_new_json(
        resolved / "started.json",
        {
            "schema": "ect.q256.g110-moment-transport-worker-start/v1",
            "experiment_id": config["experiment_id"],
            "manifest_sha256": manifest_sha256,
            "execution_commit": config["provenance"]["execution_commit"],
            "phase": phase,
            "gpu": gpu,
            "pid": os.getpid(),
            "host": platform.node(),
            "started_utc": utc_now(),
        },
    )
    return resolved


def _write_worker_exit(
    config: Mapping[str, Any],
    record_dir: Path,
    *,
    phase: str,
    gpu: int,
    manifest_sha256: str,
    status: str,
    error: str | None,
) -> None:
    _write_new_json(
        record_dir / "exit.json",
        {
            "schema": "ect.q256.g110-moment-transport-worker-exit/v1",
            "experiment_id": config["experiment_id"],
            "manifest_sha256": manifest_sha256,
            "execution_commit": config["provenance"]["execution_commit"],
            "phase": phase,
            "gpu": gpu,
            "pid": os.getpid(),
            "host": platform.node(),
            "status": status,
            "error": error,
            "finished_utc": utc_now(),
        },
    )


def status_report(config: Mapping[str, Any], manifest_sha256: str) -> Mapping[str, Any]:
    phases: MutableMapping[str, Any] = {}
    for phase in ("prepare", "smoke", "formal"):
        try:
            plan = build_plan(config, manifest_sha256, phase)
        except RunnerError as exc:
            phases[phase] = {"error": str(exc)}
            continue
        counts = {"completed": 0, "pending": 0, "failed_or_ambiguous": 0}
        jobs = []
        for job in plan["jobs"]:
            exit_path = Path(job["record_dir"]) / "exit.json"
            if not exit_path.exists():
                state = (
                    "failed_or_ambiguous"
                    if Path(job["record_dir"]).exists()
                    else "pending"
                )
            else:
                try:
                    receipt = load_json(exit_path, "exit receipt")
                    state = (
                        "completed"
                        if receipt.get("status") == "completed"
                        else "failed_or_ambiguous"
                    )
                except RunnerError:
                    state = "failed_or_ambiguous"
            counts[state] += 1
            jobs.append(
                {
                    "job_id": job["job_id"],
                    "state": state,
                    "exit_receipt": str(exit_path),
                }
            )
        gate = _phase_gate_path(config, phase)
        phases[phase] = {
            "counts": counts,
            "gate": str(gate) if gate.exists() else None,
            "jobs": jobs,
        }
    sessions = []
    binary = config["runtime"]["tmux_binary"]
    if shutil.which(binary) is not None:
        for phase in ("smoke", "formal"):
            plan = build_plan(config, manifest_sha256, phase)
            for gpu in sorted({job["gpu"] for job in plan["jobs"]}):
                name = _tmux_session_name(config, phase, int(gpu))
                sessions.append(
                    {"session": name, "running": _tmux_has_session(binary, name)}
                )
    return {
        "schema": "ect.q256.g110-moment-transport-status/v1",
        "experiment_id": config["experiment_id"],
        "manifest_sha256": manifest_sha256,
        "captured_utc": utc_now(),
        "phases": phases,
        "tmux": sessions,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "smoke", "formal", "status"), required=True
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print; create nothing"
    )
    parser.add_argument(
        "--foreground", action="store_true", help="run queues in this process"
    )
    parser.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-manifest-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-record-dir", type=Path, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config: dict[str, Any] | None = None
    manifest_sha: str | None = None
    worker_record: Path | None = None
    try:
        config, manifest_sha = load_manifest(
            args.manifest.resolve(), verify_artifacts=True
        )
        if args.phase == "status" and (
            args.dry_run
            or args.foreground
            or args.worker_gpu is not None
            or args.worker_record_dir is not None
            or args.expected_manifest_sha256 is not None
        ):
            raise RunnerError("status does not accept execution or worker options")
        if args.dry_run and (args.foreground or args.worker_gpu is not None):
            raise RunnerError("--dry-run cannot be combined with worker options")
        if args.worker_gpu is None and (
            args.worker_record_dir is not None
            or args.expected_manifest_sha256 is not None
        ):
            raise RunnerError("worker identity options require --worker-gpu")
        if args.phase == "prepare" and args.worker_gpu is not None:
            raise RunnerError("prepare does not accept --worker-gpu")
        if (
            args.worker_gpu is not None
            and args.foreground
            and args.worker_record_dir is None
            and args.expected_manifest_sha256 is None
        ):
            raise RunnerError(
                "externally managed --foreground --worker-gpu requires "
                "--expected-manifest-sha256"
            )
        if (
            args.worker_gpu is not None
            and not args.foreground
            and args.worker_record_dir is None
        ):
            raise RunnerError(
                "detached workers require --worker-record-dir; externally managed workers "
                "must pass --foreground --worker-gpu"
            )
        if (
            args.expected_manifest_sha256 is not None
            and _require_sha256(
                args.expected_manifest_sha256, "--expected-manifest-sha256"
            )
            != manifest_sha
        ):
            raise RunnerError(
                "worker manifest SHA256 changed after tmux dispatch: "
                f"{manifest_sha} != {args.expected_manifest_sha256}"
            )
        if args.worker_gpu is not None and args.worker_record_dir is not None:
            worker_record = _start_worker_record(
                config,
                record_dir=args.worker_record_dir,
                phase=args.phase,
                gpu=args.worker_gpu,
                manifest_sha256=manifest_sha,
            )
        elif args.worker_gpu is not None:
            process_argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                *(list(argv) if argv is not None else sys.argv[1:]),
            ]
            worker_record = _register_external_worker(
                config,
                phase=args.phase,
                gpu=args.worker_gpu,
                manifest_sha256=manifest_sha,
                process_argv=process_argv,
            )
        if args.phase == "status":
            print(
                json.dumps(
                    status_report(config, manifest_sha), indent=2, sort_keys=True
                )
            )
            return 0
        plan = build_plan(config, manifest_sha, args.phase)
        if args.phase in ("prepare", "smoke", "formal"):
            verify_git_provenance(config)
        if args.phase == "smoke":
            _validate_prepare_receipt(config, manifest_sha)
            verify_external_gates(config["smoke_gates"], config)
        elif args.phase == "formal":
            _validate_prepare_receipt(config, manifest_sha)
            _validate_internal_gate(config, "smoke", manifest_sha)
            verify_compatibility_fresh_arms(config)
            verify_external_gates(config["formal_gates"], config)

        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        plan_path = _plan_path(config, args.phase)
        _write_or_verify_json(plan_path, plan)
        if args.phase == "prepare":
            with phase_lock(config, "prepare"):
                receipt = run_prepare(config, plan, manifest_sha)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0

        if args.worker_gpu is not None:
            run_worker(
                config,
                plan,
                manifest_sha,
                phase=args.phase,
                gpu=args.worker_gpu,
            )
            if worker_record is not None:
                _write_worker_exit(
                    config,
                    worker_record,
                    phase=args.phase,
                    gpu=args.worker_gpu,
                    manifest_sha256=manifest_sha,
                    status="completed",
                    error=None,
                )
            return 0
        if args.foreground:
            for gpu in sorted({int(job["gpu"]) for job in plan["jobs"]}):
                run_worker(config, plan, manifest_sha, phase=args.phase, gpu=gpu)
            _write_phase_gate(config, plan, manifest_sha, args.phase)
            return 0
        with phase_lock(config, f"{args.phase}-dispatch"):
            launches = launch_tmux_workers(
                config, args.manifest.resolve(), plan, args.phase
            )
        print(
            json.dumps(
                {"phase": args.phase, "tmux": launches}, indent=2, sort_keys=True
            )
        )
        return 0
    except RunnerError as exc:
        if (
            config is not None
            and manifest_sha is not None
            and worker_record is not None
            and args.worker_gpu is not None
            and not (worker_record / "exit.json").exists()
        ):
            try:
                _write_worker_exit(
                    config,
                    worker_record,
                    phase=args.phase,
                    gpu=args.worker_gpu,
                    manifest_sha256=manifest_sha,
                    status="failed",
                    error=str(exc),
                )
            except RunnerError:
                pass
        print(f"[q256-moment-transport-runner] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
