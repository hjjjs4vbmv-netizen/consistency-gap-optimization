#!/usr/bin/env python3
"""Bind seed6/7 extension outputs and run the frozen q256 evaluator.

This adapter changes only the matrix membership (seeds 6 and 7), provenance
classification, and primary-first job ordering.  Sampling and metric commands
are copied exactly from the frozen evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_HEAD = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
SEEDS = (6, 7)
ARMS = ("A", "B", "C", "D")
EXTENSION_PROTOCOL = "q256-target-weight-secondary-extension-frozen-evaluation-v1"
BINDING_SCHEMA = "ect.q256.target-weight-secondary-extension-evaluation-binding/v1"
CELL_SCHEMA = "ect.q256.target-weight-secondary-extension-cell-binding/v1"


class ExtensionEvaluationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ExtensionEvaluationError(message)


def load_module(path: Path, name: str) -> Any:
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot import adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def regular_binding(path: Path, helper: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing immutable artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": helper.sha256_file(path),
    }


def extension_cell_record(
    helper: Any,
    run_dir: Path,
    seed: int,
    arm: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    record = helper.verify_direct_cell(run_dir, seed, arm)
    record["schema"] = CELL_SCHEMA
    record["extension_classification"] = (
        "secondary_precision_extension_not_original_preregistration"
    )
    record["replaces_preregistered_seed"] = False
    audit_cells = audit.get("cells")
    audit_cell = audit_cells.get(arm) if isinstance(audit_cells, dict) else None
    if not isinstance(audit_cell, dict):
        fail(f"seed{seed} audit has no arm {arm}")
    audit_hashes = audit_cell.get("artifact_hashes")
    checkpoint_audit = (
        audit_hashes.get("network-snapshot-latest.pkl")
        if isinstance(audit_hashes, dict)
        else None
    )
    if (
        audit_cell.get("attempts") != 2000
        or audit_cell.get("accepted_updates") != record["successful_optimizer_steps"]
        or audit_cell.get("processed_kimg") != 256.0
        or audit_cell.get("semantic_nonfinite_count") != 0
        or audit_cell.get("nonpositive_denominator_count") != 0
        or audit_cell.get("raw_grad_skip_mismatch_count") != 0
        or not isinstance(checkpoint_audit, dict)
        or checkpoint_audit.get("sha256") != record["checkpoint_sha256"]
        or checkpoint_audit.get("bytes") != record["checkpoint_bytes"]
    ):
        fail(f"seed{seed}/{arm} audit/checkpoint binding mismatch")
    record["integrity_audit_checkpoint_binding"] = dict(checkpoint_audit)
    return record


def validate_seed_audit(path: Path, seed: int) -> dict[str, Any]:
    audit = load_json(path)
    if (
        audit.get("status") != "PASS"
        or audit.get("seed") != seed
        or audit.get("source_head") != EXPECTED_HEAD
        or audit.get("four_arm_complete") is not True
        or audit.get("denominator_integrity") is not True
        or audit.get("common_initial_state_identity") is not True
        or audit.get("extension_classification")
        != "secondary_precision_extension_not_original_preregistration"
        or audit.get("replaces_preregistered_seed") is not False
    ):
        fail(f"seed{seed} integrity audit is not an exact extension PASS")
    telemetry = audit.get("telemetry_identity_checks")
    if not isinstance(telemetry, dict) or telemetry.get("all_pass") is not True:
        fail(f"seed{seed} telemetry identity audit did not pass")
    return audit


def validate_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if report.get("status") != "PASS":
        fail("extension training report is not PASS")
    if report.get("extension_classification") not in (
        None,
        "secondary_precision_extension_not_original_preregistration",
    ):
        fail("extension report has the wrong classification")
    return report


def create_binding(
    matrix_dir: Path,
    extension_root: Path,
    evaluator: Any,
    helper: Any,
) -> None:
    if matrix_dir.exists():
        fail(f"refuse to reuse matrix binding directory: {matrix_dir}")
    if not matrix_dir.parent.is_dir():
        fail(f"matrix binding parent is missing: {matrix_dir.parent}")
    extension_root = extension_root.resolve(strict=True)
    if extension_root.is_symlink() or not extension_root.is_dir():
        fail(f"invalid extension root: {extension_root}")
    source = evaluator.source_snapshot(require_clean=True)
    if source.get("git_branch") != evaluator.EXPECTED_BRANCH:
        fail("wrong frozen evaluator branch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"],
        cwd=evaluator.REPO_ROOT,
    )

    audits: dict[int, dict[str, Any]] = {}
    audit_receipts = []
    for seed in SEEDS:
        path = extension_root / "integrity" / f"seed{seed}_integrity_audit.json"
        audits[seed] = validate_seed_audit(path, seed)
        audit_receipts.append(regular_binding(path, helper))
    report_path = extension_root / "q256_factorial_seed6_7_extension_report.json"
    validate_report(report_path)
    report_receipt = regular_binding(report_path, helper)

    cells = [
        extension_cell_record(
            helper,
            extension_root / f"seed{seed}" / f"arm{arm}",
            seed,
            arm,
            audits[seed],
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    for seed in SEEDS:
        identities = {
            cell["initial_common_state_sha256"]
            for cell in cells
            if cell["seed"] == seed
        }
        if len(identities) != 1:
            fail(f"seed{seed} arms do not share one common initial state")

    matrix_dir.mkdir(mode=0o750)
    (matrix_dir / "cells").mkdir(mode=0o750)
    cell_receipts = []
    for cell in cells:
        path = matrix_dir / "cells" / f"seed{cell['seed']}-arm{cell['arm']}.json"
        helper.write_exclusive_json(path, cell)
        cell_receipts.append(regular_binding(path, helper))

    adapter_path = Path(__file__).resolve(strict=True)
    helper_path = Path(helper.__file__).resolve(strict=True)
    payload = {
        "schema": BINDING_SCHEMA,
        "status": "PASS",
        "created_utc": helper.utc_now(),
        "extension_classification": (
            "secondary_precision_extension_not_original_preregistration"
        ),
        "replaces_preregistered_seed": False,
        "selection_policy": "all_8_extension_final_256kimg_checkpoints",
        "extension_root": str(extension_root),
        "cell_count": 8,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "cell_receipts": cell_receipts,
        "cell_receipts_tree_sha256": helper.canonical_sha256(cell_receipts),
        "integrity_audit_receipts": audit_receipts,
        "extension_report_receipt": report_receipt,
        "training_source_git_head": EXPECTED_HEAD,
        "evaluator_source_git_head": source["git_head"],
        "evaluator_source_content_sha256": source["content_sha256"],
        "adapter": regular_binding(adapter_path, helper),
        "formal_numerical_adapter": regular_binding(helper_path, helper),
        "metric_numerical_semantics_changed": False,
    }
    helper.write_exclusive_json(matrix_dir / "extension_matrix_binding.json", payload)


def load_bound_matrix(
    matrix_dir: Path,
    evaluator: Any,
    helper: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_dir = matrix_dir.resolve(strict=True)
    binding_path = matrix_dir / "extension_matrix_binding.json"
    binding = load_json(binding_path)
    if (
        binding.get("schema") != BINDING_SCHEMA
        or binding.get("status") != "PASS"
        or binding.get("cell_count") != 8
        or binding.get("seeds") != list(SEEDS)
        or binding.get("arms") != list(ARMS)
        or binding.get("training_source_git_head") != EXPECTED_HEAD
        or binding.get("extension_classification")
        != "secondary_precision_extension_not_original_preregistration"
    ):
        fail("extension matrix binding identity mismatch")
    for key in ("adapter", "formal_numerical_adapter", "extension_report_receipt"):
        receipt = binding.get(key)
        if not isinstance(receipt, dict):
            fail(f"matrix binding lacks {key}")
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if (
            receipt.get("bytes") != path.stat().st_size
            or receipt.get("sha256") != helper.sha256_file(path)
        ):
            fail(f"bound artifact changed: {path}")
    audits = binding.get("integrity_audit_receipts")
    if not isinstance(audits, list) or len(audits) != 2:
        fail("matrix binding lacks two audit receipts")
    audit_by_seed: dict[int, dict[str, Any]] = {}
    for receipt, seed in zip(audits, SEEDS):
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if receipt.get("sha256") != helper.sha256_file(path):
            fail(f"integrity audit changed: {path}")
        audit_by_seed[seed] = validate_seed_audit(path, seed)

    receipts = binding.get("cell_receipts")
    if not isinstance(receipts, list) or len(receipts) != 8:
        fail("extension matrix binding must contain eight cells")
    if binding.get("cell_receipts_tree_sha256") != helper.canonical_sha256(receipts):
        fail("extension cell receipt tree is stale")
    cells = []
    for receipt in receipts:
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if path.parent != matrix_dir / "cells":
            fail(f"cell binding escapes matrix directory: {path}")
        if (
            receipt.get("bytes") != path.stat().st_size
            or receipt.get("sha256") != helper.sha256_file(path)
        ):
            fail(f"cell binding changed: {path}")
        recorded = load_json(path)
        seed, arm = int(recorded["seed"]), str(recorded["arm"])
        current = extension_cell_record(
            helper, Path(recorded["run_dir"]), seed, arm, audit_by_seed[seed]
        )
        comparable = set(recorded) - {"verified_utc"}
        if {key: recorded[key] for key in comparable} != {
            key: current[key] for key in comparable
        }:
            fail(f"immutable extension evidence changed: {path}")
        cells.append(
            {
                "arm": arm,
                "seed": seed,
                "run_dir": recorded["run_dir"],
                "checkpoint": recorded["checkpoint"],
                "checkpoint_sha256": recorded["checkpoint_sha256"],
                "checkpoint_bytes": recorded["checkpoint_bytes"],
                "training_validation_receipt": str(path),
                "training_validation_receipt_sha256": receipt["sha256"],
                "training_hash_receipt": str(path),
                "training_hash_receipt_sha256": receipt["sha256"],
                "training_source_git_head": EXPECTED_HEAD,
                "training_source_content_sha256": binding[
                    "evaluator_source_content_sha256"
                ],
                "initial_common_state_sha256": recorded[
                    "initial_common_state_sha256"
                ],
                "amp_skip_attempts": recorded["amp_skip_attempts"],
                "successful_optimizer_steps": recorded[
                    "successful_optimizer_steps"
                ],
                "amp_skip_signature_expected_value_enforced": False,
                "extension_classification": binding["extension_classification"],
            }
        )
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if {(cell["seed"], cell["arm"]) for cell in cells} != expected:
        fail("extension matrix cell set is incomplete")
    matrix = {
        "matrix_dir": str(matrix_dir),
        "extension_matrix_binding": str(binding_path),
        "extension_matrix_binding_sha256": helper.sha256_file(binding_path),
        "training_source_git_head": EXPECTED_HEAD,
        "training_source_content_sha256": binding[
            "evaluator_source_content_sha256"
        ],
        "cell_count": 8,
        "expected_amp_skip_attempts": None,
        "selection_policy": binding["selection_policy"],
        "extension_classification": binding["extension_classification"],
        "replaces_preregistered_seed": False,
        "provenance_adapter": binding["adapter"],
        "formal_numerical_adapter": binding["formal_numerical_adapter"],
    }
    return sorted(cells, key=lambda item: (item["seed"], ARMS.index(item["arm"]))), matrix


def build_jobs(
    evaluator: Any,
    cells: Sequence[Mapping[str, Any]],
    output_root: Path,
    base_port: int,
) -> list[dict[str, Any]]:
    jobs = []
    for nfe, mid_t in evaluator.NFE_SETTINGS.items():
        for cell in cells:
            job_id = f"seed{cell['seed']}-arm{cell['arm']}-nfe{nfe}"
            target = output_root / "jobs" / job_id
            command = [
                "bash",
                str(evaluator.REPO_ROOT / "scripts" / "evaluate_checkpoint.sh"),
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
                f"--metrics={','.join(evaluator.METRICS)}",
                "--metric-repeats=1",
                f"--sample-seeds={evaluator.SAMPLE_SEEDS}",
                f"--seed={evaluator.METRIC_SEED}",
                "--retain-generated-artifacts",
                f"--desc={EXTENSION_PROTOCOL}-{job_id}",
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
                    "training_validation_receipt": cell[
                        "training_validation_receipt"
                    ],
                    "training_validation_receipt_sha256": cell[
                        "training_validation_receipt_sha256"
                    ],
                    "training_hash_receipt": cell["training_hash_receipt"],
                    "training_hash_receipt_sha256": cell[
                        "training_hash_receipt_sha256"
                    ],
                    "sample_count": evaluator.SAMPLE_COUNT,
                    "sample_seeds": evaluator.SAMPLE_SEEDS,
                    "metric_seed": evaluator.METRIC_SEED,
                    "metrics": list(evaluator.METRICS),
                    "precision": "fp32",
                    "output_directory": str(target),
                    "command_argv_template": command,
                }
            )
    if len(jobs) != 16:
        fail(f"extension evaluation must contain exactly 16 jobs, got {len(jobs)}")
    return jobs


def install_gpu_audit_adapter(evaluator: Any, gpu_uuid: str) -> Any:
    import pynvml

    pynvml.nvmlInit()
    handles: dict[str, Any] = {}
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        observed = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(observed, bytes):
            observed = observed.decode("ascii")
        handles[str(observed)] = handle
    if gpu_uuid not in handles:
        fail(f"NVML cannot resolve selected GPU: {gpu_uuid}")

    original_stream = evaluator.training_launcher.stream_process
    original_query = evaluator.training_launcher.query_gpu_compute_processes
    original_tree = evaluator.training_launcher.process_tree_pids

    def query(selected: str, *, timeout_seconds: float = 5.0) -> list[dict[str, object]]:
        del timeout_seconds
        if selected != gpu_uuid:
            fail(f"GPU audit received another UUID: {selected}")
        records = []
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handles[selected]):
            pid = int(process.pid)
            proc = Path("/proc") / str(pid)
            try:
                name = (proc / "comm").read_text(encoding="utf-8").strip()
            except OSError:
                name = "<exited>"
            metadata: dict[str, object] = {}
            try:
                raw = (proc / "stat").read_bytes()
                fields = raw[raw.rfind(b")") + 2 :].split()
                metadata.update(
                    ppid=int(fields[1]),
                    process_group_id=int(fields[2]),
                    session_id=int(fields[3]),
                )
            except (OSError, ValueError, IndexError):
                pass
            try:
                metadata["executable"] = os.readlink(proc / "exe")
            except OSError:
                pass
            try:
                metadata["command_line"] = (
                    (proc / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode("utf-8", "replace")[:1024]
                )
            except OSError:
                pass
            records.append(
                {
                    "pid": pid,
                    "process_name": name,
                    "used_gpu_memory_mib": str(
                        int(process.usedGpuMemory) // (1024 * 1024)
                    ),
                    **metadata,
                }
            )
        return records

    def tree(root_pid: int, *, timeout_seconds: float = 5.0) -> set[int]:
        del timeout_seconds
        snapshot = evaluator.training_launcher.linux_process_snapshot()
        return {
            root_pid,
            *evaluator.training_launcher.snapshot_descendants(snapshot, root_pid),
        }

    def stream(*args: Any, **kwargs: Any) -> int:
        evaluator.training_launcher.query_gpu_compute_processes = query
        evaluator.training_launcher.process_tree_pids = tree
        try:
            return original_stream(*args, **kwargs)
        finally:
            monitor = kwargs.get("gpu_monitor_record")
            if isinstance(monitor, dict):
                monitor["gpu_audit_backend"] = {
                    "schema": "ect.q256.gpu-audit-backend/v1",
                    "gpu_process_source": "pynvml",
                    "process_tree_source": "linux-procfs",
                    "poll_interval_seconds": 1.0,
                    "metric_numerical_semantics_changed": False,
                }
            evaluator.training_launcher.query_gpu_compute_processes = original_query
            evaluator.training_launcher.process_tree_pids = original_tree

    evaluator.training_launcher.stream_process = stream
    return pynvml


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--formal-adapter", type=Path, required=True)
    parser.add_argument("--reuse-bound-matrix", action="store_true")
    parser.add_argument("--bind-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-port", type=int, default=32900)
    parser.add_argument(
        "--lock-root", type=Path, default=Path("/data/temp/ECT001-q256-evaluation-locks")
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts import run_q256_target_weight_evaluation as evaluator

    helper = load_module(args.formal_adapter, "q256_formal_adapter_helper")
    if evaluator.REPO_ROOT.resolve() != repo:
        fail("loaded evaluator from unexpected repository")
    if args.reuse_bound_matrix:
        if not args.matrix_dir.resolve().is_dir():
            fail("bound extension matrix is missing")
    else:
        create_binding(args.matrix_dir.resolve(), args.extension_root, evaluator, helper)
    if args.bind_only:
        cells, matrix = load_bound_matrix(args.matrix_dir, evaluator, helper)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "bind-only",
                    "cell_count": len(cells),
                    "seeds": list(SEEDS),
                    "arms": list(ARMS),
                    "matrix": matrix,
                },
                sort_keys=True,
            )
        )
        return 0

    original_plan = evaluator.build_plan
    original_write = evaluator.write_json_exclusive

    evaluator.SEEDS = SEEDS
    evaluator.PROTOCOL = EXTENSION_PROTOCOL
    evaluator.load_training_matrix = lambda matrix: load_bound_matrix(
        matrix, evaluator, helper
    )
    evaluator.build_jobs = lambda cells, root, port: build_jobs(
        evaluator, cells, root, port
    )

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        plan = original_plan(**kwargs)
        plan["selection_policy"] = "all_8_extension_final_256kimg_checkpoints"
        plan["independent_unit"] = {
            "name": "training_seed",
            "values": list(SEEDS),
            "n": 2,
        }
        plan["extension_classification"] = (
            "secondary_precision_extension_not_original_preregistration"
        )
        plan["replaces_preregistered_seed"] = False
        return plan

    def write_json(path: Path, value: Mapping[str, Any]) -> None:
        payload = dict(value)
        if path.name == "evaluation_completion.json":
            payload["selection_policy"] = (
                "all_8_extension_final_256kimg_checkpoints"
            )
            payload["independent_training_seed_n"] = 2
            payload["extension_classification"] = (
                "secondary_precision_extension_not_original_preregistration"
            )
            payload["replaces_preregistered_seed"] = False
        original_write(path, payload)

    def revalidate(
        run_dir: Path,
        *,
        phase: str,
        arm: str,
        seed: int,
        expected_skip_attempts: list[int] | None,
        runtime_command: Sequence[str] | None,
        process_env: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        del runtime_command, process_env
        if phase != "formal" or expected_skip_attempts is not None:
            fail("extension revalidation received a non-frozen contract")
        record = helper.verify_direct_cell(run_dir, seed, arm)
        return {
            "status": "PASS",
            "kind": "secondary-extension-read-only-revalidation",
            "checkpoint_sha256": record["checkpoint_sha256"],
            "artifact_count": len(record["artifacts"]),
            "checkpoint_mutation": False,
        }

    evaluator.build_plan = build_plan
    evaluator.write_json_exclusive = write_json
    evaluator.training_launcher.deep_revalidate_existing_arm = revalidate
    pynvml = install_gpu_audit_adapter(evaluator, args.gpu)
    execute_args = argparse.Namespace(
        matrix_dir=args.matrix_dir,
        data=args.data or evaluator.DEFAULT_DATASET,
        outdir=args.outdir,
        gpu=args.gpu,
        base_port=args.base_port,
        lock_root=args.lock_root,
        evaluator_repair_base_git_head=EXPECTED_HEAD,
    )
    try:
        return evaluator.execute(execute_args)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtensionEvaluationError as exc:
        raise SystemExit(f"[q256-extension-evaluation-adapter] ERROR: {exc}") from exc
