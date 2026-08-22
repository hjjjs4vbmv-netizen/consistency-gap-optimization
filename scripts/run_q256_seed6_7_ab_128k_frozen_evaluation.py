#!/usr/bin/env python3
"""Run all 24 seed6/7 A/B 128-kimg checkpoints through frozen NFE1 evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (6, 7)
ARMS = ("A", "B")
BUDGETS_KIMG = (384, 512, 640, 768, 896, 1024)
TRAINING_COMMIT = "12b0036fee8ef09a72a6d40c9ba3e699cfd15759"
TRAINING_NUMERICAL_BASE = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EVALUATOR_HEAD = "9d06ccc72545d4189af1b86de7f629f9c09d3f73"
PROTOCOL = "q256-seed6-7-ab-128k-learning-curve-frozen-nfe1-v1"
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
BINDING_SCHEMA = "ect.q256.seed6-7-ab-128k-evaluation-binding/v1"
CELL_SCHEMA = "ect.q256.seed6-7-ab-128k-evaluation-cell/v1"
INVENTORY_SCHEMA = "ect.q256.seed6-7-ab-128k-checkpoint-inventory/v1"
SOURCE_SCHEMA = "ect.q256.seed6-7-ab-source-state-audit/v1"
JOB_RE = re.compile(r"^seed(6|7)-arm(A|B)-k(384|512|640|768|896|1024)-nfe1$")


class AdapterError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AdapterError(message)


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
        fail(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def regular_binding(path: Path, helper: Any) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing immutable artifact: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": helper.sha256_file(path),
    }


def validate_inventory(path: Path, seed: int) -> dict[str, Any]:
    inventory = load_json(path)
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("status") != "PASS"
        or inventory.get("training_commit") != TRAINING_COMMIT
        or inventory.get("seeds") != [seed]
        or inventory.get("arms") != list(ARMS)
        or inventory.get("budgets_kimg") != list(BUDGETS_KIMG)
        or inventory.get("checkpoint_count") != 12
        or inventory.get("extension_classification") != CLASSIFICATION
        or inventory.get("replaces_preregistered_seed") is not False
    ):
        fail(f"seed{seed} checkpoint inventory is not exact PASS")
    return inventory


def cell_from_inventory(
    row: Mapping[str, Any], source: Mapping[str, Any], helper: Any
) -> dict[str, Any]:
    seed, arm, budget = int(row["seed"]), str(row["arm"]), int(row["budget_kimg"])
    if seed not in SEEDS or arm not in ARMS or budget not in BUDGETS_KIMG:
        fail(f"out-of-contract checkpoint row: seed={seed} arm={arm} budget={budget}")
    checkpoint_dir = Path(str(row["checkpoint_dir"])).resolve(strict=True)
    state_path = Path(str(row["training_state"])).resolve(strict=True)
    snapshot_path = Path(str(row["snapshot"])).resolve(strict=True)
    metadata_path = Path(str(row["metadata"])).resolve(strict=True)
    if state_path.parent != checkpoint_dir or snapshot_path.parent != checkpoint_dir:
        fail(f"checkpoint paths escape their immutable directory: {checkpoint_dir}")
    metadata = load_json(metadata_path)
    exact = {
        "seed": seed,
        "arm": arm,
        "budget_kimg": budget,
        "training_commit": TRAINING_COMMIT,
        "training_state_sha256": row["training_state_sha256"],
        "snapshot_sha256": row["snapshot_sha256"],
        "source_checkpoint_sha256": source["source_state_sha256"],
    }
    for field, expected in exact.items():
        if metadata.get(field) != expected:
            fail(f"checkpoint metadata {field} mismatch: {metadata_path}")
    if helper.sha256_file(state_path) != row["training_state_sha256"]:
        fail(f"training state changed after inventory: {state_path}")
    if helper.sha256_file(snapshot_path) != row["snapshot_sha256"]:
        fail(f"EMA snapshot changed after inventory: {snapshot_path}")
    if helper.sha256_file(metadata_path) != row["metadata_sha256"]:
        fail(f"metadata changed after inventory: {metadata_path}")
    return {
        "schema": CELL_SCHEMA,
        "status": "PASS",
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "seed": seed,
        "arm": arm,
        "budget_kimg": budget,
        "run_dir": str(checkpoint_dir),
        "checkpoint": str(snapshot_path),
        "checkpoint_sha256": row["snapshot_sha256"],
        "checkpoint_bytes": snapshot_path.stat().st_size,
        "training_state": str(state_path),
        "training_state_sha256": row["training_state_sha256"],
        "training_state_bytes": state_path.stat().st_size,
        "checkpoint_metadata": str(metadata_path),
        "checkpoint_metadata_sha256": row["metadata_sha256"],
        "source_checkpoint_sha256": source["source_state_sha256"],
        "initial_common_state_sha256": source.get("initial_common_state_sha256"),
        "successful_optimizer_steps": row["successful_optimizer_steps"],
        "amp_skip_count": row["amp_skips"],
        "training_commit": TRAINING_COMMIT,
        "training_numerical_base_git_head": TRAINING_NUMERICAL_BASE,
        "verified_utc": helper.utc_now(),
    }


def create_binding(
    matrix_dir: Path,
    artifact_root: Path,
    evaluator: Any,
    helper: Any,
) -> None:
    if matrix_dir.exists():
        fail(f"refuse existing matrix binding directory: {matrix_dir}")
    if not matrix_dir.parent.is_dir():
        fail(f"matrix parent is missing: {matrix_dir.parent}")
    artifact_root = artifact_root.resolve(strict=True)
    source_audit_path = artifact_root / "integrity" / "source_state_audit.json"
    source_audit = load_json(source_audit_path)
    if (
        source_audit.get("schema") != SOURCE_SCHEMA
        or source_audit.get("status") != "PASS"
        or source_audit.get("cell_count") != 4
    ):
        fail("source-state audit is not PASS")
    source_by_cell = {
        (int(row["seed"]), str(row["arm"])): row
        for row in source_audit["cells"]
    }
    source = evaluator.source_snapshot(require_clean=True)
    if source.get("git_head") != EVALUATOR_HEAD:
        fail(f"wrong frozen evaluator head: {source.get('git_head')}")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRAINING_NUMERICAL_BASE, "HEAD"],
        cwd=evaluator.REPO_ROOT,
    )
    inventories = {}
    inventory_bindings = []
    cells = []
    for seed in SEEDS:
        path = artifact_root / "integrity" / f"seed{seed}_checkpoint_inventory.json"
        inventories[seed] = validate_inventory(path, seed)
        inventory_bindings.append(regular_binding(path, helper))
        for row in inventories[seed]["checkpoints"]:
            cells.append(
                cell_from_inventory(
                    row, source_by_cell[(int(row["seed"]), str(row["arm"]))], helper
                )
            )
    expected = {
        (seed, arm, budget)
        for seed in SEEDS
        for arm in ARMS
        for budget in BUDGETS_KIMG
    }
    if {(c["seed"], c["arm"], c["budget_kimg"]) for c in cells} != expected:
        fail("checkpoint matrix is incomplete")
    matrix_dir.mkdir(mode=0o750)
    (matrix_dir / "cells").mkdir(mode=0o750)
    cell_bindings = []
    for cell in sorted(cells, key=lambda c: (c["seed"], ARMS.index(c["arm"]), c["budget_kimg"])):
        path = matrix_dir / "cells" / (
            f"seed{cell['seed']}-arm{cell['arm']}-k{cell['budget_kimg']}.json"
        )
        helper.write_exclusive_json(path, cell)
        cell_bindings.append(regular_binding(path, helper))
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "PASS",
        "created_utc": helper.utc_now(),
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "selection_policy": "all_24_new_128k_budget_checkpoints_no_selection",
        "artifact_root": str(artifact_root),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "budgets_kimg": list(BUDGETS_KIMG),
        "cell_count": 24,
        "cell_bindings": cell_bindings,
        "cell_bindings_tree_sha256": helper.canonical_sha256(cell_bindings),
        "source_audit": regular_binding(source_audit_path, helper),
        "checkpoint_inventories": inventory_bindings,
        "training_commit": TRAINING_COMMIT,
        "training_numerical_base_git_head": TRAINING_NUMERICAL_BASE,
        "evaluator_source_git_head": source["git_head"],
        "evaluator_source_content_sha256": source["content_sha256"],
        "adapter": regular_binding(Path(__file__), helper),
        "formal_numerical_adapter": regular_binding(Path(helper.__file__), helper),
        "metric_numerical_semantics_changed": False,
    }
    helper.write_exclusive_json(matrix_dir / "matrix_binding.json", binding)


def load_bound_matrix(
    matrix_dir: Path, evaluator: Any, helper: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_dir = matrix_dir.resolve(strict=True)
    binding_path = matrix_dir / "matrix_binding.json"
    binding = load_json(binding_path)
    if (
        binding.get("schema") != BINDING_SCHEMA
        or binding.get("status") != "PASS"
        or binding.get("cell_count") != 24
        or binding.get("seeds") != list(SEEDS)
        or binding.get("arms") != list(ARMS)
        or binding.get("budgets_kimg") != list(BUDGETS_KIMG)
        or binding.get("training_commit") != TRAINING_COMMIT
        or binding.get("evaluator_source_git_head") != EVALUATOR_HEAD
    ):
        fail("matrix binding identity mismatch")
    for key in ("adapter", "formal_numerical_adapter", "source_audit"):
        receipt = binding.get(key)
        if not isinstance(receipt, dict):
            fail(f"matrix binding lacks {key}")
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if receipt.get("bytes") != path.stat().st_size or receipt.get("sha256") != helper.sha256_file(path):
            fail(f"bound artifact changed: {path}")
    receipts = binding.get("cell_bindings")
    if not isinstance(receipts, list) or len(receipts) != 24:
        fail("matrix binding must contain 24 cell receipts")
    if binding.get("cell_bindings_tree_sha256") != helper.canonical_sha256(receipts):
        fail("cell binding tree digest mismatch")
    cells = []
    for receipt in receipts:
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if path.parent != matrix_dir / "cells":
            fail(f"cell binding escapes matrix directory: {path}")
        if receipt.get("sha256") != helper.sha256_file(path) or receipt.get("bytes") != path.stat().st_size:
            fail(f"cell binding changed: {path}")
        cell = load_json(path)
        for file_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("training_state", "training_state_sha256"),
            ("checkpoint_metadata", "checkpoint_metadata_sha256"),
        ):
            artifact = Path(cell[file_key]).resolve(strict=True)
            if helper.sha256_file(artifact) != cell[hash_key]:
                fail(f"checkpoint matrix artifact changed: {artifact}")
        cells.append(
            {
                **cell,
                "training_validation_receipt": str(path),
                "training_validation_receipt_sha256": receipt["sha256"],
                "training_hash_receipt": str(path),
                "training_hash_receipt_sha256": receipt["sha256"],
            }
        )
    matrix = {
        "matrix_dir": str(matrix_dir),
        "matrix_binding": str(binding_path),
        "matrix_binding_sha256": helper.sha256_file(binding_path),
        "training_source_git_head": TRAINING_NUMERICAL_BASE,
        "training_actual_git_head": TRAINING_COMMIT,
        "training_source_content_sha256": binding["evaluator_source_content_sha256"],
        "cell_count": 24,
        "expected_amp_skip_attempts": None,
        "selection_policy": binding["selection_policy"],
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
    }
    return sorted(cells, key=lambda c: (c["seed"], ARMS.index(c["arm"]), c["budget_kimg"])), matrix


def build_jobs(
    evaluator: Any,
    cells: Sequence[Mapping[str, Any]],
    output_root: Path,
    base_port: int,
) -> list[dict[str, Any]]:
    jobs = []
    for cell in cells:
        job_id = (
            f"seed{cell['seed']}-arm{cell['arm']}-k{cell['budget_kimg']}-nfe1"
        )
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
            "--nfe=1",
            f"--metrics={','.join(evaluator.METRICS)}",
            "--metric-repeats=1",
            f"--sample-seeds={evaluator.SAMPLE_SEEDS}",
            f"--seed={evaluator.METRIC_SEED}",
            "--retain-generated-artifacts",
            f"--desc={PROTOCOL}-{job_id}",
        ]
        jobs.append(
            {
                "job_id": job_id,
                "seed": cell["seed"],
                "arm": cell["arm"],
                "budget_kimg": cell["budget_kimg"],
                "nfe": 1,
                "mid_t": [],
                "checkpoint": cell["checkpoint"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "training_state": cell["training_state"],
                "training_state_sha256": cell["training_state_sha256"],
                "training_run": cell["run_dir"],
                "training_validation_receipt": cell["training_validation_receipt"],
                "training_validation_receipt_sha256": cell["training_validation_receipt_sha256"],
                "training_hash_receipt": cell["training_hash_receipt"],
                "training_hash_receipt_sha256": cell["training_hash_receipt_sha256"],
                "sample_count": evaluator.SAMPLE_COUNT,
                "sample_seeds": evaluator.SAMPLE_SEEDS,
                "metric_seed": evaluator.METRIC_SEED,
                "metrics": list(evaluator.METRICS),
                "precision": "fp32",
                "output_directory": str(target),
                "command_argv_template": command,
            }
        )
    if len(jobs) != 24:
        fail(f"evaluation must contain exactly 24 jobs, got {len(jobs)}")
    return jobs


def install_gpu_audit_adapter(evaluator: Any, gpu_uuid: str) -> Any:
    import pynvml

    pynvml.nvmlInit()
    handles = {}
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
                metadata.update(ppid=int(fields[1]), process_group_id=int(fields[2]), session_id=int(fields[3]))
            except (OSError, ValueError, IndexError):
                pass
            try:
                metadata["executable"] = os.readlink(proc / "exe")
            except OSError:
                pass
            try:
                metadata["command_line"] = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")[:1024]
            except OSError:
                pass
            records.append(
                {
                    "pid": pid,
                    "process_name": name,
                    "used_gpu_memory_mib": str(int(process.usedGpuMemory) // (1024 * 1024)),
                    **metadata,
                }
            )
        return records

    def tree(root_pid: int, *, timeout_seconds: float = 5.0) -> set[int]:
        del timeout_seconds
        snapshot = evaluator.training_launcher.linux_process_snapshot()
        return {root_pid, *evaluator.training_launcher.snapshot_descendants(snapshot, root_pid)}

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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--formal-adapter", type=Path, required=True)
    parser.add_argument("--cache-source", type=Path, required=True)
    parser.add_argument("--reuse-bound-matrix", action="store_true")
    parser.add_argument("--bind-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-port", type=int, default=33880)
    parser.add_argument("--lock-root", type=Path, default=Path("/data/temp/ECT001-q256-evaluation-locks"))
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
            fail("bound checkpoint matrix is missing")
    else:
        create_binding(args.matrix_dir.resolve(), args.artifact_root, evaluator, helper)
    if args.bind_only:
        cells, matrix = load_bound_matrix(args.matrix_dir, evaluator, helper)
        print(json.dumps({"status": "PASS", "mode": "bind-only", "cell_count": len(cells), "matrix": matrix}, sort_keys=True))
        return 0

    original_plan = evaluator.build_plan
    original_write = evaluator.write_json_exclusive
    cache_source = args.cache_source.resolve(strict=True)
    if cache_source.is_symlink() or not cache_source.is_dir():
        fail("evaluator cache source is not a regular directory")
    evaluator.SEEDS = SEEDS
    evaluator.ARMS = ARMS
    evaluator.NFE_SETTINGS = {1: []}
    evaluator.PROTOCOL = PROTOCOL
    evaluator.load_training_matrix = lambda matrix: load_bound_matrix(matrix, evaluator, helper)
    evaluator.build_jobs = lambda cells, root, port: build_jobs(evaluator, cells, root, port)

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        plan = original_plan(**kwargs)
        plan.update(
            selection_policy="all_24_new_128k_budget_checkpoints_no_selection",
            independent_unit={"name": "training_seed", "values": list(SEEDS), "n": 2},
            extension_classification=CLASSIFICATION,
            replaces_preregistered_seed=False,
            checkpoint_cadence_kimg=128,
            budgets_kimg=list(BUDGETS_KIMG),
            training_actual_git_head=TRAINING_COMMIT,
        )
        return plan

    def write_json(path: Path, value: Mapping[str, Any]) -> None:
        payload = dict(value)
        if path.name == "evaluation_plan.json":
            cache_target = path.parent / "evaluator_cache"
            if cache_target.exists():
                fail(f"refuse existing evaluator cache target: {cache_target}")
            shutil.copytree(cache_source, cache_target, symlinks=False)
        job_id = payload.get("job_id")
        if isinstance(job_id, str):
            match = JOB_RE.fullmatch(job_id)
            if match:
                payload["budget_kimg"] = int(match.group(3))
                payload["extension_classification"] = CLASSIFICATION
                payload["replaces_preregistered_seed"] = False
        if path.name == "evaluation_completion.json":
            payload.update(
                selection_policy="all_24_new_128k_budget_checkpoints_no_selection",
                independent_training_seed_n=2,
                extension_classification=CLASSIFICATION,
                replaces_preregistered_seed=False,
                checkpoint_cadence_kimg=128,
                budgets_kimg=list(BUDGETS_KIMG),
            )
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
            fail("learning-curve revalidation received a non-frozen contract")
        metadata = load_json(run_dir / "metadata.json")
        if metadata.get("seed") != seed or metadata.get("arm") != arm or metadata.get("training_commit") != TRAINING_COMMIT:
            fail(f"checkpoint revalidation identity mismatch: {run_dir}")
        snapshot = run_dir / "network-snapshot.pkl"
        state = run_dir / "training-state.pt"
        if helper.sha256_file(snapshot) != metadata.get("snapshot_sha256") or helper.sha256_file(state) != metadata.get("training_state_sha256"):
            fail(f"checkpoint revalidation hash mismatch: {run_dir}")
        return {
            "status": "PASS",
            "kind": "secondary-extension-128k-read-only-revalidation",
            "budget_kimg": metadata["budget_kimg"],
            "checkpoint_sha256": metadata["snapshot_sha256"],
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
        evaluator_repair_base_git_head=TRAINING_NUMERICAL_BASE,
    )
    try:
        return evaluator.execute(execute_args)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as exc:
        raise SystemExit(f"[q256-seed6-7-ab-128k-evaluation] ERROR: {exc}") from exc
