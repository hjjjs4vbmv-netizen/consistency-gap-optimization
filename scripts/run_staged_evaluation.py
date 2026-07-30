#!/usr/bin/env python3
"""Run the frozen staged checkpoint-evaluation matrix on a GPU server."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "staged-checkpoint-evaluation-v1"
METRIC_SEED = 20260730
NFE_SETTINGS = {1: [], 2: [0.821]}
PHASES = {
    "smoke": {
        "evidence_class": "quick",
        "sample_count": 5_000,
        "sample_seeds": "0-4999",
        "metrics": ("kid5k_full", "fid5k_full"),
    },
    "quick": {
        "evidence_class": "quick",
        "sample_count": 5_000,
        "sample_seeds": "0-4999",
        "metrics": ("kid5k_full", "fid5k_full"),
    },
    "formal": {
        "evidence_class": "formal",
        "sample_count": 50_000,
        "sample_seeds": "0-49999",
        "metrics": ("kid50k_full", "fid50k_full"),
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"[run_staged_evaluation] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return payload


def load_cells(manifest_path: Path, allow_missing_inputs: bool) -> tuple[list[dict], dict | None]:
    manifest = load_json(manifest_path, "checkpoint manifest")
    if manifest.get("protocol") not in (None, PROTOCOL_ID):
        fail(f"manifest protocol must be {PROTOCOL_ID!r}")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        fail("manifest must contain a non-empty cells list")

    cells = []
    checkpoint_ids = set()
    for raw in raw_cells:
        try:
            checkpoint_id = str(raw["checkpoint_id"])
            method = str(raw["method"])
            training_seed = int(raw["training_seed"])
            budget_kimg = int(raw["budget_kimg"])
            checkpoint = Path(raw["checkpoint"]).expanduser().resolve()
            expected_sha256 = str(raw["checkpoint_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid checkpoint cell {raw!r}: {exc}")
        if not checkpoint_id or checkpoint_id in checkpoint_ids:
            fail(f"checkpoint_id must be unique and non-empty: {checkpoint_id!r}")
        if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256.lower()):
            fail(f"checkpoint_sha256 must be a 64-character hexadecimal digest: {checkpoint_id}")
        checkpoint_ids.add(checkpoint_id)
        if checkpoint.is_file():
            actual_sha256 = sha256_file(checkpoint)
            if actual_sha256 != expected_sha256:
                fail(
                    f"checkpoint SHA256 mismatch for {checkpoint}: "
                    f"{actual_sha256} != {expected_sha256}"
                )
        elif not allow_missing_inputs:
            fail(f"checkpoint not found: {checkpoint}")
        cells.append({
            "checkpoint_id": checkpoint_id,
            "method": method,
            "training_seed": training_seed,
            "budget_kimg": budget_kimg,
            "checkpoint": checkpoint,
            "checkpoint_sha256": expected_sha256,
            "integrity_receipt": raw.get("integrity_receipt"),
        })

    comparison = manifest.get("comparison")
    if comparison is not None and not isinstance(comparison, dict):
        fail("comparison must be a JSON object when provided")
    return cells, comparison


def verify_integrity_receipt(cell: dict, allow_missing_inputs: bool) -> dict:
    receipt_value = cell.get("integrity_receipt")
    if not receipt_value:
        fail(f"formal evaluation requires integrity_receipt: {cell['checkpoint_id']}")
    receipt_path = Path(receipt_value).expanduser().resolve()
    if not receipt_path.is_file():
        if allow_missing_inputs:
            return {"path": str(receipt_path), "status": "not_checked_dry_run"}
        fail(f"training-integrity receipt not found: {receipt_path}")
    receipt = load_json(receipt_path, "training-integrity receipt")
    required_fields = (
        "schema_version", "status", "checkpoint_id", "checkpoint_path",
        "checkpoint_sha256", "training_run_id", "method", "training_seed",
        "budget_kimg", "completion_passed", "logs_state_consistent",
        "finite_loss_state_passed", "checker_version", "checker_git_commit", "checked_at_unix",
        "checkpoint_load_passed", "ema_present", "ema_finite_passed",
        "schedule_identity_passed", "global_gap_scale_identity_passed",
        "method_identity_passed",
    )
    missing = [field for field in required_fields if field not in receipt]
    if missing:
        fail(f"training-integrity receipt is incomplete ({missing}): {receipt_path}")
    if receipt.get("status") != "passed":
        fail(f"training-integrity receipt did not pass: {receipt_path}")
    if receipt.get("checkpoint_sha256") != cell["checkpoint_sha256"]:
        fail(f"training-integrity receipt SHA256 mismatch: {receipt_path}")
    if receipt.get("checkpoint_id") != cell["checkpoint_id"]:
        fail(f"training-integrity receipt checkpoint_id mismatch: {receipt_path}")
    if Path(str(receipt["checkpoint_path"])).name != cell["checkpoint"].name:
        fail(f"training-integrity receipt checkpoint path mismatch: {receipt_path}")
    if receipt.get("method") != cell["method"]:
        fail(f"training-integrity receipt method mismatch: {receipt_path}")
    if receipt.get("training_seed") != cell["training_seed"]:
        fail(f"training-integrity receipt training seed mismatch: {receipt_path}")
    if receipt.get("budget_kimg") != cell["budget_kimg"]:
        fail(f"training-integrity receipt budget mismatch: {receipt_path}")
    for field in (
        "completion_passed", "logs_state_consistent", "finite_loss_state_passed",
        "checkpoint_load_passed", "ema_present", "ema_finite_passed",
        "schedule_identity_passed", "global_gap_scale_identity_passed",
        "method_identity_passed",
    ):
        if receipt.get(field) is not True:
            fail(f"training-integrity receipt did not pass {field}: {receipt_path}")
    if (
        not str(receipt.get("training_run_id", ""))
        or not str(receipt.get("checker_version", ""))
        or not str(receipt.get("checker_git_commit", ""))
    ):
        fail(f"training-integrity receipt has an empty provenance field: {receipt_path}")
    if not isinstance(receipt.get("checked_at_unix"), (int, float)):
        fail(f"training-integrity receipt timestamp is invalid: {receipt_path}")
    return {"path": str(receipt_path), "status": "passed"}


def select_cells(cells: list[dict], phase: str, smoke_checkpoint_id: str | None) -> list[dict]:
    if phase == "smoke":
        if not smoke_checkpoint_id:
            fail("--smoke-checkpoint-id is required for --phase smoke")
        selected = [cell for cell in cells if cell["checkpoint_id"] == smoke_checkpoint_id]
        if len(selected) != 1:
            fail(f"smoke checkpoint_id not found: {smoke_checkpoint_id}")
        return selected
    if smoke_checkpoint_id:
        fail("--smoke-checkpoint-id is only valid for --phase smoke")
    return cells


def validate_formal_promotion_policy(manifest: dict, cells: list[dict]) -> None:
    """Reject formal manifests which make quick performance a selection gate."""
    policy = manifest.get("formal_promotion_policy")
    if not isinstance(policy, dict):
        fail("formal evaluation requires a formal_promotion_policy object")
    if policy.get("eligibility") != "provenance_and_integrity_only":
        fail("formal eligibility must be provenance_and_integrity_only")
    if policy.get("quick_metric_performance") != "not_an_eligibility_criterion":
        fail("quick metric performance must not be a formal eligibility criterion")
    required_ids = policy.get("required_checkpoint_ids")
    if (
        not isinstance(required_ids, list)
        or not required_ids
        or not all(isinstance(checkpoint_id, str) and checkpoint_id for checkpoint_id in required_ids)
        or len(set(required_ids)) != len(required_ids)
    ):
        fail("formal_promotion_policy required_checkpoint_ids must be a unique non-empty string list")
    actual_ids = {cell["checkpoint_id"] for cell in cells}
    required_ids_set = set(required_ids)
    if actual_ids != required_ids_set:
        fail(
            "formal manifest must contain every predeclared checkpoint; "
            f"missing={sorted(required_ids_set - actual_ids)}, "
            f"extra={sorted(actual_ids - required_ids_set)}"
        )


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        fail(f"refuse to append to non-empty output directory: {path}")


def build_jobs(
    cells: list[dict], data: Path, outdir: Path, phase: str, base_port: int,
    allow_missing_inputs: bool,
) -> list[dict]:
    config = PHASES[phase]
    jobs = []
    for cell in cells:
        receipt = (
            verify_integrity_receipt(cell, allow_missing_inputs)
            if phase == "formal"
            else {"path": None, "status": "not_required"}
        )
        for nfe in NFE_SETTINGS:
            target = outdir / cell["checkpoint_id"] / f"nfe{nfe}"
            command = [
                "bash", str(REPO_ROOT / "scripts" / "evaluate_checkpoint.sh"),
                "1", str(base_port + len(jobs)), str(cell["checkpoint"]),
                "--outdir", str(target),
                "--nosubdir",
                "--data", str(data),
                "--cond=False",
                "--arch=ddpmpp",
                "--precond=ct",
                "--dropout=0.2",
                "--augment=0",
                "--fp16=False",
                "--cache=True",
                "--workers=3",
                f"--nfe={nfe}",
                "--mid_t=0.821",
                f"--metrics={','.join(config['metrics'])}",
                "--metric-repeats=1",
                f"--sample-seeds={config['sample_seeds']}",
                f"--seed={METRIC_SEED}",
                f"--desc={PROTOCOL_ID}-{phase}-{cell['checkpoint_id']}-nfe{nfe}",
            ]
            jobs.append({
                "evidence_class": config["evidence_class"],
                "checkpoint_id": cell["checkpoint_id"],
                "method": cell["method"],
                "training_seed": cell["training_seed"],
                "budget_kimg": cell["budget_kimg"],
                "checkpoint": str(cell["checkpoint"]),
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "integrity_receipt": receipt,
                "nfe": nfe,
                "mid_t": NFE_SETTINGS[nfe],
                "sample_count": config["sample_count"],
                "sample_seeds": config["sample_seeds"],
                "metric_seed": METRIC_SEED,
                "metric_names": list(config["metrics"]),
                "output_directory": str(target),
                "command": command,
            })
    return jobs


def build_record(
    cells: list[dict], comparison: dict | None, data: Path, outdir: Path,
    phase: str, jobs: list[dict], dataset_sha256: str,
) -> dict:
    config = PHASES[phase]
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "phase": phase,
        "evidence_class": config["evidence_class"],
        "evaluation_git_commit": git_head(),
        "dataset": str(data),
        "dataset_sha256": dataset_sha256,
        "precision": "fp32",
        "nfe_modes": {str(nfe): mid_t for nfe, mid_t in NFE_SETTINGS.items()},
        "sample_count": config["sample_count"],
        "sample_seeds": config["sample_seeds"],
        "metric_seed": METRIC_SEED,
        "metric_names": list(config["metrics"]),
        "metric_repeats": 1,
        "comparison": comparison,
        "output_root": str(outdir),
        "status": "dry_run",
        "jobs": jobs,
    }


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--smoke-checkpoint-id")
    parser.add_argument("--base-port", type=int, default=29800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-inputs", action="store_true")
    args = parser.parse_args(argv)

    if args.allow_missing_inputs and not args.dry_run:
        fail("--allow-missing-inputs is only allowed with --dry-run")
    data = args.data.expanduser().resolve()
    if not data.is_file() and not (args.dry_run and args.allow_missing_inputs):
        fail(f"dataset not found: {data}")
    outdir = args.outdir.expanduser().resolve()
    if not args.dry_run:
        require_empty(outdir)

    cells, comparison = load_cells(args.manifest, args.allow_missing_inputs)
    if args.phase == "formal":
        validate_formal_promotion_policy(
            load_json(args.manifest, "checkpoint manifest"), cells
        )
    selected = select_cells(cells, args.phase, args.smoke_checkpoint_id)
    jobs = build_jobs(selected, data, outdir, args.phase, args.base_port, args.allow_missing_inputs)
    record = build_record(
        selected, comparison, data, outdir, args.phase, jobs,
        sha256_file(data) if data.is_file() else "missing",
    )
    if args.dry_run:
        print(json.dumps({key: value for key, value in record.items() if key != "jobs"}, indent=2))
        for job in jobs:
            print(shlex.join(job["command"]))
        return

    record_path = outdir / "run_manifest.json"
    record["status"] = "running"
    write_record(record_path, record)
    started = time.time()
    for index, job in enumerate(jobs, start=1):
        target = Path(job["output_directory"])
        require_empty(target)
        print(f"[{index}/{len(jobs)}] {args.phase} {job['checkpoint_id']} nfe={job['nfe']}")
        print(shlex.join(job["command"]))
        job["started_at_unix"] = time.time()
        try:
            subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            job["status"] = "failed"
            job["returncode"] = exc.returncode
            record["status"] = "failed"
            record["elapsed_seconds"] = round(time.time() - started, 3)
            write_record(record_path, record)
            raise SystemExit(exc.returncode) from exc
        job["status"] = "completed"
        job["elapsed_seconds"] = round(time.time() - job["started_at_unix"], 3)
        write_record(record_path, record)

    record["status"] = "completed"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    write_record(record_path, record)
    print(f"Completed {len(jobs)} jobs; record: {record_path}")


if __name__ == "__main__":
    main()
