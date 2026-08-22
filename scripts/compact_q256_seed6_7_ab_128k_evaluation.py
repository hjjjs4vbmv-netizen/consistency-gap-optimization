#!/usr/bin/env python3
"""Compact PASS evaluation evidence and remove only verified ephemeral arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "q256-seed6-7-ab-128k-learning-curve-frozen-nfe1-v1"
COMPACTION_SCHEMA = "ect.q256.seed6-7-ab-128k-evaluation-compaction/v1"
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
JOB_COUNT = 24
LARGE_ARTIFACTS = {
    "generated-samples.npy",
    "generated-features-kid50k_full-repeat00.npy",
    "generated-features-fid50k_full-repeat00.npy",
}


class CompactionError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CompactionError(message)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def copy_verified(source: Path, target: Path, expected: Mapping[str, Any]) -> None:
    if source.is_symlink() or not source.is_file():
        fail(f"source artifact is missing or a symlink: {source}")
    if source.stat().st_size != expected.get("bytes"):
        fail(f"source artifact byte count changed: {source}")
    observed = sha256_file(source)
    if observed != expected.get("sha256"):
        fail(f"source artifact SHA256 changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    if target.stat().st_size != expected["bytes"] or sha256_file(target) != observed:
        fail(f"durable copy verification failed: {target}")


def hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail(f"compact evidence contains a symlink: {path}")
        if path.is_file():
            records[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ephemeral-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--delete-ephemeral-on-pass", action="store_true")
    args = parser.parse_args()
    ephemeral = args.ephemeral_root.resolve(strict=True)
    durable = args.durable_root.resolve()
    if ephemeral.is_symlink() or not ephemeral.is_dir():
        fail(f"invalid ephemeral evaluation root: {ephemeral}")
    if durable.exists():
        fail(f"refuse existing durable evidence root: {durable}")
    if not durable.parent.is_dir():
        fail(f"durable evidence parent is missing: {durable.parent}")
    completion = load_json(ephemeral / "evaluation_completion.json")
    if (
        completion.get("protocol") != PROTOCOL
        or completion.get("status") != "PASS"
        or completion.get("job_count") != JOB_COUNT
        or len(completion.get("completed_job_ids", [])) != JOB_COUNT
        or completion.get("extension_classification") != CLASSIFICATION
    ):
        fail("ephemeral evaluation completion is not exact PASS")

    temporary = durable.parent / f".{durable.name}.tmp-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir(mode=0o750)
    excluded = []
    evaluation_gpu_seconds = 0.0
    job_ids = list(completion["completed_job_ids"])
    if len(set(job_ids)) != JOB_COUNT:
        fail("completion contains duplicate job IDs")
    for relative in ("evaluation_plan.json", "evaluation_completion.json"):
        source = ephemeral / relative
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
        if sha256_file(source) != sha256_file(target):
            fail(f"top-level compact copy mismatch: {relative}")

    for job_id in job_ids:
        receipt_path = ephemeral / "receipts" / f"{job_id}.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("status") != "passed"
            or receipt.get("returncode") != 0
            or receipt.get("protocol") != PROTOCOL
            or receipt.get("nfe") != 1
            or receipt.get("mid_t") != []
            or receipt.get("sample_count") != 50000
            or receipt.get("sample_seed_range") != "0-49999"
            or receipt.get("metric_seed") != 20260730
            or receipt.get("precision") != "fp32"
            or receipt.get("gpu_exclusivity_monitor", {}).get("status") != "PASS"
            or receipt.get("budget_kimg") not in (384, 512, 640, 768, 896, 1024)
        ):
            fail(f"job receipt is not exact PASS: {receipt_path}")
        metrics = receipt.get("metrics")
        if not isinstance(metrics, list) or {m.get("metric") for m in metrics} != {
            "kid50k_full",
            "fid50k_full",
        }:
            fail(f"job receipt lacks exact metrics: {receipt_path}")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            fail(f"job receipt lacks artifact tree: {receipt_path}")
        feature_bindings = [
            artifacts.get("generated-features-kid50k_full-repeat00.npy"),
            artifacts.get("generated-features-fid50k_full-repeat00.npy"),
        ]
        if not all(isinstance(item, dict) for item in feature_bindings) or feature_bindings[0].get("sha256") != feature_bindings[1].get("sha256"):
            fail(f"job generated-feature identity failed: {receipt_path}")
        evaluation_gpu_seconds += float(receipt["elapsed_seconds"])

        for category in ("manifests", "receipts", "process_logs"):
            suffix = ".json" if category != "process_logs" else ".log"
            source = ephemeral / category / f"{job_id}{suffix}"
            target = temporary / category / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
            if sha256_file(source) != sha256_file(target):
                fail(f"compact {category} copy mismatch: {job_id}")

        for relative, binding in sorted(artifacts.items()):
            source = ephemeral / "jobs" / job_id / relative
            if relative in LARGE_ARTIFACTS:
                if source.is_symlink() or not source.is_file() or source.stat().st_size != binding.get("bytes") or sha256_file(source) != binding.get("sha256"):
                    fail(f"ephemeral large artifact changed: {source}")
                excluded.append(
                    {
                        "job_id": job_id,
                        "relative_path": relative,
                        "bytes": binding["bytes"],
                        "sha256": binding["sha256"],
                        "durably_retained": False,
                        "reason": "ephemeral generated array; formal receipt and SHA256 retained",
                    }
                )
            else:
                copy_verified(source, temporary / "jobs" / job_id / relative, binding)

    tree = hash_tree(temporary)
    receipt = {
        "schema": COMPACTION_SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "protocol": PROTOCOL,
        "tool_commit": args.tool_commit,
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "ephemeral_root": str(ephemeral),
        "durable_root": str(durable),
        "job_count": JOB_COUNT,
        "metric_value_count": JOB_COUNT * 2,
        "evaluation_gpu_hours": evaluation_gpu_seconds / 3600,
        "excluded_ephemeral_artifact_count": len(excluded),
        "excluded_ephemeral_bytes": sum(row["bytes"] for row in excluded),
        "excluded_ephemeral_artifacts": excluded,
        "compact_artifact_count": len(tree),
        "compact_tree_sha256": canonical_sha256(tree),
        "ephemeral_deleted_after_pass": args.delete_ephemeral_on_pass,
    }
    write_json_exclusive(temporary / "compaction_receipt.json", receipt)
    os.rename(temporary, durable)
    parent_fd = os.open(durable.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    final_receipt = load_json(durable / "compaction_receipt.json")
    if final_receipt.get("status") != "PASS":
        fail("durable compaction receipt did not survive publication")

    if args.delete_ephemeral_on_pass:
        if ephemeral.parent != Path("/dev/shm") or ephemeral.name != "run-primary":
            fail(f"refuse cleanup outside exact private mount target: {ephemeral}")
        shutil.rmtree(ephemeral)
        if ephemeral.exists():
            fail("ephemeral evaluation root remains after cleanup")
    print(json.dumps(final_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompactionError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"[q256-seed6-7-ab-128k-compaction] ERROR: {exc}") from exc
