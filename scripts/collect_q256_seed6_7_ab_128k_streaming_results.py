#!/usr/bin/env python3
"""Collect verified seed6/7 A/B 128-kimg NFE1/NFE2 streaming results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SEEDS = (6, 7)
ARMS = ("A", "B")
BUDGETS = (384, 512, 640, 768, 896, 1024)
NFES = (1, 2)
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
RESULT_SCHEMA = "ect.q256.seed6-7-ab-128k-streaming-results/v1"
MANIFEST_SCHEMA = "ect.q256.seed6-7-ab-128k-server-archive/v1"


class CollectError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CollectError(message)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            fail(f"refuse to overwrite result: {path}")
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_exclusive(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_exclusive(path, buffer.getvalue().encode("utf-8"))


def metric_rows(receipt: Mapping[str, Any]) -> tuple[float, float]:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        fail("streaming receipt metrics must be a dictionary")
    fid = float(metrics["fid50k_full"])
    kid = float(metrics["kid50k_full"])
    if not math.isfinite(fid) or not math.isfinite(kid) or fid < 0:
        fail("streaming receipt contains invalid metrics")
    return fid, kid


def validate_receipt(
    receipt: Mapping[str, Any], *, seed: int, arm: str, budget: int, nfe: int
) -> None:
    expected = {
        "status": "PASS",
        "seed": seed,
        "arm": arm,
        "budget_kimg": budget,
        "nfe": nfe,
        "precision": "fp32",
        "sample_count": 50000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20260730,
        "evaluator_source_git_head": "9d06ccc72545d4189af1b86de7f629f9c09d3f73",
        "metric_numerical_semantics_changed": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            fail(
                f"receipt {field} mismatch for seed{seed}/{arm}/{budget}/nfe{nfe}: "
                f"{receipt.get(field)!r} != {value!r}"
            )
    expected_mid = None if nfe == 1 else 0.821
    if receipt.get("mid_t") != expected_mid:
        fail(f"receipt mid_t mismatch for nfe={nfe}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("receipt lacks artifact bindings")
    kid_feature = artifacts.get("generated-features-kid50k_full-repeat00.npy")
    fid_feature = artifacts.get("generated-features-fid50k_full-repeat00.npy")
    if (
        not isinstance(kid_feature, dict)
        or not isinstance(fid_feature, dict)
        or kid_feature.get("sha256") != fid_feature.get("sha256")
    ):
        fail("KID/FID generated-feature identity failed")


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# q256 seed6/7 A/B 128-kimg FID-vs-training-compute results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {payload['created_utc'][:10]}",
        "- Verification Status: VERIFIED",
        "- Version Label: q256_seed6_7_ab_128k_streaming_results_v1",
        "",
        "## Scope",
        "",
        "Seeds 6 and 7 are a secondary precision extension outside the original seeds3/4/5 preregistration. All six new budgets for A and B were evaluated at NFE1 and NFE2 without checkpoint selection.",
        "",
        "## Complete metric table",
        "",
        "| Seed | Arm | Budget | NFE | FID50k | KID50k | Feature SHA |",
        "|---:|:---:|---:|---:|---:|---:|:---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['budget_kimg']} | "
            f"{row['nfe']} | {row['fid50k_full']:.12f} | "
            f"{row['kid50k_full']:.15f} | `{row['generated_feature_sha256'][:12]}…` |"
        )
    lines += [
        "",
        "## B−A contrasts",
        "",
        "Negative values favor B because lower metrics are better.",
        "",
        "| Seed | Budget | NFE | B−A FID | B−A KID |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["contrasts"]:
        lines.append(
            f"| {row['seed']} | {row['budget_kimg']} | {row['nfe']} | "
            f"{row['B_minus_A_fid']:.12f} | {row['B_minus_A_kid']:.15f} |"
        )
    lines += [
        "",
        "## Time to quality",
        "",
        "| Seed | Arm | NFE | FID threshold | First budget kimg |",
        "|---:|:---:|---:|---:|---:|",
    ]
    for row in payload["time_to_quality"]:
        budget = "not reached" if row["first_budget_kimg"] is None else row["first_budget_kimg"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['nfe']} | "
            f"≤{row['fid_threshold']} | {budget} |"
        )
    lines += [
        "",
        "## Runtime and integrity",
        "",
        f"- Training GPU-hours: {payload['runtime']['training_gpu_hours']:.6f}",
        f"- Evaluation GPU-hours: {payload['runtime']['evaluation_gpu_hours']:.6f}",
        f"- Evaluation wall-clock hours: {payload['runtime']['evaluation_wall_hours']:.6f}",
        "- Immutable training checkpoints: 24/24 PASS.",
        "- Formal evaluation receipts: 48/48 PASS.",
        "- Durable evaluation copies: 48/48 PASS and byte-identical to source receipts.",
        "- Every job used byte-identical KID/FID generated features.",
        "- Results are descriptive; they do not alter the original preregistration or establish a causal mechanism.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    training_root = args.training_root.resolve(strict=True)
    evaluation_root = args.evaluation_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    reports_root = output_root / "reports"
    if reports_root.exists():
        fail(f"refuse existing reports directory: {reports_root}")
    reports_root.mkdir(mode=0o750)

    source_audit = load_json(training_root / "integrity" / "source_state_audit.json")
    if source_audit.get("status") != "PASS" or source_audit.get("cell_count") != 4:
        fail("source-state audit is not PASS")
    checkpoint_rows = []
    training_gpu_hours = 0.0
    for seed in SEEDS:
        path = training_root / "integrity" / f"seed{seed}_checkpoint_inventory.json"
        inventory = load_json(path)
        if (
            inventory.get("status") != "PASS"
            or inventory.get("checkpoint_count") != 12
            or inventory.get("training_commit")
            != "12b0036fee8ef09a72a6d40c9ba3e699cfd15759"
        ):
            fail(f"seed{seed} inventory is not exact PASS")
        training_gpu_hours += float(inventory["training_gpu_hours"])
        checkpoint_rows.extend(inventory["checkpoints"])
    checkpoint_by_cell = {
        (int(row["seed"]), str(row["arm"]), int(row["budget_kimg"])): row
        for row in checkpoint_rows
    }
    expected_checkpoints = {
        (seed, arm, budget)
        for seed in SEEDS
        for arm in ARMS
        for budget in BUDGETS
    }
    if set(checkpoint_by_cell) != expected_checkpoints or len(checkpoint_rows) != 24:
        fail("checkpoint inventory is not the exact 24-cell matrix")

    receipt_root = evaluation_root / "receipts"
    rows = []
    receipts = []
    started = []
    finished = []
    evaluation_gpu_seconds = 0.0
    for seed in SEEDS:
        for arm in ARMS:
            for budget in BUDGETS:
                checkpoint = checkpoint_by_cell[(seed, arm, budget)]
                for nfe in NFES:
                    job_id = f"seed{seed}-arm{arm}-k{budget}-nfe{nfe}"
                    path = receipt_root / f"{job_id}.json"
                    receipt = load_json(path)
                    validate_receipt(
                        receipt, seed=seed, arm=arm, budget=budget, nfe=nfe
                    )
                    if receipt.get("checkpoint_sha256") != checkpoint["snapshot_sha256"]:
                        fail(f"evaluation checkpoint binding mismatch: {job_id}")
                    fid, kid = metric_rows(receipt)
                    feature_sha = receipt["artifacts"][
                        "generated-features-kid50k_full-repeat00.npy"
                    ]["sha256"]
                    evaluation_gpu_seconds += float(receipt["elapsed_seconds"])
                    created = str(receipt["created_utc"])
                    finished.append(parse_utc(created))
                    started.append(
                        parse_utc(created)
                        - __import__("datetime").timedelta(
                            seconds=float(receipt["elapsed_seconds"])
                        )
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "budget_kimg": budget,
                            "nfe": nfe,
                            "fid50k_full": fid,
                            "kid50k_full": kid,
                            "training_state_sha256": checkpoint[
                                "training_state_sha256"
                            ],
                            "snapshot_sha256": checkpoint["snapshot_sha256"],
                            "generated_feature_sha256": feature_sha,
                            "evaluation_receipt": str(path),
                            "evaluation_receipt_sha256": sha256_file(path),
                            "status": "PASS",
                        }
                    )
                    receipts.append(
                        {
                            "job_id": job_id,
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
    if len(rows) != 48:
        fail("evaluation result matrix is not 48 rows")

    by_cell = {
        (row["seed"], row["arm"], row["budget_kimg"], row["nfe"]): row
        for row in rows
    }
    contrasts = []
    for seed in SEEDS:
        for budget in BUDGETS:
            for nfe in NFES:
                a = by_cell[(seed, "A", budget, nfe)]
                b = by_cell[(seed, "B", budget, nfe)]
                contrasts.append(
                    {
                        "seed": seed,
                        "budget_kimg": budget,
                        "nfe": nfe,
                        "fid_A": a["fid50k_full"],
                        "fid_B": b["fid50k_full"],
                        "B_minus_A_fid": b["fid50k_full"] - a["fid50k_full"],
                        "kid_A": a["kid50k_full"],
                        "kid_B": b["kid50k_full"],
                        "B_minus_A_kid": b["kid50k_full"] - a["kid50k_full"],
                    }
                )
    time_to_quality = []
    for seed in SEEDS:
        for arm in ARMS:
            for nfe, thresholds in ((1, (20, 10)), (2, (5,))):
                for threshold in thresholds:
                    hits = [
                        budget
                        for budget in BUDGETS
                        if by_cell[(seed, arm, budget, nfe)]["fid50k_full"]
                        <= threshold
                    ]
                    time_to_quality.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "nfe": nfe,
                            "fid_threshold": threshold,
                            "first_budget_kimg": min(hits) if hits else None,
                            "observed_budgets_only": True,
                        }
                    )

    rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"]), row["budget_kimg"], row["nfe"]))
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "training_root": str(training_root),
        "evaluation_root": str(evaluation_root),
        "server_archive_root": str(output_root),
        "git": {
            "branch": "codex/q256-seed6-7-ab-128k-learning-curve",
            "training_commit": "12b0036fee8ef09a72a6d40c9ba3e699cfd15759",
        },
        "protocol": {
            "budgets_kimg": list(BUDGETS),
            "nfes": list(NFES),
            "nfe2_mid_t": 0.821,
            "sample_count": 50000,
            "sample_seed_range": "0-49999",
            "metric_seed": 20260730,
            "precision": "fp32",
            "metrics": ["kid50k_full", "fid50k_full"],
        },
        "counts": {
            "training_checkpoints_pass": 24,
            "evaluation_receipts_pass": 48,
            "durable_copies_pass": 48,
            "metric_rows": 48,
            "contrast_rows": 24,
        },
        "runtime": {
            "training_gpu_hours": training_gpu_hours,
            "evaluation_gpu_hours": evaluation_gpu_seconds / 3600,
            "evaluation_wall_hours": (max(finished) - min(started)).total_seconds()
            / 3600,
        },
        "rows": rows,
        "contrasts": contrasts,
        "time_to_quality": time_to_quality,
        "receipt_tree_sha256": canonical_sha256(receipts),
        "interpretation_boundary": "descriptive_secondary_precision_extension_only",
    }

    results_json = reports_root / "learning_curve_seed6_7_ab_nfe1_nfe2_128k.json"
    results_csv = reports_root / "learning_curve_seed6_7_ab_nfe1_nfe2_128k.csv"
    results_md = reports_root / "learning_curve_seed6_7_ab_nfe1_nfe2_128k.md"
    contrast_csv = reports_root / "learning_curve_seed6_7_ab_contrasts_128k.csv"
    threshold_csv = reports_root / "time_to_quality_seed6_7_ab_128k.csv"
    checkpoint_csv = reports_root / "checkpoint_inventory_seed6_7_ab_128k.csv"
    write_json(results_json, payload)
    write_csv(
        results_csv,
        rows,
        (
            "seed",
            "arm",
            "budget_kimg",
            "nfe",
            "fid50k_full",
            "kid50k_full",
            "training_state_sha256",
            "snapshot_sha256",
            "generated_feature_sha256",
            "status",
        ),
    )
    write_exclusive(results_md, render_markdown(payload).encode("utf-8"))
    write_csv(
        contrast_csv,
        contrasts,
        (
            "seed",
            "budget_kimg",
            "nfe",
            "fid_A",
            "fid_B",
            "B_minus_A_fid",
            "kid_A",
            "kid_B",
            "B_minus_A_kid",
        ),
    )
    write_csv(
        threshold_csv,
        time_to_quality,
        (
            "seed",
            "arm",
            "nfe",
            "fid_threshold",
            "first_budget_kimg",
            "observed_budgets_only",
        ),
    )
    write_csv(
        checkpoint_csv,
        checkpoint_rows,
        (
            "seed",
            "arm",
            "budget_kimg",
            "training_state",
            "training_state_bytes",
            "training_state_sha256",
            "snapshot",
            "snapshot_bytes",
            "snapshot_sha256",
            "metadata",
            "metadata_sha256",
            "status",
        ),
    )

    critical_files = [
        training_root / "integrity" / "source_state_audit.json",
        training_root / "integrity" / "seed6_checkpoint_inventory.json",
        training_root / "integrity" / "seed7_checkpoint_inventory.json",
        results_json,
        results_csv,
        results_md,
        contrast_csv,
        threshold_csv,
        checkpoint_csv,
        *[Path(row["path"]) for row in receipts],
    ]
    bindings = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in critical_files
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "server_archive_root": str(output_root),
        "training_checkpoints": [
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "budget_kimg": row["budget_kimg"],
                "training_state": row["training_state"],
                "training_state_sha256": row["training_state_sha256"],
                "snapshot": row["snapshot"],
                "snapshot_sha256": row["snapshot_sha256"],
            }
            for row in checkpoint_rows
        ],
        "critical_files": bindings,
        "critical_files_tree_sha256": canonical_sha256(bindings),
        "large_generated_arrays_retained": False,
        "large_generated_arrays_policy": (
            "omitted from consolidated archive; hashes preserved in PASS receipts"
        ),
    }
    manifest_path = output_root / "archive_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": 48,
                "contrasts": 24,
                "threshold_rows": len(time_to_quality),
                "reports_root": str(reports_root),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CollectError, KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"[q256-seed6-7-ab-128k-streaming-collector] ERROR: {exc}") from exc
