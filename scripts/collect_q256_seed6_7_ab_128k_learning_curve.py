#!/usr/bin/env python3
"""Collect the verified 28-row seed6/7 A/B NFE1 learning curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping


SEEDS = (6, 7)
ARMS = ("A", "B")
BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
NEW_BUDGETS = BUDGETS[1:]
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
SCHEMA = "ect.q256.seed6-7-ab-128k-learning-curve/v1"
CSV_FIELDS = (
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
    "evaluation_receipt",
    "source",
)


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


def metric_values(receipt: Mapping[str, Any]) -> tuple[float, float]:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, list):
        fail("evaluation receipt lacks metrics")
    values = {str(row.get("metric")): float(row.get("value")) for row in metrics}
    if set(values) != {"fid50k_full", "kid50k_full"} or not all(
        math.isfinite(value) for value in values.values()
    ):
        fail("evaluation receipt metrics are incomplete or non-finite")
    return values["fid50k_full"], values["kid50k_full"]


def feature_sha256(receipt: Mapping[str, Any]) -> str:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("evaluation receipt lacks artifacts")
    kid = artifacts.get("generated-features-kid50k_full-repeat00.npy")
    fid = artifacts.get("generated-features-fid50k_full-repeat00.npy")
    if not isinstance(kid, dict) or not isinstance(fid, dict) or kid.get("sha256") != fid.get("sha256"):
        fail("FID/KID generated-feature identity failed")
    return str(kid["sha256"])


def validate_receipt(receipt: Mapping[str, Any], *, seed: int, arm: str, budget: int) -> None:
    expected = {
        "status": "passed",
        "returncode": 0,
        "seed": seed,
        "arm": arm,
        "nfe": 1,
        "mid_t": [],
        "sample_count": 50000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20260730,
        "precision": "fp32",
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            fail(f"evaluation receipt {field} mismatch: {receipt.get(field)!r} != {value!r}")
    if budget != 256 and receipt.get("budget_kimg") != budget:
        fail("new evaluation receipt budget mismatch")
    if receipt.get("gpu_exclusivity_monitor", {}).get("status") != "PASS":
        fail("evaluation GPU exclusivity monitor did not pass")


def sign_label(value: float) -> str:
    if value < 0:
        return "B<A"
    if value > 0:
        return "B>A"
    return "B=A"


def build_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell = {(row["seed"], row["arm"], row["budget_kimg"]): row for row in rows}
    contrasts = []
    seed_summaries = []
    thresholds = []
    for seed in SEEDS:
        fid_deltas = []
        kid_deltas = []
        for budget in BUDGETS:
            a = by_cell[(seed, "A", budget)]
            b = by_cell[(seed, "B", budget)]
            fid_delta = b["fid50k_full"] - a["fid50k_full"]
            kid_delta = b["kid50k_full"] - a["kid50k_full"]
            fid_deltas.append(fid_delta)
            kid_deltas.append(kid_delta)
            contrasts.append(
                {
                    "seed": seed,
                    "budget_kimg": budget,
                    "fid_A": a["fid50k_full"],
                    "fid_B": b["fid50k_full"],
                    "B_minus_A_fid": fid_delta,
                    "kid_A": a["kid50k_full"],
                    "kid_B": b["kid50k_full"],
                    "B_minus_A_kid": kid_delta,
                    "fid_sign": sign_label(fid_delta),
                    "kid_sign": sign_label(kid_delta),
                }
            )
        fid_reversals = [
            BUDGETS[index]
            for index in range(1, len(BUDGETS))
            if fid_deltas[index - 1] * fid_deltas[index] < 0
        ]
        kid_reversals = [
            BUDGETS[index]
            for index in range(1, len(BUDGETS))
            if kid_deltas[index - 1] * kid_deltas[index] < 0
        ]
        seed_summaries.append(
            {
                "seed": seed,
                "fid_sign_sequence": [sign_label(value) for value in fid_deltas],
                "kid_sign_sequence": [sign_label(value) for value in kid_deltas],
                "fid_sign_reversal_budgets": fid_reversals,
                "kid_sign_reversal_budgets": kid_reversals,
                "fid_absolute_gap_change_256_to_1024": abs(fid_deltas[-1]) - abs(fid_deltas[0]),
                "kid_absolute_gap_change_256_to_1024": abs(kid_deltas[-1]) - abs(kid_deltas[0]),
                "fid_gap_magnitude_label": (
                    "expanded" if abs(fid_deltas[-1]) > abs(fid_deltas[0])
                    else "decayed" if abs(fid_deltas[-1]) < abs(fid_deltas[0])
                    else "unchanged"
                ),
                "kid_gap_magnitude_label": (
                    "expanded" if abs(kid_deltas[-1]) > abs(kid_deltas[0])
                    else "decayed" if abs(kid_deltas[-1]) < abs(kid_deltas[0])
                    else "unchanged"
                ),
            }
        )
        observed_fid = [
            by_cell[(seed, arm, budget)]["fid50k_full"]
            for arm in ARMS
            for budget in BUDGETS
        ]
        lower = math.ceil(min(observed_fid) / 25) * 25
        upper = math.floor(max(observed_fid) / 25) * 25
        for threshold in range(lower, upper + 1, 25):
            tau = {}
            for arm in ARMS:
                hits = [
                    budget
                    for budget in BUDGETS
                    if by_cell[(seed, arm, budget)]["fid50k_full"] <= threshold
                ]
                tau[arm] = min(hits) if hits else None
            thresholds.append(
                {
                    "seed": seed,
                    "fid_threshold": threshold,
                    "tau_A_kimg": tau["A"],
                    "tau_B_kimg": tau["B"],
                    "tau_B_minus_tau_A_kimg": (
                        tau["B"] - tau["A"]
                        if tau["A"] is not None and tau["B"] is not None
                        else None
                    ),
                    "within_observed_curve_range": True,
                }
            )
    return contrasts, seed_summaries, thresholds


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# q256 seed6/7 A/B 128-kimg NFE1 learning curve",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {payload['created_utc'][:10]}",
        "- Verification Status: VERIFIED",
        "- Version Label: q256_seed6_7_ab_128k_learning_curve_v1",
        "",
        "## Scope",
        "",
        "Seeds 6 and 7 remain a secondary precision extension outside the original seeds3/4/5 preregistration. The 256-kimg points are bound from PR #71 and were not re-evaluated. All 24 new checkpoints were evaluated without checkpoint selection.",
        "",
        "Frozen protocol: NFE=1, FP32, 50,000 samples, sample seeds `0..49999`, metric seed `20260730`, and byte-identical retained generated features for KID/FID within each job.",
        "",
        "## Complete learning curve",
        "",
        "| Seed | Arm | Budget kimg | FID50k | KID50k | Source |",
        "|---:|:---:|---:|---:|---:|:---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['budget_kimg']} | "
            f"{row['fid50k_full']:.12f} | {row['kid50k_full']:.15f} | {row['source']} |"
        )
    lines += [
        "",
        "## B−A contrasts",
        "",
        "Negative values favor B because lower FID/KID is better.",
        "",
        "| Seed | Budget | FID A | FID B | B−A FID | KID A | KID B | B−A KID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["contrasts"]:
        lines.append(
            f"| {row['seed']} | {row['budget_kimg']} | {row['fid_A']:.12f} | "
            f"{row['fid_B']:.12f} | {row['B_minus_A_fid']:.12f} | "
            f"{row['kid_A']:.15f} | {row['kid_B']:.15f} | {row['B_minus_A_kid']:.15f} |"
        )
    lines += ["", "## Descriptive summary", ""]
    for summary in payload["seed_summaries"]:
        lines.append(
            f"- Seed {summary['seed']}: FID signs {summary['fid_sign_sequence']}; "
            f"FID reversal budgets {summary['fid_sign_reversal_budgets'] or 'none'}; "
            f"FID gap magnitude {summary['fid_gap_magnitude_label']}. "
            f"KID signs {summary['kid_sign_sequence']}; KID reversal budgets "
            f"{summary['kid_sign_reversal_budgets'] or 'none'}; KID gap magnitude "
            f"{summary['kid_gap_magnitude_label']}."
        )
    lines += [
        "",
        "Threshold hitting times use a fixed 25-FID grid restricted to each seed's observed curve range; no extrapolation is used. Full threshold rows are in the JSON.",
        "",
        "## Runtime and integrity",
        "",
        f"- Training GPU-hours: {payload['runtime']['training_gpu_hours']:.6f}",
        f"- Evaluation GPU-hours: {payload['runtime']['evaluation_gpu_hours']:.6f}",
        "- Immutable checkpoints: 24/24 PASS.",
        "- New frozen evaluation jobs: 24/24 PASS.",
        "- Final learning-curve rows: 28/28 PASS.",
        "- These are descriptive quality trajectories; they do not alter the original preregistration or establish a causal mechanism.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--baseline-eval-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    baseline = args.baseline_eval_root.resolve(strict=True)
    compact = root / "evaluation" / "run-primary-nfe1-v1"
    completion = load_json(compact / "evaluation_completion.json")
    compaction = load_json(compact / "compaction_receipt.json")
    baseline_completion = load_json(baseline / "evaluation_completion.json")
    if completion.get("status") != "PASS" or completion.get("job_count") != 24:
        fail("new evaluation completion is not 24/24 PASS")
    if compaction.get("status") != "PASS" or compaction.get("job_count") != 24:
        fail("new evaluation compaction is not PASS")
    if baseline_completion.get("status") != "PASS":
        fail("PR #71 baseline completion is not PASS")
    source_audit = load_json(root / "integrity" / "source_state_audit.json")
    source_by_cell = {
        (int(row["seed"]), str(row["arm"])): row for row in source_audit["cells"]
    }
    inventory_rows = []
    training_gpu_hours = 0.0
    for seed in SEEDS:
        inventory = load_json(root / "integrity" / f"seed{seed}_checkpoint_inventory.json")
        if inventory.get("status") != "PASS" or inventory.get("checkpoint_count") != 12:
            fail(f"seed{seed} checkpoint inventory is not PASS")
        inventory_rows.extend(inventory["checkpoints"])
        training_gpu_hours += float(inventory["training_gpu_hours"])
    inventory_by_cell = {
        (int(row["seed"]), str(row["arm"]), int(row["budget_kimg"])): row
        for row in inventory_rows
    }

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            baseline_receipt_path = baseline / "receipts" / f"seed{seed}-arm{arm}-nfe1.json"
            receipt = load_json(baseline_receipt_path)
            validate_receipt(receipt, seed=seed, arm=arm, budget=256)
            fid, kid = metric_values(receipt)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "budget_kimg": 256,
                    "nfe": 1,
                    "fid50k_full": fid,
                    "kid50k_full": kid,
                    "training_state_sha256": source_by_cell[(seed, arm)]["source_state_sha256"],
                    "snapshot_sha256": receipt["checkpoint_sha256"],
                    "generated_feature_sha256": feature_sha256(receipt),
                    "status": "PASS",
                    "evaluation_receipt": str(baseline_receipt_path),
                    "evaluation_receipt_sha256": sha256_file(baseline_receipt_path),
                    "source": "PR71_existing_256k_formal_result",
                }
            )
            for budget in NEW_BUDGETS:
                job_id = f"seed{seed}-arm{arm}-k{budget}-nfe1"
                receipt_path = compact / "receipts" / f"{job_id}.json"
                receipt = load_json(receipt_path)
                validate_receipt(receipt, seed=seed, arm=arm, budget=budget)
                inventory = inventory_by_cell[(seed, arm, budget)]
                if receipt.get("checkpoint_sha256") != inventory["snapshot_sha256"]:
                    fail(f"evaluation/checkpoint inventory mismatch: {job_id}")
                fid, kid = metric_values(receipt)
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "budget_kimg": budget,
                        "nfe": 1,
                        "fid50k_full": fid,
                        "kid50k_full": kid,
                        "training_state_sha256": inventory["training_state_sha256"],
                        "snapshot_sha256": inventory["snapshot_sha256"],
                        "generated_feature_sha256": feature_sha256(receipt),
                        "status": "PASS",
                        "evaluation_receipt": str(receipt_path),
                        "evaluation_receipt_sha256": sha256_file(receipt_path),
                        "source": "new_128k_budget_formal_evaluation",
                    }
                )
    rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"]), row["budget_kimg"]))
    expected = {(seed, arm, budget) for seed in SEEDS for arm in ARMS for budget in BUDGETS}
    if {(r["seed"], r["arm"], r["budget_kimg"]) for r in rows} != expected or len(rows) != 28:
        fail("final learning curve is not the exact 28-row matrix")
    contrasts, seed_summaries, thresholds = build_summary(rows)
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "artifact_root": str(root),
        "git": {
            "branch": "codex/q256-seed6-7-ab-128k-learning-curve",
            "training_commit": "12b0036fee8ef09a72a6d40c9ba3e699cfd15759",
            "evaluation_tool_commit": compaction["tool_commit"],
        },
        "protocol": {
            "nfe": 1,
            "precision": "fp32",
            "sample_count": 50000,
            "sample_seed_range": "0-49999",
            "metric_seed": 20260730,
            "metrics": ["kid50k_full", "fid50k_full"],
            "checkpoint_cadence_kimg": 128,
            "budgets_kimg": list(BUDGETS),
        },
        "runtime": {
            "training_gpu_hours": training_gpu_hours,
            "evaluation_gpu_hours": float(compaction["evaluation_gpu_hours"]),
        },
        "counts": {
            "immutable_checkpoints_pass": 24,
            "new_evaluation_jobs_pass": 24,
            "learning_curve_rows_pass": 28,
        },
        "rows": rows,
        "contrasts": contrasts,
        "seed_summaries": seed_summaries,
        "threshold_policy": "25-FID grid within each seed observed min/max; no extrapolation",
        "threshold_hitting_times": thresholds,
        "provenance": {
            "source_state_audit": str(root / "integrity" / "source_state_audit.json"),
            "source_state_audit_sha256": sha256_file(root / "integrity" / "source_state_audit.json"),
            "new_evaluation_completion_sha256": sha256_file(compact / "evaluation_completion.json"),
            "new_evaluation_compaction_sha256": sha256_file(compact / "compaction_receipt.json"),
            "baseline_evaluation_completion_sha256": sha256_file(baseline / "evaluation_completion.json"),
        },
        "interpretation_boundary": "descriptive_secondary_precision_extension_only",
    }
    output_json = root / "reports" / "learning_curve_seed6_7_ab_nfe1_128k.json"
    output_csv = root / "reports" / "learning_curve_seed6_7_ab_nfe1_128k.csv"
    output_md = root / "reports" / "learning_curve_seed6_7_ab_nfe1_128k.md"
    write_exclusive(
        output_json,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_exclusive(output_csv, csv_buffer.getvalue().encode("utf-8"))
    write_exclusive(output_md, render_markdown(payload).encode("utf-8"))
    print(json.dumps({"status": "PASS", "rows": 28, "contrasts": 14, "json": str(output_json), "csv": str(output_csv), "markdown": str(output_md)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CollectError, KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"[q256-seed6-7-ab-128k-collector] ERROR: {exc}") from exc
