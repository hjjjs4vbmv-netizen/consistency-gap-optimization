#!/usr/bin/env python3
"""Audit q256 target/denominator secondary-extension training seeds.

This adapter intentionally does not alter the frozen training source.  It
validates direct-worker artifacts using the production verifier's option,
initial-state, training-state, and snapshot checks, plus the telemetry policy
actually used by the durable dcca41b formal-run record.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable


SOURCE_REPO = Path(
    os.environ.get(
        "Q256_EXTENSION_SOURCE_REPO",
        "/data/temp/ECT001/q256-factorial-clean-25c3d22",
    )
).resolve()
if str(SOURCE_REPO) not in sys.path:
    sys.path.insert(0, str(SOURCE_REPO))

from scripts import verify_q256_target_weight_arm as production  # noqa: E402
from training import reproducibility  # noqa: E402


SCHEMA = "ect.q256.target-weight-secondary-extension-integrity/v1"
REPORT_SCHEMA = "ect.q256.target-weight-secondary-extension-report/v1"
SOURCE_HEAD = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
DATASET = Path(
    "/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip"
)
TRANSFER = Path(
    "/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
)
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
ARMS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}
REQUIRED_FILES = (
    "training_options.json",
    "initial_state_receipt_v1.json",
    "factorial_training_telemetry_v1.csv",
    "train_summary.csv",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "log.txt",
    "final.png",
)
SEMANTIC_NONFINITE_FIELDS = (
    "loss_nonfinite_count",
    "sanitized_grad_nonfinite_count",
    "update_nonfinite_count",
    "model_nonfinite_count",
    "ema_nonfinite_count",
    "factor_nonfinite_count",
)
COUNT_FIELDS = SEMANTIC_NONFINITE_FIELDS + (
    "raw_grad_nonfinite_count",
    "sample_count",
    "base_r_zero_count",
    "target_r_zero_count",
    "target_r_equal_t_count",
    "target_scaled_to_zero_count",
    "denominator_r_zero_count",
    "denominator_r_equal_t_count",
    "denominator_scaled_to_zero_count",
    "nonpositive_denominator_count",
)
FINITE_FIELDS = (
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
DIGEST_FIELDS = (
    "batch_sha256",
    "t_sha256",
    "base_r_sha256",
    "target_r_sha256",
    "denominator_r_sha256",
    "target_delta_sha256",
    "denominator_delta_sha256",
)
ERROR_PATTERNS = (
    "traceback (most recent call last)",
    "cuda out of memory",
    "outofmemoryerror",
    "bus error",
    "cuda error",
    "nonpositive denominator",
)


class AuditError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AuditError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected one JSON object: {path}")
    return value


def strict_int(value: Any, label: str) -> int:
    try:
        text = str(value)
        if not text.isdigit() or (len(text) > 1 and text.startswith("0")):
            raise ValueError
        return int(text)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not a canonical non-negative integer: {value!r}") from exc


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        fail(f"{label} is not finite: {value!r}")
    return result


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def verify_asset(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"{label} is missing, empty, or a symlink: {path}")
    observed = sha256_file(path)
    require_equal(observed, expected_sha256, f"{label} SHA256")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def validate_direct_telemetry(
    rows: list[dict[str, str]], arm: str
) -> dict[str, Any]:
    if len(rows) != 2000:
        fail(f"seed arm {arm} has {len(rows)} attempts instead of 2000")
    target_scale, denominator_scale = ARMS[arm]
    skips: list[int] = []
    cumulative_skips = 0
    previous_elapsed = -1.0
    counter_totals = {field: 0 for field in COUNT_FIELDS}
    losses: list[float] = []
    raw_grad_skip_mismatch_count = 0

    for attempt, row in enumerate(rows, start=1):
        label = f"arm {arm} telemetry attempt {attempt}"
        require_equal(row["schema"], production.TELEMETRY_SCHEMA, f"{label}.schema")
        require_equal(row["protocol"], production.PROTOCOL, f"{label}.protocol")
        require_equal(row["arm"], arm, f"{label}.arm")
        require_equal(float(row["target_gap_scale"]), target_scale, f"{label}.target")
        require_equal(
            float(row["denominator_gap_scale"]),
            denominator_scale,
            f"{label}.denominator",
        )
        require_equal(strict_int(row["attempted_iteration"], label), attempt, label)
        processed_nimg = strict_int(row["processed_nimg"], f"{label}.processed_nimg")
        require_equal(processed_nimg, attempt * 128, f"{label}.processed_nimg")
        require_equal(
            finite_float(row["processed_kimg"], f"{label}.processed_kimg"),
            processed_nimg / 1000,
            f"{label}.processed_kimg",
        )
        require_equal(strict_int(row["stage"], f"{label}.stage"), 0, f"{label}.stage")
        for field in DIGEST_FIELDS:
            value = row[field]
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                fail(f"{label}.{field} is not a lowercase SHA256 digest")
        if target_scale == denominator_scale:
            require_equal(
                row["target_r_sha256"],
                row["denominator_r_sha256"],
                f"{label} native r identity",
            )
            require_equal(
                row["target_delta_sha256"],
                row["denominator_delta_sha256"],
                f"{label} native delta identity",
            )

        counts = {field: strict_int(row[field], f"{label}.{field}") for field in COUNT_FIELDS}
        for field, value in counts.items():
            counter_totals[field] += value
        require_equal(counts["sample_count"], 128, f"{label}.sample_count")
        for field in SEMANTIC_NONFINITE_FIELDS + (
            "target_r_equal_t_count",
            "denominator_r_equal_t_count",
            "nonpositive_denominator_count",
        ):
            require_equal(counts[field], 0, f"{label}.{field}")

        values = {field: finite_float(row[field], f"{label}.{field}") for field in FINITE_FIELDS}
        loss = values["loss"]
        if loss < 0:
            fail(f"{label}.loss is negative")
        losses.append(loss)
        for prefix in ("target_delta", "denominator_delta"):
            minimum = values[f"{prefix}_min"]
            mean = values[f"{prefix}_mean"]
            maximum = values[f"{prefix}_max"]
            if not (0 < minimum <= mean <= maximum):
                fail(f"{label}.{prefix} violates 0 < min <= mean <= max")
        require_equal(values["learning_rate"], 1e-4, f"{label}.learning_rate")
        if values["grad_scale_before"] <= 0 or values["grad_scale_after"] <= 0:
            fail(f"{label} has a non-positive GradScaler scale")
        if values["elapsed_sec"] < previous_elapsed:
            fail(f"{label}.elapsed_sec regressed")
        previous_elapsed = values["elapsed_sec"]
        if not math.isclose(
            values["gpu_hours_cumulative"],
            values["elapsed_sec"] / 3600,
            rel_tol=0,
            abs_tol=1e-8,
        ):
            fail(f"{label}.gpu_hours_cumulative is inconsistent")

        skipped = strict_int(row["step_skipped"], f"{label}.step_skipped")
        if skipped not in (0, 1):
            fail(f"{label}.step_skipped is not 0/1")
        if skipped:
            skips.append(attempt)
            cumulative_skips += 1
        successes = strict_int(
            row["successful_optimizer_steps"],
            f"{label}.successful_optimizer_steps",
        )
        require_equal(successes, attempt - cumulative_skips, f"{label}.successful steps")
        raw_nonfinite = counts["raw_grad_nonfinite_count"] > 0
        if raw_nonfinite != bool(skipped):
            raw_grad_skip_mismatch_count += 1
        try:
            raw_norm = float(row["raw_grad_norm"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AuditError(f"{label}.raw_grad_norm is invalid") from exc
        if skipped:
            if raw_norm != float("inf"):
                fail(f"{label}.raw_grad_norm is not +inf on an AMP skip")
            if values["grad_scale_after"] >= values["grad_scale_before"]:
                fail(f"{label} AMP skip did not reduce scale")
            require_equal(values["update_norm"], 0.0, f"{label}.update_norm")
        else:
            if not math.isfinite(raw_norm) or raw_norm < 0:
                fail(f"{label}.raw_grad_norm is invalid on an accepted update")
            if values["grad_scale_after"] < values["grad_scale_before"]:
                fail(f"{label} scale fell without a recorded AMP skip")
            if values["update_norm"] <= 0:
                fail(f"{label} accepted update has non-positive update norm")
    require_equal(raw_grad_skip_mismatch_count, 0, f"arm {arm} raw-gradient/skip mismatches")
    last200 = losses[-200:]
    return {
        "attempts": 2000,
        "accepted_updates": 2000 - len(skips),
        "successful_optimizer_steps": 2000 - len(skips),
        "processed_nimg": 256000,
        "processed_kimg": 256.0,
        "amp_skip_count": len(skips),
        "amp_skip_attempts": skips,
        "final_loss": losses[-1],
        "last200_loss_mean": statistics.fmean(last200),
        "last200_loss_sd": statistics.stdev(last200),
        "counter_totals": counter_totals,
        "semantic_nonfinite_count": sum(counter_totals[field] for field in SEMANTIC_NONFINITE_FIELDS),
        "raw_grad_nonfinite_rows": len(skips),
        "raw_grad_nonfinite_elements": counter_totals["raw_grad_nonfinite_count"],
        "raw_grad_skip_mismatch_count": raw_grad_skip_mismatch_count,
        "nonpositive_denominator_count": counter_totals["nonpositive_denominator_count"],
        "elapsed_sec": previous_elapsed,
        "rows": rows,
    }


def normalized_options(options: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(options)
    value.pop("run_dir", None)
    loss = value.get("loss_kwargs", {})
    loss.pop("target_gap_scale", None)
    loss.pop("denominator_gap_scale", None)
    return value


def compare_fields(
    rows_by_arm: dict[str, list[dict[str, str]]],
    left: str,
    right: str,
    fields: Iterable[str],
    label: str,
) -> bool:
    for attempt, (left_row, right_row) in enumerate(
        zip(rows_by_arm[left], rows_by_arm[right], strict=True), start=1
    ):
        for field in fields:
            if left_row[field] != right_row[field]:
                fail(
                    f"{label} failed at attempt {attempt}, field {field}: "
                    f"{left_row[field]!r} != {right_row[field]!r}"
                )
    return True


def write_exclusive(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# seed{report['seed']} extension integrity audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {report['finished_utc'][:10]}",
        f"- Verification Status: {'VERIFIED' if report['status'] == 'PASS' else 'ANALYZED'}",
        f"- Version Label: q256_seed{report['seed']}_extension_integrity_v1",
        "",
        "This seed is a secondary precision extension. It is not part of the original",
        "seeds3/4/5 preregistration and does not replace any preregistered seed.",
        "",
        f"Overall status: **{report['status']}**.",
        "",
    ]
    if report["status"] != "PASS":
        lines += ["## Failure", "", f"- {report.get('error', 'unknown error')}", ""]
        return "\n".join(lines) + "\n"
    lines += [
        "## Cell summary",
        "",
        "| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        cell = report["cells"][arm]
        lines.append(
            f"| {arm} | {cell['attempts']} | {cell['accepted_updates']} | "
            f"{cell['amp_skip_count']} | {cell['final_loss']:.8f} | "
            f"{cell['last200_loss_mean']:.8f} ± {cell['last200_loss_sd']:.8f} | "
            f"{cell['semantic_nonfinite_count']} | {cell['nonpositive_denominator_count']} |"
        )
    lines += [
        "",
        "## Integrity conclusions",
        "",
        f"- Four-arm completion: `{report['four_arm_complete']}`.",
        f"- Denominator integrity: `{report['denominator_integrity']}`.",
        f"- Telemetry identity checks: `{report['telemetry_identity_checks']['all_pass']}`.",
        f"- Common initial-state identity: `{report['common_initial_state_identity']}`.",
        "- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;",
        "  they are reported separately from semantic non-finite counters.",
        "",
    ]
    return "\n".join(lines) + "\n"


def audit_seed(root: Path, seed: int, *, check_only: bool) -> dict[str, Any]:
    if seed not in (6, 7):
        fail("extension audit seed must be 6 or 7")
    root = root.resolve()
    seed_dir = root / f"seed{seed}"
    output_json = root / "integrity" / f"seed{seed}_integrity_audit.json"
    output_md = root / "integrity" / f"seed{seed}_integrity_audit.md"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "FAIL",
        "seed": seed,
        "extension_classification": "secondary_precision_extension_not_original_preregistration",
        "replaces_preregistered_seed": False,
        "source_head": SOURCE_HEAD,
        "source_repo": str(SOURCE_REPO),
        "root": str(root),
        "started_utc": utc_now(),
    }
    try:
        observed_head = os.popen(f"git -C {SOURCE_REPO} rev-parse HEAD").read().strip()
        require_equal(observed_head, SOURCE_HEAD, "training source HEAD")
        status = os.popen(f"git -C {SOURCE_REPO} status --porcelain").read().strip()
        require_equal(status, "", "training source cleanliness")
        report["assets"] = {
            "dataset": verify_asset(DATASET, DATASET_SHA256, "dataset"),
            "transfer": verify_asset(TRANSFER, TRANSFER_SHA256, "transfer checkpoint"),
        }
        cells: dict[str, Any] = {}
        rows_by_arm: dict[str, list[dict[str, str]]] = {}
        options_by_arm: dict[str, dict[str, Any]] = {}
        common_initial_hashes: dict[str, str] = {}
        initial_component_hashes: dict[str, Any] = {}

        for arm in ARMS:
            run_dir = seed_dir / f"arm{arm}"
            if not run_dir.is_dir() or run_dir.is_symlink():
                fail(f"missing regular run directory: {run_dir}")
            paths = {name: run_dir / name for name in REQUIRED_FILES}
            for name, path in paths.items():
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    fail(f"seed{seed}/{arm} missing non-empty regular artifact {name}")
            if list(run_dir.glob("network-snapshot-[0-9]*.pkl")):
                fail(f"seed{seed}/{arm} contains a forbidden numbered snapshot")
            if list(run_dir.glob("training-state-[0-9]*.pt")):
                fail(f"seed{seed}/{arm} contains a forbidden numbered state dump")

            options = load_json(paths["training_options.json"])
            options_info = production.validate_training_options(options, arm, seed, "formal")
            require_equal(options.get("run_dir"), str(run_dir), f"seed{seed}/{arm} run_dir")
            require_equal(options.get("cudnn_benchmark"), False, f"seed{seed}/{arm} cudnn benchmark")
            require_equal(options.get("stop_after_attempts"), None, f"seed{seed}/{arm} stop control")
            require_equal(options_info["dataset_path"], str(DATASET), f"seed{seed}/{arm} dataset")
            require_equal(options_info["transfer_path"], str(TRANSFER), f"seed{seed}/{arm} transfer")
            options_by_arm[arm] = normalized_options(options)

            initial = load_json(paths["initial_state_receipt_v1.json"])
            initial_info = production.validate_initial_receipt(
                initial, arm, seed, options_info
            )
            common_initial_hashes[arm] = initial_info["common_initial_state_sha256"]
            initial_component_hashes[arm] = initial["hashes"]

            rows = production.read_telemetry(paths["factorial_training_telemetry_v1.csv"])
            telemetry = validate_direct_telemetry(rows, arm)
            rows_by_arm[arm] = rows
            state_info = production.validate_training_state(
                paths["training-state-latest.pt"], arm, seed, "formal", telemetry
            )
            require_equal(
                state_info["trajectory_config_sha256"],
                initial_info["trajectory_config_sha256"],
                f"seed{seed}/{arm} initial/final trajectory hash",
            )
            snapshot_info = production.validate_snapshot(
                paths["network-snapshot-latest.pkl"], arm, options_info, state_info
            )
            log_text = paths["log.txt"].read_text(encoding="utf-8", errors="replace").lower()
            matches = [pattern for pattern in ERROR_PATTERNS if pattern in log_text]
            if matches:
                fail(f"seed{seed}/{arm} log contains fatal pattern(s): {matches}")
            artifact_hashes = {
                name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for name, path in paths.items()
            }
            telemetry.pop("rows")
            cells[arm] = {
                **telemetry,
                "run_dir": str(run_dir),
                "target_gap_scale": ARMS[arm][0],
                "denominator_gap_scale": ARMS[arm][1],
                "initial_common_state_sha256": initial_info["common_initial_state_sha256"],
                "trajectory_config_sha256": initial_info["trajectory_config_sha256"],
                "snapshot_ema_sha256": snapshot_info["ema_sha256"],
                "finite_tensor_checks": {
                    "state_net": state_info["net_tensors_checked"],
                    "state_ema": state_info["ema_tensors_checked"],
                    "state_optimizer": state_info["optimizer_tensors_checked"],
                    "state_gradscaler": state_info["gradscaler_tensors_checked"],
                    "snapshot_ema": snapshot_info["ema_tensors_checked"],
                },
                "artifact_hashes": artifact_hashes,
            }
            del state_info

        first_options = options_by_arm["A"]
        for arm in ("B", "C", "D"):
            require_equal(options_by_arm[arm], first_options, f"seed{seed} common options A/{arm}")
        first_common = common_initial_hashes["A"]
        first_components = initial_component_hashes["A"]
        for arm in ("B", "C", "D"):
            require_equal(common_initial_hashes[arm], first_common, f"seed{seed} common initial A/{arm}")
            require_equal(initial_component_hashes[arm], first_components, f"seed{seed} initial components A/{arm}")

        common_fields = (
            "batch_sha256",
            "t_sha256",
            "base_r_sha256",
            "base_r_zero_count",
            "sample_count",
        )
        for arm in ("B", "C", "D"):
            compare_fields(rows_by_arm, "A", arm, common_fields, f"seed{seed} common draw A/{arm}")
        target_fields = (
            "target_r_sha256",
            "target_delta_sha256",
            "target_r_zero_count",
            "target_r_equal_t_count",
            "target_scaled_to_zero_count",
            "target_delta_min",
            "target_delta_max",
            "target_delta_mean",
        )
        denominator_fields = (
            "denominator_r_sha256",
            "denominator_delta_sha256",
            "denominator_r_zero_count",
            "denominator_r_equal_t_count",
            "denominator_scaled_to_zero_count",
            "denominator_delta_min",
            "denominator_delta_max",
            "denominator_delta_mean",
        )
        identities = {
            "A_vs_D_same_target": compare_fields(rows_by_arm, "A", "D", target_fields, "A/D target"),
            "B_vs_C_same_target": compare_fields(rows_by_arm, "B", "C", target_fields, "B/C target"),
            "A_vs_C_same_denominator": compare_fields(rows_by_arm, "A", "C", denominator_fields, "A/C denominator"),
            "B_vs_D_same_denominator": compare_fields(rows_by_arm, "B", "D", denominator_fields, "B/D denominator"),
        }
        identities["all_pass"] = all(identities.values())
        report.update(
            {
                "status": "PASS",
                "cells": cells,
                "four_arm_complete": True,
                "common_initial_state_identity": True,
                "denominator_integrity": all(
                    cell["nonpositive_denominator_count"] == 0 for cell in cells.values()
                ),
                "telemetry_identity_checks": identities,
            }
        )
    except BaseException as exc:
        report.update(
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "four_arm_complete": False,
            }
        )
    report["finished_utc"] = utc_now()
    if not check_only:
        write_exclusive(output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
        write_exclusive(output_md, audit_markdown(report))
    print(json.dumps(report, sort_keys=True))
    return report


def extension_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# q256 factorial seed6/7 extension training report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {report['generated_utc'][:10]}",
        "- Verification Status: VERIFIED",
        "- Version Label: q256_factorial_seed6_7_extension_report_v1",
        "",
        "## Scope",
        "",
        "Seeds 6 and 7 are a secondary precision extension. They are independent",
        "training seeds outside the original seeds3/4/5 preregistration, do not",
        "replace seeds3/4/5, and must not be described as preregistered replications.",
        "Training used `--metrics=none`; all loss contrasts below are training-objective",
        "diagnostics and do not support a generation-quality conclusion.",
        "",
        "## Training endpoints",
        "",
        "| Seed | Arm | Attempts | Accepted updates | AMP skips | Final-row loss | Last-200 mean ± SD | Semantic non-finite | Raw-grad/skip mismatch | Nonpositive denominator |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (6, 7):
        for arm in ARMS:
            cell = report["seeds"][str(seed)]["cells"][arm]
            lines.append(
                f"| {seed} | {arm} | {cell['attempts']} | {cell['accepted_updates']} | "
                f"{cell['amp_skip_count']} ({','.join(map(str, cell['amp_skip_attempts']))}) | "
                f"{cell['final_loss']:.8f} | {cell['last200_loss_mean']:.8f} ± "
                f"{cell['last200_loss_sd']:.8f} | {cell['semantic_nonfinite_count']} | "
                f"{cell['raw_grad_skip_mismatch_count']} | {cell['nonpositive_denominator_count']} |"
            )
    lines += [
        "",
        "## Integrity",
        "",
        "- Both seeds completed all four A/B/C/D arms at exactly 2000 attempts and 256.000 kimg.",
        "- Dataset, source checkpoint, source commit, optimizer, AMP, batch, LR, gap factors,",
        "  checkpoint cadence, and `--metrics=none` identities passed the frozen-option audit.",
        "- Every semantic non-finite counter is zero. Raw-gradient non-finites occurred only",
        "  on recorded AMP-skipped attempts and never changed parameters.",
        "- All realized denominators are positive; denominator non-finite/nonpositive counters are zero.",
        "- Per-attempt identities passed for A/D same target, B/C same target, A/C same",
        "  denominator, and B/D same denominator in both seeds.",
        "- Final training states and snapshots are loadable, finite, mutually EMA-identical,",
        "  and bind the same within-seed initial state across arms.",
        "",
        "## Training-objective diagnostics",
        "",
        "The values use last-200 mean loss and the requested order",
        "`[C-A, B-D, D-A, B-C]`. Interaction is `B-C-D+A`.",
        "",
        "| Seed | C-A | B-D | D-A | B-C | B-A | Interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (6, 7):
        diag = report["diagnostics"][str(seed)]
        lines.append(
            f"| {seed} | {diag['C_minus_A']:.8f} | {diag['B_minus_D']:.8f} | "
            f"{diag['D_minus_A']:.8f} | {diag['B_minus_C']:.8f} | "
            f"{diag['B_minus_A']:.8f} | {diag['interaction']:.8f} |"
        )
    structure = report["structure_check"]
    lines += [
        "",
        f"- Target geometry stably raises the raw objective: **{structure['target_geometry_raises_raw_objective']}**.",
        f"- Denominator scaling stably lowers the raw objective: **{structure['denominator_lowers_raw_objective']}**.",
        f"- B versus A remains approximately cancelling (descriptive): **{structure['B_vs_A_near_cancellation']}**.",
        f"- The factorial interaction remains small relative to the component contrasts (descriptive): **{structure['interaction_small']}**.",
        "",
        "These statements describe only the training objective. Frozen FID/KID evaluation",
        "is required before any quality interpretation.",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_extension_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audits = {
        seed: load_json(root / "integrity" / f"seed{seed}_integrity_audit.json")
        for seed in (6, 7)
    }
    for seed, audit in audits.items():
        if audit.get("status") != "PASS" or audit.get("seed") != seed:
            fail(f"seed{seed} integrity audit is not PASS")
    diagnostics: dict[str, Any] = {}
    for seed, audit in audits.items():
        loss = {
            arm: float(audit["cells"][arm]["last200_loss_mean"])
            for arm in ARMS
        }
        diagnostics[str(seed)] = {
            "C_minus_A": loss["C"] - loss["A"],
            "B_minus_D": loss["B"] - loss["D"],
            "D_minus_A": loss["D"] - loss["A"],
            "B_minus_C": loss["B"] - loss["C"],
            "B_minus_A": loss["B"] - loss["A"],
            "interaction": loss["B"] - loss["C"] - loss["D"] + loss["A"],
        }
    target_raise = all(
        diagnostics[str(seed)][name] > 0
        for seed in (6, 7)
        for name in ("C_minus_A", "B_minus_D")
    )
    denominator_lower = all(
        diagnostics[str(seed)][name] < 0
        for seed in (6, 7)
        for name in ("D_minus_A", "B_minus_C")
    )
    cancellation = all(
        abs(diagnostics[str(seed)]["B_minus_A"])
        < 0.25
        * max(
            abs(diagnostics[str(seed)]["C_minus_A"]),
            abs(diagnostics[str(seed)]["D_minus_A"]),
        )
        for seed in (6, 7)
    )
    interaction_small = all(
        abs(diagnostics[str(seed)]["interaction"])
        < 0.25
        * max(
            abs(diagnostics[str(seed)]["C_minus_A"]),
            abs(diagnostics[str(seed)]["B_minus_D"]),
            abs(diagnostics[str(seed)]["D_minus_A"]),
            abs(diagnostics[str(seed)]["B_minus_C"]),
        )
        for seed in (6, 7)
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "generated_utc": utc_now(),
        "classification": "secondary_precision_extension_not_original_preregistration",
        "seeds": {str(seed): audits[seed] for seed in (6, 7)},
        "diagnostics": diagnostics,
        "structure_check": {
            "descriptive_threshold_fraction": 0.25,
            "target_geometry_raises_raw_objective": target_raise,
            "denominator_lowers_raw_objective": denominator_lower,
            "B_vs_A_near_cancellation": cancellation,
            "interaction_small": interaction_small,
        },
        "quality_conclusion": "not_permitted_from_training_diagnostics",
    }
    json_path = root / "q256_factorial_seed6_7_extension_report.json"
    md_path = root / "q256_factorial_seed6_7_extension_report.md"
    write_exclusive(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_exclusive(md_path, extension_report_markdown(report))
    print(json.dumps({"status": "PASS", "json": str(json_path), "md": str(md_path)}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--seed", type=int, choices=(6, 7), required=True)
    audit.add_argument("--check-only", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_seed(args.root, args.seed, check_only=args.check_only)
        return 0 if result["status"] == "PASS" else 1
    generate_extension_report(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
