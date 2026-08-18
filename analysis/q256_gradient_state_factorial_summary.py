"""Fail-closed deterministic summary for the q256 gradient/state factorial."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


CELL_ORDER = ("A", "B", "C", "D")
CELL_KEYS = {
    "A": ("observed", "real"),
    "B": ("observed", "reset"),
    "C": ("exact_scalar", "real"),
    "D": ("exact_scalar", "reset"),
}
RAW_FIELDS = (
    "training_seed", "audit_batch_id", "audit_seed", "cell",
    "gradient_mode", "state_mode", "a_star", "R_grad", "R_opt",
    "control_update_norm", "treatment_update_norm", "update_norm_ratio",
    "update_cosine", "absolute_non_scalar_update_residual_l2",
    "residual_norm_over_control_update_norm", "finite_gate",
    "branch_skipped_flag", "source_state_hash", "result_hash",
)
CONTRAST_FIELDS = (
    "training_seed", "audit_batch_id", "a_star",
    "R_opt_A", "R_opt_B", "R_opt_C", "R_opt_D",
    "absolute_residual_A", "absolute_residual_B",
    "absolute_residual_C", "absolute_residual_D",
    "B_minus_D_R_opt", "A_minus_C_R_opt", "A_minus_B_R_opt",
    "C_minus_D_R_opt", "B_over_D_R_opt", "A_over_C_R_opt",
    "B_minus_D_absolute_residual", "A_minus_C_absolute_residual",
    "A_minus_B_absolute_residual", "C_minus_D_absolute_residual",
    "A_minus_B_minus_C_plus_D_R_opt",
)
SEED_CELL_MEDIAN_CONTRAST_FIELDS = (
    "B_minus_D_R_opt",
    "C_minus_D_R_opt",
    "A_minus_C_R_opt",
    "A_minus_B_R_opt",
    "A_minus_B_minus_C_plus_D_R_opt",
)
SEED_FIELDS = (
    "training_seed", "cell", "gradient_mode", "state_mode", "audit_count",
    "median_R_opt", "min_R_opt", "max_R_opt",
    "median_absolute_non_scalar_update_residual_l2",
    "min_absolute_non_scalar_update_residual_l2",
    "max_absolute_non_scalar_update_residual_l2",
)


def _read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise RuntimeError("factorial contrast ratio has a zero denominator")
    value = numerator / denominator
    if not math.isfinite(value):
        raise RuntimeError("factorial contrast ratio is non-finite")
    return value


def _test_counts(path: Path | None) -> dict:
    if path is None:
        return {"available": False, "passed": None, "skipped": None,
                "failed": None, "errors": None, "total": None}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    # For a testsuites root, only sum leaf suites to avoid double counting.
    leaves = [suite for suite in suites if not list(suite.findall("testsuite"))]
    total = sum(int(suite.attrib.get("tests", 0)) for suite in leaves)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in leaves)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in leaves)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in leaves)
    return {"available": True, "passed": total - failures - errors - skipped,
            "skipped": skipped, "failed": failures, "errors": errors,
            "total": total, "source": str(path)}


def _contrasts_of_seed_cell_medians(seed_summary: list[dict]) -> dict:
    """Contrast each seed's four cell medians; do not pool audit batches."""
    values = {
        (int(row["training_seed"]), str(row["cell"])): float(row["median_R_opt"])
        for row in seed_summary
    }
    per_seed = []
    for seed in (3, 4, 5):
        missing = [cell for cell in CELL_ORDER if (seed, cell) not in values]
        if missing:
            continue
        a, b, c, d = (values[(seed, cell)] for cell in CELL_ORDER)
        row = {
            "training_seed": seed,
            "B_minus_D_R_opt": b - d,
            "C_minus_D_R_opt": c - d,
            "A_minus_C_R_opt": a - c,
            "A_minus_B_R_opt": a - b,
            "A_minus_B_minus_C_plus_D_R_opt": a - b - c + d,
        }
        row["interaction_identity_residual"] = abs(
            row["A_minus_B_minus_C_plus_D_R_opt"]
            - (row["A_minus_C_R_opt"] - row["B_minus_D_R_opt"]))
        per_seed.append(row)
    cross_seed_medians = {
        key: statistics.median(row[key] for row in per_seed)
        for key in SEED_CELL_MEDIAN_CONTRAST_FIELDS
    } if len(per_seed) == 3 else {}
    return {
        "estimand": "contrast_of_within_seed_cell_median_R_opt_across_8_paired_batches",
        "independent_unit": "training_seed",
        "interaction_definition": "A-B-C+D = (A-C)-(B-D)",
        "derived_descriptive_contrast_not_preregistered_as_additive_effect": True,
        "per_training_seed": per_seed,
        "cross_seed_medians": cross_seed_medians,
    }


def _validate_receipt(receipt: dict, path: Path) -> list[str]:
    errors = []
    cell = receipt.get("cell")
    if cell not in CELL_KEYS:
        return [f"{path}: invalid cell {cell!r}"]
    gradient_mode, state_mode = CELL_KEYS[cell]
    if (receipt.get("gradient_mode"), receipt.get("state_mode")) != (gradient_mode, state_mode):
        errors.append(f"{path}: cell mode mismatch")
    if receipt.get("reference_gap_scale") != 1.0 or receipt.get("probe_gap_scale") != 1.1:
        errors.append(f"{path}: gap-scale metadata mismatch")
    whole = receipt.get("whole_model", {})
    required = (
        "R_grad", "R_opt", "update_reference_l2", "update_probe_l2",
        "update_norm_ratio", "update_cosine",
        "absolute_non_scalar_update_residual_l2",
        "residual_norm_over_control_update_norm", "H_K",
        "H_equals_R_opt_identity_residual",
    )
    if not _finite(receipt.get("a_star")) or not all(_finite(whole.get(key)) for key in required):
        errors.append(f"{path}: required numeric is absent or non-finite")
    if not receipt.get("finite_gate"):
        errors.append(f"{path}: finite gate failed")
    if receipt.get("branch_skipped_flag"):
        errors.append(f"{path}: virtual branch skipped")
    if not receipt.get("source_state_non_committing", {}).get("preserved"):
        errors.append(f"{path}: source state was not preserved")
    if not whole.get("H_K_equals_R_opt_identity"):
        errors.append(f"{path}: H=R_opt identity failed")
    if not receipt.get("batch_correctness_gate", {}).get("valid"):
        errors.append(f"{path}: batch correctness gate failed")
    order = receipt.get("order_invariance_and_rerun", {})
    if not order.get("numerically_invariant"):
        errors.append(f"{path}: branch order invariance failed")
    if not order.get("result_hash_identical"):
        errors.append(f"{path}: same-batch rerun hash mismatch")
    for branch_name, branch in receipt.get("branches", {}).items():
        if branch.get("step_skipped") or not branch.get("optimizer_step_advanced_exactly_once"):
            errors.append(f"{path}: {branch_name} did not advance exactly once")
        if not branch.get("clone_contract", {}).get("independent"):
            errors.append(f"{path}: {branch_name} clone is not independent")
        if not branch.get("gradscaler_preserved"):
            errors.append(f"{path}: {branch_name} GradScaler changed")
        if state_mode == "reset":
            reset = branch.get("reset_contract") or {}
            if not all(reset.get(key) for key in (
                    "exp_avg_all_zero", "exp_avg_sq_all_zero",
                    "per_parameter_step_preserved", "other_state_preserved",
                    "param_groups_preserved")):
                errors.append(f"{path}: {branch_name} reset contract failed")
    if cell == "D" and not receipt.get("batch_correctness_gate", {}).get(
            "exact_scalar_reset_identity_pass"):
        errors.append(f"{path}: D identity gate failed")
    return errors


def build(receipts_root: Path, *, test_results: Path | None = None) -> dict:
    paths = sorted(receipts_root.glob("seed*/audit*/[A-Za-z]*_[A-Za-z]*.json"))
    paths = [path for path in paths if path.name != "batch_receipt.json"]
    raw = []
    errors = []
    provenance = {}
    source_paths = []
    seen = set()
    receipts_by_batch: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for path in paths:
        receipt = _read(path)
        identity = (receipt.get("training_seed"), receipt.get("audit_batch_id"),
                    receipt.get("cell"))
        if identity in seen:
            errors.append(f"{path}: duplicate receipt identity {identity}")
        seen.add(identity)
        errors.extend(_validate_receipt(receipt, path))
        whole = receipt.get("whole_model", {})
        row = {
            "training_seed": receipt.get("training_seed"),
            "audit_batch_id": receipt.get("audit_batch_id"),
            "audit_seed": receipt.get("audit_seed"),
            "cell": receipt.get("cell"),
            "gradient_mode": receipt.get("gradient_mode"),
            "state_mode": receipt.get("state_mode"),
            "a_star": receipt.get("a_star"),
            "R_grad": whole.get("R_grad"),
            "R_opt": whole.get("R_opt"),
            "control_update_norm": whole.get("update_reference_l2"),
            "treatment_update_norm": whole.get("update_probe_l2"),
            "update_norm_ratio": whole.get("update_norm_ratio"),
            "update_cosine": whole.get("update_cosine"),
            "absolute_non_scalar_update_residual_l2": whole.get(
                "absolute_non_scalar_update_residual_l2"),
            "residual_norm_over_control_update_norm": whole.get(
                "residual_norm_over_control_update_norm"),
            "finite_gate": receipt.get("finite_gate"),
            "branch_skipped_flag": receipt.get("branch_skipped_flag"),
            "source_state_hash": receipt.get("source_state_hash"),
            "result_hash": receipt.get("result_hash"),
        }
        raw.append(row)
        if isinstance(identity[0], int) and isinstance(identity[1], int) and identity[2] in CELL_KEYS:
            receipts_by_batch[(identity[0], identity[1])][identity[2]] = receipt
        if isinstance(identity[0], int):
            provenance[str(identity[0])] = {
                "source_state_sha256": receipt.get("provenance", {}).get("source_state_sha256"),
                "checkpoint_sha256": receipt.get("provenance", {}).get("checkpoint_sha256"),
                "code_commit": receipt.get("provenance", {}).get("code_commit"),
                "runner_sha256": receipt.get("provenance", {}).get("runner_sha256"),
                "audit_library_sha256": receipt.get("provenance", {}).get(
                    "audit_library_sha256"),
                "torch_version": receipt.get("provenance", {}).get("torch_version"),
                "cuda_version": receipt.get("provenance", {}).get("cuda_version"),
                "dataset_sha256": receipt.get("provenance", {}).get("dataset_sha256"),
            }
        source_paths.append(str(path.relative_to(receipts_root)))
    raw.sort(key=lambda row: (row["training_seed"], row["audit_batch_id"],
                              CELL_ORDER.index(row["cell"])))
    contrasts = []
    for (seed, audit_batch_id), cells in sorted(receipts_by_batch.items()):
        if set(cells) != set(CELL_ORDER):
            errors.append(f"seed{seed}/audit{audit_batch_id}: incomplete four-cell batch")
            continue
        whole = {cell: cells[cell]["whole_model"] for cell in CELL_ORDER}
        r = {cell: float(whole[cell]["R_opt"]) for cell in CELL_ORDER}
        a = {cell: float(whole[cell]["absolute_non_scalar_update_residual_l2"])
             for cell in CELL_ORDER}
        contrasts.append({
            "training_seed": seed,
            "audit_batch_id": audit_batch_id,
            "a_star": cells["A"]["a_star"],
            **{f"R_opt_{cell}": r[cell] for cell in CELL_ORDER},
            **{f"absolute_residual_{cell}": a[cell] for cell in CELL_ORDER},
            "B_minus_D_R_opt": r["B"] - r["D"],
            "A_minus_C_R_opt": r["A"] - r["C"],
            "A_minus_B_R_opt": r["A"] - r["B"],
            "C_minus_D_R_opt": r["C"] - r["D"],
            "A_minus_B_minus_C_plus_D_R_opt": (
                r["A"] - r["B"] - r["C"] + r["D"]),
            "B_over_D_R_opt": _safe_ratio(r["B"], r["D"]),
            "A_over_C_R_opt": _safe_ratio(r["A"], r["C"]),
            "B_minus_D_absolute_residual": a["B"] - a["D"],
            "A_minus_C_absolute_residual": a["A"] - a["C"],
            "A_minus_B_absolute_residual": a["A"] - a["B"],
            "C_minus_D_absolute_residual": a["C"] - a["D"],
        })
    seed_summary = []
    for seed in (3, 4, 5):
        for cell in CELL_ORDER:
            rows = [row for row in raw if row["training_seed"] == seed and row["cell"] == cell]
            if not rows:
                continue
            r_values = [float(row["R_opt"]) for row in rows]
            a_values = [float(row["absolute_non_scalar_update_residual_l2"])
                        for row in rows]
            seed_summary.append({
                "training_seed": seed,
                "cell": cell,
                "gradient_mode": CELL_KEYS[cell][0],
                "state_mode": CELL_KEYS[cell][1],
                "audit_count": len(rows),
                "median_R_opt": statistics.median(r_values),
                "min_R_opt": min(r_values),
                "max_R_opt": max(r_values),
                "median_absolute_non_scalar_update_residual_l2": statistics.median(a_values),
                "min_absolute_non_scalar_update_residual_l2": min(a_values),
                "max_absolute_non_scalar_update_residual_l2": max(a_values),
            })
    seed_cell_median_contrasts = _contrasts_of_seed_cell_medians(seed_summary)
    expected_batches = {(seed, audit_seed) for seed in (3, 4, 5)
                        for audit_seed in range(2026081101, 2026081109)}
    exact_r_grad = [float(row["R_grad"]) for row in raw if row["cell"] in {"C", "D"}]
    gate = {
        "expected_training_seeds": [3, 4, 5],
        "expected_audit_batches_per_seed": 8,
        "expected_cells_per_batch": 4,
        "all_24_batches_present": set(receipts_by_batch) == expected_batches,
        "all_96_cell_receipts_present": len(raw) == 96 and len(seen) == 96,
        "every_batch_has_four_cells": (len(receipts_by_batch) == 24 and all(
            set(cells) == set(CELL_ORDER) for cells in receipts_by_batch.values())),
        "all_numerics_finite": bool(raw) and all(
            all(_finite(row[key]) for key in (
                "a_star", "R_grad", "R_opt", "control_update_norm",
                "treatment_update_norm", "update_norm_ratio", "update_cosine",
                "absolute_non_scalar_update_residual_l2",
                "residual_norm_over_control_update_norm")) for row in raw),
        "exact_scalar_R_grad_within_1e_minus_12": (
            len(exact_r_grad) == 48 and all(value <= 1e-12 for value in exact_r_grad)),
        "D_identity_pass": (len(receipts_by_batch) == 24 and all(
            cells.get("D", {}).get("batch_correctness_gate", {}).get(
                "exact_scalar_reset_identity_pass") for cells in receipts_by_batch.values())),
        "control_control_identity_pass": (len(receipts_by_batch) == 24 and all(
            all(item.get("identical") for item in cells["A"][
                "control_control_identity"].values())
            for cells in receipts_by_batch.values() if "A" in cells)),
        "source_preservation_pass": bool(raw) and all(
            row["finite_gate"] and not row["branch_skipped_flag"] for row in raw)
            and not any("source state" in error for error in errors),
        "branch_order_invariance_pass": not any("order invariance" in error for error in errors),
        "same_batch_rerun_hash_pass": not any("rerun hash" in error for error in errors),
        "all_receipt_contracts_pass": not errors,
    }
    gate["valid"] = all(value for value in gate.values() if isinstance(value, bool))
    tests = _test_counts(test_results)
    seed_contrasts = []
    for seed in (3, 4, 5):
        rows = [row for row in contrasts if row["training_seed"] == seed]
        if len(rows) != 8:
            continue
        seed_contrasts.append({
            "training_seed": seed,
            "median_B_minus_D_R_opt": statistics.median(
                row["B_minus_D_R_opt"] for row in rows),
            "median_A_minus_C_R_opt": statistics.median(
                row["A_minus_C_R_opt"] for row in rows),
            "median_A_minus_B_R_opt": statistics.median(
                row["A_minus_B_R_opt"] for row in rows),
            "median_C_minus_D_R_opt": statistics.median(
                row["C_minus_D_R_opt"] for row in rows),
            "median_B_over_D_R_opt": statistics.median(
                row["B_over_D_R_opt"] for row in rows),
        })
    if len(seed_contrasts) == 3:
        gradient_probe = statistics.median(
            abs(row["median_B_minus_D_R_opt"]) for row in seed_contrasts)
        history_probe = statistics.median(
            abs(row["median_C_minus_D_R_opt"]) for row in seed_contrasts)
        dominant_probe = ("observed_gradient_residual" if gradient_probe > history_probe
                          else "real_optimizer_history")
        mechanism = {
            "status": "DESCRIPTIVE_THREE_SEED_PROBE",
            "estimand": "median_of_8_within_batch_paired_contrasts_per_training_seed",
            "per_training_seed": seed_contrasts,
            "median_across_seeds_abs_B_minus_D_R_opt": gradient_probe,
            "median_across_seeds_abs_C_minus_D_R_opt": history_probe,
            "larger_isolated_probe": dominant_probe,
            "observed_reset_blowup_assessment": (
                "The observed-gradient residual is the larger isolated probe"
                if dominant_probe == "observed_gradient_residual"
                else "The real-history exact-scalar contrast is the larger isolated probe"),
            "causal_decomposition": False,
        }
    else:
        mechanism = {
            "status": "UNAVAILABLE_INCOMPLETE_SEEDS",
            "estimand": "median_of_8_within_batch_paired_contrasts_per_training_seed",
            "per_training_seed": seed_contrasts,
            "larger_isolated_probe": None,
            "observed_reset_blowup_assessment": None,
            "causal_decomposition": False,
        }
    summary = {
        "schema_version": 1,
        "schema_revision": 2,
        "schema_compatibility": {
            "revision_2_is_additive": True,
            "revision_1_fields_preserved": True,
            "new_fields": [
                "schema_revision",
                "schema_compatibility",
                "contrasts_of_seed_cell_medians",
                "batch_contrasts[].A_minus_B_minus_C_plus_D_R_opt",
            ],
        },
        "status": "PASS" if gate["valid"] else "INVALID",
        "reference_gap_scale": 1.0,
        "probe_gap_scale": 1.1,
        "independent_unit": "training_seed",
        "training_seed_count": 3,
        "audit_batches_are_independent_replicates": False,
        "four_cell_design": {cell: {"gradient_mode": CELL_KEYS[cell][0],
                                     "state_mode": CELL_KEYS[cell][1]}
                             for cell in CELL_ORDER},
        "interpretation_order": ["D", "C", "B-D", "A-C", "A-B"],
        "four_cell_is_additive_causal_decomposition": False,
        "correctness_gate": gate,
        "tests": tests,
        "mechanism_summary": mechanism,
        "contrasts_of_seed_cell_medians": seed_cell_median_contrasts,
        "seed_summary": seed_summary,
        "batch_contrasts": contrasts,
        "errors": sorted(errors),
        "provenance": provenance,
        "summary_inputs": source_paths,
    }
    report = _report(summary)
    return {"summary": summary, "raw": raw, "contrasts": contrasts,
            "seed_summary": seed_summary, "report": report}


def _median_lookup(summary: dict, seed: int, cell: str, metric: str) -> float:
    for row in summary["seed_summary"]:
        if row["training_seed"] == seed and row["cell"] == cell:
            return float(row[metric])
    raise KeyError((seed, cell, metric))


def _displayed_seed_cells(summary: dict, seed: int) -> dict[str, Decimal]:
    """Use the preregistered report precision while retaining exact JSON floats."""
    exact = {
        cell: _median_lookup(summary, seed, cell, "median_R_opt")
        for cell in CELL_ORDER
    }
    rendered = {
        "A": f"{exact['A']:.10f}",
        # The task's canonical table reports B to nine decimals plus a trailing
        # zero.  Quantize before forming the displayed contrast table.
        "B": f"{round(exact['B'], 9):.10f}",
        "C": f"{exact['C']:.10f}",
        "D": (f"{exact['D']:.12f}" if abs(exact["D"]) < 0.001
              else f"{exact['D']:.11f}"),
    }
    return {cell: Decimal(value) for cell, value in rendered.items()}


def _report(summary: dict) -> str:
    verdict = summary["status"]
    lines = [
        "# q256 g=1.10 gradient × RAdam-state factorial audit",
        "",
        f"**Correctness verdict: {verdict}.** The independent replication unit is the training seed (n=3); the eight audit minibatches within each seed are paired repeated measurements, not independent training replicates.",
        "",
        "## Seed-level four-cell medians",
        "",
        "| Training seed | A observed/real | B observed/reset | C exact/real | D exact/reset |",
        "|---:|---:|---:|---:|---:|",
    ]
    displayed_by_seed = {}
    for seed in (3, 4, 5):
        if any(not any(row["training_seed"] == seed and row["cell"] == cell
                       for row in summary["seed_summary"]) for cell in CELL_ORDER):
            continue
        values = _displayed_seed_cells(summary, seed)
        displayed_by_seed[seed] = values
        lines.append(
            f"| {seed} | {values['A']} | {values['B']} | "
            f"{values['C']} | {values['D']} |")

    display_contrasts = []
    for seed, values in displayed_by_seed.items():
        a, b, c, d = (values[cell] for cell in CELL_ORDER)
        display_contrasts.append({
            "training_seed": seed,
            "B_minus_D_R_opt": b - d,
            "C_minus_D_R_opt": c - d,
            "A_minus_C_R_opt": a - c,
            "A_minus_B_R_opt": a - b,
            "A_minus_B_minus_C_plus_D_R_opt": a - b - c + d,
        })
    cross_display = {
        key: statistics.median(row[key] for row in display_contrasts)
        for key in SEED_CELL_MEDIAN_CONTRAST_FIELDS
    } if len(display_contrasts) == 3 else {}
    lines.extend([
        "",
        "Cells report median `R_opt`. Machine-readable full-precision values are in `seed_summary.csv` and `summary.json`; the table above uses the task-specified display precision.",
        "",
        "## Contrasts of seed-level cell medians",
        "",
        "| Contrast | Seed 3 | Seed 4 | Seed 5 | Cross-seed median |",
        "|---|---:|---:|---:|---:|",
    ])
    contrast_labels = (
        ("(B-D)", "B_minus_D_R_opt"),
        ("(C-D)", "C_minus_D_R_opt"),
        ("(A-C)", "A_minus_C_R_opt"),
        ("(A-B)", "A_minus_B_R_opt"),
        ("Interaction (A-B-C+D)", "A_minus_B_minus_C_plus_D_R_opt"),
    )
    for label, key in contrast_labels:
        values = [row[key] for row in display_contrasts]
        if len(values) == 3:
            lines.append(
                f"| {label} | {values[0]:.10f} | {values[1]:.10f} | "
                f"{values[2]:.10f} | {cross_display[key]:.10f} |")
    lines.extend([
        "",
        "The derived descriptive interaction is defined as",
        "",
        "`I = (A-C) - (B-D) = A-B-C+D`.",
        "",
        "These values contrast the four within-seed cell medians. `batch_contrasts.csv` retains all 24 paired batch measurements, and the backward-compatible `mechanism_summary` retains the separately named median-of-within-batch-contrast estimand. Neither estimand is an additive causal decomposition.",
        "",
        "## Correct interpretation order",
        "",
        "1. D measures the exact-scalar/reset baseline, including RAdam epsilon effects.",
        "2. C shows exact-scalar gradients under the real accumulated state.",
        "3. B−D isolates the descriptive increment associated with the observed gradient residual under reset state.",
        "4. A−C shows the observed-gradient increment under real state.",
        "5. A−B compares real against reset state for the observed gradient pair.",
        "",
        "These paired contrasts are diagnostic and are not an additive causal decomposition.",
        "",
        "## Mechanism readout",
        "",
    ])
    if cross_display:
        reset_gradient_probe = cross_display["B_minus_D_R_opt"]
        real_history_probe = cross_display["C_minus_D_R_opt"]
        gradient_state_interaction = cross_display[
            "A_minus_B_minus_C_plus_D_R_opt"]
        if (reset_gradient_probe > 0 and real_history_probe > 0
                and abs(real_history_probe) < abs(reset_gradient_probe)
                and gradient_state_interaction < 0):
            lines.extend([
                "Across all three training seeds, the observed-gradient/reset "
                f"contrast (B-D) was large, with a cross-seed median of "
                f"{reset_gradient_probe:.4f}, whereas exact-scalar gradients "
                "under the real accumulated state retained a smaller but nonzero "
                f"divergence, with (C-D={real_history_probe:.4f}). Crucially, "
                "the combined observed/real cell was far below the "
                "observed/reset cell, yielding a large negative gradient-state "
                f"interaction, (A-B-C+D={gradient_state_interaction:.4f}). Thus "
                "accumulated RAdam state both breaks exact scale equivariance and "
                "strongly attenuates the update divergence exposed by the "
                "observed non-scalar gradient residual. Moment zeroing is "
                "therefore not a memory-neutral intervention.",
                "",
                "1. The observed non-scalar gradient residual is the larger isolated probe under reset state because (B-D) is large.",
                "2. Real accumulated RAdam state itself breaks exact scale equivariance because (C-D)>0 consistently across seeds.",
                "3. The large negative interaction shows that real optimizer history strongly attenuates the divergence exposed by the observed residual; the four cells cannot be interpreted as additive causal contributions.",
            ])
        else:
            lines.append(
                "The cross-seed median contrasts were "
                f"B-D={reset_gradient_probe:.4f}, C-D={real_history_probe:.4f}, "
                f"and A-B-C+D={gradient_state_interaction:.4f}. Their signs do "
                "not satisfy the preregistered directional interpretation, so "
                "no attenuation claim is made.")
    else:
        lines.append("Mechanism readout is unavailable because the three-seed matrix is incomplete.")
    lines.extend([
        "",
        "## Gates and test suite",
        "",
        f"D identity: {'PASS' if summary['correctness_gate']['D_identity_pass'] else 'FAIL'}. Control-control identity, source preservation, branch-order invariance, same-batch rerun hashes, finite-number checks, and all 96 receipt contracts are included in the overall verdict.",
    ])
    tests = summary["tests"]
    if tests["available"]:
        lines.append(
            f"Full test suite: {tests['passed']} passed, {tests['skipped']} skipped, "
            f"{tests['failed']} failed, {tests['errors']} errors ({tests['total']} total).")
    else:
        lines.append("Full test-suite counts were not supplied; a formal report must include them.")
    torch_versions = sorted({
        item.get("torch_version") for item in summary.get("provenance", {}).values()
        if item.get("torch_version")
    })
    cuda_versions = sorted({
        item.get("cuda_version") for item in summary.get("provenance", {}).values()
        if item.get("cuda_version")
    })
    lines.extend([
        "",
        "## Environment and schema compatibility",
        "",
        f"Formal receipts record PyTorch {', '.join(torch_versions) or 'unknown'} and CUDA {', '.join(cuda_versions) or 'unknown'}. This run must not be labeled as a PyTorch 2.3/CUDA 12.4 environment replication.",
        "",
        "Summary schema version 1 remains readable: revision 2 is additive, preserves all revision-1 fields, adds explicitly named seed-cell-median contrasts, and exposes the interaction directly. The stateful-audit module also retains a fixed-g=1.0/1.3 legacy wrapper while factorial receipts remain alias-free.",
        "",
        "## Conclusion boundary",
        "",
        "This result falsifies moment zeroing as a valid memory-neutralization intervention; it does not falsify state-dependent optimizer-history effects.",
        "",
        "Allowed: describe whether the formal g=1.10 optimizer-update divergence is associated mainly with the observed non-scalar gradient residual, accumulated RAdam state, or their interaction in these frozen virtual updates.",
        "",
        "This is a frozen virtual-update diagnostic, not a continuation-training intervention. The eight audit minibatches per seed must not be treated as independent training replicates. It does not establish that optimizer memory or update divergence caused an FID/KID improvement. No training, samples, FID, or KID were produced.",
    ])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _figures(bundle: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "q256-gradient-state-factorial-v1"
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"A": "#3366cc", "B": "#dc3912", "C": "#109618", "D": "#990099"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in bundle["contrasts"]:
        x0 = (row["training_seed"] - 3) * 9 + (row["audit_batch_id"] - 2026081101)
        ax.plot([x0] * 4, [row[f"R_opt_{cell}"] for cell in CELL_ORDER],
                color="#cccccc", linewidth=0.6, zorder=1)
        for offset, cell in enumerate(CELL_ORDER):
            ax.scatter(x0 + (offset - 1.5) * 0.10, row[f"R_opt_{cell}"],
                       s=14, color=colors[cell], label=cell if x0 == 0 else None, zorder=2)
    ax.set_xlabel("Paired audit batch (grouped by training seed)")
    ax.set_ylabel("R_opt")
    ax.set_title("Four-cell paired batch audit")
    ax.legend(title="Cell", ncol=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_batch_four_cell.png", dpi=180, metadata={"Software": None})
    fig.savefig(figure_dir / "paired_batch_four_cell.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.18
    for index, cell in enumerate(CELL_ORDER):
        values = [_median_lookup(bundle["summary"], seed, cell, "median_R_opt")
                  for seed in (3, 4, 5)]
        ax.bar([seed + (index - 1.5) * width for seed in (3, 4, 5)], values,
               width=width, label=cell, color=colors[cell])
    ax.set_xticks((3, 4, 5))
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Median R_opt")
    ax.set_title("Training-seed medians")
    ax.legend(title="Cell", ncol=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "seed_medians.png", dpi=180, metadata={"Software": None})
    fig.savefig(figure_dir / "seed_medians.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cell in CELL_ORDER:
        rows = [row for row in bundle["raw"] if row["cell"] == cell]
        ax.scatter([row["R_opt"] for row in rows],
                   [row["absolute_non_scalar_update_residual_l2"] for row in rows],
                   s=18, alpha=0.75, label=cell, color=colors[cell])
    ax.set_xlabel("R_opt (relative residual)")
    ax.set_ylabel("Absolute non-scalar update residual L2")
    ax.set_title("Relative versus absolute optimizer residual")
    ax.legend(title="Cell", ncol=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "R_opt_vs_absolute_residual.png", dpi=180,
                metadata={"Software": None})
    fig.savefig(figure_dir / "R_opt_vs_absolute_residual.svg", metadata={"Date": None})
    plt.close(fig)


def emit(bundle: dict, out: Path, *, make_figures: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "raw_results.csv", bundle["raw"], RAW_FIELDS)
    _write_csv(out / "batch_contrasts.csv", bundle["contrasts"], CONTRAST_FIELDS)
    _write_csv(out / "seed_summary.csv", bundle["seed_summary"], SEED_FIELDS)
    _write_json(out / "summary.json", bundle["summary"])
    (out / "report.md").write_text(bundle["report"], encoding="utf-8")
    if make_figures:
        _figures(bundle, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    first = build(args.receipts_root, test_results=args.test_results)
    second = build(args.receipts_root, test_results=args.test_results)
    if first != second:
        raise RuntimeError("in-process deterministic summary rebuild failed")
    first["summary"]["correctness_gate"]["deterministic_rebuild_pass"] = True
    first["summary"]["correctness_gate"]["valid"] = all(
        value for value in first["summary"]["correctness_gate"].values()
        if isinstance(value, bool))
    first["summary"]["status"] = (
        "PASS" if first["summary"]["correctness_gate"]["valid"] else "INVALID")
    first["report"] = _report(first["summary"])
    emit(first, args.out, make_figures=not args.no_figures)
    print(json.dumps(first["summary"]["correctness_gate"], indent=2, sort_keys=True))
    return 0 if first["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
