#!/usr/bin/env python3
"""Summarize the frozen q256 target-component audit and align it with NFE1 FID.

This is a descriptive analysis.  The independent unit remains the training seed;
budget points are repeated measurements, not additional replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = (3, 4, 5)
BUDGETS = (256, 512, 768, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fid(path: Path) -> dict[tuple[int, str, int], float]:
    values: dict[tuple[int, str, int], float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            arm = row["arm"]
            budget = int(float(row["budget_kimg"]))
            nfe = int(row["nfe"])
            if seed in SEEDS and arm in {"A", "B"} and budget in BUDGETS and nfe == 1:
                values[(seed, arm, budget)] = float(row["fid50k_full"])
    expected = {(s, a, k) for s in SEEDS for a in ("A", "B") for k in BUDGETS}
    missing = sorted(expected - values.keys())
    if missing:
        raise ValueError(f"missing FID rows: {missing}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--fid-csv", type=Path, required=True)
    args = parser.parse_args()

    root = args.results_root.resolve()
    primary = root / "v2_primary"
    matrix_path = root / "v2_primary_matrix_validation.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["status"] != "PASS_PRIMARY_12_STATE_MATRIX":
        raise ValueError(f"matrix gate did not pass: {matrix['status']}")

    fid = read_fid(args.fid_csv.resolve())
    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        for budget in BUDGETS:
            manifest_path = primary / f"seed{seed}_A_k{budget}" / "target_component_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            aggregate = manifest["aggregate"]
            layers = manifest["layerwise_summary"]["r_tar_unweighted_across_nonzero_reference_layers"]
            fid_a = fid[(seed, "A", budget)]
            fid_b = fid[(seed, "B", budget)]
            rows.append(
                {
                    "seed": seed,
                    "budget_kimg": budget,
                    "r_tar": aggregate["r_tar"],
                    "cos_tau_g_a": aggregate["cos_tau_g_a"],
                    "a_star": aggregate["a_star"],
                    "s_explicit": aggregate["s_explicit"],
                    "a_star_minus_s": aggregate["a_star"] - aggregate["s_explicit"],
                    "best_scalar_residual_over_g_b": aggregate["r_best_over_g_b"],
                    "identity_error_max": max(
                        aggregate["max_identity_b_equals_s_c_relative_l2"],
                        aggregate["max_identity_d_equals_s_a_relative_l2"],
                    ),
                    "layer_r_tar_median": layers["median"],
                    "layer_r_tar_q90": layers["q90"],
                    "layer_r_tar_max": layers["max"],
                    "fid_a_nfe1": fid_a,
                    "fid_b_nfe1": fid_b,
                    "log_fid_b_minus_a": math.log(fid_b) - math.log(fid_a),
                }
            )

    aligned_path = root / "aligned_gradient_fid_nfe1.csv"
    with aligned_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    budget_rows = []
    for budget in BUDGETS:
        subset = [row for row in rows if row["budget_kimg"] == budget]
        r_values = [float(row["r_tar"]) for row in subset]
        d_values = [float(row["log_fid_b_minus_a"]) for row in subset]
        budget_rows.append(
            {
                "budget_kimg": budget,
                "seed_count": len(subset),
                "r_tar_mean": mean(r_values),
                "r_tar_sd": stdev(r_values),
                "log_fid_b_minus_a_mean": mean(d_values),
                "log_fid_b_minus_a_sd": stdev(d_values),
            }
        )
    budget_path = root / "budget_descriptive_summary.csv"
    with budget_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)

    seed_changes = []
    for seed in SEEDS:
        first = next(row for row in rows if row["seed"] == seed and row["budget_kimg"] == 256)
        last = next(row for row in rows if row["seed"] == seed and row["budget_kimg"] == 1024)
        seed_changes.append(
            {
                "seed": seed,
                "r_tar_1024_minus_256": float(last["r_tar"]) - float(first["r_tar"]),
                "abs_log_fid_contrast_1024_minus_256": abs(float(last["log_fid_b_minus_a"]))
                - abs(float(first["log_fid_b_minus_a"])),
            }
        )
    seed_path = root / "seed_endpoint_changes.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_changes[0]))
        writer.writeheader()
        writer.writerows(seed_changes)

    summary = {
        "schema": "ect.q256.target-component-audit-descriptive-summary/v1",
        "status": "PASS_DESCRIPTIVE_SUMMARY",
        "independent_unit": "training_seed",
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "budgets_kimg": list(BUDGETS),
        "matrix_validation_sha256": sha256(matrix_path),
        "fid_source_sha256": sha256(args.fid_csv.resolve()),
        "aligned_csv_sha256": sha256(aligned_path),
        "budget_csv_sha256": sha256(budget_path),
        "seed_changes_csv_sha256": sha256(seed_path),
        "descriptive_only": True,
        "claim_boundary": (
            "The target component is small and nonzero in this fixed-batch whole-model mean-gradient audit. "
            "Its magnitude does not show a seed-consistent contraction from 256 to 1024 kimg. "
            "This does not identify optimizer causality or prove that batch-level target effects are negligible."
        ),
    }
    (root / "summary_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
