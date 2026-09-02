#!/usr/bin/env python3
"""Join verified controls/new receipts and analyze five-seed switch trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility


SEEDS = tuple(range(14, 19))
BUDGETS = (512, 640, 768, 896, 1024)
NEW_BUDGETS = BUDGETS[1:]
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
TRAJECTORIES = ("AA", "AB", "BA", "BB")
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_receipt(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one control receipt {name}, got {len(matches)}")
    return matches[0]


def load_controls(results_csv: Path, receipts_root: Path) -> dict:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if int(row["seed"]) in SEEDS and row["arm"] in ("A", "B")
        and int(row["budget_kimg"]) in BUDGETS and int(row["nfe"]) in NFES
    ]
    if len(selected) != 100:
        raise RuntimeError(f"expected 100 A/B control rows, got {len(selected)}")
    controls = {}
    for row in selected:
        key = (int(row["seed"]), row["arm"], int(row["budget_kimg"]), int(row["nfe"]))
        if key in controls:
            raise RuntimeError(f"duplicate control cell: {key}")
        receipt_path = find_receipt(receipts_root, row["receipt_file"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "PASS"
            or receipt.get("seed") != key[0]
            or receipt.get("arm") != key[1]
            or receipt.get("budget_kimg") != key[2]
            or receipt.get("nfe") != key[3]
            or receipt.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or receipt.get("evaluator_source_commit") != EVALUATOR_COMMIT
            or receipt.get("dataset_sha256") != DATASET_SHA256
            or receipt.get("generated_feature_sha256")
            != row["generated_feature_sha256"]
        ):
            raise RuntimeError(f"control receipt identity mismatch: {receipt_path}")
        metrics = receipt["metrics"]
        for metric in METRICS:
            if not math.isclose(float(metrics[metric]), float(row[metric]),
                                rel_tol=0, abs_tol=0):
                raise RuntimeError(f"control metric mismatch: {receipt_path}")
        controls[key] = {
            metric: float(row[metric]) for metric in METRICS
        }
    return controls


def load_new(receipts_root: Path, protocol_sha: str) -> dict:
    paths = sorted(receipts_root.glob("*.json"))
    if len(paths) != 80:
        raise RuntimeError(f"expected 80 new receipts, got {len(paths)}")
    values = {}
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        key = (
            int(receipt["seed"]), receipt["branch"],
            int(receipt["budget_kimg"]), int(receipt["nfe"]),
        )
        if key in values:
            raise RuntimeError(f"duplicate new evaluation cell: {key}")
        if (
            receipt.get("status") != "PASS"
            or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("evaluator_commit") != EVALUATOR_COMMIT
            or receipt.get("dataset_sha256") != DATASET_SHA256
            or receipt.get("kid_fid_shared_feature_identity") is not True
        ):
            raise RuntimeError(f"new evaluation receipt identity mismatch: {path}")
        values[key] = {
            metric: float(receipt["metrics"][metric]) for metric in METRICS
        }
    expected = {
        (seed, branch, budget, nfe)
        for seed in SEEDS for branch in ("A_to_B", "B_to_A")
        for budget in NEW_BUDGETS for nfe in NFES
    }
    if set(values) != expected:
        raise RuntimeError("new evaluation matrix has missing or extra cells")
    return values


def normalized_aulc(points: list[tuple[int, float]]) -> float:
    if [budget for budget, _ in points] != list(BUDGETS):
        raise RuntimeError("AULC budget grid mismatch")
    area = sum(
        (right_budget - left_budget) * (left_value + right_value) / 2
        for (left_budget, left_value), (right_budget, right_value)
        in zip(points, points[1:])
    )
    return area / (BUDGETS[-1] - BUDGETS[0])


def create_plots(output: Path, trajectories: list[dict], contrasts: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    colors = {"AA": "#1f77b4", "AB": "#ff7f0e", "BA": "#2ca02c", "BB": "#d62728"}
    for metric in METRICS:
        for nfe in NFES:
            subset = [row for row in trajectories if row["metric"] == metric and row["nfe"] == nfe]
            fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True)
            for axis, seed in zip(axes, SEEDS):
                seed_rows = sorted((row for row in subset if row["seed"] == seed), key=lambda row: row["budget_kimg"])
                for trajectory in TRAJECTORIES:
                    axis.plot(BUDGETS, [row[trajectory] for row in seed_rows], marker="o", label=trajectory, color=colors[trajectory])
                axis.set_title(f"seed {seed}")
                axis.set_xlabel("kimg")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel(metric)
            axes[-1].legend(fontsize=8)
            fig.tight_layout()
            path = output / f"per_seed_four_trajectories_{metric}_nfe{nfe}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path.name)

            contrast_subset = [row for row in contrasts if row["metric"] == metric and row["nfe"] == nfe]
            for names, stem in (("S_A S_B".split(), "current_schedule_effects"),
                                ("H_A H_B I_switch".split(), "history_and_interaction")):
                fig, axis = plt.subplots(figsize=(7.2, 4.5))
                for name in names:
                    for seed in SEEDS:
                        seed_rows = sorted((row for row in contrast_subset if row["seed"] == seed), key=lambda row: row["budget_kimg"])
                        axis.plot(BUDGETS, [row[name] for row in seed_rows], alpha=0.18, linewidth=1)
                    means = [statistics.mean(row[name] for row in contrast_subset if row["budget_kimg"] == budget) for budget in BUDGETS]
                    axis.plot(BUDGETS, means, marker="o", linewidth=2.4, label=f"{name} mean")
                axis.axhline(0, color="black", linewidth=0.8)
                axis.set_xlabel("kimg")
                axis.set_ylabel(f"{metric} contrast")
                axis.grid(alpha=0.25)
                axis.legend()
                fig.tight_layout()
                path = output / f"{stem}_{metric}_nfe{nfe}.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                paths.append(path.name)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--control-receipts", type=Path, required=True)
    parser.add_argument("--new-receipts", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import hashlib
    protocol_sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    controls = load_controls(args.control_results, args.control_receipts)
    new = load_new(args.new_receipts, protocol_sha)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    trajectories = []
    contrasts = []
    for seed in SEEDS:
        for nfe in NFES:
            for metric in METRICS:
                for budget in BUDGETS:
                    aa = controls[(seed, "A", budget, nfe)][metric]
                    bb = controls[(seed, "B", budget, nfe)][metric]
                    ab = aa if budget == 512 else new[(seed, "A_to_B", budget, nfe)][metric]
                    ba = bb if budget == 512 else new[(seed, "B_to_A", budget, nfe)][metric]
                    trajectory = {"seed": seed, "nfe": nfe, "metric": metric,
                                  "budget_kimg": budget, "AA": aa, "AB": ab,
                                  "BA": ba, "BB": bb}
                    trajectories.append(trajectory)
                    contrasts.append({
                        "seed": seed, "nfe": nfe, "metric": metric,
                        "budget_kimg": budget,
                        "S_A": ab - aa,
                        "S_B": bb - ba,
                        "H_A": ba - aa,
                        "H_B": bb - ab,
                        "I_switch": bb - ba - ab + aa,
                    })
    if len(trajectories) != 100 or len(contrasts) != 100:
        raise RuntimeError("seed-level trajectory table is incomplete")
    aulc = []
    for seed in SEEDS:
        for nfe in NFES:
            for metric in METRICS:
                rows = sorted((row for row in trajectories if row["seed"] == seed and row["nfe"] == nfe and row["metric"] == metric), key=lambda row: row["budget_kimg"])
                values = {trajectory: normalized_aulc([(row["budget_kimg"], row[trajectory]) for row in rows]) for trajectory in TRAJECTORIES}
                aulc.append({"seed": seed, "nfe": nfe, "metric": metric, **values,
                             "S_A": values["AB"] - values["AA"],
                             "S_B": values["BB"] - values["BA"],
                             "H_A": values["BA"] - values["AA"],
                             "H_B": values["BB"] - values["AB"],
                             "I_switch": values["BB"] - values["BA"] - values["AB"] + values["AA"]})
    summaries = []
    for nfe in NFES:
        for metric in METRICS:
            for budget in BUDGETS:
                rows = [row for row in contrasts if row["nfe"] == nfe and row["metric"] == metric and row["budget_kimg"] == budget]
                for name in ("S_A", "S_B", "H_A", "H_B", "I_switch"):
                    values = [row[name] for row in rows]
                    summaries.append({"nfe": nfe, "metric": metric,
                                      "budget_kimg": budget, "contrast": name,
                                      "mean": statistics.mean(values),
                                      "median": statistics.median(values), "n_seeds": 5})
    write_csv(output / "per_seed_trajectories.csv", trajectories)
    write_csv(output / "per_seed_contrasts.csv", contrasts)
    write_csv(output / "per_seed_aulc.csv", aulc)
    write_csv(output / "contrast_summaries.csv", summaries)
    plots = create_plots(output, trajectories, contrasts)
    audit = {"schema": "ect.q256.schedule-switch-analysis/v1", "status": "PASS",
             "protocol_sha256": protocol_sha, "new_evaluation_jobs": 80,
             "verified_control_cells": 100, "trajectory_rows": 100,
             "contrast_rows": 100, "aulc_rows": len(aulc),
             "statistical_unit": "training seed", "plots": plots,
             "claim_boundary": "descriptive five-seed conditional post-switch evidence; no global causal percentage or universal arm ranking"}
    reproducibility.atomic_json_dump(audit, output / "analysis_audit.json", overwrite=False)
    report = ["# q256 512-kimg crossed schedule-switch results", "",
              "Status: **PASS**", "", f"Protocol SHA256: `{protocol_sha}`", "",
              "The analysis contains all five seeds, both crossed branches, four post-switch budgets, both NFEs, and both KID/FID metrics. Controls were imported only after receipt-level checkpoint, evaluator, dataset, and generated-feature verification.", "",
              "Contrasts follow the frozen definitions `S_A=AB-AA`, `S_B=BB-BA`, `H_A=BA-AA`, `H_B=BB-AB`, and `I_switch=BB-BA-AB+AA`. Lower FID/KID is better.", "",
              "All per-seed trajectories, contrasts, and normalized trapezoidal AULCs are in the adjacent CSV files. Means and medians are descriptive summaries over five training seeds; budget × NFE cells are not treated as independent samples.", "",
              "Claim boundary: these are conditional post-switch intervention contrasts and descriptive five-seed quality evidence. They do not support a universal schedule ranking, a target/weight causal percentage, or unrestricted global extrapolation.", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report)); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "jobs": 80, "plots": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
