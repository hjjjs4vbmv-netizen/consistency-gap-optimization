#!/usr/bin/env python3
"""Analyze the verified seed3-7 crossed schedule-switch experiment."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.q256_schedule_switch_v1 import analyze_results as common
from training import reproducibility


SEEDS = (3, 4, 5, 6, 7)
BUDGETS = (512, 640, 768, 896, 1024)
NEW_BUDGETS = BUDGETS[1:]
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
TRAJECTORIES = ("AA", "AB", "BA", "BB")


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_new(root: Path, protocol_sha: str) -> dict:
    paths = sorted(root.glob("*.json"))
    if len(paths) != 80:
        raise RuntimeError(f"expected 80 switched receipts, got {len(paths)}")
    values = {}
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (
            int(item["seed"]), item["branch"],
            int(item["budget_kimg"]), int(item["nfe"]),
        )
        if (
            item.get("status") != "PASS"
            or item.get("protocol_sha256") != protocol_sha
            or item.get("kid_fid_shared_feature_identity") is not True
        ):
            raise RuntimeError(f"invalid switched receipt: {path}")
        if key in values:
            raise RuntimeError("duplicate switched evaluation cell")
        values[key] = {
            metric: float(item["metrics"][metric]) for metric in METRICS
        }
    expected = {
        (seed, branch, budget, nfe)
        for seed in SEEDS for branch in ("A_to_B", "B_to_A")
        for budget in NEW_BUDGETS for nfe in NFES
    }
    if set(values) != expected:
        raise RuntimeError("switched evaluation matrix is incomplete")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--new-receipts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol"] not in {
        "q256_ab_crossed_switch_seed3_7_v2",
        "q256_ab_crossed_switch_seed3_7_v3",
    }:
        raise RuntimeError("wrong protocol")
    protocol_sha = sha256_file(protocol_path)
    control_audit_path = args.control_audit.resolve(strict=True)
    control_audit = json.loads(control_audit_path.read_text(encoding="utf-8"))
    if (
        control_audit.get("status") != "PASS"
        or control_audit.get("protocol_sha256") != protocol_sha
        or control_audit.get("control_cells") != 100
    ):
        raise RuntimeError("control compatibility audit is not PASS")
    controls = {
        (
            int(row["seed"]), row["arm"], int(row["budget_kimg"]),
            int(row["nfe"]), metric,
        ): float(row[metric])
        for row in control_audit["controls"] for metric in METRICS
    }
    new = load_new(args.new_receipts.resolve(strict=True), protocol_sha)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    trajectories = []
    contrasts = []
    for seed in SEEDS:
        for nfe in NFES:
            for metric in METRICS:
                for budget in BUDGETS:
                    aa = controls[(seed, "A", budget, nfe, metric)]
                    bb = controls[(seed, "B", budget, nfe, metric)]
                    ab = aa if budget == 512 else new[(seed, "A_to_B", budget, nfe)][metric]
                    ba = bb if budget == 512 else new[(seed, "B_to_A", budget, nfe)][metric]
                    trajectories.append({
                        "seed": seed, "nfe": nfe, "metric": metric,
                        "budget_kimg": budget, "AA": aa, "AB": ab,
                        "BA": ba, "BB": bb,
                    })
                    contrasts.append({
                        "seed": seed, "nfe": nfe, "metric": metric,
                        "budget_kimg": budget, "S_A": ab - aa,
                        "S_B": bb - ba, "H_A": ba - aa,
                        "H_B": bb - ab, "I_switch": bb - ba - ab + aa,
                    })
    if len(trajectories) != 100 or len(contrasts) != 100:
        raise RuntimeError("seed-level tables are incomplete")

    aulc = []
    for seed in SEEDS:
        for nfe in NFES:
            for metric in METRICS:
                rows = sorted(
                    (row for row in trajectories if row["seed"] == seed
                     and row["nfe"] == nfe and row["metric"] == metric),
                    key=lambda row: row["budget_kimg"],
                )
                values = {
                    name: common.normalized_aulc([
                        (row["budget_kimg"], row[name]) for row in rows
                    ])
                    for name in TRAJECTORIES
                }
                aulc.append({
                    "seed": seed, "nfe": nfe, "metric": metric, **values,
                    "S_A": values["AB"] - values["AA"],
                    "S_B": values["BB"] - values["BA"],
                    "H_A": values["BA"] - values["AA"],
                    "H_B": values["BB"] - values["AB"],
                    "I_switch": (
                        values["BB"] - values["BA"]
                        - values["AB"] + values["AA"]
                    ),
                })
    summaries = []
    for nfe in NFES:
        for metric in METRICS:
            for budget in BUDGETS:
                rows = [
                    row for row in contrasts if row["nfe"] == nfe
                    and row["metric"] == metric and row["budget_kimg"] == budget
                ]
                for name in ("S_A", "S_B", "H_A", "H_B", "I_switch"):
                    values = [row[name] for row in rows]
                    summaries.append({
                        "nfe": nfe, "metric": metric,
                        "budget_kimg": budget, "contrast": name,
                        "mean": statistics.mean(values),
                        "median": statistics.median(values), "n_seeds": 5,
                    })
    common.write_csv(output / "per_seed_trajectories.csv", trajectories)
    common.write_csv(output / "per_seed_contrasts.csv", contrasts)
    common.write_csv(output / "per_seed_aulc.csv", aulc)
    common.write_csv(output / "contrast_summaries.csv", summaries)
    common.SEEDS = SEEDS
    plots = common.create_plots(output, trajectories, contrasts)
    audit = {
        "schema": "ect.q256.schedule-switch-seed3-7-analysis/v1",
        "status": "PASS", "protocol_sha256": protocol_sha,
        "control_audit_sha256": sha256_file(control_audit_path),
        "new_evaluation_jobs": 80, "control_cells": 100,
        "trajectory_rows": 100, "contrast_rows": 100,
        "aulc_rows": len(aulc), "statistical_unit": "training seed",
        "plots": plots,
        "claim_boundary": (
            "descriptive availability-selected five-seed conditional "
            "post-switch evidence; no global causal percentage"
        ),
    }
    reproducibility.atomic_json_dump(audit, output / "analysis_audit.json", overwrite=False)
    report = [
        "# q256 seed3-7 crossed schedule-switch results", "",
        "Status: **PASS**", "", f"Protocol SHA256: `{protocol_sha}`", "",
        "All five training seeds, four trajectories, five budgets, two NFEs, "
        "and both KID/FID metrics are reported in the adjacent CSV files.", "",
        "Means and medians are descriptive summaries over five training seeds. "
        "Budget × NFE cells are not independent samples.", "",
        "Claim boundary: availability-selected seed3-7 conditional intervention "
        "evidence only; no universal schedule ranking or causal percentage.", "",
    ]
    with (output / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report)); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "jobs": 80, "plots": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
