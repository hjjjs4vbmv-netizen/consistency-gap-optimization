#!/usr/bin/env python3
"""Aggregate the immutable 3x4 P0 branch matrix and sparse probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


SEEDS = (3, 4, 5)
ARMS = ("A", "B", "C", "D")
EXOGENOUS = (
    "batch_sha256", "t_sha256", "base_r_sha256", "input_noise_sha256",
    "dropout_rng_sha256", "augmentation_rng_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classification_summary(probe_payloads: list[dict]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for payload in probe_payloads:
        for key, item in payload["mechanism_by_arm_and_block"].items():
            grouped[key].append((int(payload["seed"]), item["classification"]))
    result = {}
    for key, values in sorted(grouped.items()):
        counts = Counter(label for _, label in values)
        replicated = [
            (label, count) for label, count in counts.items()
            if label != "mixed_or_inconclusive" and count >= 2
        ]
        if len(replicated) == 1:
            status = "cross-seed replicated"
            label = replicated[0][0]
        else:
            status = "mixed"
            label = None
        result[key] = {
            "by_seed": {str(seed): value for seed, value in values},
            "counts": dict(counts), "replication_status": status,
            "replicated_classification": label,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parity-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha256_file(args.protocol)
    parity = json.loads(args.parity_json.read_text(encoding="utf-8"))
    branch_rows = []
    compute_rows = []
    exogenous_failures = []
    telemetry_by_seed_arm = {}
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = args.output_root / "runs" / f"seed{seed}" / f"B384_to_{arm}"
            receipt = json.loads((run_dir / "branch_receipt.json").read_text(encoding="utf-8"))
            if receipt["status"] != "PASS" or receipt["protocol_sha256"] != protocol_sha:
                raise RuntimeError(f"branch receipt failed or protocol drifted: seed{seed} {arm}")
            env = dict(
                line.split("=", 1) for line in
                (run_dir / "launch_environment.txt").read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            source = torch.load(receipt["source_state"], map_location="cpu", weights_only=False)
            final_kimg = 512 if arm == "B" else 448
            final_path = run_dir / f"training-state-kimg{final_kimg:06d}.pt"
            final = torch.load(final_path, map_location="cpu", weights_only=False)
            seconds = float(final["elapsed_sec"]) - float(source.get("elapsed_sec", 0.0))
            branch_rows.append({
                "seed": seed, "origin_arm": "B", "continuation_arm": arm,
                "branch_label": f"B384_to_{arm}", "status": receipt["status"],
                "source_sha256": receipt["source_sha256_before_and_after"],
                "formal_state_448_sha256": receipt["formal_448_state_sha256"],
                "formal_ema_448_sha256": receipt["formal_448_snapshot_sha256"],
                "engineering_final_kimg": final_kimg,
                "engineering_final_state_sha256": receipt["final_state_sha256"],
                "gpu_uuid": env["gpu_uuid"], "protocol_sha256": protocol_sha,
            })
            compute_rows.append({
                "seed": seed, "branch_label": f"B384_to_{arm}",
                "attempted_iterations": final_kimg * 1000 // 128 - 3000,
                "additional_kimg": final_kimg - 384,
                "elapsed_seconds": f"{seconds:.6f}",
                "a100_gpu_hours": f"{seconds / 3600:.9f}",
                "gpu_uuid": env["gpu_uuid"],
                "scope": "parity+formal-control" if arm == "B" else "formal",
            })
            telemetry = [
                row for row in read_csv(run_dir / "matched_training_telemetry_v1.csv")
                if 3001 <= int(row["attempted_iteration"]) <= 3500
            ]
            if len(telemetry) != 500:
                raise RuntimeError(f"formal telemetry length mismatch: seed{seed} {arm}")
            telemetry_by_seed_arm[(seed, arm)] = telemetry
    if len(branch_rows) != 12 or len({(row["seed"], row["continuation_arm"]) for row in branch_rows}) != 12:
        raise RuntimeError("formal branch matrix is not exactly 12 unique branches")
    for seed in SEEDS:
        for index in range(500):
            for field in EXOGENOUS:
                values = {
                    telemetry_by_seed_arm[(seed, arm)][index][field]
                    for arm in ARMS
                }
                if len(values) != 1:
                    exogenous_failures.append({
                        "seed": seed, "completed_step": index + 1, "field": field,
                    })
    if exogenous_failures:
        raise RuntimeError(f"production exogenous pairing failed: {exogenous_failures[:3]}")

    forcing_rows = []
    horizon_rows = []
    probe_payloads = []
    for seed in SEEDS:
        forcing_rows.extend(read_csv(args.probe_root / f"seed{seed}-forcing.csv"))
        horizon_rows.extend(read_csv(args.probe_root / f"seed{seed}-horizons.csv"))
        payload = json.loads((args.probe_root / f"seed{seed}-summary.json").read_text(encoding="utf-8"))
        if payload["status"] != "PASS":
            raise RuntimeError(f"sparse probe failed for seed{seed}")
        probe_payloads.append(payload)
    classifications = classification_summary(probe_payloads)
    indexed = {
        (int(row["seed"]), int(row["horizon"]), row["space"], row["block"], row["arm"]): float(row["value_norm"])
        for row in horizon_rows
    }
    contrast_rows = []
    groups = sorted({key[:4] for key in indexed})
    for seed, horizon, space, block in groups:
        y = {arm: indexed[(seed, horizon, space, block, arm)] for arm in ARMS}
        contrast_rows.extend([
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast": "target_effect_denominator_1p0", "value": y["C"] - y["A"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast": "target_effect_denominator_1p1", "value": y["B"] - y["D"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast": "denominator_effect_target_1p0", "value": y["D"] - y["A"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast": "denominator_effect_target_1p1", "value": y["B"] - y["C"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast": "conditional_interaction", "value": y["B"] - y["C"] - y["D"] + y["A"]},
        ])
    total_gpu_hours = sum(float(row["a100_gpu_hours"]) for row in compute_rows)
    closure_pass = all(payload["all_exact_closures_pass"] for payload in probe_payloads)
    replicated_core = [
        key for key, item in classifications.items()
        if any(f":state:{block}" in key for block in ("theta", "EMA", "m", "v"))
        and item["replication_status"] == "cross-seed replicated"
    ]
    residual_feature_mixed = all(
        item["replication_status"] == "mixed"
        for key, item in classifications.items()
        if ":observable:" in key
    )
    p1_worthwhile = bool(replicated_core) and closure_pass
    summary = {
        "schema": "ect.q256.b384-same-state-p0-summary/v1",
        "status": "PASS", "protocol_sha256": protocol_sha,
        "parity_3_of_3": parity["status"] == "PASS",
        "formal_branches_12_of_12": len(branch_rows) == 12,
        "all_exact_closures_pass": closure_pass,
        "max_closure_relative": max(payload["max_closure_relative"] for payload in probe_payloads),
        "production_exogenous_pairing_500_steps": not exogenous_failures,
        "cross_seed_classifications": classifications,
        "replicated_core_entries": replicated_core,
        "residual_features_mixed": residual_feature_mixed,
        "actual_a100_gpu_hours_training": total_gpu_hours,
        "p1_worthwhile": p1_worthwhile,
        "p1_started": False,
        "claim_boundary": {
            "allowed": [
                "exact same-state forcing identity",
                "mechanical carryover separated from corrected feedback",
                "conditional replication across three B-history states",
            ],
            "withheld": [
                "feedback causes FID improvement", "RAdam is the unique mechanism",
                "ImageNet extrapolation", "universal amplification",
                "norm ratios as causal percentages",
            ],
        },
    }
    write_csv(args.outdir / "branch_manifest.csv", branch_rows)
    write_csv(args.outdir / "training_compute.csv", compute_rows)
    write_csv(args.outdir / "matched_horizon_results.csv", horizon_rows)
    write_csv(args.outdir / "forcing_feedback_per_horizon.csv", forcing_rows)
    write_csv(args.outdir / "factorial_contrasts.csv", contrast_rows)
    (args.outdir / "forcing_feedback_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    core_lines = []
    for key, item in classifications.items():
        if any(f":state:{block}" in key for block in ("theta", "EMA", "m", "v")):
            core_lines.append(
                f"| {key} | {item['counts']} | {item['replication_status']} | "
                f"{item['replicated_classification'] or 'mixed'} |"
            )
    report = [
        "# q256 B@384 same-state A/B/C/D P0 report", "",
        f"1. B no-op parity: **{'3/3 PASS' if summary['parity_3_of_3'] else 'FAIL_CLOSED'}**.",
        f"2. Formal branches: **{'12/12 PASS' if summary['formal_branches_12_of_12'] else 'FAIL_CLOSED'}**.",
        f"3. Exact closure: **{'all PASS' if closure_pass else 'FAIL_CLOSED'}**.",
        "4. Late-horizon theta/EMA/m/v classifications are tabulated below.",
        f"5. Cross-seed replicated core entries: `{replicated_core}`.",
        f"6. Residual/features remain mixed: **{residual_feature_mixed}**.",
        "7. The paper may claim conditional same-state persistence/feedback replication only where the 2/3 rule passes; no quality or global causal claim is licensed.",
        f"8. P1 is **{'worth protocol consideration' if p1_worthwhile else 'not yet justified'}**; P1 was not started.",
        "", f"Actual training compute: `{total_gpu_hours:.6f}` A100 GPU-hours.", "",
        "| arm:space:block | counts | replication | label |",
        "|---|---|---|---|", *core_lines, "",
        "All factorial contrasts are conditional on a B@384 history and are not independent-training arm rankings.",
    ]
    (args.outdir / "P0_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = sorted(path for path in args.outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (args.outdir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
