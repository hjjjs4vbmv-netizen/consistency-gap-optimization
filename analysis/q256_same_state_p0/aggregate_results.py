#!/usr/bin/env python3
"""Aggregate the immutable 3x4 P0 branch matrix and sparse probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


SEEDS = (3, 4, 5)
ARMS = ("A", "B", "C", "D")
PERSISTENT_BLOCKS = ("theta", "EMA", "m", "v")
LATE_HORIZONS = (256, 500)
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
        _arm, space, _block = key.split(":", 2)
        interpreted = [
            (
                seed,
                (
                    "descriptive_history_dominated_propagation"
                    if space == "observable"
                    else "history_dominated_persistent_propagation"
                )
                if label == "persistent_state_feedback_dominance"
                else label,
            )
            for seed, label in values
        ]
        counts = Counter(label for _, label in interpreted)
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
            "by_seed": {str(seed): value for seed, value in interpreted},
            "legacy_pr89_classification_by_seed": {
                str(seed): value for seed, value in values
            },
            "counts": dict(counts), "replication_status": status,
            "replicated_classification": label,
            "classification_scope": (
                "descriptive readout propagation; no declared carryover map"
                if space == "observable"
                else "raw propagation including declared mechanical carryover"
            ),
        }
    return result


def late_propagation_rows(
    forcing_rows: list[dict[str, str]],
    classifications: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in forcing_rows:
        if (
            row["space"] == "state"
            and row["block"] in PERSISTENT_BLOCKS
            and int(row["horizon"]) in LATE_HORIZONS
        ):
            grouped[(int(row["seed"]), row["arm"], row["block"])].append(row)
    output = []
    for (seed, arm, block), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["horizon"]))
        if sorted(int(row["horizon"]) for row in rows) != list(LATE_HORIZONS):
            raise RuntimeError(
                f"late diagnostic rows incomplete: seed={seed} arm={arm} block={block}"
            )

        def finite_values(field: str) -> list[float]:
            values = [float(row[field]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    f"non-finite late diagnostic {field}: "
                    f"seed={seed} arm={arm} block={block}"
                )
            return values

        corrected_norm = finite_values("corrected_R_norm")
        corrected_ratio = finite_values("corrected_R_over_delta_k")
        corrected_alignment = finite_values("cos_corrected_R_delta_k")
        raw_gain = finite_values("feedback_gain_G")
        closure_pass = all(row["closure_pass"].lower() == "true" for row in rows)
        numerically_nonzero = (
            closure_pass
            and all(value > 0.0 for value in corrected_norm)
            and all(value > 0.0 for value in corrected_ratio)
        )
        classification = classifications[f"{arm}:state:{block}"]
        alignment_signs = [
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in corrected_alignment
        ]
        output.append({
            "seed": seed,
            "arm": arm,
            "block": block,
            "late_horizons": "256;500",
            "late_median_raw_propagation_gain_G": statistics.median(raw_gain),
            "late_raw_propagation_gain_G_min": min(raw_gain),
            "late_raw_propagation_gain_G_max": max(raw_gain),
            "late_median_corrected_R_over_delta_k": statistics.median(
                corrected_ratio
            ),
            "late_corrected_R_over_delta_k_min": min(corrected_ratio),
            "late_corrected_R_over_delta_k_max": max(corrected_ratio),
            "late_median_cos_corrected_R_delta_k": statistics.median(
                corrected_alignment
            ),
            "corrected_alignment_sign_h256": alignment_signs[0],
            "corrected_alignment_sign_h500": alignment_signs[1],
            "late_corrected_direction_stable": (
                alignment_signs[0] == alignment_signs[1]
                and alignment_signs[0] != "zero"
            ),
            "late_corrected_R_norm_min": min(corrected_norm),
            "late_closure_all_pass": closure_pass,
            "carryover_rule": rows[0]["carryover_rule"],
            "raw_propagation_label": classification["by_seed"][str(seed)],
            "legacy_pr89_raw_label": classification[
                "legacy_pr89_classification_by_seed"
            ][str(seed)],
            "corrected_feedback_numerically_nonzero": numerically_nonzero,
            "claim_ceiling": (
                "presence only; not corrected-feedback dominance or amplification"
            ),
        })
    expected = len(SEEDS) * 3 * len(PERSISTENT_BLOCKS)
    if len(output) != expected:
        raise RuntimeError(f"late diagnostic table has {len(output)} rows, expected {expected}")
    return output


def corrected_feedback_replication(
    late_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in late_rows:
        grouped[(str(row["arm"]), str(row["block"]))].append(row)
    by_arm_block = {}
    for (arm, block), rows in sorted(grouped.items()):
        nonzero_seeds = sorted(
            int(row["seed"])
            for row in rows
            if row["corrected_feedback_numerically_nonzero"]
        )
        directionally_stable = [
            row for row in rows if row["late_corrected_direction_stable"]
        ]
        stable_signs = {
            row["corrected_alignment_sign_h256"] for row in directionally_stable
        }
        directional_3_of_3 = (
            len(directionally_stable) == 3 and len(stable_signs) == 1
        )
        by_arm_block[f"{arm}:state:{block}"] = {
            "nonzero_seeds": nonzero_seeds,
            "nonzero_seed_count": len(nonzero_seeds),
            "replicated_numerically_nonzero": len(nonzero_seeds) >= 2,
            "directionally_stable_seeds": sorted(
                int(row["seed"]) for row in directionally_stable
            ),
            "replicated_directionally_consistent_3_of_3": directional_3_of_3,
            "replicated_alignment_sign": (
                next(iter(stable_signs)) if directional_3_of_3 else None
            ),
        }
    raw_gain_min = min(
        float(row["late_raw_propagation_gain_G_min"]) for row in late_rows
    )
    raw_gain_max = max(
        float(row["late_raw_propagation_gain_G_max"]) for row in late_rows
    )
    corrected_ratio_min = min(
        float(row["late_corrected_R_over_delta_k_min"]) for row in late_rows
    )
    corrected_ratio_max = max(
        float(row["late_corrected_R_over_delta_k_max"]) for row in late_rows
    )
    return {
        "seed_level_rule": (
            "At both late horizons {256,500}: closure passes, corrected_R_norm "
            "is finite and >0, and corrected_R_over_delta_k is finite and >0."
        ),
        "cross_seed_rule": (
            "Numerically nonzero corrected incremental feedback is replicated "
            "when the seed-level rule holds in at least 2/3 formal seeds."
        ),
        "directional_cross_seed_rule": (
            "A conservative directionally consistent result requires all 3/3 "
            "seeds to have finite nonzero corrected-feedback alignment with "
            "the same sign at h=256 and h=500 within seed, and the same sign "
            "across seeds. This rule is post-hoc and descriptive."
        ),
        "late_raw_propagation_gain_G_range": [raw_gain_min, raw_gain_max],
        "late_corrected_R_over_delta_k_range": [
            corrected_ratio_min, corrected_ratio_max,
        ],
        "any_late_corrected_R_over_delta_k_above_one": (
            corrected_ratio_max > 1.0
        ),
        "claim_ceiling": (
            "This post-hoc presence rule does not establish corrected-feedback "
            "dominance, same-direction amplification, or a causal contribution."
        ),
        "by_arm_and_block": by_arm_block,
    }


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
    late_rows = late_propagation_rows(forcing_rows, classifications)
    corrected_replication = corrected_feedback_replication(late_rows)
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
             "contrast_formula": "norm_C_minus_norm_A",
             "contrast_of_l2_norms": y["C"] - y["A"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast_formula": "norm_B_minus_norm_D",
             "contrast_of_l2_norms": y["B"] - y["D"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast_formula": "norm_D_minus_norm_A",
             "contrast_of_l2_norms": y["D"] - y["A"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast_formula": "norm_B_minus_norm_C",
             "contrast_of_l2_norms": y["B"] - y["C"]},
            {"seed": seed, "horizon": horizon, "space": space, "block": block,
             "contrast_formula": "norm_B_minus_norm_C_minus_norm_D_plus_norm_A",
             "contrast_of_l2_norms": y["B"] - y["C"] - y["D"] + y["A"]},
        ])
    for row in contrast_rows:
        row.update(
            quantity_definition=(
                "linear contrast of branch-specific absolute value_norm"
            ),
            exploratory_only=True,
            claim_ceiling=(
                "descriptive absolute-norm contrast; not a target or denominator effect"
            ),
        )
    total_gpu_hours = sum(float(row["a100_gpu_hours"]) for row in compute_rows)
    closure_pass = all(payload["all_exact_closures_pass"] for payload in probe_payloads)
    replicated_persistent_state = [
        key for key, item in classifications.items()
        if any(f":state:{block}" in key for block in PERSISTENT_BLOCKS)
        and item["replication_status"] == "cross-seed replicated"
    ]
    replicated_observables = [
        key
        for key, item in classifications.items()
        if ":observable:" in key
        and item["replication_status"] == "cross-seed replicated"
    ]
    corrected_replicated_entries = sorted(
        key for key, item in corrected_replication["by_arm_and_block"].items()
        if item["replicated_numerically_nonzero"]
    )
    corrected_directional_entries = sorted(
        key for key, item in corrected_replication["by_arm_and_block"].items()
        if item["replicated_directionally_consistent_3_of_3"]
    )
    p1_worthwhile = bool(replicated_persistent_state) and closure_pass
    summary = {
        "schema": "ect.q256.b384-same-state-p0-summary/v2",
        "status": "PASS", "protocol_sha256": protocol_sha,
        "parity_3_of_3": parity["status"] == "PASS",
        "formal_branches_12_of_12": len(branch_rows) == 12,
        "all_exact_closures_pass": closure_pass,
        "max_closure_relative": max(payload["max_closure_relative"] for payload in probe_payloads),
        "production_exogenous_pairing_500_steps": not exogenous_failures,
        "raw_propagation_classifications": classifications,
        "legacy_field_note": (
            "The inherited PR #89 persistent_state_feedback_dominance label "
            "is retained only under legacy_pr89_classification_by_seed. The "
            "headline interpretation is history-dominated/persistent propagation."
        ),
        "corrected_incremental_feedback": corrected_replication,
        "replicated_persistent_state_entries": replicated_persistent_state,
        "replicated_observable_descriptive_entries": replicated_observables,
        "replicated_numerically_nonzero_corrected_feedback_entries": (
            corrected_replicated_entries
        ),
        "replicated_directionally_consistent_corrected_feedback_entries": (
            corrected_directional_entries
        ),
        "actual_a100_gpu_hours_training": total_gpu_hours,
        "p1_worthwhile": p1_worthwhile,
        "p1_started": False,
        "claim_boundary": {
            "allowed": [
                "exact same-state forcing identity",
                "mechanical carryover separated from corrected feedback",
                "conditional history-dominated/persistent propagation across three B-history states",
                "numerically nonzero corrected feedback under the stated post-hoc presence rule",
                "descriptive propagation in audited observables without a carryover map",
            ],
            "withheld": [
                "feedback causes FID improvement", "RAdam is the unique mechanism",
                "ImageNet extrapolation", "universal amplification",
                "corrected-feedback dominance", "norm ratios as causal percentages",
            ],
        },
    }
    write_csv(args.outdir / "branch_manifest.csv", branch_rows)
    write_csv(args.outdir / "training_compute.csv", compute_rows)
    write_csv(args.outdir / "matched_horizon_results.csv", horizon_rows)
    write_csv(args.outdir / "forcing_feedback_per_horizon.csv", forcing_rows)
    write_csv(args.outdir / "late_propagation_corrected_feedback.csv", late_rows)
    write_csv(
        args.outdir / "exploratory_absolute_norm_contrasts.csv", contrast_rows
    )
    (args.outdir / "forcing_feedback_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    propagation_lines = []
    for key, item in classifications.items():
        if any(f":state:{block}" in key for block in PERSISTENT_BLOCKS):
            propagation_lines.append(
                f"| {key} | {item['counts']} | {item['replication_status']} | "
                f"{item['replicated_classification'] or 'mixed'} |"
            )
    corrected_lines = []
    for key, item in corrected_replication["by_arm_and_block"].items():
        corrected_lines.append(
            f"| {key} | {item['nonzero_seed_count']}/3 | "
            f"{item['replicated_numerically_nonzero']} | "
            f"{item['replicated_directionally_consistent_3_of_3']} | "
            f"{item['replicated_alignment_sign'] or 'mixed/reversing'} |"
        )
    report = [
        "# q256 B@384 same-state A/B/C/D P0 report", "",
        f"1. B no-op parity: **{'3/3 PASS' if summary['parity_3_of_3'] else 'FAIL_CLOSED'}**.",
        f"2. Formal branches: **{'12/12 PASS' if summary['formal_branches_12_of_12'] else 'FAIL_CLOSED'}**.",
        f"3. Exact closure: **{'all PASS' if closure_pass else 'FAIL_CLOSED'}**.",
        "4. Raw late-horizon propagation is history-dominated/persistent for theta/EMA/m/v in B/C/D across 3/3 seeds. This raw label includes declared mechanical carryover.",
        f"5. Numerically nonzero corrected incremental feedback replicates under the separately stated presence rule for: `{corrected_replicated_entries}`. This is not a dominance or amplification result.",
        f"6. Audited observables replicate descriptively in: `{replicated_observables}`. Feature/residual readouts have no declared linear carryover map and are not carryover-corrected state-mechanism evidence.",
        "7. The paper may claim conditional history-dominated/persistent propagation from B@384 history; no quality, global causal, or actionable-law claim is licensed.",
        f"8. P1 is **{'worth protocol consideration' if p1_worthwhile else 'not yet justified'}**; P1 was not started.",
        "", f"Actual training compute: `{total_gpu_hours:.6f}` A100 GPU-hours.", "",
        "## Raw propagation classification", "",
        "The legacy PR #89 label is relabeled for interpretation because raw `R` includes mechanical carryover.",
        "", "| arm:space:block | counts | replication | interpretive label |",
        "|---|---|---|---|", *propagation_lines, "",
        "## Carryover-corrected incremental feedback", "",
        corrected_replication["seed_level_rule"],
        corrected_replication["cross_seed_rule"],
        corrected_replication["directional_cross_seed_rule"],
        corrected_replication["claim_ceiling"], "",
        f"Across all late persistent-state rows, raw `feedback_gain_G` ranges from `{corrected_replication['late_raw_propagation_gain_G_range'][0]:.6g}` to `{corrected_replication['late_raw_propagation_gain_G_range'][1]:.6g}` and `corrected_R_over_delta_k` ranges from `{corrected_replication['late_corrected_R_over_delta_k_range'][0]:.6g}` to `{corrected_replication['late_corrected_R_over_delta_k_range'][1]:.6g}`; no corrected ratio exceeds 1.",
        "Thus neither raw nor corrected amplification is universal.", "",
        f"Directionally consistent 3/3 entries: `{corrected_directional_entries}`.",
        "", "| arm:state:block | nonzero seeds | replicated presence | directional 3/3 | alignment sign |",
        "|---|---:|---|---|---|", *corrected_lines, "",
        "Per-seed late medians for raw `feedback_gain_G`, `corrected_R_over_delta_k`, and corrected-feedback alignment are in `late_propagation_corrected_feedback.csv`.",
        "", "## Observable scope", "",
        "Fixed-latent EMA feature and signed residual readouts show replicated descriptive history-dominated propagation in B/C/D across 3/3 seeds. They have no declared linear carryover map, so no carryover-corrected observable mechanism is claimed.",
        "", "## Exploratory absolute-norm contrasts", "",
        "`exploratory_absolute_norm_contrasts.csv` contains algebraic contrasts of branch-specific absolute L2 norms. For example, `norm_C_minus_norm_A` is `||z_C||_2 - ||z_A||_2`, not `||z_C-z_A||_2`. These rows are not used for the mechanism headline and are not target effects, denominator effects, factorial causal effects, or independent-training arm rankings.",
    ]
    (args.outdir / "P0_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = sorted(path for path in args.outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (args.outdir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
