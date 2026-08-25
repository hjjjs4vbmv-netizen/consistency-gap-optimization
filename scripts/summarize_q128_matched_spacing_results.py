#!/usr/bin/env python3
"""Build the frozen q128 matched-spacing summaries from the audited CSV export."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ARMS = ["A", "Bsame", "Bmatch", "Cmatch", "Dmatch"]
SEEDS = [3, 4, 5]
BUDGETS = [256, 384, 512, 640, 768, 896, 1024]
NFES = [1, 2]
PAIR_CONTRASTS = [
    ("Bmatch-Bsame", "Bmatch", "Bsame"),
    ("Bmatch-A", "Bmatch", "A"),
    ("Bsame-A", "Bsame", "A"),
    ("Cmatch-A", "Cmatch", "A"),
    ("Dmatch-A", "Dmatch", "A"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def read_rows(path):
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["seed"]), row["arm"], int(row["kimg"]), int(row["nfe"])
            )
            if key in rows:
                raise RuntimeError("duplicate row: {}".format(key))
            rows[key] = {
                "fid": float(row["fid50k_full"]),
                "kid": float(row["kid50k_full"]),
                "status": row["status"],
            }
    expected = {
        (seed, arm, budget, nfe)
        for seed in SEEDS
        for arm in ARMS
        for budget in BUDGETS
        for nfe in NFES
    }
    if set(rows) != expected or any(row["status"] != "SEALED_PASS" for row in rows.values()):
        raise RuntimeError("input is not the complete 210-job SEALED_PASS matrix")
    return rows


def aulc(rows, seed, arm, nfe):
    log_fid = [math.log(rows[(seed, arm, budget, nfe)]["fid"]) for budget in BUDGETS]
    area = sum(
        (BUDGETS[index + 1] - BUDGETS[index])
        * (log_fid[index] + log_fid[index + 1])
        / 2.0
        for index in range(len(BUDGETS) - 1)
    )
    return area / float(BUDGETS[-1] - BUDGETS[0])


def summary(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "negative": sum(value < 0 for value in values),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
    }


def ttq(rows, seed, arm, nfe):
    values = [rows[(seed, arm, budget, nfe)]["fid"] for budget in BUDGETS]
    first = next((budget for budget, value in zip(BUDGETS, values) if value <= 10), None)
    sustained = None
    for index in range(len(BUDGETS) - 1):
        if (
            values[index] <= 10
            and values[index + 1] <= 10
            and all(value <= 10 for value in values[index:])
        ):
            sustained = BUDGETS[index]
            break
    return first, sustained


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return ("{:0." + str(digits) + "f}").format(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    rows = read_rows(args.input)
    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "PASS" or audit.get("jobs") != 210:
        raise RuntimeError("audit did not pass")
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_seed_aulc = []
    aulc_values = {}
    for seed in SEEDS:
        for arm in ARMS:
            for nfe in NFES:
                value = aulc(rows, seed, arm, nfe)
                aulc_values[(seed, arm, nfe)] = value
                per_seed_aulc.append(
                    {"seed": seed, "arm": arm, "nfe": nfe, "aulc_log_fid": value}
                )
    write_csv(
        args.outdir / "per_seed_aulc.csv",
        ["seed", "arm", "nfe", "aulc_log_fid"],
        per_seed_aulc,
    )

    arm_summary = []
    for arm in ARMS:
        for nfe in NFES:
            aulc_summary = summary([aulc_values[(seed, arm, nfe)] for seed in SEEDS])
            fid_summary = summary(
                [rows[(seed, arm, 1024, nfe)]["fid"] for seed in SEEDS]
            )
            kid_summary = summary(
                [rows[(seed, arm, 1024, nfe)]["kid"] for seed in SEEDS]
            )
            arm_summary.append(
                {
                    "arm": arm,
                    "nfe": nfe,
                    "aulc_mean": aulc_summary["mean"],
                    "aulc_median": aulc_summary["median"],
                    "aulc_min": aulc_summary["min"],
                    "aulc_max": aulc_summary["max"],
                    "fid1024_mean": fid_summary["mean"],
                    "fid1024_median": fid_summary["median"],
                    "fid1024_min": fid_summary["min"],
                    "fid1024_max": fid_summary["max"],
                    "kid1024_mean": kid_summary["mean"],
                    "kid1024_median": kid_summary["median"],
                    "kid1024_min": kid_summary["min"],
                    "kid1024_max": kid_summary["max"],
                }
            )
    write_csv(args.outdir / "arm_summary.csv", list(arm_summary[0]), arm_summary)

    contrast_rows = []
    contrast_summary = []
    for nfe in NFES:
        definitions = list(PAIR_CONTRASTS) + [("interaction", None, None)]
        for name, left, right in definitions:
            values = []
            for seed in SEEDS:
                if name == "interaction":
                    value = (
                        aulc_values[(seed, "Bmatch", nfe)]
                        - aulc_values[(seed, "Cmatch", nfe)]
                        - aulc_values[(seed, "Dmatch", nfe)]
                        + aulc_values[(seed, "A", nfe)]
                    )
                else:
                    value = aulc_values[(seed, left, nfe)] - aulc_values[(seed, right, nfe)]
                values.append(value)
                contrast_rows.append(
                    {"contrast": name, "nfe": nfe, "seed": seed, "delta_aulc": value}
                )
            stats = summary(values)
            contrast_summary.append(
                {
                    "contrast": name,
                    "nfe": nfe,
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "negative_seeds": stats["negative"],
                    "positive_seeds": stats["positive"],
                }
            )
    write_csv(
        args.outdir / "per_seed_aulc_contrasts.csv",
        ["contrast", "nfe", "seed", "delta_aulc"],
        contrast_rows,
    )
    write_csv(
        args.outdir / "contrast_summary.csv",
        list(contrast_summary[0]),
        contrast_summary,
    )

    direction_rows = []
    for name, left, right in PAIR_CONTRASTS:
        fid_deltas = []
        kid_deltas = []
        for seed in SEEDS:
            for budget in BUDGETS:
                for nfe in NFES:
                    fid_deltas.append(
                        rows[(seed, left, budget, nfe)]["fid"]
                        - rows[(seed, right, budget, nfe)]["fid"]
                    )
                    kid_deltas.append(
                        rows[(seed, left, budget, nfe)]["kid"]
                        - rows[(seed, right, budget, nfe)]["kid"]
                    )
        direction_rows.append(
            {
                "contrast": name,
                "cells": len(fid_deltas),
                "fid_negative": sum(value < 0 for value in fid_deltas),
                "kid_negative": sum(value < 0 for value in kid_deltas),
                "fid_kid_sign_agree": sum(
                    (fid < 0) == (kid < 0)
                    for fid, kid in zip(fid_deltas, kid_deltas)
                ),
            }
        )
    write_csv(
        args.outdir / "direction_consistency.csv",
        list(direction_rows[0]),
        direction_rows,
    )

    trajectory_rows = []
    for seed in SEEDS:
        for budget in BUDGETS:
            row = {"seed": seed, "kimg": budget}
            for nfe in NFES:
                for arm in ("A", "Bsame", "Bmatch"):
                    row["{}_nfe{}_fid".format(arm, nfe)] = rows[
                        (seed, arm, budget, nfe)
                    ]["fid"]
                    row["{}_nfe{}_kid".format(arm, nfe)] = rows[
                        (seed, arm, budget, nfe)
                    ]["kid"]
            trajectory_rows.append(row)
    write_csv(
        args.outdir / "a_bsame_bmatch_trajectories.csv",
        list(trajectory_rows[0]),
        trajectory_rows,
    )

    ttq_rows = []
    for arm in ("A", "Bsame", "Bmatch"):
        for seed in SEEDS:
            for nfe in NFES:
                first, sustained = ttq(rows, seed, arm, nfe)
                ttq_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "nfe": nfe,
                        "first_ttq_fid_le_10": first if first is not None else ">1024",
                        "sustained_ttq_fid_le_10": (
                            sustained if sustained is not None else ">1024"
                        ),
                        "analysis_status": "exploratory_not_preregistered_for_q128",
                    }
                )
    write_csv(args.outdir / "ttq_exploratory.csv", list(ttq_rows[0]), ttq_rows)

    validation = {
        "schema": "ect.q128-matched-spacing-validation/v1",
        "verification_status": "ANALYZED",
        "overall_confidence": "CAUTION",
        "aulc_evidence_status": "post_unblind_deterministic_descriptive",
        "aulc_preregistered": False,
        "jobs": 210,
        "metric_values": 420,
        "fallacy_scan_coverage": "11/11",
        "reproducibility": "not rerun; sealed artifact and matrix audit only",
        "limitations": [
            "Only three training seeds; paired outcomes are descriptive.",
            "AULC and its contrasts were defined after metric unblinding; they are deterministic descriptive summaries, not preregistered primary outcomes.",
            "The Bmatch-Bsame AULC direction is seed-sensitive.",
            "q128 TTQ was not preregistered and is exploratory.",
            "Redundant attempts were resolved by frozen partition ownership, never quality.",
        ],
    }
    (args.outdir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )

    arm_lookup = {(row["arm"], row["nfe"]): row for row in arm_summary}
    contrast_lookup = {
        (row["contrast"], row["nfe"]): row for row in contrast_summary
    }
    report = [
        "# q128 matched-spacing five-arm results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-24",
        "- Verification Status: ANALYZED",
        "- Version Label: q128_matched_spacing_results_v1",
        "",
        "## Audit status",
        "",
        "- 210/210 unique `SEALED_PASS` jobs and 420/420 metric values.",
        "- FP32, 50,000 samples, sample seeds 0-49999, metric seed 20260730.",
        "- NFE2 uses `mid_t=0.821`; invalidated/pre-reuse directories are excluded.",
        "- Preassigned server/data partitions override redundant attempts without quality selection.",
        "- The deployed evaluator is bound by full source SHA256 in `audit.json`.",
        "",
        "## Evidence status and blindness chronology",
        "",
        "Historical canonical q128 `A/Bsame` quality was known before the matched-spacing calibration and five-arm freeze. The matched-arm outcomes were unknown, and the fresh 210-job matrix was not unblinded until all jobs were sealed. The exact first-unblind wall-clock time was not independently logged; `provenance/blindness_chronology.json` records a bounded interval rather than inventing a timestamp.",
        "",
        "AULC was first defined in the post-unblind analysis commit. It is therefore deterministic descriptive evidence, not a frozen, preregistered, or primary outcome.",
        "",
        "## Calibration identity",
        "",
        "At stage 0 with `c=0` and no clipping, the sigmoid gap is proportional to `g/q`. Therefore `0.55/128 = 1.10/256`, making the selected q128 scale an analytic exact match to the q256 `g=1.10` reference, not merely a numerical distribution fit. The quality-blind million-sample calibration independently reports objective 0, zero clipping, and zero residual quantiles. All 15 formal trajectories remained at stage 0.",
        "",
        "## Training integrity",
        "",
        "- 15/15 fresh trajectories and 105/105 immutable state/snapshot pairs pass.",
        "- 120,000 attempts produced 119,825 accepted optimizer steps and 175 AMP skips.",
        "- All state and snapshot SHA256 values were recomputed against their receipts.",
        "- Within each seed, initialization, optimizer, GradScaler, RNG, sampler/minibatch order, and normalized non-arm configuration hashes match across all five arms.",
        "- Every preflight records the same A100-40GB model, runtime, dataset, and transfer hashes. The v1 schema did not record GPU UUID; this is disclosed and not reconstructed.",
        "- All three `A` and all three `Bsame` runs are fresh five-arm cells, not historical-output reuse.",
        "",
        "## Arm summaries",
        "",
        "AULC is normalized trapezoidal area under the natural-log FID curve; lower is better.",
        "Values below are three-seed means.",
        "",
        "| Arm | NFE1 AULC | NFE2 AULC | 1024 NFE1 FID / KID | 1024 NFE2 FID / KID |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        n1 = arm_lookup[(arm, 1)]
        n2 = arm_lookup[(arm, 2)]
        report.append(
            "| {} | {} | {} | {} / {} | {} / {} |".format(
                arm,
                fmt(n1["aulc_mean"], 4),
                fmt(n2["aulc_mean"], 4),
                fmt(n1["fid1024_mean"], 3),
                fmt(n1["kid1024_mean"], 6),
                fmt(n2["fid1024_mean"], 3),
                fmt(n2["kid1024_mean"], 6),
            )
        )
    report.extend(
        [
            "",
            "## Post-unblind deterministic AULC contrasts",
            "",
            "AULC was not present in the immutable pre-unblind protocol. It is reported as a deterministic, full-curve descriptive summary, not a frozen or preregistered primary outcome. Negative values favor the first named arm because lower AULC is better.",
            "",
            "| Contrast | NFE | Mean | Median | Range | Negative seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in [item[0] for item in PAIR_CONTRASTS] + ["interaction"]:
        for nfe in NFES:
            row = contrast_lookup[(name, nfe)]
            report.append(
                "| {} | {} | {} | {} | [{}, {}] | {}/3 |".format(
                    name,
                    nfe,
                    fmt(row["mean"], 4),
                    fmt(row["median"], 4),
                    fmt(row["min"], 4),
                    fmt(row["max"], 4),
                    row["negative_seeds"],
                )
            )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The descriptive `Bmatch-Bsame` AULC contrast is not directionally stable: NFE1 is negative for 1/3 seeds and NFE2 for 2/3 seeds. Seed 3 has a large early-curve penalty.",
            "- At 1024 kimg, `Bmatch` has lower FID and KID than `Bsame` for all three seeds at both NFEs.",
            "- `Cmatch-A` improves NFE1 AULC for 3/3 seeds and NFE2 for 2/3 seeds.",
            "- `Dmatch-A` is worse for 3/3 seeds at both NFEs.",
            "- The outcome-level interaction is negative for 2/3 seeds at NFE1 and 3/3 at NFE2; it is not an objective-level causal decomposition.",
            "",
            "## Direction consistency",
            "",
            "FID and KID agree in direction for 36/42 `Bmatch-Bsame` cells. The terminal `Bmatch-Bsame` contrast agrees in all 6 seed-by-NFE cells.",
            "",
            "## Exploratory TTQ",
            "",
            "q128 TTQ was not preregistered. `ttq_exploratory.csv` therefore reports it only as a descriptive auxiliary analysis.",
            "",
            "## Limitations and fallacy scan",
            "",
            "- Three seeds are insufficient for population-level significance claims; no p-value is used as the primary narrative.",
            "- AULC and its contrasts were defined after unblinding, so they do not mitigate researcher-degrees-of-freedom or look-elsewhere risk. Their value is deterministic complete-curve summarization, not confirmatory status.",
            "- Structural and causal fallacies are not indicated by this paired controlled design, but the 11/11 statistical fallacy checklist was reviewed.",
            "- Verification status is `ANALYZED`, not `VERIFIED`, because metrics were not independently rerun in this report step.",
            "",
            "## Files",
            "",
            "- `evaluation_results.csv`: authoritative 210-job raw matrix.",
            "- `audit.json` and `duplicate_attempts.json`: matrix/protocol audit and redundant-attempt record.",
            "- `per_seed_aulc.csv`, `per_seed_aulc_contrasts.csv`, `contrast_summary.csv`: post-unblind deterministic descriptive AULC analysis.",
            "- `arm_summary.csv` and `direction_consistency.csv`: arm-level and FID/KID summaries.",
            "- `a_bsame_bmatch_trajectories.csv`: requested per-seed, per-checkpoint comparison.",
            "- `ttq_exploratory.csv`: non-preregistered descriptive TTQ.",
            "- `validation_summary.json`: machine-readable validation status and limitations.",
            "- `provenance/calibration_manifest.json` and `calibration_exact_match.json`: compact quality-blind calibration and analytic equality audit.",
            "- `provenance/blindness_chronology.json` / `.md`: historical-vs-fresh blindness chronology.",
            "- `provenance/training/`: 15-trajectory integrity, hardware assignment, and 210-file hash audit.",
            "- `provenance/INTEGRITY_REVIEW.md`: data checks, seven-mode failure audit, and residual limitations.",
        ]
    )
    (args.outdir / "REPORT.md").write_text("\n".join(report) + "\n")

    artifact_paths = sorted(
        path
        for path in args.outdir.rglob("*")
        if path.is_file() and path != args.outdir / "SHA256SUMS.txt"
    )
    with (args.outdir / "SHA256SUMS.txt").open("w") as handle:
        for path in artifact_paths:
            handle.write(
                "{}  {}\n".format(sha256(path), path.relative_to(args.outdir))
            )


if __name__ == "__main__":
    main()
