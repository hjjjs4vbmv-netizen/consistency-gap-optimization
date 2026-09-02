#!/usr/bin/env python3
"""Descriptive structure diagnostic for the amended n=11 fresh replication.

This script decomposes the seed-level primary quantity H into structure that
the pooled mean conceals: the relationship between H (endpoint history
contrast) and Q (switch-point gap), threshold-based subgroup summaries, and
the delayed-rank-reversal replication status among reversal-eligible seeds.

Everything here is DESCRIPTIVE ONLY. Per the frozen claim boundary in
REPORT_11SEED.md, no quantity computed by this script can alter, rescue, or
overturn the frozen primary verdict (INCONCLUSIVE). The script is bound
fail-closed to the exact preserved per-seed table by SHA-256; if the table
changes, this diagnostic must be re-authored, not silently re-run.

Column definitions (from the preserved table):
  H    = 0.5 * [(Y_BA - Y_AA) + (Y_BB - Y_AB)]  endpoint history contrast
  H_A  = Y_BA - Y_AA                            history contrast, A continuation
  Q    = logFID_B_512 - logFID_A_512            switch-point gap (B - A)
  G    = H_A - Q                                gap change 512 -> 1024 under A
so G = H_A - Q holds by definition and is verified below as a data-integrity
check, not as a finding.

Interpretation note that motivates this diagnostic: a negative H can arise
from two different histories -- (i) B already better at the switch (Q < 0)
and the advantage persisting, or (ii) B worse at the switch (Q > 0) and the
ordering reversing by the endpoint. Only (ii) is a delayed rank reversal.
The pooled H mean does not distinguish them; the subgroup tables below do.

Usage:
  python structure_diagnostic.py \
      --per-seed-csv results/q256_fresh_crossed_switch_n12_matpool_v1/final_11seed/H_C_I_Q_G_per_seed.csv \
      --output-dir   results/q256_fresh_crossed_switch_n12_matpool_v1/structure_diagnostic_v1

Dependency-free (Python standard library only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "ect.q256.fresh-crossed-switch-structure-diagnostic/v1"

# Fail-closed binding to the preserved formal per-seed table.
EXPECTED_PER_SEED_CSV_SHA256 = (
    "4d8bc83f7e9254878294a38fc3ad2ac40c84445d497dd166cec4f37b2e197461"
)

# Symmetric, pre-stated subgroup thresholds on Q (log-FID units). 0.0 is the
# sign boundary; 0.5 marks a switch-point gap large enough (>= ~65% FID ratio)
# that persistence and reversal cannot be confused with evaluation noise.
Q_THRESHOLDS = (0.0, 0.5)

# Two-sided 95% Student-t critical values, df 1..15 (descriptive CIs only).
T_CRIT_95 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def ci95(values: list[float]) -> list[float] | None:
    n = len(values)
    if n < 2:
        return None
    critical = T_CRIT_95[n - 1]
    half = critical * sample_sd(values) / math.sqrt(n)
    center = mean(values)
    return [center - half, center + half]


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ols_slope(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    return cov / vx


def summarize(seeds: list[int], values: list[float]) -> dict:
    return {
        "seeds": seeds,
        "n": len(values),
        "H_values": values,
        "H_mean": mean(values) if values else None,
        "H_ci95": ci95(values),
        "negative_count": sum(v < 0 for v in values),
        "positive_count": sum(v > 0 for v in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.per_seed_csv.resolve(strict=True)
    actual_sha = sha256_file(csv_path)
    if actual_sha != EXPECTED_PER_SEED_CSV_SHA256:
        raise SystemExit(
            "per-seed table SHA256 mismatch: expected "
            f"{EXPECTED_PER_SEED_CSV_SHA256}, got {actual_sha}; "
            "this diagnostic is bound to the preserved formal table and must "
            "be re-authored for any other input"
        )

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 11:
        raise SystemExit(f"expected 11 seeds, found {len(rows)}")

    seeds = [int(row["seed"]) for row in rows]
    H = [float(row["H"]) for row in rows]
    H_A = [float(row["H_A"]) for row in rows]
    Q = [float(row["Q"]) for row in rows]
    G = [float(row["G"]) for row in rows]

    # Data-integrity check of the definitional identity G = H_A - Q.
    worst_residual = max(abs(g - (ha - q)) for g, ha, q in zip(G, H_A, Q))
    if worst_residual > 1e-9:
        raise SystemExit(
            f"G = H_A - Q identity violated (max residual {worst_residual})"
        )

    association = {
        "pearson_corr_H_Q": pearson(Q, H),
        "spearman_corr_H_Q": spearman(Q, H),
        "ols_slope_H_on_Q": ols_slope(Q, H),
        "pearson_corr_G_Q": pearson(Q, G),
    }

    subgroups = {}
    for threshold in Q_THRESHOLDS:
        below_idx = [i for i in range(11) if Q[i] < -threshold]
        above_idx = [i for i in range(11) if Q[i] > threshold]
        rest_idx = [i for i in range(11) if i not in below_idx]
        key = f"abs_Q_gt_{threshold:g}"
        subgroups[key] = {
            "threshold": threshold,
            "B_better_at_switch_Q_below_neg_threshold": summarize(
                [seeds[i] for i in below_idx], [H[i] for i in below_idx]
            ),
            "B_worse_at_switch_Q_above_threshold": summarize(
                [seeds[i] for i in above_idx], [H[i] for i in above_idx]
            ),
            "sensitivity_excluding_Q_below_neg_threshold": summarize(
                [seeds[i] for i in rest_idx], [H[i] for i in rest_idx]
            ),
        }

    # Delayed-rank-reversal replication among reversal-eligible seeds
    # (B worse at the switch): a reversal is H < 0 at the endpoint.
    reversal = {}
    for threshold in Q_THRESHOLDS:
        eligible = [i for i in range(11) if Q[i] > threshold]
        reversal[f"Q_gt_{threshold:g}"] = {
            "eligible_seeds": [seeds[i] for i in eligible],
            "reversed_seeds": [seeds[i] for i in eligible if H[i] < 0],
            "non_reversed_seeds": [seeds[i] for i in eligible if H[i] >= 0],
            "per_seed": [
                {"seed": seeds[i], "Q": Q[i], "H": H[i], "reversed": H[i] < 0}
                for i in eligible
            ],
        }

    report = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_ONLY",
        "claim_boundary": (
            "Descriptive structure diagnostic. Cannot alter, rescue, or "
            "overturn the frozen primary verdict (INCONCLUSIVE) in "
            "primary_decision.json."
        ),
        "per_seed_csv_sha256": actual_sha,
        "n": 11,
        "pooled_H": summarize(seeds, H),
        "association": association,
        "subgroups": subgroups,
        "reversal_replication": reversal,
        "identity_check_max_abs_residual": worst_residual,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "structure_diagnostic_v1.json"
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"wrote {json_path}")

    def fmt(value: float) -> str:
        return f"{value:+.4f}"

    lines = []
    lines.append("# H structure diagnostic (descriptive only)\n")
    lines.append(
        "Bound to `H_C_I_Q_G_per_seed.csv` SHA256 `" + actual_sha + "`. "
        "Nothing below can alter the frozen primary verdict "
        "(**INCONCLUSIVE**).\n"
    )
    lines.append("## Association between H and the switch-point gap Q\n")
    lines.append(
        f"- Pearson corr(H, Q) = {association['pearson_corr_H_Q']:+.3f}; "
        f"Spearman = {association['spearman_corr_H_Q']:+.3f}; "
        f"OLS slope of H on Q = {association['ols_slope_H_on_Q']:+.3f}.\n"
    )
    lines.append("## Subgroups by switch-point gap\n")
    lines.append(
        "| Subgroup | Seeds | n | H mean | 95% CI | H < 0 |\n"
        "| --- | --- | ---: | ---: | --- | ---: |"
    )
    for threshold in Q_THRESHOLDS:
        neg_threshold = -threshold if threshold else 0.0
        block = subgroups[f"abs_Q_gt_{threshold:g}"]
        for label, part in (
            (f"Q < {neg_threshold:g} (B better at switch)",
             block["B_better_at_switch_Q_below_neg_threshold"]),
            (f"Q > {threshold:g} (B worse at switch)",
             block["B_worse_at_switch_Q_above_threshold"]),
            (f"All except Q < {neg_threshold:g}",
             block["sensitivity_excluding_Q_below_neg_threshold"]),
        ):
            ci = part["H_ci95"]
            ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "--"
            lines.append(
                f"| {label} | {', '.join(map(str, part['seeds']))} "
                f"| {part['n']} | {fmt(part['H_mean'])} | {ci_str} "
                f"| {part['negative_count']}/{part['n']} |"
            )
    lines.append("")
    lines.append("## Delayed-rank-reversal replication\n")
    for threshold in Q_THRESHOLDS:
        block = reversal[f"Q_gt_{threshold:g}"]
        lines.append(
            f"Eligible (Q > {threshold:g}): "
            f"{len(block['reversed_seeds'])}/{len(block['eligible_seeds'])} "
            "reversed at 1024 kimg."
        )
        for row in block["per_seed"]:
            lines.append(
                f"- seed{row['seed']}: Q = {fmt(row['Q'])}, "
                f"H = {fmt(row['H'])} -> "
                + ("reversed" if row["reversed"] else "not reversed")
            )
        lines.append("")
    lines.append("## Reading\n")
    lines.append(
        "A negative pooled H mixes two distinct seed-level histories: "
        "persistence of an advantage B already held at the switch (Q < 0) "
        "and genuine delayed reversal (Q > 0 with H < 0). The subgroup rows "
        "separate them; only the reversal-eligible rows bear on the "
        "mid-training-misranking reading. All quantities are descriptive "
        "and post-unblind.\n"
    )

    md_path = args.output_dir / "STRUCTURE_DIAGNOSTIC_V1.md"
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
