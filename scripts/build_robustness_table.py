#!/usr/bin/env python3
"""Build a cross-seed robustness table from frozen evidence.

The three metric blocks are independent sampling-seed blocks, not independent
training runs.  This script first averages those blocks within each training
seed, then reports descriptive mean/sample-SD across the three training seeds.
It intentionally leaves R_opt unavailable for seeds 4/5: #49 measured one
Arm-A seed-3 trajectory only.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKWISE_INPUT = ROOT / "results/robustness/disjoint_5k_blockwise_results.csv"
LONGITUDINAL_INPUT = ROOT / "analysis/same_trajectory_longitudinal/longitudinal_summary.csv"
OUT_DIR = ROOT / "results/robustness"
OUT_CSV = OUT_DIR / "robustness_table.csv"
OUT_MD = OUT_DIR / "ROBUSTNESS_TABLE.md"

SEEDS = (3, 4, 5)
METRICS = ("fid5k_full", "kid5k_full")
BLOCKS = ("block_5000_9999", "block_10000_14999", "block_15000_19999")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def sign(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "−"
    return "0"


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def ratio_text(fid_ratio: float, kid_ratio: float) -> str:
    return f"FID {fid_ratio * 100:+.1f}%; KID {kid_ratio * 100:+.1f}%".replace("-", "−")


def load_blockwise() -> dict[tuple[int, str], dict[str, list[float]]]:
    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    blocks_by_cell: dict[tuple[int, str], set[str]] = defaultdict(set)
    with BLOCKWISE_INPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["training_seed"])
            metric = row["metric"]
            if seed not in SEEDS or metric not in METRICS:
                raise ValueError(f"unexpected cell: seed={seed}, metric={metric}")
            a, b, c = (float(row[key]) for key in ("A", "B", "C"))
            gap = float(row["delta_gap_B_minus_A"])
            ctrl = float(row["delta_ctrl_C_minus_B"])
            if not math.isclose(gap, b - a, abs_tol=1e-12):
                raise ValueError(f"bad B-A delta in {row}")
            if not math.isclose(ctrl, c - b, abs_tol=1e-12):
                raise ValueError(f"bad C-B delta in {row}")
            key = (seed, metric)
            blocks_by_cell[key].add(row["block"])
            grouped[key]["gap"].append(gap)
            grouped[key]["ctrl"].append(ctrl)
    for key, seen_blocks in blocks_by_cell.items():
        if set(BLOCKS) != seen_blocks:
            raise ValueError(f"incomplete block coverage for {key}: {seen_blocks}")
        if len(grouped[key]["gap"]) != 3 or len(grouped[key]["ctrl"]) != 3:
            raise ValueError(f"wrong repeat count for {key}")
    if set(grouped) != {(seed, metric) for seed in SEEDS for metric in METRICS}:
        raise ValueError("the 3-seed × 2-metric input matrix is incomplete")
    return grouped


def load_seed3_ropt() -> tuple[float, tuple[float, float]]:
    with LONGITUDINAL_INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or {float(row["K_kimg"]) for row in rows} != {32.128, 64.128, 128.128, 256.0}:
        raise ValueError("unexpected #49 longitudinal coverage")
    if any(row["source_preserved"] != "True" for row in rows):
        raise ValueError("#49 source-preservation gate failed")
    if any(row["step_skipped_g1_0"] != "False" or row["step_skipped_g1_3"] != "False" for row in rows):
        raise ValueError("#49 virtual optimizer step was skipped")
    ropt_by_k = {float(row["K_kimg"]): float(row["R_opt"]) for row in rows}
    values = tuple(ropt_by_k.values())
    return ropt_by_k[256.0], (min(values), max(values))


def block_sign_agreement(values: list[float]) -> str:
    signs = {sign(value) for value in values}
    return f"{len(values)}/3 blocks {next(iter(signs))}" if len(signs) == 1 else "mixed blocks"


def cross_seed_sign_agreement(values: list[float]) -> str:
    counts = {symbol: sum(sign(value) == symbol for value in values) for symbol in ("+", "−", "0")}
    return "/".join(f"{symbol}{count}" for symbol, count in counts.items() if count)


def build() -> None:
    grouped = load_blockwise()
    seed3_ropt, ropt_range = load_seed3_ropt()

    summary: dict[int, dict[str, float | str]] = {}
    for seed in SEEDS:
        fid_gap = mean(grouped[(seed, "fid5k_full")]["gap"])
        fid_ctrl = mean(grouped[(seed, "fid5k_full")]["ctrl"])
        kid_gap = mean(grouped[(seed, "kid5k_full")]["gap"])
        kid_ctrl = mean(grouped[(seed, "kid5k_full")]["ctrl"])
        summary[seed] = {
            "fid_gap": fid_gap,
            "fid_ctrl": fid_ctrl,
            "kid_gap": kid_gap,
            "kid_ctrl": kid_ctrl,
            "fid_absorption": fid_ctrl / -fid_gap,
            "kid_absorption": kid_ctrl / -kid_gap,
            "gap_effect": "improves" if fid_gap < 0 and kid_gap < 0 else "mixed",
            "control_effect": "regresses" if fid_ctrl > 0 and kid_ctrl > 0 else "improves",
            "gap_blocks": f"FID {block_sign_agreement(grouped[(seed, 'fid5k_full')]['gap'])}; KID {block_sign_agreement(grouped[(seed, 'kid5k_full')]['gap'])}",
            "control_blocks": f"FID {block_sign_agreement(grouped[(seed, 'fid5k_full')]['ctrl'])}; KID {block_sign_agreement(grouped[(seed, 'kid5k_full')]['ctrl'])}",
        }

    numeric_keys = ("fid_gap", "fid_ctrl", "kid_gap", "kid_ctrl", "fid_absorption", "kid_absorption")
    aggregate = {
        "mean": {key: mean([float(summary[seed][key]) for seed in SEEDS]) for key in numeric_keys},
        "std": {key: sample_sd([float(summary[seed][key]) for seed in SEEDS]) for key in numeric_keys},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "gap_effect",
        "control_effect",
        "R_opt",
        "absorption_ratio_fid",
        "absorption_ratio_kid",
        "fid_delta_gap_B_minus_A",
        "fid_delta_control_C_minus_B",
        "kid_delta_gap_B_minus_A",
        "kid_delta_control_C_minus_B",
        "sign_agreement",
        "observation_status",
    ]
    csv_rows: list[dict[str, str]] = []
    for seed in SEEDS:
        data = summary[seed]
        ropt = (
            f"{seed3_ropt:.6f}; range {ropt_range[0]:.6f}-{ropt_range[1]:.6f} over K"
            if seed == 3
            else "not measured"
        )
        csv_rows.append(
            {
                "row": f"seed {seed}",
                "gap_effect": str(data["gap_effect"]),
                "control_effect": str(data["control_effect"]),
                "R_opt": ropt,
                "absorption_ratio_fid": f"{float(data['fid_absorption']):.9f}",
                "absorption_ratio_kid": f"{float(data['kid_absorption']):.9f}",
                "fid_delta_gap_B_minus_A": f"{float(data['fid_gap']):.9f}",
                "fid_delta_control_C_minus_B": f"{float(data['fid_ctrl']):.9f}",
                "kid_delta_gap_B_minus_A": f"{float(data['kid_gap']):.9f}",
                "kid_delta_control_C_minus_B": f"{float(data['kid_ctrl']):.9f}",
                "sign_agreement": f"gap: {data['gap_blocks']}; control: {data['control_blocks']}",
                "observation_status": "per-seed descriptive",
            }
        )
    for row_name, label in (("mean", "descriptive mean across training seeds"), ("std", "sample SD across training seeds")):
        values = aggregate[row_name]
        csv_rows.append(
            {
                "row": row_name,
                "gap_effect": "all seed effects improve" if row_name == "mean" else "between-seed dispersion",
                "control_effect": "mean masks sign reversal" if row_name == "mean" else "between-seed dispersion",
                "R_opt": "not estimable (1/3 seeds measured)",
                "absorption_ratio_fid": f"{values['fid_absorption']:.9f}",
                "absorption_ratio_kid": f"{values['kid_absorption']:.9f}",
                "fid_delta_gap_B_minus_A": f"{values['fid_gap']:.9f}",
                "fid_delta_control_C_minus_B": f"{values['fid_ctrl']:.9f}",
                "kid_delta_gap_B_minus_A": f"{values['kid_gap']:.9f}",
                "kid_delta_control_C_minus_B": f"{values['kid_ctrl']:.9f}",
                "sign_agreement": label,
                "observation_status": "descriptive only",
            }
        )
    csv_rows.append(
        {
            "row": "sign agreement",
            "gap_effect": "Robust: FID −3/3; KID −3/3",
            "control_effect": "Seed-dependent: FID +2/−1; KID +2/−1",
            "R_opt": "Not reproduced: measured on seed 3 only",
            "absorption_ratio_fid": cross_seed_sign_agreement([float(summary[seed]["fid_absorption"]) for seed in SEEDS]),
            "absorption_ratio_kid": cross_seed_sign_agreement([float(summary[seed]["kid_absorption"]) for seed in SEEDS]),
            "fid_delta_gap_B_minus_A": "gap −3/3; control +2/−1",
            "fid_delta_control_C_minus_B": "gap −3/3; control +2/−1",
            "kid_delta_gap_B_minus_A": "gap −3/3; control +2/−1",
            "kid_delta_control_C_minus_B": "gap −3/3; control +2/−1",
            "sign_agreement": "sign agreement is across training seeds; each seed's three blocks agree internally",
            "observation_status": "see observation verdicts",
        }
    )
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    def row_markdown(seed: int) -> str:
        data = summary[seed]
        ropt = (
            f"{seed3_ropt * 100:.2f}% @256k (range {ropt_range[0] * 100:.2f}-{ropt_range[1] * 100:.2f}% over K)"
            if seed == 3
            else "not measured"
        )
        return (
            f"| seed {seed} | {data['gap_effect']} | {data['control_effect']} | {ropt} | "
            f"{ratio_text(float(data['fid_absorption']), float(data['kid_absorption']))} | "
            f"{signed(float(data['fid_gap']), 2)}; {signed(float(data['fid_ctrl']), 2)} | "
            f"{signed(float(data['kid_gap']), 6)}; {signed(float(data['kid_ctrl']), 6)} | "
            f"gap {data['gap_blocks']}; ctrl {data['control_blocks']} |"
        )

    mean_values = aggregate["mean"]
    std_values = aggregate["std"]
    markdown = f"""# Cross-seed robustness table

## Scope and conventions

This table consolidates the frozen evidence from PR #49 (single-trajectory
stateful RAdam audit), PR #51 (the seed-4/5 evaluation handoff), and PR #53
(three disjoint NFE=1 FID/KID-5k blocks for training seeds 3/4/5).

`gap` is B−A: fixed learning rate, `g=1.3` minus `g=1.0`. `control` is C−B:
fresh-state LR-matched `g=1.3` minus fixed-LR `g=1.3`. Lower FID/KID is
better, so a negative delta is an improvement. Absorption is
`control / (−gap)`: positive means the matching control removes part of the
gap-associated advantage; negative means it strengthens that advantage.

The three 5k blocks are independent *sampling* blocks, not extra training
seeds. Each seed row is its mean across blocks; `mean` and `std` are
descriptive mean and sample SD across the three training seeds. `R_opt` is
reported only where it was actually measured; missing seed-4/5 measurements
are never filled with the seed-3 result.

The absorption entry in the `mean` row is the arithmetic mean of the three
per-seed absorption ratios. It is deliberately **not** the ratio of the mean
control delta to the mean gap delta, because the former preserves the unit of
cross-seed heterogeneity being summarized.

## Combined table

| row | gap effect | fresh-state control | $R_{{opt}}$ | absorption ratio | FID delta (gap; control) | KID delta (gap; control) | sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row_markdown(3)}
{row_markdown(4)}
{row_markdown(5)}
| mean | all seed effects improve | **mean masks sign reversal** | not estimable (1/3 seeds measured) | {ratio_text(mean_values['fid_absorption'], mean_values['kid_absorption'])} | {signed(mean_values['fid_gap'], 2)}; {signed(mean_values['fid_ctrl'], 2)} | {signed(mean_values['kid_gap'], 6)}; {signed(mean_values['kid_ctrl'], 6)} | gap −3/3; control +2/−1 |
| std | between-seed dispersion | between-seed dispersion | not estimable | FID {std_values['fid_absorption'] * 100:.1f}%; KID {std_values['kid_absorption'] * 100:.1f}% | {std_values['fid_gap']:.2f}; {std_values['fid_ctrl']:.2f} | {std_values['kid_gap']:.6f}; {std_values['kid_ctrl']:.6f} | sample SD over 3 training seeds |
| sign agreement | **Robust: FID −3/3; KID −3/3** | **Seed-dependent: FID +2/−1; KID +2/−1** | **Not reproduced: seed 3 only** | FID +2/−1; KID +2/−1 | gap −3/3; control +2/−1 | gap −3/3; control +2/−1 | within every seed, all 3 disjoint blocks agree |

## Observation verdicts

| Observation | Tag | Evidence-supported statement |
| --- | --- | --- |
| Fixed-LR gap effect (B−A) | **Robust** | B improves both FID and KID versus A in every training seed and every disjoint block: 3/3 negative FID deltas and 3/3 negative KID deltas across seeds. |
| Fresh-state control effect (C−B) | **Seed-dependent / sign-unstable** | Seed 3 is strongly positive, seed 4 weakly positive, and seed 5 negative for both FID and KID. The sign is stable within each seed's three disjoint sampling blocks, so sampling-block noise does not explain the seed-4/5 reversal. Do not summarize this as a single positive mean. |
| Absorption ratio | **Seed-dependent / sign-unstable** | It inherits the control reversal: seed 3 removes about 84% of the gap advantage, seed 4 about 9-13%, and seed 5 has negative absorption (the matched control strengthens the advantage). |
| Stateful $R_{{opt}}$ along the #49 trajectory | **Not reproduced** | Seed 3 has a preserved, non-skipped four-state audit with $R_{{opt}}=8.22\\%-9.90\\%$; seeds 4 and 5 were not measured. It is longitudinally observed, but it is not a cross-seed result. |

## What survives across seeds?

Only the fixed-LR larger-gap comparison survives this three-seed check: it
improves both disjoint-block FID-5k and KID-5k in all seeds. The fresh-state
LR control does **not** have a seed-invariant direction. Its positive average
is therefore descriptive bookkeeping, not a robustness claim, and it must be
reported as **seed-dependent / sign-unstable**. The $R_{{opt}}$ longitudinal
pattern currently has no seeds-4/5 replication and remains **not reproduced**
across seeds.

## Provenance

- PR #49 / merge `5561ca3`: `analysis/same_trajectory_longitudinal/longitudinal_summary.csv` (SHA-256 `864fc251bf9e7ef300117c11b7e91e02b6b1268c7d94bedb19545c53ee12f435`).
- PR #51 / merge `1db0322`: frozen seed-4/5 handoff in `docs/ROLE_E_DISJOINT_5K_HANDOFF.md`; it establishes the admissible endpoint and disjoint-block contract.
- PR #53 / merge `6b0a110`: `results/gap_lr_matched/disjoint_5k_0813/blockwise_results.csv`, copied verbatim here as `disjoint_5k_blockwise_results.csv` for this table's self-contained calculation (SHA-256 `d748a4dcb33589276b82d7c1825b1fce8cfbf2c7a14e0bbe857de150bc960189`).

All FID/KID entries are NFE=1, FP32, 5k-sample proxy evaluations. They are not
FID-50k or KID-50k claims.
"""
    OUT_MD.write_text(markdown)


if __name__ == "__main__":
    build()
