#!/usr/bin/env python3
"""Validate and summarize the q256 seed6--13 longitudinal factorial receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BUDGETS = (384, 512, 640, 768, 896, 1024)
BALANCED_SEEDS = tuple(range(8, 14))
EXTENSION_SEEDS = (6, 7)
ARMS = ("A", "B", "C", "D")
NFES = (1, 2)
RECEIPT_RE = re.compile(r"^seed(?P<seed>\d+)-arm(?P<arm>[A-D])-k(?P<budget>\d+)-nfe(?P<nfe>[12])\.json$")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], *, samples: int = 20_000) -> tuple[float, float]:
    seed = 20260824 + len(values) + int(abs(sum(values)) * 1000) % 100_000
    rng = random.Random(seed)
    means = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return quantile(means, 0.025), quantile(means, 0.975)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipts(receipts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, int]] = set()
    for path in sorted(receipts_dir.glob("*.json")):
        match = RECEIPT_RE.match(path.name)
        if match is None:
            continue
        row = json.loads(path.read_text())
        key = (int(row["seed"]), str(row["arm"]), int(row["budget_kimg"]), int(row["nfe"]))
        filename_key = (
            int(match.group("seed")),
            match.group("arm"),
            int(match.group("budget")),
            int(match.group("nfe")),
        )
        if key != filename_key:
            raise RuntimeError(f"receipt filename/content mismatch: {path}")
        if key in seen:
            raise RuntimeError(f"duplicate receipt key: {key}")
        if row.get("status") != "PASS":
            raise RuntimeError(f"non-PASS receipt: {path}")
        if row.get("sample_count") != 50_000 or row.get("metric_seed") != 20260730:
            raise RuntimeError(f"frozen evaluation mismatch: {path}")
        if row.get("sample_seed_range") != "0-49999":
            raise RuntimeError(f"sample seed range mismatch: {path}")
        expected_mid = None if key[3] == 1 else 0.821
        if row.get("mid_t") != expected_mid:
            raise RuntimeError(f"mid_t mismatch: {path}")
        row = dict(row)
        row["receipt_file"] = path.name
        seen.add(key)
        rows.append(row)

    if len(rows) != 336:
        raise RuntimeError(f"expected 336 receipts, found {len(rows)}")
    expected = {
        *((seed, arm, budget, nfe) for seed in EXTENSION_SEEDS for arm in ("C", "D") for budget in BUDGETS for nfe in NFES),
        *((seed, arm, budget, nfe) for seed in BALANCED_SEEDS for arm in ARMS for budget in BUDGETS for nfe in NFES),
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise RuntimeError(f"coverage mismatch: missing={missing} extra={extra}")
    return rows


def normalized_aulc(index: dict[tuple[int, str, int, int], dict[str, Any]], seed: int, arm: str, nfe: int) -> float:
    values = [float(index[(seed, arm, budget, nfe)]["fid50k_full"]) for budget in BUDGETS]
    area = sum(
        (BUDGETS[i + 1] - BUDGETS[i]) * (values[i + 1] + values[i]) / 2
        for i in range(len(BUDGETS) - 1)
    )
    return area / (BUDGETS[-1] - BUDGETS[0])


def sustained_time(index: dict[tuple[int, str, int, int], dict[str, Any]], seed: int, arm: str, nfe: int) -> int | None:
    values = [float(index[(seed, arm, budget, nfe)]["fid50k_full"]) for budget in BUDGETS]
    for position, budget in enumerate(BUDGETS):
        if all(value <= 10 for value in values[position:]):
            return budget
    return None


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(receipts_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = load_receipts(receipts_dir)
    receipts.sort(key=lambda row: (row["seed"], row["arm"], row["budget_kimg"], row["nfe"]))
    index = {
        (int(row["seed"]), str(row["arm"]), int(row["budget_kimg"]), int(row["nfe"])): row
        for row in receipts
    }

    receipt_payload = {
        "schema": "ect.q256.seed6-13-longitudinal-factorial-receipts/v1",
        "receipt_count": len(receipts),
        "trajectory_count": 28,
        "budgets_kimg": list(BUDGETS),
        "nfes": list(NFES),
        "metric_seed": 20260730,
        "sample_seed_range": "0-49999",
        "sample_count": 50_000,
        "receipts": receipts,
    }
    (output_dir / "evaluation_receipts.json").write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")

    result_rows = [
        {
            "seed": row["seed"],
            "arm": row["arm"],
            "budget_kimg": row["budget_kimg"],
            "nfe": row["nfe"],
            "mid_t": "" if row["mid_t"] is None else row["mid_t"],
            "fid50k_full": row["fid50k_full"],
            "kid50k_full": row["kid50k_full"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "generated_feature_sha256": row["generated_feature_sha256"],
            "receipt_file": row["receipt_file"],
        }
        for row in receipts
    ]
    write_csv(
        output_dir / "evaluation_results.csv",
        ["seed", "arm", "budget_kimg", "nfe", "mid_t", "fid50k_full", "kid50k_full", "checkpoint_sha256", "generated_feature_sha256", "receipt_file"],
        result_rows,
    )

    curve_rows: list[dict[str, Any]] = []
    for nfe in NFES:
        for arm in ARMS:
            for budget in BUDGETS:
                values = [float(index[(seed, arm, budget, nfe)]["fid50k_full"]) for seed in BALANCED_SEEDS]
                kids = [float(index[(seed, arm, budget, nfe)]["kid50k_full"]) for seed in BALANCED_SEEDS]
                curve_rows.append(
                    {
                        "cohort": "balanced_seed8_13",
                        "nfe": nfe,
                        "arm": arm,
                        "budget_kimg": budget,
                        "n": len(values),
                        "fid_mean": mean(values),
                        "fid_sd": statistics.stdev(values),
                        "fid_median": statistics.median(values),
                        "kid_mean": mean(kids),
                        "kid_sd": statistics.stdev(kids),
                    }
                )
    write_csv(
        output_dir / "longitudinal_summary.csv",
        ["cohort", "nfe", "arm", "budget_kimg", "n", "fid_mean", "fid_sd", "fid_median", "kid_mean", "kid_sd"],
        curve_rows,
    )

    aulc_rows: list[dict[str, Any]] = []
    for seed in BALANCED_SEEDS:
        for nfe in NFES:
            for arm in ARMS:
                aulc_rows.append(
                    {
                        "seed": seed,
                        "nfe": nfe,
                        "arm": arm,
                        "normalized_fid_aulc": normalized_aulc(index, seed, arm, nfe),
                        "endpoint_fid_1024": index[(seed, arm, 1024, nfe)]["fid50k_full"],
                        "endpoint_kid_1024": index[(seed, arm, 1024, nfe)]["kid50k_full"],
                        "sustained_fid10_kimg": sustained_time(index, seed, arm, nfe),
                    }
                )
    write_csv(
        output_dir / "aulc_per_seed.csv",
        ["seed", "nfe", "arm", "normalized_fid_aulc", "endpoint_fid_1024", "endpoint_kid_1024", "sustained_fid10_kimg"],
        aulc_rows,
    )

    contrast_rows: list[dict[str, Any]] = []
    for seed in BALANCED_SEEDS:
        for nfe in NFES:
            aulcs = {arm: normalized_aulc(index, seed, arm, nfe) for arm in ARMS}
            endpoints = {arm: float(index[(seed, arm, 1024, nfe)]["fid50k_full"]) for arm in ARMS}
            contrast_rows.append(
                {
                    "seed": seed,
                    "nfe": nfe,
                    "aulc_B_minus_A": aulcs["B"] - aulcs["A"],
                    "aulc_C_minus_A": aulcs["C"] - aulcs["A"],
                    "aulc_D_minus_A": aulcs["D"] - aulcs["A"],
                    "aulc_interaction_B_minus_C_minus_D_plus_A": aulcs["B"] - aulcs["C"] - aulcs["D"] + aulcs["A"],
                    "endpoint_B_minus_A": endpoints["B"] - endpoints["A"],
                    "endpoint_C_minus_A": endpoints["C"] - endpoints["A"],
                    "endpoint_D_minus_A": endpoints["D"] - endpoints["A"],
                    "endpoint_interaction_B_minus_C_minus_D_plus_A": endpoints["B"] - endpoints["C"] - endpoints["D"] + endpoints["A"],
                }
            )
    write_csv(
        output_dir / "factorial_contrasts.csv",
        list(contrast_rows[0]),
        contrast_rows,
    )

    summary: dict[str, Any] = {
        "schema": "ect.q256.seed6-13-longitudinal-factorial-summary/v1",
        "balanced_seeds": list(BALANCED_SEEDS),
        "extension_seeds": list(EXTENSION_SEEDS),
        "normalized_aulc_interval_kimg": [BUDGETS[0], BUDGETS[-1]],
        "balanced": {},
        "seed6_7_cd_extension": {},
    }
    for nfe in NFES:
        nfe_key = f"nfe{nfe}"
        arm_summary: dict[str, Any] = {}
        for arm in ARMS:
            aulcs = [normalized_aulc(index, seed, arm, nfe) for seed in BALANCED_SEEDS]
            endpoints = [float(index[(seed, arm, 1024, nfe)]["fid50k_full"]) for seed in BALANCED_SEEDS]
            times = [sustained_time(index, seed, arm, nfe) for seed in BALANCED_SEEDS]
            attained = [time for time in times if time is not None]
            arm_summary[arm] = {
                "aulc_mean": mean(aulcs),
                "aulc_sd": statistics.stdev(aulcs),
                "aulc_median": statistics.median(aulcs),
                "endpoint_fid_mean": mean(endpoints),
                "endpoint_fid_sd": statistics.stdev(endpoints),
                "endpoint_fid_median": statistics.median(endpoints),
                "sustained_fid10_attained": len(attained),
                "sustained_fid10_median_kimg": statistics.median(attained) if attained else None,
                "sustained_fid10_per_seed": times,
            }
        contrast_summary: dict[str, Any] = {}
        contrast_functions = {
            "B_minus_A": lambda seed: normalized_aulc(index, seed, "B", nfe) - normalized_aulc(index, seed, "A", nfe),
            "C_minus_A": lambda seed: normalized_aulc(index, seed, "C", nfe) - normalized_aulc(index, seed, "A", nfe),
            "D_minus_A": lambda seed: normalized_aulc(index, seed, "D", nfe) - normalized_aulc(index, seed, "A", nfe),
            "interaction": lambda seed: normalized_aulc(index, seed, "B", nfe) - normalized_aulc(index, seed, "C", nfe) - normalized_aulc(index, seed, "D", nfe) + normalized_aulc(index, seed, "A", nfe),
        }
        for name, function in contrast_functions.items():
            values = [function(seed) for seed in BALANCED_SEEDS]
            lower, upper = bootstrap_mean_ci(values)
            contrast_summary[name] = {
                "mean": mean(values),
                "median": statistics.median(values),
                "bootstrap_mean_ci95": [lower, upper],
                "negative_seed_count": sum(value < 0 for value in values),
                "values": values,
            }
        endpoint_winners = Counter(
            min(ARMS, key=lambda arm: float(index[(seed, arm, 1024, nfe)]["fid50k_full"]))
            for seed in BALANCED_SEEDS
        )
        aulc_winners = Counter(
            min(ARMS, key=lambda arm: normalized_aulc(index, seed, arm, nfe))
            for seed in BALANCED_SEEDS
        )
        summary["balanced"][nfe_key] = {
            "arms": arm_summary,
            "aulc_contrasts": contrast_summary,
            "endpoint_winner_counts": dict(endpoint_winners),
            "aulc_winner_counts": dict(aulc_winners),
        }

        extension: dict[str, Any] = {}
        for arm in ("C", "D"):
            aulcs = [normalized_aulc(index, seed, arm, nfe) for seed in EXTENSION_SEEDS]
            endpoints = [float(index[(seed, arm, 1024, nfe)]["fid50k_full"]) for seed in EXTENSION_SEEDS]
            extension[arm] = {"aulc_mean": mean(aulcs), "aulc_values": aulcs, "endpoint_mean": mean(endpoints), "endpoint_values": endpoints}
        extension["C_minus_D"] = {
            "aulc_mean": mean(normalized_aulc(index, seed, "C", nfe) - normalized_aulc(index, seed, "D", nfe) for seed in EXTENSION_SEEDS),
            "endpoint_mean": mean(float(index[(seed, "C", 1024, nfe)]["fid50k_full"]) - float(index[(seed, "D", 1024, nfe)]["fid50k_full"]) for seed in EXTENSION_SEEDS),
        }
        summary["seed6_7_cd_extension"][nfe_key] = extension

    (output_dir / "factorial_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    generated = [
        output_dir / "evaluation_receipts.json",
        output_dir / "evaluation_results.csv",
        output_dir / "longitudinal_summary.csv",
        output_dir / "aulc_per_seed.csv",
        output_dir / "factorial_contrasts.csv",
        output_dir / "factorial_summary.json",
    ]
    generated.extend(
        path
        for path in (output_dir / "README.md", output_dir / "REPORT_MANIFEST.json")
        if path.is_file()
    )
    with (output_dir / "SHA256SUMS.txt").open("w") as handle:
        for path in generated:
            handle.write(f"{sha256(path)}  {path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.receipts_dir, args.output_dir)


if __name__ == "__main__":
    main()
