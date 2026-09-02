#!/usr/bin/env python3
"""Frozen post-seal H/C/I/Q/G analysis for the fresh n=12 replication."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402

DELTA = math.log(1.03)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def sample_sd(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(mean(values))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean(sign * value for sign, value in zip(signs, values)))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(values))


def interval(values: list[float], confidence: float) -> list[float]:
    n = len(values)
    center = mean(values)
    sd = sample_sd(values)
    half = float(scipy.stats.t.ppf((1 + confidence) / 2, n - 1) * sd / math.sqrt(n))
    return [center - half, center + half]


def summarize(values: list[float], *, sign_flip: bool = True) -> dict:
    if len(values) not in {11, 12} or any(not math.isfinite(value) for value in values):
        raise RuntimeError("summary requires eleven or twelve finite seed-level values")
    n = len(values)
    result = {
        "n": n, "mean": mean(values), "median": median(values),
        "sample_sd": sample_sd(values), "ci95_two_sided": interval(values, 0.95),
        "ci90_two_sided": interval(values, 0.90),
        "negative_count": sum(value < 0 for value in values),
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "range": [min(values), max(values)],
        "leave_one_seed_out_means": [mean(values[:i] + values[i + 1:]) for i in range(n)],
    }
    if sign_flip:
        result["exact_two_sided_sign_flip_p"] = exact_sign_flip(values)
    return result


def primary_verdict(summary: dict) -> tuple[str, dict]:
    lo95, hi95 = summary["ci95_two_sided"]
    lo90, hi90 = summary["ci90_two_sided"]
    strong = bool(hi95 < -DELTA and summary["negative_count"] >= 10
                  and all(value < 0 for value in summary["leave_one_seed_out_means"]))
    equivalent = bool(lo90 > -DELTA and hi90 < DELTA)
    weak = bool(not strong and not equivalent and hi95 < 0)
    opposite = bool(lo95 > 0)
    inconclusive = bool(lo95 <= 0 <= hi95 and not equivalent)
    # The equivalence and opposite-direction conditions can mathematically
    # overlap for a very precise but practically tiny positive effect.  The
    # frozen order makes practical equivalence take precedence in that case.
    if strong:
        verdict = "STRONG_SUCCESS"
    elif equivalent:
        verdict = "INFORMATIVE_PRACTICAL_NULL"
    elif weak:
        verdict = "WEAK_DIRECTIONAL_REPLICATION"
    elif opposite:
        verdict = "OPPOSITE_DIRECTION_FALSIFICATION"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        raise RuntimeError("frozen primary categories are not exhaustive")
    return verdict, {
        "strong_success": strong, "informative_practical_null": equivalent,
        "weak_directional_replication": weak, "inconclusive": inconclusive,
        "opposite_direction_falsification": opposite,
    }


def holm(raw: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalue))
        adjusted[name] = running
    return {name: {"raw_p": raw[name], "holm_adjusted_p": adjusted[name],
                   "reject_at_0.05": adjusted[name] <= alpha} for name in raw}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoded-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eleven-seed-authorization", type=Path)
    parser.add_argument("--evaluation-recovery-authorization", type=Path)
    parser.add_argument("--postseal-report-recovery-authorization", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    seeds = experiment.SEEDS
    expected_jobs = 264
    authorization_sha = None
    recovery_authorization_sha = None
    postseal_authorization_sha = None
    if args.eleven_seed_authorization is not None:
        authorization_path = args.eleven_seed_authorization.resolve(strict=True)
        experiment.validate_eleven_seed_authorization(
            authorization_path, protocol_path,
            require_commit=args.evaluation_recovery_authorization is None
        )
        seeds = experiment.ELEVEN_SEEDS
        expected_jobs = experiment.ELEVEN_JOB_COUNT
        authorization_sha = experiment.sha256_file(authorization_path)
    if args.evaluation_recovery_authorization is not None:
        if authorization_sha is None:
            raise RuntimeError("evaluation recovery statistics require eleven-seed authorization")
        recovery_path = args.evaluation_recovery_authorization.resolve(strict=True)
        recovery = experiment.validate_evaluation_recovery1_authorization(
            recovery_path, protocol_path, require_commit=True
        )
        if recovery.get("eleven_seed_authorization_sha256") != authorization_sha:
            raise RuntimeError("evaluation recovery statistics amendment binding mismatch")
        recovery_authorization_sha = experiment.sha256_file(recovery_path)
    if args.postseal_report_recovery_authorization is not None:
        from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import postseal_recovery
        postseal_path = args.postseal_report_recovery_authorization.resolve(strict=True)
        postseal_recovery.validate_authorization(postseal_path, protocol_path,
                                                 require_commit=True)
        postseal_authorization_sha = experiment.sha256_file(postseal_path)
    decoded_path = args.decoded_results.resolve(strict=True)
    decoded = experiment.load_json(decoded_path)
    if (decoded.get("status") != "PASS" or not decoded.get("decoded_after_full_seal")
            or decoded.get("job_count") != expected_jobs
            or len(decoded.get("results", [])) != expected_jobs
            or decoded.get("eleven_seed_authorization_sha256") != authorization_sha
            or decoded.get("evaluation_recovery_authorization_sha256")
            != recovery_authorization_sha):
        raise RuntimeError("statistics require the complete post-seal decoded matrix")
    index = {}
    for row in decoded["results"]:
        key = (row["seed"], row["kind"], row["cell"], row["budget_kimg"], row["nfe"])
        if key in index:
            raise RuntimeError(f"duplicate decoded evaluation cell: {key}")
        index[key] = row
    expected = expected_jobs
    if len(index) != expected:
        raise RuntimeError("decoded matrix identity count mismatch")

    rows = []
    for seed in seeds:
        y = {cell: math.log(index[(seed, "suffix", cell, 1024, 1)]["fid50k_full"])
             for cell in experiment.CELLS}
        source = {arm: math.log(index[(seed, "prefix", arm, 512, 1)]["fid50k_full"])
                  for arm in experiment.ARMS}
        h = 0.5 * ((y["BA"] - y["AA"]) + (y["BB"] - y["AB"]))
        c = 0.5 * ((y["AB"] - y["AA"]) + (y["BB"] - y["BA"]))
        interaction = y["BB"] - y["BA"] - y["AB"] + y["AA"]
        q = source["B"] - source["A"]
        h_a = y["BA"] - y["AA"]
        g = h_a - q
        rows.append({"seed": seed, "H": h, "C": c, "I": interaction,
                     "Q": q, "H_A": h_a, "G": g, **{f"Y_{cell}": y[cell] for cell in y},
                     "logFID_A_512": source["A"], "logFID_B_512": source["B"]})
    summaries = {name: summarize([row[name] for row in rows]) for name in ("H", "C", "I", "Q", "H_A", "G")}
    verdict, checks = primary_verdict(summaries["H"])
    interaction_equivalent = bool(summaries["I"]["ci90_two_sided"][0] > -DELTA
                                  and summaries["I"]["ci90_two_sided"][1] < DELTA)
    secondary_holm = holm({name: summaries[name]["exact_two_sided_sign_flip_p"] for name in ("C", "I")})
    checkpoint_diagnostic_count = sum(row["Q"] >= 0 and row["H_A"] < 0 for row in rows)

    aulc_rows = []
    for seed in seeds:
        for cell, (origin, _) in experiment.CELLS.items():
            points = [(512, math.log(index[(seed, "prefix", origin, 512, 1)]["fid50k_full"]))]
            points += [(budget, math.log(index[(seed, "suffix", cell, budget, 1)]["fid50k_full"]))
                       for budget in (640, 768, 896, 1024)]
            area = sum((x1 - x0) * (y0 + y1) / 2
                       for (x0, y0), (x1, y1) in zip(points, points[1:])) / (1024 - 512)
            aulc_rows.append({"seed": seed, "cell": cell, "normalized_log_fid_aulc_512_1024": area})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed_path = args.output_dir / "H_C_I_Q_G_per_seed.csv"
    with per_seed_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    raw_path = args.output_dir / "decoded_evaluation_results.csv"
    with raw_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decoded["results"][0]))
        writer.writeheader(); writer.writerows(decoded["results"])
    aulc_path = args.output_dir / "AULC_diagnostic.csv"
    with aulc_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aulc_rows[0]))
        writer.writeheader(); writer.writerows(aulc_rows)
    analysis = {
        "schema": "ect.q256.fresh-crossed-switch-statistics/v1", "status": "PASS",
        "protocol_sha256": experiment.sha256_file(protocol_path),
        "eleven_seed_authorization_sha256": authorization_sha,
        "evaluation_recovery_authorization_sha256": recovery_authorization_sha,
        "postseal_report_recovery_authorization_sha256": postseal_authorization_sha,
        "included_seeds": list(seeds),
        "excluded_seed": (experiment.ELEVEN_SEED_EXCLUSION if authorization_sha else None),
        "original_n12_claim_abandoned": authorization_sha is not None,
        "analysis_population": ("AUTHOR_AMENDED_N11_COMPLETE_CASE"
                                if authorization_sha else "FROZEN_N12"),
        "decoded_results_sha256": experiment.sha256_file(decoded_path), "delta_log_1p03": DELTA,
        "primary_outcome": "H from log FID50k at 1024 kimg, NFE1 only",
        "summaries": summaries, "primary_verdict": verdict,
        "primary_category_checks": checks,
        "interaction_90pct_ci_inside_3pct_equivalence_band": interaction_equivalent,
        "interaction_claim": ("NO_INTERACTION_EXCEEDING_3_PERCENT_SUPPORTED"
                              if interaction_equivalent else "THREE_PERCENT_INTERACTION_NULL_NOT_ESTABLISHED"),
        "checkpoint_quality_diagnostic": {
            "count_Q_nonnegative_and_H_A_negative": checkpoint_diagnostic_count,
            "no_posthoc_subgroup_significance_test": True,
        },
        "formal_secondary_family": "two-sided exact sign-flip tests for C and I",
        "formal_secondary_holm": secondary_holm,
        "descriptive_only_cannot_change_primary": ["NFE2", "KID", "640/768/896", "AULC", "BA single cell", "interaction"],
        "per_seed_csv_sha256": experiment.sha256_file(per_seed_path),
        "raw_csv_sha256": experiment.sha256_file(raw_path),
        "aulc_csv_sha256": experiment.sha256_file(aulc_path),
    }
    experiment.atomic_json(args.output_dir / "analysis.json", analysis)
    experiment.atomic_json(args.output_dir / "primary_decision.json", {
        "schema": "ect.q256.fresh-crossed-switch-primary-decision/v1", "status": "PASS",
        "decision": verdict, "H_summary": summaries["H"], "delta_log_1p03": DELTA,
        "category_checks": checks, "primary_only": True,
        "included_seeds": list(seeds),
        "eleven_seed_authorization_sha256": authorization_sha,
        "evaluation_recovery_authorization_sha256": recovery_authorization_sha,
        "postseal_report_recovery_authorization_sha256": postseal_authorization_sha,
        "original_n12_claim_abandoned": authorization_sha is not None,
    })
    print(json.dumps({"status": "PASS", "primary_verdict": verdict}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
