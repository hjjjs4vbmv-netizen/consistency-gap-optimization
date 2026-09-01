#!/usr/bin/env python3
"""Unseal only after 60/60 PASS and apply the frozen seed-level decision rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


DELTA = math.log(1.03)
T975_DF9 = 2.2621571627409915
T950_DF9 = 1.8331129326536335


def read_metric(path: Path, metric: str) -> float:
    lines = [line for line in path.read_text().splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"invalid metric file: {path}")
    value = float(json.loads(lines[0])["results"][metric])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid positive metric: {path}")
    return value


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        candidate = abs(statistics.mean(v * s for v, s in zip(values, signs)))
        hits += candidate >= observed - 1e-15
    return hits / (2 ** len(values))


def summarize(name: str, values: list[float]) -> dict:
    if len(values) != 10:
        raise RuntimeError(f"{name} does not contain exactly 10 seeds")
    mean = statistics.mean(values)
    median = statistics.median(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(10)
    ci95 = [mean - T975_DF9 * se, mean + T975_DF9 * se]
    ci90 = [mean - T950_DF9 * se, mean + T950_DF9 * se]
    direction = 1 if mean > 0 else -1 if mean < 0 else 0
    same_direction = sum(
        1 for value in values if (value > 0) - (value < 0) == direction
    )
    loo = [statistics.mean(values[:i] + values[i + 1:]) for i in range(10)]
    loo_stable = direction != 0 and all(
        ((value > 0) - (value < 0)) == direction for value in loo
    )
    ci_material = ci95[0] > DELTA or ci95[1] < -DELTA
    material = ci_material and same_direction >= 8 and loo_stable
    equivalent = ci90[0] >= -DELTA and ci90[1] <= DELTA
    classification = "MATERIAL" if material else "EQUIVALENT" if equivalent else "UNRESOLVED"
    return {
        "contrast": name,
        "n_training_seeds": 10,
        "mean": mean,
        "median": median,
        "sample_sd": sd,
        "ci95_low": ci95[0], "ci95_high": ci95[1],
        "ci90_low": ci90[0], "ci90_high": ci90[1],
        "delta_log_1p03": DELTA,
        "exact_two_sided_sign_flip_p": exact_sign_flip(values),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "same_direction_count": same_direction,
        "leave_one_seed_out_means": loo,
        "leave_one_seed_out_sign_stable": loo_stable,
        "material": material,
        "equivalent_tost_alpha_0p05": equivalent,
        "classification": classification,
        "values": values,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seal-audit", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    seal_path = args.seal_audit.resolve(strict=True)
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "ALL_60_SEALED_PASS" or seal.get("unseal_authorized") is not True:
        raise RuntimeError("numeric unseal is not authorized")
    if seal.get("manifest_sha256") != pulse_chase.sha256_file(manifest_path):
        raise RuntimeError("seal audit/manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol_sha = pulse_chase.sha256_file(args.protocol.resolve(strict=True))
    if manifest.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("evaluation/protocol mismatch")
    receipt_paths = sorted(args.receipts.resolve(strict=True).glob("*.json"))
    receipts = {json.loads(path.read_text())["job_index"]: json.loads(path.read_text())
                for path in receipt_paths}
    if set(receipts) != set(range(60)):
        raise RuntimeError("sealed receipt matrix changed before unseal")
    raw = []
    for job in manifest["jobs"]:
        receipt = receipts[job["job_index"]]
        if receipt.get("status") != "SEALED_PASS":
            raise RuntimeError("non-PASS receipt at unseal")
        job_dir = Path(receipt["job_dir"])
        values = {
            metric: read_metric(job_dir / f"metric-{metric}.jsonl", metric)
            for metric in ("kid50k_full", "fid50k_full")
        }
        raw.append({
            "job_index": job["job_index"], "seed": job["seed"],
            "branch": job["branch"], "budget_kimg": job["budget_kimg"],
            "nfe": job["nfe"], "mid_t": "" if job["nfe"] == 1 else 0.821,
            "fid50k_full": values["fid50k_full"],
            "kid50k_full": values["kid50k_full"],
            "generated_feature_sha256": receipt["generated_feature_sha256"],
            "receipt_sha256": pulse_chase.sha256_file(receipt_paths[job["job_index"]]),
        })
    indexed = {(row["seed"], row["branch"], row["budget_kimg"], row["nfe"]): row
               for row in raw}
    contrasts = []
    for seed in pulse_chase.SEEDS:
        y = {}
        for branch in pulse_chase.BRANCHES:
            for budget in (512, 640):
                fid = indexed[(seed, branch, budget, 1)]["fid50k_full"]
                y[(branch, budget)] = math.log(fid)
        d512 = y[("Late-switch", 512)] - y[("Early-switch", 512)]
        d640 = y[("Late-switch", 640)] - y[("Early-switch", 640)]
        contrasts.append({"seed": seed, "D512": d512, "D640": d640,
                          "J": d640 - d512})
    summaries = {
        name: summarize(name, [row[name] for row in contrasts])
        for name in ("D512", "D640", "J")
    }
    d512, d640, jump = summaries["D512"], summaries["D640"], summaries["J"]
    primary = (
        "PRIMARY_SUCCESS" if d640["material"]
        else "INFORMATIVE_NULL" if d640["equivalent_tost_alpha_0p05"]
        else "INCONCLUSIVE"
    )
    ordinary = (
        d512["material"] and d640["material"]
        and math.copysign(1, d512["mean"]) == math.copysign(1, d640["mean"])
        and jump["equivalent_tost_alpha_0p05"]
    )
    delayed = d512["equivalent_tost_alpha_0p05"] and d640["material"]
    chase_modified = d640["material"] and jump["material"]
    amplification = (
        chase_modified
        and math.copysign(1, jump["mean"]) == math.copysign(1, d640["mean"])
        and abs(d640["mean"]) > abs(d512["mean"])
    )
    mechanism = {
        "ordinary_checkpoint_quality_carryover": ordinary,
        "delayed_or_emergent_carryover": delayed,
        "chase_modified_contrast": chase_modified,
        "amplification_permitted": amplification,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "raw_per_seed.csv", raw)
    write_csv(output / "D512_D640_J.csv", contrasts)
    reproducibility.atomic_json_dump(
        {
            "schema": "ect.q256.p2-frozen-analysis/v1",
            "status": "PASS",
            "protocol_sha256": protocol_sha,
            "independent_unit": "training_seed",
            "n_independent_units": 10,
            "primary_decision": primary,
            "summaries": summaries,
            "mechanism": mechanism,
            "secondary_cannot_rescue_primary": True,
            "claim_ceiling": (
                "full-state effect of temporary spacing history on common-A "
                "finite-horizon quality evolution; no history×current interaction "
                "and no optimizer/EMA mediation claim"
            ),
        },
        output / "analysis.json",
        overwrite=False,
    )
    report = [
        "# q256 P2 B@384 pulse/chase report", "",
        f"Primary decision: **{primary}**", "",
        f"- D640: {d640['classification']}",
        f"- D512: {d512['classification']}",
        f"- J: {jump['classification']}", "",
        f"Ordinary checkpoint-quality carryover: **{ordinary}**",
        f"Delayed/emergent carryover: **{delayed}**",
        f"Chase-modified contrast: **{chase_modified}**",
        f"Amplification wording permitted: **{amplification}**", "",
        "All decisions use the ten training seeds as the only independent units. "
        "NFE2 and KID are secondary and cannot rescue the NFE1 D640 primary.", "",
        "Claim ceiling: the experiment identifies the full-state effect of temporary "
        "spacing history on common-A finite-horizon quality evolution. It does not "
        "identify a history×current interaction or optimizer/EMA mediation.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file()]
    sums = "".join(f"{pulse_chase.sha256_file(path)}  {path.name}\n"
                   for path in sorted(files))
    (output / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    print(json.dumps({"status": "PASS", "primary_decision": primary,
                      "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
