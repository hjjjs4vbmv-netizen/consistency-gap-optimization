"""Seed-level analysis of the prespecified common four-path set."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
from scipy import stats
from .protocol import SEEDS, READOUT_BLOCKS, evaluation_slots, import_old_controls, write_csv

PATHS = ("AA", "BA", "CA", "DA")


def interval(values):
    n = len(values)
    mean = statistics.mean(values) if n else None
    result = {"n": n, "mean": mean, "sd": None, "ci95_nominal": None, "p_raw": None,
              "negative": sum(x < 0 for x in values), "positive": sum(x > 0 for x in values),
              "zero": sum(x == 0 for x in values)}
    if n < 2:
        return dict(result, status="INSUFFICIENT_PAIRS")
    sd = statistics.stdev(values)
    result["sd"] = sd
    if sd == 0:
        return dict(result, status="ZERO_VARIANCE")
    sem = sd / math.sqrt(n)
    half = float(stats.t.ppf(.975, n-1)) * sem
    return dict(result, status="ESTIMATED", ci95_nominal=[mean-half, mean+half],
                p_raw=float(2 * stats.t.sf(abs(mean / sem), n-1)))


def holm_three(pvalues):
    if len(pvalues) != 3:
        raise ValueError("Holm family is exactly the three primary contrasts")
    result = [None] * 3
    previous = 0.0
    for rank, (index, value) in enumerate(sorted(enumerate(pvalues), key=lambda pair: 1 if pair[1] is None else pair[1])):
        if value is None:
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid p-value")
        previous = max(previous, min(1.0, (3-rank)*value))
        result[index] = previous
    return result


def validate_rows(rows):
    expected = {(s,p,r,b) for s in SEEDS for p in PATHS for r, blocks in READOUT_BLOCKS.items() for b in blocks}
    mapping = {}
    for row in rows:
        key = (int(row["seed"]), row["path"], row["readout"], row["block"])
        if key not in expected or key in mapping:
            raise ValueError("noncanonical or duplicate quality slot")
        if row["source"] != ("M1_original" if row["path"] in {"AA", "BA"} else "new"):
            raise ValueError("source population mismatch")
        if row["status"] == "PASS":
            if row.get("FID") is None or not math.isfinite(float(row["FID"])) or float(row["FID"]) <= 0:
                raise ValueError("PASS requires a finite positive FID")
            if row.get("KID") is None or not math.isfinite(float(row["KID"])):
                raise ValueError("PASS requires finite KID")
        elif row["status"] == "NO_ENDPOINT":
            if row.get("FID") is not None or row.get("KID") is not None:
                raise ValueError("missing endpoints must never be imputed")
        mapping[key] = row
    if set(mapping) != expected:
        raise ValueError("expected all 320 planned slots, including pending/missing")
    for old in import_old_controls():
        row = mapping[(old["seed"], old["path"], old["readout"], old["block"])]
        if any(row.get(k) != old.get(k) for k in ("status", "FID", "KID", "receipt", "source_commit")):
            raise ValueError("old K values/provenance must remain unchanged")
    return mapping


def summarize(rows):
    data = validate_rows(rows)
    pending = [r["job_id"] for r in rows if r["status"] not in {"PASS", "NO_ENDPOINT"}]
    eligible = [s for s in SEEDS if all(data[(s,p,"E_512",b)]["status"] == "PASS" for p in PATHS for b in ("B0","B1","B2"))]
    result = {"status": "INCOMPLETE_TECHNICAL" if pending else "COMPLETE",
              "pending_slots": pending, "complete_four_path_seeds": eligible, "n": len(eligible),
              "qualification": "conditional on complete original four-path training and all primary blocks",
              "cohort": "prespecified component study reusing previously observed A/B controls",
              "primary": {}, "per_seed": [], "descriptive": {}}
    if any(r["status"] == "BUDGET_PAUSED" for r in rows):
        result["status"] = "INCOMPLETE_BUDGET"
    # Do not issue inferential completion on a selectively completed subqueue.
    if pending:
        return result
    for s in eligible:
        y = {p: statistics.mean(math.log(float(data[(s,p,"E_512",b)]["FID"])) for b in ("B0","B1","B2")) for p in PATHS}
        ht, hw = y["CA"]-y["AA"], y["DA"]-y["AA"]
        interaction = y["BA"]-y["CA"]-y["DA"]+y["AA"]
        joint = y["BA"]-y["AA"]
        if not math.isclose(ht+hw+interaction, joint, rel_tol=0, abs_tol=1e-12):
            raise ArithmeticError("seed-level algebra does not close")
        result["per_seed"].append({"seed":s, **y, "H_T":ht, "H_W":hw, "I":interaction,
                                   "H_J":joint, "CA-DA":y["CA"]-y["DA"],
                                   "BA-CA":y["BA"]-y["CA"], "BA-DA":y["BA"]-y["DA"]})
    for name in ("H_T", "H_W", "I"):
        estimate = interval([r[name] for r in result["per_seed"]])
        estimate["ratio_type"] = "ratio-of-ratios" if name == "I" else "geometric FID ratio"
        estimate["ratio"] = math.exp(estimate["mean"]) if estimate["mean"] is not None else None
        estimate["ratio_ci95_nominal"] = ([math.exp(v) for v in estimate["ci95_nominal"]]
                                           if estimate["ci95_nominal"] is not None else None)
        result["primary"][name] = estimate
    adjusted = holm_three([result["primary"][n]["p_raw"] for n in ("H_T", "H_W", "I")])
    for name, p in zip(("H_T", "H_W", "I"), adjusted):
        result["primary"][name]["p_holm"] = p
        result["primary"][name]["reject_holm_0.05"] = p < .05 if p is not None else None
    for name in ("H_J", "CA-DA", "BA-CA", "BA-DA"):
        result["descriptive"][name] = interval([r[name] for r in result["per_seed"]])
    result["algebraic_closure"] = "PASS (identity only; not mechanism evidence)"
    result["absolute_quality"] = {}
    for p in PATHS:
        for readout, blocks in READOUT_BLOCKS.items():
            seeds = [s for s in SEEDS if all(data[(s,p,readout,b)]["status"] == "PASS" for b in blocks)]
            per_seed = {s: statistics.mean(float(data[(s,p,readout,b)]["FID"]) for b in blocks) for s in seeds}
            result["absolute_quality"][p+"/"+readout] = {
                "seed_FID_arithmetic_mean": per_seed, "n":len(seeds),
                "mean":statistics.mean(per_seed.values()) if seeds else None,
                "sd":statistics.stdev(per_seed.values()) if len(seeds)>1 else None}
    for readout, blocks in READOUT_BLOCKS.items():
        for metric in ("FID", "KID"):
            for left, right in (("CA","AA"), ("DA","AA"), ("BA","AA"), ("CA","DA"), ("BA","CA"), ("BA","DA")):
                seeds = [s for s in SEEDS if all(data[(s,p,readout,b)]["status"] == "PASS" for p in (left,right) for b in blocks)]
                transform = math.log if metric == "FID" else float
                diffs = [statistics.mean(transform(float(data[(s,left,readout,b)][metric])) - transform(float(data[(s,right,readout,b)][metric])) for b in blocks) for s in seeds]
                result["descriptive"][f"available_pairs/{readout}/{metric}/{left}-{right}"] = {**interval(diffs), "seeds":seeds, "values":diffs}
    # All three readouts use B0 for horizontal comparisons.
    result["readout_B0"] = [{"seed":s, "path":p, "values":{r:data[(s,p,r,"B0")]["FID"] for r in READOUT_BLOCKS}}
                            for s in SEEDS for p in PATHS]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    result = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    write_csv(args.output/"quality_slots_320.csv", rows)
    if result["per_seed"]:
        write_csv(args.output/"per_seed_contrasts.csv", result["per_seed"])
    print(result["status"], "n =", result["n"])


if __name__ == "__main__":
    main()
