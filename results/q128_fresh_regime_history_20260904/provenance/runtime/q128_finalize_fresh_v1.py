#!/usr/bin/env python3
"""Fail-closed finalizer for q128 fresh regime/history study.

The script writes and fsyncs the pre-decode gate and deviation adjudication
before decrypting any scalar quality result. Sealed inputs are decrypted only
in memory. Frozen manifests and job receipts are never modified.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("/root/q128_fresh_regime_history_n8_v1")
REPO = Path("/root/consistency-gap-optimization")
EVAL = ROOT / "evaluation"
OUT = ROOT / "final_analysis"
FINALIZER = Path(__file__).resolve()
BASE_MANIFEST = EVAL / "control/bound_manifest_completed7.json"
EXT_MANIFEST = EVAL / "replacement209/control/replacement209_evaluation_extension.json"
ANALYSIS_PLAN = REPO / "analysis/q128_fresh_regime_history_n8_v1/analysis_plan.json"
PROTOCOL = REPO / "analysis/q128_fresh_regime_history_n8_v1/protocol.json"
PROTOCOL_SHA = "e908da54c47d2f3faa9dd699f7d1a345446fdb11a71ad32642ec74197f59b3bf"
FROZEN_MANIFEST_SHA = "34f296c1eac0042bb6d37cd692c3a1ab008ef87caa7417b4ce9f0f33dde6e17e"
CT_EVAL_SHA = "938941b612bd766fbf552e84d4e127daedb19594b39a5603b7c735d89d47d325"
SOURCE_COMMIT = "5abd4bd074f6987110f29a0adb93e24e842450bd"
DATASET_SHA = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
SEEDS = [201, 202, 203, 204, 206, 207, 208, 209]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_new_json(path: Path, obj: object) -> None:
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    data = canonical_bytes(obj)
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def load(path: Path):
    return json.loads(path.read_text())


def check(cond: bool, message: str, failures: list[str]) -> None:
    if not cond:
        failures.append(message)


def predecode_gate() -> tuple[list[dict], dict]:
    base, ext = load(BASE_MANIFEST), load(EXT_MANIFEST)
    jobs = base["jobs"] + ext["jobs"]
    failures: list[str] = []
    check(sha256(PROTOCOL) == PROTOCOL_SHA, "protocol SHA mismatch", failures)
    check(base.get("frozen_manifest_sha256") == FROZEN_MANIFEST_SHA, "base frozen manifest SHA mismatch", failures)
    check(ext.get("frozen_manifest_sha256") == FROZEN_MANIFEST_SHA, "extension frozen manifest SHA mismatch", failures)
    check(len(base["jobs"]) == 238 and len(ext["jobs"]) == 34 and len(jobs) == 272, "job count mismatch", failures)
    check(ext.get("effective_cohort") == SEEDS, "effective cohort mismatch", failures)
    check(ext.get("old_seed205_jobs_excluded") is True, "seed205 exclusion not frozen", failures)
    ids = [j["opaque_id"] for j in jobs]
    check(len(set(ids)) == 272, "opaque IDs are not unique", failures)
    seed_counts = Counter(j["seed"] for j in jobs)
    check(seed_counts == Counter({s: 34 for s in SEEDS}), f"per-seed jobs mismatch: {seed_counts}", failures)
    check(not any(j["seed"] == 205 for j in jobs), "failed seed205 appears in effective manifest", failures)
    category_counts = Counter(j["category"] for j in jobs)
    check(category_counts == Counter({"PRIMARY": 48, "KEY_SECONDARY": 32, "DESCRIPTIVE": 192}),
          f"category counts mismatch: {category_counts}", failures)
    base_sha, ext_sha = sha256(BASE_MANIFEST), sha256(EXT_MANIFEST)
    sealed_checked = 0
    artifact_size_checked = 0
    receipt_shas = {}
    for job in jobs:
        receipt_dir = EVAL / ("replacement209/receipts" if job["seed"] == 209 else "receipts")
        rp = receipt_dir / f"{job['opaque_id']}.json"
        check(rp.is_file(), f"missing receipt {job['opaque_id']}", failures)
        if not rp.is_file():
            continue
        r = load(rp)
        receipt_shas[job["opaque_id"]] = sha256(rp)
        expected_bound = ext_sha if job["seed"] == 209 else base_sha
        for key, expected in [
            ("status", "SEALED_PASS"), ("opaque_id", job["opaque_id"]),
            ("category", job["category"]), ("checkpoint_sha256", job["checkpoint_sha256"]),
            ("ct_eval_sha256", CT_EVAL_SHA), ("evaluator_commit", SOURCE_COMMIT),
            ("dataset_sha256", DATASET_SHA), ("precision", "fp32"),
            ("sample_count", 50000), ("sample_seed_range", "0-49999"),
            ("metric_seed", 20260730), ("nfe", job["nfe"]),
            ("bound_manifest_sha256", expected_bound),
        ]:
            check(r.get(key) == expected, f"receipt {job['opaque_id']} {key} mismatch", failures)
        check(r.get("quality_values_decoded") is False, f"receipt {job['opaque_id']} was decoded early", failures)
        check(r.get("quality_values_in_receipt") is False, f"receipt {job['opaque_id']} discloses quality", failures)
        check(r.get("kid_fid_shared_feature_identity") is True, f"receipt {job['opaque_id']} feature identity failed", failures)
        check(r.get("generated_feature_sha256") == r.get("artifacts", {}).get("generated-features-fid50k_full-repeat00.npy", {}).get("sha256"),
              f"receipt {job['opaque_id']} generated feature hash mismatch", failures)
        check(r.get("artifacts", {}).get("generated-features-fid50k_full-repeat00.npy", {}).get("sha256") ==
              r.get("artifacts", {}).get("generated-features-kid50k_full-repeat00.npy", {}).get("sha256"),
              f"receipt {job['opaque_id']} KID/FID features differ", failures)
        sealed = r.get("sealed_scalar_artifacts", [])
        names = {Path(a["sealed_path"]).name for a in sealed}
        check("metric-fid50k_full.jsonl.sealed" in names and "metric-kid50k_full.jsonl.sealed" in names,
              f"receipt {job['opaque_id']} lacks sealed metrics", failures)
        for a in sealed:
            p = Path(a["sealed_path"])
            check(p.is_file(), f"missing sealed artifact {p}", failures)
            if p.is_file():
                check(p.stat().st_size == a["bytes"], f"sealed artifact size mismatch {p}", failures)
                check(sha256(p) == a["sha256"], f"sealed artifact SHA mismatch {p}", failures)
                sealed_checked += 1
        job_dir = EVAL / ("replacement209/jobs" if job["seed"] == 209 else "jobs") / job["opaque_id"]
        for name, meta in r.get("artifacts", {}).items():
            p = job_dir / name
            check(p.is_file(), f"missing generated artifact {p}", failures)
            if p.is_file():
                check(p.stat().st_size == meta["bytes"], f"generated artifact size mismatch {p}", failures)
                artifact_size_checked += 1
    launch7 = load(EVAL / "control/rolling_launch_receipt.json")
    launch209 = load(EVAL / "replacement209/control/launch_receipt.json")
    check(launch7.get("status") == "SEALED_PASS_COMPLETED7" and launch7.get("sealed_pass_count") == 238,
          "seven-seed launch receipt not complete", failures)
    check(launch209.get("status") == "SEALED_PASS_SEED209" and launch209.get("sealed_pass_count") == 34,
          "seed209 launch receipt not complete", failures)
    completed = subprocess.run(["pgrep", "-af", "q128_.*opaque_eval|ct_eval.py"], text=True, capture_output=True)
    active_lines = [x for x in completed.stdout.splitlines() if "pgrep -af" not in x]
    check(not active_lines, f"evaluation process still active: {active_lines}", failures)
    report = {
        "schema": "ect.q128-fresh-predecode-gate/v1",
        "created_utc": utc_now(),
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "quality_values_observed_before_gate": False,
        "effective_cohort": SEEDS,
        "failed_seed_excluded": 205,
        "job_counts": {"total": len(jobs), "by_category": dict(category_counts), "by_seed": dict(sorted(seed_counts.items()))},
        "all_primary_and_key_secondary_sealed_pass": not failures,
        "all_jobs_sealed_pass": not failures,
        "sealed_scalar_artifacts_sha256_rechecked": sealed_checked,
        "large_generated_artifacts_existence_and_size_rechecked": artifact_size_checked,
        "large_generated_artifact_content_hashes": "validated during each SEALED_PASS job and bound in public receipt; not redundantly rehashed at decode gate",
        "manifest_sha256": {"base_238": base_sha, "replacement_34": ext_sha, "frozen_272_template": FROZEN_MANIFEST_SHA},
        "protocol_sha256": PROTOCOL_SHA,
        "ct_eval_sha256": CT_EVAL_SHA,
        "evaluator_commit": SOURCE_COMMIT,
        "finalizer_path": str(FINALIZER),
        "finalizer_sha256": sha256(FINALIZER),
        "receipt_sha256": receipt_shas,
        "failures": failures,
    }
    return jobs, report


def deviation_adjudication() -> dict:
    base = load(EVAL / "control/pre_quality_integrity_deviation_record.json")
    rep = load(EVAL / "replacement209/control/pre_quality_integrity_deviation_record.json")
    events = []
    for e in base["events"]:
        events.append({**e, "trajectory": e.get("trajectory", e.get("trajectory_dir"))})
    for e in rep["events"]:
        events.append({**e, "seed": 209})
    by_seed_traj = Counter((e["seed"], e["trajectory"]) for e in events)
    scientific_state_nonfinite = sum(e.get(k, 0) for e in events for k in
                                     ("loss_nonfinite_count", "model_nonfinite_count", "ema_nonfinite_count", "update_nonfinite_count"))
    return {
        "schema": "ect.q128-fresh-protocol-deviation-adjudication/v1",
        "created_utc": utc_now(),
        "status": "ADJUDICATED_BEFORE_DECODE",
        "quality_values_observed": False,
        "issue": "AMP GradScaler skips occurred after the frozen 10000-nimg exclusive warm-up bound",
        "frozen_contract_evidence": {
            "source": "scripts/run_q128_fresh_regime_history_n8_v1.py:telemetry_gate",
            "rule": "processed_nimg >= 10000 on a skipped step is a failure",
            "regression_test": "test_late_amp_skip_fails_closed",
        },
        "late_amp_skip_event_count": len(events),
        "by_seed_trajectory": {f"seed{s}/{t}": n for (s, t), n in sorted(by_seed_traj.items())},
        "loss_model_ema_update_nonfinite_count": scientific_state_nonfinite,
        "state_on_skipped_attempts": "loss/model/EMA/update remained finite and skipped attempts had zero parameter update",
        "governance_conclusion": "CONFIRMATORY_IDENTITY_FAILED_CLOSED",
        "analysis_disposition": "Run the frozen statistical analysis and report its planned verdicts, but label them protocol-deviated fresh-cohort results; do not claim a clean confirmatory replication.",
        "rationale": [
            "The late-skip rule was frozen before formal outcomes and explicitly tested as fail-closed.",
            "The orchestrator completion receipts did not invoke the full telemetry gate, so PASS completion does not cure the violation.",
            "The deviation and this adjudication were recorded before any formal scalar quality value was decoded.",
            "No outcome-dependent seed, arm, metric, margin, NFE, budget, or analysis change is permitted.",
        ],
    }


def decrypt_jsonl(path: Path) -> list[dict]:
    proc = subprocess.run([
        "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
        "-pass", f"file:{EVAL / 'control/decode.key'}", "-in", str(path),
    ], capture_output=True, check=True)
    out = []
    for line in proc.stdout.decode("utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    if not out:
        raise RuntimeError(f"empty decoded JSONL: {path}")
    return out


def extract_metric(objs: list[dict], metric: str) -> float:
    found = []
    def walk(x):
        if isinstance(x, dict):
            if metric in x and isinstance(x[metric], (int, float)):
                found.append(float(x[metric]))
            if x.get("metric") == metric and isinstance(x.get("value"), (int, float)):
                found.append(float(x["value"]))
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(objs)
    vals = sorted(set(found))
    if len(vals) != 1 or not math.isfinite(vals[0]):
        raise RuntimeError(f"expected one finite {metric}, found {vals}")
    return vals[0]


def decode_records(jobs: list[dict]) -> list[dict]:
    records = []
    for j in sorted(jobs, key=lambda x: (x["seed"], x["job_index"], x["opaque_id"])):
        job_dir = EVAL / ("replacement209/jobs" if j["seed"] == 209 else "jobs") / j["opaque_id"]
        fid = extract_metric(decrypt_jsonl(job_dir / "metric-fid50k_full.jsonl.sealed"), "fid50k_full")
        kid = extract_metric(decrypt_jsonl(job_dir / "metric-kid50k_full.jsonl.sealed"), "kid50k_full")
        if fid <= 0:
            raise RuntimeError(f"non-positive FID for {j['opaque_id']}")
        records.append({
            "opaque_id": j["opaque_id"], "seed": j["seed"], "trajectory": j["trajectory"],
            "budget_kimg": j["budget_kimg"], "nfe": j["nfe"], "mid_t": j.get("mid_t"),
            "category": j["category"], "analysis_roles": j.get("analysis_roles", []),
            "fid50k": fid, "kid50k": kid,
        })
    return records


def ci_summary(values: list[float], alpha: float = 0.05) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x); mean = float(np.mean(x)); sd = float(np.std(x, ddof=1)); se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        tstat = math.copysign(math.inf, mean) if mean else 0.0
        pneg = 0.0 if mean < 0 else (1.0 if mean > 0 else 0.5)
        ppos = 1.0 - pneg
    else:
        tstat = mean / se
        pneg = float(stats.t.cdf(tstat, df))
        ppos = float(stats.t.sf(tstat, df))
    q975 = float(stats.t.ppf(1 - alpha / 2, df)); q95 = float(stats.t.ppf(1 - alpha, df))
    obs = mean
    flipped = [statistics.mean(v * s for v, s in zip(values, signs))
               for signs in itertools.product((-1, 1), repeat=n)]
    return {
        "n": n, "mean": mean, "median": float(np.median(x)), "sd": sd, "se": se,
        "t_statistic": float(tstat), "df": df,
        "two_sided_95_ci": [mean - q975 * se, mean + q975 * se],
        "one_sided_95_upper_bound": mean + q95 * se,
        "one_sided_p_negative": pneg, "one_sided_p_positive": ppos,
        "negative_count": int(np.sum(x < 0)), "positive_count": int(np.sum(x > 0)),
        "zero_count": int(np.sum(x == 0)),
        "exact_one_sided_signflip_p_negative": sum(v <= obs + 1e-15 for v in flipped) / len(flipped),
        "exact_one_sided_signflip_p_positive": sum(v >= obs - 1e-15 for v in flipped) / len(flipped),
        "loso_means": [float(np.mean(np.delete(x, i))) for i in range(n)],
    }


def analyze(records: list[dict]) -> dict:
    idx = {(r["seed"], r["trajectory"], r["budget_kimg"], r["nfe"]): r for r in records}
    def logfid(s, t, b, n): return math.log(idx[(s, t, b, n)]["fid50k"])
    points = []
    for s in SEEDS:
        aa = logfid(s, "A", 1024, 1); bb = logfid(s, "Bsame", 1024, 1)
        ab = logfid(s, "AB", 1024, 1); ba = logfid(s, "BA", 1024, 1)
        ha = ba - aa; hb = bb - ab; sa = ab - aa; sb = bb - ba
        p = (bb - aa) - (logfid(s, "Bsame", 512, 1) - logfid(s, "A", 512, 1))
        r = ((logfid(s, "Cmatch", 1024, 2) - logfid(s, "Bmatch", 1024, 2)) -
             (logfid(s, "Cmatch", 1024, 1) - logfid(s, "Bmatch", 1024, 1)))
        points.append({"seed": s, "H_A": ha, "H_B": hb, "S_A": sa, "S_B": sb,
                       "I": bb - ba - ab + aa, "P": p, "R": r})
    by = {name: [p[name] for p in points] for name in ("H_A", "H_B", "S_A", "S_B", "I", "P", "R")}
    summaries = {name: ci_summary(vals) for name, vals in by.items()}
    ha = summaries["H_A"]
    directional = ("NEGATIVE_DIRECTION_SUPPORTED" if ha["mean"] < 0 and ha["one_sided_p_negative"] < 0.05
                   else "POSITIVE_DIRECTION_SUPPORTED" if ha["mean"] > 0 and ha["one_sided_p_positive"] < 0.05
                   else "DIRECTION_UNRESOLVED")
    band = math.log(1.03); mean = ha["mean"]; se = ha["se"]; df = ha["df"]
    if se == 0:
        p_lower = 0.0 if mean > -band else 1.0
        p_upper = 0.0 if mean < band else 1.0
    else:
        p_lower = float(stats.t.sf((mean + band) / se, df))
        p_upper = float(stats.t.cdf((mean - band) / se, df))
    q90 = float(stats.t.ppf(0.95, df)); ci90 = [mean - q90 * se, mean + q90 * se]
    practical = "PRACTICALLY_EQUIVALENT" if p_lower < .05 and p_upper < .05 else "PRACTICAL_MAGNITUDE_UNRESOLVED"
    material = ha["one_sided_95_upper_bound"] < -band
    phase = summaries["P"]
    phase_verdict = "NEGATIVE_DIRECTION_SUPPORTED" if phase["mean"] < 0 and phase["one_sided_p_negative"] < .05 else "DIRECTION_UNRESOLVED"
    cell_groups = defaultdict(list)
    for r in records:
        cell_groups[(r["trajectory"], r["budget_kimg"], r["nfe"])].append(r)
    cell_summary = []
    for (traj, budget, nfe), rs in sorted(cell_groups.items()):
        fids = [r["fid50k"] for r in rs]; kids = [r["kid50k"] for r in rs]
        cell_summary.append({"trajectory": traj, "budget_kimg": budget, "nfe": nfe, "n": len(rs),
                             "fid_arithmetic_mean": float(np.mean(fids)), "fid_sd": float(np.std(fids, ddof=1)),
                             "fid_geometric_mean": math.exp(float(np.mean(np.log(fids)))),
                             "kid_arithmetic_mean": float(np.mean(kids)), "kid_sd": float(np.std(kids, ddof=1))})
    return {
        "schema": "ect.q128-fresh-statistical-results/v1", "created_utc": utc_now(),
        "effective_cohort": SEEDS, "scale": "natural log FID50k unless otherwise stated",
        "seed_points": points,
        "sole_primary": {"estimand": "H_A = logFID(BA@1024,NFE1)-logFID(AA@1024,NFE1)",
                         "summary": ha, "directional_verdict": directional},
        "practical_magnitude": {"band": [-band, band], "tost_alpha": .05, "p_lower": p_lower,
                                "p_upper": p_upper, "two_sided_90_ci": ci90, "verdict": practical,
                                "material_negative_supported": material},
        "key_secondary_phase": {"estimand": "P=D(1024)-D(512)", "summary": phase,
                                 "directional_verdict": phase_verdict},
        "key_secondary_nfe": {"estimand": "R=(Cmatch-Bmatch)_NFE2-(Cmatch-Bmatch)_NFE1 at 1024",
                              "summary": summaries["R"],
                              "allowed_interpretation": "Bmatch/Cmatch diagnostic trajectory ranking is or is not NFE-dependent"},
        "crossed_secondary": {k: summaries[k] for k in ("H_B", "S_A", "S_B", "I")},
        "interaction_equivalence": "NOT_TESTED: no independent interaction equivalence band was frozen",
        "cell_summaries": cell_summary,
    }


def fmt(x: float) -> str:
    return f"{x:.6g}"


def report_markdown(results: dict, adjudication: dict) -> str:
    h = results["sole_primary"]["summary"]; pm = results["practical_magnitude"]
    p = results["key_secondary_phase"]["summary"]; r = results["key_secondary_nfe"]["summary"]
    rows = []
    for q in results["seed_points"]:
        rows.append(f"| {q['seed']} | {fmt(q['H_A'])} | {fmt(q['H_B'])} | {fmt(q['S_A'])} | {fmt(q['S_B'])} | {fmt(q['I'])} | {fmt(q['P'])} | {fmt(q['R'])} |")
    return f"""# Fresh q128 Regime and History Generalization Study v1 — Final Report

Generated: {utc_now()}

## Governance headline

All 272 frozen evaluation jobs for the effective cohort passed the opaque sealing gate before unified decode. The frozen statistical analysis was executed without changing the cohort, metric, hypotheses, thresholds, NFE, budgets, or practical margin.

The scientific identity is **{adjudication['governance_conclusion']}**. There were {adjudication['late_amp_skip_event_count']} AMP GradScaler skips after the frozen 10,000-nimg warm-up bound. The frozen verifier explicitly classifies such events as fail-closed. Therefore the numerical results below are pre-specified, protocol-deviated fresh-cohort results, not a clean confirmatory replication. Loss, model, EMA, and applied-update states remained finite on the recorded skip events, but that does not waive the frozen rule.

## Sole primary: H_A

- Directional axis: **{results['sole_primary']['directional_verdict']}**
- Mean H_A: {fmt(h['mean'])}; median: {fmt(h['median'])}; SD: {fmt(h['sd'])}
- Two-sided 95% CI: [{fmt(h['two_sided_95_ci'][0])}, {fmt(h['two_sided_95_ci'][1])}]
- One-sided 95% upper bound: {fmt(h['one_sided_95_upper_bound'])}
- One-sided paired-t p for E[H_A]&lt;0: {fmt(h['one_sided_p_negative'])}
- Negative seeds: {h['negative_count']}/8; exact one-sided sign-flip p: {fmt(h['exact_one_sided_signflip_p_negative'])}
- LOSO means: {', '.join(fmt(x) for x in h['loso_means'])}

Directional evidence uses only the frozen one-sided paired t-test. The two-sided CI, sign count, sign-flip test, and LOSO results are robustness summaries and do not override that verdict.

## Independent practical-magnitude axis

- Verdict: **{pm['verdict']}**
- Equivalence band: ±log(1.03) = ±{fmt(pm['band'][1])}
- Two-sided 90% CI: [{fmt(pm['two_sided_90_ci'][0])}, {fmt(pm['two_sided_90_ci'][1])}]
- TOST p-values, lower/upper: {fmt(pm['p_lower'])}, {fmt(pm['p_upper'])}
- Material negative supported: **{str(pm['material_negative_supported']).upper()}**

## Key secondary results

Phase shift P: **{results['key_secondary_phase']['directional_verdict']}**; mean {fmt(p['mean'])}, 95% CI [{fmt(p['two_sided_95_ci'][0])}, {fmt(p['two_sided_95_ci'][1])}], one-sided p {fmt(p['one_sided_p_negative'])}, negative seeds {p['negative_count']}/8, exact sign-flip p {fmt(p['exact_one_sided_signflip_p_negative'])}.

NFE diagnostic R: mean {fmt(r['mean'])}, 95% CI [{fmt(r['two_sided_95_ci'][0])}, {fmt(r['two_sided_95_ci'][1])}], positive/negative seeds {r['positive_count']}/{r['negative_count']}. This supports only a statement about whether the Bmatch/Cmatch diagnostic trajectory ranking is NFE-dependent; it is not a causal channel attribution.

## Seed-level log-FID contrasts

| Seed | H_A | H_B | S_A | S_B | I | P | R |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

H_B, S_A, S_B, and I are secondary/descriptive and cannot rescue the primary result. No interaction-equivalence claim is made because no independent interaction equivalence band was frozen.

## Missingness and replacement

- Formal seeds started: 9 (201–208 plus replacement 209).
- All-started terminal failure count: 1 (seed205).
- Arm-specific terminal failure: seed205 Dmatch, 1 event, deterministically reproduced at attempted iteration 7951 / 1,017,728 processed images.
- B-history-specific terminal failure count: 0.
- Effective complete cohort: seeds 201–204 and 206–209 (n=8); seed209 replaced seed205 under the pre-sorted pool.
- The estimand is completion-conditioned. Replacement does not eliminate possible informative missingness.

## Separation from the old q128 n=3 study

Only the fresh n=8 cohort enters inference. Old q128 seeds3–5 are not pooled, do not contribute to any p-value, and cannot change either verdict axis. Any later side-by-side comparison must be marked DESCRIPTIVE EVIDENCE SYNTHESIS.

## Integrity

The decode gate verified 272 unique opaque IDs, all SEALED_PASS receipts, effective-cohort membership, evaluator/source/dataset hashes, category coverage, KID/FID shared-feature identity, and the hashes of all sealed scalar artifacts. Full per-cell decoded values and descriptive cell summaries are retained in the immutable analysis artifacts accompanying this report.
"""


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refuse existing final directory: {OUT}")
    OUT.mkdir(mode=0o700)
    jobs, gate = predecode_gate()
    write_new_json(OUT / "predecode_gate.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError("predecode gate failed closed")
    adjudication = deviation_adjudication()
    write_new_json(OUT / "protocol_deviation_adjudication.json", adjudication)
    # First scalar decode occurs only after the two preceding records are durable.
    decode_started = utc_now()
    records = decode_records(jobs)
    analysis_input = {
        "schema": "ect.q128-fresh-immutable-analysis-input/v1", "created_utc": utc_now(),
        "decode_started_utc": decode_started, "protocol_sha256": PROTOCOL_SHA,
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA, "effective_cohort": SEEDS,
        "failed_seed_excluded": 205, "old_n3_included": False, "records": records,
    }
    write_new_json(OUT / "analysis_input.json", analysis_input)
    results = analyze(records)
    write_new_json(OUT / "statistical_results.json", results)
    verdict = {
        "schema": "ect.q128-fresh-final-verdict/v1", "created_utc": utc_now(),
        "execution_verdict": "EVALUATION_COMPLETE_272_OF_272_SEALED_PASS",
        "confirmatory_valid": False,
        "confirmatory_governance_verdict": adjudication["governance_conclusion"],
        "planned_directional_verdict": results["sole_primary"]["directional_verdict"],
        "planned_practical_magnitude_verdict": results["practical_magnitude"]["verdict"],
        "material_negative_supported": results["practical_magnitude"]["material_negative_supported"],
        "key_secondary_phase_verdict": results["key_secondary_phase"]["directional_verdict"],
        "secondary_does_not_modify_primary": True,
        "old_n3_pooled": False,
    }
    write_new_json(OUT / "final_verdict.json", verdict)
    write_new_text(OUT / "FINAL_REPORT.md", report_markdown(results, adjudication))
    decode_receipt = {
        "schema": "ect.q128-fresh-decode-receipt/v1", "created_utc": utc_now(),
        "status": "DECODED_AFTER_GATE_PASS", "predecode_gate_sha256": sha256(OUT / "predecode_gate.json"),
        "adjudication_sha256": sha256(OUT / "protocol_deviation_adjudication.json"),
        "decoded_job_count": len(records), "effective_cohort": SEEDS, "old_n3_decoded_or_used": False,
    }
    write_new_json(OUT / "decode_receipt.json", decode_receipt)
    artifact_names = ["predecode_gate.json", "protocol_deviation_adjudication.json", "analysis_input.json",
                      "statistical_results.json", "final_verdict.json", "FINAL_REPORT.md", "decode_receipt.json"]
    receipt = {
        "schema": "ect.q128-fresh-final-analysis-receipt/v1", "created_utc": utc_now(), "status": "FINALIZED",
        "artifacts": {name: {"sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
                      for name in artifact_names},
        "protocol_sha256": PROTOCOL_SHA, "source_commit": SOURCE_COMMIT,
        "finalizer_sha256": sha256(FINALIZER),
        "evaluation_jobs": 272, "effective_cohort": SEEDS,
    }
    write_new_json(OUT / "final_receipt.json", receipt)
    lines = [f"{sha256(OUT / name)}  {name}" for name in artifact_names + ["final_receipt.json"]]
    write_new_text(OUT / "hashes.sha256", "\n".join(lines) + "\n")
    for p in OUT.iterdir():
        p.chmod(0o444)
    OUT.chmod(0o555)
    print(json.dumps({"status": "FINALIZED", "out": str(OUT), "verdict": verdict,
                      "primary": results["sole_primary"], "practical": results["practical_magnitude"],
                      "phase": results["key_secondary_phase"], "nfe": results["key_secondary_nfe"],
                      "final_receipt_sha256": sha256(OUT / "final_receipt.json")}, indent=2))


if __name__ == "__main__":
    main()
