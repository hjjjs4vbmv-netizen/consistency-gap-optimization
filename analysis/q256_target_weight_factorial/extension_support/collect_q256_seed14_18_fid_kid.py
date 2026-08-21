#!/usr/bin/env python3
"""Validate and summarize q256 target-weight seed14-18 frozen evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


SEEDS = tuple(range(14, 19))
ARMS = ("A", "B", "C", "D")
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
CONTRASTS = (
    ("C-A", ("C", 1.0), ("A", -1.0)),
    ("B-D", ("B", 1.0), ("D", -1.0)),
    ("D-A", ("D", 1.0), ("A", -1.0)),
    ("B-C", ("B", 1.0), ("C", -1.0)),
    ("I=B-C-D+A", ("B", 1.0), ("C", -1.0), ("D", -1.0), ("A", 1.0)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def stats(values: list[float]) -> dict:
    if len(values) != len(SEEDS):
        raise RuntimeError(f"expected five independent seeds, got {len(values)}")
    return {
        "n": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "span": max(values) - min(values),
        "sample_sd": statistics.stdev(values),
        "negative_count": sum(value < 0 for value in values),
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
    }


def fmt(value: float, digits: int = 9) -> str:
    return f"{value:.{digits}f}"


def collect(eval_root: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    plans: list[dict] = []
    expected = {(seed, arm, nfe) for seed in SEEDS for arm in ARMS for nfe in NFES}
    seen: set[tuple[int, str, int]] = set()

    for seed in SEEDS:
        seed_root = eval_root / f"seed{seed}"
        plan_path = seed_root / "evaluation_plan.json"
        completion_path = seed_root / "WORKER_PASS.json"
        plan = load_json(plan_path)
        completion = load_json(completion_path)
        if plan.get("schema") != "q256-target-weight-seed14-18-frozen-evaluation-plan-v1":
            raise RuntimeError(f"wrong plan schema for seed{seed}")
        if plan.get("seed") != seed or plan.get("physical_gpu_index") != seed - 14:
            raise RuntimeError(f"wrong seed/GPU mapping for seed{seed}")
        protocol = plan.get("protocol", {})
        exact_protocol = {
            "precision": "fp32",
            "sample_count": 50000,
            "sample_seeds": "0-49999",
            "metric_seed": 20260730,
            "metrics": list(METRICS),
            "metric_repeats": 1,
            "eval_batch": 512,
            "metric_generator_batch": 128,
            "workers": 3,
            "nfe_modes": {"1": [], "2": [0.821]},
            "retain_generated_artifacts": True,
            "selection_policy": "all_exact_final_256kimg_cells_no_intermediate_selection",
        }
        if protocol != exact_protocol:
            raise RuntimeError(f"protocol drift for seed{seed}")
        if completion.get("status") != "WORKER_PASS" or completion.get("jobs_completed") != 8:
            raise RuntimeError(f"seed{seed} lacks exact 8-job WORKER_PASS")
        if completion.get("evaluation_plan_sha256") != sha256_file(plan_path):
            raise RuntimeError(f"seed{seed} completion does not bind its plan")
        plans.append(plan)

        completion_receipts = {
            item["job_id"]: item for item in completion.get("job_receipts", [])
        }
        if len(completion_receipts) != 8:
            raise RuntimeError(f"seed{seed} completion receipt set is not exact")

        for arm in ARMS:
            for nfe in NFES:
                key = (seed, arm, nfe)
                job_id = f"seed{seed}-arm{arm}-nfe{nfe}"
                receipt_path = seed_root / "receipts" / f"{job_id}.json"
                receipt = load_json(receipt_path)
                if receipt.get("status") != "PASS" or receipt.get("schema") != "q256-target-weight-seed14-18-evaluation-job-receipt-v1":
                    raise RuntimeError(f"non-PASS or wrong receipt: {job_id}")
                job = receipt.get("job", {})
                if (job.get("seed"), job.get("arm"), job.get("nfe")) != key:
                    raise RuntimeError(f"job identity mismatch: {job_id}")
                if job.get("mid_t") != ([] if nfe == 1 else [0.821]):
                    raise RuntimeError(f"mid_t mismatch: {job_id}")
                completion_item = completion_receipts.get(job_id, {})
                receipt_sha = sha256_file(receipt_path)
                if completion_item.get("receipt_sha256") != receipt_sha:
                    raise RuntimeError(f"completion receipt hash mismatch: {job_id}")
                metrics = receipt.get("validation", {}).get("metrics", [])
                if [item.get("metric") for item in metrics] != list(METRICS):
                    raise RuntimeError(f"metric identity/order mismatch: {job_id}")
                metric_values: dict[str, float] = {}
                for item in metrics:
                    value = float(item["value"])
                    if not math.isfinite(value) or (item["metric"].startswith("fid") and value < 0):
                        raise RuntimeError(f"invalid metric value: {job_id}")
                    raw_path = Path(item["raw_path"])
                    if sha256_file(raw_path) != item["raw_sha256"]:
                        raise RuntimeError(f"raw metric changed after PASS: {job_id}")
                    metric_values[item["metric"]] = value
                artifacts = receipt["validation"]["artifacts"]
                kid_feature = artifacts["generated-features-kid50k_full-repeat00.npy"]
                fid_feature = artifacts["generated-features-fid50k_full-repeat00.npy"]
                if kid_feature["sha256"] != fid_feature["sha256"]:
                    raise RuntimeError(f"KID/FID feature mismatch: {job_id}")
                row = {
                    "seed": seed,
                    "arm": arm,
                    "nfe": nfe,
                    "mid_t": "" if nfe == 1 else 0.821,
                    "fid50k_full": metric_values["fid50k_full"],
                    "kid50k_full": metric_values["kid50k_full"],
                    "status": "PASS",
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "receipt_sha256": receipt_sha,
                    "generated_feature_sha256": kid_feature["sha256"],
                    "generated_samples_sha256": artifacts["generated-samples.npy"]["sha256"],
                    "artifacts_tree_sha256": receipt["validation"]["artifacts_tree_sha256"],
                    "process_log_sha256": receipt["process_log_sha256"],
                    "completed_at_utc": receipt["completed_at_utc"],
                }
                rows.append(row)
                seen.add(key)

    if seen != expected or len(rows) != 40:
        raise RuntimeError(f"incomplete exact matrix: {len(rows)}/40")
    source_commits = {
        (plan.get("training_source_commit"), plan.get("evaluator_source_commit"))
        for plan in plans
    }
    datasets = {
        (plan.get("dataset", {}).get("sha256"), plan.get("dataset", {}).get("bytes"))
        for plan in plans
    }
    if len(source_commits) != 1 or len(datasets) != 1:
        raise RuntimeError("source or dataset differs across seed workers")
    rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"]), row["nfe"]))
    metadata = {
        "source_commits": list(source_commits)[0],
        "dataset": list(datasets)[0],
        "runtime_by_seed": {str(plan["seed"]): plan["runtime"] for plan in plans},
        "plan_sha256_by_seed": {
            str(seed): sha256_file(eval_root / f"seed{seed}" / "evaluation_plan.json")
            for seed in SEEDS
        },
        "completion_sha256_by_seed": {
            str(seed): sha256_file(eval_root / f"seed{seed}" / "WORKER_PASS.json")
            for seed in SEEDS
        },
    }
    return rows, metadata


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    lookup = {
        (row["seed"], row["arm"], row["nfe"]): row for row in rows
    }
    cell_summaries = []
    for nfe in NFES:
        for arm in ARMS:
            for metric in ("fid50k_full", "kid50k_full"):
                values = [lookup[(seed, arm, nfe)][metric] for seed in SEEDS]
                cell_summaries.append({"arm": arm, "nfe": nfe, "metric": metric, **stats(values)})

    per_seed_contrasts = []
    contrast_summaries = []
    for nfe in NFES:
        for metric in ("fid50k_full", "kid50k_full"):
            for name, *terms in CONTRASTS:
                values = []
                for seed in SEEDS:
                    value = sum(
                        coefficient * lookup[(seed, arm, nfe)][metric]
                        for arm, coefficient in terms
                    )
                    values.append(value)
                    per_seed_contrasts.append(
                        {"seed": seed, "nfe": nfe, "metric": metric, "contrast": name, "value": value}
                    )
                contrast_summaries.append(
                    {"nfe": nfe, "metric": metric, "contrast": name, **stats(values)}
                )
    return cell_summaries, per_seed_contrasts, contrast_summaries


def render_markdown(rows: list[dict], cell_summaries: list[dict], per_seed: list[dict], contrast_summaries: list[dict], metadata: dict, eval_root: Path) -> str:
    lookup = {(row["seed"], row["arm"], row["nfe"]): row for row in rows}
    cell_lookup = {(row["arm"], row["nfe"], row["metric"]): row for row in cell_summaries}
    per_seed_lookup = {(row["metric"], row["nfe"], row["seed"], row["contrast"]): row["value"] for row in per_seed}
    training_commit, evaluator_commit = metadata["source_commits"]
    dataset_sha, _ = metadata["dataset"]
    lines = [
        "# q256 target geometry × denominator weighting: seed14–18 formal FID/KID evaluation",
        "",
        "Verification status: **VERIFIED**. Evaluation status: **PASS (40/40)**. This is the complete seed14–18 × arm A/B/C/D × NFE1/NFE2 matrix.",
        "",
        "Frozen protocol: FP32; 50,000 generated samples per job; sample seeds `0..49999`; metric seed `20260730`; `kid50k_full` then `fid50k_full` from byte-identical retained generated Inception features; `mid_t=0.821` for NFE2; one canonical CIFAR-10 reference. Lower is better.",
        "",
        f"Training source: `{training_commit}`. Evaluator source: `{evaluator_commit}`. Dataset SHA-256: `{dataset_sha}`. Accepted evaluation root: `{eval_root}`.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for arm in ARMS:
            n1, n2 = lookup[(seed, arm, 1)], lookup[(seed, arm, 2)]
            lines.append(
                f"| {seed} | {arm} | {fmt(n1['fid50k_full'])} | {fmt(n1['kid50k_full'], 12)} | {fmt(n2['fid50k_full'])} | {fmt(n2['kid50k_full'], 12)} |"
            )
    lines += [
        "",
        "## Across-seed descriptive summary",
        "",
        "Values are mean ± sample standard deviation over seed14–18 (`n=5`).",
        "",
        "| Arm | NFE | FID mean ± SD | KID mean ± SD |",
        "|:---:|---:|---:|---:|",
    ]
    for nfe in NFES:
        for arm in ARMS:
            fid = cell_lookup[(arm, nfe, "fid50k_full")]
            kid = cell_lookup[(arm, nfe, "kid50k_full")]
            lines.append(
                f"| {arm} | {nfe} | {fmt(fid['mean'])} ± {fmt(fid['sample_sd'])} | {fmt(kid['mean'], 12)} ± {fmt(kid['sample_sd'], 12)} |"
            )
    for nfe, title in ((1, "Primary-readout factorial contrasts"), (2, "Secondary NFE2 contrasts")):
        definition_note = (
            "These use the factorial definitions originally frozen for the preregistered study. "
            "Seed14–18 themselves are post-preregistration secondary sensitivity seeds. "
            if nfe == 1
            else ""
        )
        lines += [
            "",
            f"## {title}",
            "",
            definition_note + "A negative contrast favors its first term. The interaction is `I=B−C−D+A`.",
            "",
            "| Metric | Seed | C−A | B−D | D−A | B−C | I |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
        for metric, label, digits in (
            ("fid50k_full", f"FID-50k@NFE{nfe}" + (" (primary)" if nfe == 1 else ""), 6),
            ("kid50k_full", f"KID-50k@NFE{nfe}", 9),
        ):
            for seed in SEEDS:
                values = [per_seed_lookup[(metric, nfe, seed, name)] for name, *_ in CONTRASTS]
                lines.append(f"| {label} | {seed} | " + " | ".join(fmt(value, digits) for value in values) + " |")
        lines += [
            "",
            "Cross-seed summaries use the five training seeds as independent units. `Range` is [minimum, maximum], `Span` is maximum minus minimum, and SD is sample SD.",
            "",
            "| Metric | Contrast | Mean | Median | Range | Span | SD | Direction (−/+/0) |",
            "|:---|:---|---:|---:|:---:|---:|---:|:---:|",
        ]
        for metric, label, digits in (
            ("fid50k_full", f"FID-50k@NFE{nfe}" + (" (primary)" if nfe == 1 else ""), 6),
            ("kid50k_full", f"KID-50k@NFE{nfe}", 9),
        ):
            for name, *_ in CONTRASTS:
                item = next(row for row in contrast_summaries if row["nfe"] == nfe and row["metric"] == metric and row["contrast"] == name)
                lines.append(
                    f"| {label} | {name} | {fmt(item['mean'], digits)} | {fmt(item['median'], digits)} | [{fmt(item['minimum'], digits)}, {fmt(item['maximum'], digits)}] | {fmt(item['span'], digits)} | {fmt(item['sample_sd'], digits)} | {item['negative_count']}/{item['positive_count']}/{item['zero_count']} |"
                )
    primary = [item for item in contrast_summaries if item["nfe"] == 1]
    lines += [
        "",
        "## Directional interpretation boundary",
        "",
        "The primary endpoint is FID-50k@NFE1. Direction consistency is reported descriptively over five seeds; the report does not reinterpret the frozen factorial arms, claim a causal percentage decomposition, or use the secondary NFE2 readout to overwrite the primary endpoint.",
        "",
        "Primary direction counts (negative/positive/zero): " + "; ".join(
            f"{item['metric'].replace('50k_full', '').upper()} {item['contrast']}={item['negative_count']}/{item['positive_count']}/{item['zero_count']}"
            for item in primary
        ) + ".",
        "",
        "## Integrity and provenance",
        "",
        "- All 40 job receipts have `status=PASS`; all five workers have exact 8-job `WORKER_PASS` completions.",
        "- Every receipt is SHA-bound by its worker completion; raw metric records were re-hashed during collection.",
        "- Within every job, FID and KID use byte-identical retained generated-feature SHA-256 values.",
        "- Seed-to-GPU mapping remained seed14→GPU0 through seed18→GPU4.",
        "- Earlier v1/v2/v3 orchestration attempts and their failure evidence remain preserved and are not counted as results; this report accepts only the fresh v4 native-runtime PASS chain.",
        "",
        "The CSV contains all 40 full-precision job rows and artifact hashes. The JSON contains the corresponding machine-readable protocol, descriptive summaries, contrasts, and completion bindings.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    eval_root = args.eval_root.expanduser().resolve(strict=True)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    rows, metadata = collect(eval_root)
    cell_summaries, per_seed, contrast_summaries = summarize(rows)
    payload = {
        "schema": "q256-target-weight-seed14-18-fid-kid-report-v1",
        "status": "VERIFIED_PASS_40_OF_40",
        "protocol": {
            "seeds": list(SEEDS), "arms": list(ARMS), "nfe": list(NFES),
            "metrics": list(METRICS), "sample_count": 50000,
            "sample_seeds": "0-49999", "metric_seed": 20260730,
            "precision": "fp32", "nfe2_mid_t": 0.821,
        },
        "metadata": metadata,
        "results": rows,
        "cell_summaries": cell_summaries,
        "per_seed_contrasts": per_seed,
        "contrast_summaries": contrast_summaries,
    }
    csv_path = outdir / "q256_seed14_18_fid_kid_results.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    write_json_exclusive(outdir / "q256_seed14_18_fid_kid_results.json", payload)
    report = render_markdown(rows, cell_summaries, per_seed, contrast_summaries, metadata, eval_root)
    write_text_exclusive(outdir / "q256_seed14_18_fid_kid_report.md", report)
    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(outdir.iterdir()) if path.is_file()
    }
    write_json_exclusive(outdir / "REPORT_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "jobs": len(rows), "outdir": str(outdir), "files": sorted(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
