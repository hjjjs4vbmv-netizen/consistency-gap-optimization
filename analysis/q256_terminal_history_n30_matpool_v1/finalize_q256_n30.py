#!/usr/bin/env python3
"""Audit and summarize the sealed q256 terminal-history n=30 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


PROTOCOL_SHA256 = "317d3ef93102050276c1366d9633e322d60fbc9000cd56c8fc8a24c1d4eef544"
EXPECTED_MISSING = {(58, "AA"), (58, "BA"), (65, "AA"), (67, "AA"), (68, "AA")}
FIRSTWAVE_DIRS = ("node8", "node7")
SECONDWAVE_DIRS = ("eval5", "eval6", "single1", "single2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def metric_value(path: Path, metric: str) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"metric row count mismatch: {path}")
    value = float(rows[0]["results"][metric])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric: {path}")
    return value


def audit_evaluation(root: Path) -> tuple[list[dict], dict]:
    all_rows: list[dict] = []
    audit_rows = []
    seal_records = []
    sources = []
    for wave, names in (("firstwave", FIRSTWAVE_DIRS), ("secondwave", SECONDWAVE_DIRS)):
        for name in names:
            directory = root / f"evaluation_{wave}" / name
            csv_path = directory / "decoded_results.csv"
            seal_path = directory / "control" / "evaluation_seal.json"
            if not csv_path.is_file() or not seal_path.is_file():
                raise RuntimeError(f"missing sealed evaluation directory: {directory}")
            seal = load_json(seal_path)
            if seal.get("status") != "SEALED_PASS":
                raise RuntimeError(f"seal not PASS: {seal_path}")
            rows = load_csv(csv_path)
            sources.append({"wave": wave, "worker": name, "row_count": len(rows), "csv_sha256": sha256_file(csv_path)})
            seal_records.append({"wave": wave, "worker": name, "seal_sha256": sha256_file(seal_path), "status": seal["status"]})
            for source_row in rows:
                seed, cell = int(source_row["seed"]), source_row["cell"]
                opaque = source_row["opaque_id"]
                receipt_path = directory / "receipts" / f"{opaque}.json"
                binding_path = directory / "input-bindings" / f"{opaque}.json"
                job_dir = directory / "jobs" / opaque
                receipt = load_json(receipt_path)
                binding = load_json(binding_path)
                if receipt.get("status") != "PASS" or binding.get("status") != "PASS":
                    raise RuntimeError(f"non-PASS receipt or binding: {seed}/{cell}")
                receipt_sha = sha256_file(receipt_path)
                if receipt_sha != source_row["receipt_sha256"]:
                    raise RuntimeError(f"receipt SHA mismatch: {seed}/{cell}")
                if binding.get("checkpoint_sha256") != source_row["checkpoint_sha256"]:
                    raise RuntimeError(f"binding checkpoint SHA mismatch: {seed}/{cell}")
                if receipt.get("checkpoint_sha256") != source_row["checkpoint_sha256"]:
                    raise RuntimeError(f"receipt checkpoint SHA mismatch: {seed}/{cell}")
                if not receipt.get("kid_fid_shared_features"):
                    raise RuntimeError(f"KID/FID features not shared: {seed}/{cell}")
                artifacts = receipt.get("artifact_hashes", {})
                kid_feature = artifacts.get("generated-features-kid50k_full-repeat00.npy", {}).get("sha256")
                fid_feature = artifacts.get("generated-features-fid50k_full-repeat00.npy", {}).get("sha256")
                if not kid_feature or kid_feature != fid_feature or kid_feature != receipt.get("generated_feature_sha256"):
                    raise RuntimeError(f"generated feature SHA mismatch: {seed}/{cell}")
                kid = metric_value(job_dir / "metric-kid50k_full.jsonl", "kid50k_full")
                fid = metric_value(job_dir / "metric-fid50k_full.jsonl", "fid50k_full")
                if kid != float(source_row["kid50k_full"]) or fid != float(source_row["fid50k_full"]):
                    raise RuntimeError(f"decoded metric mismatch: {seed}/{cell}")
                checkpoint = root / "training" / f"seed{seed}" / cell / "kimg1024" / "network-snapshot.pkl"
                checkpoint_sha = sha256_file(checkpoint)
                if checkpoint_sha != source_row["checkpoint_sha256"]:
                    raise RuntimeError(f"central checkpoint SHA mismatch: {seed}/{cell}")
                row = {
                    "seed": seed,
                    "cell": cell,
                    "budget_kimg": int(source_row["budget_kimg"]),
                    "nfe": int(source_row["nfe"]),
                    "kid50k_full": kid,
                    "fid50k_full": fid,
                    "opaque_id": opaque,
                    "checkpoint_sha256": checkpoint_sha,
                    "receipt_sha256": receipt_sha,
                    "wave": wave,
                    "worker": name,
                }
                all_rows.append(row)
                audit_rows.append({
                    "seed": seed, "cell": cell, "wave": wave, "worker": name,
                    "receipt": "PASS", "binding": "PASS", "checkpoint_sha": "PASS",
                    "metric_decode": "PASS", "shared_features": "PASS",
                })
    keys = [(row["seed"], row["cell"]) for row in all_rows]
    expected = {(seed, cell) for seed in range(50, 80) for cell in ("AA", "BA")} - EXPECTED_MISSING
    if len(all_rows) != 55 or len(set(keys)) != 55 or set(keys) != expected:
        raise RuntimeError(f"evaluated endpoint coverage mismatch: {len(all_rows)} rows, {len(set(keys))} unique")
    return sorted(all_rows, key=lambda row: (row["seed"], row["cell"])), {
        "status": "PASS", "evaluated_endpoint_count": len(all_rows),
        "unique_endpoint_count": len(set(keys)), "expected_missing": sorted(EXPECTED_MISSING),
        "sources": sources, "seals": seal_records, "endpoint_checks": audit_rows,
    }


def assignment_audit(assignment_dir: Path) -> dict:
    endpoints = []
    files = []
    for path in sorted(assignment_dir.glob("eval_assignments_*.json")):
        assignments = load_json(path)
        local = [(int(seed), str(cell)) for jobs in assignments.values() for seed, cell in jobs]
        endpoints.extend(local)
        files.append({"file": path.name, "sha256": sha256_file(path), "count": len(local)})
    expected = {(seed, cell) for seed in list(range(58, 66)) + list(range(73, 80)) for cell in ("AA", "BA")}
    if len(endpoints) != 30 or len(set(endpoints)) != 30 or set(endpoints) != expected:
        raise RuntimeError("second-wave assignment coverage mismatch")
    return {"status": "PASS", "planned_endpoints": 30, "unique_endpoints": 30, "files": files}


def failure_audit(root: Path) -> tuple[list[dict], dict]:
    failures = []
    for status_path in sorted((root / "control").glob("storage_sync_node*.json")):
        status = load_json(status_path)
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"storage sync not complete: {status_path}")
        for item in status["scientific_failures"]:
            failures.append({"seed": int(item["seed"]), "cell": item["cell"], "source": status["node_id"]})
    keys = {(row["seed"], row["cell"]) for row in failures}
    if keys != EXPECTED_MISSING or len(failures) != 5:
        raise RuntimeError(f"scientific failure set mismatch: {keys}")

    prefix_failures = []
    for seed in range(50, 80):
        for history in ("A", "B"):
            path = root / "training" / f"seed{seed}" / f"prefix_{history}" / "compute_completion_receipt.json"
            record = load_json(path)
            if record.get("status") == "FAIL" or record.get("exit_code") not in (None, 0):
                prefix_failures.append({"seed": seed, "history": history, "receipt_sha256": sha256_file(path)})
    if {(row["seed"], row["history"]) for row in prefix_failures} != {(67, "A")}:
        raise RuntimeError(f"prefix failure set mismatch: {prefix_failures}")
    return sorted(failures, key=lambda row: (row["seed"], row["cell"])), {
        "status": "PASS", "endpoint_failures": failures, "prefix_failures": prefix_failures,
    }


def paired_statistics(rows: list[dict]) -> tuple[list[dict], dict]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["cell"]] = row
    paired = []
    for seed in range(50, 80):
        cells = by_seed.get(seed, {})
        if "AA" in cells and "BA" in cells:
            aa, ba = cells["AA"], cells["BA"]
            paired.append({
                "seed": seed,
                "aa_kid50k": aa["kid50k_full"], "ba_kid50k": ba["kid50k_full"],
                "delta_kid_ba_minus_aa": ba["kid50k_full"] - aa["kid50k_full"],
                "aa_fid50k": aa["fid50k_full"], "ba_fid50k": ba["fid50k_full"],
                "log_fid_contrast_ba_minus_aa": math.log(ba["fid50k_full"]) - math.log(aa["fid50k_full"]),
            })
    if len(paired) != 26:
        raise RuntimeError(f"expected 26 complete pairs, got {len(paired)}")
    contrasts = np.array([row["log_fid_contrast_ba_minus_aa"] for row in paired], dtype=np.float64)
    n = contrasts.size
    mean = float(contrasts.mean())
    sd = float(contrasts.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    t95 = float(stats.t.ppf(0.975, df))
    t90 = float(stats.t.ppf(0.95, df))
    ci95 = [mean - t95 * se, mean + t95 * se]
    ci90 = [mean - t90 * se, mean + t90 * se]
    margin = math.log(1.03)
    t_stat = mean / se
    directional_p = float(2 * stats.t.sf(abs(t_stat), df))
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    tost_p = max(p_lower, p_upper)
    if ci95[1] < 0:
        classification = "DIRECTIONAL_NEGATIVE"
    elif ci95[0] > 0:
        classification = "DIRECTIONAL_POSITIVE"
    elif ci90[0] > -margin and ci90[1] < margin:
        classification = "PRACTICAL_EQUIVALENCE"
    else:
        classification = "INCONCLUSIVE"

    subgroups = {}
    for label, seeds in (("training_node8_seeds50_65", set(range(50, 66))), ("training_node7_seeds66_79", set(range(66, 80)))):
        values = np.array([row["log_fid_contrast_ba_minus_aa"] for row in paired if row["seed"] in seeds])
        subgroups[label] = {
            "n": int(values.size), "mean_log_fid_contrast": float(values.mean()),
            "negative_count": int((values < 0).sum()), "positive_count": int((values > 0).sum()),
        }

    arm_summary = {}
    for cell in ("AA", "BA"):
        arm = [row for row in rows if row["cell"] == cell]
        arm_summary[cell] = {
            "available_n": len(arm), "planned_n": 30, "missing_n": 30 - len(arm),
            "mean_kid50k": float(np.mean([row["kid50k_full"] for row in arm])),
            "mean_fid50k": float(np.mean([row["fid50k_full"] for row in arm])),
        }
    aa_table = [[4, 26], [1, 29]]
    arm_odds, arm_fisher_p = stats.fisher_exact(aa_table, alternative="two-sided")
    prefix_odds, prefix_fisher_p = stats.fisher_exact([[1, 29], [0, 30]], alternative="two-sided")
    summary = {
        "primary_estimand": "log(FID50k_BA)-log(FID50k_AA)",
        "complete_pairs_n": n, "mean_log_fid_contrast": mean,
        "geometric_fid_ratio_ba_over_aa": math.exp(mean),
        "geometric_fid_percent_change_ba_vs_aa": 100 * (math.exp(mean) - 1),
        "sd_log_fid_contrast": sd, "se_log_fid_contrast": se,
        "cohen_dz": mean / sd, "t_statistic": t_stat, "degrees_of_freedom": df,
        "two_sided_p": directional_p, "ci95": ci95,
        "equivalence_margin": [-margin, margin], "ci90": ci90,
        "tost": {"t_lower": t_lower, "p_lower": p_lower, "t_upper": t_upper, "p_upper": p_upper, "p_tost": tost_p, "equivalent_at_0.05": tost_p < 0.05},
        "classification": classification,
        "negative_pairs": int((contrasts < 0).sum()), "positive_pairs": int((contrasts > 0).sum()),
        "mean_delta_kid_ba_minus_aa": float(np.mean([row["delta_kid_ba_minus_aa"] for row in paired])),
        "arms": arm_summary,
        "planned_endpoint_failure": {
            "AA": {"failed": 4, "planned": 30, "rate": 4 / 30},
            "BA": {"failed": 1, "planned": 30, "rate": 1 / 30},
            "fisher_exact_odds_ratio": float(arm_odds), "fisher_exact_two_sided_p": float(arm_fisher_p),
        },
        "prefix_history_failure": {
            "A": {"failed": 1, "planned": 30, "rate": 1 / 30},
            "B": {"failed": 0, "planned": 30, "rate": 0.0},
            "fisher_exact_odds_ratio": None if not math.isfinite(prefix_odds) else float(prefix_odds),
            "odds_ratio_note": "infinite because the B-history failure cell is zero" if not math.isfinite(prefix_odds) else None,
            "fisher_exact_two_sided_p": float(prefix_fisher_p),
        },
        "training_node_subgroups": subgroups,
    }
    return paired, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())


def validation_report(stats_result: dict, integrity: dict, failures: list[dict]) -> str:
    ci95 = stats_result["ci95"]
    ci90 = stats_result["ci90"]
    margin = abs(stats_result["equivalence_margin"][1])
    subgroup_values = list(stats_result["training_node_subgroups"].values())
    subgroup_reversal = any(item["mean_log_fid_contrast"] >= 0 for item in subgroup_values) != (stats_result["mean_log_fid_contrast"] >= 0)
    failure_text = ", ".join(f"seed{item['seed']}-{item['cell']}" for item in failures)
    fallacies = [
        ("Simpson's paradox", "NOTE", "按两台训练节点分层后方向均已检查；未发现总体方向反转。" if not subgroup_reversal else "发现分层方向反转，需要谨慎。"),
        ("Ecological fallacy", "NOTE", "推断单位与分析单位均为 seed 级配对，不涉及由群体推断个体。"),
        ("Berkson's paradox", "CAUTION", "完整病例分析排除了4个含缺失 endpoint 的 seed，选择机制与数值稳定性相关。"),
        ("Collider bias", "NOTE", "主模型未加入后处理控制变量，未发现显式 collider 调整。"),
        ("Base-rate neglect", "NOTE", "已同时报告 AA/BA 与 A/B history 的计划分母和失败率。"),
        ("Regression to the mean", "NOTE", "连续 seed 预先选定，未按极端 FID 选择样本。"),
        ("Survivorship bias", "CAUTION", "主效应基于26/30完整配对；5/60 endpoint 缺失且集中于 AA。"),
        ("Look-elsewhere effect", "NOTE", "主 estimand、CI、TOST margin 与判定优先级均由冻结协议预设。"),
        ("Garden of forking paths", "NOTE", "使用冻结协议与连续 seeds；科学失败未补跑或替换。"),
        ("Correlation != causation", "NOTE", "报告仅解释预设配对干预对比，不外推非实验因果主张。"),
        ("Reverse causality", "NOTE", "历史臂先于终点训练与评估，未发现反向时间顺序问题。"),
    ]
    lines = [
        "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: validate",
        f"- Origin Date: {utc_now()}", "- Verification Status: ANALYZED",
        "- Version Label: validation_v1", "", "## Validation Report", "",
        "- **Source**: q256_terminal_history_n30_matpool_v1",
        "- **Overall Confidence**: CAUTION", "",
        "### Statistical Findings", "",
        "| Metric | Test | Value | Effect Size | Confidence |", "|---|---|---:|---:|---|",
        f"| Paired log-FID BA−AA | paired Student-t | t({stats_result['degrees_of_freedom']})={stats_result['t_statistic']:.6f}, p={stats_result['two_sided_p']:.8g}; 95% CI [{ci95[0]:.6f}, {ci95[1]:.6f}] | dz={stats_result['cohen_dz']:.6f} | CAUTION |",
        f"| Practical equivalence | TOST ±log(1.03) | p_TOST={stats_result['tost']['p_tost']:.8g}; 90% CI [{ci90[0]:.6f}, {ci90[1]:.6f}] | margin ±{margin:.6f} | CAUTION |",
        f"| Final classification | frozen precedence | {stats_result['classification']} | geometric FID change {stats_result['geometric_fid_percent_change_ba_vs_aa']:.3f}% | CAUTION |",
        "", "### Warnings", "",
        "| Type | Detail | Affected |", "|---|---|---|",
        f"| Informative missingness | 5/60 endpoint failures: {failure_text} | Complete-case primary analysis |",
        "| Differential failure | AA 4/30 vs BA 1/30；失败并非均匀分布。 | Failure-rate estimand |",
        "| Reproducibility scope | 未进行完整独立重跑；已完成55个 checkpoint/receipt/metric/shared-feature 哈希链审计。 | Reproducibility verdict |",
        "", "### Fallacy Scan", "", "- **Coverage**: 11/11 fallacy types checked", "",
        "| Fallacy | Severity | Detail | Recommendation |", "|---|---|---|---|",
    ]
    for name, severity, detail in fallacies:
        recommendation = "保留预注册分析，并在解释中显式报告缺失与适用边界。" if severity == "CAUTION" else "无需额外处理。"
        lines.append(f"| {name} | {severity} | {detail} | {recommendation} |")
    lines += [
        "", "### Reproducibility", "",
        "- **Method**: artifact-integrity verification; no independent full re-run",
        "- **Verdict**: CANNOT_VERIFY",
        "", "完整性审计状态：`PASS`。这验证了已运行结果的证据链，但不等同于独立重现实验。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--assignment-dir", required=True)
    parser.add_argument("--output-name", default="final_analysis")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output_name
    output.mkdir(exist_ok=False)

    rows, evaluation_audit = audit_evaluation(root)
    assignments = assignment_audit(Path(args.assignment_dir))
    failures, failures_audit = failure_audit(root)
    paired, statistics = paired_statistics(rows)
    integrity = {
        "schema": "ect.q256.terminal-history-final-integrity/v1", "status": "PASS",
        "protocol_sha256": PROTOCOL_SHA256, "evaluation": evaluation_audit,
        "assignments": assignments, "failures": failures_audit,
        "verified_at": utc_now(),
    }
    write_csv(output / "combined_results.csv", rows)
    write_json(output / "combined_results.json", {"status": "PASS", "rows": rows, "protocol_sha256": PROTOCOL_SHA256})
    write_csv(output / "paired_results.csv", paired)
    write_csv(output / "scientific_failures.csv", failures)
    write_json(output / "statistics.json", statistics)
    write_json(output / "integrity_verification.json", integrity)
    (output / "VALIDATION_REPORT.md").write_text(validation_report(statistics, integrity, failures), encoding="utf-8")

    checksum_lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    print(json.dumps({
        "status": "PASS", "evaluated_endpoints": len(rows), "complete_pairs": len(paired),
        "classification": statistics["classification"], "mean_log_fid_contrast": statistics["mean_log_fid_contrast"],
        "ci95": statistics["ci95"], "ci90": statistics["ci90"], "tost_p": statistics["tost"]["p_tost"],
        "scientific_failures": failures, "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
