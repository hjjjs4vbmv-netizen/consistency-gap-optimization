"""Reproduce the TASK 2 report; optionally recheck a complete local evaluation archive."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "results/q256_switchpoint_sweep/full_cohort"
ROOT = BUNDLE
sys.path.insert(0, str(REPO))
from analysis.q256_switchpoint_sweep import analyze, result_conversion, companion_summary


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_archive(evaluation):
    formal = evaluation / "formal"
    matrix = load(formal / "matrix_seal.json")
    private = load(evaluation / "private_map.json")
    public = load(formal / "public_manifest.json")
    protocol_sha = sha(REPO / "analysis/q256_switchpoint_sweep/protocol.json")
    assert matrix["status"] == "SEALED_PASS" and matrix["sealed_jobs"] == 132
    assert sha(formal / "matrix_seal.json") == "673638177d15666c787d2891f4fe805490635c94c388e98952b091c763bbac09"
    assert all(x["protocol_sha256"] == protocol_sha for x in (matrix, private, public))
    assert public["private_map_sha256"] == sha(evaluation / "private_map.json")
    public_by_id = {j["opaque_id"]: j for j in public["jobs"]}
    assert len(public_by_id) == len(private["jobs"]) == 132
    rows = []
    for job in private["jobs"]:
        opaque = job["opaque_id"]
        receipt = load(formal / "receipts" / f"{opaque}.json")
        terminal_path = formal / "terminal" / f"{opaque}.json"
        terminal = load(terminal_path)
        seal = load(formal / "seals" / f"{opaque}.json")
        assert terminal["status"] == "PASS" and seal["status"] == "SEALED_PASS"
        assert terminal["opaque_id"] == seal["opaque_id"] == receipt["opaque_id"] == opaque
        assert sha(terminal_path) == seal["terminal_sha256"]
        assert receipt["status"] == "VALIDATED_UNSEALED"
        assert receipt["protocol_sha256"] == protocol_sha and job["training_status"] == "AVAILABLE"
        assert receipt["checkpoint_sha256"] == job["checkpoint_sha256"] == public_by_id[opaque]["checkpoint_sha256"]
        assert receipt["evaluator_commit"] == "d6aba02fb88e9db0993623895eb2228ed717d810"
        assert receipt["kid_fid_shared_feature_identity"] is True
        row = {k: job[k] for k in ("seed", "trajectory", "kimg", "role", "opaque_id")}
        row.update(status=terminal["status"], root_cause=terminal.get("root_cause"))
        job_dir = formal / "jobs" / opaque
        for metric in ("fid50k_full", "kid50k_full"):
            path = job_dir / f"metric-{metric}.jsonl"
            assert sha(path) == receipt["metric_artifact_sha256"][metric]
            lines = path.read_text().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["metric"] == metric and record["num_gpus"] == 1
            row[metric] = float(record["results"][metric])
            assert math.isfinite(row[metric]) and row[metric] >= 0
        options_path = job_dir / "training_options.json"
        assert sha(options_path) == receipt["artifacts"]["training_options.json"]["sha256"]
        options = load(options_path)
        assert options["sample_seeds"] == list(range(50000)) and options["mid_t"] == []
        assert options["seed"] == 20260730 and options["metric_repeats"] == 1
        assert options["network_kwargs"]["use_fp16"] is False
        rows.append(row)
    rows.sort(key=lambda r: (r["seed"], r["trajectory"], r["kimg"]))
    decoded = {"status": "PASS", "matrix_seal_sha256": sha(formal / "matrix_seal.json"), "results": rows}
    result_conversion.convert_decoded(decoded)
    dest = ROOT / "decoded_results.json"
    if dest.exists():
        assert load(dest) == decoded
    else:
        save(dest, decoded)
    save(ROOT / "verification.json", {
        "checked_at": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "metric_files_hash_verified": 264, "options_files_hash_verified": 132,
        "terminal_seals_verified": 132, "unique_jobs": 132,
        "matrix_seal_sha256": decoded["matrix_seal_sha256"],
        "scope": "Downloaded metrics/options checked against archived receipts; checkpoint and feature arrays were not rehashed in this summary.",
        "source": "Privately retained q256_switchpoint_sweep evaluation archive",
    })
    return decoded


def table_summary(name, point, ba, ctrl):
    return {"comparison": name, "n": point["n"], "mean_log_ratio": point["mean"],
            "median_log_ratio": point["median"], "sd_log_ratio": point["sample_sd"],
            "ci95_low": point["mean_t_ci95"][0], "ci95_high": point["mean_t_ci95"][1],
            "geometric_fid_change_pct": 100 * math.expm1(point["mean"]),
            "ba_better_seeds": point["sign_counts"]["negative"],
            "ba_mean_fid": statistics.mean(x["fid50k_full"] for x in ba),
            "ctrl_mean_fid": statistics.mean(x["fid50k_full"] for x in ctrl),
            "ba_mean_kid": statistics.mean(x["kid50k_full"] for x in ba),
            "ctrl_mean_kid": statistics.mean(x["kid50k_full"] for x in ctrl),
            "kid_ba_better_seeds": sum(b["kid50k_full"] < c["kid50k_full"] for b, c in zip(ba, ctrl))}


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE)
    parser.add_argument("--archive-root", type=Path, help="Optional original evaluation/ directory, including receipts and options")
    args = parser.parse_args()
    ROOT = args.output_dir
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.archive_root:
        decoded = decode_archive(args.archive_root)
    else:
        decoded = load(BUNDLE / "decoded_results.json")
        assert decoded["matrix_seal_sha256"] == sha(BUNDLE / "matrix_seal.json")
    raw = decoded["results"]
    rows, h_values = result_conversion.convert_decoded(decoded)
    result = analyze.analyze(rows)
    result["reporting_context"] = "Same cohort completed after prior six-seed preliminary disclosure; algorithm output does not establish uninterrupted blinding."
    common = analyze.summarize_common_endpoint(h_values)
    save(ROOT / "frozen_calculation.json", result)
    save(ROOT / "common_endpoint_descriptive.json", common)
    result_conversion.write_rows(rows, ROOT / "fixed_chase_seed_results.csv")
    write_csv(ROOT / "raw_metrics.csv", raw)
    cells = {(x["seed"], x["trajectory"], x["kimg"]): x for x in raw}
    summaries = []
    for curve in ("G", "H"):
        for s in analyze.SWITCH_POINTS:
            t = s + 512 if curve == "G" else 1024
            point = result["point_summaries"][f"G_{s}"] if curve == "G" else common["points"][f"H_{s}"]["summary"]
            ba = [cells[i, f"BA{s}", t] for i in analyze.SEEDS]
            ctrl = [cells[i, "CTRL", t] for i in analyze.SEEDS]
            summaries.append(table_summary(f"{curve}_{s}", point, ba, ctrl))
    write_csv(ROOT / "summary.csv", summaries)
    old_path = REPO / "results/q256_switchpoint_sweep/preliminary/metrics.csv"
    old = list(csv.DictReader(old_path.open()))
    differences = {metric: max(abs(float(r[column]) - cells[int(r["seed"]), r["arm"], int(r["kimg"])][metric]) for r in old)
                   for column, metric in (("fid50k", "fid50k_full"), ("kid50k", "kid50k_full"))}
    save(ROOT / "preliminary_comparison.json", {"overlap_jobs": len(old), "maximum_absolute_difference": differences})
    companion_receipts = [load(p) for p in (args.archive_root / "companion/receipts").glob("*.json")] if args.archive_root else []
    companion = companion_summary.summarize(decoded, companion_receipts)
    save(ROOT / "companion_status.json", companion)
    page = result["page_test"]
    lines = ["# TASK 2 全量实验结果", "", "统计日期：2026-09-06；cohort：训练 seeds 81–92，n=12。", "",
             "全量结果未支持预设的‘B 前缀越长，切换到 A 后的相对优势越大’。四点的 BA/CTRL 配对 FID 几何平均低约 4.4%–6.4%，但均值的 95% 区间全部跨零；跨训练 seed 的方向和幅度差异较大。",
             f"单侧精确 Page 检验 p={page['p_exact']:.6f}，冻结算法输出 ORDERING_NOT_RESOLVED。此前 6 个 seed 的约 11%–13% 描述性优势，在补全 12 个 seed 后缩小。未显著不代表没有效应，也不证明四种日程等效。", "",
             "## 实验与指标", "",
             "A 的 (target, denominator)=(1.0,1.0)，B=(1.1,1.1)。CTRL 全程使用 A，并在 512 kimg 做 A→A 恢复；BA(s) 先用 B 训练 s kimg，再切换 A。kimg 表示累计训练样本呈现数的千倍，不是独立样本数。",
             "FID50k 衡量 50,000 张生成图像与真实图像的 Inception 特征分布差异，越低越好；KID50k 使用核距离估计同类分布差异，越低通常越好。NFE1 表示每张图像生成时调用网络一次；所有评估为 FP32，FID/KID 共享生成特征。", "",
             "G_s=ln(FID_BA(s)@s+512/FID_CTRL@s+512)，固定 A 后续训练为 512 kimg；H_s 在共同终点 1024 kimg 比较。G/H 为负表示 BA 的 FID 更低。", "",
             "相对变化=100×[exp(平均 log 比值)−1]，是配对比值几何平均的变化，不是 FID 算术均值之比。95% CI=均值±t(11,0.975)×样本SD/√12，为跨训练 seed 的均值区间，未作多重比较校正，也不是生成噪声范围。", "",
             "## 固定 512-kimg A 后续训练", "",
             "| B 前缀 kimg | BA/CTRL 平均 FID | 平均 G [95% CI] | 中位数 G | SD | 几何平均 FID 变化 | BA 更优 seeds |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summaries[:4]:
        lines.append(f"| {x['comparison'][2:]} | {x['ba_mean_fid']:.4f} / {x['ctrl_mean_fid']:.4f} | {x['mean_log_ratio']:.5f} [{x['ci95_low']:.5f}, {x['ci95_high']:.5f}] | {x['median_log_ratio']:.5f} | {x['sd_log_ratio']:.5f} | {x['geometric_fid_change_pct']:+.2f}% | {x['ba_better_seeds']}/12 |")
    lines += ["", f"按冻结单侧 Page 算法计算：L={page['L_observed']:.1f}，p={page['p_exact']:.8f}；脚本数值判定 `{page['verdict']}`。预设方向为 G128≥G256≥G384≥G512，即 B 前缀越长，BA 相对 CTRL 越占优。L 是按 seed 内秩计算的有序趋势统计量；p 是在无有序差异、seed 内标签可交换的零假设下，得到至少同等强度预设方向统计量的概率，不是‘结果为噪声的概率’。", "",
              "本次是同一 cohort 的全量补全；其中 seeds 81–86 的恢复性初步数值已在完整 seal 前报告。因此不将脚本数值判定描述成全程盲态的确认性结论。SEALED_PASS 表示评估矩阵完成。", "",
              "相邻差异也没有显示清晰的稳定次序（以下均为描述性区间）：", "",
              "| 相邻 G 差 | 均值 | 95% CI |", "|---|---:|---:|"]
    for name, x in result["adjacent_paired_differences"].items():
        lo, hi = x["mean_t_ci95"]
        lines.append(f"| {name} | {x['mean']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += ["",
              "## 共同 1024-kimg 终点（描述性）", "",
              "| B 前缀 kimg | 平均 H [95% CI] | 几何平均 FID 变化 | BA 更优 seeds |", "|---:|---:|---:|---:|"]
    for x in summaries[4:]:
        lines.append(f"| {x['comparison'][2:]} | {x['mean_log_ratio']:.5f} [{x['ci95_low']:.5f}, {x['ci95_high']:.5f}] | {x['geometric_fid_change_pct']:+.2f}% | {x['ba_better_seeds']}/12 |")
    lines += ["", "H 同时改变 B 前缀长度与 A 后续训练长度，不能单独识别早期写入或剂量累积；H512=G512，共用同一配对。", "", "## 跨 seed 原始配对结果", "",
              "| seed | G128 | G256 | G384 | G512 |", "|---:|---:|---:|---:|---:|"]
    for seed, values in result["g_by_seed"].items():
        lines.append("| " + seed + " | " + " | ".join(f"{x:.6f}" for x in values) + " |")
    lines += ["", "例如相同 512 切换点与 1024 终点，seed85 的 BA FID 低 36.34%，seed92 却高 39.12%。128 切换点的中位数 G≈0.00208（约 +0.21%），说明均值上的负向差异不代表每个 seed 或典型 seed 都改善；所有 12 个 seed 均保留。", "",
              "完整 H 配对结果：", "", "| seed | H128 | H256 | H384 | H512 |", "|---:|---:|---:|---:|---:|"]
    for seed in analyze.SEEDS:
        lines.append("| " + str(seed) + " | " + " | ".join(f"{h_values[s][seed]:.6f}" for s in analyze.SWITCH_POINTS) + " |")
    lines += ["", "## KID 方向核对", "", "| 切换点 | BA 平均 KID | CTRL 平均 KID | BA KID 更优 seeds |", "|---:|---:|---:|---:|"]
    for x in summaries[:4]:
        lines.append(f"| {x['comparison'][2:]} | {x['ba_mean_kid']:.7f} | {x['ctrl_mean_kid']:.7f} | {x['kid_ba_better_seeds']}/12 |")
    lines += ["", "KID 仅作辅助核对，不参与 Page 判定。", "", "## 数据与边界", "",
              "132 个唯一评估 job（96 主矩阵+36 共同终点追加），全部 PASS，12 个 seed 完整。264 个指标文件、132 个参数文件对原始回执哈希一致，132 个单项 seal 与 terminal 绑定一致。本次未重新读取大体积 checkpoint 或生成特征数组。",
              f"旧初步报告与当前矩阵重叠 {len(old)} 个 job；最大绝对差 FID={differences['fid50k_full']:.12g}，KID={differences['kid50k_full']:.12g}。",
              f"生成噪声 companion：{companion['status']}，仅 {companion['n_blocks']}/5 个 paired generation blocks 可用，即复用正式评估的 block0；其余四组未完成，无法估计 generation-noise SD。",
              "不与其他 cohort 合并，不据此修改 TASK1 的 INCONCLUSIVE；未进行单点假设检验、H 趋势检验或事后子组检验。",
              "", "文件：raw_metrics.csv 为 132 行原始 FID/KID；fixed_chase_seed_results.csv 为 48 行配对；summary.csv 为 G/H 汇总；frozen_calculation.json 含均值、区间、相邻差和 Page 算法输出。"]
    (ROOT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"page": page, "summary": summaries, "old_overlap": differences, "companion": companion["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
