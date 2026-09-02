# ImageNet-64 主图与 cross-dataset trajectory illustration

## Material Passport

- Origin Skill: `academic-research-suite`（experiment-agent validate + visualization）
- Origin Date: 2026-08-29
- Verification Status: `ANALYZED_AND_RENDER_VERIFIED`
- Evidence class: three-seed paired descriptive results；不作总体显著性或因果机制推断
- Integration base: PR #91 merged as `18ec62e`

## 交付物

- `../../figures/main/imagenet_per_seed_trajectories.pdf`
- `../../figures/main/cross_dataset_quality_emergence.pdf`
- `contraction_per_seed.csv`
- 本文件

可复现绘图入口为 `../../scripts/build_imagenet91_paper_figures.py`。脚本直接读取
PR #91 已提交到 main 的 canonical 120-cell table：
`../imagenet64_gap_ab_full120_20260829/per_trajectory.csv`，不再维护重复快照。

## ImageNet 数据与语义

源结果现已通过 PR #91 合入 main（squash commit `18ec62e`），canonical 路径为
`results/imagenet64_gap_ab_full120_20260829/per_trajectory.csv`。

- 训练 seed：101、102、103，IA/IB 同 seed 配对。
- IA：`global_gap_scale=1.0`；IB：`global_gap_scale=1.1`。
- checkpoint：1,280–12,800 kimg，步长 1,280 kimg。
- 每个 checkpoint 均有 NFE=1 与 NFE=2 的 FID-50k/KID-50k。
- 图中差值定义为 `Δ = FID(IA) - FID(IB)`；负值偏向 IA。
- 冻结矩阵包含 120/120 个唯一 cell，所有 240 个指标均为有限值。
- canonical input SHA-256：`975b52a18caa186bb69929367e69d4f6f36771f31026b2044fe97f32bc06a184`。

seed103 在 7,680 kimg 后出现 trajectory instability。它仍完整保留在冻结分析、曲线和 endpoint 表中；没有据此删除、换 seed 或改变停止规则。训练 provenance 表明六条 trajectory 的配对配置除 `global_gap_scale` 外一致，并且失稳不能由 cross-seed checkpoint mixing 解释。

## Contraction summary

为这张描述性图定义的两个检查点（结果已经存在后选取，未预注册）：

`K_early = 6400 kimg`，`K_late = 12800 kimg`。

逐 seed、逐 NFE 计算：

`contraction ratio = |Δ(K_late)| / |Δ(K_early)|`。

| Seed | NFE | `|Δ(6400)|` | `|Δ(12800)|` | Ratio | 解释 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 101 | 1 | 6.296289 | 0.126633 | 0.020112 | stable trajectory 上的描述性 contraction |
| 101 | 2 | 1.741970 | 0.000552 | 0.000317 | stable trajectory 上的描述性 contraction |
| 102 | 1 | 2.148009 | 0.104444 | 0.048624 | stable trajectory 上的描述性 contraction |
| 102 | 2 | 0.709616 | 0.007685 | 0.010830 | stable trajectory 上的描述性 contraction |
| 103 | 1 | 2.094881 | 37.546985 | 17.923203 | `not interpretable after trajectory instability` |
| 103 | 2 | 0.784959 | 36.138905 | 46.039237 | `not interpretable after trajectory instability` |

CSV 同时保留有符号的 early/late delta、绝对值、ratio 与解释状态。seed103 的 ratio 是算术结果，不解释为 treatment contraction。

## Figure 4 caption

**Figure 4. ImageNet-64 paired quality trajectories show early separation and late seed heterogeneity.**
Panels A and B display IA and IB FID-50k trajectories separately for each of the three frozen training seeds at NFE=1 and NFE=2; no mean trajectory is shown. Panel C reports the paired seed-level difference `Δ = FID(IA) − FID(IB)` on a symmetric-log scale. Panel D first reports the 12,800-kimg endpoints for all three frozen seeds without exclusion, then—in a visually separate blue block—reports a post hoc descriptive sensitivity mean restricted to stable seeds101/102. The shaded region begins after 7,680 kimg, where seed103 becomes unstable. FID axes in Panels A/B are logarithmic; lower is better. The seeds101/102 sensitivity summary is not the frozen three-seed estimand, and seed103 contraction after instability is not treatment-interpretable.

## Cross-dataset figure caption

**Finite-budget contrasts contract or change sign, while one ImageNet seed becomes unstable under both interventions.**
Thin colored lines are paired differences for individual training seeds and dark lines are explicitly labeled descriptive summaries. The q256 panel is a seed3–5 replay illustration: at NFE=2 (B−A), its post hoc full-curve log-FID AULC difference (`−0.181`) coexists with a small 1,024-kimg mean endpoint gap (`−0.038`). It is not a substitute for the planned balanced q256 cohort. At q128 (NFE=1, B−A), B is harmful at 512 kimg (`+5.05`), approximately neutral at 640 kimg (`−0.013`), and has a small late mean effect at 1,024 kimg (`−0.280`). On ImageNet-64 (NFE=2, IA−IB), all seeds show an early IA advantage; stable seeds101/102 approach late near-equivalence (`−0.0036` mean at 12,800 kimg), whereas seed103 becomes unstable under both IA and IB and remains displayed separately. Panels use different arm contrasts, cohorts, and NFE settings as labeled and therefore compare trajectory shapes, not a pooled treatment effect.

## Cross-dataset provenance

- q256 source: `../../results/q256_target_weight_replay_curve_seed3_5/fidkid50k-final-20260823/evaluation_results.csv`, SHA-256 `118740bfc1191671f3c409158a3b48149851a02775372a79f4923a5ca8c5d58c`.
- q128 source: `../../results/second_q_q128_ab_v2/final/paired_results.csv`, SHA-256 `b5c8271112032b7f7135cd62a60412bbaaba1b165620e3be1b7fdb1e686c0e79`.
- q256 AULC is a deterministic normalized trapezoidal area under natural-log FID over 256–1,024 kimg, calculated per seed and then averaged; it is descriptive rather than preregistered inference.
- q128 values use the prospective paired second-q experiment and its frozen NFE=1 checkpoints.
- The q256 seed3–5 replay and its log-FID AULC are descriptive illustration only; the figure should not be promoted to the manuscript's primary q256 evidence before the balanced cohort is available.
- ImageNet stable mean is explicitly post hoc sensitivity; the all-seed three-seed result is retained elsewhere in the same figure package.

## Reproduction and QA

Run from the repository root with Python containing matplotlib:

```bash
python scripts/build_imagenet91_paper_figures.py
```

The script validates exact ImageNet key coverage before writing any summary. QA performed for this package:

- 120 rows / 120 expected unique ImageNet cells；FID/KID finite-value check passed。
- contraction CSV independently checked at 6400/12800 kimg for all six seed×NFE rows。
- both PDFs reopen as one-page, unencrypted PDF 1.4 files with no forms or JavaScript。
- both PDFs were rendered through Poppler and visually checked for clipping, label readability, legend completeness, color accessibility, log/symlog disclosure, and separation of frozen versus sensitivity analyses。

## Figure/table trace

```yaml
figure_table_trace:
  - artifact_id: fig-imagenet-per-seed-trajectories
    source_data:
      dataset_id: imagenet64-gap-ab-full120-20260829
      file: results/imagenet64_gap_ab_full120_20260829/per_trajectory.csv
    transformation:
      script: scripts/build_imagenet91_paper_figures.py
    caption_claim: Early IA/IB separation contracts toward near-equivalence for stable seeds, while seed103 becomes unstable and cannot support a contraction interpretation.
    supported_manuscript_claims:
      - claim: ImageNet shows an early uniform gap followed by stable-seed near-equivalence and one unstable seed.
        locator: Figure 4
    limitations:
      - Only three training seeds are available.
      - The seeds101/102 endpoint mean is a post hoc descriptive sensitivity analysis, not the all-seed estimand.
      - Seed103 contraction ratios are arithmetic summaries only after instability.
  - artifact_id: fig-cross-dataset-quality-emergence
    source_data:
      dataset_id: q256-replay-q128-second-q-imagenet64-full120
      file: results/q256_target_weight_replay_curve_seed3_5/fidkid50k-final-20260823/evaluation_results.csv; results/second_q_q128_ab_v2/final/paired_results.csv; results/imagenet64_gap_ab_full120_20260829/per_trajectory.csv
    transformation:
      script: scripts/build_imagenet91_paper_figures.py
    caption_claim: Descriptive finite-budget contrasts vary across the three illustrated settings, while one ImageNet seed becomes unstable under both IA and IB.
    supported_manuscript_claims:
      - claim: The illustrated trajectories are heterogeneous enough that a single pooled endpoint narrative would discard their time structure.
        locator: cross-dataset supplementary illustration
    limitations:
      - Panels use different arm contrasts and NFE settings and are not pooled statistically.
      - The q256 panel uses the seed3–5 replay rather than the planned balanced cohort and must not stand in for the primary q256 result.
      - q256 AULC and ImageNet stable-seed mean are post hoc descriptive summaries.
```
