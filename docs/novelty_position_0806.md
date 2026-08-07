# Novelty position matrix（2026-08-06；更新于 2026-08-07）

## 一页结论

已有工作已经覆盖 adaptive discretization、difficulty-aware curriculum 和 consistency-training variance reduction。当前可防守的新位置更窄：**PR #38 已在一个真实 q=128 exploratory EMA checkpoint 上提供完全 paired 的 raw-gradient scalar-equivalence evidence；尚未解决的是 common fresh RAdam/GradScaler state 下的 update residual，以及它是否能解释 finite-budget ranking。**

| 工作 | 已解决什么 | 我们不能声称什么 | 我们仍可能贡献什么 |
|---|---|---|---|
| **ADCM** — *Adaptive Discretization for Consistency Models* ([NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/84706cdfc192cd0351daf48f379847e6-Paper-Conference.pdf)) | 用 local/global consistency 与 Gauss–Newton 自适应优化 discretization step，并研究有限训练预算与训练效率。 | 不能声称首次 adaptive gap；不能说 ADCM 是 loss-only 或忽略 finite budget；PR #38 也没有证明 ADCM 的瞬时准则失败。 | **Scalar-equivalence diagnosis：**PR #38 显示单 checkpoint 上 raw mean gradients 近乎共线但不完全等价；未来可检验这种 residual 是否跨 state 稳定并提供 ADCM observables 之外的信息。 |
| **CCM** — *See Further When Clear: Curriculum Consistency Model* ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_See_Further_When_Clear_Curriculum_Consistency_Model_CVPR_2025_paper.pdf)) | 用 PSNR 衡量 timestep difficulty，动态改变 teacher steps，使 consistency-distillation curriculum 更均衡。 | 不能声称首次 difficulty-aware schedule，也不能把按 `t` 分 bin 或调 pair difficulty 本身作为 novelty；CCM 与本项目训练设定不完全相同。 | **Optimizer residual：**在 difficulty/gap 已改变 raw gradient 后，测量 fresh RAdam state 对 scalar relation 的吸收或破坏；该项仍未实现。 |
| **SCT** — *Stable Consistency Tuning* ([arXiv:2410.18958](https://arxiv.org/abs/2410.18958); [OpenReview](https://openreview.net/forum?id=5RoPe2ShXx)) | 用 score identity 构造 variance-reduced target，直接研究并降低 consistency training variance。 | 不能声称首次研究 CM noise；不能把 per-example loss variance 当 gradient covariance，也不能把“variance controller”本身写成新机制。 | **Gap-conditioned noise structure：**PR #38 已报告 raw minibatch-gradient variance trace 与 layerwise residual；仍需 fresh-state、paired residual covariance、跨 checkpoint/seed 重现及与质量 ranking 的联系。 |
| **Role C toy** — bare linear stop-gradient squared-pair dynamics ([audit](../theory/theorem_audit_0805.md)) | 区分 `H_g`、mean update `A_g` 与 second-moment operator `T_g`，并给出逐样本 scalar rescaling 导致有限时域等价的充分条件。 | 不能声称 toy 存在 budget-dependent optimum；A-matched 后当前结果近乎平坦。不能把 parameter second moment 当生成误差或自动外推到 RAdam。 | **Scalar-equivalence baseline：**它为 PR #38 的 `a_g^star` 和 residual 提供 negative-control 基线；PR #38 的小而非零 residual 是深网 observation，不是新的普遍 theorem。 |

## 当前可写的证据声明

> On one exploratory q=128 EMA checkpoint, 64 paired FP32 minibatches show that changing the global gap multiplier primarily rescales the whole-model raw mean gradient: all mean-gradient cosines exceed 0.9999 and the largest whole-model directional residual is 1.35%, while some layerwise residuals reach 12.41%.

必须紧接限制：这是 [PR #38](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/38) 的 single-checkpoint supplementary raw-gradient evidence；optimizer 未创建或 step，不能推出 RAdam update equivalence、学习率替代、formal endpoint 或生成质量改善。

## 仍待完成的候选贡献

[PR #42](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/42) 所需的 fresh-state virtual RAdam audit 尚未完成。只有它给出有限、可复算的 `c0_star`、update cosine/residual、AMP unscale 与 state-invariance evidence 后，才能把 Arm C 称为 **initialization-level one-step RAdam-update matched control**；即使通过，也不能声称整个训练轨迹始终 matched。

## 禁止声明

- “We are the first to adapt the gap / use difficulty / study CM variance.”
- “ADCM ignores finite budgets”或“ADCM only looks at loss.”
- “Loss variance is gradient-noise covariance.”
- “PR #38 proves gap is equivalent to learning-rate scaling.”
- “The Role C toy proves a budget-dependent optimal gap.”
- “Raw-gradient residual equals optimizer residual.”

本矩阵只处理 ADCM、CCM、SCT、Role C toy 与当前 PR #38/#42 的直接边界；不以停止扩张搜索推断不存在其他先行工作。
