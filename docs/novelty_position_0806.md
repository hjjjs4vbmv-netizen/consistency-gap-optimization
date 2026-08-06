# Novelty position matrix（2026-08-06）

## 一页结论

当前最可防守的定位不是“首次自适应 gap / 首次 difficulty-aware schedule / 首次研究 CM noise”，而是一个更窄、仍待真实深网证据支持的问题：**在同一参数状态与完全 paired minibatch 上，gap 引起的平均 gradient 能否被单一正标量吸收；若能，scalar matching 后是否仍留下 gap-conditioned gradient-noise 或实际 optimizer update residual。** 在 Role D 通过前，这只能写成 diagnostic question / candidate contribution，不能写成已证实贡献。

| 工作 | 已解决什么 | 我们不能声称什么 | 我们仍可能贡献什么 |
|---|---|---|---|
| **ADCM** — *Adaptive Discretization for Consistency Models* ([NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/84706cdfc192cd0351daf48f379847e6-Paper-Conference.pdf)) | 以 local consistency 保证 trainability、global consistency 约束 target denoising error，并用 Gauss–Newton 自适应优化 discretization step；已把训练效率和有限 image budget 纳入实验。 | 不能声称“首次自适应 gap/discretization”；不能把 ADCM 说成 loss-only；不能仅凭“finite budget”宣称区分，也不能说 Role C toy 已证明 ADCM 的瞬时准则失败。 | **Scalar-equivalence diagnosis：**在同一深网参数状态上检验不同 gap 的 mean gradient 是否只是标量重缩放，并报告 ADCM-style observables 之外的 direction/residual。若 residual 稳定且能预测后续训练差异，才形成 optimizer-aware 区分。 |
| **CCM** — *See Further When Clear: Curriculum Consistency Model* ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_See_Further_When_Clear_Curriculum_Consistency_Model_CVPR_2025_paper.pdf)) | 用 PSNR 衡量不同 timestep 的学习难度，动态改变 teacher 的迭代步数，使 consistency distillation 的 curriculum difficulty 更均衡。 | 不能声称“首次 difficulty-aware curriculum/schedule”；不能把任何按 `t` 分 bin 或按难度调 pair 的方法本身当作 novelty；CCM 是 distillation 设定，也不能被写成与本项目完全同一训练机制。 | **Optimizer residual：**在 difficulty 已被 gap/teacher-step 调整之后，测量 scalar-matched gradient residual；若要使用“optimizer”一词，还必须在相同 RAdam state 上复算 candidate update，而不是只报告 loss 或 gradient norm。 |
| **SCT** — *Stable Consistency Tuning* ([arXiv:2410.18958](https://arxiv.org/abs/2410.18958); [OpenReview](https://openreview.net/forum?id=5RoPe2ShXx)) | 把 CM training/tuning 表述为 TD-style value estimation，并利用 score identity 构造 variance-reduced target；已直接研究和降低 consistency training variance。 | 不能声称“首次发现/研究 CM noise 或 variance matters”；不能把 per-example loss variance 当作 SCT 未覆盖的 gradient-noise covariance；也不能把“做了 variance controller”本身写成新机制。 | **Gap-conditioned noise structure：**在固定 `theta`、真实 ECT loss、完全 paired minibatch 下，分解 `Cov(G_g)` 与 scalar-matched paired residual covariance，说明 gap 改变的是哪些 layer/direction，而不只是一个总 variance 数字。 |
| **Role C toy** — bare linear stop-gradient squared-pair dynamics ([audit](../theory/theorem_audit_0805.md)) | 在线性高斯、平方 pair loss、每步独立重采样和 SGD 等条件下，区分 `H_g`、mean update `A_g` 与 second-moment operator `T_g`；给出逐样本 scalar rescaling 导致有限时域等价的充分条件。 | 不能声称 toy 已得到 budget-dependent optimum/crossover；A-matched 后现有 grid 几乎平坦且最优边界不随 budget 改变。不能把 toy 的 parameter second moment 当成深网 generation error，也不能自动外推到 RAdam。 | **Scalar-equivalence baseline / negative control：**用精确/近似等价 toy 校准 Role D estimator，验证 `a_g^star`、mean residual 与 noise residual 的实现；深网只有显著偏离该 baseline，才支持独立 gap mechanism。 |

## Claim ledger

**现在可以写：**

> Prior work already adapts discretization, balances timestep difficulty, and reduces consistency-training variance. We therefore test a narrower optimizer-aware question: whether gap-conditioned minibatch gradients at a fixed network state remain equivalent after scalar matching, and whether any residual has stable directional or covariance structure.

**Role D 通过后才可能写：**

> Under the audited checkpoints and paired minibatch distribution, gap changes leave a reproducible scalar-matched gradient/noise residual that is not explained by loss scale alone.

这仍是 checkpoint- and optimizer-specific empirical statement；除非另有跨 checkpoint、seed、budget 和 optimizer 验证，不得升级为普遍 theorem。

**禁止写：**

- “We are the first to adapt the gap / use difficulty / study CM variance.”
- “ADCM ignores finite budgets”或“ADCM only looks at loss.”
- “Loss variance is gradient-noise covariance.”
- “The Role C toy proves a budget-dependent optimal gap.”
- “Gradient residual equals optimizer residual”——对当前 RAdam 训练，后者需要克隆 optimizer state 后单独测量。

## 决策边界

- **Role D 若 scalar matching 后 residual 约为数值零：**主线收口为 negative result / diagnostic lemma；不要继续包装成新 adaptive schedule。
- **若 gradient residual 非零但 RAdam update residual 消失：**只能声称 gradient geometry 有差异，不能声称 optimizer mechanism。
- **若 paired gradient 与 optimizer residual 均稳定、跨 checkpoint/seed 重现，并能预测 budget ranking：**才有资格把“optimizer-aware gap effect”升级为论文候选贡献。

本矩阵只处理 ADCM、CCM、SCT 与 Role C toy 四个已知直接威胁；不以“未继续扩张搜索”推断不存在其他先行工作。
