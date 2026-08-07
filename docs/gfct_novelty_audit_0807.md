# GFCT novelty audit（2026-08-07）

## Decision

**GFCT novelty：CONDITIONAL GO。**

截至 2026-08-07 的定向检索没有发现一项直接前人工作同时满足以下五个条件：

1. 把 consistency-model training gap / discretization interval 当作被干预变量；
2. 固定同一 network parameter state 与同一 adaptive-optimizer state；
3. 对 image、timestep、noise、dropout 与 minibatch aggregation 做完全 paired 的 counterfactual evaluation；
4. 比较实际 optimizer-produced parameter updates，而不是只比较 loss、gradient norm 或 raw gradient；
5. 用可复算的 scalar quotient 与 residual 区分 learning-rate-equivalent 部分和非标量部分，并检验其与 finite-budget quality ranking 的关系。

这支持一个很窄的候选位置：**gap-conditioned, optimizer-state-aware counterfactual update equivalence in consistency training**。它不支持“首次 adaptive discretization”“首次 optimizer-aware update matching”或“首次研究 Adam scale invariance”等宽泛声明。

CONDITIONAL 的原因有两个：其一，bounded search 不能证明不存在未检出的工作；其二，当前 [PR #42](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/42) 的 fresh-state one-step RAdam gate 在数学上退化为 raw-gradient matching，尚未实现真正的 optimizer-state-aware diagnostic。后者见 [`radam_theory_audit_0807.md`](radam_theory_audit_0807.md)。

## Narrow search protocol

### Research question

> 是否已有工作在 consistency-model 的相同参数与 optimizer state 上，对不同 discretization gap 做完全 paired 的 counterfactual optimizer update，并以 scalar quotient / residual 判断它能否被 learning-rate change 吸收？

### Search boundary

- 检索日期：2026-08-07。
- 来源：arXiv、OpenReview、ICLR/ICML/PMLR/CVPR/NeurIPS 官方页面与论文全文；通用网页搜索只用于发现候选，结论尽量回到 primary source 核验。
- 核心 query families：
  - `consistency model + optimizer state`
  - `consistency training + learning rate equivalence`
  - `discretization + Adam scale invariance`
  - `counterfactual optimizer matching`
  - `adaptive optimizer + discretization interval`
- 补充 exact-intersection queries：`consistency models + gradient scale invariance`、`consistency models + update equivalence`、`diffusion/consistency + optimizer-aware timestep/discretization`、`model-update matching + adaptive optimizer`。
- 纳入：直接研究 gradient/update invariance、optimizer-state-aware update matching，或在 consistency/diffusion training 中把 discretization 与 optimizer 几何联系起来的 primary research。
- 排除：仅列出 Adam/RAdam 超参数、只比较最终指标、只做 loss variance、只做 adaptive discretization、或把“consistency model”用于分布式系统语义的结果。

这是有意限定范围的 novelty audit，不是 PRISMA systematic review，也不声称穷尽所有数据库或未来工作。

### Source verification status

- 已发表的 primary sources：Adam（ICLR 2015）、RAdam（ICLR 2020）、LoRA Done RITE（ICLR 2025）、ADCM（NeurIPS 2025）、CCM（CVPR 2025）与 Seesaw（ICLR 2026）。
- Workshop primary source：SCT（ICLR 2025 DeLTa workshop）。
- 尚未按正式同行评审论文计权：`Why Adam Works Better with beta1 = beta2`（arXiv preprint）、`Two-Stage Optimizer-Aware Online Data Selection`（arXiv preprint）与 `Gradient Inversion Attacks Beyond SGD`（ICLR 2026 submission）。它们仍是 novelty boundary 的有效预警，不能被当作不存在。
- 引用存在性与标题/作者/年份以 arXiv、OpenReview 或会议 proceedings 官方页面核验；没有使用 ResearchGate、博客或聚合摘要作为最终 claim 的唯一依据。

## What is already occupied

| Area | Direct work | Occupied claim | Consequence for GFCT |
|---|---|---|---|
| Adaptive discretization | [ADCM, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/84706cdfc192cd0351daf48f379847e6-Paper-Conference.pdf) | 自适应选择 consistency discretization / gap，并讨论有限训练预算 | 不能声称首次 adaptive gap 或首次考虑 finite budget |
| Difficulty curriculum | [CCM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_See_Further_When_Clear_Curriculum_Consistency_Model_CVPR_2025_paper.pdf) | 根据 timestep difficulty 调整 curriculum | 不能声称首次 difficulty-aware schedule |
| Consistency variance reduction | [SCT, ICLR 2025 DeLTa workshop](https://openreview.net/forum?id=5RoPe2ShXx) | 以 score identity 降低 consistency-training target variance | 不能声称首次研究 consistency noise/variance |
| Parameterization and stability | [Continuous-time CM](https://arxiv.org/abs/2410.11081) | 用 continuous-time formulation、parameterization、architecture 与 objective 改善 CM stability | 不能把 parameterization/stability 本身写成 GFCT novelty |

这些工作构成背景边界，但没有直接回答 fixed-state counterfactual optimizer-update equivalence。

## Closest cross-area threats

| Work | What it establishes | Why it is close | Why it does not directly occupy GFCT |
|---|---|---|---|
| [Adam](https://arxiv.org/abs/1412.6980) | Adam 使用一阶/二阶矩并具有 gradient rescaling 相关的不变性性质 | 阻止把“adaptive optimizer 会吸收 gradient scale”当成新观察 | 不研究 consistency gap、paired counterfactual update 或 finite-budget ranking |
| [RAdam](https://arxiv.org/abs/1908.03265) | 分析早期 adaptive learning-rate variance，并以 rectification 构造 RAdam | 直接决定本项目 update operator 的早期状态依赖 | 不比较由不同 discretization interval 诱导的 counterfactual updates |
| [Why Adam Works Better with beta1 = beta2](https://arxiv.org/abs/2601.21739), 2026 preprint | 定义 gradient-scale invariance，并分析 Adam 在 slowly varying scale、transient 消退后的 first-order sensitivity | 是“optimizer-state-aware scale equivalence”最接近的通用理论威胁 | 研究连续时间 Adam scale drift；不研究 RAdam、CM gap、paired data 或 projection quotient。项目配置 `beta1=0.9, beta2=0.999` 也不在其 first-order invariant 条件内 |
| [Seesaw](https://arxiv.org/abs/2510.14717), ICLR 2026 | 在 noisy linear regression 中证明 learning-rate decay 与 batch-size ramp 的有限样本等价，并扩展到 normalized-SGD proxy；另做 AdamW 实验 | 已占据“某种训练干预可与 learning-rate schedule 等价”的宽泛理论表述 | 干预变量是 batch size，理论对象是 SGD/normalized SGD，不是 gap-conditioned fixed-state RAdam quotient |
| [LoRA Done RITE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html), ICLR 2025 | 定义并实现 LoRA parameterization 下的 optimizer transformation invariance | 已占据“比较等价表示经过 optimizer 后是否保持同一 update”的一般思想 | 等价类来自 LoRA factorization，不是 consistency discretization；也不使用 paired stochastic gap counterfactual |
| [Two-Stage Optimizer-Aware Online Data Selection](https://arxiv.org/abs/2604.00001), 2026 preprint | 把 online data selection 表述为 optimizer-aware update-matching | 直接占据术语 `optimizer-aware update matching` | 目标是 LLM data selection/reweighting，不是 CM gap equivalence；因此 GFCT 不能声称首次提出 update matching，只能声称特定交叉应用与诊断设计 |
| [Gradient Inversion Attacks Beyond SGD](https://openreview.net/forum?id=uLdGZhxlxV), ICLR 2026 submission | 对 Adam/AdaGrad/RMSProp 的 observed parameter update 构造 model-update-matching objective，并显式考虑 optimizer state | 说明“从 gradient matching 转向 update matching”也已有直接方法先例 | 目标是隐私攻击与输入重建，不是控制训练干预或估计 gap quotient |

### Synthesis

文献呈现三条已经成熟但尚未在目标交叉点汇合的线：CM 文献研究 discretization、difficulty、variance 与 stability；optimizer 文献研究 scale/transformation invariance；近期工作已把 optimizer state 纳入 update matching。此次检索没有找到把三者合并为 **consistency gap × fixed optimizer state × paired counterfactual update quotient** 的直接工作。

因此，潜在 novelty 不在任一单独组件，而在以下联合对象：

```math
U_s(g;B)
=\operatorname{OptStep}(\theta,s,G_g(B))-\theta,
\qquad
c_s^\star(g)
=\frac{\langle U_s(g;B),U_s(1;B)\rangle}
{\|U_s(g;B)\|_2^2},
```

其中 `s` 必须是固定且非平凡的 optimizer state，`B` 必须在 gap arms 间完全 paired。随后报告

```math
R_s(g)
=\frac{\|c_s^\star(g)U_s(g;B)-U_s(1;B)\|_2}
{\|U_s(1;B)\|_2}.
```

这个 least-squares projection 本身是标准数学工具，不能单独声称新颖；候选贡献是它在 CM gap mechanism 上的受控使用、state-conditioned 漂移测量，以及与 finite-budget outcome 的连接。

## GFCT claim ledger

### Allowed now

- **Raw-gradient diagnostic：GO as supplementary evidence。** [PR #38](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/38) 在一个 q=128 exploratory EMA checkpoint 上使用真实 ECT loss 与 64 个 paired FP32 minibatches，得到近乎共线但非零的 whole-model/layerwise residual。
- **Idea 5 loss variance：not mechanism evidence。** 它可保留为 engineering/controller smoke，但不能支持 gradient covariance、optimizer equivalence 或 GFCT novelty。
- **GFCT novelty：CONDITIONAL GO。** 可写成尚待 state-conditioned RAdam evidence 支持的窄研究问题。

### Pending

- **Optimizer-update diagnostic：PENDING。** 当前 fresh-state first-step 设计只验证 AMP/pairing/state integrity；在 `weight_decay=0` 的 RAdam 下，它不会产生超出 raw-gradient geometry 的 adaptive-optimizer evidence。
- 必须在完全相同、非零的 RAdam moment state 上重放 counterfactual gap updates；至少包含 `successful_optimizer_steps >= 6` 的状态，最好包含预注册的 early/mid/late checkpoints。
- 必须证明 `c_s^star` 或 residual 随 optimizer state 有可复现结构，并检验该结构能否预测 finite-budget FID/KID ranking。

### Prohibited claims

- “GFCT is the first adaptive discretization/difficulty curriculum/variance-reduction method.”
- “GFCT is the first optimizer-aware update-matching method.”
- “The projection quotient itself is novel.”
- “PR #38 proves gap is equivalent to changing learning rate.”
- “A fresh-state one-step RAdam match establishes optimizer-state-aware equivalence.”
- “No prior work exists.”

## Safe novelty wording

> We study a narrowly scoped question not directly covered by the works identified in our bounded search: whether consistency-training gap interventions remain scalar-equivalent after conditioning on the same nontrivial adaptive-optimizer state. Our paired counterfactual diagnostic separates the update component absorbable by a learning-rate multiplier from the residual that cannot be so absorbed.

必须紧接限制：该定位是 **conditional**，直到 nonzero-state RAdam diagnostic、跨状态复现和 finite-budget outcome linkage 完成；若 update residual 消失，结论应收口为 negative diagnostic result，而不是新的 gap mechanism。

## Final status for PR #41

| Item | Verdict |
|---|---|
| PR #38 raw-gradient diagnostic | **GO — supplementary raw-gradient evidence** |
| Optimizer-update diagnostic | **PENDING — fresh first step is insufficient** |
| Idea 5 loss variance | **Excluded from mechanism evidence** |
| GFCT novelty | **CONDITIONAL GO — narrow intersection only** |
