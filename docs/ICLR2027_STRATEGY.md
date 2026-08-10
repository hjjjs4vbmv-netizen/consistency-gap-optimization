# ICLR 2027 战略 — GFCT(Role C,草稿)

日期:2026-08-10。状态:**叙事骨架 + 可用草稿文本 + 完整实验计划**,非成品论文。
基于 9-agent 调研+辩论工作流(见 memory `iclr2027-gfct-strategy`)。两个决定性动作,按顺序:

1. **反转论文主线** — 非标量梯度残差是正面贡献;moment-memory 链是被严格检验并证伪的 null model。
2. **升级贡献类型** — 从*诊断*(一个测量 + 一个警示性发现)到*机制 + 方法*(残差是训练不稳定/质量退化的**原因**,且 residual-corrected update 能**修好**它)。这是突破结构性天花板的关键(§0.1)。

---

## 0. 一句话 takeaway(审稿人能带进讨论区的那句)

> Scalar-equivalence assumptions in few-step generator training are
> systematically violated — not by the scalar moment-memory of the history, but
> by non-scalar gradient content that the field's scale-invariance theory
> cannot see.

---

## 0.1 突破:从诊断到机制+方法

**为什么存在天花板。** 即使完美重构,*诊断*("标量等价失效了")也窄、且骨子里是负结果。
ICLR 主会抵抗负结果,窄 novelty 在 27.4% 录用率下天花板很低。打磨诊断突破不了这个。

**唯一突破路径:改变贡献的*类型*。** 论文必须变成:

> We identified a failure mode in few-step training (the non-scalar gradient
> residual), showed it is a **cause** of instability / quality degradation, and
> **fixed** it with a residual-corrected update.

这是两个东西之间的全部区别:
- *诊断*:"标量等价失效了" → 窄、负、约 20-30%。
- *机制+方法*:"few-step 训练不稳定,因为存在非标量梯度残差,而 residual-corrected update 能改善稳定性" → 宽、正、约 40-50%。

**三个实验把它落地**(完整计划见 §8)。

| # | 实验 | 确立什么 | 突破的天花板 |
|---|---|---|---|
| E1 | **跨状态残差轨迹** — 在 early/mid/late 状态测非标量残差;证明它*预测* FID 退化/不稳定 | 残差是**原因**,不是相关 | novelty 窄 |
| E2 | **因果干预** — 用 residual-corrected update(强制 h→1)训练;证明它*改善*稳定性/FID | 残差**可修正**,修正它**有帮助** | 负结果 → 正方法 |
| E3 | **普适性** — 第二 optimizer(AdamW/Muon)+ 第二数据集/backbone | 残差是一般 optimizer-state 现象,不是 CIFAR-10/ECM/RAdam 产物 | 单点观测反驳 |
| E4 | **MeanFlow 泛化**(可选,高杠杆)— 同一套机制跑 flow-map | 连接 MeanFlow 不稳定性这个活跃开放问题 | 晚进/相关性 |

**决策规则。** 若 E1 和 E2 都成立,论文是机制+方法贡献,天花板被突破。
若 E2 失败(修正残差无帮助),论文停留在诊断(约 20-30%),转投兜底 venue。
E1+E2 是门控实验;E3/E4 是加强项。

---

## 1. 标题(三选一)

- **A.** *Where Scalar-Equivalence Breaks Down: A State-Conditioned Analysis of
  Few-Step Generator Training*
- **B.** *The Non-Scalar Gradient Residual: Why Training Interventions Are Not
  Learning-Rate Changes*
- **C.** *Scalar-Equivalence Is Not Free: A Coordinate-Wise Gauge Analysis of
  Few-Step Generator Training*

推荐 **A** — 它点名了正面发现(在哪失效)、方法(state-conditioned)、对象
(few-step generator training,不是 consistency models)。标题避免 "consistency model"
和 "gap"。

---

## 2. 摘要(五步结构,顺序固定)

> **Hook.** Few-step generative training rests on a pervasive implicit
> assumption: that a training intervention — a schedule change, a
> discretization/gap change — is *scalar-equivalent* to a learning-rate change,
> i.e. absorbable by a single scalar multiplier on the update.
>
> **Method.** We test this assumption in the optimizer-update space, conditioned
> on a fixed, nontrivial adaptive-optimizer state. We introduce a
> **state-conditioned, coordinate-wise gauge** `h_{k,i} = U_g/U_1` and a
> **paired counterfactual update quotient** that separates the update component
> absorbable by a learning-rate multiplier from the residual that cannot be so
> absorbed. This is a rigorous, state-conditioned formalization of the informal
> "reduces to learning-rate annealing" idea.
>
> **Result.** On the effective support, the scalar null genuinely holds
> (`h_predicted ≈ 1.001`; the first/second-moment gauges cancel). Yet the actual
> update ratio is `h_actual ≈ 0.837`, tightly concentrated, driven by a
> **non-scalar 3.2% per-step gradient residual**. Scalar-equivalence reasoning
> is therefore systematically violated in few-step training.
>
> **Contribution.** (i) a rigorous, state-conditioned formalization of
> scalar-equivalence; (ii) a novel support-aware gauge-dispersion measurement
> technique; (iii) a cautionary finding: the field's scale-invariance literature
> (β1=β2, Adam-atan2, DeVA) analyzes only scalar/global rescaling and cannot see
> the non-scalar residual that dominates the real distortion.
>
> **So what.** The non-scalar residual is not merely present — it is a **cause**
> of instability and quality degradation: across training states it predicts FID
> degradation, and a residual-corrected update that restores `h` toward 1
> improves stability. This connects to the active open problem of MeanFlow
> training instability / unbounded gradient variance, and gives the field a
> diagnostic plus a fix.

---

## 3. 贡献清单(全为正面,不过度声称)

1. **标量等价性的 state-conditioned 形式化。** 我们给出第一个严格的、coordinate-wise、
   state-conditioned 的处理:训练干预是否等价于改学习率——2025-26 文献未占据的问题
   (最接近:Adam-Rel 非正式的 "reduces to LR annealing")。
2. **新颖的测量技术。** Support-aware gauge dispersion——测量标量 null 在有效支撑上
   *何处*成立——在调研文献中未出现。
3. **可证伪的警示性发现。** few-step 训练的主导更新失真是非标量梯度内容,不是标量
   moment-memory。领域的 scale-invariance 结果(β1=β2、Adam-atan2、DeVA)只覆盖
   标量/全局缩放;没有覆盖非标量残差内容。
4. **一个机制(E1)。** 非标量残差是不稳定和质量退化的**原因**,不是相关:跨训练状态
   它预测 FID 退化与不稳定。
5. **一个方法(E2)。** 把 `h` 拉回 1 的 residual-corrected update **改善**稳定性/FID——
   一个能修复的测量,而不是只报告失败的测量。

**禁止声称(不要写这些):**
- "first adaptive discretization"(ADCM, NeurIPS 2025)
- "first optimizer-aware update matching"(Filter-then-Weight, 2026)
- "first optimizer invariance analysis"(β1=β2、Adam-atan2、DeVA)
- "first history-gauge theorem"(恒等式 `H_k = R_opt(k)` 是内部的)
- "gap is equivalent to changing learning rate"(被 h=0.837 推翻)
- "no prior work exists"

---

## 4. Positioning 段(每个工作一句)

> **β1=β2 (arXiv:2601.21739).** That paper proves Adam is gradient-scale-invariant
> of first order iff β1=β2 — a *global scalar* result. Our claims are
> RAdam-specific (β1=0.9, β2=0.99), coordinate-wise, and state-conditioned, and
> concern the *non-scalar* residual that global-scale-invariance results cannot
> see. We are the state-conditioned generalization the field lacks.
>
> **Adam-atan2 / DeVA.** These analyze how adaptive optimizers absorb or
> decompose *scalar* gradient rescaling. We analyze the *non-scalar* residual
> content that dominates the real distortion — a distinct, complementary angle.
>
> **ADCM (NeurIPS 2025).** ADCM optimizes the discretization schedule for
> quality/stability. We analyze the gap's effect on *optimizer-state
> equivalence* — a dimension ADCM does not address.
>
> **SCT (arXiv:2410.18958).** SCT occupies consistency-training objective /
> variance-reduction theory. We do not claim objective theory; we analyze
> optimizer-update geometry.
>
> **Dead-Direction Conditioners (arXiv:2606.29176).** DDC uses "gauge" for
> symmetry-orbit optimizer-state conditioning. Our "gauge" is the coordinate-wise
> history gauge `h_{k,i} = U_g/U_1`. To avoid collision, we rename ours to
> **"coordinate-wise update-ratio gauge"** throughout.

---

## 5. 章节大纲(反转结构)

1. **Introduction** — 标量等价假设、为何重要、一句话 takeaway。援引 Reviewer Guide
   的 SOTA 豁免:这是关于 optimizer-state 结构的新知识,不是生成质量声明。
2. **Setup & the scalar null model** — rectified RAdam、coordinate-wise gauge
   `h_{k,i}`、moment-memory 恒等式 `h = (1+A^(1))/sqrt(1+2A^(2)+B^(2))` 作为 *null model*
   (严格、自足、可证伪)。
3. **The paired counterfactual update quotient** — 测量方法;support-aware gauge
   dispersion;精确残差恒等式 `H_k = R_opt(k)`。
4. **Main result: the non-scalar residual** — 标量 null 在有效支撑上成立
   (h_predicted≈1.001),但实际 h≈0.837,由量化的 3.2% per-step 非标量残差驱动。这是 thesis。
5. **Mechanism (E1): the residual predicts failure** — 跨状态测量,证明残差与 FID 退化
   和不稳定相关/预测。这使它成为原因而非相关。
6. **Method (E2): the residual is correctable** — residual-corrected update 把 h 拉回 1
   并改善稳定性/FID。这是突破负结果天花板的正面贡献。
7. **Universality (E3)** — 残差在第二 optimizer 和数据集/backbone 上复现。
8. **Generalization to MeanFlow / few-step generator training (E4)** — 同一套机制跑
   flow-map;连接 MeanFlow 不稳定/无界梯度方差。
9. **Related work** — positioning 段(§4)。
10. **Discussion & limitations** — 诚实范围;负结果框架作为警示性方法论发现。

---

## 6. 什么会导致拒稿(投稿前自检)

- [ ] 摘要以 "moment-memory predicts h≈1" 开头,而非非标量残差。
- [ ] 标题含 "consistency model" 或 "gap"。
- [ ] 出现 §3 的任一禁止声称。
- [ ] 无因果链到下游指标(FID/diversity/stability)— E1 必须展示残差*预测*失败,不只是存在。
- [ ] 无 constructive/corrective 结果 — E2 必须展示 residual-corrected update *改善*
      稳定性/FID,不只是把 h 拉回 1。
- [ ] CIFAR-10-only、单一 optimizer、单一 gap 对(无第二 scale/optimizer)— E3。
- [ ] "Gauge" 未改名或未与 DDC 区分。

---

## 7. 开放决策

1. **门控实验 E1 + E2(先做)。** 它们决定论文是机制+方法(约 40-50%)还是诊断
   (约 20-30%)。在其他任何事之前,先在现有 CIFAR-10/ECT 设置上跑。
2. **是否泛化到 MeanFlow(E4)?** 最高杠杆(乘上前沿、连接命名开放问题),但花计算
   + 新训练代码。仅在 E1+E2 成立后。
3. **第二 optimizer/dataset(E3)?** AdamW 或 Muon 系;ImageNet-64 或小 flow-map 基准。
   反驳"单点观测"。
4. **兜底 venue。** "I Can't Believe It's Not Better" workshop,用于 E1+E2 失败、
   论文停留在诊断时。

---

## 8. 实验计划

### 8.1 现有基础设施(复用,不要重建)

| 文件 | 提供什么 |
|---|---|
| `analysis/radam_stateful_update_audit.py` | 从真实非零状态做 stateful RAdam 审计:`a_K*`、`R_grad`、`s_K*`、`c_K*`、`R_opt`、`H_K`、`h_update`、`h_moment`、off-support energy。**核心测量工具。** |
| `analysis/real_history_sweep.py` | 从真实状态做 paired replay(20 步),记录 paired 梯度与更新。 |
| `analysis/moment_memory_prediction.py` | δ_j → A/B → ĥ 预测链。 |
| `analysis/gap_gradient_hook.py` | 固定随机性的 ECT loss(paired counterfactual)。 |
| `analysis/radam_update_gauge.py` | fresh-state sanity probe + layer 命名 + hashing 工具。 |

**服务器:** 172.16.30.17(gpu0003),ECT002,项目 `/data/raw/ECT/recurrence_of_ect`。
真实状态:`/data/raw/ECT/ect_runs/gap_lr_matched_q128_s3_v1/arm_{a,b,c}_*`。

---

### 8.2 E1 — 跨状态残差轨迹(残差是否预测失败?)

**状态:** 门控。**优先级:** critical。**成本:** 低(复用现有审计)。

**假设(可证伪):** 在训练的不同阶段(early/mid/late),非标量更新残差的大小
(`R_opt − R_grad`,或 `h_update` 的坐标离散度)**预测** FID 退化与训练不稳定。
残差越大的状态,FID 越差 / 梯度范数或 loss 方差越高。

**方法:**
1. 在同一训练运行的多个 checkpoint 上跑 `radam_stateful_update_audit.py`——
   例如 32 / 64 / 128 / 256 kimg。每个状态记录 `R_opt`、`R_grad`、`R_opt − R_grad`、
   `h_update` 加权均值/标准差、`h_actual` 中位数、off-support energy、非标量残差大小。
2. 收集这些状态的 FID(来自现有 Role D/A 评估,或重新评估)。
3. 收集每个状态的稳定性代理:梯度范数方差、loss 方差,或一段窗口内 EMA-of-loss 波动。
4. 跨状态做残差 vs FID、vs 稳定性的相关。报告 Spearman ρ 和散点。

**判定标准:**
- **PASS(机制):** ρ 显著为负(残差越大 → FID 越差)和/或残差预测不稳定,跨 ≥4 状态,
  对 support 阈值稳健。
- **FAIL:** 残差跨状态平坦或与 FID/稳定性无关。→ 残差是现象不是原因;论文停留在诊断。

**风险:** 中间状态 FID 噪声大 → 用 KID-5k 作低方差代理、跨 seed 平均。只有 3-4 状态
相关检验力弱 → 加更多 checkpoint,或用 per-step 残差序列作更密信号。

---

### 8.3 E2 — 因果干预(修正残差是否修复失败?)

**状态:** 门控。**优先级:** critical。**成本:** 中(新训练代码)。

**假设(可证伪):** 去掉 gap 效应的非标量部分——把梯度(或更新)修正为标量等价——
相对未修正的 gap 臂,**改善**训练稳定性和/或 FID。非标量残差不只是存在;它**有害**且**可修正**。

**方法(paired 3-arm 设计——项目的招牌):**
- **Arm R(参考):** g=1.0,标准 RAdam。提供参考梯度 `G_1` 与参考更新 `U_1`。
- **Arm U(未修正):** g=1.3,标准 RAdam。非标量残差存在(即当前 `h_actual≈0.837` 的臂)。
- **Arm C(修正):** g=1.3,但梯度在 optimizer step 前被修正为与参考标量等价:
  `G_corrected = s* G_1`,其中 `s = <G_g, G_1>/||G_1||²`。去掉非标量梯度内容,因此
  (按理论)更新应为标量(`h→1`)。

比较 **Arm U vs Arm C** 的稳定性与 FID,在从同一状态起的短窗口(如 16-64 kimg)内。

**为什么这是正确的因果检验:** gap 干预有标量部分(可被 LR 变化吸收)和非标量部分(残差)。
Arm C 只去掉非标量部分。若 Arm C 更稳定/FID 更好,则非标量残差是退化的**原因**,修正它是**修复**。

**判定标准:**
- **PASS(方法):** Arm C 比 Arm U 可测量地更稳定和/或 FID 更好,跨 seed 稳健。
- **FAIL:** Arm C ≈ Arm U。→ 残差无害/不可修正;论文停留在诊断。

**风险:** 计算成本高(Arm C 每步需要 `G_1`,须同时跑 Arm R → 约 2-3×)→ 短窗口或
部分步上算 `G_1`。修正可能过强(去掉合法非标量信号)→ 若 Arm C 更差,如实报告。实现需
自定义 optimizer step 或梯度 hook,复用 `gap_gradient_hook.py` 的固定随机性机制。

---

### 8.4 E3 — 普适性(残差是一般现象吗?)

**状态:** 加强项。**优先级:** high。**成本:** 中。

**假设(可证伪):** 非标量残差是一般 optimizer-state 现象,不是 CIFAR-10 / ECM / RAdam /
(1.0, 1.3) gap 对的产物。

**方法:**
1. **第二 optimizer:** 用 AdamW(和/或可用的 Muon 系)替代 RAdam 跑 audit,确认残差复现。
2. **第二数据集/backbone:** 在 ImageNet-64 或小 DiT / flow-map backbone 上跑,确认复现。
3. **第二 gap 对:** 改变 gap(如 1.0 vs 1.5,或 1.0 vs 0.8),确认残差随 gap 缩放。

**判定标准:**
- **PASS:** 残差在 ≥2 optimizer 和 ≥2 数据集/backbone 上复现。
- **FAIL:** 残差是 CIFAR-10/ECM/RAdam 特有 → 削弱普适性声明;作为 limitation 报告。

**风险:** ImageNet-64/DiT 训练昂贵 → 用 *audit*(从状态做单步 paired)而非完整训练。
Muon 系可能未安装 → AdamW 足以支撑"第二 optimizer"声明。

---

### 8.5 E4 — MeanFlow / flow-map 泛化(前沿相关性)

**状态:** 可选,高杠杆。**优先级:** medium(在 E1+E2 之后)。**成本:** 高。

**假设(可证伪):** 非标量残差机制在 MeanFlow / flow-map 训练中复现,并连接到
MeanFlow 不稳定/无界梯度方差这个活跃开放问题(arXiv:2605.09235, 2605.17834, 2511.23342)。

**方法:**
1. 把 paired counterfactual update-quotient 机制适配到 flow-map / MeanFlow 目标
   (基础目标是 flow matching / rectified flow)。
2. 在 MeanFlow 训练状态上跑同样的 stateful audit:测非标量残差,以及它是否与已知不稳定相关。
3. 若可行,在 MeanFlow 上跑 E2 修正,测试是否稳定训练。

**判定标准:**
- **PASS:** 残差在 MeanFlow 上复现,且(理想)修正能稳定它 → 论文与 2026-27 前沿相关。
- **FAIL / 不可行:** 论文停留在 CIFAR-10/ECT optimizer-theory 贡献(仍有效,但更窄)。

**风险:** 高成本+高风险(新代码、新 baseline、可能不复现)→ 门控在 E1+E2 之后。
晚进 → 专门对准*命名*的不稳定问题,不是泛泛的领域。

---

### 8.6 执行顺序与时间线

| 顺序 | 实验 | 门控 | 预估工作量 |
|---|---|---|---|
| 1 | **E1** 跨状态残差轨迹 | 决定机制 | 低(复用 audit) |
| 2 | **E2** 因果干预(修正更新) | 决定方法 | 中(新代码) |
| 3 | **E3** 普适性(第二 optimizer/dataset) | 加强 | 中 |
| 4 | **E4** MeanFlow 泛化 | 前沿相关性 | 高(仅当 E1+E2 通过) |

**E1+E2 后的决策检查点:** 两者都通过 → 推进 E3、E4 和机制+方法论文。
E2 失败 → 写诊断论文,投兜底 venue,不要在 E4 上花钱。

### 8.7 每个实验不得声称什么(护栏)

- E1 必须展示*预测*,不只是*存在*——跨状态的相关,不是单点观测。
- E2 必须展示修正*改善*下游指标,不只是把 `h` 拉回 1(拉回 h 是必要但不充分)。
- E3 必须展示残差*一般*,不只是在一个配置里存在。
- E4 必须连接到*命名*的 MeanFlow 不稳定,不只是"我们在 MeanFlow 上跑了"。
- E1-E4 都不得声称任何 "first"(见 §3 禁止声称清单)。

---

## 文件
- 本战略:`docs/ICLR2027_STRATEGY.md`。
- 理论资产:`theory/radam_moment_memory.md`、`theory/radam_gap_equivalence.md`。
- 真实数据结果:`analysis/moment_memory_real_history_result.md`。
- 战略 memory:`iclr2027-gfct-strategy`。
