# Role D 深网诊断协议审计（2026-08-06）

## 结论

**正式运行：NO-GO。** 当前仓库有真实 ECT loss 和正确的训练期 AMP `unscale_` 顺序，但尚无满足研究问题的 Role D gradient diagnostic。工作树候选 `diagnostics_plan.md`（不属于本次交付）是无梯度、EMA 网络上的 consistency-error 诊断；`experiment/idea5-variance-256step` 则把 per-example loss variance 当作控制信号，并且只跑一个 adaptive arm。二者都不能回答“改变 gap 后，真实深网的平均 minibatch gradient 是否仅发生 scalar rescaling，以及 scalar matching 后是否仍有 gradient-noise / optimizer residual”。

交付基线：`origin/main` 的 `e505085de2f68dad01ef6e3bfdf28f82489c9693`；Idea 5 分支 `b20aa3dbb15542a0a17076ecc928ac5ff02299b0`；Role C clean audit `8c2a79d174cf50ad93c9e2f86b854cd837e39508`。本报告是静态 protocol/code audit，没有把未运行的 GPU 诊断写成实验结果。

## 六项审计

| 检查项 | 判定 | 证据与问题 | 正式运行前的阻断条件 |
|---|---|---|---|
| 使用真实 loss | **CONDITIONAL PASS** | 当前 [`training/loss.py`](../training/loss.py) 先采样 `t`、由 schedule 得到 `r`，对 pair 共享 `eps` 和 dropout mask，target branch 在 `no_grad()` 中，再做像素平方和、Pseudo-Huber/平方根变换和 `1/(t-r)` weighting（lines 225–280）。Idea 5 的 `observe_training_batch()` 接收的也是最终 `weighted_loss`，不是假的 proxy；但它只统计 loss，不产生所需 gradient。 | 新诊断必须直接调用仓库 `ECMLoss`，从 checkpoint/training options 恢复 `P_mean/P_std/q/k/b/c/stage`、preconditioner、label/augmentation 设定；不得以 `raw_sq_error`、unweighted residual 或手写简化 loss 代替。 |
| 完全 paired | **FAIL** | Idea 5 的 formal command 只训练 `adaptive_variance_v1` 一个 arm；`diagnostics_plan.md` 虽给 fixed/global 相同输入，却比较两个已经分化的 EMA checkpoint，仍混入参数状态差异。 | 在**同一 online 参数状态** `theta` 上做 counterfactual gap evaluation；每个 minibatch 的 image index、label、augmentation、`t`、`eps`、dropout RNG、microbatch 划分、参数集合和 precision 必须相同，唯一变化是由 `g` 决定的 `r`。诊断不得执行 optimizer step。 |
| `a_g^star` 定义正确 | **FAIL / absent** | 当前 Role D/Idea 5 路径没有定义或计算 `a_g^star`。Role C 的 scalar projection 不能由 loss ratio 或 gradient-norm ratio 替代。 | 用未缩放 minibatch gradient 的**均值向量**定义 `a_g^star`，见下式；同时输出 denominator、sign、cosine 和 projection residual。若 `a_g^star <= 0`，不得解释为正学习率重标定。 |
| variance 按 minibatch gradient 计算 | **FAIL** | Idea 5 的 `training/schedules.py:403–440` 对 per-example weighted loss 累积 `sum(loss)` 与 `sum(loss^2)`，计算的是 loss dispersion `Var(loss)/E(loss)^2`；不是 `Cov(grad L_B)`。 | variance 的抽样单位必须是独立、固定大小的 minibatch gradient。至少报告 covariance trace 和 scalar-matched paired residual 的 centered variance；不能用 per-example loss variance、per-sample gradient norm variance或跨 step training-loss variance冒充。 |
| 未把 gradient norm 当 mean operator | **FAIL / not measured** | 当前实现只记录 loss、gap、controller state 和 GradScaler，没有累加 gradient vector，因此根本没有估计 mean operator。只保存 `E||G||` 或 `E||G||^2` 也无法恢复 `||E G||`。 | 必须先向量式累加 `mu_g = mean_b G_{g,b}`，再计算投影与 cosine；gradient norm 只能作为附加 scale diagnostic。 |
| AMP scaling 已解除 | **CONDITIONAL PASS** | 正式训练路径在 backward 后先执行 `scaler.unscale_(optimizer)`，再读取/清理 gradient，并在其后 `scaler.step()`（[`training/ct_training_loop.py`](../training/ct_training_loop.py), lines 862–917），顺序正确。但当前没有 gradient diagnostic，因此尚无“采集点位于 unscale 之后”的实现证据。 | 最稳妥的主诊断用 FP32、unscaled objective；若复用 AMP 路径，必须在每个 arm 的 `scaler.unscale_(optimizer)` **之后**复制 gradients，并记录 `scale_before`、finite check 与 skip 状态。禁止读取 `scaler.scale(loss).backward()` 后、`unscale_` 前的 `.grad`。 |

## 正确的统计对象

固定一个训练 checkpoint 的 online network、optimizer state 和 schedule stage。对预先声明的 `B` 个 paired minibatch，令

```math
G_{g,b}=\nabla_\theta\left[\frac{1}{m}\sum_{i=1}^{m}
\ell_g(x_{b,i},t_{b,i},\epsilon_{b,i};\theta)\right],
\qquad
\widehat\mu_g=\frac{1}{B}\sum_{b=1}^{B}G_{g,b}.
```

这里的一个 `b` 对应一次完整 optimizer attempt（包括与正式训练相同的 microbatch accumulation）；`G_{g,b}` 是 AMP 已解除 scaling、DDP 已按训练语义聚合后的完整参数 gradient。Role C 平均算子的深网对应物是 `mu_g`，不是 `mean_b ||G_{g,b}||`。

相对 reference gap `g=1` 的 primary scalar 定义为

```math
\widehat a_g^\star
=\arg\min_{a\in\mathbb R}\|\widehat\mu_g-a\widehat\mu_1\|_2^2
=\frac{\langle\widehat\mu_g,\widehat\mu_1\rangle}
{\|\widehat\mu_1\|_2^2}.
```

不得改成 `||mu_g||/||mu_1||`：norm ratio 丢失方向信息，反向或正交的 mean gradient 也可能得到看似合理的正标量。若 `||mu_1||^2` 低于预先声明的数值阈值，该 checkpoint/bin 应报告 `INSUFFICIENT_MEAN_SIGNAL`，不能继续除法。

若另行关心“哪个 scalar 对整个 paired stochastic gradient 的均方拟合最好”，可附加报告

```math
\widetilde a_g
=\frac{\sum_b\langle G_{g,b},G_{1,b}\rangle}
{\sum_b\|G_{1,b}\|_2^2}.
```

但 `tilde a_g` 混合了 mean 与 noise，不能替代上面的 `a_g^star`，也不能被称为 mean-operator matching coefficient。

应同时报告：

```math
\text{mean_residual}_g
=\frac{\|\widehat\mu_g-\widehat a_g^\star\widehat\mu_1\|_2}
{\|\widehat\mu_g\|_2+\varepsilon},
\qquad
\cos(\widehat\mu_g,\widehat\mu_1).
```

minibatch gradient-noise 的至少一个有效标量摘要是 covariance trace：

```math
\widehat V_g
=\operatorname{tr}(\widehat\Sigma_g)
=\frac{1}{B-1}\sum_{b=1}^{B}\|G_{g,b}-\widehat\mu_g\|_2^2.
```

完全 paired 设计还应直接测量 scalar matching 后的 stochastic residual：

```math
R_{g,b}=G_{g,b}-\widehat a_g^\star G_{1,b},
\qquad
\widehat V_g^{\mathrm{res}}
=\frac{1}{B-1}\sum_b\|R_{g,b}-\overline R_g\|_2^2.
```

`V_g` 回答各 gap 自身的 minibatch noise 大小；`V_g^res` 回答共享数据、`t`、噪声和 dropout 后，gap 是否留下不能由一个标量解释的 paired noise structure。可按 layer/block 重复上述 inner product 与 covariance-trace 汇总，但不能先把每层压成 norm 再称为 operator。

## “gradient residual”与“optimizer residual”的边界

仓库正式训练使用 RAdam。上述量是 **gradient-operator diagnostics**；在没有复算 RAdam state-conditioned update 前，文稿不得称其为“optimizer residual”。若要使用后者，必须对每个 `(g,b)` 克隆完全相同的 `theta` 与 optimizer state，以 `G_{g,b}` 计算但不提交参数变更的候选 update `U_{g,b}`，再对 `U` 定义独立的 `a_{g,opt}^star`、mean residual 和 paired variance。不能把 SGD 下的 scalar-equivalence theorem 自动外推到带动量/自适应预条件的 RAdam。

## 最小可运行协议

1. **冻结来源。** 记录 code commit、training-state SHA-256、dataset SHA-256、online-network key、optimizer-state hash、stage、参数 mask、gap grid、minibatch size、`B` 和 bootstrap seed。若只有 `network-snapshot-*.pkl` 中的 EMA，则输出只能叫 EMA checkpoint gradient proxy，不能叫实际 optimizer diagnostic。
2. **显式生成 paired inputs。** 在 arm 外生成并保存/重放 image indices、labels、augmentation parameters、`t`、`eps` 与 dropout RNG state。不要只依赖“相同 seed”这一弱保证。
3. **真实 loss、单一差异。** 每个 minibatch 从同一 `theta` 分别计算 `g=1` 与候选 `g`；仅 `r_g(t)` 可变。target detach、loss transform、`1/(t-r)`、microbatch averaging 与训练保持一致。
4. **无更新采集。** 每个 arm 前 `zero_grad(set_to_none=True)`；backward 后先解除 AMP scaling，再复制相同参数顺序的 gradient；不调用 `optimizer.step()`，不更新 EMA/controller/RNG stream。
5. **两阶段估计，避免拟合后自评。** 用预先划分的 fit minibatches 估计 `a_g^star`，在独立 evaluation minibatches 报 mean/noise residual；若算力只允许单一集合，必须标注 in-sample diagnostic，并用 paired bootstrap over minibatches 给区间。
6. **输出完整而非只报 headline。** 保存 `a_g^star`、mean cosine、mean residual、`V_g`、`V_g^res`、分层摘要、finite counts、AMP scale/precision、pairing hashes；NaN/Inf、pairing mismatch 或参数顺序不一致均非零退出。

## 正式运行 acceptance tests

- `g=1` 对自身给出 `a_1^star=1`、mean/noise residual 为数值零；交换 reference/candidate 后 paired raw differences 符号翻转。
- 打乱 candidate 的 minibatch 顺序会被 pairing hash 拒绝，而不是继续聚合。
- 构造 `G_g=cG_1` 的 synthetic gradient fixture，可恢复 `a_g^star=c` 且两个 residual 近零；构造等 norm 的正交 gradient 时 norm ratio 仍为 1，但 mean residual 必须非零。
- FP32 主诊断与 AMP-unscaled smoke 在预先声明容差内一致；故意在 `unscale_` 前读取 gradient 的 negative test 必须失败。
- loss parity test 在固定输入上逐项核对仓库 `ECMLoss` 返回值；raw squared error 或 unweighted loss 不得通过。
- 使用不同 online parameter SHA、batch/microbatch、`t`、`eps`、augmentation 或 dropout state 的两个 arm 必须在 backward 前失败。

## Role D 放行条件

只有在上述 acceptance tests 通过、并且 formal command 同时含同一 checkpoint 上的 reference/candidate paired arms 后，才允许从 `NO-GO` 改为 `GO`。当前 Idea 5 的短训练可保留为 **loss-variance controller engineering smoke**，但不能计入 scalar-equivalence、gradient-noise 或 optimizer-residual 证据。
