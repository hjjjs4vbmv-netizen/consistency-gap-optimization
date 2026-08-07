# Role D 深网诊断分层审计（2026-08-06；更新于 2026-08-07）

## 修订结论

此前统一写成“Role D gradient diagnostic：NO-GO”已经落后于当前 PR 状态。正确结论必须按证据对象拆成三层：

| 对象 | 当前判定 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| Idea 5 / loss-variance controller，以及无梯度的 EMA consistency-error plan | **NO-GO** | loss-variance controller engineering smoke、consistency-error 描述 | 不能替代 paired minibatch-gradient、mean operator、gradient covariance 或 optimizer-update 证据 |
| [PR #38](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/38) paired raw-gradient diagnostic，head `5f69c0c8` | **GO — supplementary raw-gradient evidence** | 一个真实 q=128 exploratory EMA checkpoint 上，真实 ECT objective、固定参数、64×128 完全 paired minibatch 的 mean-gradient scalar fit、方向 residual、variance trace 与 layerwise residual | 不能推出 RAdam update equivalence、学习率替代、formal endpoint、跨 checkpoint/seed/q/budget 泛化或生成质量因果解释 |
| [PR #42](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/42) 所需的 common fresh-state RAdam virtual-update diagnostic | **NO-GO — 尚未实现** | 通过后可为 Arm C 提供 initialization-level `c0_star` 和 update residual | 当前不能解锁 formal training；PR #38 的 raw-gradient `a_g^star` 不得代替 `c0_star` |

因此，**不要原样合并旧版 PR #41**。修订后的 #41 只承担分层审计和 claim boundary；保持 Draft，待 #38/#42 的状态与引用再次核对后再审。

## 审计对象与版本

- PR #41 delivery base：`e505085de2f68dad01ef6e3bfdf28f82489c9693`。
- PR #38 raw-gradient head：`5f69c0c8ec732e2ed7bd9e13136fd3cd664d2f17`。
- PR #42 gap × learning-rate protocol head：`77d86df1b81f7e7616614db521b7452b2ab1c0d2`。
- Idea 5 loss-variance branch：`b20aa3dbb15542a0a17076ecc928ac5ff02299b0`。

本报告区分 repository state、open-PR evidence 与尚未实现的 gate；不会因为 #38 尚未进入 `main` 就写成“诊断不存在”，也不会把 open PR 当成已合并的正式主线证据。

## PR #38：六项 raw-gradient 审计

| 检查项 | 判定 | 核验结果 |
|---|---|---|
| 使用真实 loss | **PASS** | controlled runner 复现 checkpoint `loss_fn` 的 `c/stage/q/k/b`、target stop-gradient、shared dropout、平方 pair error、Pseudo-Huber/范数变换、`1/(t-r)` weighting 与 `ECMPrecond`。`tests/test_deepnet_gap_gradient_moments.py` 以固定随机性逐值比对 repository `ECMLoss` 的 loss 和 gradient。 |
| 完全 paired | **PASS（within-checkpoint）** | 同一 EMA 参数与 buffer 上，每个 minibatch 的 images、per-example `t`、shared `eps`、labels 和 dropout RNG 在所有 `g` 间固定；只有 `r_g(t)` 改变。参数/buffer pre/post SHA 相同，optimizer 未创建、未 step。 |
| `a_g^star` 定义 | **PASS** | 先向量式累加 minibatch gradients 得到 `mu_g`，再计算 `a_g^star=<mu_g,mu_1>/||mu_1||^2`；没有使用 gradient-norm ratio。 |
| variance 按 minibatch gradient 计算 | **PASS（descriptive）** | 使用 `B^-1 sum_b ||G_{g,b}-mu_g||^2` 的 covariance trace 和 normalized noise scale，而不是 per-example loss variance。它是固定 64 个 minibatch 的 descriptive population moment；若外推抽样总体，可另报 `B-1` 修正。 |
| 未把 gradient norm 当 mean operator | **PASS** | 实现保存每个参数张量的 gradient vector sum，并以 vector inner product 计算 whole-model/layerwise mean fit、cosine 和 residual；norm 仅作为 scale summary。 |
| AMP scaling 已解除 | **N/A for #38；optimizer gate 仍 FAIL** | #38 是 FP32 raw-gradient diagnostic，没有 `GradScaler` scaling，因此不存在读取 scaled `.grad` 的问题。但它没有 fresh AMP/GradScaler state，也没有 RAdam virtual step，不能满足 #42 的 update gate。 |

### PR #38 可报告的结果

在单个 q=128、seed 3、1000.064 kimg exploratory EMA checkpoint 上：

- 64 个确定性 minibatch，每批 128 张 CIFAR-10 图像；`g in {0.9,1.0,1.2,1.3}`。
- whole-model mean-gradient cosine 均大于 `0.9999`。
- 最大 whole-model directional residual 为 `1.349%`（`g=1.3`）。
- `g=1.3` 的 `a_g^star=0.75539`；normalized noise scale 在 gap sweep 上约为 `0.041`。
- 最大 layerwise residual 为 `12.41%`（`g=1.3`）。
- 参数 SHA、buffer SHA 在诊断前后相同；`optimizer_created=false`、`optimizer_steps=0`。

安全解释是：**在该 checkpoint 和 paired minibatch distribution 上，gap 主要重缩放 whole-model raw mean gradient，但留下小的整体方向 residual 和更明显的局部 layer residual。** 这是 supplementary mechanism observation，不是“gap 等价于 learning-rate scaling”的证明。

## 统计对象及 PR #38 覆盖范围

对固定 checkpoint 的 paired minibatch gradients：

```math
G_{g,b}=\nabla_\theta L_{g,b}(\theta),
\qquad
\widehat\mu_g=\frac1B\sum_bG_{g,b},
```

```math
\widehat a_g^\star
=\frac{\langle\widehat\mu_g,\widehat\mu_1\rangle}
{\|\widehat\mu_1\|_2^2},
\qquad
R_{\mathrm{mean}}(g)
=\frac{\|\widehat\mu_g-\widehat a_g^\star\widehat\mu_1\|_2}
{\|\widehat\mu_g\|_2}.
```

PR #38 正确实现上述 mean-operator projection，并报告每个 gap 的 raw-gradient variance trace。它还以相同 `a_g^star` 报告 per-minibatch directional residual；但没有估计 fresh RAdam state-conditioned parameter update，也没有把 raw-gradient residual 解释为 optimizer residual。

## Fresh-state RAdam update gate：仍为 NO-GO

PR #42 的 Arm C 需要在共同的 pretrained EDM initialization、fresh RAdam state 和 fresh GradScaler state 上，以完全 paired inputs 计算不提交的 virtual updates：

```math
\Delta\theta_{1.0},\qquad \Delta\theta_{1.3}.
```

其学习率 multiplier 与 residual 定义为：

```math
c_0^\star
=\frac{\langle\Delta\theta_{1.3},\Delta\theta_{1.0}\rangle}
{\|\Delta\theta_{1.3}\|_2^2},
\qquad
R_{\mathrm{update}}
=\frac{\|c_0^\star\Delta\theta_{1.3}-\Delta\theta_{1.0}\|_2}
{\|\Delta\theta_{1.0}\|_2}.
```

正式放行前仍必须：

1. 从同一 fresh network/RAdam/GradScaler state 克隆两个 arm；不得使用已训练 EMA 作为替代。
2. images、minibatch order、`t`、`eps`、dropout RNG、microbatch accumulation 与 AMP precision 完全 paired。
3. 在 `scaler.unscale_(optimizer)` 后读取 gradient，并验证两个 arm 使用同一有效 GradScaler state。
4. 复算 non-committing RAdam update，报告 update cosine、norm、whole-model/layerwise `c0_star` 与 residual。
5. 参数、optimizer state、GradScaler state 在 virtual diagnostic 前后哈希不变。
6. 按 #42 smoke 记录重放共同的 AMP-skip 前缀，并在第一个共同 non-skipped attempt 上估计 `c0_star`。

PR #38 的 `a_{1.3}^star` 只能描述 raw mean-gradient geometry，**不得填入 PR #42 的 `C0_STAR`，不得解锁 Arm C 或 formal launch**。

## Acceptance decision

- **Idea 5 / loss variance：NO-GO as gradient evidence。** 可保留 engineering smoke。
- **PR #38：GO as supplementary raw-gradient evidence。** 可引用其严格限定的单-checkpoint observation。
- **PR #42 fresh-state RAdam update gate：NO-GO。** 在 `c0_star`、update residual、AMP unscale 与 state-invariance audit 完成前，formal training 继续阻断。

PR #41 应保持 Draft；重审时只检查上述三层状态是否仍与 #38/#42 最新 head 一致。
