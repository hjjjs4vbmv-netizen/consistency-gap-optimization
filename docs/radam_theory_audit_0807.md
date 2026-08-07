# RAdam theory audit for gap equivalence（2026-08-07）

## Decision

**当前 PR #42 所定义的 fresh-state one-step RAdam diagnostic 不能被称为 optimizer-state-aware evidence。**

在仓库配置

```text
optimizer = torch.optim.RAdam
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 0  # torch default; repository does not override it
m0 = 0
v0 = 0
step0 = 0
```

下，首个成功 RAdam step 进入未整流分支，parameter update 精确退化为 `-lr × raw_gradient`。因此 fresh-state `c0_star` 与 update residual 只是同一批 raw gradients 的反向投影系数与方向 residual；它不会测到 accumulated first moment、second moment preconditioning 或 RAdam rectification。

结论分层如下：

| Diagnostic | Verdict | Meaning |
|---|---|---|
| PR #38 paired raw-gradient diagnostic | **GO** | supplementary fixed-checkpoint gradient geometry |
| Fresh-state first successful RAdam virtual step | **GO only as implementation sanity check** | 验证 real loss、pairing、AMP unscale、skip handling 与 non-commit；不是新的 optimizer geometry |
| Nonzero-state paired RAdam virtual update | **PENDING / required** | 真正回答 optimizer-state-aware gap equivalence |

## Audited implementation facts

- [`ct_train.py`](../ct_train.py) 构造 `torch.optim.RAdam`，显式传入 `lr`、`betas=[0.9,0.999]`、`eps=1e-8`，没有覆盖 `weight_decay`，因此使用 PyTorch 默认值 `0`。
- [`training/ct_training_loop.py`](../training/ct_training_loop.py) 在 AMP 路径中先执行 `scaler.unscale_(optimizer)`，再调用 `scaler.step(optimizer)`；overflow 时 optimizer step 被跳过。
- 已有 Role D manifest 记录远端诊断环境为 PyTorch `2.6.0+cu124`；本地复核环境为 PyTorch `2.13.0`。正式实现仍应把远端 `torch.optim.RAdam` source hash 或版本固定进 receipt。
- [PyTorch RAdam documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.radam.RAdam_class.html) 给出的算法在 `rho_t <= 5` 时使用未整流 update，在 `rho_t > 5` 后才引入 second-moment adaptive denominator 与 rectification factor。
- [RAdam paper](https://arxiv.org/abs/1908.03265) 的核心对象是 early-stage adaptive learning-rate variance 与 rectification；这正是不能用 raw-gradient equivalence 自动代替的后续状态效应。

## Fresh-state derivation

令 gap `g` 在 paired minibatch `B` 上产生未缩放 gradient

```math
G_g=\nabla_\theta L_g(\theta;B).
```

RAdam 的 moment recurrence 为

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)G_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)G_t^2.
```

在 fresh state `m_0=v_0=0`、`t=1`：

```math
m_1=(1-\beta_1)G_g,
\qquad
\widehat m_1=\frac{m_1}{1-\beta_1}=G_g.
```

对于 `beta2=0.999`，

```math
\rho_\infty=\frac{2}{1-\beta_2}-1=1999,
\qquad
\rho_1
=\rho_\infty-\frac{2\beta_2}{1-\beta_2}=1.
```

因为 `rho_1 <= 5`，PyTorch 采用未整流分支：

```math
\Delta\theta_g=-\eta\widehat m_1=-\eta G_g.
```

`v_1`、`eps` 与 RAdam rectification 都不进入首步 parameter update。七次 AMP overflow/skip 也不会把 optimizer step 从 0 推到 7；第一个 non-skipped optimizer step 仍然是 `t=1`。GradScaler state 发生变化，但 RAdam moment state 仍为 fresh。

## Relationship between the two quotients

PR #38 的 raw mean-gradient coefficient 定义为

```math
a_g^\star
=\frac{\langle\mu_g,\mu_1\rangle}
{\|\mu_1\|_2^2}.
```

PR #42 把 candidate update 缩放到 reference update，定义

```math
c_0^\star
=\frac{\langle\Delta\theta_g,\Delta\theta_1\rangle}
{\|\Delta\theta_g\|_2^2}.
```

如果两者使用完全相同的 gradient vector aggregation，并处于上述 fresh first-step 条件，则

```math
c_0^\star
=\frac{\langle\mu_g,\mu_1\rangle}
{\|\mu_g\|_2^2}
=\frac{\cos^2(\mu_g,\mu_1)}{a_g^\star}.
```

只有在精确共线时才有 `c0_star = 1/a_g_star`。不能把不同 checkpoint、不同 minibatch aggregation 或不同 precision 下测得的 `a_g_star` 直接取倒数作为 Arm C learning-rate multiplier。

同一条件下，两个 projection residual 都等于两向量夹角的正弦：

```math
\frac{\|\mu_g-a_g^\star\mu_1\|}{\|\mu_g\|}
=
\frac{\|c_0^\star\mu_g-\mu_1\|}{\|\mu_1\|}.
```

所以 fresh first-step virtual update 可以独立复核实现，但不会增加 optimizer-state mechanism 信息。

## When RAdam becomes state-aware

对 `beta2=0.999`，前六个 `rho_t` 约为：

| Successful RAdam step `t` | `rho_t` | Branch |
|---:|---:|---|
| 1 | 1.0000 | unrectified |
| 2 | 1.9995 | unrectified; first moment 已含 history |
| 3 | 2.9987 | unrectified; first moment 已含 history |
| 4 | 3.9975 | unrectified; first moment 已含 history |
| 5 | 4.9960 | unrectified; first moment 已含 history |
| 6 | 5.9942 | rectified/adaptive branch begins |

因此：

- `t=1` 不含 optimizer-history effect；
- `t=2..5` 已含 first-moment history，但仍未使用 second-moment adaptive denominator；
- `t>=6` 才进入完整 RAdam rectified/adaptive branch。

这不是说 `t>=6` 自动产生非零 residual，而是说只有此后 diagnostic 才有资格检验完整 RAdam state 是否吸收、扭曲或放大 gap-induced gradient relation。

## Correct state-conditioned operator

令共同状态为

```math
s_K=(\theta_K,m_K,v_K,K,\text{GradScaler}_K).
```

在同一 state clone 和完全 paired minibatch `B` 上定义不提交的 virtual update：

```math
U_{s_K}(g;B)
=\operatorname{RAdamStep}(s_K,G_g(B),\eta)-\theta_K.
```

然后计算

```math
c_K^\star(g)
=\frac{\langle U_{s_K}(g;B),U_{s_K}(1;B)\rangle}
{\|U_{s_K}(g;B)\|_2^2},
\qquad
R_K(g)
=\frac{\|c_K^\star(g)U_{s_K}(g;B)-U_{s_K}(1;B)\|_2}
{\|U_{s_K}(1;B)\|_2}.
```

`s_K` 应来自同一 canonical trajectory 的真实 training-state dump；两个 gap arms 只改变当前 loss 的 `r_g(t)`，不得分别使用已经分化的 optimizer histories。优先状态：

1. 第 6 个或更晚的 successful RAdam step，用于最早完整 adaptive-branch sanity；
2. 预注册的 early checkpoint；
3. 32、64、128、256 kimg checkpoints，用于估计 state drift。

## Revised acceptance gate

### Fresh-state sanity gate

必须通过，但只计为 implementation validation：

1. real repository ECT loss；
2. images、labels、`t`、`eps`、dropout RNG、microbatch accumulation 完全 paired；
3. `scaler.unscale_(optimizer)` 后读取 gradients；
4. 两个 arms 的 parameter、optimizer 与 GradScaler pre-state hash 相同；
5. virtual step 后原始 state hash 不变；
6. 数值上验证 `Delta_theta_g = -eta G_g`，并验证 update residual 与 raw-gradient projection residual 相等。

### Optimizer-state mechanism gate

GFCT mechanism 与 novelty 所需的真正 gate：

1. 使用 `successful_optimizer_steps >= 6` 的同一 nonzero RAdam state；
2. clone 后只改变当前 counterfactual gap；
3. 报告 whole-model 与 layerwise `c_K_star`、cosine、norm、residual；
4. 报告 raw-gradient 与 update-space residual 的差值，明确 optimizer 增加或消除了什么；
5. 至少跨三个预注册 states 重复，并给 paired uncertainty；
6. 任何 state/pairing/AMP mismatch 均非零退出。

## Consequence for PR #42

- Formal training 应继续 **BLOCKED**，但阻断理由应从“fresh-state RAdam update 尚未测量”改为“nontrivial-state RAdam quotient 尚未测量”。
- 当前 Arm C 若使用 fresh first-step `c0_star`，只能叫 **fresh raw-gradient/update-linearized matched control**；不得称为 optimizer-state-aware matched control。
- PR #42 已计划的 `c_K_star` checkpoint audit 才是有潜力承载 optimizer-state evidence 的部分，但必须保证两个 counterfactual gaps 共享同一 `theta_K,m_K,v_K,step_K,GradScaler_K`。
- 如果项目仍需在 formal launch 前确定单一 Arm C multiplier，应预先声明其只是一个 state-local control scalar；不能声称整个 trajectory matched。

## Final audit verdict

```text
raw-gradient diagnostic        GO
fresh first-step RAdam check   GO as sanity only
optimizer-state-aware quotient PENDING
Idea 5 loss variance           excluded from mechanism evidence
GFCT novelty                   CONDITIONAL GO
```
