# RAdam theory audit for gap equivalence（2026-08-07）

## Decision

**当前 PR #42 所定义的 fresh-state one-step RAdam diagnostic 不能被称为 optimizer-state-aware evidence。**

本审计已按 main commit `526cd52`（PR #43）更新。PR #43 已把 RAdam
gap-equivalence theory 正式拆成 update scale `s_k^star`、candidate LR multiplier
`c_k^star`、support-aware coordinate history gauge `h_{k,i}`，以及精确恒等式
`H_k=R_opt(k)`。因此 Role D 的合格输出不再只是一个 `c_K_star`：必须同时报告
标量部分、逐坐标/逐层 dispersion 与 off-support energy，并把 fresh-state control
和 nontrivial-state mechanism evidence 明确分开。

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
| PR #43 history-gauge theorem | **GO as theory asset** | 给出 null、support-aware iff 与 exact residual identity；不是 real-ECT mechanism evidence |

## Audited implementation facts

- [`ct_train.py`](../ct_train.py) 构造 `torch.optim.RAdam`，显式传入 `lr`、`betas=[0.9,0.999]`、`eps=1e-8`，没有覆盖 `weight_decay`，因此使用 PyTorch 默认值 `0`。
- [`training/ct_training_loop.py`](../training/ct_training_loop.py) 在 AMP 路径中先执行 `scaler.unscale_(optimizer)`，再调用 `scaler.step(optimizer)`；overflow 时 optimizer step 被跳过。
- 已有 Role D manifest 记录远端诊断环境为 PyTorch `2.6.0+cu124`；本地复核环境为 PyTorch `2.13.0`。正式实现仍应把远端 `torch.optim.RAdam` source hash 或版本固定进 receipt。
- [PyTorch RAdam documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.radam.RAdam_class.html) 给出的算法在 `rho_t <= 5` 时使用未整流 update，在 `rho_t > 5` 后才引入 second-moment adaptive denominator 与 rectification factor。
- [RAdam paper](https://arxiv.org/abs/1908.03265) 的核心对象是 early-stage adaptive learning-rate variance 与 rectification；这正是不能用 raw-gradient equivalence 自动代替的后续状态效应。
- [`theory/radam_gap_equivalence.md`](../theory/radam_gap_equivalence.md) 已在 main
  固定两个不可混用的投影方向，并给出 support-aware theorem；对应 numeric tests
  位于 [`theory/test_radam_history_gauge.py`](../theory/test_radam_history_gauge.py)。

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

## PR #43-aligned state-conditioned operator

令 canonical training trajectory 在 checkpoint `K` 的共同 pre-state 为

```math
z_K=(\theta_K,m_K,v_K,n_K,\text{GradScaler}_K,\text{EMA}_K).
```

在同一 state clone 和完全 paired minibatch `B` 上定义不提交的 virtual update：

```math
U_{g,K}
=\operatorname{RAdamStep}(z_K,G_g(B),\eta)-\theta_K,
\qquad
U_{1,K}
=\operatorname{RAdamStep}(z_K,G_1(B),\eta)-\theta_K.
```

两个 clone 的 **pre-state 必须相同**。若从已经按不同 gap 分化的两条 trajectories
各取 optimizer state，测到的是 trajectory difference，不是 current-gap
counterfactual，不能用于 Role D 因果诊断。

按 #43 分别计算 update scale 与 Arm C learning-rate multiplier：

```math
s_K^\star
=\frac{\langle U_{g,K},U_{1,K}\rangle}{\|U_{1,K}\|_2^2},
\qquad
c_K^\star
=\frac{\langle U_{g,K},U_{1,K}\rangle}{\|U_{g,K}\|_2^2}.
```

`s_K^star` 拟合 `U_g ~= s_K^star U_1`；`c_K^star` 拟合
`c_K^star U_g ~= U_1`。只有精确共线时才有 `c_K^star=1/s_K^star`，
协议、JSON schema 与表格列名不得把二者都写成泛称 `scale`。

令有效 reference support 为

```math
S_K=\{i:U_{1,K,i}\ne0\},
\qquad
h_{K,i}=U_{g,K,i}/U_{1,K,i}\quad(i\in S_K).
```

理论 reference-normalized residual 与 history-gauge dispersion 为

```math
R_{\mathrm{opt}}(K)
=\frac{\|U_{g,K}-s_K^\star U_{1,K}\|_2}{\|U_{1,K}\|_2},
```

```math
H_K^2
=\frac{
\sum_{i\in S_K}U_{1,K,i}^2(h_{K,i}-s_K^\star)^2
+\sum_{i\notin S_K}U_{g,K,i}^2
}{\sum_{i\in S_K}U_{1,K,i}^2}.
```

若 `h_{K,i}` 直接由实际 update ratio 定义，并使用精确 support，
`H_K=R_opt(K)` 是 support-aware 代数恒等式；该 equality 是 implementation
consistency check，不是第二份独立 mechanism evidence。PR #43 中由
`mhat/sqrt(vhat)` 写出的 moment-formula gauge 才需要 `eps=0`、无 weight decay
与 rectified branch 等理想化条件。正式输出必须分别命名
`h_update` 与 `h_moment_ideal`，不得把两者的差异塞进 projection residual。

`z_K` 应来自同一 canonical trajectory 的真实 training-state dump；两个 gap arms
只改变当前 loss 的 `r_g(t)`。优先状态：

1. 第 6 个或更晚的 successful RAdam step，用于最早完整 adaptive-branch sanity；
2. 预注册的 early checkpoint；
3. 32、64、128、256 kimg checkpoints，用于估计 state drift。

真实浮点实现中，exact identity 仍应以字面 nonzero support 和稳定的加权形式计算，
避免显式放大接近零坐标的 ratio。用于画图或 quantile summary 的 practical support
可以使用预注册的相对/绝对 threshold，但必须标记为 `thresholded`、报告被排除的
reference update energy，并至少对一个更严和一个更松 threshold 做敏感性检查。
thresholded `H_K` 不再被默认视为与 full-vector `R_opt(K)` 精确相等；两者差值必须报告。

## Revised acceptance gate

### Fresh-state sanity gate

必须通过，但只计为 implementation validation：

1. real repository ECT loss；
2. images、labels、`t`、`eps`、dropout RNG、microbatch accumulation 完全 paired；
3. `scaler.unscale_(optimizer)` 后读取 gradients；
4. 两个 arms 的 parameter、optimizer 与 GradScaler pre-state hash 相同；
5. virtual step 后原始 state hash 不变；
6. 数值上验证 `Delta_theta_g = -eta G_g`，并验证 update residual 与 raw-gradient projection residual 相等。

该 gate 通过后只生成 `fresh_state_sanity=PASS`。它不得生成
`optimizer_state_mechanism=PASS`。若三臂实验保留 Arm C，fresh `c0_star` 可以继续
作为解析 learning rate 的 configuration prerequisite，但不能单独解除 mechanism
blocker 或授权 formal launch。Arm C 必须重命名为
`fresh-linearized matched control`，而非 optimizer-state-matched control。

### Optimizer-state mechanism gate

GFCT mechanism 与 novelty 所需的真正 gate：

1. 使用 `successful_optimizer_steps >= 6` 的同一 nonzero RAdam state；
2. clone 后只改变当前 counterfactual gap；
3. 报告 whole-model 与 layerwise `c_K_star`、cosine、norm、residual；
4. 同时报告 `s_K_star`、actual-update `h_update`、idealized-moment
   `h_moment_ideal`、layer aggregation、off-support energy、`H_K` 与 `R_opt(K)`；
   exact-support identity 与 thresholded summary 必须分列；
5. 报告 raw-gradient 与 update-space residual 的差值，明确 optimizer 增加或消除了什么；
6. 至少跨三个预注册 states 重复，并给 paired uncertainty；
7. 对 support threshold 做预注册敏感性检查；
8. 任何 state/pairing/AMP mismatch 均非零退出。

### Required receipt per audited state

每个 `K` 至少固化以下机器可读字段：

- source commit、checkpoint hash、PyTorch/CUDA/RAdam implementation identity；
- `successful_optimizer_steps`、RAdam branch、AMP scale 与 skip history；
- pre-state parameter/optimizer/GradScaler hashes，以及 virtual step 后的 non-commit hash；
- paired minibatch/RNG identifiers 与 microbatch aggregation count；
- raw `a_K_star`/raw residual；
- update `s_K_star`、`c_K_star`、cosine、norms、`R_opt(K)`；
- layer/coordinate `h_update` 与 `h_moment_ideal` summary、support threshold、
  excluded reference energy、off-support candidate energy、exact/thresholded `H_K`
  和各自的 `abs(H_K-R_opt(K))`。

## Consequence for PR #42

- Formal training 应继续 **BLOCKED**，但阻断理由应从“fresh-state RAdam update 尚未测量”改为“nontrivial-state RAdam quotient/history gauge 尚未测量”。
- 当前 Arm C 若使用 fresh first-step `c0_star`，只能叫 **fresh raw-gradient/update-linearized matched control**；不得称为 optimizer-state-aware matched control。
- PR #42 已计划的 `c_K_star` checkpoint audit 才是有潜力承载 optimizer-state evidence 的部分，但必须保证两个 counterfactual gaps 共享同一 `theta_K,m_K,v_K,step_K,GradScaler_K`，并扩展为 `s_K_star/c_K_star/h_K/H_K/R_opt` longitudinal receipt。
- 如果项目仍需在 formal launch 前确定单一 Arm C multiplier，应预先声明其只是一个 state-local control scalar；不能声称整个 trajectory matched。
- `COLLABORATOR_AUDIT=PASS` 这类人工环境变量不能充当 formal gate。launcher
  必须读取并校验一份不可空的审计 receipt（含 schema、state/source hashes 与 verdict）。
- 配置中冻结的 source commit 必须由 launcher 与 `git rev-parse HEAD` 强制比对；仅把
  当前 HEAD 写入 provenance、但不拒绝错误 source，不能满足 training-source enforcement。

## Final audit verdict

```text
raw-gradient diagnostic        GO
fresh first-step RAdam check   GO as sanity only
PR #43 history-gauge theory    GO as theory asset
optimizer-state-aware quotient PENDING
Idea 5 loss variance           excluded from mechanism evidence
GFCT novelty                   CONDITIONAL GO
```
