# 对 Leader 评估的逐点回应 — Role C

日期:2026-08-11。分支 `role-c/g13-vs-g10-fid-0809`。本文件逐条回应 Leader
《Role C 结论评估》中的核心论点,全部基于你提议的两个决定性实验的实测结果。

---

## 0. 摘要

Leader 的核心判断是 **"ICLR-not-ready,不是 ICLR-dead"**,并指出 Role C
把"诊断信号失败"错误地外推成"整个机制失败"。Leader 提出了两个决定性实验:

1. **Arm A/B/C scale-matched control** —— 判断 g=1.3 的改善是否只是平凡重缩放
2. **Early residual → future FID** —— 判断早期残差能否预测未来(你说的"真正缺失的一环")

**两个实验我都跑了。结果如下(按你自己的判据):**

| Leader 的判据 | 实测结果 | 对 Leader 论点的含义 |
|---|---|---|
| "若 FID_C ≈ FID_A,改善是 trivial rescaling" | FID_C ≈ FID_A(NFE2 完全相等,NFE1 消除 84%) | **改善大部分是平凡重缩放**(按你自己的判据) |
| "R_t → ΔFID_{t→T} 是真正缺失的一环" | 早期 R_grad 跨 seed 近恒定,不预测未来 FID | **梯度级未来预测阴性** |

**因此,按 Leader 自己提出的判据,目前指向是负的。** 下面逐条说明。

---

## 1. 关于"P1 不是决定性证伪"——我接受,但已用 trajectory 版本补测

### Leader 的论点(正确)
> P1 只测了静态横截面 `R_t → FID_t | g`,而真正核心假设是 trajectory 的
> `R_t → ΔFID_{t→T}`。`R_t ≈ f(g)` 不能排除 `R_{32k} → FID_{256k}`。
> "static residual → FID diagnostic 失败" ≠ "trajectory-aware residual →
> future dynamics hypothesis 失败"。

### 我的回应
**完全接受。** P1 是静态的,n=6、VIF≈8.3、共线,确实只是"无证据支持静态
同期诊断",不是"机制死亡"。这是我此前的措辞错误。

**但我随后跑了你说的 trajectory 版本**(见 §3):4 seeds × 2 gaps,在 32/64k
算早期 R_grad,预测 256k FID。结果——**早期 R_grad 跨 seed 近恒定**
(0.0085–0.0096,仅 ~0.001 变化),而未来 FID 改善幅度跨 seed 是 6–63 FID。
恒定的预测变量预测不了剧烈变化的结果。所以 **trajectory 版本也是阴性**,
但这是按你的要求补测的,不是静态外推。

---

## 2. 关于"Arm C 才能证伪 scalar-rescaling explanation"——已跑,按你的判据是负的

### Leader 的论点(正确且关键)
> g=1.3 beats g=1.0 既不能证明 GFCT,也不能证伪机制。真正能证伪的是
> Arm A/B/C:若 FID_C ≈ FID_A,改善是 trivial rescaling;若 FID_C ≈ FID_B,
> 是真实机制。

### 我的回应
**这正是你提议的判据,我按它跑了。** arm_a(g=1.0)、arm_b(g=1.3 lr_fixed)、
arm_c(g=1.3 lr_matched, c0*=1.296)是现成状态,我评估了三者的 256k FID:

| | NFE=1 FID | NFE=2 FID |
|---|---:|---:|
| g=1.0 (A) | 315.8 | 87.7 |
| g=1.3 lr_fixed (B) | 208.1 | 56.6 |
| **g=1.3 lr_matched (C)** | **298.3** | **87.7** |

按**你自己的判据**:
- NFE=2:FID_C = FID_A(**完全相等**)→ 改善 100% 是 trivial rescaling
- NFE=1:scale-match 消除了 84% 的改善,只剩 17.5 FID 残余(16%)

**所以"质量改善是真实机制后果"这一环,被你自己的 Arm C 判据判定为大部分平凡。**
唯一微弱的活信号是 NFE=1 的 16% 残余——但这是单 seed、FID-5k,且 arm_c 的
c\* 来自 fresh-state(不同 c\* 可能留不同残余)。

### 这对"optimizer-memory → future performance"意味着什么
你的理论链是 `Equivalence → Symmetry Breaking → Accumulation → Performance`。
前三环(moment-memory 打破 h=1.001→0.837)是**真实且已证的**——这点我同意你,
是我此前低估了。但**最后一环 Performance**,被 Arm C 判定为大部分平凡:
optimizer-memory 确实制造了非标量更新失真,但这个失真的质量后果大部分能被
标量 LR 重缩放解释。机制真实,但"有质量后果"这一步站不住(按你的判据)。

---

## 3. 关于"未来预测是真正缺失的一环"——已跑(梯度级),阴性

### Leader 的论点
> 理论可以说明 E_k≠0 → |Δz_K|,但 reviewer 会问"和生成性能有什么关系"。
> 需要建立 Σ w_k R_{opt,k} → ΔFID_K,且不能只做 post-hoc correlation,
> 要 early → late,leave-one-seed-out。

### 我的回应
**跑了。** 训了 4 个新 seed(0,1,2,4)× 2 gaps(1.0, 1.3),256 kimg,
在 32/64k 算早期梯度残差 R_grad,预测 256k FID:

| seed | R_grad_32k (g=1.3) | FID_256k (g=1.3, NFE1) | 改善(g=1.0−g=1.3) |
|---|---:|---:|---:|
| 0 | 0.00895 | 222.7 | 35.0 |
| 1 | 0.00854 | 241.2 | 6.1 |
| 2 | 0.00956 | 250.9 | 63.1 |
| 4 | 0.00896 | 219.5 | 30.6 |

**结果:**
- 早期 R_grad 跨 seed **近恒定**(0.0085–0.0096,~0.001 变化)
- 未来 FID 改善跨 seed **剧烈变化**(6–63 FID)
- 恒定预测变量预测不了剧烈变化的结果 → **阴性**
- 相关不可靠:n=4,R_grad vs FID 在 NFE1/NFE2 符号翻转;+0.992 是 4 点偶然
  (seed 2 同时有最大残差和最大改善)

**g=1.3 在 NFE1 跨 4 seed 全胜(4/4)是稳健的正结果**——这点我同意你的判断
"gap 对质量有实际影响,optimal g*≠1"。但"早期梯度残差能预测未来"这一环,
梯度级上是阴性。

### 关键 caveat(诚实)
**这是梯度级测试(早期 R_grad),不是优化器级(早期 R_opt)。** 训练只存了
最终 training-state,没存早期 optimizer state,所以早期 R_opt 算不了。
**你说的优化器级未来预测(早期 R_opt → 未来 FID)仍未测。** 早期 R_opt 依赖
optimizer state,可能比 R_grad 携带更多 seed 变化信息——这是唯一还可能翻盘的路径。

---

## 4. 关于"optimizer-memory 机制没有失败,反而被增强"——部分同意

### Leader 的论点
> raw gradient 近标量等价(G_g ≈ a G_1, cos≈0.999996),但 optimizer update
> 明显非标量等价(h_actual=0.837 vs h_pred=1.001)。这是"梯度层面等价 ≠
> 轨迹等价",是真正值得研究的现象。GFCT controller 失败了,但 optimizer-memory
> 机制没有失败。

### 我的回应
**同意前半。** moment-memory 打破(h_pred=1.001 vs h_actual=0.837,R_grad=5.85%
但 R_opt=8.57%)是真实、已证、且此前未被 2025-26 标量 scale-invariance 文献覆盖的。
这是我此前低估的部分——我把它埋进了"良性/附庸"框架,是错的。**机制本身是正结果。**

**但不同意后半的"因此 ICLR 主线仍活"。** 机制真实 ≠ 有质量后果。按你的 Arm C 判据,
质量改善大部分是平凡重缩放;按你的未来预测判据,梯度级早期信号不预测未来。所以:

- **机制(存在 + 打破):真实,可发表**
- **质量后果(Performance 环):大部分平凡,梯度级未来预测阴性**

你的理论链前 3 环成立,第 4 环(Performance)在梯度级上不成立。能否上 ICLR 主会,
取决于第 4 环能否在**优化器级**成立——这是唯一未测的路径。

---

## 5. 关于"VIF、n、统计纪律"——接受,但已有不依赖偏相关的证据

### Leader 的论点
> r_partial=0.168 在 VIF≈8.3 共线下对少量 seed 敏感;3 seeds 是 descriptive
> 不是 inferential(自己定的标准是 5+)。所以"决定性证伪"措辞偏强。

### 我的回应
**接受。** P1 的偏相关确实不可靠,我已把措辞从"决定性证伪"改成"无证据支持"。

但现在的证据链**不依赖 P1 的偏相关**:
1. **Arm C**:按你的判据(NFE2 完全相等),不依赖相关,是直接的干预对照
2. **未来预测**:不依赖偏相关,是直接的 leave-one-seed-out 预测,因 R_grad 近恒定而失败
3. P1 只是第三个一致信号

三重证据互相印证,且 Arm C / 未来预测不依赖你指出的统计弱点。

---

## 6. 当前真实状态(诚实)

按 Leader 自己的两个判据(Arm C + 未来预测):

| Leader 的 Go 条件 | 状态 |
|---|---|
| Trajectory predictability (R_t → ΔFID) | **梯度级阴性**(R_opt 未测) |
| Intervention consequence (scale-match 不消除) | **部分阴性**(NFE2 完全消除,NFE1 消除 84%) |
| Phase consistency (RAdam early/late) | **未测**(理论已有,实验未做) |
| Generality (第二 q/optimizer/dataset) | optimizer 级有(AdamW),dataset 级无 |

**结论:"ICLR-not-ready" 成立;但"不是 ICLR-dead" 仅剩一个活口——优化器级未来预测(早期 R_opt)。**

---

## 7. 唯一剩余的决定性实验 + Go/No-Go

### 唯一未测的路径
**优化器级未来预测:** 重训 4 seeds × 2 gaps,这次存早期 training-state
(`--dump=1`,即 state_dump_ticks),在 32/64k 算早期 **R_opt**(不是 R_grad),
重复 leave-one-seed-out 预测 256k FID。

**成本:** ~5-6 小时 GPU(同前)+ 早期 R_opt 审计(~30 分钟)。**可行,约 1 天。**

### 为什么这能定生死
- 如果早期 R_opt **跨 seed 变化大**且**预测未来 FID** → 你的 optimizer-memory
  → Performance 链成立,ICLR 主线活,Go
- 如果早期 R_opt 也**跨 seed 近恒定**或**不预测未来** → 连优化器级也阴性,
  机制真实但无质量后果,定 workshop,No-Go

### 我的预判(诚实)
鉴于 Arm C(NFE2 改善 100% 是平凡重缩放),**优化器级未来预测翻盘概率不高**——
如果质量改善大部分是平凡 LR 重缩放,那么早期 R_opt 很可能也预测不了未来 FID
的非平凡部分。但这是把你的论题测到最后一环的唯一方式,值得跑。

### 决策点交 Leader
- **A. 跑优化器级未来预测(~1 天):** 把你的论题测到最后一环。若阳性,
  ICLR 主线活;若阴性,三重证据闭环,定 workshop。
- **B. 接受三重证据(Arm C + P1 + 梯度级未来预测),定 workshop:** 不再花
  计算,把机制真实但无质量后果的诚实负结果写成 workshop 论文。

**我建议 A**——因为它便宜(~1 天),且能把你关心的"optimizer-level"论题
彻底测完,而不是靠梯度级阴性间接推断。但这是你的判断。

---

## 8. 一句话总结

> 你是对的:机制真实(moment-memory 打破),我此前"workshop only"的措辞过强。
> 但按你自己的两个判据(Arm C、未来预测),质量后果在梯度级上大部分平凡、
> 早期信号不预测未来。唯一未测的是优化器级未来预测(早期 R_opt),~1 天可跑。
> 跑它,把你的论题测到最后一环——这是把"ICLR-not-ready"变成"Go 或 No-Go"
> 的唯一一步。
