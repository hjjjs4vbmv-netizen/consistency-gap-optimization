# 状态:Leader 质疑之前(Role C 原结论)

日期:2026-08-10(leader 质疑前)。分支 `role-c/g13-vs-g10-fid-0809`。
本文档记录 leader 提出质疑**之前**的完整工作与结论,与质疑后的回应分开。

---

## 1. 原 GFCT 战略轨迹

```
机制+方法(40-50%) → 被 g=1.3 赢 FID 证伪
→ 诚实诊断(20-30%) → 被"良性"非结果质疑
→ 诊断信号转向(30-35%) → 被 P1 附庸性检验证伪(偏相关≈0)
→ 原结论:workshop 级诚实负结果
```

## 2. 质疑前的实测证据(6 实验)

| 环节 | 实验 | 结论 |
|---|---|---|
| 存在 | moment-memory | 标量 null 被证伪(h 1.001 vs 0.837, Corr≈0) |
| 有结构 | E5 | 深度趋势(ρ≈0.4)+ 剂量响应,非噪声 |
| 无害 | E6 + 干净 FID | 与 FID 反相关 r=−0.99;g=1.3 赢 −34%~−45% |
| 状态条件 | E7 | R_grad 随 n_K 增长(0.031→0.091) |
| 稳健 | E11 | 跨 3 seed(1024k 5/6)+ 支撑阈值 |
| 普遍 | E3 | 梯度残差跨 optimizer 复现(AdamW 0.095 vs RAdam 0.091) |

**关键数字:** a_K*=0.761、R_grad=5.85%、R_opt=8.57%、H_K=R_opt 精确(1.7e-18)、
h_pred=1.001 vs h_actual=0.837。

## 3. 质疑前的 P1 检验(原结论的最后一环)

P1(附庸性检验):残差附庸于标量 gap——偏相关(残差, FID | \|gap−1\|)= +0.168 ≈ 0。
诊断信号论题不成立。

## 4. 质疑前的原结论

> **非标量梯度残差存在、有结构、状态条件、普遍、良性——但附庸于标量 gap。**
> 它是少步训练的真实特征,但不是质量诊断信号,也不有害。
> **论文定位:workshop 级诚实负结果,不投 ICLR 主会。**

## 5. 质疑前的交付物

- `docs/ICLR2027_DIAGNOSIS_PAPER.{md,tex,pdf}` — 诊断论文
- `docs/ICLR2027_STRATEGY.{tex,pdf}` — 战略文档
- `docs/ICLR2027_PLAN_REVIEW.md` — 对抗性审查
- 实验结果:`analysis/g13_vs_g10_fid_result.md`、`e5/e6/e7/e11/e3_*_result.md`、
  `p1_non_epiphenomenality_result.md`
- `HANDOFF_20260810.tex` + `ICLR2027_REPORT_SLIDES.tex`

---

## 6. 与质疑后的区别(一句话)

质疑前:基于 6 诊断实验 + P1,结论是 **workshop 级诚实负结果**。
质疑后:按 leader 提议跑了 **Arm C + 双级未来预测**,得到 **四重证据 + No-Go 定论**。
详见 `docs/STATUS_AFTER_LEADER_CRITIQUE.md`。
