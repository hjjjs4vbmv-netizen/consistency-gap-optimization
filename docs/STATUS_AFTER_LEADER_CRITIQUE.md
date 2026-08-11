# 状态:Leader 质疑之后(Role C 的回应与新结论)

日期:2026-08-11(leader 质疑后)。分支 `role-c/g13-vs-g10-fid-0809`。
本文档记录 leader 提出质疑**之后**的完整过程与最终结论,与质疑前分开。
质疑前状态见 `docs/STATUS_BEFORE_LEADER_CRITIQUE.md`。

---

## 1. Leader 的核心质疑(2026-08-11)

Leader 对我的"workshop only"结论提出质疑,核心论点:

1. **P1 不是决定性证伪**(n=4/6、VIF≈8.3、共线,统计不严谨;3 seeds 是
   descriptive 不是 inferential)。
2. **P1 测错假设**——只测静态 `R_t → FID_t | g`,没测 trajectory 的
   `R_t → ΔFID_{t→T}`。
3. **Arm C 才能证伪 scalar-rescaling**——"g=1.3 赢 FID"不能证伪机制,真正
   要看 scale-matched control 能否消除改善。
4. **optimizer-memory 机制没失败,反而被增强**——raw gradient 近标量等价
   (cos≈0.999996)但 optimizer update 非标量等价(h=0.837 vs 1.001),是真正
   值得研究的现象。
5. **"负结果=Workshop"不成立**——机制型/负结果型工作同样可够主会。
6. **结论**:"ICLR-not-ready,不是 ICLR-dead";我"workshop only"的措辞过强。

## 2. 我按 leader 提议跑的两个决定性实验

### 2.1 Arm C(scale-matched control)——leader 的判据

| | NFE=1 FID | NFE=2 FID |
|---|---:|---:|
| g=1.0 (A) | 315.8 | 87.7 |
| g=1.3 lr_fixed (B) | 208.1 | 56.6 |
| **g=1.3 lr_matched (C)** | **298.3** | **87.7** |

**按 leader 自己的判据:** NFE2 FID_C = FID_A(完全相等,改善 100% 消除);
NFE1 消除 84%。**质量改善大部分是平凡 LR 重缩放。**

### 2.2 未来预测(leader 说的"真正缺失的一环")

**梯度级**(重训 4 seeds 存 snapshot,早期 R_grad):
- 早期 R_grad 跨 seed 近恒定(0.0085–0.0096),未来 FID 改善 6–63 FID
- **早期 R_grad 不预测未来 FID(阴性)**

**优化器级**(重训存早期 training-state,早期 R_opt):
- 早期 R_opt 跨 seed 近恒定(0.0183–0.0207,std 0.0011)
- R_opt 确实对 gap 敏感(diff 1.3−1.0 全为正,机制存在)
- **早期 R_opt 不预测未来 FID(阴性)**

## 3. 四重实测证据(质疑后闭环)

| 实验 | 结论 |
|---|---|
| **Arm C** | g=1.3 改善 84-100% 是平凡 LR 重缩放 |
| **P1** | 残差附庸于标量 gap(偏相关≈0) |
| **梯度级未来预测** | 早期 R_grad 不预测未来 FID |
| **优化器级未来预测** | 早期 R_opt 不预测未来 FID |

## 4. 质疑后的最终定论

**No-Go for ICLR 主会机制论题。**

Leader 论题(`Equivalence → Symmetry Breaking → Accumulation → Performance`)
**前 3 环成立**(机制真实:raw gradient 近标量等价、optimizer update 非标量
等价、h_actual≠h_pred),但**最后一环 Performance 在梯度级与优化器级均不成立**
(质量改善大部分平凡,早期残差不预测未来 FID)。

**诚实可发表资产:workshop 级结构刻画**——非标量梯度/优化器内容存在、有结构、
普遍、良性、不预测未来质量。

## 5. 质疑后新增交付物

- `docs/RESPONSE_TO_LEADER.md` — 逐点针对性回应
- `analysis/arm_c_scale_matched_control_result.md` — Arm C 结果
- `analysis/future_prediction_result.md` — 梯度级未来预测(阴性)
- `analysis/future_prediction_opt_result.md` — 优化器级未来预测(阴性)
- `analysis/future_prediction_analysis.py`、`future_prediction_opt_analysis.py`

## 6. 与质疑前的区别(一句话)

质疑前:基于 6 诊断实验 + P1,结论是 **workshop 级诚实负结果**(~20-30%)。
质疑后:按 leader 提议跑了 **Arm C + 双级未来预测**,把 leader 的 optimizer-memory
→ Performance 论题测到最后一环,四重证据闭环,定论 **No-Go for ICLR 主会**。
