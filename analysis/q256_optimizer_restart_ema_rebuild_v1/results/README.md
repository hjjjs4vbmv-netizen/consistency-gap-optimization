# M1 结果：主比较 INCONCLUSIVE（7/16 完整 R 配对）

2026-09-07 从 ECT 完整归档读取。原始 320 槽 = 260 PASS + 60 NO_ENDPOINT；
另有 60 PASS 的失败后 FP32 敏感性评估，严格分开，不填回原始失败。
原始 64 分支：K_A 14/16、K_B 15/16、R_A 8/16、R_B 15/16 成功。
12 条独立 FP32 修复全部完成，不等于原始 64/64 成功。

## 核心结果

负数表示 B 历史终点 FID 更低。E_512 先在每个 training seed 内平均三个
log-FID 配对差，再在 seed 层取均值及双侧 Student-t 95% CI；不是 FID150k，
不是先平均 FID 再取 log，generation blocks 不增加训练 n。

| 比较 | 完整配对 / 16 | 平均 log-FID 差 | 95% CI | 地位 |
|---|---:|---:|---|---|
| R_B/E_512 − R_A/E_512，三 blocks | 7 | −0.916614 | [−2.038142, 0.204914] | 唯一主要比较：INCONCLUSIVE |
| K_B/E_512 − K_A/E_512，三 blocks | 14 | −0.091467 | [−0.141187, −0.041746] | 描述性，名义 CI |
| (R_B−R_A) − (K_B−K_A)，共同四臂集合 | 6 | −1.005402 | [−2.321500, 0.310696] | 关键次级，估计而非第二确认性判定 |

主集合 seeds 55、56、59、60、61、62、65；四臂集合去掉 seed65（K_A 原失败）。
主比较 SD=1.212665，5 个 seed 为负、2 个为正，方向计数仅作描述。
FID 几何比为 0.399871，区间 [0.130271, 1.227420]；
点估计低约 60.0%，但区间仍容许 B 高约 22.7%，不能宣布 B 优势成立。
该区间只在协议规定的成功条件 seed process 和 t 模型假设下解释，不覆盖失败 seed。

共同六对中的 R 差为 −1.086655，K 差为 −0.081253。interaction 必须使用这同六对，
不能把 n=7 的 R 均值减去 n=14 的 K 均值。其区间跨零，不能说 restart 增强、
削弱或擦除了历史优势，也不能把未显著解释为无影响。


## 全 16 seed 原始成败表

PASS 表示训练完成并有五个有效评估槽；FAIL 表示原科学数值失败、五槽 NO_ENDPOINT。
不含 FP32 修复，不按修复结果更改。

| seed | K_A | K_B | R_A | R_B |
|---|---|---|---|---|
| 50 | PASS | PASS | FAIL | PASS |
| 51 | PASS | PASS | FAIL | PASS |
| 52 | PASS | PASS | FAIL | PASS |
| 53 | PASS | PASS | FAIL | PASS |
| 54 | PASS | PASS | FAIL | PASS |
| 55 | PASS | PASS | PASS | PASS |
| 56 | PASS | PASS | PASS | PASS |
| 57 | PASS | PASS | FAIL | PASS |
| 58 | FAIL | FAIL | PASS | FAIL |
| 59 | PASS | PASS | PASS | PASS |
| 60 | PASS | PASS | PASS | PASS |
| 61 | PASS | PASS | PASS | PASS |
| 62 | PASS | PASS | PASS | PASS |
| 63 | PASS | PASS | FAIL | PASS |
| 64 | PASS | PASS | FAIL | PASS |
| 65 | FAIL | PASS | PASS | PASS |

## 原始主集合的绝对质量

下表只用于展示质量量级，数字是 E_512 三个 FID50k 的算术均值。
正式主统计量仍按上文逐 block 的 log 差计算。缺失不插补。

| seed | K_A FID | K_B FID | R_A FID | R_B FID |
|---|---:|---:|---:|---:|
| 55 | 10.221 | 9.495 | 103.385 | 10.504 |
| 56 | 10.645 | 8.533 | 12.050 | 9.939 |
| 59 | 7.276 | 7.355 | 62.257 | 10.898 |
| 60 | 9.526 | 8.902 | 108.677 | 138.288 |
| 61 | 8.032 | 7.715 | 9.952 | 9.731 |
| 62 | 9.918 | 9.014 | 134.031 | 10.825 |
| 65 | 原失败 | 7.399 | 8.948 | 9.924 |

大幅负差主要出现在 seed55、59、62 的 R_A 质量很差、R_B 约 10–11 的情形。
seed60 则两边质量都很差，且 R_B 更差。因此不能把巨大负均值包装成稳定的普遍收益。
原训练成功仅表示满足执行完成条件，不表示生成质量良好。
失败分布（R_A 8、R_B 1）和有限但很差的终点质量必须一起报告；
当前干预涉及完整 RAdam moments/internal-step 重启，不能单独归因于某个 moment。

## EMA 与 KID

同一完整 K 集合的共同 B0：

- E_KEEP 的 B−A log-FID 差 −0.089196，FID 几何比 0.914666。
- E_512 的差 −0.090927，FID 几何比 0.913085。
- ONLINE 的差 −0.041782，名义 CI [−0.109464, 0.025901]。

因此 K 的 B 历史优势点估计在重建 EMA 读出下仍约 8.7%；不能简单归结为继承旧 EMA 才出现。
这不是 EMA 因果中介识别，也没有证明两种 EMA 读出等效。
K 三 blocks 的几何比 0.912592，名义区间 [0.868327, 0.959113]，只是描述性支持。

R 主集合的 KID 使用原始差、绝不对 KID 取 log：
三 blocks 平均 B−A KID 差 −0.032499，名义 95% CI [−0.085859, 0.020861]。
KID 同样没有确定差异方向，不能作为替代主要检验。

## FP32 修复单独报告

所有修复标签为 POST_FAILURE_FP32_SENSITIVITY。
首两例从精确失败前状态重放；其他十例从最近保存有限 checkpoint 续跑，
精度改变的起点不同，不能混称统一从 512 开始的 FP32 对照。
不把它们与原始幸存分支拼成正式 n=16，也不对修复质量另立确认性判定。

| seed | 修复分支 | E_512 三 block FID 算术均值 |
|---|---|---:|
| 50 | R_A | 168.245 |
| 51 | R_A | 67.987 |
| 52 | R_A | 250.792 |
| 53 | R_A | 132.441 |
| 54 | R_A | 44.939 |
| 57 | R_A | 243.401 |
| 58 | K_A | 11.129 |
| 58 | K_B | 9.884 |
| 58 | R_B | 186.225 |
| 63 | R_A | 32.181 |
| 64 | R_A | 256.674 |
| 65 | K_A | 8.466 |

12/12 修复训练完成，60/60 评估有效；但 R 修复质量范围 32.18–256.67。
可以说改变精度后这些续跑到达终点；不能说已恢复良好质量，
也不能从后期干预排除此前形成的轨迹差异或识别所有失败的唯一根因。

## 证据与复算

- 原始产物保留在私有归档，以下路径均相对于归档根；公开包不含服务器路径或凭据。
- 原始来源：endpoint_evaluation/queue.csv、receipts/、jobs/*/metric-*.jsonl。
- 修复来源：fp32_repair/evaluation/*/queue.csv、receipts/、jobs/*/metric-*.jsonl。
- [提取指标](endpoint_metrics.json)：380 行；字段顺序为
  [original或repair, slot_id, training_seed, branch, readout, generation_block,
  status, metrics, archive_relative_receipt_path]。NO_ENDPOINT 的 receipt 路径为预期位置，不代表文件存在。
  这些归档索引不是仓库内附件链接；公开指标已足够离线复算，原始图像与特征不在本包内。
- [统计结果](statistics.json) 含主比较逐 seed/逐 block 原始 FID、CI、次级、KID及描述。
- [复算脚本](analyze.py)：在仓库根目录执行 `python3 analysis/q256_optimizer_restart_ema_rebuild_v1/results/analyze.py`。
  复用仓库 summarize_m1_results.py 的 seed 层 t 区间函数，不修改旧汇总器或训练代码。

本次核对：380 行无重复；原始/修复状态数完整；320 个成功评估的 640 个 FID/KID 数值
逐项与 evaluator 原始 JSONL 相等。独立 JavaScript 算式与 Python 主要/交互结果一致；
既有 14 项 M1 分析测试全部通过。以上检查不新增 hash、门禁或 GPU 任务。
尚未在本次结果读取中重新审计训练 CRN/runner 实现，也未据本结果修改任何既有协议。
