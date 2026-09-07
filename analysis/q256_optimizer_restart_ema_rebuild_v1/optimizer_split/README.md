# M1：optimizer moments / 内部 step 的短程拆分诊断

本次最重要的结果不是“找到了安全重启”，而是：**保留旧内部 step、只清 moments 虽然通过了全部 8 个短程案例，却把首步更新放大到正常继续的 12.7–14.7 倍；只重置 step 也不是无害操作，在 seed56/A 引入了原 K/R 都没有的早期故障。**

因此，这两种拆分操作都不能直接当作“温和删除历史”的处理。此次结果不选择长期方案、不启动长训练，也不改变原 M1 的主要结论 **INCONCLUSIVE（7/16 完整 R 配对）**。

## 设计与口径

- 看过原 M1 结果后选定 seed 64、55、60、56，每个 A/B@512 源状态分别施加四种操作，共 32 格。只有 4 个训练 seed，不是 32 个独立样本；8 个案例是 4 seeds × 2 histories，不能用完成率估计总体失效率。
- K：保留 moments 与内部 step；R：清空全部 optimizer state；clear_moments：只清零 exp_avg/exp_avg_sq；reset_step：只清零内部 step。
- 每格独立从 attempt4000、512 kimg 源状态开始，继续训练的目标均为 A，最多 64 次 attempted updates（4001–4064，终点 520.192 kimg）。不修改 LR、batch、GradScaler、精度、裁剪或损失策略；未跑新 FID/KID。
- 保留原始 FP16/AMP 训练和遇到 non-finite loss 的终止语义。原始 AMP 管理的梯度溢出可能只跳过该次更新而不中止。因此“跑完 64 步”不等于“64 次成功更新”或“没有数值异常”。
- 实际源 optimizer 内部 step 为 3989 或 3990，不是 attempted-iteration 的 4000；原训练已有 AMP 跳步，不能混用两个时钟。
- 所有格的首个实际更新均为 attempt4001，保存的共同输入/噪声时刻字段在同 seed 的各 history/操作之间逐步一致。首步向量只与同 seed、同 history 的 K 相比。

记首个实际参数更新为 u=θ_after−θ_before：幅度比为 ‖u_operation‖₂/‖u_K‖₂；方向相似度为 cosine(u_operation,u_K)，1 表示同向，0 表示正交。下表给出 8 个诊断案例的范围，不是置信区间。

## 核心结果

| 操作 | 完成 64 次尝试 | 因 loss 非有限终止 | 有 AMP 跳步的格 | 跳步总次数 | 首步幅度比范围 | 与 K 的首步 cosine 范围 |
|---|---:|---:|---:|---:|---:|---:|
| K | 8/8 | 0/8 | 0/8 | 0 | 1.000 | 1.000 |
| R | 7/8 | 1/8 | 6/8 | 12 | 0.904–1.910 | 0.048–0.086 |
| clear_moments | 8/8 | 0/8 | 0/8 | 0 | 12.703–14.726 | 0.304–0.377 |
| reset_step | 7/8 | 1/8 | 7/8 | 14 | 2.429–4.339 | 0.108–0.170 |

R 首步的量级可能接近 K，但方向明显不同；仅重置 step 的首步也发生明显放大与转向。clear_moments 首步幅度比中位数为 13.609，且 8 格各自的最大更新均出现在首步。没有早期溢出并不能把这一操作解释成低扰动干预。

### 两个终止案例

| 案例 | 首次 raw-gradient 非有限 | loss 非有限并终止 | 首次记录到的 forward 非有限阶段 | 最大更新范数 |
|---|---:|---:|---|---:|
| seed64/A/R | 第 5 步（4005） | 第 9 步（4009） | online_output | 3.287985 |
| seed56/A/reset_step | 第 2 步（4002） | 第 9 步（4009） | online_output | 5.402512 |

seed64/A/R 精确复现原故障。seed56/A 的 K、R 对照均完成 64 步，而 reset_step 发生新故障；这是该诊断案例内的差异，不是总体风险估计。
`online_output` 只定位到已记录的外部 forward 阶段，并不定位模型内部的第一个溢出算子。两个终止步均由 AMP 跳过 optimizer update，遥测中的 update/model 非有限计数为 0；不得写成“参数已经全部 NaN”。

## 已有完整质量读出

[QUALITY.md](QUALITY.md) 复用全部既有有效 FID/KID：64 条原始分支和 12 条 FP32 修复分支分开展示，共 76 行、228 个 B0 读出格；另列 E_512 三个 generation blocks 的 FID 均值和样本 SD。

绝对质量表也提醒我们，原 R 的大幅 B−A 差异有时来自 A 侧退化，并不等于 B 达到更好的绝对质量。例如 seed55 的 E_512 三次 FID 均值：K_A=10.221、K_B=9.495、R_A=103.385、R_B=10.504。seed60 的 R_A=108.677、R_B=138.288，两侧都差。完整表保留所有格，不能只挑这些例子报告。

不同读出使用共同 B0 比较；E_512 的三次生成 SD 不是训练 seed 不确定性。修复结果不填补原始失败，原主要推断不变。

## 证据与复核

评估结论：**可带上述边界分享，不能据此确立长期方案或机制解释。**

- 32/32 格有终态：30 个 COMPLETE_64、2 个 NUMERICAL_FAILURE，共 1938 次已观测 attempted updates；无技术故障。
- 16/16 K/R 对照均通过与原始遥测逐字段比较，仅排除 elapsed_sec 与 gpu_hours_cumulative。没有重新训练整条原轨迹。
- 32/32 的首个 forward 记录均为实际 float16 模型输入；不是只检查启动参数。
- 逐格核对源状态摘要、被修改/未修改的 moments 与 step、连续 attempts、共同随机输入字段、跳步数与实际 optimizer step 记录；由独立遥测范数复算全部首步幅度比。
- 对 seed55/A/clear_moments、seed56/A/reset_step、seed64/A/R 的完整首步向量，另外用 NumPy 拼接向量并计算范数/点积，独立复核 PyTorch 逐参数聚合结果；幅度比与余弦均一致至浮点舍入精度。
- 原报告复算测试的浮点逐位相等暴露出最大 4.44×10⁻¹⁶ 的末位差异；仅将浮点比较改为绝对容差 10⁻¹²，结构、类型、人数和判定标签仍精确比较，保存的统计值不改。
- [evidence.json.gz](evidence.json.gz) 含 32 格逐步遥测、intervention 前后摘要、首个 forward/首个非有限 forward、内部 step 记录和向量汇总。模型、首步完整向量和完整日志保留在私有 ECT 档案，不上传 GitHub。余弦依赖私有向量；公开摘要可检查报告一致性，但不能独立重算向量内积。
- [cells.md](cells.md) 为 32 格完整读数；[summary.json](summary.json) 为机器可读聚合。

运行于 2026-09-07 10:43–12:10（Asia/Shanghai）。GPU0 完成 2 格、GPU1 完成 30 格；逐格墙钟时长之和 1.546 卡小时（包含加载、记录和保存，不等于纯 GPU kernel 时间）。

### 代码与复现

运行时使用本地 M1 operational commit `1bfbd6a` 加此次短程入口。发布代码只引入诊断所需文件，不把此前恢复调度改动整体推入 PR：两段既有 forward 观察/回放比较 helper 原样内联到诊断模块/worker，避免依赖未发布的旧恢复脚本。正式训练循环不修改；运行时存在的可选恢复逻辑在这些 manifest 中未启用。

```sh
python scripts/analyze_m1_optimizer_split.py analysis/q256_optimizer_restart_ema_rebuild_v1/optimizer_split/evidence.json.gz
python -m unittest tests.test_m1_optimizer_split_results
python -m unittest tests.test_m1_optimizer_split
python scripts/m1_full_readout_table.py analysis/q256_optimizer_restart_ema_rebuild_v1/results/endpoint_metrics.json /tmp/m1-quality.md
```

前两项只需标准库；操作单元测试需要 PyTorch。私有档案持有者可运行 `python -m scripts.summarize_m1_optimizer_split <诊断根目录>` 重算向量幅度和余弦；`python -m scripts.export_m1_optimizer_split <诊断根目录> <输出JSON>` 重建完整公开证据。

ECT worker 为该服务器的路径布局编写；`--gpu 1` 运行有限队列，`--gpu 0 --wait-idle` 只在没有计算进程时接入，不抢占他人任务。这些命令会启动诊断，不是报告复核命令。自定义操作借用 K_A/K_B 容器，但路径、manifest、checkpoint metadata 均标注 `optimizer_operation` 和 `POST_OUTCOME_OPTIMIZER_DIAGNOSTIC`，不能并入正式 K 结果。

下一步先解释“大幅更新但短程未溢出”的性质，再决定是否值得进行独立长程评估；不能只按本次 8/8 选择操作并宣称长期有效。当前没有新增长训练或质量评估。
