## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-03T12:27:19.202264+00:00
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: q256_terminal_history_n30_matpool_v1
- **Overall Confidence**: CAUTION

### Statistical Findings

| Metric | Test | Value | Effect Size | Confidence |
|---|---|---:|---:|---|
| Paired log-FID BA−AA | paired Student-t | t(25)=-4.638101, p=9.5200854e-05; 95% CI [-0.129886, -0.050006] | dz=-0.909606 | CAUTION |
| Practical equivalence | TOST ±log(1.03) | p_TOST=0.99770742; 90% CI [-0.123072, -0.056820] | margin ±0.029559 | CAUTION |
| Final classification | frozen precedence | DIRECTIONAL_NEGATIVE | geometric FID change -8.602% | CAUTION |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Informative missingness | 5/60 endpoint failures: seed58-AA, seed58-BA, seed65-AA, seed67-AA, seed68-AA | Complete-case primary analysis |
| Differential failure | AA 4/30 vs BA 1/30；失败并非均匀分布。 | Failure-rate estimand |
| Reproducibility scope | 未进行完整独立重跑；已完成55个 checkpoint/receipt/metric/shared-feature 哈希链审计。 | Reproducibility verdict |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | 按两台训练节点分层后方向均已检查；未发现总体方向反转。 | 无需额外处理。 |
| Ecological fallacy | NOTE | 推断单位与分析单位均为 seed 级配对，不涉及由群体推断个体。 | 无需额外处理。 |
| Berkson's paradox | CAUTION | 完整病例分析排除了4个含缺失 endpoint 的 seed，选择机制与数值稳定性相关。 | 保留预注册分析，并在解释中显式报告缺失与适用边界。 |
| Collider bias | NOTE | 主模型未加入后处理控制变量，未发现显式 collider 调整。 | 无需额外处理。 |
| Base-rate neglect | NOTE | 已同时报告 AA/BA 与 A/B history 的计划分母和失败率。 | 无需额外处理。 |
| Regression to the mean | NOTE | 连续 seed 预先选定，未按极端 FID 选择样本。 | 无需额外处理。 |
| Survivorship bias | CAUTION | 主效应基于26/30完整配对；5/60 endpoint 缺失且集中于 AA。 | 保留预注册分析，并在解释中显式报告缺失与适用边界。 |
| Look-elsewhere effect | NOTE | 主 estimand、CI、TOST margin 与判定优先级均由冻结协议预设。 | 无需额外处理。 |
| Garden of forking paths | NOTE | 使用冻结协议与连续 seeds；科学失败未补跑或替换。 | 无需额外处理。 |
| Correlation != causation | NOTE | 报告仅解释预设配对干预对比，不外推非实验因果主张。 | 无需额外处理。 |
| Reverse causality | NOTE | 历史臂先于终点训练与评估，未发现反向时间顺序问题。 | 无需额外处理。 |

### Reproducibility

- **Method**: artifact-integrity verification; no independent full re-run
- **Verdict**: CANNOT_VERIFY

完整性审计状态：`PASS`。这验证了已运行结果的证据链，但不等同于独立重现实验。
