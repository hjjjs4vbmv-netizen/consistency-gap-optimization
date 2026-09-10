# q256 第 6–10 次成功更新窗口

新增 seeds50–57 的 A_delayed_down6_10，复用精确绑定的 AA 与 PR111 A_startup_down5。全部 loss 为原生 q256 A，降 LR 的倍率为 1/1.1，仅作用于成功 optimizer 更新 6–10；AMP skip 不推进窗口。各轨迹保留自己的初始化、optimizer、scaler、RNG、采样器与 512-kimg E_512 边界。

主要比较为 T=Y_delayed−Y_early，直接检验两个窗口的终点差异；L=Y_delayed−Y_AA 用于解释绝对变化。使用配对 t 区间和双侧检验，另报告符号翻转敏感性分析。此队列复用了已有响应信息，属于探索性扩展，与 q128 的新队列结果分开报告。

两个窗口的成功更新次数和 LR 倍率相同，实际参数扰动量可能不同。前 16 次成功更新及期间全部 attempted updates 记录 LR、AMP/scaler、RAdam 分支、活跃参数时钟与更新范数；窗口更新范数之和、净位移只称为实际更新暴露量。

完整执行约束见 [原始中文方案](../startup_window_experiments_v1/PLAN_SOURCE_ZH.md)。控制输入的 48 个块由共享协议按原始已归档记录逐项绑定，禁止改用 DA 或 D_startup_up5。
