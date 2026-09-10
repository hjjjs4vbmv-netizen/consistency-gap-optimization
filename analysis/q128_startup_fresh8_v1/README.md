# q128 新队列四臂

计划 8 个未经旧质量响应筛选的训练种子，每个种子新跑 AA、DA、A_startup_down5、D_startup_up5，均从原始 transfer 开始训练至 1024 kimg。最终种子清单依据身份审计冻结，默认候选为 301–308。

实际 loss 使用 `q128_startup_native_v1`：q=128、global=1、target=1；D 前缀只设置 denominator=1.1。在自己的 512-kimg 边界从 online 初始化 E_512 一次，并继续使用 q128 A。

每个终点仅评估 E_512、FP32、NFE1 的三个独立 FID50k/KID50k 块。主指标 S=(C−M)/2；M/C 为 Holm 校正的两项次要比较；H=DA−AA 为支持性结果。训练种子是统计单位，主表使用四臂完整的共同集合。

完整执行约束见 [原始中文方案](../startup_window_experiments_v1/PLAN_SOURCE_ZH.md)。共享执行器位于 `analysis/startup_window_experiments_v1/`；部署后独立协议、全部命令、队列、资产与工程检查的 hash 均写入运行目录的冻结回执。
