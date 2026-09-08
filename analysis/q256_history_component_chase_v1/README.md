# q256 历史成分拆分与共同 A 续训

完整结果见 [RESULTS_ZH.md](results/RESULTS_ZH.md)，数据、图表和离线复核脚本均在 results/。

实验定义见 [EXECUTION_SPEC.md](EXECUTION_SPEC.md)。科学矩阵固定为 seeds50–65 的
32 条 CA/DA 新轨迹，完整复用 PR103 original K_A/K_B，不导入重启臂或 FP32 修复。
当前目录是实现；运行状态以节点原始日志、完整 checkpoint 和输出记录为准。

## 使用

从仓库根目录以 `python -m analysis.q256_history_component_chase_v1.<module>` 执行：

- `protocol --output OUTPUT`：生成 32 条训练队列、160 条评估槽、旧 K 来源以及合计 320 槽。
- `preflight --deployment PRIVATE_JSON --output RECEIPT`：读取固定 runtime、原数据/transfer 和旧源索引。
- `short_checks --deployment PRIVATE_JSON --output NEW_ENGINEERING_DIR --gpu INDEX`：隔离的短 GPU 检查，不产生 FID。
- `preflight --deployment PRIVATE_JSON --engineering-receipt ENGINEERING_JSON --output RECEIPT`：绑定通过的短检查。
- `worker --deployment PRIVATE_JSON --seed SEED --gpu INDEX --dry-run`：输出完整命令，不训练。
- `worker --deployment PRIVATE_JSON --seed SEED --gpu INDEX`：单卡串行执行该 seed 的两条路径。
- `evaluate --deployment EVALUATION_JSON --seed SEED --path CA_OR_DA --gpu INDEX`：在独立空闲评估卡上处理一条已经到达 1024 的轨迹，不等待该 seed 的另一条轨迹。
- `analyze --input QUALITY_320_JSON --output OUTPUT`：只有所有槽状态明确后才生成主比较；技术未决保持不完整。

`PRIVATE_JSON` 包含 `runtime_python`、`dataset`、`transfer`、`old_prefix_root`、
`old_m1_root`、`output_root`、`budget_ledger` 和 `preflight_receipt` 的实际路径。
连接信息、口令、日志和大型产物不进入仓库。每个节点可使用不同本地资产路径；
预算可以使用可靠共享存储上的统一账本，或使用互斥 seed 分区的节点本地账本。
后一种方式必须先保存全局分配表，所有 `cap_gpuh` 之和不超过 200，
所有 `scoped_seeds` 无重叠且并集恰为 50–65；不得为每个节点各给 200 GPUh。
节点配额需要调整时先降低供出节点的未消耗额度，再提高接收节点，保留记录。

## 实现边界

新的正式 protocol 只允许 CA/DA 和 seeds50–65。工程 protocol 只允许 seed50，
在 8/16 或 4008/4016 attempts 暂停。所有进程仍声明 1024-kimg 总计划。
前缀 source@512 和后缀 branch-init@512 分别保存在 `prefix/` 与 `suffix/`。
状态保留真实 C/D factorial 历史；新 metadata 单独记录当前 A、keep optimizer、
零次 optimizer reset 和一次 E_512 初始化。恢复时复制 checkpoint 中已有的 E_512。

新代码沿用原 loss 实现与 AMP skip 语义。普通 AMP skip 不补样本，不对齐成功
更新数。非有限 loss、moments、模型或 EMA 不通过 FP32/降 LR 处理。
随机输入核对仅声称旧 telemetry 实际覆盖的字段；`batch_sha256` 包含原 batch/labels
指纹。没有独立旧 epsilon/dropout 字段时，不宣称这些随机量已全部逐步核验。

worker 使用原子状态与重复任务检查；科学失败保留并继续同 seed 另一预设路径。
技术失败保留日志，同参数/同状态恢复；无完整状态时不自动从头重训。
进程退出成功而收尾校验未完成时，应先核对已有终点并完成收尾，不再训练。
节点或控制进程中断留下 RUNNING 租约时，应核对原进程已停止后再清理对应租约；
不能自动偷取活动 GPU 或将未知状态标为科学失败。

评估 runner 与训练 checkout、原冻结 evaluator 分别保存。专用评估卡使用
`evaluation_only=true` 的配置和独立预算账本，并通过 `allocation_id` 指向全局
配额调整记录。先从训练节点移出相应未使用配额，再给评估节点分配，合计仍不超过
200 GPUh；不能只复制一个已有 200 GPUh 账本。`--path` 只接受实际完成、身份匹配的
CA/DA 终点，原始文件和技术失败 attempt 保留。已有成功评估不重新生成样本。
未设置独立评估模式时，原节点 runner 会等待该节点训练全部结束后再评估。

预算只读取新增进程时间和 attempted updates，不读取 FID。
预计总成本超限时停止派发新阶段，保留完整矩阵与 INCOMPLETE_BUDGET 状态。
默认预留 8 GPUh 评估与 8 GPUh 工程/转存波动；部署时记录实际工程进程成本。
租赁闲置、传输和继承的 checkpoint elapsed 必须另列，不能算作新增训练 GPUh。

## 验证

`python -m unittest tests.test_history_component tests.test_m1_training_state`

本地 CPU 检查支持接口契约，不替代固定 CUDA runtime 的数值检查。
GPU 短检查要求 A 不干预路径对旧基线、前缀中断恢复、@512 后缀中断恢复精确一致；
检查实际 inner denoiser 输入 dtype、初始状态对旧 A/B 配对、C/D factors 和一次性 EMA。
这些短检查不能外推为完整 8000-attempt 逐位重现。
