# q256_startup_step_responder_pilot_v1

## Frozen user specification

实施并执行正式质量实验：

q256_startup_step_responder_pilot_v1

仓库：
hjjjs4vbmv-netizen/consistency-gap-optimization

基线：
PR110 head:
d9e0021f7e6a8049a642f81bd22390aec0b383ac

PR110 实际工程执行实现：
4198387b514fedbe43574452b18e021923a1de58

PR109 head：
092ab9ab54e3f2dd780654b752bf85643d92af59

从 PR110 head 创建独立 stacked branch：

experiment/q256-startup-step-responder-pilot-v1

不要修改、删除或放宽 PR110 的 engineering-only 守卫。
不要把 PR110 的 64-attempt checkpoint 用作正式训练源。
不要简单把原 engineering manifest 的 attempts 从 64 改成 8000。
必须新建独立正式质量协议。

==================================================
一、固定科学矩阵
==================================================

固定 seeds：

50,51,52,53,54,55,56,57

固定新增 arms：

1. D_startup_up5

0–512 kimg：
- native target_gap_scale = 1.0
- native denominator_gap_scale = 1.1
- 使用原生 D loss
- RAdam 成功 optimizer step 1–5 临时使用 lr × 1.1
- 成功 optimizer step 6 起使用原始 lr

512–1024 kimg：
- target_gap_scale = 1.0
- denominator_gap_scale = 1.0
- 使用 A
- 原始 lr

2. A_startup_down5

0–512 kimg：
- native target_gap_scale = 1.0
- native denominator_gap_scale = 1.0
- 使用原生 A loss
- RAdam 成功 optimizer step 1–5 临时使用 lr ÷ 1.1
- 成功 optimizer step 6 起使用原始 lr

512–1024 kimg：
- 继续使用 A
- 原始 lr

base lr：

0.0001

复用旧正式对照：

AA
DA

只允许导入 seeds50–57 的旧 AA、DA。
逐 seed、逐 readout、逐 block 核对旧指标来源和 hash。
不要重算旧对照，不要替换 generation block。

实验身份必须写明：

responder-enriched exploratory mechanism pilot

不能称为：
- random seed validation
- independent new-seed confirmation
- general population mechanism estimate
- exact five-step trajectory reproduction

==================================================
二、新代码结构
==================================================

新建：

analysis/q256_startup_step_responder_pilot_v1/
    EXECUTION_SPEC.md
    protocol.py
    preflight.py
    worker.py
    evaluate.py
    analyze.py
    verify_results.py
    results/

training/startup_quality.py

tests/test_startup_quality.py

正式 CLI：

--startup-quality-manifest=<path>

正式 manifest 至少包含：

protocol_id
experiment_class
seed
arm
initialization
native_prefix_arm
continuation_arm
startup_success_steps
lr_multiplier
base_lr
switch_kimg
final_kimg
transfer_sha256
dataset_sha256
reference_initial_receipt
immutable_output_root
old_control_bindings

固定字段：

protocol_id = q256_startup_step_responder_pilot_v1
experiment_class = responder_enriched_pilot
initialization = fresh_transfer
startup_success_steps = 5
switch_kimg = 512
final_kimg = 1024

==================================================
三、学习率干预实现
==================================================

复用 PR110 已经验证过的真实 optimizer.step 包装位置，
但为正式质量实验建立独立 controller。

要求：

1. 只有 GradScaler 实际调用 optimizer.step 时才进入 controller。
2. AMP skip 不推进成功更新计数。
3. optimizer 中全部活跃参数的 step clock 必须一致，否则 fail closed。
4. next optimizer step 为 1–5 时：
   - D_startup_up5 multiplier = 1.1
   - A_startup_down5 multiplier = 1 / 1.1
5. next optimizer step 为 6 及以后：
   - multiplier = 1.0
6. step5 必须断言 RAdam 仍在未整流区间。
7. step6 必须断言 RAdam 已进入整流区间。
8. 临时 LR 必须在 finally 中恢复。
9. 不直接修改：
   - gradients
   - exp_avg
   - exp_avg_sq
   - GradScaler
   - RNG
   - sampler
10. 第六次成功更新完成后 controller 标记为 inactive。
11. inactive 后不再执行昂贵的全参数 clock 或大张量诊断。
12. 正式训练不得保存 PR110 的完整 gradient/update/parameter tensor。
13. 只保存轻量 telemetry：
   - attempted_iteration
   - successful_optimizer_steps
   - optimizer clock
   - AMP skip
   - applied multiplier
   - base LR
   - restored LR
   - scaler before/after
   - update norm
   - batch/t/base_r hashes
14. telemetry 不得增加 forward、backward 或随机数调用。

==================================================
四、完整训练路径
==================================================

两个新增臂都必须从原始 transfer checkpoint 开始。

prefix：

fresh transfer → 512 kimg

- attempted iterations 0–4000
- 保存完整 model
- 保存 EMA
- 保存 RAdam
- 保存 GradScaler
- 保存 RNG
- 保存 sampler
- 保存全部训练计数器
- 保存 startup controller 状态
- 保存 immutable 512 checkpoint

suffix：

自己的 prefix@512 → 1024 kimg

- attempted iterations 4000–8000
- 统一切换为 A
- 保留自己的完整状态
- E_KEEP 保留
- E_512 在 512 边界从 online 初始化一次
- resume 时不得再次初始化 E_512

禁止：

- 从旧 A@512 开始
- 从旧 D@512 开始
- 从 PR110 engineering checkpoint 开始
- 跨 seed 恢复
- 跨 arm 恢复
- optimizer reset
- scaler reset
- RNG reset
- 用其他 prefix 替代
- 科学失败后改 FP32
- 科学失败后降低 LR
- 科学失败后换 seed
- 根据 loss 或 FID 提前停止

技术中断只允许从本轨迹最近的完整、身份核验通过的 checkpoint 恢复。

若已有日志但没有可验证 checkpoint，不自动从头覆盖重跑；
记录 TECHNICAL_FAILURE 并停止该 slot，等待人工审计。

==================================================
五、8 GPU 派发
==================================================

固定 GPU/seed 映射：

GPU0 → seed50
GPU1 → seed51
GPU2 → seed52
GPU3 → seed53
GPU4 → seed54
GPU5 → seed55
GPU6 → seed56
GPU7 → seed57

固定 arm 顺序：

偶数 seed：
D_startup_up5
A_startup_down5

奇数 seed：
A_startup_down5
D_startup_up5

每个 GPU 同一时间只运行一个正式训练进程。
每个 GPU 使用独占文件锁和 GPU UUID 核验。
不得因为另一张卡空闲而迁移正在运行的轨迹。

生成一个总 launcher，使下面八个 worker 可并行启动：

worker --seed 50 --gpu 0
worker --seed 51 --gpu 1
worker --seed 52 --gpu 2
worker --seed 53 --gpu 3
worker --seed 54 --gpu 4
worker --seed 55 --gpu 5
worker --seed 56 --gpu 6
worker --seed 57 --gpu 7

每个 worker 顺序完成其固定的两个 arms。

==================================================
六、预算守卫
==================================================

总硬上限：

90 GPUh

账本初始化：

cap_gpuh = 90
evaluation_reserve_gpuh = 8.5
engineering_reserve_gpuh = 2.5

预算预测必须包括：

- 已完成训练进程成本
- 正在运行进程的实时成本
- 未派发训练的预测成本
- 未完成的固定评估成本
- 工程和导出预留

若预测超过 90 GPUh：

- 不再派发新的训练或评估
- 标记 INCOMPLETE_BUDGET
- 不强杀正在正常运行的科学轨迹
- 不把已完成子集伪装成完整矩阵

GPUh 使用实际进程墙钟时间。
不要使用 checkpoint 继承的 elapsed。
租赁闲置时间与 process GPUh 分开记录。

==================================================
七、训练前检查
==================================================

短检查只做一次，总目标不超过 3 GPUh。

检查：

1. seed50 原始 A no-intervention replay。
2. seed50 D_startup_up5 前 64 attempts。
3. seed50 A_startup_down5 前 64 attempts。
4. 成功 step5/step6 的 multiplier 边界。
5. AMP skip 不推进成功计数。
6. 异常退出后 LR 仍恢复。
7. uninterrupted 与 pause/resume 在 step3、step5、step6 一致。
8. 初始 model/EMA/optimizer/scaler/RNG/sampler receipt 与旧起点一致。
9. 正式 checkpoint 不包含 startup_engineering metadata。
10. PR110 engineering checkpoint 被正式入口拒绝。
11. 原 A 路径已有 telemetry 字段可精确复现。
12. 不产生正式 FID。

检查通过后冻结：

- implementation commit
- protocol JSON
- protocol SHA256
- environment
- source hashes
- dataset/transfer hashes
- 16 条正式命令
- 8 GPU 映射
- budget ledger

==================================================
八、评估
==================================================

所有训练 outcome 达到终态后生成：

training_matrix_frozen.json

终态包括：

PASS
SCIENTIFIC_FAILURE
NO_ENDPOINT
TECHNICAL_FAILURE
INCOMPLETE_BUDGET

评估入口在 training_matrix_frozen.json 生成前必须拒绝运行。

唯一主要质量读出：

1024 kimg
E_512
FP32
NFE1

每个新增 arm 固定三个 block：

B0：sample seeds 0–49999
B1：sample seeds 50000–99999
B2：sample seeds 100000–149999

固定：

metric seed = 20260730
FID50k
KID50k
metric repeats = 1
evaluator commit =
d6aba02fb88e9db0993623895eb2228ed717d810

每个 GPU 评估自己 seed 的两个 arm，共六个槽。

总新增评估槽：

8 seeds × 2 arms × 3 blocks = 48

训练中不运行 FID。
不评估中间 checkpoint。
不新增 ONLINE 或 E_KEEP FID。
但必须保留 ONLINE、E_KEEP 和 E_512 权重。

==================================================
九、冻结统计分析
==================================================

每个 seed 和 arm：

Y = mean(
    log(FID_B0),
    log(FID_B1),
    log(FID_B2)
)

定义：

C =
Y_D_startup_up5 - Y_DA

M =
Y_A_startup_down5 - Y_AA

S =
(C - M) / 2

唯一 primary：

S

prespecified simple effects：

C
M

预期方向：

S > 0
C > 0
M < 0

报告：

- n planned
- n trained
- n finite paired
- 每个 seed 的 AA、DA、D_startup_up5、A_startup_down5
- C、M、S
- mean
- SD
- nominal 95% t CI
- paired dz
- geometric FID ratio
- 正负方向计数
- exact sign-flip sensitivity
- 配对图
- 所有科学失败

如对 C、M 分别判断显著性：

- 双侧 paired t-test
- Holm correction
- 普通 95% CI 标为 nominal
- 不称为 simultaneous Holm CI

实用等效尺度：

delta = log(1.03)

p > 0.05 不代表无效。
只有 TOST 支持时才能声明落入 ±3% 实用等效范围。

KID 使用同一批 generated features，
仅作为辅助一致性结果，
不能称为独立 replication。

==================================================
十、结果解释边界
==================================================

若 S > 0 且 C > 0、M < 0：

支持：
在 seeds50–57 的 responder-enriched cohort 中，
前五次 RAdam 成功更新的方向性 LR 干预能够双向调节长期质量。

不支持：
- 解释全部 DA 收益
- 唯一 optimizer 中介
- 一般 seed 上的平均机制效应
- 跨数据集普遍性
- 五步轨迹精确复现

若只有 C > 0：

说明 DA 对启动上调干预敏感；
单独不足以证明 AA 可以复现 D 历史收益。

若只有 M < 0：

说明 A 的启动降尺度可能产生部分收益；
单独不足以证明它解释 DA。

若两者弱、接近零或反向：

降低该启动机制假说优先级；
不自动增加 seed、倍率、窗口长度或 mediator 搜索。

==================================================
十一、最终交付
==================================================

提交：

- EXECUTION_SPEC.md
- protocol.json
- protocol SHA256
- implementation/execution commits
- frozen commands
- preflight receipts
- GPU mapping
- training queue
- 全部 outcome
- budget ledger
- process GPUh
- evaluation slots
- old-control bindings
- per-seed.csv
- statistics.json
- paired figures
- RESULTS_ZH.md
- VALIDATION.md
- archive index
- large-artifact SHA256

不要提交：

- checkpoints
- generated sample arrays
- feature arrays
- 私有绝对路径
- 用户名、口令或服务器凭据

完成代码、测试和短 GPU preflight 后，直接按固定 8-GPU 队列执行。
不得根据中间训练输出或新增 FID 修改协议。

## Deployment details

The 8 logical GPU lanes span MatPool local GPUs 0–5 (seeds50–55) and ECT local GPUs 0–1 (seeds56–57). UUIDs are frozen before dispatch. Host capacity envelopes are 67.5 and 22.5 GPUh, respectively, summing to 90. Each dispatch conservatively reserves the entire other-host envelope, including its completed, live, future and evaluation work. This can stop earlier than a centralized optimistically shared budget. No normal training process is killed for exceeding a prediction.

Each of 16 formal commands runs its own fresh-transfer prefix and suffix in one process, sealing a complete immutable 512 checkpoint before its boundary operation. On the next iteration, the same native loss object switches to A and E_512 is deep-copied from online exactly once; the own full state is preserved. A technical resume restores this phase flag and E_512 state. Existing PR110 engineering-only guards are not changed.

Native strict telemetry diagnostics are retained for replay parity. The new controller adds no tensor snapshots, forwards, backwards or RNG calls; after successful step6 its telemetry clock is explicitly the successful-call counter rather than an expensive all-parameter diagnostic.

## Dispatch cost estimator

The operational initial per-trajectory forecast is the larger measured full process cost of the two intervention 64-attempt preflights, multiplied by 8000/64 and rounded upward. This extrapolates startup and checkpoint overhead as well as training time, even though startup occurs once in a formal trajectory. A separate 25%-throughput-plus-0.3-GPUh stress scenario is retained as sensitivity, not used to change the scientific matrix. The operational forecast plus 8.5 GPUh evaluation reserve and 2.5 GPUh engineering/export reserve must be <=90 before dispatch. After 200 processed attempts, each worker recomputes its remaining cost from its own current process wall time and actual attempt progress; unstarted jobs are conservatively raised to the observed full-run rate. Any over-budget live forecast latches INCOMPLETE_BUDGET and blocks further dispatch without killing normal active trajectories.

## Preflight / final-source provenance

The sole GPU preflight ran at cdd1326fea08548cd2a4b403d9c02cfd97bb37b4 (0.20301482354808187 process GPUh). It covered 64-attempt native A replay and both interventions, then each intervention's own step3/5/6 pause/resume trajectory. The final gate-only additions require six same-seed old-control bindings, prohibit training metrics, require both immutable milestones, preserve the old planned-pause predicate grouping, and check suffix E_512 finiteness. These do not change the LR controller's numerical operations or the successful preflight trajectories. They are checked with final CPU regression tests; the suffix-only E_512 guard is not claimed to have been exercised by the short GPU preflight. Final implementation and preflight execution commits are recorded separately.

## User financial clarification: owned ECT GPUs

The user clarified that ECT GPUs are owned and card-hour limits need not apply to them. This supersedes the initial shared 90-GPUh reservation split above. The 90-GPUh cap applies only to paid MatPool process time; ECT process GPUh is recorded separately with no financial dispatch limit. MatPool retains the full 8.5-GPUh evaluation and 2.5-GPUh engineering/export reserves. No seed, arm, successful-step intervention, GPU UUID, training state, readout, generation block, or statistical definition changed. Existing GPU processes were not interrupted. The old and new protocol hashes and the user's authorization are preserved in FINANCIAL_SCOPE_AMENDMENT.json.

## User scheduling amendment: separate host completion gates

The user explicitly requested immediate MatPool evaluation after all 12 MatPool trajectories reach terminal outcomes, followed by transfer of all experiment data to ECT and hash verification. The user is notified that MatPool can be released only after this verification; no automatic cancellation is authorized. ECT continues its own four trajectories and 12 fixed evaluation slots. Each host writes a separately scoped training-matrix freeze before evaluation. The global 16-trajectory matrix and combined inference are assembled only after both hosts finish. This supersedes the prior requirement to wait for all 16 trajectories before any evaluation. Seeds, GPU UUIDs, arm order, native losses, checkpoint identities, E_512/FP32/NFE1, all 48 blocks and all statistical definitions remain unchanged. No new FID had been computed when this user-authorized ordering amendment was made.

## User replacement-GPU amendment

The user supplied an additional paid A100 for the still-unstarted seed57 D_startup_up5 slot. The old waiting dispatcher was stopped only after confirming the trajectory directory was empty and its budget job was PENDING. This arm starts from the original transfer on logical GPU7, now bound to the new A100 UUID. No running trajectory moved. Seed57 A_startup_down5 already completed on ECT's original GPU1; its checkpoint, manifest and outcome retain their original identities and are copied solely for the fixed endpoint evaluation. The replacement worker is restricted to the D arm and cannot retrain A. Evaluation groups are original MatPool seeds50–55 (36 slots), ECT seed56 (6 slots), replacement seed57 (6 slots), still 48 in total. The two paid instances share the same 90-GPUh cap with 80/10 capacity envelopes; owned ECT stays exempt. Each paid group's complete data is verified on ECT before the user is notified that that specific instance may be released. Joint inference waits for all groups.

## User completion-priority amendment

The user subsequently instructed completion of the fixed experiment without stopping dispatch on GPU-hour forecasts. The 90-GPUh figure is retained as a historical reference, not an active dispatch cutoff. Actual paid and owned GPU usage remains recorded. This authorizes completion of the existing 16 trajectories and 48 fixed evaluation slots; it does not add seeds, arms, blocks or adaptive scientific changes. Existing successful runs are never retrained, and technical-recovery identity checks remain intact.
