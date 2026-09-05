# M1：优化器重启与 EMA 重建后的早期 spacing 历史对比

**Protocol ID：** `m1_r1_history_persistence_q256`
**状态：** 科学规则已定稿；尚未绑定服务器源状态、验证 runner 或授权正式训练。
**范围：** 本协议只定义 M1。M2、额外对照和新质量扫点均不随本实验自动启动。

执行摘要：从按机械规则选出的 16 对 A/B@512 源状态分别建立 K_A、K_B、R_A、R_B 四条后缀；512→1024 kimg 全部使用 A。K 保留 optimizer，R 重新初始化 RAdam moments/internal step。每条轨迹维护旧 EMA（E_KEEP）和 512 时从在线模型建立的 EMA（E_512）。唯一主要比较是完整 R 配对中 R_B/E_512 − R_A/E_512 的三 block 平均 log-FID 差。

## 1. 科学问题与结论边界

### 1.1 唯一主要问题

在共同采用 A 进行 512→1024 kimg 后续训练，并于 512 kimg 重新初始化 RAdam 的 moments/internal step、从在线模型建立 E_512 的条件下，完整 R 配对中 R_B/E_512 − R_A/E_512 的 1024-kimg 条件平均 log-FID 差是否低于 0？

- A 历史：前 512 kimg 的 target/denominator scale = (1.0, 1.0)。
- B 历史：前 512 kimg 的 target/denominator scale = (1.1, 1.1)。
- 全部 M1 后缀：512→1024 kimg 的 target/denominator scale = (1.0, 1.0)。

本实验允许报告指定 restart 操作和 E_512 读出下的条件终点历史对比。R 分支仍保留历史形成的模型参数、buffers、各自 GradScaler 和其他执行状态；E_512 也从历史条件化的在线模型初始化。RAdam internal step 重置还改变其启动阶段。因此不能从 M1 单独推出：

- 历史唯一储存在模型参数、optimizer 或 EMA 中；
- moments 是质量差异的已识别中介；
- restart 擦除了多少百分比的记忆；
- 未拒绝零差异表示两边等效，或 restart 不改变历史对比。

### 1.2 条件 estimand

令 \(C_s^R=1\) 表示 seed \(s\) 的 R_A/R_B 均按协议科学完成，且两边 E_512 的三个固定 FID50k blocks 全部有效。技术未决不记为 \(C_s^R=0\)，而使主要分析保持 `INCOMPLETE_TECHNICAL`。

主要条件估计对象为：

\[
\theta_R^{\mathrm{cond}}
=\mathbb E\!\left[d_s\mid C_s^R=1,\ s\text{ 来自冻结资格规则所代表的 training-seed process}\right].
\]

实际完整集合上的 \(\bar d\) 同时是冻结队列完整 R 配对的有限集合描述均值。Student-t 区间只有在这些 training seeds 可视为上述条件 seed process 的可交换实现、且 \(d_s\) 的 t 模型适用时，才表达 \(\theta_R^{\mathrm{cond}}\) 的抽样不确定性。它不覆盖失败 seed，也不是全 16 队列或所有训练随机性的无条件区间。

## 2. 源状态与 roster

候选 training seeds 固定为 PR101 seeds 50–79。按 seed 数值升序检查 A@512、B@512，取前 16 个同时通过源资格校验的 seed。只需检查到第 16 个合格 seed 已被确定；必须记录所有实际检查和跳过的候选，不必为本实验深验其后的候选。

若不足 16 对，不自行缩小 n、扩大范围或按旧结果换 seed；状态为 `BLOCKED_SOURCE_INVENTORY`。只能恢复同一源状态的可验证副本，或在任何新 M1 质量结果产生前明确修订协议。

源资格只看 512 起点是否可用于精确恢复：

- 文件可读取，在线模型、buffers、E_KEEP、optimizer、GradScaler、RNG/sampler、loss/schedule 和进度状态完整；
- global images = 512000，attempted iteration = 4000；
- 架构、参数组、数据与运行配置匹配；
- A/B 标识和历史 scale 正确；
- 所有应已初始化的浮点状态均有限；合法的惰性未初始化 optimizer 项不误判为损坏。

不得使用旧 FID、旧 suffix 是否存活或 A/B GradScaler 是否相等作筛选条件。已知旧风险包括 seed58-AA、seed58-BA、seed65-AA；它们不自动排除合格的 A/B@512 源。若这三条 K 失败在 M1 中重现，四臂完整集合预计最多 14 对，但 R 主集合仍可能有 16 对。

冻结 roster 时记录已检查候选、资格结论、A/B 路径和外部 checkpoint SHA。源 checkpoint 只读，每个分支使用独立输出目录。

## 3. 固定训练设置

基线代码身份：PR101 ref `890a85a8ef4d9effb48f653111a70b5f15b249de`。M1 runner 可以增加 reset-once、shadow EMA、读出导出和状态汇总，但不得改变基线训练数值语义。

| 项目 | 固定设置 |
|---|---|
| 数据/模型 | CIFAR-10，ddpmpp，ECT；继承源预处理与样本顺序语义 |
| loss/schedule | q=256，inverse-gap，sigmoid；其余参数从源实际配置读取 |
| optimizer | RAdam；lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| batch | global=128，batch_gpu=16，world_size=1 |
| 精度 | PR101 FP16+AMP；TF32=false；恢复同一 CUDA/cuDNN 确定性设置 |
| 环境 | Python 3.11.13、PyTorch 2.6.0+cu124、CUDA 12.4、NumPy 2.1.2、SciPy 1.16.1；以 receipt、pip-freeze 与启动时 live probe 绑定。若使用重建 prefix，receipt 必须标记 `REBUILT_NOT_BYTE_IDENTICAL`，结论仅条件于该重建运行时，不声称复原原 tar 字节环境 |
| 其他 | dropout=0.2，augment=0，xflip=false |
| EMA | beta=0.9993；不得被新 ramp-up/half-life 参数覆盖 |
| 进度 | attempted iterations 4000→8000，即 512→1024 kimg |
| 保存 | 专用 branch-init@512，以及 640、768、896、1024 完整状态 |

训练 ZIP SHA256：`9818e4b801a52eac437485bc8a69e40b54e9ae9c5d1427467343c91de868f1b3`；ordered-pixel/label SHA256：`ccab7d422ddf0274e3d40693bfc00a87147c21bd1ebb65a7d2aaccdc142519b7`。路径可以按部署改变，数据身份不能静默替换。

启动前把 PR101 实际数值终止、sanitization、GradScaler 和 managed-overflow 设置写入同一 run manifest；不依赖 CLI 默认值，也不为让某个分支存活而新增学习率、clipping 或精度救援。

## 4. 四分支与一次性干预

| 分支 | 源状态 | 512 optimizer 操作 | 512→1024 schedule | 计划轨迹数 |
|---|---|---|---|---:|
| K_A | A@512 | 完整保留 | A | 16 |
| K_B | B@512 | 完整保留 | A | 16 |
| R_A | A@512 | 重建 moments/internal step | A | 16 |
| R_B | B@512 | 重建 moments/internal step | A | 16 |
| 合计 | | | | 64 |

K 必须用同一 M1 runner 重跑；旧 PR101 终点只作溯源参考。每条轨迹都从自己的 512 源建立，不能串接另一分支终点。

首次建立分支的顺序固定为：

`完整 restore → 校验源 → 设置后缀 A → 执行 K/R optimizer 操作 → 初始化 E_512 → 写专用 branch-init@512 → 开始第 4001 次 attempt`

`branch-init@512` 是 M1 专用初始化 checkpoint，必须在普通 milestone 完整性检查和第 4001 次 attempt 之前直接写入；不得仅通过向 `immutable_checkpoint_kimg` 加入 512 实现。它完整保存在线模型、optimizer、GradScaler、RNG/sampler、E_KEEP、E_512 和全部计数器。

- K 不修改 optimizer，`reset_count=0`。
- R 仅将 per-parameter RAdam state 变为同参数组 fresh RAdam 的等价状态，`reset_count=1`。
- 参数、buffers、param groups、LR/schedule、global progress、RNG/sampler、GradScaler 和 E_KEEP 不因 R 操作改变。
- global successful-step 日志保留；另记 restart 后的 successful steps。

分支 checkpoint 只需增加一个紧凑的 `m1` 元数据块，至少含 protocol ID、branch、seed、source SHA、initialized_at_nimg、reset_count 和 initialized EMAs。恢复 512/640/768/896 状态时加载当前 optimizer 和全部读出，绝不再次 reset 或重建 E_512。缺少关键 `m1` 字段时直接拒绝恢复。

## 5. EMA、随机输入与时钟

| 读出 | 初始化 | M1 用途 |
|---|---|---|
| ONLINE | 源在线模型，训练后按 eval-mode 副本导出 | 1024 的 B0 描述 |
| E_KEEP | 源旧 EMA | 继续基线更新；1024 的 B0 描述 |
| E_512 | branch-init 时的在线模型完整副本 | 继续同一 EMA 时钟；三个 blocks 的主要与次级分析 |

E_KEEP 与 E_512 在每个 attempted optimizer/scaler 处理后按基线语义更新，包括正常 AMP skip；EMA buffers 不在该循环中平均。E_KEEP 保留源 EMA buffers，E_512 从 branch-init 时在线 buffers 拷贝；shadow EMA 不参与梯度、target 或在线 buffers 更新，也不消耗训练 RNG。

同一在线轨迹的 E_KEEP 与 E_512 是相关读出；其差异只描述终点质量对完整 EMA 读出状态（包括参数平均历史与 buffer 初始化）的敏感性，不识别 EMA 的训练因果作用或独立存储贡献。

各分支仍按表保存 768 完整状态；M2 只需要其中的 K@768。M1 不建立 E_768。若以后授权 M2，再从 K@768 恢复并建立 E_768；M2 准备工作不构成 M1 启动或有效性门槛。

同一 seed 的 A/B 源来自相同 PR101 schema/runtime，因此逐 rank RNG 状态和 sampler cursor/state 必须相等。四分支按相同 attempted-update index 使用相同 batch/labels、t、base_r、epsilon、实际 online/target 输入和 dropout 随机实现。若不相等或实际输入不一致，启动被阻断；不得覆盖 B 的 RNG、删 seed 或事后改用外部 tape。

各分支保留自己的源 GradScaler。A/B scaler 可以不同，不据此换 seed 或共同清零。数据和 global kimg 按 attempt 推进；AMP skip 不补跑样本，RAdam internal step 按真实 optimizer 更新推进，EMA 时钟按基线规则推进。

## 6. 终点评估矩阵

只在 1024 kimg 评估；640/768/896 不做质量评估。

| 轨迹集合 | 读出 | blocks | 计划 jobs |
|---|---|---:|---:|
| 四分支 × 16 seeds | ONLINE | B0 | 64 |
| 四分支 × 16 seeds | E_KEEP | B0 | 64 |
| 四分支 × 16 seeds | E_512 | B0、B1、B2 | 192 |
| 合计 | | | 320 |

| block | 固定 generation sample seeds（含两端） | 样本数 |
|---|---|---:|
| B0 | 0–49999 | 50000 |
| B1 | 50000–99999 | 50000 |
| B2 | 100000–149999 | 50000 |

全部 job 使用 commit `d6aba02fb88e9db0993623895eb2228ed717d810` 的 exact clean Git evaluator checkout、FP32、NFE1，以及同一冻结 evaluator 和等价参数链、feature extractor、reference statistics 与预处理；相邻 archive 本身不足以证明实际执行树未漂移。每个 block 独立计算 FID50k；KID 复用同一生成特征，metric RNG=`20260730`。不把三个 block 合并成 FID150k。

主分析平均三个 log-FID 配对差，不改成先平均 FID 再取 log。三个 blocks 是 generation 重复测量，不增加 training-seed n。不得依据 block 方向删除、替换或增加 block。ONLINE/E_KEEP/E_512 横向描述只比较共同 B0。

320 是计划槽位上限，不承诺 320 个有效数值。若三条已知 K 失败重现，将有 15 个 `NOT_RUN_NO_ENDPOINT`，最多 305 个可执行评估；空缺预算不转给新 seed、readout 或 block。

## 7. 主要与次级分析

令 \(F(s,j,e,b)\) 为有效、有限且严格为正的 FID50k，\(S_R=\{s:C_s^R=1\}\)，\(n_R=|S_R|\)。每个完整 seed 先计算：

\[
d_s=\frac13\sum_{b=0}^{2}
\left[\log F(s,R_B,E_{512},b)-\log F(s,R_A,E_{512},b)\right].
\]

随后在 training-seed 层报告 \(\bar d\)、样本 SD（ddof=1）及双侧 95% Student-t 区间：

\[
\bar d\pm t_{n_R-1,0.975}\frac{\operatorname{sd}(d_s)}{\sqrt{n_R}}.
\]

| 条件 | 主要结果标签 |
|---|---|
| CI 上界 < 0 | `B_ADVANTAGE_SUPPORTED_CONDITIONAL` |
| CI 下界 > 0 | `B_DISADVANTAGE_SUPPORTED_CONDITIONAL` |
| CI 包含 0 | `INCONCLUSIVE` |
| n_R < 2 | `INSUFFICIENT_COMPLETE_R_PAIRS` |
| 任一主要技术任务未决 | `INCOMPLETE_TECHNICAL`；不发布完成裁决 |

若 n_R<16，标题、摘要和机器结果都必须显示 `n_R/16`，并附全 16 队列的逐臂成败表。科学失败没有有限 log-FID，不插补；缺一个主要 block 不平均其余两个。观测 SD=0 时报告退化情形，不把软件 NaN 改成显著结果。

同时报告 `exp(mean(d))` 及指数化区间、`100*(1-exp(mean(d)))`、逐 seed/逐 block 原始 FID、d_s 和方向计数。仅此一个主要检验；不增加单侧 p、符号多数、LOSO、TOST 或改善百分比门槛来救回跨零区间。

关键次级分析只在四分支 E_512 三 blocks 全部有效的同一集合 \(S_4\) 内计算，\(n_4=|S_4|\)：

\[
i_s=\frac13\sum_b
\{[\log F_{R_B,b}-\log F_{R_A,b}]-[\log F_{K_B,b}-\log F_{K_A,b}]\}.
\]

任一四臂所需 E_512 block 仍有技术未决时，secondary 保持 `INCOMPLETE_TECHNICAL_SECONDARY`；技术未决不作为科学失败排出 seed。只有相关槽均有效或已证实为科学缺失后，才冻结 S_4/n_4。

报告 n_4、均值、名义 95% CI 和逐 seed 值；它不是第二项确认性门槛，也不能用不同 seed 集合的 R/K 均值相减。正的 restart×history 差分表示 restart 使 B−A log-FID 对比点估计向更高值移动；它只描述历史对比的改变，不单独证明存在或削弱了 B 优势。

其他结果均为描述性：B0 的 ONLINE/E_KEEP/E_512 原始质量与配对差、绝对质量变化、三 block 波动、KID、失败率和成本。不同读出或 K/R 对比不得写成对总改善的可加组件分解。

## 8. 失败、恢复与状态

| 事件 | 固定行动 |
|---|---|
| 正常 AMP skip | 按基线继续并记录；不是自动失败 |
| 基线规定的科学数值终止 | `SCIENTIFIC_FAILURE`；保留证据，不改参数或反复回滚直到成功 |
| 基础设施中断 | 同 run/参数/源，从最后完整状态恢复；最多 2 次，仍失败为 `INCOMPLETE_TECHNICAL` |
| 临时评估故障 | 同 checkpoint/readout/sample IDs 最多 2 次额外重试；已有有效特征可只重算指标 |
| readout state_dict 任一参数或 buffer 非有限，或预注册全零图像、sigma=1、全零标签的 FP32/eval/no-grad 固定输入 forward 产生非有限输出 | 证据确认后为 `SCIENTIFIC_READOUT_INVALID`；不换样本；forward 异常而未产生上述观测时仍为技术未决 |
| runner/CRN/导出语义错误 | `INVALID_IMPLEMENTATION`；隔离受影响产物，修复并重过 gate |
| 源文件不可读 | 只恢复同一外部 checkpoint 身份；恢复不了则技术阻断，不换 seed |

正式 1024-kimg readout 必须先以 `training_receipt.json` 中对应 branch 的 canonical 1024 milestone 路径/SHA完成固定输入分类。只有 `READOUT_VALID` classification receipt 可进入 exporter，且唯一 evaluator snapshot 由 exporter 生成；`SCIENTIFIC_READOUT_INVALID` 只生成缺失证据，不生成或进入可评估 snapshot。G4 的 attempt-4032 非质量导出不适用此正式分类门禁。

K 失败不排除同 seed 的完整 R；非主要读出或 KID 失败不自动删除有效 R/E_512/FID。某分支科学失败后其余预定分支继续。技术重放不能覆盖原科学失败。

每条训练轨迹和评估槽只需记录当前状态、最终原因及必要路径；不为同一事实维护多份账本。正式解码前核算全部计划槽。R primary 仍有技术未决时，不得通过删除未决 seed 完成主要裁决。

## 9. 最小实现与启动 gates

实现范围：M1 branch-init/恢复模式、reset-once、E_512 保存与恢复、按读出导出、三 block 评估和缺失感知汇总。功能默认关闭时必须保持旧 runner 行为。

对所有入选的 32 个源做只读检查；前两个入选 seed 先完成 G1–G3，再复用其隔离的 attempt-4032（516096 nimg）真实 M1 gate state 完成 G4：

| Gate | 必须证明的行为 |
|---|---|
| G1 基线与源 | 源身份/进度/状态正确；一个 K 连续 32 attempts 与 PR101 合法恢复一致；实际数值规则已绑定 |
| G2 干预与恢复 | K 不变；R 只重建指定 optimizer state 且等价于 fresh RAdam；branch-init、16+save+restore+16 精确恢复；reset/E_512 只发生一次 |
| G3 shadow 与 CRN | 开关 E_512 不反馈 online/E_KEEP/optimizer/scaler/RNG；四分支实际 batch、t、noise、dropout 按 attempt/microbatch 一致 |
| G4 导出评估与基本资源条件 | 从同一真实 gate state 导出 ONLINE、E_KEEP、E_512；覆盖 ONLINE-B0、E_KEEP-B0、E_512-B0/B1/B2 五类 dry-run；验证 snapshot 反序列化、NFE1/FP32、block 区间、共享特征路径和统计器配置，以及 live A100、最低空闲显存、空闲磁盘与 cache 条件；不得执行质量生成 |

“精确一致”比较规范化计算状态和张量，不要求路径、耗时或标签不同的整个 checkpoint 文件字节相同。先通过 G1–G3；随后直接从其隔离输出根中的 attempt-4032 短 gate state 生成三个非质量 readout，并完成五类反序列化/evaluator/resource canary。个体 canary 必须原子写持久 PASS/FAIL receipt；五类再收口为一个 PASS seal，绑定 training/evaluation manifest、G1–G3 receipt、gate state、branch manifest、snapshot、runtime、evaluator、数据和实际 probe。所有 gate artifact 均 `quality_eligible=false`、`quality_generation=false`，不进入 canonical 320 槽或推断集。G4 PASS 后，才从冻结 source 启动 64 条正式轨迹。dry-run 不支持实测 50k 吞吐或峰值显存的表述。无需为 G4 另跑完整 512-kimg suffix，也无需额外科学 recurrence 扫点、rectified-update 叙事检查或 E_768 gate。

## 10. 冻结、运行与预算

代码和协议用一个 Git commit 标识。冻结材料只需：已检查候选与 16-seed roster、一个可展开 64 条训练和 320 个评估槽的参数化 run manifest、外部源 checkpoint/data/evaluator 身份、运行环境、G1–G4 结果和资源绑定。训练槽与评估槽各自的原子 receipt 记录所绑定对象、UTC 启动时间和终态；不再另设重复的全局 `launch_record.json`，也不手工维护 64/320 份重复 manifest。

顺序：完成 inventory 和 runner → 形成候选 commit → 由该 commit 生成并绑定 run manifest/evaluation manifest → G1–G3 产生 attempt-4032 short states → 从其中一个 short state 导出三 readout并完成五类 G4 canary/aggregate seal → G4 PASS → 确认代码仍是同一 clean commit → 从冻结 source 运行完整 64 条正式训练并逐槽原子记录 receipt → 滚动评估并封存 canonical 320 槽状态 → 全部槽核算后统一解码。gate 后若改代码，必须形成新候选 commit、重建 manifest 并重过 G1–G4。新质量数值不得用于改 roster、blocks、读出或失败规则。

同 seed 四分支在同一物理 GPU 串行，按 roster 位置循环使用以下顺序以分散顺序影响：

1. K_A→K_B→R_A→R_B
2. K_B→R_A→R_B→K_A
3. R_A→R_B→K_A→K_B
4. R_B→K_A→K_B→R_A

八卡第一波 S01–S08，第二波 S09–S16。训练和评估不在同一卡上争抢影响冻结设置；空闲资源只执行预定剩余任务。

| 项目 | 估计 |
|---|---:|
| 64 条 512-kimg suffix | 97.336 A100 GPUh |
| 320 个终点评估（历史均值约 405.756 秒/job） | 36.067 A100 GPUh |
| raw 总量 | 133.403 A100 GPUh |
| 八卡无损并行下界 | 16.675 小时 |
| 排期假设 | 20–28 小时；正式 preflight 后更新 |
| 保守容量 | 185–225 A100 GPUh |

资源预测可以在无质量 preflight 后更新，但不能因此减少 seeds、blocks 或 readouts。

## 11. 最终交付与用语

最终包至少包含：roster/源身份、run manifest 与代码 commit、64 条轨迹和 320 个评估槽的状态表、逐 seed×branch×readout×block 指标、主要 d_s/CI、S_4 次级对比、全 16 成败表、必要的失败/恢复证据、G1–G4 结果以及实际成本。

| 结果 | 可写 | 不可写 |
|---|---|---|
| 主 CI 上界<0 | “完整 R 配对中，指定 restart 与 E_512 读出下 B 历史终点 FID 更低” | “历史唯一储存在参数中”“保留了原收益的某百分比” |
| 主 CI 跨 0 | “本样本与精度下未确认条件差异方向” | “两组等效”“restart 擦除了历史” |
| interaction 点估计>0 | “restart 后 B−A log-FID 对比向更高值移动” | “moments 解释了多少改善” |
| interaction 区间跨 0 | “影响方向与幅度尚未明确” | “optimizer 不重要” |
| K 失败、R 成功 | 分别报告稳定性与条件 R 质量 | 删除整个 seed |

M1 不运行 M2，不做 moments 移植、m/v 拆分、保留 step 的额外 reset、GradScaler 共同重置、新 spacing、NFE2、640 质量扫描、额外 generation blocks 或科学 recurrence 扫点。

## 12. 来源与尚未完成的工作

- PR101 fixed ref：`890a85a8ef4d9effb48f653111a70b5f15b249de`。
- 数据、训练配置、已知失败和基线执行语义来自该 ref 的 `analysis/q256_terminal_history_n30_matpool_v1/`、`training/ct_training_loop.py` 与既有 evaluator/exporter。
- M1 runner、评估槽与分析器已有实现候选，但尚待服务器 inventory 与 G1–G4 实机 gates；未启动正式 M1 训练或质量评估。不得把本文件或未执行的候选代码当作 gate PASS 或运行证据。
- 不需要预先附空的 rules JSON 或 64/320 行模板；inventory 完成后由唯一 run manifest 机械展开正式槽表。
