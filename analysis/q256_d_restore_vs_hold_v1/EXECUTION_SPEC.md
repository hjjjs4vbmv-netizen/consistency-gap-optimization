# Codex 任务书：D-history 后缀恢复与保持对照

实验 ID：`q256_d_restore_vs_hold_v1`

## 执行目标与范围

在 consistency-gap-optimization 仓库实施并完成唯一新增质量实验：seeds50–65 的 DD 后缀训练。复用 PR108 对应 DA 路径及其完整 D@512 源状态，比较 512–1024 kimg 恢复 A 与持续 D 的终点质量。AA 仅作辅助参考。

A 的 target/denominator factors 为 1.0/1.0；D 为 1.0/1.1。DA 与 DD 共享每个 seed 的同一个实际 D 前缀。

本轮只新增 16 条 512→1024 kimg 的 DD 后缀。不要重跑前缀，不新跑 AA、DA、B、C、AD、optimizer reset/swap、全程 FP32 或完整 scalar-loss 质量对照，不扫 seed、学习率、倍率、切换时点或终点。允许下述短工程检查，不做这些短检查的 FID。

这是在已观察 PR108 后增加一个预先固定的后缀对照，不是独立新 seed 的确认性复现。不要改写 PR108 的结果、样本集合或统计结论。

## 1. 基线、资产和旧对照

仓库：`hjjjs4vbmv-netizen/consistency-gap-optimization`。
参考 PR108 head：`90a0cd3f6ade59c04928b6e58a17bbeb95ed0705`。
冻结 evaluator：`d6aba02fb88e9db0993623895eb2228ed717d810`。

创建独立工作分支；沿用 PR108 的数值实现及已有运行环境，不合并无关实验改动。报告实现 commit 和实际部署 commit，不把结果报告 commit 误称为所有历史运行的执行 commit。

先阅读：
- `analysis/q256_history_component_chase_v1/EXECUTION_SPEC.md`
- `analysis/q256_history_component_chase_v1/protocol.py`
- `analysis/q256_history_component_chase_v1/worker.py`
- `analysis/q256_history_component_chase_v1/evaluate.py`
- `analysis/q256_history_component_chase_v1/results/RESULTS_ZH.md`
- `analysis/q256_history_component_chase_v1/results/archive_index.json`
- `analysis/q256_history_component_chase_v1/results/quality_slots_320.json`
- `analysis/q256_history_component_chase_v1/results/training_stages.csv`
- `training/history_component.py`
- `training/schedule_switch.py`
- `training/ct_training_loop.py`
- `training/loss.py`
- `training/schedules.py`
- `training/m1.py`

使用项目现成的服务器连接、数据路径和归档引用，不把私有连接或绝对私有路径写入公开 PR。先读取索引与已有 hash 回执，不反复全量扫描历史大文件。

建立 16 行 source inventory：seed、D-prefix@512 原始完整状态、旧 DA branch-init@512、旧 DA 终点、旧 DA 五个评估槽及其来源。PR108 的 DA seeds50–65 全部成功，因此全部进入本轮预定队列。seed58、65 不因 AA 失败被排除。

优先从原始 D-prefix@512 完整状态构建 DD branch-init；用旧 DA branch-init 核对相同的计算状态。仅有 evaluator snapshot 或 EMA 权重不足以恢复训练，不能代替完整源状态。

导入旧 DA 的 80 个终点评估槽，不重算、不选择生成 block、不替换 seed。旧 AA 保留其 16-seed 槽位及已知缺失；其有效集合是 seeds50–57、59–64，仅用于辅助比较。实际文件缺失属于资产/技术问题，先定位，不默默删 seed 或另造近似对照。

## 2. B 核查：数学、浮点、学习率分别判断

本节只做源码推导与短工程审计，不新增完整质量训练。

### 数学核查

依据实际 q、k、b、stage 和 clipping 实现推导 D 与 A 的关系。当前 sigmoid、q256、k8、b1、stage0 的实数公式满足：

\[
\Delta_A(t)=\frac{t[1+8\operatorname{sigmoid}(-t)]}{256},\qquad
1.1\Delta_A(t)<\frac{5.5}{256}t<t.
\]

因此对有限正 t，理论上不会触发放大间隔到 t 的裁剪；相同参数、样本、噪声和 dropout 下：

\[
\ell_D(\theta;\xi)=\frac{\ell_A(\theta;\xi)}{1.1}.
\]

用旧 telemetry 核对真实 stage、裁剪计数、无效 t/denominator；未记录的内容不能声称已全程核查。该等价是共同状态的目标等价，不能用于比较已经分叉的不同参数轨迹。

### 浮点核查

实际 D 的实现先构造 r_denominator，再做 t-r_denominator，可能与直接把 A loss 除以 1.1 的舍入顺序不同。保留生产 dtype，不能因为 denoiser 是 FP16 就把 t/base_r/denominator 也强制转成 FP16。

固定审计 seeds50、51；使用可从原资产精确恢复的 ECT 起点和 D@512。必要时使用这两条旧 DA@768，但不为取得状态新增训练。优先复用已有可信的同口径审计。

在同一完整状态、同一 effective batch 和随机输入上比较：
1. 原生 A；
2. 原生 D；
3. 原生 A 的逐样本输出除以 1.1，再按原 reduction 处理；
4. 原生 A，optimizer lr 除以 1.1。

记录 scalar 操作插入的位置；如审计 batch-reduced loss 的缩放，另列，不把两种 reduction 顺序混称逐位相同。

分层记录 target 输入/输出一致性、denominator 比例、loss/gradient 的逐位一致率及绝对/相对误差、一次实际 RAdam 更新、moments、scaler/skip/step counter。每种分支从同一状态副本恢复，审计不改变正式 source 或其 RNG。短 rollout 每个状态最多 32 attempts；不为追求“通过”不断延长，合计工程核查目标不超过 3 GPUh。

输出：
- 数学上是否等价及适用范围；
- 浮点审计范围内是否逐位相同；不相同时真实误差与离散分歧；
- 学习率替代是否等价，以及具体区别。

学习率缩放不直接改变输入 optimizer 的 gradient/moment 更新；loss 缩放会。因此不能依据 loss 数学等价宣称 RAdam 的学习率等价。另注意 RAdam 的早期非自适应分支；不要笼统写成“RAdam 全程严格 scale-invariant”。

完整 scalar-loss 质量臂默认不启动。若数学等价且短审计未见超出舍入的数值语义差异，明确取消该完整对照。若出现 skip 或明显更新分歧，原样报告、限制“浮点/训练等价”主张，不擅自扩展完整质量矩阵。DD 始终沿用原生 D 数值实现。

## 3. DD 后缀定义与实现

为新实验创建独立 protocol ID 和元数据路径，建议目录 `analysis/q256_d_restore_vs_hold_v1/`。

特别注意：PR108 的 `protocol.py` 在 suffix 分支强制使用 A，`training/history_component.py` 的 current_arm、factor 和 branch 校验也写死 A。不能只修改启动命令中的 denominator 参数。为新协议显式支持 D→D，并保留旧 PR108 协议及默认路径的行为不变。不要全局移除旧校验。

每个 DD：
- 从旧 DA 对应的完整 D-prefix@512 恢复，attempt=4000、nimg=512000；
- source_history=D，current_arm=D，target factor=1.0，denominator factor=1.1；
- 保留 online parameters/buffers、RAdam m/v/internal step、param_groups、GradScaler、CPU/CUDA/NumPy/Python RNG、sampler、训练进度与原 schedule/LR 时钟；
- 保留原 E_KEEP；
- 从该边界 online 模型创建 E_512 一次，数值上必须与旧 DA branch-init 的 E_512 一致；
- 保存新的不可覆盖 DD branch-init 完整状态；后续 resume 不再次初始化 E_512；
- 训练至 attempt=8000、nimg=1024000，使用总计划 `--duration=1.024`，不是新增 duration=0.512；
- 保持 DD 全后缀 D，不改变 target endpoint 的规则；原生 D 的实际分母计算、裁剪和 reduction 完全保留。

元数据可以因实验身份、路径和 current policy 而不同；对非干预的计算状态必须做直接核对。不得通过改写旧 DA 元数据制造“DD”，也不能为了通过哈希校验改旧回执。

正式训练保留 PR108 设置：CIFAR-10、ddpmpp/ECT、q256、batch128、batch_gpu16、world_size1、RAdam lr0.0001、betas0.9/0.999、eps1e-8、weight_decay0、dropout0.2、augment0、xflip=False、FP16+AMP、TF32=False、global_gap_scale1.0、sigmoid、k8、b1、c0、double10000、ema_beta0.9993。其余参数及软件版本以原有效命令和 runtime 为准，不使用新版默认值覆盖。

原 runtime 参考：Python3.11.13、PyTorch2.6.0+cu124、CUDA12.4、cuDNN90100、NumPy2.1.2、SciPy1.16.1。不可另换 BF16、全程 FP32 或新 optimizer 配置。

## 4. 最小正确性检查与记录

只做必要检查，不扩成新的全仓库审计工程：

在固定 seed50 源状态验证旧 DA 短路径在新代码中不被改变；验证新 DD 的实际第一次 loss 采用 target1.0/denominator1.1；验证 DD 连续短跑与分段 resume 的计算状态一致；验证 E_512 仅初始化一次。

至少核对 DD branch-init 与旧 DA branch-init 的 online、optimizer、scaler、RNG/sampler、E_KEEP、E_512 及进度相同。首次共同 batch 的 target 输入应一致，分母允许按干预不同。中途参数已分叉后不能要求两边的 target 网络输出一致。

沿用已存在的 batch/t/base_r 配对记录，旧 DA 的已记录后缀字段可作直接比对。epsilon/dropout 未有旧指纹时，说明覆盖范围，不伪称全程逐位随机性审计通过。不要为了补旧 telemetry 重跑全部 DA。

保留原 attempts、successful updates、skip、scaler、update norm 等记录；便宜的切换后短期诊断可用现有日志，不能为新日志额外做大规模 forward/backward。每个 seed 保存 512、640、768、896、1024 完整状态；中间点本轮不做 FID，不挑最佳 checkpoint。

正常 AMP skip 按旧语义继续，不补样本、不对齐成功更新数、不清空 scaler。非有限 loss/状态按原科学失败规则终止；保留失败、不改精度修复、不换 seed。外部掉线或抢占可从同一有效状态恢复，不混称科学重训。

## 5. 冻结终点评估

所有新评估都在 1024 kimg，沿用冻结 evaluator、FP32/NFE1、相同参考统计、feature extractor、sampling 参数和 metric seed20260730。

每个 DD 五个槽：
- E_512：B0、B1、B2；
- E_KEEP：B0；
- ONLINE：B0。

B0 generation seeds0–49999；B1为50000–99999；B2为100000–149999。
共计划 80 个新评估，与旧 DA 的80个槽配对。主分析只用 E_512 三个 block；其他两个读出为次级描述。每次生成复用同一 feature 计算 FID/KID；KID 不作为独立复现。

DA 原指标直接复用。不得把旧 E_KEEP 和新 E_512 对比，或把不同生成 block、不同精度/参考库拼成配对。科学失败产生相应 NO_ENDPOINT 槽，技术评估失败先解决，不把技术缺失当质量差。

不增加 NFE2、训练中 FID、生成 block 扫描或基于结果的额外评估。

## 6. 预定统计

在 DD 正式训练前冻结统计脚本、seed名单、生成块、终点和边界。已有 DA 已观察，明确记录这一事实。

对每个训练 seed，先在三个 generation blocks 内求 DA−DD 的平均 log-FID 差：

\[
d_s=\frac{1}{3}\sum_{b=0}^{2}
\left[\log F_{DA,s,b}-\log F_{DD,s,b}\right].
\]

唯一主要对比是跨训练 seed 的平均 d_s。负值表示恢复默认 A 更好，正值表示保持 D 更好。

报告配对均值、seed SD、双侧95% t区间、双侧p值、几何FID比及区间、相对变化、逐seed值和方向计数。training seed 是统计单位，不能把三个block当成三个独立seed。所有有限极端FID保留。

实用等效作为预设次级判断，使用乘法3%界限：

\[
-\log(1.03)<\mathbb E[d_s]<\log(1.03).
\]

用 TOST 或等价的90% t区间报告；预先固定边界，不根据结果更换。方向性判断和实用等效判断分别输出，小但可精确检测的差异可以同时满足方向性和实用等效。不显著本身不证明相同。

若 DD 全部完成，主配对 n=16。若发生科学失败，报告全部16个seed的成败，同时把有限FID推断限定于完整配对集合；不得填充、删坏seed后仍写n=16，也不得对全16队列宣称质量等效。

AA 辅助分析只在 AA、DA、DD 都有效的共同集合中计算。不能拿16-seed DA/DD均值与14-seed AA均值拼接差异。不据辅助结果追加新主假设，不要求新主分析沿用PR108的四臂S4。

本轮DA−DD回答“从已形成的D历史出发，512后恢复A是否改善终点质量”。不能证明早期唯一关键窗口、最佳切换点、history×current interaction、optimizer唯一中介或跨数值精度普遍性；没有AD和多切点设计。DD在±3%内接近DA，只说明这项恢复操作未带来超过预设尺度的差异，不否定PR108的DA−AA历史效应。

## 7. 算力与调度

PR108 的16条DA后缀进程占用合计约33.65 GPUh；155个新增评估合计23.716 GPUh，据此80个新评估约12.24 GPUh。它们是规划参照，不是对新任务耗时的保证。

本轮目标约50–65 A100 GPUh，新增进程预算上限80 GPUh，且不得超过用户实际剩余额度。预算包括短工程审计、DD训练、评估和导出；租赁闲置/传输另列，及时释放闲置卡。

优先用8张可用卡，每卡一个world_size1任务，16个seed分两波。已有16张卡可一波，不改变单seed batch或world size；评估复用释放的卡。仅按可用资源和吞吐调度，不按FID调度。

首波吞吐仅用于预测完成成本，不查看终点质量选择是否继续。预测超过本轮上限时暂停派发并报告具体原因；不静默缩小seed矩阵，也不把剩余额度自动转给其他实验。

## 8. 交付

提交独立 PR，保留旧 PR108 不变。交付新的协议、实际命令、源码/运行身份、源状态绑定、16-seed成败表、80个新槽与80个旧DA槽、逐seed统计、绝对FID/KID、成本和归档索引。

结果图以16-seed DA/DD配对为主；AA/DA/DD辅助图使用三路径共同集合并显式标n。分别输出数学/浮点/LR等价核查报告与质量结果，不以工程PASS代替科学阳性。

提供离线复算入口、必要针对性测试与简短中文结果总结。大模型权重、生成图像、features和私有连接保留在现成归档，不进入公开Git。不要编造已存在的CLI；实现新入口后通过实际help、dry-run和短检查，再生成最终运行命令并跑完固定DD矩阵。


## 本轮用户覆盖

使用16卡一波，每seed world_size1。用户开卡并提供可用额度后才启动GPU进程。
