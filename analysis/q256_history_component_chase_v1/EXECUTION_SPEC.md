# ECT 第一组：历史成分拆分＋共同 A 续训

**交给 Codex 的实施与执行任务书｜2026-09-08**

实验 ID：`q256_history_component_chase_v1`。

本文件是新实验设计，不是训练已经启动、代码已经实现或资产已经在本机验证的回执。正式运行前，将本文件及实际命令、资产引用、实现 commit 一并保存为执行版本。用户本轮授权范围是实施并运行第一组；不自动启动第二组、M2、SWAP、重启或新参数扫描。

## 0. 执行目标

在 CIFAR-10、q256 的既有 FP16+AMP ECT 训练条件下，补跑 seeds 50–65 的 C/D 前期历史，并在 512 kimg 后统一为 A，训练到 1024 kimg。复用 PR103/M1 同 seed 的原始正常 K_A/K_B 作为 AA/BA 对照。

核心问题：仅改变前期 target endpoint、仅改变前期 denominator，以及两者的交互，是否在共同后续 A 下留下终点差异。

这是一项在已观察 A/B 队列上补充成分臂的、预设分析的新实验。不能称为全新独立确认，也不改变 PR101、PR103 的原始统计结论。它定位训练干预来源，不识别唯一状态存储位置或自然中介比例。

## 1. 固定范围、队列与预算

### 1.1 科学矩阵

| 路径 | 前期 target scale | 前期 denominator scale | 前期预算 | 后期两 scale | 后期终点 | 来源 |
|---|---:|---:|---:|---|---:|---|
| AA | 1.0 | 1.0 | 0–512 kimg | 1.0 / 1.0 | 1024 kimg | 复用 M1 原始 K_A |
| BA | 1.1 | 1.1 | 0–512 kimg | 1.0 / 1.0 | 1024 kimg | 复用 M1 原始 K_B |
| CA | 1.1 | 1.0 | 0–512 kimg | 1.0 / 1.0 | 1024 kimg | 新跑 C 前缀＋A 后缀 |
| DA | 1.0 | 1.1 | 0–512 kimg | 1.0 / 1.0 | 1024 kimg | 新跑 D 前缀＋A 后缀 |

固定 training seeds：50、51、52、53、54、55、56、57、58、59、60、61、62、63、64、65。

新增共 32 条完整训练轨迹。每条有一个前缀进程和一个后缀进程，共 64 个训练阶段；不能将这些阶段当成 64 个独立样本。0 kimg 是相同预训练模型加载后的 ECT 起点，不是随机从头训练扩散模型。

不从 A@512/B@512 直接构造 CA/DA；C/D 必须拥有各自的 0–512 历史。不得移植、平均或清空 optimizer 来制造 C/D 历史。

### 1.2 旧对照与已知缺失

只复用 PR103/M1 `original`、branch 为 K_A/K_B 的原始运行及评估。不得混入 R、L、X 或 `POST_FAILURE_FP32_SENSITIVITY`，不得用 PR101 的相似终点代替 M1 的 E_512 读出。

已公布的旧 K 缺失：seed58/K_A、seed58/K_B、seed65/K_A。原始 K_A 成功 14/16，K_B 成功 15/16。旧 K 有效终点总数 29，五种评估槽共 145 个有效值，另有 15 个原缺失槽。

即使新增 C/D 全部成功，主完整四格集合最多也只有 14 个 seed：50–57、59–64。50–65 仍全部在本轮队列中；58、65 也跑 C/D，报告所有启动臂的成败和可用结果，不按旧或新 FID 更换 seed。

### 1.3 预算

第一组目标占用约 170–200 A100 GPUh，规划上限 200 GPUh；这不是租赁账单保证。总项目剩余 200–300 GPUh 不全部自动授权给额外实验。

基于 PR107 的实测，512-kimg 后缀约 1.94–2.72 GPUh；线性外推本轮 32 条完整轨迹，训练约 124–174 GPUh，参考均值约 142 GPUh。160 项新增终点评估参考约 8 GPUh，另留初始化检查、导出、转存和速度波动余量。实际前缀吞吐需要实测，不能把外推当实测。

用同样矩阵的已完成 attempt 耗时估计总成本，不能根据 FID 决定是否省略慢/差的 seed。若预计超过 200 GPUh，先停止派发新的阶段，保留当前有效状态并报告成本与剩余队列；不要静默缩小科学矩阵。不得为了节省预算取消已知差 seed 的评估。预算暂停标记为 `BUDGET_PAUSED`，整组保持 `INCOMPLETE_BUDGET`；它不是科学失败，不将已完成子集当作完整预定矩阵结案。

不设置按 FID 的提前停止、增 seed 或换臂。没有终点的科学失败不自动补训。

## 2. 必读代码与已有资产

仓库：`hjjjs4vbmv-netizen/consistency-gap-optimization`。

已核对的参考身份：

- PR101：`890a85a8ef4d9effb48f653111a70b5f15b249de`。
- PR103：`7af37a6460bd423ad22e0189bfddc9b713ba0c79`。
- 原冻结评估器：`d6aba02fb88e9db0993623895eb2228ed717d810`。

这些是可追踪参考；正式数值运行代码及 runtime 以归档的实际命令/receipt 为准，不能把报告 commit 自动当成所有历史任务的运行 commit。

优先阅读：

- `analysis/q256_terminal_history_n30_matpool_v1/protocol.json`
- `analysis/q256_terminal_history_n30_matpool_v1/run_node.py`
- `analysis/q256_optimizer_restart_ema_rebuild_v1/M1_PROTOCOL.md`
- `analysis/q256_optimizer_restart_ema_rebuild_v1/results/README.md`
- `analysis/q256_optimizer_restart_ema_rebuild_v1/results/endpoint_metrics.json`
- `scripts/run_m1_training_slot.py`
- `scripts/run_m1_evaluation_job.py`
- `scripts/build_m1_evaluation_slots.py`
- `analysis/q256_optimizer_restart_ema_rebuild_v1/export_readout.py`
- `training/m1.py`
- `training/schedule_switch.py`
- `training/ct_training_loop.py`
- `training/loss.py`
- `training/reproducibility.py`
- `ct_train.py`

复用原项目已有的服务器连接和归档；不要把私有路径、凭据写入公开 PR。先读取归档索引，不要反复要求用户填写已保存的信息。

需要定位的资产：原 CIFAR training archive、原 transfer checkpoint、PR101 seeds50–65 初始配置/initial receipt/前缀 telemetry、A/B@512 完整源状态、M1 K 原始运行与终点/评估引用、原评估器与 reference statistics/feature extractor。

只做一次必要核对：资产可读、身份正确、初始化和实际数值配置匹配、所需旧对照可定位。已有可信 hash 回执可复用，不对全部历史大文件反复扫描。

若跨时期旧对照无法兼容，输出具体不匹配字段并停止正式训练；不得自动改为新四臂、重跑旧 K、替换 seed 或用近似对照拼接。工程小检查通过只能支持检查覆盖的范围，不能写成全程逐位重现。

## 3. 数值设置：继承基线，不升级软件与算法

| 设置 | 固定值或规则 |
|---|---|
| 数据 | CIFAR-10 32×32；原训练 archive 与原预处理 |
| 模型 | ddpmpp，ECT，unconditional；相同 transfer checkpoint 与全参数/buffer 加载策略 |
| batch | global 128；batch_gpu 16；world_size 1；保持原梯度累积方式 |
| optimizer | RAdam，lr 0.0001，betas (0.9,0.999)，eps 1e-8，weight_decay 0 |
| loss | q256_target_weight_v1；q=256，k=8，b=1，c=0；sigmoid |
| 显式尺度 | global_gap_scale 始终 1.0；只使用两个 explicit factors |
| 噪声分布 | mean −1.1，std 2.0 |
| schedule | double=10000；其余状态与更新时钟继承原配置 |
| dropout/增强 | dropout 0.2，augment 0，xflip false |
| 精度 | 网络 FP16＋AMP；TF32 false；loss scaling 配置与实际原命令一致 |
| EMA | beta=0.9993；不改为 PowerEMA 或其他 half-life/ramp-up |
| runtime | 原部署重建 runtime：Python 3.11.13、PyTorch 2.6.0+cu124、CUDA runtime 12.4、NumPy 2.1.2、SciPy 1.16.1；同时核对归档实际版本 |
| 其他执行 | bench false，cache true，workers 1；使用原确定性配置 |
| 训练过程质量评估 | metrics none；不新增训练中的 FID、NFE2 或自适应采样 |

如表与归档实际有效命令有冲突，必须先定位原因。不能默认采用新版本 CLI 的缺省值。

已存在、可作为实现参考的公共参数组合为：

`--cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 --factorial-protocol=q256_target_weight_v1 -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --fp16=True --tf32=False --ls=1.0 --enable_amp=True --bench=False --cache=True --workers=1 --metrics=none --duration=1.024`

这是公共数值参数片段，不是完整启动命令。Codex 必须生成完整、经过实际 `--help` 和 dry-run 核对的命令，再执行。

C 前缀追加 target 1.1 / denominator 1.0；D 前缀追加 target 1.0 / denominator 1.1。所有后缀为 target 1.0 / denominator 1.0。继续调用现有 `compute_target_weight_times` 与 `apply_global_gap_scale`，保留实际裁剪、浮点运算和 loss reduction。不要用简单 `loss *= 1/1.1` 代替 D，不要再乘一次 global scale。

不继承旧 256-kimg factorial verifier 的终点数和“只准 warm-up skip／各臂 skip 次数必须相同”判定。本轮继承 PR101/M1 的实际训练语义：正常 AMP skip 记录并保留，参数更新为零；attempt、样本游标与 EMA 按旧规则推进；RAdam internal step 按实际更新推进。不补样本、不对齐各臂 successful-step 数、不共同清空 GradScaler。

原 M1/PR101 路径不允许把 non-finite loss 作为 seed38 recovery-v2 的 managed overflow 自动放行。本轮不添加该例外。非有限 loss、参数、EMA、moment 或无效 denominator 按原科学终止规则处理。

## 4. 前缀、切换和后缀的精确定义

### 4.1 前缀

从该 seed 的相同 ECT 初始化过程启动 C/D。完整在线参数、buffers、初始 optimizer、GradScaler、RNG、sampler 和预训练加载方式与旧 A/B 匹配；只改变前期的指定 factor。

前缀仍声明 `--duration=1.024`，总计划为 1024 kimg。在 attempt 4000 已处理完成、global nimg=512000 时暂停并保存完整源状态。采用已有 planned-pause 机制，为新实验显式增加允许范围；不要冒用旧 PR101/M1 protocol ID。

不要把前缀改成 `--duration=0.512` 来凑半程。暂停只停止进程，不重新定义训练计划、LR 或 schedule 时钟。

### 4.2 512-kimg 边界

顺序固定：

1. 完整恢复各自 C@512 或 D@512。
2. 校验 source history，设置下一次 loss 实际采用 A 的两个 factors。
3. 完整保留该轨迹的模型、buffers、RAdam moments/internal step、GradScaler、RNG、sampler、global progress 与旧 EMA。
4. 从该轨迹的在线模型创建一次 E_512，参数和 buffers 都复制；这一步不消耗训练 RNG。
5. 保存独立的 `branch-init@512` 完整状态，记录 history=C/D、current=A、optimizer_operation=keep、optimizer_reset_count=0、ema_512_init_count=1。
6. 开始 attempt 4001；继续至 attempt 8000、global nimg=1024000。

前缀 source@512 与后缀 branch-init@512 可以在不同目录同名，必须区分，不能互相覆盖。旧实现的一些 `factorial` metadata 保留 source history；不要为了让 validator 通过而重写成虚假历史 A。应分别记录 source history 和实际 current loss factors，并检查 attempt4001 的实际 loss factors。

恢复 640/768/896 checkpoint 或 branch-init 时，加载已有 E_512，不再初始化；同样不能再次切换或重启 optimizer。

### 4.3 保存与 EMA 更新

保存：前缀 source@512、后缀 branch-init@512、640、768、896、1024 完整状态；沿用原已有原子 latest-state 技术恢复机制。

完整状态至少包含 net/buffers、optimizer、GradScaler、旧 EMA、E_512（后缀）、RNG/sampler、loss/schedule、attempt/successful counters、运行配置和新实验身份。

E_KEEP 与 E_512 使用原 M1 每 attempted update 的参数 EMA 更新，包括合法 AMP skip。EMA buffers 遵循原实现：E_KEEP 保留原 buffers，E_512 初始化时复制在线 buffers，不擅自改成每步平均 buffers。E_512 只用于读出，不进入 target 或优化。

## 5. 最小正确性检查与随机输入配对

不构建新一轮大规模审计系统。保留三类低成本检查：

**配置/资产检查。** 每个节点确认 runtime、数据、transfer 可用；复用原 receipt，比较各 seed 新前缀初始化与旧 initial receipt。保存科学字段的差异表，允许 differences 只有设计 factors、路径、实验标识与纯观测字段。

**短程实现检查。** 用固定工程案例核对新 wrapper 的 A 不干预路径与原基线数值语义；核对 C/D 两 factor 正确进入 loss。再从同一@512源执行一段未中断和同段断点恢复，检查一次性 E_512 初始化、未重置 optimizer 和首段训练等价。记录代表性首个实际 forward 的输入dtype，不能仅凭CLI声称精度匹配。只需覆盖新增接口，不能为检查另跑完整基线。工程 stop-attempt 必须在工程模式单独隔离，不能放宽正式 roster 或预算。

**正式配对检查。** 沿用既有 telemetry。按相同 attempt 对齐 C/D 与旧 A/B 的 batch、labels、t、base_r，以及可用的噪声/RNG 字段。在512边界检查 RNG/sampler 与旧源流一致；不能把 A 的 RNG/scaler 强行灌给其他历史来“修复配对”。suffix 共用 A，因此随机输入应按同一后续流推进。

注意：前缀 C/B 的 target time 和 target noisy input 本来就与 A/D 不同，不能要求它们相等。需要相同的是干预前的基础随机输入；预测值和梯度也不要求相同。不得新增会消耗训练 RNG 的绘图、诊断、采样或DataLoader操作。

已有日志字段足够时不追加大张量 dumps。若需要补录小量 epsilon/dropout RNG 指纹，在观测层实现，先验证不改变状态；已有旧字段没有覆盖时诚实声明核对范围，不伪称所有历史随机量均已逐步核验。

若短程检查发现实质运行不兼容，先修正接口错误；不得靠降低 LR、改变 clipping、FP32 repair、放松数值失败规则来通过。

## 6. 16 卡调度

一张卡一个独立 worker；每个 worker 独占一张 GPU，world_size=1。不要用 `--nproc_per_node=16` 训练一个模型，也不要把 batch 放大为2048。

16个逻辑槽对应 seeds50–65。每张卡串行跑本 seed 的两个完整路径，交错 C/D 顺序：

| 槽位 | seed | 先执行 | 再执行 |
|---|---:|---|---|
| S01 | 50 | CA | DA |
| S02 | 51 | DA | CA |
| S03 | 52 | CA | DA |
| S04 | 53 | DA | CA |
| S05 | 54 | CA | DA |
| S06 | 55 | DA | CA |
| S07 | 56 | CA | DA |
| S08 | 57 | DA | CA |
| S09 | 58 | CA | DA |
| S10 | 59 | DA | CA |
| S11 | 60 | CA | DA |
| S12 | 61 | DA | CA |
| S13 | 62 | CA | DA |
| S14 | 63 | DA | CA |
| S15 | 64 | CA | DA |
| S16 | 65 | DA | CA |

每条路径内部依次是 prefix 0–512、full-state switch、suffix 512–1024。一个分支科学失败后，记录终态，继续同 seed 的另一条预设分支；不能因一臂失败取消另一臂。

多节点允许 2×8、4×4 或16个单卡实例；node-local GPU index 由 Codex 实际探测并写入 worker manifest。不同进程使用独立输出目录、日志与 rendezvous endpoint，避免端口冲突。不要使用机器数量、启动顺序或 arm 名称作为额外训练 seed。

若只有8张或更少，保持全部32条路径和既定顺序，在可用 worker 上排队，不能改变 world_size、batch、实验 seed 或科学矩阵。尽量让同 seed 的 C/D 使用同卡，并交错臂顺序，减少新C/D间硬件与时间混淆；这不能消除旧A/B跨时期复用的全部限制。

评估可以在该节点训练完成后的空闲卡执行，不和同卡训练抢资源。按固定清单完成全部终点评估；中途不向训练调度器反馈 FID。

## 7. 终点评估：160个新增计划槽

只评估1024终点，不增加640/768/896质量扫点，不增加NFE2。

| 新轨迹读出 | generation blocks | 数量 |
|---|---|---:|
| 32条轨迹的 E_512 | B0/B1/B2 | 96 |
| 32条轨迹的 E_KEEP | B0 | 32 |
| 32条轨迹的 ONLINE | B0 | 32 |
| 合计 | | 160 |

B0：sample seeds 0–49999；B1：50000–99999；B2：100000–149999。每个block独立FID50k/KID50k。三个block不能合成FID150k，也不能当成三个训练seed。

全部采用原冻结 evaluator `d6aba02fb88e9db0993623895eb2228ed717d810`、FP32、NFE1、相同sampler参数和特征/真实参考/预处理。metric RNG=20260730；FID和KID复用同一批生成特征。

复用旧K的145个有效评估和15个缺失标记。新旧合起来是320个计划槽；若新32条全完成，最多305个有效槽。不因某条旧K缺失跳过本轮其余有效轨迹的预设评估。

训练科学失败标记为NO_ENDPOINT，不插补FID，不把文件传输或评估技术错误当科学失败。技术问题修复后用同一checkpoint、同一generation block重做技术失败任务；保留旧错误记录。成功评估不按指标好坏重抽block。

导出与评估必须和训练分离，不改变训练RNG。大checkpoint、图像、features进入既有持久化存储，不上传Git。空间不足时先转存并确认所需资产到位，不未经授权删除旧实验文件。

## 8. 固定统计分析

### 8.1 主读出与共同集合

唯一主质量读出：1024-kimg、E_512、NFE1、三个block的log-FID均值。

对路径 X，定义：

\[
Y_{s,X}=\frac{1}{3}\sum_{b=0}^{2}\log\!\left(\operatorname{FID50k}_{s,X,b}\right).
\]

主集合 \(\mathcal S_4\) 为固定16seed中四条路径均科学完成，且四条E_512三个block均有有效评估的seed。技术未决先报告未完成，不能归类成科学失败后选择性结案。

所有主成分对比和联合对比统一使用同一 \(\mathcal S_4\)。若C/D均完成，n最多14；不能借用PR101的26对均值或M1不同集合均值来计算本轮分解。报告所有启动臂的成败、配对资格和该条件集合限制。

### 8.2 三项主要对比

\[
H_{T,s}=Y_{s,CA}-Y_{s,AA},
\]

\[
H_{W,s}=Y_{s,DA}-Y_{s,AA},
\]

\[
I_s=Y_{s,BA}-Y_{s,CA}-Y_{s,DA}+Y_{s,AA}.
\]

H_T 是基准denominator下的target历史效应；H_W 是基准target下的denominator历史效应，不混称为对另一个factor取平均的主效应。I 是终点outcome的交互对比。

对上述三项使用 seed 层双侧配对t检验，三项p值统一Holm校正，family alpha=0.05。每项报告n、均值、样本SD、普通95% t区间、原始p、Holm调整p、逐seed方向。普通95%区间明确标成逐项名义区间，不称为Holm simultaneous CI。n<2或零方差等退化情况返回明确状态，不输出虚假p值。

对H_T/H_W可报告几何FID比与其区间：对mean及CI端点取exp；下降百分数按exp(mean)−1。I若转指数，必须称为ratio-of-ratios，不能写成单臂“质量提高百分比”。

同时在完全相同集合上报告：

\[
H_{J,s}=Y_{s,BA}-Y_{s,AA}=H_{T,s}+H_{W,s}+I_s.
\]

检查逐seed及均值的代数闭合；不能把代数闭合本身当成机制验证。

由于A/B结果已经观察，本轮是带历史对照复用的预设成分研究。Holm校正处理本轮三项多重性，并不会把队列改造成独立确认性样本。

### 8.3 次级、描述性结果

完整报告CA−DA、BA−CA、BA−DA；均值、逐seed值和名义区间，标明探索/描述性，不用次级显著项替代主结果。

ONLINE/E_KEEP/E_512横向比较仅采用共同B0。KID在seed内平均原始block数值再做差，绝不取log；它与FID共享features，不称为独立复现。

报告每条路径原始FID/KID及跨seed绝对质量，保留很大但有限的FID。原始FID算术均值、SD与log-FID对比区分。补充每个直接对比的最大可用配对集合可以，但与主共同集合分开，不拼接交互。

可增加廉价的leave-one-seed-out敏感性，但不据此删除seed、重定义主要集合或挑最有利结果。

### 8.4 结论约束

- H_T为负且Holm通过：支持在本条件下target-only历史降低终点FID。
- H_W为负且Holm通过：支持denominator-only历史降低终点FID。
- I直接有证据：才讨论相应终点交互。只有BA显著、CA/DA不显著，不足以证明交互。
- 一个成分显著、另一个不显著，不等于二者差异显著；相对大小要看直接对比。
- BA−CA或BA−DA不显著不证明“保留全部收益”；本轮不设未授权的等效性结论。
- 区间跨零不证明无效；给出当前仍容许的范围。
- 不写optimizer唯一存储、自然中介比例、跨数据集普遍规律。

## 9. 工程实施要求

建立新分支，例如 `experiment/q256-history-component-chase-v1`；在其下新增 `analysis/q256_history_component_chase_v1/` 的协议、worker、队列/汇总和测试。名称可按仓库规范调整，但必须可独立复现。

必须处理的已有接口限制：

1. `scripts/run_m1_training_slot.py` 是旧四臂K/R启动器，本轮不能直接整脚本运行。
2. 已读取版本的 `training/schedule_switch.py` 中source arms只有A/B，M1 branches只有K_A/K_B/R_A/R_B。需要为新协议显式支持C/D source到A；不能把C source伪标成B以绕过校验。
3. `training/m1.py` 的once-only EMA操作可抽出或复用通用函数；新metadata应使用新experiment ID，不覆盖旧M1。
4. `validate_planned_pause` 对协议、seed和4000attempt有白名单。为新协议增加隔离条目；不能粗暴删除所有旧校验。
5. 所有新逻辑默认关闭，旧协议行为和已存结果不改变。不要merge多个结果PR只为本轮开跑，更不要修改旧统计。
6. 不直接运行旧q256整臂validator，它针对历史256-kimg终点/skip规则；新增测试只检验本轮正确契约。

最小测试覆盖：C/D factor映射；4000/4001切换无off-by-one；keep optimizer；E_512仅初始化一次；断点续训加载已有EMA；不继承错误warm-up skip门槛；只导入original K；缺失不插补；同集合分解；三个block先log后平均；Holm三项家庭；重复job不执行。

实现干净、幂等的worker状态机即可，不新增复杂seal/decode平台、多层hash门禁或大规模状态诊断。日志保留关键状态与终止原因，训练结束后统一出结果。不能为了让测试通过篡改真实结果。

## 10. 交付清单和运行结束行为

正式启动前交付/保存：执行协议、实际runtime与commit、旧K来源清单、32条训练队列、160条评估槽清单、16worker映射、短程检查结果、完整启动命令。无需再让用户确认已经固定的seed、臂、预算或读出。

运行后提交：

- 中文 `RESULTS_ZH.md`：先给成分结论、绝对质量、完整配对n、三项对比与区间；再给限制。
- 全16seed×4路径的成败/缺失表，旧与新增来源标识。
- 320个新旧计划质量槽的tidy表；旧指标保持原值。
- 逐seed三个主要对比、联合对比、探索性条件对比，以及统计JSON。
- 训练attempt、successful/skip、实际GPU进程时数和科学/技术失败记录。
- 一张终点四路径配对图、一张三项成分/交互区间图；不绘制不存在的中间质量曲线。
- 对应代码、最小测试、归档引用；不含checkpoint、图像、features、凭据或私有连接信息的增量PR。

所有计时必须区分本次新增进程GPUh、历史继承elapsed、租赁闲置和传输时间。不能把恢复checkpoint里的累计elapsed当成本轮新增成本。

本轮结案后停止。无论结果好坏，不自动增加seed、添加NFE2、转FP32补失败、跑第二组或恢复M2/SWAP。

## 核对依据

本任务书依据以下已读取的项目文件；正式执行仍需要在实际训练节点核对资产。

- PR103 / `analysis/q256_optimizer_restart_ema_rebuild_v1/M1_PROTOCOL.md`：M1数值设置、随机流、EMA与评估矩阵。
- PR103 / `analysis/q256_optimizer_restart_ema_rebuild_v1/results/README.md`：原K缺失与读出。
- PR101 / `analysis/q256_terminal_history_n30_matpool_v1/run_node.py`：前缀duration、attempt4000暂停与命令结构。
- PR103 / `training/loss.py`：A/B/C/D实际factor映射、global scale约束和realized time计算。
- PR103 / `training/m1.py`：E_512初始化/更新与optimizer操作。
- PR103 / `training/schedule_switch.py`、`training/ct_training_loop.py`：接口白名单、状态保存与运行语义。
- PR107 / `analysis/state_interventions/TRAINING.md`：GPUh参考。
- PyTorch官方Reproducibility文档：相同seed不保证跨版本/平台复现，须匹配实际运行配置。
- statsmodels官方multipletests文档：Holm step-down多重性校正。
