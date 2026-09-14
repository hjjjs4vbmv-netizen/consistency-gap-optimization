# F12@q256：未经响应筛选的新队列

来源：本任务用户提供的九节方案（2026-09-13），以及开跑前的方案审查。
状态：前瞻性预注册；本文、protocol.py、analysis.py 和训练实现须在首条 F12 轨迹（包括工程轨迹）开始前提交。
适用范围：CIFAR-10、EDM DDPM++、q=256、1024 kimg attempted images；仅此配置下的独立重复。

## 固定科学矩阵

初始种子固定为 401–412；每种子 AA、DA、A_startup_down5、D_startup_up5 四路径，共 48 条。
不得基于终点、训练损失、范数方向或其他响应纳入、排除、替换种子。
预备种子仅 413–416。初始 48 路径全部达到记录在案的终态、且全部 NFE1 完成状态确定后，
只查看完成性元数据：若四臂完整种子数 <9，一次性追加全部四个预备种子；否则不启动预备队列。
追加后合并全部合格种子分析。仍不足 9 则报告 underpowered，并且不再追加。

训练配置完整继承 analysis/startup_window_experiments_v1/trajectory_template.json；
该模板来自此前 q256 原始初始化 receipt，SHA 为
0c6b6bbb197699296d758f2bfbea505455c6a1bd045ed404f8411210df949f14。
新种子产生自己的 canonical 初始化，绝不复用历史 AA/DA 终点或历史队列初始化哈希。
仅替换种子、输出路径及预定臂的倍率；q 固定 256。
单卡单轨迹，world_size=1；batch=128，microbatch=16，8 次梯度累积；FP16 AMP。
原生 RAdam，LR=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0。
dropout=0.2，augment=0，xflip=False，TF32=False；完整模板字段逐项比较。
stage-zero 固定映射保持原实现，训练共 8000 attempted updates / 1,024,000 attempted images。
AMP skip 不推进 successful-update 或 RAdam 时钟，但消耗 attempted-image 预算。
同种子各臂必须匹配初始参数、优化器、scaler、RNG、sampler 状态。

DA 和 D_startup_up5 的 0–512 kimg 前缀用 denominator_gap_scale=1.1，target_gap_scale=1，global_gap_scale=1。
A_startup_down5 第 1–5 次真正 optimizer.step 使用 LR*(1/1.1)；D_startup_up5 使用 LR*1.1。
第 6 次成功更新恢复 LR=1e-4；不改变 RAdam 自身状态转移。
512000 attempted images 处保留本分支参数、优化器、scaler、RNG、数据位置并切回 A。
E_512 从本分支 online model 初始化一次；之后沿用原实现 EMA beta=0.9993。
终点为 1024000 attempted images。保存 512 前缀、512 分支初始化及 1024 完整可恢复状态。

## 主终点与分析

主终点 E_512 / FP32 / NFE1 FID。每臂固定 B0=0–49999、B1=50000–99999、B2=100000–149999，
各生成 50000 张；同样的生成 seed block 用于每个种子、每个臂。metric_seed=20260730。
Y 是三块 log(FID50k) 的平均，不能合为 FID150k，也不能把 block 当成训练重复。
主对比 H=Y_DA−Y_AA，种子级配对 t，双侧 alpha=.05，名义 95% 和 90% t CI。
次要族 M=Y_A_down−Y_AA，C=Y_D_up−Y_DA，S=(C−M)/2，双侧配对 t；Holm 族大小固定 3。
对 S 必须先构造逐种子复合差，保留 M/C 协方差；exp(S) 不是某个模型 FID 改善比例。
M/C/S 的普通区间标为名义区间，不称为 Holm 同时区间。
主分析仅含四臂全部训练和三块 NFE1 有效的种子；同步报告分臂科学失败、技术失败和评估失败。
H/M/C 的可用配对敏感性分析仅描述，S 仍需四臂。KID、NFE2 不检验。
NFE2 仍用同一冻结评估器、FP32、同三块生成 seeds，mid_t=[0.821]；FID/KID共用该次生成特征。
固定评估器 commit d6aba02fb88e9db0993623895eb2228ed717d810；固定资产哈希写在 protocol.py 与部署回执。

## 开跑前修订说明

本修订不改变 H、四臂、12 种子和训练配置。
1. 功效按用户给出的 C14 SE=.0203、n=14 得到 SD=.0759556，以正态配对差、双侧非中心 t 核算：
   n=12 原效应 -.0916 功效 .9665；2/3 log 效应功效 .7179；80% 最小可检出 FID 降幅 6.53%。
   代入该 SD 的 95% CI 半宽 .04826 log；不得预先保证未显著就能排除 C14 量级。
2. 等价界限明确为对称 ±log(1.03)，即约 ±.0295588 log，沿用此前对称等价口径。
   alpha=.05 TOST，90% CI 严格位于该带内则 practical equivalence。
   n=12、真实效应零、上述 SD 和正态假设下等价检验功效约 1.9%；不能把不显著解释为等价。
3. 范数和是否按预期变化是附录观察，不是排除标准。工程失败限于配置/实际 LR/成功计数错误、
   无效状态、错误输入或可证明的执行/文件错误。正确实现产生的数值发散归科学失败，完整保留。
4. 显著性与等价性分别报告：H 95% CI 上界<0 为负方向检出，下界>0 为正方向检出，否则未解析零点；
   同时独立报告 TOST。允许“统计显著且实际很小”。均未通过则未定。最终 n<9 优先标 underpowered。
   H 复现不自动复现 M/C；未复现不证明 C14 原结果由选择或侥幸造成。
   不根据本轮结果改变 F8 的报告完整性，也不由两组显著性不同推断 spacing 交互。
5. 仅按终态元数据进行排程与追加。终点指标由自动流水线保存，不向操作日志或进度摘要输出。
   完整完成性审计、预备队列决定及资产绑定封存后一次性解盲；不靠隐藏 seed ID 充当防期中查看措施。
6. 上一轮实测外推，48 条训练加 NFE1/KID 约 232–234 GPUh；NFE2、工程、预备队列另计。
   24 卡全部就绪时采用两路径一 lane 的并行排程；新增机器只填尚未分配 lane，不改科学矩阵。

## 工程与恢复

按用户“尽快开始、不要过多验证”的启动指令，工程 seed=99401，不入统计样本，无质量评估。四臂各 32 attempted updates，需达到至少 16 successful updates。
记录每次 LR、RAdam 时钟、AMP skip、前 16 次更新范数，以及一次成功更新 5 后暂停并恢复到 32 attempts 的完整状态比较（其余窗口边界由已有 CPU 测试覆盖）。
原有 CPU 精确 RAdam/no-op/倍率/skip/恢复测试继续运行，并覆盖新 F12 q256 边界和身份。
正式数据每条路径用独立 manifest、固定 source hash、GPU UUID/主机许可及文件锁，杜绝重复派发。
技术中断只允许从自己经哈希验证的 checkpoint 恢复，保留原始日志与所有中断回执；不能选择性从头重跑。
主机和路径是追加的部署元数据；每个 lane 在启动前冻结，一旦启动不改派，不静默覆盖文件。
全程按实际 process GPUh 记账，训练硬超时 12h/路径、评估 2h/block、工程 1h/进程。
不得以训练损失或终点质量早停。保存所有工程错误与科学失败。

首条 F12 轨迹启动前的 Git commit 及 source manifest 哈希写入各主机 source_freeze.json；
正式训练另需一份四臂工程 PASS 回执与每个 seed 的 canonical 初始化哈希。
