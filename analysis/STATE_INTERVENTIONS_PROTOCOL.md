# 最后两组实验：M2与双向optimizer互换（最终执行稿）

日期：2026-09-07。依据用户提供的《反馈.md》，不增加训练矩阵。

本稿包含共享设置、两项独立方案、主次比较、失败处理及预算。任务清单和检查脚本见同名交接包。


---

# 共享执行规则与预算

Package ID：`final_state_interventions_q256_v2`

## 1. 冻结对象与证据身份

固定集合 S = {55,56,59,60,61,62}，不替换、不追加。这是已观察的原M1四臂均完成条件子集，不是独立的新验证队列，也不是原16/30-seed无条件总体。

两项分别在独立命名空间发布，不能因M2一侧失败就删除其有效SWAP数据，反之亦然。每个主比较的完整集合依据所需单元决定。所有不完整seed仍保留在原始结果和成败表中。

不把原M1的8/16早期R_A失败率与本轮六个经过完成条件选择的晚期失败率比较并声称总体风险下降。不得把新结果加入原M1的主检验。

同集合基线：反馈报告这六个seed的K/E_512历史差约-0.081253 log-FID（约7.8%）。该值仅作核对线索；分析时必须从这六个原指标重算。若新有效集合缩小，则在新集合重算所有组成。不能借用14对K的8.7%或PR101 26对的8.6%作本轮分母或组件贡献。

## 2. 配置：除指定操作外不变

| 项目 | 执行值 |
|---|---|
| 数据、预处理、网络 | 原M1 CIFAR-10、ddpmpp、ECT，与原始源一致 |
| pairing / weighting | q=256，原sigmoid，inverse-gap |
| A早期历史 | 0–512 target/denominator scale = (1.0,1.0) |
| B早期历史 | 0–512 target/denominator scale = (1.1,1.1) |
| 后续策略 | 512以后所有新增及复用分支均为A；全局schedule stage继续，不从头启动curriculum |
| RAdam | lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；保留原参数组其余选项 |
| batch / world size | global=128，microbatch=16，world_size=1 |
| dropout / augment / xflip | 0.2 / 0 / false |
| 训练精度 | 原M1 FP16+AMP，TF32=false；其余CUDA/cuDNN确定性、sanitization、managed-overflow、非有限终止语义不变 |
| runtime | 复用原已核验M1运行环境；参考Python3.11.13、PyTorch2.6.0+cu124、CUDA12.4、NumPy2.1.2、SciPy1.16.1；绑定实际代码及本地改动 |
| EMA | beta=0.9993；按原attempt更新，包括原规则接受的AMP skip；buffer处理不另改 |
| 评估 | FP32、NFE1、FID50k，KID共用同批features |
| evaluator | 原已审计M1 evaluator；参考commit d6aba02fb88e9db0993623895eb2228ed717d810，现场核实clean执行树、detector和real reference |
| metric RNG | 20260730 |
| B0/B1/B2 | 0–49999 / 50000–99999 / 100000–149999（样本seed，含两端） |

这些数值承接旧M2/M1设置，不是重新测得的“最新默认配置”。现场如与原实际命令有差异，先查原记录；不静默改变数值语义然后声称旧对照匹配。

## 3. 三种不同的时钟

- 全局attempt：512=4000，768=6000，1024=8000，样本进度按attempt推进。
- RAdam per-parameter step n：来自实际成功更新；不能用4000或6000代替。PR106案例约3989/3990，不把此范围写死成所有源的验收值。
- EMA：沿用原attempt更新时钟；GradScaler接受的skip不另补样本。

M2清空(m,v,n)，保留全局计数。SWAP复制donor的(m,v,n)，保留接收方全局计数。metadata单独记录接收方原successful count、donor step摘要、操作后的新增实际updates，不能伪造计数关系。

## 4. 原始文件与复用

需要绑定：
- 12份原始A/B@512，供SWAP接收方和donor；每份在两种用途下仍保持只读。
- 24份原始K_A/K_B/R_A/R_B@768，其中12份K用于M2训练，全部24份用于补评。
- 同六个seed原始K/R@1024的120个指标槽；其中K的60个是SWAP自身对照。

禁止使用FP32 repair、允许额外clean-skip的恢复试跑、近似时刻或其他seed代替源。源缺失先查原备份；需新增长程重放恢复源时，暂停相关队列并另报成本，不自行扩大本包。

源ID与路径、模型和optimizer结构、全局进度、既有provenance应明确。沿用既有文件绑定，不新增多层hash门禁；也不能跳过已有源完整性与身份检查。

K@512专用branch-init若被用来简化SWAP初始化，必须先证明它与原接收方@512在模型、buffer、E_KEEP、optimizer、执行状态上一致，E_512与原K初始化一致；donor仍是未干预的同seed原历史optimizer。

## 5. 统一读出与任务数

1024每个新分支5槽：E_512 B0/B1/B2，ONLINE B0，E_KEEP B0。
768补评每个原K/R分支4槽：E_512 B0/B1/B2，ONLINE B0；不加E_KEEP@768。

| 任务 | 计划槽 |
|---|---:|
| M2新L@1024 | 60 |
| 原K/R@768补评 | 96 |
| SWAP新X@1024 | 60 |
| 新增评估计划合计 | 216 |
| 原K/R@1024共享复用 | 120 |
| 本包引用的唯一计划质量槽总计 | 336 |

216槽全部生成时共10,800,000样本。FID和KID不是两份生成成本。若已有严格相同checkpoint/readout/block的768指标，可在看本轮结果前核验并指定为精确复用；不按数值好坏决定是否重评。216是按无额外缓存的预算上限，不承诺216个有效结果。

## 6. 汇总规则

ell_s(X,T)=(1/3) sum_{b=0}^2 ln FID(s,X@T,E_512,b)。
- 所有正式源指标由三个block平均log计算，不是先平均FID再取log，不是FID150k。
- 完整集合上的seed级差值计算mean、SD(ddof=1)、双侧95%Student-t区间；df=n-1。
- 几何比为exp(mean contrast)。绝对质量表可另列各block FID及三block算术均值/SD，表头必须区分。
- KID用原始差，不取log；同一生成特征上的KID是辅助描述，不是独立验证。
- ONLINE与E_KEEP横向对照只用共同B0。
- n<2时只列值，标INSUFFICIENT_COMPLETE_PAIRS；n=0不填零；SD=0显式报告退化情况，不制造显著性。
- 95%区间只在条件seed过程和t模型假设下解释。选择了已观察完成样本，不能把名义覆盖当作对原始总体或独立验证的保证。

M2：唯一主要时间对比为same endpoint；same chase及其他读出只解释，不能代替主对比。
SWAP：两个接收方来源对比是两个并列的预设估计量，均完整报告名义95%区间；不设合并确认性PASS、不在两个比较中选一个“显著的”作为成功。组合模式先作条件性描述，不宣称联合95%覆盖或总体机制确证。

上述SWAP统计地位与完整集合定义是为执行补全的本次设计约定，不是反馈中的既有实验结果。

## 7. 失败与缺失

训练终态：COMPLETE / NUMERICAL_FAILURE / TECHNICAL_UNRESOLVED / NOT_RUN_RESOURCE_LIMIT。
评估槽：PASS / NO_ENDPOINT / TECHNICAL_UNRESOLVED / EVAL_NONFINITE / NOT_RUN_RESOURCE_LIMIT。

- 大但有限、指标计算有效的FID必须保留，没有“FID过大不算”的筛选。
- 科学数值失败停止并留故障进度及日志；无终点的5槽登记NO_ENDPOINT，不插补大FID、不替换seed。
- 评估自身非有限先核验源、导出、精度和计算；不能静默删block或视为普通大FID。若排除技术原因后仍科学不可评估，则记录原始证据并排除对应连续量，报告缺失。技术未决期间不封最终分析。
- 同一主要读出缺一个block，不平均其余两个。与主要读出无关的技术任务不删除有效seed，但本包交付须披露其未决状态。
- 单项分析只能使用该项需要的共同有效集合；不能减不同seed集合的均值。
- 每个recipient可用的单侧对比在附表保留，不能因不能组成完整四格而消失。
- 纯技术中断最多两次同配置精确恢复，保留原错误和尝试目录。不得借此救援可重现的科学失败。
- 不添加FP32长修复、LR/clip/scale下限、额外skip容忍、其他optimizer操作或新生成block。
- M1结果保持原状；repair不并入原对照。新分支大幅更新或科学溢出，不是换方案、换seed的理由。

## 8. 最小工程检查（不新增长期对照）

1. M2 restore@768不开重启的短程结果与原K相匹配。
2. SWAP own-state export/reload恢复原K数值轨迹；两种receiver的路径都检查。
3. reset/transplant仅修改指定optimizer state，且只在首次分支初始化执行一次；所有来源只读且深拷贝。
4. save-resume检查：同一数值环境连续32 attempts与16+保存恢复+16一致；若原规则更早科学失败，比对故障前轨迹/故障位置。不要为满足“必须跑完32步”而改训练。
5. 同一receiver配对中，batch/t/noise/dropout等外生输入按绝对attempt对齐。移植和观测代码不消耗正式训练RNG；必要时先做副本观测后恢复原RNG，不用新随机tape“修正”某条轨迹。
6. A/B源本身的RNG/sampler身份按原恢复语义核对；scaler允许各自不同，不因此剔seed。精度与其他设置不因case变化。
7. evaluator导出实际ONLINE/E_KEEP/E_512正确，旧指标为同版本兼容链。指标生成前完成上述操作测试；不以小FID作为通过条件。

新增入口需要工程人员实现/接入；协议和清单本身不声称服务器runner已部署通过。

## 9. 有限动态记录

不再重复PR106的32格诊断。两项新增分支只复用现有字段记录前64次attempt：loss、gradient、update finite/norm，scaler前后、skip、optimizer step/moment norm、EMA时钟及输入对齐。

每个source可用同一输入的own-state/K副本做一次参考更新，记录norm ratio和cosine；不得写回正式轨迹或多消耗正式RNG。若参考或干预step跳过，记录不可比，不拿不同attempt的首次成功更新强行作比。该工作包括在2–4卡时检查预算内，不作为质量准入门槛，不拓展成新长训练。

## 10. 预算、调度与收尾

| 成本 | 单位规划假设 | 合计A100 GPUh |
|---|---|---:|
| M2新训练 | 12×(0.85–1.00) | 10.2–12.0 |
| SWAP新训练 | 12×(1.7–1.9) | 20.4–22.8 |
| 216个评估槽 | 216×(0.05–0.10) | 10.8–21.6 |
| 两模块入口/有限恢复检查 | 不另跑长对照 | 2–4 |
| raw合计 | | 43.4–60.4 |
| 含约20%余量 | | 52.1–72.5 |
| 资源申请 | 中心65，上限75 | 55–75 |

以上来自反馈中的历史吞吐核算，不是本次实测。占用GPUh、总墙钟、租赁闲置分别记录。两卡、源与runtime就绪后的35–48小时是规划窗口而非承诺；若实际测速预计越过75，报告阻塞/成本并等待决定，不能擅自选结果缩小矩阵。

默认排程：GPU0负责55/59/61，GPU1负责56/60/62。每个seed四条L_A/L_B/X_A_from_B/X_B_from_A，先跑可用入口，不按结果改变顺序；同一GPU同时只运行一个训练或生成任务。两项独立准备，既定评估按空闲队列执行，CPU并行导出和写作。

所有新增操作定义、seed、读出、时间对比在开始前同时固定；任何一项结果不能改变另一项。现在即可按旧证据写作。新任务完成或明确登记科学失败后只回填结果、确定结论边界，不再自动补seed/时刻/操作。若预算阻断，按未完成报告，不以“停止扩展”等同于“全部成功”。

## 11. 最终交付

- 36个源状态绑定项、旧120指标绑定、实际运行版本及参数名称映射。
- 24条训练终态、216个新增计划评估槽终态、120个复用槽。
- M2的same-endpoint主对比、same-chase辅助对比、A/B分别损伤。
- SWAP的两个receiver来源对比、同集合匹配代价、原始FID/全部缺失。
- 原M1/repair/本轮结果分目录；单独公开scalar表和读出比较。
- 动态记录、算力占用账本、两项result PR或一个清楚分章节的结果PR。
- 回传至已获授权的ECT新目录（例如在已确认archive parent下新建本包ID），核实实际归档成功；不覆盖M1/repair，不在公共仓库存凭据和敏感连接信息。


---

# M2最终执行协议：同一种RAdam重启的两个时刻

Experiment ID：`m2_restart_timing_q256_v2`
适用：必须与`00_SHARED_RULES.md`共同执行。v2明确取代旧M2未来设计中的E_768预留，原M1历史记录不改。

## 1. 问题和规模

在M1全部四臂完成的六个seed中，同样的完整RAdam重启发生在512或768 kimg时，后续质量损伤是否不同？这种时间差是否因A/B早期spacing历史不同而改变？

固定S={55,56,59,60,61,62}。新L共12条，从原K@768运行至1024；156个新增计划评估槽，120个共享旧槽。

不预定“晚期更安全”“optimizer不再重要”或“B总是更耐受”。两个时刻只能回答这两个起点的差别，不能识别完整关键期或普遍阶段界限。

## 2. 完整轨迹矩阵

| 分支 | 0–512 | 512–768 | 768–1024 | 本轮 |
|---|---|---|---|---|
| K_A | A | A，保留optimizer | A，保留optimizer | 复用 |
| K_B | B | A，保留optimizer | A，保留optimizer | 复用 |
| R_A | A | 512完整重启后A | A，不再重启 | 复用 |
| R_B | B | 512完整重启后A | A，不再重启 | 复用 |
| L_A | 复用K_A | 复用K_A至768 | 768完整重启后A | 新6条 |
| L_B | 复用K_B | 复用K_B至768 | 768完整重启后A | 新6条 |

源只允许原K@768，不使用R@768或FP32 repair建立L。原R@768只补评。

## 3. 一次性操作

- 完整加载K@768：cur_nimg=768000，attempted_iteration=6000。
- 验证源模型/optimizer参数组/EMA/GradScaler/RNG/sampler/loss stage/全局计数。
- 与M1的R一致：清空全部per-parameter RAdam state，fresh初始化(m,v,n)；不改参数组或LR。
- 其他状态全部继承，包括E_KEEP与E_512，后续策略仍A。
- 保存L-init@768，metadata记录reset_applied_count=1，然后attempt6001–8000。
- @896和@1024保存完整状态；中断恢复绝不再次reset。
- 使用独立M2初始化入口，不把M1 SWITCH_NIMG/SWITCH_ATTEMPT全局改成768/6000，不伪造原512的curriculum切换记录。

EMA：L从K@768恢复已有E_KEEP/E_512，继续原beta=0.9993和attempt时钟。无E_768，不再次initialize E_512。

## 4. 评估

| 对象 | 状态数 | 各状态任务 | 数量 |
|---|---:|---|---:|
| L_A/L_B@1024 | 12 | E_512 B0/B1/B2 + ONLINE B0 + E_KEEP B0 | 60新 |
| 原K_A/K_B/R_A/R_B@768 | 24 | E_512 B0/B1/B2 + ONLINE B0 | 96新 |
| 原K/R@1024 | 24 | 原5槽 | 120复用 |

新槽预算156。每槽FID50k/KID共特征，全部NFE1、FP32。不同读出横比只用B0。导出逻辑不能重建任何平均状态。

## 5. 唯一主要时间估计量

ell_s(X,T) = mean_{b=0,1,2} ln FID(s,X@T,E_512,b)。

对h∈{A,B}：

 d_h^512(T) = ell_s(R_h,T) - ell_s(K_h,T)
 d_h^768(1024) = ell_s(L_h,1024) - ell_s(K_h,1024)

正值表示重启后质量更差。

历史损伤差：
 J_s^512(T) = d_B^512(T) - d_A^512(T)
 J_s^768(1024) = d_B^768(1024) - d_A^768(1024)

**主要时间对比，固定共同1024终点：**

 Theta_SE,s = J_s^768(1024) - J_s^512(1024)

同终点K代数抵消：
 Theta_SE,s = [ell_s(L_B,1024)-ell_s(L_A,1024)]
              - [ell_s(R_B,1024)-ell_s(R_A,1024)]。

仍必须保留K并报告d_A/d_B，才能知道差来自谁。

分析集合S_M2：固定六个seed中，L_A/L_B及相应原K/R@1024的E_512三blocks均有效者。所有组成在S_M2上重算。报告n_M2/6、mean/SD/95%双侧t CI、各seed值、所有成败。仅有新L失败时也不能继续套原六seed的R/K均值。

- CI完全>0：所选共同终点下，历史损伤差存在正向时间变化。
- CI完全<0：负向时间变化。
- 跨0：INCONCLUSIVE，不是等效。
- n<2或技术未决：依共享规则，不作完成性方向裁决。

不增加最小百分比、所有seed同向、LOSO全部通过或事后单侧标准；不改原M1 verdict。本对比是完成条件子集上的预设新时间估计，不是独立确认。

## 6. 辅助时间口径：同样重启后训练256 kimg

 Theta_SC,s = J_s^768(1024) - J_s^512(768)。

用L@1024和K/R@768即可得到，不新增长训练。完整集合按这些所需单元建立，报告与S_M2的重合及差别；两口径直接并列或比较方向时使用它们的交集重算，不能把样本集合变化说成时间机制。

主要SE匹配最终预算，但含恢复长度差512 vs 256。
辅助SC匹配恢复长度256，但绝对训练阶段、输入片段、读出终点与EMA年龄仍不同。

两者互补，不声称一起排除了所有时间因素。方向不一致应如实解释，不追加时刻，不用辅助显著替换主要不确定。

## 7. 解释必须同时看两边

 T_A,SE = d_A^768(1024) - d_A^512(1024)
 T_B,SE = d_B^768(1024) - d_B^512(1024)
 Theta_SE = T_B,SE - T_A,SE。

相同chase亦给两侧对应值。J/Theta为正不自动意味着“依赖减弱”：也可能是B损伤变多。

用ONLINE共同B0重复SE/SC辅助比较，E_KEEP只做1024共同B0对照。若效应主要体现在EMA，结论限定该读出，不泛化成在线学习普遍阶段性。不用“一个显著、另一个不显著”当作两者显著不同，也不因不一致扩大矩阵。

## 8. 动态记录与交付

按共享规则：新L前64attempt的既有更新/精度/skip字段；同状态单步K参考不写回正式状态。无额外多操作诊断。

输出至少含：
- m2_seed_contrasts：每seed原K/R/L三block、d_A/d_B、J^512/J^768、Theta_SE、Theta_SC及可用集合；
- 名义区间、读出辅助对照、全部六seed成败；
- 大FID不删除，科学失败留空并解释；
- 12条终态、156计划新槽、120复用绑定、实际成本；
- 本轮结论只适用于所选六seed条件状态及指定完整RAdam重启，不能证明记忆转移、唯一中介或总体失败率下降。


---

# SWAP最终执行协议：固定接收方的完整optimizer来源比较

Experiment ID：`optimizer_donor_swap_q256_v1`
适用：必须与`00_SHARED_RULES.md`共同执行。独立于M2；不将任一项完成状态用来筛另一项。

## 1. 问题、样本与命名

固定六seed：55、56、59、60、61、62。它们是原M1四臂完成条件子集；新操作的条件已预定，但旧质量已知，不称独立确认。

问题：固定接收方历史形成的模型及其余状态，把接收方自己的optimizer替换成另一历史正常积累的optimizer，会怎样改变共同续训的质量？信息来源是否有利，还是自身配对更兼容？

X_A_from_B：接收方A、donor B。
X_B_from_A：接收方B、donor A。
“第一部分是receiver，from后面是donor”，不要和spacing continuation的AB/BA重名。

## 2. 来源矩阵与训练

| 接收方状态 | A来源optimizer | B来源optimizer |
|---|---|---|
| A | 原K_A（复用） | X_A_from_B（新6条） |
| B | X_B_from_A（新6条） | 原K_B（复用） |

每条从原始@512到1024，attempt4001–8000；所有后续策略为A。共12条×512=6144 kimg，48000 attempts。原K与R不重训。

## 3. 只互换什么

完整per-parameter RAdam O=(m,v,n)：
- exp_avg（一阶moment）
- exp_avg_sq（二阶moment）
- step（内部计数）

不能只换moments却保留receiver step；不能把step强制4000，也不能清0。PR106源约3989/3990只是示例，实际逐源记录。step差异属于此次完整来源操作的一部分，结论不称“纯moments效应”。

不替换：receiver模型参数/buffers、receiver EMA、receiver GradScaler、receiver参数组及超参数、LR、loss/schedule、global counts、RNG、sampler、数据进度。

受影响的是optimizer来源；donor不贡献旧EMA或模型。首次操作后后续optimizer正常更新，不在每一步持续装入donor，也不在resume时重复移植。

## 4. 初始化和映射规则

1. 完整恢复原始、未修复的receiver A/B@512，核对512000/4000。
2. 按M1相同规则建立receiver的共同A continuation。E_KEEP为receiver原EMA；E_512从receiver online完整副本初始化一次，必须与对应原K初始化一致。
3. 独立只读加载同seed donor @512。两个方向始终用原始donor，不用已互换分支当另一方向来源。
4. 按原模型参数名称、形状、参数dtype及optimizer parameter group成员建立映射。不只依赖state_dict整数索引恰好同序；必要时重构原optimizer参数对象映射。要求映射一一对应、无遗漏/重复。
5. 将donor的每参数(m,v,n)深拷贝到receiver相应参数；保留receiver param_groups。moment张量不改变数值与dtype，设备迁移满足原optimizer语义；step容器dtype/device正确。遇到未知state keys或不能解释的缺项先停查，不默默清空。
6. 记录两份源step摘要、修改的state keys、参数映射及未修改状态的短程一致性。全局successful counter继续receiver语义，donor step单独记，不能强迫二者相等。
7. 保存X-init@512，transplant_applied_count=1，donor/receiver身份和来源清楚；开始attempt4001。
8. @640/768/896/1024保存完整状态；resume恢复X当前全部状态，不再次移植，也不重建E_512。

初始化分支即使采用已审计K-init@512简化实现，必须证明其与上述规范等价；不静默替换原始来源。

## 5. no-op与短程观察

- 用同样映射/装载代码把receiver自己的optimizer导出再装回；应与原K短程更新、EMA、scaler、RNG数值一致。A/B两路径都覆盖。
- 移植是训练干预，不要求短程更新小、方向接近K、无科学溢出才允许入队；不能事后选择表现温和的donor或seed。
- 操作错误先修，真实科学失败按原规则保留，不加精度或学习率救援。
- 按共享规则保留前64attempt遥测和有限首步参考，不重复PR106诊断。

## 6. 新评估矩阵

| 对象 | 数量 | 每条评估 | 新槽 |
|---|---:|---|---:|
| X_A_from_B / X_B_from_A@1024 | 12 | E_512 B0/B1/B2，ONLINE B0，E_KEEP B0 | 60 |
| 原K_A / K_B@1024 | 12 | 原相同5槽 | 0（60共享复用） |

不新增中间FID。R的已有结果可作为明确不同操作的参考，不能拿R当成缺失的K或swap来源对照。

## 7. 两个来源对比，分别报告

定义Y_s,h,d = mean_b ln FID(receiver h, donor d, E_512,b)，统一1024。

 Y_AA = ell(K_A,1024)
 Y_AB = ell(X_A_from_B,1024)
 Y_BA = ell(X_B_from_A,1024)
 Y_BB = ell(K_B,1024)

两个预设主要估计量：

 D_A,s = Y_AB - Y_AA
 D_B,s = Y_BB - Y_BA

两者都统一为“B来源相对A来源”的方向：负数表示B来源更有利。特别注意D_B不是X_B_from_A-K_B，那是相反号。

核心分析集合S_swap：固定六seed中这四格E_512三个blocks全部有效者。两项在相同集合各报mean/SD/95%双侧名义t区间、n/6和逐seed值。不合并成单一swap effect或单一确认性PASS；来源模式作为条件性估计解释，不声明联合95%覆盖或无条件机制。

如果某些seed只能形成一个receiver的对比，另列该receiver所有有效配对的描述性结果和集合；不静默删除，也不把不同集合的均值用于匹配/交互模式判断。S_swap不要求M2 L成功。

## 8. 预设的结果解释

| 模式（同集合估计） | 允许的解释 |
|---|---|
| D_A<0且D_B<0 | 点估计在两个receiver均倾向B来源有利；若区间宽，明确未解析，不仅凭符号称确证 |
| X_A_from_B比K_A差，且X_B_from_A比K_B差，即D_A>0且D_B<0 | 自身来源配对更有利的模式，可能反映模型—optimizer兼容性 |
| 只有一个receiver支持B来源有利 | 来源作用依赖receiver；不能说B普遍更好 |
| 双方cross都改善，即D_A<0且D_B>0 | 跨来源可能有利；不等于B来源优势 |
| 混合/区间宽/失效 | 如实报告局部敏感性或未解析，不能挑一个receiver替代全结果 |

所有文字区分点估计模式与不确定性；不能用“这边显著那边不显著”建立receiver差异。

匹配代价辅助量：
 P_match,s = [(Y_AB-Y_AA)+(Y_BA-Y_BB)]/2 = (D_A-D_B)/2。

正值只描述两种cross损伤平均偏大；不能只凭该均值正就说每侧都受损。它不是原始历史收益的中介比例。

原同集合正常历史差：T_s=Y_BB-Y_AA。报告可解释的操作效应及原始质量，不把D或P除以PR101的8.6%或14-pair的8.7%。不声称唯一存储、全历史自然中介或全部收益来源。

## 9. 读出、故障与交付

ONLINE/E_KEEP/E_512在共同B0作辅助对照；E_512三block的主估计不由其他读出替换。

大有限FID纳入；训练科学失败留下缺失与进度，不赋极大FID、不替换seed、不用repair补足四格。所有原始成败与修复身份严格分开。

交付：12条训练终态、60计划评估槽、两个receiver各自来源对比及同集合模式、匹配代价、各读出绝对FID/KID、原始/移植state摘要和step差异、算力记录。

此实验可识别所定义state transplant在固定receiver中的后续作用；来源与兼容性可以有交互。这不自动证明“原来8.6%通过optimizer中介”，也不承诺互换必然比重启或M2更有新颖性。
