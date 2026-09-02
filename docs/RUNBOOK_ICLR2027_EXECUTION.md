# RUNBOOK — ICLR 2027 投稿执行手册 v1.3

冻结日期：2026-09-01（v1.0）。v1.1 修订：2026-09-02（PR #97 对账）。v1.2 修订：2026-09-02（三席审查仲裁，见"v1.2 增补"）。**v1.3 修订：2026-09-02（route-3 草案外部 request-changes 审查的 8 条阻塞项落地，见"v1.3 增补"）**。本手册假定执行者没有任何背景知识与判断权：每一步给出前置条件、输入、动作、输出、二元验收标准、失败动作。**任何验收失败 = 立即停止该 Phase 并上报，禁止自行变通。**

仓库：`hjjjs4vbmv-netizen/consistency-gap-optimization`。所有日期为 2026 年，UTC+8，且均为**内部提前截止时间**——ICLR 官方截止为 AOE，内部日期刻意提前以留缓冲，两者不一致不是错误。

---

## v1.1 对账（2026-09-02，依据 PR #97）

PR #97 表明 fresh 实验已在 v1.0 计划之外执行完毕，且**范围超出** v1.0 的 batch-1 设计——原留给 rebuttal 期的 stage-2 臂（AB/BB）也一并跑完：

| v1.0 计划 | 实际执行（PR #97） |
|---|---|
| 6 seeds（31–36），batch 1 只跑 2 臂（AA/BA） | 12 seeds（31–42），**全 4 臂 factorial**（AA/AB/BA/BB） |
| 协议 `q256_fresh_confirmatory_v1`（Phase P，待建） | 协议 `q256_fresh_crossed_switch_n12_matpool_v1`，SHA256 `df7f584c…f167b1`，结果观测前冻结（fail-closed validator 锁定 `quality_metrics_observed_before_amendment=false`） |
| primary = H_A 单侧配对 Wilcoxon | primary = H（双 continuation 平均）冻结五类别裁决（two-sided CI + TOST 等价带） |
| stage 2（AB/BB）预注册留 rebuttal | **已执行，rebuttal 预留耗尽** |
| n 底线 5 | seed38/AB 终端数值失败 → 结果观测前作者修正为 n=11 complete-case；informative missingness 记为限制 |

**冻结裁决：INCONCLUSIVE。** H 均值 −0.0756（发现 cohort 为 ≈−0.65，塌缩约 9 倍）；95% CI [−0.1553, +0.0042] 覆盖 0；TOST 90% CI 不在 ±log(1.03) 带内；8/11 负向；exact sign-flip p=0.042（非裁决输入）。结构诊断（PR #97 分支 `structure_diagnostic_v1/`）：corr(H,Q)=+0.598，pooled H 约 2/3 由 seed31/40 的切换点既有优势持续贡献；强资格反转 2/3（seed35 为首个明确反例，幅度 ~−0.09 对 discovery 的 −0.33~−0.95）。

**对本手册的机械后果**（正文各 Phase 已就地标注）：
- Phase P / T / E：SUPERSEDED——被 PR #97 的已执行协议与证据束取代，禁止再启动 v1.0 版 fresh 训练。
- Phase S：判定已由执行协议的冻结类别完成（INCONCLUSIVE）；**禁止用 v1.0 计划的 H_A 单侧 Wilcoxon 事后重判**——那是换检验挑显著。
- Rebuttal 预注册动作：作废（已执行）；新 rebuttal 弹药是 9/22 决策会议题（见文末）。
- G5 分支进入"INCONCLUSIVE"行（失败分支总表新增），方向未反转，TMLR 非自动默认。

---

## v1.2 增补（2026-09-02，三席审查仲裁：魔鬼审稿人 / 统计方法学家 / 执行现实主义者，三席一致判"大修"后的定稿）

### 日历铁律（与实验无关，优先级最高）

- **9/18 摘要无条件提交**（可撤回，成本近零）。不开会、不讨论、机械执行。
- **全文从即日起按路线 1（ICLR 全披露）默认撰写**；9/22 决策会只做"撤/不撤 + 转轨"开关，不做"要不要开始写"的决定。9/22 后距全文截止仅 3 天，只够开关不够改稿。

### 第 0 层硬 gate（9/22 会前必须完成，~20 GPU·h，所有路线共用）

| ID | 任务 | 规格 | 门控作用 |
|---|---|---|---|
| G0-a（并入 Z 系为 Z8） | 跨 cohort H~Q 结构分析 | 0 GPU。分 cohort 着色、层内分别拟合、**禁止 pooled 拟合线**；discovery 侧 corr 做 bootstrap/剔点稳定性 | 反号不稳 → route-3 剂量设计仍成立，但一切"调制"叙事从论文与协议中删除 |
| G0-b（升级 Z2） | 噪声地板 | **3 checkpoints（fresh 最好/中位/最差 FID，规则写死）× 5 generation seeds = 15 评估** + 1–2 段同 seed 重训分解训练噪声；δ 统一为显式 σ_e 单位；误差传播 σ(H)=σ_e、σ(Q)=√2σ_e；合格反转 = Q>max(0.5, 2√2σ_e) 且 H<−2σ_e，同报零假设下期望假反转数 | 决定：反转计数能否进正文；route-3 的 SD 假设与 n |
| G0-c | EMA 半衰期解析计算 | 0 GPU。残余权重 w=2^(−512/t_half)；w<0.05 → L1c 与移植实验的 EMA 臂降级为 sanity check | L1c / 移植 EMA 臂的 go/no-go |

**L1a（=Z8）/L1b（=Z2 升级版）未完成前，禁止冻结任何 route-3 预注册。**

### 三路线最小获胜包（决策会输入，取代 v1.1 议题描述中的实验部分）

- **路线 1（ICLR）**：第 0 层 + 既有 Z1–Z7，**零新增训练**。会前书面承认：reject 侧概率主导且无 rebuttal 实验弹药。
- **路线 2（TMLR）**：同上 + L1c 子集（4 个跨 Q 谱系 seeds，~32 评估 job）。把"预注册盲评复制 + 噪声地板 + 结构诊断"的方法学严谨性当卖点。当前 EV 最高路线。
- **路线 3（顺延，目标 ICML 2027 / TMLR，不等 ICLR 2028）**：按预注册协议草案 `analysis/q256_fresh_confirmatory_v2_draft/`（v2，经 2026-09-02 外部 request-changes 审查修订）执行，要点：
  - **两臂设计**（0→512 A/B history；512→1024 均 current-A），**n=24 全新 seeds**（数据仲裁：fresh corr(C₁,C₂)=0.965，双 continuation 冗余；同预算下两臂 n=24 的功效 **0.947** > 四臂 n=16 的 **0.783**，assurance **0.823** vs **0.697**——v1.3 修正：非中心 t 精确复算，出自 `planning_calculations.py`，v1.2 的 0.96/0.82/0.83 为正态近似，作废）
  - 设计命名 = **matched two-history continuation under a fixed current-A policy**（不得再称 crossed；identification limits 写进协议）
  - Primary = 无调整 H_A 单侧配对 t（α=0.05），MDE(80%)=**0.0591**；**verdict 五类判定表全文写进协议**（含边角：单侧 p<0.05 但双侧 95% CI 跨零 → INCONCLUSIVE）
  - Gatekept 链：H_A → 剂量趋势（**within-seed 线性对照** L_s=−0.5·Y₁.₀+0.5·Y₁.₂，8 seeds 配对 t 双侧 0.05，功效 0.899；v1.2 的随机截距回归模型作废）→（可选模块）H@2048 延训
  - n=12 处 **binding futility-only 期中** + 盲式 SD 重估（SD>0.15 → 扩至上限 28）；**公式全部写死，type-I 误差经 committed Monte Carlo 验证**（无条件 0.0495 / 压力场景 0.0490，≤0.055 冻结线；`type_I_error_simulation.py`）
  - 缺失规则：**预排序替换池 primary + completion-conditioned estimand 明文**（估计的是双臂可完成 seed 总体的均值，非无条件效应）；**硬规则：>4 替换 → EXECUTION_FAILED，不裁决**；B 臂失稳计数 = 并列报告的次要终点
  - 移植实验 15 RE（2×2 reset 对照 + sham parity gate + G0-c go/no-go），exploratory + 符号预测记分卡
  - 预算 = **三命名包**（v1.3，取代一切区间表述）：MINIMAL **127 RE / 250 job / ~400 h**；WITH_HORIZON_SUBSET **149 / 300 / ~460**；WITH_HORIZON_FULL **171 / 320 / ~500**

### v1.2 处决名单（从 v1 实验方案中删除/封存）

| 项 | 处置 | 理由（席位收敛） |
|---|---|---|
| "H~Q 调制"作为确证目标 / Q-ANCOVA | 删除。Q 仅 descriptive；协议写死 "No inferential claim about H–Q moderation at any achievable n"；仅保留 EIV 校正持久性回归 (ρ, α) 做估计 | post-treatment + 均值回归伪影 + 解盲阈值 + 空 cell + 交互功效不可行（三席全中） |
| C（gap-independent regime crossed） | 封存为下届预注册 stage-2 | null 不可解释、正结果不改路线 |
| q128 crossed（现在） | 封存为下届 rebuttal 弹药 | 主效应未确认前泛化顺序颠倒；修复"弹药烧光"教训 |
| g=1.05 剂量点 | 删除 | 任何可行 n 下无功效 |
| "magnitude bracket" 措辞 | 从 W4 规则中删除；n<6 轴只报 per-seed 点值 + 预注册"不做任何区间或方向陈述" | 括号是方向声明的遮羞布 |
| L1c 全量（88 job） | 砍到 4-seed 子集，且 G0-c 前置 | 44 GPU·h 换 descriptive 附录段，负杠杆 |
| E（切换时点）、ImageNet crossed | 维持 v1 判决：不做 | — |

### W4 追加（v1.2）

- `bracket` / `幅度括号` 不得出现在正文措辞规则中；
- sign-flip p 出现处必须写检验全名（"exact permutation test on the mean under sign flips"）+ 代码指针，且带 "not a decision input" 限定（数值 86/2048 已独立枚举复核无误，问题只在标签）。

---

## v1.3 增补（2026-09-02，route-3 草案外部 request-changes 审查落地）

审查结论：方向正确但当前是设计讨论稿而非可冻结协议，8 条阻塞项。全部已落地到 `analysis/q256_fresh_confirmatory_v2_draft/`（协议 JSON v2 + PROTOCOL_DRAFT.md + `planning_calculations.py/.json` + `type_I_error_simulation.py/.json` + `protocol_lint.py`）。逐条状态：

1. **PR #96 补录（runbook 级）**：PR #96（B@384 pulse-chase，seeds 19–28，60/60 sealed，**INFORMATIVE_NULL**：384→512 短 B 暴露在共享 A continuation 至 640 kimg 后无可检测质量残留，冻结 3% margin）此前被本 runbook 整体漏掉。机械后果：
   - R1 合并顺序补入 #96（见上）。
   - W3 Scope of Evidence 表新增一行（见 W3）。
   - §6 更名并收编 PR96（见 W1 表）。
   - route-3 设计理由新增时间域边界（协议 `design.temporal_regime_boundary`）："A shorter B exposure from 384 to 512 kimg leaves no detectable quality carryover after a shared A continuation to 640 kimg. The route-3 cohort therefore tests a different temporal regime: a longer 0–512-kimg history followed to 1024 kimg." PR96 从"负结果"变成机制边界：**不是任何短暂 spacing difference 都会留下可检测的未来质量效应**——两实验合并框定 carryover 的暴露时长条件。
2. **设计命名与 claim ceiling**：两臂设计不得称 crossed（只能识别 fixed current-A 下的 H_A；不能估 current-policy 主效应、交互、跨 continuation 可迁移性）。PR95/97 仍是 crossed evidence；route 3 只是对 H_A 的独立确认。已写进协议 `design.naming_rationale` + `identification_limits`。
3. **功效数字统一**：全部改由 committed 脚本输出——两臂 n=24 功效 **0.947**（H_A 点估计；pooled 效应变体 0.938，与审查复算一致）、四臂 n=16 **0.783**、MDE(80%) **0.0591**、assurance **0.823**（flat prior → 后验 N(−0.0776, 0.1129/√11) → 200 阶 Gauss-Hermite 积非中心 t 功效）。v1.2 的 0.96/0.82/0.058 全部作废。剂量对照 n=8 功效 **0.899**（回答"8 seeds 够不够"：在线性投影下 ~90%，MDE 0.078，诚实写入协议）。
4. **verdict 判定表闭合**：五类判定的输入、条件、优先级、边角全部写死进协议 `verdict_decision_table`（n=11 frozen statistics.py 机制的同构放大：strong > equivalence > weak > opposite > inconclusive；equivalence 优先于 weak；单侧 p<0.05 但 hi95≥0 → **INCONCLUSIVE**，即 fresh n=11 实际发生的那一类；22/24 符号计数为 10/11 的等比适配，冻结前钉死）。gatekeeping step1 = verdict ∈ {STRONG, WEAK}。
5. **盲式 SD 重估 alpha 声明闭合**：s12 公式、conditional power 闭式、扩样后检验规则全部写死；新增 committed Monte Carlo（200k reps/场景，seed 20260902）：**无条件 type-I = 0.0495（planning SD）/ 0.0490（压力 σ=0.20，扩样触发 86%）**，均 ≤ 0.055 冻结线。注意：given-analyzed 率 ~0.099 是"以 interim 均值≤0 为条件"的选择效应，是诊断量不是程序 type-I，JSON 已写明防误读。
6. **informative missingness 闭合**：completion-conditioned estimand 明文化（primary 估计的是双臂可完成 seed 总体的 H_A 均值，论文必须同句陈述）；**硬终止规则：>4 替换 → EXECUTION_FAILED，不渲染任何 verdict 类别**；all-started tipping-point 保留无条件读法；B 臂失稳率并列报告。
7. **剂量 estimand 重写**：随机截距回归 + Wald 检验作废 → **within-seed 线性对照** L_s = −0.5·Y_{g=1.0} + 0·Y_{g=1.1} + 0.5·Y_{g=1.2}（终末 log-FID），8 seeds 配对 t 双侧 0.05 + 全枚举 2⁸ sign-flip p 同报（全名）。被对照的是终末 log-FID（相对该 seed 自身两端点）；缺失 = 替换池规则同 primary，池尽 → complete-contrast + tipping-point。
8. **预算三包命名**：MINIMAL 127 RE/250 job/~400 h；WITH_HORIZON_SUBSET 149/300/~460；WITH_HORIZON_FULL 171/320/~500。本 runbook、协议 JSON、PR body 只允许引用包名，禁止裸区间。

### v1.3 论文架构修正（并入 Phase W，部分推翻 9/1 定稿，G5 议题）

- **标题（W2 撤销"定稿不再讨论"）**：现冻结标题 *Mid-Training FID Misranks Training Histories…* 在 PR97 INCONCLUSIVE + PR96 INFORMATIVE_NULL 之后偏强。G5 议题新增：改用问句式、贴证据的替代标题 **"Finite-Horizon Quality Carryover from Pair-Spacing History in Consistency Training"**（方向：把"misranks"这一现象级断言降为"carryover"这一可测性质）。9/22 决定；在决定前 W 阶段所有稿面用占位标题。
- **§6 更名**："Converging evidence" → **"Regime and scale boundaries"**（q128 与 ImageNet 并不直接验证 history carryover，不得作 converging 卖点）。§6 内容 = q128 边界 + ImageNet 规模边界 + **PR96 暴露时长边界**（新增主段落）。
- **贡献结构恢复三条**（推翻 9/1 的两条打包）：(a) intervention/identification structure；(b) state propagation（机制侧）；(c) finite-horizon quality evidence。**不得重新缩成两条**——形式化、状态保存与质量证据不得混装。Introduction 结果段**只写已执行实验**：route-3 尚未执行，只能出现在本 runbook 与 future-work 措辞中，不得进入 §1。
- **W1 表 §1 行与 §6 行、W2、W3 已就地改**（见下）。
- 会前必读清单追加：`analysis/q256_fresh_confirmatory_v2_draft/PROTOCOL_DRAFT.md`（v2）与 `type_I_error_simulation.json`（若议题涉及路线 3）。

---

## 全局规则（适用于所有 Phase）

- R-1 ~~禁止在 Gate G1 通过前启动任何 fresh cohort 训练~~（v1.1：fresh 训练已完成，本条改为：**禁止启动任何新 GPU 实验，除非 9/22 决策会选择路线 3 并完成新预注册**）。
- R-2 ~~禁止在 analysis_plan.json 提交前查看 fresh 1024-kimg 评估数字~~（v1.1：已解盲，本条改为：**禁止对已解盲的 fresh 数据做任何进入推断的新检验；一切新计算只能标 descriptive**）。
- R-3 评估一律使用冻结 evaluator commit `d6aba02fb88e9db0993623895eb2228ed717d810`、FP32、FID50k+KID50k，与 PR #95 完全相同的 `run_evaluation_job.sh` 流程。禁止更改任何评估参数。
- R-4 数据集必须是 canonical CIFAR-10 32x32，SHA256 `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`。启动训练前校验一次，不符即停。
- R-5 每个产物文件生成后立即计算 SHA256 并追加到所属 Phase 的 `evidence/manifest.json`。
- R-6 所有统计只把 training seed 当独立单位。budget/NFE/generation block 都不是新样本。

---

## Phase R — 仓库整合（9/1–9/3）

| 步 | 动作 | 验收 |
|---|---|---|
| R1 | 按顺序 merge：#91 → #93（按其评审修订后）→ #94（rebase 到新 main 后）→ #95（完成其评审 7 点后 rebase）→ **#96（B@384 pulse-chase P2：10-seed 短暴露 practical-equivalence 结果，v1.3 补录——见 v1.3 增补"PR #96 补录"）** → **#97（含已并入的 #98 结构诊断；其评审 M1/M2 已由 commit `6399892`/`fb616c9` 落地，m3–m5 为 minor 可 follow-up）** | **6** 个 PR 全部 MERGED |
| R2 | 每次 merge 后在服务器运行 `pytest tests/ -q`（#95 合入后该目录自动包含 `test_q256_schedule_switch*.py`；此前已含 `test_q256_target_weight_*.py` 共 7 个文件） | 全部 passed，0 failed |
| R3 | 本地工作树 `git fetch origin && git checkout main && git pull`（当前本地落后 154 提交） | `git status` = up to date |

PR #95 评审 7 点的执行清单（每点一个 commit）：
1. REPORT.md headline 段改为 temporal main-effect decomposition（模板见 Phase W 表 T1 行序）。
2. 新增 `per_seed_delayed_reversal.csv`：列 = seed, logfid_A_512, logfid_B_512, gap512, H_A_log_1024, reversed(bool)。数据源 `results/q256_schedule_switch_seed3_7/per_seed_trajectories.csv`。
3. 新增 log-FID 对照表 `contrast_summaries_logfid.csv`，文件头注释行写 `# post-unblind descriptive`。
4. BA ranking 段替换为："BA ranks first in 4/5 seeds at NFE1/1024, the ordering predicted by combining the two main effects."
5. evidence class 段落统一为："pre-result-frozen recovery-cohort protocol; retention-based seed selection; n=5 paired seeds."
6. rebase + 回归测试（R2）。
7. 三个结论分层（history carryover / late refinement / weak interaction）写进 REPORT.md 结论节。

**Gate G0（9/3 24:00）**：R1–R3 全部验收通过。失败分支：顺延但 Phase P 不等待——协议冻结可与 R 并行，只有训练启动依赖 G1 而非 G0。

---

## Phase P — Fresh confirmatory 协议冻结（9/2–9/3）

> **v1.1 状态：SUPERSEDED。** fresh 实验已按协议 `analysis/q256_fresh_crossed_switch_n12_matpool_v1/protocol.json`（PR #97）执行完毕，其自身冻结链（协议 SHA + sealed blind evaluation + fail-closed n=11 validator）替代本 Phase 的全部功能。下文保留仅作历史记录，**禁止再执行**。G1 的语义由执行协议的冻结时间链满足。

### P1 创建协议目录

复制 `analysis/q256_schedule_switch_seed3_7_v3/` 为 `analysis/q256_fresh_confirmatory_v1/`，然后按下表修改 `protocol.json` 字段，其余字段不动：

| 字段 | 新值 |
|---|---|
| `protocol` | `q256_fresh_confirmatory_v1` |
| `seeds` | `[31,32,33,34,35,36]` |
| `origin_arms` | `["A","B"]`，配置沿用 A=(target,denominator)=(1.0,1.0)、B=(1.1,1.1)，训练 commit 沿用 `dcca41b` 语义 |
| 新增 `stage1` | `{"kimg": [0,512], "save_full_state_at": [512], "launcher_template": "scripts/run_q256_target_weight_arm.sh"}` |
| 新增 `stage1b` | `{"kimg": [512,1024], "continuation": "A", "arms": {"AA": "A-history switch-resume to A (no-op switch, MUST pass through the same resume machinery)", "BA": "B-history switch-resume to A"}, "milestones": [640,768,896,1024], "launcher_template": "analysis/q256_schedule_switch_v1/launch_training_matrix.sh"}` |
| 新增 `stage2_preregistered` | `{"status": "preregistered, executed only in rebuttal window", "arms": ["AB","BB"], "source": "saved 512 full states", "classification": "same-cohort extension, not independent confirmation"}` |
| 新增 `primary_endpoint` | 见 P2 |
| 新增 `cohort_reduction_rule` | `"If stage-1 12/12 full states are not complete by 2026-09-12 24:00 UTC+8, the cohort reduces to the first 5 seeds with complete stage-1 states, recorded in evidence/ before any 1024-kimg evaluation. Cohorts below 5 seeds abort the confirmatory claim."` |
| 新增 `continuation_choice_rationale` | 原文粘贴："Continuation A is prespecified for both arms because (i) in the discovery cohort the history effect H_A under continuation A was the smaller of the two history contrasts, making it the stricter test, and (ii) A is the canonical late-refinement schedule. This choice is frozen before any fresh-cohort training begins." |

### P2 冻结分析计划 `analysis_plan.json`

```json
{
  "primary": {
    "estimand": "H_A_s = log(FID50k_NFE1(BA_s@1024)) - log(FID50k_NFE1(AA_s@1024))",
    "test": "exact one-sided paired Wilcoxon signed-rank",
    "H1": "median(H_A) < 0",
    "alpha": 0.05,
    "n": "6 (floor 5 per cohort_reduction_rule)"
  },
  "sensitivity": ["one-sided paired t-test on H_A_s", "exact sign test"],
  "secondary": {
    "estimand": "delayed reversal: count of seeds with (logFID_B@512 - logFID_A@512 > 0) AND (H_A_s < 0)",
    "reporting": "counts only, no hypothesis test"
  },
  "excluded_from_inference": ["KID", "NFE2", "AULC", "milestones 640-896", "raw FID"],
  "missing_data_rule": "a seed missing any of the 4 required evaluations (A@512, B@512, AA@1024, BA@1024) is dropped whole; below 5 remaining seeds the confirmatory analysis aborts"
}
```

### P3 冻结与推送

1. `sha256sum protocol.json analysis_plan.json > protocol.sha256`
2. commit + push 到分支 `experiment/q256-fresh-confirmatory-v1`，开 PR（可先 draft）。
3. 记录 commit 时间戳。

**Gate G1（9/3 24:00）**：`protocol.sha256` 的 commit 时间戳早于任何 fresh 训练日志的首行时间戳。这是唯一使 "confirmatory" 一词合法的证据，失败 = 全部 fresh 结果降级为 exploratory。

---

## Phase T — Fresh 训练执行（9/3–9/19，5×A100）

> **v1.1 状态：SUPERSEDED（已由 PR #97 完成）。** 实际执行：12 seeds × 4 臂，训练完整性 PASS（22/22 prefixes，44/44 suffixes）；seed38/AB 于 attempt 4866 第二次非有限 loss 触发协议强制退出（非硬超时），观测前修正为 n=11。下文保留仅作历史记录，**禁止再执行**。

### T1 Stage 1：12 个从零训练（6 seeds × 2 arms，0→512 kimg）

- Launcher：`scripts/run_q256_target_weight_arm.sh`（与 discovery cohort 的 A/B 臂同一模板，只改 seed）。
- 每个 run 必须在 512 kimg 保存 full training state（字段清单 = protocol.json 的 `required_state_fields`，共 11 项，缺一即该 run 判 FAIL 重跑）。
- GPU 排程：12 runs / 5 GPUs = 3 波。波 1：seed31A,31B,32A,32B,33A；波 2：33B,34A,34B,35A,35B；波 3：36A,36B（余 3 卡跑波 1/2 的 512 评估）。
- **9/12 24:00 检查点**：未满 12/12 → 触发 `cohort_reduction_rule`，机械执行，不讨论。

### T2 Stage 1b：12 个切换续训（512→1024 kimg）

- Launcher：复制 `analysis/q256_schedule_switch_v1/launch_training_matrix.sh` + `run_training_cell.sh` + `prepare_run_cell.py`，把 seed/臂表改为 12 行：每 seed 两行 `(history=A, continuation=A)` 与 `(history=B, continuation=A)`。
- **AA 臂必须同样经过 switch-resume 机制（A→A no-op）**，先对每个 seed 跑 `verify_parity.py` 式 no-op 校验：A→A 续训首步与原生继续训练 COMPUTATIONAL_STATE_MATCH，12/12 通过才继续。
- Milestone 快照：640/768/896/1024；1024 处保存 full state。
- GPU 排程：3 波，9/11–9/19。

**Gate G3（9/17 检查，硬线 9/19）**：12/12 continuations PASS + parity 12/12。失败分支：任何 seed 训练崩溃 → 按 `missing_data_rule` 整 seed 剔除，剩 ≥5 继续，<5 转 FAIL 分支（见 G5）。

---

## Phase E — 评估（滚动，与 T 重叠）

> **v1.1 状态：SUPERSEDED（已由 PR #97 完成）。** 实际执行：242/242 blind jobs SEALED_PASS，decode 仅在全矩阵 seal 之后；1 次评估存储恢复（非 metric 门）。下文保留仅作历史记录。

- 每个 milestone 产出后立即入队冻结评估（R-3）。primary 所需最小集：每 seed 4 个单元（A@512, B@512, AA@1024, BA@1024）× FID50k@NFE1。其余 milestone/NFE2/KID 照常评估但按 `analysis_plan.json` 不入推断。
- 验收：每个 job 有 receipt，generated-feature hash 一致性检查通过。

---

## Phase S — 统计判定（v1.1 改写：判定已完成，剩余为决策执行）

> **v1.1 状态：判定已由执行协议的冻结类别完成，结果 = INCONCLUSIVE**（`final_11seed/primary_decision.json`）。v1.0 的 PASS/TREND/FAIL 表作废——它绑定的是未执行的 H_A 单侧 Wilcoxon 计划；**禁止用任何 v1.0 检验对已解盲数据重判**（含：不得因 sign-flip p=0.042 或"若用单侧检验则显著"而改写判定级别）。

INCONCLUSIVE 的机械措辞后果（对应 v1.0 表的"论文措辞动作"列）：

| 项 | 动作 |
|---|---|
| confirmatory 措辞 | 全文删除。fresh cohort 的唯一合法称谓："preregistered, outcome-blind replication; primary verdict INCONCLUSIVE" |
| discovery cohort 称谓 | 固定为 DISCOVERY ONLY（与 #97 中 `docs/results/q256_512k_crossed_schedule_switch_seed3_7.md` 的降级一致），不得称 "robustly replicated" |
| 幅度表述 | discovery H≈−0.65 与 fresh H=−0.076 必须同处呈现；不得只引 discovery 幅度 |
| 反转表述 | 三件套强制：资格集定义（Q 阈值）+ 频数 + 幅度（引用规则见 `structure_diagnostic_v1/M1_M2_DIAGNOSTIC_NOTES.md`）；"8/11 negative" 禁止单独出现 |
| seed38 | 出现 fresh cohort 处必须带 complete-case + informative missingness 限制 |

**Gate G5（9/22 24:00，改为决策会）**：会议输入 = 本表 + 结构诊断 + 下方"9/22 决策会议题"。输出 = 唯一投稿路线记录（ICLR discovery+INCONCLUSIVE-replication 全披露版 / TMLR 版 / 增补 seeds 后顺延），写入 `docs/DECISION_G5.md`。

---

## Phase Z — 零 GPU 分析（9/4–9/8，与 T 并行，全部只用已有数据）

| ID | 任务 | 输入 | 输出 | 二元验收 |
|---|---|---|---|---|
| Z1 | Fig.1 反转交叉曲线 | `results/q256_schedule_switch_seed3_7/per_seed_trajectories.csv` | `figures/fig1_reversal.pdf`：x=kimg{512..1024}，y=log FID50k NFE1，每 seed 一小图（AA 实蓝/AB 虚蓝/BA 实橙/BB 虚橙）+ 第 6 格为 5-seed 均值，512 处竖线标注 switch | 6 格全出图；seed7 面板中 BA/BB 与 AA/AB 曲线在 512–1024 间可见交叉 |
| Z2 | FID 噪声地板 | seed3 AA@1024 snapshot | `analysis/noise_floor/NOISE_FLOOR.md`：同一 checkpoint 用 generation seed 0–4 重复 5 次 FID50k@NFE1，报 SD(logFID)，δ_noise = 2×SD | 5/5 次评估完成；δ_noise 数值写入 |
| Z3 | Checkpoint 探针族 | 512-kimg 已有 receipts + telemetry | `analysis/probe_family/PROBE_TABLE.csv`：探针 = {FID@NFE1, FID@NFE2, KID@NFE1, KID@NFE2, 训练 loss 末 32-kimg 均值} × seeds 3–7；每格 = sign(B@512−A@512) 是否等于 sign(H_A@1024)；探针判 MISRANK 若 ≤2/5 正确 | 25 格全填；每探针一行 RANK/MISRANK 结论 |
| Z4 | Source-gap robustness | `per_seed_trajectories.csv` | `analysis/robustness/SOURCE_GAP.md`：(a) H_A,log 对 gap512 的 OLS 斜率；(b) 剔 seed7 后 H 均值与符号计数；(c) 分层：S+={5,6,7} 与 S−={3,4} 层内 H 符号一致性 | 三小节全有数字；(c) 中 S− 层 2/2 同向即封堵均值回归攻击的关键行存在 |
| Z5 | 保留决策审计 | git log / 服务器归档时间戳 | `analysis/selection_audit/SELECTION_AUDIT.md`：证明 seeds 3–7 full states 的保留决策时间早于任何 crossed FID 读出 | 每 seed 一行时间戳对比，5/5 早于 |
| Z6 | ImageNet 表述修正 | PR #91 结果文档 | 全文将 "30/30 paired cells" 处补 "(n=3 training seeds; cells within a seed are repeated measurements)" | grep 检查：`30/30` 出现处均带 n=3 限定 |

> **v1.1 备注**：Z4 的 fresh-cohort 对应物已存在（PR #97 分支 `results/q256_fresh_crossed_switch_n12_matpool_v1/structure_diagnostic_v1/`，含 corr(H,Q)、子组分解、剔除敏感性），Z4 只需对 discovery cohort 执行并在输出中交叉引用该文件。新增 **Z7**：把 Fig.1 扩为 discovery/fresh 双 panel（fresh 数据源 `final_11seed/decoded_evaluation_results.csv`），验收 = fresh panel 中 seed34/41 反转可见、seed35 不交叉可见。

**Gate G2（9/8 24:00）**：Z1–Z6 六个输出文件全部存在且验收通过。

---

## Phase W — 论文写作（9/8–9/25）

### W1 结构与页预算（LaTeX 从空骨架新建 `overleaf_iclr2027_v2/`，禁止改旧 main.tex）

| 节 | 页 | 必含元素 |
|---|---|---|
| §1 Intro | 1.25 | 贡献**三条**（v1.3：identification structure / state propagation / finite-horizon quality evidence；不得缩成两条）+ Fig.1；结果段只写已执行实验，route-3 不得进入 |
| §2 Prelim | 0.75 | ECT、pair law、A/B 与两 regime 定义 |
| §3 Identification structure | 1.0 | regime 二分 + 跨论文比较失效实例 + attribution boundary；**无 "Proposition" 环境**；恒等式引附录 |
| §4 Design | 1.0 | H/S/I 定义（照抄 PR#95 REPORT 公式）+ Scope of Evidence 表（见 W3）+ discovery/replication 分层 |
| §5 Results | 2.25 | 表 T1（行=H_A,H_B,S_A,S_B,I；列=raw FID 均值、符号计数、logFID 均值、符号计数、**fresh H 列（固定：INCONCLUSIVE，−0.076 [−0.155, +0.004]，8/11，同行给子组分解指针）**）；反转表（Z4 数据 + fresh 强资格 2/3 行）；§5.3 探针表（Z3）；§5.4 BA 半页 |
| §6 Regime and scale boundaries | 0.75 | v1.3 更名（原 "Converging evidence" 撤销——q128/ImageNet 不直接验证 carryover，不得作 converging 卖点）：q128 边界 + ImageNet 规模边界（带 Z6 限定）+ **PR96 暴露时长边界**（384→512 短 B 暴露无可检测残留；与 route-3 的 0–512 全程史构成暴露时长对照） |
| §7 Mechanism support | 0.5 | PR90/94 一图一段 |
| §8 Related work | 0.5 | TCM 专段（划界句固定："TCM establishes stage necessity by ablation within a two-stage method whose stages differ in objective; we identify history and current-schedule effects within a single objective family via a crossed design"）+ Achille/Frankle/CCM/ADCM + LR-decay/grokking 划界句（"level improvement" vs "intervention ranking"） |
| §9 Discussion | 0.5 | 中途 FID 选 schedule 的实践警告 + limitations |

### W2 标题（v1.3：撤销"定稿"，转为 G5 议题）

原冻结标题 *Mid-Training FID Misranks Training Histories: A Crossed Decomposition of Pair-Spacing Effects in Consistency Training* 在 PR97 INCONCLUSIVE + PR96 INFORMATIVE_NULL 后偏强。**G5 议题**：替代标题 *Finite-Horizon Quality Carryover from Pair-Spacing History in Consistency Training*（把现象级断言降为可测性质）。9/22 前所有稿面用占位标题，G5 决定后机械替换。

### W3 Scope of Evidence 表（§4，全文唯一 boundary 集中点）

行：q256 crossed（status=pre-result-frozen discovery, n=5 paired）；**fresh cohort（status 固定："preregistered outcome-blind replication, n=11 complete-case (seed38/AB terminal numerical failure; informative missingness), primary verdict INCONCLUSIVE"，不随 G5 变动——G5 决定的是投稿路线，不是这一行的措辞）**；**B@384 pulse-chase（v1.3 新增：status="preregistered pulse-chase, n=10, INFORMATIVE_NULL at frozen 3% margin; short-exposure boundary"，来源 PR #96）**；q128（replication, n=3）；ImageNet-64（exploratory panel, n=3）；same-state audit（descriptive/mechanism）；设计不变量行（"All arms share paired seeds, data ordering, and fixed budgets"）；selection rule 行（retention-based, audit: Z5）。

### W4 机械写作检查（提交前运行，返回 0 行才许提交）

```
grep -nE "We emphasize|We do not claim|[Ff]or a fair comparison|Although the .*(is|are) modest|Taken together" overleaf_iclr2027_v2/*.tex
```

追加规则：每段结尾不得为限定句超过 1 次/节；`availability-selected`、`post-unblind` 两词不得出现在正文（只允许出现在 Scope 表的 status 列与附录）；n=5/n=3 首次出现必须与符号计数同句；seed4 的 −0.05 必须与 5/5 计数同表呈现。

v1.1 追加（fresh cohort 引用规则，来源 `M1_M2_DIAGNOSTIC_NOTES.md`）：
- `8/11` 出现处必须同句携带子组分解或指向分解表；
- 反转频数出现处必须同句携带资格定义与幅度（三件套）；
- `0.042`（sign-flip p）与 `0.0419` 若出现，必须同句注明 "not a decision input under the frozen protocol"；
- discovery 幅度（−0.65 或逐 seed 值）出现处，同段必须有 fresh 幅度（−0.076）。
- 机械检查：`grep -nE "8/11|0\.0419|0\.042" *.tex` 每一命中行人工核对上述规则，核对记录入 evidence/。

**Gate G6（9/24 24:00）**：W4 grep = 0 行；九节页预算 ±0.25 页内；Fig.1 + 表 T1 + Scope 表齐全。9/18 摘要提交，9/25 全文提交。

---

## 失败分支总表

| 触发 | 动作 |
|---|---|
| G1 失败（协议晚于训练） | ~~fresh 全降 exploratory~~（v1.1：不适用，执行协议冻结链已核验） |
| ~~9/12 stage-1 未满~~ | v1.1：不适用（训练已完成） |
| ~~cohort < 5~~ | v1.1：不适用（n=11） |
| **G5 = INCONCLUSIVE（实际发生）** | **9/22 决策会三选一（见下方议题）；方向未反转，TMLR 非自动默认；任何路线下 fresh 结果全披露，不得省略** |
| G5 = FAIL（方向反转） | 未发生。保留：默认转 TMLR |
| G5 = FAIL（p>0.10 但同向） | 已被 INCONCLUSIVE 行取代 |
| 任何评估 receipt hash 不一致 | 该 job 重跑；重跑仍不一致 → 该 seed 剔除并记录 |

## ~~Rebuttal 期预注册动作（写入正文脚注）~~ → v1.1：已作废并替换

~~Stage 2：从保存的 512 full states 补跑 AB/BB continuations……~~ **该预留已被 PR #97 消耗**：AB/BB 与全 factorial 一并执行且结果已解盲。原正文脚注模板删除，不得再写"results will be reported during the discussion period"。

### 9/22 决策会议题（G5 输入，三选一）

1. **ICLR 全披露版**：discovery + INCONCLUSIVE replication 双 cohort 入正文；标题现象的支持形态 = discovery 3/3 + fresh 强资格 2/3（含 seed35 反例）、fresh 幅度小一个量级。风险：审稿人以"复制未确认"为由压分；无 rebuttal 实验弹药。
2. **TMLR 版**：同一内容按 "claims match evidence" 标准重排，INCONCLUSIVE 复制作为核心贡献之一（严格预注册复制本身是卖点）。
3. **增补 cohort 顺延（v1.3 更新，取代 v1.1 的"n≈16 再+5 seeds"合并设想——该设想与 no-merge 规则冲突，作废）**：执行 `analysis/q256_fresh_confirmatory_v2_draft/`（v2 协议草案，经外部审查修订）：**全新两臂 cohort n=24**（seeds 51–74，非 n=11 扩充；corr(C₁,C₂)=0.965 数据仲裁），primary = 无调整 H_A 单侧配对 t，功效 0.947 / MDE 0.0591 / assurance 0.823；n=12 binding futility 期中（type-I 已 MC 验证）；预算 MINIMAL 包 127 RE ≈ 400 A100·h；目标 ICML 2027 / TMLR。代价：错过 9/25 全文线，顺延至下一周期。
- 会前必读：`final_11seed/REPORT_11SEED.md`、`structure_diagnostic_v1/M1_M2_DIAGNOSTIC_NOTES.md`、`MEDIAN_CORRECTION_V1.md`（manuscript 禁用归档 median 字段）、PR #96 的 seed-level equivalence 报告（§6 暴露时长边界段的数据源）。若议题涉及路线 3：`analysis/q256_fresh_confirmatory_v2_draft/PROTOCOL_DRAFT.md`（v2）+ `type_I_error_simulation.json`。
