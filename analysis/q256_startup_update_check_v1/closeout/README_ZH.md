# 释放与交付收尾

用户明确要求工作完成后自动释放容器。2026-09-10 00:52:04（北京时间），矩池云官方`matncli node cancel`返回0；随后SSH连接被拒绝。调用前确认GPU没有训练进程，并完成持久化归档校验。方法依据[矩池云官方自动释放说明](https://matpool.com/supports/reference/faqs/)。未查询账户账单金额。

持久化科学交付为commit `7a6995bf4ee5219dda21754a20f153b9a788ae16`，GPU执行commit为 `4198387b514fedbe43574452b18e021923a1de58`。本收尾提交只记录释放结果，没有改变训练或分析结论。由于记录包含释放后的连接验证，本目录最终副本保存在本地仓库及GitHub；释放前的官方请求/返回回执已写入持久化`control/container_release.json`。

最终持久化清单记录1087个运行文件、25033926730字节，以及最终代码补丁、交付包哈希。每个运行的文件SHA256在复制到网盘后已与源文件逐一匹配；释放前再次检查全部文件存在，并核验交付包和36份摘要文件的哈希。没有在释放前重复读取全部25GB张量。

[完整中文报告](../results/REPORT_ZH.md)；[Draft PR #110](https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization/pull/110)，已核实为OPEN/DRAFT，目标为PR109的head分支，未合并。

矩池云持久化目录：`/mnt/q256_startup_update_check_v1_20260909`。科学报告和脚本交付包在`control/ect-delivery.tar.gz`，展开副本在`delivery/analysis/q256_startup_update_check_v1/`；原始日志、诊断张量和检查点在`runs/`。
