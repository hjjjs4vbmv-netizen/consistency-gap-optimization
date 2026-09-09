# 环境与准备过程记录

本轮基线为 PR109 `092ab9ab54e3f2dd780654b752bf85643d92af59`。
两台授权服务器在沙箱外均成功连通并完成 SSH 认证；未使用额外跳板。
密码仅用于交互认证，不在代码、归档、命令清单或本报告中保存。

A100 默认环境为 Python 3.13.5 / Torch 2.8.0+cu128，未用于训练。
恢复了原归档的 `q256-training-runtime-py311-torch260.tar.gz`，其归档哈希通过核验。
该压缩包只有基础运行时，缺少部分原部署依赖，且 Pillow 为 11.0.0。
根据原 `runtime_parity_report.json` 补齐 SciPy 1.16.1、psutil 7.0.0、Click 8.2.1，
将 Pillow 恢复为 11.3.0。最终 Python 3.11.13、Torch 2.6.0+cu124、CUDA runtime 12.4、
cuDNN 90100、NumPy 2.1.2、SciPy 1.16.1 均与要求相符。

额外缺失的 HTTP 导入依赖采用 requests 2.32.4；仅供离线绘图的 Matplotlib 为 3.10.5。
这些辅助依赖的原版本未在可读原部署回执中列明，因此不宣称整个 pip freeze 与原环境逐包相同。
完整新环境导出为 `pip-freeze.txt`；原基础环境包和依赖修复日志保存在服务器。
原部署回执记录 NVIDIA driver 610.57.04，本轮 driver 570.211.01；运行时和 cuDNN 版本保持原值。
初始化及原 A 前64步日志核对用于检查实际影响，不能仅凭版本号假定数值一致。

ECT002 常规 datasets 目录及 A100 现有同名 ZIP 的哈希均不符，均未用于本轮。
使用原归档 `assets/cifar10-32x32-training-original.zip`，核验为要求的 9818e4b8…f1b3。
A100 已有 transfer 权重哈希为要求的 4d5dcc1f…4da，因此复用该文件。
仅复制训练 ZIP、原环境包、seeds50/51 回执与配置；没有复制全量旧检查点、FID features 或整个归档。

原 `launch_receipt.json` 对 ECT002 用户不可读；可读的原训练配置、launcher 日志、原部署回执以及
仓库 `analysis/q256_history_component_chase_v1/protocol.py` 的命令构造提供命令核对依据。
本轮完整八条命令和展开后的原生训练参数保存在 `eight_commands.json`。

持久化 /mnt 网盘的原子硬链接发布检查失败。因此保留原训练原子写入函数，使用容器本地
`/root/q256_startup_update_check_v1_runs`，每个进程退出后复制完整输出到持久化实验 `runs/`，
逐文件核对 SHA256 后再启动下一臂。失败的文件系统预检没有启动 GPU 训练。

准备阶段还遇到 Git HTTPS 超时、慢速本机资产中转和缺少历史 JSON 测试资料。
分别通过按 Git blob 哈希校验原始文件、ECT002 直传 A100、补齐两份原测试资料解决。
首次 CPU 导入测试失败、资料未齐的回归测试失败均保留日志；最终相关23项测试全部通过。
这些准备操作没有启动额外科学轨迹，也没有计入不存在的“已完成 GPU 检查”。

后处理第一次独立 checkpoint 验证因脚本路径不在仓库根的 Python 导入路径中，
报 `ModuleNotFoundError: torch_utils`；保留原错误日志后设置 PYTHONPATH 完成 CPU 重试。
该问题没有启动训练或增加 GPU 成本。正式交付采用仓库模块入口执行 `verify_results`。
