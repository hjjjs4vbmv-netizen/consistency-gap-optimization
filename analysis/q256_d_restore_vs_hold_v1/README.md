# q256_d_restore_vs_hold_v1

独立于PR108的新DD后缀对照：seeds50–65，共16条，原D@512恢复至1024。正式损失始终使用原生target1.0/denominator1.1；world_size1、FP16+AMP、原RAdam和完整状态保持。

`EXECUTION_SPEC.md`保存附件及本轮16卡覆盖要求。`frozen/`包含训练前冻结分析、原DA80槽、AA辅助80槽、DD80个计划槽和16行源绑定。`B_MATH_AND_SCOPE.md`只陈述数学关系和核查范围，GPU工程结果另行出具。

真实入口（均先支持`--help`）：

```bash
python -m analysis.q256_d_restore_vs_hold_v1.worker --deployment PRIVATE_DEPLOYMENT.json --seed 50 --gpu 0 --dry-run
python -m analysis.q256_d_restore_vs_hold_v1.short_checks --deployment PRIVATE_DEPLOYMENT.json --output ENGINEERING_DIR --gpu 0
python -m analysis.q256_d_restore_vs_hold_v1.preflight --deployment PRIVATE_DEPLOYMENT.json --engineering ENGINEERING_DIR/short_checks.json --output PREFLIGHT.json
python -m analysis.q256_d_restore_vs_hold_v1.worker --deployment PRIVATE_DEPLOYMENT.json --seed 50 --gpu 0
python -m analysis.q256_d_restore_vs_hold_v1.evaluate --deployment PRIVATE_DEPLOYMENT.json --seed 50 --gpu 0
```

占位路径是部署配置，实际命令另存执行回执。分析可完全离线复算：

```bash
python -m analysis.q256_d_restore_vs_hold_v1.analyze --dd-slots RESULTS/DD_80.json --output RECOMPUTED
python -m analysis.q256_d_restore_vs_hold_v1.render_figures --results RESULTS
python -m unittest tests.test_d_restore tests.test_history_component tests.test_m1_training_state tests.test_history_component_evaluation
```

源权重、生成图、features与私有路径留在既有持久化归档。此设计不能称独立新seed复现，也不能识别早期唯一关键窗口。
