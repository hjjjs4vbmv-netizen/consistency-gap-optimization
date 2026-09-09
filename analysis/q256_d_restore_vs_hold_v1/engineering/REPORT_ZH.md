# 数学、浮点与学习率核查结果

固定 seeds50、51，原 ECT 初始化与 D@512，四变体各16 attempts，无FID。所有变体从相同完整源状态，或与旧初始化 receipt 完全匹配的重建起点执行。

**数学目标等价；浮点和训练轨迹不应称为等价；LR缩放不能替代原生D。完整scalar质量臂取消，不增加矩阵。**

实际 denoiser 输入为 torch.float16；t、base_r、denominator 与逐样本 loss 均为 torch.float32。未将时间或分母强制转成FP16。逐样本 A/1.1 插在原生返回值之后、原有每 microbatch mean 之前；8个 microbatch 按原AMP方式累积梯度，未更改 reduction。

| seed/source | loss最大绝对差 D−A/1.1 | loss最大相对差 | 首步D / scalar | 实际参数最大绝对差 |
|---|---:|---:|---|---:|
| 50/ECT初始 | 7.6294e-5 | 1.0067e-5 | skip / skip | 无首步更新，不能比较 |
| 50/D@512 | 8.7738e-5 | 1.3304e-5 | update / update | 3.1485e-5 |
| 51/ECT初始 | 1.0681e-4 | 1.3694e-5 | skip / skip | 无首步更新，不能比较 |
| 51/D@512 | 8.3923e-5 | 1.1885e-5 | update / update | 4.5450e-5 |

四个共同首batch的 D/A target 时间、输入和输出逐位相同。原生D与 A/1.1 的 loss 和梯度四例均不逐位相同；D@512 的实际RAdam参数与 moments 也不同。初始状态共同首步均被原AMP跳过，不能把缺少参数更新记录误写成“更新一致”；后续16步里的首个成功更新及skip计数另列在 summary.json。

逐坐标最大相对误差会被接近零的梯度或参数放大，因此保留绝对误差，并另提供更新向量的CPU范数复算。初始非有限梯度按原 unscale、overflow 检测、sanitization、scaler.step/skip 顺序处理；未更改scaler、样本计数或成功更新时钟。

在D@512共同首步，D与逐样本A/1.1的梯度相对L2误差分别为0.0657%和0.0508%；实际参数更新向量的相对L2误差为0.2253%和0.2852%，余弦分别为0.99999746和0.99999593。分母均为右侧变体向量的L2范数。误差虽小，仍不能推出完整训练轨迹等价。

D与A的LR/1.1变体的更新相对L2差分别为9.18%和8.58%。原生A与其LR变体的首步梯度完全一致，但实际更新不同。完整范数数据见 norms.json。

原32条已记录流（16条D前缀和16条DA后缀）每条4000 attempts，stage均为0；现存 factor/nonpositive-denominator、scaled-to-zero 计数及 summary 裁剪率均为0。该结论仅覆盖实际记录字段，不声称拥有未记录的全程 epsilon/dropout 指纹。实数推导见上级 B_MATH_AND_SCOPE.md。

原RAdam dispatcher保留foreach的生产选择，完整模块源码hash归档。LR变体仅在真实RAdam.step时将lr除1.1，保留A梯度和moment输入；原生D改变梯度及moment更新。还需区分RAdam早期非自适应分支与后期rectification，不能笼统称全程scale-invariant。

工程成功表示协议与执行检查完成，不代表浮点逐位等价或质量阳性。数值分歧按原样报告，不升级成完整scalar、LR或FP32质量臂。

全部新增工程GPU进程成本（含技术失败）为 **0.407817169 GPUh**，低于3 GPUh。37项针对性及旧协议CPU测试通过；固定DD连续16步与8+8 resume、旧DA在原PR108与新代码的16步计算状态逐位一致，E_512初始化一次。

正式训练使用独立原生DD入口，不使用任何 audit_entry 变体。

工程部署实际为 `de04bee`，由 `29b5684` 基线及跨设备比较、短检查恢复、工程入口导入这三个文件补丁组成。工程归档中逐次写入的 `source_commit` 仅指最新补丁来源，不能当作完整源树版本；已用实际Git blob核对并在 source_provenance.json 中补充准确映射。正式四节点部署均为 `28e9ffd`，对应逐文件核验的完整选定源快照 `1740b9c`。原生产loss、schedule、RAdam和AMP数学路径未改变；结果报告commit另计。

CPU离线范数复算入口（路径对应归档中的完整原D@512源清单和本次工程产物）：

```bash
python -m analysis.q256_d_restore_vs_hold_v1.audit_norms --audit-root ENGINEERING_DIR --source-inventory SOURCE_INVENTORY.json --output norms.json
```
