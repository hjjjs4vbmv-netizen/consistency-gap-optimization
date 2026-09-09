# D、A/1.1与学习率缩放：分层结论

本文件是源码数学核查。生产浮点证据由固定GPU短核查另行生成；此文件不预先声明逐位通过，也不启动完整scalar质量臂。

原sigmoid实现先算 `adj=1+8*sigmoid(-t)`、`ratio=1-adj/256`、`base_r=clamp(t*ratio,min=0)`。stage为0时，对实数有限正t，sigmoid(-t)<1/2，因此0<1.1*Delta_A<5.5*t/256<t。放大间隔不触发上界裁剪，D和A具有相同target、同一pair误差，D逐样本损失在实数算术下等于A/1.1。这只适用于共同参数状态与相同batch、噪声、dropout；不能把已分叉DA/DD轨迹的损失当作同一theta下的函数值。

生产实现的t/base_r和输出loss dtype应由实际tensor记录决定。denoiser FP16不意味着将时间或分母转成FP16。原生D先构造r_denominator，再做t-r_denominator，舍入顺序不同于逐样本A loss/1.1。B核查保留原128有效batch、8个microbatch，每microbatch依原实现取mean后调用GradScaler.backward，不另除以累积次数。scalar支路仅在原生A逐样本返回值后、原有mean之前除1.1；不混称batch-reduced scalar操作逐位一致。

学习率缩放不是一般的RAdam loss缩放替代。loss缩放先改变梯度，继而改变m/v和可能的AMP overflow/skip；LR缩放保留A的梯度和moment更新，只改变实际RAdam步长。相同已有m/v的source下，这些差别直接影响更新。即使从零moment出发，RAdam早期非自适应分支、epsilon、舍入和AMP也不支持“全程严格scale-invariant”的陈述。不得用LR或scalar实现替代正式DD的原生分母路径。

核查固定seeds50、51、原ECT初始化和原D@512。每variant每source只短跑16 attempts（低于32上限），无FID。首次共同有效batch记录输入/输出、时间和分母、loss、AMP unscale后的原梯度、实际RAdam更新与moments/scaler/skip。后续仅用这16步原有日志描述分叉。初始化重建须与对应旧初始receipt核对，不将transfer权重文件称为完整可恢复optimizer/RNG状态。

完整scalar质量臂保持取消状态。浮点差异、skip或明显更新分歧只收窄等价主张，不增加质量矩阵。工程全局目标/上限为3 GPUh。
