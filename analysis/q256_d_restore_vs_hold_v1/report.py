"""Write the final report only from completed frozen statistics and cost evidence."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path


def number(value, digits=7):
    return '—' if value is None else f'{value:.{digits}g}'


def interval(value):
    return '—' if value is None else '[' + ', '.join(number(x) for x in value) + ']'


def render(root):
    root=Path(root)
    s=json.loads((root/'statistics.json').read_text())
    cost=json.loads((root/'cost_summary.json').read_text())
    outcomes=json.loads((root/'outcomes_16.json').read_text())
    if s['status'] not in {'COMPLETE','COMPLETE_WITH_SCIENTIFIC_FAILURES'}:
        raise RuntimeError('technical or budget work remains; final reporting is blocked')
    p=s['primary']; n=p.get('n',0)
    completed=sum(r['DD']=='PASS' for r in outcomes)
    failed=sum(r['DD']=='SCIENTIFIC_FAILURE' for r in outcomes)
    direction={'A_favored':'95%区间支持恢复A降低终点FID。',
               'D_favored':'95%区间支持保持D降低终点FID。',
               'unresolved':'几何比95%区间包含1，方向性证据不足。'}
    lines=['# D历史后恢复A与保持D：完整结果', '',
           f'计划16条DD后缀，{completed}条成功、{failed}条科学失败。唯一主比较使用完整有限配对 n={n}，统计单位是training seed。', '',
           f"**DA/DD几何FID比为 {number(p.get('geometric_ratio'))}，95%配对区间 {interval(p.get('ratio_ci95'))}。** " + direction.get(p.get('direction_inference'),'配对数据不足，不能给出方向性推断。'), '',
           '## 证据与范围', '',
           '- 实验：q256_d_restore_vs_hold_v1；每个DD仅从对应原完整D@512继续至1024，不重跑前缀。',
           '- DA的80个原槽直接复用；AA仅作辅助，原缺失保留。seeds58、65始终属于DD主队列。',
           '- 分析在DD正式训练前冻结；DA既有结果已经观察。本轮不是独立新seed复现。',
           '- 正式实现源1740b9c，实际四节点部署28e9ffd；冻结FP32/NFE1 evaluator为d6aba02。结果报告commit不代表历史训练执行commit。', '',
           '## 唯一主比较', '',
           '每seed先对B0/B1/B2的log-FID作等权平均，形成DA减DD的配对差。负值表示恢复A有利，正值表示保持D有利。三个generation blocks是重复测量，不是三个training seeds。', '',
           '| 项目 | 数值 |', '|---|---:|',
           f'| 完整配对n | {n} |',
           f"| 平均log-FID差 DA−DD | {number(p.get('mean_log_difference'))} |",
           f"| seed SD（log差） | {number(p.get('seed_sd'))} |",
           f"| 双侧95% t区间 | {interval(p.get('ci95'))} |",
           f"| 双侧p值 | {number(p.get('p_two_sided'))} |",
           f"| 几何FID比 DA/DD | {number(p.get('geometric_ratio'))} |",
           f"| 几何比95%区间 | {interval(p.get('ratio_ci95'))} |",
           f"| 相对变化 | {number(p.get('relative_change_percent'))}% |",
           f"| 负/零/正方向seed数 | {p.get('direction_counts',{}).get('negative',0)} / {p.get('direction_counts',{}).get('zero',0)} / {p.get('direction_counts',{}).get('positive',0)} |", '',
           '全部逐seed值见per_seed.csv，全部原始FID/KID见metrics_160.csv。所有有限极端FID均保留，不插补缺失，不按成功更新数重加权。', '',
           '## 预设乘法3%实用等效', '',
           '边界为±log(1.03)，对应几何比范围[1/1.03, 1.03]，约[0.970874, 1.03]。以90% t区间严格落入边界作为TOST alpha=0.05的等效判断；方向判断与实用等效分别输出。', '',
           f"90% log差区间：{interval(p.get('ci90'))}；TOST p={number(p.get('tost_p'))}。",
           '完整配对集合中的实用等效'+('得到支持。' if p.get('practically_equivalent') is True else '未得到支持；不显著本身不证明相同。'), '',
           '该判断'+('基于本轮全部16个成功配对。' if n==16 else '仅限完整有限配对，不能对全16队列宣称质量等效。'), '',
           '## 16-seed全部成败', '', '| seed | 旧DA | 新DD | 旧AA辅助 |', '|---:|---|---|---|']
    for r in outcomes:lines.append(f"| {r['seed']} | {r['DA']} | {r['DD']} | {r['AA']} |")
    metrics=json.loads((root/'DA_DD_160.json').read_text())
    lines += ['', '## 绝对指标描述', '',
              '下表先在seed内汇总固定blocks，再跨seed作描述性汇总；不把blocks当独立seed。全量逐槽FID/KID均保留在metrics_160.csv。', '',
              '| 路径/读出 | n | FID算术均值 | FID几何均值 | KID原始均值 |', '|---|---:|---:|---:|---:|']
    for arm in ('DA','DD'):
        for readout in ('E_512','E_KEEP','ONLINE'):
            grouped={}
            for row in metrics:
                if row['path']==arm and row['readout']==readout and row['status']=='PASS':grouped.setdefault(row['seed'],[]).append(row)
            if not grouped:continue
            f=statistics.mean(statistics.mean(x['FID'] for x in values) for values in grouped.values())
            g=math.exp(statistics.mean(statistics.mean(math.log(x['FID']) for x in values) for values in grouped.values()))
            k=statistics.mean(statistics.mean(x['KID'] for x in values) for values in grouped.values())
            lines.append(f'| {arm}/{readout} | {len(grouped)} | {number(f)} | {number(g)} | {number(k)} |')
    a=s['auxiliary']
    lines += ['', '## AA辅助与次级读出', '',
              f"AA/DA/DD辅助比较只使用三路径共同有效集合 n={a.get('n',0)}：{a.get('common_seeds',[])}。不得拿16-seed DA/DD均值与14-seed AA均值直接拼接。辅助结果是描述性比较，不增加新的主要假设。", '',
              '| 同一共同集合的辅助对比 | n | 平均log差 | 几何比 | 名义95%区间 |', '|---|---:|---:|---:|---|']
    for name,value in a.get('comparisons',{}).items():
        lines.append(f"| {name} | {value['n']} | {number(value['mean_log_difference'])} | {number(value['geometric_ratio'])} | {interval(value['ratio_ci95'])} |")
    lines += ['',
              'E_KEEP与ONLINE均仅比较共同B0；E_512主读出使用固定三个block。KID复用相同features，按原始值汇总，不取log，也不作为独立复现。完整次级数值保存在statistics.json。', '',
              '![DA/DD配对](figures/DA_DD_paired.png)', '',
              '![逐seed配对log差](figures/paired_log_differences.png)', '',
              '![AA/DA/DD共同集合](figures/AA_DA_DD_common.png)', '',
              '## 实施、核查与成本', '',
              '独立DD协议与状态元数据明确current_arm=D、target1.0、denominator1.1。保留原模型/buffers、RAdam、GradScaler、RNG/sampler及全部训练时钟；E_KEEP保持，E_512从边界online初始化一次，resume不再初始化。正式训练始终使用原生D分母，未用scalar或LR替代。', '',
              '原FP16+AMP、batch128/microbatch16/world_size1与全部原超参数保持。正常AMP skip保留，既不补样本也不对齐成功更新数；科学失败不救援或换seed。仅对1024终点做固定80槽评估，中间完整checkpoint保留。', '',
              'B核查区分实数目标、浮点实现及LR关系；浮点loss/梯度/实际更新存在差异，未扩展完整scalar质量臂。工程结果与准确执行版本见../engineering/REPORT_ZH.md及source_provenance.json。', '',
              '| 新增GPU进程成本 | GPUh |', '|---|---:|']
    for key,label in [('training_gpuh','DD训练'),('evaluation_gpuh','80槽评估'),('export_gpuh','读出导出'),('engineering_gpuh','工程及其技术失败'),('total_gpuh','总计')]:
        lines.append(f'| {label} | {number(cost.get(key),10)} |')
    lines += ['', '以上为实际新进程墙钟GPUh；不计入复用前缀或旧DA的历史elapsed。租赁闲置、传输及账单口径另列，不能把GPU进程成本冒称实际租赁账单。80 GPUh进程上限与用户确认的200 GPUh剩余额度分开记录。', '',
              '归档状态及逐文件来源见archive_index.json。复制完成与最终hash验证分开记录；大权重、生成图、features及私有连接不进入公开Git。', '',
              '## 解释限制', '',
              '本轮回答从已形成的D历史出发，512之后恢复A相对继续D对终点质量的影响。没有AD或多切点设计，不能识别早期唯一关键窗口、最佳切换点、history×current interaction或optimizer唯一中介，也不外推跨精度普遍性。', '',
              '实用等效若成立，仅说明这项恢复操作的差异落在预设尺度内，不否定PR108的DA−AA历史效应。若有科学失败，有限FID推断条件于存活配对，全部16个结局同时报告。', '',
              '## 离线复算', '', '```bash',
              'python -m analysis.q256_d_restore_vs_hold_v1.analyze --dd-slots analysis/q256_d_restore_vs_hold_v1/results/DD_80.json --output /tmp/dd-recomputed',
              'python -m analysis.q256_d_restore_vs_hold_v1.render_figures --results analysis/q256_d_restore_vs_hold_v1/results',
              'python -m unittest tests.test_d_restore tests.test_history_component tests.test_m1_training_state tests.test_history_component_evaluation', '```', '',
              '原数值环境与数据/脚本hash见验证记录。核心分析始终采用训练前冻结版本。']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    render(parser.parse_args().results)
