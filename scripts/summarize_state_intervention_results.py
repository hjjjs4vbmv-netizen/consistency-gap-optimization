"""Reproduce seed-level estimates and readable tables from bound scalar results."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import state_intervention_contrasts as contrasts


def summarize(rows):
    expected = {(s, b, t, r, block) for s in contrasts.SEEDS
        for t, branches in ((768, ('K_A', 'K_B', 'R_A', 'R_B')),
                           (1024, ('K_A', 'K_B', 'R_A', 'R_B', 'L_A', 'L_B',
                                   'X_A_from_B', 'X_B_from_A')))
        for b in branches
        for r in (('ONLINE', 'E_512') if t == 768 else ('ONLINE', 'E_KEEP', 'E_512'))
        for block in (('B0', 'B1', 'B2') if r == 'E_512' else ('B0',))}
    keys = [(int(r['seed']), r['branch'], int(r['budget_kimg']), r['readout'], r['block']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('results must contain all 336 unique planned slots, including missing statuses')
    results = {}
    for name, readout, blocks in (('E_512_three_blocks', 'E_512', ('B0', 'B1', 'B2')),
            ('E_512_B0', 'E_512', ('B0',)), ('ONLINE_B0', 'ONLINE', ('B0',)),
            ('E_KEEP_B0', 'E_KEEP', ('B0',))):
        results[name] = {metric: contrasts.summarize_cells(
            contrasts.aggregate_cells(rows, metric, readout, blocks),
            include_same_chase=readout != 'E_KEEP')
            for metric in ('fid50k_full', 'kid50k_full')}
    return results


def number(value):
    return '缺失' if value is None else f'{value:.9g}' if isinstance(value, float) else str(value)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(number, row)) + ' |' for row in rows]]) + '\n'


def public_metadata(value):
    if isinstance(value, dict):
        return {k: public_metadata(v) for k, v in value.items()
                if 'path' not in k.lower() and k not in ('host', 'command')}
    if isinstance(value, list):
        return [public_metadata(v) for v in value]
    return value


def training_report(source, output):
    training = source.get('training')
    if not training:
        return
    fields = ('attempted_iteration', 'ema_attempt', 'step_skipped', 'successful_optimizer_steps',
        'optimizer_step_values', 'loss', 'raw_grad_norm', 'raw_grad_nonfinite_count',
        'update_norm', 'update_nonfinite_count', 'grad_scale_before', 'grad_scale_after',
        'radam_first_moment_norm', 'radam_second_moment_norm', 'batch_sha256', 't_sha256',
        'input_rng_before')
    records = []
    for row in training:
        public = public_metadata({k: v for k, v in row.items() if k != 'first64'})
        public['first64'] = [{k: obs[k] for k in fields} for obs in row['first64']]
        records.append(public)
    (output / 'training.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    train_h = sum(r['elapsed_seconds'] for r in records) / 3600
    eval_h = sum(r['elapsed_seconds'] for r in source['rows'] if r['mode'] == 'NEW') / 3600
    content = '# 训练终态、有限动态和实际成本\n\n'
    content += f'24/24条训练完成至8000 attempts。新训练进程窗口合计{train_h:.4f} GPUh；216项新增评估进程合计{eval_h:.4f} GPUh；两者合计{train_h+eval_h:.4f} GPUh。\n\n'
    content += '训练窗口含进程启动/加载/保存，评估窗口不含每项前置canary和CPU导出。这不是精确CUDA kernel占用，也不含工程检查、租赁闲置、原120项评估历史成本。工程检查为原2–4 GPUh规划额，缺少统一实测总账，不将其伪称实测；不使用含继承历史的elapsed_sec累计值充当本轮成本。租赁计费起止记录不在产物中，因此精确闲置计费时长不可得。\n\n'
    content += table(['seed', '分支', '状态', '终点attempt', '新增成功更新', '新增skip', '进程小时'],
        [[r['seed'], r['branch'], r['status'], r['attempted_iteration'],
          r['intervention']['successful_steps_since_init'],
          (2000 if r['branch'].startswith('L_') else 4000)-r['intervention']['successful_steps_since_init'],
          r['elapsed_seconds']/3600] for r in records])
    content += '\n## 前64次attempt\n\n下表只描述已预设记录的有限动态，不是全程最大更新、不作质量准入。step是实际RAdam内部值，不拿全局attempt替代。原始逐attempt标量和输入指纹见training.json；没有额外运行首步K参考向量，因此不报告不存在的norm ratio/cosine。\n\n'
    content += table(['seed', '分支', '首步step', '第64步step', '前64 skip', '首步update norm', '前64最大update norm'],
        [[r['seed'], r['branch'], str(r['first64'][0]['optimizer_step_values']),
          str(r['first64'][-1]['optimizer_step_values']), sum(int(o['step_skipped']) for o in r['first64']),
          float(r['first64'][0]['update_norm']), max(float(o['update_norm']) for o in r['first64'])] for r in records])
    content += '\n## 初始化状态摘要\n\ntraining.json保留receiver操作前、操作后moment/step摘要、donor成功更新计数和参数名映射，去除私有路径。L只清空完整optimizer，X移植完整(m,v,n)；接收方成功更新计数与donor内部step分别记录，不强行对齐。\n'
    (output / 'TRAINING.md').write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    if isinstance(source, list):
        source = dict(rows=source)
    results = summarize(source['rows'])
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = ('slot_id', 'seed', 'branch', 'budget_kimg', 'readout', 'block', 'mode', 'status',
        'fid50k_full', 'kid50k_full', 'source_state_sha256', 'snapshot_sha256', 'readout_sha256',
        'result_receipt_sha256', 'export_receipt_sha256', 'original_slot_id', 'recorded_dataset',
        'metric_json_sha256', 'elapsed_seconds', 'started_utc', 'ended_utc', 'sample_seed_start',
        'sample_seed_end', 'sample_count', 'nfe', 'precision', 'metric_seed', 'evaluator_commit')
    raw = [{k: r[k] for k in allowed if k in r} for r in source['rows']]
    for filename, data in (('raw_metrics.json', raw), ('estimates.json', results),
                           ('validation.json', source.get('validation', {'status': 'PENDING'}))):
        (args.output / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    raw_text = '# 原始质量指标：336 个固定槽位\n\n'
    raw_text += '每行是一条 checkpoint × 读出 × generation block 的 FID50k/KID50k；不是独立训练样本。KID 为原始值，没有乘以1000。NEW=本轮生成，REUSE=原M1结果原样复用。\n\n'
    raw_text += table(['seed', '分支', 'kimg', '读出', 'block', '来源', '状态', 'FID50k', 'KID50k'],
        [[r[k] for k in ('seed', 'branch', 'budget_kimg', 'readout', 'block', 'mode', 'status',
                         'fid50k_full', 'kid50k_full')] for r in raw])
    (args.output / 'RAW_METRICS.md').write_text(raw_text)
    detail = '# 配对结果明细\n\nFID列是自然对数FID之差；KID列为原始差。区间均为seed级双侧95%名义t区间；SD使用ddof=1。辅助读出不替换E_512三block主分析。\n'
    for readout, metrics in results.items():
        detail += '\n## ' + readout + '\n'
        for metric, analyses in metrics.items():
            detail += '\n### ' + metric + '\n\n'
            detail += table(['比较集合', '量', 'n', 'mean', 'SD', 'CI下界', 'CI上界', '方向'],
                [[name, key, e['n'], e['mean'], e['sample_sd'], *(e['ci95'] or [None, None]), e['ci_direction']]
                 for name, analysis in analyses.items() for key, e in analysis['estimates'].items()])
        primary = metrics['fid50k_full']
        for name in ('m2_same_endpoint', 'm2_same_chase', 'swap_common'):
            if name not in primary:
                continue
            analysis = primary[name]
            detail += '\n### 逐seed：' + name + '\n\n'
            detail += '集合：' + str(analysis['seeds']) + '；排除：' + str(analysis['excluded']) + '。\n\n'
            if analysis['per_seed']:
                fields = list(analysis['per_seed'][0])
                detail += table(fields, [[row[k] for k in fields] for row in analysis['per_seed']])
    (args.output / 'PAIRED_RESULTS.md').write_text(detail)
    absolute = []
    for key in sorted({(r['branch'], r['budget_kimg'], r['readout']) for r in raw}):
        selected = [r for r in raw if (r['branch'], r['budget_kimg'], r['readout']) == key]
        per_seed = [[r for r in selected if r['seed'] == seed] for seed in contrasts.SEEDS]
        for metric in ('fid50k_full', 'kid50k_full'):
            values = [statistics.fmean(r[metric] for r in group) for group in per_seed
                      if group and all(r['status'] == 'PASS' for r in group)]
            absolute.append([*key, metric, len(values), statistics.fmean(values) if values else None,
                             statistics.stdev(values) if len(values) > 1 else None])
    (args.output / 'ABSOLUTE_QUALITY.md').write_text('# 绝对质量汇总\n\n每个seed先平均同读出的blocks，再在六seed间计算算术均值和样本SD。这不是主估计量使用的平均log，也不是FID150k。\n\n' + table(
        ['分支', 'kimg', '读出', '指标', 'seed数', '算术均值', 'seed间SD'], absolute))
    training_report(source, args.output)


if __name__ == '__main__':
    main()
