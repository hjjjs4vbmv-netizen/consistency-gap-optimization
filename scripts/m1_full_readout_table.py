"""Reuse the completed metric extract; no evaluator calls or new fits."""

import argparse
import json
import statistics as st
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('metrics', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    rows = json.loads(args.metrics.read_text())
    cells = {(r[0], r[2], r[3], r[4], r[5]): r for r in rows}
    if len(cells) != len(rows):
        raise ValueError('duplicate evaluation cells')
    lines = ['# M1 完整绝对质量表', '',
             '复用已有 320 个有效评估；不新增生成或拟合。原始运行与 FP32 修复分开。', '',
             '三个读出的 FID/KID 均使用共同 generation block B0，便于同格比较。'
             'E_512 另列 B0/B1/B2 FID 的算术均值与样本 SD；SD 是生成重复差异，不是训练 seed 不确定性。'
             'NO_ENDPOINT 保留为缺失，不插补，不以修复结果替换。', '']
    for cohort in ('original', 'repair'):
        lines += [f'## {cohort}', '',
                  '| seed | branch | ONLINE FID / KID | E_KEEP FID / KID | E_512 FID / KID | E_512 FID 三次均值 ± SD |',
                  '|---|---|---|---|---|---|']
        pairs = sorted({(r[2], r[3]) for r in rows if r[0] == cohort})
        for seed, branch in pairs:
            formatted = []
            for readout in ('ONLINE', 'E_KEEP', 'E_512'):
                row = cells[cohort, seed, branch, readout, 'B0']
                formatted.append(f"{row[7]['fid50k_full']:.4f} / {row[7]['kid50k_full']:.6f}"
                                 if row[6] == 'PASS' else row[6])
            repeats = [cells[cohort, seed, branch, 'E_512', b] for b in ('B0','B1','B2')]
            if all(r[6] == 'PASS' for r in repeats):
                values = [r[7]['fid50k_full'] for r in repeats]
                spread = f'{st.fmean(values):.4f} ± {st.stdev(values):.4f}'
            else:
                spread = 'NO_ENDPOINT'
            lines.append('| ' + ' | '.join([str(seed), branch, *formatted, spread]) + ' |')
        lines.append('')
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
