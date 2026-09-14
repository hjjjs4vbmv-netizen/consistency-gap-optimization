"""Render the prespecified H contrast; individual seeds have no block-level CI."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / 'per_seed.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    effect = json.loads((ROOT / 'statistics.json').read_text())['effects']['H']
    changes = [100*math.expm1(float(r['H'])) for r in rows]
    mean = 100*math.expm1(effect['mean_log'])
    lower, upper = [100*math.expm1(v) for v in effect['ci95_log']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = list(range(13, 1, -1))
    ax.axvline(0, color='#999999', linestyle='--', linewidth=1)
    ax.scatter(changes, y, s=32, color='#536273', zorder=3)
    ax.errorbar(mean, 0, xerr=[[mean-lower], [upper-mean]], fmt='D',
                color='#005a8d', markersize=6, capsize=4, linewidth=1.8)
    ax.axhline(1, color='#dddddd', linewidth=.8)
    ax.set_yticks(y + [0], [r['seed'] for r in rows] + ['Overall H'])
    ax.set_xlim(-27, 15)
    ax.set_ylim(-1.8, 14)
    ax.set_xlabel('NFE1 FID change: DA relative to AA (%)')
    ax.set_ylabel('Training seed')
    ax.set_title('F12 at q = 256: fixed fresh cohort (n = 12)', loc='left', pad=16)
    ax.text(-26, -1.25, f'Overall: {mean:.2f}%   95% CI [{lower:.2f}%, {upper:.2f}%]   p = {effect["p_two_sided"]:.4f}', fontsize=9, color='#005a8d')
    ax.xaxis.grid(True, color='#eeeeee', linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / 'paired_H.png', dpi=180)
    fig.savefig(ROOT / 'paired_H.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
