"""Validate and report the outcome-selected optimizer diagnostic evidence."""
import argparse
import datetime as dt
import gzip
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path


OPERATIONS = ('K', 'R', 'clear_moments', 'reset_step')


def analyze(data):
    cells = data['cells']
    index = {(c['summary']['seed'], c['summary']['arm'], c['summary']['operation']): c for c in cells}
    expected = {(s,a,o) for s in (55,56,60,64) for a in ('A','B') for o in OPERATIONS}
    if len(cells) != 32 or set(index) != expected:
        raise ValueError('diagnostic matrix is not complete and unique')
    for (seed, arm, operation), cell in index.items():
        s, rows, intervention = cell['summary'], cell['telemetry'], cell['intervention']
        if s['status'] not in ('COMPLETE_64','NUMERICAL_FAILURE'):
            raise ValueError('nonterminal or technical failure')
        attempts = [int(r['attempted_iteration']) for r in rows]
        if attempts != list(range(4001, attempts[-1]+1)):
            raise ValueError('noncontiguous attempts')
        if s['status'] == 'COMPLETE_64' and attempts[-1] != 4064:
            raise ValueError('completed cell has wrong budget')
        if operation in ('K','R') and s.get('original_replay') != 'MATCH':
            raise ValueError('unmatched original control')
        if cell['first_forward']['model_input_dtypes'] != ['torch.float16']*2:
            raise ValueError('precision mismatch')
        before, after = intervention['before'], intervention['after']
        if before != index[seed,arm,'K']['intervention']['before']:
            raise ValueError('different source optimizer summaries')
        if operation == 'K' and before != after:
            raise ValueError('K changed optimizer')
        if operation == 'R' and (after['parameter_states'] or after['step_values'] or any(after['moment_norms'].values())):
            raise ValueError('R did not fully clear state')
        if operation == 'clear_moments' and (before['step_values'] != after['step_values'] or any(after['moment_norms'].values())):
            raise ValueError('clear_moments changed step or kept moments')
        if operation == 'reset_step' and (after['step_values'] != [0.] or before['moment_norms'] != after['moment_norms']):
            raise ValueError('reset_step changed moments or kept clock')
        baseline = {int(r['attempted_iteration']):r for r in index[seed,'A','K']['telemetry']}
        for r in rows:
            if any(r[k] != baseline[int(r['attempted_iteration'])][k] for k in r if k.endswith('sha256')):
                raise ValueError('saved common-random-input fields differ')
        first = rows[0]
        if int(first['step_skipped']) or s['first_actual_attempt'] != 4001:
            raise ValueError('first-update comparison uses different attempts')
        ratio = float(first['update_norm'])/float(index[seed,arm,'K']['telemetry'][0]['update_norm'])
        if not math.isclose(ratio,s['first_update_norm_ratio_to_K'],rel_tol=1e-10):
            raise ValueError('vector norm ratio differs from telemetry')
        if sum(int(r['step_skipped']) for r in rows) != s['skipped_attempts']:
            raise ValueError('skip count mismatch')
        if len(cell['optimizer_steps']) != len(rows)-s['skipped_attempts']:
            raise ValueError('actual step records do not reconcile')
    groups = {}
    for o in OPERATIONS:
        ss = [c['summary'] for c in cells if c['summary']['operation']==o]
        ratios = [s['first_update_norm_ratio_to_K'] for s in ss]
        cosines = [s['first_update_cosine_to_K'] for s in ss]
        groups[o] = dict(completed=sum(s['status']=='COMPLETE_64' for s in ss),
                         failed=sum(s['status']=='NUMERICAL_FAILURE' for s in ss),
                         cells_with_skips=sum(s['skipped_attempts']>0 for s in ss),
                         skipped_attempts=sum(s['skipped_attempts'] for s in ss),
                         first_update_ratio_range=[min(ratios),max(ratios)],
                         first_update_ratio_median=st.median(ratios),
                         cosine_range=[min(cosines),max(cosines)])
    seconds = sum((dt.datetime.fromisoformat(c['summary']['ended_utc'])-
                   dt.datetime.fromisoformat(c['summary']['started_utc'])).total_seconds() for c in cells)
    return dict(groups=groups, total_observed_attempts=sum(len(c['telemetry']) for c in cells),
                summed_cell_wall_hours=seconds/3600,
                gpu_cells=dict(Counter(c['summary']['gpu'] for c in cells)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('evidence',type=Path)
    args=p.parse_args()
    opener = gzip.open if args.evidence.suffix == '.gz' else open
    with opener(args.evidence, 'rt') as handle:
        data = json.load(handle)
    print(json.dumps(analyze(data),indent=2))


if __name__ == '__main__':
    main()
