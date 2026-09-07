"""Read-only M1 summary from the ECT receipt extract; prints JSON to stdout."""
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.summarize_m1_results import _interval

rows = json.loads(Path(__file__).with_name('endpoint_metrics.json').read_text())
assert len(rows) == 380 and len({(r[0], r[1]) for r in rows}) == 380
assert Counter((r[0], r[6]) for r in rows) == {
    ('original', 'PASS'): 260, ('original', 'NO_ENDPOINT'): 60, ('repair', 'PASS'): 60}
assert all(math.isfinite(v) for r in rows if r[6] == 'PASS' for v in r[7].values())
assert all(r[7]['fid50k_full'] > 0 for r in rows if r[6] == 'PASS')
original = {(r[2], r[3], r[4], r[5]): r[7] for r in rows
            if r[0] == 'original' and r[6] == 'PASS'}
blocks = ('B0', 'B1', 'B2')

def differences(a, b, readout='E_512', use_blocks=blocks, metric='fid50k_full',
                transform=math.log, seeds=range(50, 66)):
    return {s: st.fmean(transform(original[s, b, readout, k][metric]) -
                       transform(original[s, a, readout, k][metric]) for k in use_blocks)
            for s in seeds if all((s, arm, readout, k) in original
                                  for arm in (a, b) for k in use_blocks)}

def describe(values, log_fid=True):
    result = dict(n=len(values), per_seed=values, **_interval(list(values.values())))
    if log_fid:
        result['fid_ratio'] = math.exp(result['mean'])
        result['fid_ratio_ci95'] = [math.exp(v) for v in result['ci95']]
    return result

r = differences('R_A', 'R_B')
k = differences('K_A', 'K_B')
four = sorted(set(r) & set(k))
primary = describe(r)
lo, hi = primary['ci95']
primary['status'] = ('B_ADVANTAGE_SUPPORTED_CONDITIONAL' if hi < 0 else
                     'B_DISADVANTAGE_SUPPORTED_CONDITIONAL' if lo > 0 else 'INCONCLUSIVE')
primary['denominator'] = 16
primary['directions'] = dict(B_better=sum(v < 0 for v in r.values()),
                             A_better=sum(v > 0 for v in r.values()))
output = dict(primary=primary, K_E512_descriptive=describe(k),
              interaction=describe({s: r[s] - k[s] for s in four}, False),
              R_on_four=describe({s: r[s] for s in four}),
              K_on_four=describe({s: k[s] for s in four}),
              KID_R_descriptive=describe(differences('R_A', 'R_B',
                  metric='kid50k_full', transform=float), False))
output['b0_descriptive'] = {f'{kind}_{e}': describe(differences(
    kind + '_A', kind + '_B', readout=e, use_blocks=('B0',)))
    for kind in ('K', 'R') for e in ('E_KEEP', 'E_512', 'ONLINE')}
output['primary_raw'] = [{
    'seed': s, 'R_A_fid_blocks': [original[s, 'R_A', 'E_512', b]['fid50k_full'] for b in blocks],
    'R_B_fid_blocks': [original[s, 'R_B', 'E_512', b]['fid50k_full'] for b in blocks],
    'd': r[s]} for s in r]
output['original_failures'] = sorted({(r[2], r[3]) for r in rows if r[6] == 'NO_ENDPOINT'})
output['repair_descriptive'] = [{
    'seed': s, 'branch': a, 'mean_fid_E512': st.fmean(x[7]['fid50k_full'] for x in rows
        if x[0] == 'repair' and x[2] == s and x[3] == a and x[4] == 'E_512')}
    for s, a in sorted({(x[2], x[3]) for x in rows if x[0] == 'repair'})]
print(json.dumps(output, indent=2, ensure_ascii=False))
