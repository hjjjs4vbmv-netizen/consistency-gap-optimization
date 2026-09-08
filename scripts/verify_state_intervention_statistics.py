"""Independent NumPy/SciPy recomputation of the completed six-seed result tables."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def verify(raw, estimates):
    checks = 0
    largest = 0.0
    seeds = (55, 56, 59, 60, 61, 62)
    for basis, metrics in estimates.items():
        readout = 'E_512' if basis.startswith('E_512') else basis[:-3]
        blocks = ('B0', 'B1', 'B2') if basis.endswith('three_blocks') else ('B0',)
        for metric, analyses in metrics.items():
            cells = {}
            for seed in seeds:
                for branch, budget in {(r['branch'], r['budget_kimg']) for r in raw}:
                    group = [r for r in raw if (r['seed'], r['branch'], r['budget_kimg'], r['readout'])
                             == (seed, branch, budget, readout) and r['block'] in blocks]
                    if not group:
                        continue
                    assert len(group) == len(blocks) and all(r['status'] == 'PASS' for r in group)
                    values = np.array([r[metric] for r in group], dtype=float)
                    cells[seed, branch, budget] = np.mean(np.log(values) if metric == 'fid50k_full' else values)
            for name, analysis in analyses.items():
                assert analysis['seeds'] == list(seeds) and analysis['excluded'] == []
                independent = []
                for seed in seeds:
                    def y(branch, budget=1024):
                        return cells[seed, branch, budget]
                    if name.startswith('swap'):
                        da, db = y('X_A_from_B') - y('K_A'), y('K_B') - y('X_B_from_A')
                        values = {'D_A': da, 'D_B': db, 'P_match': (da - db) / 2,
                                  'T': y('K_B') - y('K_A')}
                    else:
                        values = {}
                        budgets = [('SE_', 1024), ('SC_', 768)] if name == 'm2_time_intersection' else [
                            ('', 768 if name == 'm2_same_chase' else 1024)]
                        for prefix, budget in budgets:
                            early = np.array([y('R_A', budget)-y('K_A', budget),
                                              y('R_B', budget)-y('K_B', budget)])
                            late = np.array([y('L_A')-y('K_A'), y('L_B')-y('K_B')])
                            vals = dict(d_A_512=early[0], d_B_512=early[1], d_A_768=late[0],
                                d_B_768=late[1], J_512=np.diff(early)[0], J_768=np.diff(late)[0],
                                T_A=(late-early)[0], T_B=(late-early)[1], Theta=np.diff(late-early)[0])
                            values.update({prefix+k:v for k,v in vals.items()})
                    independent.append(values)
                for field, reported in analysis['estimates'].items():
                    values = np.array([r[field] for r in independent])
                    np.testing.assert_allclose(values, [r[field] for r in analysis['per_seed']], atol=1e-12, rtol=0)
                    mean, sd = float(values.mean()), float(values.std(ddof=1))
                    lo, hi = mean + np.array([-1, 1]) * t.ppf(.975, 5) * sd / np.sqrt(6)
                    np.testing.assert_allclose([mean, sd, lo, hi],
                        [reported['mean'], reported['sample_sd'], *reported['ci95']], atol=1e-9, rtol=0)
                    expected = 'POSITIVE' if lo > 0 else 'NEGATIVE' if hi < 0 else 'INCONCLUSIVE'
                    assert reported['ci_direction'] == expected
                    largest = max(largest, abs(lo-reported['ci95'][0]), abs(hi-reported['ci95'][1]))
                    checks += 1
    return dict(status='PASS', estimates_checked=checks, training_seed_n=6,
                maximum_ci_absolute_difference=largest, method='independent NumPy aggregation and scipy.stats.t')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--estimates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.raw.read_text()), json.loads(args.estimates.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
