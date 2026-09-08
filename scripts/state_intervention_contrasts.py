"""Protocol arithmetic for validated scalar rows; does not verify source receipts."""
import math
import statistics

from scripts.summarize_m1_results import _interval

SEEDS = (55, 56, 59, 60, 61, 62)
STATUSES = {'PASS', 'NO_ENDPOINT', 'TECHNICAL_UNRESOLVED', 'EVAL_NONFINITE',
            'NOT_RUN_RESOURCE_LIMIT', 'PLANNED', 'RUNNING'}


def aggregate_cells(rows, metric, readout='E_512', blocks=('B0', 'B1', 'B2')):
    if metric not in ('fid50k_full', 'kid50k_full'):
        raise ValueError('unsupported metric')
    cells = {}
    for row in rows:
        if row['readout'] != readout or row['block'] not in blocks:
            continue
        key = (int(row['seed']), row['branch'], int(row['budget_kimg']))
        cell = cells.setdefault(key, {})
        if row['block'] in cell:
            raise ValueError(f'duplicate generation block: {key}')
        if row['status'] not in STATUSES:
            raise ValueError('unknown evaluation status')
        value = None
        if row['status'] == 'PASS':
            value = float(row[metric])
            if not math.isfinite(value) or (metric == 'fid50k_full' and value <= 0):
                raise ValueError('PASS row has an invalid metric')
            if metric == 'fid50k_full':
                value = math.log(value)
        cell[row['block']] = value
    return {key: statistics.fmean(cell.values())
            if set(cell) == set(blocks) and all(v is not None for v in cell.values()) else None
            for key, cell in cells.items()}


def paired_statistics(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, sample_sd=None, ci95=None,
                    ci_direction='INSUFFICIENT_COMPLETE_PAIRS')
    if len(values) > 6 or any(not math.isfinite(v) for v in values):
        raise ValueError('requires at most six finite seed-level values')
    result = dict(n=len(values), **_interval(values))
    if len(values) < 2:
        direction = 'INSUFFICIENT_COMPLETE_PAIRS'
    elif result['sample_sd'] == 0:
        direction = 'DEGENERATE_SD'
    else:
        lo, hi = result['ci95']
        direction = 'POSITIVE' if lo > 0 else 'NEGATIVE' if hi < 0 else 'INCONCLUSIVE'
    return dict(result, ci_direction=direction)


def m2_components(cell, early_budget):
    late = {arm: cell[('L_' + arm, 1024)] - cell[('K_' + arm, 1024)] for arm in ('A', 'B')}
    early = {arm: cell[('R_' + arm, early_budget)] - cell[('K_' + arm, early_budget)]
             for arm in ('A', 'B')}
    ta, tb = late['A'] - early['A'], late['B'] - early['B']
    return dict(d_A_512=early['A'], d_B_512=early['B'], d_A_768=late['A'],
                d_B_768=late['B'], J_512=early['B'] - early['A'],
                J_768=late['B'] - late['A'], T_A=ta, T_B=tb, Theta=tb - ta)


def swap_components(cell):
    aa, bb = cell[('K_A', 1024)], cell[('K_B', 1024)]
    ab, ba = cell[('X_A_from_B', 1024)], cell[('X_B_from_A', 1024)]
    da, db = ab - aa, bb - ba
    return dict(D_A=da, D_B=db, P_match=(da - db) / 2, T=bb - aa)


def summarize_cells(cells, *, include_same_chase=True):
    se_keys = {(b, 1024) for b in ('K_A', 'K_B', 'R_A', 'R_B', 'L_A', 'L_B')}
    sc_keys = {(b, 1024) for b in ('K_A', 'K_B', 'L_A', 'L_B')}
    sc_keys |= {(b, 768) for b in ('K_A', 'K_B', 'R_A', 'R_B')}
    swap_keys = {(b, 1024) for b in ('K_A', 'K_B', 'X_A_from_B', 'X_B_from_A')}
    definitions = {'m2_same_endpoint': se_keys, 'swap_common': swap_keys,
        'swap_A_all_pairs': {('K_A', 1024), ('X_A_from_B', 1024)},
        'swap_B_all_pairs': {('K_B', 1024), ('X_B_from_A', 1024)}}
    if include_same_chase:
        definitions.update(m2_same_chase=sc_keys, m2_time_intersection=se_keys | sc_keys)
    output = {}
    for name, required in definitions.items():
        per_seed, excluded = [], []
        for seed in SEEDS:
            missing = sorted(k for k in required if cells.get((seed, *k)) is None)
            if missing:
                excluded.append(dict(seed=seed, missing_cells=missing))
                continue
            cell = {k: cells[(seed, *k)] for k in required}
            if name == 'm2_same_endpoint':
                values = m2_components(cell, 1024)
            elif name == 'm2_same_chase':
                values = m2_components(cell, 768)
            elif name == 'm2_time_intersection':
                values = {prefix + key: value for prefix, budget in (('SE_', 1024), ('SC_', 768))
                          for key, value in m2_components(cell, budget).items()}
            elif name == 'swap_common':
                values = swap_components(cell)
            elif name == 'swap_A_all_pairs':
                values = dict(D_A=cell[('X_A_from_B', 1024)] - cell[('K_A', 1024)])
            else:
                values = dict(D_B=cell[('K_B', 1024)] - cell[('X_B_from_A', 1024)])
            per_seed.append(dict(seed=seed, **values))
        fields = [k for k in per_seed[0] if k != 'seed'] if per_seed else []
        output[name] = dict(seeds=[r['seed'] for r in per_seed], n=len(per_seed),
            per_seed=per_seed, excluded=excluded,
            estimates={k: paired_statistics(r[k] for r in per_seed) for k in fields})
    return output
