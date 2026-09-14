"""Verify the published F12 tables on CPU, without training or evaluation access."""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import math
import platform
import re
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

BUNDLE = Path(__file__).resolve().parent
SEEDS = tuple(range(401, 413))
ARMS = ('AA', 'DA', 'A_startup_down5', 'D_startup_up5')
CONTRASTS = ('H', 'M', 'C', 'S')
BLOCKS = ('B0', 'B1', 'B2')
ATOL = 1e-12
DIFFERENCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name):
    return json.loads((BUNDLE / name).read_text())


def table(name):
    with (BUNDLE / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label):
    """Compare all numeric fields, with an explicit cross-runtime float tolerance."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{label}: fields differ')
        for key in expected:
            compare(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{label}: lengths differ')
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f'{label}[{i}]')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected, f'{label}: value differs')
    else:
        difference = abs(float(actual) - float(expected))
        require(math.isfinite(difference) and difference <= ATOL, f'{label}: numeric mismatch {difference}')
        DIFFERENCES[label] = difference


def verify_files():
    manifest = {}
    for line in (BUNDLE / 'PUBLIC_SHA256SUMS.txt').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        require(relative not in manifest, 'duplicate checksum entry')
        path = BUNDLE / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(BUNDLE), 'unsafe checksum path')
        require(digest(path) == expected, f'{relative}: checksum mismatch')
        manifest[relative] = expected
    actual = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'PUBLIC_SHA256SUMS.txt'}
    require(actual == set(manifest), 'checksum manifest does not cover the exact bundle')
    provenance = load('PROVENANCE.json')
    freeze = load('frozen/source_freeze.json')
    for name, description in provenance['publication_transforms'].items():
        if description == 'Unmodified byte-for-byte copy.':
            require(digest(BUNDLE / name) == provenance['original_artifacts'][name], f'{name}: original differs')
    require(digest(BUNDLE / 'frozen/source_freeze.json') == provenance['original_artifacts']['source_freeze.json'],
            'original source freeze differs')
    for name, source in provenance['frozen_copies'].items():
        require(digest(BUNDLE / name) == source['sha256'] == freeze['source_hashes'][source['source_path']],
                f'{name}: original frozen source differs')
    require(freeze['protocol']['cohort'] == list(SEEDS) and freeze['protocol']['arms'] == list(ARMS), 'cohort differs')
    require(provenance['reserve_activated'] is False, 'reserve unexpectedly activated')
    return len(manifest), freeze


def paired(values, equivalence=False):
    x = np.asarray(values, dtype=np.float64)
    test = stats.ttest_1samp(x, popmean=0)
    ci95 = test.confidence_interval(confidence_level=.95)
    ci90 = test.confidence_interval(confidence_level=.90)
    mean = float(x.mean())
    result = dict(n=len(x), mean_log=mean, se_log=float(stats.sem(x)),
                  ci95_log=[float(ci95.low), float(ci95.high)],
                  ci90_log=[float(ci90.low), float(ci90.high)],
                  p_two_sided=float(test.pvalue), geometric_ratio=math.exp(mean),
                  ci95_ratio=[math.exp(ci95.low), math.exp(ci95.high)])
    if equivalence:
        bound = math.log(1.03)
        lower = stats.ttest_1samp(x, -bound, alternative='greater').pvalue
        upper = stats.ttest_1samp(x, bound, alternative='less').pvalue
        result['TOST'] = dict(alpha=.05, bounds_log=[-bound, bound], p=float(max(lower, upper)),
                              equivalent=bool(ci90.low > -bound and ci90.high < bound))
    return result


def reconstruct_blocks(freeze):
    rows = table('blocks.csv')
    expected = set(itertools.product(SEEDS, ARMS, (1, 2), BLOCKS))
    indexed = {}
    receipts = set()
    evaluation = freeze['protocol']['evaluation']
    for row in rows:
        seed, arm, nfe, block = int(row['seed']), row['arm'], int(row['nfe']), row['block']
        key = (seed, arm, nfe, block)
        require(key in expected and key not in indexed, 'duplicate or unexpected evaluation identity')
        require(row['status'] == 'PASS', 'non-PASS block')
        require(float(row['FID']) > 0 and math.isfinite(float(row['FID'])), 'invalid FID')
        require(math.isfinite(float(row['KID'])), 'invalid KID')
        require([int(row['sample_seed_start']), int(row['sample_seed_end'])] == evaluation['blocks'][block], 'generation range differs')
        for field in ['readout', 'precision', 'evaluator_commit']:
            require(row[field] == evaluation[field], f'{field} differs')
        require(int(row['metric_seed']) == evaluation['metric_seed'], 'metric seed differs')
        require(row['receipt_id'] == f'seed{seed}/{arm}/NFE{nfe}/{block}', 'receipt ID differs')
        for field in ['source_sha256', 'checkpoint_sha256', 'snapshot_sha256', 'feature_sha256']:
            require(re.fullmatch('[0-9a-f]{64}', row[field]) is not None, f'invalid {field}')
        receipts.add(row['source_sha256'])
        indexed[key] = row
    require(set(indexed) == expected and len(receipts) == 288, 'incomplete block matrix or duplicated receipts')
    per_seed = []
    for seed in SEEDS:
        row = dict(seed=seed)
        for arm in ARMS:
            require(len({indexed[(seed, arm, nfe, block)]['checkpoint_sha256'] for nfe in (1, 2) for block in BLOCKS}) == 1,
                    f'endpoint binding differs: {seed} {arm}')
            # Frozen worker exports E_512.pkl separately under evaluation/NFE1
            # and evaluation/NFE2. Pickle-byte hashes need only agree within NFE.
            for nfe in (1, 2):
                require(len({indexed[(seed, arm, nfe, block)]['snapshot_sha256'] for block in BLOCKS}) == 1,
                        f'NFE{nfe} export binding differs: {seed} {arm}')
            row[arm] = float(np.mean([math.log(float(indexed[(seed, arm, 1, block)]['FID'])) for block in BLOCKS]))
        row.update(H=row['DA']-row['AA'], M=row['A_startup_down5']-row['AA'], C=row['D_startup_up5']-row['DA'])
        row['S'] = (row['C']-row['M']) / 2
        per_seed.append(row)
    published = [{key: int(value) if key == 'seed' else float(value) for key, value in row.items()}
                 for row in table('per_seed.csv')]
    compare(per_seed, published, 'per_seed')
    fids = [{key: int(value) if key == 'seed' else float(value) for key, value in row.items()}
            for row in table('per_seed_fid.csv')]
    compare([dict(seed=row['seed'], **{a: math.exp(row[a]) for a in ARMS}, H_percent=100*math.expm1(row['H']))
             for row in per_seed], fids, 'per_seed_fid')
    return rows, per_seed


def verify_engineering():
    rows = table('engineering_paths.csv')
    indexed = {(int(row['seed']), row['arm']): row for row in rows}
    require(len(rows) == len(indexed) == 48 and set(indexed) == set(itertools.product(SEEDS, ARMS)), 'training matrix differs')
    for row in rows:
        require(row['clock_first10_ok'] == row['lr_window_first10_ok'] == 'True', 'clock or LR audit failed')
        require(int(row['attempted']) == 8000 == int(row['successful']) + int(row['amp_skips']), 'attempt accounting differs')
    complete = table('completion.csv')
    require(len(complete) == 48 and {(int(r['seed']), r['arm']) for r in complete} == set(indexed), 'completion matrix differs')
    for row in complete:
        require(row['training_status'] == 'PASS' and row['valid'] == 'True' and
                ast.literal_eval(row['evaluation_statuses']) == ['PASS']*3, 'incomplete primary path')
    pairs = table('engineering_pairs.csv')
    require(len(pairs) == 12 and {int(r['seed']) for r in pairs} == set(SEEDS), 'pair table differs')
    down, up = [], []
    for row in pairs:
        seed = int(row['seed'])
        norms = {arm: float(indexed[(seed, arm)]['first5_norm_sum']) for arm in ARMS}
        a_ratio, d_ratio = norms[ARMS[2]]/norms['AA'], norms[ARMS[3]]/norms['DA']
        compare(a_ratio, float(row['A_down5_to_AA_norm_sum_ratio']), f'engineering.{seed}.A_ratio')
        compare(d_ratio, float(row['D_up5_to_DA_norm_sum_ratio']), f'engineering.{seed}.D_ratio')
        if a_ratio >= 1: down.append(seed)
        if d_ratio <= 1: up.append(seed)
        require(row['same_stochastic_inputs'] == row['within_0p1_percent_relative'] == 'True', 'input or norm-ratio audit failed')
        require(abs(float(row['first_common_DA_to_AA_norm_ratio'])/(1/1.1)-1) <= .001, 'common update norm ratio differs')
    snapshots = table('switch_snapshot_hashes.csv')
    names = ('training-state-kimg000512.pt.sha256.json', 'training-state-branch-init-kimg000512.pt.sha256.json')
    require(len(snapshots) == 96 and {(int(r['seed']), r['arm'], r['snapshot']) for r in snapshots} ==
            set(itertools.product(SEEDS, ARMS, names)), 'switch snapshots incomplete')
    require(all(r['attempted_iteration'] == '4000' and re.fullmatch('[0-9a-f]{64}', r['sha256']) for r in snapshots), 'switch binding invalid')
    engineering = load('engineering_summary.json')
    compare(dict(paths=48, clock_checks_pass=48, lr_window_checks_pass=48,
                 A_down_first5_lower=12-len(down), D_up_first5_higher=12-len(up),
                 first_common_ratio_within_tolerance=12, same_stochastic_inputs=12,
                 snapshot_hashes_present=96, total_amp_skips=sum(int(r['amp_skips']) for r in rows)),
            engineering['summary'], 'engineering_summary')
    compare(dict(A_startup_down5=down, D_startup_up5=up), engineering['direction_exceptions'], 'direction_exceptions')
    failures = table('failure_counts.csv')
    require(len(failures) == 4 and {r['arm'] for r in failures} == set(ARMS), 'failure table differs')
    for row in failures:
        require([int(row[k]) for k in ['training_pass', 'NFE1_pass', 'NFE2_pass']] == [12, 36, 36], 'PASS counts differ')
        require(all(int(row[k]) == 0 for k in row if 'failure' in k), 'unexpected final failure')
        require(int(row['amp_skips']) == sum(int(r['amp_skips']) for r in rows if r['arm'] == row['arm']), 'arm skip sum differs')


def main():
    files, freeze = verify_files()
    blocks, per_seed = reconstruct_blocks(freeze)
    effects = {k: paired([r[k] for r in per_seed], equivalence=k == 'H') for k in CONTRASTS}
    running = 0.0
    for i, name in enumerate(sorted(('M', 'C', 'S'), key=lambda k: effects[k]['p_two_sided'])):
        running = max(running, (3-i)*effects[name]['p_two_sided'])
        effects[name]['p_holm_M_C_S'] = min(1.0, running)
    effects['S']['ratio_interpretation'] = 'square root of ratio of FID ratios; not a model FID percent improvement'
    original = load('statistics.json')
    compare(effects, original['effects'], 'effects')
    compare({k: paired([r[k] for r in per_seed]) for k in CONTRASTS}, original['available_pairs_descriptive'], 'available_pairs')
    require(original['complete_n'] == original['planned_n'] == 12 and original['underpowered'] is False, 'sample count differs')
    h = effects['H']
    decision = dict(direction='negative' if h['ci95_log'][1] < 0 else 'positive' if h['ci95_log'][0] > 0 else 'unresolved',
                    equivalence='equivalent' if h['TOST']['equivalent'] else 'not_demonstrated')
    compare(decision, original['decision'], 'decision')
    descriptive = {}
    for nfe in (1, 2):
        descriptive[f'NFE{nfe}'] = {}
        for arm in ARMS:
            subset = [r for r in blocks if int(r['nfe']) == nfe and r['arm'] == arm]
            descriptive[f'NFE{nfe}'][arm] = dict(FID_geometric_mean=math.exp(float(np.mean([math.log(float(r['FID'])) for r in subset]))),
                KID_mean_raw=float(np.mean([float(r['KID']) for r in subset])), blocks=len(subset))
    compare(descriptive, load('descriptive.json')['metrics'], 'descriptive')
    compare(sum(float(r['process_gpuh']) for r in blocks), load('descriptive.json')['process_gpu_hours']['evaluation'], 'evaluation_gpuh')
    verify_engineering()
    print(json.dumps(dict(status='PASS', scope='Portable table-level analysis and evidence consistency; no GPU rerun',
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.system()),
        verified_public_files=files, training_paths=48, evaluation_blocks=288, complete_seeds=12,
        numeric_comparisons=len(DIFFERENCES), numeric_atol=ATOL, max_abs_difference=max(DIFFERENCES.values()),
        main=effects['H'], secondary_holm={k: effects[k]['p_holm_M_C_S'] for k in ('M', 'C', 'S')},
        engineering=load('engineering_summary.json')['summary']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
