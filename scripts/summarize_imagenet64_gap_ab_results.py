#!/usr/bin/env python3
"""Summarize paired IA-minus-IB ImageNet-64 FID and KID results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts import score_imagenet64_feature_matrix as scorer


KEY_FIELDS = ('kimg', 'seed', 'method', 'nfe')


def result_key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def load_results(paths):
    results = {}
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        for row in payload['results']:
            key = result_key(row)
            previous = results.get(key)
            if previous is not None:
                if any(previous[field] != row[field] for field in ('fid50k', 'kid50k')):
                    raise ValueError(f'conflicting result for {key}')
                previous_sha = previous.get('generated_features', {}).get('sha256')
                row_sha = row.get('generated_features', {}).get('sha256')
                if previous_sha and row_sha and previous_sha != row_sha:
                    raise ValueError(f'conflicting generated features for {key}')
                if previous_sha and not row_sha:
                    continue
            results[key] = row
    return results


def expected_keys():
    return {
        result_key(job)
        for job in scorer.w2_revised_jobs(Path('.'))
    }


def require_revised_matrix(results):
    expected = expected_keys()
    actual = set(results)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f'revised W2 matrix mismatch: missing={len(missing)} '
            f'extra={len(extra)}'
        )


def paired_differences(results):
    paired = []
    groups = sorted({(kimg, seed, nfe) for kimg, seed, _method, nfe in results})
    for kimg, seed, nfe in groups:
        ia = results.get((kimg, seed, 'IA', nfe))
        ib = results.get((kimg, seed, 'IB', nfe))
        if ia is None or ib is None:
            continue
        paired.append({
            'kimg': kimg,
            'seed': seed,
            'nfe': nfe,
            'ia_fid50k': ia['fid50k'],
            'ib_fid50k': ib['fid50k'],
            'delta_fid50k_ia_minus_ib': ia['fid50k'] - ib['fid50k'],
            'ia_kid50k': ia['kid50k'],
            'ib_kid50k': ib['kid50k'],
            'delta_kid50k_ia_minus_ib': ia['kid50k'] - ib['kid50k'],
        })
    return paired


def paired_summary(paired):
    summaries = []
    groups = sorted({(row['kimg'], row['nfe']) for row in paired})
    for kimg, nfe in groups:
        rows = [row for row in paired if row['kimg'] == kimg and row['nfe'] == nfe]
        fid_deltas = [row['delta_fid50k_ia_minus_ib'] for row in rows]
        kid_deltas = [row['delta_kid50k_ia_minus_ib'] for row in rows]
        summaries.append({
            'kimg': kimg,
            'nfe': nfe,
            'pair_count': len(rows),
            'ia_fid_wins': sum(delta < 0 for delta in fid_deltas),
            'mean_ia_fid50k': statistics.mean(row['ia_fid50k'] for row in rows),
            'mean_ib_fid50k': statistics.mean(row['ib_fid50k'] for row in rows),
            'mean_delta_fid50k_ia_minus_ib': statistics.mean(fid_deltas),
            'sample_sd_delta_fid50k_ia_minus_ib': (
                statistics.stdev(fid_deltas) if len(fid_deltas) > 1 else None
            ),
            'ia_kid_wins': sum(delta < 0 for delta in kid_deltas),
            'mean_ia_kid50k': statistics.mean(row['ia_kid50k'] for row in rows),
            'mean_ib_kid50k': statistics.mean(row['ib_kid50k'] for row in rows),
            'mean_delta_kid50k_ia_minus_ib': statistics.mean(kid_deltas),
            'sample_sd_delta_kid50k_ia_minus_ib': (
                statistics.stdev(kid_deltas) if len(kid_deltas) > 1 else None
            ),
        })
    return summaries


def write_csv(path, rows, fieldnames):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction='ignore',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    results = load_results(args.inputs)
    require_revised_matrix(results)
    rows = [results[key] for key in sorted(results)]
    paired = paired_differences(results)
    summaries = paired_summary(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / 'per_trajectory.csv', rows,
        ('kimg', 'seed', 'method', 'nfe', 'fid50k', 'kid50k'),
    )
    write_csv(
        args.output_dir / 'paired_differences.csv', paired,
        tuple(paired[0]),
    )
    write_csv(
        args.output_dir / 'paired_summary.csv', summaries,
        tuple(summaries[0]),
    )
    print(f'rows={len(rows)} pairs={len(paired)} summaries={len(summaries)}')


if __name__ == '__main__':
    main()
