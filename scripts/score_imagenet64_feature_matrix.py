#!/usr/bin/env python3
"""Score the complete 108-job ImageNet-64 feature matrix in one shot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEEDS = (101, 102, 103)
METHODS = ('IA', 'IB')
MILESTONES = tuple(
    (iteration, iteration * 128 // 1000)
    for iteration in range(20_000, 100_001, 10_000)
)
NFES = (1, 2)
JOB_COUNT = len(SEEDS) * len(METHODS) * len(MILESTONES) * len(NFES)
REAL_COUNT = 1_281_167
GENERATED_COUNT = 50_000
FEATURE_DIM = 2_048
METRIC_SEED = 20_260_730


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_real_stats(path):
    import numpy as np

    with path.open('rb') as handle:
        reference = pickle.load(handle)
    fid = reference['fid']
    mean = np.asarray(fid['mu'])
    covariance = np.asarray(fid['sigma'])
    if (
        reference.get('num_images') != REAL_COUNT
        or mean.shape != (FEATURE_DIM,)
        or covariance.shape != (FEATURE_DIM, FEATURE_DIM)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError('real stats must be the canonical local img64.pkl')
    return mean, covariance


def feature_jobs(feature_root):
    jobs = []
    for iteration, kimg in MILESTONES:
        for seed in SEEDS:
            for method in METHODS:
                for nfe in NFES:
                    path = (
                        feature_root
                        / f'seed{seed}'
                        / method
                        / f'kimg{kimg:06d}'
                        / f'nfe{nfe}'
                        / 'features.final.npy'
                    )
                    jobs.append(dict(
                        seed=seed, method=method, iteration=iteration,
                        kimg=kimg, nfe=nfe, path=path,
                    ))
    return jobs


def require_complete_matrix(feature_root):
    jobs = feature_jobs(feature_root)
    missing = [job['path'] for job in jobs if not job['path'].is_file()]
    if missing:
        preview = ', '.join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f'feature matrix is incomplete: '
            f'missing {len(missing)}/{JOB_COUNT}; {preview}'
        )
    return jobs


def score_matrix(feature_root, real_features_path, real_stats_path):
    jobs = require_complete_matrix(feature_root)
    from metrics import (
        frechet_inception_distance,
        kernel_inception_distance,
        metric_utils,
    )

    real_stats_sha256 = sha256_file(real_stats_path)
    real_mean, real_cov = load_real_stats(real_stats_path)
    real_features_sha256 = sha256_file(real_features_path)
    real_features = metric_utils._load_precomputed_features(
        real_features_path, expected_items=REAL_COUNT,
    )
    if real_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f'real features must have shape ({REAL_COUNT}, {FEATURE_DIM})'
        )
    rows = []
    for job in jobs:
        generated_sha256 = sha256_file(job['path'])
        generated_features = metric_utils._load_precomputed_features(
            job['path'], expected_items=GENERATED_COUNT,
        )
        if generated_features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"{job['path']} must have shape "
                f'({GENERATED_COUNT}, {FEATURE_DIM})'
            )
        generated_mean, generated_cov = metric_utils.compute_feature_mean_cov(
            generated_features, unbiased=True,
        )
        fid = frechet_inception_distance.compute_fid_from_stats(
            real_mean, real_cov, generated_mean, generated_cov,
        )
        kid = kernel_inception_distance.compute_kid_from_features(
            real_features,
            generated_features,
            num_subsets=100,
            max_subset_size=1000,
            random_seed=METRIC_SEED,
        )
        rows.append({
            'seed': job['seed'],
            'method': job['method'],
            'iteration': job['iteration'],
            'kimg': job['kimg'],
            'nfe': job['nfe'],
            'generated_features': {
                'path': str(job['path']),
                'sha256': generated_sha256,
            },
            'fid50k': fid,
            'kid50k': kid,
        })
    return {
        'metric_seed': METRIC_SEED,
        'inception_detector': metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
        'real_features': {
            'path': str(real_features_path),
            'sha256': real_features_sha256,
        },
        'real_stats': {
            'path': str(real_stats_path),
            'sha256': real_stats_sha256,
        },
        'real_count': REAL_COUNT,
        'generated_count_per_job': GENERATED_COUNT,
        'feature_dim': FEATURE_DIM,
        'kid_num_subsets': 100,
        'kid_subset_size': 1000,
        'jobs_expected': JOB_COUNT,
        'jobs_scored': len(rows),
        'results': rows,
    }


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f'.{uuid.uuid4().hex}')
    try:
        with temp_path.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--real-features', type=Path, required=True)
    parser.add_argument('--real-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f'refusing to overwrite: {output_path}')
    payload = score_matrix(
        args.feature_root.resolve(),
        args.real_features.resolve(),
        args.real_stats.resolve(),
    )
    if payload['jobs_scored'] != JOB_COUNT:
        raise RuntimeError('refusing to publish a partial scoring result')
    atomic_write_json(output_path, payload)
    print(f'scored={JOB_COUNT} output={output_path}')


if __name__ == '__main__':
    main()
