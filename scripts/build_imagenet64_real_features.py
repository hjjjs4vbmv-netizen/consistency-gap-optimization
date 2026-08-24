#!/usr/bin/env python3
"""Build the canonical ImageNet-64 Inception feature bank on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REAL_COUNT = 1_281_167
FEATURE_DIM = 2_048
OFFICIAL_REFERENCE_URL = (
    'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl'
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_official_reference(source):
    if str(source).startswith(('https://', 'http://')):
        import dnnlib

        with dnnlib.util.open_url(str(source), verbose=True) as handle:
            payload = handle.read()
        reference_sha256 = hashlib.sha256(payload).hexdigest()
        reference = pickle.loads(payload)
    else:
        source = Path(source)
        reference_sha256 = sha256_file(source)
        with source.open('rb') as handle:
            reference = pickle.load(handle)
    fid = reference['fid']
    if (
        reference.get('num_images') != REAL_COUNT
        or np.asarray(fid['mu']).shape != (FEATURE_DIM,)
        or np.asarray(fid['sigma']).shape != (FEATURE_DIM, FEATURE_DIM)
        or not np.isfinite(fid['mu']).all()
        or not np.isfinite(fid['sigma']).all()
    ):
        raise RuntimeError('official img64.pkl has unexpected FID statistics')
    return fid, reference_sha256


def build_feature_bank(
    data_path, output_path, reference_source, receipt_path,
    batch_size, workers,
):
    import torch

    from metrics import frechet_inception_distance, metric_utils
    from training.dataset import ImageFolderDataset

    if output_path == receipt_path:
        raise ValueError('--output and --receipt must be different paths')
    for path in (output_path, receipt_path):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite: {path}')
    reference, reference_sha256 = load_official_reference(reference_source)
    dataset = ImageFolderDataset(
        path=str(data_path), resolution=64, use_labels=False,
        xflip=False, cache=False,
    )
    if (
        len(dataset) != REAL_COUNT
        or dataset.num_channels != 3
        or dataset.resolution != 64
    ):
        raise RuntimeError(
            f'canonical ImageNet-64 dataset must contain {REAL_COUNT} RGB images'
        )

    device = torch.device('cuda')
    detector = metric_utils.get_feature_detector(
        metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
        device=device,
        num_gpus=1,
        rank=0,
        verbose=True,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    if workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        output_path.name + f'.{uuid.uuid4().hex}.npy'
    )
    features_out = np.lib.format.open_memmap(
        temp_path, mode='w+', dtype=np.float32,
        shape=(REAL_COUNT, FEATURE_DIM),
    )
    raw_mean = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=device)
    raw_cov = torch.zeros(
        [FEATURE_DIM, FEATURE_DIM], dtype=torch.float64, device=device,
    )
    offset = 0
    try:
        with torch.no_grad():
            for images, _labels in loader:
                batch_features = detector(
                    images.to(device), return_features=True,
                ).to(torch.float32)
                if (
                    batch_features.ndim != 2
                    or batch_features.shape[1] != FEATURE_DIM
                    or not torch.isfinite(batch_features).all()
                ):
                    raise RuntimeError('Inception returned invalid features')
                end = offset + batch_features.shape[0]
                features_out[offset:end] = batch_features.cpu().numpy()
                features64 = batch_features.to(torch.float64)
                raw_mean += features64.sum(0)
                raw_cov += features64.T @ features64
                offset = end
                if offset % 100_000 < batch_features.shape[0]:
                    print(f'features={offset}/{REAL_COUNT}', flush=True)
        if offset != REAL_COUNT:
            raise RuntimeError(f'extracted {offset} features, expected {REAL_COUNT}')
        features_out.flush()
        mean = (raw_mean / REAL_COUNT).cpu().numpy()
        covariance = (
            (raw_cov - raw_mean.ger(raw_mean) / REAL_COUNT)
            / (REAL_COUNT - 1)
        ).cpu().numpy()
        comparison = {
            'num_images': REAL_COUNT,
            'mean_max_abs_error': float(
                np.max(np.abs(mean - reference['mu']))
            ),
            'covariance_max_abs_error': float(
                np.max(np.abs(covariance - reference['sigma']))
            ),
            'fid_to_official_reference': (
                frechet_inception_distance.compute_fid_from_stats(
                    reference['mu'], reference['sigma'], mean, covariance,
                )
            ),
        }
        del features_out
        os.replace(temp_path, output_path)
        receipt = {
            'feature_bank': {
                'path': str(output_path),
                'sha256': sha256_file(output_path),
                'shape': [REAL_COUNT, FEATURE_DIM],
                'dtype': 'float32',
            },
            'inception_detector': metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
            'official_reference': {
                'path': str(reference_source),
                'sha256': reference_sha256,
            },
            'comparison': comparison,
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--official-reference', default=OFFICIAL_REFERENCE_URL,
        help='Local official img64.pkl; URL default is for engineering only',
    )
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0:
        parser.error('--batch must be positive and --workers must be nonnegative')
    import torch

    torch.multiprocessing.set_start_method('spawn')
    reference_source = args.official_reference
    if not reference_source.startswith(('https://', 'http://')):
        reference_source = str(Path(reference_source).resolve())
    receipt = build_feature_bank(
        args.data.resolve(),
        args.output.resolve(),
        reference_source,
        args.receipt.resolve(),
        args.batch,
        args.workers,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
