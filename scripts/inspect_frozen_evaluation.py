"""Output checks reused from the M1 endpoint queue (commit 458e561)."""
import json
import math
from pathlib import Path

import numpy as np
from scripts import build_m1_evaluation_slots as slots


def inspect_output(row, job, snapshot, dataset):
    options = json.loads((job / 'training_options.json').read_text())
    expected = dict(sample_seeds=list(range(int(row['sample_seed_start']),
                                          int(row['sample_seed_end']) + 1)),
                    seed=slots.METRIC_SEED, metrics=['kid50k_full', 'fid50k_full'],
                    metric_repeats=1, metric_generator_batch=128,
                    retain_generated_artifacts=True, mid_t=[])
    if any(options.get(k) != v for k, v in expected.items()):
        raise ValueError('recorded evaluation options disagree with frozen command')
    if (options['network_kwargs']['use_fp16'] is not False
        or Path(options['resume_pkl']).resolve() != snapshot.resolve()
        or Path(options['dataset_kwargs']['path']).resolve() != dataset.resolve()):
        raise ValueError('snapshot, dataset or precision mismatch')
    if 'Exiting...' not in (job / 'log.txt').read_text():
        raise ValueError('missing evaluator completion marker')
    samples = np.load(job / 'generated-samples.npy', mmap_mode='r')
    if samples.shape[0] != 50000:
        raise ValueError('incomplete generated sample count')
    features, results = [], {}
    for metric in ('kid50k_full', 'fid50k_full'):
        features.append(np.load(job / f'generated-features-{metric}-repeat00.npy', mmap_mode='r'))
        records = [json.loads(s) for s in (job / f'metric-{metric}.jsonl').read_text().splitlines() if s]
        if len(records) != 1 or records[0]['metric'] != metric or records[0]['num_gpus'] != 1:
            raise ValueError('metric output identity mismatch')
        value = float(records[0]['results'][metric])
        if not math.isfinite(value) or (metric == 'fid50k_full' and value <= 0):
            raise FloatingPointError(f'invalid {metric}')
        results[metric] = value
    if features[0].shape != (50000, 2048) or features[1].shape != features[0].shape:
        raise ValueError('incomplete generated feature count')
    for start in range(0, 50000, 1000):
        a, b = (f[start:start + 1000] for f in features)
        if not np.isfinite(a).all() or not np.array_equal(a, b):
            raise ValueError('KID/FID generated features differ or are nonfinite')
    return results
