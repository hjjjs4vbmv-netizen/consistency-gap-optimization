"""Isolated short-run entry point, keeping the original FP16 training numerics."""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import ct_training_loop, m1_optimizer_split


def short_pause(**kwargs):
    if kwargs['seed'] not in m1_optimizer_split.SEEDS or kwargs['stop_after_attempts'] != 4064:
        raise ValueError('diagnostic is limited to four selected seeds and 64 attempts')
    return 4064


if __name__ == '__main__':
    ct_training_loop.validate_planned_pause = short_pause
    m1_optimizer_split.install(os.environ['M1_SPLIT_OUTPUT'], os.environ['M1_SPLIT_OPERATION'])
    runpy.run_path(str(ROOT / 'ct_train.py'), run_name='__main__')
