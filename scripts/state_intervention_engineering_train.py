"""Bounded no-op and save/resume checks; never the formal training entry point."""
import copy
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import ct_training_loop, state_interventions as interventions
from training.m1_optimizer_split import state_summary


def pause(**kwargs):
    target = kwargs['stop_after_attempts']
    if kwargs['seed'] not in interventions.SEEDS or target not in (4016, 4032, 6016, 6032):
        raise ValueError('engineering requires a bounded 16/32-attempt target')
    if not kwargs['strict_reproducibility'] or kwargs['total_kimg'] != 1024:
        raise ValueError('engineering must preserve the original trajectory configuration')
    return target


def noop(optimizer, model, manifest):
    summary = dict(receiver_before=state_summary(optimizer), engineering_noop=True)
    if manifest['experiment_protocol'] == interventions.SWAP:
        summary['parameter_mapping'] = interventions.transplant(
            optimizer, model, model, copy.deepcopy(optimizer.state_dict()))
    summary['after'] = state_summary(optimizer)
    return summary


if __name__ == '__main__':
    ct_training_loop.validate_planned_pause = pause
    if os.environ.get('STATE_INTERVENTION_ENGINEERING_NOOP') == '1':
        interventions.apply = noop
    runpy.run_path(str(ROOT / 'ct_train.py'), run_name='__main__')
