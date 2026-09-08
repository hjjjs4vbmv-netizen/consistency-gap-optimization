"""Resolve the six fixed seeds within the retained original archive layout."""
from pathlib import Path

from training import state_interventions as interventions


def prefix(root, seed, arm):
    if seed not in interventions.SEEDS or arm not in ('A', 'B'):
        raise ValueError('invalid fixed source')
    return Path(root) / 'q256_terminal_history_n30_full_archive_v1' / 'training' / f'seed{seed}' / f'prefix_{arm}' / 'training-state-kimg000512.pt'


def original_run(root, seed, branch):
    if seed not in interventions.SEEDS or branch not in ('K_A', 'K_B', 'R_A', 'R_B'):
        raise ValueError('invalid original M1 branch')
    base = Path(root) / 'm1_optimizer_restart_ema'
    if seed != 56:
        base = base / 'archive' / ('hub2_26065' if seed in (55, 62) else 'old4_29026')
    return base / 'runs' / f'S{seed - 49:02d}' / branch


def manifest(root, seed, branch, output):
    from training import schedule_switch
    if branch not in interventions.BRANCHES or seed not in interventions.SEEDS:
        raise ValueError('outside the fixed new training matrix')
    arm = interventions.BRANCHES[branch][0]
    late = branch.startswith('L_')
    source = (original_run(root, seed, 'K_' + arm) / 'training-state-kimg000768.pt'
              if late else prefix(root, seed, arm))
    result = dict(schema=schedule_switch.RUN_MANIFEST_SCHEMA,
                  experiment_protocol=interventions.M2 if late else interventions.SWAP,
                  run_kind='formal', branch=branch, seed=seed, origin_arm=arm,
                  continuation_arm='A', switch_kimg=512, final_kimg=1024,
                  source_state={'path': str(source.resolve(strict=True))},
                  immutable_output_root=str(Path(output).resolve()), m1_shadow_update=True)
    if not late:
        result['donor_state'] = {'path': str(prefix(root, seed, 'B' if arm == 'A' else 'A').resolve(strict=True))}
    return result


def training_command(python, dataset, seed, manifest_path, resume, branch):
    from scripts import run_m1_training_slot
    command = run_m1_training_slot.training_command(python, dataset, seed, manifest_path, resume)
    if branch.startswith('L_'):
        command[command.index('--immutable-checkpoint-kimg=640,768,896,1024')] = '--immutable-checkpoint-kimg=896,1024'
    return command
