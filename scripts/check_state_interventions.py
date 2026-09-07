"""Run the protocol's bounded CUDA replay checks in a separate directory."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

from scripts import check_state_intervention_replay as replay
from scripts import run_m1_training_slot as old_runner
from scripts import state_intervention_sources as sources
from training import state_interventions as interventions


def run_piece(directory, manifest, args, stop, *, noop=False, resume=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'formal_run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('engineering manifest changed across resume')
    else:
        old_runner.write_json(path, manifest)
    command = sources.training_command(Path(sys.executable), args.dataset, args.seed,
                                        path, resume or Path(manifest['source_state']['path']),
                                        manifest['branch'])
    index = command.index(str(old_runner.REPO_ROOT / 'ct_train.py'))
    command[index] = str(old_runner.REPO_ROOT / 'scripts/state_intervention_engineering_train.py')
    command.append(f'--stop-after-attempts={stop}')
    env = old_runner.runtime_environment(args.gpu, Path(sys.executable))
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               STATE_INTERVENTION_ENGINEERING_NOOP=str(int(noop)))
    log = directory / f'process-to-{stop}.log'
    with log.open('x') as handle:
        process = subprocess.run(command, cwd=old_runner.REPO_ROOT, env=env,
                                 stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode and not old_runner.scientific_failure(log):
        raise RuntimeError(f'engineering technical failure: {log}')
    rows = replay.read_rows(directory / 'schedule_switch_training_telemetry_v1.csv')
    last = max(rows) if rows else interventions.boundary(manifest) // 128
    result = dict(exit_code=process.returncode, last_attempt=last, command=command)
    if process.returncode:
        result['status'] = 'NUMERICAL_FAILURE'
    elif last != stop:
        raise RuntimeError('engineering did not reach its exact pause')
    else:
        result['status'] = 'COMPLETE_PAUSE'
    old_runner.write_json(directory / f'status-to-{stop}.json', result)
    return result


def check_branch(args, branch):
    base = args.output / f'seed{args.seed}' / branch
    if base.exists():
        raise RuntimeError(f'use a fresh engineering attempt directory: {base}')
    base.mkdir(parents=True)
    initial = 6000 if branch.startswith('L_') else 4000
    arm = interventions.BRANCHES[branch][0]
    results = {}
    for name in ('noop', 'continuous', 'split'):
        directory = base / name
        manifest = sources.manifest(args.source_root, args.seed, branch, directory)
        target = initial + (16 if name == 'split' else 32)
        result = run_piece(directory, manifest, args, target, noop=name == 'noop')
        replay.check_initialization(
            directory / f'training-state-kimg{initial * 128 // 1000:06d}.pt',
            manifest, noop=name == 'noop')
        if name == 'split' and result['status'] == 'COMPLETE_PAUSE':
            result = run_piece(directory, manifest, args, initial + 32,
                               resume=directory / 'training-state-latest.pt')
        results[name] = result
    original = sources.original_run(args.source_root, args.seed, 'K_' + arm)
    telemetry = 'schedule_switch_training_telemetry_v1.csv'
    replay.compare_telemetry(original / telemetry, base / 'noop' / telemetry,
                             initial, initial + 32)
    continuous, split = results['continuous'], results['split']
    if (continuous['status'], continuous['last_attempt']) != (split['status'], split['last_attempt']):
        raise ValueError('continuous and resumed terminal positions differ')
    replay.compare_telemetry(base / 'continuous' / telemetry, base / 'split' / telemetry,
                             initial, continuous['last_attempt'], same_branch=True)
    replay.compare_inputs(base / 'noop', base / 'continuous',
                          min(results['noop']['last_attempt'], continuous['last_attempt']))
    replay.compare_observations(base / 'continuous', base / 'split')
    if continuous['status'] == 'COMPLETE_PAUSE':
        replay.compare_states(base / 'continuous' / 'training-state-latest.pt',
                              base / 'split' / 'training-state-latest.pt')
    record = dict(seed=args.seed, branch=branch, status='PASS',
                  original_noop='MATCH', save_resume='MATCH', runs=results)
    old_runner.write_json(base / 'check.json', record)
    print(json.dumps(record), flush=True)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=interventions.SEEDS, default=55)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    args = parser.parse_args()
    lock = Path(f'/tmp/m1-gpu-{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                                        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError(f'GPU is occupied; not starting checks: {processes}')
    for branch in ('L_' + args.arm, 'X_' + args.arm + '_from_' + ('B' if args.arm == 'A' else 'A')):
        check_branch(args, branch)


if __name__ == '__main__':
    main()
