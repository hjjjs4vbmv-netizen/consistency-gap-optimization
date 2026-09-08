"""Run a fixed single-GPU queue as each branch's original sources arrive."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from scripts import run_m1_training_slot as old
from scripts import state_intervention_sources as sources
from training import m1, schedule_switch, state_interventions as interventions

LANES = {}
for role, seeds in (('ect', (59, 60)), ('cloud', (55, 56)), ('extra', (61, 62))):
    for gpu, seed in enumerate(seeds):
        LANES[f'{role}-{gpu}'] = [(seed, branch) for branch in interventions.BRANCHES]


def wait_sources(args, seed, branch):
    arm = interventions.BRANCHES[branch][0]
    paths = [sources.original_run(args.source_root, seed, 'K_' + arm) /
             'training-state-kimg000768.pt'] if branch.startswith('L_') else [
                 sources.prefix(args.source_root, seed, a) for a in ('A', 'B')]
    while not all(path.is_file() for path in paths):
        print(f'WAIT_SOURCE seed={seed} branch={branch}', flush=True)
        time.sleep(60)


def run_branch(args, seed, branch):
    protocol = interventions.M2 if branch.startswith('L_') else interventions.SWAP
    directory = args.output / protocol / f'seed{seed}' / branch
    status_path = directory / 'branch_status.json'
    if status_path.is_file():
        record = json.loads(status_path.read_text())
        if record['status'] in ('COMPLETE', 'NUMERICAL_FAILURE'):
            return record
        raise RuntimeError(f'existing nonterminal branch needs explicit inspection: {directory}')
    wait_sources(args, seed, branch)
    if directory.exists():
        raise RuntimeError(f'refusing to overwrite existing branch: {directory}')
    directory.mkdir(parents=True)
    manifest = sources.manifest(args.source_root, seed, branch, directory)
    manifest_path = directory / 'formal_run_manifest.json'
    old.write_json(manifest_path, manifest)
    schedule_switch.load_run_manifest(manifest_path)
    record = dict(seed=seed, branch=branch, protocol=protocol, gpu=args.gpu,
                  host=socket.gethostname(), code_commit=args.code_commit,
                  started_utc=old.utc_now(), status='RUNNING')
    command = sources.training_command(Path(sys.executable), args.dataset, seed,
                                       manifest_path, Path(manifest['source_state']['path']), branch)
    record['command'] = command
    old.write_json(status_path, record)
    log = directory / 'train-attempt-01.log'
    print(f'START seed={seed} branch={branch} gpu={args.gpu}', flush=True)
    environment = old.runtime_environment(args.gpu, Path(sys.executable))
    environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment.pop('STATE_INTERVENTION_ENGINEERING_NOOP', None)
    with log.open('x') as handle:
        process = subprocess.run(command, cwd=old.REPO_ROOT, env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT)
    record.update(ended_utc=old.utc_now(), exit_code=process.returncode)
    if process.returncode:
        record['status'] = 'NUMERICAL_FAILURE' if old.scientific_failure(log) else 'TECHNICAL_UNRESOLVED'
    else:
        terminal = directory / 'training-state-kimg001024.pt'
        state = torch.load(terminal, map_location='cpu', weights_only=False)
        schedule_switch.verify_switched_state(state, manifest)
        m1.validate_terminal_state(state, manifest)
        record.update(status='COMPLETE', terminal_path=str(terminal))
    old.write_json(status_path, record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lane', choices=LANES, required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--engineering', type=Path, required=True)
    args = parser.parse_args()
    args.gpu = int(args.lane[-1])
    for branch in interventions.BRANCHES:
        record = json.loads((args.engineering / 'seed55' / branch / 'check.json').read_text())
        if record['status'] != 'PASS':
            raise RuntimeError(f'engineering check did not pass: {branch}')
    args.dataset = args.dataset.resolve(strict=True)
    torch.set_num_threads(2)
    lock = Path(f'/tmp/m1-gpu-{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError(f'GPU {args.gpu} occupied: {occupied}')
    checked = set()
    for seed, branch in LANES[args.lane]:
        if seed not in checked:
            while not all(sources.prefix(args.source_root, seed, a).is_file() for a in ('A', 'B')):
                print(f'WAIT_PAIRED_SOURCE seed={seed}', flush=True)
                time.sleep(60)
            old.check_paired_random_streams(
                args.source_root / 'q256_terminal_history_n30_full_archive_v1/training', seed)
            checked.add(seed)
        run_branch(args, seed, branch)
    print('QUEUE_FINISHED', flush=True)


if __name__ == '__main__':
    main()
