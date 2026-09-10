"""Construct the local part of a global freeze after engineering and seed audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import protocol as p
from .preflight import assets
from training import startup_windows as w
from scripts.run_m1_training_slot import runtime_environment
from analysis.q256_d_restore_vs_hold_v1.evaluate import verify_evaluator
from scripts.run_m1_evaluation_job import build_command


def validate_engineering(rows):
    expected = {(w.Q128, 99001), (w.Q128, 99002), (w.Q256, 50), (w.Q256, 51)}
    if len(rows) != 4 or {(r['protocol_id'], r['seed']) for r in rows} != expected:
        raise RuntimeError('the four planned engineering sequences are required')
    for row in rows:
        if row['status'] != 'PASS' or row['no_quality_evaluation'] is not True:
            raise RuntimeError('engineering did not pass')
        arms = set(w.ARMS128) if row['protocol_id'] == w.Q128 else {'A_replay', 'A_delayed_down6_10'}
        if set(row['checks']) != arms:
            raise RuntimeError('engineering arm coverage differs')
        limit = 128 if row['protocol_id'] == w.Q128 else 64
        if any(check['attempts'] != limit or check['successful_updates'] < 16 for check in row['checks'].values()):
            raise RuntimeError('engineering attempt/diagnostic coverage differs')
        if row['seed'] in (99001, 50) and row['pause_boundaries'] != [5, 6, 10, 11]:
            raise RuntimeError('representative recovery boundaries missing')
        if row['protocol_id'] == w.Q256 and row['parity']['archived_A_all_shared_scientific_fields']['rows'] != 64:
            raise RuntimeError('original A parity missing')
    cost = sum(row['process_gpuh'] for row in rows)
    if cost > 3:
        raise RuntimeError('engineering exceeded the prepared 3-GPUh target; report before formal dispatch')
    return cost


def canonical_initializations(config, gpu, label):
    mapping = next(row for row in config['allocation'] if row['host'] == config['host'] and row['local_gpu'] == gpu)
    seed = config['cohort128'][mapping['logical_gpu']]
    directory = Path(config['reference_root']) / 'q128' / f'seed{seed}'
    root = Path(config['control_root']) / 'canonical_initializations'
    root.mkdir(parents=True, exist_ok=True)
    if (directory / 'initial_state_receipt_v1.json').exists():
        raise RuntimeError('canonical initialization already exists; verify it, do not overwrite')
    manifest = p.manifest(config, w.Q128, seed, 'AA', directory, mode='initialization')
    mp = root / f'seed{seed}.manifest.json'
    p.write(mp, manifest, True)
    command = p.command(config, manifest, mp)
    from .worker import verify_gpu
    verify_gpu(mapping)
    started = time.monotonic()
    with (root / f'seed{seed}-{label}.log').open('x') as stream:
        rc = subprocess.run(command, cwd=p.ROOT, env=runtime_environment(mapping['uuid'], Path(config['runtime_python'])),
                            stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError('canonical initialization failed; preserve its log')
    receipt_path = directory / 'initial_state_receipt_v1.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['seed'] != seed or receipt['attempted_iteration'] != 0 or receipt['processed_nimg'] != 0 or receipt['trajectory_config']['loss_kwargs']['q'] != 128:
        raise RuntimeError('canonical state is not this new q128 seed at zero updates')
    result = dict(status='PASS', seed=seed, gpu=mapping, receipt_sha256=p.digest(receipt_path),
                  process_gpuh=(time.monotonic() - started) / 3600, initialization_only=True, command=command)
    p.write(root / f'seed{seed}.json', result, True)
    print(json.dumps(result), flush=True)


def freeze(config, deployment, source_hashes, engineering_path, seed_audit_path, implementation_commit):
    control = Path(config['control_root'])
    control.mkdir(parents=True, exist_ok=True)
    if (control / 'freeze.json').exists():
        raise RuntimeError('already frozen; no replacement freeze in place')
    hashes = json.loads(source_hashes.read_text())
    for relative, expected in hashes.items():
        if p.digest(p.ROOT / relative) != expected:
            raise RuntimeError('deployed source mismatch: ' + relative)
    engineering = json.loads(engineering_path.read_text())
    engineering_cost = validate_engineering(engineering)
    audit = json.loads(seed_audit_path.read_text())
    if audit['status'] != 'PASS' or audit['final_cohort'] != config['cohort128']:
        raise RuntimeError('seed identity audit/cohort not accepted')
    actual_assets = assets(config)
    verify_evaluator(config)
    rows = p.queue(config)
    selected = [row for row in rows if row['host'] == config['host']]
    controls = p.old_controls()
    for row in controls:
        if row['arm'] == 'AA':
            path = Path(config['reference_root']) / 'q256' / row['receipt']
            if p.digest(path) != row['receipt_sha256']:
                raise RuntimeError('historical AA block receipt differs')
    p.write(control / 'old_controls_48.json', controls, True)
    p.write(control / 'training_queue_40.json', rows, True)
    p.write(control / 'evaluation_slots_120.json', p.evaluation_slots(rows), True)
    p.write(control / 'engineering_receipts.json', engineering, True)
    p.write(control / 'seed_audit.json', audit, True)
    p.write(control / 'source_hashes.json', hashes, True)
    for group in (w.Q128, w.Q256):
        p.write(control / (group + '.protocol.json'), p.specification(group, config['cohort128']), True)
    commands = []
    dry_examples = {}
    for row in selected:
        directory = Path(config['output_root']) / row['protocol_id'] / f"seed{row['seed']}" / row['arm']
        manifest = p.manifest(config, row['protocol_id'], row['seed'], row['arm'], directory)
        name = f"{row['protocol_id']}-seed{row['seed']}-{row['arm']}"
        mp = control / 'manifests' / (name + '.json')
        p.write(mp, manifest, True)
        command = p.command(config, manifest, mp)
        commands.append(dict(**row, manifest=str(mp), manifest_sha256=p.digest(mp), command=command))
        dry_examples.setdefault((row['protocol_id'], row['arm']), (row, manifest, mp))
    p.write(control / 'frozen_training_commands.json', commands, True)
    evaluation_commands = []
    for slot in p.evaluation_slots(selected):
        evaluation_root = (Path(config['output_root']) / slot['protocol_id'] /
                           f"seed{slot['seed']}" / slot['arm'] / 'evaluation')
        wire = {**{k: str(v) for k, v in slot.items()}, 'slot_id': slot['job_id'], 'metrics': 'kid50k_full,fid50k_full'}
        command = build_command(wire, str(evaluation_root / 'E_512.pkl'), Path(config['evaluation_dataset']),
            evaluation_root / slot['block'], Path(config['evaluator_repo']), Path(config['runtime_python']),
            47000 + slot['logical_gpu'])
        evaluation_commands.append(dict(**slot, command=command, snapshot_binding='own sealed 1024-kimg E_512 export'))
    p.write(control / 'frozen_evaluation_commands.json', evaluation_commands, True)
    # Every CLI structure is checked once per host; all seed/output variants above are validated individually.
    dry_root = control / 'dry_runs'
    dry_root.mkdir(exist_ok=True)
    for (group, arm), (row, manifest, mp) in dry_examples.items():
        mapping = next(gpu for gpu in config['allocation'] if gpu['logical_gpu'] == row['logical_gpu'])
        log = dry_root / f'{group}-{arm}.log'
        with log.open('x') as stream:
            rc = subprocess.run(p.command(config, manifest, mp, dry_run=True), cwd=p.ROOT,
                env=runtime_environment(mapping['uuid'], Path(config['runtime_python'])), stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError('formal command dry-run failed: ' + str(log))
    bound = {str(path.relative_to(control)): p.digest(path) for path in control.rglob('*') if path.is_file()
             and (path.suffix == '.json' or path.parent.name == 'dry_runs')
             and (path.parent == control or path.parent.name in ('manifests', 'dry_runs'))
             and path.name not in ('freeze.json', 'cluster_ready.json') and not path.name.startswith(('worker-', 'ledger-'))}
    gate = dict(status='FROZEN', engineering_status='PASS', implementation_commit=implementation_commit,
        deployment_sha256=p.digest(deployment), source_hashes=hashes, control_hashes=bound,
        engineering_process_gpuh=engineering_cost, assets=actual_assets,
        allocation=config['allocation'], host=config['host'], local_training_count=len(commands),
        total_training_count=40, total_new_evaluation_blocks=120, historical_evaluation_blocks=48,
        frozen_wall=time.time(), analysis_quality_unblinded=False)
    p.write(control / 'freeze.json', gate, True)
    print(json.dumps({k: v for k, v in gate.items() if k not in ('source_hashes', 'control_hashes', 'allocation')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--initialize-gpu', type=int)
    parser.add_argument('--label', default='v1')
    parser.add_argument('--source-hashes', type=Path)
    parser.add_argument('--engineering-receipts', type=Path)
    parser.add_argument('--seed-audit', type=Path)
    parser.add_argument('--implementation-commit')
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text())
    if args.initialize_gpu is not None:
        canonical_initializations(config, args.initialize_gpu, args.label)
    else:
        freeze(config, args.deployment, args.source_hashes, args.engineering_receipts, args.seed_audit, args.implementation_commit)


if __name__ == '__main__':
    main()
