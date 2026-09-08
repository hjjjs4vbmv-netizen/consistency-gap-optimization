"""Bind archived NEW receipts and original REUSE outputs to the fixed scalar plan."""
import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import sys

import torch

from scripts import state_intervention_evaluation_slots as slots
from scripts import state_intervention_sources as sources
from scripts.inspect_frozen_evaluation import inspect_output
from training import m1, reproducibility


@lru_cache(maxsize=None)
def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def bind_initial_sources(root, bound):
    inventory = {r['path']: r['bytes'] for r in read(root / 'sources_inventory.json')}
    records = []
    for seed in slots.experiment.SEEDS:
        for arm in ('A', 'B'):
            path = sources.prefix(root.parent, seed, arm)
            if path.stat().st_size != inventory[str(path)]:
                raise ValueError('512 source no longer matches preflight inventory size')
            records.append(dict(seed=seed, branch='prefix_' + arm, budget_kimg=512,
                bytes=path.stat().st_size, source_state_sha256=digest(path)))
        for branch in ('K_A', 'K_B', 'R_A', 'R_B'):
            rows = [r for r in bound['rows'] if (r['seed'], r['branch'], r['budget_kimg']) == (seed, branch, 768)]
            hashes = {r['source_state_sha256'] for r in rows}
            if len(hashes) != 1 or len(rows) != 4:
                raise ValueError('768 source binding is inconsistent across readouts')
            path = sources.original_run(root.parent, seed, branch) / 'training-state-kimg000768.pt'
            records.append(dict(seed=seed, branch=branch, budget_kimg=768,
                bytes=path.stat().st_size, source_state_sha256=hashes.pop()))
    return dict(binding_time='post-run source fingerprints, not a pre-run seal', sources=records)


def check_identity(record, slot, *, legacy=False):
    numeric = ('seed', 'sample_seed_start', 'sample_seed_end', 'sample_count', 'nfe', 'metric_seed')
    keys = ('branch', 'readout', 'block', 'precision', 'evaluator_commit')
    if (record['status'] != 'PASS' or any(int(record[k]) != slot[k] for k in numeric)
            or any(record[k] != slot[k] for k in keys)):
        raise ValueError('result identity or status mismatch: ' + slot['slot_id'])
    if not legacy and any(record[k] != slot[k] for k in
                          ('slot_id', 'protocol_id', 'budget_kimg', 'mode')):
        raise ValueError('new result source identity mismatch')


def archived_path(recorded, base, root):
    path = Path(recorded)
    cloud = Path('/root/final_state_interventions_q256_v2')
    if path.is_relative_to(cloud):
        return base / path.relative_to(cloud)
    if path.is_relative_to(root):
        return path
    raise ValueError('unexpected task artifact location: ' + str(path))


@lru_cache(maxsize=None)
def verify_old_state(directory, snapshot_dir, seed, branch):
    state_path = directory / 'training-state-kimg001024.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    manifest = read(directory / 'formal_run_manifest.json')
    m1.validate_terminal_state(state, manifest)
    if state['m1']['seed'] != seed or state['m1']['branch'] != branch:
        raise ValueError('old endpoint identity mismatch')
    hashes = {}
    for name in ('ONLINE', 'E_KEEP', 'E_512'):
        with (snapshot_dir / (name + '.pkl')).open('rb') as stream:
            snapshot = pickle.load(stream)
        actual = reproducibility.module_state_sha256(snapshot['ema'])
        if actual != reproducibility.module_state_sha256(m1.readout_module(state, name)):
            raise ValueError('old exported readout does not match original state')
        hashes[name] = actual
    return dict(source_state_sha256=digest(state_path), readout_hashes=hashes)


def collect(root, original):
    bases = [root, root / 'archive/cloud28453', root / 'archive/cloud29569']
    index = {}
    for base in bases:
        for path in (base / 'evaluation/receipts').glob('*.json'):
            row = read(path)
            if row['slot_id'] in index:
                raise ValueError('duplicate NEW receipt')
            index[row['slot_id']] = (base, path, row)
    plan = slots.build_slots()
    if set(index) != {r['slot_id'] for r in plan if r['mode'] == 'NEW'}:
        raise ValueError('NEW receipts do not cover the fixed plan')
    equivalence = read(root / 'evaluation/dataset_equivalence.json')
    if equivalence['status'] != 'ALL_ENTRY_BYTES_MATCH' or equivalence['entries'] != 50001:
        raise ValueError('dataset equivalence evidence is incomplete')
    for kind in ('original', 'evaluation'):
        if digest(Path(equivalence[kind + '_path'])) != equivalence[kind + '_sha256']:
            raise ValueError('dataset changed since equivalence check')
    output = []
    for slot in plan:
        if slot['mode'] == 'NEW':
            base, receipt_path, record = index[slot['slot_id']]
            check_identity(record, slot)
            job = base / 'evaluation/jobs' / slot['slot_id']
            export_path = archived_path(record['export_receipt_path'], base, root)
            export = read(export_path)
            if digest(export_path) != record['export_receipt_sha256']:
                raise ValueError('archived export receipt changed')
            for key in ('seed', 'branch', 'budget_kimg', 'readout', 'protocol_id'):
                if export[key] != slot[key]:
                    raise ValueError('export identity mismatch')
            if slot['budget_kimg'] == 768:
                directory = sources.original_run(original.parent, slot['seed'], slot['branch'])
            else:
                owner = root if slot['seed'] in (59, 60) else root / ('archive/cloud28453'
                    if slot['seed'] in (55, 56) else 'archive/cloud29569')
                directory = owner / 'runs' / slot['protocol_id'] / f"seed{slot['seed']}" / slot['branch']
            files = dict(source_state=directory / f"training-state-kimg{slot['budget_kimg']:06d}.pt",
                branch_manifest=directory / 'formal_run_manifest.json',
                branch_status=directory / 'branch_status.json', snapshot=export_path.parent / 'readout.pkl')
            for key, path in files.items():
                if digest(path) != export[key + '_sha256']:
                    raise ValueError('archived source binding changed: ' + str(path))
            if (record['source_state_sha256'] != export['source_state_sha256'] or
                    record['snapshot_sha256'] != export['snapshot_sha256'] or
                    export['source_readout_sha256'] != export['snapshot_readout_sha256']):
                raise ValueError('readout source binding mismatch')
            for name, expected in record['artifact_sha256'].items():
                if not (job / name).is_file():
                    raise ValueError('missing archived evaluation artifact: ' + name)
                if not name.endswith('.npy') and digest(job / name) != expected:
                    raise ValueError('archived evaluation artifact changed: ' + name)
            binding = dict(source_state_sha256=export['source_state_sha256'],
                snapshot_sha256=export['snapshot_sha256'], readout_sha256=export['source_readout_sha256'],
                export_receipt_sha256=digest(export_path))
        else:
            legacy_id = f"S{slot['seed'] - 49:02d}-{slot['branch']}-{slot['readout']}-{slot['block']}"
            old = original / 'endpoint_evaluation'
            receipt_path = old / 'receipts' / (legacy_id + '.json')
            record = read(receipt_path)
            check_identity(record, slot, legacy=True)
            if record['slot_id'] != legacy_id:
                raise ValueError('wrong original slot')
            directory = sources.original_run(original.parent, slot['seed'], slot['branch'])
            if Path(record['source_state']).resolve() != (directory / 'training-state-kimg001024.pt').resolve():
                raise ValueError('original receipt points at another source')
            snapshot_dir = old / 'snapshots' / record['roster_slot'] / slot['branch']
            bound = verify_old_state(directory, snapshot_dir, slot['seed'], slot['branch'])
            snapshot = snapshot_dir / (slot['readout'] + '.pkl')
            binding = dict(source_state_sha256=bound['source_state_sha256'],
                snapshot_sha256=digest(snapshot), readout_sha256=bound['readout_hashes'][slot['readout']],
                original_slot_id=legacy_id)
            job = old / 'jobs' / legacy_id
        options = read(job / 'training_options.json')
        dataset = Path(options['dataset_kwargs']['path'])
        if dataset.name not in ('cifar10-32x32-training-original.zip', 'cifar10-32x32-eval.zip'):
            raise ValueError('unexpected recorded evaluation dataset')
        metrics = inspect_output(slot, job, Path(options['resume_pkl']), dataset, scan_features=False)
        if metrics != record['metrics']:
            raise ValueError('metric JSON differs from receipt')
        output.append(dict(slot, status='PASS', **metrics, **binding,
            result_receipt_sha256=digest(receipt_path), recorded_dataset=dataset.name,
            metric_json_sha256={m: digest(job / ('metric-' + m + '.jsonl')) for m in metrics},
            elapsed_seconds=record['elapsed_seconds'], started_utc=record['started_utc'],
            ended_utc=record['ended_utc']))
        if len(output) % 24 == 0:
            print('VERIFIED', len(output), '/336', file=sys.stderr, flush=True)
    training = []
    for seed in slots.experiment.SEEDS:
        base = root if seed in (59, 60) else root / ('archive/cloud28453' if seed in (55, 56)
                                                   else 'archive/cloud29569')
        for branch in slots.experiment.BRANCHES:
            protocol = slots.experiment.M2 if branch.startswith('L_') else slots.experiment.SWAP
            directory = base / 'runs' / protocol / f'seed{seed}' / branch
            status = read(directory / 'branch_status.json')
            if status['status'] != 'COMPLETE' or status['exit_code'] != 0:
                raise ValueError('training terminal not complete')
            state = torch.load(directory / 'training-state-kimg001024.pt', map_location='cpu', weights_only=False)
            m1.validate_terminal_state(state, read(directory / 'formal_run_manifest.json'))
            obs = [json.loads(s) for s in (directory / 'first64.jsonl').read_text().splitlines()]
            if len(obs) != 64:
                raise ValueError('incomplete first64 telemetry')
            elapsed = (datetime.fromisoformat(status['ended_utc']) - datetime.fromisoformat(status['started_utc'])).total_seconds()
            training.append(dict(seed=seed, branch=branch, status=status['status'],
                started_utc=status['started_utc'], ended_utc=status['ended_utc'], elapsed_seconds=elapsed,
                code_commit=status['code_commit'], attempted_iteration=state['attempted_iteration'],
                intervention=state['m1'], first64=obs,
                source_state_sha256=digest(directory / 'training-state-kimg001024.pt')))
    return dict(rows=output, training=training,
        validation=dict(new=216, reuse=120, unique=336, duplicate=0, missing=0,
                        dataset_equivalence=equivalence['status'],
                        old_readouts_rechecked_against_states=72,
                        archive_hash_scope='sources, snapshots, manifests, statuses, metric JSONs and options',
                        generated_arrays='presence and shape rechecked; finite/equality scans reused from PASS jobs'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = collect(args.task_root, args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    args.output.with_name('source_bindings.json').write_text(json.dumps(
        bind_initial_sources(args.task_root, result), indent=2) + '\n')


if __name__ == '__main__':
    main()
