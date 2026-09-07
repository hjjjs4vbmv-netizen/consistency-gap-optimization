"""Export actual readouts from original 768 or new M2/SWAP 1024 states."""
import argparse
import json
from pathlib import Path

import torch

from scripts import classify_m1_readout, state_intervention_evaluation_slots as slots
from scripts import validate_m1_evaluation_job as validation
from training import m1, reproducibility, schedule_switch


def validate_state(state, manifest, slot):
    expected = dict(experiment_protocol=slot['protocol_id'], seed=int(slot['seed']),
                    branch=slot['branch'], run_kind='formal')
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('checkpoint manifest does not match the evaluation slot')
    if slot['mode'] != 'NEW':
        raise ValueError('original 1024 slots must be reused, not re-exported here')
    budget = int(slot['budget_kimg'])
    if (state.get('cur_nimg') != budget * 1000 or
            state.get('attempted_iteration') != budget * 1000 // 128):
        raise ValueError('checkpoint is not at the exact requested milestone')
    schedule_switch.verify_switched_state(state, manifest)
    m1.validate_resumed_state(state, manifest)
    if budget == 1024:
        m1.validate_terminal_state(state, manifest)
    else:
        if (state.get('reproducibility_schema') != reproducibility.TRAINING_STATE_SCHEMA
                or state['trajectory_config'].get('seed') != int(slot['seed'])
                or state['trajectory_config'].get('total_kimg') != 1024
                or len(state['rank_states']) != 1
                or state['rank_states'][0]['sampler_state']['consumed_samples'] != 768000):
            raise ValueError('original 768 state identity or sampler progress mismatch')
        if state['trajectory_config']['network_kwargs'].get('use_fp16') is not True:
            raise ValueError('768 state is not from the original FP16 trajectory')


def prepare_readout(state, manifest, slot):
    validate_state(state, manifest, slot)
    source_hash = reproducibility.module_state_sha256(m1.readout_module(state, slot['readout']))
    snapshot = m1.evaluator_snapshot(state, slot['readout'])
    if (snapshot['ema'].training or
            reproducibility.module_state_sha256(snapshot['ema']) != source_hash):
        raise RuntimeError('export changed the actual readout weights')
    return snapshot, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot-id', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    slot = slots.get_slot(args.slot_id, new_only=True)
    directory = slots.state_directory(slot, args.source_root, args.runs_root)
    state_path = directory / f"training-state-kimg{slot['budget_kimg']:06d}.pt"
    manifest_path = directory / 'formal_run_manifest.json'
    branch_status_path = directory / 'branch_status.json'
    manifest = schedule_switch.load_run_manifest(str(manifest_path))
    branch_status = json.loads(branch_status_path.read_text())
    if (branch_status.get('status') not in ('PASS', 'COMPLETE') or
            branch_status.get('exit_code') != 0 or
            branch_status.get('seed') != slot['seed'] or
            branch_status.get('branch') != slot['branch']):
        raise ValueError('source branch is not a matching completed formal run')
    output = args.output_root / slot['readout_id']
    if output.exists():
        raise FileExistsError(f'readout output already exists: {output}')
    torch.set_num_threads(2)
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    snapshot, source_hash = prepare_readout(state, manifest, slot)
    # Reuse the original deterministic FP32 probe on a copy, never the training state.
    observation = classify_m1_readout.classify_fixed_input(snapshot['ema'], torch.device('cpu'))
    finite = observation['classification'] == 'FINITE_READOUT'
    output.mkdir(parents=True)
    snapshot_path = output / 'readout.pkl'
    reproducibility.atomic_pickle_dump(snapshot, snapshot_path, overwrite=False)
    receipt = dict(schema='ect.state-interventions.readout-export/v1',
        status='EXPORTED' if finite else 'READOUT_INVALID_NEEDS_REVIEW',
        protocol_id=slot['protocol_id'], seed=slot['seed'], branch=slot['branch'],
        budget_kimg=slot['budget_kimg'], readout=slot['readout'],
        readout_id=slot['readout_id'], source_state_path=str(state_path.resolve()),
        source_state_sha256=validation.sha256_file(state_path),
        source_attempted_iteration=state['attempted_iteration'],
        source_cur_nimg=state['cur_nimg'],
        source_readout_sha256=source_hash,
        snapshot_readout_sha256=reproducibility.module_state_sha256(snapshot['ema']),
        snapshot_path=str(snapshot_path.resolve()),
        snapshot_sha256=validation.sha256_file(snapshot_path),
        branch_manifest_path=str(manifest_path.resolve()),
        branch_manifest_sha256=validation.sha256_file(manifest_path),
        branch_status_path=str(branch_status_path.resolve()),
        branch_status_sha256=validation.sha256_file(branch_status_path),
        training_code_commit=branch_status.get('code_commit'),
        fixed_input_observation=observation)
    receipt['export_source_files'] = {
        str(Path(module.__file__).relative_to(Path(__file__).resolve().parents[1])):
        validation.sha256_file(Path(module.__file__))
        for module in (m1, reproducibility, schedule_switch, classify_m1_readout, slots)
    }
    receipt['export_source_files']['scripts/export_state_intervention_readout.py'] = validation.sha256_file(Path(__file__))
    validation.atomic_json(output / 'receipt.json', receipt)
    print(json.dumps({'readout_id': slot['readout_id'], 'status': receipt['status']}))
    return 0 if finite else 2


if __name__ == '__main__':
    raise SystemExit(main())
