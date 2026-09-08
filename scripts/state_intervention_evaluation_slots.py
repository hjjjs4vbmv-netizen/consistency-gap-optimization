"""The fixed 216 new and 120 reused M2/SWAP quality slots."""
import argparse
import csv
from pathlib import Path

from scripts import build_m1_evaluation_slots as original
from training import state_interventions as experiment


def build_slots():
    rows = []
    groups = ((experiment.M2, ('L_A', 'L_B'), 1024, 'NEW'),
              (original.PROTOCOL_ID, original.BRANCHES, 768, 'NEW'),
              (experiment.SWAP, ('X_A_from_B', 'X_B_from_A'), 1024, 'NEW'),
              (original.PROTOCOL_ID, original.BRANCHES, 1024, 'REUSE'))
    for protocol, branches, budget, mode in groups:
        for seed in experiment.SEEDS:
            for branch in branches:
                for readout, blocks in original.READOUT_BLOCKS.items():
                    if budget == 768 and readout == 'E_KEEP':
                        continue
                    readout_id = f'seed{seed}-{branch}-kimg{budget:06d}-{readout}'
                    for block in blocks:
                        start, end = original.BLOCKS[block]
                        rows.append(dict(slot_index=len(rows),
                            slot_id=f'{readout_id}-{block}', readout_id=readout_id,
                            protocol_id=protocol, seed=seed, branch=branch,
                            budget_kimg=budget, readout=readout, block=block,
                            sample_seed_start=start, sample_seed_end=end,
                            sample_count=50000, nfe=1, precision='fp32',
                            metrics='kid50k_full,fid50k_full',
                            metric_seed=original.METRIC_SEED,
                            evaluator_commit=original.EVALUATOR_COMMIT, mode=mode))
    return rows


def get_slot(slot_id, *, new_only=False):
    matches = [row for row in build_slots() if row['slot_id'] == slot_id]
    if not matches or (new_only and matches[0]['mode'] != 'NEW'):
        raise ValueError('slot is outside the requested fixed evaluation matrix')
    return matches[0]


def state_directory(slot, source_root, runs_root):
    from scripts import state_intervention_sources
    if slot['protocol_id'] == original.PROTOCOL_ID:
        return state_intervention_sources.original_run(
            source_root, int(slot['seed']), slot['branch'])
    return Path(runs_root) / slot['protocol_id'] / f"seed{slot['seed']}" / slot['branch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build_slots()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('PLANNED new=216 reuse=120; source/result binding is a separate check')


if __name__ == '__main__':
    main()
