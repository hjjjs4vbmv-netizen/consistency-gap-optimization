"""Fixed 16 DD suffixes and 80 endpoint jobs; imports never launch computation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from analysis.q256_history_component_chase_v1 import protocol as old

EXPERIMENT_ID = 'q256_d_restore_vs_hold_v1'
ENGINEERING_ID = 'q256_d_restore_vs_hold_engineering_v1'
PR108_HEAD = '90a0cd3f6ade59c04928b6e58a17bbeb95ed0705'
ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(50, 66))
READOUT_BLOCKS = old.READOUT_BLOCKS
EVALUATOR_COMMIT = old.EVALUATOR_COMMIT
EXPECTED_RUNTIME = old.EXPECTED_RUNTIME
FACTORS = old.FACTORS
write_csv = old.write_csv
SPEC = {
    'experiment': EXPERIMENT_ID, 'pr108_head': PR108_HEAD,
    'seeds': list(SEEDS), 'training_paths': ['DD'], 'world_size': 1,
    'parallel_trajectories': 16, 'start_attempt': 4000, 'end_attempt': 8000,
    'duration': 1.024, 'target': 1.0, 'denominator': 1.1,
    'readout_blocks': READOUT_BLOCKS, 'evaluation_kimg': 1024,
    'evaluator_commit': EVALUATOR_COMMIT, 'metric_seed': 20260730,
    'generation_seeds': {'B0': [0,49999], 'B1': [50000,99999], 'B2': [100000,149999]},
    'primary': 'mean_seed(mean_block(log(FID_DA)-log(FID_DD)))',
    'unit': 'training_seed', 'negative_direction': 'restoring A improves FID',
    'interval': 'two-sided paired t 95%', 'p_value': 'two-sided t, one primary contrast',
    'equivalence': {'role': 'prespecified secondary', 'bounds': ['-log(1.03)','log(1.03)'],
                    'criterion': '90% t interval strictly inside bounds (TOST alpha .05)'},
    'scientific_missingness': 'complete pairs only; all 16 outcomes retained; no full-cohort equivalence',
    'technical_missingness': 'no final inference until resolved',
    'aa_auxiliary': 'AA/DA/DD common valid seeds only',
    'da_already_observed': True, 'independent_seed_replication': False,
    'cost_target_gpuh': [50,65], 'process_cap_gpuh': 80,
    'engineering_target_cap_gpuh': 3, 'effective_cap': 'min(80, verified actual remaining quota)',
    'b_audit_seeds': [50,51], 'b_audit_states': ['ECT_initial','D_prefix_512'],
    'b_audit_variants': ['A','D','A_per_sample_div_1.1','A_lr_div_1.1'],
    'b_audit_max_attempts_per_rollout': 32, 'full_scalar_quality_arm': False,
}


def training_queue():
    return [{'slot': f'S{s-49:02d}', 'seed': s, 'path': 'DD', 'history': 'D',
             'phase': 'suffix', 'start_attempt': 4000, 'end_attempt': 8000, 'status': 'PENDING'} for s in SEEDS]


def evaluation_slots():
    return [{**r, 'job_id':r['job_id'].replace('-DA-','-DD-'), 'path':'DD'}
            for r in old.evaluation_slots() if r['path']=='DA']


def import_controls(arm='DA'):
    rows = json.loads((ROOT/'analysis/q256_history_component_chase_v1/results/quality_slots_320.json').read_text())
    rows = [r for r in rows if r['path']==arm]
    if arm not in {'DA','AA'} or len(rows)!=80:
        raise ValueError('only the 80 original DA or AA slots are allowed')
    expected = {(s,r,b) for s in SEEDS for r,blocks in READOUT_BLOCKS.items() for b in blocks}
    if {(r['seed'],r['readout'],r['block']) for r in rows} != expected:
        raise ValueError('old control matrix changed')
    missing = {r['seed'] for r in rows if r['status']!='PASS'}
    if missing != (set() if arm=='DA' else {58,65}):
        raise ValueError('old scientific missingness changed')
    return rows  # Keep original values/provenance verbatim, including original "source".


def manifest(seed, source, output, binding, engineering=False, da_branch_init=None):
    if seed not in ((50,51) if engineering else SEEDS):
        raise ValueError('outside fixed DD cohort')
    return {'schema':'ect.q256.schedule-switch-run-manifest/v1',
            'experiment_protocol': ENGINEERING_ID if engineering else EXPERIMENT_ID,
            'run_kind':'formal', 'branch':'DD', 'seed':seed, 'origin_arm':'D', 'continuation_arm':'D',
            'switch_kimg':512, 'final_kimg':1024, 'source_state':{'path':str(Path(source).resolve())},
            'source_binding':binding, 'da_branch_init_path':str(Path(da_branch_init or Path(source).parent.parent/'suffix/training-state-kimg000512.pt').resolve()), 'immutable_output_root':str(Path(output).resolve()),
            'd_restore_shadow_update':True}


def command(*, python, dataset, output, seed, resume, switch_manifest, engineering_stop=None):
    if engineering_stop is not None and (seed not in (50,51) or engineering_stop not in (4008,4016,4032)):
        raise ValueError('DD engineering is bounded to fixed seeds and <=32 suffix attempts')
    cmd = old.command(python=python, dataset=dataset, transfer=None, output=output,
                      seed=seed, history='D', phase='suffix', resume=resume,
                      switch_manifest=switch_manifest)
    # The old builder remains unchanged; DD has its own protocol and native factors.
    cmd = ['--denominator-gap-scale=1.1' if x=='--denominator-gap-scale=1.0' else x for x in cmd]
    if engineering_stop is not None:
        cmd += [f'--stop-after-attempts={engineering_stop}',f'--planned-pause-protocol={ENGINEERING_ID}']
    return cmd


def prepare(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name, data in [('protocol.json',SPEC),('training_queue.json',training_queue()),
                       ('old_DA_80.json',import_controls()),('old_AA_auxiliary_80.json',import_controls('AA')),
                       ('new_DD_80.json',evaluation_slots()),('quality_slots_160.json',import_controls()+evaluation_slots())]:
        value=json.dumps(data,indent=2,ensure_ascii=False)+'\n'
        path=output/name
        if path.exists() and path.read_text()!=value:
            raise RuntimeError('refusing to overwrite frozen preparation: '+name)
        path.write_text(value)
    for name in ('training_queue','quality_slots_160'):
        write_csv(output/(name+'.csv'),json.loads((output/(name+'.json')).read_text()))
    files=[Path(__file__),Path(__file__).with_name('analyze.py'),output/'protocol.json',output/'old_DA_80.json',output/'old_AA_auxiliary_80.json']
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    path=output/'analysis_freeze.json'
    payload={'status':'FROZEN_BEFORE_DD_TRAINING','da_already_observed':True,'files_sha256':hashes}
    if path.exists() and json.loads(path.read_text())!=payload:
        raise RuntimeError('analysis freeze changed; requires an explicit recorded amendment')
    path.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
