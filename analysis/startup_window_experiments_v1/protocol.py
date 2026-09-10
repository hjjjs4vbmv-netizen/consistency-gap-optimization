"""Scientific specifications, exact controls and commands; no dispatch on import."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from training import startup_windows as w
from training.startup_update import DATA_SHA256, TRANSFER_SHA256

ROOT = Path(__file__).resolve().parents[2]
BASE = 'e429ab29476b23c4417cc2b1df7268a2b192f175'
EVALUATOR_COMMIT = 'd6aba02fb88e9db0993623895eb2228ed717d810'
ENVIRONMENT = dict(python='3.11.13', torch='2.6.0+cu124', torch_cuda='12.4', cudnn=90100,
                   numpy='2.1.2', scipy='1.16.1')
BLOCKS = {'B0': [0, 49999], 'B1': [50000, 99999], 'B2': [100000, 149999]}
TERMINAL = ('PASS', 'SCIENTIFIC_FAILURE', 'NO_ENDPOINT', 'TECHNICAL_FAILURE')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable:
        if path.exists():
            if path.read_text() != text:
                raise RuntimeError('immutable record differs: ' + str(path))
            return
        with path.open('x') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def specification(group, cohort128):
    q = 128 if group == w.Q128 else 256 if group == w.Q256 else None
    if q is None:
        raise ValueError('unknown protocol')
    return dict(protocol_id=group, q=q, base_commit=BASE,
        cohort=list(cohort128) if q == 128 else list(range(50, 58)),
        cohort_identity='fresh_training_seeds_not_selected_by_quality' if q == 128 else 'PR111_enriched_exploratory_extension',
        arms=list(w.ARMS128 if q == 128 else w.ARMS256),
        reused_arms=[] if q == 128 else ['AA', 'A_startup_down5'],
        initialization='fresh_transfer', final_kimg=1024, switch_kimg=512,
        batch=128, microbatch=16, world_size=1, attempted_iterations=8000,
        global_gap_scale=1.0, target_gap_scale=1.0,
        windows={a: dict(zip(('start_success_step', 'end_success_step', 'lr_multiplier'), w.WINDOWS[a]))
                 for a in (w.ARMS128 if q == 128 else w.ARMS256)},
        primary='S128=(C128-M128)/2' if q == 128 else 'T256=Y_delayed-Y_early',
        key_secondary=['M128=Y_A_startup_down5-Y_AA', 'C128=Y_D_startup_up5-Y_DA'] if q == 128 else ['L256=Y_delayed-Y_AA'],
        supportive=['H128=Y_DA-Y_AA'] if q == 128 else [],
        statistics=dict(unit='training_seed', Y='mean of three log-FID50k values',
            test='two-sided paired t', interval='95 percent t',
            secondary_multiplicity='Holm over M128 and C128 only' if q == 128 else 'none; exploratory nominal intervals',
            sign_flip='all 2^n signs; sensitivity under sign symmetry, not allocation randomization',
            missing='complete four-arm set' if q == 128 else 'complete AA/early/delayed set',
            optional_equivalence='TOST +/-log(1.03), auxiliary only' if q == 256 else None),
        evaluation=dict(endpoint_kimg=1024, readout='E_512', precision='fp32', nfe=1,
            blocks=BLOCKS, samples_per_block=50000, metric_seed=20260730,
            metrics=['fid50k_full', 'kid50k_full'], evaluator_commit=EVALUATOR_COMMIT,
            timing='all three blocks after each valid endpoint', unblind='only after this full group is terminal'),
        engineering=dict(seeds=list(w.ENGINEERING_SEEDS if q == 128 else (50, 51)),
            max_attempts=128 if q == 128 else 64, no_quality_evaluation=True),
        failure_policy='preserve all failures; no replacement, outlier removal or effect-driven changes')


def old_controls():
    sources = [
        ('analysis/q256_startup_step_responder_pilot_v1/results/old_control_bindings.json', 'AA'),
        ('analysis/q256_startup_step_responder_pilot_v1/results/evaluation_slots.json', 'A_startup_down5'),
    ]
    rows = []
    for relative, arm in sources:
        path = ROOT / relative
        source_hash = digest(path)
        for row in json.loads(path.read_text()):
            if row.get('seed') in range(50, 58) and row.get('arm', row.get('path')) == arm and row.get('readout') == 'E_512':
                record = copy.deepcopy(row)
                record.update(arm=arm, source_document=relative, source_document_sha256=source_hash,
                              source_repository_commit=BASE)
                if record.get('status') != 'PASS' or record.get('evaluator_commit') != EVALUATOR_COMMIT:
                    raise RuntimeError('historical endpoint/evaluator identity differs')
                if [record.get('sample_seed_start'), record.get('sample_seed_end')] != BLOCKS[record['block']]:
                    raise RuntimeError('historical block sample range differs')
                rows.append(record)
    expected = {(s, a, b) for s in range(50, 58) for a in ('AA', 'A_startup_down5') for b in BLOCKS}
    if len(rows) != 48 or {(r['seed'], r['arm'], r['block']) for r in rows} != expected:
        raise RuntimeError('exactly 48 distinct historical AA/early blocks required')
    return rows


def manifest(config, group, seed, arm, output, mode='formal'):
    q = 128 if group == w.Q128 else 256
    start, end, factor = w.WINDOWS[arm]
    reference = Path(config['reference_root']) / f'q{q}' / f'seed{seed}' / 'initial_state_receipt_v1.json'
    return w.validate_manifest(dict(protocol_id=group, q=q, seed=seed, arm=arm,
        cohort=list(config['cohort128']) if q == 128 else list(range(50, 58)),
        mode=mode, initialization='fresh_transfer', base_lr=1e-4, final_kimg=1024, switch_kimg=512,
        target_gap_scale=1.0, global_gap_scale=1.0, diagnostic_success_steps=16,
        native_prefix_arm=w.native_prefix(arm), start_success_step=start, end_success_step=end, lr_multiplier=factor,
        max_attempts=(128 if q == 128 else 64) if mode == 'engineering' else 0 if mode == 'initialization' else 8000,
        reference_initial_receipt=None if mode == 'initialization' else dict(path=str(reference), sha256=digest(reference)),
        dataset_sha256=DATA_SHA256, transfer_sha256=TRANSFER_SHA256,
        immutable_output_root=str(Path(output).resolve()),
        old_control_bindings=[] if q == 128 else [r for r in old_controls() if r['seed'] == seed]))


def command(config, m, manifest_path, resume=None, stop_success=None, dry_run=False):
    cmd = [config['runtime_python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1', str(ROOT / 'ct_train.py'),
        f"--data={config['dataset']}", f"--outdir={m['immutable_output_root']}", '--nosubdir',
        '--cond=False', '--arch=ddpmpp', '--precond=ect', '--batch=128', '--batch-gpu=16',
        '--optim=RAdam', '--lr=0.0001', '--dropout=0.2', '--augment=0', '--xflip=False',
        '--mean=-1.1', '--std=2.0', '--mapping=sigmoid', '--global-gap-scale=1.0',
        f"--factorial-protocol={w.native_protocol(m['q'])}", '--target-gap-scale=1.0',
        '--denominator-gap-scale=' + ('1.1' if m['native_prefix_arm'] == 'D' else '1.0'),
        '-q', str(m['q']), '-k', '8', '-b', '1', '-c', '0', '--double=10000', '--ema_beta=0.9993', f"--seed={m['seed']}",
        '--fp16=True', '--tf32=False', '--ls=1.0', '--enable_amp=True', '--bench=False', '--cache=True', '--workers=1',
        '--metrics=none', '--duration=1.024', '--tick=10', '--snap=0', '--dump=0', '--ckpt=10',
        '--sample_every=26', '--eval_every=50', '--mid_t=0.821', '--adaptive-update-kimg=0.5',
        '--immutable-checkpoint-kimg=512,1024', f'--startup-window-manifest={manifest_path}']
    cmd.append(f'--resume={resume}' if resume else f"--transfer={config['transfer']}")
    if stop_success is not None:
        cmd.append(f'--window-preflight-stop-success={stop_success}')
    if dry_run:
        cmd.append('--dry_run')
    return cmd


def queue(config):
    rows = []
    for slot in config['allocation']:
        index = slot['logical_gpu']
        seed128 = config['cohort128'][index]
        arms = w.ARMS128[index % 4:] + w.ARMS128[:index % 4]
        jobs = [(w.Q256, 50 + index, 'A_delayed_down6_10')] + [(w.Q128, seed128, arm) for arm in arms]
        for order, (group, seed, arm) in enumerate(jobs):
            rows.append(dict(protocol_id=group, seed=seed, arm=arm, order=order,
                             host=slot['host'], local_gpu=slot['local_gpu'], logical_gpu=index, gpu_uuid=slot['uuid']))
    if len(rows) != 40 or len({(r['protocol_id'], r['seed'], r['arm']) for r in rows}) != 40:
        raise RuntimeError('complete 40-trajectory queue required')
    return rows


def evaluation_slots(rows):
    return [dict(**row, block=b, sample_seed_start=limits[0], sample_seed_end=limits[1],
        metric_seed=20260730, readout='E_512', precision='fp32', nfe=1,
        job_id=f"{row['protocol_id']}-seed{row['seed']}-{row['arm']}-E_512-{b}")
        for row in rows for b, limits in BLOCKS.items()]
