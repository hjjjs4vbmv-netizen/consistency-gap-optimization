"""F12 lanes: finish assigned training first, then fixed NFE1/NFE2 evaluations."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from . import protocol as p
from .evaluation import validate_result
from training import startup_windows as w, reproducibility as repro, m1
from scripts.run_m1_training_slot import scientific_failure
from analysis.q256_d_restore_vs_hold_v1.evaluate import verify_evaluator
from analysis.startup_window_experiments_v1.worker import (
    locked, verify_gpu, run_directory, key, latest, trim_for_resume, process,
)


def verify_source(config):
    gate_path=Path(config['control_root'])/'source_freeze.json'
    frozen=json.loads(gate_path.read_text())
    if frozen['status']!='FROZEN' or frozen['protocol_id']!=p.PROTOCOL_ID:
        raise RuntimeError('F12 source preregistration missing')
    for relative,sha in frozen['source_hashes'].items():
        if p.digest(p.ROOT/relative)!=sha:raise RuntimeError('frozen source changed: '+relative)
    if frozen['protocol']!=p.specification():raise RuntimeError('frozen F12 specification changed')
    return frozen


def verify_lane(config, deployment, lane):
    frozen=verify_source(config)
    ready_path=Path(config['control_root'])/f'lane-{lane}.ready.json'
    ready=json.loads(ready_path.read_text())
    if ready['status']!='READY' or ready['deployment_sha256']!=p.digest(deployment):
        raise RuntimeError('lane deployment permit changed')
    if ready['source_freeze_sha256']!=p.digest(Path(config['control_root'])/'source_freeze.json'):
        raise RuntimeError('lane source identity changed')
    for path,sha in ready['bindings'].items():
        if p.digest(path)!=sha:raise RuntimeError('frozen lane binding changed: '+path)
    mapping=next(r for r in config['allocation'] if r['host']==config['host'] and r['logical_gpu']==lane)
    if mapping!=ready['mapping']:raise RuntimeError('GPU ownership permit differs')
    if lane>=24:
        decision=json.loads(Path(config['reserve_decision']).read_text())
        if decision.get('activate_reserve') is not True or decision['initial_complete_n']>=9:
            raise RuntimeError('reserve cohort not activated by completion-only rule')
    return mapping

def train(config, row, mapping, heartbeat):
    directory = run_directory(config, row)
    directory.mkdir(parents=True, exist_ok=True)
    mp = Path(config['control_root']) / 'manifests' / (key(row) + '.json')
    manifest = w.read_manifest(mp)
    if manifest != p.manifest(config, row['protocol_id'], row['seed'], row['arm'], directory):
        raise RuntimeError('formal manifest drift')
    outcome_path = directory / 'outcome.json'
    if outcome_path.exists():
        old = json.loads(outcome_path.read_text())
        if old['status'] in ('PASS', 'SCIENTIFIC_FAILURE', 'NO_ENDPOINT'):
            return old
    attempt, resume = latest(directory, manifest)
    if attempt == 8000:
        result = dict(**row, status='PASS', endpoint=str(resume), checkpoint_sha256=p.digest(resume),
                      attempted_iteration=8000, recovered_completed_endpoint=True,
                      process_gpuh=sum(json.loads(path.read_text()).get('session_process_gpuh', 0)
                                       for path in directory.glob('process-receipt-*.json')))
        p.write(outcome_path, result)
        return result
    if list(directory.glob('process-*.log')) and resume is None:
        result = dict(**row, status='TECHNICAL_FAILURE', reason='prior logs without a verified own state; start-over requires documented technical audit')
        p.write(outcome_path, result)
        return result
    if resume is not None:
        trim_for_resume(directory, attempt)
    log = directory / f'process-{time.time_ns()}.log'
    observation = process(config, p.command(config, manifest, mp, resume), mapping, log, heartbeat, training_dir=directory, timeout=43200)
    prior_gpuh = sum(json.loads(path.read_text()).get('session_process_gpuh', 0)
                     for path in directory.glob('process-receipt-*.json'))
    text = log.read_text(errors='replace')
    scientific = scientific_failure(log) or any(marker in text for marker in (
        'non-finite RAdam moment state', 'non-finite E_512 state'))
    result = dict(**row, **observation, status='SCIENTIFIC_FAILURE' if scientific else 'TECHNICAL_FAILURE', start_attempt=attempt)
    result.update(session_process_gpuh=observation['process_gpuh'], process_gpuh=prior_gpuh + observation['process_gpuh'])
    if observation['returncode'] == 0:
        endpoint = directory / 'training-state-kimg001024.pt'
        try:
            state = torch.load(endpoint, map_location='cpu', weights_only=False)
            meta = w.validate_state(state, manifest)
            if state['attempted_iteration'] != 8000 or meta['ema_512_init_count'] != 1:
                raise RuntimeError('not a complete own E_512 endpoint')
            if p.digest(endpoint) != json.loads(Path(str(endpoint) + '.sha256.json').read_text())['sha256']:
                raise RuntimeError('endpoint seal differs')
            result.update(status='PASS', endpoint=str(endpoint), checkpoint_sha256=p.digest(endpoint),
                attempted_iteration=8000, successful_optimizer_steps=state['successful_optimizer_steps'])
            del state
        except Exception as exc:
            result['error'] = repr(exc)
    p.write(directory / f'process-receipt-{time.time_ns()}.json', result, True)
    p.write(outcome_path, result)
    return result

def evaluate(config, row, mapping, outcome, heartbeat, nfe=1):
    directory = run_directory(config, row)
    root = directory / 'evaluation' / f'NFE{nfe}'
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'E_512.pkl'
    if outcome['status'] == 'PASS':
        verify_evaluator(config)
        source = Path(outcome['endpoint'])
        if p.digest(source) != outcome['checkpoint_sha256']:
            raise RuntimeError('own endpoint changed before export')
        manifest = w.read_manifest(Path(config['control_root']) / 'manifests' / (key(row) + '.json'))
        state = torch.load(source, map_location='cpu', weights_only=False)
        w.validate_state(state, manifest)
        exported = m1.evaluator_snapshot(state, 'E_512')
        module_hash = repro.module_state_sha256(exported['ema'])
        if not snapshot.exists():
            repro.atomic_pickle_dump(exported, snapshot, overwrite=False)
        else:
            import pickle
            with snapshot.open('rb') as handle:
                previous = pickle.load(handle)
            if repro.module_state_sha256(previous['ema']) != module_hash:
                raise RuntimeError('existing E_512 export differs')
            del previous
        p.write(root / 'export.json', dict(readout='E_512', checkpoint_sha256=outcome['checkpoint_sha256'],
            snapshot_sha256=p.digest(snapshot), module_sha256=module_hash), True)
        del state, exported
    results = []
    for slot in p.evaluation_slots([row], nfe=nfe):
        receipt = root / (slot['block'] + '.json')
        if receipt.exists():
            results.append(json.loads(receipt.read_text()))
            continue
        if outcome['status'] != 'PASS':
            result = dict(**slot, status='NO_ENDPOINT', training_status=outcome['status'])
        else:
            target = root / slot['block']
            if target.exists():
                raise RuntimeError('partial evaluation without receipt requires audit; no overwrite')
            target.mkdir()
            command = p.evaluation_command(config, slot, snapshot, target)
            observation = process(config, command, mapping, root / (slot['block'] + '.process.log'), heartbeat, timeout=7200)
            result = dict(**slot, **observation, status='TECHNICAL_FAILURE', evaluator_commit=p.EVALUATOR_COMMIT,
                checkpoint_sha256=outcome['checkpoint_sha256'], snapshot_sha256=p.digest(snapshot))
            try:
                result.update(validate_result(slot, str(snapshot), target, config['evaluation_dataset'], observation['returncode']), status='PASS')
            except Exception as exc:
                result['error'] = repr(exc)
        p.write(receipt, result, True)
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--logical-gpu',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.deployment.read_text())
    lane=args.logical_gpu;mapping=verify_lane(config,args.deployment,lane)
    control=Path(config['control_root']); heartbeat=control/f'worker-{lane}.json'
    ledger_path=control/f'ledger-{lane}.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(jobs={})
    rows=[r for r in p.queue(include_reserve=lane>=24) if r['logical_gpu']==lane]
    if len(rows)!=2:raise RuntimeError('each F12 lane must have exactly two fixed paths')
    with locked(control/(mapping['uuid']+'.lock')):
        for row in rows:
            verify_lane(config,args.deployment,lane)
            outcome=train(config,row,mapping,heartbeat)
            ledger['jobs'][key(row)]=dict(training_status=outcome['status'],training_gpuh=outcome.get('process_gpuh',0),evaluation_statuses=[])
            p.write(ledger_path,ledger)
            if outcome['status']=='TECHNICAL_FAILURE':
                p.write(heartbeat,dict(status='TECHNICAL_ATTENTION',job=row,quality_values_exposed=False));return
        p.write(control/f'lane-{lane}.training_complete.json',dict(status='TRAINING_COMPLETE',jobs=rows,ledger=str(ledger_path)),True)
        for nfe in (1,2):
            for row in rows:
                outcome=json.loads((run_directory(config,row)/'outcome.json').read_text())
                results=evaluate(config,row,mapping,outcome,heartbeat,nfe)
                entry=ledger['jobs'][key(row)]
                entry[f'evaluation_NFE{nfe}_gpuh']=sum(r.get('process_gpuh',0) for r in results)
                entry[f'evaluation_NFE{nfe}_statuses']=[r['status'] for r in results]
                p.write(ledger_path,ledger)
                if any(r['status']=='TECHNICAL_FAILURE' for r in results):
                    p.write(heartbeat,dict(status='EVALUATION_ATTENTION',job=row,nfe=nfe,quality_values_exposed=False));return
        p.write(heartbeat,dict(status='COMPLETE',jobs=rows,ledger=str(ledger_path),completed_wall=time.time(),quality_values_exposed=False))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys
        try:
            d=Path(sys.argv[sys.argv.index('--deployment')+1]);c=json.loads(d.read_text())
            lane=int(sys.argv[sys.argv.index('--logical-gpu')+1])
            p.write(Path(c['control_root'])/f'worker-{lane}.json',dict(status='EXCEPTION',error=repr(exc),wall=time.time()))
        except Exception:pass
        raise
