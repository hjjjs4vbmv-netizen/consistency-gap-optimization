"""Minimal fail-closed runtime, source, frozen-analysis and engineering checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from analysis.q256_history_component_chase_v1.preflight import runtime
from training import schedule_switch as sw
from scripts.run_m1_training_slot import write_json
from . import protocol as p


def check(config,engineering):
    if runtime()!=p.EXPECTED_RUNTIME:raise RuntimeError('production runtime differs')
    frozen=p.ROOT/'analysis/q256_d_restore_vs_hold_v1/frozen'
    freeze=json.loads((frozen/'analysis_freeze.json').read_text())
    for name,h in freeze['files_sha256'].items():
        path=p.ROOT/'analysis/q256_d_restore_vs_hold_v1'/name if name.endswith('.py') else frozen/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise RuntimeError('analysis freeze changed: '+name)
    baseline=json.loads((p.ROOT/'analysis/q256_terminal_history_n30_matpool_v1/protocol.json').read_text())
    for k in ('dataset','transfer'):
        if sw.sha256_file(config[k])!=baseline['assets'][k]['sha256']:raise RuntimeError('asset differs: '+k)
    report=json.loads(Path(engineering).read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.ROOT,text=True).strip()
    if report['status']!='PASS':raise RuntimeError('engineering checks missing')
    checked=report['implementation_commit']
    scientific_paths=['ct_train.py','training','torch_utils','dnnlib',
                      'analysis/q256_d_restore_vs_hold_v1/protocol.py',
                      'analysis/q256_d_restore_vs_hold_v1/frozen']
    changed=subprocess.check_output(['git','diff','--name-only',checked,head,'--',*scientific_paths],cwd=p.ROOT,text=True).strip()
    if changed:raise RuntimeError('scientific implementation changed after GPU checks: '+changed)
    if report['process_gpuh']>3:raise RuntimeError('engineering exceeded its fixed process scope')
    rows=json.loads(Path(config['source_inventory']).read_text())
    for seed in config['assigned_seeds']:
        row=next(r for r in rows if r['seed']==seed)
        for k,h in [('prefix_path','prefix_sha256'),('da_branch_init_path','da_branch_init_sha256')]:
            if not Path(row[k]).is_file():raise RuntimeError('required full state unavailable')
            if sw.sha256_file(row[k])!=row['binding'][h]:raise RuntimeError('staged source differs')
    return {'status':'PASS','implementation_commit':head,'runtime':runtime(),'assigned_seeds':config['assigned_seeds'],
            'engineering_implementation_commit':checked,'scientific_source_unchanged_since_checks':True,
            'analysis_freeze':freeze,'engineering_receipt':str(engineering),'short_checks':report}


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--deployment',type=Path,required=True)
    a.add_argument('--engineering',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    x=a.parse_args();write_json(x.output,check(json.loads(x.deployment.read_text()),x.engineering))
