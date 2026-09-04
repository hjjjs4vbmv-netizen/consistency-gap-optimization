#!/usr/bin/env python3
import csv, hashlib, json, math, os, statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/root/q128_fresh_regime_history_n8_v1')
OLD=Path('/root/consistency-gap-optimization/results/q128_matched_spacing_20260824')
FRESH=ROOT/'final_analysis'
OUT=ROOT/'descriptive_evidence_synthesis'

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):
    with p.open('x') as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def write_text(p,x):
    with p.open('x') as f: f.write(x); f.flush(); os.fsync(f.fileno())
def summary(points):
    x=list(points.values())
    return {'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x),
            'negative_count':sum(v<0 for v in x),'positive_count':sum(v>0 for v in x),'seed_points':points}

if OUT.exists(): raise RuntimeError(f'refuse overwrite {OUT}')
# Verify the fresh verdict was already locked before touching the discovery bundle.
fresh_verdict=json.loads((FRESH/'final_verdict.json').read_text())
fresh_receipt=json.loads((FRESH/'final_receipt.json').read_text())
assert fresh_receipt['status']=='FINALIZED' and fresh_verdict['old_n3_pooled'] is False
fresh_input=json.loads((FRESH/'analysis_input.json').read_text())
idx={(r['seed'],r['trajectory'],r['budget_kimg'],r['nfe']):r['fid50k'] for r in fresh_input['records']}
fresh_points={str(s):math.log(idx[(s,'Bsame',1024,1)])-math.log(idx[(s,'A',1024,1)]) for s in fresh_input['effective_cohort']}
old_idx={}
with (OLD/'evaluation_results.csv').open() as f:
    for r in csv.DictReader(f):
        if int(r['kimg'])==1024 and int(r['nfe'])==1 and r['arm'] in ('A','Bsame') and r['status']=='SEALED_PASS':
            old_idx[(int(r['seed']),r['arm'])]=float(r['fid50k_full'])
old_points={str(s):math.log(old_idx[(s,'Bsame')])-math.log(old_idx[(s,'A')]) for s in (3,4,5)}
obj={
 'schema':'ect.q128-fresh-descriptive-evidence-synthesis/v1','created_utc':now(),
 'status':'DESCRIPTIVE_EVIDENCE_SYNTHESIS_ONLY','pooling_performed':False,'pooled_n':None,'pooled_p_value':None,
 'fresh_verdict_locked_before_old_bundle_access':True,
 'fresh_final_verdict_sha256':sha(FRESH/'final_verdict.json'),
 'fresh_final_receipt_sha256':sha(FRESH/'final_receipt.json'),
 'shared_descriptive_estimand':'D(1024)=logFID_NFE1(Bsame@1024)-logFID_NFE1(A@1024)',
 'old_discovery_reference':summary(old_points),'fresh_effective_cohort':summary(fresh_points),
 'fresh_sole_primary_separate':{
   'estimand':'H_A=logFID_NFE1(BA@1024)-logFID_NFE1(AA@1024)',
   'mean':json.loads((FRESH/'statistical_results.json').read_text())['sole_primary']['summary']['mean'],
   'directional_verdict':fresh_verdict['planned_directional_verdict']},
 'interpretation':'The old and fresh D(1024) summaries are shown side by side only. The old n=3 cohort is not part of fresh inference and cannot alter any fresh verdict.',
}
OUT.mkdir(mode=0o700)
write_json(OUT/'descriptive_evidence_synthesis.json',obj)
o=obj['old_discovery_reference']; n=obj['fresh_effective_cohort']; h=obj['fresh_sole_primary_separate']
md=f'''# DESCRIPTIVE EVIDENCE SYNTHESIS — old q128 n=3 vs fresh q128 n=8

This comparison was generated only after the fresh n=8 verdict and receipt were locked. No pooling or cross-cohort p-value is used.

| Cohort | Estimand | n | Mean | Median | Range | Negative seeds |
|---|---|---:|---:|---:|---:|---:|
| Old discovery reference (seeds 3–5) | D(1024) | {o['n']} | {o['mean']:.6g} | {o['median']:.6g} | [{o['min']:.6g}, {o['max']:.6g}] | {o['negative_count']}/{o['n']} |
| Fresh effective cohort | D(1024) | {n['n']} | {n['mean']:.6g} | {n['median']:.6g} | [{n['min']:.6g}, {n['max']:.6g}] | {n['negative_count']}/{n['n']} |

The shared descriptive estimand is `D(1024) = logFID_NFE1(Bsame@1024) - logFID_NFE1(A@1024)`.

The fresh study's separate sole primary `H_A = BA-AA` has mean {h['mean']:.6g} and planned directional verdict **{h['directional_verdict']}**. The old five-arm cohort has no BA trajectory and therefore cannot estimate H_A.

The old n=3 values are discovery reference only. They do not contribute to the fresh n=8 estimate, test, confidence interval, or verdict.
'''
write_text(OUT/'DESCRIPTIVE_EVIDENCE_SYNTHESIS.md',md)
receipt={'schema':'ect.q128-fresh-descriptive-synthesis-receipt/v1','created_utc':now(),'status':'FINALIZED_DESCRIPTIVE_ONLY',
         'pooling_performed':False,'artifacts':{n:{'sha256':sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in ['descriptive_evidence_synthesis.json','DESCRIPTIVE_EVIDENCE_SYNTHESIS.md']}}
write_json(OUT/'receipt.json',receipt)
write_text(OUT/'hashes.sha256',''.join(f"{sha(OUT/n)}  {n}\n" for n in ['descriptive_evidence_synthesis.json','DESCRIPTIVE_EVIDENCE_SYNTHESIS.md','receipt.json']))
for p in OUT.iterdir(): p.chmod(0o444)
OUT.chmod(0o555)
print(json.dumps({'status':'FINALIZED_DESCRIPTIVE_ONLY','old':o,'fresh':n,'receipt_sha256':sha(OUT/'receipt.json')},indent=2))
