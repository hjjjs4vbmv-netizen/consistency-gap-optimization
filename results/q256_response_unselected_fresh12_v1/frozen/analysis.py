"""Frozen seed-level analysis; completion gates never expose quality values."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy import stats
from . import protocol as p


def paired(values, equivalence=False):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite seed contrast')
    n=len(x)
    out=dict(n=n,mean_log=None,se_log=None,ci95_log=None,ci90_log=None,p_two_sided=None,
             geometric_ratio=None,ci95_ratio=None)
    if n==0:return out
    mean=float(x.mean())
    out.update(mean_log=mean,geometric_ratio=math.exp(mean))
    if n<2:return out
    se=float(x.std(ddof=1)/math.sqrt(n))
    prob=float(2*stats.t.sf(abs(mean/se),n-1)) if se else (1.0 if mean==0 else 0.0)
    r95=float(stats.t.ppf(.975,n-1)*se);r90=float(stats.t.ppf(.95,n-1)*se)
    out.update(se_log=se,ci95_log=[mean-r95,mean+r95],ci90_log=[mean-r90,mean+r90],
               p_two_sided=prob,ci95_ratio=[math.exp(mean-r95),math.exp(mean+r95)])
    if equivalence:
        bound=p.EQUIVALENCE_BOUND
        tost=max(float(stats.t.sf((mean+bound)/se,n-1)),float(stats.t.cdf((mean-bound)/se,n-1))) if se else (0.0 if -bound<mean<bound else 1.0)
        out['TOST']=dict(alpha=.05,bounds_log=[-bound,bound],p=tost,
            equivalent=mean-r90>-bound and mean+r90<bound)
    return out


def holm_three(effects):
    names=('M','C','S')
    if any(effects[k]['p_two_sided'] is None for k in names):
        for k in names:effects[k]['p_holm_M_C_S']=None
        return
    running=0.0
    for i,k in enumerate(sorted(names,key=lambda k:effects[k]['p_two_sided'])):
        running=max(running,(3-i)*effects[k]['p_two_sided'])
        effects[k]['p_holm_M_C_S']=min(1.0,running)


def contrasts(row):
    pairs={'H':('DA','AA'),'M':('A_startup_down5','AA'),'C':('D_startup_up5','DA')}
    out={k:(row[a]-row[b] if row.get(a) is not None and row.get(b) is not None else None)
         for k,(a,b) in pairs.items()}
    out['S']=(out['C']-out['M'])/2 if out['C'] is not None and out['M'] is not None else None
    return out


def classify(effect):
    if effect['n']<2:return dict(direction='unresolved',equivalence='unresolved')
    lo,hi=effect['ci95_log']
    direction='negative' if hi<0 else 'positive' if lo>0 else 'unresolved'
    return dict(direction=direction,equivalence='equivalent' if effect.get('TOST',{}).get('equivalent') else 'not_demonstrated')


def summarize(per_seed):
    complete=[r for r in per_seed if all(r.get(a) is not None for a in p.ARMS)]
    effects={k:paired([contrasts(r)[k] for r in complete],equivalence=k=='H') for k in ('H','M','C','S')}
    holm_three(effects)
    effects['S']['ratio_interpretation']='square root of ratio of FID ratios; not a model FID percent improvement'
    available={k:paired([contrasts(r)[k] for r in per_seed if contrasts(r)[k] is not None]) for k in ('H','M','C','S')}
    return dict(primary='H',complete_n=len(complete),planned_n=len(per_seed),underpowered=len(complete)<9,
        effects=effects,decision=classify(effects['H']),available_pairs_descriptive=available,
        per_seed=per_seed,missing_policy='complete four-arm NFE1 set',statistical_unit='training seed')


def completion(root, seeds):
    """Inspect statuses/identity only; do not return FID, KID, loss or norm values."""
    rows=[];pending=[];complete=[];bindings={}
    for seed in seeds:
        all_valid=True
        for arm in p.ARMS:
            run=Path(root)/'runs'/p.PROTOCOL_ID/f'seed{seed}'/arm
            outcome=run/'outcome.json'
            if not outcome.exists():
                pending.append(dict(seed=seed,arm=arm,reason='training pending'));all_valid=False;continue
            train=json.loads(outcome.read_text())
            if train.get('status') not in p.TERMINAL:
                pending.append(dict(seed=seed,arm=arm,reason='training not terminal'));all_valid=False;continue
            if (train.get('seed'),train.get('arm'),train.get('protocol_id'))!=(seed,arm,p.PROTOCOL_ID):
                raise RuntimeError('training identity mismatch')
            bindings[str(outcome)]=p.digest(outcome)
            statuses=[]
            for block in p.BLOCKS:
                receipt=run/'evaluation'/'NFE1'/(block+'.json')
                if not receipt.exists():
                    statuses.append('PENDING');continue
                d=json.loads(receipt.read_text())
                if (d.get('seed'),d.get('arm'),d.get('block'),d.get('nfe'))!=(seed,arm,block,1):
                    raise RuntimeError('evaluation identity mismatch')
                statuses.append(d['status']);bindings[str(receipt)]=p.digest(receipt)
            if any(s not in p.TERMINAL for s in statuses):
                pending.append(dict(seed=seed,arm=arm,reason='NFE1 statuses pending'))
            valid=train['status']=='PASS' and statuses==['PASS']*3
            all_valid &= valid
            rows.append(dict(seed=seed,arm=arm,training_status=train['status'],evaluation_statuses=statuses,valid=valid))
        if all_valid:complete.append(seed)
    return dict(status='TERMINAL' if not pending else 'PENDING',planned_seeds=list(seeds),
        complete_seeds=complete,complete_n=len(complete),rows=rows,pending=pending,bindings=bindings,
        quality_values_exposed=False)


def gate(root):
    root=Path(root)
    first=completion(root,p.SEEDS)
    if first['status']!='TERMINAL':return dict(status='WAITING_INITIAL_COHORT',pending=len(first['pending']))
    reserve=first['complete_n']<9
    decision=dict(protocol_id=p.PROTOCOL_ID,activate_reserve=reserve,initial_complete_n=first['complete_n'],
        reserve_seeds=list(p.RESERVE_SEEDS) if reserve else [],initial_bindings=first['bindings'],quality_values_exposed=False)
    p.write(root/'control'/'reserve_decision.json',decision,True)
    seeds=p.SEEDS+p.RESERVE_SEEDS if reserve else p.SEEDS
    all_=completion(root,seeds)
    if all_['status']!='TERMINAL':return dict(status='WAITING_RESERVE_COHORT',pending=len(all_['pending']))
    # The descriptive evaluation matrix must also be terminal before the single reveal.
    for row in all_['rows']:
        run=root/'runs'/p.PROTOCOL_ID/f"seed{row['seed']}"/row['arm']
        for block in p.BLOCKS:
            path=run/'evaluation'/'NFE2'/(block+'.json')
            if not path.exists() or json.loads(path.read_text()).get('status') not in p.TERMINAL:
                return dict(status='WAITING_DESCRIPTIVE_EVALUATION')
            all_['bindings'][str(path)]=p.digest(path)
    sealed=dict(status='READY_FOR_SINGLE_UNBLIND',protocol_id=p.PROTOCOL_ID,
        seeds=list(seeds),completion=all_,reserve_decision_sha256=p.digest(root/'control'/'reserve_decision.json'))
    p.write(root/'control'/'analysis_ready.json',sealed,True)
    return dict(status=sealed['status'],complete_n=all_['complete_n'])


def analyze(root, output):
    root=Path(root);output=Path(output)
    ready=root/'control'/'analysis_ready.json'
    if not ready.exists():raise RuntimeError('cohort completion and reserve decision must be sealed before unblinding')
    frozen=json.loads(ready.read_text())
    if frozen['status']!='READY_FOR_SINGLE_UNBLIND' or frozen['protocol_id']!=p.PROTOCOL_ID:
        raise RuntimeError('analysis gate identity differs')
    for path,expected in frozen['completion']['bindings'].items():
        if p.digest(path)!=expected:raise RuntimeError('bound outcome changed after completion seal')
    if p.digest(root/'control'/'reserve_decision.json')!=frozen['reserve_decision_sha256']:
        raise RuntimeError('reserve decision changed')
    blocks=[];per_seed=[]
    for seed in frozen['seeds']:
        row={'seed':seed}
        for arm in p.ARMS:
            records=[]
            for nfe in (1,2):
                for block in p.BLOCKS:
                    path=root/'runs'/p.PROTOCOL_ID/f'seed{seed}'/arm/'evaluation'/f'NFE{nfe}'/(block+'.json')
                    d=json.loads(path.read_text())
                    r={k:d.get(k) for k in ('seed','arm','block','nfe','status','FID','KID','checkpoint_sha256')}
                    r.update(source=str(path),source_sha256=p.digest(path));blocks.append(r)
                    if nfe==1:records.append(r)
            valid=all(r['status']=='PASS' for r in records)
            if valid and any(not isinstance(r['FID'],(int,float)) or not math.isfinite(r['FID']) or r['FID']<=0 for r in records):
                raise RuntimeError('valid receipt contains FID unsuitable for log analysis')
            row[arm]=float(np.mean([math.log(r['FID']) for r in records])) if valid else None
        row.update(contrasts(row));per_seed.append(row)
    result=summarize(per_seed)
    result.update(protocol_id=p.PROTOCOL_ID,status='ANALYZED',completion=frozen['completion']['rows'],
        analysis_ready_sha256=p.digest(ready),descriptive_blocks=blocks,
        claim_limits='Complete cases may remain selected when failure depends on treatment or response; no mechanism uniqueness or spacing interaction established.')
    output.mkdir(parents=True,exist_ok=True)
    p.write(output/'statistics.json',result,True)
    for name,rows in [('per_seed.csv',per_seed),('blocks.csv',blocks),('completion.csv',frozen['completion']['rows'])]:
        path=output/name
        if path.exists():raise RuntimeError('analysis output exists; no silent replacement')
        with path.open('x') as f:
            keys=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader();writer.writerows(rows)
    return dict(status='ANALYZED',complete_n=result['complete_n'],output=str(output))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--unblind',action='store_true')
    args=parser.parse_args()
    print(json.dumps(analyze(args.root,args.output) if args.unblind else gate(args.root)))


if __name__=='__main__':main()
