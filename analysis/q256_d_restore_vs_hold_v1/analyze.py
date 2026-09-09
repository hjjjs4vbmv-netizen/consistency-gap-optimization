"""Offline prespecified seed-level inference. No training or evaluation dispatch."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
from . import protocol as p


def paired(values):
    a=np.asarray(values,dtype=float); n=len(a)
    result={'n':n,'mean_log_difference':None,'seed_sd':None,'ci95':None,'p_two_sided':None,
            'geometric_ratio':None,'ratio_ci95':None,'relative_change_percent':None,
            'ci90':None,'tost_p':None,'practically_equivalent':None,
            'equivalence_bounds':[-math.log(1.03),math.log(1.03)],
            'direction_counts':{'negative':int((a<0).sum()),'zero':int((a==0).sum()),'positive':int((a>0).sum())}}
    if not n: return result
    mean=float(a.mean()); result.update(mean_log_difference=mean,geometric_ratio=math.exp(mean),relative_change_percent=100*math.expm1(mean))
    if n<2: return result
    sd=float(a.std(ddof=1)); se=sd/math.sqrt(n); margin=math.log(1.03)
    ci95=[mean-float(t.ppf(.975,n-1))*se,mean+float(t.ppf(.975,n-1))*se]
    ci90=[mean-float(t.ppf(.95,n-1))*se,mean+float(t.ppf(.95,n-1))*se]
    if se:
        pv=float(2*t.sf(abs(mean/se),n-1))
        tost=max(float(t.sf((mean+margin)/se,n-1)),float(t.cdf((mean-margin)/se,n-1)))
    else:
        pv=1.0 if mean==0 else 0.0
        tost=0.0 if -margin<mean<margin else 1.0
    result.update(seed_sd=sd,ci95=ci95,p_two_sided=pv,ratio_ci95=[math.exp(x) for x in ci95],
                  ci90=ci90,tost_p=tost,practically_equivalent=(-margin<ci90[0] and ci90[1]<margin),
                  direction_inference=('A_favored' if ci95[1]<0 else 'D_favored' if ci95[0]>0 else 'unresolved'))
    return result


def validate(rows,arm):
    expected={(s,r,b) for s in p.SEEDS for r,blocks in p.READOUT_BLOCKS.items() for b in blocks}
    if len(rows)!=80 or {(r['seed'],r['readout'],r['block']) for r in rows}!=expected:
        raise ValueError('incomplete or duplicate '+arm+' slots')
    for r in rows:
        if r['path']!=arm: raise ValueError('wrong path')
        if r['status']=='PASS':
            if not isinstance(r.get('FID'),(float,int)) or not math.isfinite(r['FID']) or r['FID']<=0:
                raise ValueError('PASS requires positive finite FID')
            if not isinstance(r.get('KID'),(float,int)) or not math.isfinite(r['KID']):
                raise ValueError('PASS requires finite KID')
        elif r.get('FID') is not None or r.get('KID') is not None:
            raise ValueError('failed or pending slot has a metric')
        if arm!='AA':
            b=int(r['block'][1]); expected_config={'nfe':1,'precision':'fp32','metric_seed':20260730,
                    'sample_seed_start':b*50000,'sample_seed_end':(b+1)*50000-1}
            for k,v in expected_config.items():
                if r.get(k)!=v: raise ValueError('unpaired evaluation setting: '+k)
    return {(r['seed'],r['readout'],r['block']):r for r in rows}


def summarize(new):
    da=p.import_controls(); aa=p.import_controls('AA')
    indexes={arm:validate(rows,arm) for arm,rows in [('DA',da),('AA',aa),('DD',new)]}
    out={'status':'INCOMPLETE_TECHNICAL','planned_seeds':list(p.SEEDS),'primary':{},'per_seed':[],
         'outcomes':[],'auxiliary':{},'secondary_readouts':{},'full_cohort_equivalence':False,
         'inference_scope':'complete finite pairs; post-PR108 observed cohort'}
    for seed in p.SEEDS:
        rows=[r for r in new if r['seed']==seed]
        statuses={r['status'] for r in rows}
        status='PASS' if statuses=={'PASS'} else 'NO_ENDPOINT' if statuses=={'NO_ENDPOINT'} else 'TECHNICAL_OR_PENDING'
        out['outcomes'].append({'seed':seed,'DA':'PASS','DD':status,'AA':'NO_ENDPOINT' if seed in (58,65) else 'PASS'})
    if any(r['status'] not in {'PASS','NO_ENDPOINT'} for r in new): return out
    if any(r['DD']=='TECHNICAL_OR_PENDING' for r in out['outcomes']):
        raise ValueError('mixed NO_ENDPOINT/PASS cannot describe one completed training trajectory')
    def complete(arm,seed,readout='E_512'):
        return all(indexes[arm][seed,readout,b]['status']=='PASS' for b in p.READOUT_BLOCKS[readout])
    def mean(arm,seed,readout='E_512',metric='FID'):
        vals=[indexes[arm][seed,readout,b][metric] for b in p.READOUT_BLOCKS[readout]]
        return float(np.mean(np.log(vals) if metric=='FID' else vals))
    valid=[s for s in p.SEEDS if complete('DA',s) and complete('DD',s)]
    out['complete_pair_seeds']=valid
    out['per_seed']=[{'seed':s,'DA_mean_log_FID':mean('DA',s),'DD_mean_log_FID':mean('DD',s),
                      'DA_minus_DD':mean('DA',s)-mean('DD',s),
                      'geometric_ratio':math.exp(mean('DA',s)-mean('DD',s))} for s in valid]
    out['primary']=paired([r['DA_minus_DD'] for r in out['per_seed']])
    out['full_cohort_equivalence']=len(valid)==16 and out['primary']['practically_equivalent'] is True
    for readout in p.READOUT_BLOCKS:
        seeds=[s for s in p.SEEDS if complete('DA',s,readout) and complete('DD',s,readout)]
        out['secondary_readouts'][readout]={'seeds':seeds,
            'FID_DA_minus_DD':paired([mean('DA',s,readout)-mean('DD',s,readout) for s in seeds]),
            'KID_raw_DA_minus_DD':[{'seed':s,'difference':mean('DA',s,readout,'KID')-mean('DD',s,readout,'KID')} for s in seeds]}
    common=[s for s in valid if complete('AA',s)]
    out['auxiliary']={'common_seeds':common,'n':len(common),'comparisons':{},
                     'per_seed':[{'seed':s,**{arm:mean(arm,s) for arm in ('AA','DA','DD')}} for s in common]}
    for a,b in [('DA','AA'),('DD','AA'),('DA','DD')]:
        out['auxiliary']['comparisons'][a+'-'+b]=paired([mean(a,s)-mean(b,s) for s in common])
    out['status']='COMPLETE' if len(valid)==16 else 'COMPLETE_WITH_SCIENTIFIC_FAILURES'
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dd-slots',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    result=summarize(json.loads(a.dd_slots.read_text()))
    (a.output/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name in ('per_seed','outcomes'):
        if result[name]: p.write_csv(a.output/(name+'.csv'),result[name])
    print(json.dumps({'status':result['status'],'n':result['primary'].get('n')}))


if __name__=='__main__': main()
