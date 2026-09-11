"""Recompute the complete delayed-window result from committed, hash-bound blocks."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,pathlib
import numpy as np
from scipy import stats
ROOT=pathlib.Path(__file__).resolve().parents[2]
RESULTS=pathlib.Path(__file__).resolve().parent/'results'
ARMS=('AA','A_startup_down5','A_delayed_down6_10')
SEEDS=range(50,58)
BLOCKS=('B0','B1','B2')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def close(actual,expected,label):
    require(math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-12),label)
def paired(values):
    v=np.asarray(values,dtype=np.float64);mean=float(v.mean());sd=float(v.std(ddof=1));se=sd/math.sqrt(len(v));radius=float(stats.t.ppf(.975,len(v)-1)*se)
    p=float(2*stats.t.sf(abs(mean/se),len(v)-1)) if se else float(mean==0)
    return dict(mean_log_difference=mean,ci95_log=[mean-radius,mean+radius],p_two_sided=p,geometric_ratio=math.exp(mean),ci95_ratio=[math.exp(mean-radius),math.exp(mean+radius)])
def verify(root=RESULTS):
    hashes=read(root/'PUBLIC_SHA256.json')
    for rel,sha in hashes.items():
        p=root/rel
        require(p.is_file() and digest(p)==sha,'public artifact mismatch: '+rel)
    with (root/'blocks.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys={(int(r['seed']),r['arm'],r['block']) for r in rows}
    require(len(rows)==72 and keys==set(itertools.product(SEEDS,ARMS,BLOCKS)),'incomplete/duplicate fixed block matrix')
    old=read(root/'evidence/control/old_controls_48.json')
    oldmap={(r['seed'],r['arm'],r['block']):r for r in old}
    provenance=read(root/'RAW_PROVENANCE.json')
    values={}
    for r in rows:
        k=(int(r['seed']),r['arm'],r['block']);fid=float(r['FID']);kid=float(r['KID'])
        require(r['status']=='PASS' and math.isfinite(fid) and fid>0 and math.isfinite(kid),'invalid block')
        if r['arm']=='A_delayed_down6_10':
            rel=f"evaluation/seed{k[0]}-{k[2]}.json";e=read(root/'evidence'/rel)
            require(r['reused'].lower()=='false','new block marked reused')
            require(r['source_sha256']==provenance[rel]['raw_sha256'],'new block raw receipt binding changed')
        else:
            e=oldmap[k];require(r['reused'].lower()=='true','old block recomputed')
            source=ROOT/e['source_document']
            require(digest(source)==e['source_document_sha256'],'published prior-control source changed')
            published=[v for v in read(source) if v.get('seed')==k[0] and v.get('arm',v.get('path'))==k[1] and v.get('block')==k[2] and v.get('readout')=='E_512']
            require(len(published)==1,'prior published row is not unique')
            for field in ('FID','KID','readout','block','sample_seed_start','sample_seed_end','evaluator_commit'):
                require(published[0][field]==e[field],'prior published value differs: '+field)
            require(r['source_sha256']==e['source_document_sha256'],'old block source binding changed')
            if r['arm']=='AA':
                rel=f'historical_AA/seed{k[0]}-{k[2]}.json';aa=read(root/'evidence'/rel)
                require(provenance[rel]['raw_sha256']==e['receipt_sha256'],'historical AA receipt binding changed')
                close(fid,aa['metrics']['fid50k_full'],'AA raw FID differs');close(kid,aa['metrics']['kid50k_full'],'AA raw KID differs')
                e={**e,**{field:aa[field] for field in ('precision','nfe','metric_seed')}}
        close(fid,e['FID'],'FID receipt mismatch');close(kid,e['KID'],'KID receipt mismatch')
        require(e['readout']=='E_512' and e['precision']=='fp32' and int(e['nfe'])==1,'readout/precision/NFE drift')
        start={'B0':0,'B1':50000,'B2':100000}[k[2]]
        require((int(e['sample_seed_start']),int(e['sample_seed_end']),int(e['metric_seed']))==(start,start+49999,20260730),'generation block drift')
        require(e['evaluator_commit']=='d6aba02fb88e9db0993623895eb2228ed717d810','evaluator commit drift')
        values.setdefault(k[:2],[]).append(math.log(fid))
    per_seed=[]
    for seed in SEEDS:
        ys={arm:float(np.mean(values[(seed,arm)])) for arm in ARMS}
        per_seed.append(dict(seed=seed,**ys,T=ys[ARMS[2]]-ys[ARMS[1]],L=ys[ARMS[2]]-ys[ARMS[0]],M=ys[ARMS[1]]-ys[ARMS[0]]))
    saved=read(root/'statistics.json');require(saved['complete_n']==8 and saved['planned_n']==8,'wrong analysis unit count')
    for row,expected in zip(per_seed,saved['per_seed']):
        require(row['seed']==expected['seed'],'seed order differs')
        for name in (*ARMS,'T','L','M'):close(row[name],expected[name],name+' per-seed mismatch')
    for name in ('T','L','M'):
        recomputed=paired([r[name] for r in per_seed]);expected=saved['effects'][name]
        for field,value in recomputed.items():
            if isinstance(value,list):
                for i,x in enumerate(value):close(x,expected[field][i],name+' '+field)
            else:close(value,expected[field],name+' '+field)
    t=np.array([r['T'] for r in per_seed]);observed=abs(float(t.mean()));tol=np.finfo(float).eps*8*max(1.,float(np.abs(t).sum()))/8
    sign_p=sum(abs(float(np.mean(t*np.array(signs))))>=observed-tol for signs in itertools.product((-1,1),repeat=8))/256
    close(sign_p,saved['effects']['T']['sign_flip_p_two_sided'],'sign flip mismatch')
    se=float(t.std(ddof=1)/math.sqrt(8));bound=math.log(1.03);tost=max(float(stats.t.sf((float(t.mean())+bound)/se,7)),float(stats.t.cdf((float(t.mean())-bound)/se,7)))
    close(tost,saved['effects']['T']['auxiliary_TOST']['p'],'TOST mismatch')
    require(all(r['T']>0 for r in per_seed),'direction count changed')
    for seed in SEEDS:
        o=read(root/f'evidence/training/seed{seed}.json')
        require(o['status']=='PASS' and o['attempted_iteration']==8000,'incomplete trajectory')
        for b in BLOCKS:require(read(root/f'evidence/evaluation/seed{seed}-{b}.json')['checkpoint_sha256']==o['checkpoint_sha256'],'foreign endpoint')
        seals=[read(p) for p in (root/'evidence/checkpoints').glob(f'seed{seed}-*.json')]
        require(any(v['attempted_iteration']==8000 and v['sha256']==o['checkpoint_sha256'] for v in seals),'missing endpoint seal')
        require(any(v['attempted_iteration']==4000 for v in seals),'missing own 512 seal')
    sources=read(root/'update_diagnostics/source_manifest.json')['sources']
    require(len(sources)==16,'missing window diagnostics')
    for item in sources:
        p=root/'update_diagnostics'/f"seed{item['seed']}-{item['arm']}-first16.jsonl"
        require(digest(p)==item['prefix_sha256'],'diagnostic prefix binding changed')
        telemetry=[json.loads(line) for line in p.read_text().splitlines()]
        require(len(telemetry)==item['prefix_attempts'],'diagnostic prefix length differs')
        successful=0;scaled=0;lo,hi=(1,5) if item['arm']=='early' else (6,10)
        for attempt,event in enumerate(telemetry,1):
            require(event['attempted_iteration']==attempt,'attempt clock discontinuity')
            successful += int(not event['step_skipped'])
            require(event['successful_optimizer_steps']==successful and event['optimizer_clock']==successful,'AMP/success/optimizer clock disagreement')
            expected=1/1.1 if not event['step_skipped'] and lo<=successful<=hi else 1.0
            close(event['applied_multiplier'],expected,'wrong realized LR window')
            for lr in event['restored_lr']:close(lr,1e-4,'LR not restored')
            scaled += int(expected!=1.0)
        require(successful==16 and scaled==5,'realized window success count differs')
    return dict(status='PASS',training_seeds=8,new_training=8,new_blocks=24,reused_blocks=48,complete_triplets=8,sign_flip_enumerations=256,T=paired(t),sign_flip_p=sign_p,TOST_p=tost,all_eight_early_better=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=pathlib.Path,default=RESULTS);args=parser.parse_args()
    print(json.dumps(verify(args.results),indent=2))
