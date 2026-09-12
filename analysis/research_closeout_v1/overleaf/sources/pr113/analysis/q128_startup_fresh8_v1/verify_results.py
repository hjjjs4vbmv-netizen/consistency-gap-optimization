import pathlib,json,csv,hashlib,math,numpy as np
from scipy import stats
R=pathlib.Path(__file__).resolve().parent/'results'
J=pathlib.Path(__file__).resolve().parents[1]/'startup_window_experiments_v1/results'
for folder in (R,J):
 for name,sha in json.loads((folder/'PUBLIC_SHA256.json').read_text()).items():
  assert '..' not in pathlib.PurePosixPath(name).parts and not name.startswith('/')
  assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha, name
provenance=json.loads((R/'RAW_PROVENANCE.json').read_text())
index=json.loads((J/'ARCHIVE_INDEX.json').read_text()); indexed={(r['seed'],r['arm']):r for r in index['records'] if r['protocol_id']=='q128_startup_fresh8_v1'}
assert len(indexed)==32
s=json.loads((R/'statistics.json').read_text());rows=list(csv.DictReader((R/'blocks.csv').open()));arms=('AA','DA','A_startup_down5','D_startup_up5');keys={(int(r['seed']),r['arm'],r['block']) for r in rows}
assert len(rows)==96 and len(keys)==96 and keys=={(seed,arm,b) for seed in range(301,309) for arm in arms for b in ('B0','B1','B2')}
for row in rows:
 rel='seed'+row['seed']+'-'+row['arm']+'/'+row['block']+'.json';p=R/'evidence'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance[rel]['public_sha256'];assert row['source_sha256']==provenance[rel]['raw_sha256']==indexed[(int(row['seed']),row['arm'])]['evaluation_receipt_sha256'][row['block']];d=json.loads(p.read_text());assert d['status']=='PASS' and d['readout']=='E_512' and d['precision']=='fp32' and d['nfe']==1
 assert d['checkpoint_sha256']==indexed[(int(row['seed']),row['arm'])]['checkpoint_sha256']
 start={'B0':0,'B1':50000,'B2':100000}[row['block']];assert (d['sample_seed_start'],d['sample_seed_end'],d['metric_seed'])==(start,start+49999,20260730)
 assert d['evaluator_commit']=='d6aba02fb88e9db0993623895eb2228ed717d810'
 assert float(row['FID'])==d['FID'] and float(row['KID'])==d['KID'] and math.isfinite(d['KID'])
calc=[]
for seed in range(301,309):
 y={a:float(np.log([float(r['FID']) for r in rows if int(r['seed'])==seed and r['arm']==a]).mean()) for a in arms};m=y[arms[2]]-y['AA'];c=y[arms[3]]-y['DA'];h=y['DA']-y['AA'];calc.append(dict(seed=seed,**y,M=m,C=c,H=h,S=(c-m)/2))
results={}
for name in ('S','M','C','H'):
 v=np.array([r[name] for r in calc]);test=stats.ttest_1samp(v,0);ci=test.confidence_interval(confidence_level=.95);expected=s['effects'][name]
 def check(x,y):assert math.isclose(float(x),float(y),rel_tol=1e-11,abs_tol=1e-12),(name,x,y)
 check(v.mean(),expected['mean_log_difference']);check(test.pvalue,expected['p_two_sided']);check(ci.low,expected['ci95_log'][0]);check(ci.high,expected['ci95_log'][1])
 check(np.exp(v.mean()),expected['geometric_ratio']);check(np.exp(ci.low),expected['ci95_ratio'][0]);check(np.exp(ci.high),expected['ci95_ratio'][1])
 item=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),paired_dz=float(v.mean()/v.std(ddof=1)),t=float(test.statistic),p=float(test.pvalue),ci95=[float(ci.low),float(ci.high)],positive=int((v>0).sum()),negative=int((v<0).sum()))
 if name in ('S','M','C'):
  signs=2*((np.arange(256)[:,None]>>np.arange(8))&1)-1;prob=float((np.abs((signs*v).mean(axis=1))>=abs(float(v.mean()))-1e-14).mean());check(prob,expected['sign_flip_p_two_sided']);item['sign_flip_p']=prob
 results[name]=item
ordered=sorted(('M','C'),key=lambda k:results[k]['p']);previous=0.
for i,k in enumerate(ordered):
 adjusted=min(1,max(previous,(2-i)*results[k]['p']));previous=adjusted;assert math.isclose(adjusted,s['effects'][k]['p_holm_M_C'],abs_tol=1e-12)
for expected,actual in zip(s['per_seed'],calc):
 assert expected['seed']==actual['seed']
 for field in (*arms,'S','M','C','H'):assert math.isclose(expected[field],actual[field],abs_tol=1e-12)
assert s['complete_n']==8 and len(s['outcomes'])==32 and all(r['status']=='PASS' for r in s['outcomes'])
output=dict(status='PASS',verification_scope='Independent numerical reproduction from 96 hash-bound archived blocks; not GPU training rerun',complete_seeds=8,canonical_trajectories=32,blocks=96,effects=results,holm_M_C='PASS',source_hashes='96/96 PASS',operational_duplicates_excluded=s['operational_deviations']['excluded_partial_duplicate_trajectories'])
for (seed,arm),record in indexed.items():
 p=R/'evidence'/('seed'+str(seed)+'-'+arm)/'outcome.json';o=json.loads(p.read_text());assert o['status']=='PASS' and o['attempted_iteration']==8000 and o['checkpoint_sha256']==record['checkpoint_sha256']
 assert provenance['seed'+str(seed)+'-'+arm+'/outcome.json']['raw_sha256']==record['outcome_sha256']
print(json.dumps(output,indent=2))
