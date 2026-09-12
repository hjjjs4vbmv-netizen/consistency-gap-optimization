"""Bounded CPU-only audit of published existing records; never imports training code."""
import json,hashlib,math,csv,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];E=ROOT/'evidence';O=ROOT/'diagnostics';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def dump(name,x):(O/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def near(x,y):return math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-12)
bindings=read(ROOT/'SOURCE_BINDINGS.json');bind={x['public_file']:x for x in bindings['files']}
for name,b in bind.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==b['public_sha256'],name
index=read(E/'reports/ARCHIVE_INDEX.json');records=[r for r in index['records'] if r['protocol_id']=='q128_startup_fresh8_v1']
assert {(r['seed'],r['arm']) for r in records}=={(s,a) for s in range(301,309) for a in ('AA','DA','A_startup_down5','D_startup_up5')}
checks=[];attempts=[];windows=[];identities=[];by_run={}
for r in records:
 seed,arm=r['seed'],r['arm'];d=E/r['ect_directory'].split('<ECT_ROOT>/',1)[1];problems=[]
 def ck(test,label):
  if not test:problems.append(label)
 receipt=read(d/'initial_state_receipt_v1.json');options=read(d/'training_options.json');initialcheck=read(d/'startup_quality_initial_check.json')
 outcome=read(REPO/f'analysis/q128_startup_fresh8_v1/results/evidence/seed{seed}-{arm}/outcome.json');host=outcome['host']
 candidates=list(E.glob(f'**/control/manifests/q128_startup_fresh8_v1-seed{seed}-{arm}.json'))
 if host=='owned':candidates=[p for p in candidates if p.parent.parent==E/'control']
 else:candidates=[p for p in candidates if host in p.parts]
 mp=min(candidates,key=lambda p:len(p.parts));m=read(mp)
 canonical=[p for p in E.glob(f'**/references/q128/seed{seed}/initial_state_receipt_v1.json') if bind[str(p.relative_to(ROOT))]['raw_source_sha256']==m['reference_initial_receipt']['sha256']]
 ck(bool(canonical),'canonical receipt hash binding');canon=read(canonical[0])
 ck(receipt['hashes']==canon['hashes'],'full initial component hashes');ck(initialcheck['status']=='PASS','saved initial check')
 ck(receipt['attempted_iteration']==canon['attempted_iteration']==0,'fresh initialization')
 ck(m['q']==128 and m['global_gap_scale']==m['target_gap_scale']==1 and m['max_attempts']==8000 and m['switch_kimg']==512 and m['final_kimg']==1024,'manifest fixed science')
 ck((m['seed'],m['arm'],m['mode'])==(seed,arm,'formal'),'formal identity')
 factor=1.1 if arm in ('DA','D_startup_up5') else 1.
 expected_config=copy.deepcopy(read(REPO/'analysis/startup_window_experiments_v1/trajectory_template.json')['trajectory_config'])
 expected_config.update(seed=seed,rank_seed=seed)
 expected_config['loss_kwargs'].update(q=128.,factorial_protocol='q128_startup_native_v1',target_gap_scale=1.,denominator_gap_scale=factor)
 expected_config['dataset_kwargs']['path']=receipt['trajectory_config']['dataset_kwargs']['path']
 ck(receipt['trajectory_config']==expected_config,'full recorded scientific trajectory configuration')
 loss=options['loss_kwargs'];ck(loss['q']==128 and loss['target_gap_scale']==loss['global_gap_scale']==1 and loss['denominator_gap_scale']==factor,'effective prefix q/factors')
 for n in ['training-state-kimg000512.pt.sha256.json','training-state-branch-init-kimg000512.pt.sha256.json','training-state-kimg001024.pt.sha256.json']:
  seal=read(d/n);ck((seal['q'],seal['seed'],seal['arm'])==(128,seed,arm),'checkpoint seal q/identity')
  ck(seal['attempted_iteration']==(8000 if '001024' in n else 4000),'checkpoint boundary attempt')
  if '001024' in n:ck(seal['sha256']==r['checkpoint_sha256'],'endpoint seal binding')
 process=[p for p in d.glob('process-receipt-*.json') if bind[str(p.relative_to(ROOT))]['raw_source_sha256']==r['outcome_sha256']]
 ck(len(process)==1,'readable process receipt is byte-identical archived outcome')
 # Finished worker validates actual loss phase/q, controller clocks and E_512 count before writing PASS.
 ck(read(process[0])['status']=='PASS','existing endpoint validation receipt')
 rows=[json.loads(x) for x in (d/'startup_window_telemetry.jsonl.prefix16').read_text().splitlines()];by_run[(seed,arm)]=rows
 ck([x['attempted_iteration'] for x in rows]==list(range(1,len(rows)+1)),'continuous attempts')
 prev=0;previous_scale=None
 for row in rows:
  skip=bool(row['step_skipped']);success=row['successful_optimizer_steps'];proposed=prev+1
  ck(success==prev+int(not skip) and row['optimizer_clock']==success,'clock/skip increment')
  ck(row['proposed_success_step']==proposed,'proposed successful step')
  rho=2/(1-.999)-1-2*proposed*.999**proposed/(1-.999**proposed)
  ck(near(row['rho'],rho) and row['radam_branch']==('rectified' if rho>5 else 'non_adaptive'),'native rho branch')
  expected=(1/1.1 if arm=='A_startup_down5' else 1.1 if arm=='D_startup_up5' else 1.) if proposed<=5 and not skip else 1.
  ck(near(row['applied_multiplier'],expected) and row['base_lr']==1e-4 and row['restored_lr']==[1e-4],'applied/restored LR')
  ck(row['clock_evidence']==('successful_call_counter' if skip else 'all_active_parameters'),'native clock evidence class')
  ck(float(row['grad_scale_after'])==(float(row['grad_scale_before'])/2 if skip else float(row['grad_scale_before'])),'AMP scale event')
  if previous_scale is not None:ck(float(row['grad_scale_before'])==previous_scale,'AMP continuity')
  ck(row['q']==128 and math.isfinite(float(row['update_norm'])),'q/finite update')
  if skip:ck(float(row['update_norm'])==0,'skip zero update')
  attempts.append(dict(seed=seed,arm=arm,**{k:row[k] for k in ('attempted_iteration','successful_optimizer_steps','step_skipped','optimizer_clock','clock_evidence','proposed_success_step','rho','radam_branch','applied_multiplier','base_lr','grad_scale_before','grad_scale_after','update_norm','batch_sha256','t_sha256','base_r_sha256')},actual_lr=None if skip else row['base_lr']*row['applied_multiplier']))
  prev=success;previous_scale=float(row['grad_scale_after'])
 ck(prev==16,'prefix ends at successful16')
 for name,start,end in [('1_5',1,5),('6_10',6,10)]:
  active=[x for x in rows if not x['step_skipped'] and start<=x['successful_optimizer_steps']<=end];rec=rows[-1]['update_exposure'][name];normsum=sum(float(x['update_norm']) for x in active)
  ck(len(active)==rec['count']==5 and near(normsum,rec['sum_update_norm']),'saved scalar norm sum')
  ck(0<=rec['net_displacement_norm']<=normsum+1e-10,'recorded net displacement bounds')
  windows.append(dict(seed=seed,arm=arm,q=128,window=name,sum_update_norm=normsum,net_displacement_norm=rec['net_displacement_norm'],net_source='existing float64 window-vector accumulator; not reconstructed from scalar norms',first_attempt=active[0]['attempted_iteration'],last_attempt=active[-1]['attempted_iteration']))
 checks.append(dict(seed=seed,arm=arm,status='FAIL' if problems else 'PASS',problems=sorted(set(problems)),attempts_to_success16=len(rows),skips=sum(x['step_skipped'] for x in rows),successful_updates=prev,canonical_receipt=str(canonical[0].relative_to(ROOT)),actual_manifest=str(mp.relative_to(ROOT)),readable_outcome_equivalent=str(process[0].relative_to(ROOT)),endpoint_validation='existing worker PASS receipt bound to source code; no checkpoint reload this round',boundary_state_equality='existing explicit full-state check only for seed302 DA; remaining cases not independently rehashed'))
 identities.append(dict(seed=seed,arm=arm,canonical_initial_raw_sha256=m['reference_initial_receipt']['sha256'],initial_component_hashes=receipt['hashes'],manifest_raw_sha256=bind[str(mp.relative_to(ROOT))]['raw_source_sha256'],endpoint_sha256=r['checkpoint_sha256'],status=checks[-1]['status']))
# Diagnostics are prespecified descriptive pairings; never fit or filter by quality.
pairs=[]
for seed in range(301,309):
 for arm,ref in [('A_startup_down5','AA'),('D_startup_up5','DA'),('DA','AA')]:
  for window in ('1_5','6_10'):
   a=next(x for x in windows if (x['seed'],x['arm'],x['window'])==(seed,arm,window));b=next(x for x in windows if (x['seed'],x['arm'],x['window'])==(seed,ref,window))
   pairs.append(dict(seed=seed,candidate=arm,reference=ref,window=window,sum_norm_ratio=a['sum_update_norm']/b['sum_update_norm'],net_displacement_ratio=a['net_displacement_norm']/b['net_displacement_norm'],interpretation='diverged trajectories; not a same-state counterfactual dose'))
# Reuse existing engineering checks; do not execute optimizer tests again.
eng=[x for x in read(E/'control/global_engineering_receipts.json') if x['seed'] in (99001,99002)]
assert len(eng)==2 and all(x['status']=='PASS' and set(x['checks'])=={'AA','DA','A_startup_down5','D_startup_up5'} for x in eng)
# Incident chronology from retained pre-dispatch assignment and post-incident records.
incident=read(E/'supervision/duplicate_dispatch_incidents/1789124222736135699/DUPLICATE_ARCHIVE_VERIFIED.json');routing=read(E/'supervision/extra_six_amendment.json');cost=read(E/'supervision/excluded_duplicate_process_costs.json');inc=[]
for dup in incident['runs']:
 o=dup['outcome'];seed,arm=o['seed'],o['arm'];pub=REPO/f'analysis/q128_startup_fresh8_v1/results/evidence/seed{seed}-{arm}';canon=read(pub/'outcome.json');ev=[read(pub/f'B{i}.json') for i in range(3)];lane=next(l for l in routing['lanes'] if any((j['seed'],j['arm'])==(seed,arm) for j in l['jobs']))
 actual=sum(x['process_gpuh'] for x in cost.values() if x['job']==dup['job']);tests=dict(assignment_before_canonical_start=routing['wall']<canon['started_wall'],assignment_before_relevant_quality=routing['wall']<min(x['started_wall'] for x in ev),canonical_host_matches_prior_assignment=lane['host']==canon['host'],duplicate_is_distinct_host=o['host']!=canon['host'],duplicate_stopped=o['returncode']!=0,duplicate_no_quality=not incident['quality_evaluation_performed'] and not any('evaluation/' in f['path'] for f in dup['files']),cost_matches=near(actual,o['process_gpuh']),formal_endpoint_separate=canon['status']=='PASS' and 'checkpoint_sha256' in canon)
 inc.append(dict(seed=seed,arm=arm,status='PASS' if all(tests.values()) else 'FAIL',checks=tests,assignment_wall=routing['wall'],canonical_host=canon['host'],canonical_start=canon['started_wall'],first_quality_start=min(x['started_wall'] for x in ev),first_quality_end=min(x['ended_wall'] for x in ev),duplicate_host=o['host'],duplicate_start=o['started_wall'],duplicate_end=o['ended_wall'],duplicate_gpuh=actual,duplicate_after_some_canonical_quality=o['started_wall']>min(x['ended_wall'] for x in ev)))
old=read(REPO/'analysis/startup_window_experiments_v1/results/final_process_costs.json')
assert near(sum(x['duplicate_gpuh'] for x in inc),old['excluded_duplicate_process_gpuh'])
assert near(old['total_process_gpuh'],old['canonical_process_gpuh']+old['shared_preparation_gpuh']+old['excluded_duplicate_process_gpuh'])
assert {(x['job'],x['process_gpuh']) for x in cost.values()}=={(x['job'],x['process_gpuh']) for x in old['excluded_duplicates'].values()}
assert near(old['by_physical_host']['paid-node-04']['excluded_duplicate_gpuh'],old['excluded_duplicate_process_gpuh'])
dump('q128_execution_checks.json',checks);dump('q128_identities.json',identities);dump('reused_engineering_checks.json',eng);dump('duplicate_exclusion_audit.json',{'records':inc,'extra_gpuh':sum(x['duplicate_gpuh'] for x in inc),'included_in_existing_total_gpuh':old['total_process_gpuh'],'cost_ledger_reconciliation':'PASS','interpretation':'Prior assignment precedes affected quality results. For two jobs some canonical quality results already existed when duplicate launches occurred; exclusion is based on prior assignment, not a claim of pre-outcome incident handling.','existing_cost_ledger':'analysis/startup_window_experiments_v1/results/final_process_costs.json'})
csvout('q128_attempts_first16.csv',attempts);csvout('q128_window_exposure.csv',windows);csvout('q128_within_seed_comparisons.csv',pairs)
print(json.dumps({'formal_runs':len(checks),'PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'attempts':len(attempts),'successes':32*16,'duplicates':[x['status'] for x in inc],'new_training_generation_gpuh':0},indent=2))
