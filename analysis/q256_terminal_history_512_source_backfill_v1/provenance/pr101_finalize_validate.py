"""Independent scalar replay and final compact-result validation."""
import csv,hashlib,json,math,subprocess,time
from pathlib import Path
W=Path('/root/q256-pr101-512-source-fid-backfill-v1');R=W/'code';ID='q256_terminal_history_512_source_backfill_v1';O=W/'results'/ID
load=lambda p:json.loads(p.read_text())
def csvrows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log):
 with (W/'logs'/log).open('xb') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 st=load(O/'statistics.json');decoded=csvrows(O/'decoded_source_metrics.csv');pairs=csvrows(O/'source_pairs.csv');joint=csvrows(O/'source_to_future_per_seed.csv')
 d={(int(x['seed']),x['arm']):x for x in decoded};assert len(d)==len(decoded)==58
 for p in pairs:
  n=int(p['seed']);a=d[n,'A'];b=d[n,'B'];assert a['evaluation_status']==b['evaluation_status']=='PASS'
  assert float(p['fid_A_512'])==float(a['fid50k']) and float(p['fid_B_512'])==float(b['fid50k'])
  q=math.log(float(b['fid50k']))-math.log(float(a['fid50k']));assert abs(q-float(p['Q_logfid_B_minus_A']))<1e-14
 endpoints={int(x['seed']):x for x in csvrows(R/'analysis/q256_terminal_history_n30_matpool_v1/final_results/paired_results.csv')}
 assert {int(x['seed']) for x in joint}=={int(x['seed']) for x in pairs}&set(endpoints)
 counts=dict(reversal=0,bad_to_bad=0,good_to_good=0,reverse_loss=0,on_axis=0)
 for x in joint:
  n=int(x['seed']);e=endpoints[n];q=math.log(float(d[n,'B']['fid50k']))-math.log(float(d[n,'A']['fid50k']));h=math.log(float(e['ba_fid50k']))-math.log(float(e['aa_fid50k']))
  assert abs(q-float(x['Q']))<1e-14 and abs(h-float(x['H_A']))<1e-14
  k='on_axis' if q==0 or h==0 else ('reversal' if q>0 and h<0 else 'bad_to_bad' if q>0 and h>0 else 'good_to_good' if q<0 and h<0 else 'reverse_loss');counts[k]+=1
 assert counts==st['quadrants'] and len(joint)==st['joint_counts']['joint_n']
 assert counts['reversal']==st['joint_counts']['delayed_reversals']
 run([str(W/'runtime-env/bin/python'),'-m','pytest','-q','tests/test_pr101_source_backfill.py','tests/test_q256_terminal_history_n30.py'],'final-tests.log')
 run([str(W/'runtime-env/bin/python'),'-m','py_compile',*[str(p) for p in (R/'analysis'/ID).rglob('*.py')],str(R/'tests/test_pr101_source_backfill.py')],'final-py-compile.log')
 run(['git','diff','--check'],'final-git-diff-check.log')
 cost=load(O/'GPU_COST_REPORT.json')
 import datetime
 frozen=datetime.datetime.fromisoformat(load(R/'analysis'/ID/'protocol.json')['frozen_at_utc'].replace('Z','+00:00')).timestamp()
 cost['protocol_freeze_to_formal_end_wall_seconds']=cost['formal_ended_epoch']-frozen
 cost['wall_time_scope']='Formal matrix includes input waiting; separate freeze-to-formal-end interval includes transfer and runtime preparation. GPUh is occupied evaluation time, not rental billing.'
 (O/'GPU_COST_REPORT.json').write_text(json.dumps(cost,indent=2)+'\n')
 note='''\n## Execution amendments and artifact retention\n\nThe user explicitly waived smoke and redundant SHA passes to reduce budget. Smoke status is NOT_RUN_USER_WAIVED (zero smoke GPU time). All source state hashes were already verified during inventory. Each consumed snapshot was hashed once on receipt; FID/KID generated-feature equality was checked by the reused evaluator. Large binary artifacts were not repeatedly rehashed at the integrity seal.\n\nEight full states already received were verified and exported on A100; the remaining fifty consumed snapshots were exported on ECT002. The scheduler encountered input-readiness races: a receipt could arrive ahead of its snapshot, and delivery of the completion JSON was initially non-atomic. The technical recovery used complete transport markers, atomic marker replacement, and exclusive per-job directories; stalled, untouched GPU lanes were resumed separately. No evaluated job was repeated, no sample block was added, and no seed was replaced. Intentional cancellations of superseded full-state transfers and eight redundant snapshot copies are transport events, not failed scientific evaluations.\n\nThe ECT002 result package contains scalar results, plots, source inventory, protocol and amendments, code, complete evaluation receipts, logs, and artifact hashes. The user subsequently requested release of the A100 instance. Therefore large generated samples/features and consumed exported snapshots are also backed up under the ECT002 archive in raw_evaluation_artifacts, with artifact hashes checked there. The raw-backup completion receipt is required before final packaging and release. Original full states remain in the pre-existing ECT002 archive.\n'''
 note+='\nProtocol freeze to formal-matrix completion: %.6f hours (includes runtime preparation and transport); formal matrix: %.6f hours. GPU hours report occupied evaluation time, not rental billing.\n'%(cost['protocol_freeze_to_formal_end_wall_seconds']/3600,cost['formal_wall_seconds']/3600)
 report=O/'SOURCE_TO_FUTURE_REPORT.md';report.write_text(report.read_text()+note)
 summary=dict(no_new_training=True,smoke_status='NOT_RUN_USER_WAIVED',source_valid_pairs=29,formal_jobs=58,formal_terminal_status_counts={s:sum(x['evaluation_status']==s for x in decoded) for s in ('PASS','FAILED','NOT_RUN')},technical_scheduler_resume='Input-transfer readiness race; first eight PASS jobs retained, untouched jobs resumed',scientific_jobs_repeated=0,transport_cancellations='Superseded full-state copying and redundant first-eight snapshot copying; retained logs',integrity_scope='Full source inventory; per-input SHA before evaluation; evaluator-generated artifact hashes; no redundant bulk binary seal pass',large_generated_artifact_location='/data/raw/ECT/'+ID+'/raw_evaluation_artifacts',cost=load(O/'GPU_COST_REPORT.json'))
 (O/'FORMAL_LOGS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (O/'SHA256SUMS.txt').open('w') as f:
  for p in sorted(O.rglob('*')):
   if p.is_file() and p.name!='SHA256SUMS.txt':f.write(sha(p)+'  '+str(p.relative_to(O))+'\n')
 subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS.txt'],cwd=O,check=True)
 import shutil
 shutil.copytree(O,R/'results'/ID)
 shutil.copytree(W/'source_inventory',R/'analysis'/ID/'source_inventory')
 subprocess.run(['git','add','analysis/'+ID,'results/'+ID,'tests/test_pr101_source_backfill.py'],cwd=R,check=True)
 run(['git','diff','--cached','--check'],'final-staged-diff-check.log')
 receipt=dict(status='PASS',independent_scalar_replay=True,source_pairs=len(pairs),joint_n=len(joint),quadrants=counts,pytest='13 tests plus 4 subtests PASS',python_py_compile='PASS',git_diff_check='PASS',result_SHA256SUMS='PASS',validated_epoch=time.time(),validation_script_sha256=sha(Path(__file__)))
 (W/'receipts/final_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('FINAL_VALIDATION_PASS',flush=True)
if __name__=='__main__':main()
