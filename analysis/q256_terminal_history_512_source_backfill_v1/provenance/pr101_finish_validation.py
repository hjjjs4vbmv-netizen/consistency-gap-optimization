import csv,json,math,subprocess,hashlib,time,shutil
from pathlib import Path
W=Path('/root/q256-pr101-512-source-fid-backfill-v1');R=W/'code';ID='q256_terminal_history_512_source_backfill_v1';O=W/'results'/ID
load=lambda p:json.loads(p.read_text())
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=load(O/'statistics.json');d={(int(x['seed']),x['arm']):x for x in rows(O/'decoded_source_metrics.csv')};p=rows(O/'source_pairs.csv');j=rows(O/'source_to_future_per_seed.csv');e={int(x['seed']):x for x in rows(R/'analysis/q256_terminal_history_n30_matpool_v1/final_results/paired_results.csv')}
assert len(d)==58 and all(x['evaluation_status']=='PASS' for x in d.values())
for x in p:
 n=int(x['seed']);q=math.log(float(d[n,'B']['fid50k']))-math.log(float(d[n,'A']['fid50k']));assert abs(q-float(x['Q_logfid_B_minus_A']))<1e-14
assert {int(x['seed']) for x in j}=={int(x['seed']) for x in p}&set(e)
count=dict(reversal=0,bad_to_bad=0,good_to_good=0,reverse_loss=0,on_axis=0)
for x in j:
 n=int(x['seed']);q=math.log(float(d[n,'B']['fid50k']))-math.log(float(d[n,'A']['fid50k']));h=math.log(float(e[n]['ba_fid50k']))-math.log(float(e[n]['aa_fid50k']));assert abs(q-float(x['Q']))<1e-14 and abs(h-float(x['H_A']))<1e-14
 k='on_axis' if q==0 or h==0 else ('reversal' if q>0 and h<0 else 'bad_to_bad' if q>0 and h>0 else 'good_to_good' if q<0 and h<0 else 'reverse_loss');count[k]+=1
assert count==s['quadrants']
report=O/'SOURCE_TO_FUTURE_REPORT.md';t=report.read_text();start=t.find('\n## Execution amendments and artifact retention')
if start>=0:t=t[:start]
t+='''\n## Execution amendments and release preparation\n\nSmoke was NOT_RUN_USER_WAIVED before formal results. Source SHA verification was completed during inventory. Each consumed snapshot passed its input hash and EMA binding check before generation; the reused evaluator checked shared FID/KID feature hashes. Repeated bulk binary hashes at the seal were omitted under the budget waiver.\n\nEight already received full states were exported on A100; fifty source snapshots were exported on ECT002. Input-readiness races (receipt arrival before snapshot completion and initially non-atomic JSON completion delivery) interrupted only untouched jobs. Atomic completion markers and exclusive job directories resolved the scheduling issues. No scientific job was repeated, no sample block added, and no seed replaced. Original error logs and recovery scripts are retained.\n\nThe user subsequently requested release of the A100 instance. Generated samples/features and consumed exported snapshots are therefore also returned and hash-verified under the ECT002 archive's raw_evaluation_artifacts directory. Raw-backup completion, compact-package remote SHA verification, and a new results PR are required before release. Original source checkpoints remain in the original ECT002 archive.\n\nThe final archive timing receipt separately records preparation and return delays. Evaluation GPUh measures exclusive evaluation occupancy, not rental billing.\n'''
report.write_text(t)
(O/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
summary=load(O/'FORMAL_LOGS_SUMMARY.json');summary['large_generated_artifact_location']='/data/raw/ECT/'+ID+'/raw_evaluation_artifacts';summary['technical_scheduler_resume']='Input readiness and non-atomic marker races; untouched lanes resumed without rerunning scientific jobs';(O/'FORMAL_LOGS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
with (O/'SHA256SUMS.txt').open('w') as f:
 for x in sorted(O.rglob('*')):
  if x.is_file() and x.name!='SHA256SUMS.txt':f.write(sha(x)+'  '+str(x.relative_to(O))+'\n')
shutil.copytree(O,R/'results'/ID,dirs_exist_ok=True)
commands=[([str(W/'runtime-env/bin/python'),'-m','pytest','-q','tests/test_pr101_source_backfill.py','tests/test_q256_terminal_history_n30.py'],'final-v2-tests.log'),([str(W/'runtime-env/bin/python'),'-m','py_compile',*[str(x) for x in (R/'analysis'/ID).rglob('*.py')]],'final-v2-pycompile.log'),(['git','add','--sparse','analysis/'+ID,'tests/test_pr101_source_backfill.py'],'final-v2-stage-analysis.log'),(['git','add','--sparse','-f','results/'+ID],'final-v2-stage-results.log'),(['git','--no-pager','diff','--cached','--check'],'final-v2-staged-diff.log'),(['git','--no-pager','diff','--check'],'final-v2-unstaged-diff.log')]
for cmd,name in commands:
 with (W/'logs'/name).open('xb') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS.txt'],cwd=O,check=True)
(W/'receipts/final_validation.json').write_text(json.dumps(dict(status='PASS',independent_scalar_replay=True,source_pairs=len(p),joint_n=len(j),quadrants=count,pytest='13 tests and 4 subtests PASS',python_py_compile='PASS',git_diff_check='PASS including staged files; CSV CRLF preserved via scoped attributes',result_SHA256SUMS='PASS',plot_visual_review='PASS: all seed labels, strict-sign axes, no regression line',validation_script_sha256=sha(Path(__file__)),validated_epoch=time.time()),indent=2)+'\n')
print('FINAL_VALIDATION_PASS')
