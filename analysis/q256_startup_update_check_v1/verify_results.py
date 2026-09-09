"""Run as a module from the repository; checkpoints load on CPU only."""
import csv,json,hashlib,pathlib,torch
import argparse
parser=argparse.ArgumentParser(description='CPU-only independent final checkpoint and original A telemetry verification')
parser.add_argument('--root',type=pathlib.Path,required=True)
ROOT=parser.parse_args().root
records=[];original=[];summary=[]
for seed in (50,51):
 for arm in ('A','D','D_compensate','A_mimic'):
  d=ROOT/'runs'/f'seed{seed}-{arm}'
  rows=[json.loads(x) for x in (d/'startup_attempts.jsonl').read_text().splitlines()]
  state=torch.load(d/'training-state-latest.pt',map_location='cpu',weights_only=False)
  result=dict(seed=seed,arm=arm,ordered_64_attempts=[r['attempted_index'] for r in rows]==list(range(1,65)),
      checkpoint_attempts=state['attempted_iteration'],checkpoint_success=state['successful_optimizer_steps'],
      processed_nimg=state['cur_nimg'],total_kimg=state['trajectory_config']['total_kimg'],
      engineering_only=state['startup_engineering']['engineering_only'],metadata_arm=state['startup_engineering']['arm'],
      sampler_consumed=[r['sampler_state']['consumed_samples'] for r in state['rank_states']],
      final_learning_rates=[g['lr'] for g in state['optimizer_state']['param_groups']],
      scaler_final=state['gradscaler_state']['scale'],skip_attempts=[r['attempted_index'] for r in rows if r['skip']],
      all_losses_finite=all(int(r['loss_nonfinite_count'])==0 for r in rows),
      all_updates_models_emas_finite=all(int(r[k])==0 for r in rows for k in ('update_nonfinite_count','model_nonfinite_count','ema_nonfinite_count')))
  assert result['ordered_64_attempts'] and result['checkpoint_attempts']==64 and result['processed_nimg']==8192 and result['total_kimg']==1024
  assert result['checkpoint_success']==rows[-1]['successful_optimizer_steps'] and result['engineering_only'] and result['metadata_arm']==arm
  assert result['sampler_consumed']==[8192] and result['final_learning_rates']==[1e-4]
  assert result['all_losses_finite'] and result['all_updates_models_emas_finite']
  result['status']='PASS';records.append(result);del state
  for r in rows:
   summary.append({k:r[k] for k in ('seed','arm','attempted_index','successful_optimizer_steps','skip','scale_before','scale_after','update_norm','raw_grad_norm','raw_grad_nonfinite_count','first_moment_norm','second_moment_norm')}|
      {'lr_multiplier':r['actual_step_plan'][0]['multiplier'] if r['actual_step_plan'] else None,
       'actual_lr':r['actual_step_plan'][0]['lr'] if r['actual_step_plan'] else None,
       'internal_steps':','.join(map(str,sorted(set(r['clocks_after']))))})
  if arm=='A':
   reference=ROOT/'provenance'/f'seed{seed}-original-first64.csv'
   old=list(csv.DictReader(reference.open()))
   keys=['batch_sha256','t_sha256','base_r_sha256','target_r_sha256','denominator_r_sha256','loss','raw_grad_norm','raw_grad_nonfinite_count','update_norm','model_norm','ema_norm','grad_scale_before','grad_scale_after','successful_optimizer_steps']
   original.append(dict(seed=seed,original_csv_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),fields={k:sum(str(x[k])==str(y[k]) for x,y in zip(rows,old)) for k in keys},rows=len(rows)))
out=ROOT/'results';out.mkdir(exist_ok=True)
(out/'final_state_verification.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'original_A_replay.json').write_text(json.dumps(original,indent=2)+'\n')
with (out/'diagnostic_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Verified final checkpoints:',len(records),'Attempts:',len(summary),'Original A replay:',original)
