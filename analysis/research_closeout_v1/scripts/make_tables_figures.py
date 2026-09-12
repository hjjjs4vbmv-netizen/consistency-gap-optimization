"""CPU-only tables and figures from the fixed, already verified data; no selection."""
import csv,json,math,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];REPO=R.parents[1];D=R/'diagnostics';F=R/'figures';F.mkdir(exist_ok=True);T=R/'tables';T.mkdir(exist_ok=True)
ARMS=['AA','DA','A_startup_down5','D_startup_up5'];LABELS=['AA','DA','A down (1–5)','D up (1–5)'];COLORS=['#355070','#80738b','#118a7e','#c46b35'];SEEDS=list(range(301,309))
rows=list(csv.DictReader((REPO/'analysis/q128_startup_fresh8_v1/results/blocks.csv').open()));stats=json.loads((REPO/'analysis/q128_startup_fresh8_v1/results/statistics.json').read_text())
def csvwrite(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def save(fig,name):
 for ext in ('png','svg','pdf'):fig.savefig(F/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
ys={};out=[]
for seed in SEEDS:
 y={a:float(np.log([float(r['FID']) for r in rows if int(r['seed'])==seed and r['arm']==a]).mean()) for a in ARMS};ys[seed]=y;m=y[ARMS[2]]-y['AA'];c=y[ARMS[3]]-y['DA'];h=y['DA']-y['AA'];out.append(dict(seed=seed,**{a:math.exp(v) for a,v in y.items()},M=m,C=c,H=h,S=(c-m)/2))
csvwrite(T/'q128_complete_per_seed.csv',out)
effects=[]
for name in ['S','M','C','H']:
 x=stats['effects'][name];effects.append(dict(contrast=name,mean_log=x['mean_log_difference'],lower=x['ci95_log'][0],upper=x['ci95_log'][1],p=x['p_two_sided'],holm_p=x.get('p_holm_M_C'),geometric_ratio=x['geometric_ratio'],percent_change=None if name=='S' else 100*(x['geometric_ratio']-1)))
csvwrite(T/'q128_effects.csv',effects)
(T/'q128_rows.tex').write_text(''.join(f"{r['seed']} & "+' & '.join(f"{r[a]:.4f}" for a in ARMS)+f" & {r['M']:+.5f} & {r['C']:+.5f} & {r['H']:+.5f} & {r['S']:+.5f} \\\\\n" for r in out))
(T/'q128_effect_rows.tex').write_text(''.join(f"${x['contrast']}$ & {x['mean_log']:+.6f} & $[{x['lower']:+.6f},{x['upper']:+.6f}]$ & {x['p']:.6f} & "+(f"{x['holm_p']:.6f}" if x['holm_p'] is not None else '---')+' \\\\\n' for x in effects))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.35,1]},constrained_layout=True)
for i,seed in enumerate(SEEDS):
 vals=[math.exp(ys[seed][a]) for a in ARMS];axs[0].plot(range(4),vals,'-o',alpha=.8,lw=1,label=str(seed),markersize=4)
axs[0].set(xticks=range(4),xticklabels=LABELS,ylabel='Geometric mean of three FID50k blocks',title='Fresh q128: all eight training seeds');axs[0].legend(title='Training seed',ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15));axs[0].grid(axis='y',alpha=.2)
for i,e in enumerate(effects):
 axs[1].errorbar(e['mean_log'],i,xerr=[[e['mean_log']-e['lower']],[e['upper']-e['mean_log']]],fmt='o',color='#355070',capsize=4)
axs[1].axvline(0,color='grey',ls='--',lw=1);axs[1].set(yticks=range(4),yticklabels=['S (primary)','M (secondary)','C (secondary)','H (supportive)'],xlabel='Paired log-FID contrast; nominal 95% t CI',title='Training seed is the unit of inference');axs[1].invert_yaxis();axs[1].grid(axis='x',alpha=.2)
fig.suptitle('q128: 32 trajectories, 96 blocks, n = 8');save(fig,'q128_endpoints_effects')
comp=list(csv.DictReader((D/'q128_within_seed_comparisons.csv').open()));fig,axs=plt.subplots(2,2,figsize=(10,6.5),sharex=True,constrained_layout=True)
for j,window in enumerate(['1_5','6_10']):
 for i,key in enumerate(['sum_norm_ratio','net_displacement_ratio']):
  ax=axs[i,j]
  for arm,ref,color in [('A_startup_down5','AA',COLORS[2]),('D_startup_up5','DA',COLORS[3]),('DA','AA',COLORS[1])]:
   selected=[x for x in comp if x['candidate']==arm and x['window']==window];ax.plot([int(x['seed']) for x in selected],[float(x[key]) for x in selected],'-o',label=f'{LABELS[ARMS.index(arm)]} / {ref}',color=color,markersize=4)
  ax.axhline(1,color='grey',ls='--',lw=1);ax.grid(alpha=.2);ax.set(xticks=SEEDS,title=f"Successful updates {window.replace('_','–')}" if i==0 else None,ylabel='Sum of update norms: ratio' if i==0 else 'Recorded net displacement: ratio',xlabel='Training seed' if i==1 else None)
axs[0,0].legend(fontsize=8);fig.suptitle('q128 within-seed trajectory comparisons (not same-state counterfactual doses)');save(fig,'q128_local_response')
win=list(csv.DictReader((D/'q128_window_exposure.csv').open()));old=list(csv.DictReader((REPO/'analysis/q256_startup_delayed6_10_v1/results/update_diagnostics/window_update_exposure.csv').open()))
combined=[dict(q=128,cohort='fresh301_308',seed=x['seed'],arm=x['arm'],window=x['window'],sum_update_norm=x['sum_update_norm'],net_displacement_norm=x['net_displacement_norm'],net_source=x['net_source']) for x in win]
combined += [dict(q=256,cohort='responder_enriched50_57',seed=x['seed'],arm=x['arm'],window=x['window'],sum_update_norm=x['sum_update_norm'],net_displacement_norm=x['net_displacement_norm'] or None,net_source=x['net_displacement_note']) for x in old]
csvwrite(T/'descriptive_q128_q256_exposure.csv',combined)
exposure_tex=[]
for seed in range(301,309):
 for arm,label in [('AA','AA'),('DA','DA'),('A_startup_down5',r'$\Adown$'),('D_startup_up5',r'$\Dup$')]:
  readings=[next(x for x in win if int(x['seed'])==seed and x['arm']==arm and x['window']==w) for w in ('1_5','6_10')]
  values=[float(x[k]) for x in readings for k in ('sum_update_norm','net_displacement_norm')]
  exposure_tex.append(str(seed)+' & '+label+' & '+' & '.join(f'{v:.6f}' for v in values)+r' \\'+'\n')
(T/'q128_exposure_rows.tex').write_text(''.join(exposure_tex))
# Additive copies only; original PR outputs remain untouched.
over=R/'overleaf'
for f in T.glob('q128*rows.tex'):shutil.copyfile(f,over/'data'/f.name)
for f in F.glob('*'):shutil.copyfile(f,over/'figures/en'/f.name)
for f in [T/'q128_complete_per_seed.csv',T/'q128_effects.csv',D/'q128_window_exposure.csv',T/'descriptive_q128_q256_exposure.csv']:shutil.copyfile(f,over/'data/raw'/f.name)
print(json.dumps({'seeds':8,'trajectories':32,'blocks':96,'figures':2,'exposure_rows':len(combined),'missing_q256_early_net_displacements':sum(x['net_displacement_norm'] is None for x in combined),'effects':effects},indent=2))
