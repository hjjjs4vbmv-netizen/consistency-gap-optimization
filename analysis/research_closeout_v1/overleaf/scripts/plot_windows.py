#!/usr/bin/env python3
"""Recompute PR112 contrasts and draw window figures from the pinned records.

No training, sample generation, or imputation. Success and attempt clocks remain
separate. Raw prefixes provide norms and AMP values; early rho is derived from
the observed clock using the native RAdam formula, not claimed as logged rho.
"""
from pathlib import Path
import io, itertools, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

P = Path(__file__).resolve().parents[1]
R = P/'sources/pr112/analysis/q256_startup_delayed6_10_v1/results'
HEAD = '3d1330f5d794d05fe63b67aa40e2dc6213f1387f'
SCIENTIFIC = 'fcc94d26b1085011ac935084f7c1cf4c149e2947'
ARMS = ['AA', 'A_startup_down5', 'A_delayed_down6_10']
SEEDS = list(range(50,58))
archive = json.loads((R/'statistics.json').read_text())
blocks = pd.read_csv(R/'blocks.csv', float_precision='round_trip')
assert len(blocks)==72 and not blocks.duplicated(['seed','arm','block']).any()
assert blocks.status.eq('PASS').all()
assert np.isfinite(blocks.FID).all() and (blocks.FID>0).all()
assert blocks.groupby(['seed','arm']).size().eq(3).all()
assert int(blocks.reused.sum())==48
Y = np.array([[np.log(blocks[(blocks.seed==seed)&(blocks.arm==arm)].FID).mean()
               for arm in ARMS] for seed in SEEDS])
old = pd.read_csv(P/'data/raw/startup_per_seed.csv', float_precision='round_trip').set_index('seed')
for i,seed in enumerate(SEEDS):
    for j,arm in enumerate(ARMS):
        np.testing.assert_allclose(Y[i,j], archive['per_seed'][i][arm], rtol=0, atol=2e-14)
        if arm!='A_delayed_down6_10':
            for b in ['B0','B1','B2']:
                new_fid = blocks[(blocks.seed==seed)&(blocks.arm==arm)&(blocks.block==b)].FID.item()
                assert new_fid==old.loc[seed,f'{arm}_{b}_FID']
values={'T':Y[:,2]-Y[:,1], 'L':Y[:,2]-Y[:,0], 'M':Y[:,1]-Y[:,0]}
assert np.all(values['T']>0)
recomputed={}
for key,v in values.items():
    mean=float(v.mean()); sem=stats.sem(v)
    ci=np.array([mean,mean])+np.array([-1,1])*stats.t.ppf(.975,7)*sem
    p=float(stats.ttest_1samp(v,0).pvalue)
    a=archive['effects'][key]
    np.testing.assert_allclose([mean,*ci,p],[a['mean_log_difference'],*a['ci95_log'],a['p_two_sided']],rtol=0,atol=2e-13)
    recomputed[key]={'mean_log_difference':mean,'ci95_log':ci.tolist(),
      'two_sided_t_p':p,'ratio':float(np.exp(mean)), 'ratio_ci95':np.exp(ci).tolist(),
      'percent':float(100*np.expm1(mean)), 'percent_ci95':(100*np.expm1(ci)).tolist()}
v=values['T'];m=float(v.mean());sem=stats.sem(v);margin=np.log(1.03)
flips=np.array(list(itertools.product([-1,1],repeat=8)))
pflip=float(np.mean(np.abs(flips@v/8)>=abs(m)-1e-14))
ptost=float(max(stats.t.sf((m+margin)/sem,7),stats.t.cdf((m-margin)/sem,7)))
np.testing.assert_allclose([pflip,ptost],[archive['effects']['T']['sign_flip_p_two_sided'],archive['effects']['T']['auxiliary_TOST']['p']],atol=2e-13,rtol=0)
recomputed['T'].update(sign_flip_p=pflip,tost_p=ptost)

prefixes={}; traces={}; exposures={}; total=0; skips=0
published_exposure=pd.read_csv(R/'update_diagnostics/window_update_exposure.csv',float_precision='round_trip')
for seed in SEEDS:
    for arm in ['early','delayed']:
        rows=[json.loads(s) for s in (R/f'update_diagnostics/seed{seed}-{arm}-first16.jsonl').read_text().splitlines()]
        prefixes[seed,arm]=rows; total+=len(rows)
        good=[r for r in rows if not bool(int(r['step_skipped']))]
        skips+=len(rows)-len(good)
        assert [int(r['successful_optimizer_steps']) for r in good]==list(range(1,17))
        expected=np.ones(16)
        expected[0:5 if arm=='early' else 0]=1/1.1
        if arm=='delayed': expected[5:10]=1/1.1
        multipliers=np.array([float(r['applied_multiplier']) for r in good])
        np.testing.assert_allclose(multipliers,expected,atol=1e-14,rtol=0)
        norms=np.array([float(r['update_norm']) for r in good])
        assert np.isfinite(norms).all() and (norms>0).all()
        if arm=='delayed':
            for row in good:
                step=int(row['successful_optimizer_steps'])
                rho=1999-2*step*.999**step/(1-.999**step)
                np.testing.assert_allclose(float(row['rho']),rho,atol=2e-10,rtol=0)
                assert row['radam_branch']==('rectified' if rho>5 else 'non_adaptive')
        traces[seed,arm]={'norms':norms,'multipliers':multipliers}
        for window,slice_ in [('1_5',slice(0,5)),('6_10',slice(5,10))]:
            val=float(norms[slice_].sum());exposures[seed,arm,window]=val
            record=published_exposure[(published_exposure.seed==seed)&(published_exposure.arm==arm)&(published_exposure.window==window)].iloc[0]
            np.testing.assert_allclose(val,record.sum_update_norm,atol=1e-13,rtol=0)
            if arm=='early': assert pd.isna(record.net_displacement_norm)
assert total==406 and skips==150
ratios=[exposures[seed,arm,'1_5']/exposures[seed,arm,'6_10'] for seed in SEEDS for arm in ['early','delayed']]
report={'status':'PASS','source_commit':HEAD,'scientific_commit':SCIENTIFIC,
 'n':8,'block_count':72,'new_blocks':24,'reused_blocks':48,'control_values_exactly_reconciled_with_PR111':48,
 'effects':recomputed,'T_all_positive':True,
 'telemetry':{'prefixes':16,'attempts':total,'successful_updates':256,'AMP_skips':skips,
 'within_path_norm_sum_ratio_1_5_over_6_10_range':[min(ratios),max(ratios)],
 'early_net_displacement':'missing; not inferred from norm sums',
 'early_rho':'derived using native RAdam formula and observed successful clock'},
 'scope':'Public scalar and telemetry reconstruction; no private checkpoint rehash or GPU rerun.'}
(P/'data/window_recomputation.json').write_text(json.dumps(report,indent=2)+'\n')

# Round only the displayed rows. Source FIDs and calculations remain full precision.
(P/'data/window_rows.tex').write_text(''.join(
 f'{seed} & {np.exp(Y[i,0]):.3f} & {np.exp(Y[i,1]):.3f} & {np.exp(Y[i,2]):.3f} & {values["T"][i]:.6f} '+r'\\'+'\n'
 for i,seed in enumerate(SEEDS)))
names={'T':r'$T$: delayed $-$ early','L':r'$L$: delayed $-$ AA','M':r'$M$: early $-$ AA (reused)'}
(P/'data/window_effect_rows.tex').write_text(''.join(
 f'{names[k]} & ${v["mean_log_difference"]:.6f}$ & $[{v["ci95_log"][0]:.6f},{v["ci95_log"][1]:.6f}]$ & {v["two_sided_t_p"]:.6f} '+r'\\'+'\n'
 for k,v in recomputed.items()))
(P/'data/window_exposure_rows.tex').write_text(''.join(
 str(seed)+' & '+' & '.join(f'{exposures[seed,arm,w]:.6f}' for arm in ['early','delayed'] for w in ['1_5','6_10'])+' '+r'\\'+'\n' for seed in SEEDS))

BLUE,TEAL,AMBER,INK='#48647A','#008681','#C88730','#203348'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
 'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#607080',
 'axes.labelcolor':INK,'text.color':INK,'legend.fontsize':8.5,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
def save(fig,name):
    for ext in ['pdf','svg','png']:
        buf=io.BytesIO(); fig.savefig(buf,format=ext,dpi=200,bbox_inches='tight',pad_inches=.07,facecolor='white')
        (P/f'figures/en/{name}.{ext}').write_bytes(buf.getvalue())
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.8,3.25),gridspec_kw={'width_ratios':[1,1.08]},layout='constrained')
ax=axes[0];fids=np.exp(Y)
for row in fids: ax.plot(range(3),row,color='#9BA9B1',lw=.8,alpha=.8,zorder=1)
for j,color in enumerate([BLUE,TEAL,AMBER]): ax.scatter(np.full(8,j),fids[:,j],s=19,color=color,zorder=3)
ax.plot(range(3),np.exp(Y.mean(axis=0)),color=INK,lw=1.7,marker='D',ms=4.5,zorder=4,label='Across-seed geometric mean')
ax.set_xticks(range(3),['AA','Early\n1-5','Delayed\n6-10'])
ax.set_xlim(-.25,2.25);ax.set_ylim(6.7,11.55)
ax.set_ylabel('Endpoint geometric FID')
ax.set_title('(a) Complete paired endpoints, n = 8',loc='left',fontsize=10)
ax.legend(loc='upper right',frameon=False,fontsize=7.7)
ax.grid(axis='y',alpha=.15)
ax=axes[1];ax.axvline(0,lw=1,color='#9BA9B1')
for y,k,color in [(1,'T',TEAL),(0,'L',BLUE)]:
    v=recomputed[k];mean=v['percent'];lo,hi=v['percent_ci95']
    ax.errorbar(mean,y,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=color,capsize=3,lw=1.8,ms=5)
    ax.text(mean,y+.19,f'{mean:+.2f}%',ha='center',va='bottom',fontsize=10,color=color)
ax.set_yticks([1,0],['Delayed / early\n(primary T)','Delayed / AA\n(secondary L)'])
ax.set_ylim(-.55,1.6);ax.set_xlim(-6,21);ax.set_xticks([-5,0,5,10,15,20])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:+.0f}%' if v else '0%'))
ax.set_xlabel('Geometric FID change (nominal 95% CI)')
ax.set_title('(b) Direct and baseline comparisons',loc='left',fontsize=10)
ax.grid(axis='x',alpha=.15)
save(fig,'windows')

fig,axs=plt.subplots(2,2,figsize=(7.8,5.65),layout='constrained')
x=np.arange(1,17);rho=1999-2*x*.999**x/(1-.999**x)
for ax in axs[0]:
    ax.axvspan(.5,5.5,color=TEAL,alpha=.065,lw=0)
    ax.axvspan(5.5,10.5,color=AMBER,alpha=.07,lw=0)
    ax.axvline(5.5,color='#7A8B95',lw=.8,ls='--')
    ax.set_xlim(.5,16.5);ax.set_xticks([1,5,6,10,16]);ax.set_xlabel('Successful update')
for arm,color,label in [('early',TEAL,'Early: updates 1-5'),('delayed',AMBER,'Delayed: updates 6-10')]:
    ax=axs[0,0];ax.step(x,traces[50,arm]['multipliers'],where='mid',color=color,lw=1.6,label=label)
    ax=axs[0,1];norms=np.array([traces[seed,arm]['norms'] for seed in SEEDS])
    for row in norms:ax.plot(x,row,color=color,alpha=.20,lw=.65)
    ax.plot(x,norms.mean(axis=0),color=color,lw=1.7,label=label)
    ax=axs[1,1]
    for seed in SEEDS:
        rs=prefixes[seed,arm];attempt=np.array([int(r['attempted_iteration']) for r in rs]);scaler=np.array([float(r['grad_scale_after']) for r in rs]);skip=np.array([bool(int(r['step_skipped'])) for r in rs])
        ax.plot(attempt,scaler,color=color,lw=.65,alpha=.28)
        ax.scatter(attempt[skip],scaler[skip],marker='x',s=12,color=color,alpha=.35,lw=.65)
axs[0,0].set(ylim=(.89,1.02),ylabel='Applied LR / base LR')
axs[0,0].set_title('(a) Exactly five reduced-LR updates',loc='left',fontsize=10)
axs[0,0].legend(loc='lower right',frameon=False,fontsize=7.5)
axs[0,1].set_yscale('log');axs[0,1].set_ylabel('Actual parameter-update norm')
axs[0,1].set_title('(b) Recorded parameter-update norms',loc='left',fontsize=10)
ax=axs[1,0];ax.plot(x,rho,color=BLUE,marker='o',ms=2.5,lw=1.3)
ax.axhline(5,color='#7A8B95',ls='--',lw=1,label=r'Rectify only when $\rho_n>5$')
ax.axvline(5.5,color='#7A8B95',ls='--',lw=.8)
ax.set(xlim=(.5,16.5),ylim=(0,17),xticks=[1,5,6,10,16],xlabel='Successful update',ylabel=r'Rectification statistic $\rho_n$')
ax.set_title('(c) Native RAdam branch boundary',loc='left',fontsize=10)
ax.legend(loc='upper left',frameon=False,fontsize=7.5)
ax=axs[1,1];ax.set_yscale('log',base=2);ax.set(xlabel='Attempt (skips retained)',ylabel='AMP scaler after attempt')
ax.set_title('(d) Crosses mark skipped attempts',loc='left',fontsize=10)
ax.set_yticks([128,1024,8192,32768]);ax.set_yticklabels(['128','1,024','8,192','32,768'])
for ax in axs.flat:ax.grid(axis='y',alpha=.15)
save(fig,'window_diagnostics')
print(json.dumps(report,indent=2))
