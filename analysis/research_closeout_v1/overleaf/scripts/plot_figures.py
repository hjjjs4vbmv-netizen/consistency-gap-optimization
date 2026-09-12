#!/usr/bin/env python3
"""Rebuild all manuscript figures from included observations; no training needed.

Run from anywhere: python scripts/plot_figures.py
This English-only version does not require a CJK font.
"""
from pathlib import Path
import argparse, json, subprocess, io, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats

P=Path(__file__).resolve().parents[1]
cjk='DejaVu Sans'  # English-only revision: no CJK font dependency.
COL={'A':'#48647A','B':'#7F4DAD','C':'#C88730','D':'#008681'}
raw=P/'data/raw'
comp=pd.read_csv(raw/'component_contrasts.csv')
hist=pd.read_csv(raw/'history_pairs.csv')
sf=pd.read_csv(raw/'source_future.csv')
pilot_raw=pd.read_csv(raw/'startup_per_seed.csv',float_precision='round_trip')
pilot_archive=json.loads((raw/'startup_statistics.json').read_text())
pilot=pd.DataFrame({'seed':pilot_raw.seed})
pilot_y={}
for short,arm in [('AA','AA'),('DA','DA'),('A_down5','A_startup_down5'),('D_up5','D_startup_up5')]:
    blocks=pilot_raw[[f'{arm}_{block}_FID' for block in ['B0','B1','B2']]].to_numpy()
    if not np.isfinite(blocks).all() or not (blocks>0).all():
        raise ValueError('Startup block FIDs must be finite and positive')
    pilot_y[short]=np.log(blocks).mean(axis=1)
    np.testing.assert_allclose(pilot_y[short],pilot_raw[f'{arm}_Y'],rtol=0,atol=2e-14)
    pilot[short]=np.exp(pilot_y[short])
    np.testing.assert_allclose(pilot[short],pilot_raw[f'{arm}_geometric_FID'],rtol=0,atol=5e-13)
assert list(pilot.seed)==list(range(50,58))
new_slots=json.loads((raw/'startup_evaluation_slots.json').read_text())
old_slots=json.loads((raw/'startup_old_control_bindings.json').read_text())
training=json.loads((raw/'startup_training_matrix.json').read_text())['outcomes']
assert len(new_slots)==48 and len(old_slots)==48 and len(training)==16
training_by_id={(v['seed'],v['arm']):v for v in training}
assert len(training_by_id)==16
assert all(v['status']=='PASS' and v['start_attempt']==0 and v['attempted_iteration']==8000 for v in training)
slots_by_id={(v['seed'],v['arm'],v['block']):v for v in new_slots+old_slots}
assert len(slots_by_id)==96
for _,row in pilot_raw.iterrows():
    for arm in ['AA','DA','A_startup_down5','D_startup_up5']:
        for block in ['B0','B1','B2']:
            slot=slots_by_id[(int(row.seed),arm,block)]
            assert slot['status']=='PASS' and slot['readout']=='E_512'
            assert slot['evaluator_commit']=='d6aba02fb88e9db0993623895eb2228ed717d810'
            assert slot['FID']==row[f'{arm}_{block}_FID']
            if arm not in ['AA','DA']:
                assert slot['checkpoint_sha256']==training_by_id[(int(row.seed),arm)]['checkpoint_sha256']
pilot_contrasts={'M':pilot_y['A_down5']-pilot_y['AA'],'C':pilot_y['D_up5']-pilot_y['DA']}
pilot_contrasts['S']=(pilot_contrasts['C']-pilot_contrasts['M'])/2
im=pd.read_csv(raw/'imagenet_trajectories.csv')
long=pd.read_csv(raw/'longitudinal_evaluations.csv')
long=long[long.seed.between(8,13)]
hold=pd.read_csv(raw/'restore_hold_per_seed.csv')

def ci(v):
    v=np.asarray(v,float)
    m=v.mean();h=stats.t.ppf(.975,len(v)-1)*stats.sem(v)
    return m,m-h,m+h

reported={'status':'completed_responder_enriched_pilot','n':len(pilot),'source_commit':'e429ab29476b23c4417cc2b1df7268a2b192f175',
          'interim_look_disclosed':True,'sequential_adjustment':False}
pilot_check={}
for key,v in pilot_contrasts.items():
    m,l,h=ci(v)
    archived=pilot_archive['statistics'][key]
    pval=float(stats.ttest_1samp(v,0).pvalue)
    means=np.asarray([np.mean(v*np.asarray(sign)) for sign in itertools.product([-1,1],repeat=len(v))])
    pflip=float(np.mean(np.abs(means)>=abs(m)-1e-14))
    np.testing.assert_allclose([m,l,h,pval,pflip],
        [archived['mean'],*archived['nominal_95_t_CI'],archived['two_sided_t_p'],archived['exact_sign_flip_p']],rtol=0,atol=2e-12)
    pilot_check[key]={'mean':m,'ci95':[l,h],'two_sided_t_p':pval,'exact_sign_flip_p':pflip}
    if key in ['M','C']:
        reported[key+'_percent']=float(100*np.expm1(m))
        reported[key+'_percent_ci95']=(100*np.expm1([l,h])).tolist()
        reported[key+'_Holm_p']=archived['Holm_adjusted_p']
for i,key in enumerate(sorted(['M','C'],key=lambda k:pilot_check[k]['two_sided_t_p'])):
    adjusted=max((2-j)*pilot_check[k]['two_sided_t_p'] for j,k in enumerate(sorted(['M','C'],key=lambda k:pilot_check[k]['two_sided_t_p'])[:i+1]))
    np.testing.assert_allclose(min(adjusted,1),pilot_archive['statistics'][key]['Holm_adjusted_p'],rtol=0,atol=2e-12)
for name,v in {'DA_minus_AA':pilot_y['DA']-pilot_y['AA'],'Adown_minus_DA':pilot_y['A_down5']-pilot_y['DA']}.items():
    m,l,h=ci(v)
    pilot_check[name]={'mean':m,'ci95':[l,h],'ratio':float(np.exp(m)),'ratio_ci95':np.exp([l,h]).tolist(),'role':'descriptive same-cohort comparison'}
(P/'data/startup_summary.json').write_text(json.dumps(reported,indent=2)+'\n')

def style(zh):
    plt.rcParams.update({'font.family':[cjk,'DejaVu Sans'] if zh else 'DejaVu Sans',
      'font.size':11,'axes.labelsize':11,'axes.titlesize':11.5,'xtick.labelsize':10,
      'ytick.labelsize':10,'legend.fontsize':9.5,'axes.spines.top':False,'axes.spines.right':False,
      'axes.edgecolor':'#607080','axes.labelcolor':'#203348','text.color':'#203348',
      # Fandol is a CFF OpenType font: embed vector glyphs, not a TrueType wrapper.
      'axes.titleweight':'normal','pdf.fonttype':3 if zh else 42,'ps.fonttype':3 if zh else 42,'axes.unicode_minus':False,
      'savefig.facecolor':'white','figure.facecolor':'white'})

def save(fig,name,lang):
    out=P/'figures'/lang
    out.mkdir(parents=True,exist_ok=True)
    pdf_buffer=io.BytesIO()
    fig.savefig(pdf_buffer,format='pdf',bbox_inches='tight',pad_inches=.07)
    pdf_bytes=pdf_buffer.getvalue()
    if not pdf_bytes.startswith(b'%PDF') or len(pdf_bytes)<500:
        raise RuntimeError('Invalid PDF figure: '+name)
    temp=out/(name+'.pdf.tmp')
    temp.write_bytes(pdf_bytes)
    temp.replace(out/(name+'.pdf'))
    fig.savefig(out/(name+'.png'),bbox_inches='tight',pad_inches=.07,dpi=180)
    plt.close(fig)

for lang in ['en']:
    zh=lang=='zh';style(zh)
    tr=lambda en,cn:cn if zh else en

    fig,axs=plt.subplots(1,2,figsize=(7.6,3.7),gridspec_kw={'width_ratios':[1,1.08]},layout='constrained')
    ax=axs[0];v=hist.log_fid_contrast_ba_minus_aa.to_numpy();order=np.argsort(v)
    ax.axvline(0,color='#AAB5BF',lw=1)
    for j,i in enumerate(order):
        ax.plot([0,v[i]],[j,j],color=COL['D'] if v[i]<0 else COL['C'],lw=1.2)
        ax.scatter(v[i],j,s=15,color=COL['D'] if v[i]<0 else COL['C'],zorder=3)
    ax.set_yticks(np.arange(len(v))[::2],[str(int(hist.seed.iloc[i])) for i in order[::2]])
    ax.set_xlabel(tr('Endpoint log-FID: BA - AA','终点 log-FID：BA - AA'))
    ax.set_ylabel(tr('Training seed (sorted)','训练种子（按效应排序）'))
    ax.set_title(tr('(a) Joint history, n = 26','(a) 联合历史，n = 26'),loc='left')
    ax=axs[1];good=(sf.Q>0)&(sf.H_A<0)
    ax.axhline(0,color='#AAB5BF',lw=1);ax.axvline(0,color='#AAB5BF',lw=1)
    ax.scatter(sf.loc[~good,'Q'],sf.loc[~good,'H_A'],s=25,color=COL['A'],alpha=.8)
    ax.scatter(sf.loc[good,'Q'],sf.loc[good,'H_A'],s=30,color=COL['D'],zorder=3)
    ax.set_xlabel(tr('Source log-FID ratio Q (512 kimg)','源质量 log-FID 比 Q（512 kimg）'))
    ax.set_ylabel(tr('Future log-FID ratio H','后续 log-FID 比 H'))
    ax.set_title(tr('(b) Source vs. future, same 26 seeds','(b) 源质量与后续质量，同一 26 种子'),loc='left')
    ax.text(.97,.06,tr('11 worse-to-better pairs','11 对由较差变为较好'),transform=ax.transAxes,ha='right',color=COL['D'],fontsize=10)
    save(fig,'history',lang)

    fig=plt.figure(figsize=(7.6,4.1),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[.55,3])
    ax=fig.add_subplot(gs[0,:]);ax.axis('off')
    ax.text(.01,.60,tr('Early history: A / C / D / B','前期历史：A / C / D / B'),va='center',fontsize=12)
    ax.annotate('',xy=(.60,.60),xytext=(.43,.60),arrowprops={'arrowstyle':'->','color':'#48647A','lw':1.6})
    ax.text(.62,.60,tr('Common A continuation','共同 A 续训'),va='center',fontsize=12)
    ax.text(.01,.04,'0-512 kimg',fontsize=10,color='#667788')
    ax.text(.62,.04,tr('512-1024 kimg; rebuilt EMA','512-1024 kimg；重建 EMA'),fontsize=10,color='#667788')
    ax=fig.add_subplot(gs[1,0]);names=['AA','CA','DA','BA'];values=np.exp(comp[names].to_numpy())
    for row in values:ax.plot(range(4),row,color='#BFCAD3',lw=.8,alpha=.65,zorder=1)
    for i,name in enumerate(names):ax.scatter(np.full(len(comp),i),values[:,i],s=17,color=COL[name[0]],zorder=2)
    means=np.exp(comp[names].mean().to_numpy());ax.plot(range(4),means,color='#203348',lw=1.8,marker='D',ms=5,zorder=3,label=tr('Geometric mean','几何均值'))
    ax.set_xticks(range(4),names);ax.set_ylabel(tr('Endpoint FID (lower is better)','终点 FID（越低越好）'));ax.legend(frameon=False,loc='upper right')
    ax.set_title(tr('(a) All 14 complete pairs','(a) 全部 14 个完整配对'),loc='left')
    ax=fig.add_subplot(gs[1,1]);ax.axvline(0,color='#9EAAB7',lw=1)
    for j,(name,color) in enumerate([('H_T',COL['C']),('H_W',COL['D']),('I',COL['B'])]):
        m,l,h=ci(comp[name]);ax.errorbar(m,2-j,xerr=[[m-l],[h-m]],fmt='o',color=color,capsize=3,lw=1.7)
    ax.set_yticks([2,1,0],[r'$H_T$',r'$H_W$',r'$I$']);ax.set_ylim(-.6,2.6)
    ax.set_xlabel(tr('Mean log-FID contrast; 95% CI','平均 log-FID 对比；95% 区间'))
    ax.set_title(tr('(b) Prespecified contrasts','(b) 预设成分对比'),loc='left')
    ax.text(.02,.03,tr('Denominator: Holm p = 0.00178','分母效应：Holm p = 0.00178'),transform=ax.transAxes,fontsize=9.6,color=COL['D'])
    save(fig,'components',lang)

    fig,axs=plt.subplots(1,3,figsize=(7.6,3.0),gridspec_kw={'width_ratios':[1,1,1.18]},layout='constrained')
    for ax,(base,new,color,label) in zip(axs[:2],[('AA','A_down5',COL['D'],tr('(a) Reduce A startup LR','(a) A 启动降 LR')),('DA','D_up5',COL['C'],tr('(b) Raise D startup LR','(b) D 启动升 LR'))]):
        for _,row in pilot.iterrows():
            ax.plot([0,1],[row[base],row[new]],color=color,alpha=.50,lw=1)
            ax.scatter([0,1],[row[base],row[new]],color=color,s=17)
        ax.set_xticks([0,1],[base,r'$A_{\downarrow5}$' if new=='A_down5' else r'$D_{\uparrow5}$'])
        ax.set_ylim(6.7,11.6);ax.set_title(label,loc='left',fontsize=10.5)
        ax.set_ylabel(tr('Endpoint FID','终点 FID') if ax is axs[0] else '')
        count=int((pilot[new]<pilot[base]).sum()) if base=='AA' else int((pilot[new]>pilot[base]).sum())
        ax.text(.03,.04,tr(f'{count}/{len(pilot)} improve',f'{count}/{len(pilot)} 改善') if base=='AA' else tr(f'{count}/{len(pilot)} worsen',f'{count}/{len(pilot)} 变差'),transform=ax.transAxes,color=color,fontsize=10)
    ax=axs[2];ax.axvline(0,color='#AAB5BF',lw=1)
    for y,key,color in [(1,'M',COL['D']),(0,'C',COL['C'])]:
        m=reported[key+'_percent'];l,h=reported[key+'_percent_ci95']
        ax.errorbar(m,y,xerr=[[m-l],[h-m]],fmt='o',color=color,lw=1.7,capsize=3)
    ax.set_yticks([1,0],['M','C']);ax.set_ylim(-.6,1.6);ax.set_xlim(-21,16)
    ax.set_xticks([-20,-10,0,10]);ax.set_xlabel(tr('Paired FID change (%)','配对 FID 变化（%）'))
    ax.set_title(tr('(c) Complete, n = 8','(c) 完整结果，n = 8'),loc='left',fontsize=10.5)
    ax.text(.03,.04,tr('Nominal 95% intervals','名义 95% 区间'),transform=ax.transAxes,fontsize=9,color='#667788')
    save(fig,'startup',lang)

    fig,axs=plt.subplots(2,3,figsize=(7.6,4.45),sharex=True,sharey='row',layout='constrained')
    for col,seed in enumerate([101,102,103]):
        for row,nfe in enumerate([1,2]):
            ax=axs[row,col]
            for method,color,marker in [('IA',COL['A'],'o'),('IB',COL['B'],'s')]:
                d=im[(im.seed==seed)&(im.nfe==nfe)&(im.method==method)].sort_values('kimg')
                ax.plot(d.kimg/1000,d.fid50k,color=color,marker=marker,ms=3,lw=1.2,label=method)
            ax.set_yscale('log');ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis='y',which='major',alpha=.14)
            ax.set_title(tr(f'Seed {seed} | NFE{nfe}',f'种子 {seed} | NFE{nfe}'),loc='left',fontsize=10.5)
            ax.set_xticks([1.28,6.4,12.8],['1.28','6.4','12.8'])
            if col==0:ax.set_ylabel('FID')
            if row==1:ax.set_xlabel(tr('Training images (million)','训练图像（百万）'))
    axs[0,0].legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
    save(fig,'imagenet',lang)

    fig,axs=plt.subplots(1,2,figsize=(7.6,3.2),layout='constrained')
    for ax,nfe in zip(axs,[1,2]):
        for arm,color in COL.items():
            d=long[(long.nfe==nfe)&(long.arm==arm)].groupby('budget_kimg').fid50k_full.mean()
            ax.plot(d.index,d.values,color=color,marker='o',ms=3,lw=1.4,label=arm)
        ax.set_yscale('log');ax.set_xticks([384,640,896,1024]);ax.set_xlabel(tr('Training images (kimg)','训练图像（kimg）'))
        ax.set_ylabel(tr('Arithmetic mean FID','FID 算术均值'));ax.set_title(f'NFE{nfe}',loc='left');ax.grid(axis='y',alpha=.15)
        ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0].legend(ncol=4,frameon=False,loc='upper right')
    save(fig,'longitudinal',lang)

    fig,axs=plt.subplots(1,2,figsize=(7.6,3.25),layout='constrained')
    ax=axs[0]
    for _,row in hold.iterrows():
        color=COL['C'] if int(row.seed)==58 else '#94ACBA'
        ax.plot([0,1],np.exp([row.DA_mean_log_FID,row.DD_mean_log_FID]),color=color,lw=1,marker='o',ms=3)
    ax.set_xticks([0,1],['DA','DD']);ax.set_ylabel(tr('Endpoint FID','终点 FID'));ax.set_title(tr('(a) All 16 D-history pairs','(a) 全部 16 个 D 历史配对'),loc='left')
    ax.text(.03,.95,tr('Seed 58 retained','保留种子 58'),transform=ax.transAxes,va='top',color=COL['C'],fontsize=10)
    ax=axs[1];ax.axhline(0,color='#AAB5BF',lw=1)
    ax.axhspan(-np.log(1.03),np.log(1.03),color='#E6F0EF',label=tr('3% equivalence margin','3% 等效界限'))
    ax.scatter(hold.seed,hold.DA_minus_DD,c=[COL['C'] if s==58 else COL['A'] for s in hold.seed],s=25)
    ax.set_xlabel(tr('Training seed','训练种子'));ax.set_ylabel(tr('log-FID: DA - DD','log-FID：DA - DD'))
    ax.set_title(tr('(b) Individual effects','(b) 逐种子效应'),loc='left');ax.legend(frameon=False,loc='upper left',fontsize=9)
    save(fig,'restore_hold',lang)

def table_rows_component():
    return ['% Values are exponentiated seed-mean log-FID from the archived component table.']+[f"{int(r.seed)} & {np.exp(r.AA):.3f} & {np.exp(r.CA):.3f} & {np.exp(r.DA):.3f} & {np.exp(r.BA):.3f} & {r.H_W:+.5f} \\\\" for _,r in comp.iterrows()]
(P/'data/component_rows.tex').write_text('\n'.join(table_rows_component())+'\n')
rows=[f"{int(r.seed)} & {r.AA:.3f} & {r.A_down5:.3f} & {r.DA:.3f} & {r.D_up5:.3f} \\\\" for _,r in pilot.iterrows()]
(P/'data/startup_rows.tex').write_text('\n'.join(rows)+'\n')
hold_records=list(hold.sort_values('seed').itertuples())
half=(len(hold_records)+1)//2
def hold_cells(r):
    return f"{int(r.seed)} & {np.exp(r.DA_mean_log_FID):.3f} & {np.exp(r.DD_mean_log_FID):.3f} & {r.geometric_ratio:.5f}"
rows=[]
for i in range(half):
    right=hold_cells(hold_records[i+half]) if i+half<len(hold_records) else ' & & & '
    rows.append(hold_cells(hold_records[i])+' & '+right+' '+chr(92)*2)
(P/'data/restore_rows.tex').write_text('\n'.join(rows)+'\n')
for metric,label in [('fid50k','fid'),('kid50k','kid')]:
    rows=[]
    for (seed,kimg),d in im.groupby(['seed','kimg']):
        cells=[float(d[(d.method==method)&(d.nfe==nfe)][metric].iloc[0]) for nfe in [1,2] for method in ['IA','IB']]
        fmt='.3f' if label=='fid' else '.5f'
        rows.append(f'{seed} & {kimg} & '+' & '.join(format(v,fmt) for v in cells)+r' \\')
    (P/f'data/imagenet_{label}_rows.tex').write_text('\n'.join(rows)+'\n')
check={}
for name in ['H_T','H_W','I']:
    m,l,h=ci(comp[name]);check[name]={'mean':m,'ci95':[l,h]}
check['startup_full_precision_recomputation']=pilot_check
(P/'data/plot_recomputation.json').write_text(json.dumps(check,indent=2)+'\n')

completion=pd.read_csv(raw/'component_outcomes.csv')
completion_rows=[]
for seed,group in completion.groupby('seed',sort=True):
    present={row.path:str(row.evaluation_status)=='PASS' for row in group.itertuples()}
    if set(present)!=set(['AA','BA','CA','DA']):raise ValueError('Incomplete completion record')
    cells=['Yes' if present[arm] else '--' for arm in ['AA','BA','CA','DA']]
    complete=all(present.values())
    assert complete==(seed in set(comp.seed))
    completion_rows.append(str(seed)+' & '+' & '.join(cells)+' & '+('Yes' if complete else '--')+r' \\')
(P/'data/completion_rows.tex').write_text('\n'.join(completion_rows)+'\n')

print('Built six English result figures, six table-row files, and recomputation summary.')
