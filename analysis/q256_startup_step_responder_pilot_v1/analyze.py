"""Frozen seed-paired analysis, reporting all outcomes and missingness."""
from __future__ import annotations
import argparse,csv,itertools,json,math
from pathlib import Path
import numpy as np
from scipy import stats
from . import protocol as p


def summarize(values):
    x=np.asarray(values,dtype=float);n=len(x)
    if n<2:return dict(n=n,mean=float(x.mean()) if n else None,SD=None,nominal_95_t_CI=None,paired_dz=None,two_sided_t_p=None,exact_sign_flip_p=None,TOST=None)
    mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(n);tcrit=float(stats.t.ppf(.975,n-1));delta=math.log(1.03)
    if not se:
        pv=1. if mean==0 else 0.;dz=None
        tost=0. if -delta<mean<delta else 1.
    else:
        pv=float(2*stats.t.sf(abs(mean/se),n-1));dz=mean/sd
        tost=max(float(stats.t.sf((mean+delta)/se,n-1)),float(stats.t.cdf((mean-delta)/se,n-1)))
    flips=[abs(float((x*np.array(sign)).mean())) for sign in itertools.product((-1,1),repeat=n)]
    exact=sum(v>=abs(mean)-1e-14 for v in flips)/len(flips)
    ci=[mean-tcrit*se,mean+tcrit*se]
    return dict(n=n,mean=mean,SD=sd,nominal_95_t_CI=ci,paired_dz=dz,
        geometric_FID_contrast_ratio=math.exp(mean),geometric_ratio_nominal_95_CI=[math.exp(v) for v in ci],
        positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()),
        two_sided_t_p=pv,exact_sign_flip_p=exact,
        TOST=dict(delta=delta,p=tost,equivalent=tost<.05,nominal_90_t_CI=[mean-float(stats.t.ppf(.95,n-1))*se,mean+float(stats.t.ppf(.95,n-1))*se]))


def analyze(root,output):
    matrix=json.loads((root/'training_matrix_frozen.json').read_text())
    if len(matrix['outcomes'])!=16 or any(r['status'] not in p.SPEC['terminal_statuses'] for r in matrix['outcomes']):raise RuntimeError('training matrix not terminal')
    outcomes={(r['seed'],r['arm']):r for r in matrix['outcomes']}
    controls=json.loads((root/'old_control_bindings.json').read_text())
    new=json.loads((root/'evaluation_slots.json').read_text())
    if len(new)!=48 or {(r['seed'],r['arm'],r['block']) for r in new}!={(s,a,b) for s in p.SEEDS for a in p.ARMS for b in ('B0','B1','B2')}:
        raise RuntimeError('48 fixed evaluation outcomes required')
    allrows=controls+new;lookup={(r['seed'],r['arm'],r['block']):r for r in allrows}
    result=[]
    for seed in p.SEEDS:
        row=dict(seed=seed)
        for arm in ('AA','DA')+p.ARMS:
            rr=[lookup[(seed,arm,b)] for b in ('B0','B1','B2')]
            valid=all(r['status']=='PASS' and r.get('FID') is not None and math.isfinite(r['FID']) and r['FID']>0 for r in rr)
            row[arm+'_Y']=float(np.log([r['FID'] for r in rr]).mean()) if valid else None
            row[arm+'_geometric_FID']=math.exp(row[arm+'_Y']) if valid else None
            row[arm+'_KID_mean']=float(np.mean([r['KID'] for r in rr])) if valid and all(r.get('KID') is not None and math.isfinite(r['KID']) for r in rr) else None
            if arm in p.ARMS:row[arm+'_training_status']=outcomes[(seed,arm)]['status']
            for b,r in zip(('B0','B1','B2'),rr):row[arm+'_'+b+'_FID']=r.get('FID')
        finite=all(row[a+'_Y'] is not None for a in ('AA','DA')+p.ARMS)
        row['finite_paired']=finite
        row['C']=row[p.ARMS[0]+'_Y']-row['DA_Y'] if finite else None
        row['M']=row[p.ARMS[1]+'_Y']-row['AA_Y'] if finite else None
        row['S']=(row['C']-row['M'])/2 if finite else None
        result.append(row)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per-seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary={key:summarize([r[key] for r in result if r['finite_paired']]) for key in ('S','C','M')}
    raw=[summary[k]['two_sided_t_p'] for k in ('C','M')]
    if all(x is not None for x in raw):
        order=np.argsort(raw);adjusted=[0.,0.];last=0.
        for rank,index in enumerate(order):last=max(last,min(1.,(2-rank)*raw[index]));adjusted[index]=last
        for key,adj in zip(('C','M'),adjusted):summary[key]['Holm_adjusted_p']=adj
    report=dict(protocol_id=p.PROTOCOL,experiment_identity=p.SPEC['experiment_identity'],n_planned=8,
        n_trained_trajectories=sum(x['status']=='PASS' for x in matrix['outcomes']),
        n_trained_paired=sum(all(outcomes[(s,a)]['status']=='PASS' for a in p.ARMS) for s in p.SEEDS),
        n_finite_paired=sum(r['finite_paired'] for r in result),primary='S',statistics=summary,
        failures=[x for x in matrix['outcomes'] if x['status']!='PASS'],evaluation_failures=[x for x in new if x['status']!='PASS'],
        KID_role='auxiliary consistency on the same generated features, not independent replication',
        interpretation_scope='responder-enriched seeds50-57 only; no population or exact-trajectory claim',
        missingness_policy='all planned slots retained; finite complete pairs only; incomplete matrix never presented as complete')
    p.write(output/'statistics.json',report,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(left,right,title) in zip(axes[:2],[('DA',p.ARMS[0],'C: D startup up'),('AA',p.ARMS[1],'M: A startup down')]):
        for r in result:
            if r[left+'_geometric_FID'] is not None and r[right+'_geometric_FID'] is not None:
                ax.plot([0,1],[r[left+'_geometric_FID'],r[right+'_geometric_FID']],'-o',alpha=.7,label=str(r['seed']))
        ax.set_xticks([0,1],[left,right],rotation=15);ax.set_ylabel('Geometric FID across 3 blocks');ax.set_title(title)
    valid=[r for r in result if r['finite_paired']];axes[2].axhline(0,color='gray',lw=1)
    axes[2].scatter([r['seed'] for r in valid],[r['S'] for r in valid]);axes[2].set(xlabel='Training seed',ylabel='S (log-FID contrast)',title='Primary S')
    fig.savefig(output/'paired.png',dpi=180);fig.savefig(output/'paired.pdf');plt.close(fig)
    lines=['# 实验结果', '',p.SPEC['experiment_identity'],'',f"计划 8 seeds / 16 轨迹；训练完成 {report['n_trained_trajectories']} 轨迹；有限完整配对 {report['n_finite_paired']} seeds。",'', '| Contrast | mean | nominal 95% t CI | paired dz | ratio |','|---|---:|---|---:|---:|']
    for k,v in summary.items():lines.append(f"| {k} | {v['mean']} | {v['nominal_95_t_CI']} | {v['paired_dz']} | {v.get('geometric_FID_contrast_ratio')} |")
    lines+=['','唯一 primary 为 S；C、M 显著性判断使用双侧 paired t-test 和 Holm 校正。普通区间为 nominal，不能称为 simultaneous Holm CI。',
            'p>0.05 不等于无效；只有 TOST 支持才可声明 ±3% 实用等效。KID 与 FID 共用生成特征，仅为辅助一致性结果。',
            '解释仅限 responder-enriched seeds50–57，不推广到一般 seed，不宣称解释全部 DA 收益、唯一 optimizer 中介、跨数据集普遍性或五步轨迹精确复现。',
            '',f"所有非 PASS 训练：{json.dumps(report['failures'],ensure_ascii=False)}",f"所有非 PASS 评估：{json.dumps(report['evaluation_failures'],ensure_ascii=False)}"]
    (output/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();print(json.dumps(analyze(args.root,args.output),indent=2))
