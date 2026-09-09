"""Standalone scientific figures from saved seed-level metrics; no sampling/training."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(results):
    results=Path(results);s=json.loads((results/'statistics.json').read_text())
    if s['status'] not in {'COMPLETE','COMPLETE_WITH_SCIENTIFIC_FAILURES'}:raise RuntimeError('cannot plot unfinished results as final')
    figures=results/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    rows=s['per_seed'];fig,ax=plt.subplots(figsize=(8,6))
    for y,r in enumerate(rows):
        values=[math.exp(r['DA_mean_log_FID']),math.exp(r['DD_mean_log_FID'])]
        ax.plot(values,[y,y],color='#999',lw=1)
        ax.scatter([values[0]],[y],s=45,facecolors='none',edgecolors='#27648e',linewidths=1.5,label='DA: restore A' if y==0 else None,zorder=3)
        ax.scatter([values[1]],[y],s=23,marker='D',color='#bb663a',label='DD: hold D' if y==0 else None,zorder=4)
    ax.set_yticks(range(len(rows)),[str(r['seed']) for r in rows]);ax.invert_yaxis()
    ax.set_ylabel('Training seed');ax.set_xlabel('Endpoint geometric FID (B0/B1/B2); lower is better')
    ax.set_title(f'Original D@512 shared within seed; complete pairs n={len(rows)}')
    ax.legend(loc='best',fontsize=9)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(figures/f'DA_DD_paired.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6));margin=math.log(1.03)
    ax.axvspan(-margin,margin,color='#dcebdc',label='Prespecified multiplicative 3% range')
    ax.axvline(0,color='#888',lw=1)
    ax.scatter([r['DA_minus_DD'] for r in rows],range(len(rows)),color='#27648e')
    labels=[str(r['seed']) for r in rows]
    positions=list(range(len(rows)))
    primary=s['primary']
    if primary.get('mean_log_difference') is not None:
        y=len(rows)+1; mean=primary['mean_log_difference']
        ax.axhline(len(rows)-.35,color='#bbb',lw=.7,ls=':')
        if primary.get('ci95'):
            ax.hlines(y,*primary['ci95'],color='#222',lw=1.2,label='Aggregate 95% t interval')
        if primary.get('ci90'):
            ax.hlines(y,*primary['ci90'],color='#222',lw=4,label='Aggregate 90% t interval')
        ax.scatter([mean],[y],marker='D',color='#222',zorder=4)
        labels.append('Mean');positions.append(y)
    ax.set_yticks(positions,labels);ax.invert_yaxis()
    ax.set_xlabel('Mean block log(FID_DA / FID_DD); negative favors restoring A')
    ax.set_ylabel('Training seed');ax.set_title('Equivalence bounds apply to the aggregate mean')
    ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(figures/f'paired_log_differences.{ext}',dpi=180)
    plt.close(fig)
    common=s['auxiliary']['per_seed'];fig,ax=plt.subplots(figsize=(8,5))
    for r in common:ax.plot(range(3),[math.exp(r[a]) for a in ('AA','DA','DD')],'o-',color='#27648e',alpha=.4)
    ax.set_xticks(range(3),['AA (auxiliary)','DA','DD']);ax.set_ylabel('Endpoint geometric FID')
    ax.set_title(f'Only common valid AA/DA/DD seeds, n={len(common)}');fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(figures/f'AA_DA_DD_common.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);render(p.parse_args().results)
