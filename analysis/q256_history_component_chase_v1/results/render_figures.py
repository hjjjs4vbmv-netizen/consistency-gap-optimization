"""Render the two prespecified figures from saved statistics only."""
import argparse,json,math,os,statistics
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ect-history-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--preview-dir',type=Path);args=p.parse_args()
    data=json.loads((ROOT/'statistics.json').read_text());assert data['status']=='COMPLETE'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'q256-history-component-v1','axes.spines.top':False,'axes.spines.right':False})
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    colors=['#354F6A','#AA6C39','#26796D','#735B96'];paths=['AA','CA','DA','BA']
    fig,ax=plt.subplots(figsize=(7.6,5),layout='constrained')
    for row in data['per_seed']:
        ys=[math.exp(row[p]) for p in paths];ax.plot(range(4),ys,color='#A0A8AE',linewidth=.8,alpha=.65,zorder=1)
        for i,y in enumerate(ys):ax.scatter(i,y,s=21,color=colors[i],alpha=.82,zorder=2)
    gm=[math.exp(statistics.mean(row[p] for row in data['per_seed'])) for p in paths]
    ax.plot(range(4),gm,color='#17242C',linestyle='--',marker='D',markersize=7,linewidth=1.4,zorder=4,label='Geometric mean across training seeds')
    ax.set_xticks(range(4),['AA\nbaseline history','CA\ntarget-only history','DA\ndenominator-only history','BA\nboth histories'])
    ax.set_ylabel('E_512 FID at 1024 kimg (log axis)');ax.set_yscale('log');ax.set_ylim(5.8,16)
    ax.set_yticks([6,8,10,12,15],labels=['6','8','10','12','15']);ax.grid(axis='y',alpha=.2)
    ax.set_title('Paired endpoints after common A continuation',loc='left',fontweight='bold');ax.legend(loc='upper left',frameon=False,fontsize=9)
    fig.supxlabel('Same 14 complete training seeds; each point combines three fixed generation blocks.',fontsize=9)
    fig.savefig(out/'paired_endpoints.svg',metadata={'Date':None})
    if args.preview_dir:args.preview_dir.mkdir(parents=True,exist_ok=True);fig.savefig(args.preview_dir/'paired_endpoints.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained');names=['H_T','H_W','I'];labels=['Target history: CA - AA','Denominator history: DA - AA','Interaction: BA - CA - DA + AA']
    for i,name in enumerate(names):
        v=data['primary'][name];m=v['mean'];lo,hi=v['ci95_nominal'];ax.errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[i+1],capsize=4,markersize=7,linewidth=2)
        ax.text(.165,2-i,f"Holm p = {v['p_holm']:.5f}",va='center',fontsize=9)
    ax.axvline(0,color='#44515B',linestyle='--',linewidth=1);ax.set_yticks([2,1,0],labels);ax.set_xlim(-.16,.28);ax.set_ylim(-.55,2.55)
    ax.set_xticks([-.15,-.10,-.05,0,.05,.10,.15]);ax.grid(axis='x',alpha=.18)
    ax.set_xlabel('Mean log-FID contrast; individual nominal 95% t intervals');ax.set_title('Prespecified component contrasts, same n = 14',loc='left',fontweight='bold')
    fig.supxlabel('Intervals are not Holm simultaneous CIs. Holm correction applies to the three p-values.',fontsize=9)
    fig.savefig(out/'component_intervals.svg',metadata={'Date':None})
    if args.preview_dir:fig.savefig(args.preview_dir/'component_intervals.png',dpi=180)
    plt.close(fig)
    for svg in out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
if __name__=='__main__':main()
