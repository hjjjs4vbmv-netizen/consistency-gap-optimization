"""Decode only after integrity sealing; all source-to-future analysis is descriptive."""
import argparse
import math
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from analysis.q256_terminal_history_512_source_backfill_v1.common import (
    EXPERIMENT_ID, PROTOCOL_SHA, check_hash, frozen_inputs, join_future, load,
    quadrants, read_csv, require, sha256, source_pairs, write_csv, write_json)

def summarize(q):
    import numpy as np
    from scipy import stats
    n=len(q)
    if not n: return dict(n=0,mean=None,median=None,sd=None,ci95=None,Q_gt_zero=0,Q_lt_zero=0,Q_eq_zero=0)
    sd=float(np.std(q,ddof=1)) if n>1 else None
    mean=float(np.mean(q))
    half=float(stats.t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,median=float(np.median(q)),sd=sd,
                ci95=[mean-half,mean+half] if half is not None else None,
                Q_gt_zero=sum(x>0 for x in q),Q_lt_zero=sum(x<0 for x in q),Q_eq_zero=sum(x==0 for x in q))

def correlation(rows):
    from scipy import stats
    q=[r['Q'] for r in rows]; h=[r['H_A'] for r in rows]
    if len(rows)<3 or len(set(q))<2 or len(set(h))<2:
        return dict(pearson=None,spearman=None,role='descriptive',reason='too few observations or constant variable')
    return dict(pearson=float(stats.pearsonr(q,h).statistic),
                spearman=float(stats.spearmanr(q,h).statistic),role='descriptive; no confirmatory p-value')

def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    colors={'reversal':'#d55e00','bad_to_bad':'#8a7ca8','good_to_good':'#0072b2','reverse_loss':'#009e73','on_axis':'#777777'}
    labels={'reversal':'Reversal','bad_to_bad':'Bad → bad','good_to_good':'Good → good','reverse_loss':'Reverse-loss','on_axis':'On axis'}
    for key in colors:
        subset=[r for r in rows if quadrants([r])[key]]
        if subset:
            ax.scatter([r['Q'] for r in subset],[r['H_A'] for r in subset],s=48,
                       color=colors[key],label='%s (n=%d)'%(labels[key],len(subset)),zorder=3)
    for r in rows:
        ax.annotate(str(r['seed']),(r['Q'],r['H_A']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axvline(0,color='#555555',linewidth=1); ax.axhline(0,color='#555555',linewidth=1)
    ax.set_xlabel(r'$Q_s=\ln\mathrm{FID}_{B,512}-\ln\mathrm{FID}_{A,512}$'+'\nPositive: B worse at the switch')
    ax.set_ylabel(r'$H_A(s)=\ln\mathrm{FID}_{BA,1024}-\ln\mathrm{FID}_{AA,1024}$'+'\nNegative: B history better after A continuation')
    ax.set_title('PR101 source-to-future ranking audit\nPost-hoc descriptive analysis; joint n=%d'%len(rows))
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15); ax.margins(.15)
    if rows: ax.legend(fontsize=9,frameon=False)
    fig.savefig(out/'source_to_future_scatter.png',dpi=220)
    fig.savefig(out/'source_to_future_scatter.pdf')
    plt.close(fig)

def number(x): return 'NA' if x is None else '%.8g'%x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work-root',required=True,type=Path); args=ap.parse_args()
    work=args.work_root; out=work/'results'/EXPERIMENT_ID
    protocol,inv,source_rows=frozen_inputs(REPO,work)
    seal=load(out/'integrity_verification.json')
    require(seal['status']=='PASS' and seal['all_formal_jobs_terminal'],'integrity seal required before decoding')
    require(seal['protocol_sha256']==PROTOCOL_SHA,'seal protocol mismatch')
    require(not (out/'decoded_source_metrics.csv').exists(),'decoded results already exist')
    decoded=[]
    for r in seal['jobs']:
        receipt_path=work/'receipts/formal_jobs'/(r['opaque_id']+'.json')
        check_hash(receipt_path,r['evaluation_receipt_sha256'])
        d=dict(seed=r['seed'],arm=r['arm'],fid50k='',kid50k='',
               source_checkpoint_sha256=r['source_checkpoint_sha256'],exported_model_sha256=r['exported_model_sha256'],
               generated_features_sha256=r.get('generated_features_sha256',''),evaluation_status=r['status'])
        if r['status']=='PASS':
            jobdir=work/'evaluation/formal/jobs'/r['opaque_id']
            for metric,field in [('fid50k_full','fid50k'),('kid50k_full','kid50k')]:
                p=jobdir/('metric-'+metric+'.jsonl'); check_hash(p,r['metric_artifact_sha256'][metric])
                lines=[x for x in p.read_text().splitlines() if x.strip()]
                require(len(lines)==1,'metric line count mismatch')
                import json
                d[field]=float(json.loads(lines[0])['results'][metric])
        decoded.append(d)
    write_csv(out/'decoded_source_metrics.csv',decoded,['seed','arm','fid50k','kid50k','source_checkpoint_sha256',
              'exported_model_sha256','generated_features_sha256','evaluation_status'])
    pairs=source_pairs(decoded)
    write_csv(out/'source_pairs.csv',pairs,['seed','fid_A_512','fid_B_512','kid_A_512','kid_B_512',
              'Q_logfid_B_minus_A','B_worse_at_512'])
    source_stats=summarize([r['Q_logfid_B_minus_A'] for r in pairs])
    endpoint=REPO/protocol['joint_analysis']['endpoint_file']
    check_hash(endpoint,protocol['joint_analysis']['endpoint_file_sha256'])
    endpoints=read_csv(endpoint)
    joint=join_future(pairs,endpoints)
    write_csv(out/'source_to_future_per_seed.csv',joint,['seed','fid_A_512','fid_B_512','Q','fid_AA_1024','fid_BA_1024',
              'H_A','source_B_worse','future_B_history_better','delayed_reversal'])
    quad=quadrants(joint); corr=correlation(joint)
    counts=dict(joint_n=len(joint),Q_gt_zero=sum(r['Q']>0 for r in joint),H_A_lt_zero=sum(r['H_A']<0 for r in joint),
                delayed_reversals=sum(r['delayed_reversal'] for r in joint))
    require(sum(quad.values())==len(joint) and quad['reversal']==counts['delayed_reversals'],'quadrant reconciliation failed')
    excluded=[dict(seed=r['seed'],arm=r['arm'],reason=r['exclusion_reason']) for r in inv['rows'] if not r['source_valid']]
    unmatched=sorted({r['seed'] for r in pairs}-{r['seed'] for r in joint})
    failed=[dict(seed=r['seed'],arm=r['arm'],status=r['evaluation_status']) for r in decoded if r['evaluation_status']!='PASS']
    stats=dict(source_valid_pairs=protocol['cohort']['n_source_pairs'],formal_jobs=len(decoded),
       pass_count=sum(r['evaluation_status']=='PASS' for r in decoded),failed_count=sum(r['evaluation_status']=='FAILED' for r in decoded),
       not_run_count=sum(r['evaluation_status']=='NOT_RUN' for r in decoded),source_quality=source_stats,
       joint_counts=counts,quadrants=quad,correlation=corr,source_exclusions=excluded,
       source_pairs_without_complete_frozen_endpoint=unmatched,evaluation_failures=failed,
       source_pair_seeds=[r['seed'] for r in pairs],joint_seeds=[r['seed'] for r in joint],
       protocol_sha256=PROTOCOL_SHA,pr101_primary_verdict_unchanged=True,analysis_role='post-hoc descriptive')
    write_json(out/'statistics.json',stats)
    plot(joint,out)
    cost=load(out/'GPU_COST_REPORT.json')
    ci=source_stats['ci95']
    ci_text='NA' if ci is None else '[%s, %s]'%(number(ci[0]),number(ci[1]))
    claim=('Within the PR101 q256 cohort, one-step FID at 512 kimg does not always preserve the ordering of downstream continuation value. '
           'Some enlarged-spacing histories that are worse by FID at the switch later yield better final quality under the same native-spacing continuation.'
           if counts['delayed_reversals'] else 'No Q>0, H_A<0 reversal was observed in the available joint cohort.')
    report=f'''# PR101 512-kimg source-to-future ranking audit

This is a post-hoc source-quality backfill and structural audit. No new training was performed. The frozen PR101 primary verdict is unchanged; this analysis was not its preregistered primary endpoint.

## Cohort and evidence

- Source-valid A/B@512 pairs: {protocol['cohort']['n_source_pairs']}.
- Formal jobs: {len(decoded)}; PASS / FAILED / NOT_RUN: {stats['pass_count']} / {stats['failed_count']} / {stats['not_run_count']}.
- Source pairs with complete finite FID/KID: {len(pairs)}.
- Joint source plus frozen AA/BA@1024 cohort: {len(joint)}.
- Source exclusions: {excluded}.
- Valid source pairs without a complete frozen endpoint: {unmatched}.
- Evaluation failures: {failed}.

Source inclusion used only the retained A/B@512 states. All selected full-state file hashes, 512000-image / 4000-attempt counters, seed and factorial identity, EMA finiteness, internal hashes, and telemetry were verified on ECT002 before freezing. Extracted source receipts were unreadable to ECT002, so their copies were read from the historical audit tar after verifying its recorded SHA256. This permission limitation did not remove a seed.

The unchanged evaluator, detector, and real-feature bytes are bound by protocol hashes. Dataset-path cache aliases preserve the frozen real-feature bytes. Each evaluation uses sample IDs 0–49999, metric seed 20260730, NFE1 and FP32. FID50k and KID50k share generated features, with equality checked by SHA256. The integrity seal predates scalar decoding.

## Source-quality descriptive summary

Q = ln(FID_B@512) − ln(FID_A@512); positive Q means B has worse source FID.

| Statistic | Value |
|---|---:|
| n | {source_stats['n']} |
| Mean Q | {number(source_stats['mean'])} |
| Median Q | {number(source_stats['median'])} |
| Sample SD | {number(source_stats['sd'])} |
| 95% Student-t CI for mean Q | {ci_text} |
| Q > 0 | {source_stats['Q_gt_zero']} |
| Q < 0 | {source_stats['Q_lt_zero']} |
| Q = 0 | {source_stats['Q_eq_zero']} |

## Source-to-future descriptive comparison

H_A = ln(FID_BA@1024) − ln(FID_AA@1024), using the frozen PR101 complete endpoint file. Both scalars independently reproduce its stored contrast. The join uses actual seed overlap; no joint sample size was imposed.

Joint Q>0: {counts['Q_gt_zero']}/{len(joint)}. Joint H_A<0: {counts['H_A_lt_zero']}/{len(joint)}. Delayed reversals: **{counts['delayed_reversals']}/{len(joint)}**.

| Source ordering | H_A < 0 | H_A > 0 |
|---|---:|---:|
| Q > 0 | {quad['reversal']} reversal | {quad['bad_to_bad']} bad → bad |
| Q < 0 | {quad['good_to_good']} good → good | {quad['reverse_loss']} reverse-loss |

On-axis observations: {quad['on_axis']}. Exact ties are retained separately, without an epsilon relabeling rule.

Pearson(Q,H_A): {number(corr['pearson'])}. Spearman(Q,H_A): {number(corr['spearman'])}. These correlations are descriptive. No confirmatory correlation test or causal regression was performed.

![Seed-labeled source-to-future scatter](source_to_future_scatter.png)

## Interpretation boundary

{claim}

The result is restricted to this PR101 cohort and available complete joint observations. It does not establish general FID unreliability, systematic misranking, or a causal effect of source FID on future quality. PR95/97/101 were not pooled into an unplanned confirmatory p-value. Missing endpoints are reported rather than used to exclude valid sources.

## Measured compute

- Formal A100 GPUh: {cost['formal_gpu_hours']:.6f}.
- Smoke A100 GPUh: {cost['smoke_gpu_seconds']/3600:.6f}.
- Total evaluation A100 GPUh: {cost['total_a100_gpu_hours']:.6f}.
- Formal matrix wall time: {cost['formal_wall_seconds']:.3f} seconds ({cost['formal_wall_seconds']/3600:.6f} hours).
- Accounting: {cost['accounting']}

The scalar CSVs independently reproduce Q, the join, and the quadrant table. Protocol SHA256: `{PROTOCOL_SHA}`. Final packaging and ECT002 return verification are recorded separately after this report is sealed.
'''
    (out/'SOURCE_TO_FUTURE_REPORT.md').write_text(report)
    er=out/'evaluation_receipts'; er.mkdir()
    for r in seal['jobs']:
        shutil.copy2(work/'receipts/formal_jobs'/(r['opaque_id']+'.json'),er/(r['opaque_id']+'.json'))
    for name in ('smoke_attempt00.json','environment_parity.json','transfer_sha_verification.json','execution_code_freeze.json'):
        shutil.copy2(work/'receipts'/name,er/name)
    with (out/'SHA256SUMS.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt': f.write(sha256(p)+'  '+str(p.relative_to(out))+'\n')
    print('POST_HOC_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__': main()
