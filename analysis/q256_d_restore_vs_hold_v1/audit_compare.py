"""Quantify arithmetic/gradient/update differences without quality evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from training import m1


def error(a,b):
    a=a.detach().cpu();b=b.detach().cpu()
    if a.shape!=b.shape or a.dtype!=b.dtype:return {'shape_or_dtype_mismatch':True}
    bits=(a.contiguous().reshape(-1).view(torch.uint8)==b.contiguous().reshape(-1).view(torch.uint8)).reshape(-1,a.element_size()).all(dim=1)
    same=bits; finite=torch.isfinite(a)&torch.isfinite(b)
    x=a[finite].double();y=b[finite].double();diff=(x-y).abs()
    return {'dtype':str(a.dtype),'elements':a.numel(),'bitwise_equal':bool(bits.all()),
            'equal_fraction':float(same.double().mean()) if a.numel() else 1,
            'nonfinite_a':int((~torch.isfinite(a)).sum()),'nonfinite_b':int((~torch.isfinite(b)).sum()),
            'max_abs':float(diff.max()) if diff.numel() else None,
            'max_relative_to_nonzero_b':float((diff[y!=0]/y[y!=0].abs()).max()) if bool((y!=0).any()) else None,
            'zero_reference_nonzero_difference_count':int(((y==0)&(diff!=0)).sum())}


def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,dict):
        for k in sorted(x,key=str):yield from tensors(x[k])
    elif isinstance(x,(list,tuple)):
        for value in x:yield from tensors(value)


def collection(a,b):
    a=list(tensors(a));b=list(tensors(b))
    if len(a)!=len(b):return {'tensor_count_mismatch':[len(a),len(b)]}
    reports=[error(x,y) for x,y in zip(a,b)]
    return {'all_bitwise_equal':all(r.get('bitwise_equal',False) for r in reports),'tensors':reports}


def compare(paths):
    loaded={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in paths.items()}
    result={}
    for left,right in [('D','A_scalar'),('D','A'),('D','A_lr'),('A','A_lr')]:
        a,b=loaded[left],loaded[right]
        row={}
        for field in ('loss','raw_gradients','parameters_after','optimizer_after','forward'):
            row[field]=collection(a.get(field,[]),b.get(field,[]))
        row['scaler_before_equal']=m1._equal_state(a['scaler_before'],b['scaler_before'])
        row['scaler_after_equal']=m1._equal_state(a['scaler_after'],b['scaler_after'])
        row['step_called']=[a['step_called'],b['step_called']]
        result[left+'_vs_'+right]=row
    a,d=loaded['A'],loaded['D']
    result['denominator_D_vs_1.1_A']=[error(x['delta_denominator'],y['delta_denominator']*1.1) for x,y in zip(d['pairs'],a['pairs'])]
    result['target_times_D_vs_A']=collection([x['r_target'] for x in d['pairs']],[x['r_target'] for x in a['pairs']])
    result['scope']='first common effective batch; separate 16-attempt rollout telemetry; no full-trajectory equivalence claim'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); result=compare({v:a.root/v/'first_effective_batch.pt' for v in ('A','D','A_scalar','A_lr')})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
