"""Engineering-only process instrumentation. Never used by the formal DD launcher.

Variant A_scalar divides native A's per-sample output before the existing per-
microbatch mean and GradScaler backward. Variant A_lr divides LR only inside
actual RAdam.step; gradients and moments use unscaled A. No replacement dtype,
reduction, optimizer or AMP behavior is introduced. At most 16 attempts here.
"""
from __future__ import annotations
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from training import loss as loss_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('A','D','A_scalar','A_lr'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args,cli=parser.parse_known_args()
    if cli and cli[0]=='--':cli=cli[1:]
    stops=[x for x in cli if x.startswith('--stop-after-attempts=')]
    if len(stops)!=1 or int(stops[0].split('=')[1]) not in (16,4016):
        raise RuntimeError('audit entry only permits fixed 16-attempt short runs')
    if '--metrics=none' not in cli: raise RuntimeError('engineering never computes FID')
    args.output.mkdir(parents=True,exist_ok=True)
    first={'variant':args.variant,'position':'native per-sample A output / 1.1 BEFORE microbatch mean' if args.variant=='A_scalar' else 'native',
           'lr_position':'inside RAdam.step after AMP unscale and original gradient sanitization' if args.variant=='A_lr' else 'original',
           'loss':[],'pairs':[],'forward':[]}
    calls=0; unscales=0; steps=0
    original_loss=loss_module.ECMLoss.__call__
    original_times=loss_module.compute_target_weight_times
    def times(*a,**kw):
        result=original_times(*a,**kw)
        if calls<=8:
            first['pairs'].append({'t':a[0].detach().cpu(),'base_r':a[1].detach().cpu(),
                                   'r_target':result[0].detach().cpu(),'r_denominator':result[1].detach().cpu(),
                                   'delta_target':result[2].detach().cpu(),'delta_denominator':result[3].detach().cpu()})
        return result
    loss_module.compute_target_weight_times=times
    def call(self,net,*a,**kw):
        nonlocal calls
        calls+=1; saved=self.factorial
        factor='D' if args.variant=='D' else 'A'
        self.factorial={**saved,'arm':factor,'target_gap_scale':1.0,'denominator_gap_scale':1.1 if factor=='D' else 1.0}
        handle=None
        if calls==1:
            inner=net.module.model if hasattr(net,'module') else net.model
            def dtype_hook(module,inputs):
                first['actual_inner_input_dtype']=str(inputs[0].dtype)
                dtype_handle.remove()
            dtype_handle=inner.register_forward_pre_hook(dtype_hook)
        if calls<=8:
            def hook(module,inputs,output):
                first['forward'].append({'input':inputs[0].detach().cpu(),'t':inputs[1].detach().cpu(),'output':output.detach().cpu()})
            handle=net.register_forward_hook(hook)
        try:
            value=original_loss(self,net,*a,**kw)
            if args.variant=='A_scalar':value=value/1.1
            if calls<=8:first['loss'].append(value.detach().cpu())
            return value
        finally:
            self.factorial=saved
            if handle:handle.remove()
    loss_module.ECMLoss.__call__=call
    original_unscale=torch.cuda.amp.GradScaler.unscale_
    def unscale(self,optimizer):
        nonlocal unscales
        result=original_unscale(self,optimizer); unscales+=1
        if unscales==1:
            first['scaler_before']=copy.deepcopy(self.state_dict())
            first['raw_gradients']=[p.grad.detach().cpu().clone() if p.grad is not None else None for g in optimizer.param_groups for p in g['params']]
        return result
    torch.cuda.amp.GradScaler.unscale_=unscale
    original_step=torch.optim.RAdam.step
    def step(self,*a,**kw):
        nonlocal steps
        steps+=1
        rates=[g['lr'] for g in self.param_groups]
        if args.variant=='A_lr':
            for g in self.param_groups:g['lr']/=1.1
        try:
            result=original_step(self,*a,**kw)
            if unscales==1:
                first['parameters_after']=[p.detach().cpu().clone() for g in self.param_groups for p in g['params']]
                first['optimizer_after']=copy.deepcopy(self.state_dict())
                # torch.save handles original CUDA moments; analysis loads on CPU.
            return result
        finally:
            for g,lr in zip(self.param_groups,rates):g['lr']=lr
    torch.optim.RAdam.step=step
    original_update=torch.cuda.amp.GradScaler.update
    def update(self,*a,**kw):
        result=original_update(self,*a,**kw)
        if unscales==1:
            first['scaler_after']=copy.deepcopy(self.state_dict()); first['step_called']=steps>0
            target=args.output/'first_effective_batch.pt'
            if target.exists():raise RuntimeError('audit artifact already exists')
            torch.save(first,target)
            first.clear() # The remaining 15 attempts are a bounded rollout with ordinary telemetry.
        return result
    torch.cuda.amp.GradScaler.update=update
    sys.argv=[str(Path(__file__).resolve().parents[2]/'ct_train.py'),*cli]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
