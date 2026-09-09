"""Recompute saved B update/gradient norms without any forward/backward or FID."""
import torch,json,pathlib,math,argparse

parser=argparse.ArgumentParser(description="Offline CPU norms from the saved first-batch audit and original D@512 states")
parser.add_argument("--audit-root",type=pathlib.Path,required=True)
parser.add_argument("--source-inventory",type=pathlib.Path,required=True)
parser.add_argument("--output",type=pathlib.Path,required=True)
args=parser.parse_args()
torch.set_num_threads(1)
source=json.loads(args.source_inventory.read_text());rows=[]
def compare(a,b):
 aa=bb=ee=ab=0.;count=0;excluded=0
 for x,y in zip(a,b):
  if x is None or y is None:continue
  x=x.detach().double().reshape(-1);y=y.detach().double().reshape(-1);f=torch.isfinite(x)&torch.isfinite(y)
  excluded+=int((~f).sum());x=x[f];y=y[f];d=x-y
  aa+=float((x*x).sum());bb+=float((y*y).sum());ee+=float((d*d).sum());ab+=float((x*y).sum());count+=x.numel()
 return {'l2_left':math.sqrt(aa),'l2_right':math.sqrt(bb),'l2_error':math.sqrt(ee),'relative_l2_error_to_right':math.sqrt(ee/bb) if bb else None,
         'cosine':ab/math.sqrt(aa*bb) if aa and bb else None,'finite_elements':count,'excluded_nonfinite_elements':excluded}
for seed in (50,51):
 root=args.audit_root/'B'/f'seed{seed}'/'D_prefix_512'
 loaded={v:torch.load(root/v/'first_effective_batch.pt',map_location='cpu',weights_only=False,mmap=True) for v in ('A','D','A_scalar','A_lr')}
 src=torch.load(next(x for x in source if x['seed']==seed)['prefix_path'],map_location='cpu',weights_only=False,mmap=True)
 params=list(src['net'].parameters())
 for a,b in [('D','A_scalar'),('D','A_lr'),('A','A_lr')]:
  x,y=loaded[a],loaded[b]
  assert x['step_called'] and y['step_called'], 'a common actual update is required'
  assert len(x['parameters_after'])==len(y['parameters_after'])==len(params)
  assert all(v.shape==w.shape==z.shape for v,w,z in zip(x['parameters_after'],y['parameters_after'],params))
  update_a=[v.double()-s.double() for v,s in zip(x['parameters_after'],params)]
  update_b=[v.double()-s.double() for v,s in zip(y['parameters_after'],params)]
  rows.append({'seed':seed,'comparison':a+'_vs_'+b,'gradient':compare(x['raw_gradients'],y['raw_gradients']),
               'actual_parameter_update':compare(update_a,update_b)})
  del update_a,update_b
 del src,params,loaded
args.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
print(json.dumps(rows))
