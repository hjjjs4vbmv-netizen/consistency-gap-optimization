"""64-sample smoke adapter; imports unchanged frozen FID/KID implementations."""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0,os.environ['Q256_FROZEN_EVALUATOR'])
import ct_eval
from metrics import metric_main, kernel_inception_distance, frechet_inception_distance
from training import reproducibility

_copy=ct_eval.misc.copy_params_and_buffers

def checked_copy(src_module,dst_module,require_all=False):
    source=src_module.state_dict(); target=dst_module.state_dict()
    if set(source)!=set(target): raise RuntimeError('smoke evaluator parameter/buffer keys differ from exported EMA')
    if any(source[k].shape!=target[k].shape or source[k].dtype!=target[k].dtype for k in source):
        raise RuntimeError('smoke evaluator parameter/buffer shape or dtype mismatch')
    _copy(src_module,dst_module,require_all=True)
    source_sha=reproducibility.module_state_sha256(src_module)
    target_sha=reproducibility.module_state_sha256(dst_module)
    if source_sha!=target_sha: raise RuntimeError('smoke evaluator did not load the exact exported EMA state')
    out=Path(sys.argv[sys.argv.index('--outdir')+1])
    (out/'model-transfer-binding.json').write_text(json.dumps(dict(status='PASS',
        source_ema_canonical_sha256=source_sha,evaluation_model_canonical_sha256=target_sha,
        parameter_buffer_count=len(source),strict_key_shape_dtype_match=True),indent=2)+'\n')

ct_eval.misc.copy_params_and_buffers=checked_copy

@metric_main.register_metric
def source_smoke_kid64(opts):
    opts.dataset_kwargs.update(max_size=None,xflip=False)
    value=kernel_inception_distance.compute_kid(opts,max_real=1000000,num_gen=64,
             num_subsets=100,max_subset_size=1000,random_seed=opts.metric_seed)
    return dict(source_smoke_kid64=value)

@metric_main.register_metric
def source_smoke_fid64(opts):
    opts.dataset_kwargs.update(max_size=None,xflip=False)
    return dict(source_smoke_fid64=frechet_inception_distance.compute_fid(opts,max_real=None,num_gen=64))

if __name__=='__main__': ct_eval.main()
