"""The frozen metric validation extended only to prespecified NFE2 mid_t."""
import json
from pathlib import Path
import numpy as np
from scripts import validate_m1_evaluation_job as validation
from training import schedule_switch

def validate_result(slot, snapshot, directory, dataset, exit_code):
    if exit_code!=0: raise RuntimeError(f'evaluator process exit {exit_code}')
    if 'Exiting...' not in (directory/'log.txt').read_text(errors='replace'):
        raise RuntimeError('evaluator did not finish')
    options=json.loads((directory/'training_options.json').read_text())
    start,end=slot['sample_seed_start'],slot['sample_seed_end']
    expected={'sample_seeds':list(range(start,end+1)),'seed':20260730,
              'metrics':['kid50k_full','fid50k_full'],'metric_repeats':1,
              'metric_generator_batch':128,'retain_generated_artifacts':True,'mid_t':[] if slot['nfe']==1 else [.821]}
    for key,value in expected.items():
        if options.get(key)!=value: raise RuntimeError('evaluator option mismatch: '+key)
    if options['network_kwargs'].get('use_fp16') is not False: raise RuntimeError('evaluator is not FP32')
    if Path(options['resume_pkl']).resolve()!=Path(snapshot).resolve(): raise RuntimeError('wrong snapshot')
    if Path(options['dataset_kwargs']['path']).resolve()!=Path(dataset).resolve(): raise RuntimeError('wrong reference dataset')
    samples=np.load(directory/'generated-samples.npy',mmap_mode='r')
    if samples.shape[0]!=50000: raise RuntimeError('not a FID50k block')
    del samples
    results={}; hashes=[]
    for metric in ('kid50k_full','fid50k_full'):
        row=validation.read_metric(directory/f'metric-{metric}.jsonl',metric)
        if row['status']!='SEALED_PASS': raise RuntimeError('invalid '+metric+': '+str(row))
        feature=directory/f'generated-features-{metric}-repeat00.npy'
        if np.load(feature,mmap_mode='r').shape[0]!=50000: raise RuntimeError('wrong feature count')
        hashes.append(schedule_switch.sha256_file(feature))
        results[metric]=row['value']
    if hashes[0]!=hashes[1]: raise RuntimeError('FID and KID did not share generated features')
    return {'FID':results['fid50k_full'],'KID':results['kid50k_full'],'feature_sha256':hashes[0]}

