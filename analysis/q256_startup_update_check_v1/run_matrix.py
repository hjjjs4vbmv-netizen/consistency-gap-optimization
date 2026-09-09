"""Prepare immutable inputs or execute the eight slots serially under a hard budget."""
import argparse,datetime,fcntl,hashlib,inspect,json,os,platform,signal,subprocess,time
from pathlib import Path
from analysis.q256_startup_update_check_v1.protocol import ARMS,BASE,PROTOCOL,EXPECTED_ENV,native_cli

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj,exclusive=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def environment(python,gpu='0'):
    prefix=Path(python).parent.parent;site=prefix/'lib/python3.11/site-packages'
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               CUDA_CACHE_DISABLE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
               LD_LIBRARY_PATH=':'.join(map(str,[prefix/'lib',site/'torch/lib',*sorted((site/'nvidia').glob('*/lib'))])),
               PATH=f'{prefix}/bin:/usr/bin:/bin')
    return env

def prepare(root,python,commit):
    import torch,numpy,scipy
    import importlib.metadata
    from torch.optim import radam
    from training.startup_update import DATA_SHA256,TRANSFER_SHA256
    actual=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),numpy=numpy.__version__,scipy=scipy.__version__)
    if actual!=EXPECTED_ENV:raise RuntimeError(f'environment mismatch: {actual}')
    code=Path(__file__).resolve().parents[2];control=root/'control'
    dataset=root/'assets/cifar10-32x32-training-original.zip'
    transfer=Path('/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl')
    for path,expected in ((dataset,DATA_SHA256),(transfer,TRANSFER_SHA256)):
        if sha(path)!=expected:raise ValueError('asset mismatch '+str(path))
    for key,value in dict(click='8.2.1',pillow='11.3.0',psutil='7.0.0').items():
        if importlib.metadata.version(key)!=value:raise RuntimeError('original auxiliary runtime mismatch: '+key)
    source=inspect.getsource(radam)
    (control/'native_torch_radam.py.txt').write_text(source)
    write(control/'environment.json',{**actual,'platform':platform.platform(),'radam_source_sha256':hashlib.sha256(source.encode()).hexdigest()},True)
    freeze=dict(protocol=PROTOCOL,engineering_only=True,seeds=[50,51],arms=list(ARMS),attempts_per_arm=64,
                max_total_attempts=512,total_kimg=1024,base_lr=1e-4,budget_gpuh=3,per_process_timeout_seconds=1200,
                base_commit=BASE,implementation_commit=commit,created_utc=now(),
                numerical_hit_threshold=None,numerical_policy='Descriptive continuous norm ratio, cosine and relative L2; no post-hoc pass threshold.',
                inference_scope='startup implementation and local updates only; no FID or endpoint mechanism claims',
                initialization='original transfer + seed + configuration; archived full receipt must pass BEFORE attempt1',
                tensor_successful_steps=[1,5,6],comparison_alignment=['same_attempt','same_successful_step'],
                zero_vector_metrics='relative error/cosine N/A when a required norm is zero',
                asset_hashes={str(dataset):DATA_SHA256,str(transfer):TRANSFER_SHA256},
                cost_policy='conservative full GPU-training process-group wall duration, including setup/failures; idle rental not GPUh',
                retry_policy='no automatic scientific trajectory retries')
    write(control/'fixed_protocol.json',freeze,True)
    jobs=[]
    for seed in (50,51):
        receipt=root/f'receipts/seed{seed}/prefix_A/initial_state_receipt_v1.json'
        for arm in ARMS:
            slot=f'seed{seed}-{arm}';mp=control/f'{slot}.json'
            m=dict(protocol=PROTOCOL,engineering_only=True,seed=seed,arm=arm,attempts=64,total_kimg=1024,
                   initialization='fresh_transfer',reference_receipt=str(receipt),reference_receipt_sha256=sha(receipt),
                   dataset=str(dataset),transfer=str(transfer),output=str(root/'runs'/slot),manifest_path=str(mp))
            write(mp,m,True)
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
                     str(code/'analysis/q256_startup_update_check_v1/entry.py'),'--manifest',str(mp)]
            jobs.append(dict(slot=slot,seed=seed,arm=arm,status='NOT_RUN',command=command,
                             native_training_cli=native_cli(m),manifest_sha256=sha(mp),attempts=0))
    write(control/'eight_commands.json',jobs,True)
    files={str(p.relative_to(code)):sha(p) for d in ('training','torch_utils','dnnlib','metrics','analysis/q256_startup_update_check_v1')
           for p in (code/d).rglob('*.py') if not p.name.startswith('._')}
    files['ct_train.py']=sha(code/'ct_train.py')
    write(control/'source_files.json',files,True)
    write(control/'run_status.json',jobs,True)
    write(control/'gpu_ledger.json',dict(processes=[],process_gpuh=0.,gpu_budget=3),True)

def run(root,python):
    control=root/'control';code=Path(__file__).resolve().parents[2]
    lock=(control/'matrix.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in json.loads((control/'source_files.json').read_text()).items():
        if sha(code/name)!=h:raise RuntimeError('code changed after freeze: '+name)
    jobs=json.loads((control/'run_status.json').read_text())
    ledger=json.loads((control/'gpu_ledger.json').read_text())
    if ledger['processes']:raise RuntimeError('this driver never automatically retries or restarts the matrix')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('A100 not exclusive: '+active)
    for row in jobs:
        if sha(control/f"{row['slot']}.json")!=row['manifest_sha256']:raise RuntimeError('manifest changed')
        remaining=10800-sum(x['charged_seconds'] for x in ledger['processes'])
        if remaining<=10:
            row['status']='NOT_RUN_BUDGET';continue
        timeout=min(1200.,remaining-10)
        record=dict(slot=row['slot'],started_utc=now(),hard_timeout_seconds=timeout,command=row['command'])
        log=control/f"{row['slot']}.log"
        start=time.monotonic();proc=None
        try:
            with log.open('x') as handle:
                proc=subprocess.Popen(row['command'],cwd=code,env=environment(python),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=proc.pid;row['status']='RUNNING'
                write(control/'current_process.json',record);write(control/'run_status.json',jobs)
                try:rc=proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait();record['hard_timeout']=True
                record['exit_code']=rc
                row['status']='COMPLETED' if rc==0 else 'FAILED_TIMEOUT' if record.get('hard_timeout') else 'FAILED'
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            row['status']='FAILED_DRIVER';record['error']=repr(exc)
            raise
        finally:
            record.update(ended_utc=now(),charged_seconds=time.monotonic()-start)
            record['process_gpuh']=record['charged_seconds']/3600
            data=root/'runs'/row['slot']/'startup_attempts.jsonl'
            rows=[json.loads(x) for x in data.read_text().splitlines()] if data.exists() else []
            row['attempts']=len(rows);row['successful_steps']=rows[-1]['successful_optimizer_steps'] if rows else 0
            row['skips']=sum(x['skip'] for x in rows)
            if row['status']=='COMPLETED' and len(rows)!=64:row['status']='FAILED_COVERAGE'
            record['status']=row['status'];ledger['processes'].append(record)
            ledger['process_gpuh']=sum(x['process_gpuh'] for x in ledger['processes'])
            write(control/'gpu_ledger.json',ledger);write(control/'run_status.json',jobs)
    write(control/'matrix_finished.json',dict(finished_utc=now(),attempts=sum(r['attempts'] for r in jobs),
          all_eight_completed=all(r['status']=='COMPLETED' for r in jobs),process_gpuh=ledger['process_gpuh']))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',required=True);parser.add_argument('--commit');a=parser.parse_args()
    if a.mode=='prepare':prepare(a.root,a.python,a.commit)
    else:run(a.root,a.python)
