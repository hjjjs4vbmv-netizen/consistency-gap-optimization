"""Assemble and checksum a complete result archive after descriptive analysis."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from analysis.q256_terminal_history_512_source_backfill_v1.common import (
    EXPERIMENT_ID,PROTOCOL_SHA,check_hash,load,require,sha256,write_json)

def copy_tree(source,target,exclude=()):
    shutil.copytree(source,target,copy_function=os.link,symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc',*exclude))

import os

def main():
    p=argparse.ArgumentParser(); p.add_argument('--work-root',type=Path,required=True); args=p.parse_args()
    work=args.work_root; result=work/'results'/EXPERIMENT_ID
    require(load(result/'integrity_verification.json')['status']=='PASS','integrity seal required')
    require((result/'SOURCE_TO_FUTURE_REPORT.md').is_file(),'descriptive report required')
    require(load(work/'receipts/final_validation.json')['status']=='PASS','final tests and scalar replay gate required')
    subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS.txt'],cwd=result,check=True)
    payload=work/'archive_payload'/EXPERIMENT_ID
    require(not payload.exists(),'archive payload already exists')
    payload.mkdir(parents=True)
    (payload/'analysis').mkdir(); (payload/'results').mkdir(); (payload/'tests').mkdir()
    copy_tree(REPO/'analysis'/EXPERIMENT_ID,payload/'analysis'/EXPERIMENT_ID)
    copy_tree(result,payload/'results'/EXPERIMENT_ID)
    copy_tree(work/'source_inventory',payload/'source_inventory')
    copy_tree(work/'receipts',payload/'receipts')
    copy_tree(work/'logs',payload/'logs')
    copy_tree(work/'evaluation',payload/'evaluation',exclude=('caches','*.npy','*.pkl'))
    os.link(REPO/'tests/test_pr101_source_backfill.py',payload/'tests/test_pr101_source_backfill.py')
    stats=load(result/'statistics.json'); cost=load(result/'GPU_COST_REPORT.json')
    write_json(payload/'ARCHIVE_CONTENTS.json',dict(experiment_id=EXPERIMENT_ID,protocol_sha256=PROTOCOL_SHA,
       no_new_training=True,pr101_primary_verdict_unchanged=True,
       includes=['analysis protocol/code','all source inventory records','decoded and paired scalar CSVs',
                 'post-hoc joint report and plots','evaluation and EMA export receipts','formal scalar metrics and artifact hashes','formal and preparation logs','compute accounting'],
       large_generated_artifacts='A100 working copy retains generated samples/features and exported snapshots; archive includes their immutable hashes and receipts',
       large_original_states='Not duplicated: preserved read-only in the original ECT002 archive, with verified source SHA bindings',
       frozen_assets='Preserved in original ECT002 archive and referenced by protocol SHA; no new real features',
       source_valid_pairs=stats['source_valid_pairs'],formal_jobs=stats['formal_jobs'],
       joint_n=stats['joint_counts']['joint_n'],delayed_reversals=stats['joint_counts']['delayed_reversals'],
       total_a100_gpu_hours=cost['total_a100_gpu_hours'],formal_wall_seconds=cost['formal_wall_seconds']))
    manifest=payload/'SHA256SUMS.txt'
    with manifest.open('x') as f:
        for item in sorted(payload.rglob('*')):
            if item.is_file() and item!=manifest:
                f.write(sha256(item)+'  '+str(item.relative_to(payload))+'\n')
    subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS.txt'],cwd=payload,check=True)
    archive=work/(EXPERIMENT_ID+'.tar.zst')
    require(not archive.exists(),'archive file already exists')
    started=time.time()
    subprocess.run(['tar','--sort=name','--numeric-owner','--owner=0','--group=0','--format=posix',
        '-I','zstd -T8 -3','-cf',str(archive),'-C',str(payload.parent),EXPERIMENT_ID],check=True)
    digest=sha256(archive)
    write_json(work/'PACKAGE_RECEIPT.json',dict(status='PASS',experiment_id=EXPERIMENT_ID,
        archive_path=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
        protocol_sha256=PROTOCOL_SHA,payload_manifest_sha256=sha256(manifest),
        packaging_started_epoch=started,packaging_ended_epoch=time.time(),
        return_directory='/data/raw/ECT/'+EXPERIMENT_ID,remote_sha_verified=False))
    (work/(archive.name+'.sha256')).write_text(digest+'  '+archive.name+'\n')
    print('ARCHIVE_SEALED',digest,flush=True)

if __name__=='__main__': main()
