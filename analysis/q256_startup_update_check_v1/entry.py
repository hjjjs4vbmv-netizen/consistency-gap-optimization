"""Dedicated engineering entry, accepting only a pre-frozen manifest."""
import argparse,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.q256_startup_update_check_v1.protocol import native_cli
from training.startup_update import validate_pause

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    m=json.loads(args.manifest.read_text())
    validate_pause(seed=m['seed'],attempts=m['attempts'],total_kimg=m['total_kimg'],
                   resume_state_dump=None,schedule_switch_manifest=None,manifest=m)
    if Path(m['manifest_path']).resolve()!=args.manifest.resolve():raise ValueError('manifest identity mismatch')
    if Path(m['output']).exists():raise FileExistsError('never overwrite an existing trajectory')
    sys.argv=[str(ROOT/'ct_train.py'),*native_cli(m)]
    runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
