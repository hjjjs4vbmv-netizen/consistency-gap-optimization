"""Run sealing and descriptive analysis only after every formal job terminates."""
import json,subprocess,time
from pathlib import Path
w=Path('/root/q256-pr101-512-source-fid-backfill-v1');r=w/'code';py=str(w/'runtime-env/bin/python');ID='q256_terminal_history_512_source_backfill_v1'
p=w/'receipts/formal_matrix_terminal.json'
while not p.exists():time.sleep(5)
s=json.loads(p.read_text());assert s['all_jobs_terminal'] and s['pass_count']==58 and s['failed_count']==0
commands=[[py,'-m','analysis.'+ID+'.evaluate','seal','--work-root',str(w)],[py,'-m','analysis.'+ID+'.analyze','--work-root',str(w)],[py,str(r/'analysis'/ID/'provenance/pr101_finalize_validate.py')]]
for command in commands:
 print('RUN',command,flush=True);subprocess.run(command,cwd=r,check=True)
print('RESULTS_READY_FOR_VISUAL_REVIEW_AND_PACKAGING',flush=True)
