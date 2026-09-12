"""Verify recorded scientific Git hashes and retrieved duplicate archive files on CPU."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];REPO=R.parents[1]
def read(p):return json.loads(p.read_text())
freeze=read(R/'evidence/control/freeze.json');commit=freeze['implementation_commit']
checks=[]
for path,expected in read(R/'evidence/control/source_hashes.json').items():
 raw=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=REPO)
 actual=hashlib.sha256(raw).hexdigest()
 checks.append(dict(path=path,recorded_sha256=expected,git_sha256=actual,status='PASS' if actual==expected else 'FAIL'))
assert all(x['status']=='PASS' for x in checks)
(R/'validation/frozen_source_binding.json').write_text(json.dumps(dict(scientific_commit=commit,checks=checks),indent=2)+'\n')
bindings=read(R/'SOURCE_BINDINGS.json')['files'];incident=read(R/'evidence/supervision/duplicate_dispatch_incidents/1789124222736135699/DUPLICATE_ARCHIVE_VERIFIED.json');result=[]
for run in incident['runs']:
 files=[]
 for source in run['files']:
  matches=[x for x in bindings if run['job'] in x['source_identity'] and x['source_identity'].endswith('/'+source['path']) and x['raw_source_sha256']==source['sha256']]
  if matches:files.append(dict(file=source['path'],sha256=source['sha256'],public_files=[x['public_file'] for x in matches]))
 assert any(x['file'].startswith('process-receipt-') for x in files),run['job']
 result.append(dict(job=run['job'],status='PASS',matched_small_files=files,scope='retrieved small files only; images and checkpoints deliberately not transferred'))
(R/'validation/duplicate_archive_bindings.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='PASS',frozen_source_files=len(checks),duplicate_archive_receipts=len(result)),indent=2))
