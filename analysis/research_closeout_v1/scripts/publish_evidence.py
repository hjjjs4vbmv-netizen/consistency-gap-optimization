"""CPU-only publication of an existing read-only collection; no server access."""
import argparse,json,hashlib,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
raw=a.private_root/'raw';index=json.loads((raw/'reports/ARCHIVE_INDEX.json').read_text())
repo=Path(__file__).resolve().parents[3]
prior=json.loads((repo/'analysis/startup_window_experiments_v1/results/ARCHIVE_INDEX.json').read_text())
lookup={(r['protocol_id'],r['seed'],r['arm']):r for r in prior['records']}
aliases={'owned_ect':'owned'};roots=set()
for r in index['records']:
 q=lookup[(r['protocol_id'],r['seed'],r['arm'])]
 if '/archive/' in r['ect_directory']:
  aliases[r['ect_directory'].split('/archive/')[1].split('/')[0]]=q['ect_directory'].split('/archive/')[1].split('/')[0]
 roots.add(r['ect_directory'].split('/archive/')[0] if '/archive/' in r['ect_directory'] else r['ect_directory'].split('/runs/')[0])

def text(s):
 for root in sorted(roots,key=len,reverse=True):s=s.replace(root,'<ECT_ROOT>')
 # Preserve within-project file identities while removing private mount/user prefixes.
 s=re.sub(r'/(?:root|mnt|home|data)(?:/[A-Za-z0-9_.-]+)*/ect_q128_fourarm_q256_delayed_20260911', '<TRAINING_ROOT>',s)
 for src,dst in sorted(aliases.items(),key=lambda x:len(x[0]),reverse=True):s=s.replace(src,dst)
 s=re.sub(r'GPU-[0-9a-fA-F-]{20,}', '<GPU_UUID>',s)
 s=re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])','<PRIVATE_ADDRESS>',s)
 s=re.sub(r'[\w.-]+\.(?:matpool\.com|autodl\.com|autodl\.top)', '<PRIVATE_HOST>',s)
 s=re.sub(r'(?<![\w>])/(?:root|mnt|home|data|tmp|opt|usr|public)/[^\s\"\'<>;,\]\)]+',lambda m:'<PRIVATE_PATH>/'+m[0].rsplit('/',1)[-1],s)
 return s

def redact(v):
 if isinstance(v,dict):return {text(k):('<REDACTED>' if re.search(r'password|passwd|secret|token|private_key',k,re.I) else '<PORT>' if k.lower() in ('port','ssh_port') else '<USER>' if k.lower() in ('user','username') else redact(x)) for k,x in v.items()}
 if isinstance(v,list):return [redact(x) for x in v]
 return text(v) if isinstance(v,str) else v
m=json.loads((a.private_root/'raw_collection_manifest.json').read_text());records=[]
for e in m['entries']:
 source=a.private_root/e['bundle_file'];data=source.read_bytes()
 assert hashlib.sha256(data).hexdigest()==e['extracted_raw_sha256']
 rel=text(e['source_relative_path'])+('.prefix16' if e['extraction']['kind']!='whole_file' else '')
 if rel.endswith('.json'):
  public=(json.dumps(redact(json.loads(data)),indent=2,ensure_ascii=False)+'\n').encode()
 elif rel.endswith('.prefix16'):
  public=(''.join(json.dumps(redact(json.loads(line)),ensure_ascii=False)+'\n' for line in data.splitlines())).encode()
 else:public=text(data.decode()).encode()
 dest=a.output/'evidence'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(public)
 records.append({'public_file':str(dest.relative_to(a.output)),'source_identity':'<ECT_ROOT>/'+text(e['source_relative_path']),'raw_source_sha256':e['raw_source_sha256'],'source_bytes':e['source_bytes'],'source_mtime_untrusted':e['source_mtime'],'extraction':e['extraction'],'extracted_raw_sha256':e['extracted_raw_sha256'],'public_sha256':hashlib.sha256(public).hexdigest()})
manifest={'schema':'ect.research-closeout.source-bindings/v1','collection_wall':m['collected_at'],'additional_collection_wall':m.get('additional_collection_wall'),'publication_is_posthoc':True,'training_or_generation_gpuh':0,'redaction':'Stable host aliases aligned with PR113; physical GPU UUIDs, private paths/addresses and credential fields removed. Original hashes and byte/line extraction boundaries retained. No scientific numeric fields modified.','files':records,'unreadable_sources':redact(m['missing'])}
(a.output/'SOURCE_BINDINGS.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('Published',len(records),'small evidence files; raw source and public hashes kept separate.')
