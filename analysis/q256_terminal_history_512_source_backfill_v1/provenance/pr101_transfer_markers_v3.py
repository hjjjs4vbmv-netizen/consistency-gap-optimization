import concurrent.futures,json,subprocess,time
from pathlib import Path
R=Path('/home/ECT002/q256-pr101-512-source-fid-backfill-v1')
def lane(n):
 sock=R/('a100-control.sock' if n==0 else 'a100-lane%d.sock'%n)
 for i in range(n,58,32):
  if i<8:continue
  p=R/'receipts'/('ema_job%03d.json'%i)
  while not p.exists():time.sleep(2)
  r=json.loads(p.read_text());assert r['status']=='PASS'
  subprocess.run(['scp','-o','BatchMode=yes','-o','ControlPath='+str(sock),'-P','27815',str(p),'root@px-cloud1.matpool.com:/root/q256-pr101-512-source-fid-backfill-v1/receipts/transfers/.'+p.name+'.tmp'],check=True,stdout=subprocess.DEVNULL)
  subprocess.run(['ssh','-o','BatchMode=yes','-S',str(sock),'-p','27815','root@px-cloud1.matpool.com','mv /root/q256-pr101-512-source-fid-backfill-v1/receipts/transfers/.'+p.name+'.tmp /root/q256-pr101-512-source-fid-backfill-v1/receipts/transfers/'+p.name],check=True)
  print(i,'TRANSFER_COMPLETION_DELIVERED',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as p:list(p.map(lane,range(32)))
