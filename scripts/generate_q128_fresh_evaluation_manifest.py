#!/usr/bin/env python3
"""Generate the complete frozen 272-job q128 evaluation matrix."""

from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/q128_fresh_regime_history_n8_v1/evaluation_manifest.json"
PROTOCOL = ROOT / "analysis/q128_fresh_regime_history_n8_v1/protocol.json"

def oid(parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]

def main():
    protocol = json.loads(PROTOCOL.read_text())
    jobs=[]
    def add(seed, trajectory, budget, nfe, path, roles):
        category = "PRIMARY" if "PRIMARY" in roles else ("KEY_SECONDARY" if any(r.startswith("KEY_SECONDARY") for r in roles) else "DESCRIPTIVE")
        jobs.append({"job_index":len(jobs),"opaque_id":oid([protocol["protocol_id"],seed,trajectory,budget,nfe]),
                     "seed":seed,"trajectory":trajectory,"budget_kimg":budget,"nfe":nfe,
                     "mid_t":0.821 if nfe==2 else None,"category":category,"analysis_roles":roles,
                     "checkpoint_path":path,"checkpoint_sha256":None,"status":"FROZEN_NOT_RUN"})
    base="/root/q128_fresh_regime_history_n8_v1/formal"
    for seed in protocol["cohort"]["formal_seeds"]:
        for arm in ("A","Bsame","Bmatch","Cmatch","Dmatch"):
            for budget in (512,768,1024):
                roles=["DESCRIPTIVE_FIVE_ARM"]
                if arm in ("A","Bsame") and budget in (512,1024): roles += ["PRIMARY","KEY_SECONDARY_PHASE"]
                if arm in ("Bmatch","Cmatch") and budget==1024: roles += ["KEY_SECONDARY_NFE"]
                add(seed,arm,budget,1,f"{base}/seed{seed}/arm{arm}/network-snapshot-kimg{budget:06d}.pkl",roles)
            roles=["DESCRIPTIVE_FIVE_ARM"] + (["KEY_SECONDARY_NFE"] if arm in ("Bmatch","Cmatch") else [])
            add(seed,arm,1024,2,f"{base}/seed{seed}/arm{arm}/network-snapshot-kimg001024.pkl",roles)
        for arm in ("A","Bsame"):
            for budget in (640,896):
                add(seed,arm,budget,1,f"{base}/seed{seed}/arm{arm}/network-snapshot-kimg{budget:06d}.pkl",["DESCRIPTIVE_CROSSED_NATIVE"])
        for branch in ("AB","BA"):
            for budget in (640,768,896,1024):
                roles=["DESCRIPTIVE_CROSSED"]
                if budget==1024: roles += ["PRIMARY","CROSSED_SECONDARY"]
                add(seed,branch,budget,1,f"{base}/seed{seed}/{branch}/network-snapshot-kimg{budget:06d}.pkl",roles)
            add(seed,branch,1024,2,f"{base}/seed{seed}/{branch}/network-snapshot-kimg001024.pkl",["DESCRIPTIVE_CROSSED"])
    payload={"schema":"ect.q128-fresh-evaluation-manifest/v1","protocol_id":protocol["protocol_id"],
             "status":"FROZEN_NOT_RUN","job_count":len(jobs),"quality_values_decoded":False,
             "decode_gate":"all jobs whose category is PRIMARY or KEY_SECONDARY must be SEALED_PASS",
             "evaluation":{"precision":"fp32","metrics":["fid50k_full","kid50k_full"],"sample_seeds":"0-49999","metric_seed":20260730,"nfe2_mid_t":0.821},
             "jobs":jobs}
    if len(jobs)!=272: raise RuntimeError(len(jobs))
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"PASS","job_count":len(jobs),"manifest":str(OUT)}))
if __name__=="__main__": main()
