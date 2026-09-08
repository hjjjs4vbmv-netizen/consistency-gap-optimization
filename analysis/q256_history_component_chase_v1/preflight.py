"""Node-local asset/configuration checks; GPU implementation checks are separate."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import scipy
import torch
from training import history_component as hc, schedule_switch
from . import protocol as p
from .worker import write


def runtime():
    return {"python":platform.python_version(), "torch":torch.__version__, "torch_cuda":torch.version.cuda,
            "numpy":np.__version__, "scipy":scipy.__version__, "cudnn":torch.backends.cudnn.version()}


def check(config, *, engineering=None):
    actual = runtime()
    mismatches = {k:{"expected":v,"actual":actual.get(k)} for k,v in p.EXPECTED_RUNTIME.items() if actual.get(k)!=v}
    if mismatches:
        raise RuntimeError("runtime differs: "+json.dumps(mismatches))
    baseline = json.loads((p.ROOT/"analysis/q256_terminal_history_n30_matpool_v1/protocol.json").read_text())
    assets = {}
    for name in ("dataset","transfer"):
        path = Path(config[name]).resolve(strict=True)
        digest = schedule_switch.sha256_file(path)
        if digest != baseline["assets"][name]["sha256"]:
            raise RuntimeError(f"{name} differs from the original PR101 asset")
        assets[name] = {"path":str(path),"sha256":digest}
    configs = []
    for seed in p.SEEDS:
        initials = []
        for arm in ("A","B"):
            directory = Path(config["old_prefix_root"])/f"seed{seed}"/f"prefix_{arm}"
            for filename in ("initial_state_receipt_v1.json","factorial_training_telemetry_v1.csv"):
                path = directory/filename
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f"required old source asset is missing: {path}")
            if seed in config.get("assigned_seeds",p.SEEDS) or (seed==50 and arm=="A"):
                if not (directory/"training-state-kimg000512.pt").is_file() and not (directory/"rank_streams.pt").is_file():
                    raise RuntimeError(f"required old 512 stream state is missing: {directory}")
            initials.append(json.loads((directory/"initial_state_receipt_v1.json").read_text()))
        hc.validate_initial_receipt(initials[0],initials[1])
        configs.append({"seed":seed,"old_initial_pair":"PASS"})
    for row in p.import_old_controls():
        if row["status"]=="PASS" and not (Path(config["old_m1_root"])/row["receipt"]).is_file():
            raise RuntimeError(f"old original K receipt unavailable: {row['job_id']}")
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=p.ROOT,text=True).strip()
    result = {"status":"ASSETS_READY_GPU_CHECKS_PENDING", "implementation_commit":head,
              "runtime":actual,"assets":assets,"old_prefix_configs":configs,"deployment":config,
              "old_K_valid_slots":145,"old_K_missing_slots":15}
    # This command validates the node and requires an actual engineering receipt.
    if engineering:
        report = json.loads(Path(engineering).read_text())
        if report.get("status")!="PASS" or report.get("implementation_commit")!=head:
            raise RuntimeError("short engineering checks did not pass on this implementation")
        result.update(status="PASS",short_checks=report)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment",type=Path,required=True)
    parser.add_argument("--engineering-receipt",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text())
    write(args.output,check(config,engineering=args.engineering_receipt))
