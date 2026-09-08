"""Small GPU-only implementation checks; isolated engineering outputs, no FID."""
from __future__ import annotations
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path
import torch
from training import history_component as hc, m1, schedule_switch
from scripts.run_m1_training_slot import runtime_environment
from . import protocol as p
from .worker import write, compare_telemetry

EXACT_KEYS = ("optimizer_state", "gradscaler_state", "rank_states", "loss_fn_state",
              "attempted_iteration", "successful_optimizer_steps", "cur_nimg", "cur_tick", "tick_start_nimg")


def compare_states(left, right, shadow=False):
    a = torch.load(left, map_location="cpu", weights_only=False)
    b = torch.load(right, map_location="cpu", weights_only=False)
    keys = ["net", "ema"] + (["ema_512"] if shadow else [])
    for key in keys:
        if not m1._equal_state(a[key].state_dict(),b[key].state_dict()):
            raise RuntimeError(f"short check differs in {key}")
    for key in EXACT_KEYS:
        if not m1._equal_state(a[key],b[key]):
            raise RuntimeError(f"short check differs in {key}")
    if shadow and a["history_component"] != b["history_component"]:
        # Source path is identical; metadata must also survive resume unchanged.
        raise RuntimeError("short check E_512/optimizer metadata differs")
    return {"status":"PASS", "state_fields":keys+list(EXACT_KEYS), "exact":True}


def execute(config, output, gpu):
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError("engineering output already exists; preserve prior attempt")
    output.mkdir(parents=True)
    started = time.monotonic()
    records = []
    env = runtime_environment(gpu,Path(config["runtime_python"]))
    def run(name, history, stop, resume=None, suffix=False, baseline=False):
        directory = output/name
        directory.mkdir(exist_ok=True)
        manifest_path = directory/"engineering_manifest.json"
        if suffix:
            original = Path(config["old_prefix_root"])/"seed50/prefix_A/training-state-kimg000512.pt"
            value = p.manifest(50,"AA",original,directory,engineering=True)
            write(manifest_path,value)
            resume = resume or original
        cmd = p.command(python=config["runtime_python"],dataset=config["dataset"],transfer=config["transfer"],
             output=directory,seed=50,history=history,phase="suffix" if suffix else "prefix",
             resume=resume,switch_manifest=manifest_path if suffix else None,engineering_stop=stop)
        if baseline:
            cmd = [item for item in cmd if not item.startswith("--planned-pause-protocol=")]
        log = directory/f"attempt-{stop}.log"
        t = time.monotonic()
        with log.open("xb") as handle:
            result = subprocess.run(cmd,cwd=p.ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        records.append({"name":name,"command":cmd,"exit_code":result.returncode,"process_gpuh":(time.monotonic()-t)/3600})
        write(output/"processes.json",records)
        if result.returncode:
            raise RuntimeError(f"engineering process failed: {log}")
        return directory/"training-state-latest.pt"
    report = {"status":"RUNNING", "checks":{}, "scope":"short engineering checks only, not full-run bitwise reproduction"}
    try:
        baseline = run("baseline_A16","A",16,baseline=True)
        a = run("wrapper_A16","A",16)
        split = run("wrapper_A_split","A",8)
        split = run("wrapper_A_split","A",16,resume=split)
        report["checks"]["A_no_intervention"] = compare_states(baseline,a)
        report["checks"]["prefix_resume"] = compare_states(a,split)
        for arm in ("A","C","D"):
            path = a if arm=="A" else run(f"wrapper_{arm}16",arm,16)
            current = json.loads((path.parent/"initial_state_receipt_v1.json").read_text())
            for old in ("A","B"):
                reference = json.loads((Path(config["old_prefix_root"])/f"seed50/prefix_{old}/initial_state_receipt_v1.json").read_text())
                hc.validate_initial_receipt(current,reference)
            state = torch.load(path,map_location="cpu",weights_only=False)
            if (state["factorial"]["target_gap_scale"],state["factorial"]["denominator_gap_scale"]) != p.FACTORS[arm]:
                raise RuntimeError("actual short prefix factor mismatch")
            del state
            dtype = json.loads((path.parent/"first_actual_forward.json").read_text())
            if dtype["input_dtype"] != "torch.float16" or dtype["training"] is not True:
                raise RuntimeError("actual denoiser training input is not FP16")
            report["checks"][arm+"_dtype"] = dtype
            report["checks"][arm+"_base_inputs"] = compare_telemetry(path.parent/"factorial_training_telemetry_v1.csv",
                baseline.parent/"factorial_training_telemetry_v1.csv")
        continuous = run("suffix_continuous","A",4016,suffix=True)
        split = run("suffix_split","A",4008,suffix=True)
        split = run("suffix_split","A",4016,resume=split,suffix=True)
        report["checks"]["suffix_resume"] = compare_states(continuous,split,shadow=True)
        for directory in (continuous.parent,split.parent):
            manifest = json.loads((directory/"engineering_manifest.json").read_text())
            source = torch.load(manifest["source_state"]["path"],map_location="cpu",weights_only=False)
            init = torch.load(directory/"training-state-kimg000512.pt",map_location="cpu",weights_only=False)
            hc.validate_branch_init_against_source(init,source,manifest)
            del source,init
        report["checks"]["branch_init_once_keep"] = {"status":"PASS"}
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        report["process_gpuh"] = sum(row["process_gpuh"] for row in records)
        report["wall_hours"] = (time.monotonic()-started)/3600
        report["implementation_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=p.ROOT,text=True).strip()
        write(output/"short_checks.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args = parser.parse_args()
    execute(json.loads(args.deployment.read_text()),args.output,args.gpu)
