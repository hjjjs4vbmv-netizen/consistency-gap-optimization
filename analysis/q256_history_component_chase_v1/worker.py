"""One exclusive GPU, fixed seed order, resumable stages and shared cost accounting.

All nodes must use the same budget ledger on shared persistent storage. Node-local
data/runtime/output paths live only in the deployment JSON, outside the repository.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
import torch
from scripts import run_m1_training_slot as old_worker
from training import history_component as hc, schedule_switch
from . import protocol as p


def write(path, value):
    old_worker.write_json(Path(path), value)


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def budget_admit(path, key):
    """Reserve a stage using process time, never inherited checkpoint elapsed/FID."""
    path = Path(path)
    with locked(str(path)+".lock"):
        data = json.loads(path.read_text()) if path.exists() else {"cap_gpuh":200, "scoped_seeds":list(p.SEEDS), "evaluation_reserve_gpuh":8,
            "engineering_transfer_reserve_gpuh":8, "stages":{}, "status":"RUNNING"}
        scope = data.get("scoped_seeds", list(p.SEEDS))
        if int(key.split('/')[0][1:])+49 not in scope:
            raise RuntimeError("stage is outside this node's allocated budget scope")
        stage = data["stages"].get(key, {})
        if stage.get("status") == "RUNNING":
            raise RuntimeError("stage already leased; reconcile its process before recovery")
        if stage.get("status") in {"PASS", "SCIENTIFIC_FAILURE", "NO_ENDPOINT"}:
            return False
        done = [a for s in data["stages"].values() for a in s.get("attempts", [])]
        spent = sum(a["process_gpuh"] for a in done)
        samples = [a for a in done if a.get("processed_attempts", 0) > 0]
        rate = (sum(a["process_gpuh"] for a in samples)/sum(a["processed_attempts"] for a in samples)
                if samples else 2.72/4000)
        remaining = 0
        for row in p.training_queue():
            if row['seed'] not in scope:
                continue
            for phase in ("prefix", "suffix"):
                record = data["stages"].get(f"{row['slot']}/{row['path']}/{phase}", {})
                if record.get("status") not in {"PASS", "SCIENTIFIC_FAILURE", "NO_ENDPOINT"}:
                    remaining += 4000-record.get("completed_attempts", 0)
        active_extra = sum(max(0, time.time()-s["started_unix"])/3600 for s in data["stages"].values() if s.get("status")=="RUNNING")
        projected = spent + active_extra + remaining*rate + data["evaluation_reserve_gpuh"] + data["engineering_transfer_reserve_gpuh"]
        data["forecast_gpuh"] = projected
        if projected > data["cap_gpuh"] or data["status"] == "INCOMPLETE_BUDGET":
            data["status"] = "INCOMPLETE_BUDGET"
            write(path, data)
            return False
        data["stages"][key] = {**stage, "status":"RUNNING", "host":socket.gethostname(),
                                "pid":os.getpid(), "started_unix":time.time()}
        write(path, data)
        return True


def budget_finish(path, key, record):
    path = Path(path)
    with locked(str(path)+".lock"):
        data = json.loads(path.read_text())
        stage = data["stages"][key]
        stage.setdefault("attempts", []).append(record)
        stage.update(status=record["status"], completed_attempts=record["end_attempt"]-(0 if key.endswith("prefix") else 4000))
        if key.endswith("prefix") and record["status"] == "SCIENTIFIC_FAILURE":
            data["stages"][key.removesuffix("prefix")+"suffix"] = {"status":"NO_ENDPOINT"}
        write(path, data)


def checkpoint(run_dir, phase, manifest):
    target = 4000 if phase == "prefix" else 8000
    candidate = [run_dir/"training-state-latest.pt"] + sorted(run_dir.glob("training-state-kimg*.pt"), reverse=True)
    for path in candidate:
        if not path.is_file():
            continue
        # Atomic latest is authoritative; do not silently fall back from a corrupt state.
        state = torch.load(path, map_location="cpu", weights_only=False)
        attempt = state["attempted_iteration"]
        if state["cur_nimg"] != attempt*128 or not (0 <= attempt <= target):
            raise RuntimeError("checkpoint progress mismatch")
        if phase == "suffix":
            hc.validate_resumed_state(state, manifest)
        else:
            if state.get("history_component_prefix", {}).get("protocol_id") != p.EXPERIMENT_ID:
                raise RuntimeError("resume is not a formal new prefix")
            if state["factorial"]["arm"] != manifest["origin_arm"] or state["trajectory_config"]["seed"] != manifest["seed"]:
                raise RuntimeError("prefix resume identity mismatch")
        del state
        return path, attempt
    return None, 0 if phase == "prefix" else 4000


def paired_boundary(source, old_root, seed):
    state = torch.load(source, map_location="cpu", weights_only=False)
    stream = state["rank_states"]
    del state
    for arm in ("A", "B"):
        directory = Path(old_root)/f"seed{seed}"/f"prefix_{arm}"
        stream_path = directory/"rank_streams.pt"
        old = torch.load(stream_path if stream_path.exists() else directory/"training-state-kimg000512.pt", map_location="cpu", weights_only=False)
        if not old_worker.equal_state(stream, old["rank_states"]):
            raise RuntimeError(f"512 RNG/sampler mismatch with old {arm}; never overwrite streams")
        del old


def compare_telemetry(current, reference):
    fields = ("batch_sha256", "labels_sha256", "t_sha256", "base_r_sha256")
    with Path(current).open() as a, Path(reference).open() as b:
        new, old = list(csv.DictReader(a)), list(csv.DictReader(b))
    old = {int(r["attempted_iteration"]):r for r in old}
    checked = [f for f in fields if new and f in new[0] and old and f in next(iter(old.values()))]
    if not {"t_sha256", "base_r_sha256"} <= set(checked):
        raise RuntimeError("required base-random telemetry is absent")
    for row in new:
        previous = old[int(row["attempted_iteration"])]
        if any(row[f] != previous[f] for f in checked):
            raise RuntimeError(f"base random input mismatch at {row['attempted_iteration']}")
    return {"attempts_compared":len(new), "fields":checked,
            "scope":"existing telemetry only; unrecorded epsilon/dropout not asserted"}


def run_stage(config, row, phase, gpu):
    seed, branch = row["seed"], row["path"]
    root = Path(config["output_root"])/row["slot"]/branch
    run_dir = root/phase
    run_dir.mkdir(parents=True, exist_ok=True)
    source = root/"prefix"/"training-state-kimg000512.pt"
    manifest = p.manifest(seed, branch, source, root/"suffix")
    status_path = run_dir/"stage_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["status"] in {"PASS", "SCIENTIFIC_FAILURE", "NO_ENDPOINT"}:
            return previous
    manifest_path = root/"suffix"/"formal_run_manifest.json"
    if phase == "suffix":
        prior = json.loads((root/"prefix"/"stage_status.json").read_text())
        if prior["status"] == "SCIENTIFIC_FAILURE":
            record = {"status":"NO_ENDPOINT", "reason":"own prefix scientific failure"}
            write(status_path, record)
            return record
        if prior["status"] != "PASS":
            raise RuntimeError("own prefix is not complete")
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("suffix manifest changed")
        write(manifest_path, manifest)
        schedule_switch.load_run_manifest(manifest_path)
    resume, start = checkpoint(run_dir, phase, manifest)
    if start == (4000 if phase == "prefix" else 8000):
        raise RuntimeError("completed state without finalized receipt: reconcile before dispatch")
    if phase == "suffix" and resume is None:
        paired_boundary(source, config["old_prefix_root"], seed)
        resume = source
    if any(run_dir.glob("train-attempt-*.log")) and resume is None:
        raise RuntimeError("technical interruption has no complete state; fresh rerun is not automatic")
    if resume and resume != source:
        # Preserve original logs before aligning CSV tail to the last committed state.
        for name in ("train_summary.csv", "factorial_training_telemetry_v1.csv", "schedule_switch_training_telemetry_v1.csv"):
            path = run_dir/name
            if path.exists():
                shutil.copy2(path, path.with_name(path.name+f".before-recovery-{time.time_ns()}"))
                old_worker.truncate_attempt_csv(path, start)
    cmd = p.command(python=config["runtime_python"], dataset=config["dataset"], transfer=config["transfer"],
                    output=run_dir, seed=seed, history=row["history"], phase=phase,
                    resume=resume, switch_manifest=manifest_path if phase=="suffix" else None)
    key = f"{row['slot']}/{branch}/{phase}"
    if not budget_admit(config["budget_ledger"], key):
        record = {"status":"BUDGET_PAUSED", "command":cmd}
        write(status_path, record)
        return record
    log = run_dir/f"train-attempt-{len(list(run_dir.glob('train-attempt-*.log')))+1:02d}.log"
    env = old_worker.runtime_environment(gpu, Path(config["runtime_python"]))
    env["ECT_HISTORY_OLD_PREFIX_ROOT"] = config["old_prefix_root"]
    started = time.monotonic()
    record = {"command":cmd, "status":"RUNNING", "started_utc":old_worker.utc_now(),
              "start_attempt":start, "gpu":gpu, "host":socket.gethostname(), "log":str(log)}
    write(status_path, record)
    with log.open("xb") as handle:
        process = subprocess.Popen(cmd, cwd=p.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        write(status_path, record)
        # Process-liveness checks; no FID/loss early stop and no inherited elapsed timer.
        while process.poll() is None:
            time.sleep(5)
    record.update(exit_code=process.returncode, ended_utc=old_worker.utc_now(),
                  process_gpuh=(time.monotonic()-started)/3600, end_attempt=start, processed_attempts=0)
    record["status"] = "TECHNICAL_FAILURE"
    try:
        if process.returncode != 0:
            if old_worker.scientific_failure(log):
                record["status"] = "SCIENTIFIC_FAILURE"
            latest, end = checkpoint(run_dir, phase, manifest)
            record.update(end_attempt=end, processed_attempts=max(0, end-start))
        else:
            end = 4000 if phase == "prefix" else 8000
            terminal = run_dir/f"training-state-kimg{end*128//1000:06d}.pt"
            state = torch.load(terminal, map_location="cpu", weights_only=False)
            if phase == "prefix":
                schedule_switch.verify_source_state(state, manifest)
            else:
                hc.validate_terminal_state(state, manifest)
                init = torch.load(run_dir/"training-state-kimg000512.pt", map_location="cpu", weights_only=False)
                original = torch.load(source, map_location="cpu", weights_only=False)
                hc.validate_branch_init_against_source(init, original, manifest)
                del init, original
                first = next(csv.DictReader((run_dir/"schedule_switch_training_telemetry_v1.csv").open()))
                if int(first["attempted_iteration"]) != 4001 or first["arm"] != "A" or any(float(first[k]) != 1.0 for k in ("target_gap_scale", "denominator_gap_scale")):
                    raise RuntimeError("attempt4001 did not execute A")
            del state
            if phase == "prefix":
                checks = [compare_telemetry(run_dir/"factorial_training_telemetry_v1.csv",
                    Path(config["old_prefix_root"])/f"seed{seed}"/f"prefix_{arm}"/"factorial_training_telemetry_v1.csv") for arm in ("A", "B")]
                write(run_dir/"base_random_pairing.json", {"status":"PASS", "comparisons":checks})
            record.update(status="PASS", end_attempt=end, processed_attempts=end-start, endpoint=str(terminal))
    except Exception as exc:
        record["postcheck_error"] = repr(exc)
        if record["status"] != "SCIENTIFIC_FAILURE":
            record["status"] = "TECHNICAL_FAILURE"
    finally:
        write(status_path, record)
        write(log.with_suffix(".json"), record)
        budget_finish(config["budget_ledger"], key, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=p.SEEDS)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text())
    rows = [r for r in p.training_queue() if r["seed"] == args.seed]
    if args.dry_run:
        for row in rows:
            root = Path(config["output_root"])/row["slot"]/row["path"]
            for phase in ("prefix", "suffix"):
                print(json.dumps(p.command(python=config["runtime_python"], dataset=config["dataset"], transfer=config["transfer"],
                      output=root/phase, seed=args.seed, history=row["history"], phase=phase,
                      resume=root/"prefix"/"training-state-kimg000512.pt" if phase=="suffix" else None,
                      switch_manifest=root/"suffix"/"formal_run_manifest.json" if phase=="suffix" else None)))
        return
    preflight = json.loads(Path(config["preflight_receipt"]).read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p.ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=p.ROOT, text=True)
    if preflight.get("status") != "PASS" or preflight.get("implementation_commit") != head or dirty:
        raise RuntimeError("formal dispatch requires passed node preflight and its clean implementation")
    if preflight.get("deployment") != config:
        raise RuntimeError("deployment changed after preflight")
    from scripts.run_m1_evaluation_job import gpu_resource_probe
    gpu = gpu_resource_probe(args.gpu)
    with locked(f"/tmp/ect-history-component-{gpu['uuid']}.lock"):
        for row in rows:
            for phase in ("prefix", "suffix"):
                result = run_stage(config, row, phase, args.gpu)
                print(json.dumps({"seed":args.seed, "path":row["path"], "phase":phase, "status":result["status"]}), flush=True)
                if result["status"] in {"TECHNICAL_FAILURE", "BUDGET_PAUSED"}:
                    return


if __name__ == "__main__":
    main()
