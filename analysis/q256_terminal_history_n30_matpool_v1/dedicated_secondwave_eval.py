#!/usr/bin/env python3
"""Run a static q256 second-wave shard on dedicated evaluation GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


PROTOCOL_SHA256 = "317d3ef93102050276c1366d9633e322d60fbc9000cd56c8fc8a24c1d4eef544"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
EVALUATOR_CT_EVAL_SHA256 = "8e17e4cd4e12097e12659a9c8849d42554f24efb25e5255261383d952d878c95"
RUNTIME = Path("/root/q256-training-runtime-env")
RUNTIME_BASE = Path("/root/q256-training-runtime-base")
EVALUATOR = Path("/root/q256-evaluator-d6aba02")
KEY = Path("/root/.ssh/q256_hub_ed25519")
KNOWN_HOSTS = Path("/root/q256_hub_known_hosts")
METRICS = ("kid50k_full", "fid50k_full")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def opaque_id(seed: int, cell: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_SHA256}|secondwave|{seed}|{cell}".encode()).hexdigest()[:24]


def endpoint_probe_script(hub_root: str, seed: int, cell: str) -> str:
    directory = f"{hub_root}/training/seed{seed}/{cell}"
    return f"""
import hashlib,json,pathlib
d=pathlib.Path({directory!r})
cp=d/'compute_completion_receipt.json'
tp=d/'trajectory_completion_receipt.json'
sp=d/'kimg1024'/'network-snapshot.pkl'
mp=d/'checkpoint_manifest.json'
out={{'state':'WAITING'}}
if cp.is_file():
    c=json.load(open(cp))
    if c.get('status')=='FAIL' or c.get('exit_code') not in (None,0):
        out={{'state':'SCIENTIFIC_FAILURE','exit_code':c.get('exit_code'),'compute_receipt_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}}
    elif c.get('status')=='PASS' and tp.is_file() and sp.is_file() and mp.is_file():
        t=json.load(open(tp)); m=json.load(open(mp))
        expected=next((x.get('network_snapshot_sha256') for x in m.get('milestones',[]) if x.get('kimg')==1024),None)
        h=hashlib.sha256()
        with open(sp,'rb') as f:
            for b in iter(lambda:f.read(8388608),b''): h.update(b)
        actual=h.hexdigest()
        if t.get('status')=='PASS' and expected and actual==expected:
            out={{'state':'READY','checkpoint':str(sp),'checkpoint_bytes':sp.stat().st_size,'checkpoint_sha256':actual,'compute_receipt_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'trajectory_receipt_sha256':hashlib.sha256(tp.read_bytes()).hexdigest()}}
        else:
            out={{'state':'WAITING_SYNC','actual_sha256':actual,'expected_sha256':expected}}
print(json.dumps(out,sort_keys=True))
""".strip()


class Hub:
    def __init__(self, host: str, port: int, root: str, local: bool):
        self.host = host
        self.port = port
        self.root = root
        self.local = local

    def ssh_base(self, accept_new: bool = False) -> list[str]:
        return [
            "ssh", "-i", str(KEY), "-p", str(self.port), "-o", "BatchMode=yes",
            "-o", f"StrictHostKeyChecking={'accept-new' if accept_new else 'yes'}",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", f"root@{self.host}",
        ]

    def ensure_remote(self, worker_id: str) -> None:
        if self.local:
            return
        subprocess.run(
            self.ssh_base(True) + [f"mkdir -p {shlex.quote(self.root)}/evaluation_secondwave/{shlex.quote(worker_id)}"],
            check=True,
        )

    def probe(self, seed: int, cell: str) -> dict:
        script = endpoint_probe_script(self.root, seed, cell)
        if self.local:
            output = subprocess.check_output(["/root/miniconda3/bin/python", "-c", script], text=True)
        else:
            output = subprocess.check_output(self.ssh_base() + [f"/root/miniconda3/bin/python -c {shlex.quote(script)}"], text=True)
        return json.loads(output)

    def checkpoint(self, record: dict, destination: Path) -> Path:
        if self.local:
            return Path(record["checkpoint"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".pkl.partial")
        subprocess.run([
            "rsync", "-a", "--partial", "--append-verify", "-e",
            f"ssh -i {KEY} -p {self.port} -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}",
            f"root@{self.host}:{record['checkpoint']}", str(partial),
        ], check=True)
        if partial.stat().st_size != record["checkpoint_bytes"] or sha256_file(partial) != record["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint transfer mismatch: {destination}")
        os.replace(partial, destination)
        return destination

    def sync_worker(self, worker_id: str, work_root: Path) -> None:
        if self.local:
            return
        subprocess.run([
            "rsync", "-aH", "--partial", "--append-verify",
            "--exclude=inputs", "--exclude=job-caches", "-e",
            f"ssh -i {KEY} -p {self.port} -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}",
            f"{work_root}/", f"root@{self.host}:{self.root}/evaluation_secondwave/{worker_id}/",
        ], check=True)


def runtime_env(gpu: int, cache: Path) -> dict[str, str]:
    paths = [
        RUNTIME_BASE / "lib/python3.11/site-packages/torch/lib",
        RUNTIME_BASE / "lib",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
    ]
    env = os.environ.copy()
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        DNNLIB_CACHE_DIR=str(cache), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(56000 + gpu),
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
        LD_LIBRARY_PATH=":".join(map(str, paths)),
        PATH=f"{RUNTIME / 'bin'}:{RUNTIME_BASE / 'bin'}:/usr/bin:/bin",
    )
    return env


def metric_value(path: Path, metric: str) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"metric row count mismatch: {path}")
    value = float(rows[0]["results"][metric])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric: {path}")
    return value


def validate_job(job: dict, output: Path, elapsed: float, gpu: int) -> dict:
    required = [
        "log.txt", "training_options.json", "generated-samples.npy",
        "generated-features-kid50k_full-repeat00.npy",
        "generated-features-fid50k_full-repeat00.npy",
        "metric-kid50k_full.jsonl", "metric-fid50k_full.jsonl",
    ]
    artifacts = {}
    for name in required:
        path = output / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing evaluation artifact: {path}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if "Exiting..." not in (output / "log.txt").read_text(errors="replace"):
        raise RuntimeError("evaluation log lacks completion marker")
    kid = output / "generated-features-kid50k_full-repeat00.npy"
    fid = output / "generated-features-fid50k_full-repeat00.npy"
    if artifacts[kid.name]["sha256"] != artifacts[fid.name]["sha256"]:
        raise RuntimeError("KID/FID generated features differ")
    if kid.stat().st_ino != fid.stat().st_ino:
        temporary = output / ".shared-features.tmp"
        os.link(kid, temporary); os.replace(temporary, fid)
    values = {metric: metric_value(output / f"metric-{metric}.jsonl", metric) for metric in METRICS}
    options = load(output / "training_options.json")
    if options.get("sample_seeds") != list(range(50000)) or options.get("seed") != 20260730 or options.get("metrics") != list(METRICS) or options.get("mid_t") != []:
        raise RuntimeError("evaluation option contract mismatch")
    return {
        "schema": "ect.q256.terminal-history-secondwave-evaluation-job/v1",
        "status": "PASS", "seed": job["seed"], "cell": job["cell"],
        "opaque_id": job["opaque_id"], "gpu_index": gpu,
        "checkpoint_sha256": job["checkpoint_sha256"], "elapsed_seconds": elapsed,
        "artifact_hashes": artifacts, "generated_feature_sha256": artifacts[kid.name]["sha256"],
        "kid_fid_shared_features": True, "values": values,
        "protocol_sha256": PROTOCOL_SHA256, "automatic_retry_count": 0,
        "completed_at": utc_now(),
    }


def run_job(asset_root: Path, work_root: Path, job: dict, checkpoint: Path, gpu: int) -> dict:
    output = work_root / "jobs" / job["opaque_id"]
    receipt_path = work_root / "receipts" / f"{job['opaque_id']}.json"
    output.mkdir()
    cache = work_root / "job-caches" / job["opaque_id"]
    shutil.copytree(asset_root / "cache-template", cache, copy_function=os.link)
    command = [
        str(RUNTIME / "bin/python"), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        f"--master_port={56000 + gpu}", str(EVALUATOR / "ct_eval.py"),
        "--resume", str(checkpoint), "--outdir", str(output), "--nosubdir",
        "--data", str(asset_root / "cifar10-32x32-eval.zip"), "--cond=False",
        "--arch=ddpmpp", "--precond=ct", "--dropout=0.2", "--augment=0",
        "--xflip=False", "--fp16=False", "--cache=True", "--workers=1",
        "--eval-batch=512", "--metric-generator-batch=128", "--nfe=1",
        "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
        "--sample-seeds=0-49999", "--seed=20260730", "--retain-generated-artifacts",
        f"--desc=blind-{job['opaque_id']}",
    ]
    launcher = work_root / "logs" / f"{job['opaque_id']}.launcher.log"
    started = time.monotonic()
    with launcher.open("xb") as log:
        process = subprocess.Popen(command, cwd=EVALUATOR, env=runtime_env(gpu, cache), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=4 * 3600)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); code = process.wait()
    if code != 0 or timed_out:
        failure = {
            "schema": "ect.q256.terminal-history-secondwave-evaluation-job/v1",
            "status": "FAIL", "seed": job["seed"], "cell": job["cell"],
            "opaque_id": job["opaque_id"], "gpu_index": gpu, "exit_code": code,
            "hard_timeout": timed_out, "automatic_retry_count": 0,
            "protocol_sha256": PROTOCOL_SHA256, "completed_at": utc_now(),
        }
        atomic_json(output / "failure_receipt.json", failure)
        return failure
    try:
        receipt = validate_job(job, output, time.monotonic() - started, gpu)
        atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema": "ect.q256.terminal-history-secondwave-evaluation-job/v1",
            "status": "FAIL_POSTCHECK", "seed": job["seed"], "cell": job["cell"],
            "opaque_id": job["opaque_id"], "gpu_index": gpu,
            "error": repr(error), "automatic_retry_count": 0,
            "protocol_sha256": PROTOCOL_SHA256, "completed_at": utc_now(),
        }
        atomic_json(output / "postcheck_failure_receipt.json", failure)
        return failure


def worker_loop(gpu: int, assignments: list[list], asset_root: Path, work_root: Path, hub: Hub, worker_id: str, failures: list[dict], lock: threading.Lock) -> None:
    remaining = [(int(seed), str(cell)) for seed, cell in assignments]
    while remaining:
        made_progress = False
        for seed, cell in list(remaining):
            opaque = opaque_id(seed, cell)
            receipt_path = work_root / "receipts" / f"{opaque}.json"
            science_path = work_root / "scientific-failures" / f"seed{seed}-{cell}.json"
            if receipt_path.exists() or science_path.exists():
                remaining.remove((seed, cell)); made_progress = True; continue
            record = hub.probe(seed, cell)
            if record["state"] == "SCIENTIFIC_FAILURE":
                atomic_json(science_path, {
                    "schema": "ect.q256.terminal-history-secondwave-scientific-failure/v1",
                    "status": "SCIENTIFIC_FAILURE", "seed": seed, "cell": cell,
                    "automatic_retry_count": 0, "protocol_sha256": PROTOCOL_SHA256,
                    **record, "recorded_at": utc_now(),
                })
                hub.sync_worker(worker_id, work_root)
                remaining.remove((seed, cell)); made_progress = True; continue
            if record["state"] != "READY":
                continue
            job = {"seed": seed, "cell": cell, "opaque_id": opaque, **record}
            binding = work_root / "input-bindings" / f"{opaque}.json"
            if not binding.exists():
                atomic_json(binding, {
                    "schema": "ect.q256.terminal-history-secondwave-input-binding/v1",
                    "status": "PASS", **job, "protocol_sha256": PROTOCOL_SHA256,
                    "bound_at": utc_now(),
                })
            checkpoint = hub.checkpoint(record, work_root / "inputs" / f"seed{seed}-{cell}.pkl")
            result = run_job(asset_root, work_root, job, checkpoint, gpu)
            hub.sync_worker(worker_id, work_root)
            if result["status"] != "PASS":
                with lock:
                    failures.append(result)
                return
            remaining.remove((seed, cell)); made_progress = True
            break
        if not made_progress:
            time.sleep(30)


def decode(work_root: Path, worker_id: str) -> None:
    rows = []
    for receipt_path in sorted((work_root / "receipts").glob("*.json")):
        receipt = load(receipt_path)
        if receipt.get("status") != "PASS":
            continue
        rows.append({
            "seed": receipt["seed"], "cell": receipt["cell"], "budget_kimg": 1024,
            "nfe": 1, "kid50k_full": receipt["values"]["kid50k_full"],
            "fid50k_full": receipt["values"]["fid50k_full"],
            "opaque_id": receipt["opaque_id"], "checkpoint_sha256": receipt["checkpoint_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        })
    rows.sort(key=lambda row: (row["seed"], row["cell"]))
    csv_path = work_root / "decoded_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["seed", "cell"])
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    atomic_json(work_root / "control" / "evaluation_seal.json", {
        "schema": "ect.q256.terminal-history-secondwave-worker-seal/v1",
        "status": "SEALED_PASS", "worker_id": worker_id, "evaluated_jobs": len(rows),
        "scientific_failures": len(list((work_root / "scientific-failures").glob("*.json"))),
        "protocol_sha256": PROTOCOL_SHA256, "sealed_at": utc_now(),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--hub-host", required=True)
    parser.add_argument("--hub-port", type=int, required=True)
    parser.add_argument("--hub-root", default="/root/q256-n30-central-store-v1")
    parser.add_argument("--local-hub", action="store_true")
    args = parser.parse_args()

    asset_root = Path(args.asset_root)
    if sha256_file(asset_root / "cifar10-32x32-eval.zip") != DATASET_SHA256:
        raise RuntimeError("evaluation dataset SHA mismatch")
    if sha256_file(EVALUATOR / "ct_eval.py") != EVALUATOR_CT_EVAL_SHA256:
        raise RuntimeError("ct_eval.py SHA mismatch")
    assignments = load(Path(args.assignments))
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True).splitlines()
    gpu_count = len(gpu_rows)
    expected_keys = {str(index) for index in range(gpu_count)}
    if set(assignments) != expected_keys:
        raise RuntimeError(f"assignment keys {set(assignments)} != {expected_keys}")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if apps:
        raise RuntimeError("dedicated evaluation launch requires idle GPUs")

    hub = Hub(args.hub_host, args.hub_port, args.hub_root, args.local_hub)
    hub.ensure_remote(args.worker_id)
    work_root = Path(args.hub_root) / "evaluation_secondwave" / args.worker_id if args.local_hub else Path("/root/q256-n30-secondwave-eval-v1") / args.worker_id
    for name in ("inputs", "jobs", "receipts", "logs", "job-caches", "input-bindings", "scientific-failures", "control"):
        (work_root / name).mkdir(parents=True, exist_ok=True)
    atomic_json(work_root / "control" / "launch_receipt.json", {
        "schema": "ect.q256.terminal-history-secondwave-dedicated-worker/v1",
        "status": "RUNNING", "worker_id": args.worker_id, "gpu_count": gpu_count,
        "assignment_count": sum(map(len, assignments.values())),
        "protocol_sha256": PROTOCOL_SHA256, "launched_at": utc_now(),
    })
    hub.sync_worker(args.worker_id, work_root)

    failures: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker_loop,
            args=(gpu, assignments[str(gpu)], asset_root, work_root, hub, args.worker_id, failures, lock),
            daemon=False,
        )
        for gpu in range(gpu_count)
    ]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        atomic_json(work_root / "control" / "evaluation_failure_summary.json", {
            "schema": "ect.q256.terminal-history-secondwave-evaluation-failure/v1",
            "status": "FAIL", "worker_id": args.worker_id, "failures": failures,
            "automatic_retry_count": 0, "protocol_sha256": PROTOCOL_SHA256,
        })
        hub.sync_worker(args.worker_id, work_root)
        raise RuntimeError("evaluation failed closed; no automatic retry")
    decode(work_root, args.worker_id)
    hub.sync_worker(args.worker_id, work_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
