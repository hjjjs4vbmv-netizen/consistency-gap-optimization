#!/usr/bin/env python3
"""Run and return the frozen first-wave q256 NFE1 KID50k/FID50k evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


PROTOCOL_SHA256 = "317d3ef93102050276c1366d9633e322d60fbc9000cd56c8fc8a24c1d4eef544"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
EVALUATOR_ARCHIVE_SHA256 = "7ef8a1b22af9beab106ad3adbac6474608f27e74c43629a95fcc71738dab0a6f"
EVALUATOR_CT_EVAL_SHA256 = "8e17e4cd4e12097e12659a9c8849d42554f24efb25e5255261383d952d878c95"
WORK_ROOT = Path("/root/q256-n30-firstwave-eval-v3")
SOURCE_ROOT = Path("/root/q256-terminal-history-n30-v1")
DATASET = Path("/mnt/ect_project/datasets/cifar10-32x32.zip")
EVALUATOR = Path("/root/q256-evaluator-d6aba02")
TRAIN_RUNTIME_BASE = Path("/root/q256-training-runtime-base")
TRAIN_RUNTIME_ENV = Path("/root/q256-training-runtime-env")
RUNTIME_RECEIPT = Path("/root/q256-training-runtime-transfer-receipt.json")
TRANSFER_KEY = Path("/root/q256_eval_transfer_ed25519")
KNOWN_HOSTS = Path("/root/q256_eval_known_hosts")
METRICS = ("kid50k_full", "fid50k_full")
MISSING = {(67, "AA"), (68, "AA")}
NODE_CONFIG = {
    "node8": {
        "gpu_count": 8,
        "source_host": "px-cloud2.matpool.com",
        "source_port": 28062,
        "seeds": tuple(range(50, 58)),
        "return_label": "eval-node8-firstwave",
    },
    "node6": {
        "gpu_count": 6,
        "source_host": "px-cloud2.matpool.com",
        "source_port": 28798,
        "seeds": tuple(range(66, 73)),
        "return_label": "eval-node6-firstwave",
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    os.unlink(temporary)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def ssh_base(config: dict) -> list[str]:
    return [
        "ssh", "-i", str(TRANSFER_KEY), "-p", str(config["source_port"]),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        f"root@{config['source_host']}",
    ]


def source_path(seed: int, cell: str) -> Path:
    return SOURCE_ROOT / "training" / f"seed{seed}" / cell


def expected_cells(node_id: str) -> list[tuple[int, str]]:
    result = []
    for seed in NODE_CONFIG[node_id]["seeds"]:
        for cell in ("AA", "BA"):
            if (seed, cell) not in MISSING:
                result.append((seed, cell))
    expected = 16 if node_id == "node8" else 12
    if len(result) != expected:
        raise AssertionError("first-wave cell count mismatch")
    return result


def remote_probe(config: dict, seed: int, cell: str) -> dict | None:
    cell_dir = source_path(seed, cell)
    script = (
        "import hashlib,json,pathlib,sys;"
        f"d=pathlib.Path({str(cell_dir)!r});"
        "cp=d/'compute_completion_receipt.json';"
        "tp=d/'trajectory_completion_receipt.json';"
        "sp=d/'kimg1024'/'network-snapshot.pkl';"
        "ok=cp.is_file() and tp.is_file() and sp.is_file();"
        "x=json.load(open(cp)) if cp.is_file() else {};"
        "y=json.load(open(tp)) if tp.is_file() else {};"
        "ok=ok and x.get('status')=='PASS' and x.get('exit_code')==0 and y.get('status')=='PASS';"
        "h=hashlib.sha256();"
        "[(h.update(b)) for b in iter(lambda:open(sp,'rb').read(0),b'')] if False else None;"
        "\nif not ok: raise SystemExit(3)\n"
        "with open(sp,'rb') as f:\n"
        "  for b in iter(lambda:f.read(8388608),b''): h.update(b)\n"
        "print(json.dumps({'path':str(sp),'bytes':sp.stat().st_size,'sha256':h.hexdigest(),"
        "'compute_receipt_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),"
        "'trajectory_receipt_sha256':hashlib.sha256(tp.read_bytes()).hexdigest()}))"
    )
    result = subprocess.run(
        ssh_base(config) + [f"python3 -c {shlex.quote(script)}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode == 3:
        return None
    if result.returncode != 0:
        raise RuntimeError(f"remote probe failed seed{seed}/{cell}: {result.stderr}")
    return json.loads(result.stdout)


def pull_file(config: dict, record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    remote = f"root@{config['source_host']}:{record['path']}"
    command = [
        "rsync", "-a", "--partial", "--append-verify",
        "-e",
        (
            f"ssh -i {TRANSFER_KEY} -p {config['source_port']} -o BatchMode=yes "
            f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}"
        ),
        remote, str(partial),
    ]
    subprocess.run(command, check=True)
    if partial.stat().st_size != int(record["bytes"]) or sha256_file(partial) != record["sha256"]:
        raise RuntimeError(f"transferred checkpoint hash mismatch: {destination}")
    os.replace(partial, destination)


def validate_environment(node_id: str) -> None:
    config = NODE_CONFIG[node_id]
    if sha256_file(DATASET) != DATASET_SHA256:
        raise RuntimeError("canonical evaluation dataset SHA mismatch")
    if sha256_file(Path("/root/q256-evaluator-d6aba02.tar.gz")) != EVALUATOR_ARCHIVE_SHA256:
        raise RuntimeError("evaluator archive SHA mismatch")
    if sha256_file(EVALUATOR / "ct_eval.py") != EVALUATOR_CT_EVAL_SHA256:
        raise RuntimeError("evaluator ct_eval.py SHA mismatch")
    runtime_receipt = load(RUNTIME_RECEIPT)
    if runtime_receipt.get("status") != "PASS":
        raise RuntimeError("training-compatible runtime receipt is not PASS")
    archive = Path(runtime_receipt["archive_path"])
    if sha256_file(archive) != runtime_receipt.get("archive_sha256"):
        raise RuntimeError("training-compatible runtime archive changed")
    python = TRAIN_RUNTIME_ENV / "bin" / "python"
    if not python.is_file() or not TRANSFER_KEY.is_file() or not KNOWN_HOSTS.is_file():
        raise RuntimeError("runtime or transfer identity missing")
    probe = json.loads(subprocess.check_output([
        str(python), "-c",
        (
            "import json,platform,numpy,scipy,torch;"
            "print(json.dumps({'python':platform.python_version(),"
            "'numpy':numpy.__version__,'scipy':scipy.__version__,"
            "'torch':torch.__version__,'torch_cuda':torch.version.cuda},sort_keys=True))"
        ),
    ], text=True))
    expected_probe = {
        "python": "3.11.13", "numpy": "2.1.2", "scipy": "1.16.1",
        "torch": "2.6.0+cu124", "torch_cuda": "12.4",
    }
    if probe != expected_probe or runtime_receipt.get("runtime_probe") != probe:
        raise RuntimeError(f"training-compatible runtime probe mismatch: {probe}")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if len(output) != config["gpu_count"] or any("A100" not in row for row in output):
        raise RuntimeError("evaluation GPU inventory mismatch")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if apps:
        raise RuntimeError("evaluation launch requires all GPUs idle")


def prepare_inputs(node_id: str) -> tuple[Path, list[dict]]:
    config = NODE_CONFIG[node_id]
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    for name in (
        "inputs", "jobs", "receipts", "logs", "job-caches", "control",
        "input-bindings",
    ):
        (WORK_ROOT / name).mkdir()
    control = WORK_ROOT / "control"
    shutil.copy2(RUNTIME_RECEIPT, control / "runtime_transfer_receipt.json")
    missing = [
        {"seed": seed, "cell": cell, "reason": "protocol-observed numerical failure"}
        for seed, cell in sorted(MISSING)
        if seed in config["seeds"]
    ]
    remote_protocol = SOURCE_ROOT / "control" / "protocol.json"
    protocol_dest = control / "training_protocol.json"
    protocol_record = subprocess.check_output(
        ssh_base(config) + ["sha256sum", str(remote_protocol)], text=True
    ).split()[0]
    if protocol_record != PROTOCOL_SHA256:
        raise RuntimeError("source training protocol SHA mismatch")
    subprocess.run(
        [
            "rsync", "-a", "-e",
            (
                f"ssh -i {TRANSFER_KEY} -p {config['source_port']} -o BatchMode=yes "
                f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}"
            ),
            f"root@{config['source_host']}:{remote_protocol}", str(protocol_dest),
        ],
        check=True,
    )
    if sha256_file(protocol_dest) != PROTOCOL_SHA256:
        raise RuntimeError("transferred protocol hash mismatch")
    jobs = []
    private = []
    for index, (seed, cell) in enumerate(expected_cells(node_id)):
        opaque = hashlib.sha256(
            f"{PROTOCOL_SHA256}|firstwave|{seed}|{cell}".encode()
        ).hexdigest()[:24]
        job = {
            "queue_index": index,
            "opaque_id": opaque,
        }
        jobs.append(job)
        private.append({**job, "seed": seed, "cell": cell})
    public_manifest = {
        "schema": "ect.q256.terminal-history-firstwave-public-evaluation/v1",
        "status": "FROZEN_WAITING_CHECKPOINT_BINDINGS", "node_id": node_id,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "runtime_transfer_receipt_sha256": sha256_file(
            control / "runtime_transfer_receipt.json"
        ),
        "evaluator_commit": EVALUATOR_COMMIT,
        "nfe": 1, "sample_seeds": [0, 49999], "sample_count": 50000,
        "metric_seed": 20260730, "metrics": list(METRICS),
        "missing_cells": missing, "job_count": len(jobs), "jobs": jobs,
    }
    private_map = {
        "schema": "ect.q256.terminal-history-firstwave-private-map/v1",
        "node_id": node_id, "protocol_sha256": PROTOCOL_SHA256,
        "jobs": private,
    }
    atomic_json(control / "private_map.json", private_map)
    public_manifest["private_map_sha256"] = sha256_file(control / "private_map.json")
    atomic_json(control / "public_manifest.json", public_manifest)
    return control / "private_map.json", private


def bind_input(node_id: str, job: dict) -> dict:
    config = NODE_CONFIG[node_id]
    seed, cell = job["seed"], job["cell"]
    while True:
        record = remote_probe(config, seed, cell)
        if record is not None:
            break
        print(f"[{utc_now()}] waiting for source endpoint: seed{seed}/{cell}", flush=True)
        time.sleep(60)
    destination = WORK_ROOT / "inputs" / f"seed{seed}-{cell}.pkl"
    pull_file(config, record, destination)
    binding = {
        "schema": "ect.q256.terminal-history-firstwave-input-binding/v1",
        "status": "PASS", "node_id": node_id,
        "seed": seed, "cell": cell, "opaque_id": job["opaque_id"],
        "checkpoint": str(destination),
        "checkpoint_bytes": record["bytes"],
        "checkpoint_sha256": record["sha256"],
        "source_path": record["path"],
        "compute_receipt_sha256": record["compute_receipt_sha256"],
        "trajectory_receipt_sha256": record["trajectory_receipt_sha256"],
        "protocol_sha256": PROTOCOL_SHA256,
        "bound_at": utc_now(),
    }
    binding_path = WORK_ROOT / "input-bindings" / f"{job['opaque_id']}.json"
    atomic_json(binding_path, binding)
    return {**job, **binding}


def runtime_env(gpu: int, cache: Path, port: int) -> dict[str, str]:
    root = TRAIN_RUNTIME_BASE
    paths = [
        root / "lib/python3.11/site-packages/torch/lib",
        root / "lib",
        Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"),
    ]
    env = os.environ.copy()
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        DNNLIB_CACHE_DIR=str(cache), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
        LD_LIBRARY_PATH=":".join(map(str, paths)),
        PATH=f"{TRAIN_RUNTIME_ENV / 'bin'}:{root / 'bin'}:/usr/bin:/bin",
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


def validate_job(job: dict, output: Path, elapsed: float) -> dict:
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
        os.link(kid, temporary)
        os.replace(temporary, fid)
    values = {metric: metric_value(output / f"metric-{metric}.jsonl", metric) for metric in METRICS}
    options = load(output / "training_options.json")
    if (options.get("sample_seeds") != list(range(50000))
            or options.get("seed") != 20260730
            or options.get("metrics") != list(METRICS)
            or options.get("mid_t") != []):
        raise RuntimeError("evaluation option contract mismatch")
    return {
        "schema": "ect.q256.terminal-history-firstwave-evaluation-job/v1",
        "status": "PASS", "opaque_id": job["opaque_id"],
        "queue_index": job["queue_index"], "checkpoint_sha256": job["checkpoint_sha256"],
        "elapsed_seconds": elapsed, "artifact_hashes": artifacts,
        "generated_feature_sha256": artifacts[kid.name]["sha256"],
        "kid_fid_shared_features": True,
        "metric_artifact_sha256": {
            metric: artifacts[f"metric-{metric}.jsonl"]["sha256"] for metric in METRICS
        },
        "values_sealed_in_metric_artifacts": True,
        "protocol_sha256": PROTOCOL_SHA256,
    }


def run_job(job: dict, gpu: int, cache: Path) -> dict:
    output = WORK_ROOT / "jobs" / job["opaque_id"]
    receipt_path = WORK_ROOT / "receipts" / f"{job['opaque_id']}.json"
    if output.exists() or receipt_path.exists():
        raise RuntimeError(f"refuse pre-existing evaluation job: {job['opaque_id']}")
    output.mkdir()
    cache.mkdir(parents=True, exist_ok=True)
    port = 53000 + gpu
    python = TRAIN_RUNTIME_ENV / "bin/python"
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        f"--master_port={port}", str(EVALUATOR / "ct_eval.py"),
        "--resume", job["checkpoint"], "--outdir", str(output), "--nosubdir",
        "--data", str(DATASET), "--cond=False", "--arch=ddpmpp", "--precond=ct",
        "--dropout=0.2", "--augment=0", "--xflip=False", "--fp16=False",
        "--cache=True", "--workers=1", "--eval-batch=512", "--metric-generator-batch=128",
        "--nfe=1", "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
        "--sample-seeds=0-49999", "--seed=20260730", "--retain-generated-artifacts",
        f"--desc=blind-{job['opaque_id']}",
    ]
    log_path = WORK_ROOT / "logs" / f"{job['opaque_id']}.launcher.log"
    start = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, cwd=EVALUATOR, env=runtime_env(gpu, cache, port),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        timed_out = False
        try:
            code = process.wait(timeout=4 * 3600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    if code != 0 or timed_out:
        failure = {
            "schema": "ect.q256.terminal-history-firstwave-evaluation-job/v1",
            "status": "FAIL", "opaque_id": job["opaque_id"], "gpu_index": gpu,
            "exit_code": code, "hard_timeout": timed_out, "automatic_retry_count": 0,
            "protocol_sha256": PROTOCOL_SHA256,
        }
        atomic_json(output / "failure_receipt.json", failure)
        return failure
    try:
        receipt = validate_job(job, output, time.monotonic() - start)
        receipt["gpu_index"] = gpu
        receipt["automatic_retry_count"] = 0
        atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema": "ect.q256.terminal-history-firstwave-evaluation-job/v1",
            "status": "FAIL_POSTCHECK", "opaque_id": job["opaque_id"],
            "gpu_index": gpu, "error": repr(error), "automatic_retry_count": 0,
            "protocol_sha256": PROTOCOL_SHA256,
        }
        atomic_json(output / "postcheck_failure_receipt.json", failure)
        return failure


def copy_cache_template(destination: Path) -> None:
    source = WORK_ROOT / "cache-template"
    shutil.copytree(source, destination, copy_function=os.link)


def execute_jobs(node_id: str, private: list[dict]) -> None:
    config = NODE_CONFIG[node_id]
    prewarm = bind_input(node_id, private[0])
    first = run_job(prewarm, 0, WORK_ROOT / "cache-template")
    if first["status"] != "PASS":
        raise RuntimeError("cache-prewarm evaluation job failed; no retry")
    queues = [[] for _ in range(config["gpu_count"])]
    for index, job in enumerate(private[1:]):
        queues[index % config["gpu_count"]].append(job)

    def worker(gpu: int, jobs: list[dict]) -> list[dict]:
        results = []
        for job in jobs:
            job = bind_input(node_id, job)
            cache = WORK_ROOT / "job-caches" / job["opaque_id"]
            copy_cache_template(cache)
            results.append(run_job(job, gpu, cache))
        return results

    results = [first]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["gpu_count"]) as executor:
        futures = [executor.submit(worker, gpu, jobs) for gpu, jobs in enumerate(queues) if jobs]
        for future in futures:
            results.extend(future.result())
    failures = [record for record in results if record.get("status") != "PASS"]
    atomic_json(WORK_ROOT / "control" / "evaluation_matrix_completion.json", {
        "schema": "ect.q256.terminal-history-firstwave-evaluation-matrix/v1",
        "status": "PASS" if not failures and len(results) == len(private) else "FAIL",
        "node_id": node_id, "expected_jobs": len(private), "completed_jobs": len(results),
        "failures": failures, "automatic_retry_count": 0,
        "protocol_sha256": PROTOCOL_SHA256, "ended_at": utc_now(),
    })
    if failures or len(results) != len(private):
        raise RuntimeError("evaluation matrix failed closed")
    bindings = sorted((WORK_ROOT / "input-bindings").glob("*.json"))
    if len(bindings) != len(private):
        raise RuntimeError("checkpoint binding count mismatch")
    atomic_json(WORK_ROOT / "control" / "input_transfer_receipt.json", {
        "schema": "ect.q256.terminal-history-firstwave-input-transfer/v1",
        "status": "PASS", "node_id": node_id,
        "binding_hashes": {path.name: sha256_file(path) for path in bindings},
        "public_manifest_sha256": sha256_file(WORK_ROOT / "control" / "public_manifest.json"),
        "protocol_sha256": PROTOCOL_SHA256,
    })


def seal_and_decode(node_id: str, private: list[dict]) -> None:
    matrix = load(WORK_ROOT / "control" / "evaluation_matrix_completion.json")
    if matrix.get("status") != "PASS":
        raise RuntimeError("cannot seal incomplete matrix")
    receipts = sorted((WORK_ROOT / "receipts").glob("*.json"))
    if len(receipts) != len(private):
        raise RuntimeError("receipt count mismatch before seal")
    receipt_hashes = {path.name: sha256_file(path) for path in receipts}
    seal = {
        "schema": "ect.q256.terminal-history-firstwave-evaluation-seal/v1",
        "status": "SEALED_PASS", "node_id": node_id,
        "job_count": len(private), "receipt_hashes": receipt_hashes,
        "public_manifest_sha256": sha256_file(WORK_ROOT / "control" / "public_manifest.json"),
        "private_map_sha256": sha256_file(WORK_ROOT / "control" / "private_map.json"),
        "protocol_sha256": PROTOCOL_SHA256, "sealed_at": utc_now(),
    }
    atomic_json(WORK_ROOT / "control" / "evaluation_seal.json", seal)
    rows = []
    for job in private:
        binding = load(WORK_ROOT / "input-bindings" / f"{job['opaque_id']}.json")
        receipt = load(WORK_ROOT / "receipts" / f"{job['opaque_id']}.json")
        if receipt_hashes[f"{job['opaque_id']}.json"] != sha256_file(
            WORK_ROOT / "receipts" / f"{job['opaque_id']}.json"
        ):
            raise RuntimeError("receipt changed after seal")
        job_dir = WORK_ROOT / "jobs" / job["opaque_id"]
        rows.append({
            "seed": job["seed"], "cell": job["cell"], "budget_kimg": 1024,
            "nfe": 1, "kid50k_full": metric_value(job_dir / "metric-kid50k_full.jsonl", "kid50k_full"),
            "fid50k_full": metric_value(job_dir / "metric-fid50k_full.jsonl", "fid50k_full"),
            "opaque_id": job["opaque_id"], "checkpoint_sha256": binding["checkpoint_sha256"],
            "receipt_sha256": receipt_hashes[f"{job['opaque_id']}.json"],
        })
    rows.sort(key=lambda row: (row["seed"], row["cell"]))
    csv_path = WORK_ROOT / "decoded_results.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    atomic_json(WORK_ROOT / "decoded_results.json", {
        "schema": "ect.q256.terminal-history-firstwave-decoded-results/v1",
        "status": "PASS", "node_id": node_id,
        "seal_sha256": sha256_file(WORK_ROOT / "control" / "evaluation_seal.json"),
        "rows": rows, "protocol_sha256": PROTOCOL_SHA256,
    })


def build_return_manifest() -> Path:
    excluded = {"inputs", "cache-template", "job-caches"}
    files = []
    for path in WORK_ROOT.rglob("*"):
        if not path.is_file() or path.name == "RETURN_SHA256SUMS.txt":
            continue
        relative = path.relative_to(WORK_ROOT)
        if relative.parts[0] in excluded:
            continue
        files.append((relative, sha256_file(path)))
    destination = WORK_ROOT / "RETURN_SHA256SUMS.txt"
    with destination.open("x", encoding="ascii") as handle:
        for relative, digest in sorted(files):
            handle.write(f"{digest}  {relative}\n")
        handle.flush(); os.fsync(handle.fileno())
    return destination


def push_back(node_id: str) -> None:
    config = NODE_CONFIG[node_id]
    build_return_manifest()
    target = SOURCE_ROOT / "evaluation_firstwave" / config["return_label"]
    parent = target.parent
    prepare_remote = (
        f"mkdir -p {shlex.quote(str(parent))} && "
        f"test ! -e {shlex.quote(str(target))} && "
        f"mkdir {shlex.quote(str(target))}"
    )
    subprocess.run(ssh_base(config) + [prepare_remote], check=True)
    subprocess.run([
        "rsync", "-a", "--exclude=inputs", "--exclude=cache-template", "--exclude=job-caches",
        "-e",
        (
            f"ssh -i {TRANSFER_KEY} -p {config['source_port']} -o BatchMode=yes "
            f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}"
        ),
        f"{WORK_ROOT}/", f"root@{config['source_host']}:{target}/",
    ], check=True)
    verify_remote = (
        f"cd {shlex.quote(str(target))} && sha256sum -c RETURN_SHA256SUMS.txt"
    )
    subprocess.run(ssh_base(config) + [verify_remote], check=True)
    atomic_json(WORK_ROOT / "control" / "return_verification_receipt.json", {
        "schema": "ect.q256.terminal-history-firstwave-return/v1",
        "status": "PASS", "node_id": node_id, "target": str(target),
        "return_manifest_sha256": sha256_file(WORK_ROOT / "RETURN_SHA256SUMS.txt"),
        "verified_at": utc_now(), "protocol_sha256": PROTOCOL_SHA256,
    })
    # Push the final return receipt separately and verify its hash.
    receipt = WORK_ROOT / "control" / "return_verification_receipt.json"
    subprocess.run([
        "rsync", "-a", "-e",
        (
            f"ssh -i {TRANSFER_KEY} -p {config['source_port']} -o BatchMode=yes "
            f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN_HOSTS}"
        ), str(receipt), f"root@{config['source_host']}:{target}/control/",
    ], check=True)
    remote_hash = subprocess.check_output(
        ssh_base(config) + ["sha256sum", str(target / "control" / receipt.name)], text=True
    ).split()[0]
    if remote_hash != sha256_file(receipt):
        raise RuntimeError("return receipt hash mismatch")


def wait_run(node_id: str) -> None:
    validate_environment(node_id)
    _, private = prepare_inputs(node_id)
    execute_jobs(node_id, private)
    seal_and_decode(node_id, private)
    push_back(node_id)
    print(json.dumps({"status": "PASS", "node_id": node_id, "jobs": len(private),
                      "returned_to_source": True}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", choices=tuple(NODE_CONFIG), required=True)
    return parser


def main() -> int:
    args = parser().parse_args()
    wait_run(args.node_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
