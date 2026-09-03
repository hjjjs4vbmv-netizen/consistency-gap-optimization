#!/usr/bin/env python3
"""Run the generation-block sensitivity matrix."""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .validation import sha256, validate_attempt, validate_receipt


def parse_gpus(value):
    gpus = [int(item) for item in value.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise RuntimeError("exactly two distinct GPU indices are required")
    return gpus


def gpu_slot(job_index):
    return ((job_index // 2) + (job_index % 2)) % 2


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(path):
    manifest = json.loads(path.read_text())
    checkpoints = manifest["checkpoints"]
    blocks = manifest["blocks"]
    expected_checkpoints = manifest["expected_checkpoints"]
    expected_jobs = expected_checkpoints * len(blocks) * 2
    if (len(checkpoints) != expected_checkpoints
            or len({item["id"] for item in checkpoints}) != expected_checkpoints):
        raise RuntimeError("manifest checkpoint count or identity mismatch")
    ranges = [(item["start"], item["end"]) for item in blocks]
    expected = [(start, start + 49999) for start in range(50000, 300000, 50000)]
    if ranges != expected:
        raise RuntimeError("generation blocks are not the frozen B1-B5 ranges")
    jobs = build_jobs(manifest)
    if len(jobs) != expected_jobs or len({job["job_id"] for job in jobs}) != expected_jobs:
        raise RuntimeError("matrix job count or identity mismatch")
    if (len(manifest["contrasts"]) != manifest["expected_contrasts"]
            or len(manifest["rotations"]) != manifest["expected_rotations"]):
        raise RuntimeError("unexpected contrast or rotation count")
    if manifest.get("metric_seed") != 20260730:
        raise RuntimeError("metric seed is not frozen to 20260730")
    return manifest


def build_jobs(manifest):
    jobs = []
    for checkpoint in manifest["checkpoints"]:
        for block in manifest["blocks"]:
            for nfe in (1, 2):
                jobs.append({
                    "job_index": len(jobs),
                    "job_id": f"{checkpoint['id']}-{block['id'].lower()}-nfe{nfe}",
                    "checkpoint": checkpoint,
                    "block": block,
                    "nfe": nfe,
                })
    return jobs


def select_jobs(jobs, checkpoint_ids):
    if not checkpoint_ids:
        return jobs
    requested = {item.strip() for item in checkpoint_ids.split(",") if item.strip()}
    known = {job["checkpoint"]["id"] for job in jobs}
    unknown = requested - known
    if unknown:
        raise RuntimeError("unknown checkpoint ids: {}".format(",".join(sorted(unknown))))
    return [job for job in jobs if job["checkpoint"]["id"] in requested]


def git_output(repo, *args):
    return subprocess.check_output(
        ["git", *args], cwd=str(repo), universal_newlines=True).strip()


def preflight(manifest, gpus):
    evaluator = Path(manifest["evaluator_repo"])
    if git_output(evaluator, "rev-parse", "HEAD") != manifest["evaluator_commit"]:
        raise RuntimeError("evaluator commit mismatch")
    if git_output(evaluator, "status", "--porcelain"):
        raise RuntimeError("evaluator worktree is dirty")
    bindings = (
        (Path(manifest["dataset"]), manifest["dataset_sha256"]),
        (Path(manifest["runtime_sif"]), manifest["runtime_sha256"]),
        *((Path(item["path"]), item["sha256"]) for item in manifest["checkpoints"]),
    )
    for path, expected in bindings:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"file binding mismatch: {path}")
    cache = Path(manifest["detector_cache"])
    if not cache.is_dir() or not any(cache.iterdir()):
        raise RuntimeError("detector cache is missing")
    output_parent = Path(manifest["output_root"]).parent
    if not os.access(output_parent, os.W_OK):
        raise RuntimeError(f"output parent is not writable: {output_parent}")
    if shutil.disk_usage(output_parent).free < 180 * 1024**3:
        raise RuntimeError("less than 180 GiB free for retained artifacts")
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free",
        "--format=csv,noheader,nounits",
    ], universal_newlines=True).splitlines()
    gpu_rows = {int(row.split(",", 1)[0]): row for row in rows}
    for gpu in gpus:
        if gpu not in gpu_rows or "A100" not in gpu_rows[gpu]:
            raise RuntimeError(f"GPU {gpu} is not an A100")


def prepare_caches(manifest, gpus):
    source = Path(manifest["detector_cache"])
    cache_root = Path(manifest["output_root"]) / "worker-cache"
    caches = {}
    for gpu in gpus:
        downloads = cache_root / f"gpu{gpu}" / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = downloads / item.name
            if not target.exists():
                shutil.copy2(item, target)
        caches[gpu] = downloads.parent
    return caches


def command(manifest, job, target, port):
    mid_t = [] if job["nfe"] == 1 else ["--mid_t=0.821"]
    block = job["block"]
    return [
        "singularity", "exec", "--nv", "--bind", "/data:/data",
        manifest["runtime_sif"], "python", "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=1", f"--master_port={port}",
        str(Path(manifest["evaluator_repo"]) / "ct_eval.py"),
        "--resume", job["checkpoint"]["path"], "--outdir", str(target), "--nosubdir",
        "--data", manifest["dataset"], "--cond=False", "--arch=ddpmpp",
        "--precond=ct", "--dropout=0.2", "--augment=0", "--xflip=False",
        "--fp16=False", "--cache=True", "--workers=1", "--eval-batch=512",
        "--metric-generator-batch=128", f"--nfe={job['nfe']}", *mid_t,
        "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
        f"--sample-seeds={block['start']}-{block['end']}", "--seed=20260730",
        "--retain-generated-artifacts", f"--desc=noise-floor-{job['job_id']}",
    ]


def run_job(manifest, job, gpu, cache):
    root = Path(manifest["output_root"])
    receipt = root / "receipts" / f"{job['job_id']}.json"
    if receipt.exists():
        validate_receipt(manifest, job, json.loads(receipt.read_text()))
        return "PASS"
    errors = []
    for attempt in (1, 2):
        target = root / "jobs" / job["job_id"] / f"attempt-{attempt:02d}"
        if target.exists():
            errors.append(f"attempt-{attempt:02d}: pre-existing nonterminal output")
            continue
        log = root / "logs" / f"{job['job_id']}.attempt-{attempt:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "DNNLIB_CACHE_DIR": str(cache),
            "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        })
        started = time.time()
        try:
            with log.open("wb") as handle:
                subprocess.run(command(manifest, job, target, 52000 + job["job_index"]),
                               cwd=manifest["evaluator_repo"], env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True, timeout=21600)
            metrics, feature_sha = validate_attempt(manifest, job, target)
            atomic_json(receipt, {
                "status": "PASS", "job_id": job["job_id"],
                "job_index": job["job_index"], "checkpoint_id": job["checkpoint"]["id"],
                "checkpoint": job["checkpoint"]["path"],
                "checkpoint_sha256": job["checkpoint"]["sha256"],
                "block": job["block"], "nfe": job["nfe"],
                "mid_t": None if job["nfe"] == 1 else 0.821,
                "attempt": attempt, "gpu": gpu, "metrics": metrics,
                "generated_feature_sha256": feature_sha,
                "metric_seed": manifest["metric_seed"],
                "evaluator_commit": manifest["evaluator_commit"],
                "dataset_sha256": manifest["dataset_sha256"],
                "runtime_sha256": manifest["runtime_sha256"],
                "elapsed_seconds": round(time.time() - started, 3),
                "job_dir": str(target),
            })
            print(f"PASS {job['job_id']} gpu={gpu} attempt={attempt}", flush=True)
            return "PASS"
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(f"attempt-{attempt:02d}: {type(error).__name__}: {error}")
            print(f"RETRY {job['job_id']} {errors[-1]}", flush=True)
    atomic_json(receipt, {"status": "EVAL_FAIL", "job_id": job["job_id"],
                          "job_index": job["job_index"], "errors": errors})
    return "EVAL_FAIL"


def execute(manifest, jobs, gpus):
    caches = prepare_caches(manifest, gpus)
    groups = {gpu: [] for gpu in gpus}
    for job in jobs:
        groups[gpus[gpu_slot(job["job_index"])]].append(job)

    def worker(gpu):
        return [run_job(manifest, job, gpu, caches[gpu]) for job in groups[gpu]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = [status for result in pool.map(worker, gpus) for status in result]
    print(json.dumps({"jobs": len(jobs), "PASS": statuses.count("PASS"),
                      "EVAL_FAIL": statuses.count("EVAL_FAIL")}))
    if any(status != "PASS" for status in statuses):
        raise RuntimeError("one or more evaluation jobs failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--mode", choices=("list", "check", "canary", "run"), required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--checkpoint-ids")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest.resolve())
    jobs = select_jobs(build_jobs(manifest), args.checkpoint_ids)
    if args.mode == "list":
        selected = len({job["checkpoint"]["id"] for job in jobs})
        print(json.dumps({"checkpoints": selected, "jobs": len(jobs),
                          "job_ids": [job["job_id"] for job in jobs]}, indent=2))
        return
    gpus = parse_gpus(args.gpus)
    preflight(manifest, gpus)
    if args.mode == "check":
        print(json.dumps({"status": "PREFLIGHT_PASS", "jobs": len(jobs), "gpus": gpus}))
        return
    if args.mode == "canary":
        canaries = set(manifest["canary_jobs"])
        jobs = [job for job in jobs if job["job_id"] in canaries]
    elif args.mode == "run":
        job_index = {job["job_id"]: job for job in jobs}
        for job_id in manifest["canary_jobs"]:
            job = job_index[job_id]
            receipt = Path(manifest["output_root"]) / "receipts" / f"{job_id}.json"
            if not receipt.is_file():
                raise RuntimeError("full run requires validated canary receipts")
            validate_receipt(manifest, job, json.loads(receipt.read_text()))
    execute(manifest, jobs, gpus)


if __name__ == "__main__":
    main()
