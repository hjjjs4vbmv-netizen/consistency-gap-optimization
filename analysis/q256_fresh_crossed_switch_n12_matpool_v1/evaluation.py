#!/usr/bin/env python3
"""Prepare, run, seal, and only then decode the blinded 264-job evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402

PUBLIC_SCHEMA = "ect.q256.fresh-crossed-switch-blind-evaluation/v1"
PRIVATE_SCHEMA = "ect.q256.fresh-crossed-switch-private-evaluation-map/v1"
JOB_SCHEMA = "ect.q256.fresh-crossed-switch-blind-job/v1"
METRICS = ("kid50k_full", "fid50k_full")
BUDGETS = (640, 768, 896, 1024)


def load(path: Path) -> dict:
    return experiment.load_json(path)


def sha256_file(path: Path) -> str:
    return experiment.sha256_file(path)


def evaluation_profile(protocol_path: Path, authorization_path: Path | None,
                       recovery_authorization_path: Path | None = None) -> dict:
    if authorization_path is None:
        if recovery_authorization_path is not None:
            raise RuntimeError("evaluation recovery requires the eleven-seed authorization")
        protocol = load(protocol_path)
        return {"seeds": experiment.SEEDS, "job_count": 264,
                "evaluation_dir": "evaluation", "matrix": "training_matrix_completion_receipt.json",
                "integrity": "training_integrity_report.json", "authorization_sha256": None,
                "recovery_authorization_sha256": None,
                "cache_root": str(Path(protocol["paths"].get("evaluator_cache_root", ".")).resolve())}
    authorization_path = authorization_path.resolve(strict=True)
    protocol = load(protocol_path)
    experiment.validate_eleven_seed_authorization(
        authorization_path, protocol_path, require_commit=recovery_authorization_path is None
    )
    recovery_sha = None
    evaluation_dir = "evaluation_11seed"
    cache_root = str(Path(protocol["paths"].get("evaluator_cache_root", ".")).resolve())
    if recovery_authorization_path is not None:
        recovery_authorization_path = recovery_authorization_path.resolve(strict=True)
        recovery = experiment.validate_evaluation_recovery1_authorization(
            recovery_authorization_path, protocol_path, require_commit=True
        )
        if recovery.get("eleven_seed_authorization_sha256") != sha256_file(authorization_path):
            raise RuntimeError("evaluation recovery has the wrong eleven-seed binding")
        recovery_sha = sha256_file(recovery_authorization_path)
        evaluation_dir = recovery["evaluation_dir"]
        cache_root = str(Path(recovery["cache_destination"]).resolve(strict=True))
    return {"seeds": experiment.ELEVEN_SEEDS, "job_count": experiment.ELEVEN_JOB_COUNT,
            "evaluation_dir": evaluation_dir,
            "matrix": "training_matrix_11seed_completion_receipt.json",
            "integrity": "training_integrity_11seed_report.json",
            "authorization_sha256": sha256_file(authorization_path),
            "recovery_authorization_sha256": recovery_sha, "cache_root": cache_root}


def metric_value(path: Path, metric: str) -> float:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"metric file must have one record: {path}")
    record = json.loads(lines[0])
    if record.get("metric") != metric or record.get("num_gpus") != 1:
        raise RuntimeError(f"metric identity mismatch: {path}")
    value = float(record["results"][metric])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid metric result: {path}")
    return value


def training_jobs(protocol: dict, protocol_path: Path, profile: dict) -> list[dict]:
    root = Path(protocol["paths"]["formal_output_root"])
    completion = load(root / profile["matrix"])
    if completion.get("status") != "PASS" or completion.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("training matrix is not a protocol-bound PASS")
    integrity = load(root / profile["integrity"])
    if integrity.get("status") != "PASS" or integrity.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("blind evaluation requires a protocol-bound training integrity PASS")
    jobs: list[dict] = []
    if profile["authorization_sha256"] is not None:
        if (completion.get("eleven_seed_authorization_sha256") != profile["authorization_sha256"]
                or integrity.get("eleven_seed_authorization_sha256") != profile["authorization_sha256"]):
            raise RuntimeError("eleven-seed training gates are not authorization-bound")
    for seed in profile["seeds"]:
        seed_root = root / "training" / f"seed{seed}"
        if load(seed_root / "seed_completion_receipt.json").get("status") != "PASS":
            raise RuntimeError(f"seed {seed} is not PASS")
        for arm in experiment.ARMS:
            path = seed_root / f"prefix_{arm}" / "kimg0512" / "network-snapshot.pkl"
            jobs.append({"seed": seed, "kind": "prefix", "cell": arm, "budget_kimg": 512,
                         "nfe": 1, "checkpoint": path})
        for cell in experiment.CELLS:
            for budget in BUDGETS:
                path = seed_root / cell / f"kimg{budget:04d}" / "network-snapshot.pkl"
                jobs.append({"seed": seed, "kind": "suffix", "cell": cell,
                             "budget_kimg": budget, "nfe": 1, "checkpoint": path})
            path = seed_root / cell / "kimg1024" / "network-snapshot.pkl"
            jobs.append({"seed": seed, "kind": "suffix", "cell": cell,
                         "budget_kimg": 1024, "nfe": 2, "checkpoint": path})
    identities = {(j["seed"], j["kind"], j["cell"], j["budget_kimg"], j["nfe"]) for j in jobs}
    if len(jobs) != profile["job_count"] or len(identities) != profile["job_count"]:
        raise RuntimeError("evaluation matrix has the wrong unique job count")
    return jobs


def prepare(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    profile = evaluation_profile(
        protocol_path, getattr(args, "eleven_seed_authorization", None),
        getattr(args, "evaluation_recovery_authorization", None)
    )
    eval_root = Path(protocol["paths"]["formal_output_root"]) / profile["evaluation_dir"]
    if eval_root.exists() or args.private_map.exists() or args.public_manifest.exists():
        raise RuntimeError("blind evaluation preparation requires fresh destinations")
    eval_root.mkdir(parents=True, exist_ok=False)
    aliases = eval_root / "checkpoints"
    aliases.mkdir()
    jobs = training_jobs(protocol, protocol_path, profile)
    shuffled = list(jobs)
    random.Random(protocol["evaluation"]["shuffle_seed"]).shuffle(shuffled)
    private_jobs = []
    public_jobs = []
    used_ids: set[str] = set()
    for queue_index, job in enumerate(shuffled):
        opaque = secrets.token_hex(12)
        while opaque in used_ids:
            opaque = secrets.token_hex(12)
        used_ids.add(opaque)
        source = Path(job.pop("checkpoint")).resolve(strict=True)
        if source.is_symlink() or source.stat().st_size <= 0:
            raise RuntimeError(f"invalid evaluation checkpoint: {source}")
        alias = aliases / f"{opaque}.pkl"
        os.link(source, alias)
        checkpoint_sha = sha256_file(alias)
        gpu = queue_index % 6
        private_jobs.append({
            "queue_index": queue_index, "opaque_id": opaque, "gpu_index": gpu,
            **job, "source_checkpoint": str(source), "checkpoint_alias": str(alias),
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        })
        public_jobs.append({
            "queue_index": queue_index, "opaque_id": opaque, "gpu_index": gpu,
            "checkpoint_alias": str(alias), "checkpoint_sha256": checkpoint_sha,
            "status": "FROZEN_NOT_RUN",
        })
    private_payload = {
        "schema": PRIVATE_SCHEMA, "status": "SEALED_PRIVATE_MAP",
        "protocol_sha256": sha256_file(protocol_path), "job_count": profile["job_count"],
        "eleven_seed_authorization_sha256": profile["authorization_sha256"],
        "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        "decode_forbidden_before_matrix_seal": True, "jobs": private_jobs,
    }
    args.private_map.parent.mkdir(parents=True, exist_ok=True)
    experiment.atomic_json(args.private_map, private_payload)
    os.chmod(args.private_map, 0o400)
    private_sha = sha256_file(args.private_map)
    public_payload = {
        "schema": PUBLIC_SCHEMA, "status": "FROZEN_NOT_RUN", "job_count": profile["job_count"],
        "protocol_sha256": sha256_file(protocol_path),
        "eleven_seed_authorization_sha256": profile["authorization_sha256"],
        "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        "private_map_sha256": private_sha,
        "shuffle_seed": protocol["evaluation"]["shuffle_seed"],
        "gpu_assignment": "queue_index modulo six after frozen shuffle",
        "metrics_executed": False, "identities_disclosed": False,
        "jobs": public_jobs,
    }
    args.public_manifest.parent.mkdir(parents=True, exist_ok=True)
    experiment.atomic_json(args.public_manifest, public_payload)
    experiment.atomic_json(eval_root / "preparation_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-blind-preparation/v1", "status": "PASS",
        "job_count": profile["job_count"], "protocol_sha256": sha256_file(protocol_path),
        "eleven_seed_authorization_sha256": profile["authorization_sha256"],
        "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        "public_manifest": str(args.public_manifest.resolve()),
        "public_manifest_sha256": sha256_file(args.public_manifest),
        "private_map": str(args.private_map.resolve()), "private_map_sha256": private_sha,
        "opaque_checkpoint_aliases": profile["job_count"],
    })
    print(json.dumps({"status": "FROZEN_NOT_RUN", "job_count": profile["job_count"],
                      "public_manifest_sha256": sha256_file(args.public_manifest)}))


def validate_bindings(protocol_path: Path, public_path: Path, private_path: Path,
                      authorization_path: Path | None = None,
                      recovery_authorization_path: Path | None = None) -> tuple[dict, dict, dict, dict]:
    protocol = load(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    public = load(public_path)
    private = load(private_path)
    protocol_sha = sha256_file(protocol_path)
    profile = evaluation_profile(protocol_path, authorization_path, recovery_authorization_path)
    if public.get("schema") != PUBLIC_SCHEMA or public.get("job_count") != profile["job_count"]:
        raise RuntimeError("invalid public blind manifest")
    if private.get("schema") != PRIVATE_SCHEMA or private.get("job_count") != profile["job_count"]:
        raise RuntimeError("invalid private blind map")
    if public.get("protocol_sha256") != protocol_sha or private.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("blind evaluation protocol mismatch")
    if public.get("private_map_sha256") != sha256_file(private_path):
        raise RuntimeError("private mapping hash mismatch")
    if (public.get("eleven_seed_authorization_sha256") != profile["authorization_sha256"]
            or private.get("eleven_seed_authorization_sha256") != profile["authorization_sha256"]):
        raise RuntimeError("blind manifests have the wrong amendment binding")
    if (public.get("evaluation_recovery_authorization_sha256")
            != profile["recovery_authorization_sha256"]
            or private.get("evaluation_recovery_authorization_sha256")
            != profile["recovery_authorization_sha256"]):
        raise RuntimeError("blind manifests have the wrong evaluation recovery binding")
    pub = {(j["queue_index"], j["opaque_id"], j["gpu_index"], j["checkpoint_sha256"])
           for j in public["jobs"]}
    prv = {(j["queue_index"], j["opaque_id"], j["gpu_index"], j["checkpoint_sha256"])
           for j in private["jobs"]}
    if len(pub) != profile["job_count"] or pub != prv:
        raise RuntimeError("public/private blind job mismatch")
    return protocol, public, private, profile


def evaluator_ok(path: Path) -> None:
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True).strip()
    if head != experiment.EVALUATOR_COMMIT or dirty:
        raise RuntimeError("evaluator repository is not the frozen clean commit")


def validate_job(job: dict, job_dir: Path, protocol: dict, evaluator_repo: Path,
                 gpu: int, elapsed: float, receipt: Path) -> None:
    required = ["log.txt", "training_options.json", "generated-samples.npy",
                "generated-features-kid50k_full-repeat00.npy",
                "generated-features-fid50k_full-repeat00.npy",
                "metric-kid50k_full.jsonl", "metric-fid50k_full.jsonl"]
    artifacts = {}
    for name in required:
        path = job_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing evaluation artifact: {path}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    kid_sha = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
    fid_sha = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
    if kid_sha != fid_sha:
        raise RuntimeError("KID/FID were not computed from identical generated features")
    kid_feature = job_dir / "generated-features-kid50k_full-repeat00.npy"
    fid_feature = job_dir / "generated-features-fid50k_full-repeat00.npy"
    shared_tmp = job_dir / ".generated-features-shared.tmp"
    os.link(kid_feature, shared_tmp)
    os.replace(shared_tmp, fid_feature)
    directory = os.open(job_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if kid_feature.stat().st_ino != fid_feature.stat().st_ino:
        raise RuntimeError("generated-feature hardlink deduplication failed")
    for metric in METRICS:
        metric_value(job_dir / f"metric-{metric}.jsonl", metric)
    options = load(job_dir / "training_options.json")
    expected = {"batch_size": 512, "metrics": list(METRICS), "metric_repeats": 1,
                "metric_generator_batch": 128, "retain_generated_artifacts": True,
                "seed": 20260730}
    for key, value in expected.items():
        if options.get(key) != value:
            raise RuntimeError(f"evaluation option mismatch: {key}")
    if options.get("sample_seeds") != list(range(50000)):
        raise RuntimeError("generation seeds are not exactly 0..49999")
    if options.get("mid_t") != ([] if job["nfe"] == 1 else [0.821]):
        raise RuntimeError("NFE/mid_t mismatch")
    alias = Path(job["checkpoint_alias"]).resolve(strict=True)
    if Path(options["resume_pkl"]).resolve() != alias or sha256_file(alias) != job["checkpoint_sha256"]:
        raise RuntimeError("checkpoint alias binding mismatch")
    dataset = Path(protocol["assets"]["dataset"]["path"]).resolve(strict=True)
    if Path(options["dataset_kwargs"]["path"]).resolve() != dataset:
        raise RuntimeError("dataset binding mismatch")
    experiment.atomic_json(receipt, {
        "schema": JOB_SCHEMA, "status": "VALIDATED_UNSEALED",
        "opaque_id": job["opaque_id"], "queue_index": job["queue_index"],
        "gpu_index": gpu, "gpu_uuid": protocol["gpus"][gpu]["uuid"],
        "elapsed_seconds": elapsed, "checkpoint_sha256": job["checkpoint_sha256"],
        "generated_feature_sha256": kid_sha, "kid_fid_shared_feature_identity": True,
        "kid_fid_shared_feature_inode": True,
        "metric_artifact_sha256": {metric: artifacts[f"metric-{metric}.jsonl"]["sha256"] for metric in METRICS},
        "artifacts": artifacts, "evaluator_commit": experiment.EVALUATOR_COMMIT,
        "protocol_sha256": sha256_file(Path(protocol["frozen_protocol_path"]))
            if "frozen_protocol_path" in protocol else job["protocol_sha256"],
        "result_values_disclosed": False,
        "evaluation_recovery_authorization_sha256":
            job.get("evaluation_recovery_authorization_sha256"),
    })


def run_one(protocol_path: Path, protocol: dict, job: dict, evaluator_repo: Path,
            cache_root: Path, eval_root: Path) -> None:
    gpu = int(job["gpu_index"])
    opaque = job["opaque_id"]
    run_root = eval_root / "jobs" / opaque
    receipt = eval_root / "receipts" / f"{opaque}.json"
    if run_root.exists() or receipt.exists():
        raise RuntimeError(f"refuse existing blind evaluation output: {opaque}")
    run_root.mkdir(parents=True, exist_ok=False)
    experiment.assert_gpu_exclusive(protocol["gpus"][gpu]["uuid"],
                                    run_root / "gpu_exclusivity_before.json", "before")
    runtime = load(Path(protocol["assets"]["runtime_manifest"]["path"]))
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    command = [
        python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        f"--master_port={52000 + gpu}", str(evaluator_repo / "ct_eval.py"),
        "--resume", job["checkpoint_alias"], "--outdir", str(run_root), "--nosubdir",
        "--data", protocol["assets"]["dataset"]["path"], "--cond=False", "--arch=ddpmpp",
        "--precond=ct", "--dropout=0.2", "--augment=0", "--xflip=False", "--fp16=False",
        "--cache=True", "--workers=1", "--eval-batch=512", "--metric-generator-batch=128",
        f"--nfe={job['nfe']}", "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
        "--sample-seeds=0-49999", "--seed=20260730", "--retain-generated-artifacts",
        f"--desc=blind-{opaque}",
    ]
    if job["nfe"] == 2:
        command.append("--mid_t=0.821")
    env = experiment.cell_environment(gpu, runtime)
    env["DNNLIB_CACHE_DIR"] = str(cache_root)
    log_handle = (run_root / "launcher.log").open("xb")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    monitor = subprocess.Popen(
        [python, str(Path(__file__).with_name("monitor.py")), "--pid", str(process.pid),
         "--run-dir", str(run_root), "--gpu-index", str(gpu),
         "--gpu-uuid", protocol["gpus"][gpu]["uuid"], "--total-attempts", "0",
         "--log-name", "launcher.log", "--interval-seconds", "30",
         "--stall-seconds", "300", "--min-free-bytes", str(100 * 1024**3)],
        cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    timed_out = False
    try:
        code = process.wait(timeout=6 * 3600)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    finally:
        log_handle.close()
    monitor.wait(timeout=90)
    if code != 0 or timed_out:
        experiment.atomic_json(run_root / "failure_receipt.json", {
            "schema": JOB_SCHEMA, "status": "FAIL", "opaque_id": opaque,
            "exit_code": code, "hard_timeout": timed_out, "automatic_retry_count": 0,
            "evaluation_recovery_authorization_sha256":
                job.get("evaluation_recovery_authorization_sha256"),
        })
        raise RuntimeError(f"blind evaluation failed without retry: {opaque}")
    job = dict(job)
    job["protocol_sha256"] = sha256_file(protocol_path)
    validate_job(job, run_root, protocol, evaluator_repo, gpu, time.monotonic() - started, receipt)
    experiment.assert_gpu_exclusive(protocol["gpus"][gpu]["uuid"],
                                    run_root / "gpu_exclusivity_after.json", "after",
                                    release_grace_seconds=30)


def worker(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol, _, private, profile = validate_bindings(
        protocol_path, args.public_manifest.resolve(strict=True),
        args.private_map.resolve(strict=True), getattr(args, "eleven_seed_authorization", None),
        getattr(args, "evaluation_recovery_authorization", None)
    )
    evaluator_repo = args.evaluator_repo.resolve(strict=True)
    evaluator_ok(evaluator_repo)
    cache_root = args.cache_root.resolve(strict=True)
    if cache_root != Path(profile["cache_root"]):
        raise RuntimeError("worker evaluator cache differs from authorized profile")
    eval_root = Path(protocol["paths"]["formal_output_root"]) / profile["evaluation_dir"]
    jobs = [job for job in private["jobs"] if job["gpu_index"] == args.gpu_index]
    expected_for_gpu = sum(job["gpu_index"] == args.gpu_index for job in private["jobs"])
    if len(jobs) != expected_for_gpu or expected_for_gpu not in {40, 41, 44}:
        raise RuntimeError("GPU blind queue length mismatch")
    for job in sorted(jobs, key=lambda item: item["queue_index"]):
        run_one(protocol_path, protocol, job, evaluator_repo, cache_root, eval_root)


def launch(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    public_path = args.public_manifest.resolve(strict=True)
    private_path = args.private_map.resolve(strict=True)
    protocol, _, _, profile = validate_bindings(
        protocol_path, public_path, private_path, getattr(args, "eleven_seed_authorization", None),
        getattr(args, "evaluation_recovery_authorization", None)
    )
    evaluator_repo = args.evaluator_repo.resolve(strict=True)
    evaluator_ok(evaluator_repo)
    cache_root = args.cache_root.resolve(strict=True)
    if cache_root != Path(profile["cache_root"]):
        raise RuntimeError("launcher evaluator cache differs from authorized profile")
    runtime = load(Path(protocol["assets"]["runtime_manifest"]["path"]))
    experiment.validate_runtime(runtime)
    if experiment.compute_apps():
        raise RuntimeError("blind evaluation launch requires six exclusive GPUs")
    eval_root = Path(protocol["paths"]["formal_output_root"]) / profile["evaluation_dir"]
    for name in ("jobs", "receipts", "seals", "logs"):
        (eval_root / name).mkdir(exist_ok=False)
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    processes = []
    for gpu in range(6):
        command = [python, str(Path(__file__).resolve()), "worker", "--protocol", str(protocol_path),
                   "--public-manifest", str(public_path), "--private-map", str(private_path),
                   "--evaluator-repo", str(evaluator_repo), "--cache-root", str(cache_root),
                   "--gpu-index", str(gpu)]
        if getattr(args, "eleven_seed_authorization", None) is not None:
            command += ["--eleven-seed-authorization",
                        str(args.eleven_seed_authorization.resolve(strict=True))]
        if getattr(args, "evaluation_recovery_authorization", None) is not None:
            command += ["--evaluation-recovery-authorization",
                        str(args.evaluation_recovery_authorization.resolve(strict=True))]
        handle = (eval_root / "logs" / f"gpu{gpu}.queue.log").open("xb")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=os.environ.copy(),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((gpu, process, handle))
    failures = []
    for gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"gpu_index": gpu, "exit_code": code})
    receipts = sorted((eval_root / "receipts").glob("*.json"))
    validated = [path for path in receipts
                 if load(path).get("status") == "VALIDATED_UNSEALED"
                 and load(path).get("evaluation_recovery_authorization_sha256")
                 == profile["recovery_authorization_sha256"]]
    if failures or len(validated) != profile["job_count"]:
        experiment.atomic_json(eval_root / "evaluation_matrix_failure_receipt.json", {
            "schema": "ect.q256.fresh-crossed-switch-evaluation-matrix/v1", "status": "FAIL",
            "worker_failures": failures, "validated_receipts": len(validated),
            "expected_receipts": profile["job_count"], "automatic_retry_count": 0,
            "eleven_seed_authorization_sha256": profile["authorization_sha256"],
            "evaluation_recovery_authorization_sha256":
                profile["recovery_authorization_sha256"],
        })
        raise RuntimeError("blind evaluation matrix failed closed; no decoding permitted")
    for path in validated:
        record = load(path)
        experiment.atomic_json(eval_root / "seals" / f"{record['opaque_id']}.json", {
            "schema": "ect.q256.fresh-crossed-switch-evaluation-job-seal/v1",
            "status": "SEALED_PASS", "opaque_id": record["opaque_id"],
            "validation_receipt_sha256": sha256_file(path), "result_values_disclosed": False,
            "evaluation_recovery_authorization_sha256":
                profile["recovery_authorization_sha256"],
        })
    seal = {
        "schema": "ect.q256.fresh-crossed-switch-evaluation-matrix-seal/v1",
        "status": "SEALED_PASS", "sealed_jobs": profile["job_count"],
        "expected_jobs": profile["job_count"],
        "protocol_sha256": sha256_file(protocol_path),
        "eleven_seed_authorization_sha256": profile["authorization_sha256"],
        "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        "public_manifest_sha256": sha256_file(public_path),
        "private_map_sha256": sha256_file(private_path), "decoded": False,
        "automatic_retry_count": 0,
    }
    experiment.atomic_json(eval_root / "evaluation_matrix_seal.json", seal)
    print(json.dumps({"status": "SEALED_PASS", "sealed_jobs": profile["job_count"],
                      "decoded": False}))


def decode(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol, _, private, profile = validate_bindings(
        protocol_path, args.public_manifest.resolve(strict=True),
        args.private_map.resolve(strict=True), getattr(args, "eleven_seed_authorization", None),
        getattr(args, "evaluation_recovery_authorization", None)
    )
    eval_root = Path(protocol["paths"]["formal_output_root"]) / profile["evaluation_dir"]
    seal = load(eval_root / "evaluation_matrix_seal.json")
    if (seal.get("status") != "SEALED_PASS"
            or seal.get("sealed_jobs") != profile["job_count"]
            or seal.get("eleven_seed_authorization_sha256") != profile["authorization_sha256"]
            or seal.get("evaluation_recovery_authorization_sha256")
            != profile["recovery_authorization_sha256"]):
        raise RuntimeError("all amended jobs must be sealed before decoding")
    decoded = []
    for job in private["jobs"]:
        opaque = job["opaque_id"]
        job_seal = load(eval_root / "seals" / f"{opaque}.json")
        receipt_path = eval_root / "receipts" / f"{opaque}.json"
        if (job_seal.get("status") != "SEALED_PASS"
                or job_seal.get("validation_receipt_sha256") != sha256_file(receipt_path)):
            raise RuntimeError(f"job seal mismatch: {opaque}")
        job_dir = eval_root / "jobs" / opaque
        decoded.append({
            "seed": job["seed"], "kind": job["kind"], "cell": job["cell"],
            "budget_kimg": job["budget_kimg"], "nfe": job["nfe"],
            "fid50k_full": metric_value(job_dir / "metric-fid50k_full.jsonl", "fid50k_full"),
            "kid50k_full": metric_value(job_dir / "metric-kid50k_full.jsonl", "kid50k_full"),
            "opaque_id": opaque,
        })
    payload = {
        "schema": "ect.q256.fresh-crossed-switch-decoded-results/v1", "status": "PASS",
        "decoded_after_full_seal": True, "job_count": profile["job_count"],
        "eleven_seed_authorization_sha256": profile["authorization_sha256"],
        "evaluation_recovery_authorization_sha256": profile["recovery_authorization_sha256"],
        "matrix_seal_sha256": sha256_file(eval_root / "evaluation_matrix_seal.json"),
        "protocol_sha256": sha256_file(protocol_path), "results": decoded,
    }
    experiment.atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "decoded_jobs": profile["job_count"],
                      "output": str(args.output)}))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subs = root.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--public-manifest", type=Path, required=True)
    prep.add_argument("--private-map", type=Path, required=True)
    prep.add_argument("--eleven-seed-authorization", type=Path)
    prep.add_argument("--evaluation-recovery-authorization", type=Path)
    prep.set_defaults(func=prepare)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", type=Path, required=True)
    common.add_argument("--public-manifest", type=Path, required=True)
    common.add_argument("--private-map", type=Path, required=True)
    common.add_argument("--eleven-seed-authorization", type=Path)
    common.add_argument("--evaluation-recovery-authorization", type=Path)
    work = subs.add_parser("worker", parents=[common])
    work.add_argument("--evaluator-repo", type=Path, required=True)
    work.add_argument("--cache-root", type=Path, required=True)
    work.add_argument("--gpu-index", type=int, choices=range(6), required=True)
    work.set_defaults(func=worker)
    launch_parser = subs.add_parser("launch", parents=[common])
    launch_parser.add_argument("--evaluator-repo", type=Path, required=True)
    launch_parser.add_argument("--cache-root", type=Path, required=True)
    launch_parser.set_defaults(func=launch)
    decoder = subs.add_parser("decode", parents=[common])
    decoder.add_argument("--output", type=Path, required=True)
    decoder.set_defaults(func=decode)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
