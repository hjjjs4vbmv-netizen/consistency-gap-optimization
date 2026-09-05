#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import evaluation, experiment
from analysis.q256_switchpoint_sweep import companion_summary
from analysis.q256_switchpoint_sweep import train


METRICS = ("kid50k_full", "fid50k_full")
BLOCKS = ((50000, 99999), (100000, 149999), (150000, 199999), (200000, 249999))


def formal_jobs(training_root: Path) -> list[dict]:
    jobs = []
    for seed in range(81, 93):
        cells = [("CTRL", kimg, "primary") for kimg in (640, 768, 896, 1024)]
        cells += [(f"BA{s}", s + 512, "primary") for s in (128, 256, 384, 512)]
        cells += [(f"BA{s}", 1024, "secondary") for s in (128, 256, 384)]
        for trajectory, kimg, role in cells:
            jobs.append({"seed": seed, "trajectory": trajectory, "kimg": kimg, "role": role,
                         "nfe": 1, "checkpoint": str(training_root / f"seed{seed:03d}" / trajectory / f"kimg{kimg:04d}" / "network-snapshot.pkl")})
    return jobs


def companion_jobs(training_root: Path) -> list[dict]:
    jobs = []
    for trajectory in ("CTRL", "BA512"):
        checkpoint = training_root / "seed081" / trajectory / "kimg1024" / "network-snapshot.pkl"
        for block, (start, end) in enumerate(BLOCKS, 1):
            jobs.append({"seed": 81, "trajectory": trajectory, "kimg": 1024, "block": block,
                         "sample_start": start, "sample_end": end, "gpu": len(jobs), "checkpoint": str(checkpoint)})
    return jobs


def prepare(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text()); eval_root = Path(protocol["paths"]["evaluation"])
    training_root = Path(protocol["paths"]["training"])
    matrix = json.loads((training_root / "training_matrix_receipt.json").read_text())
    if (matrix.get("jobs") != 84 or matrix.get("expected_jobs") != 84
            or matrix.get("protocol_sha256") != experiment.sha256_file(protocol_path)
            or matrix.get("status") not in {"PASS", "COMPLETE_WITH_FAILURES"}):
        raise RuntimeError("evaluation requires a complete protocol-bound training matrix")
    formal_root = eval_root / "formal"
    if formal_root.exists():
        raise RuntimeError("formal evaluation destination already exists")
    aliases = formal_root / "checkpoints"; aliases.mkdir(parents=True)
    logical = formal_jobs(Path(protocol["paths"]["training"]))
    random.Random(20260903).shuffle(logical)
    private_jobs, public_jobs = [], []
    for queue, job in enumerate(logical):
        opaque = secrets.token_hex(12); source = Path(job.pop("checkpoint"))
        terminal = training_root / "terminal" / f"continuation-seed{job['seed']:03d}-{job['trajectory']}.json"
        training_receipt = json.loads(terminal.read_text())
        training_pass = training_receipt.get("status") == "PASS"
        alias = aliases / f"{opaque}.pkl"; available = training_pass and source.is_file() and not source.is_symlink()
        checkpoint_sha = None
        if available:
            source = source.resolve(strict=True); os.link(source, alias)
            checkpoint_sha = experiment.sha256_file(alias)
        shared = {"queue_index": queue, "opaque_id": opaque, "gpu_index": queue % 8,
                  "checkpoint_alias": str(alias), "checkpoint_sha256": checkpoint_sha}
        unavailable_cause = None if available else training_receipt.get("root_cause", "CHECKPOINT_ARTIFACT_MISSING")
        private_jobs.append({**shared, **job, "training_status": "AVAILABLE" if available else "TRAINING_UNAVAILABLE",
                             "training_root_cause": unavailable_cause,
                             "training_terminal_sha256": experiment.sha256_file(terminal),
                             "protocol_sha256": experiment.sha256_file(protocol_path)})
        public_jobs.append(shared)
    private = eval_root / "private_map.json"
    experiment.atomic_json(private, {"status": "SEALED_PRIVATE_MAP", "job_count": 132,
                           "protocol_sha256": experiment.sha256_file(protocol_path), "jobs": private_jobs})
    os.chmod(private, 0o400)
    experiment.atomic_json(formal_root / "public_manifest.json", {
        "status": "FROZEN_NOT_RUN", "job_count": 132, "private_map_sha256": experiment.sha256_file(private),
        "protocol_sha256": experiment.sha256_file(protocol_path), "identities_disclosed": False, "jobs": public_jobs})


def _formal_attempt(protocol_path: Path, job: dict, evaluator_repo: Path, cache: Path, root: Path) -> None:
    protocol = train.bound_protocol(json.loads(protocol_path.read_text()))
    evaluation.run_one(protocol_path, protocol, job, evaluator_repo, cache, root)


def formal_worker(protocol_path: Path, evaluator_repo: Path, gpu: int) -> None:
    protocol = json.loads(protocol_path.read_text()); root = Path(protocol["paths"]["evaluation"]) / "formal"
    evaluation.evaluator_ok(evaluator_repo)
    private = json.loads((Path(protocol["paths"]["evaluation"]) / "private_map.json").read_text())
    cache = Path(protocol["paths"]["assets"]) / "cache"; (root / "terminal").mkdir(exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)
    for job in private["jobs"]:
        if job["gpu_index"] != gpu:
            continue
        terminal_path = root / "terminal" / f"{job['opaque_id']}.json"
        if job["training_status"] != "AVAILABLE":
            experiment.atomic_json(terminal_path, {"status": "EXHAUSTED_FAILURE",
                "opaque_id": job["opaque_id"], "attempts": 0, "errors": [],
                "root_cause": job["training_root_cause"],
                "training_terminal_sha256": job["training_terminal_sha256"]})
            continue
        status, attempts, errors, root_cause = "PASS", 1, [], None
        try:
            _formal_attempt(protocol_path, job, evaluator_repo, cache, root)
        except Exception as error:
            errors.append(repr(error)); attempts = 2
            first = root / "jobs" / job["opaque_id"]
            if first.exists():
                shutil.move(first, root / "attempts" / f"{job['opaque_id']}-attempt1")
            first_receipt = root / "receipts" / f"{job['opaque_id']}.json"
            if first_receipt.exists():
                shutil.move(
                    first_receipt,
                    root / "attempts" / f"{job['opaque_id']}-attempt1-receipt.json",
                )
            try:
                _formal_attempt(protocol_path, job, evaluator_repo, cache, root)
            except Exception as error:
                errors.append(repr(error))
                status, root_cause = "EXHAUSTED_FAILURE", "EVALUATION_FAILURE"
        experiment.atomic_json(terminal_path, {"status": status, "opaque_id": job["opaque_id"],
                               "attempts": attempts, "errors": errors, "root_cause": root_cause})


def seal(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text()); root = Path(protocol["paths"]["evaluation"]) / "formal"
    public = json.loads((root / "public_manifest.json").read_text()); terminals = []
    private = json.loads((Path(protocol["paths"]["evaluation"]) / "private_map.json").read_text())
    private_by_id = {job["opaque_id"]: job for job in private["jobs"]}
    (root / "seals").mkdir(exist_ok=True)
    for job in public["jobs"]:
        terminal_path = root / "terminal" / f"{job['opaque_id']}.json"
        terminal = json.loads(terminal_path.read_text()); terminals.append(terminal)
        if terminal["status"] not in {"PASS", "EXHAUSTED_FAILURE"}:
            raise RuntimeError("invalid formal terminal status")
        if terminal["status"] == "PASS":
            if terminal.get("attempts") not in {1, 2} or terminal.get("root_cause") is not None:
                raise RuntimeError("invalid PASS terminal documentation")
            if not (root / "receipts" / f"{job['opaque_id']}.json").is_file():
                raise RuntimeError("PASS job lacks validation receipt")
        elif terminal.get("attempts") == 0:
            source = private_by_id[job["opaque_id"]]
            if (not terminal.get("root_cause") or terminal.get("training_terminal_sha256")
                    != source.get("training_terminal_sha256")):
                raise RuntimeError("training-unavailable failure lacks receipt binding")
        elif (terminal.get("attempts") != 2 or terminal.get("root_cause") != "EVALUATION_FAILURE"
              or len(terminal.get("errors", [])) != 2):
            raise RuntimeError("evaluation failure is not documented after two attempts")
        experiment.atomic_json(root / "seals" / f"{job['opaque_id']}.json", {
            "status": "SEALED_PASS" if terminal["status"] == "PASS" else "SEALED_FAILURE",
            "opaque_id": job["opaque_id"], "terminal_sha256": experiment.sha256_file(terminal_path)})
    failures = sum(row["status"] != "PASS" for row in terminals)
    experiment.atomic_json(root / "matrix_seal.json", {
        "status": "SEALED_PASS" if failures == 0 else "SEALED_WITH_DOCUMENTED_FAILURES",
        "expected_jobs": 132, "sealed_jobs": len(terminals), "failures": failures,
        "protocol_sha256": experiment.sha256_file(protocol_path), "decoded": False})


def launch(protocol_path: Path, evaluator_repo: Path) -> None:
    protocol = json.loads(protocol_path.read_text()); root = Path(protocol["paths"]["evaluation"]) / "formal"
    for name in ("jobs", "receipts", "logs"):
        (root / name).mkdir(exist_ok=False)
    runtime = json.loads((Path(protocol["paths"]["runtime"]) / "runtime-manifest.json").read_text())
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python"); processes = []
    for gpu in range(8):
        log = (root / "logs" / f"gpu{gpu}.log").open("xb")
        command = [python, str(Path(__file__).resolve()), "worker", "--protocol", str(protocol_path),
                   "--evaluator-repo", str(evaluator_repo), "--gpu", str(gpu)]
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    for process, log in processes:
        code = process.wait(); log.close()
        if code:
            raise RuntimeError(f"evaluation worker exited {code}")
    seal(protocol_path)


def decode(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text()); eval_root = Path(protocol["paths"]["evaluation"])
    root = eval_root / "formal"; output = Path(protocol["paths"]["analysis"]) / "decoded_results.json"
    if output.exists():
        raise RuntimeError("formal matrix may be decoded only once")
    matrix = json.loads((root / "matrix_seal.json").read_text())
    if matrix["status"] not in {"SEALED_PASS", "SEALED_WITH_DOCUMENTED_FAILURES"} or matrix["sealed_jobs"] != 132:
        raise RuntimeError("decode requires terminal 132-job seal")
    private = json.loads((eval_root / "private_map.json").read_text()); rows = []
    for job in private["jobs"]:
        terminal = json.loads((root / "terminal" / f"{job['opaque_id']}.json").read_text())
        row = {key: job[key] for key in ("seed", "trajectory", "kimg", "role", "opaque_id")}
        row.update(status=terminal["status"], root_cause=terminal.get("root_cause"))
        if terminal["status"] == "PASS":
            job_dir = root / "jobs" / job["opaque_id"]
            row.update({metric: evaluation.metric_value(job_dir / f"metric-{metric}.jsonl", metric) for metric in METRICS})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    experiment.atomic_json(output, {"status": "PASS", "matrix_seal_sha256": experiment.sha256_file(root / "matrix_seal.json"), "results": rows})


def companion_command(protocol: dict, job: dict, evaluator_repo: Path, output: Path) -> list[str]:
    runtime = json.loads((Path(protocol["paths"]["runtime"]) / "runtime-manifest.json").read_text())
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    return [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", f"--master_port={53100 + job['gpu']}",
            str(evaluator_repo / "ct_eval.py"), "--resume", job["checkpoint"], "--outdir", str(output), "--nosubdir",
            "--data", protocol["assets"]["dataset"]["path"], "--cond=False", "--arch=ddpmpp", "--precond=ct", "--dropout=0.2",
            "--augment=0", "--xflip=False", "--fp16=False", "--cache=True", "--workers=1", "--eval-batch=512",
            "--metric-generator-batch=128", "--nfe=1", "--metrics=kid50k_full,fid50k_full", "--metric-repeats=1",
            f"--sample-seeds={job['sample_start']}-{job['sample_end']}", "--seed=20260730", "--retain-generated-artifacts",
            f"--desc=companion-{job['trajectory']}-b{job['block']}"]


def _validate_companion(job: dict, output: Path) -> dict:
    options = json.loads((output / "training_options.json").read_text())
    expected = {"batch_size": 512, "metrics": list(METRICS), "metric_repeats": 1,
                "metric_generator_batch": 128, "retain_generated_artifacts": True,
                "seed": 20260730, "mid_t": []}
    if any(options.get(key) != value for key, value in expected.items()):
        raise RuntimeError("companion evaluator option mismatch")
    seeds = options.get("sample_seeds", [])
    if len(seeds) != 50000 or seeds[0] != job["sample_start"] or seeds[-1] != job["sample_end"]:
        raise RuntimeError("companion generation block mismatch")
    required = ["generated-features-kid50k_full-repeat00.npy", "generated-features-fid50k_full-repeat00.npy"]
    hashes = [experiment.sha256_file(output / name) for name in required]
    if hashes[0] != hashes[1]:
        raise RuntimeError("companion KID/FID feature identity mismatch")
    fid = output / required[1]; temporary = output / ".shared-feature.tmp"
    os.link(output / required[0], temporary); os.replace(temporary, fid)
    values = {metric: evaluation.metric_value(output / f"metric-{metric}.jsonl", metric) for metric in METRICS}
    return {"status": "PASS", "sample_seed_range": [job["sample_start"], job["sample_end"]],
            "generated_feature_sha256": hashes[0], "values": values}


def companion_worker(protocol_path: Path, evaluator_repo: Path, gpu: int) -> None:
    protocol = json.loads(protocol_path.read_text()); evaluation.evaluator_ok(evaluator_repo)
    formal = Path(protocol["paths"]["evaluation"]) / "formal"
    seal_record = json.loads((formal / "matrix_seal.json").read_text())
    if not seal_record["status"].startswith("SEALED_") or not (Path(protocol["paths"]["analysis"]) / "decoded_results.json").is_file():
        raise RuntimeError("companion requires sealed and decoded formal matrix")
    root = Path(protocol["paths"]["evaluation"]) / "companion"; job = companion_jobs(Path(protocol["paths"]["training"]))[gpu]
    name = f"{job['trajectory']}-b{job['block']}"; destination = root / "jobs" / name
    bound = train.bound_protocol(protocol); runtime = json.loads((Path(protocol["paths"]["runtime"]) / "runtime-manifest.json").read_text())
    env = experiment.cell_environment(gpu, runtime); env["DNNLIB_CACHE_DIR"] = str(Path(protocol["paths"]["assets"]) / "cache")
    errors = []
    for attempt in (1, 2):
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "launcher.log").open("xb") as log:
            code = subprocess.run(companion_command(bound, job, evaluator_repo, destination), env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        try:
            if code:
                raise RuntimeError(f"evaluator exit {code}")
            receipt = _validate_companion(job, destination); receipt.update(job={k: job[k] for k in ("seed", "trajectory", "kimg", "block")}, attempts=attempt)
            experiment.atomic_json(root / "receipts" / f"{name}.json", receipt); return
        except Exception as error:
            errors.append(repr(error))
            if attempt == 1:
                shutil.move(destination, root / "attempts" / f"{name}-attempt1")
    experiment.atomic_json(root / "receipts" / f"{name}.json", {
        "status": "EXHAUSTED_FAILURE", "job": name, "attempts": 2, "errors": errors})


def companion_launch(protocol_path: Path, evaluator_repo: Path) -> None:
    protocol = json.loads(protocol_path.read_text()); root = Path(protocol["paths"]["evaluation"]) / "companion"
    for name in ("jobs", "receipts", "attempts", "logs"):
        (root / name).mkdir(parents=True, exist_ok=False)
    runtime = json.loads((Path(protocol["paths"]["runtime"]) / "runtime-manifest.json").read_text())
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python"); processes = []
    for gpu in range(8):
        log = (root / "logs" / f"gpu{gpu}.log").open("xb")
        command = [python, str(Path(__file__).resolve()), "companion-worker", "--protocol", str(protocol_path),
                   "--evaluator-repo", str(evaluator_repo), "--gpu", str(gpu)]
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    for process, log in processes:
        code = process.wait(); log.close()
        if code:
            raise RuntimeError(f"companion worker exited {code}")
    companion_summary.write(
        Path(protocol["paths"]["analysis"]) / "decoded_results.json",
        root / "receipts", Path(protocol["paths"]["analysis"]) / "generation_noise_companion.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "launch", "worker", "seal", "decode", "companion", "companion-worker", "plan"))
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json")); parser.add_argument("--evaluator-repo", type=Path)
    parser.add_argument("--gpu", type=int, choices=range(8)); args = parser.parse_args(); protocol = args.protocol.resolve(strict=True)
    if args.command == "plan":
        value = json.loads(protocol.read_text()); print(json.dumps({"formal": formal_jobs(Path(value["paths"]["training"])), "companion": companion_jobs(Path(value["paths"]["training"]))}, indent=2)); return 0
    if args.command == "prepare": prepare(protocol)
    elif args.command == "seal": seal(protocol)
    elif args.command == "decode": decode(protocol)
    elif args.command == "worker": formal_worker(protocol, args.evaluator_repo.resolve(strict=True), args.gpu)
    elif args.command == "launch": launch(protocol, args.evaluator_repo.resolve(strict=True))
    elif args.command == "companion-worker": companion_worker(protocol, args.evaluator_repo.resolve(strict=True), args.gpu)
    elif args.command == "companion": companion_launch(protocol, args.evaluator_repo.resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
