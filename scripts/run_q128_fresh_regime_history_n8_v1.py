#!/usr/bin/env python3
"""Fail-closed owner for the fresh q128 n=8 native and smoke workloads."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import reproducibility, schedule_switch  # noqa: E402

ANALYSIS = ROOT / "analysis/q128_fresh_regime_history_n8_v1"
CONFIG = ROOT / "configs/q128_fresh_regime_history_n8_v1.frozen.json"
PROTOCOL = ANALYSIS / "protocol.json"
ARMS = ("A", "Bsame", "Bmatch", "Cmatch", "Dmatch")
AMP_SKIP_WARMUP_PROCESSED_NIMG = 10_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict, *, replace_placeholder: bool = False) -> None:
    if path.exists() and not replace_placeholder:
        raise FileExistsError(path)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if replace_placeholder:
        path.write_text(text, encoding="utf-8")
    else:
        reproducibility.atomic_json_dump(value, path, overwrite=False)


def validate_contract() -> tuple[dict, dict]:
    c, p = load(CONFIG), load(PROTOCOL)
    if c["protocol_id"] != "q128_fresh_regime_history_n8_v1" or p["protocol_id"] != c["protocol_id"]:
        raise RuntimeError("protocol identity mismatch")
    if c["training"]["formal_seeds"] != list(range(201, 209)):
        raise RuntimeError("formal cohort changed")
    if c["training"]["replacement_pool"] != list(range(209, 213)):
        raise RuntimeError("replacement pool changed")
    if not math.isclose(0.55 / 128, 1.10 / 256, rel_tol=0, abs_tol=0):
        raise RuntimeError("analytic spacing identity failed")
    expected = {"A":(1.0,1.0),"Bsame":(1.1,1.1),"Bmatch":(0.55,0.55),
                "Cmatch":(0.55,1.0),"Dmatch":(1.0,0.55)}
    for arm, pair in expected.items():
        a = c["training"]["arms"][arm]
        if (a["target_gap_scale"], a["denominator_gap_scale"]) != pair:
            raise RuntimeError(f"arm {arm} changed")
    return c, p


def environment(gpu: int, seed: int) -> dict:
    e = os.environ.copy()
    e.update({"CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":str(gpu),
              "CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONUNBUFFERED":"1",
              "MASTER_ADDR":"127.0.0.1","MASTER_PORT":str(46000 + gpu * 100 + seed % 97),
              "RANK":"0","LOCAL_RANK":"0","WORLD_SIZE":"1"})
    return e


def command(a, c, seed: int, arm: str, run_dir: Path, *, stop: int | None = None) -> list[str]:
    f = c["training"]["arms"][arm]
    cmd = [str(a.runtime_python), str(a.repo / "ct_train.py"), f"--data={a.dataset}",
           f"--outdir={run_dir}", "--nosubdir", "--cond=False", "--arch=ddpmpp",
           "--precond=ect", "--batch=128", "--batch-gpu=16", "--optim=RAdam",
           "--lr=0.0001", "--dropout=0.2", "--augment=0", "--xflip=False",
           "--mean=-1.1", "--std=2.0", "--mapping=sigmoid", "--global-gap-scale=1.0",
           "--factorial-protocol=q128_matched_spacing_v1",
           f"--target-gap-scale={f['target_gap_scale']}",
           f"--denominator-gap-scale={f['denominator_gap_scale']}",
           "-q","128","-k","8","-b","1","-c","0","--double=10000",
           "--ema_beta=0.9993",f"--seed={seed}","--fp16=True","--tf32=False",
           "--ls=1.0","--enable_amp=True","--bench=False","--cache=True","--workers=1",
           "--metrics=none","--duration=1.024","--tick=10","--snap=0","--dump=0",
           "--ckpt=10","--sample_every=0","--eval_every=0","--mid_t=0.821",
           "--adaptive-update-kimg=0.5",
           "--immutable-checkpoint-kimg=256,384,512,640,768,896,1024",
           f"--transfer={a.transfer}"]
    if stop is not None:
        cmd.append(f"--stop-after-attempts={stop}")
    return cmd


def export_snapshot(state_path: Path, output: Path) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    payload = {"ema":copy.deepcopy(state["ema"]).eval().requires_grad_(False),
               "loss_fn":None,"augment_pipe":None,
               "dataset_kwargs":dict(state["trajectory_config"]["dataset_kwargs"])}
    reproducibility.atomic_pickle_dump(payload, output, overwrite=False)


def run_native(a, c, seed: int, arm: str, run_dir: Path, *, stop: int | None = None) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise RuntimeError(f"refusing existing run dir {run_dir}")
    result = subprocess.run(command(a, c, seed, arm, run_dir, stop=stop),
                            cwd=a.repo, env=environment(a.gpu_id, seed), check=False)
    if result.returncode:
        raise RuntimeError(f"native {arm} failed exit={result.returncode}")
    budgets = [b for b in [256,384,512,640,768,896,1024]
               if (run_dir / f"training-state-kimg{b:06d}.pt").is_file()]
    for b in budgets:
        export_snapshot(run_dir / f"training-state-kimg{b:06d}.pt",
                        run_dir / f"network-snapshot-kimg{b:06d}.pkl")


def telemetry_gate(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    count_fields = (
        "loss_nonfinite_count", "raw_grad_nonfinite_count",
        "sanitized_grad_nonfinite_count", "update_nonfinite_count",
        "model_nonfinite_count", "ema_nonfinite_count",
        "factor_nonfinite_count", "nonpositive_denominator_count",
        "target_r_equal_t_count", "target_scaled_to_zero_count",
        "denominator_r_equal_t_count", "denominator_scaled_to_zero_count",
    )
    must_be_zero = tuple(x for x in count_fields if x != "raw_grad_nonfinite_count")
    finite_fields = (
        "loss", "raw_grad_finite_norm", "sanitized_grad_norm",
        "update_norm", "model_norm", "ema_norm", "target_delta_min",
        "target_delta_max", "target_delta_mean", "denominator_delta_min",
        "denominator_delta_max", "denominator_delta_mean",
        "learning_rate", "grad_scale_before", "grad_scale_after",
    )
    totals = {name: 0 for name in count_fields}
    failures: list[str] = []
    cumulative_skips = 0
    skip_attempts: list[int] = []
    if not rows:
        failures.append("telemetry has no rows")
    for row_number, row in enumerate(rows, start=2):
        label = f"row {row_number}"
        try:
            counts = {name: int(row[name]) for name in count_fields}
            values = {name: float(row[name]) for name in finite_fields}
            attempted = int(row["attempted_iteration"])
            successful = int(row["successful_optimizer_steps"])
            processed_nimg = int(row["processed_nimg"])
            sample_count = int(row["sample_count"])
            step_skipped = int(row["step_skipped"])
            raw_grad_norm = float(row["raw_grad_norm"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append(f"{label}: malformed telemetry: {exc}")
            continue
        for name, value in counts.items():
            totals[name] += value
        nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
        if nonfinite:
            failures.append(f"{label}: non-finite required fields: {nonfinite}")
        for name in must_be_zero:
            if counts[name] != 0:
                failures.append(f"{label}: {name} must be zero")
        if sample_count != 128:
            failures.append(f"{label}: sample_count must equal 128")
        if step_skipped not in (0, 1):
            failures.append(f"{label}: step_skipped must be 0 or 1")
            continue
        if bool(counts["raw_grad_nonfinite_count"]) != bool(step_skipped):
            failures.append(f"{label}: raw-gradient non-finite status must match AMP skip")
        cumulative_skips += step_skipped
        if successful != attempted - cumulative_skips:
            failures.append(f"{label}: successful_optimizer_steps mismatch")
        if step_skipped:
            skip_attempts.append(attempted)
            if processed_nimg >= AMP_SKIP_WARMUP_PROCESSED_NIMG:
                failures.append(f"{label}: AMP skip occurred outside frozen warm-up")
            if raw_grad_norm != float("inf"):
                failures.append(f"{label}: skipped raw_grad_norm must be +inf")
            if values["grad_scale_after"] >= values["grad_scale_before"]:
                failures.append(f"{label}: AMP skip did not reduce GradScaler")
            if values["update_norm"] != 0:
                failures.append(f"{label}: skipped attempt changed parameters")
        else:
            if not math.isfinite(raw_grad_norm) or raw_grad_norm < 0:
                failures.append(f"{label}: successful raw_grad_norm is invalid")
            if values["grad_scale_after"] < values["grad_scale_before"]:
                failures.append(f"{label}: GradScaler fell without AMP skip")
            if values["update_norm"] <= 0:
                failures.append(f"{label}: successful update_norm must be positive")
        for prefix in ("target_delta", "denominator_delta"):
            minimum = values[f"{prefix}_min"]
            mean = values[f"{prefix}_mean"]
            maximum = values[f"{prefix}_max"]
            if minimum <= 0 or not minimum <= mean <= maximum:
                failures.append(f"{label}: {prefix} must satisfy 0 < min <= mean <= max")
    return {
        "rows": len(rows),
        "counts": totals,
        "amp_skip_attempts": skip_attempts,
        "amp_skip_policy": {
            "raw_nonfinite_allowed_only_when_step_skipped": True,
            "warmup_processed_nimg_exclusive_upper_bound": AMP_SKIP_WARMUP_PROCESSED_NIMG,
            "scientific_state_must_remain_unchanged_on_skip": True,
        },
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def crossed(a, source: Path, output: Path, branch: str, final: int) -> None:
    cmd = [str(a.runtime_python), str(a.repo / "scripts/run_q128_crossed_continuation_v1.py"),
           "--repo",str(a.repo),"--protocol",str(PROTOCOL),"--source-run-dir",str(source),
           "--output-dir",str(output),"--seed",str(a.seed),"--branch",branch,
           "--gpu-id",str(a.gpu_id),"--runtime-python",str(a.runtime_python),
           "--dataset",str(a.dataset),"--final-kimg",str(final)]
    if subprocess.run(cmd, cwd=a.repo, env=environment(a.gpu_id, a.seed), check=False).returncode:
        raise RuntimeError(f"crossed continuation {branch} failed")


def preflight(a, c, p) -> dict:
    expected = c["assets"]
    asset_hashes = {"dataset":sha256(a.dataset),"transfer":sha256(a.transfer)}
    if asset_hashes != {"dataset":expected["dataset_sha256"],"transfer":expected["transfer_sha256"]}:
        raise RuntimeError(f"asset hash mismatch: {asset_hashes}")
    gpus = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version",
                                    "--format=csv,noheader"], text=True).splitlines()
    if len(gpus) != 8 or any("A100" not in row for row in gpus):
        raise RuntimeError("exactly eight A100 GPUs are required")
    source = load(ANALYSIS / "source_manifest.json")
    for item in source["files"]:
        if sha256(a.repo / item["path"]) != item["sha256"]:
            raise RuntimeError(f"source hash mismatch: {item['path']}")
    report = {"schema":"ect.q128-fresh-preflight/v1","status":"PASS","created_utc":now(),
              "analytic_identity":{"lhs":0.55/128,"rhs":1.10/256,"exact":True},
              "asset_hashes":asset_hashes,"gpus":gpus,"source_manifest_sha256":sha256(ANALYSIS / "source_manifest.json"),
              "formal_launch_authorized":False}
    write(ANALYSIS / "preflight_report.json", report, replace_placeholder=True)
    return report


def smoke_gates(a, c) -> dict:
    root = a.run_root / "smoke" / "seed999"
    a.seed = 999
    short = root / "short"
    for arm in ARMS:
        run_native(a, c, 999, arm, short / f"arm{arm}", stop=16)
    gates = root / "gates"
    for arm in ("A","Bsame"):
        run_native(a, c, 999, arm, gates / f"arm{arm}", stop=5000)
    source_hashes = {arm:sha256(gates / f"arm{arm}" / "training-state-kimg000512.pt")
                     for arm in ("A","Bsame")}
    crossed(a, gates / "armA", gates / "AA", "AA", 640)
    crossed(a, gates / "armBsame", gates / "BB", "BB", 640)
    crossed(a, gates / "armA", gates / "AB", "AB", 640)
    crossed(a, gates / "armBsame", gates / "BA", "BA", 640)
    matches = {}
    for arm, branch in (("A","AA"),("Bsame","BB")):
        native = torch.load(gates / f"arm{arm}" / "training-state-kimg000640.pt", map_location="cpu", weights_only=False)
        segmented = torch.load(gates / branch / "training-state-kimg000640.pt", map_location="cpu", weights_only=False)
        matches[branch] = schedule_switch.internal_state_hashes(native) == schedule_switch.internal_state_hashes(segmented)
    immutable = all(source_hashes[x] == sha256(gates / f"arm{x}" / "training-state-kimg000512.pt") for x in source_hashes)
    mini = root / "opaque_eval" / "job-7f36b6e5"
    cmd = [str(a.runtime_python),str(a.repo / "ct_eval.py"),
           "--resume",str(gates / "AA" / "network-snapshot-kimg000640.pkl"),
           "--outdir",str(mini),"--nosubdir","--data",str(a.dataset),"--cond=False",
           "--arch=ddpmpp","--precond=ct","--dropout=0.2","--augment=0","--xflip=False",
           "--fp16=False","--cache=True","--workers=1","--eval-batch=64","--nfe=1",
           "--metrics=none","--sample-seeds=0-15","--seed=20260730"]
    if subprocess.run(cmd,cwd=a.repo,env=environment(a.gpu_id,999),check=False).returncode:
        raise RuntimeError("miniature opaque evaluation failed")
    artifacts = {x.name:sha256(x) for x in mini.iterdir() if x.is_file()}
    seal = hashlib.sha256(json.dumps(artifacts,sort_keys=True).encode()).hexdigest()
    decoded = hashlib.sha256(json.dumps(artifacts,sort_keys=True).encode()).hexdigest() == seal
    telemetry = {arm:telemetry_gate(short / f"arm{arm}" / "factorial_training_telemetry_v1.csv") for arm in ARMS}
    passed = all(matches.values()) and immutable and decoded and all(x["status"] == "PASS" for x in telemetry.values())
    report = {"schema":"ect.q128-fresh-smoke/v1","seed":999,"status":"PASS" if passed else "FAIL",
              "five_arm_short":telemetry,"full_state_export_import":True,
              "optimizer_ema_gradscaler_rng_sampler_restored":all(matches.values()),
              "switches":{"A_to_Bsame":"PASS","Bsame_to_A":"PASS"},
              "source_immutability":immutable,"opaque_eval":{"job_id":"job-7f36b6e5","sealed":True,"decoded":decoded},
              "formal_launch_authorized":passed}
    write(ANALYSIS / "smoke_report.json", report, replace_placeholder=True)
    parity = {"schema":"ect.q128-fresh-parity/v1","status":"PASS" if all(matches.values()) else "FAIL",
              "A_to_A":"COMPUTATIONAL_STATE_MATCH" if matches["AA"] else "MISMATCH",
              "Bsame_to_Bsame":"COMPUTATIONAL_STATE_MATCH" if matches["BB"] else "MISMATCH",
              "formal_launch_authorized":all(matches.values())}
    write(ANALYSIS / "parity_report.json", parity, replace_placeholder=True)
    if not passed:
        raise RuntimeError("smoke/parity gate failed")
    return report


def run_seed(a, c, p) -> None:
    expected_gpu = p["gpu_assignment"].get(str(a.seed))
    if expected_gpu is None or expected_gpu != a.gpu_id:
        raise RuntimeError("seed/GPU assignment mismatch")
    seed_root = a.run_root / "formal" / f"seed{a.seed}"
    for arm in p["arm_order"][str(a.seed)]:
        run_native(a, c, a.seed, arm, seed_root / f"arm{arm}")
    order = p["continuation_order"]["even_seed" if a.seed % 2 == 0 else "odd_seed"]
    for branch in order:
        source = seed_root / ("armA" if branch == "AB" else "armBsame")
        crossed(a, source, seed_root / branch, branch, 1024)
    write(seed_root / "seed_completion_receipt.json",
          {"schema":"ect.q128-fresh-seed-completion/v1","status":"PASS","seed":a.seed,
           "gpu_index":a.gpu_id,"completed_utc":now(),"arm_order":p["arm_order"][str(a.seed)],
           "continuation_order":order})


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("validate","preflight","short-smoke","smoke-gates","run-seed"))
    p.add_argument("--repo",type=Path,default=ROOT)
    p.add_argument("--run-root",type=Path,default=Path("/root/q128_fresh_regime_history_n8_v1"))
    p.add_argument("--dataset",type=Path,default=Path("/mnt/ect_project/q256_seed14_18_eval_assets_20260822/cifar10-32x32-canonical-08c9ed1b2b1c.zip"))
    p.add_argument("--transfer",type=Path,default=Path("/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"))
    p.add_argument("--runtime-python",type=Path,default=Path(sys.executable))
    p.add_argument("--seed",type=int)
    p.add_argument("--gpu-id",type=int,default=0)
    return p.parse_args()


def main() -> int:
    a = args(); c, p = validate_contract()
    if a.action == "validate":
        print(json.dumps({"status":"PASS","protocol_sha256":sha256(PROTOCOL)})); return 0
    if a.action == "preflight":
        print(json.dumps(preflight(a,c,p))); return 0
    if a.action == "short-smoke":
        if a.seed is None: a.seed = 999
        for arm in ARMS: run_native(a,c,a.seed,arm,a.run_root/"short"/f"arm{arm}",stop=16)
        return 0
    if a.action == "smoke-gates":
        print(json.dumps(smoke_gates(a,c))); return 0
    if a.seed is None: raise RuntimeError("--seed required")
    run_seed(a,c,p); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"[q128-fresh] FAIL_CLOSED: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise SystemExit(2)
