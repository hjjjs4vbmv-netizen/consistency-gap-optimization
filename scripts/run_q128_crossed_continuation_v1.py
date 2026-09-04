#!/usr/bin/env python3
"""Prepare and execute a protocol-bound q128 A/Bsame crossed continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import reproducibility, schedule_switch  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_prefix(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(float(r["attempted_iteration"])) <= 4000]
        fields = reader.fieldnames
    if len(rows) != 4000 or int(float(rows[-1]["processed_nimg"])) != 512000:
        raise RuntimeError(f"source prefix is not exact through 512 kimg: {source}")
    with destination.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_snapshot(state_path: Path, output: Path) -> dict:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    payload = {"ema": state["ema"].eval().requires_grad_(False), "loss_fn": None,
               "augment_pipe": None,
               "dataset_kwargs": dict(state["trajectory_config"]["dataset_kwargs"])}
    reproducibility.atomic_pickle_dump(payload, output, overwrite=False)
    return {"state": str(state_path), "state_sha256": sha256(state_path),
            "snapshot": str(output), "snapshot_sha256": sha256(output)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source-run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--branch", choices=("AB", "BA", "AA", "BB"), required=True)
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--runtime-python", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--final-kimg", type=int, choices=(640, 1024), required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    if a.seed not in protocol["cohort"]["formal_seeds"] + protocol["cohort"]["replacement_pool"] + [protocol["cohort"]["smoke_seed"]]:
        raise RuntimeError("seed outside frozen protocol")
    mapping = {"AB": ("A", "Bsame", "A_to_Bsame"),
               "BA": ("Bsame", "A", "Bsame_to_A"),
               "AA": ("A", "A", "A_to_A"),
               "BB": ("Bsame", "Bsame", "Bsame_to_Bsame")}
    origin, continuation, manifest_branch = mapping[a.branch]
    run_kind = ("parity" if a.branch in {"AA", "BB"}
                else ("smoke" if a.final_kimg == 640 else "formal"))
    if run_kind == "parity" and a.final_kimg != 640:
        raise RuntimeError("parity must end at 640")
    source_dir = a.source_run_dir.resolve(strict=True)
    source_state = source_dir / "training-state-kimg000512.pt"
    state = torch.load(source_state, map_location="cpu", weights_only=False)
    before_sha = sha256(source_state)
    if state.get("factorial", {}).get("arm") != origin:
        raise RuntimeError("source arm mismatch")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    copy_prefix(source_dir / "train_summary.csv", out / "train_summary.csv")
    copy_prefix(source_dir / "factorial_training_telemetry_v1.csv",
                out / "source_factorial_training_telemetry_v1.csv")
    shutil.copyfile(source_dir / "initial_state_receipt_v1.json",
                    out / "initial_state_receipt_v1.json")
    manifest = {
        "schema": schedule_switch.Q128_RUN_MANIFEST_SCHEMA,
        "experiment_protocol": schedule_switch.Q128_FRESH_PROTOCOL,
        "run_kind": run_kind, "branch": manifest_branch, "seed": a.seed,
        "origin_arm": origin, "continuation_arm": continuation,
        "switch_kimg": 512, "final_kimg": a.final_kimg,
        "protocol_sha256": sha256(a.protocol),
        "implementation_commit": subprocess.check_output(
            ["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_checkpoint_manifest_sha256": before_sha,
        "source_state": {"path": str(source_state), "bytes": source_state.stat().st_size,
                         "sha256": before_sha,
                         "internal_state_sha256": schedule_switch.internal_state_hashes(state)},
    }
    manifest_path = out / "formal_run_manifest.json"
    reproducibility.atomic_json_dump(manifest, manifest_path, overwrite=False)
    factors = {"A": (1.0, 1.0), "Bsame": (1.1, 1.1)}[continuation]
    milestones = "640" if a.final_kimg == 640 else "640,768,896,1024"
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(a.gpu_id),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(48000 + a.gpu_id * 100 + a.seed % 97),
                "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})
    cmd = [str(a.runtime_python), str(a.repo / "ct_train.py"),
           f"--data={a.dataset}", f"--outdir={out}", "--nosubdir", "--cond=False",
           "--arch=ddpmpp", "--precond=ect", "--batch=128", "--batch-gpu=16",
           "--optim=RAdam", "--lr=0.0001", "--dropout=0.2", "--augment=0", "--xflip=False",
           "--mean=-1.1", "--std=2.0", "--mapping=sigmoid", "--global-gap-scale=1.0",
           "--factorial-protocol=q128_matched_spacing_v1",
           f"--target-gap-scale={factors[0]}", f"--denominator-gap-scale={factors[1]}",
           "-q", "128", "-k", "8", "-b", "1", "-c", "0", "--double=10000",
           "--ema_beta=0.9993", f"--seed={a.seed}", "--fp16=True", "--tf32=False",
           "--ls=1.0", "--enable_amp=True", "--bench=False", "--cache=True", "--workers=1",
           "--metrics=none", f"--duration={a.final_kimg / 1000}", "--tick=10", "--snap=0",
           "--dump=0", "--ckpt=10", "--sample_every=0", "--eval_every=0", "--mid_t=0.821",
           "--adaptive-update-kimg=0.5", f"--immutable-checkpoint-kimg={milestones}",
           f"--schedule-switch-manifest={manifest_path}", f"--resume={source_state}"]
    result = subprocess.run(cmd, cwd=a.repo, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"continuation failed with exit {result.returncode}")
    if sha256(source_state) != before_sha:
        raise RuntimeError("source state mutated")
    receipts = []
    for budget in ([640] if a.final_kimg == 640 else [640, 768, 896, 1024]):
        state_path = out / f"training-state-kimg{budget:06d}.pt"
        receipts.append(export_snapshot(state_path, out / f"network-snapshot-kimg{budget:06d}.pkl"))
    reproducibility.atomic_json_dump(
        {"schema":"ect.q128-crossed-continuation-receipt/v1","status":"PASS",
         "branch":a.branch,"seed":a.seed,"gpu_index":a.gpu_id,
         "source_immutable":True,"artifacts":receipts},
        out / "trajectory_completion_receipt.json", overwrite=False)
    print(json.dumps({"status":"PASS","branch":a.branch,"output":str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
