#!/usr/bin/env python3
"""Launch one fixed two-GPU ImageNet-64 IA/IB training cell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 102, 103)
GAP_SCALES = {"IA": 1.0, "IB": 1.1}
MILESTONES_KIMG = tuple(range(1280, 12801, 1280))


def parse_gpus(value: str) -> tuple[str, str]:
    gpus = tuple(token.strip() for token in value.split(",") if token.strip())
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise argparse.ArgumentTypeError("--gpus requires two distinct GPU IDs")
    return gpus


def training_command(args: argparse.Namespace) -> list[str]:
    run_dir = args.run_root / f"seed{args.seed}" / args.method
    command = [
        str(args.python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        f"--master_port={args.master_port}",
        str(args.repo / "ct_train.py"),
        f"--outdir={run_dir}",
        "--nosubdir",
        f"--data={args.data}",
        "--cond=True",
        "--arch=edm2",
        "--preset=edm2-img64-s",
        "--batch=128",
        "--batch-gpu=32",
        "--optim=Adam",
        "--lr=0.001",
        "--betas=0.9",
        "--betas=0.99",
        "--decay=2000",
        "--dropout=0.4",
        "--dropres=16",
        "--mean=-0.8",
        "--std=1.6",
        "--schedule=global_sigmoid",
        f"--global-gap-scale={GAP_SCALES[args.method]}",
        "--wt=snrpk",
        "-q",
        "4",
        "-k",
        "8",
        "-b",
        "1",
        "-c",
        "0.06",
        "--double=500",
        "--duration=12.8",
        "--tick=6.4",
        "--snap=0",
        "--dump=0",
        "--ckpt=200",
        "--immutable-checkpoint-kimg="
        + ",".join(str(value) for value in MILESTONES_KIMG),
        "--power-ema-stds=0.01,0.05,0.1",
        "--exact-resume=True",
        "--global-batch-mean=True",
        f"--seed={args.seed}",
        "--fp16=False",
        "--tf32=False",
        "--enable_amp=False",
        "--bench=False",
        "--cache=True",
        "--workers=1",
        "--metrics=none",
        "--startup-preview=False",
        "--sample_every=0",
        "--eval_every=0",
        "--mid_t=1.526",
        f"--desc=imagenet64-gap-{args.method.lower()}-seed{args.seed}",
    ]
    command.append(
        f"--resume={args.resume}" if args.resume is not None
        else f"--transfer={args.transfer}"
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(GAP_SCALES), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--gpus", type=parse_gpus, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in ("data", "run_root", "repo", "python", "transfer", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    if (args.transfer is None) == (args.resume is None):
        parser.error("specify exactly one of --transfer or --resume")

    command = training_command(args)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.dry_run:
        print(
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"CUBLAS_WORKSPACE_CONFIG={env['CUBLAS_WORKSPACE_CONFIG']} "
            f"{shlex.join(command)}"
        )
        return

    if not (args.repo / "ct_train.py").is_file():
        parser.error(f"missing trainer: {args.repo / 'ct_train.py'}")
    if not args.data.exists():
        parser.error(f"missing dataset: {args.data}")
    checkpoint = args.resume if args.resume is not None else args.transfer
    if checkpoint is not None and not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")
    subprocess.run(command, cwd=args.repo, env=env, check=True)


if __name__ == "__main__":
    main()
