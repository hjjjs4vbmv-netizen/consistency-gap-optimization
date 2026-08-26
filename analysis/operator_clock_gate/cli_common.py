"""Shared loading and provenance helpers for operator-clock runners."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import torch

from .core import (
    AlgorithmicState,
    AuditBatchGroup,
    freeze_batch_groups,
    write_json,
)


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
DEFAULT_OUT = HERE / "results" / "raw_receipts"


def configure_determinism() -> dict[str, Any]:
    expected_workspace = ":4096:8"
    actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_workspace != expected_workspace:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 before Python starts; "
            f"got {actual_workspace!r}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    return {
        "CUBLAS_WORKSPACE_CONFIG": actual_workspace,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
    }


def protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_epsilons(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epsilons must be comma-separated floats") from exc
    if len(result) < 3 or len(set(result)) != len(result) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("provide at least three unique positive epsilons")
    return result


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-state", type=Path, required=True,
                        help="Trusted training-state-*.pt containing net and optimizer_state")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Matching trusted network snapshot containing ema and loss_fn")
    parser.add_argument("--batch-file", type=Path, required=True,
                        help="Trusted torch file with four fixed image/label minibatches")
    parser.add_argument("--expected-training-state-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-batch-file-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ema-beta", type=float, default=0.9993)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def select_shard(tasks: Sequence[Any], *, shard_index: int,
                 num_shards: int) -> list[Any]:
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [task for index, task in enumerate(tasks) if index % num_shards == shard_index]


def _check_expected(path: Path, expected: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None:
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected hashes must be lowercase SHA256")
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual,
            "expected_sha256": expected, "matched": expected is None or actual == expected}


def source_assets(args) -> dict[str, Any]:
    return {
        "training_state": _check_expected(
            args.training_state, args.expected_training_state_sha256),
        "checkpoint": _check_expected(args.checkpoint, args.expected_checkpoint_sha256),
        "batch_file": _check_expected(args.batch_file, args.expected_batch_file_sha256),
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256_file(PROTOCOL_PATH)},
        "implementation": {
            path.name: sha256_file(path)
            for path in (
                HERE / "core.py", HERE / "cli_common.py",
                HERE / "prepare_frozen_batches.py",
                HERE / "run_field_jvp.py", HERE / "run_algorithmic_jvp.py",
                HERE / "run_matched_micro_rollout.py",
            )
        },
    }


def _optimizer_from_state(net: torch.nn.Module, optimizer_state: dict[str, Any]):
    groups = optimizer_state.get("param_groups", [])
    if not groups:
        raise RuntimeError("optimizer_state has no parameter groups")
    first = groups[0]
    if len(groups) != 1:
        raise RuntimeError("operator gate currently requires one RAdam parameter group")
    optimizer = torch.optim.RAdam(
        net.parameters(), lr=float(first["lr"]), betas=tuple(first["betas"]),
        eps=float(first["eps"]), weight_decay=float(first.get("weight_decay", 0.0)),
        decoupled_weight_decay=bool(first.get("decoupled_weight_decay", False)),
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    if len(optimizer.state) != len(tuple(net.parameters())):
        raise RuntimeError(
            "full algorithmic audit requires initialized optimizer state for every parameter")
    return optimizer


def _load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint must contain a dictionary")
    return value


def load_algorithmic_state(args) -> AlgorithmicState:
    device = torch.device(args.device)
    training = torch.load(
        args.training_state, map_location="cpu", weights_only=False)
    snapshot = _load_snapshot(args.checkpoint)
    for key in ("net", "optimizer_state"):
        if key not in training:
            raise RuntimeError(f"training state is missing {key!r}")
    for key in ("ema", "loss_fn"):
        if key not in snapshot:
            raise RuntimeError(f"checkpoint is missing {key!r}")
    net = copy.deepcopy(training["net"]).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(snapshot["ema"]).to(device).eval().requires_grad_(False)
    if set(net.state_dict()) != set(ema.state_dict()):
        raise RuntimeError("net and EMA state schemas differ")
    optimizer = _optimizer_from_state(net, training["optimizer_state"])
    loss_fn = copy.deepcopy(snapshot["loss_fn"])
    if "loss_fn_state" in training and hasattr(loss_fn, "load_schedule_state_dict"):
        if not loss_fn.load_schedule_state_dict(copy.deepcopy(training["loss_fn_state"])):
            raise RuntimeError("training loss state is incompatible with checkpoint loss")
    scaler = None
    if "gradscaler_state" in training:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("checkpoint has GradScaler state but CUDA is unavailable")
        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:  # pragma: no cover - older supported PyTorch.
            scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(copy.deepcopy(training["gradscaler_state"]))
    return AlgorithmicState(
        net=net, optimizer=optimizer, ema=ema, loss_fn=loss_fn,
        scaler=scaler, ema_beta=float(args.ema_beta),
    )


def _raw_batches(value: Any, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(value, dict):
        value = value.get("batches", value.get("audit_batches"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("batch file must contain a list under 'batches'")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            images, labels = item.get("images"), item.get("labels")
        else:
            images, labels = item
        if not isinstance(images, torch.Tensor):
            raise RuntimeError(f"batch {index} images are not a tensor")
        images = images.to(device)
        if not images.is_floating_point():
            images = images.to(torch.float32) / 127.5 - 1.0
        else:
            images = images.to(torch.float32)
        if labels is None:
            labels = torch.empty(images.shape[0], 0)
        labels = labels.to(device)
        if labels.shape[0] != images.shape[0]:
            raise RuntimeError(f"batch {index} image/label counts differ")
        result.append((images, labels))
    return result


def load_frozen_batches(args, loss_fn: Any) -> list[AuditBatchGroup]:
    value = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    raw = _raw_batches(value, torch.device(args.device))
    audit_ids = protocol()["audit_minibatch_ids"]
    if len(raw) != len(audit_ids):
        raise RuntimeError(
            f"formal protocol requires exactly {len(audit_ids)} minibatches, got {len(raw)}")
    return freeze_batch_groups(
        raw, loss_fn, audit_ids,
        microbatch_size=int(protocol()["batch_construction"]["microbatch_size"]),
    )


def write_run_manifest(out: Path, kind: str, assets: dict[str, Any],
                       receipts: Sequence[str], status: str,
                       *, assets_after: dict[str, Any] | None = None) -> None:
    assets_after = assets if assets_after is None else assets_after
    source_files_preserved = assets == assets_after
    if not source_files_preserved:
        status = "FAIL_CLOSED"
    write_json(out / f"{kind}_manifest.json", {
        "schema_version": 1, "kind": kind, "status": status,
        "protocol": protocol(), "assets_before": assets,
        "assets_after": assets_after,
        "source_files_preserved": source_files_preserved,
        "receipts": list(receipts),
    })
