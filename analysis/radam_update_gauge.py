"""One-step, non-committing RAdam update gauge for the ECT gap intervention.

This is deliberately an *update* probe, rather than the gradient-only probe in
``gap_gradient_hook.py``.  It is a fresh-state sanity probe: it makes two
disposable copies of one pretrained EDM, starts a fresh RAdam on each
(``m=v=0``, optimizer step zero), and runs the training-loop AMP order once
for g=1.0 and g=1.3.  It is not an audit of a resumed training optimizer
state.  The source model, optimizer, and GradScaler are never stepped.

The CLI accepts any of the 32/64/128/256 kimg snapshots: checkpoint age is
only provenance, not an assumption of this diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from training.schedules import get_schedule

LAYERWISE_FIELDS = (
    "layer", "update_1_l2", "update_1p3_l2", "update_cosine",
    "c0_star", "layerwise_residual",
    "layerwise_residual_with_model_c0_star",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash a dataset directory by sorted relative path and file content.

    Absolute locations and metadata are intentionally excluded so that a copied
    dataset has the same provenance hash.  Directory entries are included to
    distinguish otherwise identical file sets with different structure.
    """
    digest = hashlib.sha256()
    digest.update(b"radam-update-gauge-directory-sha256-v1\\0")
    entries = sorted(path.rglob("*"), key=lambda entry: entry.relative_to(path).as_posix())
    for entry in entries:
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        if entry.is_dir():
            digest.update(b"directory\\0" + relative + b"\\0")
        elif entry.is_file():
            digest.update(b"file\\0" + relative + b"\\0")
            with entry.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise ValueError(f"dataset directory contains unsupported entry: {entry}")
    return digest.hexdigest()


def dataset_sha256(path: Path) -> tuple[str, str]:
    """Return a deterministic provenance hash and its algorithm identifier."""
    if path.is_file():
        return sha256_file(path), "sha256_file"
    if path.is_dir():
        return sha256_directory(path), "sha256_directory_v1"
    raise FileNotFoundError(f"dataset path does not exist or is not a regular file/directory: {path}")


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested optimizer/scaler state without relying on pickle format."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
    elif value is None:
        digest.update(b"none\0")
    else:
        digest.update((type(value).__qualname__ + ":" + repr(value)).encode("utf-8"))


def state_sha256(state: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, state)
    return digest.hexdigest()


def tensor_collection_sha256(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return tensor_collection_sha256((("value", tensor),))


def module_state_hashes(net: torch.nn.Module) -> dict[str, str]:
    return {
        "parameter_sha256": tensor_collection_sha256(net.named_parameters()),
        "buffer_sha256": tensor_collection_sha256(net.named_buffers()),
    }


def layer_name(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0] if "." in parameter_name else parameter_name


def get_rng_state(device: torch.device) -> torch.Tensor:
    return torch.cuda.get_rng_state(device=device) if device.type == "cuda" else torch.get_rng_state()


def set_rng_state(state: torch.Tensor, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device=device)
    else:
        torch.set_rng_state(state)


def fixed_ect_loss(net, loss_template, schedule, images, labels, t, eps, dropout_rng_state):
    """ECT loss with t/noise/dropout supplied by the shared pair contract."""
    r = schedule.compute_r(t=t, stage=loss_template.stage)
    set_rng_state(dropout_rng_state, images.device)
    d_yt = net(images + eps * t, t, labels, augment_labels=None)
    if bool((r > 0).any()):
        set_rng_state(dropout_rng_state, images.device)
        with torch.no_grad():
            d_yr = net(images + eps * r, r, labels, augment_labels=None)
        d_yr = torch.nan_to_num(d_yr)
        d_yr = (r > 0) * d_yr + (~(r > 0)) * images
    else:
        d_yr = images
    raw = (d_yt - d_yr).square().reshape(images.shape[0], -1).sum(dim=1)
    loss = torch.sqrt(raw + loss_template.c ** 2) - loss_template.c if loss_template.c > 0 else torch.sqrt(raw)
    return loss / (t - r).flatten()


class _CPUGradScaler:
    """Minimal deterministic scaler for CPU-only audit tests on Torch 2.2a0."""

    def __init__(self, enabled: bool, initial_scale: float):
        self._enabled = bool(enabled)
        self._scale = float(initial_scale)
        self._optimizer = None

    def state_dict(self):
        return {"enabled": self._enabled, "scale": self._scale}

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss * self._scale if self._enabled else loss

    def unscale_(self, optimizer):
        self._optimizer = optimizer
        if self._enabled:
            reciprocal = 1.0 / self._scale
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.mul_(reciprocal)

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _new_scaler(device: torch.device, enabled: bool, initial_scale: float):
    modern = getattr(torch.amp, "GradScaler", None)
    if modern is not None:
        return modern(device.type, enabled=enabled, init_scale=initial_scale)
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler(
            enabled=enabled, init_scale=initial_scale
        )
    return _CPUGradScaler(enabled=enabled, initial_scale=initial_scale)


def _delta_by_name(before: torch.nn.Module, after: torch.nn.Module) -> dict[str, torch.Tensor]:
    before_params = dict(before.named_parameters())
    return {
        name: (param.detach().double().cpu() - before_params[name].detach().double().cpu())
        for name, param in after.named_parameters()
    }


def _norm_sq(values: Iterable[torch.Tensor]) -> float:
    return sum(float(value.square().sum()) for value in values)


def _dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return sum(float((left[name] * right[name]).sum()) for name in left)


def gauge_metrics(update_1: dict[str, torch.Tensor], update_13: dict[str, torch.Tensor]) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute the requested gauge and whole-model/per-layer residuals.

    ``c_star`` is the least-squares coefficient for the stated target
    ``c_star * d_1.3 ≈ d_1``: <d_1.3,d_1> / ||d_1.3||².  The residual below
    uses that same coefficient.
    """
    n1_sq, n13_sq = _norm_sq(update_1.values()), _norm_sq(update_13.values())
    dot = _dot(update_13, update_1)
    if n1_sq <= 0 or n13_sq <= 0 or dot == 0:
        raise RuntimeError("zero or orthogonal virtual update; gauge is undefined")
    c_star = dot / n13_sq
    cosine = dot / math.sqrt(n1_sq * n13_sq)
    residual_sq = max(_norm_sq((c_star * update_13[name] - update_1[name] for name in update_1)), 0.0)
    whole = {
        "gauge_defined": True,
        "gauge_error": None,
        "update_1_l2": math.sqrt(n1_sq),
        "update_1p3_l2": math.sqrt(n13_sq),
        "update_dot": dot,
        "update_cosine": cosine,
        "c0_star": c_star,
        "whole_model_residual": math.sqrt(residual_sq) / math.sqrt(n1_sq),
    }
    by_layer: dict[str, list[str]] = defaultdict(list)
    for name in update_1:
        by_layer[layer_name(name)].append(name)
    layers = []
    for layer, names in sorted(by_layer.items()):
        l1_sq = _norm_sq(update_1[name] for name in names)
        l13_sq = _norm_sq(update_13[name] for name in names)
        ldot = sum(float((update_13[name] * update_1[name]).sum()) for name in names)
        if l1_sq == 0 or l13_sq == 0 or ldot == 0:
            c_layer, cosine_layer, residual, residual_with_model_c = (math.nan,) * 4
        else:
            c_layer = ldot / l13_sq
            cosine_layer = ldot / math.sqrt(l1_sq * l13_sq)
            lres_sq = _norm_sq(c_layer * update_13[name] - update_1[name] for name in names)
            residual = math.sqrt(max(lres_sq, 0.0)) / math.sqrt(l1_sq)
            model_lres_sq = _norm_sq(c_star * update_13[name] - update_1[name] for name in names)
            residual_with_model_c = math.sqrt(max(model_lres_sq, 0.0)) / math.sqrt(l1_sq)
        layers.append({
            "layer": layer,
            "update_1_l2": math.sqrt(l1_sq),
            "update_1p3_l2": math.sqrt(l13_sq),
            "update_cosine": cosine_layer,
            "c0_star": c_layer,
            "layerwise_residual": residual,
            "layerwise_residual_with_model_c0_star": residual_with_model_c,
        })
    return whole, layers


def undefined_gauge_metrics(update_1: dict[str, torch.Tensor], update_13: dict[str, torch.Tensor],
                            error: str) -> dict[str, Any]:
    """Retain update telemetry when a skipped/degenerate step has no gauge."""
    n1_sq, n13_sq = _norm_sq(update_1.values()), _norm_sq(update_13.values())
    dot = _dot(update_13, update_1)
    cosine = dot / math.sqrt(n1_sq * n13_sq) if n1_sq and n13_sq else None
    return {
        "gauge_defined": False, "gauge_error": error,
        "update_1_l2": math.sqrt(n1_sq), "update_1p3_l2": math.sqrt(n13_sq),
        "update_dot": dot, "update_cosine": cosine,
        "c0_star": None,
        "whole_model_residual": None,
    }


def virtual_radam_update(common_net, loss_template, microbatches, *, gain: float,
                         lr: float, betas: tuple[float, float], eps_opt: float,
                         scaler_template, amp: bool) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Execute a real, disposable RAdam step and return its parameter delta."""
    net = copy.deepcopy(common_net).train().requires_grad_(True)
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, betas=betas, eps=eps_opt)
    scaler = copy.deepcopy(scaler_template)
    param_before = module_state_hashes(net)
    optimizer_before = state_sha256(optimizer.state_dict())
    scaler_before = state_sha256(scaler.state_dict())
    schedule = get_schedule("global_sigmoid", q=float(loss_template.q), k=float(loss_template.k),
                            b=float(loss_template.b), global_gap_scale=gain)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, loss_count = 0.0, 0
    # Deliberately no autocast context: ct_training_loop uses GradScaler but
    # relies on the EDM network's own fp16 selection, not torch.autocast.
    for images, labels, t, eps, dropout_rng_state in microbatches:
        loss = fixed_ect_loss(net, loss_template, schedule, images, labels, t, eps, dropout_rng_state)
        loss_mean = loss.mean()
        scaler.scale(loss_mean).backward() if amp else loss_mean.backward()
        loss_sum += float(loss.detach().double().sum().cpu())
        loss_count += loss.numel()
    if amp:
        scaler.unscale_(optimizer)
    nonfinite_before_sanitize = False
    for parameter in net.parameters():
        if parameter.grad is not None:
            nonfinite_before_sanitize |= not bool(torch.isfinite(parameter.grad).all())
            torch.nan_to_num(parameter.grad, nan=0, posinf=1e5, neginf=-1e5, out=parameter.grad)
    if amp:
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        step_skipped = scale_after < scale_before
    else:
        scale_before = scale_after = None
        optimizer.step()
        step_skipped = False
    delta = _delta_by_name(common_net, net)
    detail = {
        "gain": gain,
        "loss_mean": loss_sum / loss_count,
        "accumulation_rounds": len(microbatches),
        "amp_enabled": amp,
        "amp_unscale_called": amp,
        "nonfinite_before_sanitize": nonfinite_before_sanitize,
        "step_skipped": step_skipped,
        "grad_scale_before": scale_before,
        "grad_scale_after": scale_after,
        "parameter_hash_before": param_before,
        "parameter_hash_after_virtual_step": module_state_hashes(net),
        "optimizer_state_hash_before": optimizer_before,
        "optimizer_state_hash_after_virtual_step": state_sha256(optimizer.state_dict()),
        "gradscaler_hash_before": scaler_before,
        "gradscaler_hash_after_virtual_step": state_sha256(scaler.state_dict()),
        "optimizer_step_before": 0,
        "optimizer_step_after": int(next(iter(optimizer.state.values()))["step"].item()) if optimizer.state else 0,
    }
    return delta, detail


def run_pair(common_net, loss_template, images, labels, *, gains=(1.0, 1.3), lr=1e-4,
             betas=(0.9, 0.999), eps_opt=1e-8, amp=True, initial_scale=65536.0,
             random_seed: int | None = None,
             microbatch_size: int | None = None) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Run the pair while proving the common source objects are unchanged."""
    if tuple(gains) != (1.0, 1.3):
        raise ValueError("this audit is defined for exactly gains (1.0, 1.3)")
    device = images.device
    microbatch_size = images.shape[0] if microbatch_size is None else microbatch_size
    if microbatch_size < 1 or images.shape[0] % microbatch_size:
        raise ValueError("microbatch_size must be a positive divisor of batch size")
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device=device).clone() if device.type == "cuda" else None
    try:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(random_seed)
        source_before = module_state_hashes(common_net)
        source_optimizer = torch.optim.RAdam(common_net.parameters(), lr=lr, betas=betas, eps=eps_opt)
        source_optimizer_before = state_sha256(source_optimizer.state_dict())
        scaler_template = _new_scaler(device, amp, initial_scale)
        source_scaler_before = state_sha256(scaler_template.state_dict())
        microbatches = []
        # Sampling per microbatch intentionally mirrors ct_training_loop: a
        # global batch with --batch-gpu=16 makes eight separate RNG calls.
        for start in range(0, images.shape[0], microbatch_size):
            image_micro = images[start:start + microbatch_size]
            label_micro = labels[start:start + microbatch_size]
            t = (torch.randn(image_micro.shape[0], 1, 1, 1, device=device)
                 * loss_template.P_std + loss_template.P_mean).exp()
            eps = torch.randn_like(image_micro)
            microbatches.append((image_micro, label_micro, t, eps, get_rng_state(device).clone()))
        deltas, branches = {}, []
        for gain in gains:
            delta, detail = virtual_radam_update(common_net, loss_template, microbatches, gain=gain,
                lr=lr, betas=betas, eps_opt=eps_opt, scaler_template=scaler_template, amp=amp)
            deltas[gain] = delta
            branches.append(detail)
        try:
            whole, layers = gauge_metrics(deltas[1.0], deltas[1.3])
        except RuntimeError as exc:
            whole, layers = undefined_gauge_metrics(deltas[1.0], deltas[1.3], str(exc)), []
        source_after = module_state_hashes(common_net)
        source_optimizer_after = state_sha256(source_optimizer.state_dict())
        source_scaler_after = state_sha256(scaler_template.state_dict())
    finally:
        # The caller's RNG stream is state, too: a diagnostic must not consume
        # it, including when an overflow or another exception is raised.
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, device=device)
    audit = {
        "gains": list(gains),
        "fresh_radam": {"lr": lr, "betas": list(betas), "eps": eps_opt,
                        "m0_v0": True, "optimizer_step": 0},
        "randomness_contract": {"same_minibatch": True, "same_t": True, "same_noise": True,
                                "same_dropout_rng_state": True,
                                "minibatch_images_sha256": tensor_sha256(images),
                                "minibatch_labels_sha256": tensor_sha256(labels),
                                "microbatch_size": microbatch_size,
                                "accumulation_rounds": len(microbatches),
                                "t_sha256": state_sha256([t for _, _, t, _, _ in microbatches]),
                                "noise_sha256": state_sha256([eps for _, _, _, eps, _ in microbatches]),
                                "dropout_rng_state_sha256": state_sha256(
                                    [state for _, _, _, _, state in microbatches])},
        "source_state_non_committing": {
            "parameter_hash_before": source_before, "parameter_hash_after": source_after,
            "optimizer_state_hash_before": source_optimizer_before,
            "optimizer_state_hash_after": source_optimizer_after,
            "gradscaler_hash_before": source_scaler_before,
            "gradscaler_hash_after": source_scaler_after,
            "preserved": source_before == source_after and source_optimizer_before == source_optimizer_after
                         and source_scaler_before == source_scaler_after,
        },
        "branches": branches,
        "whole_model": whole,
    }
    return audit, layers


def load_checkpoint(path: Path, device: torch.device):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "ema" not in payload or "loss_fn" not in payload:
        raise SystemExit("checkpoint must be an ECT network snapshot containing ema and loss_fn")
    if payload.get("augment_pipe") is not None:
        raise SystemExit("augmentation-enabled checkpoint is unsupported: paired augmentation is not implemented")
    loss = payload["loss_fn"]
    if loss.schedule.name != "sigmoid":
        raise SystemExit(f"checkpoint schedule must be 'sigmoid', got {loss.schedule.name!r}")
    net = payload["ema"].to(device).train().requires_grad_(True)
    return net, loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, help="EDM ImageFolderDataset zip/directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-gpu", type=int, default=None,
                        help="training microbatch size; reproduces gradient accumulation")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--state-kimg", type=float, default=None,
                        help="provenance label; accepts 32/64/128/256 or any future checkpoint age")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-scale", type=float, default=65536.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", default="0.9,0.999")
    parser.add_argument("--eps", dest="eps_opt", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "analysis")
    args = parser.parse_args(argv)
    try:
        args.betas = tuple(float(value) for value in args.betas.split(","))
    except ValueError as exc:
        raise SystemExit("--betas must be beta1,beta2") from exc
    if len(args.betas) != 2:
        raise SystemExit("--betas must contain exactly two values")
    if (args.batch_size < 1 or args.lr <= 0 or args.initial_scale <= 0
            or (args.batch_gpu is not None and args.batch_gpu < 1)):
        raise SystemExit("batch size, lr, and initial scale must be positive")
    if args.batch_gpu is not None and args.batch_size % args.batch_gpu:
        raise SystemExit("--batch-size must be divisible by --batch-gpu")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if args.amp and device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--amp requires an available CUDA device; use --no-amp only for CPU test runs")
    net, loss = load_checkpoint(args.checkpoint, device)
    from training.dataset import ImageFolderDataset
    from torch.utils.data import DataLoader
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dataset = ImageFolderDataset(path=args.data, use_labels=False, xflip=False, cache=True,
                                resolution=net.img_resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=generator)
    images, labels = next(iter(loader))
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)
    audit, layers = run_pair(net, loss, images, labels, lr=args.lr, betas=args.betas,
                             eps_opt=args.eps_opt, amp=args.amp,
                             initial_scale=args.initial_scale, random_seed=args.seed,
                             microbatch_size=args.batch_gpu)
    data_sha256, dataset_hash_algorithm = dataset_sha256(Path(args.data))
    audit["provenance"] = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "data": str(args.data), "dataset_sha256": data_sha256,
        "dataset_hash_algorithm": dataset_hash_algorithm,
        "state_kimg": args.state_kimg, "batch_size": args.batch_size, "batch_gpu": args.batch_gpu,
        "seed": args.seed,
        "device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "schedule": loss.schedule.name, "q": float(loss.q), "k": float(loss.k), "b": float(loss.b),
        "stage": int(loss.stage), "amp_training_order": "scale, backward, unscale, sanitize, step, update",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "radam_update_audit_fresh.json"
    layer_path = args.out / "radam_update_layerwise.csv"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAYERWISE_FIELDS)
        writer.writeheader(); writer.writerows(layers)
    print(json.dumps(audit["whole_model"], indent=2))
    print(f"source state preserved: {audit['source_state_non_committing']['preserved']}")
    print(f"wrote {audit_path} and {layer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
