#!/usr/bin/env python3
"""Measure gradient moments for a fixed ECT checkpoint under a gap sweep.

This is a gradient-only diagnostic.  It loads one checkpoint, holds its
parameters fixed, and never constructs or steps an optimizer.  For each
minibatch it samples the training-distribution timestep vector and shared
noise exactly once, then reuses them (and the dropout RNG state) for every
gap multiplier.  Consequently, within a minibatch the only intentional
difference between sweep points is ``global_gap_scale``.

The diagnostic is supplementary evidence.  It is not an evaluation run and
does not establish formal eligibility for a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dnnlib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.schedules import get_schedule


SCRIPT_VERSION = 2


def fail(message: str) -> None:
    raise SystemExit(f"[deepnet_gap_gradient_moments] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_collection_sha256(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash names, metadata, and exact values without depending on device."""
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        # Scalar integer buffers (e.g. BatchNorm num_batches_tracked) cannot
        # be byte-viewed until flattened.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_state_hashes(net: torch.nn.Module) -> dict[str, str]:
    """Return separate immutable-value hashes for model parameters and buffers."""
    return {
        "parameter_sha256": tensor_collection_sha256(net.named_parameters()),
        "buffer_sha256": tensor_collection_sha256(net.named_buffers()),
    }


def set_dropout_rng_state(state: torch.Tensor, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device=device)
    else:
        torch.set_rng_state(state)


def parse_gaps(value: str) -> list[float]:
    try:
        gaps = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gaps must be comma-separated floats") from exc
    if not gaps or any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        raise argparse.ArgumentTypeError("--gaps must contain finite values > 0")
    if len(set(gaps)) != len(gaps):
        raise argparse.ArgumentTypeError("--gaps must not contain duplicates")
    if 1.0 not in gaps:
        raise argparse.ArgumentTypeError("--gaps must include the reference value 1.0")
    return gaps


def output_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "moments": repo_root / "analysis" / "deepnet_gap_gradient_moments.csv",
        "layerwise": repo_root / "analysis" / "deepnet_layerwise_residual.csv",
        "batch": repo_root / "analysis" / "deepnet_gap_gradient_batch_residual.csv",
        "manifest": repo_root / "analysis" / "deepnet_gap_gradient_moments_manifest.json",
        "report": repo_root / "analysis" / "deepnet_scalar_residual.md",
        "figure": repo_root / "figures" / "deepnet_scalar_residual_vs_g.pdf",
    }


def layer_name(parameter_name: str) -> str:
    """Group a weight and bias under their enclosing module path."""
    return parameter_name.rsplit(".", 1)[0] if "." in parameter_name else parameter_name


def finite_scalar(value: torch.Tensor, label: str) -> float:
    result = float(value.detach().double().cpu())
    if not math.isfinite(result):
        fail(f"non-finite {label}")
    return result


def ect_loss_with_fixed_randomness(
    net: torch.nn.Module,
    loss_template: Any,
    schedule: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ECMLoss with supplied t/noise/dropout randomness.

    ``loss_template`` supplies c and stage from the actual snapshot.  The
    caller constructs ``global_sigmoid`` schedules once: it equals the
    official sigmoid mapping at g=1 and applies the requested multiplicative
    gap intervention otherwise.
    """
    r = schedule.compute_r(t=t, stage=loss_template.stage)

    # There is no augmentation in the saved run.  Resetting before both
    # network calls exactly mirrors ECMLoss' shared dropout-mask protocol.
    device = images.device
    set_dropout_rng_state(dropout_rng_state, device)
    d_yt = net(images + eps * t, t, labels, augment_labels=None)
    if bool((r > 0).any()):
        set_dropout_rng_state(dropout_rng_state, device)
        with torch.no_grad():
            d_yr = net(images + eps * r, r, labels, augment_labels=None)
        d_yr = torch.nan_to_num(d_yr)
        d_yr = (r > 0) * d_yr + (~(r > 0)) * images
    else:
        d_yr = images

    raw = (d_yt - d_yr).square().reshape(images.shape[0], -1).sum(dim=1)
    loss = torch.sqrt(raw + loss_template.c**2) - loss_template.c if loss_template.c > 0 else torch.sqrt(raw)
    return loss / (t - r).flatten(), r


def gradient_statistics(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[float, float, dict[str, float]]:
    """Return whole-model norm², L∞ norm, and per-layer norm² for gradients."""
    total_norm_sq = 0.0
    inf_norm = 0.0
    per_layer: dict[str, float] = defaultdict(float)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            fail(f"non-finite gradient for {name}")
        norm_sq = finite_scalar(gradient.float().square().sum(), f"gradient norm for {name}")
        total_norm_sq += norm_sq
        inf_norm = max(inf_norm, finite_scalar(gradient.float().abs().max(), f"gradient inf norm for {name}"))
        per_layer[layer_name(name)] += norm_sq
    if total_norm_sq <= 0:
        fail("all gradients are zero")
    return total_norm_sq, inf_norm, dict(per_layer)


def vector_dot(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    reference: dict[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    total = 0.0
    per_layer: dict[str, float] = defaultdict(float)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        dot = finite_scalar((parameter.grad.detach().float() * reference[name]).sum(), f"gradient dot for {name}")
        total += dot
        per_layer[layer_name(name)] += dot
    return total, dict(per_layer)


def mean_vector_statistics(
    mean_sums: dict[str, dict[str, torch.Tensor]],
    reference_gap: float,
    batches: int,
    batch_norm_sq_sums: dict[str, float],
    batch_layer_norm_sq_sums: dict[str, dict[str, float]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compute model and layer moments from accumulated minibatch gradients."""
    reference = mean_sums[str(reference_gap)]
    reference_norm_sq = sum(finite_scalar((value / batches).double().square().sum(), "reference mean norm") for value in reference.values())
    if reference_norm_sq <= 0:
        fail("reference mean gradient is zero")

    rows: list[dict[str, float]] = []
    layer_rows: list[dict[str, float]] = []
    for gap_key, sums in mean_sums.items():
        gap = float(gap_key)
        mean_norm_sq = 0.0
        dot_reference = 0.0
        layer_values: dict[str, dict[str, float]] = defaultdict(lambda: {"norm_sq": 0.0, "reference_norm_sq": 0.0, "dot": 0.0})
        for name, value in sums.items():
            mean = value / batches
            reference_mean = reference[name] / batches
            current_norm_sq = finite_scalar(mean.double().square().sum(), f"mean norm for {name}")
            reference_component_sq = finite_scalar(reference_mean.double().square().sum(), f"reference mean norm for {name}")
            dot = finite_scalar((mean.double() * reference_mean.double()).sum(), f"mean dot for {name}")
            mean_norm_sq += current_norm_sq
            dot_reference += dot
            layer = layer_name(name)
            layer_values[layer]["norm_sq"] += current_norm_sq
            layer_values[layer]["reference_norm_sq"] += reference_component_sq
            layer_values[layer]["dot"] += dot

        scalar_fit = dot_reference / reference_norm_sq
        residual_sq = max(mean_norm_sq + scalar_fit**2 * reference_norm_sq - 2 * scalar_fit * dot_reference, 0.0)
        mean_norm = math.sqrt(mean_norm_sq)
        residual = math.sqrt(residual_sq) / mean_norm if mean_norm else math.nan
        cosine = dot_reference / math.sqrt(mean_norm_sq * reference_norm_sq) if mean_norm_sq else math.nan
        variance_trace = max(batch_norm_sq_sums[gap_key] / batches - mean_norm_sq, 0.0)
        noise_scale = variance_trace / mean_norm_sq if mean_norm_sq else math.nan
        rows.append({
            "gap": gap,
            "mean_grad_l2": mean_norm,
            "mean_grad_l2_sq": mean_norm_sq,
            "scalar_fit_to_g1": scalar_fit,
            "cosine_to_g1": cosine,
            "direction_residual": residual,
            "gradient_variance_trace": variance_trace,
            "normalized_noise_scale": noise_scale,
        })

        for layer, values in layer_values.items():
            layer_reference_sq = values["reference_norm_sq"]
            layer_norm_sq = values["norm_sq"]
            layer_dot = values["dot"]
            layer_fit = layer_dot / layer_reference_sq if layer_reference_sq else math.nan
            layer_residual_sq = max(layer_norm_sq + layer_fit**2 * layer_reference_sq - 2 * layer_fit * layer_dot, 0.0)
            layer_norm = math.sqrt(layer_norm_sq)
            layer_variance = max(batch_layer_norm_sq_sums[gap_key][layer] / batches - layer_norm_sq, 0.0)
            layer_rows.append({
                "layer": layer,
                "gap": gap,
                "mean_grad_l2": layer_norm,
                "scalar_fit_to_g1": layer_fit,
                "cosine_to_g1": layer_dot / math.sqrt(layer_norm_sq * layer_reference_sq) if layer_norm_sq and layer_reference_sq else math.nan,
                "direction_residual": math.sqrt(layer_residual_sq) / layer_norm if layer_norm else math.nan,
                "gradient_variance_trace": layer_variance,
                "normalized_noise_scale": layer_variance / layer_norm_sq if layer_norm_sq else math.nan,
            })
    return sorted(rows, key=lambda row: row["gap"]), sorted(layer_rows, key=lambda row: (row["layer"], row["gap"]))


def batch_residual_rows(
    batch_rows: list[dict[str, float]],
    moment_rows: list[dict[str, float]],
    reference_gap: float,
) -> list[dict[str, float]]:
    scalar_fits = {row["gap"]: row["scalar_fit_to_g1"] for row in moment_rows}
    reference_by_batch = {int(row["batch_index"]): row for row in batch_rows if row["gap"] == reference_gap}
    output: list[dict[str, float]] = []
    for row in batch_rows:
        reference = reference_by_batch[int(row["batch_index"])]
        fit = scalar_fits[row["gap"]]
        residual_sq = max(row["grad_l2_sq"] + fit**2 * reference["grad_l2_sq"] - 2 * fit * row["dot_to_g1"], 0.0)
        result = dict(row)
        result["scalar_fit_to_g1"] = fit
        result["direction_residual_to_g1"] = math.sqrt(residual_sq) / math.sqrt(row["grad_l2_sq"])
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fail(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["gap"] for row in rows], [row["direction_residual"] for row in rows], marker="o")
    axis.axvline(1.0, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("global gap scale g")
    axis.set_ylabel(r"$\|\mu_g-a_g^*\mu_1\|/\|\mu_g\|$")
    axis.set_title("Scalar-fit directional residual")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path: Path, manifest: dict[str, Any], moments: list[dict[str, float]], batch_rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Deep-network gap gradient moments",
        "",
        "This is a gradient-only supplementary diagnostic. The checkpoint parameters were held fixed; no optimizer was created or stepped.",
        "",
        "For each minibatch, the image batch, per-example timestep vector, shared noise tensor, and dropout RNG state were reused for every gap value.",
        "",
        "The scalar fit is the least-squares projection `a_g*=<mu_g,mu_1>/||mu_1||^2`; the directional residual is `||mu_g-a_g*mu_1||/||mu_g||`.",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{manifest['checkpoint']}`",
        f"- checkpoint SHA256: `{manifest['checkpoint_sha256']}`",
        f"- dataset SHA256: `{manifest['dataset_sha256']}`",
        f"- batches: {manifest['batches']}; batch size: {manifest['batch_size']}; seed: {manifest['seed']}",
        f"- parameter values unchanged: `{manifest['parameter_values_unchanged']}`",
        f"- buffers unchanged: `{manifest['buffer_values_unchanged']}`",
        f"- parameter SHA256 (before/after): `{manifest['parameter_sha256_before']}` / `{manifest['parameter_sha256_after']}`",
        f"- buffer SHA256 (before/after): `{manifest['buffer_sha256_before']}` / `{manifest['buffer_sha256_after']}`",
        f"- gradient slots populated (before/after): {manifest['gradient_slots_populated_before']} / {manifest['gradient_slots_populated_after']}",
        "",
        "## Whole-model moments",
        "",
        "| g | ||mu_g|| | a* | cos(mu_g,mu_1) | residual | variance trace | normalized noise scale | batch residual mean +/- sd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for moment in moments:
        values = [row["direction_residual_to_g1"] for row in batch_rows if row["gap"] == moment["gap"]]
        lines.append(
            "| {gap:.3g} | {mean_grad_l2:.6g} | {scalar_fit_to_g1:.6g} | {cosine_to_g1:.6g} | {direction_residual:.6g} | {gradient_variance_trace:.6g} | {normalized_noise_scale:.6g} | {batch_mean:.6g} +/- {batch_std:.6g} |".format(
                **moment,
                batch_mean=float(np.mean(values)),
                batch_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("ema"), torch.nn.Module):
        fail(f"checkpoint lacks EMA network: {path}")
    if "loss_fn" not in payload:
        fail(f"checkpoint lacks persisted loss_fn: {path}")
    net = payload["ema"].to(device)
    # Snapshots are exported for inference and commonly freeze EMA
    # parameters.  Re-enabling autograd changes only the tensor flag; no
    # parameter value, buffer, or optimizer state is mutated by this script.
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    net.train()
    return net, payload["loss_fn"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gaps", type=parse_gaps, default=parse_gaps("0.9,1.0,1.2,1.3"))
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.batches < 32:
        fail("--batches must be at least 32")
    if args.batch_size < 1:
        fail("--batch-size must be positive")
    if not torch.cuda.is_available():
        fail("this diagnostic requires CUDA")
    checkpoint = args.checkpoint.resolve()
    dataset_path = args.data.resolve()
    if not checkpoint.is_file() or not dataset_path.is_file():
        fail("--checkpoint and --data must name existing files")
    repo_root = args.repo_root.resolve()
    paths = output_paths(repo_root)
    if any(path.exists() for path in paths.values()):
        fail("refusing to overwrite existing diagnostic outputs; move them aside first")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net, loss_template = load_checkpoint(checkpoint, device)
    if not hasattr(loss_template, "schedule") or getattr(loss_template.schedule, "name", None) != "sigmoid":
        fail("this diagnostic expects a checkpoint trained with the fixed sigmoid schedule")
    dataset = dnnlib.util.construct_class_by_name(
        class_name="training.dataset.ImageFolderDataset",
        path=str(dataset_path), use_labels=False, xflip=False, cache=True,
        resolution=32, max_size=50000,
    )
    if len(dataset) < args.batch_size:
        fail("dataset is smaller than one requested batch")

    named_parameters = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        fail("EMA network has no trainable parameters")
    net.zero_grad(set_to_none=True)
    state_before = module_state_hashes(net)
    gradient_slots_before = sum(parameter.grad is not None for _, parameter in named_parameters)
    mean_sums: dict[str, dict[str, torch.Tensor]] = {
        str(gap): {name: torch.zeros_like(parameter, dtype=torch.float64) for name, parameter in named_parameters}
        for gap in args.gaps
    }
    batch_norm_sq_sums: dict[str, float] = defaultdict(float)
    batch_layer_norm_sq_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    batch_rows: list[dict[str, float]] = []
    sample_count = args.batches * args.batch_size
    index_generator = np.random.default_rng(args.seed)
    if sample_count <= len(dataset):
        selected_indices = index_generator.choice(len(dataset), size=sample_count, replace=False)
    else:
        selected_indices = index_generator.integers(0, len(dataset), size=sample_count, endpoint=False)
    selected_indices = selected_indices.reshape(args.batches, args.batch_size)
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    schedules = {
        str(gap): get_schedule("global_sigmoid", q=loss_template.q, k=loss_template.k, b=loss_template.b, global_gap_scale=gap)
        for gap in args.gaps
    }
    started = time.time()

    for batch_index, indices in enumerate(selected_indices):
        image_array = np.stack([dataset[int(index)][0] for index in indices])
        images = torch.from_numpy(image_array).to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
        labels = torch.empty((args.batch_size, 0), device=device)
        rnd_normal = torch.randn((args.batch_size, 1, 1, 1), device=device, generator=random_generator)
        t = (rnd_normal * loss_template.P_std + loss_template.P_mean).exp()
        eps = torch.randn(images.shape, device=device, generator=random_generator)
        dropout_state = torch.cuda.get_rng_state(device=device) if device.type == "cuda" else torch.get_rng_state()
        baseline_gradients: dict[str, torch.Tensor] | None = None

        # Run g=1 first so every other gap can record an exact per-batch dot.
        ordered_gaps = [1.0] + [gap for gap in args.gaps if gap != 1.0]
        for gap in ordered_gaps:
            net.zero_grad(set_to_none=True)
            losses, r = ect_loss_with_fixed_randomness(net, loss_template, schedules[str(gap)], images, labels, t, eps, dropout_state)
            mean_loss = losses.mean()
            if not bool(torch.isfinite(mean_loss)):
                fail(f"non-finite loss at batch={batch_index}, g={gap}")
            mean_loss.backward()
            norm_sq, inf_norm, layer_norm_sq = gradient_statistics(named_parameters)
            gap_key = str(gap)
            if gap == 1.0:
                baseline_gradients = {name: parameter.grad.detach().float().clone() for name, parameter in named_parameters}
                dot_to_reference = norm_sq
            else:
                assert baseline_gradients is not None
                dot_to_reference, _ = vector_dot(named_parameters, baseline_gradients)
            for name, parameter in named_parameters:
                mean_sums[gap_key][name].add_(parameter.grad.detach().double())
            batch_norm_sq_sums[gap_key] += norm_sq
            for layer, value in layer_norm_sq.items():
                batch_layer_norm_sq_sums[gap_key][layer] += value
            batch_rows.append({
                "batch_index": batch_index,
                "gap": gap,
                "loss_mean": finite_scalar(mean_loss, "mean loss"),
                "grad_l2": math.sqrt(norm_sq),
                "grad_l2_sq": norm_sq,
                "grad_linf": inf_norm,
                "dot_to_g1": dot_to_reference,
                "t_mean": finite_scalar(t.mean(), "t mean"),
                "t_min": finite_scalar(t.min(), "t min"),
                "t_max": finite_scalar(t.max(), "t max"),
                "gap_mean": finite_scalar(((t - r) / t).mean(), "gap mean"),
            })
        del baseline_gradients, images, labels, t, eps

    state_after = module_state_hashes(net)
    gradient_slots_after = sum(parameter.grad is not None for _, parameter in named_parameters)
    if state_before != state_after:
        fail("diagnostic changed checkpoint parameter values or buffers")

    moments, layerwise = mean_vector_statistics(mean_sums, 1.0, args.batches, batch_norm_sq_sums, batch_layer_norm_sq_sums)
    batches = batch_residual_rows(batch_rows, moments, 1.0)
    manifest = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "gaps": args.gaps,
        "seed": args.seed,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "parameter_sha256_before": state_before["parameter_sha256"],
        "parameter_sha256_after": state_after["parameter_sha256"],
        "buffer_sha256_before": state_before["buffer_sha256"],
        "buffer_sha256_after": state_after["buffer_sha256"],
        "parameter_values_unchanged": state_before["parameter_sha256"] == state_after["parameter_sha256"],
        "buffer_values_unchanged": state_before["buffer_sha256"] == state_after["buffer_sha256"],
        "gradient_slots_populated_before": gradient_slots_before,
        "gradient_slots_populated_after": gradient_slots_after,
        "randomness_contract": "same images, per-example t, shared epsilon, and dropout RNG state for every g within each minibatch",
        "scalar_fit_formula": "dot(mu_g, mu_1) / dot(mu_1, mu_1)",
        "elapsed_seconds": time.time() - started,
    }
    write_csv(paths["moments"], moments)
    write_csv(paths["layerwise"], layerwise)
    write_csv(paths["batch"], batches)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figure(paths["figure"], moments)
    write_report(paths["report"], manifest, moments, batches)
    print(json.dumps({"status": "passed", "outputs": {key: str(path) for key, path in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
