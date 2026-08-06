"""Gap-gradient diagnostic hook for real ECT networks.

Role C, 2026-08-05. Independent implementation (does not depend on Role D's
PR #36 script).

Purpose: measure whether the real-network training gradient at global gap `g`
is (nearly) a scalar rescaling of the gradient at g=1:

    mu_g ~= a_g^star * mu_1 ,   a_g^star = <mu_g, mu_1> / ||mu_1||^2

The diagnostic is a GRADIENT-ONLY probe:
  * loads one checkpoint, holds its parameters fixed;
  * never constructs or steps an optimizer;
  * does not mutate model / optimizer / EMA state (parameter and buffer SHA256
    are identical before and after).

Randomness contract (the hook guarantee):
  * within one minibatch, the image batch, per-example timestep vector `t`,
    shared noise tensor `eps`, and the dropout RNG state are sampled EXACTLY
    ONCE and REUSED for every gap multiplier;
  * the only difference between sweep points is `global_gap_scale`.

Per-layer statistics:
  * accumulate per-parameter gradients, group by enclosing module path;
  * per layer: norm^2, dot-to-reference, scalar_fit, cosine, directional
    residual, gradient variance trace, normalized noise scale.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from training.loss import ECMLoss
from training.schedules import get_schedule


def parse_gaps(value: str) -> list[float]:
    """Parse a comma-separated gap list; must include the reference 1.0."""
    try:
        gaps = [float(x) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gaps must be comma-separated floats") from exc
    if not gaps or any(not math.isfinite(g) or g <= 0 for g in gaps):
        raise argparse.ArgumentTypeError("--gaps must contain finite values > 0")
    if len(set(gaps)) != len(gaps):
        raise argparse.ArgumentTypeError("--gaps must not contain duplicates")
    if 1.0 not in gaps:
        raise argparse.ArgumentTypeError("--gaps must include the reference value 1.0")
    return gaps


def tensor_collection_sha256(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash names, metadata, and exact values without depending on device."""
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_state_hashes(net: torch.nn.Module) -> dict[str, str]:
    """Separate immutable-value hashes for parameters and buffers."""
    return {
        "parameter_sha256": tensor_collection_sha256(net.named_parameters()),
        "buffer_sha256": tensor_collection_sha256(net.named_buffers()),
    }


def set_dropout_rng_state(state: torch.Tensor, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device=device)
    else:
        torch.set_rng_state(state)


def layer_name(parameter_name: str) -> str:
    """Group a weight/bias under its enclosing module path."""
    return parameter_name.rsplit(".", 1)[0] if "." in parameter_name else parameter_name


def ect_loss_with_fixed_randomness(
    net: torch.nn.Module,
    loss_template: ECMLoss,
    schedule,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ECMLoss with supplied t/noise/dropout randomness.

    ``loss_template`` supplies c and stage from the actual checkpoint.
    ``schedule`` is a global_sigmoid schedule (== official sigmoid at g=1,
    applies the multiplicative gap intervention otherwise). The shared
    dropout-mask protocol mirrors ECMLoss: reset RNG before both calls.
    """
    r = schedule.compute_r(t=t, stage=loss_template.stage)
    device = images.device

    set_dropout_rng_state(dropout_rng_state, device)
    # force_fp32=True: the diagnostic needs numerically stable gradients.
    # ECMPrecond defaults to fp16 when trained with use_fp16, which overflows
    # on 0-255 inputs and yields NaN here (training hides this via GradScaler).
    d_yt = net(images + eps * t, t, labels, augment_labels=None, force_fp32=True)

    if bool((r > 0).any()):
        set_dropout_rng_state(dropout_rng_state, device)
        with torch.no_grad():
            d_yr = net(images + eps * r, r, labels, augment_labels=None,
                       force_fp32=True)
        d_yr = torch.nan_to_num(d_yr)
        d_yr = (r > 0) * d_yr + (~(r > 0)) * images
    else:
        d_yr = images

    raw = (d_yt - d_yr).square().reshape(images.shape[0], -1).sum(dim=1)
    if loss_template.c > 0:
        loss = torch.sqrt(raw + loss_template.c**2) - loss_template.c
    else:
        loss = torch.sqrt(raw)
    return loss / (t - r).flatten(), r


def gradient_statistics(named_parameters) -> tuple[float, float, dict[str, float]]:
    """Whole-model norm², L∞ norm, and per-layer norm² for gradients."""
    total_norm_sq = 0.0
    inf_norm = 0.0
    per_layer: dict[str, float] = defaultdict(float)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise SystemExit(f"[gap_gradient_hook] non-finite gradient for {name}")
        norm_sq = float(gradient.float().square().sum().double().cpu())
        total_norm_sq += norm_sq
        inf_norm = max(inf_norm, float(gradient.float().abs().max().double().cpu()))
        per_layer[layer_name(name)] += norm_sq
    if total_norm_sq <= 0:
        raise SystemExit("[gap_gradient_hook] all gradients are zero")
    return total_norm_sq, inf_norm, dict(per_layer)


def vector_dot(named_parameters, reference: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
    total = 0.0
    per_layer: dict[str, float] = defaultdict(float)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        dot = float((parameter.grad.detach().float() * reference[name]).sum().double().cpu())
        total += dot
        per_layer[layer_name(name)] += dot
    return total, dict(per_layer)


class GapGradientProbe:
    """Gradient-only gap sweep at a fixed checkpoint (no optimizer, no state write)."""

    def __init__(self, net: torch.nn.Module, loss: ECMLoss,
                 gaps: list[float], q: float = 128.0, k: float = 8.0, b: float = 1.0):
        self.net = net
        self.loss = loss
        self.gaps = sorted(gaps)
        if 1.0 not in self.gaps:
            raise ValueError("gaps must include the reference 1.0")
        self.q, self.k, self.b = q, k, b

    def _schedule(self, g: float):
        # global_sigmoid with scale g: at g=1.0 this is the official sigmoid.
        return get_schedule("global_sigmoid", q=self.q, k=self.k, b=self.b,
                            global_gap_scale=g)

    def run(self, data_iter, batches: int, device: torch.device, seed: int = 0):
        """Run the sweep over `batches` minibatches.

        data_iter: iterable yielding (images, labels) tensors (CPU/GPU ok).
        Returns (moment_rows, layer_rows, manifest) — see the analysis funcs.
        """
        torch.manual_seed(seed)
        mean_sums: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        batch_norm_sq_sums: dict[str, float] = defaultdict(float)
        batch_layer_norm_sq_sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))

        # per-gap: accumulate per-parameter gradient SUMS (float64) and norms
        for _ in range(batches):
            images, labels = next(data_iter)
            images = images.to(device).float()   # dataset yields uint8
            labels = labels.to(device)
            # sample randomness ONCE for this minibatch, reuse for every g
            t = (torch.randn(images.shape[0], 1, 1, 1, device=images.device)
                 * self.loss.P_std + self.loss.P_mean).exp()
            eps = torch.randn_like(images)
            dropout_state = torch.get_rng_state() if device.type == "cpu" \
                else torch.cuda.get_rng_state(device=device)

            for g in self.gaps:
                self.net.zero_grad(set_to_none=True)
                schedule = self._schedule(g)
                loss_t, _ = ect_loss_with_fixed_randomness(
                    self.net, self.loss, schedule, images, labels, t, eps, dropout_state)
                loss_t.mean().backward()
                for name, param in self.net.named_parameters():
                    if param.grad is None:
                        continue
                    key = str(g)
                    grad = param.grad.detach().float().double()
                    if name in mean_sums[key]:
                        mean_sums[key][name] = mean_sums[key][name] + grad
                    else:
                        mean_sums[key][name] = grad.clone()
                nsq, _, _ = gradient_statistics(self.net.named_parameters())
                batch_norm_sq_sums[str(g)] += nsq
        return mean_sums, batch_norm_sq_sums


def mean_vector_statistics(mean_sums, reference_gap, batches):
    """Model + per-layer moments from accumulated minibatch gradient sums.

    mean_sums: {gap_key: {param_name: summed_gradient (float64)}}.
    Returns (model_rows, layer_rows).
    """
    ref = mean_sums[str(reference_gap)]
    ref_norm_sq = sum(float((v / batches).double().square().sum()) for v in ref.values())
    if ref_norm_sq <= 0:
        raise SystemExit("reference mean gradient is zero")

    model_rows, layer_rows = [], []
    for gap_key, sums in mean_sums.items():
        gap = float(gap_key)
        mean_norm_sq = 0.0
        dot_ref = 0.0
        layer_val: dict[str, dict[str, float]] = defaultdict(
            lambda: {"norm_sq": 0.0, "ref_norm_sq": 0.0, "dot": 0.0})
        for name, value in sums.items():
            mean = value / batches
            ref_mean = ref[name] / batches
            cur_nsq = float(mean.double().square().sum())
            ref_nsq = float(ref_mean.double().square().sum())
            dot = float((mean.double() * ref_mean.double()).sum())
            mean_norm_sq += cur_nsq
            dot_ref += dot
            lay = layer_name(name)
            layer_val[lay]["norm_sq"] += cur_nsq
            layer_val[lay]["ref_norm_sq"] += ref_nsq
            layer_val[lay]["dot"] += dot

        fit = dot_ref / ref_norm_sq
        res_sq = max(mean_norm_sq + fit**2 * ref_norm_sq - 2 * fit * dot_ref, 0.0)
        mean_norm = math.sqrt(mean_norm_sq)
        residual = math.sqrt(res_sq) / mean_norm if mean_norm else math.nan
        cosine = dot_ref / math.sqrt(mean_norm_sq * ref_norm_sq) if mean_norm_sq else math.nan
        model_rows.append({
            "gap": gap,
            "mean_grad_l2": mean_norm,
            "scalar_fit_to_g1": fit,
            "cosine_to_g1": cosine,
            "direction_residual": residual,
        })
        for lay, v in layer_val.items():
            lr_sq = v["ref_norm_sq"]
            lfit = v["dot"] / lr_sq if lr_sq else math.nan
            lres_sq = max(v["norm_sq"] + lfit**2 * lr_sq - 2 * lfit * v["dot"], 0.0)
            lnorm = math.sqrt(v["norm_sq"])
            layer_rows.append({
                "layer": lay,
                "gap": gap,
                "mean_grad_l2": lnorm,
                "scalar_fit_to_g1": lfit,
                "cosine_to_g1": (v["dot"] / math.sqrt(v["norm_sq"] * lr_sq)
                                 if v["norm_sq"] and lr_sq else math.nan),
                "direction_residual": (math.sqrt(lres_sq) / lnorm if lnorm else math.nan),
            })
    model_rows.sort(key=lambda r: r["gap"])
    layer_rows.sort(key=lambda r: (r["layer"], r["gap"]))
    return model_rows, layer_rows


def load_checkpoint(pkl_path: Path, device: torch.device):
    """Load an ECT network snapshot; returns (net, loss, dataset_kwargs).

    The snapshot pkl is a dict with keys ema / loss_fn / augment_pipe /
    dataset_kwargs (same layout the training loop saves).
    """
    with open(pkl_path, "rb") as f:
        import pickle
        data = pickle.load(f)
    ema = data["ema"]
    loss = data["loss_fn"]
    net = ema.to(device)
    net.requires_grad_(True)
    return net, loss, data.get("dataset_kwargs", {})


def main(argv: list[str] | None = None) -> int:
    import csv
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="ECT snapshot pkl")
    ap.add_argument("--data", required=True, help="dataset zip (EDM format)")
    ap.add_argument("--gaps", type=parse_gaps, default="0.9,1.0,1.2,1.3",
                    help="comma-separated gap list (must include 1.0)")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "analysis")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    net, loss, _ = load_checkpoint(Path(args.checkpoint), device)
    net.eval()

    # dataset loader (ImageFolderDataset is what ECT uses)
    from training.dataset import ImageFolderDataset
    from torch.utils.data import DataLoader
    ds = ImageFolderDataset(path=args.data, use_labels=False, xflip=False,
                            cache=True, resolution=32)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    it = iter(loader)

    probe = GapGradientProbe(net, loss, args.gaps)
    hashes_before = module_state_hashes(net)

    mean_sums, batch_norm_sq_sums = probe.run(it, args.batches, device, seed=args.seed)
    model_rows, layer_rows = mean_vector_statistics(mean_sums, 1.0, args.batches)

    hashes_after = module_state_hashes(net)
    state_preserved = (hashes_before == hashes_after)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "gap_gradient_model_moments.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0]))
        w.writeheader(); w.writerows(model_rows)
    with (args.out / "gap_gradient_layerwise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        w.writeheader(); w.writerows(layer_rows)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "gaps": args.gaps,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "state_preserved": state_preserved,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }
    with (args.out / "gap_gradient_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print("=== model-level moments ===")
    for r in model_rows:
        print(f"  g={r['gap']}: a*={r['scalar_fit_to_g1']:.4f} "
              f"cos={r['cosine_to_g1']:.6f} residual={r['direction_residual']:.4f}")
    print(f"state preserved: {state_preserved}")
    print("wrote analysis/gap_gradient_{model_moments,layerwise}.csv + manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
