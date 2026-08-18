"""Real-history paired sweep: recover δ_j from a real ECT state.

Loads a real training state (e.g. arm_a g=1.0 at 256 kimg), clones it into two
arms that differ ONLY by the gap scale (g=1.0 reference vs g=1.3 candidate),
and replays N optimizer steps with identical (images, t, noise, dropout) per
step. Each step records the paired gradients (G_j, G^g_j) and the single-step
parameter updates (u1_j, ug_j).

Outputs the inputs for moment_memory_prediction.py:
  - grad_history_1.npy   (N, d) reference gradients
  - grad_history_g.npy   (N, d) candidate gradients
  - u1.npy / ug.npy      final-step updates
plus a JSON with per-step deltas and the actual h^update series.

This is the real-data analogue of the #47 self-check: it measures whether the
#45 moment-memory chain predicts the real optimizer distortion from the real
δ_j history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dnnlib
from training.schedules import get_schedule
from training.dataset import ImageFolderDataset
from torch.utils.data import DataLoader

from analysis.radam_stateful_update_audit import load_training_state
from analysis import radam_update_gauge as gauge


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grad_flat(net: torch.nn.Module) -> np.ndarray:
    parts = []
    for p in net.parameters():
        if p.grad is None:
            parts.append(np.zeros(p.numel(), dtype=np.float64))
        else:
            parts.append(p.grad.detach().float().cpu().numpy().reshape(-1).astype(np.float64))
    return np.concatenate(parts)


def _update_flat(before: list[np.ndarray], after_net: torch.nn.Module) -> np.ndarray:
    after = [p.detach().cpu().numpy().reshape(-1).astype(np.float64)
             for p in after_net.parameters()]
    return np.concatenate(after) - np.concatenate(before)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-state", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g-candidate", type=float, default=1.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=Path("analysis/real_history"))
    a = ap.parse_args(argv)
    if a.n_steps < 1 or a.batch_size < 1 or a.lr <= 0:
        raise SystemExit("--n-steps, --batch-size, and --lr must be positive")
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite existing raw-history directory: {a.out}")

    device = torch.device(a.device)
    net, optimizer, scaler_state, loss_fn_state, meta = load_training_state(
        a.training_state, device, lr=a.lr, betas=(0.9, 0.999), eps_opt=1e-8)
    # The prospective branches are clones.  Keep the restored source object
    # untouched so the receipt can prove the replay did not commit a step.
    source_params_before = gauge.module_state_hashes(net)
    source_optimizer_before = gauge.state_sha256(optimizer.state_dict())

    # loss_fn from the checkpoint (q/k/b/stage) — for schedule construction
    with open(a.checkpoint, "rb") as f:
        payload = pickle.load(f)
    loss_fn = payload["loss_fn"]
    q, k, b = float(loss_fn.q), float(loss_fn.k), float(loss_fn.b)
    loss_template = payload["loss_fn"]  # provides c, stage
    print(f"checkpoint loss: q={q} k={k} b={b} c={loss_template.c} schedule={loss_template.schedule.name}")

    sched_ref = get_schedule("global_sigmoid", q=q, k=k, b=b, global_gap_scale=1.0)
    sched_cand = get_schedule("global_sigmoid", q=q, k=k, b=b, global_gap_scale=a.g_candidate)

    # Both prospective arms are cloned from the same real state (identical
    # params, moments, step count; only the schedule/gap differs).
    import copy
    net_ref = copy.deepcopy(net)
    net_cand = copy.deepcopy(net)
    opt_ref = torch.optim.RAdam(net_ref.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_ref.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    opt_cand = torch.optim.RAdam(net_cand.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_cand.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    # data loader
    torch.manual_seed(a.seed)
    ds = ImageFolderDataset(path=a.data, use_labels=False, xflip=False, cache=True, resolution=32)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
                        drop_last=True, generator=torch.Generator().manual_seed(a.seed))
    it = iter(loader)

    from analysis.gap_gradient_hook import ect_loss_with_fixed_randomness as fixed_loss

    G1_hist = []; Gg_hist = []; u1_hist = []; ug_hist = []; delta_hist = []
    for step in range(a.n_steps):
        images, labels = next(it)
        images = images.to(device).float() / 127.5 - 1
        labels = labels.to(device)
        t = (torch.randn(images.shape[0], 1, 1, 1, device=device) * loss_fn.P_std + loss_fn.P_mean).exp()
        eps = torch.randn_like(images)
        dropout = torch.cuda.get_rng_state(device=device) if device.type == "cuda" else torch.get_rng_state()

        # reference arm
        opt_ref.zero_grad(set_to_none=True)
        l1, _ = fixed_loss(net_ref, loss_template, sched_ref, images, labels, t, eps, dropout)
        l1.mean().backward()
        G1 = _grad_flat(net_ref)
        b1 = [p.detach().cpu().numpy().reshape(-1).copy() for p in net_ref.parameters()]
        opt_ref.step()
        u1 = _update_flat(b1, net_ref)

        # candidate arm (g=1.3), same randomness
        opt_cand.zero_grad(set_to_none=True)
        lc, _ = fixed_loss(net_cand, loss_template, sched_cand, images, labels, t, eps, dropout)
        lc.mean().backward()
        Gg = _grad_flat(net_cand)
        bc = [p.detach().cpu().numpy().reshape(-1).copy() for p in net_cand.parameters()]
        opt_cand.step()
        ug = _update_flat(bc, net_cand)

        G1_hist.append(G1); Gg_hist.append(Gg); u1_hist.append(u1); ug_hist.append(ug)
        dj = float(np.sum(Gg * G1) / max(np.sum(G1 * G1), 1e-30)) - 1.0
        delta_hist.append(dj)

    G1_arr = np.stack(G1_hist); Gg_arr = np.stack(Gg_hist)
    u1_arr = np.stack(u1_hist); ug_arr = np.stack(ug_hist)
    u1_final = u1_arr[-1]; ug_final = ug_arr[-1]

    a.out.mkdir(parents=True, exist_ok=False)
    np.save(a.out / "grad_history_1.npy", G1_arr)
    np.save(a.out / "grad_history_g.npy", Gg_arr)
    np.save(a.out / "u1.npy", u1_final)
    np.save(a.out / "ug.npy", ug_final)
    # Full per-step update history (T, d) so the predictor can evaluate h^t vs
    # h^actual at the SAME step t (fixes the 1-step control endpoint mismatch).
    np.save(a.out / "u1_history.npy", u1_arr)
    np.save(a.out / "ug_history.npy", ug_arr)
    source_params_after = gauge.module_state_hashes(net)
    source_optimizer_after = gauge.state_sha256(optimizer.state_dict())
    raw_paths = {
        name: a.out / name for name in (
            "grad_history_1.npy", "grad_history_g.npy", "u1.npy", "ug.npy",
            "u1_history.npy", "ug_history.npy",
        )
    }
    with (a.out / "sweep_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 2,
            "protocol": "canonical-pr47-pr58-prospective-scalar-history-v1",
            "training_state": str(a.training_state),
            "training_state_sha256": sha256_file(a.training_state),
            "checkpoint": str(a.checkpoint),
            "checkpoint_sha256": sha256_file(a.checkpoint),
            "dataset": str(a.data),
            "dataset_sha256": sha256_file(Path(a.data)),
            "source_state_non_committing": {
                "parameter_hash_before": source_params_before,
                "parameter_hash_after": source_params_after,
                "optimizer_state_hash_before": source_optimizer_before,
                "optimizer_state_hash_after": source_optimizer_after,
                "preserved": source_params_before == source_params_after
                and source_optimizer_before == source_optimizer_after,
            },
            "gradscaler_state_present": scaler_state is not None,
            "training_state_meta": meta,
            "n_steps": a.n_steps,
            "g_candidate": a.g_candidate,
            "reference_gain": 1.0,
            "batch_size": a.batch_size,
            "probe_rng_seed": a.seed,
            "lr": a.lr,
            "q": q, "k": k, "b": b,
            "loss_c": loss_template.c,
            "delta_hist": delta_hist,
            "delta_mean": float(np.mean(delta_hist)),
            "delta_std": float(np.std(delta_hist)),
            "raw_artifacts": {
                name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for name, path in raw_paths.items()
            },
        }, f, indent=2)

    print(f"=== real paired sweep ({a.n_steps} steps, g=1.0 vs {a.g_candidate}) ===")
    print(f"δ_j: mean={np.mean(delta_hist):.4f}, std={np.std(delta_hist):.4f}")
    print(f"final u1 norm={np.linalg.norm(u1_final):.3f}, ug norm={np.linalg.norm(ug_final):.3f}")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
