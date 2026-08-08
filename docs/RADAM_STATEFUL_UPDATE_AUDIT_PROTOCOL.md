# Stateful RAdam update audit protocol

## Purpose

`analysis/radam_stateful_update_audit.py` is the core optimizer-level check.
Fresh-state `c_0^*` from `analysis/radam_update_gauge.py` is only an
implementation sanity probe (empty moments, new GradScaler).  This audit starts
from a **real non-zero** training optimizer state

```text
z_K = (θ_K, m_K, v_K, n_K, GradScaler_K)
```

restored from `training-state-*.pt`, then asks whether the gap intervention at
that state remains a near-scalar update rematch once RAdam moments are live.

## Required inputs

| Input | Role |
|---|---|
| `--training-state` | `net` (θ_K), `optimizer_state` (m_K, v_K, n_K), `gradscaler_state` (GradScaler_K when AMP) |
| `--checkpoint` | `loss_fn` / schedule hyperparameters; `loss_fn_state` from the training-state is overlaid when present |
| `--data` | CIFAR (or matching) ImageFolderDataset zip/directory |

The CLI fails closed if any model parameter is missing RAdam state, a moment is
zero/NaN at the collection level, the per-parameter counters are not the same
positive integer `n_K`, or `--amp` is on without `gradscaler_state`.  There is
no silent fresh-scaler fallback.

## Invariants

It creates two disposable branches that share:

- the exact same restored `θ_K`, `m_K`, `v_K`, `n_K`, and GradScaler state;
- the same one-time minibatch, `t`, noise, and dropout RNG state.

The sole branch difference is `global_gap_scale`: `1.0` versus `1.3`.
Augmentation-enabled checkpoints fail closed.  The update follows the training
loop order exactly: `scale → backward → unscale → sanitize → step → update`.
The tool does **not** enter `torch.autocast`.

The source `z_K` is never stepped.  Before/after SHA-256 hashes of parameters,
optimizer state, and GradScaler must match.

If AMP skips `optimizer.step` on either branch, the gauge is undefined: actual
`Δθ` is zero while moment-predicted updates are not, so `R_opt` and
predicted-vs-actual `h` are not comparable.

## Metrics

### Shared residual convention

`R_grad` and `R_opt` use the **same** least-squares algebra
(`residual_convention = c_star_probe_to_reference`):

```text
scale * probe_{1.3} ≈ reference_1
R = ||scale * probe_{1.3} - reference_1|| / ||reference_1||
```

so `R_opt(K) - R_grad(K)` subtracts isomorphic quantities.  Both equal
`|sin θ|` of the angle between the two vectors.

### Gradient side

```text
a_K^*   = <g_{1.3}, g_1> / ||g_1||²          # gap-hook / LR-matching coefficient
R_grad  = ||c_g g_{1.3} - g_1|| / ||g_1||    # c-convention, matched to R_opt
          where c_g = <g_{1.3}, g_1> / ||g_{1.3}||²
```

`R_grad_a_convention = ||g_{1.3} - a_K^* g_1|| / ||g_{1.3}||` is retained as
telemetry; it equals `R_grad` up to numerics (`R_grad_a_c_abs_gap`).

### Optimizer side (actual update Δθ)

```text
c_K^*   = <d_{1.3}, d_1> / ||d_{1.3}||²      # c_K^* d_{1.3} ≈ d_1
R_opt   = ||c_K^* d_{1.3} - d_1|| / ||d_1||
```

### Idealized moment-predicted update

Using the restored moments and the post-sanitize gradients, the tool evaluates
the analytical RAdam map (matching `torch.optim.RAdam` with `weight_decay=0`)
**without** stepping the source optimizer:

```text
s_K^*   = <p_{1.3}, p_1> / ||p_{1.3}||²      # s_K^* p_{1.3} ≈ p_1
R_pred  = ||s_K^* p_{1.3} - p_1|| / ||p_1||
```

### Layer / coordinate summary

For each enclosing module path `i`:

- `h_{K,i}` actual: layer residual of `d` under whole-model `c_K^*`
- `h_{K,i}` predicted: layer residual of `p` under whole-model `s_K^*`
- `H_K`: energy-weighted RMS of the actual layer residual energies

`H_K = R_opt` is an **identity** (same residual energy, two aggregations).  Do
not treat equality as evidence.  Off-support energy is the relative energy of
`d_1` orthogonal to `span(d_{1.3})`:

```text
off_support = ||d_1 - c_K^* d_{1.3}||² / ||d_1||² = R_opt²
```

### Critical comparisons

1. `R_opt(K) - R_grad(K)` — under the shared `c`-convention, how much live
   moments inflate (or shrink) the directional residual relative to the raw
   gradient residual.
2. Whether idealized moment-predicted `h_{K,i}` matches actual-update
   `h_{K,i}` (and whether `predicted_vs_actual_relative_l2` is near zero on
   each branch).

## Run

```bash
python analysis/radam_stateful_update_audit.py \
  --training-state /path/to/training-state-XXXXXX.pt \
  --checkpoint /path/to/network-snapshot.pkl \
  --data /path/to/cifar10-32x32.zip \
  --state-kimg 128 --batch-size 128 --batch-gpu 16 \
  --seed 20260808 --device cuda
```

Outputs (overwritten only on a successful run):

- `analysis/radam_update_audit_stateful.json`
- `analysis/radam_update_stateful_layerwise.csv`
