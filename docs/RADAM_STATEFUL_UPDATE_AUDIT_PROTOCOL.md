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
It is a **single-state measurement**: it does not by itself establish the
#45 temporal-history mechanism.

## Required inputs

| Input | Role |
|---|---|
| `--training-state` | `net` (θ_K), `optimizer_state` (m_K, v_K, n_K), `gradscaler_state` (GradScaler_K when AMP) |
| `--checkpoint` | `loss_fn` / schedule hyperparameters; `loss_fn_state` from the training-state is overlaid when present |
| `--data` | CIFAR (or matching) ImageFolderDataset zip/directory |

The CLI fails closed if any model parameter is missing RAdam state, a moment is
zero/NaN at the collection level, the per-parameter counters are not the same
positive integer `n_K`, or `--amp` is on without `gradscaler_state`.  There is
no silent fresh-scaler fallback.  A mechanism receipt additionally requires
`n_K >= 6` and serialized `successful_optimizer_steps >= 6`, so a warm-up or
fresh-state measurement cannot be mislabelled as Role-D evidence.

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
`Δθ` is zero while the candidate moment map is not a comparable completed
update. The receipt records the skip status rather than emitting a gauge.

## Metrics

### Scalar conventions

The measurement follows #43/#45 exactly.  The two directions are both
reported and never relabelled:

```text
s_K^* = <U_g, U_1> / ||U_1||²       # U_g ≈ s_K^* U_1, update scale
c_K^* = <U_g, U_1> / ||U_g||²       # c_K^* U_g ≈ U_1, candidate LR multiplier
R_opt = ||U_g - s_K^* U_1|| / ||U_1||
```

For the raw gradients the analogous `a_K^*` and `R_grad` use the update-scale
direction: `g_g ≈ a_K^* g_1`, reference-normalized.  Consequently,
`R_opt(K) - R_grad(K)` is an empirical diagnostic with **no claimed sign** on
real ECT, not an invariance theorem.

The analytical RAdam update is retained as an implementation check and reports
its own `s_K_star_predicted`, `c_K_star_predicted`, and `R_pred`; it is not a
substitute for the measured moment-memory gauge below.

### Support-aware coordinate update gauge

With `U_1 = Δθ_1` and `U_g = Δθ_1.3`, the theorem gauge on support is

```text
h_update_i = U_g,i / U_1,i.
```

The receipt records both the exact support `U_1,i != 0` (used for the exact
decomposition) and an optional effective support
`|U_1,i| > --support-atol` for robust coordinate summaries.  Each layer reports
support coordinate/energy coverage plus weighted mean and standard deviation
of `h_update`, and unweighted p05/p50/p95 quantiles.  The old quantity
`||c_K^* U_g^(l) - U_1^(l)|| / ||U_1^(l)||` is retained only under its accurate
name: `layer_residual_with_global_c_star`; it is not `h_{K,i}`.  This field
uses the reverse/LR-matching `c_K^*` convention, whereas `R_opt` uses `s_K^*`.

On the exact support, the receipt separately reports

```text
on_support_gauge_dispersion
  = Σ_{U_1,i≠0} U_1,i² (h_update_i - s_K^*)² / ||U_1||²
off_support_candidate_energy_exact
  = Σ_{U_1,i=0} U_g,i² / ||U_1||².
```

Their sum reconstructs `R_opt²` numerically.  `H_K` is the square root of
that sum and is flagged as an **identity check**, never independent evidence.

### Moment/update mapping consistency check

For effective coordinates with `m_1,i != 0`, `v_1,i > 0`, and `v_g,i > 0`,
the audit reports

```text
h_moment_i = (m_g,i / m_1,i) * sqrt(v_1,i / v_g,i).
```

It also reports the implementation-aware `h_moment_eps` variant and the
update-energy weighted RMSE of `h_update - h_moment_eps` (whole model and per
layer).  For the same rectified RAdam step this is primarily an
**optimizer implementation/algebra consistency check**: both quantities are
two expressions of the same current-step mapping, subject to the recorded
support, epsilon, weight-decay, and numerical details.  It must not be framed
as independent evidence of #45's temporal-history mechanism.

### Critical comparisons

1. `R_opt(K) - R_grad(K)` as a state-conditioned non-scalar-effect diagnostic.
2. Actual `h_update` dispersion, exact off-support candidate energy, and their
   reconstruction of `R_opt²`.
3. `h_update` versus `h_moment_eps` as an optimizer mapping-consistency check.
4. Analytical predicted-vs-actual update agreement, as a separate optimizer
   implementation check.

Temporal-history mechanism evidence requires a **cross-state** design not
implemented by this one-state receipt: measure how `a_K^*` changes with `K`
and test the #45 coordinate-history `D_i` prediction from the corresponding
per-step scale history.  Do not infer that evidence merely from a nonzero
`h_update - h_moment_eps` value at one state.

## Run

```bash
python analysis/radam_stateful_update_audit.py \
  --training-state /path/to/training-state-XXXXXX.pt \
  --checkpoint /path/to/network-snapshot.pkl \
  --data /path/to/cifar10-32x32.zip \
  --state-kimg 128 --batch-size 128 --batch-gpu 16 --support-atol 0 \
  --seed 20260808 --device cuda
```

Outputs (overwritten only on a successful run):

- `analysis/radam_update_audit_stateful.json`
- `analysis/radam_update_stateful_layerwise.csv`

The JSON receipt includes the source commit plus analysis-script SHA-256,
training-state/checkpoint/dataset hashes, `n_K`, serialized successful-step
count, AMP skip telemetry, effective-support threshold/coverage, and
source-state preservation hashes.
