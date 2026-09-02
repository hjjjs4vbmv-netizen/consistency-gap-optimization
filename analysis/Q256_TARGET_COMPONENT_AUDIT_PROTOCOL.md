# q=256 target-component audit protocol

Status: **v3 preflight passed; Amendment 001 was frozen after the first full
matrix validation exposed a horizon-hash specification error and before any
matrix-level result analysis; v2 artifacts must be regenerated**  
Scope: descriptive, frozen-state diagnostics on already trained models; no
training, optimizer step, sampling, FID, or KID.  The estimand is the
**FP32 reference one-sided-stop-gradient objective field evaluated at the
realized online state**.  Because the formal runs used an FP16-capable network
and AMP/GradScaler, this is not labeled as the native mixed-precision training
gradient.

## Question and estimands

At one common online model state, one common minibatch, and one common
realization of (t), noise, and dropout, virtual cells (A,B,C,D) use the
frozen q=256 target-by-denominator definitions in `training/loss.py`.  Let

\[
s=1/1.1,\qquad
\tau_{\mathrm{tar}}=G_B-sG_A.
\]

The primary descriptive quantities are computed on the gradient of the
equal-weight mean over all audit examples, not by averaging nonlinear
batch-level summaries:

\[
R_{\mathrm{tar},A}
=\frac{\lVert\tau_{\mathrm{tar}}\rVert_2}{\lVert G_A\rVert_2},\qquad
\cos(\tau_{\mathrm{tar}},G_A),\qquad
a^\star=\frac{\langle G_B,G_A\rangle}{\lVert G_A\rVert_2^2}.
\]

`R_tar` in the output means (R_{\mathrm{tar},A}).  To prevent denominator
ambiguity, the runner also reports the same fixed-(s) residual normalized by
(lVert G_B\rVert_2) and by (lVert sG_A\rVert_2), plus the best scalar-fit
residual normalized by (lVert G_B\rVert_2).  The fixed-(s) target component
must not be conflated with the best-fit residual.

## Frozen measurement matrix

- Training seeds: 3, 4, 5.
- State arm: A (baseline trajectory) only for the primary audit.
- Budgets: 256, 512, 768, and 1024 kimg.
- Model tensor source: the online `state["net"]`, not EMA.
- Audit examples per state: 8 fixed minibatches x 16 examples = 128.  This
  matches the formal run's per-GPU microbatch size and is the primary matrix;
  a larger diagnostic sample must be labeled as sensitivity analysis.
- Audit RNG seed: 20260823 for dataset order and stochastic inputs.
- Arithmetic: forced-FP32 forward/backward; float64 gradient accumulation and
  metric reductions; no AMP or GradScaler.  The manifest records the source
  run's `use_fp16` and AMP configuration separately.
- Dataset: the same immutable CIFAR-10 archive and ordering rule at every
  state.  Batch image, label, (t), noise, and dropout-state hashes are saved.

The runner requires an explicit `--run-kind`.  A `primary` run is admitted
only for the frozen A-state matrix above, exactly 8x16 examples, audit seed
20260823, the canonical dataset digest, and the (10^{-4}) identity tolerance.
A one-batch `smoke` run has a different schema and the non-primary status
`PASS_SMOKE_NOT_PRIMARY`; it cannot be ingested as a primary measurement.
Primary gradient measurements require CUDA; CPU is admitted only for a
no-gradient preflight or a run labeled `smoke`.

An optional B-state sensitivity audit may be designed after inspecting the
primary A-state results.  It is not part of this protocol and must not be
silently pooled with the primary matrix.

## Admission and identity gates

Each run fails closed unless all of the following hold:

1. The strict training state contains the expected arm and exact `cur_nimg`,
   q=256, (c=0), native sigmoid schedule, stage 0, no augmentation, and a
   present, valid trajectory-config hash.  The formal loss contract is frozen
   at (P_mean=-1.1), (P_std=2), (sigma_data=0.5), (k=8), and (b=1).
2. The loss is reconstructed from
   `training_state.trajectory_config.loss_kwargs` and then receives the saved
   `loss_fn_state`.  A snapshot loss, when present, must agree with this
   contract.
3. The snapshot EMA and training-state EMA have identical parameter and buffer
   hashes.  The measured network remains the online network.
4. The realized samplewise denominator ratio is constant at (s=1/1.1); any
   clipping-induced sample dependence rejects the constant-(s) audit.
5. Both loss and gradient identities
   (G_D=sG_A) and (G_B=sG_C) have relative L2 error at most (10^{-4}).
6. Gradients are finite and (G_A\ne0).  With the production (c=0) norm
   loss, an exactly zero pair residual has no unique ordinary derivative and
   is an audit failure rather than an imputed value.
7. Parameter and buffer hashes, pre-existing gradient-buffer emptiness, and
   CPU/CUDA RNG state are preserved by the diagnostic process.  No optimizer
   is constructed or stepped.
8. CUDA runs require deterministic algorithms, cuDNN benchmark off, cuDNN
   deterministic on, TF32 off, and a cuBLAS workspace configuration set before
   process launch.  The primary matrix requires
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `:16:8` is admitted only for smoke
   runs.  These settings, GPU name, Python/platform, and
   PyTorch/CUDA/cuDNN versions are written to the manifest.
9. The two CSV artifacts are SHA256-bound by the manifest and published with
   the manifest as one completed directory.  CUDA manifests record peak
   allocated and reserved memory so the one-batch smoke can gate the matrix.
10. The manifest binds the runner, protocol, matrix validator, loss/schedule,
    dataset/model, construction, and persistence code used on the execution
    path.  Matrix admission requires these recorded digests to agree with the
    validator's current implementation closure.

## Matrix-level admission gate

The 12 primary cells are admitted as one frozen matrix, not as independently
selectable successful runs.  Before executing the matrix, freeze and test a
matrix validator.  Before any result is summarized, that validator must reject
the matrix unless all of the following hold:

1. there is exactly one manifest for every seed-by-budget cell in
   ({3,4,5} x {256,512,768,1024}) and no additional cell;
2. every manifest has schema
   `ect.q256.target-component-audit-primary/v2`, status
   `PASS_PRIMARY_COMMON_STATE_GRADIENT_AUDIT`, `run_kind=primary`, and a
   passing identity, energy-reconstruction, and overall audit gate;
3. the dataset digest, implementation digests, estimand, arm definitions,
   audit seed, batch count, batch size, identity tolerance, state loss
   contract, and deterministic runtime settings agree across all cells;
4. both CSV digests verify against every manifest, each batch CSV contains
   exactly 8 batches of 16 examples, and the per-batch image, label, (t),
   noise, dropout-state, target-endpoint, and denominator-endpoint hashes are
   identical across all 12 cells; and
5. the raw trajectory-config digest is retained per cell, while a second
   `trajectory_dynamics_sha256` computed after removing only `total_kimg` must
   agree across budgets within each seed; `trajectory_total_kimg` must be an
   integer no smaller than the audited budget; and
6. cell-varying provenance is confined to the declared training seed, budget,
   checkpoint/state/receipt paths and hashes, declared terminal horizon, and
   measurement resource fields. A CUDA ordinal may differ, but the recorded
   GPU model and software/runtime contract must agree.

The validator's own version or digest and its pass receipt accompany any
matrix-level table or figure.  A complete set of per-cell manifests is
therefore necessary but not sufficient for matrix admission.

## Interpretation contract

The audit measures common-state objective-gradient geometry.  It may show
whether the realized target component changes across budgets or seeds.  It
does not estimate mediation, attribute a percentage of FID change, turn
audit batches into independent training replicates, or establish that an
instantaneous component causes learning-curve contraction.  Comparisons with
paired B-minus-A FID curves are seed-resolved and descriptive.

## Runner

`analysis/q256_target_component_audit.py` writes:

- `target_component_manifest.json`: provenance, state contract, global
  quantities, and pass/fail gates;
- `target_component_batches.csv`: fixed-input hashes and batch diagnostics;
- `target_component_layers.csv`: layerwise geometry, with zero-reference
  layers marked undefined rather than dropped.  Each layer's `a_star` and
  best-fit residual use a layer-specific scalar; they are not a decomposition
  of the whole-model best-fit scalar.  The manifest reports the median, upper
  quantiles, and maximum layerwise (R_tar), and verifies that layer energies
  reconstruct the whole-model norms.

The focused CPU tests and the final read-only remote v3 asset/state preflight
are complete for the implementation hashes recorded by the v3 receipt; see
`analysis/Q256_TARGET_COMPONENT_AUDIT_PREFLIGHT_REPORT.md` and its local
manifest copy.  This closes the no-gradient implementation gate, including
cuDNN-version provenance, the primary cuBLAS workspace policy, and the matrix
validator.  Running either a CUDA smoke measurement or the frozen matrix still
required a separate explicit go decision. That decision was supplied on
2026-08-25. Amendment 001 records the first matrix-validation failure and the
bounded v2 provenance correction; v1 measurements are retained as failed-gate
provenance and are not admissible as the primary matrix.

The audit runner requires Python >=3.10.  The completed server preflight used
Python 3.10 with PyTorch 2.3; the repository's legacy Python 3.9 environment
declaration is not the runtime contract for this diagnostic.
