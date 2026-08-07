# Fresh-RAdam update gauge protocol

## Purpose

`analysis/radam_update_gauge.py` answers the optimizer-level question that the
raw-gradient diagnostic cannot: for one common pretrained EDM state, how close
are the *actual first RAdam updates* for `g=1.0` and `g=1.3`?

It works directly with any ECT snapshot, including the prospective
32/64/128/256 kimg states.  `--state-kimg` is provenance only; it does not
change parameters, optimizer initialization, or the calculation.

## Invariants

For a single invocation the tool loads one `ema` network snapshot and creates
two disposable branches.  Both branches have:

- the exact same pretrained parameter and buffer values;
- a newly constructed `torch.optim.RAdam(lr=1e-4, betas=(0.9,0.999), eps=1e-8)`;
- empty RAdam state (`m_0=v_0=0`, optimizer step zero);
- a GradScaler cloned from the same initial state (default scale 65536);
- the same, one-time sampled minibatch, `t`, shared noise tensor, and dropout
  RNG state.

The sole branch difference is `global_gap_scale`: `1.0` versus `1.3`.
Augmentation-enabled checkpoints fail closed because paired augmentation is
not implemented.  The update follows the training-loop order exactly:
`scale → backward → unscale → sanitize → step → update`.  It intentionally
does **not** enter `torch.autocast`: the repository's training loop uses a
GradScaler while the EDM network itself selects fp16.

When the training run used `batch-gpu < batch`, provide `--batch-gpu` as well.
The probe then samples `t`, noise, and dropout state separately for each
microbatch, accumulates their gradients in the training-loop order, and makes
one RAdam step.  The audit records the number of accumulation rounds.

The source model, source fresh optimizer, and source GradScaler never receive
a step.  Their before/after SHA-256 hashes must match.  Each branch also
records its virtual post-step hashes, which normally differ from the branch's
pre-step values; this proves that a real RAdam update occurred instead of a
gradient proxy.

## Calculation

Let `d1 = Δθ_1` and `d13 = Δθ_1.3`.  The audit reports their norms and cosine,
then uses the requested convention:

```
c0_star = ||d13||² / <d13, d1>
rho0    = ||c0_star * d13 - d1|| / ||d1||
```

`whole_model_residual` is `rho0`; `radam_update_layerwise.csv` contains the
same calculation within each enclosing parameter module.  Its
`layerwise_residual_with_model_c0_star` applies the single whole-model
`c0_star` to every layer, while `layerwise_residual` uses each layer's own
requested `c0_star` to localize direction mismatch.  For interpretation, the
output additionally reports `least_squares_scale_1p3_to_1`, which is
`<d13,d1>/||d13||²`.  It is the reciprocal of `c0_star` for exactly collinear
updates.  Keeping both fields avoids silently changing the requested formula.

At the first RAdam step, the optimizer is in its unrectified regime.  Therefore
the updates are expected to remain close to a gradient-scale relation, but the
audit, not that expectation, is the evidence.

## Run

```bash
python analysis/radam_update_gauge.py \
  --checkpoint /path/to/network-snapshot.pkl \
  --data /path/to/cifar10-32x32.zip \
  --state-kimg 128 --batch-size 128 --batch-gpu 16 \
  --seed 20260807 --device cuda
```

The command overwrites only these two explicit analysis outputs:

- `analysis/radam_update_audit_fresh.json`
- `analysis/radam_update_layerwise.csv`

The JSON includes update cosine/norms, whole-model residual, AMP unscale and
skipped-step telemetry, all source and virtual branch hashes, optimizer steps,
checkpoint/data hashes, and runtime provenance.  The CSV includes layerwise
norms, cosine, both scale conventions, and residuals.
