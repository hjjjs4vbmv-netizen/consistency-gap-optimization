# Gap-gradient diagnostic protocol (Role C, rev.2)

Date: 2026-08-06. PR #40 rev.2 — after the review that flagged the protocol
mismatch vs the real training path.

## Question

At a fixed real ECT checkpoint, is the training gradient under global gap `g`
(nearly) a scalar rescaling of the gradient at g=1? I.e. does
`mu_g ~= a_g^star * mu_1` hold, so that a single scalar learning-rate matching
`eta_g = eta_1 / a_g^star` is numerically applicable (SGD sense)?

## Protocol (now matches the real training path)

1. **Input domain**: images are normalized exactly like
   `training/ct_training_loop.py:870`:
   `images = images.to(device).to(torch.float32) / 127.5 - 1` → range **[-1,1]**.
2. **Network mode**: the EMA is switched to **`net.train()`** (dropout active),
   matching `ct_training_loop.py:577` (`net.train().requires_grad_(True)`).
   Dropout RNG state is captured once per minibatch and REUSED across gaps.
3. **Schedule**: `global_sigmoid` with q/k/b/stage read from the checkpoint's
   `loss_fn` (NOT hardcoded); the checkpoint's own schedule is verified to be
   the official `sigmoid` (the fixed baseline), and g=1 reproduces it exactly.
4. **Augmentation**: the saved `augment_pipe` is loaded; both checkpoints used
   here have `augment_pipe=None`, recorded in the manifest
   (`augment_used=false`, `augment_randomness_fixed=true`). If a checkpoint had
   augment, its randomness would be fixed and reused across gaps.
5. **Precision**: the probe runs FP32 (`force_fp32=True`) as the numerically
   stable **reference**. This is deliberate: the diagnostic needs the *unscaled*
   finite gradient, and AMP GradScaler in training would hide fp16 overflow.
   The FP32-vs-AMP direction equivalence is an open caveat (the protocol note
   below), not asserted here.
6. **No state mutation**: no optimizer created/stepped; parameter and buffer
   SHA256 identical before/after (`state_preserved: true`).

## Metrics

Per gap (model-level and per-layer):
- `mu_g = B^-1 sum_b grad_b(g)` (mean gradient over B minibatches)
- `a_g^star = <mu_g, mu_1> / ||mu_1||^2` (scalar fit to g=1)
- `R_mean(g) = ||mu_g - a_g^star mu_1|| / ||mu_g||` (directional residual)
- per-layer: same quantities per module path; plus cosine.

LR matching (SGD, corrected per review):
`eta_g * mu_g = eta_1 * mu_1`  ⇒  `eta_g = eta_1 / a_g^star`  (inverse, not a*).

## Results (q128, 256 kimg, seed 3, g_screen g1_0 checkpoint)

| g | a* | cos | residual |
|---:|---:|---:|---:|
| 0.9 | 1.1104 | 0.999999 | 0.0013 |
| 1.0 | 1.0000 | 1.000000 | 0 |
| 1.2 | 0.8340 | 0.999998 | 0.0019 |
| 1.3 | 0.7705 | 0.999996 | 0.0028 |

16 minibatches, batch size 64, seed 20260806. Per-layer residuals (208 layers,
g=1.3): mean 0.0016, max 0.009. `state_preserved: true`.

## Comparison with Role D (valid baseline)

| source | g=0.9 residual | g=1.2 | g=1.3 | per-layer max |
|---|---:|---:|---:|---:|
| Role D (1000 kimg, normalized+train) | 0.48% | 0.88% | 1.35% | ~12.4% |
| Role C rev.2 (256 kimg, normalized+train) | 0.13% | 0.19% | 0.28% | ~0.9% |

Under the corrected protocol, Role C's residual is small but **no longer the
near-1e-4 perfect value of the buggy run**; it is 3-5x smaller than Role D's.
The remaining difference is consistent with the 256-kimg vs 1000-kimg stage
difference, but with a single seed and different batch counts this is a
**descriptive** comparison, not an inference.

## Protocol caveats / open items

- FP32 reference only; AMP-path gradient equivalence not asserted (needs an
  unscaled-gradient comparison, open).
- RAdam-update equivalence is NOT licensed by gradient collinearity; an
  optimizer-state diagnostic (Role D boundary) is required before claiming LR
  matching works for the actual RAdam training.
- Single seed, one checkpoint; per-layer max ~0.9% is small but nonzero —
  whether it is "scalar-matchable" depends on the RAdam check.

## Files
- `analysis/gap_gradient_hook.py` — the probe (protocol above).
- `analysis/gap_gradient_model_moments.csv`, `gap_gradient_layerwise.csv`.
- `analysis/gap_gradient_manifest.json` — provenance (hashes, mode, domain, etc).
- `tests/test_gap_gradient_hook.py` — 10 tests incl. real-ECMLoss parity.
