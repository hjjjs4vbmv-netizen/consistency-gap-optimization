# Gap-gradient diagnostic protocol (Role C, rev.3)

Date: 2026-08-06. PR #40 rev.3, after correcting the training-path and
evidence-consistency issues identified during review.

## Question

At a fixed real ECT checkpoint, is the training gradient under global gap `g`
(nearly) a scalar rescaling of the gradient at g=1? I.e. does
`mu_g ~= a_g^star * mu_1` hold, so that a single scalar learning-rate matching
`eta_g = eta_1 / a_g^star` is numerically applicable in the SGD sense?

## Protocol

1. **Input domain**: images are normalized exactly like
   `training/ct_training_loop.py:870`:
   `images = images.to(device).to(torch.float32) / 127.5 - 1`, giving range
   **[-1,1]**.
2. **Network mode**: the EMA is switched to **`net.train()`** with dropout
   active, matching `ct_training_loop.py:577`
   (`net.train().requires_grad_(True)`). Dropout RNG state is captured once per
   minibatch and reused across gaps.
3. **Schedule**: `global_sigmoid` uses q/k/b/stage read from the checkpoint's
   `loss_fn`, not hardcoded values. The checkpoint schedule is verified to be
   the official `sigmoid` baseline, and g=1 reproduces it exactly.
4. **Augmentation**: the checkpoint used here has `augment_pipe=None`.
   Augmentation-enabled checkpoints are rejected because paired augmentation
   is not implemented by this probe.
5. **Precision**: the probe uses FP32 (`force_fp32=True`) as a numerically stable
   unscaled-gradient reference. Equivalence to the actual AMP training path is
   not asserted and remains an open check.
6. **No state mutation**: no optimizer is created or stepped. Parameter and
   buffer SHA256 values are identical before and after the diagnostic
   (`state_preserved: true`).
7. **Determinism**: the DataLoader is created with a seeded generator before
   iteration; t, epsilon, and dropout randomness are paired across all gaps.

## Metrics

Per gap, at model and module-path level:

- `mu_g = B^-1 sum_b grad_b(g)`
- `a_g^star = <mu_g, mu_1> / ||mu_1||^2`
- `R_mean(g) = ||mu_g - a_g^star mu_1|| / ||mu_g||`
- cosine similarity to the g=1 mean gradient

For SGD, matching the mean parameter update requires

`eta_g * mu_g = eta_1 * mu_1`, hence
`eta_g = eta_1 / a_g^star`.

This relation is not automatically valid for RAdam because the real update also
depends on optimizer moment state and rectification.

## Results

Checkpoint: q128, 256 kimg, seed 3, `g_screen/g1_0`.
Execution: 16 minibatches, batch size 64, seed 20260806.

| g | a* | cosine | directional residual |
|---:|---:|---:|---:|
| 0.9 | 1.1109 | 0.999999 | 0.0012 |
| 1.0 | 1.0000 | 1.000000 | 0 |
| 1.2 | 0.8339 | 0.999998 | 0.0022 |
| 1.3 | 0.7700 | 0.999996 | 0.0030 |

The whole-model FP32 mean gradient is therefore predominantly an approximately
`1/g` scalar rescaling over this gap range.

### Layerwise residuals

The following values were recomputed from the committed CSV over 208 module
paths:

| g | mean | median | max | max layer | >2% | >3% |
|---:|---:|---:|---:|---|---:|---:|
| 0.9 | 1.16% | 1.21% | 2.31% | dec.8x8_block3.conv0 | 26 | 0 |
| 1.2 | 1.36% | 1.51% | 2.56% | dec.16x16_up.norm0 | 28 | 0 |
| 1.3 | 1.73% | 1.85% | 3.76% | dec.16x16_block1.norm0 | 83 | 6 |

Thus the whole-model mean gradient is near-scalar, while individual layers
retain directional deviations up to approximately 3.8% at g=1.3.

## Comparison with Role D

| source | g=0.9 residual | g=1.2 | g=1.3 | per-layer max |
|---|---:|---:|---:|---:|
| Role D, about 1000 kimg | 0.48% | 0.88% | 1.35% | about 12.4% |
| Role C rev.3, 256 kimg | 0.12% | 0.22% | 0.30% | about 3.8% |

The difference is descriptive only. The runs differ in checkpoint, batch count,
batch size, data permutation, and implementation details, so it cannot be
attributed to training stage without a controlled checkpoint sweep.

## Scope and open items

- Single seed and one checkpoint.
- FP32 reference only; AMP-path gradient equivalence is unverified.
- RAdam-update equivalence is not licensed by raw-gradient collinearity.
- The largest observed layerwise residual is approximately 3.8%, so scalar
  matching is an accurate whole-model description but not an exact layerwise
  identity.
- The diagnostic does not establish that the residual predicts FID or causes
  the observed global-gap performance difference.

## Files

- `analysis/gap_gradient_hook.py`
- `analysis/gap_gradient_model_moments.csv`
- `analysis/gap_gradient_layerwise.csv`
- `analysis/gap_gradient_manifest.json`
- `tests/test_gap_gradient_hook.py`
