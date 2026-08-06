# Deep-network gap gradient moments (supplementary)

## Status and scope

This is an **exploratory supplementary diagnostic**, not a formal evaluation.
It uses one q=128 fixed-sigmoid checkpoint trained from the official EDM
transfer model (seed 3, 1000.064 kimg), rather than a checkpoint in the
frozen q=128 256-kimg matrix.  It does not establish a 1024-kimg formal
result, a prospective result, cross-q generalization, or an explanation of
the q=256 FID values.

The portable evidence binding is
`analysis/deepnet_gap_gradient_moments_provenance.json`.  It records the
checkpoint, training-state, data, sender-receipt, checkpoint-archive,
evaluation-commit, runner, and result-archive hashes.  The post-hoc
machine-readable training-integrity receipt is present and passed at
`analysis/deepnet_training_integrity_receipt.posthoc.json` (SHA256
`e02de8e126ac28a1be6650bf8a3c25c8c0a32de0533c66b3004e10a946a63e8c`).
It verifies clean completion, loss/state finiteness, EMA, and schedule
identity, but cannot attest an expected training commit because the completed
run did not persist `commit_sha.txt`.  Independent Role D receiver
verification passed for this one exploratory checkpoint at
`analysis/deepnet_role_d_receiver_verification.json` (SHA256
`e092db3727de6915ed0e3b6113da9744d2ea9dda5e60d7fe84697756fd2ad8c5`):
ECT002 recomputed both the received archive digest and checkpoint digest.
This closes the receiver-identity gap for this supplementary diagnostic only;
it does not make the checkpoint prospective or formal, and does not verify
the six historical `q128-1024k-*` checkpoints.

The sender handoff declaration is
`analysis/q128_transfer_1000k_sender_handoff.json`; the received Role D
receipt above was generated with `scripts/verify_checkpoint_handoff.py`.

## v2 validation extension

The v2 script adds a unit test that compares the manually controlled loss and
gradient with the repository's `ECMLoss` under identical randomness, and it
records parameter/buffer SHA256 values before and after the diagnostic.  The
64-batch v2 rerun recorded identical parameter hashes
`d1b0823277dc592c14b4167a31136cbabc60739e6967bac7b35c1dfe4f9b7270` and
identical buffer hashes
`30b9349e40d1d53010d9764911771919a12bf32fa719ba30063956a111ce0cee`;
all 416 parameter gradient slots were empty before the diagnostic and present
after it.  No optimizer was constructed or stepped.

## Question and controlled protocol

The question is local: at a fixed real ECT checkpoint, does changing the
global gap multiplier mainly rescale the training gradient or materially
rotate it?  The sweep used `g in {0.9, 1.0, 1.2, 1.3}` and 64 deterministic
minibatches of 128 CIFAR-10 images.  For every minibatch and every g, it held
fixed:

- image minibatch;
- per-example timestep vector sampled from the actual training distribution;
- shared noise tensor;
- dropout RNG state; and
- EMA parameter values.

The runner set the inference-exported EMA tensors to require gradients, but
did not create an optimizer or execute an optimizer step.  The manifest
records `optimizer_created=false` and `optimizer_steps=0`.

For minibatch gradient `nabla_b(g)`, it calculates

`mu_g = B^-1 sum_b nabla_b(g)`,

`a*_g = <mu_g, mu_1> / ||mu_1||^2`, and

`R_mean(g) = ||mu_g - a*_g mu_1|| / ||mu_g||`.

The normalized noise scale is
`B^-1 sum_b ||nabla_b(g)-mu_g||^2 / ||mu_g||^2`.  The raw 256 batch-by-gap
records and 832 layer-by-gap records are retained alongside the aggregate
table.

## Whole-model result

| g | `||mu_g||` | `a*_g` | cosine to `mu_1` | `R_mean(g)` | variance trace | normalized noise scale |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 591.004 | 1.10915 | 0.999988 | 0.00481 | 14286.39 | 0.04090 |
| 1.0 | 532.838 | 1.00000 | 1.000000 | 0 | 11594.19 | 0.04084 |
| 1.2 | 439.745 | 0.82526 | 0.999962 | 0.00876 | 8040.80 | 0.04158 |
| 1.3 | 402.538 | 0.75539 | 0.999909 | 0.01349 | 6667.41 | 0.04115 |

At whole-model scale, the sweep is close to a scalar gradient rescaling:
all mean-gradient cosines exceed 0.9999 and the largest directional residual
is 1.35% at g=1.3.  Smaller g increases gradient magnitude, while larger g
reduces it.  Although the unnormalized variance trace drops with gradient
magnitude, the normalized noise scale remains near 0.041 across the sweep.

The batch-level directional residual is stable in magnitude, but not zero:

| g | mean | standard deviation | minimum | maximum |
|---:|---:|---:|---:|---:|
| 0.9 | 0.01503 | 0.00557 | 0.00795 | 0.03782 |
| 1.2 | 0.01961 | 0.00557 | 0.01171 | 0.03546 |
| 1.3 | 0.02435 | 0.00578 | 0.01549 | 0.04362 |

## Layerwise qualification

The whole-model scalar relation should not be mistaken for exact
layer-by-layer collinearity.  Across 208 module-level parameter groups, the
largest layerwise residuals were 7.37% at g=0.9, 8.89% at g=1.2, and 12.41%
at g=1.3.  The largest deviations at g=1.3 occurred in late 32x32 decoder
layers (notably `model.dec.32x32_block3.norm1`, 12.41%).  Thus the aggregate
gradient is almost collinear while some individual layers exhibit modest but
non-negligible directional changes.

The directional-residual plot is
`figures/deepnet_scalar_residual_vs_g.pdf`.  It has been rendered and checked
for a labeled axis, visible g=1 baseline, and legible layout.

## Validation and caveats

The v2 result archive SHA256 is
`18470fe78e3b99b3daace10b0b5721e97d057afeedb89d59c4270bda9eaed10d`.
All six extracted artifact SHA256 values match the sender record.  The
validation pass found four whole-model rows, exactly 256 unique
`minibatch x gap` records, 832 layer-by-gap rows, no missing sweep cell, and
only finite numeric values.  Batch residual means and standard deviations
were independently recomputed from the raw batch CSV.

This is adequate to share as a controlled, within-checkpoint mechanism
observation.  It is not adequate to claim an optimization or sample-quality
benefit: it has one checkpoint, one seed, no global110 training counterpart,
no formal metric endpoint, and no q=128/q=256 dataset-semantic equivalence
result.  Independent receiver verification passed for this exploratory
checkpoint only; it does not verify the historical six `q128-1024k-*`
checkpoints or promote this result to formal evidence.
