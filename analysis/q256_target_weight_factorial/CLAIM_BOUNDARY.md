# Claim boundary: q256 target geometry × denominator weighting

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-19
- Verification Status: VERIFIED
- Version Label: q256_target_weight_factorial_claim_boundary_v2

## Supported by the completed formal matrix and evaluation

The 12/12 fresh training matrix and 24/24 frozen evaluation jobs are complete.
The experiment may describe the observed seed-level and cross-seed patterns of
the four frozen 256-kimg endpoints, their preregistered contrasts, and the
factorial interaction. It may say that the NFE1 pattern is consistent with the
preregistered target-geometry-dominant branch, while retaining the `n=3`
uncertainty, the mixed denominator contrasts, the unstable interaction sign,
and the strong NFE2 heterogeneity.

## Unsupported

The experiment does not support any of the following claims:

- optimizer history causes or explains a FID/KID improvement;
- moment state mediates the gap-quality relationship;
- a contrast is a percentage of a total effect explained;
- minibatches, checkpoints, sample blocks, or metric repeats raise the
  independent sample size above three;
- an exploratory checkpoint, NFE, seed block, or metric can replace the frozen
  primary endpoint;
- failure of fresh B to beat A can be hidden by emphasizing C, D, a secondary
  metric, or an earlier checkpoint.

## Reporting labels

Every final statement must be marked as one of: observed result,
preregistered interpretation, exploratory observation, or unsupported claim.
