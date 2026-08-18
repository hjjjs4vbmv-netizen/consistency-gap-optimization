# q256 g=1.00/1.10 RAdam moment-reset manipulation check

**Pre-registered verdict: NO-GO.**

## Seed-level results

| Training seed | Audit batches | Median suppression | Min | Max |
|---:|---:|---:|---:|---:|
| 3 | 8 | -4.624537 | -5.385773 | -4.070444 |
| 4 | 8 | -4.586929 | -5.395765 | -3.365010 |
| 5 | 8 | -5.110132 | -5.616396 | -4.400603 |

Suppression is defined as `1 - R_opt_reset / R_opt_real`.

## Cross-seed R_grad vs R_opt_real

| Training seed | Median R_grad | Median R_opt_real | Paired median (R_opt_real - R_grad) | R_opt_real < R_grad |
|---:|---:|---:|---:|---:|
| 3 | 0.164929 | 0.083028 | -0.086425 | 8/8 |
| 4 | 0.230686 | 0.087762 | -0.145301 | 8/8 |
| 5 | 0.103997 | 0.074052 | -0.036439 | 8/8 |

Across the frozen real optimizer states, `R_opt_real < R_grad` in 24/24 paired audit minibatches. Thus the stateful optimizer history consistently attenuated, rather than amplified, the instantaneous gradient directional residual in this audit.

The result falsifies moment zeroing as a valid memory-neutralization intervention; it does not falsify state-dependent optimizer-history effects.

## Interpretation boundary

This audit can determine whether clearing accumulated RAdam moments lowers optimizer-update divergence for the formal g=1.10 treatment at the frozen q256 source states. It does not establish that optimizer memory caused an FID improvement. Audit minibatches are not independent training replicates, and this is not a full-training intervention.

No training, sample generation, FID, or KID computation was performed.
