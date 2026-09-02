# q256 g=1.00/1.10 RAdam moment-reset manipulation check

**Pre-registered verdict: NO-GO.**

## Seed-level results

| Training seed | Audit batches | Median suppression | Min | Max |
|---:|---:|---:|---:|---:|
| 3 | 8 | -4.624537 | -5.385773 | -4.070444 |
| 4 | 8 | -4.586929 | -5.395765 | -3.365010 |
| 5 | 8 | -5.110132 | -5.616396 | -4.400603 |

Suppression is defined as `1 - R_opt_reset / R_opt_real`.

## Interpretation boundary

This audit can determine whether clearing accumulated RAdam moments lowers optimizer-update divergence for the formal g=1.10 treatment at the frozen q256 source states. It does not establish that optimizer memory caused an FID improvement. Audit minibatches are not independent training replicates, and this is not a full-training intervention.

No training, sample generation, FID, or KID computation was performed.
