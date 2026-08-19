# Independent audit of the held-out NO-GO

Two independent read-only reviews recomputed the result from raw receipts and
batch rows. Both concluded that `heldout-attempt-002` is an internally
consistent scientific NO-GO, not an engineering false negative.

## Split and construction

- Calibration IDs were exactly `2026081101/03/05/07`; held-out IDs were
  exactly `2026081102/04/06/08` (canonical ranks 1/3/5/7).
- Every seed coefficient was the unweighted median of its four calibration
  values, with no held-out value entering calibration.
- The four virtual arms were reconstructed as:
  F = original moments plus `G_1.00`; G = original moments plus `G_1.10`;
  T = `(a*m, a^2*v)` plus `G_1.10`; and T-exact = `(a*m, a^2*v)` plus
  `a*G_1.00`.
- Every virtual optimizer step used an independent deep clone. Primary and
  reverse branch orders produced identical update hashes.

The transform covered all 416 optimizer parameter states. Observed median
moment norm ratios were consistent with the frozen coefficients:

| Seed | `exp_avg` ratio | `exp_avg_sq` ratio |
| ---: | ---: | ---: |
| 3 | 0.8370121118 | 0.7005892943 |
| 4 | 0.8073491451 | 0.6518126767 |
| 5 | 0.8233457212 | 0.6778981710 |

Non-moment optimizer state hashes were unchanged. These checkpoints did not
contain `max_exp_avg_sq`.

## Independent numerical reconstruction

The reviews found 12 unique `(training_seed, audit_batch_id)` rows, four per
seed, with no skip and all finite values. They independently recomputed the
per-seed medians and suppression values shown in `outcome.md` and obtained an
exact match. They also reconstructed

`s = q*cos(theta)`, `c = cos(theta)/q`, and
`R_opt = q*sqrt(1-cos(theta)^2)`

from the raw norm/cosine quantities. Maximum discrepancies from the archived
values were `2.52e-14` for G, `1.97e-14` for T, and `2.89e-11` for T-exact.
Reaggregation from all 208 layer summaries per arm and row agreed to about
`1.1e-12`.

The strongest positive control was T-exact: its median residual was
`2.80e-05`, `3.99e-05`, and `5.65e-05` for seeds 3, 4, and 5, while
`||U_T_exact||/||U_F||` was approximately `0.999999`. Ordinary-G gradients,
updates, residuals, and hashes also matched their original factorial receipts
exactly. This rules out a reversed moment scale, F/T label swap, wrong branch
order, or changed G baseline as an explanation.

All 12 held-out batches had `R_opt_T > R_opt_G`; the individual worsening was
approximately 1.9% to 29.8%. Thus the scalar-covariance construction and its
positive control worked as implemented, but the frozen whole-model transport
did not generalize to the real `G_1.10` gradients.

## Provenance and stop decision

The execution HEAD and code hashes matched the final receipt, all six source
state/snapshot files and the audit dataset retained their expected SHA256,
and the server worktree remained clean. No smoke/formal directory, training
artifact, continuation process, or related tmux session was present.

The preregistered decision is therefore final: do not start smoke, formal
continuation, sampling, FID, or KID for this intervention.
