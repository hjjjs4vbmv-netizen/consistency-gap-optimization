# Forcing–feedback decomposition v2

Status: **PASS**

## State recoverability gate

The PR #87 matched-rollout receipt is **not** recoverable at every k=0,…,64. It stores projections, summaries and hashes, not full augmented states.
The decomposition therefore comes from a matched 64-step replay of the same hash-pinned frozen source and is instrumentation, not a new scientific experiment.

## Exact identity

For each X∈{B,C,D} and k, the run evaluates `Phi_A(zA)`, `Phi_X(zA)` and `Phi_X(zX)` using the same frozen batch/noise/dropout receipt and pairing seed.
All block/observable closures pass: **True**; maximum relative closure error: `0`.

The CSV also records the pre-transition separation `Delta_k`, measured propagation gain `G_k=||R_k||/max(||Delta_k||,epsilon)`, and `cos(R_k,Delta_k)` for every state block and observable.

## Propagation and incremental-feedback diagnostics

| arm | space | block | classification | late G | late cos(R,Delta_k) | median corrected R/Delta_k | median corrected R/b | median cos(corrected R,Delta_k) |
|---|---|---|---|---:|---:|---:|---:|---:|
| B | observable | feature | mixed_or_inconclusive | 0.999 | 0.01584 | NA | NA | NA |
| B | observable | residual | mixed_or_inconclusive | 0.9973 | 0.1997 | NA | NA | NA |
| B | state | EMA | persistent_state_feedback_dominance | 1.033 | 0.9993 | 0.09827 | 10.02 | 0.5447 |
| B | state | m | persistent_state_feedback_dominance | 0.9835 | 0.9408 | 0.3217 | 1.123 | -0.003705 |
| B | state | net_buffer | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| B | state | scaler | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| B | state | theta | persistent_state_feedback_dominance | 1.014 | 0.9995 | 0.0462 | 3.857 | 0.5973 |
| B | state | v | persistent_state_feedback_dominance | 1.001 | 0.9999 | 0.02448 | 0.7208 | -0.1176 |
| C | observable | feature | mixed_or_inconclusive | 0.9991 | -0.004699 | NA | NA | NA |
| C | observable | residual | mixed_or_inconclusive | 1.013 | 0.07615 | NA | NA | NA |
| C | state | EMA | persistent_state_feedback_dominance | 1.033 | 0.999 | 0.1084 | 8.3 | 0.4522 |
| C | state | m | persistent_state_feedback_dominance | 0.9979 | 0.8886 | 0.4583 | 1.516 | -0.02602 |
| C | state | net_buffer | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| C | state | scaler | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| C | state | theta | persistent_state_feedback_dominance | 1.017 | 0.9994 | 0.05715 | 3.398 | 0.4821 |
| C | state | v | persistent_state_feedback_dominance | 0.9781 | 0.9971 | 0.1039 | 1.418 | -0.1006 |
| D | observable | feature | mixed_or_inconclusive | 0.991 | 0.02579 | NA | NA | NA |
| D | observable | residual | mixed_or_inconclusive | 1.008 | 0.2439 | NA | NA | NA |
| D | state | EMA | persistent_state_feedback_dominance | 1.034 | 0.9993 | 0.102 | 12.56 | 0.539 |
| D | state | m | persistent_state_feedback_dominance | 0.9492 | 0.9554 | 0.3458 | 1.289 | -0.03609 |
| D | state | net_buffer | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| D | state | scaler | mixed_or_inconclusive | 0 | NA | NA | NA | NA |
| D | state | theta | persistent_state_feedback_dominance | 1.017 | 0.9995 | 0.04848 | 6.751 | 0.6022 |
| D | state | v | persistent_state_feedback_dominance | 0.9988 | 0.9999 | 0.02135 | 0.7592 | -0.1868 |

`G≈1` with alignment near one indicates persistence; `G<1` indicates contractive propagation. `G>1` with high alignment is only possible same-direction expansion. Low alignment indicates rotation or more complex deformation.

For `theta`, corrected feedback subtracts `Delta_k`. For RAdam `m` and `v`, it subtracts the actual per-parameter-group `beta1*Delta_k` and `beta2*Delta_k`. EMA uses the implemented transition map: parameters retain `AlgorithmicState.ema_beta`, while EMA buffers are unchanged and therefore retain their full incoming separation. No arbitrary EMA beta is introduced. The CSV additionally records corrected alignment against the incoming separation, current forcing, and raw propagation term.

No mechanism winner is selected. Only propagation/persistence wording is used: this audit has no second independent state replication, so the stronger causal claim gate is not satisfied.

## Interpretation boundary

`R/b` is reported only as a scale diagnostic. It is not a contribution percentage: large forcing and feedback terms can be nearly antiparallel and cancel.
For persistent state blocks, the dominance label means that accumulated state history is larger than the current common-state forcing; it is a propagation label.
Residual closure uses a common signed arm-A validation residual map for all three post-transition states; feature closure uses the same fixed-latent EMA map.
