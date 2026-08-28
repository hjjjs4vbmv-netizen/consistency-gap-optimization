# Exact nonlinear forcing–feedback decomposition

Status: **PASS**

## State recoverability gate

The PR #87 matched-rollout receipt is **not** recoverable at every k=0,…,64. It stores projections, summaries and hashes, not full augmented states.
The decomposition therefore comes from a matched 64-step replay of the same hash-pinned frozen source and is instrumentation, not a new scientific experiment.

## Exact identity

For each X∈{B,C,D} and k, the run evaluates `Phi_A(zA)`, `Phi_X(zA)` and `Phi_X(zX)` using the same frozen batch/noise/dropout receipt and pairing seed.
All block/observable closures pass: **True**; maximum relative closure error: `0`.

## Mechanism diagnostics

| arm | space | block | classification | early median R/b | late median R/b | late cos(R,Δ) |
|---|---|---|---|---:|---:|---:|
| B | observable | feature | mixed_or_inconclusive | 1.026 | 1.065 | 0.5487 |
| B | observable | residual | mixed_or_inconclusive | 1.121 | 1.306 | 0.7124 |
| B | state | EMA | trajectory_feedback_amplification | 9.575 | 273.1 | 1 |
| B | state | m | trajectory_feedback_amplification | 2.424 | 3.728 | 0.9752 |
| B | state | net_buffer | mixed_or_inconclusive | NA | NA | NA |
| B | state | scaler | mixed_or_inconclusive | NA | NA | NA |
| B | state | theta | trajectory_feedback_amplification | 17.04 | 138.6 | 1 |
| B | state | v | trajectory_feedback_amplification | 6.34 | 61.37 | 1 |
| C | observable | feature | mixed_or_inconclusive | 1.045 | 1.081 | 0.5676 |
| C | observable | residual | mixed_or_inconclusive | 1.05 | 1.224 | 0.6586 |
| C | state | EMA | trajectory_feedback_amplification | 9.5 | 199.3 | 1 |
| C | state | m | trajectory_feedback_amplification | 2.812 | 3.727 | 0.9648 |
| C | state | net_buffer | mixed_or_inconclusive | NA | NA | NA |
| C | state | scaler | mixed_or_inconclusive | NA | NA | NA |
| C | state | theta | trajectory_feedback_amplification | 15.97 | 99.85 | 1 |
| C | state | v | trajectory_feedback_amplification | 4.66 | 20.45 | 0.9991 |
| D | observable | feature | mixed_or_inconclusive | 1.034 | 1.083 | 0.5779 |
| D | observable | residual | mixed_or_inconclusive | 1.065 | 1.325 | 0.7129 |
| D | state | EMA | trajectory_feedback_amplification | 11.51 | 359.3 | 1 |
| D | state | m | trajectory_feedback_amplification | 2.15 | 5.704 | 0.9862 |
| D | state | net_buffer | mixed_or_inconclusive | NA | NA | NA |
| D | state | scaler | mixed_or_inconclusive | NA | NA | NA |
| D | state | theta | trajectory_feedback_amplification | 25.87 | 260.2 | 1 |
| D | state | v | trajectory_feedback_amplification | 4.223 | 76.3 | 1 |

## Interpretation boundary

`R/b` is reported only as a scale diagnostic. It is not a contribution percentage: large forcing and feedback terms can be nearly antiparallel and cancel.
Residual closure uses a common signed arm-A validation residual map for all three post-transition states; feature closure uses the same fixed-latent EMA map.
