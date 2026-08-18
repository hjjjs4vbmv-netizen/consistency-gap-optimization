# Cross-seed replication of gap-induced optimizer divergence

The table contains one same-state Layer A receipt and one 20-step Layer B receipt per training seed. A row is a training-trajectory observation, not a repeated-minibatch estimate.

| seed | row | K | schedule q | a* | R_grad | R_opt | h_i disp. (on support) | scalar-history R² | Corr | wRMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | existing_artifact | 256 | 128 | 0.748928 | 0.115157 | 0.098956 | 0.098922 | 0.735011 | 0.858201 | 0.029542 |
| 4 | new_independent_training_trajectory | 256 | 256 | 0.614878 | 0.275529 | 0.115198 | 0.115163 | 0.638644 | 0.809086 | 0.041223 |
| 5 | new_independent_training_trajectory | 256 | 256 | 0.840351 | 0.148584 | 0.092422 | 0.092392 | 0.081677 | 0.419873 | 0.071467 |

`h_i` dispersion is the square root of the exact on-support dispersion energy. `H_K` is retained in the CSV as an algebraic identity check (`H_K = R_opt` after off-support energy is included), not as a second mechanism measurement.

## Accounting and claim boundary

- K=256 for every row: `True`.
- Distinct named training trajectories: `True`.
- Layer A uses shared minibatch/t/noise/dropout **within** each seed; this pairing does not substitute for training-seed replication.
- schedule q by seed: `{'3': 128, '4': 256, '5': 256}`; all schedules equal: `False`.

The historical seed-3 anchor is explicitly retained with its recorded schedule q. If the schedule-q accounting is false, do not pool all three rows as a pure same-configuration seed effect; report the q256 seed4/5 trajectories as independent replications and the seed-3 row as a hash-bound mechanism anchor. In every case, the scalar-history values quantify prospective update-ratio explanatory power, not endpoint-quality causality.
