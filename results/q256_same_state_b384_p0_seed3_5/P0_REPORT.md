# q256 B@384 same-state A/B/C/D P0 report

1. B no-op parity: **3/3 PASS**.
2. Formal branches: **12/12 PASS**.
3. Exact closure: **all PASS**.
4. Raw late-horizon propagation is history-dominated/persistent for theta/EMA/m/v in B/C/D across 3/3 seeds. This raw label includes declared mechanical carryover.
5. Numerically nonzero corrected incremental feedback replicates under the separately stated presence rule for: `['B:state:EMA', 'B:state:m', 'B:state:theta', 'B:state:v', 'C:state:EMA', 'C:state:m', 'C:state:theta', 'C:state:v', 'D:state:EMA', 'D:state:m', 'D:state:theta', 'D:state:v']`. This is not a dominance or amplification result.
6. Audited observables replicate descriptively in: `['B:observable:feature', 'B:observable:residual', 'C:observable:feature', 'C:observable:residual', 'D:observable:feature', 'D:observable:residual']`. Feature/residual readouts have no declared linear carryover map and are not carryover-corrected state-mechanism evidence.
7. The paper may claim conditional history-dominated/persistent propagation from B@384 history; no quality, global causal, or actionable-law claim is licensed.
8. P1 is **worth protocol consideration**; P1 was not started.

Actual training compute: `3.952445` A100 GPU-hours.

## Raw propagation classification

The legacy PR #89 label is relabeled for interpretation because raw `R` includes mechanical carryover.

| arm:space:block | counts | replication | interpretive label |
|---|---|---|---|
| B:state:EMA | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| B:state:m | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| B:state:theta | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| B:state:v | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| C:state:EMA | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| C:state:m | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| C:state:theta | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| C:state:v | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| D:state:EMA | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| D:state:m | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| D:state:theta | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |
| D:state:v | {'history_dominated_persistent_propagation': 3} | cross-seed replicated | history_dominated_persistent_propagation |

## Carryover-corrected incremental feedback

At both late horizons {256,500}: closure passes, corrected_R_norm is finite and >0, and corrected_R_over_delta_k is finite and >0.
Numerically nonzero corrected incremental feedback is replicated when the seed-level rule holds in at least 2/3 formal seeds.
A conservative directionally consistent result requires all 3/3 seeds to have finite nonzero corrected-feedback alignment with the same sign at h=256 and h=500 within seed, and the same sign across seeds. This rule is post-hoc and descriptive.
This post-hoc presence rule does not establish corrected-feedback dominance, same-direction amplification, or a causal contribution.

Across all late persistent-state rows, raw `feedback_gain_G` ranges from `0.881234` to `1.00707` and `corrected_R_over_delta_k` ranges from `0.000129996` to `0.0967489`; no corrected ratio exceeds 1.
Thus neither raw nor corrected amplification is universal.

Directionally consistent 3/3 entries: `['B:state:EMA', 'B:state:theta', 'B:state:v', 'C:state:EMA', 'C:state:theta', 'D:state:EMA', 'D:state:theta']`.

| arm:state:block | nonzero seeds | replicated presence | directional 3/3 | alignment sign |
|---|---:|---|---|---|
| B:state:theta | 3/3 | True | True | positive |
| B:state:EMA | 3/3 | True | True | positive |
| B:state:m | 3/3 | True | False | mixed/reversing |
| B:state:v | 3/3 | True | True | positive |
| C:state:theta | 3/3 | True | True | positive |
| C:state:EMA | 3/3 | True | True | positive |
| C:state:m | 3/3 | True | False | mixed/reversing |
| C:state:v | 3/3 | True | False | mixed/reversing |
| D:state:theta | 3/3 | True | True | positive |
| D:state:EMA | 3/3 | True | True | positive |
| D:state:m | 3/3 | True | False | mixed/reversing |
| D:state:v | 3/3 | True | False | mixed/reversing |

Per-seed late medians for raw `feedback_gain_G`, `corrected_R_over_delta_k`, and corrected-feedback alignment are in `late_propagation_corrected_feedback.csv`.

## Observable scope

Fixed-latent EMA feature and signed residual readouts show replicated descriptive history-dominated propagation in B/C/D across 3/3 seeds. They have no declared linear carryover map, so no carryover-corrected observable mechanism is claimed.

## Exploratory absolute-norm contrasts

`exploratory_absolute_norm_contrasts.csv` contains algebraic contrasts of branch-specific absolute L2 norms. For example, `norm_C_minus_norm_A` is `||z_C||_2 - ||z_A||_2`, not `||z_C-z_A||_2`. These rows are conditional on B@384 history, are not used for the mechanism headline, and are not target effects, denominator effects, factorial causal effects, or independent-training arm rankings.
