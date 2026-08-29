# q256 B@384 same-state A/B/C/D P0 report

1. B no-op parity: **3/3 PASS**.
2. Formal branches: **12/12 PASS**.
3. Exact closure: **all PASS**.
4. Late-horizon theta/EMA/m/v classifications are tabulated below.
5. Cross-seed replicated core entries: `['B:state:EMA', 'B:state:m', 'B:state:theta', 'B:state:v', 'C:state:EMA', 'C:state:m', 'C:state:theta', 'C:state:v', 'D:state:EMA', 'D:state:m', 'D:state:theta', 'D:state:v']`.
6. Residual/features remain mixed: **False**. Both observables are classified as `persistent_state_feedback_dominance` for B/C/D in all 3/3 seeds.
7. The paper may claim conditional same-state persistence/feedback replication only where the 2/3 rule passes; no quality or global causal claim is licensed.
8. P1 is **worth protocol consideration**; P1 was not started.

Actual training compute: `3.952445` A100 GPU-hours.

| arm:space:block | counts | replication | label |
|---|---|---|---|
| B:state:EMA | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| B:state:m | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| B:state:theta | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| B:state:v | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| C:state:EMA | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| C:state:m | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| C:state:theta | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| C:state:v | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| D:state:EMA | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| D:state:m | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| D:state:theta | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |
| D:state:v | {'persistent_state_feedback_dominance': 3} | cross-seed replicated | persistent_state_feedback_dominance |

For the fixed-latent EMA feature map and common signed validation residual map,
all B/C/D comparisons are `persistent_state_feedback_dominance` in seeds 3, 4,
and 5. Network buffers and GradScaler are `mixed_or_inconclusive` in all seeds;
they do not support a replicated mechanism label.

These classifications describe conditional propagation from a B@384 history.
They do not show that feedback improves FID, that RAdam is unique, or that any
norm ratio is a causal contribution percentage.

All factorial contrasts are conditional on a B@384 history and are not independent-training arm rankings.
