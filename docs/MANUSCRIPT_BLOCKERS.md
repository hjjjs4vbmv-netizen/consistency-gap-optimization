# Manuscript Blockers

Status on 2026-08-23: **no hard blocker prevents the first submission-ready draft or its local compilation.**

## Verified omissions

| Item | Status | Decision |
|---|---|---|
| End-to-end wall-clock training runtime | The compact formal record contains per-job timing and queue/recovery events, but not one clean, comparable total training runtime for every arm after the audited interruption/recovery sequence. | Omitted rather than reconstructed from incomplete timing evidence. Hardware and software environment remain reported. |
| Predeclared FID $\leq 10$ threshold | No timestamped artifact was found that froze this threshold before the relevant curves were visible. | Labeled descriptive/post-hoc; observed checkpoints only; no interpolation or significance claim. |
| Cross-$q$ or effective-spacing transfer | No completed matched control exists. | Stated as the highest-priority validation, not as a result. |
| Second dataset or weighting rule | No completed audited result exists. | Listed as future validation only. |
| Optimizer-to-FID causal closure | The held-out moment-transport manipulation failed its gate and launched no continuation/FID evaluation. | Preserved as a scientific NO-GO; RAdam remains a supporting boundary. |

These omissions constrain the claim ceiling but do not create missing values in any reported formula, table, figure, or headline count.
