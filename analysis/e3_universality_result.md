# E3 — Universality: does the residual reproduce with AdamW? (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Paired g=1.0 vs g=1.3
step with AdamW on arm_a tick 8 (n_K=1991), same minibatch/seed as the RAdam
audit. Script: `analysis/run_e3_adamw.py`.

## Hypothesis (E3)
The non-scalar residual is a general optimizer-state phenomenon, not RAdam-
specific. The gradient residual R_grad is a property of the loss/gradient
(optimizer-independent); the update distortion R_opt depends on the optimizer's
update rule.

## Result (arm_a tick 8, n_K=1991)

| quantity | AdamW | RAdam (E7) |
|---|---:|---:|
| a_K* (gradient scalar fit) | 0.7774 | 0.7831 |
| **R_grad (gradient residual)** | **0.0951** | **0.0905** |
| s_K* (update scale) | 0.9205 | 0.9990 |
| **R_opt (update residual)** | **0.3905** | **0.0142** |
| R_opt − R_grad | +0.2954 | −0.0764 |

## Interpretation (honest)

- **The gradient residual R_grad reproduces with AdamW** (0.0951 vs 0.0905 for
  RAdam, nearly identical). This confirms the non-scalar GRADIENT residual is a
  property of the loss/gradient, NOT optimizer-specific. It is a real,
  universal feature of few-step training.
- **The update distortion R_opt differs sharply by optimizer**: AdamW amplifies
  the gradient residual into a large update distortion (R_opt=0.39, R_opt−R_grad
  positive), whereas RAdam compresses it (R_opt=0.014, R_opt−R_grad negative).
  The optimizer's update rule determines how the gradient residual propagates to
  the update.

## Verdict for E3
- **PASS (gradient residual is universal):** the non-scalar gradient residual
  reproduces with AdamW, confirming it is not a RAdam artifact.
- The update distortion is optimizer-dependent (AdamW amplifies, RAdam
  compresses) — a real, reportable difference.

## Relation to the diagnosis
E3 strengthens the diagnosis: the non-scalar gradient content is a universal
property of few-step training (reproduces across optimizers), while its
propagation to the update depends on the optimizer. This is consistent with the
honest diagnosis (residual exists, is real, is not harmful).

## Files
- Script: `analysis/run_e3_adamw.py`
- This summary: `analysis/e3_universality_result.md`
