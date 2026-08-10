# E7 — Optimizer-state dependence of the residual (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Direct method: calls
`run_stateful_pair` on each arm_a training-state tick (different n_K / optimizer
maturity), bypassing the audit's sigmoid gate (uses the arm's own global_sigmoid
checkpoint for loss_fn). Script: `analysis/run_e7_direct.py`.

## Hypothesis (E7)
The non-scalar residual and the update distortion depend on the optimizer state
(moments m, v magnitude / step count n_K), not just the gradient.

## Results (arm_a, g=1.0 reference, paired g=1.3 step, seed 20260808)

| tick | n_K | a_K* | R_grad | R_opt | R_opt−R_grad |
|---|---:|---:|---:|---:|---:|
| 1 | 243 | 0.7754 | 0.0306 | 0.0120 | −0.0185 |
| 2 | 493 | 0.7915 | 0.0466 | 0.0148 | −0.0318 |
| 3 | 743 | 0.7814 | 0.0580 | 0.0157 | −0.0423 |
| 4 | 993 | 0.7642 | 0.0837 | 0.0174 | −0.0663 |
| 5 | 1243 | 0.7663 | 0.0839 | 0.0169 | −0.0670 |
| 6 | 1492 | 0.7601 | 0.0614 | 0.0170 | −0.0444 |
| 7 | 1742 | 0.7763 | 0.0765 | 0.0160 | −0.0604 |
| 8 | 1991 | 0.7831 | 0.0905 | 0.0142 | −0.0764 |

## Interpretation (honest)

- **The gradient residual R_grad grows with optimizer maturity (n_K):**
  0.031 (n_K=243) → 0.091 (n_K=1991), roughly monotone (dip at tick 6). The
  non-scalar residual is **optimizer-state-dependent**, not a fixed property of
  the network. This confirms the "state-conditioned" characterization.
- **The update distortion R_opt is small and roughly flat** (0.012–0.017), and
  R_opt−R_grad is NEGATIVE (the optimizer compresses the gradient residual, not
  amplifies it).
- **Measurement-sensitivity caveat:** the earlier stateful audit
  (`radam_update_audit_stateful.json`) reported R_opt=0.0857, R_opt−R_grad=+0.027
  at n_K=1991, but this E7 run gives R_opt=0.0142, R_opt−R_grad=−0.076 at the
  same n_K. The difference is the loss_fn checkpoint (earlier: g_screen sigmoid;
  here: arm_a global_sigmoid) and/or minibatch. The residual magnitude and the
  sign of R_opt−R_grad are **measurement-convention sensitive** — consistent
  with the earlier "3.2% vs 5.85%/8.57%" discrepancy.

## Verdict for E7
- **PASS (state dependence):** the residual R_grad grows with optimizer state
  (n_K), confirming the residual is optimizer-state-conditioned.
- The update distortion R_opt is small and its sign is measurement-sensitive;
  report this honestly.

## Relation to the diagnosis
E7 adds: the non-scalar residual is not just structured (E5) and dose-responsive
(E6) — it is also **optimizer-state-dependent** (grows with n_K). This enriches
the honest diagnosis: the residual is a real, state-conditioned, structured
phenomenon, but benign (g=1.3 wins FID).

## Files
- Script: `analysis/run_e7_direct.py`
- This summary: `analysis/e7_optimizer_state_result.md`
