# Arm C — Scale-matched control (does g=1.3 improvement survive LR matching?)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Run on 172.16.30.17
(gpu0003, ECT002). The leader's decisive test: is the g=1.3 FID improvement a
genuine trajectory mechanism, or trivial scalar learning-rate rescaling?

## Design (the leader's Arm A/B/C)

| arm | gap | lr | purpose |
|---|---|---|---|
| arm_a (A) | g=1.0 | 1e-4 | reference |
| arm_b (B) | g=1.3 | 1e-4 (fixed) | uncorrected gap |
| arm_c (C) | g=1.3 | 1.296e-4 (matched, c0_star×1e-4) | scale-matched control |

arm_c compensates the scalar gradient scale (c0_star=1.296 from the fresh-state
audit). If FID_C ≈ FID_A, the improvement is trivial rescaling; if FID_C ≈ FID_B,
it is a genuine mechanism.

## Results (FID-5k, lower better; 3 repeats)

| arm | NFE=1 FID | NFE=2 FID |
|---|---:|---:|
| g=1.0 (A) | 315.8 | 87.7 |
| g=1.3 lr_fixed (B) | 208.1 | 56.6 |
| **g=1.3 lr_matched (C)** | **298.3** | **87.7** |

## Arm C question

| | NFE=1 | NFE=2 |
|---|---:|---:|
| g=1.3 improvement over g=1.0 | 107.7 FID | 31.1 FID |
| remaining after scale-match (C) | 17.5 FID | 0.0 FID |
| **% removed by scale-match** | **84%** | **100%** |

## Interpretation (honest)

- **The g=1.3 FID improvement is LARGELY (84-100%) explained by trivial scalar
  learning-rate rescaling.** At NFE=2, scale-matching removes the improvement
  COMPLETELY (arm_c = arm_a exactly). At NFE=1, it removes 84%, leaving a small
  17.5 FID residual.
- This is evidence AGAINST the "optimizer-memory mechanism has quality
  consequences" thesis: the quality improvement from g=1.3 is mostly trivial
  rescaling, not a genuine trajectory mechanism.
- The optimizer-memory mechanism itself is still REAL (h_actual=0.837 vs
  h_pred=1.001, R_opt≠0) — it creates non-scalar update distortion. But its
  QUALITY consequence is largely trivial.

## Caveats

- arm_c's lr_matched (1.296e-4) uses c0_star from the fresh-state audit; a
  different scale match could leave a different residual.
- NFE=1 has a small residual improvement (17.5 FID, 16%) not explained by scale
  matching — the only weak signal for a genuine mechanism.
- Seed 3 only, 256 kimg, FID-5k (not FID-50k).

## Implication for the ICLR plan

The Arm C control weakens the leader's "optimizer-memory → future performance"
link: the quality consequence is mostly trivial rescaling. The decisive
remaining test is the future-prediction experiment (early residual → future
FID, out-of-sample), which requires new training.

## Files
- Eval driver (server): `analysis/run_arm_c_control.sh`
- Server results: `/data/raw/ECT/ect_runs/arm_c_control_0809/`
- This summary: `analysis/arm_c_scale_matched_control_result.md`
