# Clean g=1.3 vs g=1.0 FID/KID comparison at the residual-measurement states

Date: 2026-08-10. Branch: `role-c/g13-vs-g10-fid-0809` (NOT merged to main).
Run on 172.16.30.17 (gpu0003, ECT002), per HANDOFF_20260804.md server norms.

## Motivation

The ICLR 2027 strategy's "mechanism+method" premise is that the non-scalar
gradient residual is HARMFUL, so a residual-corrected update would IMPROVE
quality. The repo analysis workflow flagged that no clean g=1.3-vs-1.0 FID
comparison existed at the states where the residual was measured
(`gap_lr_matched_q128_s3_v1`). This experiment runs that missing comparison.

## Setup (clean, matched)

The two arms are the EXACT states used in the stateful RAdam audit
(`radam_update_audit_stateful.json`, which measured a_K*=0.761, R_grad=5.85%,
R_opt=8.57%, h_actual≈0.837):

| arm | gap | schedule | lr | total_kimg | snapshot |
|---|---|---|---|---|---|
| `arm_a_g1_0_lr_fixed_s3` | g=1.0 | `global_sigmoid` | 1e-4 (fixed) | 256 | `network-snapshot-latest.pkl` |
| `arm_b_g1_3_lr_fixed_s3` | g=1.3 | `global_sigmoid` | 1e-4 (fixed) | 256 | `network-snapshot-latest.pkl` |

Both use `global_sigmoid`; they differ ONLY by `global_gap_scale` (1.0 vs 1.3),
with identical lr. This is the cleanest available g=1.3-vs-1.0 comparison at the
exact states carrying the measured non-scalar residual.

Evaluation: `ct_eval.py --nfe={1,2} --mid_t=0.821 --metrics=fid5k_full,kid5k_full
--seed=3`, 3 repeats. Outputs under
`/data/raw/ECT/ect_runs/g13_vs_g10_fid_0809/`.

## Results (FID-5k / KID-5k, lower = better)

| arm | NFE | FID-5k (mean of 3) | KID-5k (mean) |
|---|---|---|---|
| g=1.0 | 1 | **315.81** | 0.3250 |
| g=1.3 | 1 | **208.08** | 0.2033 |
| g=1.0 | 2 | **87.73** | 0.0709 |
| g=1.3 | 2 | **56.60** | 0.0392 |

Relative improvement of g=1.3 over g=1.0:
- NFE=1: FID **−34.1%**, KID −37.4%
- NFE=2: FID **−35.5%**, KID −44.7%

Raw per-repeat values:
- g=1.0 nfe1 FID: 316.47, 315.07, 315.89
- g=1.0 nfe2 FID: 87.89, 87.35, 87.94
- g=1.3 nfe1 FID: 208.33, 207.69, 208.23
- g=1.3 nfe2 FID: 56.08, 56.79, 56.94

## Interpretation (honest)

The g=1.3 arm — which carries the non-scalar gradient residual
(a_K*=0.761, R_grad=5.85%, R_opt=8.57%, h_actual≈0.837) — **massively beats
g=1.0 on FID and KID at both NFE levels** (~35% FID, ~40% KID). The advantage
is large, consistent across both metrics and both NFE.

Implication for the strategy's mechanism+method premise:
- The premise that "the non-scalar residual is harmful" has **no support** in
  this data. The arm carrying the residual is dramatically BETTER.
- A residual-corrected update (E2's Arm C, which projects the g=1.3 gradient
  toward the g=1.0 reference direction) would move the model TOWARD g=1.0's
  much worse FID. So E2 is not just unsupported — it is contraindicated.

Caveats on interpretation:
- g=1.3 differs from g=1.0 in the WHOLE gap (scalar + non-scalar components),
  so this does not isolate the non-scalar residual's specific contribution. The
  dominant effect is likely the scalar gap change. But it is decisive on the
  point that matters for the strategy: there is NO evidence the residual is
  harmful, and the residual-laden arm is the better one.
- FID-5k (not FID-50k), 256-kimg runs, seed 3 only, single schedule pair.
- The g=1.3-vs-g=1.0 gap is large and not noise (3 repeats, tight spread).

## Conclusion for the ICLR 2027 plan

This resolves the open contradiction in the review (g=1.3-wins-FID was
previously only asserted, not measured): **now it is measured, and it holds
strongly.** The mechanism+method strategy is not supported by this data. The
defensible contribution is the honest diagnosis: the non-scalar gradient
residual EXISTS and is structured, but it is NOT harmful — the arm carrying it
is much better on quality. The paper should NOT claim E2 (correct-the-residual
improves quality); it should report the structural characterization + the
quality direction as honest evidence.

## Files
- Eval driver (server): `analysis/run_arm_a_fid.sh`, `analysis/run_g13_vs_g10_fid.sh`
- Server results: `/data/raw/ECT/ect_runs/g13_vs_g10_fid_0809/`
- This summary: `analysis/g13_vs_g10_fid_result.md`
