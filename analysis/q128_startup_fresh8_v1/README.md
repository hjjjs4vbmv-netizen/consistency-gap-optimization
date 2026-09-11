# Complete fresh q128 four-arm experiment

All 32 planned trajectories and 96 fixed endpoint evaluations completed, with a complete four-arm common set of eight prespecified seeds (301–308).

The primary contrast **S=(C−M)/2 is 0.004235**, nominal paired-t 95% CI **[−0.034159, 0.042629]**, two-sided **p=0.801741**. Four seeds have positive S and four negative S. M/C Holm-adjusted p-values are both 1.0; the supportive H contrast has p=0.660248. The fresh cohort does not support the predicted bidirectional startup effect. Non-significance does not establish zero effect or equivalence.

[中文完整结果](results/SUMMARY_ZH.md) · [Validation](results/VALIDATION.md) · [Both experiments and final costs](../startup_window_experiments_v1/results/FINAL_SUMMARY_ZH.md)

![Paired four-arm endpoints](results/paired.png)

![Prespecified effect intervals](results/effects.png)

## Design and interpretation

AA and DA use native updates; A_startup_down5 applies LR/1.1 and D_startup_up5 applies LR×1.1 only on the first five successful optimizer updates. AMP skips do not advance the window. Native q128 is preserved throughout. Each trajectory keeps its own optimizer/scaler/RNG/sampler and initializes its own E_512 exactly once at 512 kimg; the only quality readout is its 1024-kimg E_512, FP32, NFE1 endpoint.

Each seed/arm uses the mean of three log-FID50k blocks, not FID150k or 24 independent training replicates. S is a composite log contrast: exp(S) is the square root of a ratio of FID ratios, not a model FID improvement. M/C intervals are nominal and the two secondary tests use Holm; H is supportive. Exhaustive sign flips are sensitivity checks under sign symmetry, not randomized arm-assignment tests.

The q256 result in PR112 remains unchanged: its prior responder-enriched cohort favored early over delayed windows. Both q and cohort differ, and S and T are different contrasts; their comparison cannot identify a causal q interaction, a unique rectification mechanism, or a natural mediation fraction.

## Reproduction and audit

```sh
python -m pip install numpy==2.1.2 scipy==1.16.1
python analysis/q128_startup_fresh8_v1/verify_results.py
```

The verifier checks public SHA manifests, 128 redacted outcome/evaluation receipts, all 96 raw receipt-hash bindings against the final archive index, endpoint and sampling identities, eight complete four-arm sets, and independently recomputes S/M/C/H, paired-t intervals, Holm and all 256 sign patterns. It does not rerun GPU training or modify committed outputs. Raw archival hashes are distinguished from redacted public-file hashes. Checkpoints, feature/sample arrays, credentials, private paths and physical GPU UUIDs are excluded from the PR.

An operational host-permit bug caused three unplanned partial duplicate starts. They were stopped, separately archived and excluded according to the pre-existing canonical host assignment, without replacing or rerunning a canonical trajectory or using duplicate quality results. Their **0.400088 GPUh** is included in the final **195.740269 GPUh** process total. The shared final report and cost ledger disclose this deviation. All canonical data were verified on ECT before final analysis; no additional seeds or windows were selected after seeing these results.
