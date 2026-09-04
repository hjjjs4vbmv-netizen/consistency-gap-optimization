# Fresh q128 Regime and History Generalization Study v1

This is a fresh, independent n=8 cohort. Old q128 seeds 3–5 are discovery reference only and cannot enter primary inference.

Each seed runs A, Bsame, Bmatch, Cmatch, and Dmatch from transfer initialization to 1024 kimg in the frozen counterbalanced order. A@512 and Bsame@512 full states additionally generate AB and BA crossed continuations. AA and BB denote the corresponding native A and Bsame trajectories. All work for one seed is pinned to its assigned GPU.

The sole primary estimand is `H_A = log(FID50k_NFE1(BA@1024)) - log(FID50k_NFE1(AA@1024))`, with `E[H_A] < 0`. The complete statistical and missingness rules are in `analysis_plan.json`. All quality jobs use opaque IDs and remain undecoded until every primary and key-secondary job is sealed.

Formal launch is fail-closed on protocol freeze/push, asset and source hashes, zero clipping, stage-0, native A and Bsame parity, tensor correctness, finite loss/gradient, five-arm smoke, exact source export/import, A→A and Bsame→Bsame computational-state parity, legal switches, optimizer/EMA/GradScaler/RNG/sampler restoration, source immutability, miniature opaque evaluation seal/decode smoke, and the frozen test suites.
