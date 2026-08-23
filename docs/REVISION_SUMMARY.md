# Revision Summary

## New paper spine

The manuscript now follows one argument from beginning to end:

> In fixed-$q$ ECT, pair spacing is an exact composite intervention on the detached target endpoint and explicit denominator weighting. Its matched-state action is exactly characterizable, while its finite-training quality consequences depend on trajectory and horizon. The complete intervention can advance entry into a high-quality one-step regime on several trajectories without defining a universally best static gap.

## Major revisions

- Rewrote the title, 224-word abstract, Introduction, Related Work/Setup, theory, design, results, optimizer boundary, discussion, conclusion, and appendix.
- Integrated the exact one-sided derivative, finite-gap decomposition, four-arm identities, and conditional local law $\kappa=\nu(p-1)-\alpha$; the ECT specialization is explicitly local and does not validate $\nu$.
- Replaced endpoint-only motivation with seven-budget formal replay curves and seed-resolved observed time-to-quality.
- Kept original formal, formal replay, secondary A/B, and secondary factorial evidence classes separate.
- Added all adverse cases: two ties, two later crossings, one censored seed, changing four-arm winners, and the failed moment-transport gate.
- Added a primary-source literature audit, claim/evidence matrix, source map, compact unified CSV, figure builder, and submission QA records.
- Added a complete AI use statement, ethics statement, and reproducibility statement.

## Claims removed or downgraded

- Removed any universal claim that $g=1.10$ improves consistency training.
- Removed the universal or pooled “13% compute saving” framing; the manuscript reports one 128-kimg observed stage only for identified seeds.
- Replaced target-dominant language with seed-, budget-, and sampler-dependent component rankings.
- Removed denominator-irrelevance language.
- Downgraded RAdam from mechanism/novelty to a frozen-state supporting boundary; no causal FID claim remains.
- Kept ideal endpoint invariance conditional and prevented the local factorization/scaling law from becoming an endpoint-ranking theorem.
- Labeled extended cohorts as secondary/post-preregistration rather than prospective confirmation.
- Added explicit fixed-$q=256$, single-dataset, single-configuration limits.

## Main-text evidence objects

1. Figure 1: composite pair-spacing intervention and local-to-trajectory distinction.
2. Table 2: complete seed-level formal 256-kimg endpoint cells and B$-$A contrasts.
3. Figure 2: formal seed-resolved four-arm exact-budget curves.
4. Figure 3: seed-resolved first observed NFE1 FID $\leq 10$ budget, including censoring and adverse seeds.
5. Figure 4: formal budget-dependent factorial contrasts.
