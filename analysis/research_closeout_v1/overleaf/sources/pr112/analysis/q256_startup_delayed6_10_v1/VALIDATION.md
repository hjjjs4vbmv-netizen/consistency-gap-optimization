# Validation of q256 delayed-window results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-11
- Verification Status: VERIFIED (numerical reproduction and evidence integrity; GPU training was not rerun)
- Version Label: q256_delayed_complete8_v1

The scope is numerical reproduction and evidence integrity of the completed q256 experiment. It is not a rerun of eight GPU trajectories or a new independent scientific replication. No q128 quality outputs were inspected for this report.

## Evidence and interpretation

The primary paired log difference is T=0.1065959153, nominal 95% CI [0.0428479558, 0.1703438748], two-sided p=0.0055037724. Its exponentiated ratio is 1.1124846: delayed-window geometric FID is 11.25% higher than early-window FID in this selected cohort. All 8/8 seeds have positive T. The paired-t interval relies on assumptions about seed-level differences; n=8 does not reliably establish their sampling distribution. Exhaustive sign flips are reported as a sensitivity check under sign symmetry, not as randomization inference. Overall interpretation remains **CAUTION** because of cohort enrichment and the small sample.

L and M intervals are nominal exploratory/contextual comparisons; no q128 Holm family is borrowed or pooled here. p>.05 for L is not evidence of no effect. Auxiliary TOST does not support ±3% equivalence. KID is a correlated auxiliary metric. Reported GPUh describe measured process time, not rental billing.

## Fallacy scan: 11/11 checked

| Item | Assessment |
|---|---|
| Simpson's paradox | No reversal in the reported comparison: every paired seed has T>0; no subgroup search performed. |
| Ecological fallacy | Inference and analysis both use training seeds, not generated images or blocks as independent units. |
| Berkson/selection bias | CAUTION: the prior responder-enriched selection limits generalization to unselected seeds. |
| Collider bias | No post-treatment covariate adjustment is performed; prior selection still limits causal population claims. |
| Base-rate neglect | Not applicable to this continuous paired metric; no diagnostic probabilities are claimed. |
| Regression to the mean | CAUTION: prior enrichment and reused controls preclude calling this fresh independent confirmation. |
| Survivorship bias | All 8 planned seeds and 24 new blocks are retained; technical interruptions and complete common-set counts are reported. |
| Look-elsewhere effect | Direct T is the fixed primary contrast; L, reused M, KID and TOST are labeled secondary/contextual/auxiliary. |
| Forking paths | Fixed windows, seed roster, endpoint and blocks preserved; no effect-driven replacements, exclusions or extra arms. |
| Correlation/causation overreach | Within-cohort controlled window intervention is distinguished from unique rectification, natural mediation or causal q-interaction claims. |
| Reverse causality | Window placement precedes the endpoint; no retrospective mechanism identification is claimed. |

## Verification procedure

`verify_results.py` recomputes the paired analysis from the complete public blocks and checks receipt/source identities, endpoint and diagnostic bindings. The shared controller's CPU tests check successful-step windows, AMP-skip behavior, native no-op parity, exception-safe LR restoration and full-state pause/resume behavior. The frozen engineering runs and original-source receipts are retained in the evidence package. No new full training or quality evaluation is performed for this PR.

Raw archival hashes and public-redacted hashes are kept separate. Published source control values are compared directly with their PR111 repository documents, not merely accepted because a source hash string is present. q256 collection covered verified paid-node archives and freshly hashed owned-node files.

## Executed checks

- The standalone public result verifier passed: 8 complete seeds, 24 new and 48 reused blocks, all endpoint/source bindings, paired statistics, 256 sign patterns, TOST, and all 16 realized LR-window telemetry prefixes. Numerical tolerance is 1e-12 absolute / 1e-11 relative.
- All 33 CPU tests matching `test_startup*.py` passed in the frozen execution environment (2.696 seconds). This covers the old engineering and quality guards as well as the new window controller.
- Publication scan found no credentials, private absolute paths or physical GPU UUIDs in the added result bundle.
- `git diff --check` passed. GitHub Actions repeats the standalone public evidence/statistics check on the PR.
