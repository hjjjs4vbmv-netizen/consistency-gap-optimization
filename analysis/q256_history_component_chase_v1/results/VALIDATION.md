## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-08
- Verification Status: ANALYZED
- Version Label: component_results_v1

## Statistical validation

This reviews the completed fixed experiment; it does not replay all GPU training. All 320 slots are terminal: 300 PASS and 20 NO_ENDPOINT. The 145 old K metrics remain unchanged. New failures were not repaired or imputed. All three primary contrasts and H_J use the same prespecified 14 seeds, with three block log-FIDs averaged within each seed before inference.

| Finding | Effect and nominal 95% CI | Holm p | Interpretation |
|---|---|---:|---|
| H_T | FID ratio 0.95417 [0.88015, 1.03443] | 0.30631 | Inconclusive; not evidence of no effect. |
| H_W | FID ratio 0.91253 [0.87334, 0.95349] | 0.0017786 | Evidence of lower endpoint FID within the fixed conditions and complete-pair qualification. |
| I | Ratio-of-ratios 1.04809 [0.98028, 1.12060] | 0.30631 | Interaction not established; not a single-arm improvement percentage. |

No practical small/medium/large thresholds for relative FID were prespecified, so no arbitrary classification is added. These intervals are not familywise intervals. Paired-t assumptions concern seed-level differences, not equal variance between independent groups. With n=14, finite-sample assumptions and selection into a successful four-path set limit generalization. No diagnostic altered the test, transformation or cohort.

## Fallacy scan — 11/11 checked

| Type | Assessment | Scope and handling |
|---|---|---|
| Simpson's paradox | No aggregate substitution | Primary and joint contrasts retain the same paired set; available-pair supplements remain separate. This is not proof against all possible subgroup reversals. |
| Ecological fallacy | Avoided | The unit is a training seed, not 50,000 images or three independent seed replicates. |
| Berkson's paradox | Caution | Successful completion can select a nonrepresentative cohort; effects are explicitly conditional. |
| Collider bias | Caution | Completion may depend on treatment and latent training properties. No unconditional effect is claimed. |
| Base-rate neglect | Not a diagnostic-accuracy study | All 16 seeds, 64 path outcomes and 320 quality slots, including failures, are disclosed. |
| Regression to the mean | Controlled scope | Fixed seeds were not chosen for extreme FID and comparisons retain controls. Previously observed A/B still prevents independent confirmation. |
| Survivorship bias | Documented limitation | AA/BA/CA/DA availability is 14/15/15/16, common n=14; exclusions are explicit. |
| Look-elsewhere effect | Primary family controlled | Exactly three main p-values receive Holm correction; all secondary/KID/readout results remain descriptive. |
| Garden of forking paths | Fixed protocol retained | No seed replacement, outlier removal, FID-driven training changes or numerical repairs. Leave-one-seed-out only reports means, without picking an analysis. |
| Correlation versus causation | Restricted experimental interpretation | Intervention precedes outcome and suffix A is held common; claims remain within the fixed conditional experiment. No unique storage or mediator proportion is identified. |
| Reverse causality | No applicable temporal reversal | Endpoint FID did not feed training scheduling; historical A/B reuse is disclosed. |

## Reproducibility and integrity boundaries

- Five original-runtime short GPU check sets passed, including exact short no-intervention and resume comparisons; they are not a full 8000-attempt bitwise replay.
- Formal receipts validated FP32/NFE1, 50,000 samples, fixed generation blocks and metric seed, finite FID/KID, and matching feature hashes.
- `reproduce.py` checks the complete statistical structure, distinguishing exact JSON equality from 1e-12 numerical tolerance across library versions.
- The old M1 published-statistics exact JSON test retains 15 last-bit differences, maximum 4.44e-16; old metrics, results and tests were not changed. It is not called a passing old test.
- BA−DA near zero with p=0.99711 does not establish equivalence. CA−DA's interval crosses zero; different individual significance decisions do not establish different component effects.
- KID shares features with FID and is not an independent replication. Algebraic closure is an identity, not mechanism validation.
