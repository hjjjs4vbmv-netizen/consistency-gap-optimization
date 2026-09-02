# Fresh q256 crossed-switch replication: frozen n=12 design and amended n=11 result

This directory contains the frozen implementation and the complete execution
history for `q256_fresh_crossed_switch_n12_matpool_v1`. The original design used
seeds 31--42 (n=12) and 264 blind evaluation jobs. That design did **not** reach
finalization. Before any fresh quality metric was observed, the authors abandoned
the original n=12 claim after a terminal failure in the seed38/AB training cell
and authorized an n=11 complete-case analysis over seeds
31--37 and 39--42. The amended blind matrix contained 242 jobs.

The final n=11 verdict is **INCONCLUSIVE** under the frozen decision rule. This
does not confirm directional history carryover and does not establish a practical
null. The seed3--7 crossed-switch result is discovery evidence; the fresh n=11
run is an outcome-blind, author-amended replication that did not confirm the
directional effect and did not meet equivalence.

No command in this runbook reads the independent P2 experiment. No private
evaluation map or unsealed metric artifact belongs in the public evidence bundle.

## Execution timeline and deviations

1. **Frozen n=12 design.** The rebuilt runtime, exact-resume parity, protocol,
   six-GPU assignment, 12-seed training plan, 264-job blind manifest design, and
   statistical rule were frozen before fresh FID/KID observation.
2. **Formal training.** Eleven seeds completed. Seed38/AB encountered a terminal
   numerical failure. One author-approved bounded numeric recovery was attempted;
   the original failure was archived and preserved. The cell again terminated,
   and the missing seed38/AA cell was not run.
3. **Outcome-blind n=11 amendment.** Before evaluation started and with
   `quality_metrics_observed_before_amendment == false`, the authors explicitly
   abandoned the n=12 claim and excluded seed38 as an incomplete seed. This is a
   complete-case amendment, not a random missing-seed design. Because the failure
   occurred in a specific treatment trajectory (AB), informative missingness
   cannot be ruled out.
4. **Blind evaluation recovery.** The first 242-job launch failed before any job
   was sealed because the evaluator cache exhausted local storage. The failed
   public/secret control artifacts and six worker failures were archived. One
   author-approved storage recovery created a fresh cache and a fresh opaque
   manifest/map pair; it did not inspect the old private map or metric values.
5. **Full seal.** The recovery matrix completed 242/242 opaque jobs with zero
   failures. Decoding was allowed only after all individual receipts were
   `SEALED_PASS` and `evaluation_matrix_seal.json` was `SEALED_PASS`.
6. **Post-seal reporting.** Evaluation was not rerun. Two author-approved
   post-seal reporting recoveries preserved the failed reporting attempts. The
   second corrected NumPy scalar JSON serialization and produced the final report,
   statistics, completion receipt, and SHA-256 manifest.

The recovery counts are therefore: one bounded training numeric recovery, one
evaluation storage recovery, and two post-seal reporting recoveries. The phrase
"single formal pipeline with no retry" describes the original launch policy,
not the complete execution history. All recoveries were manual, explicit,
hash-bound, and preserved their predecessor failure artifacts; no automatic
retry loop was used.

## Final evidence and claim boundary

The reviewable, post-seal public bundle is committed under
`results/q256_fresh_crossed_switch_n12_matpool_v1/final_11seed/`. Its public
manifest binds the copied files to the formal-node SHA-256 values. It includes
the final report, completion receipt, matrix seal, integrity report, decoded
results, frozen statistics, primary decision, and per-seed H/C/I/Q/G table. It
does not include the sealed private evaluation map, P2 content, credentials,
checkpoints, or temporary `metric-*.jsonl` files.

The primary decision uses seed-level H at 1024-kimg NFE1 FID-50k. `INCONCLUSIVE`
means the two-sided 95% confidence interval covers zero while the two-sided 90%
confidence interval is not wholly contained in the frozen +/-3% equivalence
band. NFE2, KID, intermediate milestones, AULC, individual cells, interaction,
and checkpoint-ranking diagnostics are descriptive and cannot rescue or alter
the primary verdict.

Accordingly, manuscript language must not state that fresh replication confirmed
the claim that B history improves future quality under both current schedules.
The supported wording is: a seed3--7 discovery was followed by an outcome-blind,
author-amended n=11 complete-case replication whose primary result was
inconclusive. The seed38 trajectory-specific failure remains an explicit
missingness limitation.

## Fail-closed amendment validation

`authorization.py` is dependency-free and is called by every downstream n=11
consumer through `experiment.validate_eleven_seed_authorization`. It checks the
authorization schema and frozen fields, including the no-quality-observation
attestation and amended threshold, then rehashes and validates:

- the numeric-recovery-v2 authorization;
- the unique terminal seed38/AB failure receipt;
- all eleven seed completion receipts.

A hand-written minimal JSON cannot authorize evaluation or analysis. The CI
workflow `.github/workflows/q256-fresh-protocol.yml` exercises field and source
tampering tests without importing Torch or requiring a GPU. It also verifies the
frozen protocol hash, compiles all q256 Python sources, and parses every shell
entry point.

## Original frozen gate order

The original pre-execution gates were:

1. build and freeze the isolated Conda-pack runtime;
2. run the relevant unit/integration suite;
3. run exact A and B uninterrupted-vs-resumed engineering parity;
4. freeze `protocol.json` against the implementation commit and remote assets;
5. commit the protocol, require a clean worktree, and run preflight;
6. launch the six-worker fail-closed pipeline.

Formal workers used independent single-GPU processes (`world_size=1`), fixed GPU
UUID assignments, deterministic algorithms, immutable source states, matched
randomness receipts, and advisory-only monitors. Only the protocol hard timeout
could automatically terminate a job.

The frozen protocol remains the historical n=12 protocol and must not be edited
after the fact. The n=11 amendment and recovery receipts extend it through
hash-bound authorizations rather than rewriting the preregistration.
