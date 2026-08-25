# Integrity review after requested revisions

Status: **PASS WITH DISCLOSED LIMITATIONS**

## Evidence checks

- Every reported metric is traceable to the committed 210-row authoritative
  CSV; matrix cardinality is 210/210 and metric cardinality is 420/420.
- The result collector resolves redundant attempts by preassigned partition
  ownership, never by metric quality.
- The deployed evaluator is bound by six full source-file SHA256 values,
  including both metric implementations and the shared feature utility.
- The calibration manifest is byte-bound, the reproduction command reproduces
  all scientific fields and sample hashes, and the exact `g/q` identity is
  stated separately from the numerical diagnostic.
- All 105 training-state files and all 105 EMA snapshots were rehashed against
  immutable receipts.
- AULC and TTQ are explicitly post-unblind descriptive analyses. Neither is
  presented as a preregistered primary outcome.

## AI research failure-mode checklist

| Mode | Status | Evidence |
| --- | --- | --- |
| 1. Implementation bug passing self-review | CLEAR | Focused production-path tests cover native parity, target hashes, inverse-gap scaling, metric-reuse fail-closed behavior, and deterministic result regeneration. |
| 2. Hallucinated citation | CLEAR / not applicable | The result report contains no external literature claims or bibliography. |
| 3. Hallucinated experimental result | CLEAR | Raw metrics, sealed receipts, 15 run summaries, and 210 artifact hashes are committed and cross-checked. |
| 4. Shortcut reliance | CLEAR for the claimed control comparison | The five-arm design changes only frozen target/denominator factors within seed; the report does not claim broad dataset generalization. |
| 5. Implementation bug reframed as insight | CLEAR | Unexpected seed-3 behavior is reported as heterogeneity; correctness gates and redundant-attempt audit do not expose a mechanism-changing bug. |
| 6. Methodology fabrication | CLEAR | Run commands, initial-state receipts, optimizer/EMA/RNG/sampler hashes, hardware model, attempts, skips, budgets, and evaluation settings are all artifact-backed. |
| 7. Early frame-lock | CLEAR with scope caveat | The review corrected the evidence framing: historical `A/Bsame` knowledge and post-unblind AULC status are now explicit rather than forced into the original blindness narrative. |

## Disclosed limitations

- The exact conversational timestamp of first metric unblinding was not logged;
  the chronology reports a bounded interval.
- The original calibration shell command was not preserved literally. The
  manifest plus script defaults define the committed reproduction command,
  which reproduces all scientific fields and sample hashes.
- GPU UUID was absent from the immutable v1 preflight schema and cannot be
  recovered after node release. Hardware model/memory, node hostname, GPU index,
  runtime, dataset, transfer, and arm assignment remain recorded.
- Three seeds support paired descriptive evidence, not population-level
  significance claims.
