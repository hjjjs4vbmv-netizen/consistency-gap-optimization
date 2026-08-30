# q256 seed3-7 archived-control compatibility audit

Status: **PASS**

Protocol SHA256: `195ca2843791c0ea28ac5a87f3c9e0fb24a4fb8c9214b665331dbbd92648b32d`

- 100/100 A/B control cells joined to PASS receipts.
- Checkpoint, generated-feature, dataset, sampling, and metric values match.
- Seed6-7 evaluator code is identical to the frozen evaluator in all relevant files.
- Seed3-5 differs only by later shared-feature reuse; its original 168/168 jobs already produced byte-identical KID/FID features.
- No control was selected or discarded by quality.
