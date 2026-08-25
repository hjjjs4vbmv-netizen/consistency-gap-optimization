# PR79 focused validation

Validated 2026-08-23 against head `ba9889fb0f8a3b13b64d373062847eeb994868eb`.

| Gate | Result |
|---|---|
| Replay/smoke/evaluation launcher `bash -n` | PASS |
| Training, saver, exporter, audit, collector Python compilation | PASS |
| Structured checkpoint inventory | PASS: 84 rows |
| Canonical endpoint parity | PASS: 12/12 `BITWISE_EQUIVALENT` |
| Training integrity | PASS: 12/12, zero formal crash recoveries |
| Formal FID/KID results | PASS: 168/168 |
| Seed-level factorial contrasts | PASS: 504 rows |
| Paired contrast summaries | PASS: 168 rows |
| Compact evaluation package SHA256 | PASS: 5/5 top-level files |
| Server artifact-manifest receipt | PASS: 1,487 entries |
| Server archive verification receipt | PASS |
| Git whitespace check | PASS |

The training/saver source files are byte-unchanged from executed training commit
`c8721a05227f3ff171f8dc1f559a64d58281c0ae`; subsequent commits add or normalize
result evidence only. Before formal launch, the deterministic MatPool runtime
passed the milestone saver unit checks and the real seed3/armA 384 kimg
save→reload→same-budget-resume smoke. The completed 12-trajectory replay then
provided the stronger end-to-end regression gate: all 12 canonical 1024 kimg
computational endpoints matched PR76 bitwise.

No training or metric job was rerun during this focused validation.
