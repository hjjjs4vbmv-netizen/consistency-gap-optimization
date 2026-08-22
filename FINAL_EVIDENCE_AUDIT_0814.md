# FINAL EVIDENCE AUDIT — 2026-08-14

## BLOCK #54

Audit target: PR #54 head `fe17b32823961257d98e7a787acc00d11466b0b0`; status `ANALYZED` (no training or metric generation performed).

The scoped checks pass: #54 preserves #50's complete adjudication tree byte-for-byte (`3805b1c52cb4115d333e2035a2427354ac5c8fcc`), including explicit D1–D5, `protocol_exact=false`, 6/6 passed receipts, `failed_conditions=[]`, and `rerun_affected_runs=[]`. It also preserves #51's handoff JSON and #53's result tree byte-for-byte. The only evaluated block directories are `5000–9999`, `10000–14999`, and `15000–19999`; all 54/54 metric JSONLs pass the committed SHA256 manifest, and both CSVs rebuild exactly from them.

Blocking provenance gaps remain:

- `31/31 passed, 0 skipped` exists only in the #50 PR text/comment; no raw test transcript or GitHub check is retained. The fixed suite collects 31 tests, but the present host reproduces only `23 passed, 8 skipped` because the required PyTorch audit runtime is unavailable.
- The 54 FID/KID receipts do not retain or hash-bind generated samples/features (`NPZ`), and each JSONL omits the explicit sample range and checkpoint SHA256. Exact seed-3 Arm B/C EMA checkpoint hashes are also absent. Therefore the scientific numbers stop at metric receipts, not the required checkpoint → samples/features → metric chain.
- The seed-3 FID-50k SHA manifest names stale `arm_*/nfe1/` paths; the three hashes match the committed JSONLs at `arm_*/metric-fid50k_full.jsonl`, but `shasum -c` cannot resolve the manifest as written.

| Main claim/number | Closest committed artifact at #54 head |
| --- | --- |
| D1–D5; `protocol_exact=false` | [blind evidence][blind-evidence], [adjudication][adjudication], [provenance appendix][appendix] |
| 6/6 integrity PASS | [six public receipts][receipts] and `blind_evidence.json → per_run_integrity` |
| 31/31; zero reruns | [#50 claim text][pr50] (not a raw log); `blind_adjudication.json → decision` for zero failed/rerun lists |
| Frozen blocks `5000–9999`, `10000–14999`, `15000–19999` | [#51 machine handoff][handoff] |
| 27 cells; 54 metric files; exact block usage | [raw metric JSONLs][metric-tree], [SHA256 manifest][metric-hashes] |
| Seed-3 FID-50k A/B/C `314.5528 / 205.6356 / 296.8897`; absorption `0.8378298468` | [three FID JSONLs][fid-tree], [quality decision][fid-decision] |
| FID `delta_ctrl` mean ± SD: seed 3 `+90.4944±0.4670`, seed 4 `+4.9459±0.1526`, seed 5 `−14.5170±0.2728` | [blockwise CSV][blockwise], [summary CSV][summary], [54 JSONLs][metric-tree] |
| KID has the same signs; `delta_gap < 0` for all 18 seed×block×metric rows | [blockwise CSV][blockwise], [summary CSV][summary], [quality decision][block-decision] |

[blind-evidence]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_seed_replication_blind_adjudication/blind_evidence.json
[adjudication]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_seed_replication_blind_adjudication/blind_adjudication.json
[appendix]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/supplementary/GAP_LR_SEED_REPLICATION_PROVENANCE_APPENDIX.md
[receipts]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/tree/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_seed_replication_blind_adjudication/public_receipts
[pr50]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/50
[handoff]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_seed_replication_role_e_handoff/role_e_disjoint_5k_handoff.json
[metric-tree]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/tree/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/disjoint_5k_0813/blocks
[metric-hashes]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/disjoint_5k_0813/RESULT_SHA256SUMS.txt
[fid-tree]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/tree/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/nfe1_fid50k
[fid-decision]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/nfe1_fid50k/quality_decision.json
[blockwise]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/disjoint_5k_0813/blockwise_results.csv
[summary]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/disjoint_5k_0813/disjoint_block_summary.csv
[block-decision]: https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/blob/fe17b32823961257d98e7a787acc00d11466b0b0/results/gap_lr_matched/disjoint_5k_0813/quality_decision.json
