# q256 512-kimg schedule-switch source inventory

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-29
- Verification Status: ANALYZED
- Version Label: q256_schedule_switch_source_inventory_fail_closed_v1

## Verdict

**FAIL CLOSED — 0/10 canonical 512-kimg full training states are present.**

The authoritative host `gpu0003` was reached through `ECT001@172.16.30.17:22`. The PR #80 canonical root still exists, but now contains only `release_archive_v1/`. Its 26,952,376,320-byte `q256_seed14_18_network_snapshots_and_evidence_v1.tar` retains 120 EMA evaluation snapshots, 20 checkpoint manifests, histories, milestone receipts, and other evidence. It contains **zero** `training-state.pt` entries.

The archived checkpoint manifests identify the exact expected 512-kimg states, including their file SHA256 and internal online/EMA/optimizer/GradScaler/RNG/sampler hashes. Direct checks show all ten recorded paths are absent. A wider inventory of `/data/raw/ECT` and `/data/temp/ECT001` found no seed14–18 canonical 512-kimg full state elsewhere.

| Seed | Origin | Expected state SHA256 | Present |
|---:|:---:|---|:---:|
| 14 | A | `abf7117fbea162b2d223616df18c17859567f67608995b02e81363ddd6b9c8d2` | No |
| 14 | B | `8420577e5748db1c152317a60d178a4a6e6d7eb4de0c3296dedf1075b0f1f2c4` | No |
| 15 | A | `df56152e51e9b98dedf3ff575ac6645b56f9f5648a2e92a3e16418b55485e548` | No |
| 15 | B | `63f324c4a4d970e4b93a0432718fafcbd52d0a545a0b773f47e4b593fac31b97` | No |
| 16 | A | `db68a6251c266059861356c4af7495eef1ca06e0830d48cebb2c21151ad9ce2c` | No |
| 16 | B | `5b0f443b69a6d40087455e3bdc47e02d2cf291526cb2c3a6a5a3dcfa7c0c5a0a` | No |
| 17 | A | `3b0d1c309382c4fe2ac1e9cbc62e13a7c4f3661522ebb2d01fd92a17c32efb73` | No |
| 17 | B | `58e3e92bbe32521b9a2eddc726eb4eca7ff4fc0ce8b906f57a38af9a4812cda6` | No |
| 18 | A | `9bc4d7c9b597659dd6d9a7a7f40ad5f14a0bfb4232210898d3f47e7e3eeb7f92` | No |
| 18 | B | `482b933b51e4acba4f1ebcc8d8e1bd8289eb43a433745a2ce3e4e5b4351ce8d5` | No |

## Gate consequences

- Source inventory: `FAIL_CLOSED`
- No-op parity: `NOT_STARTED`
- Formal crossed training: `NOT_STARTED`
- Formal 80-job evaluation: `NOT_STARTED`
- Result analysis: `NOT_STARTED`

The retained snapshots cannot legally substitute for the missing full states because they contain neither RAdam moments/step counters nor GradScaler, RNG, and sampler state. No seed was replaced; no state was synthesized; no replay was silently relabeled as canonical; and no switch quality result was generated or inspected.

The experiment may resume only if all ten byte-identical full states are restored and match the recorded file and internal-state hashes in `source_inventory.json`.
