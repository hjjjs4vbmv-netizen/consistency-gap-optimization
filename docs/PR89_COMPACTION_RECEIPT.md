# PR #89 compaction receipt

Date: 2026-08-28

Branch: `codex/nonlinear-forcing-feedback-v1`

Base: `codex/operator-algorithmic-jacobian-v2`

## Git retention boundary

The PR retains frozen protocols, analysis code, tests, compact CSV/JSON
results, reports, manifests with SHA256 values, correctness receipts, and only
small exemplar formal receipts. It does not retain the complete 160-cell
Jacobian receipt matrix, full 64-step forcing-feedback telemetry, raw tensor
arrays, or large execution payloads.

At the start of this revision, the already-compacted remote PR contained 66
changed files and 13,092 additions. This superseded the earlier expanded
working state described in the review request (216 changed files and roughly
960,000 additions).

With this staged revision applied, the full PR diff contains 71 changed files,
17,536 additions, and 0 deletions relative to the PR base. Only seven formal
Jacobian files remain under the v2 formal raw-receipt tree: two manifests and
five regime exemplars.

## Existing PR-wide archive

The prior compaction archive remains the retrieval source for the complete
160-cell Jacobian receipts and the original full forcing-feedback telemetry.

- Release tag: `pr89-audit-artifacts-v2`
- Asset: [`pr89-full-audit-artifacts-v2.zip`](https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization/releases/download/pr89-audit-artifacts-v2/pr89-full-audit-artifacts-v2.zip)
- File count: 166
- Uncompressed bytes: 26,711,088
- Compressed bytes: 953,604
- SHA256: `22dee9eba6caacf37bf5baeba35cc723df4b9883668d14f9dcd5c4c9beffad63`

## Forcing-feedback v2 full telemetry archive

- Asset: [`pr89-forcing-feedback-v2-full.zip`](https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization/releases/download/pr89-audit-artifacts-v2/pr89-forcing-feedback-v2-full.zip)
- Contents: `analysis/nonlinear_dynamics_gate/forcing_feedback_summary_v2_full.json`
- File count: 1
- Uncompressed bytes: 23,362,382
- Compressed bytes: 565,985
- SHA256: `d4eeb19e47ff898d5688158ce8e6133e492337e645b83ef007140b15684e8e01`
- Upload status: **UPLOADED_AND_VERIFIED**
- Release tag: `pr89-audit-artifacts-v2`
- Remote GitHub digest:
  `sha256:d4eeb19e47ff898d5688158ce8e6133e492337e645b83ef007140b15684e8e01`

The archive contains complete step replay telemetry but no credentials. The
remote size and GitHub-provided SHA256 digest match the local archive receipt.
The A100 execution copy also remains at
`/root/pr89-role-b-v2-20260828/formal/forcing_feedback_summary_v2.json` on the
recorded `px-cloud1.matpool.com:26078` execution endpoint, but the GitHub
Release asset is the durable retrieval source.

## Compact v2 artifacts retained in Git

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `forcing_feedback_per_step_v2.csv` | 469,641 | `98fdd94275efea90bedf1708198e68930c3e299dfda774e45a27dbfe4617ae3d` |
| `forcing_feedback_summary_v2.json` | 91,866 | `952bafb58bc551a5948fc6cc30e3709bb856caebb6bc3d6ed32f3646c9f0a691` |
| `FORCING_FEEDBACK_REPORT_V2.md` | 4,695 | `33698d4194443f72fc08de320511702ee24c8c653d3df25a924dd28958921d70` |

The compact summary stores the full 64-step receipt sequence count and
canonical SHA256 while omitting the expanded sequence itself.

## Replay and content-equivalence receipt

- Formal training state SHA256:
  `fbda746805e6614319b96653563757f9e48670339e8f275f018194ebe19c9575`
- Formal checkpoint SHA256:
  `09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9`
- Original frozen-batch container SHA256:
  `6751e98fcc6c91bff83fe96976453535256c2de940b3e4a5fc8b3384e7c24929`
- Rebuilt frozen-batch container SHA256:
  `b8c42e3522dc14a9b75659d4b61dad94dfe638cd55361779a532b97293fb8987`
- Canonical receipt SHA256 over all 32 microbatches' images, labels, `t`,
  noise, and dropout RNG states:
  `b1eb60e44bdd7f4e6648d2af1439cf36a3873de20b6009d963295ab3abb804e9`
- Content equivalence against the published formal replay: **PASS**

The differing batch-container hash is not treated as byte identity. The v2
runner separately fails closed on the transition-input tensor receipt hash;
that hash matches the published formal replay exactly.

## Verification

- Formal replay implementation SHA256:
  `eaad0e4bb1e63cce4c63eaaf07a3b880bb301f432571374944a0a357fd95699b`
- Final revision implementation SHA256 (metric kernel unchanged; claim keys,
  report alignment column, and LF CSV serialization finalized afterward):
  `ed71437369b2c5aadd4c1edcbde095082231852922a7563e21bf7ce59d0ce41a`
- Compact/migration implementation SHA256:
  `6431a3a3d223ae9f503b54521febae715c95f1e3b10913296164705d5ec07ce1`
- 1,536 per-step rows; 768 carryover-corrected `theta`/`EMA`/`m`/`v` rows.
- All exact closures pass and the frozen source state/RNG are preserved.
- All old v1 columns are exactly unchanged across all 1,536 rows.
- RAdam retention is read from actual parameter groups (`beta1=0.9`,
  `beta2=0.999`).
- EMA carryover uses the implemented map (`ema_beta=0.9993` for parameters;
  identity for untouched EMA buffers).
- No mechanism winner is selected; stronger expansion wording is withheld
  because no second independent state replication is present.
- PR #89 focused regression set: **36 passed**.
- Full repository test attempt: **429 passed, 12 skipped, 17 failed**. The 17
  failures are outside this revision's files and concern host/runtime sampler
  compatibility, an MPS metric path, q256 temporary-run path binding, a frozen
  unrelated file hash, and a global `sys.path` assertion.

After retrieval, verify the archive with:

```bash
shasum -a 256 pr89-forcing-feedback-v2-full.zip
unzip -t pr89-forcing-feedback-v2-full.zip
```
