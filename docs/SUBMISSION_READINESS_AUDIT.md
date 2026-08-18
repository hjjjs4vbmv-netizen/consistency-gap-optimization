# Submission Readiness Audit

**Audit date:** 2026-08-18

**Audited collaboration head:** `1c8971e78637a31f044a5e05e262c08adf15c5d2`

**Scope:** publication engineering only; no new theory

**Verdict:** **PASS — anonymous release candidate is ready for upload**

This audit does not add a novelty claim. Generic Adam scale sensitivity and
the `beta1 = beta2` first-order invariance remain outside novelty. B002 is an
appendix limitation. B003, B005, and B006 are no longer open or ambiguous.

## Executive decision

The publication gate is closed successfully at the artifact level:

- an allowlisted, history-free anonymous code artifact was built without
  rewriting the private research repository or its evidence history;
- a separate 2.5 GB anonymous data payload was built on NFS with relative
  paths only;
- the code and data packages independently pass the manifest, array, receipt,
  table, figure, unit-test, and anonymity gates described below;
- the anonymous remote has not been created or pushed. Uploading the two
  frozen artifacts is a release action, not unresolved scientific work.

## Final B003/B005/B006 decisions

### B003 — resolved by regenerated evaluation provenance

Recovery of the old samples/features is formally closed. The publication-v2
run regenerated all 27 declared evaluation cells from one frozen contract and
retained:

- 27 ordered generated-sample arrays, each shaped `(5000, 3, 32, 32)` with
  dtype `uint8`;
- 54 generated-feature arrays, each shaped `(5000, 2048)` with dtype
  `float32`;
- 54 one-row FID/KID receipts;
- SHA256 bindings for all 81 arrays and all 54 receipts;
- exact sample ranges, checkpoint hashes, dataset hash, and zero retries.

Status: `RESOLVED_BY_REGENERATED_PROVENANCE`.

### B005 — resolved by recovery of the original bytes

The exact original seed-3 Arm B/C numbered EMA files were recovered, copied to
immutable NFS storage, and independently rehashed. They were not inferred and
were not regenerated:

- seed-3 Arm B: `a698182f3bbc8307fe1c36c229e5b50772f7fe7e532868353ddf5e395c0ee4db`;
- seed-3 Arm C: `0caf658fdffc30a5d9fd3d143da1a86a7cf40152403e7235cb2b8ae392bc1639`.

Status: `RESOLVED_BY_RECOVERED_ORIGINAL_BYTES`.

### B006 — resolved at 54/54

The recovered seed-3 B/C hashes bind the final 12 historical cells. The
historical cell manifest and the regenerated publication-v2 manifest both
report 54/54 checkpoint-hash-bound metric receipts and zero unbound cells.

Status: `RESOLVED_54_OF_54_CHECKPOINT_HASH_BOUND`.

## Frozen release candidates

### Anonymous code artifact

- local directory: `/Users/wpb/Desktop/SRT/anonymous_submission_export_v4`
- archive: `/Users/wpb/Desktop/SRT/anonymous_submission_export_v4.tar.gz`
- archive SHA256:
  `ab24dc4b087a660d670914bdcc9a0c9a39a6dc18a74450e7b2d493998479f80c`
- release-manifest SHA256:
  `f7c0d87e5241a70dfded94bd335f9e313c760d80ff0fe115c3d2708d7c6183ca`
- unpacked size: 48 MB; 84 manifest-listed files
- contains no `.git/` or `.agents/` tree

### Anonymous data artifact

- private staging location:
  `${PUBLICATION_ROOT}/output/anonymous-publication-v2-data`
- release-manifest SHA256:
  `0e9dd05096e15ce911556e188c0c4225ad22a9854d68f91deb3616a4c87d239f`
- publication-v2 cell-manifest SHA256:
  `deb846e36ac01104ae91de6744a64675a51e5f5a2b89b2c455d0f5c2db6c6ce8`
- size: 2.5 GB; 143 files total
- contents: 81 arrays, 54 metric receipts, 6 lightweight table/manifest files,
  root README, and root release manifest

The data package uses hard links in private staging to avoid duplicating 2.5
GB on NFS. This does not change the bytes or the artifact hashes. A submitted
archive must dereference these files normally.

## Gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| Clean collaboration clone | **PASS** | Anonymous remote read and exact PR head checkout succeeded. Scientific checks used the declared commit rather than an existing development worktree. |
| Core scientific tests | **PASS** | 14 tests passed locally, including the strict evidence manifest and anonymity scanner tests. The GPU-compatible Python 3.10 environment also passed the two retained-artifact unit tests. |
| Canonical manifest | **PASS** | `publication_ready=true`; `blocking_findings=[]`; 54/54 checkpoint bindings; 27 publication-v2 cells; 54 receipts; 81 retained arrays. |
| Data-plane verification | **PASS** | An independent verifier reread every array and receipt from the anonymous data package and checked SHA256, shape, dtype, metric field, and matrix accounting: 81/81 arrays and 54/54 receipts. |
| Headline table reconstruction | **PASS** | `blockwise_results.csv` and `disjoint_block_summary.csv` reconstruct byte-exactly with LF line endings from the anonymous publication-v2 manifest. The older PR #53 table is archival only. |
| Same-trajectory figures | **PASS** | PDF, SVG, and PNG reconstruct byte-exactly from the bundled CSV/receipts under Matplotlib 3.10.6. |
| Cross-K figures | **PASS under declared normalized rule** | All six PDF/PNG files render. The bundled h=20 raw arrays independently recompute `R2={0.8608, 0.9109, 0.9182, 0.7355}` for `K={32,64,128,256}`. The older figure masters were produced by Matplotlib 3.11.1, so cross-version PDF/PNG bytes are not an acceptance criterion. |
| History-free clean-copy tests | **PASS** | A new temporary copy outside both repositories passed the self-contained verifier, all 7 non-Torch export tests, table reconstruction, and both figure reconstruction commands. The two Torch retention tests passed separately on the Python 3.10 GPU runtime. |
| Anonymous code scan | **PASS** | 0 findings across 13 text rules plus explicit `.git/` and `.agents/` structural rejection. |
| Anonymous data scan | **PASS** | 0 findings; metric receipts contain relative checkpoint paths only. |
| Environment declaration | **PASS WITH RECORDED RUNTIME SPLIT** | `environment-publication.yml` now requires Python 3.10.18 and pins pytest, NumPy, SciPy, pandas, Matplotlib 3.10.6, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 11.8. GPU/data tests ran in the compatible Python 3.10/PyTorch runtime; visual checks ran in the pinned Matplotlib 3.10.6 audit runtime. A fresh network solve was not repeated during this audit. |
| B002 | **APPENDIX LIMITATION** | Only the Git/self-contained h=20 subfinding is headline-eligible. The remaining full `R2(K,h)` matrix is not used as headline evidence. |

## Reviewer-visible commands

From a fresh unpack of the anonymous code artifact:

```bash
conda env create -f environment-publication.yml

conda run -n anonymous-publication-audit \
  python scripts/verify_submission_export.py --root .

conda run -n anonymous-publication-audit \
  python scripts/verify_submission_export.py --root . \
  --data-root /path/to/anonymous-publication-v2-data --require-data

conda run -n anonymous-publication-audit \
  pytest -q -p no:cacheprovider \
  tests/test_crossk_h20_recompute.py \
  tests/test_same_trajectory_figure.py \
  tests/test_metric_artifact_retention.py \
  tests/test_submission_export.py

python scripts/reconstruct_headline_tables.py \
  --manifest results/publication_v2_regenerated/publication_v2_cell_manifest.json \
  --outdir rebuilt/tables \
  --verify-against results/publication_v2_regenerated

python scripts/plot_same_trajectory_longitudinal.py \
  --outdir rebuilt/same_trajectory

python analysis/plot_crossk.py --out rebuilt/cross_k
```

The internal release builder and external scanner are intentionally retained
in the private research repository, not shipped in the anonymous artifact.

## Anonymization architecture

The private repository was not rewritten:

```text
private research repository and immutable evidence history
                         |
                         | allowlisted one-way export
                         v
history-free anonymous code artifact + separate anonymous data payload
```

The export excludes collaboration Git history, PR/issue URLs, contributor
identity, machine paths, server accounts, `.agents/`, and development records.
Private commit/checkpoint mapping remains in the private audit ledger.

## Remaining release action

No additional samples, features, EMA checkpoints, or theory are required from
the user. The only remaining action is operational: upload the frozen code
archive and dereferenced data payload to the chosen anonymous artifact host,
then record the public artifact identifier and hashes in the private ledger.
That upload was not performed because no anonymous destination or authorization
to publish externally was supplied.
