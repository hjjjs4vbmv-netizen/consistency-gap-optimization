# PR101 512-kimg source-quality backfill

This is a post-hoc descriptive audit of retained PR101 source checkpoints. It performs no training and does not modify PR101's frozen primary verdict.

The source-only inventory covers seeds 50–79 and verifies state file SHA256, 512000 processed images, 4000 attempted iterations, seed, factorial arm, finite EMA, all recorded internal-state hashes, and the source telemetry hash/coverage. It found 59 valid individual states and 29 valid A/B pairs. Seed 67 A@512 was absent. Later AA/BA endpoint failure was not a source exclusion criterion.

`protocol.json` and `protocol.sha256` were frozen on ECT002 after this inventory and before selective checkpoint transfer or formal source evaluation. The formal matrix contains exactly 58 jobs, all NFE1, FP32, FID50k/KID50k, sample IDs 0–49999, metric seed 20260730, one generation job per GPU. Technical transfer resumptions do not change the scientific matrix.

The evaluator archive is the unchanged PR101/PR97 evaluator at `d6aba02fb88e9db0993623895eb2228ed717d810`. The wrapper reuses PR101's `firstwave_eval_pool.run_job` and its shared-feature validator. Snapshot export uses the PR97/replay exporter payload and hashing utilities, with strict source-to-exported-EMA verification. The user explicitly cancelled smoke before formal evaluation. The immutable budget and pipeline amendments record this waiver, EMA-only transfer, reuse of eight full states already received, and per-job input verification before generation while the remaining transfer continues. No smoke result is claimed.

Frozen detector and real-feature bytes are preserved. Because dataset paths enter the original cache key, the independent work directory uses verified hard-link cache aliases; their bytes and SHA256 remain unchanged. A same-named file found on the evaluation machine's shared volume had a different SHA and was not substituted.

Source receipts extracted in the historical archive were unreadable to ECT002. Their identical logical records were recovered from the readable historical audit tar after its recorded SHA256 was verified. The old archive and all original checkpoints remain unchanged. The original runtime tar was also unreadable, so the independent runtime is reconstructed at the protocol's exact Python, PyTorch, CUDA, NumPy, and SciPy versions, with setup logs and a package-freeze receipt.

## Execution order

The environment is provisioned once. Each formal worker waits for its own snapshot receipt and verifies its input hash before generation. Source inventory hashes are reused; repeated full-state and binary seal hashing is omitted under the user budget waiver.

```sh
python -m analysis.q256_terminal_history_512_source_backfill_v1.evaluate environment --work-root "$WORK_ROOT"
python -m analysis.q256_terminal_history_512_source_backfill_v1.evaluate waive-smoke --work-root "$WORK_ROOT"
python -m analysis.q256_terminal_history_512_source_backfill_v1.evaluate formal --work-root "$WORK_ROOT"
python -m analysis.q256_terminal_history_512_source_backfill_v1.evaluate seal --work-root "$WORK_ROOT"
python -m analysis.q256_terminal_history_512_source_backfill_v1.analyze --work-root "$WORK_ROOT"
```

Each attempt has an independent output directory and receipt. The wrapper refuses existing formal runs, retains failures, and performs no automatic scientific retry or seed replacement. It does not decode formal scalar results until every job has a terminal status and the integrity seal has been written.

The scalar CSVs independently reproduce Q = ln(FID_B512) − ln(FID_A512), the actual seed join with frozen H_A, and the four strict-sign quadrants. Exact ties are recorded separately. Correlations are descriptive and no regression line or pooled confirmatory p-value is introduced.

The result package, including scalar artifacts, receipts, plots, compute accounting and checksum manifests, must be returned to a new ECT002 archive and its remote package SHA checked before a results PR is opened. The repository retains compact audit artifacts; large generated samples/features and consumed snapshots are additionally backed up and hash-verified under the new ECT002 archive raw_evaluation_artifacts directory before releasing A100. Original source states remain in their ECT002 archive.

## Final execution and release record

The original frozen driver is retained as audit evidence. Its streaming-readiness races were resolved by the provenance recovery scripts and atomic completion markers, without rerunning evaluated jobs. For a fresh reproduction, complete and verify all inputs before launching the original formal entrypoint. Final validation uses the sparse-checkout-aware finishing script. Scoped CSV Git attributes preserve the frozen CRLF bytes.

The user authorized releasing the A100 instance after all results return. Both the compact tar package and all large generated artifacts must be verified on ECT002 before the official MatPool release command is executed. The original ECT002 source archive remains untouched.
