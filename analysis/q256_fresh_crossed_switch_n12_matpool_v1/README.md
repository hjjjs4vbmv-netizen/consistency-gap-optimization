# Fresh q256 crossed-switch n=12 MatPool runbook

This directory implements the author-approved rebuilt-runtime amendment for
`q256_fresh_crossed_switch_n12_matpool_v1`. Formal seeds remain exactly 31--42.
No command in this runbook reads the independent P2 experiment.

The gate order is fixed:

1. build and freeze the isolated Conda-pack runtime;
2. run the relevant unit/integration suite;
3. run exact A and B uninterrupted-vs-resumed engineering parity;
4. freeze `protocol.json` against the implementation commit and remote assets;
5. commit the protocol, require a clean worktree, and run preflight;
6. launch the single fail-closed tmux pipeline.

The formal pipeline performs training, training-integrity audit, blind manifest
preparation, 264 evaluations, full-matrix sealing, decoding, frozen statistics,
and final reporting. It contains no retry loop. A failed formal cell or evaluation
job retains its receipts and stops that worker; user authorization is required
before an identical retry. Only the six-hour hard timeout can automatically kill
a job.

```bash
RUNTIME=/root/q256_fresh_crossed_switch_n12_matpool_v1/runtime
REPO=/root/q256_fresh_crossed_switch_n12_matpool_v1/repo
CONTROL=/root/q256_fresh_crossed_switch_n12_matpool_v1/control

bash "$REPO/analysis/q256_fresh_crossed_switch_n12_matpool_v1/build_runtime.sh" \
  "$REPO" "$RUNTIME"

"$RUNTIME/env/bin/python" \
  "$REPO/analysis/q256_fresh_crossed_switch_n12_matpool_v1/parity.py" launch \
  --runtime-manifest "$RUNTIME/runtime-manifest.json" \
  --dataset /mnt/ect_project/q256_seed14_18_eval_assets_20260822/cifar10-32x32-canonical-08c9ed1b2b1c.zip \
  --transfer /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
  --implementation-commit IMPLEMENTATION_COMMIT \
  --output-root /root/q256_fresh_crossed_switch_n12_matpool_v1/engineering-parity

# Freeze protocol, commit it, then create a PASS preflight receipt. The exact
# freeze command is recorded with the protocol commit because evaluator cache
# paths are resolved from the current node. The freeze command requires and
# hash-binds the first-attempt PASS parity_report.json.

tmux new-session -d -s q256_fresh_n12_formal \
  "bash '$REPO/analysis/q256_fresh_crossed_switch_n12_matpool_v1/formal_pipeline.sh' \
  '$REPO/analysis/q256_fresh_crossed_switch_n12_matpool_v1/protocol.json' \
  '$CONTROL/preflight.json' EVALUATOR_REPO EVALUATOR_CACHE \
  >> '$CONTROL/formal_pipeline.log' 2>&1"
```

Per-cell advisory monitors record process trees, GPU identity/utilization/memory/
temperature/power, foreign processes, log growth, disk availability, progress,
accepted steps, and ETA every 30 seconds. They never kill a job. Evaluation
receipts omit FID/KID values; decoding is impossible through the supplied command
until 264 individual `SEALED_PASS` receipts and the matrix seal exist.
