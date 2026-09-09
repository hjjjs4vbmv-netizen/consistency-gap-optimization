# ECT / RAdam startup engineering check

Engineering only. Fixed seeds 50/51, A/D/D_compensate/A_mimic, 64 attempted
updates each, original 1024-kimg schedule. No FID and no endpoint inference.

`entry.py --manifest ...` constructs the original native training command.
The manifest and loop reject other seeds, lengths, source checkpoints and
scientific configurations. Initial model/EMA/optimizer/scaler/RNG/sampler
hashes are checked against the archived receipt before the first attempt.
Only the original A and D losses are used; microbatch means are accumulated
with eight backwards, without division by eight.

`training/startup_update.py` wraps the actual native RAdam step. It computes
rho from each active parameter's current step plus one, rejects unequal
clocks, and applies a temporary LR only when rho <= 5. LR is restored in
`finally`. AMP skips do not enter this wrapper. No moments are altered by the
intervention. Native RAdam still performs its usual gradient/moment updates.

Diagnostics reuse native batch/time/gradient fields, add RNG hashes and all
internal clocks, and save pre-update parameters, finite gradients and actual
update vectors at successful steps 1/5/6. No extra forward/backward or random
draw is inserted. Zero-vector cosine and relative errors are N/A. Comparisons
at different successful-step attempts or diverged parameter states are not
same-state counterfactuals. There is no numerical pass threshold; report
continuous errors separately from engineering correctness.

Run from repository root with the recovered Python 3.11.13 / Torch 2.6.0
runtime. `run_matrix.py prepare` freezes protocol, environment, source hashes,
receipts and all eight commands. `run_matrix.py run` runs them sequentially
with exclusive-GPU checks, process group hard timeouts and a conservative
3 GPUh wall-time ledger. It does not automatically restart a matrix or retry
a failed trajectory. `analyze.py --root EXPERIMENT --output RESULTS` runs
entirely on CPU and regenerates numerical tables and the four-panel figure.

All large tensors and raw logs belong in the persistent experiment directory,
not Git. Startup checkpoints carry `startup_engineering` metadata and are
rejected as formal training resumes.
