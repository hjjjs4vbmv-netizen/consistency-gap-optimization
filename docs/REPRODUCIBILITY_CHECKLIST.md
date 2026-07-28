# Reproducibility Checklist

Complete this checklist from a clean clone before the anonymous release is frozen. `PASS` requires recorded evidence; an unchecked item is not implicitly satisfied.

## Source and anonymity

- [ ] Anonymous repository starts from the accepted paper commit.
- [ ] Working tree is clean and the exact Git SHA is recorded.
- [ ] No author names, private usernames, email addresses, tokens, SSH hosts, or private repository URLs remain.
- [ ] No absolute paths such as `/root/...`, `/mnt/...`, `C:\Users\...`, or editor-specific paths appear in tracked files.
- [ ] Git history and release artifacts have been reviewed for identity leaks.
- [ ] Third-party attribution and licenses remain intact.

## Environment

- [ ] A fresh environment can be created from the tracked environment specification.
- [ ] Python, PyTorch, CUDA, cuDNN, driver, and GPU versions are recorded.
- [ ] Required imports pass in a clean shell.
- [ ] CPU-only inspection commands do not require CUDA initialization.
- [ ] The expected GPU count and memory requirement are documented.

## Assets

- [ ] Dataset acquisition and conversion commands are documented.
- [ ] Dataset SHA256 is recorded and verified.
- [ ] Transfer checkpoint acquisition is documented.
- [ ] Transfer checkpoint SHA256 is recorded and verified.
- [ ] Large checkpoints and generated image sets are excluded from Git.
- [ ] Persistent and temporary output locations are distinguished.

## Training

- [ ] Primary fixed and global-only commands are present.
- [ ] Secondary-setting commands are present.
- [ ] Seeds, budgets, batch settings, optimizer, precision, and schedule parameters are explicit.
- [ ] `g=1.10` is fixed and not selected on confirmatory seeds.
- [ ] Resume behavior is documented and smoke tested.
- [ ] Runs write immutable metadata before training starts.
- [ ] Checkpoint paths include method, setting, seed, budget, and checkpoint SHA prefix.
- [ ] Successful, skipped, NaN, Inf, loss, and controller telemetry fields are collected.

## Evaluation

- [ ] Sampling seeds are fixed and shared across methods.
- [ ] NFE=1 records `mid_t=[]`.
- [ ] NFE=2 records `mid_t=[0.821]`.
- [ ] KID/FID-5k is labeled as screening or proxy evidence.
- [ ] KID/FID-50k is used for confirmatory claims.
- [ ] Dataset reference statistics and their SHA256 are identical across compared methods.
- [ ] Metric code commit and sample count are recorded.
- [ ] Repeated metric smoke produces the expected reproducibility evidence.
- [ ] Per-seed values, paired deltas, mean, and dispersion are reported.

## Results and claims

- [ ] Primary and secondary endpoints are identified before confirmatory results.
- [ ] Results include all declared seeds, including unfavorable outcomes.
- [ ] Missing or failed cells are reported rather than silently removed.
- [ ] Main tables can be regenerated from tracked lightweight inputs.
- [ ] Proxy and formal results are visually and textually separated.
- [ ] The second setting is described as generalization evidence, not additional tuning.
- [ ] Claim wording matches the number of seeds and evaluation scale.

## Clean-clone audit record

| Field | Value |
| --- | --- |
| Auditor | anonymous reviewer |
| Date (UTC) | pending |
| Repository commit | pending |
| Environment creation | pending |
| Asset verification | pending |
| Training smoke | pending |
| Sampling smoke | pending |
| Metric smoke | pending |
| Overall result | pending |
| Unresolved issues | pending |
