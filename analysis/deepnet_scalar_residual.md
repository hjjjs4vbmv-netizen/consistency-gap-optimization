# Deep-network gap gradient moments

This is a gradient-only supplementary diagnostic. The checkpoint parameters were held fixed; no optimizer was created or stepped.

For each minibatch, the image batch, per-example timestep vector, shared noise tensor, and dropout RNG state were reused for every gap value.

The scalar fit is the least-squares projection `a_g*=<mu_g,mu_1>/||mu_1||^2`; the directional residual is `||mu_g-a_g*mu_1||/||mu_g||`.

## Provenance

- checkpoint: `/data/raw/ECT/ect_runs/pr_gpu_smoke_role_d_v4_20260806/network-snapshot-000020.pkl`
- checkpoint SHA256: `eda4c12e95bca6308b897e54dbc3e61b93d0f46c9299a113b7eb4f2bd36b83a5`
- dataset SHA256: `a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1`
- batches: 64; batch size: 128; seed: 20260806

## Whole-model moments

| g | ||mu_g|| | a* | cos(mu_g,mu_1) | residual | variance trace | normalized noise scale | batch residual mean +/- sd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 591.004 | 1.10915 | 0.999988 | 0.00480567 | 14286.4 | 0.0409017 | 0.0150341 +/- 0.00557131 |
| 1 | 532.838 | 1 | 1 | 0 | 11594.2 | 0.0408366 | 0 +/- 0 |
| 1.2 | 439.745 | 0.825257 | 0.999962 | 0.00876349 | 8040.8 | 0.0415812 | 0.0196102 +/- 0.00556753 |
| 1.3 | 402.538 | 0.755391 | 0.999909 | 0.0134902 | 6667.41 | 0.0411475 | 0.0243458 +/- 0.00577681 |
