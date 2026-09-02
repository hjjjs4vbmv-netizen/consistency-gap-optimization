# Live server audit 001

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_live_server_audit_001

The same `gpu0003` host became reachable through its internal endpoint at
`ECT001@172.16.30.17:22`. The former public mapping at
`region-9.autodl.pro:34360` remained unavailable and is not used.

## Verified assets

- Canonical CIFAR-10 archive:
  `/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip`
  - size: 166,000,134 bytes
  - SHA256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`
- Authoritative initial transfer:
  `/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl`
  - size: 223,173,327 bytes
  - SHA256: `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`

The byte-distinct `/data/raw/ECT/datasets/cifar10-32x32.zip` is not eligible
for formal use.

## Runtime and storage

The historical-compatible sandbox is
`/data/temp/ect001-pytorch2401-sandbox`: Python 3.10.12, PyTorch
`2.2.0a0+81ea7a4`, torch CUDA 12.3, and cuDNN 8.9.0.7. Host Python has no
PyTorch and the shared repository virtual environment is not host-compatible.
Formal commands must use the sandbox and explicitly disable TF32.

The root filesystem had only 36 GB free and is ineligible for run artifacts.
`/data/temp` had approximately 5 TB free and is appropriate for isolated code;
`/data/raw/ECT/ect_runs` had approximately 33 TB free and is the durable formal
run root. The formal disk gate is at least 60 GB free on the selected run
filesystem.

## GPU and process stop

The host has two NVIDIA A100 80GB PCIe devices:

- GPU0 UUID `GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859`;
- GPU1 UUID `GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10`.

At audit time another user's long-lived Ray workers held memory on both GPUs.
An independent ECT001 gradient-audit smoke also appeared on GPU0. These are not
this experiment and were not stopped or modified. The shared
`/data/raw/ECT/recurrence_of_ect` checkout was materially dirty and is
ineligible as execution source.

Status remains `formal_training_authorized=false`. Formal smoke and training
must wait for a fresh live audit showing no foreign compute process on the
assigned GPU, a clean committed isolated source, the 60 GB disk gate, and all
versioned correctness receipts. A zero instantaneous utilization reading does
not override an active compute PID.
