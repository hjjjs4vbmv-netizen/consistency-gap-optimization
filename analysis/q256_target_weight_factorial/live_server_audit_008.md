# Live server audit 008: authoritative transfer source extra

The first A smoke under commit `3b0ad3c` reached authoritative-transfer loading
and stopped before the first optimizer attempt. No factorial telemetry file was
created. B/C/D were not started and the GPU was released.

The target was missing no tensor from the authoritative EMA. The source had one
extra tensor: `model.map_augment.weight`, shape `[128, 9]`, dtype
`torch.float32`, raw contiguous tensor SHA256
`4500f8ac1eb5cc8dd4096595a798c8ea4793d42f8433014ab67e41d5ceb70de0`.
It is the checkpoint's augmentation map; the frozen protocol uses `augment=0`
and therefore the target network does not instantiate that map.

The correction does not restore generic partial loading. Every destination
parameter and buffer remains mandatory with exact name, shape, and dtype. The
only permitted source extra is the content-bound tensor above, and that policy
is persisted in trajectory identity and checked by the arm verifier. Any other
missing, extra, shape, dtype, or content difference remains fail-closed. All
Role E and smoke authorization evidence must be regenerated.
