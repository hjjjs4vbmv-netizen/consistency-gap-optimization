# q256 P2 formal result provenance

- Formal run root: `/data/raw/ECT/ect_runs/q256-p2-b384-pulse-chase-v1-formal-20260901-rerun3`
- Execution commit: `8741ca4d494fc124f10d05aaab9d93243003b9b7`
- Implementation commit: `9026fb930853be005b030a2d87614dfea38c5565`
- Protocol SHA256: `2791223f6195d565a4f1c63758dfc91f62c4c91c6d0951304c96aa5eb9a33366`
- Training integrity: 10/10 sources, 20/20 branches, 10/10 paired seeds, `PASS`
- Evaluation integrity: 60/60 jobs `SEALED_PASS`; numeric results were unsealed only after the seal audit passed
- Primary decision: `INFORMATIVE_NULL` (`D640` is equivalent within the frozen 3% margin)
- Total measured compute: 40.60 A100 GPU-hours

`SHA256SUMS.txt` is the compact manifest from the completed server run. The
large checkpoints, generated samples, and feature arrays remain on the server
and are bound through the committed receipts and manifests rather than copied
into Git.

The earlier infrastructure-only attempts remain preserved on the server. They
ended before producing an eligible formal result and were not substituted,
selected, or analyzed as scientific cells.
