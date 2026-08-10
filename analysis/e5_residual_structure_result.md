# E5 — Non-scalar residual structure decomposition (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Free analysis on existing
layerwise CSVs (`radam_update_stateful_layerwise.csv`, `gap_gradient_layerwise.csv`),
no GPU. Script: `analysis/e5_residual_structure.py`.

## Hypothesis (E5)
The non-scalar residual is STRUCTURED (layer-depth / magnitude / direction), not
per-coordinate noise.

## Results

### (a) Layer-depth structure
| quantity | pearson vs depth | spearman vs depth |
|---|---|---|
| `R_opt_layer` | +0.067 | +0.062 |
| `\|h_mean − 1\|` (gauge deviation from 1) | **+0.461** | **+0.403** |

- The update residual `R_opt_layer` has no strong depth trend.
- BUT the coordinate gauge's deviation from scalar equivalence (`|h_mean−1|`)
  **grows with layer depth** (moderate, ρ≈0.4). Coarse/deep 8×8 layers deviate
  more; fine/shallow 32×32 layers deviate less.

### (b) Magnitude correlation
| quantity | pearson vs update_mag | spearman |
|---|---|---|
| `R_opt_layer` | +0.025 | −0.070 |
| `\|h_mean−1\|` | −0.107 | −0.103 |

The residual is **NOT magnitude-correlated**. It is not simply "big-gradient
layers distort more."

### (c) Direction residual vs gap (dose-response, gradient level)
| gap | mean direction_residual (208 layers) |
|---|---:|
| 0.9 | 0.0116 |
| 1.0 | 0.0000 (reference) |
| 1.2 | 0.0136 |
| 1.3 | 0.0173 |

The gradient direction residual is a **monotone, systematic function of
|gap−1|** — it grows as the gap deviates from 1. This confirms the residual is a
deterministic consequence of the intervention (dose-response), not noise.

### Gauge deviation spread (layer level)
`|h_mean−1|`: n=208, mean=0.037, std=0.008, p05=0.027, p95=0.050, frac>0.1=0.
- Layer-level gauge deviation is **small (~3–5%) and tightly concentrated**, and
  systematically below 1 (h_mean ≈ 0.93–0.97, consistent with coordinate-level
  h_actual≈0.837 but partially averaged by layer aggregation).
- No single dominant "bad" layer; the residual is spread across layers with a mild
  depth trend.

## Verdict for E5
- **PASS (structure, not noise):** the residual is structured — it has a layer-depth
  trend (deeper → larger gauge deviation, ρ≈0.4), is a monotone dose-response of the
  gap, and is not magnitude-driven.
- It does NOT localize to a few layers and is NOT large/magnitude-concentrated.

## Relation to the ICLR plan
This supports the **honest diagnosis** framing: the non-scalar residual exists, is
reproducible, and is *structured* (depth + gap dose-response). It does **not**
support the mechanism+method framing (already falsified by the clean g=1.3-vs-1.0
FID result — the residual-laden arm is much better on quality). Combined:
- residual EXISTS + STRUCTURED → a real, characterizable phenomenon (diagnosis)
- residual NOT harmful (g=1.3 wins FID) → no causal-quality claim is defensible
- residual has a mild depth trend + gap dose-response → can be described, not "fixed"

## Files
- Analysis: `analysis/e5_residual_structure.py`
- This summary: `analysis/e5_residual_structure_result.md`
- Data: `analysis/radam_update_stateful_layerwise.csv`, `analysis/gap_gradient_layerwise.csv`
