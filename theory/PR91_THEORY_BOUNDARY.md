# PR #91 theory and claim boundary

Status: manuscript integration boundary for the frozen ImageNet-64 IA/IB
matrix in PR #91.

## 1. Theory retained in the manuscript

The manuscript uses three bounded objects.

1. **Realized pair spacing.** Parameters such as `q` and
   `global_gap_scale=g` jointly construct the law of realized pairs. The
   scientific intervention is that realized pair-spacing law, not an
   independent mechanism attributed to nominal `g`.
2. **Exact local objective structure.** Write the one-sided stop-gradient ECT
   objective as $w(t,\Delta)\rho_c(e_r)$. The matched-state decomposition and
   target-by-weight factorial identities are exact at fixed parameters,
   sample, and RNG. In legacy CIFAR ECT, $c=0$ and $w=1/\Delta$, so spacing
   changes both target endpoint and explicit weight. In the PR #91 ImageNet
   configuration, $c=0.06$ and the SNR-style weight is independent of
   $\Delta$ at matched $t$, so the local scalar ratio is one and IA--IB changes
   the target endpoint without inverse-gap rescaling.
3. **Carryover-corrected state recursion.** For a block with a shared declared
   carryover operator,

   $$
   \Delta x_{k+1}
   =b_k^x+C_k\Delta x_k+\widetilde R_k^x.
   $$

   This identity separates common-state schedule forcing, mechanical state
   retention, and state-dependent incremental feedback. It is an attribution
   boundary, not a quality-prediction model.

The main text does not use a fixed global schedule operator, a matrix
exponential as a production model, a nonnormal-amplification headline, or a
universal learning-clock claim.

## 2. What PR #91 establishes

The frozen matrix contains three paired training seeds, IA and IB, ten
checkpoints from 1,280 through 12,800 kimg, and NFE1/2 FID/KID evaluations.
The paired configurations differ scientifically only in
`loss_kwargs.global_gap_scale` (`1.0` for IA and `1.1` for IB).

- IA wins every one of the 30 same-seed FID comparisons and every one of the
  30 same-seed KID comparisons through 6,400 kimg.
- Seed 103 becomes unstable after 7,680 kimg in both arms. IB deteriorates more
  sharply at 8,960 kimg; IA also collapses over the later checkpoints.
- Seeds 101 and 102 remain well behaved through 12,800 kimg.
- Three-seed means after 8,960 kimg are dominated by seed 103 and cannot serve
  as a typical late-training ranking.

The first item is evidence for a transient realized-spacing intervention
effect. The second is evidence for trajectory instability in one paired seed.
They are distinct empirical statements.

## 3. Claim ceiling

| Statement | Status | Reason |
|---|---|---|
| In legacy $c=0$ inverse-gap ECT, realized spacing changes the detached target and explicit weight. | Exact for the legacy CIFAR objective | Here $w(t,\Delta)=1/\Delta$. |
| In PR #91 ImageNet training, realized spacing changes the detached target while the SNR-style weight is gap-independent at matched $t$. | Exact for the evaluated PR #91 objective | Here $c=0.06$ and $w(t,\Delta)=\omega_{\mathrm{snr}+k}(t)$, so $s=1$. |
| The target-by-weight identities hold at a matched one-sided-SG objective state. | Exact | Algebraic finite-spacing identity for parameter-independent $w(t,\Delta)$ and differentiable $\rho_c$ at the evaluated residual. |
| The carryover-corrected recursion separates forcing, retention, and incremental feedback. | Exact under shared declared carryover | No differentiability assumption is needed. |
| IA has a consistent transient advantage through 6,400 kimg in the frozen PR #91 matrix. | Supported in the evaluated matrix | All 30 paired comparisons favor IA for each metric. |
| Seed 103 exhibits late trajectory instability in both IA and IB. | Supported descriptively | Both arms deteriorate after 7,680 kimg. |
| The spacing intervention causes or prevents the seed103 collapse. | Not supported | Instability occurs in both treatments and no collapse-specific intervention was run. |
| The exact recursion explains seed103 collapse. | Not supported | The required ImageNet forcing/carryover/feedback terms were not measured across the transition. |
| Corrected feedback mediates FID or KID. | Not supported | No state-to-quality mediation design is available. |
| The CIFAR local production-transition audit identifies an ImageNet instability mechanism. | Not supported | Dataset, model, precision path, optimizer trajectory, and audited state differ. |
| A late three-seed endpoint mean identifies a stable schedule ranking. | Not supported | The mean is dominated by the unstable seed103 trajectory. |

## 4. Main-text wording licensed by the evidence

> The realized-spacing target intervention produces a consistent transient IA
> advantage through 6,400 kimg in the frozen ImageNet-64 matrix. Late training
> exposes a separate stability phenomenon: seed 103 degrades in both arms,
> causing the three-seed endpoint mean to conflate intervention contrast with
> trajectory instability.

> The exact carryover-corrected recursion defines how schedule forcing,
> mechanical retention, and state-dependent incremental feedback would be
> separated at a measured state. It does not identify the cause of the
> ImageNet seed103 collapse or mediate its FID/KID consequences.

## 5. Evidence required to raise the ceiling

A collapse-mechanism claim requires pre-transition and post-transition
ImageNet checkpoints with paired replay of the relevant state blocks,
gradients, optimizer increments, data/RNG inputs, and numerical execution
regime. Independent trajectory replication is required to distinguish a
repeatable instability mechanism from a seed-specific event. A mediation
claim additionally requires a prospectively defined link from those state
diagnostics to seed-resolved quality outcomes.

## 6. Role C verdict

**GO** for the bounded Theory and Discussion integration above.

**HOLD** for any treatment-specific collapse mechanism, FID mediation, or
cross-dataset extrapolation from the CIFAR transition audit.
