# Literature audit: consistency discretization, curricula, and weighting

**Audit date:** 2026-08-24
**Scope:** prior work that could collide with claims about pair spacing,
discretization, curriculum, or explicit loss weighting in consistency-model
training. Results were deduplicated by DOI or normalized title. Only primary
paper or venue sources were used.

## Findings

| Work | Relevant prior art | Collision with PR #81 | Safe distinction |
|---|---|---|---|
| [Consistency Models (ICML 2023)](https://proceedings.mlr.press/v202/song23a.html) | Formulates consistency training on neighboring discretized noise levels and reports that fewer intervals converge faster but to worse samples, whereas more intervals converge more slowly and reach better quality. | High for any claim that discretization-dependent learning speed or endpoint quality is newly discovered. | PR #81 studies exact common-state one-sided training-field identities for a particular finite-spacing ECT intervention and what remains after separately trained factorial trajectories diverge. |
| [Improved Techniques for Training Consistency Models (ICLR 2024)](https://arxiv.org/html/2310.14189) | Introduces inverse-interval weighting $1/(t_{i+1}-t_i)$, changes the discretization curriculum, and studies weighting empirically. | High for claims of first separating curriculum and weighting design. | It does not provide the PR #81 matched target-endpoint/weight training-field identities or four-arm finite-training intervention. |
| [Consistency Models Made Easy / ECT (ICLR 2025)](https://arxiv.org/html/2406.14548) | Defines continuous pairs through $p(r\mid t,\mathrm{iters})$, uses $q$ to shrink $\Delta=t-r$, states that $1/(t-r)$ couples weighting to the mapping, explicitly evaluates decoupled weighting alternatives, and reports compute-budget dependence. | **Highest collision.** PR #81 cannot claim first recognition that gap, curriculum, weighting, and training speed are coupled, or say the intervention is “not merely discretization.” | The defensible distinction is the exact cross-spacing one-sided SG decomposition into detached-target displacement and explicit reweighting, exact common-state A/B/C/D identities, and the test of their finite-trajectory consequences. |
| [Truncated Consistency Models (ICLR 2025)](https://openreview.net/forum?id=ZYDEJEvCbv) | Reproduces the one-sided CT objective with $\theta^- = \operatorname{stopgrad}(\theta)$ and weight $\omega(t)/\Delta_t$, derives its parameter gradient and $\Delta_t\to0$ limit by Taylor/chain-rule calculations, and changes the trained time range plus boundary weighting to reallocate model capacity. | **Highest collision** for presenting stop-gradient differentiation, inverse-gap normalization, a chain-rule gradient derivation, or target/boundary design as new in itself. | PR #81 concerns an exact finite-spacing difference between controlled ECT cells at one common state; it neither relies on the $\Delta_t\to0$ limit nor proposes truncated-time training. |
| [Stable Consistency Tuning (arXiv:2410.18958; ICLR 2025 DeLTa workshop)](https://arxiv.org/abs/2410.18958) | Interprets consistency training as TD-style bootstrapping, identifies a smaller-step performance-ceiling versus optimization/propagation trade-off, and modifies ECT with a smoother progressive schedule and a variance-reduced target. | High for a first bootstrapping interpretation, a first spacing--stability trade-off, or a generic claim that schedules reshape convergence. | PR #81 isolates a finite pair-spacing intervention into cross-assigned target and denominator diagnostics and states what its common-state identities do and do not determine after separate training. |
| [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (2024)](https://arxiv.org/abs/2410.11081) | Avoids discrete-timestep schedules through a continuous-time formulation and introduces stabilization/weighting choices. | Medium for broad claims that finite discretization is the only way to formulate consistency training. | PR #81 is explicitly scoped to finite-spacing ECT. |
| [Improved Discretization Complexity Analysis of Consistency Models (ICML 2025)](https://proceedings.mlr.press/v267/yang25l.html) | Analyzes VE consistency-model discretization complexity under a decay stepsize. | High for a general discretization-theory novelty claim. | PR #81 does not claim a new complexity rate; it characterizes the instantaneous ECT objective and finite-training trajectories. |
| [Adaptive Discretization for Consistency Models / ADCMs (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html) | Treats $\Delta t$ as determining the cleaner-side training target, selects it from local consistency under a global-consistency constraint, and introduces an adaptive weighting function and adaptive distance metric. | **Highest collision** for adaptive gap selection, trainability/stability trade-offs, or claims that spacing control itself is new. | PR #81 does not optimize spacing or propose an adaptive controller; it uses a fixed controlled probe to expose exact common-state identities and then measures separately trained trajectories. |
| [See Further When Clear: Curriculum Consistency Model (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_See_Further_When_Clear_Curriculum_Consistency_Model_CVPR_2025_paper.html) | Treats teacher--student target construction as a curriculum and adapts distillation step size using clarity/difficulty signals. | High for a first claim that target spacing is a curriculum variable or controls training efficiency. | It studies consistency distillation and does not couple the endpoint intervention to ECT's inverse-spacing denominator or derive the common-state factorial. |
| [Target-Driven Distillation (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32820) | Selects target timesteps to improve distillation efficiency and distinguishes single- from multi-target training; its ``decoupled guidance'' concerns classifier-free guidance rather than loss-weight decoupling. | Medium for a first controlled target-endpoint-selection claim. | Its reported loss weight is constant; it does not study the ECT target $\times$ denominator intervention or one-sided-SG identities. |
| [How to Build a Consistency Model: Learning Flow Maps via Self-Distillation (NeurIPS 2025)](https://arxiv.org/abs/2505.18825) | Gives a broad flow-map/self-distillation framework based on tangent, Lagrangian/Eulerian, and semigroup characterizations. | Medium for claims that ideal flow-map or semigroup invariance is new. | Endpoint invariance in PR #81 is supporting intuition under explicit assumptions, not the headline novelty. |
| [Flow Map Matching with Stochastic Interpolants (TMLR 2025)](https://openreview.net/forum?id=cqDH0e6ak2) | Formalizes two-time flow maps, path composition, and consistency through stochastic interpolants. | High for an ideal pathwise endpoint-invariance novelty claim. | PR #81 uses asymptotic-anchor identification only as a conditional reference mapping. |
| [VCT (ICML 2025)](https://proceedings.mlr.press/v267/silvestri25a.html) | Discusses inverse-gap and gap-independent weighting regimes and their interaction with consistency training. | High for a first spacing--weight-scale interaction claim. | The general-weight proposition characterizes a fixed parameter-independent weight ratio; a second weighting regime would test that term rather than propose a new weight. |
| [Towards a Mathematical Theory for Consistency Training (AISTATS 2025)](https://proceedings.mlr.press/v258/li25c.html) | Establishes convergence/discretization guarantees for consistency training. | High for broad claims of a first mathematical theory of consistency training. | PR #81's theory is a local finite-intervention characterization, not a convergence-rate theorem. |
| [Stabilizing Consistency Training (arXiv 2026)](https://arxiv.org/abs/2601.22679) | Studies stop-gradient flow-map dynamics, fixed points, finite-batch perturbations, and training stability. | High for broad claims that stop-gradient objectives were not previously linked to dynamics. | PR #81 uses a narrower target-by-weight common-state identity and a standard coupled-SGD recurrence to state its finite-step implication boundary. |
| [Align Your Tangent (arXiv:2510.00658)](https://arxiv.org/abs/2510.00658) | Studies consistency-model training dynamics through output-time tangent alignment and reports substantial convergence acceleration. | **Highest title/narrative collision** for a first study of consistency-model ``optimization geometry'' or training dynamics. | Its geometry concerns output tangents and manifold alignment. PR #81 concerns the parameter-space common-state training field of a finite-spacing target/weight intervention and makes no first-geometry claim. |
| [Elucidating the Preconditioning in Consistency Distillation (ICLR 2025)](https://arxiv.org/abs/2502.02922) | Uses ``consistency gap'' for the discrepancy between a teacher denoiser and an optimal student denoiser, then designs analytic preconditioning and reports faster training. | **Highest terminology collision** with an undefined or domain-general use of ``consistency gap.'' | PR #81 must define its gap as temporal ECT pair spacing; it does not study the teacher--optimal-student denoiser discrepancy. |
| [Dual-End Consistency Model (arXiv:2602.10764)](https://arxiv.org/abs/2602.10764) | Selects critical sub-trajectory clusters as optimization targets and links trajectory selection to stability and flexible few-step sampling. | Medium for a first endpoint/trajectory-selection or training-stability claim. | It is a continuous-time distillation/flow-map construction, not a finite-spacing ECT target/denominator factorial. |
| [Consistent Diffusion Language Models (arXiv:2605.00161)](https://arxiv.org/abs/2605.00161) | In discrete diffusion, explicitly separates the step-size scheduler $p(\delta)$ from the weighting scheduler $w(\delta)$, uses $w(\delta)=1/\delta$, trains against a stop-gradient target network, and proves a boundary-anchored path-invariant population fixed point. | **Highest conceptual collision** for a first scheduler-versus-weighting distinction, first inverse-spacing normalization, or a domain-general path-invariance claim. | Its objective uses stochastic discrete posterior bridges. PR #81's bounded contribution is the controlled cross-assigned factorial and exact common-state identities for continuous one-sided-SG ECT, followed by the separately-trained trajectory boundary. |
| [A Guide to Training Consistency Models (public ICLR 2026 submission)](https://openreview.net/pdf?id=1SHdqm7Eaa) | Deconstructs time discretization, preconditioning, weighting, time sampling, and auxiliary tasks as interacting training modules. | **Highest narrative collision** with any broad “we disentangle the consistency training pipeline” claim. The public version is anonymous, so metadata and final status should be rechecked before citation. | Keep the paper narrowly about the exact ECT target/weight decomposition and budget-resolved failure of outcome additivity. |

## Novelty boundary after the audit

Do not claim:

- that $g$ is independent of $q$;
- that $g=1.10$ is a new schedule family or universally superior setting;
- first recognition that pair spacing/discretization affects training;
- first study of consistency-model optimization geometry or training dynamics;
- ownership of the term ``consistency gap'' without defining it as temporal
  ECT pair spacing;
- first discovery that spacing changes convergence speed or time-to-quality;
- first recognition that spacing and $1/\Delta$ weighting are coupled;
- first separation of a step-size scheduler from a weighting scheduler;
- first use of stop-gradient asymmetry in a consistency objective;
- novelty from the chain rule or from differentiating a detached-target loss;
- first disentanglement of consistency-model discretization and weighting;
- a new adaptive discretization, curriculum, or optimizer mechanism; or
- a general flow-map endpoint-invariance theorem.

The algebraic observation that moving the cleaner endpoint changes both the
detached target and an inverse-spacing weight is prior-art territory. The
chain-rule calculation that turns such an objective into a parameter gradient
is likewise not a standalone novelty claim. The contribution must be attached
to the controlled intervention and its implication boundary:

> For the evaluated one-sided stop-gradient ECT objective, we construct a
> controlled cross-assigned target $\times$ denominator factorial for a realized
> pair-spacing intervention. At a common model state and matched sample/RNG,
> its four cells satisfy exact finite-spacing training-field identities. Those
> identities delimit, rather than prescribe, the behavior of arms that
> subsequently follow separate finite-training trajectories; they do not imply
> additive or seed-invariant quality effects.

This is a three-part boundary, and all three parts should remain visible in the
paper:

1. **Object of study:** realized pair spacing, with $g$ used as a controlled
   implementation probe rather than a new schedule family.
2. **Theory object:** exact common-state identities for the one-sided-SG ECT
   target $\times$ denominator factorial at finite spacing.
3. **Empirical boundary:** separately trained arms leave the common state, so
   objective-level factorization is not an outcome-level causal decomposition.

In the actual single-stage, unclipped protocol,
$g=1.10$ is exactly the pair-construction reparameterization
$q_{\mathrm{eff}}=256/1.10\approx232.727$ in real arithmetic. A matched-$q$
design is a control for whether implementation and floating-point paths preserve
that equality, not evidence for a distinct mechanism. Until the frozen
pair/loss/field gates pass, no full-training equivalence claim is licensed.

## Manuscript action

The Introduction treats $g$ as a pair-spacing implementation probe at fixed
$q=256$, records its exact stage-0 $q_{\mathrm{eff}}$ equivalence, and removes
the phrase “not merely a different discretization step.” The Related Work
section should credit prior inverse-gap objectives, stop-gradient gradient
derivations, spacing curricula, and scheduler/weighting distinctions before
stating the narrower common-state factorial contribution.

## Primary-source metadata checked in this update

| Work | Verified status used here | Primary source |
|---|---|---|
| Truncated Consistency Models | ICLR 2025 conference paper; arXiv:2410.14895v2 | [OpenReview paper](https://openreview.net/forum?id=ZYDEJEvCbv), [arXiv](https://arxiv.org/abs/2410.14895) |
| Stable Consistency Tuning | arXiv:2410.18958v3; also listed by ICLR as a 2025 DeLTa workshop poster | [arXiv](https://arxiv.org/abs/2410.18958), [ICLR virtual program](https://www.iclr.cc/virtual/2025/35357) |
| Adaptive Discretization for Consistency Models | NeurIPS 2025 main track, *Advances in Neural Information Processing Systems* 38 | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html) |
| Consistent Diffusion Language Models | arXiv:2605.00161v2; no proceedings venue asserted | [arXiv](https://arxiv.org/abs/2605.00161) |
| Align Your Tangent | arXiv:2510.00658v1; preprint status asserted | [arXiv](https://arxiv.org/abs/2510.00658) |
| Elucidating the Preconditioning in Consistency Distillation | arXiv:2502.02922v3; author metadata states accepted at ICLR 2025 | [arXiv](https://arxiv.org/abs/2502.02922) |
| Dual-End Consistency Model | arXiv:2602.10764v3; no proceedings venue asserted | [arXiv](https://arxiv.org/abs/2602.10764) |
