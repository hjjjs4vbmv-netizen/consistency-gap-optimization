# Literature audit: consistency discretization, curricula, and weighting

**Audit date:** 2026-08-23
**Scope:** prior work that could collide with claims about pair spacing,
discretization, curriculum, or explicit loss weighting in consistency-model
training. Results were deduplicated by DOI or normalized title. Only primary
paper or venue sources were used.

## Findings

| Work | Relevant prior art | Collision with PR #81 | Safe distinction |
|---|---|---|---|
| [Consistency Models (ICML 2023)](https://proceedings.mlr.press/v202/song23a.html) | Formulates consistency training on neighboring discretized noise levels with an explicit schedule and weighting function. | High for any claim that pair discretization or weighting is newly recognized as important. | PR #81 studies the exact one-sided stop-gradient derivative of a particular finite-spacing ECT objective and its trained factorial trajectories. |
| [Improved Techniques for Training Consistency Models (ICLR 2024)](https://arxiv.org/html/2310.14189) | Introduces inverse-interval weighting $1/(t_{i+1}-t_i)$, changes the discretization curriculum, and studies weighting empirically. | High for claims of first separating curriculum and weighting design. | It does not provide the PR #81 matched target-endpoint/denominator gradient identities or four-arm finite-training factorization test. |
| [Consistency Models Made Easy / ECT (ICLR 2025)](https://arxiv.org/html/2406.14548) | Defines continuous pairs through $p(r\mid t,\mathrm{iters})$, uses $q$ to shrink $\Delta=t-r$, explicitly states that $1/(t-r)$ couples weighting to the mapping, and reports that fixed $q=256$ can improve quickly but converge differently from a gradual curriculum. | **Highest collision.** PR #81 cannot claim first recognition that gap, curriculum, and weighting are coupled, or say the intervention is “not merely discretization.” | The defensible novelty is the exact one-sided SG loss/gradient decomposition into detached-target and explicit-rescaling components, exact A/B/C/D identities, and the separation between objective factorization and trained outcome factorization. |
| [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (2024)](https://arxiv.org/abs/2410.11081) | Avoids discrete-timestep schedules through a continuous-time formulation and introduces stabilization/weighting choices. | Medium for broad claims that finite discretization is the only way to formulate consistency training. | PR #81 is explicitly scoped to finite-spacing ECT. |
| [Improved Discretization Complexity Analysis of Consistency Models (ICML 2025)](https://proceedings.mlr.press/v267/yang25l.html) | Analyzes VE consistency-model discretization complexity under a decay stepsize. | High for a general discretization-theory novelty claim. | PR #81 does not claim a new complexity rate; it characterizes the instantaneous ECT objective and finite-training trajectories. |
| [Adaptive Discretization for Consistency Models (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/file/84706cdfc192cd0351daf48f379847e6-Paper-Conference.pdf) | Selects discretization adaptively using local/global consistency and combines it with an adaptive weighting rule. | **Highest collision** for adaptive gap selection, trainability/stability trade-offs, or claims that spacing control itself is new. | PR #81 should not propose an adaptive controller as its novelty. Its current contribution is analytical factorization plus controlled trajectory evidence. |
| [How to Build a Consistency Model: Learning Flow Maps via Self-Distillation (NeurIPS 2025)](https://arxiv.org/abs/2505.18825) | Gives a broad flow-map/self-distillation framework based on tangent, Lagrangian/Eulerian, and semigroup characterizations. | Medium for claims that ideal flow-map or semigroup invariance is new. | Endpoint invariance in PR #81 is supporting intuition under explicit assumptions, not the headline novelty. |
| [A Guide to Training Consistency Models (public ICLR 2026 submission)](https://openreview.net/pdf?id=1SHdqm7Eaa) | Deconstructs time discretization, preconditioning, weighting, time sampling, and auxiliary tasks as interacting training modules. | **Highest narrative collision** with any broad “we disentangle the consistency training pipeline” claim. The public version is anonymous, so metadata and final status should be rechecked before citation. | Keep the paper narrowly about the exact ECT target/weight decomposition and budget-resolved failure of outcome additivity. |

## Novelty boundary after the audit

Do not claim:

- that $g$ is independent of $q$;
- that $g=1.10$ is a new schedule family or universally superior setting;
- first recognition that pair spacing/discretization affects training;
- first recognition that spacing and $1/\Delta$ weighting are coupled;
- first disentanglement of consistency-model discretization and weighting;
- a new adaptive discretization, curriculum, or optimizer mechanism; or
- a general flow-map endpoint-invariance theorem.

The bounded claim supported by PR #81 is:

> For the evaluated one-sided stop-gradient ECT objective, a controlled
> realized pair-spacing change exactly decomposes at a matched model state into
> a detached target-endpoint intervention and an explicit loss-rescaling
> intervention. The corresponding objective-gradient identities do not imply
> an additive or seed-invariant factorization after the arms enter separate
> finite-training trajectories.

## Manuscript action

The Introduction now treats $g$ as a controlled probe at fixed $q=256$, defines
the estimand as a pair-spacing intervention, and removes the phrase “not merely
a different discretization step.” The Related Work section credits the
established discretization, curriculum, and weighting literature before stating
the narrower contribution.
