---
name: non-defensive-scientific-writing
description: Write and revise machine learning and scientific papers in a claim-first, evidence-calibrated, non-defensive style. Use when drafting or revising abstracts, introductions, methods, experiments, discussions, conclusions, captions, or contribution statements. Remove unnecessary hedging, anticipatory rebuttals, apology-like language, excessive caveats, and self-undermining formulations while preserving scientific accuracy and appropriate uncertainty.
---

# Non-Defensive Scientific Writing

## Objective

Write scientific papers that are:

- claim-first;
- evidence-calibrated;
- precise;
- concise;
- confident without overclaiming;
- appropriately scoped without sounding apologetic;
- focused on what the work establishes rather than on preemptively defending it from hypothetical reviewers.

The goal is NOT to make every claim stronger.

The goal is to express each claim at the strongest level justified by the available evidence, without surrounding it with unnecessary defensive rhetoric.

---

# Core Principle

Separate:

1. **epistemic uncertainty**, which is scientifically necessary;
2. **scope qualification**, which is sometimes necessary;
3. **defensive rhetoric**, which should usually be removed.

Uncertainty must reflect evidence.

Scope must reflect the actual domain of validity.

Defensive language must not be added merely because a reviewer might object.

---

# Primary Writing Rule

Use this default structure:

> Claim → Evidence → Interpretation → Scope, if necessary.

Do NOT default to:

> Caveat → qualification → anticipated objection → weak claim → another caveat.

The reader should encounter the scientific result before encountering its limitations.

---

# Evidence-Calibrated Confidence

Before writing a claim, determine its evidence level.

## Strong evidence

Use direct formulations:

- "We find that..."
- "Our results show that..."
- "X improves Y across..."
- "X consistently outperforms..."
- "These results demonstrate..."
- "The experiments reveal..."
- "This effect persists across..."

Do not weaken strong evidence with unnecessary modifiers such as:

- "appears to"
- "seems to"
- "may potentially"
- "might perhaps"
- "to some extent"
- "we cautiously suggest"

unless the uncertainty is scientifically real.

## Moderate evidence

Use calibrated formulations:

- "Our results suggest that..."
- "The evidence supports..."
- "We observe a consistent association between..."
- "These results are consistent with..."

Do not compensate for moderate evidence by adding long defensive explanations.

## Preliminary or incomplete evidence

State the uncertainty directly and briefly:

- "These results provide preliminary evidence that..."
- "We observe this effect under the evaluated settings."
- "Whether this generalizes to X remains an open question."

Do not disguise weak evidence as certainty.

Do not replace a properly scoped claim with vague defensive prose.

---

# Anti-Defensive Writing Rules

## 1. State the positive result before its qualification

Avoid:

> Although our method is evaluated only on several architectures and we do not claim universal applicability, our results nevertheless suggest that...

Prefer:

> Our method consistently improves performance across the evaluated architectures.

If the scope matters:

> Our method consistently improves performance across the evaluated architectures. Whether the effect extends to substantially different architectures remains open.

---

## 2. Do not preemptively argue with imaginary reviewers

Avoid sentences whose main purpose is anticipating criticism that has not been raised.

Examples:

- "One might argue that..."
- "A possible criticism is..."
- "We acknowledge that a skeptical reader may..."
- "It could be questioned whether..."
- "We emphasize that we do not claim..."
- "We would like to clarify that..."
- "It is important to note that we are not suggesting..."

Delete these unless the objection corresponds to a genuine conceptual ambiguity that must be resolved for the paper to be understood.

Write the scientific position directly instead.

---

## 3. Prefer positive scope statements over negative disclaimers

Avoid:

> We do not claim that the method works for all possible distributions.

Prefer:

> We study the method under distribution families X, Y, and Z.

Avoid:

> Our goal is not to provide a complete theory of optimization dynamics.

Prefer:

> We characterize the optimization dynamics relevant to X.

Avoid:

> This result should not be interpreted as proving that X always occurs.

Prefer:

> The result establishes X under assumptions A–C.

Describe what the paper establishes rather than listing everything it does not establish.

---

## 4. Do not apologize for legitimate scope choices

Avoid:

- "Unfortunately, we cannot..."
- "Due to limited resources..."
- "We only evaluate..."
- "We merely consider..."
- "We restrict ourselves to..."
- "Admittedly..."
- "Of course, our experiments are limited to..."

Prefer neutral statements:

- "We evaluate..."
- "Our experiments cover..."
- "We consider..."
- "We focus on..."
- "The present study examines..."

A bounded experimental scope is not automatically a weakness.

---

## 5. Never weaken a claim twice

If a claim is already scoped, do not add another hedge unless it changes the scientific meaning.

Bad:

> Our results appear to suggest that the proposed mechanism may improve robustness in the evaluated settings.

Potential rewrites:

> Our results suggest that the proposed mechanism improves robustness in the evaluated settings.

or, if evidence is strong:

> The proposed mechanism improves robustness across the evaluated settings.

One epistemic qualifier is usually enough.

---

## 6. Avoid caveat stacking

Do not write constructions such as:

> While X, and although Y, we acknowledge Z; nevertheless, our results may still suggest...

This structure makes the paper sound as though it is defending itself before presenting the result.

Rewrite into separate logical statements:

> We observe X across all evaluated settings. The effect is smaller under Y. We discuss the implications for Z in Section 6.

---

## 7. Do not use limitations as transition filler

Avoid unnecessary sentences such as:

- "Despite these limitations, our approach still..."
- "Nevertheless, we believe..."
- "Even though the method is imperfect..."
- "While much remains to be explored..."
- "Although further investigation is needed..."

If the result is supported, state it directly.

Further work is almost always possible; its existence does not need to accompany every contribution.

---

## 8. Replace author belief with evidence

Avoid:

- "We believe that..."
- "We feel that..."
- "We think that..."
- "In our opinion..."
- "We hope that these results..."

Prefer:

- "The results indicate..."
- "The analysis shows..."
- "This observation motivates..."
- "These findings suggest..."

Scientific arguments should be grounded in evidence rather than author confidence.

Use "we hypothesize" only when explicitly introducing a hypothesis.

---

## 9. Avoid unnecessary importance disclaimers

Avoid:

> It is worth noting that...
> It should be emphasized that...
> Importantly, it should be noted that...

Usually state the fact directly.

Bad:

> It is important to note that the effect persists at larger scales.

Better:

> The effect persists at larger scales.

Use "Importantly" only when it genuinely communicates argumentative structure.

---

## 10. Do not turn every limitation into a threat to validity

Distinguish between:

- scope;
- limitation;
- failure mode;
- unresolved question;
- threat to validity.

Do not automatically label every untested setting a "limitation."

For example:

Bad:

> A limitation of our work is that we do not test every possible architecture.

Better:

> We evaluate ResNet, ViT, and ConvNeXt architectures.

If generalization beyond these architectures is scientifically central, mention it once in the limitations or discussion section.

---

# Lexical Audit

When revising text, actively inspect occurrences of:

- only
- merely
- simply
- unfortunately
- admittedly
- arguably
- perhaps
- possibly
- potentially
- may
- might
- could
- seems
- appears
- somewhat
- relatively
- nevertheless
- nonetheless
- despite
- although
- while
- even though
- we believe
- we acknowledge
- we do not claim
- we do not intend
- should not be interpreted
- it is important to note
- it is worth noting
- a possible concern
- one might argue
- a potential criticism
- due to limitations
- despite these limitations

Do NOT blindly delete these words.

For every occurrence ask:

> Does removing this expression change the scientifically justified level of uncertainty or scope?

If no, remove or rewrite it.

---

# Sentence-Level Revision Procedure

When revising an existing paragraph, internally classify every sentence as one of:

- CLAIM
- EVIDENCE
- INTERPRETATION
- SCOPE
- METHOD
- DEFENSE

A DEFENSE sentence is one whose primary function is to:

- anticipate hypothetical criticism;
- apologize for scope;
- deny an unnecessarily broad interpretation;
- reassure the reviewer;
- weaken an already qualified claim;
- repeat a limitation already stated elsewhere.

For every DEFENSE sentence:

1. determine whether it contains scientifically necessary information;
2. delete it if not;
3. if necessary information exists, convert it into CLAIM, EVIDENCE, or SCOPE;
4. place it where it logically belongs.

Do not output these labels unless explicitly requested.

---

# Paragraph Structure

Prefer paragraphs with a clear argumentative trajectory:

> Topic claim  
> Evidence or explanation  
> Mechanism / interpretation  
> Consequence

Avoid paragraphs structured as:

> Disclaimer  
> caveat  
> background  
> another disclaimer  
> eventual result  
> reassurance

Each paragraph should have a scientific job, not a reputational-defense job.

---

# Section-Specific Rules

## Abstract

The abstract should communicate:

1. problem;
2. gap;
3. approach;
4. principal result;
5. implication.

Do not spend abstract space defending against possible limitations.

Avoid:

> While our study is limited to X and does not aim to fully characterize Y...

unless the qualification is essential to prevent a materially false interpretation.

Prefer concrete results over cautious meta-commentary.

---

## Introduction

Present the scientific problem and contribution assertively.

Avoid making contributions sound provisional when the paper provides direct evidence.

Bad:

> We hope that our work may provide some initial insight into...

Better:

> Our analysis provides insight into...

Bad:

> Without claiming to fully resolve this question, we attempt to investigate...

Better:

> We investigate...

Contribution statements should begin with the contribution itself.

Prefer:

> We identify...
> We show...
> We derive...
> We demonstrate...
> We introduce...
> We establish...
> We systematically evaluate...

---

## Related Work

Do not use defensive comparison against prior work.

Avoid:

> Unlike prior work, which fails to...

unless the failure is directly established and relevant.

Prefer precise distinctions:

> Prior work studies X under A; we study X under B.

Do not inflate novelty by attacking adjacent literature.

Confidence should come from precise differentiation.

---

## Method

Explain design decisions through their scientific or computational rationale.

Avoid:

> Although this design may seem simplistic...

Prefer:

> We use this design to isolate X from Y.

Avoid defending methodological simplicity. Simplicity can be a deliberate experimental control.

---

## Experiments

Report results before explanations for why they may not be stronger.

Prefer:

> Method A improves accuracy by 4.2 points over baseline B.

not:

> Although the absolute improvement is not extremely large and varies somewhat across seeds, Method A still achieves a 4.2-point improvement...

If variability is scientifically material, report it quantitatively.

Prefer:

> Method A improves accuracy by 4.2 ± 0.6 points.

Quantification is better than rhetorical defensiveness.

---

## Negative Results

Do not apologize for negative or mixed results.

Use matter-of-fact language:

> The effect disappears under X.

> We do not observe a statistically distinguishable difference under Y.

> Performance degrades when Z exceeds 0.5.

Then interpret the result scientifically.

Negative results should constrain the claim, not trigger defensive prose.

---

## Discussion

Use the discussion section to interpret boundaries of the result.

Prefer:

> These results indicate that X depends on Y.

instead of:

> Although this limitation means that our conclusions should be interpreted cautiously, we nevertheless believe...

---

## Limitations

Limitations should be:

- specific;
- scientifically meaningful;
- non-redundant;
- proportional to their actual importance.

A limitation section is NOT a place to list every conceivable extension of the work.

For each limitation state:

1. what is not established;
2. why it matters;
3. what evidence would resolve it.

Example:

> We evaluate models up to 7B parameters; the scaling behavior beyond this regime remains untested. Experiments at larger scales would determine whether the observed trend persists.

Avoid:

> We acknowledge that our work has several important limitations. First, due to computational constraints, we are unfortunately unable to...

---

# Claim–Evidence Alignment

Non-defensive writing MUST NOT become overclaiming.

Before strengthening any sentence, verify that the repository contains evidence supporting it.

Whenever possible inspect:

- experiment outputs;
- tables;
- figures;
- logs;
- configs;
- statistical tests;
- ablations;
- theoretical statements;
- proofs.

Never strengthen:

> "suggests"

into:

> "demonstrates"

unless the evidence justifies the stronger statement.

Never strengthen:

> "in the evaluated settings"

into an unrestricted universal claim.

If evidence is insufficient, narrow the claim rather than surrounding it with defensive prose.

Prefer:

> X improves Y on datasets A–D.

over:

> X may improve Y more generally, although we cannot rule out exceptions and do not claim universal generalization.

---

# Scope Compression Rule

When qualification is necessary, express it using the shortest scientifically complete formulation.

Prefer:

> Under Assumptions 1–3, X converges to Y.

over:

> We emphasize that our convergence result should be interpreted specifically within the context of Assumptions 1–3 and should not necessarily be taken to imply convergence outside these conditions.

Prefer:

> We observe this behavior for CNNs and ViTs.

over:

> While it remains unclear whether this phenomenon generalizes to every possible architecture, our current experiments suggest that it may hold for the CNNs and ViTs considered here.

---

# Reviewer-Awareness Rule

Write for reviewers, but do not write *to* an imaginary reviewer.

A paper should make objections difficult by being:

- precise;
- well-evidenced;
- reproducible;
- appropriately scoped.

It should NOT make objections difficult by repeatedly asserting that the authors are aware of every possible weakness.

Do not include sentences whose implicit message is:

> "Please do not reject us for this."

Replace them with scientific information.

---

# Before / After Examples

## Example 1

Defensive:

> While our experiments are necessarily limited in scope, and we do not claim that the proposed phenomenon universally applies to all neural networks, our results nevertheless suggest that it may occur across several architectures.

Preferred:

> We observe the phenomenon consistently across the evaluated architectures.

If scope is important:

> We observe the phenomenon consistently across the evaluated architectures; its behavior in substantially different model families remains open.

---

## Example 2

Defensive:

> Although our method does not outperform the baseline in every single setting, it nevertheless achieves relatively strong performance overall.

Preferred:

> Our method outperforms the baseline in 14 of 18 settings, with the largest gains occurring under high distribution shift.

---

## Example 3

Defensive:

> We acknowledge that the number of random seeds is somewhat limited due to computational constraints.

Preferred:

> Results are averaged over three random seeds.

If statistical reliability is a real issue:

> Results are averaged over three random seeds; increasing the number of seeds would provide tighter uncertainty estimates.

---

## Example 4

Defensive:

> We do not intend to claim that simplicity bias is the sole explanation for the observed phenomenon.

Preferred:

> Our experiments isolate simplicity bias as one mechanism contributing to the observed phenomenon.

---

## Example 5

Defensive:

> Despite the simplicity of this benchmark, we believe that it can still provide meaningful insight into shortcut learning.

Preferred:

> The benchmark deliberately isolates shortcut complexity, enabling controlled measurement of shortcut preference.

---

## Example 6

Defensive:

> Although these results are preliminary and further investigation is certainly required, they may potentially indicate that...

Preferred:

> These preliminary results suggest that...

---

# Revision Workflow

When asked to revise paper text:

## Step 1 — Recover scientific intent

Determine:

- What is the claim?
- What evidence supports it?
- What is the actual scope?
- What uncertainty is real?

Do not edit rhetoric before understanding these four items.

## Step 2 — Identify defensive language

Search for:

- anticipatory rebuttal;
- unnecessary caveats;
- apology;
- self-undermining wording;
- redundant uncertainty;
- author-belief statements;
- negative formulations of scope.

## Step 3 — Remove defense

Delete rhetoric that does not alter scientific content.

## Step 4 — Calibrate the claim

Set claim strength according to evidence.

Use:

strong evidence → direct claim

moderate evidence → "suggests" / scoped claim

weak evidence → preliminary claim or explicit uncertainty

## Step 5 — Reorder

Prefer:

Claim → Evidence → Interpretation → Scope

## Step 6 — Compress caveats

Move secondary caveats to the discussion or limitations section when appropriate.

Do not repeat the same caveat across Abstract, Introduction, Experiments, Discussion, and Conclusion.

## Step 7 — Verify fidelity

Ensure that revision has NOT:

- invented evidence;
- broadened experimental scope;
- changed statistical meaning;
- converted correlation into causation;
- converted observation into proof;
- removed a necessary assumption;
- concealed a genuine limitation.

---

# Interaction With Repository Evidence

When working inside a paper repository:

1. Treat experimental outputs and source files as ground truth for empirical claims.
2. Verify important numerical claims against repository evidence whenever feasible.
3. Preserve exact experimental scope.
4. Do not invent stronger results for rhetorical confidence.
5. If manuscript wording is substantially weaker than the evidence, strengthen the wording.
6. If manuscript wording is stronger than the evidence, narrow the claim.
7. Prefer changing claim scope over adding defensive paragraphs.

---

# Output Behavior

Unless the user asks for commentary:

- return polished scientific prose;
- do not explain every hedge that was removed;
- do not insert meta-comments about being "more confident";
- preserve LaTeX commands, citations, labels, equations, and references;
- preserve technical terminology;
- maintain the logical meaning of mathematical statements.

When major claim strength changes are necessary, briefly flag them separately after the revision.

---

# Final Defensive-Writing Audit

Before completing any scientific writing task, silently check:

### Claims

- Does each major paragraph state its scientific point directly?
- Is every claim as strong as the evidence permits?
- Is any claim stronger than the evidence permits?

### Hedging

- Is there more than one hedge attached to the same proposition?
- Can "may potentially suggest" become "suggests"?
- Can a scoped direct statement replace a vague hedge?

### Scope

- Is scope expressed positively where possible?
- Are we saying what was studied rather than apologizing for what was not studied?

### Reviewer anxiety

- Are there sentences answering criticisms nobody has raised?
- Are there statements whose real purpose is "please do not reject this paper"?
- Are limitations repeated across multiple sections?

### Evidence

- Could quantitative evidence replace rhetorical qualification?
- Are variability, uncertainty, and failure modes stated numerically where possible?

### Style

- Does the paper sound like authors reporting scientific findings rather than defendants answering accusations?
- Would deleting a caveat change the actual scientific meaning? If not, delete it.

---

# Hard Constraints

Never use this skill to:

- hide genuine limitations;
- exaggerate novelty;
- fabricate experimental support;
- imply causality from correlation;
- claim generality beyond tested or proven settings;
- suppress negative results that materially affect the conclusion;
- convert hypotheses into established findings.

Confidence must come from precision and evidence, not exaggeration.

The desired style is:

> assertive about evidence,
> precise about scope,
> neutral about limitations,
> and uninterested in imaginary objections.
