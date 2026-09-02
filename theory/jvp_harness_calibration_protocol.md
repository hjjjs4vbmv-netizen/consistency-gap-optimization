# Squared-GN JVP harness calibration protocol

Status: frozen before calibration results.

The failed PR #87 correctness cell does not identify a source of Jacobian
failure. Its finest two finite-difference estimates were directionally aligned,
but their relative magnitude change exceeded the frozen threshold. The next
question is therefore narrower: does the current parameter-space central
difference approximate the directional derivative that automatic
differentiation assigns to the same executed FP32 graph?

For the same state, arm, frozen batch, and direction, define

\[
v_{\mathrm{AD}}=(J_i-J_j)u,
\qquad
h_{\mathrm{AD}}=J_i^\top v_{\mathrm{AD}}.
\]

The oracle computes (v_{\mathrm{AD}}) by reverse-over-reverse automatic
differentiation and then computes the online-branch VJP (h_{\mathrm{AD}}).
For every preregistered binary scale (\epsilon), the finite-difference harness
computes

\[
v_\epsilon=
\frac{r(\theta+\epsilon u)-r(\theta-\epsilon u)}{2\epsilon},
\qquad
h_\epsilon=J_i^\top v_\epsilon.
\]

The primary diagnostic is the oracle-relative error of both (v_\epsilon) and
(h_\epsilon). A calibration plateau requires three consecutive frozen scales
with both errors at most five percent. We additionally record cosine, norm
ratio, and the fraction of FP32 parameter coordinates that actually change.

Automatic differentiation is an executed-graph oracle, not a proof that a
classical derivative exists at every activation boundary. Agreement establishes
only that the finite difference resolves the tangent selected by the current
graph. Disagreement can reflect truncation, quantization, or crossings of local
nonsmooth boundaries; this one-cell calibration does not distinguish those
causes globally.

Regardless of the result, the calibration cannot retroactively change the old
gate. A passing plateau permits only a separately frozen v2 correctness
protocol. It does not admit the old factorial, explain FID, or establish a
schedule mechanism.
