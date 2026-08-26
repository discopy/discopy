# Independence of Vilmart's pure-qubit ZX axioms

This is a research log for an independent investigation of the presentation in
Renaud Vilmart, *A Near-Optimal Axiomatisation of ZX-Calculus for Pure Qubit
Quantum Mechanics*, arXiv:1812.09114v1 (source dated 24 December 2018 and
published in 2019).  The source boundary is the end of 2019.  In particular,
this investigation does not inspect or use later proposed solutions, including
the separate post-2019 work already visible in the repository's remote refs.

The only paper used initially besides Vilmart is the version available to him:
M. Backens, S. Perdrix, and Q. Wang, *Towards a Minimal Stabilizer
ZX-calculus*, arXiv:1709.08903v1 (27 September 2017).  Later revisions are not
used.

## The presentation under test

Write (Z^a_{m,n}) and (X^a_{m,n}) for green and red spiders with
(m) inputs, (n) outputs, and phase (a) in radians.  Write (H) for
Hadamard, juxtaposition for tensor, and `;` for composition from top to
bottom.  Define the scalar

\[
q = X^0_{0,1};Z^0_{1,0}.
\]

Figure 1 of arXiv:1812.09114v1 has ten equations under nine labels:

| Label | Equation schema |
|---|---|
| (S) | (Z^a_{m,k};Z^b_{k,n}=Z^{a+b}_{m,n}), for the displayed spider-fusion arities |
| (I_g) | (Z^0_{1,1}=1_1) |
| (I_r) | (X^0_{1,1}=1_1) |
| (CP) | (q\,(X^0_{0,1};Z^0_{1,2})=X^0_{0,1}X^0_{0,1}) |
| (B) | (q\,(Z^0_{1,2}Z^0_{1,2});(1\,\sigma\,1);(X^0_{2,1}X^0_{2,1})=X^0_{2,1};Z^0_{1,2}) |
| (HD) | (H=(Z^{\pi/2}_{1,1}Z^{-\pi/2}_{0,1});X^0_{2,1};Z^{\pi/2}_{1,1}) |
| (H) | (H^{\otimes m};X^a_{m,n};H^{\otimes n}=Z^a_{m,n}) |
| (E) | (Z^{\pi/4}_{0,1};X^{-\pi/4}_{1,0}=1_0) |
| (EU) | (Z^{a_1}_{1,1};X^{a_2}_{1,1};Z^{a_3}_{1,1}=s_g\,(X^{b_1}_{1,1};Z^{b_2}_{1,1};X^{b_3}_{1,1})) with Vilmart's nonlinear side condition |

Here \(\sigma\) is the wire swap and

\[
s_g=(X^\pi_{0,1};Z^g_{1,0})
    (X^0_{0,3};Z^0_{3,0}).
\]

Under the standard qubit interpretation, (q=\sqrt2) and (s_g=e^{ig}).
DisCoPy records phases in half-turns, so a spider with radian phase (a)
is `Z(m, n, a / pi)` or `X(m, n, a / pi)`.

Vilmart's own minimality section claims independence for (S), (CP),
(HD), (H), (E), and (EU), and proves only that at least one of
(I_g,I_r) is needed.  It explicitly leaves (B) and the remaining identity
question open.  Those claims are hypotheses here until their proposed models
have been checked against every other schema.

## 2026-08-26: source reconstruction

The arXiv v1 source was inspected, not a later rendering.  The diagrams in the
table above were reconstructed from the TikZ node and edge data and checked
visually against page 4 of the v1 PDF.  The nonlinear side condition implies,
as a determinant check,

\[
a_1+a_2+a_3=b_1+b_2+b_3+2g \pmod {2\pi}.
\]

The 2017 stabilizer paper explains the unresolved pair categorically: its
analogue of the second identity says that the compact structures induced by
the two observable structures coincide.  Its degree-twist interpretation
violates the bialgebra and that compact-structure equation together, so it
does not settle either one separately.

## First failed approach: scalar gradings

The first model search multiplied every generator by a central scalar.  In
additive notation, the most general cancellative degree-affine deformation
compatible with green spider fusion has the form

\[
w(Z^a_d)=u(d-2)+f(a),\qquad
w(X^a_d)=(u-h)d-2u+f(a),\qquad w(H)=h,
\]

where (d=m+n), and the colour-change rule gives the second formula.  The
copy rule is automatic.  Direct counting gives

\[
\operatorname{res}(I_r)=-2h,
\qquad
\operatorname{res}(B)=-4h=2\operatorname{res}(I_r).
\]

The tempting 2017 twist (w(X_d)=d, w(H)=-1\pmod4) therefore makes (B)
hold while (I_r) fails, but it violates Vilmart's scalar rule (E).
Allowing an additive phase character does not repair it: (E), (HD), and
the continuum of (EU) instances force the relevant residuals back to zero.
Thus a countermodel for the unresolved rules cannot be merely a nonzero
degree/phase rescaling of the standard semantics.  This failure motivates
searching for non-scalar deformations, extra summands, altered compact forms,
and noncancellative coefficient systems.

## Next experiments

1. Encode both sides of all rules as DisCoPy diagrams and evaluate them with
   an explicit tensor functor, avoiding `zx.Diagram.eval` until its current
   phase-handling issue is resolved.
2. Solve finite-dimensional polynomial models first over small finite fields
   and then lift any witness to an exact ring or categorical construction.
3. Linearise the equations at the standard model.  A tangent direction that
   preserves all but one rule is a candidate infinitesimal countermodel over
   dual numbers.
4. Search noncancellative and direct-sum models to determine whether the
   identity equations can fail individually rather than only together.
