---
title: Vilmart 2019 ZX minimality
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

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

The following evaluator deliberately uses a symmetric functor rather than
`zx.Diagram.eval`: it treats phased spiders as ordinary generating boxes, so
their arrays are supplied explicitly.

```python {.marimo}
import cmath
import math
import numpy as np

from discopy.quantum.zx import H, SWAP, Id, Scalar, X, Z
from discopy.symmetric import Functor
from discopy.tensor import Dim, Tensor

hadamard_array = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def z_array(n_inputs, n_outputs, phase):
    """Return the standard array of a green spider; phase is in half-turns."""
    degree = n_inputs + n_outputs
    if degree == 0:
        return np.array(1 + np.exp(1j * math.pi * phase))
    result = np.zeros((2,) * degree, dtype=complex)
    result[(0,) * degree] = 1
    result[(1,) * degree] = np.exp(1j * math.pi * phase)
    return result


def spider_array(box):
    """Return the standard array of a DisCoPy green or red spider."""
    degree = len(box.dom) + len(box.cod)
    result = z_array(len(box.dom), len(box.cod), box.phase)
    if isinstance(box, X):
        colour_change = np.array([1], dtype=complex)
        for _ in range(degree):
            colour_change = np.kron(colour_change, hadamard_array)
        result = (colour_change @ result.reshape(-1)).reshape(result.shape)
    return result


def standard_arrow(box):
    """Interpret a ZX generator as a complex tensor."""
    n_inputs, n_outputs = len(box.dom), len(box.cod)
    if isinstance(box, (Z, X)):
        array = spider_array(box)
    elif box == H:
        array = hadamard_array
    elif isinstance(box, Scalar):
        array = np.array(box.data, dtype=complex)
    else:
        raise TypeError(box)
    return Tensor[complex](
        array, Dim(2) ** n_inputs, Dim(2) ** n_outputs)


standard = Functor(
    ob_map=lambda _: Dim(2), ar_map=standard_arrow,
    dom=type(Id(1)), cod=Tensor[complex])


def evaluate(diagram):
    """Evaluate a ZX diagram under the explicit standard interpretation."""
    return np.asarray(standard(diagram).array)


def close(left, right, atol=1e-10):
    """Check an interpreted diagram equation entry by entry."""
    return np.allclose(evaluate(left), evaluate(right), atol=atol, rtol=0)
```

Here are the eight linear rule families and representative spider-fusion and
colour-change arities as typed DisCoPy diagrams.

```python {.marimo}
copy_scalar = X(0, 1) >> Z(1, 0)
copy_left = copy_scalar @ (X(0, 1) >> Z(1, 2))
copy_right = X(0, 1) @ X(0, 1)

bialgebra_square = (
    Z(1, 2) @ Z(1, 2)
    >> Id(1) @ SWAP @ Id(1)
    >> X(2, 1) @ X(2, 1))
bialgebra_left = copy_scalar @ bialgebra_square
bialgebra_right = X(2, 1) >> Z(1, 2)

hadamard_decomposition = (
    Z(1, 1, .5) @ Z(0, 1, -.5)
    >> X(2, 1)
    >> Z(1, 1, .5))

assert close(Z(1, 1), Id(1))
assert close(X(1, 1), Id(1))
assert close(copy_left, copy_right)
assert close(bialgebra_left, bialgebra_right)
assert close(H, hadamard_decomposition)
assert close(Z(0, 1, .25) >> X(1, 0, -.25), Id(0))

for n_inputs, n_outputs, middle in (
        (0, 0, 1), (1, 1, 1), (2, 3, 1), (0, 2, 3)):
    assert close(
        Z(n_inputs, middle, .37) >> Z(middle, n_outputs, -.19),
        Z(n_inputs, n_outputs, .18))

for n_inputs, n_outputs in ((0, 1), (1, 0), (1, 1), (2, 3)):
    hadamards_in = Id().tensor(*([H] * n_inputs))
    hadamards_out = Id().tensor(*([H] * n_outputs))
    assert close(
        hadamards_in >> X(n_inputs, n_outputs, .31) >> hadamards_out,
        Z(n_inputs, n_outputs, .31))
```

The nonlinear rule is checked independently from its side-condition formula.
The test includes the convention at (z'=0).

```python {.marimo}
def euler_angles(alpha_1, alpha_2, alpha_3):
    """Compute Vilmart's beta angles and global phase in radians."""
    x_plus = (alpha_1 + alpha_3) / 2
    x_minus = x_plus - alpha_3
    z = (
        math.cos(alpha_2 / 2) * math.cos(x_plus)
        + 1j * math.sin(alpha_2 / 2) * math.cos(x_minus))
    z_prime = (
        math.cos(alpha_2 / 2) * math.sin(x_plus)
        - 1j * math.sin(alpha_2 / 2) * math.sin(x_minus))
    arg = lambda value: cmath.phase(value) if abs(value) > 1e-12 else 0
    beta_1 = arg(z) + arg(z_prime)
    beta_2 = (
        0 if abs(z_prime) < 1e-12
        else 2 * arg(1j + abs(z / z_prime)))
    beta_3 = arg(z) - arg(z_prime)
    gamma = x_plus - arg(z) + (alpha_2 - beta_2) / 2
    return beta_1, beta_2, beta_3, gamma


def euler_rule(alpha_1, alpha_2, alpha_3):
    """Return the two typed DisCoPy diagrams in one EU instance."""
    beta_1, beta_2, beta_3, gamma = euler_angles(
        alpha_1, alpha_2, alpha_3)
    left = (
        Z(1, 1, alpha_1 / math.pi)
        >> X(1, 1, alpha_2 / math.pi)
        >> Z(1, 1, alpha_3 / math.pi))
    right = (
        (X(0, 1, 1) >> Z(1, 0, gamma / math.pi))
        @ (X(0, 3) >> Z(3, 0))
        @ (X(1, 1, beta_1 / math.pi)
           >> Z(1, 1, beta_2 / math.pi)
           >> X(1, 1, beta_3 / math.pi)))
    return left, right


for angles in ((.2, .7, -.4), (0, 0, 0), (1.1, -2.2, .3)):
    assert close(*euler_rule(*angles))
```

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

1. Solve finite-dimensional polynomial models first over small finite fields
   and then lift any witness to an exact ring or categorical construction.
2. Linearise the equations at the standard model.  A tangent direction that
   preserves all but one rule is a candidate infinitesimal countermodel over
   dual numbers.
3. Search noncancellative and direct-sum models to determine whether the
   identity equations can fail individually rather than only together.
