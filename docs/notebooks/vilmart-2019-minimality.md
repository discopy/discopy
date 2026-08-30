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

Figure 1 of arXiv:1812.09114v1 has nine displayed equation schemas when the
two identity equations are counted separately:

| Label | Equation schema |
|---|---|
| (S) | \(Z^a_{m,k};Z^b_{k,n}=Z^{a+b}_{m,n}\), with \(m,n\geq0\), \(k\geq1\), and arbitrary orientations obtained by bending |
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

## Small grading and connectivity countermodels

Five independence questions have canonical combinatorial models.  The S and H
gradings and the E support quotient reduce to two values, CP records boundary
connectivity, and EU records an additive phase class.  The gradings are not
merely occurrence-count heuristics: each is a compact functor to the
one-object category whose endomorphisms are \(\mathbb Z/2\mathbb Z\).
Structural wires, cups, caps, and swaps map to zero.

For (S), count all red and green spiders of total degree at least four modulo
two.  Every other rule preserves this count: (H) changes the colour but not the
degree, and no other displayed rule contains a spider of degree four.  Fusion
of two trivalent green spiders along one leg violates it.

For (H), count only red spiders of degree at least four.  This works precisely
because Vilmart's (S) schema is green-only.  The degree-four colour-change
instance violates it.

```python {.marimo}
def high_spider_parity(diagram):
    """The Z/2-valued model which separates spider fusion."""
    return sum(
        isinstance(box, (Z, X))
        and len(box.dom) + len(box.cod) >= 4
        for box in diagram.boxes) % 2


def high_red_parity(diagram):
    """The Z/2-valued model which separates colour change."""
    return sum(
        isinstance(box, X)
        and len(box.dom) + len(box.cod) >= 4
        for box in diagram.boxes) % 2


s_counterexample = (
    Z(1, 2) >> Z(1, 2) @ Id(1),
    Z(1, 3))
h_counterexample = (
    H >> X(1, 3) >> H @ H @ H,
    Z(1, 3))

assert tuple(map(high_spider_parity, s_counterexample)) == (0, 1)
assert tuple(map(high_red_parity, h_counterexample)) == (1, 0)
```

For (CP), use finite corelations.  Send every positive-degree spider to the
indiscrete partition of its boundary, every degree-zero spider to the unique
scalar, and H to the identity corelation.  Fusion has \(k\geq1\), so its two
spiders share a component.  The identity rules are identity corelations, and
B, HD, H, and both sides of EU have one block on their external boundary;
FinCorel has only one scalar, so all boundaryless cases and E also hold.  The
left side of (CP) connects its two outputs, while its right side leaves them in
two components.  A singleton is the smallest nonempty object on which this
distinction can occur.

```python {.marimo}
from discopy.frobenius import Diagram as FrobeniusDiagram
from discopy.frobenius import Ty as FrobeniusTy

corel_object = FrobeniusTy("x")


def corel_arrow(box):
    """Map a ZX generator to its boundary-connectivity corelation."""
    if isinstance(box, (Z, X)):
        return FrobeniusDiagram.spiders(
            len(box.dom), len(box.cod), corel_object)
    if box == H:
        return FrobeniusDiagram.id(corel_object)
    if isinstance(box, Scalar):
        return FrobeniusDiagram.id(FrobeniusTy())
    raise TypeError(box)


corel_functor = Functor(
    ob_map=lambda _: corel_object,
    ar_map=corel_arrow,
    dom=type(Id(1)),
    cod=FrobeniusDiagram)


def mapped_hypergraph(diagram):
    """Return the finite-cospan hypergraph underlying the model."""
    return corel_functor(diagram).to_hypergraph()


def boundary_partition(diagram):
    """Forget internal components and canonically label boundary blocks."""
    hypergraph = mapped_hypergraph(diagram)
    left, right = hypergraph.wires[0], hypergraph.wires[2]
    names = {}

    def rename(wires):
        return tuple(
            names.setdefault(wire, len(names))
            for wire in wires)

    return rename(left), rename(right)


cp_connected = FrobeniusDiagram.spiders(
    0, 2, corel_object).to_hypergraph()
cp_separate = (
    FrobeniusDiagram.spiders(0, 1, corel_object)
    @ FrobeniusDiagram.spiders(0, 1, corel_object)
).to_hypergraph()

assert cp_connected.wires == ((), (), (0, 0))
assert cp_separate.wires == ((), (), (0, 1))
assert cp_connected != cp_separate
assert boundary_partition(copy_left) != boundary_partition(copy_right)

for _left, _right in (
        (Z(1, 1), Id(1)),
        (X(1, 1), Id(1)),
        (Z(1, 2, .25) >> Z(2, 1, -.5), Z(1, 1, -.25)),
        (bialgebra_left, bialgebra_right),
        (H, hadamard_decomposition),
        (H >> X(1, 3, .25) >> H @ H @ H, Z(1, 3, .25)),
        (Z(0, 1, .25) >> X(1, 0, -.25), Id(0)),
        euler_rule(.2, .7, -.4)):
    assert boundary_partition(_left) == boundary_partition(_right)
```

For (E), take the support quotient of finite cospans.  Every hom-set with
nonempty outer boundary is collapsed to one element, while
\(\operatorname{Hom}(0,0)=\{0,1\}\) records whether the cospan apex is empty
or nonempty.  Scalar composition and tensor are Boolean OR.  This is a
congruence: a composite closed through a nonempty middle boundary has a
nonempty pushout, while composition through the empty boundary is just OR.
It is therefore a compact quotient.  Map every positive-degree spider to the
unique arrow, every zero-legged spider to 1, H to the unique arrow, and the
empty diagram to 0.  Every scalar instance of every other rule has equal
support on both sides; (E) alone asserts \(1=0\).

```python {.marimo}
cospan_unit = FrobeniusTy()
empty_support = FrobeniusDiagram.id(cospan_unit).to_hypergraph()
nonempty_support = FrobeniusDiagram.spiders(
    0, 0, corel_object).to_hypergraph()

assert empty_support.n_spiders == 0
assert nonempty_support.n_spiders == 1


def support_value(diagram):
    """Evaluate the two-scalar finite-cospan support quotient."""
    if len(diagram.dom) + len(diagram.cod):
        return "*"
    return int(mapped_hypergraph(diagram).n_spiders > 0)


assert support_value(Z(0, 1, .25) >> X(1, 0, -.25)) == 1
assert support_value(Id(0)) == 0
assert support_value(Z(0, 2) >> Z(2, 0)) == support_value(Z(0, 0))
assert support_value(Id(1).trace()) == 1
```

Finally, remove (EU) and use the one-object compact category whose morphism
group is

\[
A=\mathbb R/(\tfrac\pi2\mathbb Z).
\]

Both tensor and composition are addition, cups, caps, swaps, and H map to
zero, and every spider maps to the class of its phase.  The phase sum of (HD)
is \(\pi/2\), hence zero in A; (E) has phase sum zero, and all remaining
rules, including degree-zero cases, preserve the grading.  In the exact EU
subfamily \((a_1,a_2,a_3)=(t,\pi,t)\), Vilmart's convention gives
\((b_1,b_2,b_3,g)=(\pi/2,0,\pi/2,t)\).  At \(t=\pi/4\), the two phase sums
differ by the nonzero class \([\pi/4]\).

```python {.marimo}
from fractions import Fraction


def phase_grade(diagram):
    """Evaluate the universal additive phase grade in half-turn units."""
    total = sum(
        (Fraction(str(box.phase)) for box in diagram.boxes
         if isinstance(box, (Z, X))),
        start=Fraction())
    return total % Fraction(1, 2)


eu_grade_left = Z(1, 1, .25) >> X(1, 1, 1) >> Z(1, 1, .25)
eu_grade_right = (
    (X(0, 1, 1) >> Z(1, 0, .25))
    @ (X(0, 3) >> Z(3, 0))
    @ (X(1, 1, .5) >> Z(1, 1) >> X(1, 1, .5)))

assert (phase_grade(eu_grade_left), phase_grade(eu_grade_right)) == (
    Fraction(), Fraction(1, 4))

for _left, _right in (
        (Z(1, 1), Id(1)),
        (X(1, 1), Id(1)),
        (Z(1, 2, .25) >> Z(2, 1, -.5), Z(1, 1, -.25)),
        (copy_left, copy_right),
        (bialgebra_left, bialgebra_right),
        (H, hadamard_decomposition),
        (H >> X(1, 3, .25) >> H @ H @ H, Z(1, 3, .25)),
        (Z(0, 1, .25) >> X(1, 0, -.25), Id(0))):
    assert phase_grade(_left) == phase_grade(_right)
```

These quotients are minimal in a precise modest sense: the three Boolean
models have the smallest possible nontrivial hom-set, and the corelation
witness uses the smallest boundary with two distinct partitions.  The full
group \(A\) is the universal colour- and arity-independent additive phase
grading with H assigned zero; (HD) imposes exactly \([\pi/2]=0\), while the
displayed EU family kills every remaining class.

## The green identity is independently necessary

There is a smallest finite-dimensional linear countermodel for \(I_g\).  On
\(V=\mathbb C^2\), let \(p=|0\rangle\langle0|\).  Interpret every
positive-degree green spider, independently of phase, as the rank-one all-zero
tensor, and every green scalar as one.  Interpret H as \(p\).  Do the same for
red spiders except that a phase-zero red spider of total degree two is the
full ambient Kronecker delta (identity, cup, or cap according to its bending).

Green fusion contracts rank-one tensors.  H fixes every active tensor and
compresses each exceptional red delta to the active delta, proving every (H)
instance.  The equations CP, B, HD, and E live entirely on the active line.
In EU, the green unary spider in each main chain compresses any exceptional
red identity; both closed scalars are one.  Thus all rules except \(I_g\)
hold for all phases and arities, while
\(Z^0_{1,1}=p\ne 1_V=X^0_{1,1}\).

There is a source-language subtlety in the executable check.  DisCoPy's
quantum.zx class implements its structural cup as the green zero-phase
two-leg spider, an identification which itself uses \(I_g\).  In this
countermodel the structural cup is instead the full Kronecker delta, while
the green two-leg tensor is \(|00\rangle\).  The symmetric functor below
therefore checks oriented rule schemas, not cups() or trace() calls.  The
compact extension is nevertheless canonical and checkable tensorially: all
assigned spider tensors and p are totally symmetric, so their mates under the
full Kronecker cup are exactly the assigned bent orientations.

```python {.marimo}
active_projection = np.diag([1, 0]).astype(complex)


def ig_countermodel_arrow(box):
    """The minimal two-dimensional tensor model separating I_g."""
    n_inputs, n_outputs = len(box.dom), len(box.cod)
    degree = n_inputs + n_outputs
    if isinstance(box, (Z, X)):
        if degree == 0:
            array = np.array(1, dtype=complex)
        elif (
                isinstance(box, X) and degree == 2
                and np.isclose(float(box.phase) % 2, 0)):
            array = np.eye(2, dtype=complex)
        else:
            array = np.zeros((2,) * degree, dtype=complex)
            array[(0,) * degree] = 1
    elif box == H:
        array = active_projection
    elif isinstance(box, Scalar):
        array = np.array(box.data, dtype=complex)
    else:
        raise TypeError(box)
    return Tensor[complex](
        array, Dim(2) ** n_inputs, Dim(2) ** n_outputs)


ig_countermodel = Functor(
    ob_map=lambda _: Dim(2), ar_map=ig_countermodel_arrow,
    dom=type(Id(1)), cod=Tensor[complex])


def ig_evaluate(diagram):
    return np.asarray(ig_countermodel(diagram).array)


def ig_close(left, right):
    return np.allclose(ig_evaluate(left), ig_evaluate(right), rtol=0)


assert not ig_close(Z(1, 1), Id(1))
assert ig_close(X(1, 1), Id(1))
assert ig_close(copy_left, copy_right)
assert ig_close(bialgebra_left, bialgebra_right)
assert ig_close(H, hadamard_decomposition)
assert ig_close(Z(0, 1, .25) >> X(1, 0, -.25), Id(0))
for _angles in ((.2, .7, -.4), (0, 0, 0), (1.1, -2.2, .3)):
    assert ig_close(*euler_rule(*_angles))
```

Dimension two is globally smallest among vector-space models over a field.
Indeed, in dimension one write \(p\) for the value of \(Z^0_{1,1}\) and
\(g\) for the green \(\pi/4\) state.  Fusion gives \(pg=g\), while (E)
makes \(g\) invertible.  Hence \(p=1\), so a one-dimensional model cannot
violate \(I_g\).

## The red identity is derivable

The apparent symmetry between the two identity equations is misleading.  The
green-only spider axiom permits the support countermodel above, but the red
identity is forced by the remaining rules.  The following proof takes place
entirely in the free compact graphical theory: it does not use a basis,
positivity, or cancellation of arbitrary points.

Put

\[
 R=X^0_{1,1},\qquad
 u=Z^0_{0,1},\quad \epsilon=Z^0_{1,0},\quad
 x=X^0_{0,1},\quad \delta=Z^0_{1,2},\quad \mu=X^0_{2,1},
\]

and put

\[
 q_a=X^a_{0,1};Z^0_{1,0},\qquad
 c=X^0_{0,3};Z^0_{3,0},\qquad
 \omega=q_\pi c.
\]

All spiders and H are symmetric under bending, as required by Only
Connectivity Matters.

1.  The degree-two (H) instance and \(I_g\) give
    \(HRH=1\).  Thus H has both a left inverse \(HR\) and a right inverse
    \(RH\); they coincide, H is invertible, and \(R=H^{-2}\).
2.  For the zero-angle EU instance, Vilmart's side condition gives
    \(b_1=b_2=b_3=g=0\).  Its equation is
    \(R=\omega R^2\).  Cancelling the now-invertible R gives
    \(\omega R=1\).
3.  This also makes \(\omega\) a genuine invertible scalar, not just an
    invertible scalar action on the qubit object.  Indeed, let
    \(v=Z^{\pi/4}_{0,1}\) and \(w=X^{-\pi/4}_{1,0}\).  Rule (E) says
    \(v;w=1\), and \(v;R;w\) is a scalar inverse for \(\omega\).  Since
    scalars commute, both factors \(q_\pi\) and c are invertible.
4.  Write \(\rho=\omega^{-1}\), so \(R=\rho\,1\).  Colour change transports
    green fusion to a *projective* red fusion law: joining two red spiders
    along k legs contributes \(\rho^k\).  Each internal pair contributes
    \(H^{-2}=R=\rho\,1\).

There are two scalar-cancellation points in what follows.  Both have explicit
diagrammatic witnesses.  Let

\[
 P_a=Z^a_{1,1},\qquad w=X^{-\pi/4}_{1,0},\qquad
 a=H;P_{\pi/4};w.
\]

The degree-one (H) instance says \(x;H=u\), so green fusion and (E) give
\(x;a=1\).  Postcomposing (CP),

\[
q(x;\delta)=x\otimes x,
\]

by \(a\otimes a\) gives

\[
q\,t=1,\qquad t=x;\delta;(a\otimes a). \tag{1}
\]

Thus q is a unit in the scalar monoid; no field cancellation has been used.

Next, projective red fusion gives
\((x\otimes1);\mu=\rho^2 1\), up to the harmless choice of which input is
drawn first.  Precompose (B) by \(x\otimes1\).  On its left, (CP) removes q
and the two red multiplications each contribute \(\rho^2\), giving
\(\rho^4\delta\).  Its right side is \(\rho^2\delta\).  Postcompose with the
green multiplication and use (S) and \(I_g\), obtaining
\(\rho^4 1_A=\rho^2 1_A\).  Rule (E) exhibits the tensor unit as a retract
of A, so equality of these scalar actions implies equality of the scalars.
Cancelling the already-invertible \(\rho\) yields

\[
\rho^2=1. \tag{2}
\]

It remains to evaluate one closed Hopf diagram.  Vilmart's own appendix gives,
from (S), (CP), and (B), the typed equation

\[
q^2(\delta;\mu)=\epsilon;x:A\longrightarrow A. \tag{3}
\]

That printed derivation uses the red identity only in its two red-unit
reductions.  Each replacement is a degree-one red spider joined once to a
degree-three one, so projective fusion evaluates it as
\(\rho X^0_{1,1}=\rho^2 1=1\).  The intervening rewrites are precisely (S),
(B), and (CP).  Thus the six-diagram proof replays unchanged after (2), and
(3) is a derivation from the rules still available here.  Take its compact
trace.  Bending the two sides gives respectively c and q, hence

\[
q^2c=q.
\]

The explicit inverse (1) permits cancellation of q, proving

\[
qc=1. \tag{4}
\]

For comparison, a finite-dimensional component calculation verifies the
same closed argument independently.  In a green classical basis set
\(K=H^{-1}=K^T\), \(K^2=\kappa I\), and

\[
T_{abc}=\sum_jK_{aj}K_{bj}K_{cj},\qquad K\mathbf1=qe.
\]

B is exactly

\[
qT_{abc}T_{abd}=\delta_{cd}T_{abc}.
\]

Since \(K e=(\kappa/q)\mathbf1\), one has
\(T_{abe}=\kappa^2\delta_{ab}/q\).  The B components with
\((a,b,c,d)=(a,a,e,e)\) first give \(\kappa^2=1\); those with
\((a,a,e,a)\) then give \(T_{aaa}=0\) for \(a\ne e\), while
\(T_{eee}=1/q\).  Therefore \(c=\sum_aT_{aaa}=1/q\).  This check does use a
split vector-space basis and field cancellation; equation (3) is the
basis-free proof.

It remains to account for phases.  Let
\(\eta=q^{-1}x=cx\) be the normalized CP point and define
\(\chi_a=\eta;P_a;\epsilon\).  Green fusion and CP give scalars
\(\chi_a\) such that

\[
\eta;P_a=\chi_a\eta,\qquad
\chi_a\chi_b=\chi_{a+b},\qquad
q_a=q\chi_a. \tag{5}
\]

The first equality follows by copying \(\eta;P_a\) and applying the green
counit to one leg; the second is spider fusion, and the third is bending plus
the degree-one (H) instance.  Explicitly, transposing the closed scalar gives
\(q_a=(u;P_a;H^{-1};\epsilon)^T=x;P_a;\epsilon=q\chi_a\).

Now apply (HD) to \(\eta\), and set \(t=\pi/2\).  Its left side is
\(\eta;H=cu\).  On the right, \(\eta;P_t=\chi_t\eta\), while (2) makes
\((\eta\otimes Z^{-t}_{0,1});\mu=cZ^{-t}_{0,1}\).  The final \(P_t\)
fuses with that state to give u.  Thus (HD) says

\[
cu=\chi_tcu.
\]

The state u is cancellable: its composite with the transpose of \(\eta\) is
\(cq=1\).  Since c is a unit, it follows that

\[
\chi_{\pi/2}=1.
\]

Equation (5) now gives \(q_\pi=q\).  Combining this with (4) yields
\(\omega=q_\pi c=1\), and step 2 finally gives

\[
X^0_{1,1}=R=1.
\]

Thus \(I_r\) is redundant, whereas \(I_g\) is not.  The projective assignment
\(H=iH_{\rm std}\), with every red degree-d spider multiplied by
\((-i)^d\), illustrates the obstruction before the scalar normalizers are
imposed: it satisfies S, \(I_g\), CP, B, HD, and H and gives \(R=-1\), but
violates both E and EU.

The typed scalar and Hopf diagrams in the derivation are represented directly
below.

```python {.marimo}
red_green_phase_scalar = X(0, 1, 1) >> Z(1, 0)
triple_edge_scalar = X(0, 3) >> Z(3, 0)
euler_zero_scalar = red_green_phase_scalar @ triple_edge_scalar
split_effect = H >> Z(1, 1, .25) >> X(1, 0, -.25)
copy_scalar_inverse = X(0, 1) >> Z(1, 2) >> split_effect @ split_effect
hopf_left = copy_scalar @ copy_scalar @ (Z(1, 2) >> X(2, 1))
hopf_right = Z(1, 0) >> X(0, 1)

assert close(copy_scalar @ triple_edge_scalar, Id(0))
assert close(euler_zero_scalar, Id(0))
assert close(*euler_rule(0, 0, 0))
assert close(X(0, 1) >> split_effect, Id(0))
assert close(copy_scalar @ copy_scalar_inverse, Id(0))
assert close(hopf_left, hopf_right)
assert close(hopf_left.trace(), hopf_right.trace())
```

## Hadamard decomposition: the doubled-colour model

Let \(W=\mathbb C^2\otimes\mathbb C^2\).  A green spider is interpreted as
the tensor product of its standard green and red interpretations; a red
spider uses the opposite order.  Canonical unshuffle permutations regroup the
two factors belonging to each boundary wire.  Interpret the H generator as
the swap of the two tensor factors.

Every H-free equation is then the tensor product of its standard semantics
and its colour swap.  The (H) rule is exactly naturality of the factor swap.
Consequently every rule other than (HD) holds, including both factors of E's
and EU's closed scalars.  The two sides of (HD), however, become respectively
the factor swap and \(H\otimes H\), which already differ on \(|00\rangle\).

```python {.marimo}
factor_swap_array = np.asarray(
    Tensor.swap(Dim(2), Dim(2)).array).reshape(4, 4)


def interleaved_product(left, right):
    """Tensor two symmetric arrays, pairing corresponding boundary legs."""
    degree = left.ndim
    if degree == 0:
        return left * right
    outer = np.multiply.outer(left, right)
    order = sum(([index, degree + index] for index in range(degree)), [])
    return outer.transpose(order).reshape((4,) * degree)


def doubled_arrow(box):
    """Vilmart's doubled-colour interpretation separating HD."""
    n_inputs, n_outputs = len(box.dom), len(box.cod)
    if isinstance(box, Z):
        array = interleaved_product(
            spider_array(box),
            spider_array(X(n_inputs, n_outputs, box.phase)))
    elif isinstance(box, X):
        array = interleaved_product(
            spider_array(box),
            spider_array(Z(n_inputs, n_outputs, box.phase)))
    elif box == H:
        array = factor_swap_array
    elif isinstance(box, Scalar):
        array = np.array(box.data, dtype=complex)
    else:
        raise TypeError(box)
    return Tensor[complex](
        array, Dim(4) ** n_inputs, Dim(4) ** n_outputs)


doubled = Functor(
    ob_map=lambda _: Dim(4), ar_map=doubled_arrow,
    dom=type(Id(1)), cod=Tensor[complex])


def doubled_evaluate(diagram):
    return np.asarray(doubled(diagram).array)


def doubled_close(left, right):
    return np.allclose(
        doubled_evaluate(left), doubled_evaluate(right), atol=1e-9, rtol=0)


assert doubled_close(Z(1, 1), Id(1))
assert doubled_close(X(1, 1), Id(1))
assert doubled_close(copy_left, copy_right)
assert doubled_close(bialgebra_left, bialgebra_right)
assert not doubled_close(H, hadamard_decomposition)
assert doubled_close(Z(0, 1, .25) >> X(1, 0, -.25), Id(0))
for _angles in ((.2, .7, -.4), (0, 0, 0), (1.1, -2.2, .3)):
    assert doubled_close(*euler_rule(*_angles))

assert np.array_equal(
    doubled_evaluate(H).reshape(4, 4)[:, 0],
    np.array([1, 0, 0, 0]))
assert np.allclose(
    doubled_evaluate(hadamard_decomposition).reshape(4, 4)[:, 0],
    np.array([.5, .5, .5, .5]))
```

This construction works over any nontrivial colour-symmetric model.  It
separates (HD) whenever the factor swap differs from the tensor square of H.
Dimension four is minimal within this tensor-square construction; no global
lower bound on unrelated tensor countermodels is claimed.

## Bialgebra is independent: a phase-central deformation

The obstruction left open in 2019 is not a consequence of the other rules.
The countermodel is a real-orthogonal deformation on \(\mathbb C^{16}\);
its structure is more useful than its size.

For an integer \(N\), index the standard basis of
\(V=\mathbb C^{2^N}\) by bit strings \(x\in\{0,1\}^N\), and put

\[
 p_a=\sum_x e^{i|x|a}|x\rangle,
 \qquad P_a=\operatorname{diag}(e^{i|x|a}),
 \qquad H_N=H^{\otimes N}.
\]

Interpret a green degree-\(k\) spider as
\(\sum_xe^{i|x|a}|x\rangle^{\otimes k}\).  Let \(Q\) be complex
orthogonal, commute with every \(P_a\), and fix every \(p_a\).  Put
\(K=QH_NQ^T\), interpret H as K, and obtain each red spider by applying K
on every leg of its green counterpart.

The apparently ad hoc conditions on Q have a complete description.  Over the
reals, commuting with all phases makes Q block diagonal by Hamming weight,
and fixing all \(p_a\) fixes the uniform vector in every block.  Hence the
full deformation group is

\[
G_N=\prod_{w=0}^N O\left(\binom{N}{w}-1\right),
\]

acting on the orthogonal complements of those uniform vectors.  The example
below is a one-plane rotation in the \(O(5)\) factor of \(G_4\).

This entire deformation family satisfies (S), both identities, (CP), (H),
(HD), and (E).  For example, \(Kp_0=2^{N/2}|0^N\rangle\), which proves
(CP), while phase commutation and fixed phase states make (HD) and (E)
orthogonal conjugates of their \(N\)-fold qubit instances.  Every matrix path
in (EU) is likewise Q-conjugate to the \(N\)-fold tensor power of the qubit
path.  Only its disconnected three-legged scalar can change.  Define

\[
 C_3(K)=\sum_{x,y}K_{xy}^3.
\]

The full continuum of (EU) instances holds exactly when

\[
 C_3(K)=2^{-N/2}.
\]

In contrast, if
\(X_{abc}=\sum_jK_{aj}K_{bj}K_{cj}\), (B) is the much stronger family

\[
 2^{N/2}X_{abc}X_{abd}=\delta_{cd}X_{abc}
 \quad\text{for every }a,b,c,d.
\]

Thus, within this phase-central family, all the non-B rules leave a large
orthogonal deformation group, and EU cuts it by only one cubic scalar
equation.

For \(N=4\), act nontrivially only on the six-dimensional weight-two block,
ordered

\[
(0011,0101,0110,1001,1010,1100).
\]

Let

\[
 f_1=(5,-1,-1,-1,-1,-1)/\sqrt{30},\qquad
 f_2=(0,4,-1,-1,-1,-1)/(2\sqrt5)
\]

and rotate their span by cosine \(c\) and positive sine
\(s=\sqrt{1-c^2}\).  Choose any root in \((-1/2,-2/5)\) of

\[
P(c)=25c^5+25c^4+25c^3-335c^2-398c-110.
\]

Such a root exists because \(P(-1/2)=93/32>0\) and
\(P(-2/5)=-702/125<0\).  Exact expansion modulo \(c^2+s^2=1\) gives

\[
C_3(K)=\frac14+\frac{(1-c)P(c)}{768}=\frac14,
\]

so every non-B axiom holds.  But

\[
X_{10,10,0}=\frac14,qquad
X_{10,10,15}
=\frac{7-6c-c^2+4\sqrt6s}{96}>0.
\]

The (B) component with inputs \((1010,1010)\) and outputs
\((0000,1111)\) is therefore positive on the left and zero on the right.
The exact polynomial reduction is checked independently by
`vilmart_2019_bialgebra_symbolic.py`; run it with
`uv run --with sympy python docs/notebooks/vilmart_2019_bialgebra_symbolic.py`.

```python {.marimo}
b_n, b_dimension = 4, 16
b_polynomial = np.array([25, 25, 25, -335, -398, -110])
b_roots = np.roots(b_polynomial)
b_cosine = next(
    root.real for root in b_roots
    if abs(root.imag) < 1e-10 and -.5 < root.real < -.4)
b_sine = math.sqrt(1 - b_cosine ** 2)

b_qubit_h = hadamard_array
b_h0 = b_qubit_h
for _index in range(3):
    b_h0 = np.kron(b_h0, b_qubit_h)

b_weights = np.array([
    index.bit_count() for index in range(b_dimension)])
b_weight_two = np.flatnonzero(b_weights == 2)
b_f1 = np.array([5, -1, -1, -1, -1, -1]) / math.sqrt(30)
b_f2 = np.array([0, 4, -1, -1, -1, -1]) / (2 * math.sqrt(5))
b_q2 = (
    np.eye(6)
    + (b_cosine - 1) * (
        np.outer(b_f1, b_f1) + np.outer(b_f2, b_f2))
    + b_sine * (
        np.outer(b_f2, b_f1) - np.outer(b_f1, b_f2)))
b_q = np.eye(b_dimension)
b_q[np.ix_(b_weight_two, b_weight_two)] = b_q2
b_k = b_q @ b_h0 @ b_q.T


def b_phase_state(angle):
    return np.exp(1j * b_weights * angle)


def b_phase_gate(angle):
    return np.diag(b_phase_state(angle))


def b_red_gate(angle):
    return b_k @ b_phase_gate(angle) @ b_k


b_u = np.ones(b_dimension, dtype=complex)
b_x = np.einsum("aj,bj,cj->abc", b_k, b_k, b_k, optimize=True)
b_cubic_scalar = np.sum(b_k ** 3)

assert np.allclose(b_q.T @ b_q, np.eye(b_dimension))
assert np.allclose(b_k.T, b_k)
assert np.allclose(b_k @ b_k, np.eye(b_dimension))
assert np.allclose(b_k @ b_u, 4 * np.eye(b_dimension)[0])
assert np.allclose(b_cubic_scalar, .25)
assert np.isclose(
    np.polyval(b_polynomial, b_cosine), 0, atol=1e-10)
```

The following evaluator uses DisCoPy tensors for the two fixed-arity rules and
ordinary matrices for the phase-parametric rules.  The random EU instances are
regression tests only; the proof for all real angles is the tensor-power and
Q-conjugacy argument above.

```python {.marimo}
b_dim = Dim(b_dimension)
b_unit = Dim(1)
b_hadamard_tensor = Tensor[complex](b_k, b_dim, b_dim)


def b_tensor_power(tensor, exponent):
    result = Tensor[complex]([1], b_unit, b_unit)
    for _factor in range(exponent):
        result = result @ tensor
    return result


def b_green(n_inputs, n_outputs, angle=0):
    degree = n_inputs + n_outputs
    if degree == 0:
        return Tensor[complex](
            [np.sum(b_phase_state(angle))], b_unit, b_unit)
    array = np.zeros((b_dimension,) * degree, dtype=complex)
    for _basis in range(b_dimension):
        array[(_basis,) * degree] = b_phase_state(angle)[_basis]
    return Tensor[complex](
        array, b_dim ** n_inputs, b_dim ** n_outputs)


def b_red(n_inputs, n_outputs, angle=0):
    return (
        b_tensor_power(b_hadamard_tensor, n_inputs)
        >> b_green(n_inputs, n_outputs, angle)
        >> b_tensor_power(b_hadamard_tensor, n_outputs))


b_copy_scalar = b_red(0, 1) >> b_green(1, 0)
assert np.allclose(
    (b_copy_scalar @ (b_red(0, 1) >> b_green(1, 2))).array,
    (b_red(0, 1) @ b_red(0, 1)).array)

b_z3 = b_green(1, 2).array
b_r3 = b_red(2, 1).array
b_bialgebra_left = (
    b_copy_scalar.array.item()
    * np.einsum(
        "axy,bzw,xzc,ywd->abcd",
        b_z3, b_z3, b_r3, b_r3, optimize=True))
b_bialgebra_right = np.einsum(
    "abx,xcd->abcd", b_r3, b_z3, optimize=True)

assert not np.allclose(b_bialgebra_left, b_bialgebra_right)
assert np.allclose(
    b_bialgebra_left[10, 10, 0, 15],
    (7 - 6 * b_cosine - b_cosine ** 2
     + 4 * math.sqrt(6) * b_sine) / 96)
assert b_bialgebra_right[10, 10, 0, 15] == 0

# E and HD.
assert np.allclose(
    b_phase_state(-math.pi / 4)
    @ b_k @ b_phase_state(math.pi / 4),
    1)
b_hd_central = np.einsum(
    "abc,c->ab", b_x, b_phase_state(-math.pi / 2))
assert np.allclose(
    b_phase_gate(math.pi / 2)
    @ b_hd_central @ b_phase_gate(math.pi / 2),
    b_k)

# EU, including its two disconnected scalar components.
b_rng = np.random.default_rng(20260826)
for _triple in b_rng.uniform(-4 * math.pi, 4 * math.pi, (100, 3)):
    _a1, _a2, _a3 = _triple
    _b1, _b2, _b3, _gamma = euler_angles(_a1, _a2, _a3)
    _left = b_phase_gate(_a3) @ b_red_gate(_a2) @ b_phase_gate(_a1)
    _right = b_red_gate(_b3) @ b_phase_gate(_b2) @ b_red_gate(_b1)
    _scalar = (
        b_cubic_scalar
        * (b_phase_state(_gamma) @ b_k @ b_phase_state(math.pi)))
    assert np.allclose(_scalar, np.exp(4j * _gamma))
    assert np.allclose(_left, _scalar * _right)
```

The example is not essentially unique.  The displayed root is simple; adding
another small phase-central rotation and applying the real implicit-function
theorem gives local continuous families on the hypersurface
\(C_3(K)=1/4\), and the strict B violation persists.

There is also a sharp lower bound inside this deformation family.  For
\(N\le2\), the group \(G_N\) contains only the relevant basis permutations.
For \(N=3\), its identity component is \(SO(2)\times SO(2)\).  If the two
block angles are \(\theta_1,\theta_2\), direct block multiplication shows that
K depends only on \(\theta=\theta_1+\theta_2\), and

\[
C_3(K)-2^{-3/2}
=\frac{\sqrt2}{3}(\cos\theta-1)(2\cos\theta+1)^2
=-\frac{2\sqrt2}{3}\sin^2(3\theta/2).
\]

Thus EU permits only \(\theta\in(2\pi/3)\mathbb Z\); the three resulting K
matrices are Walsh matrices with permuted characters and all satisfy B.
Consequently \(N=4\), or dimension 16, is minimal in the connected real
phase-central tensor-power family.  This is not a global lower bound:
unrelated countermodels in dimensions 3 through 15 are not excluded.

## Failed approaches and what they exposed

The first model search multiplied every generator by a central scalar.  In
additive notation, a degree-affine deformation compatible with one-wire green
fusion has the form

\[
w(Z^a_d)=u(d-2)+f(a),\qquad
w(X^a_d)=(u-h)d-2u+f(a),\qquad w(H)=h,
\]

where \(d=m+n\), and colour change gives the second formula.  Vilmart's
multiwire form of (S) additionally imposes \(2u=0\).  The copy rule is
automatic.  Direct counting gives

\[
\operatorname{res}(I_r)=-2h,
\qquad
\operatorname{res}(B)=-4h=2\operatorname{res}(I_r).
\]

The tempting 2017 twist (w(X_d)=d, w(H)=-1\pmod4) therefore makes (B)
hold while (I_r) fails, but it violates Vilmart's scalar rule (E).
In fact, after \(2u=0\), the residual of (E) is \(-h\), so E kills the
projective H scale immediately.  Additive phase characters cannot repair it.
This proved that neither unresolved rule could be settled by a central
degree/phase grading.

Several less trivial searches were also informative.

- In a standard copy basis, write green phases as diagonal characters and
  let a symmetric H relate the two colours.  CP forces H applied to the
  all-ones vector to be supported on one basis vector.  In dimension two this
  determines the usual Hadamard up to the projective scale above.  Searches in
  dimensions three through six found no solution of E and HD beyond the
  qubit block.  This rigidity motivated moving phases through degenerate
  eigenspaces rather than altering their characters.
- Adding a tensor-direct junk summand first seemed capable of separating both
  identities.  It works for \(I_g\), because (S) is green-only.  Reversing
  the construction cannot work in finite-dimensional vector spaces: the
  degree-two (H) equation with \(I_g\) makes H and the red unary spider
  invertible.  The zero-angle EU instance then reduces every possible failure
  of \(I_r\) to a scalar, leading to the derivation above.
- Phase-trivial relational and one-dimensional semiring models were too
  rigid.  CP makes the red zero state a singleton classical point, while the
  all-leg H equation cannot map a smaller support onto every green diagonal.
  In a one-object commutative monoid, E makes its two factors units and forces
  H's residual to be the unit.
- Higher-spin and symmetric-power representations make Euler conversion look
  automatic, but their binomial coefficients violate multiwire special spider
  fusion.  Plain tensor powers retain fusion and every rule, but also retain
  B because their H is the Fourier transform of \((\mathbb Z/2)^N\).
- For B, random orthogonal searches inside Hamming-weight eigenspaces first
  found only Walsh permutations at \(N=3\).  The exact two-block calculation
  above then explained this rigidity.  At \(N=4\), numerical roots exposed
  the quintic; exact symbolic reduction revealed that EU imposes only the
  single cubic constraint \(C_3(K)=1/4\), explaining both the counterexample
  and its continuous generalizations.
- The cleanest near-countermodel for \(I_r\) is projective:
  \(H=iH_{\rm std}\) and a red degree-d spider scaled by \((-i)^d\).
  It satisfies S, \(I_g\), CP, B, HD, and H, but has
  \(X^0_{1,1}=-1\).  E evaluates to \(-i\), and EU differs by a sign.  This
  experiment suggested the projective reduction used in the proof instead of
  an independence model.

```python {.marimo}
def projective_arrow(box):
    """The near-countermodel which isolates the projective obstruction."""
    n_inputs, n_outputs = len(box.dom), len(box.cod)
    degree = n_inputs + n_outputs
    if isinstance(box, Z):
        array = spider_array(box)
    elif isinstance(box, X):
        array = (-1j) ** degree * spider_array(box)
    elif box == H:
        array = 1j * hadamard_array
    elif isinstance(box, Scalar):
        array = np.array(box.data, dtype=complex)
    else:
        raise TypeError(box)
    return Tensor[complex](
        array, Dim(2) ** n_inputs, Dim(2) ** n_outputs)


projective = Functor(
    ob_map=lambda _: Dim(2), ar_map=projective_arrow,
    dom=type(Id(1)), cod=Tensor[complex])


def projective_evaluate(diagram):
    return np.asarray(projective(diagram).array)


def projective_close(left, right):
    return np.allclose(
        projective_evaluate(left), projective_evaluate(right), rtol=0)


assert projective_close(Z(1, 1), Id(1))
assert not projective_close(X(1, 1), Id(1))
assert projective_close(copy_left, copy_right)
assert projective_close(bialgebra_left, bialgebra_right)
assert projective_close(H, hadamard_decomposition)
assert not projective_close(
    Z(0, 1, .25) >> X(1, 0, -.25), Id(0))
assert not projective_close(*euler_rule(0, 0, 0))
```

## Result and minimality statement

The investigation gives the following status for every displayed equation.

| Equation | Status | Separating model or derivation |
|---|---|---|
| S | independent | degree-at-least-four parity |
| \(I_g\) | independent | minimal two-dimensional support model |
| \(I_r\) | derivable | projective-fusion and Hopf derivation above |
| CP | independent | finite corelations |
| B | independent | 16-dimensional phase-central orthogonal deformation |
| HD | independent | doubled-colour tensor model |
| H | independent | high-degree red parity |
| E | independent | two-scalar support-cospan quotient |
| EU | independent | total phase in \(\mathbb R/(\pi/2\mathbb Z)\) |

Consequently Vilmart's presentation is not minimal as printed: \(I_r\) can be
deleted.  After deleting it, every remaining displayed equation has a model
of all the others which violates that equation, so the resulting eight-rule
presentation is minimal.  Here “rule” counts \(I_g\) and \(I_r\) separately
and treats Only Connectivity Matters as the ambient compact graphical theory,
as Vilmart's own minimality discussion does.

## Post hoc comparison: an independent rediscovery

This section was added on 30 August 2026, after the investigation above had
been completed, committed, and opened as pull request
[#688](https://github.com/discopy/discopy/pull/688).  The discovery phase was
deliberately restricted to Vilmart's 2019 arXiv v1 and sources available
before 2020.  Only afterwards did we read Stoltz and Vilmart's
[*Minimality of the Pure Qubit ZX Calculus*](https://arxiv.org/abs/2608.14872)
and the separate DisCoPy reproduction in pull request
[#626](https://github.com/discopy/discopy/pull/626).  Thus the result above is
an independent rediscovery in the sense of information provenance; this is
not a claim of chronological priority over Stoltz and Vilmart, whose preprint
was already public.

For the presentation studied here, the conclusions coincide exactly.
Stoltz and Vilmart call the eight-rule calculus obtained by deleting \(I_r\)
\(\mathrm{ZX}_{\mathrm{opt}}\), and prove that it is complete and minimal.
They additionally establish a second minimal seven-rule presentation
\(\mathrm{ZX}'_{\mathrm{opt}}\), involving \(IV\) and \(EU'\), which is not
analysed in this notebook.

### The bialgebra countermodels are finite and infinitesimal versions of one idea

Stoltz and Vilmart give the simpler countermodel if arbitrary coefficient
rings are allowed.  Their wire is the rank-eight module

\[
 \mathbb D^8,\qquad \mathbb D=\mathbb C[\varepsilon]/(\varepsilon^2),
\]

and their Hadamard is

\[
 H'=(I+\varepsilon N)^{-1}H_3(I+\varepsilon N),
 \qquad H_3=H^{\otimes3},
\]

where \(N\) is a sparse skew three-cycle on the weight-one basis states.
The bialgebra already fails to first order, with witness coefficient
\(-\sqrt2\varepsilon/4\), while every other rule remains valid.

This is the tangent version of the phase-central deformation constructed
above.  Indeed, at \(N=3\) our finite Euler obstruction is

\[
 C_3(K)-2^{-3/2}
 =-\frac{2\sqrt2}{3}\sin^2(3\theta/2)=O(\theta^2),
\]

whereas the bialgebra defect has a nonzero linear term.  Substituting an
infinitesimal \(\theta\) proportional to \(\varepsilon\) kills the Euler
obstruction because \(\varepsilon^2=0\), without killing the bialgebra
defect.  In ordinary complex vector spaces that quadratic obstruction must
instead vanish at an actual finite point.  In the connected real \(N=3\)
family classified above, the exact Euler solutions are Walsh points and all
satisfy (B); the first countermodel in that family occurs at \(N=4\).  The
construction above achieves this on \(\mathbb C^{16}\) by solving the exact
cubic-scalar constraint.  This explains both the close structural agreement
and the extra algebraic complexity of our original witness.

The sizes are therefore not directly comparable.  Stoltz and Vilmart use
rank eight over a nonreduced ring; after forgetting the module structure,
\(\mathbb D^8\) has complex dimension sixteen, but its tensor product and
monoidal unit are still those of \(\mathbb D\)-modules.  Our model is an
ordinary finite-dimensional \(\mathbb C\)-linear model with no nilpotent
scalars.

Pull request #626 makes the infinitesimal structure explicit by evaluating
diagrams and their first derivatives.  It further computes the effective
spaces of admissible bialgebra-breaking tangent directions to have dimensions
\(0,0,1,12\) for one through four qubits.  These are quotient dimensions of
tangent parameters, not wire dimensions.  In our language its admissibility
conditions are precisely the Lie-algebra versions of commuting with all phase
gates and fixing every uniform weight vector.

### The two derivations of the red identity track the same scalar anomaly

Both proofs isolate the failure of naive red fusion before proving that its
scalar residue is trivial.  With

\[
 R=X^0_{1,1},\qquad q_a=X^a_{0,1};Z^0_{1,0},\qquad
 c=X^0_{0,3};Z^0_{3,0},
\]

Stoltz and Vilmart's residual scalar \(cq_\pi\) is our \(\omega\).  Their
scalar-cleanup lemma \(cq_0=1\) is the diagrammatic counterpart of our traced
Hopf calculation \(qc=1\), and their phase-deletion lemmas play the role of
our character calculation \(\chi_{\pi/2}=1\).  Both routes conclude that
\(cq_\pi=1\), hence \(R=1\).

The presentation is different.  Their Appendix A gives a longer, completely
diagrammatic chain of named rewrites, making every scalar movement visible.
Our proof packages the same obstruction into invertibility, projective red
fusion, a typed Hopf trace, and a phase character.  It is shorter and exposes
the algebraic mechanism, but refers back to Vilmart's printed Hopf derivation
for one intermediate equation.  Pull request #626 transcribes the endpoints
of the appendix lemmas and checks their standard semantics; the syntactic
rewrite derivations themselves remain the argument in Stoltz and Vilmart's
paper.

### Scope beyond the common theorem

Stoltz and Vilmart introduce the two missing independence models, for \(B\)
and \(I_g\), and use Vilmart's earlier results for the other six rules.  This
notebook instead supplies an explicit separating model for every surviving
rule.  Their Boolean-relational \(I_g\) model and our two-dimensional linear
support model have the same support pattern; our additional result is the
one-dimensional impossibility statement for field-valued tensor models.
Conversely, their treatment is broader because it proves minimality of both
\(\mathrm{ZX}_{\mathrm{opt}}\) and \(\mathrm{ZX}'_{\mathrm{opt}}\).

The independent proofs therefore meet at the same theorem but contribute
different refinements: Stoltz and Vilmart give a clean rank-eight
infinitesimal bialgebra witness and fully graphical identity proofs, while
this investigation gives ordinary-complex finite deformations, countermodels
for the complete rule table, and lower-dimensional classifications inside
several model families.

## Reproduction

From the repository root, the executable claims are checked with:

1. `uv run python docs/export_notebooks.py --check vilmart-2019-minimality`
2. `uv run --with sympy python docs/notebooks/vilmart_2019_bialgebra_symbolic.py`
3. `uv run pflake8 discopy docs/notebooks/vilmart_2019_bialgebra_symbolic.py`
4. `uv run coverage run -m pytest --skip-extra`

The first command executes every notebook cell, including the DisCoPy
functors and all numerical witnesses.  The second independently checks the
exact quintic reduction, the violating B component, simplicity of the root,
and the complete connected N=3 calculation.

The final verification on 29 August 2026 gave 680 passed and 50 skipped for
the documented no-extras test suite.  The nominal unskipped suite stopped at
collection because the checkout lacks the optional `pennylane` and `pytket`
packages; these failures are unrelated to the files in this investigation.
