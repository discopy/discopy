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

## Small grading and connectivity countermodels

Four of the independence questions have canonical two-valued models.  These
are not merely occurrence-count heuristics: an additive invariant is a compact
functor to the one-object category whose endomorphisms are
\(\mathbb Z/2\mathbb Z\).  Structural wires, cups, caps, and swaps map to zero.

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
indiscrete partition of its boundary and send H to the identity corelation.
All other equations connect all their external ports on both sides.  The left
side of (CP) connects its two outputs, while its right side leaves them in two
components.  A singleton is the smallest nonempty object on which this
distinction can occur.

```python {.marimo}
from discopy.frobenius import Diagram as FrobeniusDiagram
from discopy.frobenius import Ty as FrobeniusTy

corel_object = FrobeniusTy("x")
cp_connected = FrobeniusDiagram.spiders(
    0, 2, corel_object).to_hypergraph()
cp_separate = (
    FrobeniusDiagram.spiders(0, 1, corel_object)
    @ FrobeniusDiagram.spiders(0, 1, corel_object)
).to_hypergraph()

assert cp_connected.wires == ((), (), (0, 0))
assert cp_separate.wires == ((), (), (0, 1))
assert cp_connected != cp_separate
```

For (E), take the support quotient of finite cospans: every hom-set with
nonempty outer boundary is collapsed to one element, while scalars retain only
whether the cospan apex is empty or nonempty.  This quotient is compact and
has exactly two scalars.  Every scalar instance of every other rule has
nonempty support on both sides; (E) alone equates nonempty support with the
empty apex.

```python {.marimo}
cospan_unit = FrobeniusTy()
empty_support = FrobeniusDiagram.id(cospan_unit).to_hypergraph()
nonempty_support = FrobeniusDiagram.spiders(
    0, 0, corel_object).to_hypergraph()

assert empty_support.n_spiders == 0
assert nonempty_support.n_spiders == 1
```

Finally, remove (EU) and grade a diagram by the sum of all its spider phases in

\[
A=\mathbb R/(\tfrac\pi2\mathbb Z).
\]

The phase sum of (HD) is \(\pi/2\), hence zero in \(A\); (E) has phase sum
zero, and all remaining rules visibly preserve the grading.  In the exact EU
subfamily \((a_1,a_2,a_3)=(t,\pi,t)\), Vilmart's convention gives
\((b_1,b_2,b_3,g)=(\pi/2,0,\pi/2,t)\).  At \(t=\pi/4\), the two phase sums
differ by the nonzero class \([\pi/4]\).

```python {.marimo}
# Work in units of pi/4, so quotienting by pi/2 is reduction modulo 2.
eu_phase_left = (1 + 4 + 1) % 2
eu_phase_right = (2 + 0 + 2 + 4 + 1) % 2
assert (eu_phase_left, eu_phase_right) == (0, 1)
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
The countermodel is real and 16-dimensional, but its structure is more useful
than its size.

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

Thus all the non-B rules leave a large orthogonal deformation group, and EU
cuts it by only one cubic scalar equation.

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
\(C_3(K)=1/4\), and the strict B violation persists.  Dimension 16 is the
smallest example found, not a global minimum.  Within the connected
tensor-power rotation search, \(N\le2\) offers only permutations and the
\(N=3\) scalar equation reduces to
\(\sqrt2(\cos\theta-1)(2\cos\theta+1)^2/3=0\); the resulting rotations are
again permutations satisfying B.  This explains the first occurrence at
\(N=4\) in this ansatz but does not exclude unrelated dimensions 3 through 15.

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
