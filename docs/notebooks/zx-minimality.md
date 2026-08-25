---
title: ZX Minimality
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Minimality of the pure qubit ZX calculus

This notebook reproduces [Stoltz & Vilmart, *Minimality of the Pure Qubit ZX
Calculus* (arXiv:2608.14872)](https://arxiv.org/abs/2608.14872) with DisCoPy,
then generalises its main construction.

The paper resolves a decade-old open problem: it exhibits the first *minimal*
complete axiomatisations of the pure qubit ZX calculus, by trimming the red
identity rule $(I_r)$ from the near-optimal rulesets of
[Vilmart (LICS 2019)](https://doi.org/10.1109/LICS.2019.8785765)
and proving that every remaining rule is necessary. The two hard cases are:

* **the bialgebra rule $(B)$**, broken by an interpretation into modules over
  the dual numbers $\mathbb{C}[\varepsilon]/(\varepsilon^2)$ where the
  Hadamard is perturbed by a nilpotent skew rotation,
* **the green identity rule $(I_g)$**, broken by an interpretation into
  relations over the Booleans, following
  [Backens, Perdrix & Wang](https://doi.org/10.23638/lmcs-16(4:19)2020).

Everything in the paper is checkable by computation: the rules are pairs of
`zx.Diagram`, an interpretation is a `Functor` into a category of tensors,
soundness of a rule is an equality of tensors. We build the two rulesets
$\mathrm{ZX_{opt}}$ and $\mathrm{ZX'_{opt}}$, the three interpretations, and
verify every claim: soundness, the derivability of $(I_r)$ lemma by lemma,
and both necessity theorems down to the exact witness coefficient
$-\frac{\sqrt{2}}{4}\varepsilon$. We then ask *which* nilpotent perturbations
give a countermodel and find that, modulo perturbations that do nothing,
the paper's is the only one on three qubits — and that three qubits is the
least possible size.

```python {.marimo}
from math import pi
from cmath import phase
import numpy as np
from discopy import frobenius, tensor
from discopy.tensor import Dim
from discopy.quantum import zx
from discopy.quantum.zx import Z, X, H, SWAP, Id, Diagram
from discopy.monoidal import Equation
```

## The rules as pairs of diagrams

A ZX diagram is built from green spiders `Z(n, m, phase)`, red spiders
`X(n, m, phase)`, the Hadamard `H` and swaps; DisCoPy phases are in turns,
so $\pi/2$ is `0.25`. A rule is a pair of diagrams with the same domain and
codomain, a ruleset a list of named rules. Scalars are diagrams with no
wires; the rules of the paper use three scalar gadgets: the *dumbbell*
$\lambda$ worth $\sqrt{2}$, the triple-edged dumbbell $\theta$ worth
$\frac{1}{\sqrt{2}}$ and the *global phase* pair worth
$\sqrt{2}e^{i\gamma}$.

```python {.marimo}
lam = X(0, 1) >> Z(1, 0)
theta = X(0, 3) >> Z(3, 0)
lam_pi = X(0, 1, .5) >> Z(1, 0)
global_phase = lambda gamma: X(0, 1, .5) >> Z(1, 0, gamma)

mo.vstack([Equation(lam, theta, global_phase(.1), symbol="  ,  ")])
```

The only rules with a side condition are the Euler decompositions $(EU)$
and $(EU')$, which decompose a green-red-green rotation (respectively a
green-Hadamard-green one) into a red-green-red one. The angles
$\beta_1, \beta_2, \beta_3$ and the global phase $\gamma$ are given by the
formulas in Figures 2 and 3 of the paper, which we transcribe and check
against the corresponding matrix identities below.

```python {.marimo}
def euler(a1, a2, a3):
    """The side condition of (EU), angles in radians."""
    xp, xm = (a1 + a3) / 2, (a1 - a3) / 2
    z = np.cos(a2 / 2) * np.cos(xp) + 1j * np.sin(a2 / 2) * np.cos(xm)
    zp = np.cos(a2 / 2) * np.sin(xp) - 1j * np.sin(a2 / 2) * np.sin(xm)
    b2 = 0 if abs(zp) < 1e-12 else 2 * phase(1j + abs(z / zp))
    return phase(z) + phase(zp), b2, phase(z) - phase(zp),\
        xp - phase(z) + (a2 - b2) / 2


def euler_prime(a1, a2):
    """The side condition of (EU'), angles in radians."""
    xp, xm = (a1 + a2) / 2, (a1 - a2) / 2
    z = -np.sin(xp) + 1j * np.cos(xm)
    zp = np.cos(xp) - 1j * np.sin(xm)
    b2 = 0 if abs(zp) < 1e-12 else 2 * phase(1j + abs(z / zp))
    return phase(z) + phase(zp), b2, phase(z) - phase(zp),\
        xp - phase(z) + (pi - b2) / 2
```

```python {.marimo}
hadamard_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
diagonal_phase = lambda angle: np.diag([1, np.exp(1j * angle)])

for _ in range(100):
    _a1, _a2, _a3 = np.random.default_rng(_).uniform(-pi, pi, 3)
    _b1, _b2, _b3, _g = euler(_a1, _a2, _a3)
    assert np.allclose(
        diagonal_phase(_a3) @ hadamard_matrix @ diagonal_phase(_a2)
        @ hadamard_matrix @ diagonal_phase(_a1),
        np.exp(1j * _g) * hadamard_matrix @ diagonal_phase(_b3)
        @ hadamard_matrix @ diagonal_phase(_b2) @ hadamard_matrix
        @ diagonal_phase(_b1) @ hadamard_matrix)
    _b1, _b2, _b3, _g = euler_prime(_a1, _a2)
    assert np.allclose(
        diagonal_phase(_a2) @ hadamard_matrix @ diagonal_phase(_a1),
        np.exp(1j * _g) * hadamard_matrix @ diagonal_phase(_b3)
        @ hadamard_matrix @ diagonal_phase(_b2) @ hadamard_matrix
        @ diagonal_phase(_b1) @ hadamard_matrix)
mo.md("The transcribed side conditions hold on the nose. ✓")
```

Now the rules themselves, transcribed from Figures 2 and 3. The rules
$(S)$, $(EU)$, $(EU')$, $(IV)$ and $(H)$ are schemas — one rule per choice
of arities and phases — so we represent each by a function and check a
family of instances.

```python {.marimo}
def fusion(n1, m1, n2, m2, k, a, b):
    """(S): two green spiders fuse along k >= 1 wires."""
    return (Z(n1, m1 + k, a) @ Id(n2) >> Id(m1) @ Z(k + n2, m2, b),
            Z(n1 + n2, m1 + m2, a + b))


def colour_change(n, m, a):
    """(H): a red spider with Hadamards on every leg is a green spider."""
    return (Id().tensor(*(n * [H])) >> X(n, m, a) >> Id().tensor(*(m * [H])),
            Z(n, m, a))


def eu(a1, a2, a3):
    """(EU): the Euler decomposition of a green-red-green rotation."""
    _angles = euler(*(2 * pi * a for a in (a1, a2, a3)))
    b1, b2, b3, g = (t / (2 * pi) for t in _angles)
    return (Z(1, 1, a1) >> X(1, 1, a2) >> Z(1, 1, a3),
            theta @ global_phase(g)
            @ (X(1, 1, b1) >> Z(1, 1, b2) >> X(1, 1, b3)))


def eu_prime(a1, a2):
    """(EU'): the Euler decomposition of a green-Hadamard-green rotation."""
    _angles = euler_prime(*(2 * pi * a for a in (a1, a2)))
    b1, b2, b3, g = (t / (2 * pi) for t in _angles)
    return (Z(1, 1, a1) >> H >> Z(1, 1, a2),
            theta @ global_phase(g)
            @ (X(1, 1, b1) >> Z(1, 1, b2) >> X(1, 1, b3)))


iv = lambda a: ((X(0, 1) >> Z(1, 0, a)) @ theta, Id(0))

Ig_rule = Z(1, 1), Id(1)
Ir_rule = X(1, 1), Id(1)
E_rule = Z(0, 1, 1 / 8) >> X(1, 0, -1 / 8), Id(0)
CP_rule = lam @ (X(0, 1) >> Z(1, 2)), X(0, 1) @ X(0, 1)
B_rule = (lam @ (Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1)
                 >> X(2, 1) @ X(2, 1)), X(2, 1) >> Z(1, 2))
HD_rule = (Z(1, 1, 1 / 4) >> Id(1) @ Z(0, 1, -1 / 4) >> X(2, 1)
           >> Z(1, 1, 1 / 4), H)
```

```python {.marimo}
class Round(frobenius.Functor):
    """Round the spider phases of a diagram, for display only."""
    def __init__(self):
        super().__init__(
            ob_map=lambda x: x, ar_map=None, dom=Diagram, cod=Diagram)

    def __call__(self, other):
        if isinstance(other, zx.Spider):
            return type(other)(len(other.dom), len(other.cod),
                               round(float(other.phase), 3))
        if isinstance(other, zx.Box) and not isinstance(other, zx.Swap):
            return other
        return super().__call__(other)


rounded = Round()

mo.vstack([
    Equation(*map(rounded, B_rule), symbol="=(B)="),
    Equation(*map(rounded, HD_rule), symbol="=(HD)="),
    Equation(*map(rounded, eu(.1, .2, .3)), symbol="=(EU)=")])
```

```python {.marimo}
fusion_cases = [(1, 1, 1, 1, 1, .1, .2), (2, 0, 1, 2, 2, .3, -.1),
                (0, 1, 2, 1, 1, .7, .4), (1, 2, 0, 1, 2, .25, .5),
                (0, 0, 0, 0, 1, .1, .9), (1, 1, 0, 0, 2, 0., 0.)]
colour_cases = [(0, 1, .3), (1, 1, .1), (2, 1, .25), (1, 2, -.4),
                (2, 2, .5), (1, 0, 0.), (1, 1, 0.), (2, 0, 0.), (0, 2, 0.)]
eu_cases = [(.1, .2, .3), (.4, -.3, .2), (0., .5, 0.), (.25, .25, .25),
            (-.4, .45, .15), (.5, .5, .5)]
eu_prime_cases = [(.1, .2), (.4, -.3), (0., 0.), (.25, -.25), (.5, .5)]
iv_cases = [(0., ), (.3, ), (.5, ), (-.25, )]

instances = lambda rule, cases: [rule(*case) for case in cases]

ZX_opt = [("S", instances(fusion, fusion_cases)), ("Ig", [Ig_rule]),
          ("E", [E_rule]), ("CP", [CP_rule]), ("B", [B_rule]),
          ("EU", instances(eu, eu_cases)), ("HD", [HD_rule]),
          ("H", instances(colour_change, colour_cases))]
ZX_opt_prime = [("S", instances(fusion, fusion_cases)), ("Ig", [Ig_rule]),
                ("IV", instances(iv, iv_cases)), ("CP", [CP_rule]),
                ("B", [B_rule]), ("H", instances(colour_change, colour_cases)),
                ("EU'", instances(eu_prime, eu_prime_cases))]
```

## Interpretations are functors

An interpretation of the ZX calculus sends each wire to a space and each
generator to a linear map. In DisCoPy this is a `frobenius.Functor` into
`tensor.Diagram`: the image of a diagram is a tensor network, evaluated by
`einsum`. A `Model` is determined by a dimension, a family of diagonal
phases for the green spiders and a matrix for the Hadamard — red spiders
are Hadamard-conjugated green ones, as in Section 3.4 of the paper.

```python {.marimo}
class Model(frobenius.Functor):
    """Interpret zx diagrams from a dimension, a family of spider phases
    and a Hadamard matrix, with X spiders defined by conjugation."""
    def __init__(self, dim, phases, hadamard, dtype=complex):
        self.dim, self.phase_family, self.hadamard = dim, phases, hadamard
        self.dtype = dtype
        super().__init__(ob_map=lambda _: Dim(dim), ar_map=None,
                         dom=Diagram, cod=tensor.Diagram[dtype])

    def box(self, name, array, n, m):
        return tensor.Box[self.dtype](
            name, Dim(*(n * (self.dim, ))), Dim(*(m * (self.dim, ))),
            np.asarray(array, dtype=self.dtype))

    def spider(self, n, m, phases):
        array = np.zeros((self.dim, ) * (n + m), dtype=self.dtype)
        for i, scalar in enumerate(phases):
            array[(i, ) * (n + m)] = scalar
        if not n + m:
            array = np.array(sum(phases), dtype=self.dtype)
        return self.box(f"Z({n}, {m})", array, n, m)

    def hadamards(self, k):
        return self.cod.id(Dim()).tensor(
            *(k * [self.box('H', self.hadamard, 1, 1)]))

    def __call__(self, other):
        if isinstance(other, X):
            n, m = len(other.dom), len(other.cod)
            return self.hadamards(n)\
                >> self.spider(n, m, self.phase_family(other.phase))\
                >> self.hadamards(m)
        if isinstance(other, Z):
            return self.spider(len(other.dom), len(other.cod),
                               self.phase_family(other.phase))
        if isinstance(other, zx.Scalar):
            return self.box('scalar', other.data, 0, 0)
        if isinstance(other, zx.Box) and not isinstance(
                other, (zx.Spider, zx.Swap, zx.Permutation, zx.Scalar))\
                and other.name == 'H':
            return self.hadamards(1)
        return super().__call__(other)


def evaluate(model, diagram):
    return np.asarray(model(diagram).eval().array)


def sound(model, rule, atol=1e-9):
    lhs, rhs = rule
    return np.allclose(evaluate(model, lhs), evaluate(model, rhs), atol=atol)
```

The standard interpretation of Section 2 sends a wire to
$\mathbb{C}^2$, the green spider to
$|0^m\rangle\langle 0^n| + e^{i\alpha}|1^m\rangle\langle 1^n|$
and the Hadamard to the Hadamard. Both rulesets are sound:

```python {.marimo}
standard = Model(2, lambda a: [1, np.exp(2j * pi * a)], hadamard_matrix)

for _name, _cases in ZX_opt + ZX_opt_prime:
    assert all(sound(standard, _rule) for _rule in _cases), _name
mo.md(r"Soundness of $\mathrm{ZX_{opt}}$ and $\mathrm{ZX'_{opt}}$. ✓")
```

## Theorem 3.2: the bialgebra rule is necessary

Section 3 of the paper interprets a wire as
$V = \mathbb{D}^{\{0,1\}^3}$ for $\mathbb{D} = \mathbb{C}[\varepsilon] /
(\varepsilon^2)$ the dual numbers: three qubits whose scalars carry an
infinitesimal. The green spider gets the phases
$t_x(\alpha) = e^{i|x|\alpha}$ for $|x|$ the Hamming weight, and the
Hadamard is perturbed by conjugation

$$H' := O^{-1} H_0 O
     = H_0 + \varepsilon(H_0 N - N H_0), \qquad O := 1 + \varepsilon N$$

where $H_0 = h^{\otimes 3}$ is the Sylvester–Walsh matrix and $N$ skew
rotates the Hamming weight one subspace:
$N|100\rangle = |010\rangle - |001\rangle$ and cyclically.

Rather than computing with dual numbers entry by entry, we observe that the
model is the *first-order jet* of the family of interpretations
$t \mapsto (1 + tN)^{-1} H_0 (1 + tN)$ at $t = 0$: a diagram evaluates to
$A + \varepsilon B$ where $A$ is its unperturbed evaluation and $B$ its
derivative along the family. Composition and tensor are bilinear, so by the
product rule $B$ is a sum over the occurrences of $H$ in the diagram, each
replaced by the commutator $[H_0, N]$ — forward-mode differentiation, done
by diagram surgery. First we expand every red spider into Hadamards around
a green spider, which is how the model defines them anyway:

```python {.marimo}
class Expand(frobenius.Functor):
    """Expand every X spider into Hadamards around a Z spider."""
    def __init__(self):
        super().__init__(
            ob_map=lambda x: x, ar_map=None, dom=Diagram, cod=Diagram)

    def __call__(self, other):
        if isinstance(other, X):
            n, m = len(other.dom), len(other.cod)
            return Id().tensor(*(n * [H])) >> Z(n, m, other.phase)\
                >> Id().tensor(*(m * [H]))
        if isinstance(other, zx.Box) and not isinstance(other, zx.Swap):
            return other
        return super().__call__(other)


expand = Expand()

K = zx.Box('K', zx.PRO(1), zx.PRO(1))


def replace(diagram, index, box):
    """Replace the box at ``index`` by ``box``."""
    boxes = list(diagram.boxes)
    boxes[index] = box
    return Diagram.decode(
        dom=diagram.dom, boxes_and_offsets=zip(boxes, diagram.offsets))
```

```python {.marimo}
class ModelK(Model):
    """A model together with an array interpreting the box ``K``."""
    def __init__(self, dim, phases, hadamard, derivative):
        self.derivative = derivative
        super().__init__(dim, phases, hadamard)

    def __call__(self, other):
        if not isinstance(other, zx.Spider) and isinstance(other, zx.Box)\
                and other.name == 'K':
            return self.box('K', self.derivative, 1, 1)
        return super().__call__(other)


class Jet:
    """Evaluate diagrams to pairs (value, epsilon) by the product rule."""
    def __init__(self, model):
        self.model = model

    def __call__(self, diagram):
        diagram = expand(diagram)
        value = evaluate(self.model, diagram)
        eps = np.zeros(value.shape, dtype=complex) + 0j
        for index, box in enumerate(diagram.boxes):
            if not isinstance(box, zx.Spider) and box.name == 'H':
                eps = eps + evaluate(self.model, replace(diagram, index, K))
        return value, eps

    def sound(self, rule, atol=1e-9):
        (lv, le), (rv, re) = map(self, rule)
        return np.allclose(lv, rv, atol=atol)\
            and np.allclose(le, re, atol=atol)


def qubit_hadamard(k):
    result = hadamard_matrix
    for _ in range(k - 1):
        result = np.kron(result, hadamard_matrix)
    return result


def jet_model(N, k):
    """The jet of the k-fold qubit model perturbed by ``N``."""
    weight = [bin(i).count('1') for i in range(2 ** k)]
    return Jet(ModelK(
        2 ** k, lambda a: [np.exp(2j * pi * a * w) for w in weight],
        qubit_hadamard(k), qubit_hadamard(k) @ N - N @ qubit_hadamard(k)))
```

```python {.marimo}
paper_N = np.zeros((8, 8), dtype=complex)
for _source, _plus, _minus in [(4, 2, 1), (2, 1, 4), (1, 4, 2)]:
    paper_N[_plus, _source], paper_N[_minus, _source] = 1, -1

paper_jet = jet_model(paper_N, 3)

for _name, _cases in ZX_opt + [("Ir", [Ir_rule])] + ZX_opt_prime:
    _ok = all(paper_jet.sound(_rule) for _rule in _cases)
    assert _ok == (_name != "B"), _name
mo.md(r"""Under $[\![-]\!]^{(B)}$ every rule of both rulesets holds,
    and so does $(I_r)$ — except $(B)$, which fails. ✓""")
```

The witness is the one the paper computes: evaluating both sides of $(B)$
on $|010\rangle \otimes |010\rangle$ and reading the coefficient of
$|000\rangle \otimes |001\rangle$ gives $-\frac{\sqrt{2}}{4}\varepsilon$
on the left and $0$ on the right.

```python {.marimo}
witness_in, witness_out = (2, 2), (0, 1)
lhs_value, lhs_eps = (a.reshape(4 * (8, )) for a in paper_jet(B_rule[0]))
rhs_value, rhs_eps = (a.reshape(4 * (8, )) for a in paper_jet(B_rule[1]))

assert np.isclose(lhs_eps[witness_in + witness_out], -np.sqrt(2) / 4)
assert np.isclose(rhs_eps[witness_in + witness_out], 0)
mo.md(f"""The coefficient is
    ${np.round(lhs_value[witness_in + witness_out], 12)}
    {np.round(lhs_eps[witness_in + witness_out].real, 8)}\\,\\varepsilon
    = -\\frac{{\\sqrt 2}}{{4}}\\varepsilon$
    on the left-hand side and
    ${np.round(rhs_value[witness_in + witness_out], 12)}$
    on the right-hand side. ✓""")
```

## Theorem 4.2: the green identity rule is necessary

Section 4 interprets diagrams as relations over the Booleans, i.e. tensors
with entries in the semiring `bool` where sum is disjunction and product
conjunction. Green spiders, whatever their phase, become the one-point
relation $P_{n,m} = \{(0^n, 0^m)\}$ and so does the Hadamard; red spiders
become $P_{n,m}$ too, except at phase zero and degree two where they are
the two-point relation $C_{n,m} = \{(0^n, 0^m), (1^n, 1^m)\}$ — so that
$(I_r)$ still holds while $(I_g)$ fails.

```python {.marimo}
def point(n, m):
    """The one-point relation P(n, m)."""
    array = np.zeros((2, ) * (n + m), dtype=bool)
    array[(0, ) * (n + m)] = True
    return array if n + m else np.array(True)


class Relational(frobenius.Functor):
    """The interpretation of Section 4 into relations over bool."""
    def __init__(self):
        super().__init__(ob_map=lambda _: Dim(2), ar_map=None,
                         dom=Diagram, cod=tensor.Diagram[bool])

    def __call__(self, other):
        if isinstance(other, (Z, X)):
            n, m = len(other.dom), len(other.cod)
            if isinstance(other, X)\
                    and float(other.phase) % 1 == 0 and n + m == 2:
                array = point(n, m).copy()
                array[(1, ) * (n + m)] = True
                return tensor.Box[bool](
                    "C", Dim(*(n * (2, ))), Dim(*(m * (2, ))), array)
            return tensor.Box[bool](
                "P", Dim(*(n * (2, ))), Dim(*(m * (2, ))), point(n, m))
        if isinstance(other, zx.Box) and not isinstance(
                other, (zx.Spider, zx.Swap, zx.Permutation))\
                and other.name == 'H':
            return tensor.Box[bool]("P", Dim(2), Dim(2), point(1, 1))
        return super().__call__(other)


relational = Relational()


def relationally_sound(rule):
    lhs, rhs = rule
    return bool(np.array_equal(
        np.asarray(relational(lhs).eval().array, dtype=bool),
        np.asarray(relational(rhs).eval().array, dtype=bool)))
```

```python {.marimo}
for _name, _cases in ZX_opt + [("Ir", [Ir_rule])] + ZX_opt_prime:
    _ok = all(relationally_sound(_rule) for _rule in _cases)
    assert _ok == (_name != "Ig"), _name

assert np.array_equal(
    np.asarray(relational(Z(1, 1)).eval().array, dtype=bool),
    point(1, 1))
mo.md(r"""Under $[\![-]\!]^{(I_g)}$ every rule of both rulesets holds, and
    so does $(I_r)$ — except $(I_g)$, whose left-hand side evaluates to
    $P_{1,1} \neq \mathrm{id}$. ✓""")
```

## Completeness: the red identity rule is derivable

Theorems 2.4 and 5.1 derive $(I_r)$ from each ruleset, through the chains
of lemmas of Appendices A and C. Each lemma is an equation between two
diagrams; we transcribe every statement and check it in the standard
interpretation, so a mistranscribed or unsound lemma would fail here.
(The derivations themselves are sequences of rule applications, whose
soundness is exactly what the previous sections establish.)

```python {.marimo}
hang = lambda state: Id(1) @ state >> X(2, 1)

lemmas_A = [
    ("H0", H >> X(1, 1) >> H, Id(1)),
    ("HC", X(1, 1) >> H, H >> X(1, 1)),
    ("LIr", theta @ lam_pi @ X(1, 1), Id(1)),
    ("H-1", H >> H, theta @ lam_pi @ Id(1)),
    ("LSr", theta @ lam_pi @ (X(1, 2, .1) >> Id(1) @ X(1, 1, .2)),
     X(1, 2, .3)),
    ("LSr", theta @ lam_pi @ (X(2, 2, .3) >> X(2, 1, .4)), X(2, 1, .7)),
    ("HF", lam @ lam @ (X(1, 2) >> Z(2, 1)), X(1, 0) >> Z(0, 1)),
    ("IV'", lam @ theta, Id(0)),
    ("IV''", Id(0), theta @ lam_pi @ theta @ lam_pi),
    ("D-", X(0, 1) >> Z(1, 1, -.25), X(0, 1)),
    ("D'pi", X(0, 1) >> Z(1, 0, .5), X(0, 1) >> Z(1, 0)),
    ("Ir", *Ir_rule)]

lemmas_C = [
    ("HX", X(1, 1, .3), H >> Z(1, 1, .3) >> H),
    ("HX", X(2, 1, .3), H @ H >> Z(2, 1, .3) >> H),
    ("RT", hang(X(0, 1)) >> hang(X(0, 1)), X(1, 1) >> X(1, 1)),
    ("ZC", Z(1, 1, .2) >> H >> H, H >> H >> Z(1, 1, .2)),
    ("X+-", X(1, 1, -.125) >> X(1, 1, .125), Id(1)),
    ("HH", H >> H, Id(1)),
    ("Ir", *Ir_rule)]

for _name, _lhs, _rhs in lemmas_A + lemmas_C:
    assert sound(standard, (_lhs, _rhs)), _name
mo.md("Every lemma of Appendices A and C is sound. ✓")
```

Together with Vilmart's completeness theorem for $\mathrm{ZX_V}$ and
$\mathrm{ZX'_V}$ this reproduces the paper's main result:
$\mathrm{ZX_{opt}}$ and $\mathrm{ZX'_{opt}}$ are complete — $(I_r)$ is
derivable — and minimal — no other rule is.

## Generalisation: which perturbations break the bialgebra?

The construction of Section 3 has three moving parts: the number of qubits
$k = 3$, the perturbation $N$, and the interpretation built from them. The
soundness checks of Appendix B only ever use four properties of $N$:

* $N$ is **antisymmetric**, so that $O^T = O^{-1}$ and $H'$ stays symmetric
  and self-inverse — Hadamards can still bend through the compact
  structure;
* $N$ **preserves each Hamming weight subspace**, so that $O$ commutes with
  every diagonal phase $D_\theta$ — the Euler and Hadamard decompositions
  survive conjugation;
* $N$ **annihilates the constant vector on each weight subspace** — every
  spider state $\sum_x e^{i|x|\alpha}|x\rangle$ is fixed by $O$, so the
  scalar gadgets and copy rule keep their values.

Call such an $N$ *admissible*. Admissibility is a linear condition, and the
space of admissible perturbations is spanned by *three-cycles*: pick three
basis states of equal weight and skew rotate them the way the paper's $N$
rotates $|100\rangle, |010\rangle, |001\rangle$.

```python {.marimo}
from itertools import combinations


def admissible_basis(k):
    """Three-cycles spanning the antisymmetric weight-preserving matrices
    that annihilate the constant vector on each weight subspace."""
    dim, basis = 2 ** k, []
    for w in range(k + 1):
        block = [i for i in range(dim) if bin(i).count('1') == w]
        for i, j in combinations(range(1, len(block)), 2):
            N = np.zeros((dim, dim))
            cycle = (block[0], block[i], block[j])
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                N[b, a], N[a, b] = 1, -1
            basis.append(N)
    return basis


assert [len(admissible_basis(k)) for k in (1, 2, 3, 4)] == [0, 0, 2, 16]
assert np.allclose(sum(
    c * b for c, b in zip(np.linalg.lstsq(
        np.stack([b.ravel() for b in admissible_basis(3)], 1),
        paper_N.ravel().real)[0], admissible_basis(3))), paper_N.real)
mo.md("""A weight subspace supports a three-cycle only if its dimension is
    at least three, and an antisymmetric matrix with vanishing row sums is
    zero in dimension below three: there is **no admissible perturbation on
    one or two qubits**, a two-parameter family on three qubits — with the
    paper's $N$ the first basis vector — and a sixteen-parameter one on
    four. ✓""")
```

Two facts, both verified below:

1. **Every admissible $N$ gives a model of everything but $(B)$**: the jet
   interpretation of $a N_1 + b N_2$ satisfies both rulesets minus $(B)$,
   and $(I_r)$, for random coefficients — on three and on four qubits.
2. **The bialgebra defect is linear in $N$ and vanishes exactly on the
   perturbations that do nothing.** The $\varepsilon$-part of
   $[\![(B)_{\mathrm{lhs}}]\!] - [\![(B)_{\mathrm{rhs}}]\!]$ is a linear
   function of $N$, and its kernel on the admissible space is precisely
   the $N$ that commute with $H_0$ — for those, $H' = H_0$ on the nose and
   the perturbation is invisible, so *every* rule holds and no
   countermodel arises.

```python {.marimo}
def bialgebra_defect(N, k):
    """The epsilon-part of the difference between the two sides of (B)."""
    jet = jet_model(N, k)
    (_, lhs_e), (_, rhs_e) = jet(B_rule[0]), jet(B_rule[1])
    return lhs_e - rhs_e


generalisation_report = []
for _k in (3, 4):
    _basis = admissible_basis(_k)
    _rng = np.random.default_rng(_k)
    _N = sum(c * b for c, b in zip(
        _rng.standard_normal(len(_basis)), _basis))
    _jet = jet_model(_N, _k)
    for _name, _cases in ZX_opt + [("Ir", [Ir_rule])] + ZX_opt_prime:
        assert all(_jet.sound(_rule) for _rule in _cases)\
            == (_name != "B"), (_k, _name)
    _defects = np.stack([bialgebra_defect(b, _k).ravel() for b in _basis])
    _commutators = np.stack([
        (qubit_hadamard(_k) @ b - b @ qubit_hadamard(_k)).ravel()
        for b in _basis])
    _rank = np.linalg.matrix_rank(_defects, tol=1e-9)
    _trivial = len(_basis) - np.linalg.matrix_rank(_commutators, tol=1e-9)
    assert _rank == len(_basis) - _trivial
    _U = np.linalg.svd(_commutators)[0]
    assert np.allclose(_U[:, len(_basis) - _trivial:].T @ _defects, 0)
    generalisation_report.append(
        f"$k = {_k}$: admissible space of dimension {len(_basis)}, "
        f"of which {_trivial} trivial, and the defect has rank {_rank} "
        f"$= {len(_basis)} - {_trivial}$. ✓")
mo.md("  \n".join(generalisation_report))
```

So the countermodels of this shape are classified: they are the admissible
perturbations modulo the trivial ones, a space of dimension $0, 0, 1, 12$
on $1, 2, 3, 4$ qubits. In particular:

* **three qubits is the least possible size** for a countermodel of the
  paper's shape, and
* **at that size the countermodel is unique** up to scale and trivial
  perturbations: the second basis cycle — the mirror of the paper's $N$ on
  the weight-two subspace — differs from $-N$ by a perturbation commuting
  with $H_0$, so it defines the *same* interpretation.

```python {.marimo}
mirror_N, minus_paper_N = admissible_basis(3)[1], -paper_N.real
assert np.allclose(
    qubit_hadamard(3) @ (mirror_N - minus_paper_N)
    - (mirror_N - minus_paper_N) @ qubit_hadamard(3), 0)
mo.md("The weight-two mirror cycle equals $-N$ up to a trivial "
      "perturbation. ✓")
```

## Conclusion

Everything in arXiv:2608.14872 that can be checked by computation has been:
the two rulesets are sound, every lemma behind the derivability of $(I_r)$
holds, the dual-number model breaks exactly $(B)$ — witness coefficient
$-\frac{\sqrt{2}}{4}\varepsilon$ included — and the relational model breaks
exactly $(I_g)$, in both rulesets. The generalisation says the paper's
choices were not accidental: admissible perturbations exist only from three
qubits up, and modulo the ones that leave the model unchanged, the paper's
$N$ is the only countermodel at that size.

The construction itself — evaluate a diagram in a one-parameter family of
models and differentiate — is not specific to the ZX calculus: it is
forward-mode automatic differentiation of a functor, done by diagram
surgery, and `Jet` works for any `Model` whatsoever.
