---
title: Learning Intertwiners
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Learning the fusion rules of a topological quantum field theory

Suppose we want to learn a linear map that commutes with the action of a Hopf
algebra $H$ — an *intertwiner*. The usual approach penalises the violation
$\lVert \rho_W(h) \circ T - T \circ \rho_V(h) \rVert$ during training, so
equivariance holds only approximately. `Intertwiner.chart` takes the opposite
route: it computes an isometry $Q$ onto the space
$\mathrm{Hom}_H(V, W)$, so that every value of the parameters gives an
intertwiner by construction and every intertwiner arises this way.

This power comes with a warning attached. The number of parameters of the
chart of $\mathrm{Hom}_H(V_a \otimes V_b, V_c)$ **is** the fusion multiplicity
$N_{ab}^c$ of the underlying anyon theory: whoever builds the chart has already
computed the fusion rules, before seeing any data. An experiment that
"learns" a fusion map from examples generated inside the chart would only be
solving a least-squares problem whose answer was baked into its own
parametrisation.

So this notebook plays the game the way experiments have to play it. The
fusion rules of a topological quantum field theory are inferred from
*braiding data*: monodromy measurements give the modular S-matrix, and the
Verlinde formula turns it into integers. The chart never enters the loop —
which frees it to serve as an independent check, and, in the last sections, to
say something precise about quantum error correction.

```python {.marimo}
import numpy as np
from discopy.tensor import Dim, Id, Diagram
from discopy.hopf import Algebra, Double, Representation, Intertwiner
```

## The toric code and its four anyons

Kitaev's toric code (Kitaev 2003) is governed by the quantum double
$D(k[\mathbb{Z}/2])$: its anyons — the vacuum $1$, the charge $e$, the flux
$m$ and their composite $\psi$ — are the four irreducible representations of
$D(k[\mathbb{Z}/2])$, labelled by a flux sector and a character.

```python {.marimo}
D2 = Double(Algebra.cyclic(2))
toric = [Representation[D2].anyon(f, c)
         for f, c in [(0, 1), (0, -1), (1, 1), (1, -1)]]
toric_labels = ["1", "e", "m", "ψ"]
assert all(a.is_module() for a in toric)
```

## The chart computes the fusion rules

Before doing any inference, here is the circularity in the open: asking the
chart for the dimension of every channel $\mathrm{Hom}(V_a \otimes V_b, V_c)$
*is* computing the fusion table. A `ValueError` means the space of
intertwiners is zero.

```python {.marimo}
def multiplicity(double, dom, cod, size):
    try:
        chart = Intertwiner[double].chart(dom, cod)
    except ValueError:
        return 0
    return chart.eval(dtype=complex).array.size // size


def fusion_from_charts(double, anyons):
    dims = [int(np.prod(a.inside)) for a in anyons]
    return np.array([[[
        multiplicity(double, a @ b, c, da * db * dc)
        for c, dc in zip(anyons, dims)]
        for b, db in zip(anyons, dims)]
        for a, da in zip(anyons, dims)])


def fusion_table(N, labels):
    lines = [
        f"{labels[a]} ⊗ {labels[b]} = " + " + ".join(
            (f"{N[a, b, c]}·" if N[a, b, c] > 1 else "") + labels[c]
            for c in range(len(labels)) if N[a, b, c])
        for a in range(len(labels)) for b in range(len(labels))]
    return "\n".join(lines)


N_charts = fusion_from_charts(D2, toric)
print(fusion_table(N_charts, toric_labels))
```

The table is the multiplication of $\mathbb{Z}/2 \times \mathbb{Z}/2$ — each
ordered pair of anyons fuses to a unique third. This is a computation, not
learning: nothing was measured, no data was seen.

## Fusion rules from braiding, exactly

An experiment cannot solve for nullspaces of an action it does not know. What
it can do is *braid*: wind one anyon around another and read off the phase.
The trace of the double braiding (the monodromy) over every pair of sectors
assembles into the modular S-matrix,

$$S_{ab} = \frac{1}{|G|}\,
\mathrm{tr}\left(c_{b,a} \circ c_{a,b}\right),$$

and the Verlinde formula (Verlinde 1988) recovers the fusion rules from $S$
alone:

$$N_{ab}^c = \sum_x \frac{S_{ax} S_{bx} \bar{S}_{cx}}{S_{0x}}.$$

Nothing on the right-hand side mentions intertwiner spaces — this is how
ground-state and interferometry experiments infer fusion rules without
presupposing them (Zhang, Grover, Turner, Oshikawa, Vishwanath 2012; Iqbal et
al. 2023).

```python {.marimo}
def s_matrix(double, anyons):
    dims = [int(np.prod(a.inside)) for a in anyons]
    braids = [[Intertwiner[double].braid(a, b).eval(dtype=complex).array
               for b in anyons] for a in anyons]
    return np.array([[
        np.trace(braids[a][b].reshape(dims[a] * dims[b], -1)
                 @ braids[b][a].reshape(dims[b] * dims[a], -1))
        for b in range(len(anyons))]
        for a in range(len(anyons))]) / double.base.dim


def verlinde(S):
    return np.einsum('ax,bx,cx,x->abc', S, S, S.conj(), 1 / S[0])


def fusion_rules(S):
    return np.round(verlinde(S).real).astype(int)


S_toric = s_matrix(D2, toric)
assert np.allclose(S_toric @ S_toric.conj().T, np.eye(4))
assert np.allclose(verlinde(S_toric), fusion_rules(S_toric))
assert (fusion_rules(S_toric) == N_charts).all()
print(np.round(2 * S_toric.real).astype(int))
```

The S-matrix is unitary, and the fusion rules it generates agree with the
chart dimensions channel by channel — two computations that share no code
path: one solves the commutant, the other traces monodromies.

## Learning the fusion rules from measurements

Now make it a learning problem. An interferometer does not output traces: it
outputs clicks, with the visibility of the interference fringe given by the
normalised monodromy $M_{ab} = \mathrm{tr}(c_{b,a} \circ c_{a,b}) / d_a d_b$.
We simulate a finite number of shots per pair of anyons — one Bernoulli
sample per shot for each quadrature — estimate $\hat{S}$ from the counts, and
push the noisy estimate through the Verlinde formula. The fusion multiplicities
are integers, so rounding gives an exact success criterion: either the
statistics suffice to identify the theory, or they do not.

```python {.marimo}
def sample_s_matrix(S, dims, group_order, shots, seed):
    rng = np.random.default_rng(seed)
    visibility = S * group_order / np.outer(dims, dims)
    real = 2 * rng.binomial(shots, (1 + visibility.real) / 2) / shots - 1
    imag = 2 * rng.binomial(shots, (1 + visibility.imag) / 2) / shots - 1
    return (real + 1j * imag) * np.outer(dims, dims) / group_order


S_sampled = sample_s_matrix(S_toric, [1, 1, 1, 1], 2, shots=500, seed=7)
assert not np.allclose(S_sampled, S_toric)
assert (fusion_rules(S_sampled) == N_charts).all()
print("largest Verlinde residual before rounding:",
      np.round(np.abs(verlinde(S_sampled) - fusion_rules(S_sampled)).max(), 3))
```

With 500 shots per pair the estimated S-matrix is off in the second decimal,
and the rounded fusion table is already exact. The fusion rules of the toric
code have been learned from (simulated) measurement statistics — the chart
only graded the answer.

## A fusion table that is no group law

For the toric code, one could object that the fusion rules are just the group
law of $\mathbb{Z}/2 \times \mathbb{Z}/2$. The smallest quantum double where
fusion is *not* a group multiplication is $D(k[S_3])$: eight anyons, labelled
by a conjugacy class of $S_3$ (the flux) and an irreducible representation of
its centraliser (the charge). $A$, $B$, $C$ are the pure charges — trivial,
sign, and the two-dimensional representation of $S_3$; $D$, $E$ carry the
transposition flux with centraliser $\mathbb{Z}/2$; $F$, $G$, $H$ carry the
three-cycle flux with centraliser $\mathbb{Z}/3$. Their dimensions
$1, 1, 2, 3, 3, 2, 2, 2$ square-sum to $|D(k[S_3])| = 36$.

```python {.marimo}
D3 = Double(Algebra.symmetric(3))
permutation_matrices = {
    i: np.eye(3)[list(p)].T for i, p in enumerate(
        sorted(__import__('itertools').permutations(range(3))))}
omega = np.exp(2j * np.pi / 3)
plane = np.array([
    [1, omega, omega ** 2], [1, omega ** 2, omega]]) / np.sqrt(3)
s3_anyons = [
    Representation[D3].anyon(0, 1),
    Representation[D3].anyon(0, {
        i: float(np.linalg.det(m))
        for i, m in permutation_matrices.items()}),
    Representation[D3].anyon(0, {
        i: plane @ m @ plane.conj().T
        for i, m in permutation_matrices.items()}),
    Representation[D3].anyon(1, {0: 1, 1: 1}),
    Representation[D3].anyon(1, {0: 1, 1: -1}),
    Representation[D3].anyon(3, {0: 1, 3: 1, 4: 1}),
    Representation[D3].anyon(3, {0: 1, 3: omega, 4: omega ** 2}),
    Representation[D3].anyon(3, {0: 1, 3: omega ** 2, 4: omega})]
s3_labels = list("ABCDEFGH")
assert [int(np.prod(a.inside)) for a in s3_anyons] == [1, 1, 2, 3, 3, 2, 2, 2]
```

The same pipeline — braid, trace, Verlinde — generates the fusion table of
$D(k[S_3])$ from its braiding data:

```python {.marimo}
S_s3 = s_matrix(D3, s3_anyons)
assert np.allclose(S_s3 @ S_s3.conj().T, np.eye(8))
N_s3 = fusion_rules(S_s3)
assert np.allclose(verlinde(S_s3), N_s3) and (N_s3 >= 0).all()
print(fusion_table(N_s3, s3_labels))
```

The two-dimensional charge $C$ fuses with itself into three different anyons,
$C \otimes C = A + B + C$ — no group multiplication does that. The chart
agrees on every channel we ask it about, again through entirely different
linear algebra:

```python {.marimo}
s3_dims = [int(np.prod(a.inside)) for a in s3_anyons]
assert all(
    multiplicity(D3, s3_anyons[a] @ s3_anyons[b], s3_anyons[c],
                 s3_dims[a] * s3_dims[b] * s3_dims[c]) == N_s3[a, b, c]
    for a, b in [(2, 2), (3, 4), (3, 3), (2, 6)] for c in range(8))
```

## Anyons as an error-correcting code

Back to the quantum computing context of the toric code. The anyon sectors of
Kitaev's model form exactly the category of representations of the quantum
double (Bols, Chen, Naaijkens 2025), and the *code space* of $n$ anyons with
trivial total charge is an intertwiner space:

$$\mathcal{C} \;=\; \mathrm{Hom}\!\left(1,\; V_1 \otimes \cdots \otimes
V_n\right).$$

Its dimension is the topological degeneracy — the protected qubits. The chart
of this space is not just a basis of states: it is the *encoding isometry*
from the logical space into the physical one.

```python {.marimo}
V2 = Representation[D2].direct_sum(toric)
vacuum = toric[0]
code = Intertwiner[D2].chart(vacuum, V2 @ V2)
logical = code.eval(dtype=complex).array.size // 16
encoder = code.eval(dtype=complex).array.reshape(logical, 16)
assert logical == 4
assert np.allclose(encoder @ encoder.conj().T, np.eye(4))
```

Two anyons of the toric code protect a four-dimensional logical space — one
state for each anyon type paired with itself — and the chart's slices are an
orthonormal encoding of it.

This is where learning intertwiners meets error correction, and the statement
is sharper than an analogy. A code is called *covariant* for a symmetry when
its encoding isometry commutes with the group action on the logical and
physical spaces — that is, when the encoder **is an intertwiner** (Faist,
Nezami, Albert, Salton, Pastawski, Hayden, Preskill 2020; Zhou, Liu, Jiang
2021). Fitting inside the chart, as in the sections above, is therefore not
just a regularisation trick: it is a search over exactly the valid covariant
encoders, with the symmetry holding by construction rather than by penalty.
The Eastin–Knill theorem forbids such exact covariance for continuous
symmetries acting transversally (Eastin, Knill 2009; Kubica,
Demkowicz-Dobrzański 2021) — the quantum double of a finite group is on the
right side of that boundary, which is one reason topological codes can have
exactly covariant encoders at all.

## The braid group acts on the parameters

The parametrisation is not passive under the topology. Pre-composing an
intertwiner with the braiding of two strands preserves the space
$\mathrm{Hom}(V^{\otimes 3}, V)$, so the braid group $B_3$ acts on the
parameters of its chart. Because $D(k[\mathbb{Z}/2])$ is quasitriangular and
not a plain group algebra, the representation is genuinely braided: the
generators are unitary and satisfy Yang–Baxter, but their square is not the
identity — the $e$–$m$ double braiding is $-1$.

```python {.marimo}
Q3 = Intertwiner[D2].chart(V2 @ V2 @ V2, V2)
fusion_states = Q3.eval(dtype=complex).array.size // (64 * 4)
slices = Q3.eval(dtype=complex).array.reshape(fusion_states, 64, 4)
crossing = Intertwiner[D2].braid(V2, V2).eval(
    dtype=complex).array.reshape(16, 16).T
b1, b2 = [
    np.einsum('kva,vw,lwa->kl', slices.conj(), c, slices)
    for c in [np.kron(crossing, np.eye(4)), np.kron(np.eye(4), crossing)]]
assert fusion_states == 64
assert np.allclose(b1 @ b1.conj().T, np.eye(64))
assert np.allclose(b1 @ b2 @ b1, b2 @ b1 @ b2)
assert not np.allclose(b1 @ b1, np.eye(64))
```

## Where this goes

- The chart needs no semisimplicity — it is a nullspace, so it works for
  Sweedler's Hopf algebra and other non-semisimple examples where no
  equivariant projector exists.
- Everything above is multiplicity-free: every $N_{ab}^c$ is $0$ or $1$, so
  each fusion channel fixes its intertwiner up to scale. In a theory with
  multiplicities the chart of a single channel has genuine parameters, and
  *which* intertwiner nature picks becomes a learnable question — the honest
  successor to the circular fit this notebook replaced.
- A chart is an ordinary box: it can sit inside any diagram evaluated by the
  ribbon `Functor`, e.g. as a learnable fusion vertex in a knot invariant or
  a tensor-network model.

## References

- A. Kitaev, *Fault-tolerant quantum computation by anyons*, Annals of
  Physics 303, 2–30 (2003),
  [quant-ph/9707021](https://arxiv.org/abs/quant-ph/9707021).
- E. Verlinde, *Fusion rules and modular transformations in 2D conformal
  field theory*, Nuclear Physics B 300, 360–376 (1988).
- Y. Zhang, T. Grover, A. Turner, M. Oshikawa, A. Vishwanath,
  *Quasiparticle statistics and braiding from ground-state entanglement*,
  Physical Review B 85, 235151 (2012),
  [arXiv:1111.2342](https://arxiv.org/abs/1111.2342).
- M. Iqbal et al., *Non-Abelian topological order and anyons on a
  trapped-ion processor*, Nature 626, 505–511 (2024),
  [arXiv:2305.03766](https://arxiv.org/abs/2305.03766).
- A. Bols, S. Chen, P. Naaijkens, *The category of anyon sectors for
  non-Abelian quantum double models*,
  [arXiv:2503.15611](https://arxiv.org/abs/2503.15611).
- P. Faist, S. Nezami, V. V. Albert, G. Salton, F. Pastawski, P. Hayden,
  J. Preskill, *Continuous symmetries and approximate quantum error
  correction*, Physical Review X 10, 041018 (2020),
  [arXiv:1902.07714](https://arxiv.org/abs/1902.07714).
- S. Zhou, Z.-W. Liu, L. Jiang, *New perspectives on covariant quantum error
  correction*, Quantum 5, 521 (2021),
  [arXiv:2005.11918](https://arxiv.org/abs/2005.11918).
- B. Eastin, E. Knill, *Restrictions on transversal encoded quantum gate
  sets*, Physical Review Letters 102, 110502 (2009),
  [arXiv:0811.4262](https://arxiv.org/abs/0811.4262).
- A. Kubica, R. Demkowicz-Dobrzański, *Using quantum metrological bounds in
  quantum error correction: a simple proof of the approximate Eastin–Knill
  theorem*, Physical Review Letters 126, 150503 (2021),
  [arXiv:2004.11893](https://arxiv.org/abs/2004.11893).
