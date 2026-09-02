"""P11: encodings agree with a numeric oracle — to_hypergraph, to_map and simplify preserve the value of one tensor functor, equal diagrams have equal values and unequal things compare unequal.
Miniature of the property over curated examples; red while its bullets (B47, B48, B50, B71, B74, B75) are live — issue #699.
"""
import numpy as np

from discopy import compact, frobenius, pivotal
from discopy.grammar import pregroup
from discopy.tensor import Dim, Tensor

x, y = map(compact.Ty, "xy")
f, g = compact.Box('f', x, y), compact.Box('g', x @ y, x @ y)
s, t = compact.Box('s', compact.Ty(), x), compact.Box('t', x @ y.r, x @ y.r)
F = compact.Functor(
    ob_map={x: 2, y: 3},
    ar_map={f: np.arange(6.), g: np.arange(36.),
            s: np.array([1., 2.]), t: np.arange(36.)},
    cod=Tensor)

F_compound = compact.Functor(ob_map={x: Dim(2, 2)}, ar_map={}, cod=Tensor)
cup, cap = compact.Cup(x, x.r), compact.Cap(x.r, x)
snake = x @ cap >> cup @ x

xp = pivotal.Ty('x')
sp = pivotal.Box('s', pivotal.Ty(), xp)
F_pivotal = pivotal.Functor(
    ob_map={xp: 2}, ar_map={sp: np.array([1., 2.])}, cod=Tensor)

a, b = map(frobenius.Ty, "ab")
fa = frobenius.Box('f', a, a)
G = frobenius.Functor(
    ob_map={a: 2, b: 3}, ar_map={fa: np.array([[1., 2.], [3., 4.]])},
    cod=Tensor)

n, sentence = pregroup.Ty('n'), pregroup.Ty('s')
loves = pregroup.Box('loves', n @ n, sentence)
wiring = pregroup.Cap(n.r, n) @ pregroup.Cap(n, n.l) >> n.r @ loves @ n.l
P = pregroup.Functor(
    ob_map={n: 2, sentence: 1}, ar_map={loves: np.arange(4.)}, cod=Tensor)

ENCODINGS = [
    ("to_hypergraph", lambda d: d.to_hypergraph().to_diagram()),
    ("to_map", lambda d: d.to_map().to_diagram()),
    ("simplify", lambda d: d.simplify()),
]

PRESERVED = [
    ("rotated box f.r", F, f.r),
    ("rotated box g.r", F, g.r),
    ("transpose of a rotated box", F, f.r.transpose()),
    ("trace over an adjoint wire", F, t.trace()),
    ("state on the left", F, s @ f),
    ("pivotal state on the left", F_pivotal, sp @ xp),
    ("frobenius rotated box", G, fa.r),
    ("pregroup cap wiring", P, wiring),
]

EQUAL = [
    ("snake on a compound image", F_compound, snake, compact.Id(x)),
    ("cup alone and cup inside a diagram on a compound image", F_compound,
     cup, snake @ x.r >> cup),
]

UNEQUAL = [
    ("scalar spider types differ", frobenius.Equation(
        frobenius.Spider(0, 0, a) @ fa, frobenius.Spider(0, 0, b) @ fa)),
    ("a box and its rotation differ", compact.Equation(g, g.r)),
]


def same(lhs, rhs):
    return (lhs.dom, lhs.cod) == (rhs.dom, rhs.cod) and np.allclose(
        np.asarray(lhs.array, dtype=float), np.asarray(rhs.array, dtype=float))


def test_p11():
    failures = []
    for label, functor, diagram in PRESERVED:
        expected = functor(diagram)
        for name, encode in ENCODINGS:
            if not hasattr(diagram, name):
                continue
            try:
                if not same(functor(encode(diagram)), expected):
                    failures.append(f"{label}: {name} changes the value")
            except Exception as error:
                failures.append(
                    f"{label}: {name} raised {type(error).__name__}")
    for label, functor, lhs, rhs in EQUAL:
        try:
            if not same(functor(lhs), functor(rhs)):
                failures.append(f"{label}: equal diagrams, unequal values")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    for label, equation in UNEQUAL:
        try:
            if bool(equation):
                failures.append(f"{label}: Equation says equal")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    assert not failures, failures
