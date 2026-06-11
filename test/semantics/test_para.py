from discopy.monoidal import Ty
from discopy.symmetric import Box as SBox
from discopy.para import Box, Reparam, Diagram
from discopy import python


def test_para_composition_metadata():
    a, b, c = map(Ty, "ABC")
    p, q = map(Ty, "PQ")

    f = Box(name="f", dom=a, cod=b, params=p)
    g = Box(name="g", dom=b, cod=c, params=q)

    composite = f >> g

    assert composite.dom == a
    assert composite.cod == c
    assert composite.params == p @ q
    assert composite.inside.dom == a @ p @ q
    assert composite.inside.cod == c


def test_para_tensor_metadata():
    a, b, c, d = map(Ty, "ABCD")
    p, q = map(Ty, "PQ")

    f = Box(name="f", dom=a, cod=b, params=p)
    g = Box(name="g", dom=c, cod=d, params=q)

    composite = f @ g

    assert composite.dom == a @ c
    assert composite.cod == b @ d
    assert composite.params == p @ q
    assert composite.inside.dom == a @ c @ p @ q
    assert composite.inside.cod == b @ d


def test_para_reparam_metadata():
    a, b = map(Ty, "AB")
    p, p_prime = map(Ty, ("P", "P'"))

    f = Box(name="f", dom=a, cod=b, params=p)
    r = SBox(name="r", dom=p_prime, cod=p)
    target = f.reparam(r)
    reparam = Reparam(f, target, r)

    assert reparam.source == f
    assert reparam.target == target
    assert reparam.reparam_box == r
    assert reparam.source.params == p
    assert reparam.target.params == p_prime
    assert reparam.reparam_box.dom == p_prime
    assert reparam.reparam_box.cod == p


def test_para_reparam_vertical_composition():
    a, b = map(Ty, "AB")
    p, p_prime, p_double_prime = map(Ty, ("P", "P'", "P''"))

    f = Box(name="f", dom=a, cod=b, params=p)
    r = SBox(name="r", dom=p_prime, cod=p)
    s = SBox(name="s", dom=p_double_prime, cod=p_prime)

    f_r = f.reparam(r)
    f_rs = f_r.reparam(s)
    alpha = Reparam(f, f_r, r)
    beta = Reparam(f_r, f_rs, s)

    composite = alpha >> beta

    assert composite.source == f
    assert composite.target == f_rs
    assert composite.reparam_box == s >> r


def test_para_reparam_horizontal_composition():
    a, b, c, d = map(Ty, "ABCD")
    p, p_prime = map(Ty, ("P", "P'"))
    q, q_prime = map(Ty, ("Q", "Q'"))

    f = Box(name="f", dom=a, cod=b, params=p)
    g = Box(name="g", dom=c, cod=d, params=q)
    r = SBox(name="r", dom=p_prime, cod=p)
    s = SBox(name="s", dom=q_prime, cod=q)

    f_r = f.reparam(r)
    g_s = g.reparam(s)
    alpha = Reparam(f, f_r, r)
    beta = Reparam(g, g_s, s)

    composite = alpha @ beta

    assert composite.source == f @ g
    assert composite.target == f_r @ g_s
    assert composite.reparam_box == r @ s


def test_para_python():
    A, B, P = int, float, str
    f_inside = python.Function(lambda x, s: float(len(s) + x), (A, P), (B, ))
    f = Diagram[python.Category](f_inside, (A, ), (B, ), (P, ))
    
    Q = str
    g_inside = python.Function(lambda b, q: b + len(q), (B, Q), (float, ))
    g = Diagram[python.Category](g_inside, (B, ), (float, ), (Q, ))
    
    h = f >> g

    assert h.dom == (int, )
    assert h.cod == (float, )
    assert h.params == (P, Q)
    assert h.inside(1, "abc", "defg") == 1 + 3 + 4

    # Data input: A=1; P = "abc"; Q = "defg"
    # f runs: f(1, "abc") -> float(len("abc") + 1) = 4.0
    # g runs: g(4,0, "defg") -> 4.0 + len("defg") = 4.0 + 4 = 8.0
