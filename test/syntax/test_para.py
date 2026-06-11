import os
from discopy.monoidal import Ty
from discopy.symmetric import Box as SBox
from discopy.para import Box, Reparam


def test_para_composition_metadata():
    a, b, c = map(Ty, "ABC")
    p, q = map(Ty, "PQ")

    f = Box(name="f", data_dom=a, data_cod=b, params=p)
    g = Box(name="g", data_dom=b, data_cod=c, params=q)

    composite = f >> g

    assert composite.data_dom == a
    assert composite.data_cod == c
    assert composite.params == p @ q
    assert composite.dom == a @ p @ q
    assert composite.cod == c


def test_para_tensor_metadata():
    a, b, c, d = map(Ty, "ABCD")
    p, q = map(Ty, "PQ")

    f = Box(name="f", data_dom=a, data_cod=b, params=p)
    g = Box(name="g", data_dom=c, data_cod=d, params=q)

    composite = f @ g

    assert composite.data_dom == a @ c
    assert composite.data_cod == b @ d
    assert composite.params == p @ q
    assert composite.dom == a @ c @ p @ q
    assert composite.cod == b @ d


def test_para_reparam_metadata():
    a, b = map(Ty, "AB")
    p, p_prime = map(Ty, ("P", "P'"))

    f = Box(name="f", data_dom=a, data_cod=b, params=p)
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

    f = Box(name="f", data_dom=a, data_cod=b, params=p)
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

    f = Box(name="f", data_dom=a, data_cod=b, params=p)
    g = Box(name="g", data_dom=c, data_cod=d, params=q)
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
