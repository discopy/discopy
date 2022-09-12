from pytest import raises

import numpy as np
from sympy.abc import phi

from discopy.quantum.optics import *
from discopy.quantum.zx import decomp, H, X, Z
from discopy.monoidal import Swap

unit = Unit()
counit = Counit()

monoid = Monoid()
comonoid = Comonoid()

swap = Swap(PRO(1), PRO(1))


def check(d1, d2):
    assert (to_matrix(d1).array == to_matrix(d2).array).all()


def test_bialgebra():
    check(monoid >> comonoid, (comonoid @ comonoid).permute(2, 1)
          >> monoid @ monoid)
    check(unit >> comonoid, unit @ unit)
    check(counit << monoid, counit @ counit)


def test_monoid():
    check(unit @ Id(PRO(1)) >> monoid, Id(PRO(1)))
    check(monoid @ Id(PRO(1)) >> monoid, Id(PRO(1)) @ monoid >> monoid)

    check(swap >> monoid, monoid)


def test_comonoid():
    check(counit @ Id(PRO(1)) << comonoid, Id(PRO(1)))
    check(comonoid @ Id(PRO(1)) << comonoid, Id(PRO(1)) @ comonoid << comonoid)

    check(swap << comonoid, comonoid)


def test_homomorphism():
    x = Endo(0.123)
    y = Endo(0.321)

    check(x >> comonoid, comonoid >> x @ x)
    check(y @ y >> monoid, monoid >> y)

    check(unit >> x, unit)
    check(counit << y, counit)

    check(comonoid >> x @ y >> monoid, Endo(0.123 + 0.321))
    check(x >> y, Endo(0.123 * 0.321))


def test_scalar():
    assert Scalar(1j).dagger() == Scalar(-1j)


def test_bad_queries():
    network = MZI(0.2, 0.4) @ MZI(0.2, 0.4) >> Id(1) @ MZI(0.2, 0.4) @ Id(1)
    with raises(ValueError):
        network.amp([0, 1], [2, 3])
    with raises(ValueError):
        network.pdist_prob([0, 1], [2, 3], np.array([[1, 0.1], [0.1, 1]]))
    with raises(ValueError):
        network.dist_prob([0, 1], [2, 3], np.array([[1, 0.1], [0.1, 1]]))


def test_bad_box():
    with raises(TypeError):
        Box('use PRO', 1, PRO(2))
    with raises(TypeError):
        Box('use PRO', PRO(1), 2)


def test_box_repr():
    box1 = Box('repr box', PRO(1), PRO(2))
    assert repr(box1) == "optics.Box('repr box', PRO(1), PRO(2))"

    box2 = PathBox('repr path box', PRO(1), PRO(2))
    assert repr(box2) == "optics.PathBox('repr path box', PRO(1), PRO(2))"


def test_to_matrix():
    mzi = MZI(0.18, 0.97)
    bbs1 = BBS(0.123)
    tbs = TBS(0.321)
    bbs2 = BBS(0)
    network = mzi >> bbs1 >> tbs >> bbs2 >> Phase(0.34) @ Id(1)
    path = optics2path(network)
    assert np.allclose(to_matrix(path).array, to_matrix(network).array)

    assert np.allclose((mzi >> mzi.dagger()).array, np.eye(2))

    with raises(NotImplementedError):
        ar_optics2path(123)
    with raises(Exception):
        to_matrix(annil)
    with raises(Exception):
        to_matrix(create)
    with raises(Exception):
        to_matrix(Scalar(2.))


def test_fusion_zx2path():
    fusion = Z(2, 1)
    path_fusion = zx2path(fusion)
    expect = Diagram(dom=PRO(4), cod=PRO(2),
                     boxes=[monoid, annil], offsets=[1, 1])
    assert path_fusion == expect
    assert evaluate(path_fusion, [1, 0, 1, 0], [1, 0]) == 1.0
    assert evaluate(path_fusion, [0, 1, 0, 1], [0, 1]) == 1.0
    assert evaluate(path_fusion, [0, 1, 1, 0], [0, 1]) == 0.0


def test_bell_zx2path():
    from discopy.quantum import zx

    zx_circs = [
        Z(0, 2),
        Z(0, 1) >> Z(1, 1) >> Z(1, 2),
        Z(0, 1) >> X(1, 1, 0.25) >> X(1, 1, -0.25) >> Z(1, 2),
        X(0, 1) >> H >> Z(1, 2),
        (X(0, 1, 0.5) >> decomp(X(1, 2)) >> X(1, 0, 0.5) @ zx.Id(1)
                      >> decomp(X(1, 2))),
        decomp(Z(0, 3) >> Z(1, 0) @ zx.Id(2)),
        Z(0, 2) >> decomp(X(1, 2) >> X(1, 0) @ zx.Id(1)) @ zx.Id(1),
        (X(0, 1) >> H >> Z(1, 2)
            >> zx.Id(2) @ X(0, 1) >> zx.Id(1) @ decomp(X(2, 1)))
    ]
    zx_circs += [decomp(zx_circ) for zx_circ in zx_circs]
    for zx_circ in zx_circs:
        path = zx2path(zx_circ)
        a = evaluate(path, [], [1, 0, 1, 0])
        b = evaluate(path, [], [1, 0, 0, 1])
        c = evaluate(path, [], [0, 1, 1, 0])
        d = evaluate(path, [], [0, 1, 0, 1])
        with raises(ValueError):
            evaluate(path, [], [1, 2, 3, 4])
        assert np.round(a, 5) == np.round(d, 5) == 1
        assert np.round(b, 5) == np.round(c, 5) == 0


def test_tk_ghz_zx2path():
    from discopy.quantum import zx, Ket

    tket_circ = Ket(0, 0, 0).H(0).CX(0, 1).CX(0, 2)
    path = zx2path(decomp(zx.circuit2zx(tket_circ)))
    encodings = [
        [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]
    ]

    amps = [np.round(evaluate(path, [], code), 5) for code in encodings]
    print(amps)
    assert amps[0] == amps[-1] == np.round(2 ** -0.5, 5)
    assert all(amp == 0 for amp in amps[1:-1])


def test_bad_zx2path():
    with raises(NotImplementedError):
        zx2path(Z(42, 21))


def test_endo_repr():
    assert Endo(phi).name == 'Endo(phi)'
    assert Endo(1).dagger() == Endo(1)


def test_bad_drags():
    with raises(ValueError):
        swap_right(comonoid, 0)
    with raises(ValueError):
        drag_out(comonoid, 0)


def test_make_square():
    d = comonoid >> Phase(.25) @ Id(1) >> monoid
    assert np.allclose(evaluate(d, [1], [1]),
                       evaluate(make_square(d), [1], [1]))


def test_ansatz():
    dims = [(1, 2), (2, 1), (3, 1), (2, 2)]
    for width, depth in dims:
        ansatz(width, depth, np.zeros(params_shape(width, depth)))
