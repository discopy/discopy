from pytest import raises

import numpy as np

from discopy.quantum.optics import (ar_to_path, BBS, Box, Comonoid, Counit, Endo, Monoid,
                                    Unit, to_matrix, to_path, Id, MZI, PathBox,
                                    PRO, TBS)
from discopy.monoidal import Swap

unit = Unit()
counit = Counit()

monoid = Monoid()
comonoid = Comonoid()

swap = Swap(PRO(1), PRO(1))


def check(d1, d2):
    assert to_matrix(d1) == to_matrix(d2)


def test_bialgebra():
    check(monoid >> comonoid, comonoid @ comonoid >> monoid @ monoid)

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
    network = mzi >> bbs1 >> tbs >> bbs2
    assert to_matrix(to_path(network)) == to_matrix(network)
    assert to_matrix(to_path(network).dagger()) == to_matrix(network).dagger()

    assert np.allclose((mzi >> mzi.dagger()).array, np.eye(2))

    with raises(NotImplementedError):
        ar_to_path(123)
