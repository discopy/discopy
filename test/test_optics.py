from discopy.quantum.optics import Comonoid, Counit, Endo, Monoid, Unit, to_matrix, Id, PRO
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
