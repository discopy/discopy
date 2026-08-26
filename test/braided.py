from discopy.braided import *


x, y, z = map(Ty, "xyz")
a, b = Ty('a'), Ty('b')
f = Box('f', a, b)

def test_hexagon():
    assert Diagram.braid(x, y @ z) == Braid(x, y) @ z >> y @ Braid(x, z)
    assert Diagram.braid(x @ y, z) == x @ Braid(y, z) >> Braid(x, z) @ y


def test_simplify():
    assert (Diagram.braid(x, y @ z) >> Diagram.braid(x, y @ z)[::-1]).simplify()\
        == Diagram.id(x @ y @ z)\
        == (Diagram.braid(y @ z, x)[::-1] >> Diagram.braid(y @ z, x)).simplify()


def test_strategy():
    from hypothesis import find

    from discopy import testing

    testing.assert_strategy_finds(Diagram, Braid)
    for is_dagger in (False, True):
        box = find(Box.strategy(), lambda value: isinstance(value, Braid)
                   and value.is_dagger == is_dagger)
        assert box.is_dagger == is_dagger


def test_axioms():
    from discopy import testing

    testing.assert_canonical_axioms(Diagram, Functor)
