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

    from discopy import axioms

    axioms.assert_strategy_finds(Diagram, Braid)
    x, y = Diagram.ob('x'), Diagram.ob('y')
    for is_dagger in (False, True):
        braided = find(
            Diagram.strategy(dom=x @ y, cod=y @ x),
            lambda value: any(isinstance(box, Braid)
                              and box.is_dagger == is_dagger
                              for box in value.boxes))
        assert (braided.dom, braided.cod) == (x @ y, y @ x)
