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
    braided = find(
        Diagram.strategy(dom=x @ y, cod=y @ x),
        lambda value: any(isinstance(box, Braid) for box in value.boxes))
    assert (braided.dom, braided.cod) == (x @ y, y @ x)


def test_dagger_braid_serialisation():
    from discopy.utils import dumps, from_tree, loads

    x, y = Ty('x'), Ty('y')
    braid = Braid(x, y, is_dagger=True)
    assert from_tree(braid.to_tree()) == braid == loads(dumps(braid))


def test_axioms():
    from discopy import axioms

    axioms.assert_axioms(Diagram)
