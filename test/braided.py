from discopy.braided import *
from discopy.axioms import assert_axioms
from hypothesis import find
from discopy.utils import dumps, loads


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


def test_axioms():
    assert_axioms(Diagram)
    for generator in Diagram.generators:
        box = find(generator.strategy(), lambda _: True)
        assert box == loads(dumps(box)) == eval(repr(box), box.environment())
