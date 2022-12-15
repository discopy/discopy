from discopy.closed import *


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert repr(Over(x, y))\
        == "closed.Over(closed.Ty(cat.Ob('x')), closed.Ty(cat.Ob('y')))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
    assert repr(Under(x, y))\
        == "closed.Under(closed.Ty(cat.Ob('x')), closed.Ty(cat.Ob('y')))"
    assert {Under(x, y): 42}[Under(x, y)] == 42
    assert Under(x, y) != Over(x, y)


def test_to_rigid():
    from discopy import rigid

    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> Diagram.ev(x, y)
    assert Diagram.to_rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert Diagram.to_rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', y, z << x), Box('g', y, z >> x)

    from discopy.python import Function
    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda y: lambda z: z + 1j if y else -1j},
        cod=Category(tuple[type, ...], Function))

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.uncurry(left=False).curry(left=False))(True)(1.2) == F(g)(True)(1.2)
