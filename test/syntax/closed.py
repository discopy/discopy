from pytest import raises
from discopy.utils import from_tree
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


def test_Diagram():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.id(x) == Id(x)
    assert Diagram.ba(x, y) == BA(x >> y)
    assert Diagram.fa(x, y) == FA(x << y)
    assert Diagram.fc(x, y, z) == FC(x << y, y << z)
    assert Diagram.bc(x, y, z) == BC(x >> y, y >> z)
    assert Diagram.fx(x, y, z) == FX(x << y, z >> y)
    assert Diagram.bx(x, y, z) == BX(y << x, y >> z)


def test_BA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        BA(x << y)
    assert "BA(closed.Ty(closed.Under(" in repr(BA(x >> y))


def test_FA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        FA(x >> y)
    assert "FA(closed.Ty(closed.Over" in repr(FA(x << y))


def test_FC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        FC(x >> y, y >> x)
    with raises(TypeError):
        FC(x << y, y >> x)
    with raises(AxiomError):
        FC(x << y, z << y)


def test_BC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        BC(x << y, y << x)
    with raises(TypeError):
        BC(x >> y, y << x)
    with raises(AxiomError):
        BC(x >> y, z >> y)


def test_FX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        FX(x >> y, y >> x)
    with raises(TypeError):
        FX(x << y, y << x)
    with raises(AxiomError):
        FX(x << y, y >> x)


def test_BX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        BX(x >> y, y >> x)
    with raises(TypeError):
        BX(x << y, y << x)
    with raises(AxiomError):
        BX(x << y, y >> x)


def test_Functor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    IdF = Functor(lambda x: x, lambda f: f)
    assert IdF(x >> y << x) == x >> y << x
    assert IdF(Curry(f)) == Curry(f)
    assert IdF(FA(x << y)) == FA(x << y)
    assert IdF(BA(x >> y)) == BA(x >> y)
    assert IdF(FC(x << y, y << x)) == FC(x << y, y << x)
    assert IdF(BC(x >> y, y >> x)) == BC(x >> y, y >> x)
    assert IdF(FX(x << y, z >> y)) == FX(x << y, z >> y)
    assert IdF(BX(y << x, y >> z)) == BX(y << x, y >> z)


def test_to_rigid():
    from discopy import rigid

    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> FA(x << y)
    assert Diagram.to_rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert Diagram.to_rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)
    assert Diagram.to_rigid(Curry(BA(x >> y))).normal_form()\
        == rigid.Cap(y_, y_.l) @ rigid.Id(x_)
    assert closed2rigid(Curry(FA(x << y), left=True)).normal_form()\
        == rigid.Id(y_) @ rigid.Cap(x_.r, x_)
    assert closed2rigid(FC(x << y, y << x))\
        == rigid.Id(x_) @ rigid.Cup(y_.l, y_) @ rigid.Id(x_.l)
    assert closed2rigid(BC(x >> y, y >> x))\
        == rigid.Id(x_.r) @ rigid.Cup(y_, y_.r) @ rigid.Id(x_)
    assert closed2rigid(FX(x << y, x >> y))\
        == rigid.Id(x_) @ rigid.Swap(y_.l, x_.r) @ Id(y_) >>\
        rigid.Swap(x_, x_.r) @ rigid.Cup(y_.l, y_)
    assert closed2rigid(BX(y << x, y >> x))\
        == rigid.Id(y_) @ rigid.Swap(x_.l, y_.r) @ Id(x_) >>\
        rigid.Cup(y_, y_.r) @ rigid.Swap(x_.l, x_)


def test_to_tree():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    for diagram in [
            FA(x << y),
            BA(x >> y),
            FC(x << y, y << x),
            BC(x >> y, y >> x),
            FX(x << y, z >> y),
            BX(y << x, y >> z)]:
        assert from_tree(diagram.to_tree()) == diagram


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
