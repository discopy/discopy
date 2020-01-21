# -*- coding: utf-8 -*-

from pytest import raises
from discopy.moncat import *


def test_Ty():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert x @ y != y @ x
    assert x @ Ty() == x == Ty() @ x
    assert (x @ y) @ z == x @ y @ z == x @ (y @ z)


def test_Ty_init():
    assert list(Ty('x', 'y', 'z')) == [Ob('x'), Ob('y'), Ob('z')]


def test_Ty_eq():
    assert Ty('x') != 'x'


def test_Ty_repr():
    assert repr(Ty('x', 'y')) == "Ty('x', 'y')"


def test_Ty_str():
    str(Ty('x')) == 'x'


def test_Ty_getitem():
    assert Ty('x', 'y', 'z')[:1] == Ty('x')


def test_Ty_pow():
    assert Ty('x') ** 42 == Ty('x') ** 21 @ Ty('x') ** 21
    with raises(TypeError) as err:
        Ty('x') ** Ty('y')
    assert messages.type_err(int, Ty('y'))


def test_Diagram_init():
    with raises(TypeError) as err:
        Diagram('x', Ty('x'), [], [])
    assert str(err.value) == messages.type_err(Ty, 'x')
    with raises(TypeError) as err:
        Diagram(Ty('x'), 'x', [], [])
    assert str(err.value) == messages.type_err(Ty, 'x')
    with raises(ValueError) as err:
        Diagram(Ty('x'), Ty('x'), [], [1])
    assert "Boxes and offsets must have the same length." in str(err.value)
    with raises(TypeError) as err:
        Diagram(Ty('x'), Ty('x'), [1], [1])
    assert str(err.value) == messages.type_err(Diagram, 1)
    with raises(TypeError) as err:
        Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [Ty('x')])
    assert str(err.value) == messages.type_err(int, Ty('x'))


def test_Diagram_eq():
    assert Diagram(Ty('x'), Ty('x'), [], []) != Ty('x')
    assert Diagram(Ty('x'), Ty('x'), [], []) == Id(Ty('x'))


def test_Diagram_getitem():
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x @ y, x), Box('f1', x, y)
    d = f0 @ Id(y @ y) >> f0 @ Id(y) >> f1 @ f1.dagger()
    with raises(IndexError) as err:
        d[5]
    assert "IndexError" in str(err)
    with raises(IndexError) as err:
        d[1:3:2]
    assert "IndexError" in str(err)
    with raises(ValueError) as err:
        d[x]
    assert "ValueError" in str(err)


def test_Diagram_tensor():
    with raises(TypeError) as err:
        Id(Ty('x')) @ Ty('x')
    assert str(err.value) == messages.type_err(Diagram, Ty('x'))


def test_Diagram_offsets():
    assert Diagram(Ty('x'), Ty('x'), [], []).offsets == []


def test_Diagram_repr():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert repr(Diagram(x, x, [], [])) == "Id(Ty('x'))"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert repr(Diagram(x, y, [f0], [0])) == "Box('f0', Ty('x'), Ty('y'))"
    assert "Diagram(dom=Ty('x', 'z'), cod=Ty('y', 'w')" in repr(f0 @ f1)
    assert "offsets=[0, 1]" in repr(f0 @ f1)


def test_Diagram_hash():
    assert {Id(Ty('x')): 42}[Id(Ty('x'))] == 42


def test_Diagram_str():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert str(Diagram(x, x, [], [])) == "Id(x)"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert str(Diagram(x, y, [f0], [0])) == "f0"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ Id(z) >> Id(y) @ f1"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ Id(z) >> Id(y) @ f1"


def test_Diagram_matmul():
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x', 'y'))
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x')).tensor(Id(Ty('y')))


def test_Diagram_interchange():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    d = f @ f.dagger()
    assert d.interchange(0, 0) == f @ Id(y) >> Id(y) @ f.dagger()
    assert d.interchange(0, 1) == Id(x) @ f.dagger() >> f @ Id(x)
    assert (d >> d.dagger()).interchange(0, 2) ==\
        Id(x) @ f.dagger() >> Id(x) @ f >> f @ Id(y) >> f.dagger() @ Id(y)
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    assert (cup >> cap).interchange(0, 1) == cap @ Id(x @ x) >> Id(x @ x) @ cup
    assert (cup >> cap).interchange(0, 1, left=True) ==\
        Id(x @ x) @ cap >> cup @ Id(x @ x)
    f0, f1 = Box('f0', x, y), Box('f1', y, x)
    d = f0 @ Id(y) >> f1 @ f1 >> Id(x) @ f0
    with raises(IndexError):
        d.interchange(42, 43)
    with raises(InterchangerError) as err:
        d.interchange(0, 2)
    assert str(err.value) == str(InterchangerError(f0, f1))
    assert d.interchange(2, 0) == Id(x) @ f1 >> f0 @ Id(x) >> f1 @ f0


def test_Diagram_normalize():
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, x)
    assert list((f0 >> f1).normalize()) == []


def test_Diagram_normal_form():
    assert Id(Ty()).normal_form() == Id(Ty())
    assert Id(Ty('x', 'y')).normal_form() == Id(Ty('x', 'y'))
    s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        (s0 >> s1).normal_form()
    assert str(err.value) == messages.is_not_connected(s0 >> s1)
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, x)
    assert f0.normal_form() == f0
    assert (f0 >> f1).normal_form() == f0 >> f1
    assert (Id(x) @ f1 >> f0 @ Id(x)).normal_form() == f0 @ f1
    assert (f0 @ f1).normal_form(left=True) == Id(x) @ f1 >> f0 @ Id(x)


def test_AxiomError():
    with raises(AxiomError) as err:
        Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [0])
    assert str(err.value) == messages.does_not_compose(Ty('y'), Ty('x'))
    with raises(AxiomError) as err:
        Diagram(Ty('y'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])
    assert str(err.value) == messages.does_not_compose(Ty('y'), Ty('x'))


def test_InterchangerError():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(InterchangerError) as err:
        (f >> g).interchange(0, 1)
    assert str(err.value) == str(InterchangerError(f, g))


def test_spiral(n_cups=2):
    diagram = spiral(n_cups)
    unit, counit = diagram.boxes[0], diagram.boxes[n_cups + 1]
    spiral_nf = diagram.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n_cups] == unit


def test_Box_init():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert (f.name, f.dom, f.cod, f.data) == ('f', Ty('x', 'y'), Ty('z'), 42)


def test_Box_hash():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert {f: 42}[f] == 42


def test_Box_eq():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert f == Diagram(Ty('x', 'y'), Ty('z'), [f], [0]) and f != 'f'


def test_Functor_init():
    F = MonoidalFunctor({Ty('x'): Ty('y')}, {})
    assert F(Id(Ty('x'))) == Id(Ty('y'))


def test_Functor_repr():
    assert repr(MonoidalFunctor({Ty('x'): Ty('y')}, {})) ==\
        "MonoidalFunctor(ob={Ty('x'): Ty('y')}, ar={})"


def test_Functor_call():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    F = MonoidalFunctor({x: y, y: x}, {f: f.dagger()})
    assert F(x) == y
    assert F(f) == f.dagger()
    assert F(F(f)) == f
    assert F(f >> f.dagger()) == f.dagger() >> f
    assert F(f @ f.dagger()) == f.dagger() @ Id(x) >> Id(x) @ f
    with raises(TypeError) as err:
        F(F)
    assert str(err.value) == messages.type_err(Diagram, F)
