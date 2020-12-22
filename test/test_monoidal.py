# -*- coding: utf-8 -*-

from pytest import raises
from discopy.cat import *
from discopy.monoidal import *
from discopy.rewriting import *


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


def test_PRO_init():
    assert list(PRO(0)) == []
    assert PRO(PRO(2)) == PRO(2)
    assert all(len(PRO(n)) == n for n in range(5))
    with raises(TypeError):
        PRO.upgrade(Ty('x'))


def test_PRO_tensor():
    assert PRO(2) @ PRO(3) @ PRO(7) == PRO(12)
    assert PRO(2) @ Ty(1) == Ty(1) @ PRO(2) == Ty(1, 1, 1)
    with raises(TypeError) as err:
        PRO(2) @ 3


def test_PRO_repr():
    assert repr((PRO(0), PRO(1))) == "(PRO(0), PRO(1))"


def test_PRO_str():
    assert str(PRO(2 * 3 * 7)) == "42"


def test_PRO_getitem():
    assert PRO(42)[2: 4] == PRO(2)
    assert all(PRO(42)[i].name == 1 for i in range(42))


def test_Layer_getitem():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', y, z)
    layer = Layer(x, f, z)
    assert layer[::-1] == Layer(x, f[::-1], z)
    assert layer[0] == layer


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


def test_Diagram_iter():
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, y)
    g0 = Box('g', y @ y, x)
    g1 = g0.dagger()
    d = (f0 >> f1) @ Id(y @ x) >> g0 @ g1 >> f0 @ g0
    assert Id(x @ y @ x).then(*(layer for layer in d)) == d


def test_Diagram_getitem():
    x, y = Ty('x'), Ty('y')
    f0, f1, g = Box('f0', x, y), Box('f1', y, y), Box('g', y @ y, x)
    diagram = f0 @ Id(y @ x)\
        >> f1 @ Id(y @ x)\
        >> g @ Id(x)\
        >> Id(x) @ g.dagger()\
        >> f0 @ Id(y @ y)\
        >> Id(y) @ g
    assert diagram[:] == diagram
    assert diagram[::-1] == diagram.dagger()
    with raises(TypeError):
        diagram["Alice"]
    for depth, (left, box, right) in enumerate(diagram.layers):
        layer = Id(left) @ box @ Id(right)
        assert diagram[depth] == layer
        assert (diagram[-depth], ) == tuple(
            Id(left) @ box @ Id(right)
            for left, box, right in (diagram.layers[-depth], ))
        assert diagram[depth:depth] == Id(layer.dom)
        assert diagram[depth:] == Id(layer.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.layers[depth:]))
        assert diagram[:depth] == Id(diagram.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.layers[:depth]))
        assert diagram[depth: depth + 2] == Id(layer.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.layers[depth: depth + 2]))


def test_Diagram_permutation():
    x = PRO(1)
    assert Diagram.swap(x, x ** 2)\
        == Diagram.swap(x, x) @ Id(x) >> Id(x) @ Diagram.swap(x, x)\
        == Diagram.permutation([2, 0, 1])
    with raises(ValueError):
        Diagram.permutation([2, 0])
    with raises(ValueError):
        Diagram.permutation([2, 0, 1], x ** 2)


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
    with raises(AxiomError) as err:
        Diagram(Ty('y'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])


def test_InterchangerError():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(InterchangerError) as err:
        (f >> g).interchange(0, 1)
    assert str(err.value) == str(InterchangerError(f, g))


def spiral(n_cups, _type=Ty('x')):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    unit, counit = Box('unit', Ty(), _type), Box('counit', _type, Ty())
    cup, cap = Box('cup', _type @ _type, Ty()), Box('cap', Ty(), _type @ _type)
    result = unit
    for i in range(n_cups):
        result = result >> Id(_type ** i) @ cap @ Id(_type ** (i + 1))
    result = result >> Id(_type ** n_cups) @ counit @ Id(_type ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(_type ** (n_cups - i - 1)) @ cup @ Id(_type ** (n_cups - i - 1))
    return result


def test_spiral(n_cups=2):
    diagram = spiral(n_cups)
    unit, counit = diagram.boxes[0], diagram.boxes[n_cups + 1]
    spiral_nf = diagram.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n_cups] == unit


def test_Id_init():
    assert Id(Ty('x')) == Diagram.id(Ty('x'))


def test_Id_repr():
    assert repr(Id(Ty('x'))) == "Id(Ty('x'))"


def test_Id_str():
    assert str(Id(Ty('x'))) == "Id(x)"


def test_Box_init():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert (f.name, f.dom, f.cod, f.data) == ('f', Ty('x', 'y'), Ty('z'), 42)


def test_Box_hash():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert {f: 42}[f] == 42


def test_Box_eq():
    f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
    assert f == Diagram(Ty('x', 'y'), Ty('z'), [f], [0]) and f != 'f'


def test_Swap():
    x, y = Ty('x'), Ty('y')
    assert repr(Swap(x, y)) == "Swap(Ty('x'), Ty('y'))"
    assert Swap(x, y).dagger() == Swap(y, x)
    with raises(ValueError):
        Swap(x ** 2, Ty())


def test_Functor_init():
    F = Functor({Ty('x'): Ty('y')}, {})
    assert F(Id(Ty('x'))) == Id(Ty('y'))


def test_Functor_repr():
    assert repr(Functor({Ty('x'): Ty('y')}, {})) ==\
        "Functor(ob={Ty('x'): Ty('y')}, ar={})"


def test_Functor_call():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    F = Functor({x: y, y: x}, {f: f.dagger()})
    assert F(x) == y
    assert F(f) == f.dagger()
    assert F(F(f)) == f
    assert F(f >> f.dagger()) == f.dagger() >> f
    assert F(f @ f.dagger()) == f.dagger() @ Id(x) >> Id(x) @ f
    with raises(TypeError) as err:
        F(F)
    assert str(err.value) == messages.type_err(Diagram, F)


def test_Functor_sum():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', x, y)
    F = Functor(ob={x: y, y: x}, ar={f: g[::-1], g: f[::-1]})
    assert F(f + g) == F(f) + F(g)


def test_Sum():
    x = Ty('x')
    f = Box('f', x, x)
    with raises(ValueError):
        Sum([])
    with raises(AxiomError):
        Sum([f], dom=Ty())
    with raises(AxiomError):
        f + Box('g', Ty(), x)
    with raises(TypeError):
        Sum.upgrade(f)
    assert Sum([f]) != f
    assert {Sum([f]): 42}[Sum([f])] == 42
    assert Id(x).then(*(3 * (f + f, ))) == sum(8 * [f >> f >> f])
    assert Id(Ty()).tensor(*(3 * (f + f, ))) == sum(8 * [f @ f @ f])
