# -*- coding: utf-8 -*-

from pytest import raises
from discopy.cat import *


def test_main():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    assert Id(x) >> f == f == f >> Id(y)
    assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
    assert f >> g >> h == f >> (g >> h)
    F = Functor(ob={x: y, y: z, z: x}, ar={f: g, g: h})
    assert F(Id(x)) == Id(F(x))
    assert F(f >> g) == F(f) >> F(g)


def test_Ob():
    assert Ob('x') == Ob('x') and Ob('x') != Ob('y')


def test_Ob_init():
    assert (Ob('x'), Ob(42), Ob('Alice')) == (Ob('x'), Ob(42), Ob('Alice'))
    with raises(ValueError) as err:
        Ob('')
    assert str(err.value) == messages.empty_name('')


def test_Ob_name():
    assert Ob('x').name == 'x'


def test_Ob_repr():
    assert repr(Ob('x')) == "Ob('x')"


def test_Ob_str():
    str(Ob('x')) == 'x'


def test_Ob_eq():
    x, x1, y = Ob('x'), Ob('x'), Ob('y')
    assert x == x1 and x != y and x != 'x'
    assert 'x' != Ob('x')


def test_Ob_hash():
    assert {Ob('x'): 42}[Ob('x')] == 42


def test_Diagram():
    x, y, z, w = Ob('x'), Ob('y'), Ob('z'), Ob('w')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, w)
    assert f >> g >> h == Diagram(x, w, [f, g, h])


def test_Diagram_init():
    with raises(TypeError) as err:
        Diagram('x', Ob('x'), [])
    assert str(err.value) == messages.type_err(Ob, 'x')
    with raises(TypeError) as err:
        Diagram(Ob('x'), 'x', [])
    assert str(err.value) == messages.type_err(Ob, 'x')
    with raises(TypeError) as err:
        Diagram(Ob('x'), Ob('x'), [Ob('x')])
    assert str(err.value) == messages.type_err(Diagram, Ob('x'))


def test_Diagram_len():
    assert len(Diagram(Ob('x'), Ob('x'), [])) == 0


def test_Diagram_getitem():
    f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
    diagram = 2 * (f >> g >> g.dagger() >> f.dagger())
    with raises(TypeError):
        diagram["Alice"]
    with raises(IndexError):
        diagram[9]
    with raises(IndexError):
        diagram[::-2]
    assert diagram[:] == diagram
    assert diagram[::-1] == diagram.dagger()
    assert diagram[:0] == diagram[:-8] == diagram[-9:-9] == Id(diagram.dom)
    for depth, box in enumerate(diagram):
        assert diagram[depth] == box
        assert diagram[-depth] == diagram.boxes[-depth]
        assert diagram[depth:depth] == Id(box.dom)
        assert diagram[depth:] == Id(box.dom).compose(*diagram.boxes[depth:])
        assert diagram[:depth] == Id(diagram.dom).compose(
            *diagram.boxes[:depth])
        assert diagram[depth: depth + 2] == Id(box.dom).compose(
            *diagram.boxes[depth: depth + 2])


def test_Diagram_repr():
    assert repr(Diagram(Ob('x'), Ob('x'), [])) == "Id(Ob('x'))"
    assert repr(Diagram(Ob('x'), Ob('y'), [Box('f', Ob('x'), Ob('y'))]))\
        == "Box('f', Ob('x'), Ob('y'))"
    assert repr(Diagram(Ob('x'), Ob('z'),
                [Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))]))\
        == "cat.Diagram(dom=Ob('x'), cod=Ob('z'), "\
           "boxes=[Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))])"


def test_Diagram_str():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert str(Diagram(x, x, []) == "Id(x)")
    assert str(Diagram(x, y, [f]) == "f")
    assert str(Diagram(x, z, [f, g])) == "f >> g"


def test_Diagram_eq():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f >> g == Diagram(x, z, [f, g])


def test_Diagram_hash():
    assert {Id(Ob('x')): 42}[Id(Ob('x'))] == 42


def test_Diagram_then():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f.then(g) == f >> g == g << f
    with raises(TypeError) as err:
        f >> x
    assert str(err.value) == messages.type_err(Diagram, x)


def test_Diagram_dagger():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    h = Diagram(x, z, [f, g])
    assert h.dagger() == g.dagger() >> f.dagger()
    assert h.dagger().dagger() == h


def test_Id_init():
    idx = Id(Ob('x'))
    assert idx >> idx == idx
    assert idx.dagger() == idx


def test_Id_repr():
    repr(Id(Ob('x'))) == Id(Ob('x'))


def test_Id_str():
    x = Ob('x')
    assert str(Id(x)) == "Id(x)"


def test_AxiomError():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(AxiomError) as err:
        Diagram(x, y, [g])
    assert str(err.value) == messages.does_not_compose(Id(x), g)
    with raises(AxiomError) as err:
        Diagram(x, z, [f])
    assert str(err.value) == messages.does_not_compose(f, Id(z))
    with raises(AxiomError) as err:
        g >> f
    assert str(err.value) == messages.does_not_compose(g, f)


def test_Box():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    assert f >> Id(Ob('y')) == f == Id(Ob('x')) >> f
    with raises(ValueError) as err:
        Box('', '', '')


def test_Box_dagger():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f.dom == f.dagger().cod and f.cod == f.dagger().dom
    assert f == f.dagger().dagger()


def test_Box_repr():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert repr(f) == "Box('f', Ob('x'), Ob('y'), data=42)"
    assert repr(f.dagger()) == "Box('f', Ob('x'), Ob('y'), data=42).dagger()"


def test_Box_str():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert str(f) == "f"
    assert str(f.dagger()) == "f[::-1]"


def test_Box_hash():
    assert {Box('f', Ob('x'), Ob('y')): 42}[Box('f', Ob('x'), Ob('y'))] == 42


def test_Box_eq():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f == Diagram(Ob('x'), Ob('y'), [f]) and f != Ob('x')


def test_Functor():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    assert F((f >> g).dagger()) == F(f >> g).dagger()
    assert F(Id(Ob('x'))) == Id(Ob('y'))


def test_Functor_eq():
    x, y = Ob('x'), Ob('y')
    assert Functor({x: y, y: x}, {}) == Functor({y: x, x: y}, {})


def test_Functor_repr():
    assert repr(Functor({}, {})) == "Functor(ob={}, ar={})"


def test_Functor_call():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    with raises(TypeError) as err:
        F(F)
    assert str(err.value) == messages.type_err(Diagram, F)
    assert F(x) == y
    assert F(f) == f.dagger()
    assert F(f.dagger()) == f
    assert F(g) == f >> g
    assert F(f >> g) == f.dagger() >> f >> g


def test_Quiver():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    F = Functor({x: x, y: y, z: z}, Quiver(lambda x: x))
    f = Box('f', x, y, data=[0, 1])
    assert F(f) == Box('f', Ob('x'), Ob('y'), data=[0, 1])
    f.data.append(2)
    assert F(f) == Box('f', Ob('x'), Ob('y'), data=[0, 1, 2])


def test_Quiver_init():
    ar = Quiver(lambda x: x ** 2)
    assert ar[3] == 9


def test_Quiver_getitem():
    assert Quiver(lambda x: x * 10)[42] == 420
    with raises(TypeError) as err:
        Quiver(lambda x: x * 10)[42] = 421
    "does not support item assignment" in str(err.value)


def test_Quiver_repr():
    assert "Quiver(<function " in repr(Quiver(lambda x: x))
