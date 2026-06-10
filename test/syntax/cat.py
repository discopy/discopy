# -*- coding: utf-8 -*-

from pytest import raises

from discopy.cat import *
from discopy.utils import AxiomError


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
    assert (Ob('x'), Ob('Alice')) == (Ob('x'), Ob('Alice'))


def test_Ob_name():
    assert Ob('x').name == 'x'


def test_Ob_repr():
    assert repr(Ob('x')) == "cat.Ob('x')"


def test_Ob_str():
    assert str(Ob('x')) == 'x'


def test_Ob_eq():
    x, x1, y = Ob('x'), Ob('x'), Ob('y')
    assert x == x1 and x != y and x != 'x'
    assert 'x' != Ob('x')


def test_Ob_hash():
    assert {Ob('x'): 42}[Ob('x')] == 42


def test_Arrow():
    x, y, z, w = Ob('x'), Ob('y'), Ob('z'), Ob('w')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, w)
    assert f >> g >> h == Arrow((f, g, h), x, w)


def test_Arrow_init():
    with raises(TypeError) as err:
        Arrow((), 1, Ob('x'))
    with raises(TypeError) as err:
        Arrow((), Ob('x'), 1)
    with raises(TypeError) as err:
        Arrow((Ob('x'), ), Ob('x'), Ob('x'))


def test_Arrow_len():
    assert len(Arrow((), Ob('x'), Ob('x'))) == 0


def test_Arrow_getitem():
    f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
    arrow = f >> g >> g.dagger() >> f.dagger()\
        >> f >> g >> g.dagger() >> f.dagger()
    with raises(TypeError):
        arrow["Alice"]
    with raises(IndexError):
        arrow[9]
    with raises(IndexError):
        arrow[::-2]
    assert arrow[:] == arrow
    assert arrow[::-1] == arrow.dagger()
    assert arrow[:0] == arrow[:-8] == arrow[-9:-9] == Id(arrow.dom)
    for depth, box in enumerate(arrow):
        assert arrow[depth] == box
        assert arrow[-depth] == arrow.inside[-depth]
        assert arrow[depth:depth] == Id(box.dom)
        assert arrow[depth:] == Id(box.dom).then(*arrow.inside[depth:])
        assert arrow[:depth] == Id(arrow.dom).then(
            *arrow.inside[:depth])
        assert arrow[depth: depth + 2] == Id(box.dom).then(
            *arrow.inside[depth: depth + 2])


def test_Arrow_repr():
    assert repr(Arrow((), Ob('x'), Ob('x'))) == "cat.Arrow.id(cat.Ob('x'))"
    inside = (Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z')))
    assert repr(Arrow(inside, Ob('x'), Ob('z')))\
        == "cat.Arrow(inside=(cat.Box('f', cat.Ob('x'), cat.Ob('y')), "\
           "cat.Box('g', cat.Ob('y'), cat.Ob('z'))), dom=cat.Ob('x'), "\
           "cod=cat.Ob('z'))"


def test_Arrow_str():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert str(Arrow((), x, x) == "Id(x)")
    assert str(Arrow((f, ), x, y) == "f")
    assert str(Arrow((f, g), x, z)) == "f >> g"


def test_Arrow_eq():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f >> g == Arrow((f, g), x, z)


def test_Arrow_hash():
    assert {Id(Ob('x')): 42}[Id(Ob('x'))] == 42


def test_Arrow_then():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f.then(g) == f >> g == g << f
    with raises(TypeError) as err:
        f >> x


def test_Arrow_dagger():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    h = Arrow((f, g), x, z)
    assert h.dagger() == g.dagger() >> f.dagger()
    assert h.dagger().dagger() == h


def test_Id_init():
    idx = Id(Ob('x'))
    assert idx >> idx == idx
    assert idx.dagger() == idx


def test_Id_repr():
    assert repr(Id(Ob('x'))) == "cat.Arrow.id(cat.Ob('x'))"


def test_Id_str():
    x = Ob('x')
    assert str(Id(x)) == "Id(x)"


def test_AxiomError():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(AxiomError) as err:
        Arrow((g, ), x, y)
    with raises(AxiomError) as err:
        Arrow((f, ), x, z)
    with raises(AxiomError) as err:
        g >> f


def test_Box():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    assert f >> Id(Ob('y')) == f == Id(Ob('x')) >> f


def test_Box_dagger():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f.dom == f.dagger().cod and f.cod == f.dagger().dom
    assert f == f.dagger().dagger()


def test_Box_repr():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert repr(f) == "cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42)"
    assert repr(f.dagger())\
        == "cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42).dagger()"


def test_Box_str():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert str(f) == "f"
    assert str(f.dagger()) == "f[::-1]"


def test_Box_hash():
    assert {Box('f', Ob('x'), Ob('y')): 42}[Box('f', Ob('x'), Ob('y'))] == 42


def test_Box_eq():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f == Arrow((f, ), Ob('x'), Ob('y')) and f != Ob('x')


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
    assert repr(Functor({}, {})) == "cat.Functor(ob={}, ar={})"


def test_Functor_call():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    with raises(TypeError) as err:
        F(F)
    assert F(x) == y
    assert F(f) == f.dagger()
    assert F(f.dagger()) == f
    assert F(g) == f >> g
    assert F(f >> g) == f.dagger() >> f >> g


def test_total_ordering():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    assert sorted([z, y, x]) == [x, y, z]
    f, g = Box('f', x, y), Box('g', y, z)
    assert f < g


def test_Bubble():
    f = Box('f', Ob('x'), Ob('y'))
    assert repr((f).bubble())\
        == "cat.Bubble(cat.Box('f', cat.Ob('x'), cat.Ob('y')))"
    assert str(f.bubble()) == "(f).bubble()"


def test_Box_call():
    f = Box('f', Ob('x'), Ob('y'))
    with raises(TypeError):
        f(42)


def test_from_tree():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    d = (f >> f[::-1].bubble()) + Id(Ob('x'))
    assert from_tree(d.to_tree()) == d


def test_sum_lambdify():
    from sympy.abc import phi
    f = Box('f', Ob('x'), Ob('y'), data=[phi])
    g = Box('g', Ob('x'), Ob('y'), data=[phi])

    assert (f + g).free_symbols == {phi}
    assert (f + g).lambdify(phi)(1) == f.lambdify(phi)(1) + g.lambdify(phi)(1)

    empty_sum = Sum((), Ob('x'), Ob('y'))
    assert empty_sum.lambdify(phi)(123) == empty_sum

    assert empty_sum.subs(phi, 0) == empty_sum


def test_Sum():
    x, y, z = map(Ob, "xyz")
    f = Box('f', x, y)
    g = Box('g', y, z)
    with raises(ValueError):
        Sum(())
    with raises(AxiomError):
        Sum((), x, y) + Sum((), y, z)
    with raises(AxiomError):
        Sum((f, ), y, z)
    with raises(AxiomError):
        Sum((f, g))
    assert hash(Sum((), x, y)) == hash(Sum((), x, y))
    assert repr(Sum((), x, y))\
        == "cat.Sum(terms=(), dom=cat.Ob('x'), cod=cat.Ob('y'))"
    assert list(Sum((f, ), x, y)) == [f]
    assert len(Sum((), x, y)) == 0
    assert Sum((), x, x).then(f, g) == Sum((), x, z)
    assert Sum((), x, y).dagger() == Sum((), y, x)
