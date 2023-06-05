# -*- coding: utf-8 -*-

from pytest import raises

from discopy.cat import *
from discopy.monoidal import *
from discopy.drawing import spiral
from discopy.utils import AxiomError


def test_Ty():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert x @ y != y @ x
    assert x @ Ty() == x == Ty() @ x
    assert (x @ y) @ z == x @ y @ z == x @ (y @ z)


def test_Ty_init():
    assert list(Ty('x', 'y', 'z')) == [Ty('x'), Ty('y'), Ty('z')]


def test_Ty_eq():
    assert Ty('x') != 'x'


def test_Ty_repr():
    assert repr(Ty('x', 'y')) == "monoidal.Ty(cat.Ob('x'), cat.Ob('y'))"


def test_Ty_str():
    assert str(Ty('x')) == 'x'


def test_Ty_getitem():
    assert Ty('x', 'y', 'z')[:1] == Ty('x')


def test_Ty_pow():
    assert Ty('x') ** 42 == Ty('x') ** 21 @ Ty('x') ** 21
    with raises(TypeError) as err:
        Ty('x') ** Ty('y')


def test_PRO_init():
    assert list(PRO(0)) == []
    assert all(len(PRO(n)) == n for n in range(5))


def test_PRO_tensor():
    assert PRO(2) @ PRO(3) @ PRO(7) == PRO(12) == PRO(2).tensor(PRO(3), PRO(7))
    with raises(TypeError) as err:
        PRO(2) @ Ty('x')


def test_PRO_repr():
    assert repr((PRO(0), PRO(1))) == "(monoidal.PRO(0), monoidal.PRO(1))"


def test_PRO_hash():
    assert hash(PRO(0)) == hash(PRO(0)) != hash(PRO(1))


def test_PRO_to_tree():
    assert PRO(0).to_tree() == {'factory': 'monoidal.PRO', 'n': 0}
    assert PRO.from_tree(PRO(0).to_tree()) == PRO(0)


def test_PRO_str():
    assert str(PRO(2 * 3 * 7)) == "PRO(42)"


def test_PRO_getitem():
    assert PRO(42)[2: 4] == PRO(2)
    assert all(PRO(42)[i] == PRO(1) for i in range(42))


def test_Layer_init():
    with raises(ValueError):
        Layer(1, 2, 3, 4)


def test_Layer_getitem():
    f = Box('f', 'x', 'x')
    layer = Layer(Ty(), f, Ty())
    assert layer[1] == f and layer[0] == layer[2] == Ty()


def test_Diagram_init():
    with raises(TypeError) as err:
        Diagram((), 1, Ty('x'))
    with raises(TypeError) as err:
        Diagram((), Ty('x'), 1)
    with raises(TypeError) as err:
        Diagram((1, ), Ty('x'), Ty('x'))


def test_Diagram_eq():
    assert Diagram((), Ty('x'), Ty('x')) != Ty('x')
    assert Diagram((), Ty('x'), Ty('x')) == Id(Ty('x'))


def test_Diagram_iter():
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, y)
    g0 = Box('g', y @ y, x)
    g1 = g0.dagger()
    d = (f0 >> f1) @ Id(y @ x) >> g0 @ g1 >> f0 @ g0
    assert Id(x @ y @ x).then(*(left @ box @ right for left, box, right in d))\
        == d


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
    for depth, (left, box, right) in enumerate(diagram.inside):
        layer = Id(left) @ box @ Id(right)
        assert diagram[depth] == layer
        assert (diagram[-depth], ) == tuple(
            Id(left) @ box @ Id(right)
            for left, box, right in (diagram.inside[-depth], ))
        assert diagram[depth:depth] == Id(layer.dom)
        assert diagram[depth:] == Id(layer.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.inside[depth:]))
        assert diagram[:depth] == Id(diagram.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.inside[:depth]))
        assert diagram[depth: depth + 2] == Id(layer.dom).then(*(
            Id(left) @ box @ Id(right)
            for left, box, right in diagram.inside[depth: depth + 2]))


def test_Diagram_offsets():
    assert Diagram((), Ty('x'), Ty('x')).offsets == []


def test_Diagram_hash():
    assert {Id(Ty('x')): 42}[Id(Ty('x'))] == 42


def test_Diagram_str():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert str(Diagram((), x, x)) == "Id(x)"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert str(Diagram((Layer.cast(f0), ), x, y)) == "f0"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ z >> y @ f1"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ z >> y @ f1"


def test_Diagram_matmul():
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x', 'y'))
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x')).tensor(Id(Ty('y')))


def test_Diagram_interchange():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    d = f @ f.dagger()
    with raises(NotImplementedError):
        d.foliation().interchange(0, 1)
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
    with raises(AxiomError) as err:
        d.interchange(0, 2)
    assert str(err.value) == messages.INTERCHANGER_ERROR.format(f0, f1)
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
    assert str(err.value) == messages.NOT_CONNECTED.format(s0 >> s1)
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, x)
    assert f0.normal_form() == f0
    assert (f0 >> f1).normal_form() == f0 >> f1
    assert (Id(x) @ f1 >> f0 @ Id(x)).normal_form() == f0 @ f1
    assert (f0 @ f1).normal_form(left=True) == Id(x) @ f1 >> f0 @ Id(x)


def test_AxiomError():
    inside = (Layer.cast(Box('f', Ty('x'), Ty('y'))), )
    with raises(AxiomError) as err:
        Diagram(inside, Ty('x'), Ty('x'))
    with raises(AxiomError) as err:
        Diagram(inside, Ty('y'), Ty('y'))


def test_InterchangerError():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(AxiomError) as err:
        (f >> g).interchange(0, 1)
    assert str(err.value) == messages.INTERCHANGER_ERROR.format(f, g)


def test_spiral(n_cups=2):
    diagram = spiral(n_cups)
    unit, counit = diagram.boxes[0], diagram.boxes[n_cups + 1]
    spiral_nf = diagram.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n_cups] == unit


def test_Id_init():
    assert Id(Ty('x')) == Diagram.id(Ty('x'))


def test_Id_repr():
    assert repr(Id(Ty('x')))\
        == "monoidal.Diagram.id(monoidal.Ty(cat.Ob('x')))"


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
    assert f == Diagram((Layer.cast(f), ), Ty('x', 'y'), Ty('z')) and f != 'f'


def test_Functor_init():
    F = Functor({Ty('x'): Ty('y')}, {})
    assert F(Id(Ty('x'))) == Id(Ty('y'))


def test_Functor_repr():
    assert repr(Functor({Ty('x'): Ty('y')}, {})) ==\
        "monoidal.Functor("\
        "ob={monoidal.Ty(cat.Ob('x')): monoidal.Ty(cat.Ob('y'))}, ar={})"


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


def test_PRO_Functor():
    G = Functor(lambda x: x @ x, lambda f: f, cod=Category(PRO, Diagram))
    assert G(PRO(2)) == PRO(4)


def test_Functor_sum():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', x, y)
    F = Functor(ob={x: y, y: x}, ar={f: g[::-1], g: f[::-1]})
    assert F(f + g) == F(f) + F(g)


def test_Sum():
    x = Ty('x')
    f = Box('f', x, x)
    with raises(ValueError):
        Sum(())
    with raises(AxiomError):
        Sum((f, ), dom=Ty())
    with raises(AxiomError):
        f + Box('g', Ty(), x)
    assert Sum([f]) == f
    assert {Sum([f]): 42}[Sum([f])] == 42
    assert Id(x).then(*(3 * (f + f, ))) == sum(8 * [f >> f >> f])
    assert Id(Ty()).tensor(*(3 * (f + f, ))) == sum(8 * [f @ f @ f])


def test_Layer_merge_cup_cap():
    unit, counit = Box("unit", Ty(), 'x'), Box("counit", 'x', Ty())
    layer0, layer1 = Layer.cast(unit), Layer.cast(counit)
    with raises(AxiomError):
        layer0.merge(layer1)
    assert layer1.merge(layer0) == Layer(Ty(), unit, Ty(), counit, Ty())


def test_Layer_scalars():
    a, b = Box("a", Ty(), Ty()), Box("b", Ty(), Ty())
    assert Layer.cast(a).merge(Layer.cast(b)) == Layer(Ty(), a, Ty(), b, Ty())


def test_Diagram_from_callable():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    with raises(AxiomError):
        @Diagram.from_callable(y, x)
        def diagram(wire):
            return f(wire)
    with raises(AxiomError):
        @Diagram.from_callable(x, x)
        def diagram(wire):
            return f(wire)
    with raises(AxiomError):
        @Diagram.from_callable(x @ x, x)
        def diagram(left, right):
            return f(left, right)
    with raises(AxiomError):
        @Diagram.from_callable(x, x @ y)
        def diagram(wire):
            return wire, f(offset=0)
    with raises(TypeError):
        @Diagram.from_callable(x, y)
        def diagram(wire):
            return f(x)
