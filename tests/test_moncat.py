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


def test_Ty_repr():
    assert repr(Ty('x', 'y')) == "Ty('x', 'y')"


def test_Ty_str():
    str(Ty('x')) == 'x'


def test_Ty_getitem():
    assert Ty('x', 'y', 'z')[:1] == Ty('x')


def test_Ty_pow():
    assert Ty('x') ** 42 == Ty('x') ** 21 @ Ty('x') ** 21


def test_Diagram_init():
    with raises(TypeError) as err:
        Diagram('x', Ty('x'), [], [])
    assert str(err.value) == config.Msg.type_err(Ty, 'x')
    with raises(ValueError) as err:
        Diagram(Ty('x'), Ty('x'), [], [1])
    assert "Boxes and offsets must have the same length." in str(err.value)
    with raises(TypeError) as err:
        Diagram(Ty('x'), Ty('x'), [1], [1])
    assert str(err.value) == config.Msg.type_err(Diagram, 1)
    with raises(TypeError) as err:
        Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [Ty('x')])
    assert str(err.value) == config.Msg.type_err(int, Ty('x'))


def test_Diagram_offsets():
    assert Diagram(Ty('x'), Ty('x'), [], []).offsets == []


def test_Diagram_repr():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert repr(Diagram(x, x, [], [])) == "Id(Ty('x'))"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert "Diagram(dom=Ty('x'), cod=Ty('y')" in repr(Diagram(x, y, [f0], [0]))
    assert "offsets=[0]" in repr(Diagram(x, y, [f0], [0]))
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
    with raises(InterchangerError) as err:
        d.interchange(0, 2)
    assert str(err.value) == str(InterchangerError(f0, f1))
    assert d.interchange(2, 0) == Id(x) @ f1 >> f0 @ Id(x) >> f1 @ f0


def test_Diagram_normal_form():
    assert Id(Ty()).normal_form() == Id(Ty())
    assert Id(Ty('x', 'y')).normal_form() == Id(Ty('x', 'y'))
    s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        (s0 >> s1).normal_form()
    assert str(err.value) == config.Msg.is_not_connected(s0 >> s1)
    x, y = Ty('x'), Ty('y')
    f0, f1 = Box('f0', x, y), Box('f1', y, x)
    assert f0.normal_form() == f0
    assert (f0 >> f1).normal_form() == f0 >> f1
    assert (Id(x) @ f1 >> f0 @ Id(x)).normal_form() == f0 @ f1
    assert (f0 @ f1).normal_form(left=True) == Id(x) @ f1 >> f0 @ Id(x)


def test_AxiomError():
    with raises(AxiomError) as err:
        Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [0])
    assert str(err.value) == config.Msg.does_not_compose(Ty('y'), Ty('x'))
    with raises(AxiomError) as err:
        Diagram(Ty('y'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])
    assert str(err.value) == config.Msg.does_not_compose(Ty('y'), Ty('x'))


def test_InterchangerError():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(InterchangerError) as err:
        (f >> g).interchange(0, 1)
    assert str(err.value) == str(InterchangerError(f, g))


def build_spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    x = Ty('x')  # pylint: disable=invalid-name
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result


def test_spiral(n=2):
    spiral = build_spiral(n)
    unit, counit = Box('unit', Ty(), Ty('x')), Box('counit', Ty('x'), Ty())
    assert spiral.boxes[0] == unit and spiral.boxes[n + 1] == counit
    spiral_nf = spiral.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n] == unit


def test_Diagram_draw():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    diagram = Box('f0', x, y) @ Box('f1', z, w)
    graph, positions, labels = diagram.draw(_test=True)
    assert sorted(labels.items()) == [
        ('box_0', 'f0'),
        ('box_1', 'f1'),
        ('input_0', 'x'),
        ('input_1', 'z'),
        ('output_0', 'y'),
        ('output_1', 'w')
    ]
    assert sorted(positions.items()) == [
        ('box_0', (-1.0, 2)),
        ('box_1', (0.0, 1)),
        ('input_0', (-1.0, 3)),
        ('input_1', (0.0, 3)),
        ('output_0', (-1.0, 0)),
        ('output_1', (0.0, 0)),
        ('wire_0_1', (0.0, 2)),
        ('wire_1_0', (-1.0, 1))
    ]
    assert sorted(graph.edges()) == [
        ('box_0', 'wire_1_0'),
        ('box_1', 'output_1'),
        ('input_0', 'box_0'),
        ('input_1', 'wire_0_1'),
        ('wire_0_1', 'box_1'),
        ('wire_1_0', 'output_0')
    ]
