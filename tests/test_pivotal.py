from pytest import raises
from discopy.pivotal import *


def test_Diagram_normal_form():
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    twist = Cap(x, x.r) @ Id(x.r.r) >> Id(x) @ Cup(x.r, x.r.r)
    assert twist.dom != twist.cod and twist.normal_form() == twist
    d = Cap(x, x.l) @ unit >> counit @ Cup(x.l, x)
    assert d.normal_form(left=True) == unit >> counit
    assert d.dagger().normal_form() == counit.dagger() >> unit.dagger()
    a, b, c = Ty('a'), Ty('b'), Ty('c')
    f = Box('f', a, b @ c)
    assert f.normal_form() == f
    transpose_rl = f.transpose_r().transpose_l()
    assert transpose_rl.normal_form() == f
    transpose_lr = f.transpose_l().transpose_r()
    assert transpose_lr.normal_form() == f
    more_complicated = f
    more_complicated = more_complicated.transpose_l().transpose_l()
    more_complicated = more_complicated.transpose_r().transpose_r()
    assert more_complicated.normal_form() == f
    Eckmann_Hilton = Box('s0', Ty(), Ty()) @ Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        Eckmann_Hilton.normal_form()
    assert str(err.value) == config.Msg.is_not_connected(Eckmann_Hilton)


def test_Diagram_draw():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    graph, positions, labels = f.transpose_l().draw(_test=True)
    assert sorted(labels.items()) == [
        ('box_1', 'f'),
        ('input_0', 'y.l'),
        ('output_0', 'x.l'),
    ]
    assert sorted(positions.items()) == [
        ('box_1', (-0.5, 2)),
        ('input_0', (-0.5, 4)),
        ('output_0', (-0.5, 0)),
        ('wire_0_0', (-1.0, 3)),
        ('wire_0_1', (0.0, 3)),
        ('wire_1_0', (-1.5, 2)),
        ('wire_1_2', (0.5, 2)),
        ('wire_2_0', (-1.0, 1)),
        ('wire_2_1', (0.0, 1)),
    ]
    assert sorted(graph.edges()) == [
        ('box_1', 'wire_2_0'),
        ('input_0', 'wire_0_0'),
        ('wire_0_0', 'wire_1_0'),
        ('wire_0_1', 'box_1'),
        ('wire_0_1', 'wire_1_2'),
        ('wire_1_0', 'wire_2_0'),
        ('wire_1_2', 'wire_2_1'),
        ('wire_2_1', 'output_0'),
    ]


def test_Cup_init():
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cup(t, t.l)
    assert str(err.value) == config.Msg.cup_vs_cups(t, t.l)
    with raises(ValueError) as err:
        Cup(Ty(), Ty().l)
    assert str(err.value) == config.Msg.cup_vs_cups(Ty(), Ty().l)


def test_Cap_init():
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cap(t, t.l)
    assert str(err.value) == config.Msg.cap_vs_caps(t, t.l)
    with raises(ValueError) as err:
        Cap(Ty(), Ty().l)
    assert str(err.value) == config.Msg.cap_vs_caps(Ty(), Ty().l)


def test_AxiomError():
    n, s = Ty('n'), Ty('s')
    with raises(AxiomError) as err:
        Cup(n, n)
        assert str(err.value) == config.Msg.are_not_adjoints(n, n)
    with raises(AxiomError) as err:
        Cup(n, s)
        assert str(err.value) == config.Msg.are_not_adjoints(n, s)
    with raises(AxiomError) as err:
        Cup(n, n.l.l)
        assert str(err.value) == config.Msg.are_not_adjoints(n, n.l.l)
