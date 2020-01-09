from pytest import raises
from discopy.rigidcat import *


def test_Diagram_normal_form():
    x = Ty('x')
    assert Id(x).transpose_l().normal_form() == Id(x.l)
    assert Id(x).transpose_r().normal_form() == Id(x.r)

    f = Box('f', Ty('a'), Ty('b') @ Ty('c'))
    assert f.normal_form() == f
    assert f.transpose_r().transpose_l().normal_form() == f
    assert f.transpose_l().transpose_r().normal_form() == f
    diagram = f.transpose_l().transpose_l().transpose_r().transpose_r()
    assert diagram.normal_form() == f

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
        Cup(t, t.r)
    assert str(err.value) == config.Msg.cup_vs_cups(t, t.r)
    with raises(ValueError) as err:
        Cup(Ty(), Ty())
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
