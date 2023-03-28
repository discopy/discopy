from pytest import raises

from discopy.rigid import *


def test_Ob_init():
    with raises(TypeError) as err:
        Ob('x', z='y')


def test_Ob_eq():
    assert Ob('a') == Ob('a').l.r and Ob('a') != 'a'


def test_Ob_hash():
    a = Ob('a')
    assert {a: 42}[a] == 42


def test_Ob_repr():
    assert repr(Ob('a', z=42)) == "rigid.Ob('a', z=42)"


def test_Ob_str():
    a = Ob('a')
    assert str(a) == "a" and str(a.r) == "a.r" and str(a.l) == "a.l"


def test_Ty_z():
    with raises(ValueError):
        Ty('x', 'y').z
    with raises(ValueError):
        Ty().z
    assert Ty('x').l.z == -1


def test_PRO_r():
    assert PRO(2).r == PRO(2)


def test_Diagram_cups():
    with raises(TypeError) as err:
        Diagram.cups('x', Ty('x'))
    with raises(TypeError) as err:
        Diagram.cups(Ty('x'), 'x')


def test_Diagram_caps():
    with raises(TypeError) as err:
        Diagram.caps('x', Ty('x'))
    with raises(TypeError) as err:
        Diagram.caps(Ty('x'), 'x')


def test_Diagram_normal_form():
    x = Ty('x')
    assert Id(x).transpose(left=True).normal_form() == Id(x.l)
    assert Id(x).transpose().normal_form() == Id(x.r)

    f = Box('f', Ty('a'), Ty('b') @ Ty('c'))
    assert f.normal_form() == f
    assert f.transpose().transpose(left=True).normal_form() == f
    assert f.transpose(left=True).transpose().normal_form() == f
    diagram = f\
        .transpose(left=True).transpose(left=True).transpose().transpose()
    assert diagram.normal_form() == f

    Eckmann_Hilton = Box('s0', Ty(), Ty()) @ Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        Eckmann_Hilton.normal_form()
    assert str(err.value) == messages.NOT_CONNECTED.format(Eckmann_Hilton)


def test_Cup_init():
    with raises(TypeError):
        Cup('x', Ty('y'))
    with raises(TypeError):
        Cup(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cup(t, t.r)
    with raises(ValueError) as err:
        Cup(Ty(), Ty())


def test_Cap_init():
    with raises(TypeError):
        Cap('x', Ty('y'))
    with raises(TypeError):
        Cap(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cap(t, t.l)
    with raises(ValueError) as err:
        Cap(Ty(), Ty())


def test_Cup_Cap_adjoint():
    n = Ty('n')
    assert Cap(n, n.l).l == Cup(n.l.l, n.l)
    assert Cap(n, n.l).r == Cup(n, n.r)
    assert Cup(n, n.r).l == Cap(n, n.l)
    assert Cup(n, n.r).r == Cap(n.r.r, n.r)


def test_AxiomError():
    n, s = Ty('n'), Ty('s')
    with raises(AxiomError) as err:
        Cup(n, n)
    with raises(AxiomError) as err:
        Cup(n, s)
    with raises(AxiomError) as err:
        Cup(n, n.l)
    with raises(AxiomError) as err:
        Cup(n, n.r).dagger()
    with raises(AxiomError) as err:
        Cap(n, n.l).dagger()
    with raises(AxiomError) as err:
        Cup(n, n.l.l)
    with raises(AxiomError) as err:
        Cap(n, n.l.l)


def test_id_adjoint():
    assert Id(Ty('n')).r == Id(Ty('n').r)
    assert Id(Ty('n')).l == Id(Ty('n').l)
    assert Id().l == Id() == Id().r


def test_sum_adjoint():
    x = Ty('x')
    two, boxes = Box('two', x, x), Box('boxes', x, x)
    two_boxes = two + boxes
    assert two_boxes.l == two.l + boxes.l
    assert two_boxes.l.r == two_boxes
