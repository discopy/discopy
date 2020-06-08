from pytest import raises
from discopy.rigid import *


def test_Ob_init():
    with raises(TypeError) as err:
        Ob('x', z='y')
    assert str(err.value) == messages.type_err(int, 'y')
    assert cat.Ob('x') == Ob('x')


def test_Ob_eq():
    assert Ob('a') == Ob('a').l.r and Ob('a') != 'a'


def test_Ob_hash():
    a = Ob('a')
    assert {a: 42}[a] == 42


def test_Ob_repr():
    assert repr(Ob('a', z=42)) == "Ob('a', z=42)"


def test_Ob_str():
    a = Ob('a')
    assert str(a) == "a" and str(a.r) == "a.r" and str(a.l) == "a.l"


def test_PRO_r():
    assert PRO(2).r == PRO(2)


def test_Diagram_cups():
    with raises(TypeError) as err:
        Diagram.cups('x', Ty('x'))
    assert str(err.value) == messages.type_err(Ty, 'x')
    with raises(TypeError) as err:
        Diagram.cups(Ty('x'), 'x')
    assert str(err.value) == messages.type_err(Ty, 'x')


def test_Diagram_caps():
    with raises(TypeError) as err:
        Diagram.caps('x', Ty('x'))
    assert str(err.value) == messages.type_err(Ty, 'x')
    with raises(TypeError) as err:
        Diagram.caps(Ty('x'), 'x')
    assert str(err.value) == messages.type_err(Ty, 'x')


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
    assert str(err.value) == messages.is_not_connected(Eckmann_Hilton)


def test_Cup_init():
    with raises(TypeError):
        Cup('x', Ty('y'))
    with raises(TypeError):
        Cup(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cup(t, t.r)
    assert str(err.value) == messages.cup_vs_cups(t, t.r)
    with raises(ValueError) as err:
        Cup(Ty(), Ty())
    assert str(err.value) == messages.cup_vs_cups(Ty(), Ty().l)
    with raises(NotImplementedError):
        Cup(Ty('n'), Ty('n').l)
    with raises(NotImplementedError):
        Cup(Ty('n'), Ty('n').r).dagger()


def test_Cap_init():
    with raises(TypeError):
        Cap('x', Ty('y'))
    with raises(TypeError):
        Cap(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cap(t, t.l)
    assert str(err.value) == messages.cap_vs_caps(t, t.l)
    with raises(ValueError) as err:
        Cap(Ty(), Ty())
    assert str(err.value) == messages.cap_vs_caps(Ty(), Ty())
    with raises(NotImplementedError):
        Cap(Ty('n'), Ty('n').r)
    with raises(NotImplementedError):
        Cap(Ty('n'), Ty('n').l).dagger()


def test_AxiomError():
    n, s = Ty('n'), Ty('s')
    with raises(AxiomError) as err:
        Cup(n, n)
    assert str(err.value) == messages.are_not_adjoints(n, n)
    with raises(AxiomError) as err:
        Cup(n, s)
    assert str(err.value) == messages.are_not_adjoints(n, s)
    with raises(AxiomError) as err:
        Cup(n, n.l.l)
    assert str(err.value) == messages.are_not_adjoints(n, n.l.l)
    with raises(AxiomError) as err:
        Cap(n, n.l.l)
    assert str(err.value) == messages.are_not_adjoints(n, n.l.l)


def test_Functor_call():
    F = Functor({}, {})
    with raises(TypeError):
        F(F)
