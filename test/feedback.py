from pytest import raises

from discopy import *
from discopy.feedback import *


def test_delayed_monoid():
    from discopy.abc import DelayedMonoid
    x = Ty('x')
    assert isinstance(x, DelayedMonoid)
    assert x.d == Ty(Wire('x', time_step=1))
    assert x.d.d == Ty(Wire('x', time_step=2))


def test_invalid_inputs():
    with raises(NotImplementedError):
        Wire('x', time_step=-1)
    with raises(ValueError):
        HeadOb(Wire('x').d)
    with raises(ValueError):
        TailOb(Wire('x').d)


def test_functor_python_stream():
    x = Ty('x')
    zero, wait = Box('zero', Ty(), x), Diagram.wait(x)
    F = Functor(
        ob_map={x: int},
        ar_map={zero: lambda: 0},
        cod=stream.Stream[python.Function])
    assert F(wait @ zero).unroll(2).now(1, 2, 3) == (0, ) + (1, 0) + (2, 0) + (3, )


def test_Permutation_d():
    x, y, z = map(Ty, "xyz")
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert perm.d == Permutation((x @ y @ z).d, [2, 0, 1])
    assert perm.d.d == Permutation((x @ y @ z).d.d, [2, 0, 1])
    assert (perm >> Swap(z, x) @ y).d\
        == perm.d >> Swap(z, x).d @ y.d
