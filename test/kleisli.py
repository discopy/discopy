# -*- coding: utf-8 -*-

from pytest import raises

from discopy.cat import Transformation
from discopy.utils import AxiomError
from discopy.python.function import Function, EndoFunctor
from discopy.kleisli.monad import Monad, Maybe, Powerset, Subdistribution
from discopy.kleisli.channel import Channel


def test_EndoFunctor():
    assert EndoFunctor.id()(int) == (int, )

    listing = EndoFunctor(
        lambda X: (list[X[0]], ),
        lambda f: Function(
            lambda xs: [f(x) for x in xs], list[f.dom[0]], list[f.cod[0]]))
    increment = Function(lambda x: x + 1, int, int)

    assert listing(int) == (list[int], )
    assert listing(increment)([1, 2, 3]) == [2, 3, 4]
    assert listing.then(listing)(int) == listing(listing(int))


def test_Monad_type_errors():
    with raises(TypeError):
        Monad("bad", "not-a-functor", Maybe.unit, Maybe.mult)
    with raises(TypeError):
        Monad("bad", Maybe.functor, "not-a-transformation", Maybe.mult)
    with raises(TypeError):
        Monad("bad", Maybe.functor, Maybe.unit, "not-a-transformation")


def test_Monad_repr():
    assert repr(Maybe) == "Monad('Maybe')"
    assert str(Maybe) == Maybe.__name__ == "Maybe"


def unit_laws(monad: Monad, X: type, values: list):
    """ Check the unit laws of a monad at a type ``X`` on some ``values``. """
    MX = monad(X)
    for value in values:
        assert monad.mult(X)(monad.unit(MX)(value)) == value
        assert monad.mult(X)(monad.functor(monad.unit(X))(value)) == value


def test_maybe_laws():
    unit_laws(Maybe, int, [0, 1, -5, None])
    lifted = Maybe.functor(Function(lambda x: x + 1, int, int))
    assert lifted(1) == 2 and lifted(None) is None


def test_powerset_laws():
    unit_laws(Powerset, int, [
        frozenset(), frozenset({1}), frozenset({1, 2, 3})])
    lifted = Powerset.functor(Function(lambda x: x % 2, int, int))
    assert lifted(frozenset({1, 2, 3, 4})) == frozenset({0, 1})


def test_subdistribution_laws():
    unit_laws(Subdistribution, int, [
        frozenset(), frozenset({(1, 1.)}),
        frozenset({(1, .5), (2, .5)})])
    lifted = Subdistribution.functor(Function(lambda x: x % 2, int, int))
    assert lifted(frozenset({(1, .5), (3, .5)})) == frozenset({(1, 1.)})


def test_associativity():
    mx = frozenset({frozenset({1, 2}), frozenset({3})})
    lhs = Powerset.mult(int)(
        Powerset.mult(Powerset(int))(frozenset({mx})))
    rhs = Powerset.mult(int)(
        Powerset.functor(Powerset.mult(int))(frozenset({mx})))
    assert lhs == rhs == frozenset({1, 2, 3})


def test_Channel_maybe():
    Safe = Channel[Maybe]
    assert Safe is Channel[Maybe]
    assert Safe.monad is Maybe

    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    increment = Safe(lambda x: x + 1, int, int)

    assert (half >> increment)(4) == 3
    assert (half >> increment)(3) is None

    identity = Safe.id(int)
    assert identity(5) == 5
    assert (identity >> half)(4) == half(4) == (half >> identity)(4)

    with raises(TypeError):
        half.then("not-a-channel")

    Nondet = Channel[Powerset]
    other = Nondet(lambda x: frozenset({x}), int, int)
    with raises(AxiomError):  # Mismatched monads, caught at Function level.
        half.then(other)


def test_Channel_powerset():
    Nondet = Channel[Powerset]

    def divisors(n: int) -> frozenset:
        return frozenset(d for d in range(1, n + 1) if n % d == 0)

    def successors(n: int) -> frozenset:
        return frozenset({n, n + 1})

    d, s = Nondet(divisors, int, int), Nondet(successors, int, int)
    expected = frozenset().union(*(successors(n) for n in divisors(6)))
    assert (d >> s)(6) == expected


def test_Channel_repr():
    Safe = Channel[Maybe]
    increment = Safe(lambda x: x + 1, int, int)
    assert repr(increment) == f"Channel[Maybe]("\
        f"{increment.inside!r}, dom={int!r}, cod={int!r})"


def test_Transformation_reuse():
    assert isinstance(Maybe.unit, Transformation)
    assert isinstance(Maybe.mult, Transformation)
