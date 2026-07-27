# -*- coding: utf-8 -*-

from pytest import raises

from discopy.cat import Transformation
from discopy.utils import AxiomError
from discopy.python.function import Function, EndoFunctor
from discopy.kleisli.monad import (
    Monad, Maybe, Powerset, Subdistribution, make_monad)
from discopy.kleisli.channel import Channel
from discopy.kleisli import multiplicative
from discopy.kleisli.multiplicative import Row


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


def test_make_monad():
    identity = make_monad(
        "Identity", lambda X: X, lambda f: f,
        lambda X: Function.id(X), lambda X: Function.id(X))
    assert identity(int) == (int, )
    assert identity.unit(int)(42) == identity.mult(int)(42) == 42


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


def test_Row():
    assert Row(1, 2) == eval(repr(Row(1, 2))) == Row(1, 2)
    assert Row(1, 2) != Row(2, 1) and Row(1, 2) != (1, 2)
    assert hash(Row(1, 2)) == hash(Row(1, 2))


def test_pack():
    assert multiplicative.pack(()) is Row
    assert multiplicative.pack((int, )) is int
    assert multiplicative.pack((int, str)) is Row
    assert multiplicative.pack_value(()) == Row()
    assert multiplicative.pack_value((1, )) == 1
    assert multiplicative.pack_value((1, 2)) == Row(1, 2)
    assert multiplicative.unpack_value(Row()) == ()
    assert multiplicative.unpack_value(1) == (1, )
    assert multiplicative.unpack_value(Row(1, 2)) == (1, 2)


def test_multiplicative_Channel_tensor():
    Safe = multiplicative.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    increment = Safe(lambda x: x + 1, int, int)

    parallel = half @ increment
    assert parallel.dom == (int, int) == parallel.cod
    assert parallel(4, 10) == (2, 11)
    assert parallel(3, 10) is None

    triple = half @ increment @ half
    assert triple(4, 10, 6) == (2, 11, 3)
    assert triple(3, 10, 6) is None

    left_whiskered = (int, ) @ half
    assert left_whiskered.dom == (int, int) == left_whiskered.cod
    assert left_whiskered(99, 4) == (99, 2)

    with raises(TypeError):
        half.tensor("not-a-channel")


def test_multiplicative_Channel_identity_and_composition():
    Safe = multiplicative.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)

    identity = Safe.id((int, ))
    assert identity(4) == 4
    assert (identity >> half)(4) == half(4) == (half >> identity)(4)


def test_multiplicative_Channel_copy_discard():
    Nondet = multiplicative.Channel[Powerset]

    copy = Nondet.copy((int, ))
    assert copy(3) == frozenset({(3, 3)})

    copy_two_wires = Nondet.copy((int, int))
    assert copy_two_wires(1, 2) == frozenset({(1, 2, 1, 2)})

    discard = Nondet.discard((int, ))
    assert discard(3) == frozenset({()})


def test_multiplicative_Channel_repr():
    Safe = multiplicative.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    assert repr(half) == f"multiplicative.Channel[Maybe]("\
        f"{half.inside!r}, dom={half.dom!r}, cod={half.cod!r})"


def swapped_tensor(f, g):
    """
    The tensor of two channels biased the other way round from
    :meth:`~discopy.kleisli.multiplicative.Channel.tensor`, i.e. running
    ``g``'s effects before ``f``'s. Agreeing with ``f @ g`` for every
    ``f, g`` is exactly the condition for the Kleisli category to be
    monoidal, see the module docstring of
    :mod:`discopy.kleisli.multiplicative`.
    """
    return g.left_whisker(f.dom).then(f.right_whisker(g.cod))


def make_writer(append, empty):
    """
    The writer monad over a monoid ``(append, empty)``, representing a
    computation together with a log: not part of the public API, this is
    only used here to exhibit a *non-commutative* monad, e.g. taking
    ``append`` to be string concatenation.
    """
    return make_monad(
        "Writer",
        ob_map=lambda X: Row,
        lift=lambda f: Function(
            lambda xw: Row(f(xw.values[0]), xw.values[1]), Row, Row),
        unit_map=lambda X: Function(lambda x: Row(x, empty), X, Row),
        mult_map=lambda X: Function(
            lambda mmx: Row(
                mmx.values[0].values[0],
                append(mmx.values[1], mmx.values[0].values[1])),
            Row, Row))


def test_interchange_holds_for_commutative_monads():
    Safe = multiplicative.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    increment = Safe(lambda x: x + 1, int, int)
    assert (half @ increment)(4, 10) == swapped_tensor(half, increment)(4, 10)
    assert (half @ increment)(3, 10) == swapped_tensor(half, increment)(3, 10)

    Nondet = multiplicative.Channel[Powerset]

    def divisors(n: int) -> frozenset:
        return frozenset(d for d in range(1, n + 1) if n % d == 0)

    def successors(n: int) -> frozenset:
        return frozenset({n, n + 1})

    d, s = Nondet(divisors, int, int), Nondet(successors, int, int)
    assert (d @ s)(6, 3) == swapped_tensor(d, s)(6, 3)


def test_interchange_fails_for_noncommutative_monad():
    Writer = make_writer(lambda w1, w2: w1 + w2, "")
    Log = multiplicative.Channel[Writer]
    tag_a = Log(lambda x: Row(x, "a"), int, int)
    tag_b = Log(lambda x: Row(x, "b"), int, int)

    f_first, g_first = tag_a @ tag_b, swapped_tensor(tag_a, tag_b)
    assert f_first(1, 2) == Row((1, 2), "ab")
    assert g_first(1, 2) == Row((1, 2), "ba")
    assert f_first(1, 2) != g_first(1, 2)
