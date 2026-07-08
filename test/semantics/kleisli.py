# -*- coding: utf-8 -*-

from math import sqrt

from pytest import raises

from discopy import closed, markov, symmetric
from discopy.kleisli import (
    additive, multiplicative, Maybe, Nothing, Powerset, Subdistribution,
    Writer)
from discopy.kleisli.additive import token_passing
from discopy.kleisli.multiplicative import Channel, message_passing


MONADS = {
    Maybe: (42, Nothing),
    Powerset: (frozenset({1, 2}), frozenset()),
    Subdistribution: ({0: 0.5, 1: 0.25}, {}),
    Writer: (("log", 3), ), }

KLEISLI_ARROWS = {
    Maybe: (
        lambda a: Maybe.pure(a + 1),
        lambda a: Nothing if a % 2 else Maybe.pure(a // 2)),
    Powerset: (
        lambda a: frozenset({a, a + 1}),
        lambda a: frozenset() if a % 2 else frozenset({a // 2})),
    Subdistribution: (
        lambda a: {a: 0.5, a + 1: 0.5},
        lambda a: {} if a % 2 else {a // 2: 1.}),
    Writer: (
        lambda a: ("f", a + 1),
        lambda a: ("g", 2 * a)), }


def test_monad_laws():
    for monad, samples in MONADS.items():
        f, g = KLEISLI_ARROWS[monad]
        for a in (0, 1, 6):
            assert monad.allclose(monad.bind(monad.pure(a), f), f(a))
        for ma in samples:
            assert monad.allclose(monad.bind(ma, monad.pure), ma)
            lhs = monad.bind(monad.bind(ma, f), g)
            rhs = monad.bind(ma, lambda a: monad.bind(f(a), g))
            assert monad.allclose(lhs, rhs)


def test_transformation_components():
    x = (int, )
    for monad in MONADS:
        unit, mult = monad.unit, monad.mult
        assert unit(x).dom == x and unit(x).cod == monad(x)
        assert mult(x).dom == (monad >> monad)(x)
        assert mult(x).cod == monad(x)
    assert Maybe.unit(x)(42) == 42
    assert Maybe.mult(x)(Nothing) is Nothing
    assert Powerset.mult(x)(
        frozenset({frozenset({1}), frozenset({2, 3})})) == {1, 2, 3}
    assert Writer.mult(x)(("out", ("in", 5))) == ("outin", 5)


def test_double_strength():
    pairs = {
        Maybe: (2, 3),
        Powerset: (frozenset({1, 2}), frozenset({3})),
        Subdistribution: ({0: 0.5, 1: 0.5}, {2: 1.}),
        Writer: (("a", 1), ("b", 2)), }
    for monad, (ma, mb) in pairs.items():
        left = monad.double_strength(ma, mb)
        right = monad.double_strength(ma, mb, left=False)
        assert monad.commutative == monad.allclose(left, right)


def test_endofunctor():
    from discopy.python import function
    succ = function.Function(lambda n: n + 1, int, int)
    assert Maybe(int) == (object, )
    assert Maybe(succ)(41) == 42 and Maybe(succ)(Nothing) is Nothing
    assert Powerset(succ)(frozenset({1, 2})) == {2, 3}
    square = Powerset >> Powerset
    assert square(int) == (frozenset, )
    assert square(succ)(frozenset({frozenset({1})}))\
        == {frozenset({2})}


def test_channel():
    C = Channel[Maybe]
    half = C(lambda n: n // 2 if n % 2 == 0 else Nothing, int, int)
    assert (C.id(int) >> half)(4) == half(4) == (half >> C.id(int))(4)
    assert ((half >> half) >> half)(8) == (half >> (half >> half))(8) == 1
    assert (half >> half)(6) is Nothing
    assert C.lift(len, str, int)("abc") == 3
    with raises(ValueError):
        half >> Channel[Powerset].id(int)


def test_interchange():
    for monad in (Maybe, Powerset, Subdistribution, Writer):
        C = Channel[monad]
        f = C(lambda: monad.pure(1), (), int)
        g = C(lambda: monad.pure(2), (), int)
        if monad is Writer:
            f = C(lambda: ("f", 1), (), int)
            g = C(lambda: ("g", 2), (), int)
        f_first = f @ g
        g_first = C.id(()) @ g >> f @ C.id(int)
        assert f_first.dom == g_first.dom and f_first.cod == g_first.cod
        assert monad.commutative == monad.allclose(f_first(), g_first())


def test_copy_correlation():
    C = Channel[Subdistribution]
    coin = C(lambda: {0: 0.5, 1: 0.5}, (), int)
    assert (coin >> C.copy(int))() == {(0, 0): 0.5, (1, 1): 0.5}
    xor = C.lift(lambda a, b: a ^ b, (int, int), int)
    assert (coin >> C.copy(int) >> xor)() == {0: 1.0}


def test_permutation_dagger():
    C = Channel[Maybe]
    p = C.permutation([2, 0, 1], (int, bool, str))
    assert p.cod == (str, int, bool) and p(1, True, "x") == ("x", 1, True)
    assert (p >> p.dagger())(1, True, "x") == (1, True, "x")
    assert C.swap(int, bool).dagger()(True, 1) == (1, True)
    with raises(ValueError):
        C.id(int).dagger()


def test_curry_uncurry():
    monad = Subdistribution
    C = Channel[monad]
    add = C.lift(lambda a, b: a + b, (int, int), int)
    curried = add.curry()
    assert curried.uncurry()(1, 2) == {3: 1.}
    assert add.curry(left=False).uncurry(left=False)(1, 2) == {3: 1.}
    assert monad.bind(curried(1), lambda f: f(2)) == {3: 1.}


def golden_functor(cod):
    x = symmetric.Ty('x')
    f = symmetric.Box('f', x, x @ x)
    g = symmetric.Box('g', symmetric.Ty(), x)
    functor = symmetric.Functor(
        ob={x: (float, )},
        ar={f: lambda x=1.: (x, 1 + 1. / x), g: lambda: (1 + sqrt(5)) / 2},
        cod=cod)
    return functor, f, g


def test_golden_ratio_trace():
    functor, f, g = golden_functor(Channel[Maybe])
    assert functor(f.trace())() == functor(g)() == (1 + sqrt(5)) / 2


def test_golden_ratio_message_passing():
    functor, f, g = golden_functor(Channel[Maybe])
    assert message_passing(functor, f.trace())() == functor(g)()


def test_message_passing_acyclic():
    x = markov.Ty('x')
    c, h = markov.Box('c', markov.Ty(), x), markov.Box('h', x, x)
    diagram = c >> markov.Copy(x) >> h @ x >> markov.Swap(x, x)\
        >> x @ (h >> markov.Copy(x, 0))
    ars = {
        Maybe: {c: lambda: 5, h: lambda a: a + 1},
        Powerset: {
            c: lambda: frozenset({0, 5}),
            h: lambda a: frozenset({a + 1})},
        Subdistribution: {
            c: lambda: {0: 0.5, 5: 0.5}, h: lambda a: {a + 1: 1.}},
        Writer: {c: lambda: ("c", 5), h: lambda a: ("h", a + 1)}, }
    for monad, ar in ars.items():
        functor = markov.Functor({x: int}, ar, cod=Channel[monad])
        assert monad.allclose(
            functor(diagram)(), message_passing(functor, diagram)())


def test_hypergraph_well_definedness():
    x = markov.Ty('x')
    a, b = markov.Box('a', x, x), markov.Box('b', x, x)
    a_first = a @ b
    b_first = x @ b >> a @ x
    assert a_first.to_hypergraph() == b_first.to_hypergraph()
    ars = {
        Subdistribution: {
            a: lambda v: {v + 1: 1.}, b: lambda v: {2 * v: 1.}},
        Writer: {
            a: lambda v: ("a", v + 1), b: lambda v: ("b", 2 * v)}, }
    for monad, ar in ars.items():
        functor = markov.Functor({x: int}, ar, cod=Channel[monad])
        results = [
            functor(a_first)(1, 2), message_passing(functor, a_first)(1, 2),
            functor(b_first)(1, 2), message_passing(functor, b_first)(1, 2)]
        assert monad.allclose(results[0], results[1])
        assert monad.allclose(results[2], results[3])
        assert monad.commutative == monad.allclose(results[0], results[2])


def lambda_functor(monad, constants):
    X = closed.Ty('X')
    boxes = {closed.Constant(name, typ): value
             for name, typ, value in constants(X)}
    functor = multiplicative.Functor(
        {X: int}, boxes, cod=Channel[monad])
    return X, {box.name: box for box in boxes}, functor


def test_lambda_call_by_value():
    monad = Subdistribution
    X, consts, functor = lambda_functor(monad, lambda X: [
        ('coin', X, lambda: {0: 0.5, 1: 0.5}),
        ('add', X >> (X >> X), lambda: monad.pure(
            lambda a: monad.pure(lambda b: monad.pure(a + b))))])
    coin, add = consts['coin'], consts['add']
    term = X(lambda v: add(v)(v))(coin)
    assert functor(term)() == {0: 0.5, 2: 0.5}
    assert message_passing(functor, term.eval())() == {0: 0.5, 2: 0.5}


def test_lambda_maybe():
    monad = Maybe
    X, consts, functor = lambda_functor(monad, lambda X: [
        ('one', X, lambda: 1),
        ('zero', X, lambda: 0),
        ('div', X >> (X >> X), lambda: monad.pure(
            lambda a: monad.pure(
                lambda b: Nothing if b == 0 else a // b)))])
    one, zero, div = consts['one'], consts['zero'], consts['div']
    term = X(lambda v: div(one)(v))(zero)
    assert functor(term)() is Nothing
    assert message_passing(functor, term.eval())() is Nothing
    happy = X(lambda v: div(v)(one))(one)
    assert functor(happy)() == 1


def test_lambda_writer_order():
    monad = Writer
    X, consts, functor = lambda_functor(monad, lambda X: [
        ('c', X, lambda: ("c", 0)),
        ('q', X >> X, lambda: ("", lambda a: ("q", a))),
        ('p', X >> X, lambda: ("", lambda a: ("p", a)))])
    c, q, p = consts['c'], consts['q'], consts['p']
    term = p(q(c))
    assert functor(term)() == ("cqp", 0)
    assert message_passing(functor, term.eval())() == ("cqp", 0)


def countdown_setup(monad):
    x = symmetric.Ty('x')
    count = symmetric.Box('count', x @ x, x @ x)
    functor = symmetric.Functor(
        {x: int},
        {count: lambda n, tag: monad.pure(
            (n, 0) if n <= 0 else (n - 1, 1))},
        cod=additive.Channel[monad])
    return functor, count


def test_additive_countdown():
    functor, count = countdown_setup(Maybe)
    diagram = count.trace()
    assert functor(diagram)(3) == (0, 0)
    assert token_passing(functor, diagram)(3) == (0, 0)


def test_additive_deterministic_cycle():
    monad = Maybe
    x = symmetric.Ty('x')
    loop = symmetric.Box('loop', x @ x, x @ x)
    functor = symmetric.Functor(
        {x: int}, {loop: lambda n, tag: (n, 1)},
        cod=additive.Channel[monad])
    assert functor(loop.trace())(5) is Nothing
    assert token_passing(functor, loop.trace())(5) is Nothing


def test_additive_reachability():
    monad = Powerset
    graph = {0: (1, ), 1: (0, 2), 2: (3, 4), 3: (), 4: ()}

    def step(v, tag):
        if not graph[v]:
            return frozenset({(v, 0)})
        return frozenset({(s, 1) for s in graph[v]})
    x = symmetric.Ty('x')
    box = symmetric.Box('step', x @ x, x @ x)
    functor = symmetric.Functor(
        {x: int}, {box: step}, cod=additive.Channel[monad])
    assert functor(box.trace())(0) == {(3, 0), (4, 0)}
    assert token_passing(functor, box.trace())(0) == {(3, 0), (4, 0)}


def test_additive_geometric():
    monad = Subdistribution
    x = symmetric.Ty('x')
    geo = symmetric.Box('geo', x @ x, x @ x)
    functor = symmetric.Functor(
        {x: int}, {geo: lambda n, tag: {(n, 0): 0.5, (n + 1, 1): 0.5}},
        cod=additive.Channel[monad])
    tol = 1e-9
    recursive = functor(geo.trace())(0)
    tokens = token_passing(functor, geo.trace(), tol=tol)(0)
    assert abs(sum(recursive.values()) - 1.) <= 1e-6
    for k in range(10):
        assert abs(recursive[(k, 0)] - 2. ** -(k + 1)) <= 1e-6
    assert monad.allclose(recursive, tokens, 1e-6)


def test_additive_swap_merge_inject():
    C = additive.Channel[Powerset]
    swap = C.swap((int, ), (str, ))
    assert swap("x", 1) == {("x", 0)} and swap(5, 0) == {(5, 1)}
    assert swap.dagger()(5, 1) == {(5, 0)}
    merge = C.merge((int, ), 2)
    assert merge(5, 1) == {(5, 0)}
    assert C.inject((int, str), 1)("x") == {("x", 1)}
    with raises(ValueError):
        C.id(int).dagger()


def test_additive_trace_requires_additive_monad():
    C = additive.Channel[Writer]
    f = C(lambda n, tag: ("", (n, tag)), (int, int), (int, int))
    with raises(ValueError):
        f.trace()
