# -*- coding: utf-8 -*-

from pytest import approx, raises

from discopy.cat import Transformation
from discopy.utils import AxiomError
from discopy.python.function import Function, EndoFunctor
from discopy.python.additive import Function as AdditiveFunction
from discopy.kleisli.monad import (
    Monad, Maybe, Powerset, Subdistribution, Seed,
    make_monad, make_state, merge, sample)
from discopy.kleisli.channel import Channel
from discopy.kleisli import additive, multiplicative, token
from discopy.kleisli.additive import Tagged
from discopy.kleisli.multiplicative import Row
from discopy.tensor import Dim, Tensor
from discopy.closed import Box as ClosedBox, Ty, Variable


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
    assert repr(increment) == f"kleisli.channel.Channel[Maybe]("\
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
    assert repr(half) == f"kleisli.multiplicative.Channel[Maybe]("\
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


def stochastic_tensor(array, n: int, m: int) -> Tensor:
    """ A stochastic matrix from ``n`` outcomes to ``m``, as a Tensor. """
    return Tensor[float](array, Dim(n), Dim(m))


def stochastic_channel(tensor: Tensor) -> multiplicative.Channel:
    """
    The :class:`Subdistribution`-valued channel reading its conditional
    probabilities off a stochastic ``tensor``, used below to cross-check
    Kleisli composition and tensor against matrix multiplication and the
    Kronecker product.
    """
    Prob = multiplicative.Channel[Subdistribution]
    m = tensor.cod.inside[0]

    def inside(x):
        return frozenset(
            (y, float(tensor.array[x, y]))
            for y in range(m) if tensor.array[x, y])
    return Prob(inside, int, int)


def test_multiplicative_tensor_contraction():
    """
    Kleisli composition and tensor of subdistribution channels agree with
    matrix multiplication and the Kronecker product of the corresponding
    stochastic matrices, i.e. probabilistic Kleisli channels contract like
    tensor networks on small enough models -- the stress-case suggested on
    issue #374.
    """
    xy = stochastic_tensor([[.5, .5, 0.], [0., .25, .75]], 2, 3)
    yz = stochastic_tensor([[1., 0.], [.5, .5], [0., 1.]], 3, 2)
    f, g = stochastic_channel(xy), stochastic_channel(yz)
    composite = stochastic_channel(xy >> yz)

    for x in range(2):
        assert (f >> g)(x) == composite(x)

    identity = stochastic_tensor([[1., 0.], [0., 1.]], 2, 2)
    h = stochastic_channel(identity)
    parallel = xy @ identity

    for x in range(2):
        for x_ in range(2):
            expected = frozenset(
                ((y, y_), float(parallel.array[x, x_, y, y_]))
                for y in range(3) for y_ in range(2)
                if parallel.array[x, x_, y, y_])
            assert (f @ h)(x, x_) == expected


def tick(value):
    """ The state-monad computation returning ``value`` and ticking a log. """
    return lambda state: (value, state + 1)


def run(computation, states: list) -> list:
    """ Run a state-monad ``computation`` on each state, for equality. """
    return [computation(state) for state in states]


def test_state_laws():
    State = make_state(int)
    states = [0, 1, 42]
    for value in [tick("egg"), tick("yolk")]:
        assert run(State.mult(str)(State.unit(State(str))(value)), states)\
            == run(value, states)\
            == run(State.mult(str)(State.functor(State.unit(str))(value)),
                   states)
    lifted = State.functor(Function(lambda x: x + "s", str, str))
    assert lifted(tick("egg"))(0) == ("eggs", 1)


def test_state_is_not_commutative():
    """
    The state monad is not commutative: whiskering on the left and on the
    right of a channel do not commute, so the two biased tensors differ.
    """
    Stateful = multiplicative.Channel[make_state(str)]
    tag_a = Stateful(lambda x: lambda s: (x, s + "a"), int, int)
    tag_b = Stateful(lambda x: lambda s: (x, s + "b"), int, int)

    assert (tag_a @ tag_b)(1, 2)("") == ((1, 2), "ab")
    assert swapped_tensor(tag_a, tag_b)(1, 2)("") == ((1, 2), "ba")


def empirical(computation, seed: int, size: int) -> frozenset:
    """
    The empirical subdistribution of ``size`` samples drawn by a ``Seed``
    computation, i.e. the outcomes with their observed frequency.
    """
    counts = {}
    for _ in range(size):
        outcome, seed = computation(seed)
        counts[outcome] = counts.get(outcome, 0) + 1
    return frozenset((x, n / size) for x, n in counts.items())


def assert_close(sampled: frozenset, exact: frozenset, tolerance=.02):
    """ Assert two subdistributions agree outcome by outcome. """
    weights = dict(sampled)
    assert set(weights) == {x for x, _ in exact}
    for outcome, weight in exact:
        assert abs(weights[outcome] - weight) < tolerance


def test_sample_matches_subdistribution():
    """
    Sampling a Kleisli composite in the ``Seed`` monad converges to the
    exact composite computed in the ``Subdistribution`` monad, i.e. seeded
    randomness simulates sub-distribution semantics -- the comparison asked
    for on issue #374.
    """
    weather = [frozenset({(0, .7), (1, .3)}), frozenset({(0, .4), (1, .6)})]
    sensor = [frozenset({(0, .9), (1, .1)}), frozenset({(0, .2), (1, .8)})]

    exact = Channel[Subdistribution](weather.__getitem__, int, int)\
        >> Channel[Subdistribution](sensor.__getitem__, int, int)
    sampled = Channel[Seed](lambda x: sample(weather[x]), int, int)\
        >> Channel[Seed](lambda x: sample(sensor[x]), int, int)

    assert_close(empirical(sampled(0), 420, 10000), exact(0))


def test_sample_misses_the_missing_mass():
    """ The missing mass of a subdistribution is sampled as ``None``. """
    half = frozenset({(0, .25), (1, .25)})
    assert_close(
        empirical(sample(half), 420, 10000),
        frozenset({(0, .25), (1, .25), (None, .5)}))


def test_Tagged():
    assert Tagged("x", 1) == eval(repr(Tagged("x", 1)))
    assert Tagged("x", 1) != Tagged("x", 0) and Tagged("x", 1) != ("x", 1)
    assert hash(Tagged("x", 1)) == hash(Tagged("x", 1))


def test_additive_pack():
    assert additive.pack(()) is Tagged
    assert additive.pack((int, )) is int
    assert additive.pack((int, str)) is Tagged
    assert additive.pack_value("x", 0, (str, )) == "x"
    assert additive.pack_value("x", 1, (int, str)) == Tagged("x", 1)
    assert additive.unpack_value("x") == ("x", 0)
    assert additive.unpack_value(Tagged("x", 1)) == ("x", 1)


def test_additive_injection():
    assert additive.injection(1, (str, ), (int, str))("x") == Tagged("x", 1)
    assert additive.injection(0, (int, str), (int, ))(Tagged(42, 0)) == 42
    assert additive.injection(2, (int, str), 2 * (int, str))(
        Tagged("x", 1)) == Tagged("x", 3)


def test_additive_Channel_identity_and_composition():
    Safe = additive.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)

    identity = Safe.id((int, ))
    assert identity(4) == 4
    assert (identity >> half)(4) == half(4) == (half >> identity)(4)

    with raises(AxiomError):
        half >> Safe(lambda x: x, str, str)


def test_additive_Channel_tensor():
    Safe = additive.Channel[Maybe]
    half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    increment = Safe(lambda x: x + 1, int, int)

    either = half @ increment
    assert either.dom == (int, int) == either.cod
    assert either(4) == Tagged(2, 0) and either(4, 1) == Tagged(5, 1)
    assert either(3) is None and either(3, 1) == Tagged(4, 1)

    with raises(TypeError):
        half.tensor("not-a-channel")

    # Whiskering on either side comes from `MonoidalCategory`.
    assert ((str, ) @ increment).dom == (str, int)
    assert (increment @ (str, )).dom == (int, str)


def test_additive_tensor_is_a_bifunctor():
    """
    Only one side of a disjoint union ever runs, so unlike the tuple tensor
    of ``multiplicative`` this one is a bifunctor for every monad.
    """
    Nondet = additive.Channel[Powerset]
    f, g, h, k = [
        Nondet(lambda x, step=step: frozenset({x + step, x * step}), int, int)
        for step in (1, 2, 3, 4)]

    lhs, rhs = (f @ g) >> (h @ k), (f >> h) @ (g >> k)
    for tag in (0, 1):
        assert lhs(3, tag) == rhs(3, tag)


def test_additive_Channel_swap_and_merge():
    Nondet = additive.Channel[Powerset]

    swap = Nondet.swap((int, ), (str, str))
    assert swap.dom == (int, str, str) and swap.cod == (str, str, int)
    assert swap(1, 0) == frozenset({Tagged(1, 2)})
    assert swap("x", 1) == frozenset({Tagged("x", 0)})
    assert swap("y", 2) == frozenset({Tagged("y", 1)})

    codiagonal = Nondet.merge((int, str))
    assert codiagonal.dom == 2 * (int, str) and codiagonal.cod == (int, str)
    assert codiagonal(1, 0) == frozenset({Tagged(1, 0)}) == codiagonal(1, 2)
    assert codiagonal("x", 1) == frozenset(
        {Tagged("x", 1)}) == codiagonal("x", 3)


def test_additive_trace_agrees_with_the_pure_trace():
    """
    A channel with no effect traces to the same thing as the plain Python
    function it comes from, i.e. ``python.additive.Function.trace``.
    """
    halve = lambda x, tag=0: (x // 2, 1) if x % 2 == 0 else (x, 0)
    pure = AdditiveFunction(halve, (int, int), (int, int))
    channel = additive.Channel[Maybe](
        lambda x, tag=0: Tagged(*halve(x, tag)), (int, int), (int, int))

    assert all(channel.trace()(x) == pure.trace()(x) for x in range(1, 12))


def test_additive_trace_converges_for_powerset():
    """
    A nondeterministic walk exiting as soon as it leaves ``range(3)``: the
    trace converges although the loop cycles, since an outcome that has
    already been stepped is never stepped again.
    """
    Nondet = additive.Channel[Powerset]
    walk = Nondet(lambda x, tag=0: frozenset({
        Tagged(y, 0) if y not in range(3) else Tagged(y, 1)
        for y in (x - 1, x + 1)}), (int, int), (int, int))

    assert walk.trace()(1) == frozenset({-1, 3})


def test_additive_trace_converges_for_subdistribution():
    """
    Gambler's ruin: a fair random walk absorbed at ``0`` and ``3`` traces to
    the exact ruin probabilities, i.e. ``2 / 3`` and ``1 / 3`` from a start
    of ``1``, although it loops unboundedly often.
    """
    walk = lambda x, tag=0: merge((
        Tagged("ruin" if y == 0 else "rich", 0) if y in (0, 3)
        else Tagged(y, 1), .5) for y in (x - 1, x + 1))
    chain = additive.Channel[Subdistribution](walk, (int, int), (str, int))

    outcomes = dict(chain.trace()(1))
    assert outcomes == {"ruin": approx(2 / 3), "rich": approx(1 / 3)}


def test_additive_trace_loses_the_diverging_mass():
    """
    A loop that goes around with probability ``1 / 4`` and loses a quarter
    of its mass on the way exits with probability ``.5 / (1 - .25)``, the
    rest going missing as the subdistribution monad allows.
    """
    leaky = additive.Channel[Subdistribution](lambda x, tag=0: frozenset({
        (Tagged(x, 0), .5), (Tagged(x + 1, 1), .25)}), (int, int), (int, int))

    assert sum(p for _, p in leaky.trace()(0)) == approx(2 / 3)


def test_additive_trace_zero_is_the_identity():
    """
    Tracing no summand at all is the vanishing axiom, ``f.trace(0) == f``,
    even for a monad with no iteration operator: see issue #578, the tenth
    ``self.dom[:-n]`` site left to this branch.
    """
    channel = additive.Channel[Maybe](
        lambda x, tag=0: Tagged(x, tag), (int, int), (int, int))
    assert channel.trace(0) == channel

    stateful = additive.Channel[Seed](
        lambda x, tag=0: lambda s: (Tagged(x, tag), s),
        (int, int), (int, int))
    assert stateful.trace(0) == stateful


def test_additive_trace_needs_an_iteration_operator():
    """
    The trace is extra structure on the monad: a monad that does not supply
    an iteration operator raises rather than guessing, see issue #374.
    """
    stateful = additive.Channel[Seed](
        lambda x, tag=0: lambda s: (Tagged(x, 0), s),
        (int, int), (int, int))
    with raises(ValueError):
        stateful.trace()

    assert Seed.iterate is None and Maybe.iterate is not None
    with raises(NotImplementedError):
        additive.Channel[Maybe](
            lambda x, tag=0: Tagged(x, 0), (int, int), (int, int)
        ).trace(left=True)


def test_additive_token_machine_computes_the_posterior():
    """
    Dal Lago-Hoshino's token machine for a coin drawn from a prior and
    flipped: one step loses the mass of the tails, i.e. the evidence is
    what exits, and tracing the internal wire feeds the tails back into
    the draw. That loop is rejection sampling, so the traced machine
    computes Bayes' rule exactly, see the module docstring.
    """
    Machine, Hyp, Unit = additive.Channel[Subdistribution], str, type(None)
    prior = {"fair": .5, "biased": .5}
    likelihood = {"fair": .5, "biased": .8}

    draw = Machine(
        lambda _: frozenset(prior.items()), (Unit, ), (Hyp, ))
    flip = Machine(lambda h: frozenset({
        (Tagged(h, 0), likelihood[h]),
        (Tagged(None, 1), 1 - likelihood[h])}), (Hyp, ), (Hyp, Unit))
    net = Machine.merge((Unit, ), 2) >> draw >> flip

    evidence = sum(prior[h] * likelihood[h] for h in prior)
    assert sum(
        p for port, p in (draw >> flip)(None) if port.tag == 0
    ) == approx(evidence)

    posterior = dict(net.trace()(None))
    assert posterior == {
        h: approx(prior[h] * likelihood[h] / evidence) for h in prior}
    assert sum(posterior.values()) == approx(1)


def test_additive_Channel_repr():
    half = additive.Channel[Maybe](
        lambda x: x // 2 if x % 2 == 0 else None, int, int)
    assert repr(half).startswith("kleisli.additive.Channel[Maybe](")


def token_machine(**constants):
    """
    A probabilistic token machine, i.e. the one of the module docstring of
    :mod:`discopy.kleisli.token` with the constants given as keywords.
    """
    return token.Machine[Subdistribution](constants)


def test_token_lookup_reads_the_innermost_binder():
    x, y = Variable("x", Ty("X")), Variable("y", Ty("Y"))
    assert token.lookup(((x, "inner"), (x, "outer")), x) == "inner"
    assert token.lookup(((x, 0), (y, 1)), y) == 1
    with raises(ValueError):
        token.lookup(((x, 0), ), y)


def test_token_evidence_and_posterior():
    values = frozenset({("heads", .1), ("tails", .3)})
    assert token.evidence(values) == approx(.4)
    assert dict(token.posterior(values)) == {
        "heads": approx(.25), "tails": approx(.75)}
    with raises(ValueError):
        token.posterior(frozenset())


def test_token_machine_evaluates_pure_terms():
    """
    Beta reduction with no effect at all: the machine walks a closed term
    of the simply-typed lambda calculus to its value, which is the constant
    itself for a base type and a closure for a function type.
    """
    X = Ty("X")
    identity, c = X(lambda x: x), X("c")
    machine = token.Machine[Maybe]()

    assert machine(identity(c)) == c
    assert machine(identity) == token.Closure(identity, ())
    assert machine((X >> X)(lambda f: f(f(c)))(identity)) == c
    assert machine(X(lambda x: X(lambda y: x))(c)(X("d"))) == c
    assert machine(X(lambda x: X(lambda x: x)(X("d")))(c)) == X("d")


def test_token_machine_needs_a_closed_term_of_the_calculus():
    X = Ty("X")
    machine = token.Machine[Maybe]()
    with raises(ValueError):
        machine(Variable("z", X))
    with raises(ValueError):
        machine((X >> X)("f")(X("c")))
    with raises(ValueError):
        machine.apply(42, 0, ())
    with raises(NotImplementedError):
        machine.step(token.Down(ClosedBox("f", X, X)))


def test_token_machine_needs_an_iteration_operator():
    """
    The machine is a trace, so it is defined exactly for the monads that
    are Elgot, see :mod:`discopy.kleisli.additive`.
    """
    with raises(ValueError):
        token.Machine[Seed]()(Ty("X")("c"))


def test_token_machine_is_a_traced_channel():
    """
    One transition is a channel from an entry plus the two directions of
    the token to an exit plus the two directions, and the machine is its
    trace over the two directions.
    """
    X = Ty("X")
    machine = token.Machine[Maybe]()
    channel = machine.channel

    assert channel.dom == (token.Down, token.Down, token.Up)
    assert channel.cod == (token.Value, token.Down, token.Up)
    assert channel.trace(2)(token.Down(X("c"))) == machine(X("c"))
    assert channel(token.Down(X("c"))) == Tagged(token.Up(X("c")), 2)
    assert channel(token.Up(X("c")), 2) == Tagged(X("c"), 0)


def test_token_machine_samples_and_scores():
    """
    Dal Lago-Hoshino's two constants over the subdistribution monad: the
    mass of a draw whose outcome is discarded merges back into the branch
    it came from, and scoring multiplies the weight of the branch it runs
    in, so that the machine returns the unnormalised posterior of Bayes.
    """
    B, U = Ty("B"), Ty("U")
    flip, star = B("flip"), U("*")
    score = (B >> U)("score")
    machine = token_machine(
        flip=frozenset({(True, .5), (False, .5)}),
        score=lambda weight: frozenset({(star, .5 if weight else 1.)}))

    discarded = B(lambda x: B(lambda y: x)(flip))(flip)
    assert dict(machine(discarded)) == {
        True: approx(.5), False: approx(.5)}

    conditioned = B(lambda x: U(lambda _: x)(score(x)))(flip)
    assert dict(machine(conditioned)) == {
        True: approx(.25), False: approx(.5)}
    assert token.evidence(machine(conditioned)) == approx(.75)
    assert dict(token.posterior(machine(conditioned))) == {
        True: approx(1 / 3), False: approx(2 / 3)}


def test_token_machine_loses_all_its_mass_on_an_impossible_observation():
    B, U = Ty("B"), Ty("U")
    flip, star = B("flip"), U("*")
    machine = token_machine(
        flip=frozenset({(True, .5), (False, .5)}),
        score=lambda weight: frozenset({(star, 0.)}))
    term = B(lambda x: U(lambda _: x)((B >> U)("score")(x)))(flip)

    assert machine(term) == frozenset()
    assert token.evidence(machine(term)) == 0
    with raises(ValueError):
        token.posterior(machine(term))


def test_token_machine_is_nondeterministic_over_the_powerset_monad():
    """
    The same machine over another Elgot monad: the powerset monad forgets
    the weights, so a coin comes back as the set of its outcomes.
    """
    B = Ty("B")
    flip = B("flip")
    machine = token.Machine[Powerset]({
        "flip": frozenset({"heads", "tails"}),
        "twice": lambda x: frozenset({x + x})})
    term = B(lambda x: (B >> B)("twice")(x))(flip)

    assert machine(term) == frozenset({"headsheads", "tailstails"})
