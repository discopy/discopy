# -*- coding: utf-8 -*-

from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.python import *
from discopy.utils import AxiomError


def test_Function():
    x, y, z = (complex, ), (bool, ), (float, )
    f = Function(dom=y, cod=exp(z, x),
                 inside=lambda y: lambda x: abs(x) ** 2 if y else 0)
    g = Function(dom=x + y, cod=z, inside=lambda x, y: f(y)(x))

    assert f.uncurry().curry()(True)(1j) == f(True)(1j)
    assert f.uncurry(left=False).curry(left=False)(True)(1j) == f(True)(1j)
    assert g.curry().uncurry()(1j, True) == g(1j, True)
    assert g.curry(left=False).uncurry(left=False)(1j, True) == g(1j, True)


def test_fixed_point():
    from math import sqrt
    phi = Function(lambda x=1: 1 + 1 / x, dom=(float,), cod=(float,)).fix()
    assert phi() == (1 + sqrt(5)) / 2


def test_trace():
    with raises(NotImplementedError):
        Function.id(int).trace(left=True)


def test_FinSet():
    from discopy.markov import Ty, Diagram, Functor
    from discopy.python import finset

    x = Ty('x')
    copy, discard, swap = Diagram.copy(x), Diagram.copy(x, 0), Diagram.swap(x, x)
    F = Functor({x: 1}, {}, cod=finset.Function)

    assert F(copy >> discard @ x) == F(Diagram.id(x)) == F(copy >> x @ discard)
    assert F(copy >> copy @ x) == F(Diagram.copy(x, 3)) == F(copy >> x @ copy)
    assert F(copy >> swap) == F(copy)


def test_additive_Function():
    from discopy.interaction import Ty, Diagram
    from discopy.python.additive import Ty as T, Function, Id, Swap, Merge

    X, xs = (int, ), []
    m, e = Function.merge(X, n=2), Function.merge(X, n=0)

    def f_inside(m, n=0):
        xs.append(m)
        return 3 * m + 1 if m % 2 else m // 2, 0 if n == 1 and m == 2 else 1

    f = Function(f_inside, X + X, X + X)
    g = Function(lambda m: m // 2, X, X)

    # This converges if https://en.wikipedia.org/wiki/Collatz_conjecture holds.
    assert f.trace()(42) == 1 and xs == [42, 21, 64, 32, 16, 8, 4, 2]

    eq = lambda *fs: all(fs[0].is_parallel(f) for f in fs) and all(
        len(set(f(42, i) for f in fs)) == 1 for i in range(len(fs[0].dom)))

    assert eq(Swap(X, X) >> m, m)
    assert eq(X @ e >> m, Id(X), e @ X >> m)
    assert eq(m @ X >> m, X @ m >> m, Function.merge(X, n=3))
    assert eq(Function.merge(X + X), X @ Swap(X, X) @ X >> m @ m)

    assert eq(Swap(X, X).trace(), Id(X))  # Yanking
    assert eq((f >> X @ g).trace(), (X @ g >> f).trace())  # Sliding
    assert eq((g @ X >> f).trace(), g >> f.trace())  # Left-naturality
    assert eq((f >> g @ X).trace(), f.trace() >> g)  # Right-naturality
    
    T, D = Ty[tuple], Diagram[Function]

    assert eq(D.id(T(X, X)).transpose().inside, Id(X + X))

def test_list_generic_in_function():
    func = Function(sum, List[int], int)
    assert func([1, 2, 3]) == 6


def test_Hypergraph_call():
    add = Function(lambda x, y: x + y, (int, int), (int, ))

    # A wire copied into both addends and the other one forwarded: the value
    # on wire 0 is read twice (by the box and by the output) and wire 1 once.
    f = Hypergraph(
        dom=(int, int), cod=(int, int),
        boxes=(add, ),
        wires=((0, 1), (((1, 0), (2, )), ), (2, 0)))
    assert f(2, 3) == (5, 2)

    # A discarded input (wire 1 is read by nothing) and a single output.
    g = Hypergraph(
        dom=(int, int), cod=(int, ),
        boxes=(), wires=((0, 1), (), (0, )))
    assert g(2, 3) == 2


def test_Hypergraph_axioms():
    swap = Function(lambda x, y: (y, x), (int, int), (int, int))
    add = Function(lambda x, y: x + y, (int, int), (int, ))

    # Not causal: the swap reads wires that only its own output produces.
    with raises(AxiomError):
        Hypergraph((int, ), (int, ),
                   (swap, ), ((0, ), (((0, 1), (1, 2)), ), (2, )))

    # Not left-monogamous: wire 0 is produced by the input and the box.
    with raises(AxiomError):
        Hypergraph((int, ), (int, ),
                   (add, ), ((0, ), (((0, 0), (0, )), ), (0, )))


def _permutation(factory, xs, dom):
    """ A permutation arrow, as in the arXiv:2105.09257 benchmark. """
    if len(dom) <= 1:
        return factory.id(dom)
    i = xs[0]
    head = factory.swap(dom[:i], dom[i:i + 1]).tensor(factory.id(dom[i + 1:]))
    tail = factory.id(dom[i:i + 1]).tensor(_permutation(
        factory, [x - 1 if x > i else x for x in xs[1:]],
        dom[:i] + dom[i + 1:]))
    return head.then(tail)


def _adder_step(full_adder, adder, k):
    """ One incremental ripple-carry step: adder(k) -> adder(k + 1). """
    factory = type(full_adder)
    bit = full_adder.dom[:1]
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    step = adder.tensor(factory.id(bit @ bit))
    step = step.then(_permutation(factory, reorder1, step.cod))
    step = step.then(factory.id(bit ** k).tensor(full_adder))
    return step.then(_permutation(factory, reorder2, step.cod))


def test_Hypergraph_adder():
    """ The carry-save adder of the #346 benchmark, evaluated directly as a
    hypergraph of Python functions rather than compiled to a diagram. """
    import itertools
    from discopy.symmetric import Ty, Box

    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)

    def full_adder_function(a, b, carry_in):
        return a ^ b ^ carry_in, (a & b) | (carry_in & (a ^ b))

    fa = Function(full_adder_function, (int, int, int), (int, int))

    def carry_save_value(outputs):
        return outputs[0] + 2 * sum(outputs[1:])

    full_adder_hg = full_adder.to_hypergraph()
    adder = full_adder_hg
    for k in range(1, 5):
        compiled = Hypergraph.from_hypergraph(
            adder, ob={bit: int}, ar={full_adder: fa})
        # The swaps of the reordering permutations are absorbed into the
        # wiring, so the only boxes left are the full adders themselves.
        assert all(box == fa for box in compiled.boxes)
        for bits in itertools.product((0, 1), repeat=len(compiled.dom)):
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = _adder_step(full_adder_hg, adder, k)
