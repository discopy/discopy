# -*- coding: utf-8 -*-


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


def test_trace_unequal_arity():
    from discopy.python.additive import Function

    # f : A + U -> B + C + U, traced over U: tag 0 (A) loops back to U,
    # any other tag (C) exits. len(dom) == 2 != 3 == len(cod), so the
    # traced summand sits at a different tag on each side.
    def inside(obj, tag=0):
        return obj, 2 if tag == 0 else 1

    f = Function(inside, (int, int), (int, int, int))
    assert f.trace()(7) == (7, 1)


def test_strategy():
    from hypothesis import find
    from discopy.python.additive import Function

    find(Function.strategy(), lambda f: len(f.dom) > len(f.cod) >= 1)


def test_axioms():
    from discopy import testing
    from discopy.python.additive import Function

    testing.assert_axioms(Function)
