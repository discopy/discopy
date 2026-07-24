# -*- coding: utf-8 -*-

from pytest import raises


def test_FinSet():
    from discopy.markov import Ty, Diagram, Functor
    from discopy.python import finset

    p = finset.Permutation.swap(2, 3)
    assert isinstance(p, finset.Function)
    assert isinstance(p, finset.SymmetricCategory)
    assert p == (3, 4, 0, 1, 2)
    assert p[-1] == 2
    assert p >> p.dagger() == finset.Permutation.id(5)
    assert p @ finset.Permutation((1, 0)) == (3, 4, 0, 1, 2, 6, 5)
    assert finset.Function.swap(2, 3).inside == list(p)
    assert finset.Permutation((1, 0, 3, 2)).cycles() == ((0, 1), (2, 3))
    assert finset.Permutation.from_cycles([(0, 1), (2, 3)], 4)\
        == (1, 0, 3, 2)
    assert finset.Permutation((1, 0)).is_fixpoint_free_involution()
    assert not finset.Permutation((0,)).is_fixpoint_free_involution()
    assert finset.Permutation.identity(2) == (0, 1)
    assert finset.Permutation((1, 0)).then((1, 0)) == (0, 1)
    assert finset.Permutation((1, 0, 2)).then((1, 2, 0)) == (2, 1, 0)
    assert finset.Permutation((1, 2, 0)).dagger() == (2, 0, 1)
    assert finset.Permutation((1, 0, 2)).conjugate((2, 0, 1))\
        == (2, 1, 0)
    assert finset.Permutation((1, 2, 0)).cycle(1) == (1, 2, 0)
    assert finset.Permutation((1, 2, 0)).cycle(-1) == (2, 0, 1)
    assert finset.Permutation((1, 0, 3, 2)).coequalizer(
        finset.Permutation((0, 2, 1, 3))) == {0: 0, 1: 0, 2: 0, 3: 0}
    with raises(ValueError):
        finset.Permutation((0, 0))
    with raises(ValueError):
        finset.Permutation((0,), size=2)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 0)], 1)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 2)], 2)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 1), (1, 2)], 3)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 0)], 2)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 2)], 2)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 1), (1, 2)], 4)
    with raises(ValueError):
        finset.Permutation((0,)).cycle(1)
    function = finset.Function([1, 0], 2, 2)
    permutation = finset.Permutation((1, 0))
    assert list(function) == [1, 0]
    assert function.index(0) == permutation.index(0) == 1
    assert function[-1] == permutation[-1] == 0
    with raises(IndexError):
        permutation[2]
    with raises(ValueError):
        permutation.index(2)

    x = Ty('x')
    copy, discard, swap = Diagram.copy(x), Diagram.copy(x, 0), Diagram.swap(x, x)
    F = Functor({x: 1}, {}, cod=finset.Function)

    assert F(copy >> discard @ x) == F(Diagram.id(x)) == F(copy >> x @ discard)
    assert F(copy >> copy @ x) == F(Diagram.copy(x, 3)) == F(copy >> x @ copy)
    assert F(copy >> swap) == F(copy)
