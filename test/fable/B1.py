"""B1: Function.swap returns the inverse permutation (discopy/python/finset.py:85).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.python import finset


def test_b1_swap_inside():
    # inside maps range(cod) to range(dom): output block order is y then x,
    # so out0 pulls from in2, out1 from in0, out2 from in1.
    assert list(finset.Function.swap(2, 1).inside) == [2, 0, 1]


def test_b1_swap_naturality():
    # (f @ g) ; swap(2, 1) == swap(1, 1) ; (g @ f) with f: 1 -> 2 the copy.
    f = finset.Function([0, 0], 1, 2)
    g = finset.Function.id(1)
    lhs = f.tensor(g).then(finset.Function.swap(2, 1))
    rhs = finset.Function.swap(1, 1).then(g.tensor(f))
    assert lhs == rhs
