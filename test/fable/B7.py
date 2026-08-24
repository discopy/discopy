"""B7: spiders(0, 0, Dim(n)) returns 1 instead of n (discopy/tensor.py:203).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.tensor import Dim, Tensor


def test_b7_zero_legged_spider_is_the_dimension():
    assert Tensor.spiders(0, 0, Dim(2)).array == 2


def test_b7_zero_legged_spider_composition():
    circle = Tensor.spiders(0, 1, Dim(2)) >> Tensor.spiders(1, 0, Dim(2))
    assert Tensor.spiders(0, 0, Dim(2)).array == circle.array
