"""B8: cup_factory is wrong on non-atomic dimensions (discopy/tensor.py:160).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.tensor import Dim, Tensor


def test_b8_cup_factory_non_atomic():
    left, right = Dim(2, 3), Dim(3, 2)
    assert Tensor.cup_factory(left, right).is_close(Tensor.cups(left, right))
