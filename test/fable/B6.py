"""B6: Matrix.trace ignores its left argument (discopy/matrix.py:373).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.matrix import Matrix


def test_b6_trace_left():
    h = Matrix[bool]([1, 0, 0, 0], 2, 2)
    assert h.trace(1, left=True).array.tolist() == [[False]]


def test_b6_trace_right():
    h = Matrix[bool]([1, 0, 0, 0], 2, 2)
    assert h.trace(1, left=False).array.tolist() == [[True]]
