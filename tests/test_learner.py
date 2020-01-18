from pytest import raises
from discopy.learner import *


def test_Box_function():
    assert "<lambda>" in repr(COPY.function)


def test_Box_repr():
    assert "Box('COPY', 1, 2, function=" in repr(COPY)
