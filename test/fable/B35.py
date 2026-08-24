"""B35: Hypergraph.from_callable leaks a monkey-patched __call__ on failure (discopy/hypergraph.py:1524).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import pytest

from discopy import frobenius


def test_b35_from_callable_cleans_up_on_failure():
    x = frobenius.Ty('x')
    with pytest.raises(RuntimeError):
        @frobenius.Diagram.from_callable(x, x)
        def diagram(wire):
            raise RuntimeError("boom")
    assert '__call__' not in vars(frobenius.Diagram)
