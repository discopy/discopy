"""B4: Function.permutation ignores block structure (discopy/python/finset.py:90).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.python import finset


def test_b4_block_permutation():
    # Permuting blocks of sizes (2, 3) by (1, 0) is the swap of 2 and 3.
    p = finset.Function.permutation((1, 0), (2, 3))
    assert p.dom == p.cod == 5
    assert p == finset.Function.swap(2, 3)
