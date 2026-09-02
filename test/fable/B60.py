"""B60: the identity functor is not the identity on a Permutation (discopy/symmetric.py:655).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import feedback, markov, symmetric


def _identity_on_permutation(module):
    x, y, z = map(module.Ty, "xyz")
    perm = module.Permutation(x @ y @ z, [1, 2, 0])
    assert module.Functor.id(module.Diagram)(perm) == perm


def test_b60_symmetric():
    _identity_on_permutation(symmetric)


def test_b60_markov():
    _identity_on_permutation(markov)


def test_b60_feedback():
    _identity_on_permutation(feedback)
