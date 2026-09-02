"""B47: a cup on a non-atomic object image is Tensor.cups alone and CMap wiring in a diagram, so F is not a functor (discopy/tensor.py:428 vs 176).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

from discopy import compact, frobenius, pivotal
from discopy.tensor import Dim, Functor
from discopy.utils import AxiomError

MODULES = [frobenius, compact, pivotal]


def functor(module, image):
    n = module.Ty('n')
    Alice, Bob = [module.Box(s, module.Ty(), n) for s in ('Alice', 'Bob')]
    loves = module.Box('loves', module.Ty(), n.r @ n.l)
    rng = np.random.RandomState(1)
    F = Functor({n: image}, {
        Alice: rng.rand(4).tolist(), loves: rng.rand(16).tolist(),
        Bob: rng.rand(4).tolist()}, dom=module.Diagram, dtype=float)
    return n, Alice, loves, Bob, F


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_b47_cup_alone_equals_cup_in_a_diagram(module):
    n, _, _, _, F = functor(module, Dim(2, 2))
    cup = module.Cup(n, n.r)
    alone, composed = F(cup), F(module.Id(n @ n.r) >> cup)
    assert np.allclose(alone.array, composed.array)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_b47_sentence_evaluates_the_same_alone_and_composed(module):
    n, Alice, loves, Bob, F = functor(module, Dim(2, 2))
    cups = module.Cup(n, n.r) @ module.Cup(n.l, n)
    sentence = Alice @ loves @ Bob >> cups
    boxwise = F(Alice) @ F(loves) @ F(Bob) \
        >> F(module.Cup(n, n.r)) @ F(module.Cup(n.l, n))
    assert np.isclose(float(F(sentence).array), float(boxwise.array))


def outcome(thunk):
    try:
        return thunk().array.tolist()
    except AxiomError:
        return AxiomError


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_b47_non_adjoint_image_raises_consistently_or_agrees(module):
    n, _, _, _, F = functor(module, Dim(2, 3))
    cup = module.Cup(n, n.r)
    alone = outcome(lambda: F(cup))
    composed = outcome(lambda: F(module.Id(n @ n.r) >> cup))
    assert alone == composed
