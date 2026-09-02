"""B50: Hypergraph equality and hash ignore the type of a scalar spider (discopy/hypergraph.py:606).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.frobenius import Box, Equation, Hypergraph as H, Spider, Ty

x, y = map(Ty, "xy")
f = Box('f', x, x)


def test_b50_scalar_spiders_of_different_types_are_unequal():
    assert H.spiders(0, 0, x) != H.spiders(0, 0, y)


def test_b50_hash_sees_the_spider_type():
    assert hash(H.spiders(0, 0, x)) != hash(H.spiders(0, 0, y))


def test_b50_equation_sees_the_spider_type():
    assert not Equation(Spider(0, 0, x) @ f, Spider(0, 0, y) @ f)
