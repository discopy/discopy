"""B64: to_tn builds a 0-input spider from the wire to its right (discopy/tensor.py:629).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy.tensor import Dim, Id, Spider

tn = pytest.importorskip("tensornetwork")


def contract(diagram):
    nodes, order = diagram.to_tn()
    return tn.contractors.auto(nodes, output_edge_order=order).tensor


def test_b64_state_spider_beside_another_wire_has_the_right_shape():
    d = Spider(0, 1, Dim(2)) @ Id(Dim(3))
    assert contract(d).shape == (3, 2, 3)
    assert contract(d).tolist() == d.eval().array.tolist()


def test_b64_state_spider_at_the_end_of_the_scan_builds():
    d = Spider(0, 2, Dim(2))
    assert contract(d).tolist() == d.eval().array.tolist()
