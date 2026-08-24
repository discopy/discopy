"""B38: the instance-level lambda dagger on zx.H makes H-containing diagrams unpicklable (discopy/quantum/zx.py:386).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import pickle

from discopy.quantum import zx


def test_b38_h_diagram_pickles():
    diagram = zx.Id(1) @ zx.H
    assert pickle.loads(pickle.dumps(diagram)) == diagram
