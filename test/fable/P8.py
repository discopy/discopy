"""P8: purity, no operation mutates global or shared state.
Miniature of the property over curated examples; red while its bullets (B35, B36, B37, B38) are live — issue #606.
"""
import os
import pickle
import tempfile

import numpy

from discopy import frobenius
from discopy.matrix import Matrix
from discopy.quantum import zx


def test_p8():
    failures = []
    numpy.set_printoptions(threshold=1000)
    before = numpy.get_printoptions()
    repr(Matrix(list(range(25)), 5, 5))
    if numpy.get_printoptions() != before:
        failures.append("repr(Matrix): numpy printoptions changed")
    numpy.set_printoptions(**before)

    x = frobenius.Ty('x')
    f = frobenius.Box('f', x @ x, x)
    try:
        @frobenius.Diagram.from_callable(x, x)
        def bad(wire):
            return f(wire)  # f wants two inputs, so this raises inside
    except Exception:
        pass
    if "__call__" in vars(frobenius.Diagram):
        failures.append("failing from_callable: __call__ left on Diagram")
        delattr(frobenius.Diagram, "__call__")

    g = frobenius.Box('g', x, x)
    one, two = g.to_drawing(), g.to_drawing()
    if one == two:
        with tempfile.TemporaryDirectory() as tmp:
            one.draw(path=os.path.join(tmp, "out.svg"))
        if one != two:
            failures.append("draw(path=...): equal drawings became unequal")
    else:
        failures.append("to_drawing: two equal drawings are not equal")

    try:
        pickle.dumps(zx.Id(1) @ zx.H)
    except Exception as error:
        failures.append(
            f"pickle.dumps(zx.Id(1) @ zx.H) raised {type(error).__name__}")
    assert not failures, failures
