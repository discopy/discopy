"""
Property tests for the drawing pipeline: the layout functor preserves the
boundary of every generated diagram, and both backends render it without
a baseline — Matplotlib on Agg into an in-memory buffer, TikZ into a
throwaway file.
"""

import io
import os
import tempfile

import matplotlib
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import monoidal
from discopy.utils import factory_name

from proptest.test_axioms import CARRIERS

matplotlib.use("Agg")

DIAGRAMS = tuple(
    carrier for carrier in CARRIERS
    if isinstance(carrier, type) and issubclass(carrier, monoidal.Diagram))


@pytest.mark.parametrize("carrier", DIAGRAMS, ids=factory_name)
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_to_drawing(carrier, data):
    """ Check that the layout functor preserves the boundary. """
    diagram = data.draw(carrier.strategy())
    drawing = diagram.to_drawing()
    assert drawing.dom == diagram.dom.to_drawing()
    assert drawing.cod == diagram.cod.to_drawing()


@pytest.mark.parametrize("carrier", DIAGRAMS, ids=factory_name)
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_draw(carrier, data):
    """ Check that both backends render a diagram without a baseline. """
    diagram = data.draw(carrier.strategy())
    diagram.draw(path=io.BytesIO(), format="png")
    with tempfile.TemporaryDirectory() as directory:
        diagram.draw(
            path=os.path.join(directory, "diagram.tikz"), to_tikz=True)
