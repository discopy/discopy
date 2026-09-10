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
from hypothesis import given
from hypothesis import strategies as st

from discopy import monoidal
from discopy.utils import factory_name

from proptest.categories import CATEGORIES

matplotlib.use("Agg")

DIAGRAMS = tuple(
    category for category in CATEGORIES
    if isinstance(category, type) and issubclass(category, monoidal.Diagram))


@pytest.mark.parametrize("category", DIAGRAMS, ids=factory_name)
@given(data=st.data())
def test_to_drawing(category, data):
    """ Check that the layout functor preserves the boundary. """
    diagram = data.draw(category.strategy())
    drawing = diagram.to_drawing()
    assert drawing.dom == diagram.dom.to_drawing()
    assert drawing.cod == diagram.cod.to_drawing()


@pytest.mark.parametrize("category", DIAGRAMS, ids=factory_name)
@given(data=st.data())
def test_draw(category, data):
    """ Check that both backends render a diagram without a baseline. """
    diagram = data.draw(category.strategy())
    diagram.draw(path=io.BytesIO(), format="png")
    with tempfile.TemporaryDirectory() as directory:
        diagram.draw(
            path=os.path.join(directory, "diagram.tikz"), to_tikz=True)
