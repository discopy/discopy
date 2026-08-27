"""
Property tests for transparency: ``eval(repr(x)) == x`` in a fresh
environment, for every carrier of the property matrix.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import monoidal, tensor
from discopy.python import function
from discopy.quantum import zx
from discopy.utils import factory_name

from proptest.test_axioms import CARRIERS

IMPORTS = (
    "from discopy import *",
    "import numpy as np",
    "from numpy import complex128, float64",
    "from discopy.quantum.gates import *",
    "from discopy.matrix import Matrix",
    "from discopy.tensor import Tensor",
    "from discopy.frobenius import Dim",
    "from discopy.python.finset import Function, Permutation",
    "from discopy.testing import Relabelling, Relabelled",
)
"""
What the fresh environment loads: the package itself, plus the obvious
import for each carrier whose ``repr`` uses its bare class name — ``Matrix``
and ``finset`` print unqualified, and a generated functor relabels the
generators through :class:`discopy.testing.Relabelling`.
"""

EXTRA_IMPORTS = {zx.Diagram: "from discopy.quantum.zx import *"}
"""
The extra import a carrier's environment needs when it collides with the
shared ones, e.g. the ZX generators shadow the quantum gates.
"""

ENVIRONMENT = {}
for statement in IMPORTS:
    exec(statement, ENVIRONMENT)


def environment(carrier):
    """ The fresh environment a carrier's reprs evaluate in. """
    env = dict(ENVIRONMENT)
    if carrier in EXTRA_IMPORTS:
        exec(EXTRA_IMPORTS[carrier], env)
    return env


def carrier_parameters():
    """ One parameter per carrier, the known violations expected failures. """
    for carrier in CARRIERS:
        if carrier is monoidal.Wire:
            marks = pytest.mark.xfail(reason=(
                "An uncoloured wire reprs as the cat.Ob that Ty coerces, "
                "which Wire.__eq__ rejects."))
        elif carrier is tensor.Tensor[int]:
            marks = pytest.mark.xfail(reason=(
                "A tensor with more than config.NUMPY_THRESHOLD entries "
                "elides its array as a literal ellipsis."))
        elif isinstance(carrier, type)\
                and issubclass(carrier, function.Function):
            marks = pytest.mark.xfail(
                reason="A closure does not repr its body.")
        else:
            marks = ()
        yield pytest.param(carrier, marks=marks, id=factory_name(carrier))


@pytest.mark.parametrize("carrier", carrier_parameters())
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_repr(carrier, data):
    """ Check that ``repr`` evaluates back to the value it describes. """
    value = data.draw(carrier.strategy())
    assert eval(repr(value), environment(carrier)) == value
