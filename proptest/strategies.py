"""Resolve Hypothesis strategies through :mod:`discopy.testing`."""

import inspect
from typing import get_args, get_origin

from hypothesis import strategies as st

from discopy.testing import Strategy


def strategy(annotation, **params):
    """Resolve the strategy implemented by an annotated type."""
    origin = get_origin(annotation) or annotation
    if not isinstance(origin, type) or not issubclass(origin, Strategy):
        raise TypeError(
            f"Expected a Strategy annotation, got {annotation!r}.")
    if args := get_args(annotation):
        params["factory"] = args[-1]
    return origin.strategy(**params)


def arguments(axiom):
    """Generate the explicit arguments expected by a bound axiom."""
    function = axiom.equation.__func__
    annotations = inspect.get_annotations(
        function, globals=function.__globals__,
        locals={"C0": axiom.carrier.ob, "C1": axiom.carrier.ar},
        eval_str=True)
    required = (
        parameter for parameter in axiom.parameters
        if parameter.default is inspect.Parameter.empty)
    return st.tuples(*(
        strategy(annotations[parameter.name]) for parameter in required))
