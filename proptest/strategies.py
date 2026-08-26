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
    """
    Generate the arguments expected by a bound axiom.

    Both resolve ``C0`` and ``C1`` to objects and arrows, of the carrier for a
    law of a category and of the carrier's domain for a law of an element:
    the arguments a functor is applied to live in the category it maps from,
    and its codomain is reachable as ``self.cod`` from the body.
    """
    function = axiom.equation
    source = axiom.carrier.dom if axiom.is_method else axiom.carrier
    scope = {"C0": source.ob, "C1": source.ar}
    annotations = inspect.get_annotations(
        function, globals=function.__globals__, locals=scope, eval_str=True)
    annotations[axiom.receiver] = axiom.carrier
    required = (
        parameter for parameter in axiom.parameters
        if parameter.default is inspect.Parameter.empty)
    return st.tuples(*(
        strategy(annotations[parameter.name]) for parameter in required))
