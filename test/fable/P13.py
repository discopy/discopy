"""P13: levels conform — every level sets the factories its abc methods need, bubbles compose and rotate on every level, and every abstract method in discopy.abc has the same kind in every concrete subclass.
Miniature of the property over the package's own classes; red while its bullets (B76, B88) are live — issue #699.
"""
import inspect

from discopy import (abc, cat, monoidal, braided, balanced, symmetric, traced,
                     markov, closed, biclosed, rigid, pivotal, ribbon, compact,
                     frobenius, hypergraph, cmap, matrix, tensor, para, stream,
                     interaction, feedback)
from discopy.drawing import drawing
from discopy.python import function, additive, multiplicative, finset
from discopy.quantum import channel
from discopy.grammar import cfg

LEVELS = [monoidal, braided, balanced, symmetric, traced, markov, closed,
          biclosed, rigid, pivotal, ribbon, compact, frobenius]
MODULES = LEVELS + [cat, hypergraph, cmap, matrix, tensor, para, stream,
                    interaction, feedback, drawing, function, additive,
                    multiplicative, finset, channel, cfg]


def _kind(attribute):
    if isinstance(attribute, (classmethod, staticmethod)):
        return "class"
    if isinstance(attribute, property):
        return "property"
    return "instance"


def _abstract_methods():
    for klass in vars(abc).values():
        if not (inspect.isclass(klass) and issubclass(klass, abc.Category)):
            continue
        for name, attribute in vars(klass).items():
            function_ = getattr(attribute, "__func__", attribute)
            if getattr(function_, "__isabstractmethod__", False):
                yield klass, name, _kind(attribute)


def _concrete_classes():
    for module in MODULES:
        for klass in vars(module).values():
            if inspect.isclass(klass) and issubclass(klass, abc.Category)\
                    and klass.__module__ == module.__name__:
                yield klass


def _level_failures():
    failures = []
    for level in LEVELS:
        x = level.Ty('x')
        f, h = level.Box('f', x, x), level.Box('h', x, x)
        if not issubclass(level.Diagram.bubble_factory, level.Diagram):
            failures.append(f"{level.__name__}: bubble_factory is not a "
                            f"{level.__name__}.Diagram")
        try:
            f.bubble() >> h
        except Exception as error:
            failures.append(f"{level.__name__}: Box.bubble() >> Box raised "
                            f"{type(error).__name__}")
        if hasattr(level.Box, 'r'):
            try:
                f.bubble().r
            except Exception as error:
                failures.append(f"{level.__name__}: Box.bubble().r raised "
                                f"{type(error).__name__}")
    return failures


def _abstract_failures():
    failures, seen = [], set()
    for base, name, kind in _abstract_methods():
        for klass in _concrete_classes():
            if not issubclass(klass, base):
                continue
            owner = next(k for k in klass.__mro__ if name in vars(k))
            if owner.__module__ == abc.__name__ or (owner, name) in seen:
                continue
            seen.add((owner, name))
            found = _kind(vars(owner)[name])
            if found not in (kind, "property"):
                failures.append(
                    f"abc.{base.__name__}.{name} is a {kind} method, "
                    f"{owner.__module__}.{owner.__name__}.{name} is {found}")
    return failures


def _default_left(method):
    return inspect.signature(method).parameters["left"].default


def test_p13():
    failures = _level_failures() + _abstract_failures()
    for klass in (para.Closed, para.Compact):
        if _default_left(klass.curry) != _default_left(abc.BiclosedCategory.curry):
            failures.append(f"para.{klass.__name__}.curry defaults left to "
                            f"{_default_left(klass.curry)}")
    assert not failures, failures
