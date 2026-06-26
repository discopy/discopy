# -*- coding: utf-8 -*-

"""
Random generation of diagrams, for property-based testing with shrinking.

This module provides functions for generating seeded random objects and
diagrams, as well as composable and parallel tuples of diagrams. These can
be fed into the ``check_*`` methods defined on the classes of
:mod:`discopy.abc`, in order to test that the axioms of a category hold
(possibly up to some notion of equality) for randomly generated examples.

The entry point is :func:`check_property`, which takes a ``generator`` from a
random source to an example and a ``predicate`` from an example to a boolean.
When the predicate fails, the example is **shrunk** to a (locally) minimal
counterexample. This uses integrated shrinking, i.e. we record the sequence of
random draws made by the generator, shrink that sequence and regenerate, so
that structural invariants such as composability are preserved automatically.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    random_ty
    random_diagram
    random_parallel_pair
    random_composable_pair
    random_composable_triple
    check_property

Example
-------
We can check the associativity and unitality of composition for randomly
generated diagrams in any monoidal category, e.g. :mod:`discopy.symmetric`.

>>> from discopy import symmetric

>>> assert check_property(
...     lambda rng: random_composable_triple(symmetric.Box, rng=rng),
...     lambda fgh: fgh[0].check_associativity(fgh[1], fgh[2]))

>>> assert check_property(
...     lambda rng: random_diagram(symmetric.Box, rng=rng),
...     lambda f: f.check_unitality())

Some axioms only hold up to some notion of equality, e.g. the swap is its
own inverse only up to :attr:`symmetric.Diagram.hypergraph_equality`.

>>> def swap_is_inverse(xy):
...     with symmetric.Diagram.hypergraph_equality:
...         return symmetric.Diagram.check_swap_inverse(*xy)
>>> assert check_property(
...     lambda rng: (random_ty(symmetric.Ty, min_length=1, rng=rng),
...                  random_ty(symmetric.Ty, min_length=1, rng=rng)),
...     swap_is_inverse)
"""

from __future__ import annotations

import random as _random
from string import ascii_lowercase as ALPHABET
from typing import Callable

DEFAULT_OBJECTS = ALPHABET


class Falsified(AssertionError):
    """
    Raised by :func:`check_property` when a predicate is falsified, carrying
    the minimal counterexample found by shrinking as its ``example`` attribute.
    """
    def __init__(self, example):
        self.example = example
        super().__init__(repr(example))


class _Source:
    """
    A source of randomness that records the integers it draws, so that they
    can be replayed and shrunk for integrated shrinking.

    Parameters:
        draws : A sequence of integers to replay, clamped to the bounds asked.
        rng : The generator used once ``draws`` is exhausted, or ``None`` to
            fall back to the minimum, i.e. deterministic shrinking.
    """
    def __init__(self, draws=(), rng: _random.Random = None):
        self.draws, self.rng, self.taken = list(draws), rng, []

    def _next(self, low: int, high: int) -> int:
        index = len(self.taken)
        if index < len(self.draws):
            value = min(max(self.draws[index], low), high)
        elif self.rng is not None:
            value = self.rng.randint(low, high)
        else:
            value = low
        self.taken.append(value)
        return value

    def randint(self, low: int, high: int) -> int:
        """ Draw an integer in ``range(low, high + 1)``. """
        return self._next(low, high)

    def choice(self, seq):
        """ Draw an element from a non-empty sequence. """
        return seq[self._next(0, len(seq) - 1)]


def random_ty(
        ty: type, objects=DEFAULT_OBJECTS,
        min_length: int = 0, max_length: int = 3,
        seed: int = None, rng=None):
    """
    Generate a random type.

    Parameters:
        ty : The class of types to generate, e.g. :class:`monoidal.Ty`.
        objects : The pool of names to draw atomic objects from.
        min_length : The minimum length of the type.
        max_length : The maximum length of the type.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random source, ``random.Random(seed)`` by default.

    Example
    -------
    >>> from discopy.monoidal import Ty
    >>> assert random_ty(Ty, seed=420) == random_ty(Ty, seed=420)
    """
    rng = rng or _random.Random(seed)
    length = rng.randint(min_length, max_length)
    return ty(*(rng.choice(objects) for _ in range(length)))


def random_diagram(
        box: type, dom=None, n_boxes: int = 5, objects=DEFAULT_OBJECTS,
        max_cod_length: int = 2, seed: int = None, rng=None):
    """
    Generate a random diagram, by repeatedly whiskering and composing with
    boxes of a random domain (a sub-type of the current codomain) and a
    random codomain.

    Parameters:
        box : The class of boxes to generate, e.g. :class:`monoidal.Box`.
        dom : The domain of the diagram, random by default.
        n_boxes : The maximum number of boxes to compose, the actual number
            being drawn at random so that it can be shrunk.
        objects : The pool of names to draw atomic objects from.
        max_cod_length : The maximum length of the codomain of each box.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random source, ``random.Random(seed)`` by default.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> diagram = random_diagram(Box, Ty('x'), n_boxes=5, seed=420)
    >>> assert diagram.dom == Ty('x')
    """
    rng = rng or _random.Random(seed)
    if dom is None:
        dom = random_ty(box.ob, objects, 1, max_cod_length, rng=rng)
    ty = type(dom)
    diagram, cod = box.id(dom), dom
    for i in range(rng.randint(0, n_boxes)):
        left = rng.randint(0, len(cod))
        right = rng.randint(left, len(cod))
        box_cod = random_ty(ty, objects, 0, max_cod_length, rng=rng)
        new_box = box(f"f{i}", cod[left:right], box_cod)
        diagram = diagram >> cod[:left] @ new_box @ cod[right:]
        cod = cod[:left] @ box_cod @ cod[right:]
    return diagram


def random_parallel_pair(
        box: type, dom=None, n_boxes: int = 3, seed: int = None,
        rng=None, **kwargs):
    """
    Generate a random pair of parallel diagrams, i.e. with the same domain
    and codomain.

    Parameters:
        box : The class of boxes to generate.
        dom : The shared domain of the two diagrams, random by default.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random source, ``random.Random(seed)`` by default.
        kwargs : Passed to :func:`random_diagram`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f, g = random_parallel_pair(Box, Ty('x'), seed=420)
    >>> assert f.is_parallel(g)
    """
    rng = rng or _random.Random(seed)
    f = random_diagram(box, dom, n_boxes, rng=rng, **kwargs)
    g = random_diagram(box, f.dom, n_boxes, rng=rng, **kwargs)
    return f, g >> box("g", g.cod, f.cod)


def random_composable_pair(
        box: type, dom=None, n_boxes: int = 3, seed: int = None,
        rng=None, **kwargs):
    """
    Generate a random pair of composable diagrams, i.e. such that the
    codomain of the first is the domain of the second.

    Parameters:
        box : The class of boxes to generate.
        dom : The domain of the first diagram, random by default.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random source, ``random.Random(seed)`` by default.
        kwargs : Passed to :func:`random_diagram`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f, g = random_composable_pair(Box, Ty('x'), seed=420)
    >>> assert f.is_composable(g)
    """
    rng = rng or _random.Random(seed)
    f = random_diagram(box, dom, n_boxes, rng=rng, **kwargs)
    g = random_diagram(box, f.cod, n_boxes, rng=rng, **kwargs)
    return f, g


def random_composable_triple(
        box: type, dom=None, n_boxes: int = 3, seed: int = None,
        rng=None, **kwargs):
    """
    Generate a random triple of pairwise composable diagrams, e.g. for
    checking associativity of composition.

    Parameters:
        box : The class of boxes to generate.
        dom : The domain of the first diagram, random by default.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random source, ``random.Random(seed)`` by default.
        kwargs : Passed to :func:`random_diagram`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f, g, h = random_composable_triple(Box, Ty('x'), seed=420)
    >>> assert f.check_associativity(g, h)
    """
    rng = rng or _random.Random(seed)
    f, g = random_composable_pair(box, dom, n_boxes, rng=rng, **kwargs)
    h = random_diagram(box, g.cod, n_boxes, rng=rng, **kwargs)
    return f, g, h


def _replay(generator, predicate, draws):
    """ Replay ``draws`` through the generator and test the predicate. """
    source = _Source(draws)
    try:
        passed = bool(predicate(generator(source)))
    except Exception:  # An invalid example does not count as a counterexample.
        passed = True
    return passed, source.taken


def _candidates(draws):
    """ The smaller draw sequences to try when shrinking ``draws``. """
    for i in range(len(draws)):
        yield draws[:i] + draws[i + 1:]
    for i, value in enumerate(draws):
        for smaller in {value // 2, value - 1} if value > 0 else ():
            yield draws[:i] + [smaller] + draws[i + 1:]


def _shrink(generator, predicate, draws):
    """ Greedily shrink ``draws`` to a local minimum that still fails. """
    shrinking = True
    while shrinking:
        shrinking = False
        for candidate in _candidates(draws):
            passed, taken = _replay(generator, predicate, candidate)
            if not passed and (len(taken), sum(taken)) < (
                    len(draws), sum(draws)):
                draws, shrinking = taken, True
                break
    return draws


def check_property(
        generator: Callable[[_Source], object],
        predicate: Callable[[object], bool],
        n_trials: int = 100, seed: int = 0) -> bool:
    """
    Run a property-based test: check that ``predicate`` holds for examples
    drawn by ``generator`` over ``n_trials`` seeds, shrinking to a minimal
    counterexample if it does not.

    Parameters:
        generator : A function from a random source to an example.
        predicate : A function from an example to a boolean.
        n_trials : The number of random trials to run.
        seed : The first seed to try.

    Returns:
        ``True`` if the predicate held for every trial.

    Raises:
        Falsified : With the minimal counterexample, if the predicate failed.

    Example
    -------
    The interchange law does not hold on the nose for monoidal diagrams, only
    up to interchanger normal form. We generate a pair of boxes with atomic
    domain and codomain, then check that their two interchanger forms are
    syntactically equal:

    >>> from discopy.monoidal import Ty, Box
    >>> def two_boxes(rng):
    ...     atom = lambda: random_ty(Ty, min_length=1, max_length=1, rng=rng)
    ...     return Box('f', atom(), atom()), Box('g', atom(), atom())
    >>> def interchange_on_the_nose(boxes):
    ...     f, g = boxes
    ...     return f @ g.dom >> f.cod @ g == f.dom @ g >> f @ g.cod

    The smallest counterexample is a pair of boxes on a single atomic wire:

    >>> try:
    ...     check_property(two_boxes, interchange_on_the_nose)
    ... except Falsified as error:
    ...     f, g = error.example
    >>> assert f.dom == f.cod == g.dom == g.cod == Ty('a')

    These two diagrams are not equal on the nose, but they do become equal
    once we put them in interchanger normal form:

    >>> lhs, rhs = f @ g.dom >> f.cod @ g, f.dom @ g >> f @ g.cod
    >>> assert lhs != rhs
    >>> assert lhs.normal_form() == rhs.normal_form()
    """
    rng = _random.Random(seed)
    for _ in range(n_trials):
        source = _Source(rng=rng)
        if not predicate(generator(source)):
            minimal = _shrink(generator, predicate, source.taken)
            raise Falsified(generator(_Source(minimal)))
    return True
