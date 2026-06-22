# -*- coding: utf-8 -*-

"""
Random generation of diagrams, for property-based testing.

This module provides functions for generating seeded random objects and
diagrams, as well as composable and parallel tuples of diagrams. These can
be fed into the ``check_*`` methods defined on the classes of
:mod:`discopy.abc`, in order to test that the axioms of a category hold
(possibly up to some notion of equality) for randomly generated examples.

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

>>> def associativity_holds(seed):
...     f, g, h = random_composable_triple(
...         symmetric.Box, symmetric.Ty('x'), seed=seed)
...     return f.check_associativity(g, h)
>>> assert not check_property(associativity_holds)

>>> def unitality_holds(seed):
...     f = random_diagram(symmetric.Box, symmetric.Ty('x'), seed=seed)
...     return f.check_unitality()
>>> assert not check_property(unitality_holds)

Some axioms only hold up to some notion of equality, e.g. the swap is its
own inverse only up to :attr:`symmetric.Diagram.hypergraph_equality`.

>>> def swap_inverse_holds(seed):
...     x = random_ty(symmetric.Ty, seed=seed)
...     y = random_ty(symmetric.Ty, seed=seed + 1)
...     with symmetric.Diagram.hypergraph_equality:
...         return symmetric.Diagram.check_swap_inverse(x, y)
>>> assert not check_property(swap_inverse_holds)
"""

from __future__ import annotations

import random as _random
from string import ascii_lowercase as ALPHABET
from typing import Callable

DEFAULT_OBJECTS = ALPHABET


def random_ty(
        ty: type, objects=DEFAULT_OBJECTS,
        min_length: int = 0, max_length: int = 3,
        seed: int = None, rng: _random.Random = None):
    """
    Generate a random type.

    Parameters:
        ty : The class of types to generate, e.g. :class:`monoidal.Ty`.
        objects : The pool of names to draw atomic objects from.
        min_length : The minimum length of the type.
        max_length : The maximum length of the type.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random number generator, ``random.Random(seed)`` by default.

    Example
    -------
    >>> from discopy.monoidal import Ty
    >>> assert random_ty(Ty, seed=420) == random_ty(Ty, seed=420)
    """
    rng = rng or _random.Random(seed)
    length = rng.randint(min_length, max_length)
    return ty(*(rng.choice(objects) for _ in range(length)))


def random_diagram(
        box: type, dom, n_boxes: int = 5, objects=DEFAULT_OBJECTS,
        max_cod_length: int = 2,
        seed: int = None, rng: _random.Random = None):
    """
    Generate a random diagram with domain ``dom``, by repeatedly whiskering
    and composing with boxes of a random domain (a sub-type of the current
    codomain) and a random codomain.

    Parameters:
        box : The class of boxes to generate, e.g. :class:`monoidal.Box`.
        dom : The domain of the diagram.
        n_boxes : The number of boxes to compose.
        objects : The pool of names to draw atomic objects from.
        max_cod_length : The maximum length of the codomain of each box.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random number generator, ``random.Random(seed)`` by default.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> diagram = random_diagram(Box, Ty('x'), n_boxes=5, seed=420)
    >>> assert diagram.dom == Ty('x')
    """
    rng = rng or _random.Random(seed)
    ty = type(dom)
    diagram, cod = box.id(dom), dom
    for i in range(n_boxes):
        left = rng.randint(0, len(cod))
        right = rng.randint(left, len(cod))
        box_dom, box_cod = cod[left:right], random_ty(
            ty, objects, 0, max_cod_length, rng=rng)
        new_box = box(f"f{i}", box_dom, box_cod)
        diagram = diagram >> cod[:left] @ new_box @ cod[right:]
        cod = cod[:left] @ box_cod @ cod[right:]
    return diagram


def random_parallel_pair(
        box: type, dom, n_boxes: int = 3,
        seed: int = None, rng: _random.Random = None, **kwargs):
    """
    Generate a random pair of parallel diagrams, i.e. with the same domain.

    Parameters:
        box : The class of boxes to generate.
        dom : The shared domain of the two diagrams.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random number generator, ``random.Random(seed)`` by default.
        kwargs : Passed to :func:`random_diagram`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f, g = random_parallel_pair(Box, Ty('x'), seed=420)
    >>> assert f.dom == g.dom == Ty('x')
    """
    rng = rng or _random.Random(seed)
    f = random_diagram(box, dom, n_boxes, rng=rng, **kwargs)
    g = random_diagram(box, dom, n_boxes, rng=rng, **kwargs)
    return f, g


def random_composable_pair(
        box: type, dom, n_boxes: int = 3,
        seed: int = None, rng: _random.Random = None, **kwargs):
    """
    Generate a random pair of composable diagrams, i.e. such that the
    codomain of the first is the domain of the second.

    Parameters:
        box : The class of boxes to generate.
        dom : The domain of the first diagram.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random number generator, ``random.Random(seed)`` by default.
        kwargs : Passed to :func:`random_diagram`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f, g = random_composable_pair(Box, Ty('x'), seed=420)
    >>> assert f.cod == g.dom
    """
    rng = rng or _random.Random(seed)
    f = random_diagram(box, dom, n_boxes, rng=rng, **kwargs)
    g = random_diagram(box, f.cod, n_boxes, rng=rng, **kwargs)
    return f, g


def random_composable_triple(
        box: type, dom, n_boxes: int = 3,
        seed: int = None, rng: _random.Random = None, **kwargs):
    """
    Generate a random triple of pairwise composable diagrams, e.g. for
    checking associativity of composition.

    Parameters:
        box : The class of boxes to generate.
        dom : The domain of the first diagram.
        n_boxes : The number of boxes to compose for each diagram.
        seed : A seed for reproducibility, ignored if ``rng`` is given.
        rng : A random number generator, ``random.Random(seed)`` by default.
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


def check_property(
        predicate: Callable[[int], bool],
        n_trials: int = 100, seed: int = 0) -> list[int]:
    """
    Run a property-based test, i.e. check that ``predicate`` holds for
    ``n_trials`` random seeds, returning the list of seeds for which it
    fails (so that an empty list means the property held for every trial).

    Parameters:
        predicate : A function from a seed to a boolean.
        n_trials : The number of random trials to run.
        seed : The first seed to try, the following ones being ``seed + 1``,
            ``seed + 2``, etc.

    Example
    -------
    >>> assert not check_property(lambda seed: seed >= 0, n_trials=10)
    >>> failures = check_property(lambda seed: seed >= 5, n_trials=10)
    >>> assert failures == [0, 1, 2, 3, 4]
    """
    return [seed + i for i in range(n_trials) if not predicate(seed + i)]
