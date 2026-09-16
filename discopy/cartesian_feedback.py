# -*- coding: utf-8 -*-

"""
The free cartesian feedback category, i.e. feedback diagrams with a supply
of :class:`Copy` and :class:`Merge` borrowed from :mod:`discopy.markov`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Copy
    Merge
    Functor
"""

from __future__ import annotations

from discopy import markov, feedback, hypergraph
from discopy.abc import CartesianCategory
from discopy.cat import factory, Generator
from discopy.feedback import Ty, Wire, HeadOb, TailOb  # noqa: F401


@factory
class Diagram(feedback.Diagram, markov.Diagram, CartesianCategory):
    """
    A cartesian feedback diagram is a feedback diagram with a supply of
    :class:`Copy` read as deterministic, e.g. to output a stream and feed
    it back at once.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> x = Ty('x')
    >>> zero = Box('0', Ty(), x.head)
    >>> rand = Box('rand', Ty(), x)
    >>> plus = Box('+', x @ x, x)
    >>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
    ...         >> FollowedBy(x) >> Copy(x)).feedback()
    >>> walk.draw(doctest="docs/_static/feedback/feedback-random-walk.svg")

    .. image:: /_static/feedback/feedback-random-walk.svg
        :align: center
    """
    ob = Ty

    @Generator()
    def copy_factory(cls):
        return Copy

    @Generator()
    def merge_factory(cls):
        return Merge


Box, Permutation, Swap, Head, Tail, Feedback, FollowedBy = (
    Diagram.generator_factory, Diagram.permutation_factory,
    Diagram.swap_factory, Diagram.head_factory, Diagram.tail_factory,
    Diagram.feedback_factory, Diagram.followed_by)


class Copy(markov.Copy, Box):
    "A :class:`markov.Copy` in a cartesian feedback diagram."
    def delay(self, n_steps=1):
        return type(self)(self.dom.delay(n_steps), len(self.cod))


class Merge(markov.Merge, Box):
    "A :class:`markov.Merge` in a cartesian feedback diagram."
    def delay(self, n_steps=1):
        return type(self)(self.cod.delay(n_steps), len(self.dom))


Discard, Sum, Bubble = (
    Diagram.discard_factory, Diagram.sum_factory, Diagram.bubble_factory)


class Functor(feedback.Functor, markov.Functor):
    """
    A cartesian feedback functor is a feedback functor that also preserves
    copies.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(feedback.Equation):
    """ The :class:`feedback.Equation` of cartesian feedback diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
