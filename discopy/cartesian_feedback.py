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
    Box
    Swap
    Copy
    Merge
    Discard
    Functor
"""

from __future__ import annotations

from discopy import markov, feedback, hypergraph
from discopy.abc import CartesianCategory
from discopy.cat import factory
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

    @property
    def head(self):
        """ Syntactic sugar for :class:`Head`. """
        return Head(self)

    @property
    def tail(self):
        """ Syntactic sugar for :class:`Tail`. """
        return Tail(self)


class Box(feedback.Box, markov.Box, Diagram):
    """
    A cartesian feedback box is a feedback box in a cartesian feedback
    diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(feedback.Permutation, Box):
    "A permutation in a cartesian feedback diagram."


class Swap(feedback.Swap, Permutation):
    "A swap in a cartesian feedback diagram."


class Copy(Box, markov.Copy):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    The :class:`Box` comes first so that ``factory`` resolves to
    :class:`Diagram` rather than :class:`markov.Diagram`.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: Ty, n: int = 2):
        markov.Copy.__init__(self, x, n)
        Box.__init__(self, self.name, self.dom, self.cod)

    def dagger(self) -> Merge:
        return Merge(self.dom, len(self.cod))

    __repr__ = markov.Copy.__repr__

    def delay(self, n_steps=1):
        return type(self)(self.dom.delay(n_steps), len(self.cod))


class Merge(Box, markov.Merge):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: Ty, n: int = 2):
        markov.Merge.__init__(self, x, n)
        Box.__init__(self, self.name, self.dom, self.cod)

    def dagger(self) -> Copy:
        return Copy(self.cod, len(self.dom))

    __repr__ = markov.Merge.__repr__

    def delay(self, n_steps=1):
        return type(self)(self.cod.delay(n_steps), len(self.dom))


class Discard(Copy):
    """
    The discard of an atomic type :code:`x`.

    Parameters:
        x : The type to discard.
    """
    def __init__(self, x: Ty, *args, **kwargs):
        super().__init__(x, 0)


class Head(feedback.Head, Box):
    "The head of a cartesian feedback diagram."


class Tail(feedback.Tail, Box):
    "The tail of a cartesian feedback diagram."


class Feedback(feedback.Feedback, Box):
    "The feedback bubble on a cartesian feedback diagram."


class FollowedBy(feedback.FollowedBy, Box):
    "The isomorphism between `x.head @ x.tail.delay()` and `x`."


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
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.discard_factory = Discard
Diagram.feedback_factory, Diagram.followed_by = Feedback, FollowedBy
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(feedback.Equation):
    """ The :class:`feedback.Equation` of cartesian feedback diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
