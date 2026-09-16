# -*- coding: utf-8 -*-

"""
The free cartesian feedback category, i.e. feedback diagrams with a supply
of copy, e.g. to output a stream and feed it back at once.

The module is the meet of :mod:`discopy.feedback` and
:mod:`discopy.cartesian`: every class is a subclass of its two namesakes
and the module ends with the factory assignments. There is nothing else to
define because every generator refers to the others through a factory
resolved on the instance — e.g. :meth:`discopy.markov.Copy.dagger` builds
`self.merge_factory` — so that the reference lands back in this module.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Permutation
    Swap
    Copy
    Merge
    Discard
    Head
    Tail
    Feedback
    FollowedBy
    Functor
"""

from __future__ import annotations

from discopy import cartesian, feedback, hypergraph
from discopy.cat import factory
from discopy.feedback import Ty, Wire, HeadOb, TailOb  # noqa: F401


@factory
class Diagram(feedback.Diagram, cartesian.Diagram):
    """
    A cartesian feedback diagram is a feedback diagram with a supply of
    :class:`Copy`.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> x = Ty('x')
    >>> zero = Box('0', Ty(), x.head)
    >>> rand = Box('rand', Ty(), x)
    >>> plus = Box('+', x @ x, x)
    >>> walk = (rand.d @ x.d >> zero @ plus.d
    ...         >> FollowedBy(x) >> Copy(x)).feedback()
    >>> walk.draw(doctest="docs/_static/feedback/feedback-random-walk.svg")

    .. image:: /_static/feedback/feedback-random-walk.svg
        :align: center
    """
    ob = Ty


class Box(feedback.Box, cartesian.Box, Diagram):
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


class Copy(cartesian.Copy, Box):
    "The copy of an atomic feedback type some number of times."

    @property
    def d(self) -> Copy:
        return type(self)(self.dom.d, len(self.cod))


class Merge(cartesian.Merge, Box):
    "The merge of an atomic feedback type some number of times."

    @property
    def d(self) -> Merge:
        return type(self)(self.cod.d, len(self.dom))


class Discard(cartesian.Discard, Copy):
    "The discard of an atomic feedback type."


class Head(feedback.Head, Box):
    "The head of a cartesian feedback diagram."


class Tail(feedback.Tail, Box):
    "The tail of a cartesian feedback diagram."


class Feedback(feedback.Feedback, Box):
    "The feedback bubble of a cartesian feedback diagram."


class FollowedBy(feedback.FollowedBy, Box):
    "The isomorphism between `x.head @ x.tail.d` and `x`."


class Functor(feedback.Functor, cartesian.Functor):
    """
    A cartesian feedback functor is a feedback functor that also preserves
    copies.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.discard_factory = Discard
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.feedback_factory, Diagram.followed_by = Feedback, FollowedBy
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(feedback.Equation):
    """ The :class:`feedback.Equation` of cartesian feedback diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
