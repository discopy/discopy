# -*- coding: utf-8 -*-

"""
The traced category of feedforward neural networks: lists of shapes as
objects, layers as boxes, with copy, discard, swap and trace.

A :class:`Dims` is a list of :class:`Dim`, i.e. a tuple of tensors with one
shape per leg. A :class:`Box` from ``Dims(a_1, ..., a_m)`` to
``Dims(b_1, ..., b_n)`` is a layer taking ``m`` tensors and returning ``n``,
computed by its optional :attr:`Box.module`. A :class:`Network` is a diagram
of such layers: composition ``>>`` feeds the outputs of one into the inputs
of the next, tensor ``@`` runs two side by side, :meth:`Network.copy` sends
an activation to several layers, :meth:`Network.discard` drops one and
:meth:`Network.trace` feeds an output back into an input, as a recurrent
network does. Networks form the free traced Markov category on their boxes,
see :mod:`discopy.markov` and :mod:`discopy.traced`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dims
    Network
    Box
    Swap
    Permutation
    Copy
    Merge
    Discard
    Trace
    Functor

Example
-------

A residual block copies its input, applies a layer to one copy and adds the
two back together; a recurrent cell feeds one of its outputs back in.

>>> x, h = Dims(4), Dims(8)
>>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
>>> residual = Network.copy(x) >> layer @ x >> add
>>> assert residual.dom == residual.cod == x
>>> cell = Box('cell', x @ h, x @ h)
>>> recurrent = cell.trace()
>>> assert recurrent.dom == recurrent.cod == x
>>> Equation(residual, recurrent, symbol='').draw(
...     doctest="docs/_static/neural/residual-and-recurrent.svg")

.. image:: /_static/neural/residual-and-recurrent.svg
    :align: center
"""

from __future__ import annotations

from discopy import markov, monoidal
from discopy.cat import factory
from discopy.monoidal import Dim
from discopy.utils import assert_isinstance, factory_name


@factory
class Dims(monoidal.Ty):
    """
    A list of shapes, i.e. the free monoid over :class:`Dim` with
    concatenation as tensor ``@`` and the empty list as unit: a tuple of
    tensors, one per leg.

    Together with the product ``*`` of shapes, which distributes over the
    concatenation, this is the free rig on the natural numbers: ``Dims()``
    is the zero and ``Dims(1)`` the one, both operations commutative only
    up to :class:`Swap`.

    Parameters:
        inside : The shape of each leg, an integer standing for a vector.

    Example
    -------
    >>> assert Dims(2, 3) == Dims(Dim(2), Dim(3)) == Dims(2) @ Dims(3)
    >>> assert Dims(Dim(2, 3)) != Dims(2, 3) and Dims(1) == Dims(Dim())
    >>> Dims(2, 3) * Dims(4)
    Dims(Dim(2, 4), Dim(3, 4))
    >>> from discopy.utils import dumps, loads
    >>> assert loads(dumps(Dims(2, Dim(2, 3)))) == Dims(2, Dim(2, 3))
    """
    generator_factory = Dim

    def __init__(self, *inside: int | Dim, **kwargs):
        inside = kwargs.pop('inside', inside)
        super().__init__(
            *(Dim(x) if isinstance(x, int) else x for x in inside), **kwargs)

    def __mul__(self, other: Dims) -> Dims:
        assert_isinstance(other, Dims)
        return type(self)(*(x @ y for x in self.inside for y in other.inside))

    def __repr__(self):
        return f"Dims({', '.join(map(repr, self.inside))})"

    __str__ = __repr__


@factory
class Network(markov.Diagram):
    """
    A network is a Markov diagram with lists of shapes as objects and
    :class:`Box` layers inside, together with the trace.

    Parameters:
        inside (Layer) : The layers of the network.
        dom (Dims) : The input legs of the network.
        cod (Dims) : The output legs of the network.

    Note
    ----
    Networks can be written as Python functions of their input legs.

    >>> x = Dims(4)
    >>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
    >>> @Network.from_callable(x, x)
    ... def residual(v):
    ...     return add(layer(v), v)
    >>> assert residual == Network.copy(x) >> layer @ x >> add
    """
    ob = Dims


class Box(markov.Box, Network):
    """
    A box is a layer taking one tensor per leg of ``dom`` and returning one
    per leg of ``cod``, computed by its ``module`` when it has one.

    Parameters:
        name : The name of the layer.
        dom : The input legs of the layer.
        cod : The output legs of the layer.
        module : The object computing the layer, e.g. a ``torch.nn.Module``.

    Note
    ----
    The module is the ``data`` of the box, so boxes compare equal when they
    have the same name, legs and module, a framework's modules comparing by
    identity. The repr and the serialisation omit the module, which has no
    eval-able representation, so ``eval(repr(f)) == f`` and
    ``loads(dumps(f)) == f`` hold for a box without one and give the shape
    of one with.

    Example
    -------
    >>> f = Box('f', Dims(2), Dims(3, 3), module=lambda x: (x, x))
    >>> assert f.module is f.data is f.dagger().module
    >>> f
    neural.network.Box('f', Dims(Dim(2)), Dims(Dim(3), Dim(3)))
    """
    module = None

    def __init__(self, name: str, dom: Dims, cod: Dims,
                 module: object = None, data=None, **params):
        self.module = data if module is None else module
        super().__init__(name, dom, cod, data=self.module, **params)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return f"{factory_name(type(self))}("\
            f"{self.name!r}, {self.dom!r}, {self.cod!r})"

    def to_tree(self) -> dict:
        tree = super().to_tree()
        tree.pop('data', None)
        return tree


class Permutation(markov.Permutation, Box):
    """
    A permutation of the legs of a network.

    Parameters:
        dom (Dims) : The legs to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """


class Swap(Permutation, markov.Swap, Box):
    """
    The swap of two legs.

    Parameters:
        left (Dims) : The leg on the top left and bottom right.
        right (Dims) : The leg on the top right and bottom left.
    """


class Copy(markov.Copy, Box):
    """
    The copy of a leg some ``n`` number of times.

    Parameters:
        x (Dims) : The leg to copy.
        n : The number of copies.
    """


class Merge(markov.Merge, Box):
    """
    The merge of ``n`` copies of a leg, the dagger of :class:`Copy`.

    Parameters:
        x (Dims) : The leg to merge.
        n : The number of copies.
    """


class Discard(markov.Discard, Copy):
    """
    The discard of a leg.

    Parameters:
        x (Dims) : The leg to discard.
    """


class Trace(markov.Trace, Box):
    """
    A trace feeds an output leg of a network back into an input leg.

    Parameters:
        arg : The network to trace.
        left : Whether to trace the leftmost or the rightmost leg.
    """


class Functor(markov.Functor):
    """
    A functor from networks to any Markov category with traces, e.g. an
    evaluation of the layers as functions.

    Parameters:
        ob_map (Mapping[Dims, Dims]) :
            Map from atomic :class:`Dims` to :code:`cod.ob`.
        ar_map (Mapping[Box, Network]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Network` by default.

    Example
    -------
    >>> from discopy import python
    >>> x = Dims(2)
    >>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
    >>> residual = Network.copy(x) >> layer @ x >> add
    >>> F = Functor(
    ...     ob_map={x: (list, )},
    ...     ar_map={layer: lambda v: [2 * i for i in v],
    ...             add: lambda v, w: [i + j for i, j in zip(v, w)]},
    ...     cod=python.Function)
    >>> assert F(residual)([1, 2]) == [3, 6]
    """
    dom = cod = Network


Id = Network.id
Equation = markov.Equation

Network.functor_factory = Functor
Network.copy_factory, Network.merge_factory = Copy, Merge
Network.discard_factory = Discard
Network.swap_factory = Swap
Network.permutation_factory = Permutation
Network.trace_factory = Trace
