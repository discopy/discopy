# -*- coding: utf-8 -*-

"""
What a box is, before there is a tensor: a parametric map.

Two notions, and the difference between them is the whole point of this
package.

An **ordinary parametric map** :math:`(P, f) : X \\to Y` is a map

.. math:: f : P \\otimes X \\to Y

from a parameter object :math:`P` and an input to an output.  That is a
feed-forward layer, and it is *not* what a
:class:`~discopy.neural.Network` is.

A **parametric interaction map** :math:`(P, \\Phi) : X \\to Y` is a map on
the *boundary* of a box.  Write the boundary as

.. math:: \\partial f = X^* \\otimes Y,

the inputs read as the outputs of whatever is upstream, together with the
outputs; then a local neural interaction is

.. math:: \\Phi : P \\otimes \\partial f \\to \\partial f,

reading one incoming message on every port and emitting one outgoing
message on every port.  This is the local half of the execution formula of
the geometry of interaction, and it is exactly what the torch module of a
:class:`~discopy.neural.Network` computes: ``R ** width -> R ** width`` for
``width`` the sum of the domain and codomain dimensions.  A cell of
:mod:`discopy.neural.cells` answers *every* leg of its orbit, its inputs
included, which no ``X -> Y`` signature can say.

Nothing here computes.  These are specifications: they carry a domain, a
codomain, a parameter object, the identity of the generator and the laws
it promises, and they say how those compose.  The numerics stay in
:class:`~discopy.neural.CMap`, which remains the only implementation of a
forward pass; :func:`interaction_spec` reads the metadata off a
:class:`~discopy.neural.Network` without touching it.

Note
----
``X*`` is represented by the same :class:`~discopy.neural.Dim` data as
``X``, because every atomic dimension is self-dual.  The *order* is where
the two differ: ``Dim(2, 3).r == Dim(3, 2)`` reverses a composite type,
whereas a module reads its domain ports in domain order -- which is what
:meth:`~discopy.neural.CMap.box_ports` restores when it un-reverses the
clockwise storage.  So :attr:`InteractionMap.boundary` is ``dom @ cod``:
the same object as :math:`X^* \\otimes Y` up to the symmetry that puts the
duals back in order, and the one whose port order the executable layout
actually uses.

Like :mod:`discopy.neural.core`, this module does not import ``torch``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Parametric
    ParamMap
    InteractionMap

Example
-------

>>> from discopy.neural import Dim, Network
>>> f = Network("f", Dim(2), Dim(3))
>>> spec = interaction_spec(f)
>>> spec.boundary, spec.width
(Dim(2, 3), 5)
>>> interaction_spec(f.dagger()) == spec.dagger()
True
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy import messages
from discopy.neural.core import Dim
from discopy.utils import AxiomError


@dataclass(frozen=True)
class Parametric:
    """
    The bookkeeping the two parametric notions share: a name, a domain, a
    codomain, a parameter object and the laws promised.

    Composition and tensor put the parameter objects side by side, left
    then right, and that order is the whole content of the definition:
    :math:`(Q, g) \\circ (P, f)` has parameter object :math:`P \\otimes Q`.
    Since :class:`~discopy.neural.Dim` is a strict monoid this is strictly
    associative and strictly unital -- on the *parametric data*, i.e. on
    the domain, the codomain and the parameter object.  The name is a
    label recording how a composite was built, so it is the one field the
    unit law does not leave alone.

    Parameters:
        name : The identity of the generator.
        dom : The domain :math:`X`.
        cod : The codomain :math:`Y`.
        params : The parameter object :math:`P`, the unit by default.
        laws : The laws the map promises, see :mod:`discopy.neural.laws`.

    Note
    ----
    Composition and tensor forget the laws.  A law is a statement about the
    legs of *one* map, and the laws of a composite are not the union of its
    parts': there is nothing left to say about the ports that were glued,
    and the legs that survive have been renumbered.

    The two subclasses never compose with one another: they read the same
    bookkeeping in two different ways, and mixing the readings would be a
    category error rather than a shorthand.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> ParamMap("f", Dim(2), Dim(3)) >> InteractionMap("g", Dim(3), Dim(4))
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for >>: 'ParamMap' and \
'InteractionMap'
    """

    name: str
    dom: Dim
    cod: Dim
    params: Dim = Dim()
    laws: tuple = ()

    @classmethod
    def id(cls, dom: Dim = Dim()) -> Parametric:
        """
        The identity map on an object: no parameters, nothing promised.

        Parameters:
            dom : The object.

        Example
        -------
        >>> from discopy.neural import Dim
        >>> f = ParamMap("f", Dim(2), Dim(3), Dim(6))
        >>> def data(one):
        ...     return one.dom, one.cod, one.params
        >>> data(ParamMap.id(Dim(2)) >> f) == data(f)
        True
        >>> data(f >> ParamMap.id(Dim(3))) == data(f)
        True
        """
        return cls("Id", dom, dom)

    def __rshift__(self, other: Parametric) -> Parametric:
        if type(other) is not type(self):
            return NotImplemented
        if self.cod != other.dom:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.name, other.name, self.cod, other.dom))
        return type(self)(f"({self.name} >> {other.name})", self.dom,
                          other.cod, self.params @ other.params)

    def __matmul__(self, other: Parametric) -> Parametric:
        if type(other) is not type(self):
            return NotImplemented
        return type(self)(f"({self.name} @ {other.name})",
                          self.dom @ other.dom, self.cod @ other.cod,
                          self.params @ other.params)


@dataclass(frozen=True)
class ParamMap(Parametric):
    """
    An ordinary parametric map :math:`(P, f) : X \\to Y`, i.e. a map
    :math:`f : P \\otimes X \\to Y`.

    A layer of a feed-forward network is one of these.  A
    :class:`~discopy.neural.Network` is *not*: see :class:`InteractionMap`.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> f = ParamMap("f", Dim(2), Dim(3), Dim(6))
    >>> g = ParamMap("g", Dim(3), Dim(4), Dim(12))
    >>> (f >> g).name, (f >> g).dom, (f >> g).cod, (f >> g).params
    ('(f >> g)', Dim(2), Dim(4), Dim(6, 12))
    >>> (f @ g).dom, (f @ g).cod, (f @ g).params
    (Dim(2, 3), Dim(3, 4), Dim(6, 12))
    >>> f >> f
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: f does not compose with f: Dim(3) != Dim(2).
    """


@dataclass(frozen=True)
class InteractionMap(Parametric):
    """
    A parametric interaction map :math:`(P, \\Phi) : X \\to Y`, i.e. a map
    :math:`\\Phi : P \\otimes (X^* \\otimes Y) \\to X^* \\otimes Y` on the
    boundary of a box.

    This is the formal reading of a :class:`~discopy.neural.Network`: same
    name, same domain, same codomain, and a module whose input and output
    both live on :attr:`boundary`.

    Note
    ----
    :meth:`~Parametric.__rshift__` records the boundary bookkeeping of a
    composite and nothing more.  Two interaction maps glued along a shared
    :math:`Y` do not compose by substitution: they talk to each other along
    the wires, which is symmetric feedback -- the trace of the two boxes
    over the shared boundary -- and what computes it is a finite number of
    rounds of :meth:`~discopy.neural.CMap.forward`.  See
    :mod:`discopy.neural.dynamics`.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> f = InteractionMap("f", Dim(2), Dim(3), Dim(25))
    >>> g = InteractionMap("g", Dim(5), Dim(7), Dim(49))
    >>> f.boundary, f.width
    (Dim(2, 3), 5)
    >>> (f >> InteractionMap("h", Dim(3), Dim(4))).boundary
    Dim(2, 4)

    The boundary of a tensor is not the tensor of the boundaries -- it
    interleaves, which is why a map lays out one contiguous block per
    *box* rather than one per boundary:

    >>> (f @ g).boundary, f.boundary @ g.boundary
    (Dim(2, 5, 3, 7), Dim(2, 3, 5, 7))
    >>> (f @ g).width == f.width + g.width
    True
    """

    @property
    def boundary(self) -> Dim:
        """
        The boundary :math:`\\partial f = X^* \\otimes Y`, in the port order
        the executable module reads: the domain then the codomain.
        """
        return self.dom @ self.cod

    @property
    def width(self) -> int:
        """ The flat width of the boundary, i.e. of the module. """
        return sum(self.boundary.inside)

    def dagger(self) -> InteractionMap:
        """
        The same interaction read backwards, :math:`(P, \\Phi)^\\dagger : Y
        \\to X`.

        Its boundary is the same object up to the symmetry exchanging the
        two halves, which is why
        :meth:`~discopy.neural.Network.dagger` can reuse the module: the
        weights are read in the new port order.

        Example
        -------
        >>> from discopy.neural import Dim
        >>> f = InteractionMap("f", Dim(2), Dim(3), Dim(25))
        >>> f.dagger().boundary, f.dagger().width
        (Dim(3, 2), 5)
        >>> f.dagger().dagger() == f
        True
        """
        return type(self)(self.name, self.cod, self.dom, self.params,
                          self.laws)


def interaction_spec(network, laws: tuple = ()) -> InteractionMap:
    """
    The formal interaction map a :class:`~discopy.neural.Network` realises.

    Read-only and side-effect free: it registers nothing, wraps nothing,
    owns no parameter and takes no part in a forward pass.  The network is
    the implementation; this is what it implements.

    The parameter object is ``Dim(n)`` for ``n`` the number of scalars the
    module holds, and the unit when the network has no module or its data
    is not a torch one -- reading it needs no ``torch`` import, only
    ``module.parameters()``.

    Parameters:
        network : The network to read.
        laws : The laws to attach, e.g.
               :func:`discopy.neural.laws.symmetry` of the signature of the
               site the network fills.

    Example
    -------
    >>> import torch
    >>> from discopy.neural import Dim, Network
    >>> f = Network("f", Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    >>> spec = interaction_spec(f)
    >>> spec.name, spec.boundary, spec.params
    ('f', Dim(2, 3), Dim(30))
    >>> spec.width == f.module.in_features == f.module.out_features
    True

    A network with no module, or with data that is not a torch module,
    still has a boundary:

    >>> interaction_spec(Network("g", Dim(2), Dim(3))).params
    Dim(0)
    >>> interaction_spec(Network("g", Dim(2), Dim(3), module=object())).params
    Dim(0)
    """
    module = getattr(network, "module", None)
    parameters = getattr(module, "parameters", None)
    params = Dim() if parameters is None else Dim(
        sum(parameter.numel() for parameter in parameters()))
    return InteractionMap(network.name, network.dom, network.cod, params,
                          tuple(laws))
