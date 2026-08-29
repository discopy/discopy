# -*- coding: utf-8 -*-

"""
From a diagram to a global interaction: what a generator means, and what a
whole diagram compiles to.

Two notions, and the difference between them is the whole point of this
package.

An **ordinary parametric map** :math:`(P, f) : X \\to Y` is a map

.. math:: f : P \\otimes X \\to Y

from a parameter object and an input to an output.  That is a feed-forward
layer, and it is *not* what a :class:`~discopy.neural.Network` is.  These
are the morphisms of :math:`\\mathrm{Para}`, they compose by substitution,
and :class:`ParamMap` records them.

A **parametric interaction map** :math:`(P, \\Phi) : X \\to Y` is a map on
the *boundary* of a box.  Write the boundary as

.. math:: \\partial f = X^* \\otimes Y,

the inputs read as the outputs of whatever is upstream, together with the
outputs; then a local neural interaction is

.. math:: \\Phi_f : P_f \\otimes \\partial f \\to \\partial f,

reading one incoming message on every port and emitting one outgoing
message on every port.  This is the local half of the execution formula of
the geometry of interaction :cite:p:`Abramsky96`, and it is exactly what
the torch module of a :class:`~discopy.neural.Network` computes:
``R ** width -> R ** width`` for ``width`` the sum of the domain and
codomain dimensions.  A cell of :mod:`discopy.neural.cells` answers *every*
leg of its orbit, its inputs included, which no ``X -> Y`` signature can
say.  :class:`InteractionMap` records these, and deliberately does **not**
compose: two interaction maps glued along a shared object do not compose by
substitution, they talk to each other along wires.  That is the next
paragraph.

The global interaction
----------------------

A diagram wires the boundaries together.  Interpreting it -- each atomic
role to the :class:`~discopy.neural.Dim` it carries, each generator name to
the torch module computing it -- gives a closed
:class:`~discopy.neural.CMap`, and one synchronous round of message passing
is the two halves in sequence: every box interacts with the messages on its
own ports, then the wires carry each emission to the other end.  Writing
:math:`\\Phi_\\theta` for the parallel application of every local
interaction and :math:`\\sigma_D` for the permutation of the flat state
induced by the wiring,

.. math:: T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S_D \\to S_D

on the state object :math:`S_D = \\bigoplus_p \\mathbb{R}^{w_p}`, one summand
per port.  :class:`Interaction` is that compiled object: the closed map, the
port index of every ``(generator name, role)`` family, and
:meth:`Interaction.advance`, the one implementation of :math:`T^n`.

When an initial message vector :math:`i` is re-injected -- ``inject=True``
-- the round is

.. math:: T_{D,\\theta,i}(s) = \\sigma_D(\\Phi_\\theta(s)) + i,

an affine, not a linear, dependence on :math:`i`: the vector is added back
to the *whole* state after routing, every round.

Four notions that are easy to conflate, kept apart
--------------------------------------------------

* a **categorical trace** is a structural operation on wiring.  A
  self-wired pair of ports *is* the trace of the compact target, and a
  functor into :mod:`discopy.neural` preserves it strictly and for free.
* a **persistent state channel** -- delayed feedback in the sense of
  :mod:`discopy.feedback` -- is what that same pair does across rounds:
  what a box writes on one end it reads on the other one round later.
* **finite iteration** is what running the map computes.  ``n`` rounds
  compute :math:`T^n(s_0)` and nothing more.  What holds unconditionally is
  resumption, :math:`T^{a+b} = T^b \\circ T^a`, which is why a segmented
  solver can stop and carry on -- and it holds for *one* transition, so a
  run resumed from its own carried state only resumes when ``inject`` is
  off.
* a **fixed point** of :math:`T` is a fourth thing.  If some :math:`T`
  happens to be a contraction then :math:`T^n` converges, but that is an
  analytic property of the learned weights, to be measured -- never
  something the category supplies.  :class:`~discopy.neural.FixedPoint` is
  the solver that looks for one and :meth:`Interaction.residual` is the
  number that says whether it found one.

Note
----
``X*`` is represented by the same :class:`~discopy.neural.Dim` data as
``X``, because every atomic dimension is self-dual.  The *order* is where
the two differ: ``Dim(2, 3).r == Dim(3, 2)`` reverses a composite type,
whereas a module reads its domain ports in domain order -- which is what
:func:`box_ports` restores when it un-reverses the clockwise storage.  So
:attr:`InteractionMap.boundary` is ``dom @ cod``.

This module imports ``torch`` lazily, inside the methods that need it, so
that a diagram can be compiled and inspected on a machine without it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    ParamMap
    InteractionMap
    Interaction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from discopy import messages
from discopy.neural.core import CMap, Dim, Functor, Network, box_ports
from discopy.utils import AxiomError, assert_isinstance

if TYPE_CHECKING:
    import torch


# --- what a generator is ---------------------------------------------------

@dataclass(frozen=True)
class Parametric:
    """
    The bookkeeping the two parametric notions share: a name, a domain, a
    codomain, a parameter object and the laws promised.

    The tensor puts the parameter objects side by side, left then right,
    and that order is the whole content of the definition.  Since
    :class:`~discopy.neural.Dim` is a strict monoid this is strictly
    associative and strictly unital on the parametric data, i.e. on the
    domain, the codomain and the parameter object.

    Parameters:
        name : The identity of the generator.
        dom : The domain :math:`X`.
        cod : The codomain :math:`Y`.
        params : The parameter object :math:`P`, the unit by default.
        laws : The laws promised, see :mod:`discopy.neural.laws`.

    Note
    ----
    The tensor forgets the laws.  A law is a statement about the legs of
    *one* map, and the laws of a product are not the union of its parts':
    the legs have been renumbered.
    """

    name: str
    dom: Dim
    cod: Dim
    params: Dim = Dim()
    laws: tuple = ()

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

    A layer of a feed-forward network is one of these, and they compose by
    substitution: :math:`(Q, g) \\circ (P, f)` has parameter object
    :math:`P \\otimes Q`.  A :class:`~discopy.neural.Network` is *not* one:
    see :class:`InteractionMap`.

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

    @classmethod
    def id(cls, dom: Dim = Dim()) -> ParamMap:
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
        >>> data(ParamMap.id(Dim(2)) >> f) == data(f) == data(
        ...     f >> ParamMap.id(Dim(3)))
        True
        """
        return cls("Id", dom, dom)

    def __rshift__(self, other: ParamMap) -> ParamMap:
        if type(other) is not type(self):
            return NotImplemented
        if self.cod != other.dom:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.name, other.name, self.cod, other.dom))
        return type(self)(f"({self.name} >> {other.name})", self.dom,
                          other.cod, self.params @ other.params)


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
    Interaction maps do **not** compose by substitution, and ``>>`` refuses
    rather than pretending.  Two of them glued along a shared object talk
    to each other along the wires: that is symmetric feedback -- the trace
    of the two boxes over the shared boundary -- and what computes it is a
    finite number of rounds of :meth:`Interaction.advance`.  Their *tensor*
    is meaningful and is kept, because :math:`\\Phi_\\theta` is exactly the
    parallel application of every local interaction.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> f = InteractionMap("f", Dim(2), Dim(3), Dim(25))
    >>> g = InteractionMap("g", Dim(5), Dim(7), Dim(49))
    >>> f.boundary, f.width
    (Dim(2, 3), 5)
    >>> f >> g
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: interaction maps do not compose by \
substitution: wire them together and iterate, see Interaction.

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

    def __rshift__(self, other) -> InteractionMap:
        raise AxiomError(
            "interaction maps do not compose by substitution: wire them "
            "together and iterate, see Interaction.")


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
    >>> interaction_spec(Network("g", Dim(2), Dim(3))).params
    Dim(0)
    """
    module = getattr(network, "module", None)
    parameters = getattr(module, "parameters", None)
    params = Dim() if parameters is None else Dim(
        sum(parameter.numel() for parameter in parameters()))
    return InteractionMap(network.name, network.dom, network.cod, params,
                          tuple(laws))


# --- what a diagram compiles to --------------------------------------------

class Interaction:
    """
    A compiled diagram: the global interaction
    :math:`T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S_D \\to S_D`
    together with the port index a caller reads and writes it through.

    This is what :meth:`~discopy.neural.MapNN.compile` returns and what a
    :class:`~discopy.neural.Solver` executes.  A state is one flat tensor of
    shape ``(rows, total)``, the messages of every port in port order;
    :meth:`read` and :meth:`write` address it by ``(generator name, role)``
    rather than by offset, so no port arithmetic is ever written by hand.
    The repr is a human summary rather than an eval-able expression: a
    compiled object holds torch modules, which have no eval-able
    representation, so the transparency rule stops at :attr:`cmap`.

    Parameters:
        cmap : The closed :class:`~discopy.neural.CMap` to run.
        ports : The global port indices of each ``(name, role)`` family, in
                box order then position order.
        heads : The subset of :attr:`ports` a module *reads*: one port per
                traced pair, every port of an untraced family.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> kept = interpret(pair, {peer: Dim(3), state: Dim(5)}, {"cell": None})
    >>> kept.widths  # ports are stored clockwise, codomain last
    (5, 5, 3, 5, 5, 3)
    >>> kept.heads["cell", state], kept.ports["cell", state]
    ((1, 4), (1, 0, 4, 3))

    Sending a role to ``Dim(0)`` erases its ports and the wires on them,
    which is how one diagram serves two models:

    >>> interpret(pair, {peer: Dim(3), state: Dim(0)}, {"cell": None}).widths
    (3, 3)
    """
    def __init__(self, cmap: CMap, ports: Mapping = (), heads: Mapping = ()):
        if len(cmap.dom) or len(cmap.cod):
            raise ValueError(
                "a compiled interaction is closed; trace the boundary first")
        self.cmap = cmap
        self.ports = dict(ports)
        self.heads = dict(heads)
        self.widths = cmap.port_widths
        offsets, total = [], 0
        for port, width in enumerate(self.widths):
            offsets.append(total)
            total += width
            if self.widths[cmap.edges[port]] != width:
                raise ValueError("a wire changes width")
        self.offsets, self.total = offsets, total
        self._cache: dict = {}

    # --- what it denotes ---------------------------------------------------

    @property
    def local(self) -> tuple:
        """ The interaction :math:`\\Phi_f` of each box, in box order. """
        return tuple(interaction_spec(box) for box in self.cmap.boxes)

    @property
    def routing(self) -> tuple[int, ...]:
        """ The wiring :math:`\\sigma_D`, as an involution on ports. """
        return tuple(self.cmap.edges)

    @property
    def state(self) -> Dim:
        """ The state object :math:`S_D`, one summand per port. """
        return Dim(*self.widths)

    @property
    def n_wires(self) -> int:
        """ The number of wires, i.e. half the number of ports. """
        return self.cmap.n_ports // 2

    def is_involution(self) -> bool:
        """
        Whether :attr:`routing` is a fixpoint-free involution, i.e. whether
        every port is wired to exactly one other and never to itself.
        """
        return all(self.routing[other] == port and other != port
                   for port, other in enumerate(self.routing))

    def sites(self, key) -> int:
        """
        How many sites a family has, i.e. how many generators of that name
        carry that role.

        Parameters:
            key : A ``(generator name, role)`` pair.
        """
        return len(self.heads[key])

    def __repr__(self):
        return f"Interaction({len(self.cmap.boxes)} boxes, " \
               f"{self.n_wires} wires, state width {self.total})"

    # --- how it runs -------------------------------------------------------

    def compile(self, **kwargs) -> Interaction:
        """
        Compile the round step with ``torch.compile``; see
        :meth:`discopy.neural.CMap.compile`.  Message passing on these maps
        is launch-bound -- many small kernels per round -- so this is a
        several-fold wall-clock speedup on a GPU at identical numerics up
        to rounding error.
        """
        self.cmap.compile(**kwargs)
        return self

    def zeros(self, rows: int, dtype=None, device=None) -> "torch.Tensor":
        """ An all-zero state of ``rows`` rows. """
        import torch
        return torch.zeros(rows, self.total, dtype=dtype, device=device)

    def advance(self, state, rounds: int, inject: bool = False,
                per_round: bool = False):
        """
        :math:`T^{\\text{rounds}}(s)`: that many rounds of synchronous
        message passing from a flat state, back to a flat state.

        Parameters:
            state : The flat incoming messages, ``(rows, total)``.
            rounds : The number of rounds.
            inject : Whether to re-add ``state`` after routing every round,
                     i.e. whether the transition is
                     :math:`\\sigma(\\Phi(s)) + i` rather than
                     :math:`\\sigma(\\Phi(s))`.
            per_round : Whether to return the state after every round
                        rather than only the last.
        """
        return self.cmap(init=state, n_rounds=rounds, inject=inject,
                         return_rounds=per_round, return_flat=True)

    def residual(self, state, inject: bool = False) -> "torch.Tensor":
        """
        :math:`\\|T(s) - s\\|_\\infty` per row: how far a state is from
        being a fixed point of the transition.

        Nothing makes this go to zero.  It is the number a
        :class:`~discopy.neural.FixedPoint` solver stops on and the honest
        answer to "does this map converge", which is a property of the
        learned weights rather than of the category.

        Parameters:
            state : The flat incoming messages.
            inject : Whether the transition re-adds the initial messages.
        """
        return (self.advance(state, 1, inject) - state).abs().amax(dim=-1)

    def route(self, outgoing) -> "torch.Tensor":
        """
        The flat state that starts the next round, from the per-box
        outgoing messages of this one: one application of the ``edges``
        involution, as a single cached permutation.

        Parameters:
            outgoing : One tensor per box, in the logical port order of
                       that box.
        """
        import torch
        flat = torch.cat(list(outgoing), -1)
        key = ("route", flat.device)
        if key not in self._cache:
            position, cursor = [0] * len(self.widths), 0
            for index in range(len(self.cmap.boxes)):
                for port in box_ports(self.cmap, index):
                    position[port] = cursor
                    cursor += self.widths[port]
            self._cache[key] = torch.tensor([
                k for port, width in enumerate(self.widths)
                for k in range(position[self.cmap.edges[port]],
                               position[self.cmap.edges[port]] + width)],
                dtype=torch.long, device=flat.device)
        return flat[:, self._cache[key]]

    # --- how it is addressed -----------------------------------------------

    def index(self, ports: tuple[int, ...], device=None) -> "torch.Tensor":
        """
        The flat indices of a family of equally wide ports, cached.

        Parameters:
            ports : The global port indices.
            device : The device the index tensor lives on.
        """
        import torch
        key = (tuple(ports), device)
        if key not in self._cache:
            width = self.widths[ports[0]]
            if any(self.widths[port] != width for port in ports):
                raise ValueError(
                    "ports of different widths cannot be read as one block")
            self._cache[key] = torch.tensor(
                [k for port in ports
                 for k in range(self.offsets[port],
                                self.offsets[port] + width)],
                dtype=torch.long, device=device)
        return self._cache[key]

    def read(self, state, key, every: bool = False):
        """
        The messages of a family, of shape ``(rows, sites, width)``.

        Parameters:
            state : The flat incoming messages.
            key : A ``(generator name, role)`` pair, or a tuple of port
                  indices.
            every : Whether to read every port of the family rather than
                    one per traced pair.
        """
        ports = self._family(key, every)
        return state.index_select(1, self.index(ports, state.device)) \
            .reshape(len(state), len(ports), self.widths[ports[0]])

    def write(self, state, key, values):
        """
        A copy of ``state`` with ``values`` written on *every* port of a
        family, one value per site broadcast to each copy of its trace.

        Parameters:
            state : The flat incoming messages.
            key : A ``(generator name, role)`` pair, or a tuple of port
                  indices.
            values : A tensor of shape ``(rows, sites, width)``.
        """
        ports = self._family(key, every=True)
        copies = len(ports) // len(self._family(key, every=False))
        if copies > 1:
            values = values.repeat_interleave(copies, dim=1)
        return state.index_copy(
            1, self.index(ports, state.device),
            values.reshape(len(state), -1).to(state.dtype))

    def _family(self, key, every: bool) -> tuple[int, ...]:
        """ A ``(name, role)`` key or a bare tuple of ports, as ports. """
        if isinstance(key, tuple) and key and isinstance(key[0], int):
            return key
        return (self.ports if every else self.heads)[key]


def functor(source, ob: Mapping, ar: Mapping) -> Functor:
    """
    The neural functor :math:`F_\\theta` of an interpretation: each atomic
    role to the :class:`~discopy.neural.Dim` it carries, each generator to
    the :class:`~discopy.neural.Network` of the image type around the
    module of the same name.

    Parameters:
        source : The closed map in the source category to interpret.
        ob : The ``Dim`` each atomic role carries, ``Dim(0)`` to erase it.
        ar : The torch module filling each generator name.  One shared
             module means one shared box, hence one batched call per round
             for a whole family of sites.
    """
    category = type(source).category.ar
    types = Functor(ob_map=dict(ob), dom=category)
    networks = {
        box: Network(box.name, types(box.dom), types(box.cod),
                     module=ar[box.name])
        for box in dict.fromkeys(source.boxes)}
    return Functor(ob_map=dict(ob), ar_map=networks, dom=category)


def interpret(source, ob: Mapping, ar: Mapping) -> Interaction:
    """
    Compile a diagram into a global interaction, port by port: each
    generator becomes its image :class:`~discopy.neural.Network`, each wire
    a wire between the image ports.

    A role must go to an atomic ``Dim`` -- one abstract port becomes one
    concrete port -- or to ``Dim(0)``, in which case the port vanishes and
    the wire is erased with it.  A wire may only be erased whole: a role
    wired to a surviving role cannot be erased.

    Parameters:
        source : The closed map in the source category, whose atomic types
                 name the *role* a port plays rather than its width; a
                 diagram is read through
                 :meth:`~discopy.cmap.CMap.from_diagram`.
        ob : The ``Dim`` each atomic role carries.
        ar : The torch module filling each generator name.

    Example
    -------
    >>> from discopy.frobenius import Box, Diagram, Ty
    >>> from discopy.neural import Dim
    >>> x = Ty("x")
    >>> f, g = Box("f", Ty(), x @ x), Box("g", x @ x, Ty())
    >>> compiled = interpret(f >> g, {x: Dim(2)}, {"f": None, "g": None})
    >>> compiled.routing, compiled.state
    ((3, 2, 1, 0), Dim(2, 2, 2, 2))
    >>> interpret(f >> Diagram.swap(x, x) >> g,
    ...           {x: Dim(2)}, {"f": None, "g": None}).routing
    (2, 3, 0, 1)
    """
    if not hasattr(source, "edges"):
        source = source.to_map()
    if len(source.dom) or len(source.cod):
        raise ValueError("only a closed diagram compiles to an interaction")
    image = functor(source, ob, ar)
    boxes = tuple(image(box) for box in source.boxes)
    for box, network in zip(source.boxes, boxes):
        assert_isinstance(network, Network)
        if (network.dom, network.cod) != (image(box.dom), image(box.cod)):
            raise ValueError(f"the image of {box} has the wrong type")

    erased, position = {}, {}
    for index, box in enumerate(source.boxes):
        cursor = 0
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            width = image(role)
            if len(width) > 1:
                raise ValueError(f"{role} maps to the non-atomic {width}")
            erased[index, place] = not len(width)
            position[index, place] = cursor
            cursor += len(width)

    logical = {port: (index, place)
               for index in range(len(source.boxes))
               for place, port in enumerate(box_ports(source, index))}
    wires = []
    for port, other in enumerate(source.edges):
        if port > other:
            continue
        one, two = logical[port], logical[other]
        if erased[one] and erased[two]:
            continue
        if erased[one] or erased[two]:
            raise ValueError(
                f"the wire {one} -- {two} is only erased at one end")
        wires.append(((one[0], position[one]), (two[0], position[two])))
    cmap = CMap.from_wiring(boxes, wires)
    return Interaction(cmap, *_families(source, cmap, erased, position))


def _families(source, cmap: CMap, erased: Mapping, position: Mapping):
    """
    The global port indices of each ``(generator name, role)`` pair, in box
    order then position order.

    A port is a *head* -- one a module reads a value off, rather than the
    far end of its own loop -- unless it is wired to an earlier port of the
    same box.  That is exactly the second copy of a traced leg, read off
    the wiring rather than off a declaration.
    """
    ports: dict = {}
    heads: dict = {}
    for index, box in enumerate(source.boxes):
        abstract, concrete = box_ports(source, index), box_ports(cmap, index)
        place_of = {port: place for place, port in enumerate(abstract)}
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            if erased[index, place]:
                continue
            port = concrete[position[index, place]]
            ports.setdefault((box.name, role), []).append(port)
            if place_of.get(source.edges[abstract[place]], place) >= place:
                heads.setdefault((box.name, role), []).append(port)
    return ({key: tuple(value) for key, value in ports.items()},
            {key: tuple(value) for key, value in heads.items()})


def route(cmap: CMap, outgoing) -> list:
    """
    The readable spelling of :meth:`Interaction.route`: one application of
    the ``edges`` involution, box by box and port by port.

    Composing ``forward`` with ``route`` resumes message passing exactly,
    which is what a segmented solver relies on.

    Parameters:
        cmap : The closed map whose boxes emitted the messages.
        outgoing : One tensor per box, in the logical port order of that
                   box.
    """
    import torch
    widths = cmap.port_widths
    per_port: list = [None] * len(widths)
    for index, emitted in enumerate(outgoing):
        ports = box_ports(cmap, index)
        chunks = torch.split(emitted, [widths[port] for port in ports], -1)
        for port, chunk in zip(ports, chunks):
            per_port[port] = chunk
    return [per_port[cmap.edges[port]] for port in range(len(widths))]
