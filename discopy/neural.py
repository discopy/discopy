# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, with additive
dimensions as objects and concatenation as tensor.

A :class:`Network` with domain ``Dim(a_1, ..., a_m)`` and codomain
``Dim(b_1, ..., b_n)`` carries one :class:`torch.nn.Module` from ``R ** w``
to ``R ** w`` for ``w = a_1 + ... + a_m + b_1 + ... + b_n``, reading incoming
messages on all its ports and emitting outgoing messages on all its ports.
Networks compose with the cartesian product of vector spaces, so the tensor
of dimensions is their sum with the zero-dimensional space ``Dim(0)`` as
unit; dimensions are self-dual so that cups, caps and swaps are pure
rerouting.

The combinatorial maps of this category are graph neural networks: the
:meth:`CMap.forward` pass does synchronous message passing along the wires,
which implements the execution formula of the geometry of interaction, see
:cite:t:`Abramsky96` and :mod:`discopy.interaction` for the Int-construction
of Joyal, Street & Verity :cite:p:`JoyalEtAl96`.

Note that ``import discopy.neural`` does not import ``torch``: networks can
be built, composed and rewired without it, only evaluating their modules
requires it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dim
    Seq
    Diagram
    Box
    Network
    Cup
    Cap
    Swap
    Functor
    Hypergraph
    CMap

Example
-------

Message passing on the combinatorial map of a diagram computes its image
under the execution formula, e.g. rerouting for a snake:

>>> import torch
>>> snake = Id(Dim(2)).transpose().to_map()
>>> snake.boxes
()
>>> x = torch.tensor([[0.1, 0.2]])
>>> assert (snake(x) == x).all()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import cached_property

from discopy import compact, monoidal
from discopy.cat import factory
from discopy.cmap import PortKind
from discopy.pivotal import Ty
from discopy.utils import assert_isinstance

if TYPE_CHECKING:
    import torch


class Seq(int):
    """
    The free monoid on vectors of a fixed dimension: an atomic dimension
    whose wires carry variable-length sequences of vectors rather than
    single vectors, see :meth:`CMap.pass_messages`.

    Example
    -------
    >>> assert Dim(Seq(2), 3).r == Dim(3, Seq(2))
    >>> assert Seq(2) != 2 and 2 != Seq(2) and Seq(2) == Seq(2)
    """
    def __eq__(self, other):
        return isinstance(other, Seq) and int(self) == int(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Seq, int(self)))

    def __repr__(self):
        return f"Seq({int(self)})"

    __str__ = __repr__


@factory
class Dim(monoidal.Dim, Ty):
    """
    A dimension is a tuple of positive integers seen as a self-dual type,
    with addition as tensor and the zero-dimensional space as unit; the
    atoms can also be free monoids on a fixed dimension, see :class:`Seq`.

    Example
    -------
    >>> assert Dim(0) == Dim() and Dim(0) @ Dim(2) @ Dim(3) == Dim(2, 3)
    >>> assert Dim(2, 3).l == Dim(2, 3).r == Dim(3, 2)
    """
    unit = 0
    l = r = property(lambda self: self.factory(*self.inside[::-1]))
    z = property(lambda self: 0)


@factory
class Diagram(compact.Diagram):
    """
    A neural diagram is a compact diagram with dimensions as objects.

    Parameters:
        inside (Layer) : The layers of the diagram.
        dom (Dim) : The domain of the diagram, i.e. its input.
        cod (Dim) : The codomain of the diagram, i.e. its output.
    """
    ob = Dim


class Box(compact.Box, Diagram):
    """
    A neural box is a compact box between dimensions.

    Parameters:
        name (str) : The name of the box.
        dom (Dim) : The domain of the box, i.e. its input.
        cod (Dim) : The codomain of the box, i.e. its output.
    """


class Cup(compact.Cup, Box):
    """
    A neural cup is a compact cup between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Cap(compact.Cap, Box):
    """
    A neural cap is a compact cap between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Swap(compact.Swap, Box):
    """
    A neural swap is a compact swap between dimensions.

    Parameters:
        left (Dim) : The dimension on the top left and bottom right.
        right (Dim) : The dimension on the top right and bottom left.
    """


class Network(Box):
    """
    A network is a neural box together with a torch module computing it.

    The module maps ``R ** width`` to ``R ** width`` for ``width`` the sum
    of the domain and codomain dimensions, reading one incoming message and
    emitting one outgoing message on every port, in the order given by the
    domain followed by the codomain. Reusing the same network instance, or
    the same module, as several boxes shares its weights.

    Parameters:
        name : The name of the network.
        dom : The domain of the network, i.e. its input.
        cod : The codomain of the network, i.e. its output.
        module : The torch module of the network.

    Note
    ----
    Networks compare equal when they have the same name, shape and module,
    where missing modules compare equal and given modules compare by
    identity. The dagger and rotation of a network reuse its module, with
    the weights read in the new port order.

    Example
    -------
    >>> import torch
    >>> f = Network('f', Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    >>> g = Network('g', Dim(3), Dim(2), module=torch.nn.Linear(5, 5))
    >>> (f >> g).dom == (f >> g).cod == Dim(2)
    True
    >>> f.module(torch.ones(1, 5)).shape
    torch.Size([1, 5])
    >>> assert f[::-1].module is f.module
    """
    def __init__(self, name: str, dom: Dim, cod: Dim,
                 module: "torch.nn.Module" = None, data=None, **params):
        self.module = module if module is not None else data
        super().__init__(name, dom, cod, data=self.module, **params)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __repr__(self):
        return f"neural.Network({self.name!r}, " \
               f"dom={self.dom!r}, cod={self.cod!r})"


class Functor(compact.Functor):
    """
    A neural functor is a compact functor between neural diagrams.

    Parameters:
        ob (Mapping[Dim, Dim]) : Map from atomic :class:`Dim` to `cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


class Hypergraph(compact.Hypergraph):
    """ A neural hypergraph is a compact hypergraph between dimensions. """
    functor = Functor


class CMap(compact.CMap):
    """
    A neural combinatorial map is a compact map with networks as boxes,
    which computes as a graph neural network.

    The :meth:`forward` pass does synchronous message passing: one message
    per port, travelling along the wires given by the ``edges`` involution.
    An optimizer only needs :meth:`parameters` and a training loop only
    needs to call the map, so it can be trained like any torch module;
    :meth:`as_network` wraps it back into a :class:`Network` with a fresh
    module inside, for use inside a larger model.

    Example
    -------
    >>> f = Network('f', Dim(2), Dim(3, 2))
    >>> fm = f.to_map()
    >>> fm.box_ports(0)
    (1, 3, 2)
    >>> fm.port_widths
    (2, 2, 2, 3, 3, 2)
    """
    category = Diagram
    functor = Functor

    def box_ports(self, index: int) -> tuple[int, ...]:
        """
        The global port indices of a box in logical order, i.e. its domain
        ports followed by its codomain ports, undoing the clockwise order
        which stores the codomain ports reversed.

        Parameters:
            index : The index of the box.
        """
        return self._logical_box_ports[index]

    @cached_property
    def _logical_box_ports(self) -> tuple[tuple[int, ...], ...]:
        """ The ports of each box in logical order, see :meth:`box_ports`. """
        return tuple(
            ports[:len(box.dom)] + tuple(reversed(ports[len(box.dom):]))
            for ports, box in zip(self._box_port_indices, self.boxes))

    @cached_property
    def port_widths(self) -> tuple[int, ...]:
        """ The dimension carried by each port of the map. """
        return tuple(
            sum(getattr(port.obj, "inside", (port.obj, )))
            for port in self.ports)

    @cached_property
    def module_list(self) -> "torch.nn.ModuleList":
        """ The distinct torch modules of the networks inside the map. """
        import torch
        modules, seen = [], set()
        for box in self.boxes:
            assert_isinstance(box, Network)
            if box.module is None:
                raise ValueError(f"{box!r} has no module.")
            if id(box.module) not in seen:
                seen.add(id(box.module))
                modules.append(box.module)
        return torch.nn.ModuleList(modules)

    def parameters(self, recurse: bool = True):
        """ The parameters of the networks inside the map. """
        return self.module_list.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """ The named parameters of the networks inside the map. """
        return self.module_list.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        """ The state dict of the networks inside the map. """
        return self.module_list.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        """ Load a state dict into the networks inside the map. """
        return self.module_list.load_state_dict(state_dict, **kwargs)

    def train(self, mode: bool = True) -> CMap:
        """ Set the networks inside the map to training mode. """
        self.module_list.train(mode)
        return self

    def eval(self) -> CMap:
        """ Set the networks inside the map to evaluation mode. """
        return self.train(False)

    def to(self, *args, **kwargs) -> CMap:
        """ Move the networks inside the map to a device or dtype. """
        self.module_list.to(*args, **kwargs)
        return self

    def as_network(self, name: str = "network") -> Network:
        """
        Wrap the map back into a :class:`Network` with a fresh torch module
        inside, whose forward pass is the message passing of the map. The
        module registers the modules of the networks inside the map, so
        that the result can be used inside a larger model.

        Parameters:
            name : The name of the network.
        """
        import torch
        cmap = self

        class CMapModule(torch.nn.Module):
            """ A combinatorial map wrapped as a torch module. """
            def __init__(self):
                super().__init__()
                self.networks = cmap.module_list

            def forward(self, *args, **kwargs):
                """ Message passing over the wrapped map. """
                return cmap.forward(*args, **kwargs)

        return Network(name, self.dom, self.cod, module=CMapModule())

    def forward(self, x: "torch.Tensor" = None, init=None,
                n_rounds: int = None, inject: bool = True):
        """
        Synchronous message passing along the wires of the map, i.e. the
        execution formula of the geometry of interaction.

        Every port carries one message of shape ``(batch_size, width)``.
        The boundary input ports deliver the corresponding slice of ``x``
        along their wires before the first round and after every round.
        Each round, every network reads the incoming messages on its ports
        and emits outgoing ones, then all messages travel along the wires.

        Parameters:
            x : The input, of shape ``(batch_size, sum of domain widths)``.
            init : The initial incoming messages, given per port or as one
                   tensor of shape ``(batch_size, sum of port widths)``.
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-add ``init`` to the incoming messages at
                     every round rather than just the first.

        Returns:
            The final messages at the boundary output ports, concatenated,
            or the tuple of final per-box outgoing messages in logical port
            order when the map is closed.
        """
        import torch
        ports = self.ports
        widths = self.port_widths
        n_rounds = len(self.boxes) if n_rounds is None else n_rounds

        given = [x] + (list(init)
                       if isinstance(init, (list, tuple)) else [init])
        ref = next((t for t in given if t is not None), None)
        batch_size = 1 if ref is None else ref.shape[0]
        proto = ref if ref is not None else next(
            iter(self.parameters()), None) if self.boxes else None

        def zeros(width):
            kwargs = {} if proto is None else {
                "dtype": proto.dtype, "device": proto.device}
            return torch.zeros(batch_size, width, **kwargs)

        input_ports = [i for i, port in enumerate(ports)
                       if port.kind == PortKind.INPUT]
        x_slices = dict(zip(input_ports, torch.split(
            x, [widths[i] for i in input_ports], dim=-1))) if x is not None\
            else {i: zeros(widths[i]) for i in input_ports}
        if init is not None and isinstance(init, torch.Tensor):
            init = torch.split(init, list(widths), dim=-1)

        incoming = [zeros(width) for width in widths]\
            if init is None else list(init)
        for i in input_ports:
            incoming[self.edges[i]] = incoming[self.edges[i]] + x_slices[i]
        box_outputs = [None] * len(self.boxes)

        for _ in range(n_rounds):
            outgoing = [None] * len(ports)
            for i in input_ports:
                outgoing[i] = x_slices[i]
            for box_index, box in enumerate(self.boxes):
                assert_isinstance(box, Network)
                box_ports = self.box_ports(box_index)
                output = box.module(torch.cat(
                    [incoming[i] for i in box_ports], dim=-1))
                box_outputs[box_index] = output
                for i, chunk in zip(box_ports, torch.split(
                        output, [widths[i] for i in box_ports], dim=-1)):
                    outgoing[i] = chunk
            incoming = [
                outgoing[self.edges[i]]
                if outgoing[self.edges[i]] is not None else zeros(width)
                for i, width in enumerate(widths)]
            if inject and init is not None:
                incoming = [message + initial
                            for message, initial in zip(incoming, init)]

        if len(self.dom) or len(self.cod):
            output_ports = [i for i, port in enumerate(ports)
                            if port.kind == PortKind.OUTPUT]
            return torch.cat([incoming[i] for i in output_ports], dim=-1)\
                if output_ports else zeros(0)
        return tuple(box_outputs)

    __call__ = forward

    def pass_messages(self, init: dict = None, n_rounds: int = None,
                      inject: bool = True) -> list:
        """
        Synchronous message passing with one opaque message per port.

        Each round, the module of every network is called with one incoming
        message per port in logical order, ``None`` for the empty message,
        and must return one outgoing message per port; then the messages
        travel along the wires. Messages can be of any type, e.g. pairs of
        token sequences for the geometry of interaction, so the messages in
        ``init`` overwrite the incoming ones rather than being added.

        Parameters:
            init : A mapping from port indices to the messages delivered as
                   incoming at these ports.
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-deliver ``init`` after every round rather
                     than just before the first.

        Returns:
            The final incoming messages, one per port.

        Example
        -------
        A network whose module reflects its domain port onto its codomain
        port, receiving a token and bouncing it back out to the root:

        >>> bounce = Network('bounce', Dim(Seq(1)), Dim(Seq(1)),
        ...                  module=lambda left, right: (right, left))
        >>> bm = bounce.to_map()
        >>> bm.pass_messages(init={bm.edges[0]: "token"}, n_rounds=1)[-1]
        'token'
        """
        n_rounds = len(self.boxes) if n_rounds is None else n_rounds
        n_ports, edges = self.n_ports, [self.edges[i]
                                        for i in range(self.n_ports)]
        for box in self.boxes:
            assert_isinstance(box, Network)
        modules = [(box.module, self.box_ports(box_index))
                   for box_index, box in enumerate(self.boxes)]
        incoming = [None] * n_ports
        for port, message in (init or {}).items():
            incoming[port] = message
        for _ in range(n_rounds):
            outgoing = [None] * n_ports
            for module, box_ports in modules:
                outputs = module(*[incoming[i] for i in box_ports])
                for i, message in zip(box_ports, outputs):
                    outgoing[i] = message
            incoming = [outgoing[i] for i in edges]
            if inject and init:
                for port, message in init.items():
                    incoming[port] = message
        return incoming


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
