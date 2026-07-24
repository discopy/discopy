# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, with additive
dimensions as objects and concatenation as tensor.

A :class:`Network` with domain ``Dim(a_1, ..., a_m)``, codomain
``Dim(b_1, ..., b_n)`` and private memory ``Dim(c_1, ..., c_k)`` carries one
:class:`torch.nn.Module` from ``R ** w`` to ``R ** w`` for
``w = sum(a_i) + sum(b_i) + sum(c_i)``. It reads incoming messages on all its
public ports followed by its previous private memory, and emits outgoing
public messages followed by its next private memory.
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
requires it. This mirrors :func:`discopy.matrix.backend`, where ``numpy``,
``jax`` and ``tensorflow`` are each imported lazily inside a
:class:`~discopy.matrix.Backend` subclass rather than at module scope, so
importing e.g. :mod:`discopy.tensor` never forces an unused array library
onto the user. :class:`Execution` isolates the backend operations from the
geometry-of-interaction stages, while :meth:`CMap.as_network` delegates
framework-specific parameter management to a lazily imported module wrapper.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dim
    Diagram
    Network
    Cup
    Cap
    Swap
    Functor
    Hypergraph
    CMap
    Execution
    Backend
    PyTorch

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        backend
        get_backend

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

from contextlib import contextmanager
from functools import cached_property
from types import ModuleType
from typing import TYPE_CHECKING

from discopy import compact, hypergraph, monoidal
from discopy.cat import factory
from discopy.cmap import PortKind
from discopy.pivotal import Ty
from discopy.utils import assert_isinstance

if TYPE_CHECKING:
    import torch


class Backend:
    """
    A neural execution backend.

    Parameters:
        module : The backend module implementing tensor operations and
                 framework-specific wrapping.
    """
    def __init__(self, module: ModuleType):
        self.module = module

    def __getattr__(self, attr):
        return getattr(self.module, attr)


class PyTorch(Backend):
    """ The PyTorch neural execution backend, imported lazily. """
    def __init__(self):
        from discopy import neural_torch
        super().__init__(neural_torch)


BACKENDS = {
    'pytorch': PyTorch,
}


@contextmanager
def backend(name: str = None, _stack=['pytorch'], _cache=dict()):
    """
    Context manager for neural execution backends.

    Parameters:
        name : The backend name, ``"pytorch"`` by default.
    """
    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()


def get_backend() -> Backend:
    """ Get the current neural execution backend. """
    with backend() as result:
        return result


@factory
class Dim(monoidal.Dim, Ty):
    """
    A dimension is a tuple of positive integers seen as a self-dual type,
    with addition as tensor and the zero-dimensional space as unit.

    Example
    -------
    >>> assert Dim(0) == Dim() and Dim(0) @ Dim(2) @ Dim(3) == Dim(2, 3)
    >>> assert Dim(2, 3).l == Dim(2, 3).r == Dim(3, 2)
    """
    unit = 0
    l = r = property(lambda self: self.ar(*self.inside[::-1]))

    def unwind(self) -> Dim:
        """ Dimensions have no winding number to normalize. """
        return self


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


class Network(compact.Box, Diagram):
    """
    A network is a neural box together with a torch module computing it.

    The module maps ``R ** width`` to ``R ** width`` for ``width`` the sum
    of the domain, codomain and private memory dimensions. It reads one
    incoming message and emits one outgoing message on every public port,
    in the order given by the domain followed by the codomain, then reads
    the previous memory and emits the next memory. Reusing the same network
    instance, or the same module, as several boxes shares its weights but
    each box occurrence has its own memory.

    Cups, caps and swaps are networks with ``module`` left to ``None``,
    since they are pure rerouting.

    Parameters:
        name : The name of the network.
        dom : The domain of the network, i.e. its input.
        cod : The codomain of the network, i.e. its output.
        module : The torch module of the network.
        mem : The private memory dimension.

    Note
    ----
    Networks compare equal when they have the same name, shape, memory and
    module, where missing modules compare equal and given modules compare
    by identity. The dagger and rotation of a network reuse its module and
    preserve its private memory, with the public ports read in the new order.

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
    module, mem = None, Dim()

    def __init__(self, name: str, dom: Dim, cod: Dim,
                 module: "torch.nn.Module" = None, mem: Dim = Dim(),
                 data=None, **params):
        assert_isinstance(mem, Dim)
        self.mem = mem
        self.module = module if module is not None else data
        super().__init__(name, dom, cod, data=self.module, **params)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def dagger(self) -> Network:
        """ Reverse the public ports while preserving module and memory. """
        return type(self)(
            self.name, dom=self.cod, cod=self.dom, module=self.module,
            mem=self.mem, is_dagger=not self.is_dagger, z=self.z)

    def rotate(self, left=False) -> Network:
        """ Rotate the public ports while preserving module and memory. """
        del left
        return type(self)(
            self.name, dom=self.cod.r, cod=self.dom.r, module=self.module,
            mem=self.mem, is_dagger=self.is_dagger, z=(self.z + 1) % 2)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        result = super().__repr__()
        return result if not self.mem\
            else result[:-1] + f", mem={self.mem!r})"

    def setoid(self):
        """ Include the private memory dimension in equality and hashing. """
        return super().setoid() + (self.mem, )


class Cup(compact.Cup, Network):
    """
    A neural cup is a compact cup between self-dual dimensions, i.e. a
    network with no module since it is pure rerouting.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Cap(compact.Cap, Network):
    """
    A neural cap is a compact cap between self-dual dimensions, i.e. a
    network with no module since it is pure rerouting.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Swap(compact.Swap, Network):
    """
    A neural swap is a compact swap between dimensions, i.e. a network
    with no module since it is pure rerouting.

    Parameters:
        left (Dim) : The dimension on the top left and bottom right.
        right (Dim) : The dimension on the top right and bottom left.
    """


class Functor(compact.Functor):
    """
    A neural functor is a compact functor between neural diagrams.

    Parameters:
        ob (Mapping[Dim, Dim]) : Map from atomic :class:`Dim` to `cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


Hypergraph = hypergraph.Hypergraph[Diagram]


class CMap(compact.CMap):
    """
    A neural combinatorial map is a compact map with networks as boxes,
    which computes as a graph neural network.

    The :meth:`forward` pass does synchronous message passing: one message
    per port, travelling along the wires given by the ``edges`` involution.
    :meth:`as_network` wraps the map into a :class:`Network` with a fresh
    backend module inside, which owns parameter and training state.

    :attr:`ports` lists the diagram's input ports, then each box's domain
    ports followed by its codomain ports (reversed), then the diagram's
    output ports, see :attr:`discopy.cmap.CMap.ports`.

    Example
    -------
    >>> f = Network('f', Dim(2), Dim(3, 2))
    >>> fm = f.to_map()
    >>> fm.port_dims  # f's dom, then f's dom, f's cod (reversed), f's cod
    (2, 2, 2, 3, 3, 2)
    """
    category = Diagram

    @property
    def port_dims(self) -> tuple[int, ...]:
        """ The dimension carried by each port of the map. """
        return tuple(sum(port.obj.inside) for port in self.ports)

    @cached_property
    def modules(self) -> tuple["torch.nn.Module", ...]:
        """ The distinct modules of the networks inside the map. """
        modules, seen = [], set()
        for box in self.boxes:
            assert_isinstance(box, Network)
            if box.module is None:
                raise ValueError(f"{box!r} has no module.")
            if id(box.module) not in seen:
                seen.add(id(box.module))
                modules.append(box.module)
        return tuple(modules)

    def as_network(self, name: str = "network") -> Network:
        """
        Wrap the map back into a :class:`Network` with a fresh backend module
        inside. The wrapper registers the modules of the networks in the map,
        so that it can be trained or nested inside a larger model.

        Parameters:
            name : The name of the network.
        """
        return Network(
            name, self.dom, self.cod, module=get_backend().wrap(self))

    def forward(self, x: "torch.Tensor" = None, init=None,
                n_rounds: int = None, inject: bool = True,
                memory=None, return_memory: bool = False):
        """
        Apply the geometry-of-interaction :class:`Execution` of the map.

        Parameters:
            x : The input, of shape ``(batch_size, sum of domain widths)``.
            init : The initial incoming messages, given per port or as one
                   tensor of shape ``(batch_size, sum of port widths)``.
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-add ``init`` to the incoming messages at
                     every round rather than just the first.
            memory : Initial private memory, given per box occurrence or as
                     one tensor of their concatenated memory dimensions.
            return_memory : Whether to return the final per-box memories
                            together with the usual result.

        Returns:
            The final messages at the boundary output ports, concatenated,
            or the tuple of final per-box outgoing messages in logical port
            order when the map is closed. If ``return_memory`` is true, this
            result is paired with the tuple of final per-box memories.
        """
        return Execution(self, x, init, memory).forward(
            n_rounds, inject, return_memory)

    __call__ = forward


class Execution:
    """
    The geometry-of-interaction execution of a neural combinatorial map.

    Let ``edge`` be the fixpoint-free involution on ports and ``activate``
    apply every network independently. One synchronous round first activates
    the boxes, then routes their outgoing messages with
    ``incoming[i] = outgoing[edge[i]]``. Boundary inputs are emitted before
    every round, while ``init`` is optionally injected after every routing
    step as well as before the first round.

    Parameters:
        inside : The combinatorial map to execute.
        x : The boundary input.
        init : The initial incoming messages.
        memory : The initial private memory, one tensor per box occurrence.
        backend : The execution backend, the current backend by default.
    """
    def __init__(
            self, inside: CMap, x=None, init=None, memory=None,
            backend: Backend = None):
        assert_isinstance(inside, CMap)
        self.inside, self.x, self.init = inside, x, init
        self.memory = memory
        self.backend = get_backend() if backend is None else backend
        self.modules = inside.modules
        self.batch_size, self.prototype = 1, None
        self.initial = self.boundary = self.incoming = ()
        self.outgoing = self.box_outputs = ()
        self.memories = ()

    @cached_property
    def input_ports(self) -> tuple[int, ...]:
        """ The indices of boundary input ports. """
        return tuple(
            i for i, port in enumerate(self.inside.ports)
            if port.kind == PortKind.INPUT)

    @cached_property
    def output_ports(self) -> tuple[int, ...]:
        """ The indices of boundary output ports. """
        return tuple(
            i for i, port in enumerate(self.inside.ports)
            if port.kind == PortKind.OUTPUT)

    @cached_property
    def box_ports(self) -> tuple[tuple[int, ...], ...]:
        """ The ports of each box in domain-then-codomain order. """
        result = []
        for box, indices in zip(
                self.inside.boxes, self.inside._box_port_indices):
            arity = len(box.dom)
            result.append(
                indices[:arity] + tuple(reversed(indices[arity:])))
        return tuple(result)

    @cached_property
    def memory_widths(self) -> tuple[int, ...]:
        """ The private memory width of each box occurrence. """
        return tuple(sum(box.mem.inside) for box in self.inside.boxes)

    def zeros(self, width: int):
        """ A zero message with the execution's batch size and prototype. """
        return self.backend.zeros(
            self.batch_size, width, like=self.prototype)

    def validate(self, value, width: int, label: str):
        """ Validate the rank, batch size and width of a message tensor. """
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"{label} must have shape (batch_size, {width}).")
        if shape[0] != self.batch_size or shape[1] != width:
            raise ValueError(
                f"{label} has shape {tuple(shape)}, expected "
                f"({self.batch_size}, {width}).")
        return value

    @staticmethod
    def _values(given):
        """ Yield non-null tensors from a tensor or per-item sequence. """
        if isinstance(given, (list, tuple)):
            return (value for value in given if value is not None)
        return iter(()) if given is None else iter((given, ))

    def _initialize_messages(self, given, widths, label):
        """ Normalize a tensor or nullable per-item sequence to messages. """
        if given is None:
            values = len(widths) * (None, )
        elif isinstance(given, (list, tuple)):
            if len(given) != len(widths):
                raise ValueError(
                    f"{label} must contain {len(widths)} messages, "
                    f"got {len(given)}.")
            values = given
        else:
            self.validate(given, sum(widths), label)
            values = self.backend.split(given, widths) if widths else ()
        return tuple(
            self.zeros(width) if value is None
            else self.validate(value, width, f"{label}[{i}]")
            for i, (value, width) in enumerate(zip(values, widths)))

    def initialize(self) -> tuple:
        """ Initialize public messages and per-box private memories. """
        widths = self.inside.port_dims
        given = (
            self._values(self.x), self._values(self.init),
            self._values(self.memory))
        reference = next((
            value for values in given for value in values), None)
        if reference is not None:
            shape = getattr(reference, "shape", None)
            if shape is None or len(shape) != 2:
                raise ValueError(
                    "Messages must have shape (batch_size, width).")
        self.batch_size = 1 if reference is None else shape[0]
        self.prototype = reference if reference is not None\
            else self.backend.prototype(self.modules)

        boundary = [self.zeros(width) for width in widths]
        if self.x is not None:
            self.validate(
                self.x, sum(widths[i] for i in self.input_ports), "x")
            slices = self.backend.split(
                self.x, tuple(widths[i] for i in self.input_ports))
            for i, message in zip(self.input_ports, slices):
                boundary[i] = message

        initial = self._initialize_messages(self.init, widths, "init")
        memories = self._initialize_messages(
            self.memory, self.memory_widths, "memory")

        incoming = list(initial)
        for i in self.input_ports:
            edge = self.inside.edges[i]
            incoming[edge] = incoming[edge] + boundary[i]

        self.initial = tuple(initial)
        self.boundary = tuple(boundary)
        self.incoming = tuple(incoming)
        self.outgoing = ()
        self.box_outputs = len(self.inside.boxes) * (None, )
        self.memories = memories
        return self.incoming

    def activate(self) -> tuple:
        """ Apply each network to its public messages and private memory. """
        widths = self.inside.port_dims
        outgoing = [self.zeros(width) for width in widths]
        for i in self.input_ports:
            outgoing[i] = self.boundary[i]

        box_outputs, memories = [], []
        for box_index, (box, ports) in enumerate(zip(
                self.inside.boxes, self.box_ports)):
            public_widths = tuple(widths[i] for i in ports)
            memory_width = self.memory_widths[box_index]
            values = tuple(self.incoming[i] for i in ports)\
                + (self.memories[box_index], )
            output = box.module(self.backend.concatenate(values))
            self.validate(
                output, sum(public_widths) + memory_width,
                f"output of box {box_index}")
            chunks = self.backend.split(
                output, public_widths + (memory_width, ))
            public, next_memory = chunks[:-1], chunks[-1]
            box_outputs.append(
                self.backend.concatenate(public) if public else self.zeros(0))
            memories.append(next_memory)
            for i, chunk in zip(ports, public):
                outgoing[i] = chunk

        self.outgoing = tuple(outgoing)
        self.box_outputs = tuple(box_outputs)
        self.memories = tuple(memories)
        return self.outgoing

    def route(self) -> tuple:
        """ Route outgoing messages along the edge involution. """
        self.incoming = tuple(
            self.outgoing[self.inside.edges[i]]
            for i in range(self.inside.n_ports))
        return self.incoming

    def inject(self) -> tuple:
        """ Add the initial messages to the current incoming messages. """
        self.incoming = tuple(
            message + initial
            for message, initial in zip(self.incoming, self.initial))
        return self.incoming

    def readout(self):
        """ Read boundary outputs, or final box outputs for a closed map. """
        if len(self.inside.dom) or len(self.inside.cod):
            return self.backend.concatenate(tuple(
                self.incoming[i] for i in self.output_ports))\
                if self.output_ports else self.zeros(0)
        return self.box_outputs

    def forward(
            self, n_rounds: int = None, inject: bool = True,
            return_memory: bool = False):
        """ Execute synchronous activation and routing rounds. """
        self.initialize()
        n_rounds = len(self.inside.boxes) if n_rounds is None else n_rounds
        for _ in range(n_rounds):
            self.activate()
            self.route()
            if inject and self.init is not None:
                self.inject()
        result = self.readout()
        return (result, self.memories) if return_memory else result

    __call__ = forward


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap


class Equation(compact.Equation):
    """ An equation between neural diagrams, compared up to maps. """
    up_to = staticmethod(Diagram.to_map)
