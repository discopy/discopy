# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, with additive
dimensions as objects and concatenation as tensor.

A :class:`Network` with domain ``Dim(a_1, ..., a_m)``, codomain
``Dim(b_1, ..., b_n)`` and private memory ``Dim(c_1, ..., c_k)`` carries one
backend-owned callable from ``R ** w`` to ``R ** w`` for
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

Note that ``import discopy.neural`` imports neither ``torch`` nor ``jax``:
networks can be built, composed and rewired without either library, and only
evaluation requires a concrete :class:`Backend`. The abstract backend owns
the tensor primitives and module protocol, while :class:`ExecutionPlan` keeps
graph geometry independent from runtime parameters. :class:`Execution`
interprets that plan and :meth:`CMap.as_network` delegates framework-specific
parameter management to a lazily imported module wrapper. :class:`PyTorch`
is the default concrete backend and :class:`JAX` is available with
``backend="jax"``.

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
    ExecutionPlan
    Execution
    Backend
    PyTorch
    JAX

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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property

from discopy import compact, hypergraph, monoidal
from discopy.cat import factory
from discopy.cmap import PortKind
from discopy.pivotal import Ty
from discopy.utils import (
    assert_isinstance, factory_name, from_tree as decode)


class Backend(ABC):
    """
    An abstract neural execution backend.

    A backend supplies array operations, activation of backend-owned callable
    modules, parameter prototypes for allocating messages, and a trainable
    wrapper for compiled maps.
    """

    @abstractmethod
    def zeros(self, batch_size: int, width: int, like=None):
        """ Return a batch of zero messages. """

    @abstractmethod
    def split(self, value, widths: tuple[int, ...]) -> tuple:
        """ Split a batch into messages of the given widths. """

    @abstractmethod
    def concatenate(self, values: tuple):
        """ Concatenate messages along their final dimension. """

    @abstractmethod
    def activate(self, module, value):
        """ Apply a backend-owned module to an all-port message. """

    @abstractmethod
    def prototype(self, modules: tuple):
        """ Find a value whose dtype and device zero messages should use. """

    @abstractmethod
    def wrap(self, inside: CMap):
        """ Wrap a combinatorial map as a backend-owned module. """

    @abstractmethod
    def zeros_module(self):
        """ Return a parameter-free all-port zero module. """


class PyTorch(Backend):
    """ The PyTorch neural execution backend, imported lazily. """

    def zeros(self, batch_size: int, width: int, like=None):
        from discopy import neural_torch
        return neural_torch.zeros(batch_size, width, like=like)

    def split(self, value, widths: tuple[int, ...]) -> tuple:
        from discopy import neural_torch
        return neural_torch.split(value, widths)

    def concatenate(self, values: tuple):
        from discopy import neural_torch
        return neural_torch.concatenate(values)

    def activate(self, module, value):
        from discopy import neural_torch
        return neural_torch.activate(module, value)

    def prototype(self, modules: tuple):
        from discopy import neural_torch
        return neural_torch.prototype(modules)

    def wrap(self, inside: CMap):
        from discopy import neural_torch
        return neural_torch.wrap(inside, backend=self)

    def zeros_module(self):
        from discopy import neural_torch
        return neural_torch.zeros_module()


class JAX(Backend):
    """
    The JAX neural execution backend, imported lazily.

    A module is a callable PyTree from one batched all-port array to an array
    of the same width. ``jax.tree_util.Partial`` can bind parameter arrays
    to a function while leaving them visible to JAX transformations.
    :meth:`CMap.as_network` returns a callable PyTree whose immutable
    :class:`ExecutionPlan` is static metadata and whose distinct modules are
    dynamic children. Pass that wrapper as an argument to transformations,
    for example ``jax.jit(lambda model, x: model(x))(model, x)``. Execution
    controls such as ``n_rounds``, ``inject``, ``causal`` and
    ``return_memory`` remain static Python values under JIT. Passing the
    wrapper itself directly to ``jax.jit`` closes over its current parameters;
    pass it as an argument when parameters should remain dynamic.
    """

    def zeros(self, batch_size: int, width: int, like=None):
        from discopy import neural_jax
        return neural_jax.zeros(batch_size, width, like=like)

    def split(self, value, widths: tuple[int, ...]) -> tuple:
        from discopy import neural_jax
        return neural_jax.split(value, widths)

    def concatenate(self, values: tuple):
        from discopy import neural_jax
        return neural_jax.concatenate(values)

    def activate(self, module, value):
        from discopy import neural_jax
        return neural_jax.activate(module, value)

    def prototype(self, modules: tuple):
        from discopy import neural_jax
        return neural_jax.prototype(modules)

    def wrap(self, inside: CMap):
        from discopy import neural_jax
        return neural_jax.wrap(inside, backend=self)

    def zeros_module(self):
        from discopy import neural_jax
        return neural_jax.zeros_module()


BACKENDS = {
    'jax': JAX,
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


def get_backend(name: str | Backend = None) -> Backend:
    """
    Get a neural execution backend by name, or return a given backend.

    Parameters:
        name : The backend name or instance, the current backend by default.
    """
    if isinstance(name, Backend):
        return name
    with backend(name) as result:
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

    def to_tree(self) -> dict:
        """ Serialize an additive dimension. """
        return {
            'factory': factory_name(type(self)),
            'inside': list(self.inside)}

    @classmethod
    def from_tree(cls, tree: dict) -> Dim:
        """ Deserialize an additive dimension. """
        return cls(*tree['inside'])


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
    A network is a neural box together with a backend module computing it.

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
        module : The backend-owned module of the network.
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
                 module: object = None, mem: Dim = Dim(),
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

    def setoid(self):
        """ Compare given modules by identity and include private memory. """
        result = super().setoid()
        module = None if self.module is None else id(self.module)
        return result[:5] + (module, ) + result[6:] + (self.mem, )

    def to_tree(self) -> dict:
        """ Serialize the network shape, including its private memory. """
        tree = super().to_tree()
        tree['mem'] = self.mem.to_tree()
        if self.z:
            tree['z'] = self.z
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Network:
        """ Deserialize a network, accepting trees without private memory. """
        dom, cod = map(decode, (tree['dom'], tree['cod']))
        mem = decode(tree['mem']) if 'mem' in tree else Dim()
        return cls(
            tree['name'], dom, cod, data=tree.get('data'), mem=mem,
            is_dagger='is_dagger' in tree, z=tree.get('z', 0))


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


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Immutable backend-neutral data needed to execute a neural map.

    A plan contains only graph topology, dimensions and indices into the
    separate tuple of runtime modules. Thus backend wrappers can expose those
    modules as trainable state without retaining them in compiled graph data.
    """
    port_dims: tuple[int, ...]
    port_kinds: tuple[PortKind, ...]
    edges: tuple[int, ...]
    box_ports: tuple[tuple[int, ...], ...]
    box_dom_arities: tuple[int, ...]
    memory_widths: tuple[int, ...]
    module_indices: tuple[int, ...]
    input_ports: tuple[int, ...]
    output_ports: tuple[int, ...]
    has_boundary: bool
    n_modules: int

    @property
    def n_ports(self) -> int:
        """ The number of ports in the plan. """
        return len(self.port_dims)

    @property
    def n_boxes(self) -> int:
        """ The number of box occurrences in the plan. """
        return len(self.box_ports)


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
    def modules(self) -> tuple:
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

    @cached_property
    def module_indices(self) -> tuple[int, ...]:
        """ The unique-module index used by each box occurrence. """
        indices = {id(module): i for i, module in enumerate(self.modules)}
        return tuple(indices[id(box.module)] for box in self.boxes)

    @cached_property
    def execution_plan(self) -> ExecutionPlan:
        """ Compile the map into immutable backend-neutral execution data. """
        modules = self.modules
        box_ports = []
        for box, indices in zip(self.boxes, self._box_port_indices):
            arity = len(box.dom)
            box_ports.append(
                indices[:arity] + tuple(reversed(indices[arity:])))
        return ExecutionPlan(
            port_dims=self.port_dims,
            port_kinds=tuple(port.kind for port in self.ports),
            edges=tuple(self.edges),
            box_ports=tuple(box_ports),
            box_dom_arities=tuple(len(box.dom) for box in self.boxes),
            memory_widths=tuple(
                sum(box.mem.inside) for box in self.boxes),
            module_indices=self.module_indices,
            input_ports=tuple(
                i for i, port in enumerate(self.ports)
                if port.kind == PortKind.INPUT),
            output_ports=tuple(
                i for i, port in enumerate(self.ports)
                if port.kind == PortKind.OUTPUT),
            has_boundary=bool(len(self.dom) or len(self.cod)),
            n_modules=len(modules))

    def as_network(
            self, name: str = "network",
            backend: str | Backend = None) -> Network:
        """
        Wrap the map back into a :class:`Network` with a fresh backend module
        inside. The wrapper registers the modules of the networks in the map,
        so that it can be trained or nested inside a larger model.

        Parameters:
            name : The name of the network.
            backend : The backend name or instance, the current backend by
                      default.
        """
        backend = get_backend(backend)
        memory_width = sum(
            sum(box.mem.inside) for box in self.boxes)
        return Network(
            name, self.dom, self.cod, module=backend.wrap(self),
            mem=Dim(memory_width))

    def forward(self, x=None, init=None,
                n_rounds: int = None, inject: bool = True,
                memory=None, return_memory: bool = False,
                causal: bool = False, backend: str | Backend = None,
                modules=None):
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
            causal : Whether to activate every box once in topological order.
                     This is only valid for feed-forward maps and cannot be
                     combined with ``n_rounds``.
            backend : The backend name or instance, the current backend by
                      default.
            modules : The backend-owned modules, in :attr:`module_indices`
                      order. The modules in the boxes are used by default.

        Returns:
            The final messages at the boundary output ports, concatenated,
            or the tuple of final per-box outgoing messages in logical port
            order when the map is closed. If ``return_memory`` is true, this
            result is paired with the tuple of final per-box memories.
        """
        execution = Execution(
            self, x, init, memory, backend=backend, modules=modules)
        if causal:
            if n_rounds is not None:
                raise ValueError(
                    "A causal schedule cannot be combined with n_rounds.")
            return execution.forward_causal(inject, return_memory)
        return execution.forward(n_rounds, inject, return_memory)

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
        inside : The combinatorial map or compiled execution plan to execute.
        x : The boundary input.
        init : The initial incoming messages.
        memory : The initial private memory, one tensor per box occurrence.
        backend : The execution backend, the current backend by default.
        modules : The backend-owned modules, in the map's unique-module order.
    """
    def __init__(
            self, inside: CMap | ExecutionPlan, x=None, init=None, memory=None,
            backend: str | Backend = None, modules=None):
        if isinstance(inside, CMap):
            self.inside, self.plan = inside, inside.execution_plan
            expected_modules = inside.modules
        else:
            assert_isinstance(inside, ExecutionPlan)
            self.inside, self.plan = None, inside
            expected_modules = None
        self.x, self.init = x, init
        self.memory = memory
        self.backend = get_backend(backend)
        if modules is None and expected_modules is None:
            raise ValueError(
                "Runtime modules are required with an execution plan.")
        self.modules = expected_modules\
            if modules is None else tuple(modules)
        if len(self.modules) != self.plan.n_modules:
            raise ValueError(
                f"Expected {self.plan.n_modules} modules, "
                f"got {len(self.modules)}.")
        self.batch_size, self.prototype = 1, None
        self.initial = self.boundary = self.incoming = ()
        self.outgoing = self.box_outputs = ()
        self.memories = ()

    @cached_property
    def input_ports(self) -> tuple[int, ...]:
        """ The indices of boundary input ports. """
        return self.plan.input_ports

    @cached_property
    def output_ports(self) -> tuple[int, ...]:
        """ The indices of boundary output ports. """
        return self.plan.output_ports

    @cached_property
    def box_ports(self) -> tuple[tuple[int, ...], ...]:
        """ The ports of each box in domain-then-codomain order. """
        return self.plan.box_ports

    @cached_property
    def memory_widths(self) -> tuple[int, ...]:
        """ The private memory width of each box occurrence. """
        return self.plan.memory_widths

    @cached_property
    def topological_order(self) -> tuple[int, ...]:
        """ Order boxes from boundary inputs towards boundary outputs. """
        domain_owner, codomain_owner, box_port_owner = {}, {}, {}
        for box_index, (arity, ports) in enumerate(zip(
                self.plan.box_dom_arities, self.box_ports)):
            for port in ports:
                box_port_owner[port] = box_index
            for port in ports[:arity]:
                domain_owner[port] = box_index
            for port in ports[arity:]:
                codomain_owner[port] = box_index

        dependencies = []
        for box_index, (arity, ports) in enumerate(zip(
                self.plan.box_dom_arities, self.box_ports)):
            current = set()
            for port in ports[:arity]:
                source = self.plan.edges[port]
                if source in codomain_owner:
                    current.add(codomain_owner[source])
                elif source in box_port_owner:
                    raise ValueError(
                        "A causal schedule requires every box input to be "
                        "wired from a box output or boundary input.")
                elif self.plan.port_kinds[source] != PortKind.INPUT:
                    raise ValueError(
                        "A causal schedule requires every box input to be "
                        "wired from a box output or boundary input.")
            dependencies.append(current)

            for port in ports[arity:]:
                target = self.plan.edges[port]
                if target in box_port_owner and target not in domain_owner:
                    raise ValueError(
                        "A causal schedule requires every box output to be "
                        "wired to a box input or boundary output.")
                if target not in box_port_owner\
                        and self.plan.port_kinds[target] != PortKind.OUTPUT:
                    raise ValueError(
                        "A causal schedule requires every box output to be "
                        "wired to a box input or boundary output.")

        remaining = [set(items) for items in dependencies]
        ready = [
            box_index for box_index, items in enumerate(remaining)
            if not items]
        order = []
        while ready:
            box_index = ready.pop(0)
            order.append(box_index)
            for target, items in enumerate(remaining):
                if box_index in items:
                    items.remove(box_index)
                    if not items and target not in order\
                            and target not in ready:
                        ready.append(target)
        if len(order) != self.plan.n_boxes:
            raise ValueError(
                "A causal schedule requires an acyclic box dependency graph.")
        return tuple(order)

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
        widths = self.plan.port_dims
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
            edge = self.plan.edges[i]
            incoming[edge] = incoming[edge] + boundary[i]

        self.initial = tuple(initial)
        self.boundary = tuple(boundary)
        self.incoming = tuple(incoming)
        self.outgoing = ()
        self.box_outputs = self.plan.n_boxes * (None, )
        self.memories = memories
        return self.incoming

    def activate_box(self, box_index: int, incoming):
        """ Apply one network to its public messages and private memory. """
        widths = self.plan.port_dims
        ports = self.box_ports[box_index]
        public_widths = tuple(widths[i] for i in ports)
        memory_width = self.memory_widths[box_index]
        values = tuple(incoming[i] for i in ports)\
            + (self.memories[box_index], )
        module = self.modules[self.plan.module_indices[box_index]]
        output = self.backend.activate(
            module, self.backend.concatenate(values))
        self.validate(
            output, sum(public_widths) + memory_width,
            f"output of box {box_index}")
        chunks = self.backend.split(
            output, public_widths + (memory_width, ))
        public, next_memory = chunks[:-1], chunks[-1]
        box_output = (
            self.backend.concatenate(public) if public else self.zeros(0))
        return public, next_memory, box_output

    def activate(self) -> tuple:
        """ Apply each network to its public messages and private memory. """
        widths = self.plan.port_dims
        outgoing = [self.zeros(width) for width in widths]
        for i in self.input_ports:
            outgoing[i] = self.boundary[i]

        box_outputs, memories = [], []
        for box_index, ports in enumerate(self.box_ports):
            public, next_memory, box_output = self.activate_box(
                box_index, self.incoming)
            box_outputs.append(box_output)
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
            self.outgoing[self.plan.edges[i]]
            for i in range(self.plan.n_ports))
        return self.incoming

    def inject(self) -> tuple:
        """ Add the initial messages to the current incoming messages. """
        self.incoming = tuple(
            message + initial
            for message, initial in zip(self.incoming, self.initial))
        return self.incoming

    def readout(self):
        """ Read boundary outputs, or final box outputs for a closed map. """
        if self.plan.has_boundary:
            return self.backend.concatenate(tuple(
                self.incoming[i] for i in self.output_ports))\
                if self.output_ports else self.zeros(0)
        return self.box_outputs

    def forward(
            self, n_rounds: int = None, inject: bool = True,
            return_memory: bool = False):
        """ Execute synchronous activation and routing rounds. """
        self.initialize()
        n_rounds = self.plan.n_boxes if n_rounds is None else n_rounds
        for _ in range(n_rounds):
            self.activate()
            self.route()
            if inject and self.init is not None:
                self.inject()
        result = self.readout()
        return (result, self.memories) if return_memory else result

    def forward_causal(
            self, inject: bool = True, return_memory: bool = False):
        """ Execute every box once in topological order. """
        self.initialize()
        widths = self.plan.port_dims
        incoming = list(self.incoming)
        outgoing = [self.zeros(width) for width in widths]
        for port in self.input_ports:
            outgoing[port] = self.boundary[port]
        box_outputs = list(self.box_outputs)
        memories = list(self.memories)

        for box_index in self.topological_order:
            ports = self.box_ports[box_index]
            public, next_memory, box_output = self.activate_box(
                box_index, incoming)
            box_outputs[box_index] = box_output
            memories[box_index] = next_memory
            for port, chunk in zip(ports, public):
                outgoing[port] = chunk
                target = self.plan.edges[port]
                incoming[target] = chunk + self.initial[target]\
                    if inject and self.init is not None else chunk

        self.incoming, self.outgoing = tuple(incoming), tuple(outgoing)
        self.box_outputs, self.memories = (
            tuple(box_outputs), tuple(memories))
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
